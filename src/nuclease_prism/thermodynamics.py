"""
RNA:DNA R-loop nearest-neighbor thermodynamics and Markov kinetic cleavage master equation.

Implements:
1. Sugimoto et al. (Biochemistry 1995) nearest-neighbor enthalpy (deltaH) and entropy (deltaS)
   parameters for all 16 RNA:DNA dinucleotide steps (5'-rX rY-3' paired to 3'-dZ dW-5').
2. SantaLucia (PNAS 1998) nearest-neighbor parameters for displaced DNA:DNA duplex steps.
3. Mismatch free-energy destabilization tensor ddG_mm(r_base, d_base, position) capturing
   rG:dT wobble stability versus purine-purine (rA:dG, rG:dA) steric clashes.
4. Discrete-state continuous-time Markov R-loop propagation master equation across states
   {0 (PAM-bound), 1, ..., L (fully zippered)} with backward/forward step rates governed by
   Boltzmann energy barriers, PAM dissociation rate k_off, and terminal cleavage rate k_cat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import solve

from .nucleases import pam_zipping_order
from .types import NucleaseProfile, ThermodynamicProfile


GAS_CONSTANT_KCAL = 1.98720425864083e-3  # kcal / (mol * K)
DEFAULT_TEMP_K = 310.15  # 37.0 C


# Sugimoto et al. (1995) RNA:DNA nearest-neighbor parameters:
# Key is RNA dinucleotide (5'->3', using T for U in DNA-equivalent indexing)
# Value is (deltaH in kcal/mol, deltaS in cal/(mol*K))
SUGIMOTO_RNA_DNA_NN: Dict[str, Tuple[float, float]] = {
    "AA": (-7.8, -21.9),
    "AC": (-5.9, -12.3),
    "AG": (-9.1, -23.5),
    "AT": (-8.3, -23.9),
    "CA": (-9.0, -26.1),
    "CC": (-9.3, -23.2),
    "CG": (-16.3, -47.1),
    "CT": (-7.0, -19.7),
    "GA": (-5.5, -13.5),
    "GC": (-8.0, -17.1),
    "GG": (-12.8, -31.9),
    "GT": (-7.8, -21.6),
    "TA": (-7.8, -23.2),
    "TC": (-8.6, -22.9),
    "TG": (-10.4, -28.4),
    "TT": (-11.5, -36.4),
}

# Initiation term for RNA:DNA hybrid formation (Sugimoto 1995)
RNA_DNA_INIT: Tuple[float, float] = (1.9, -3.9)

# SantaLucia (1998) unified DNA:DNA nearest-neighbor parameters:
# Key is DNA dinucleotide (5'->3'), value is (deltaH kcal/mol, deltaS cal/(mol*K))
SANTALUCIA_DNA_DNA_NN: Dict[str, Tuple[float, float]] = {
    "AA": (-7.9, -22.2),
    "TT": (-7.9, -22.2),
    "AT": (-7.2, -20.4),
    "TA": (-7.2, -21.3),
    "CA": (-8.5, -22.7),
    "TG": (-8.5, -22.7),
    "GT": (-8.4, -22.4),
    "AC": (-8.4, -22.4),
    "CT": (-7.8, -21.0),
    "AG": (-7.8, -21.0),
    "GA": (-8.2, -22.2),
    "TC": (-8.2, -22.2),
    "CG": (-10.6, -27.2),
    "GC": (-9.8, -24.4),
    "GG": (-8.0, -19.9),
    "CC": (-8.0, -19.9),
}

DNA_DNA_INIT_AT: Tuple[float, float] = (2.3, 4.1)
DNA_DNA_INIT_GC: Tuple[float, float] = (0.1, -2.8)

# RNA:DNA single-mismatch penalty delta-delta-G (kcal/mol) indexed by (rna_base, target_dna_sense_base)
# Note: matched base means rna_base == target_dna_sense_base (which is complementary to target strand).
MISMATCH_PENALTY_KCAL: Dict[Tuple[str, str], float] = {
    # Wobble-like tolerated pairs (rG:dT mismatch -> spacer G vs target-sense A)
    ("G", "A"): 1.05,  # rG pairs with dT on antisense strand
    ("T", "G"): 1.35,  # rU pairs with dG on antisense strand
    # Pyrimidine-pyrimidine moderate destabilization
    ("C", "T"): 2.25,
    ("T", "C"): 2.15,
    ("C", "A"): 2.40,
    ("A", "C"): 2.35,
    # Purine-purine steric clashes (strongest disruption)
    ("A", "G"): 3.15,
    ("G", "T"): 2.85,
    ("A", "T"): 2.65,
    ("T", "A"): 2.75,
    ("G", "C"): 3.45,
    ("C", "G"): 3.40,
}


@dataclass(frozen=True)
class KineticParameters:
    """
    Calibratable biophysical kinetic parameters for the R-loop master equation.

    Attributes:
        k_zip_forward: Base forward zipping rate per nt step (s^-1).
        rloop_barrier_scale: Dimensionless scaling factor on Boltzmann free-energy step differences.
        protein_stabilization_per_nt: Mean Cas protein R-loop stabilization energy per base pair (kcal/mol).
        seed_mismatch_multiplier: Extra energetic penalty multiplier inside PAM-proximal seed.
        temperature_k: Simulation temperature in Kelvin.
    """

    k_zip_forward: float = 120.0
    rloop_barrier_scale: float = 1.0
    protein_stabilization_per_nt: float = -0.32
    seed_mismatch_multiplier: float = 1.45
    temperature_k: float = DEFAULT_TEMP_K


def compute_duplex_enthalpy_entropy(
    seq: str,
    nn_table: Dict[str, Tuple[float, float]],
    init_dh: float,
    init_ds: float,
) -> Tuple[float, float]:
    """
    Sum nearest-neighbor enthalpy (kcal/mol) and entropy (cal/(mol*K)) along ``seq``.

    Args:
        seq: Uppercase nucleotide sequence.
        nn_table: Dinucleotide lookup table.
        init_dh: Initiation enthalpy in kcal/mol.
        init_ds: Initiation entropy in cal/(mol*K).

    Returns:
        Tuple ``(total_deltaH_kcal, total_deltaS_cal_per_K)``.
    """
    upper = seq.upper()
    dh = init_dh
    ds = init_ds
    for i in range(len(upper) - 1):
        dinuc = upper[i : i + 2]
        step_dh, step_ds = nn_table.get(dinuc, (-8.0, -22.0))
        dh += step_dh
        ds += step_ds
    return dh, ds


def compute_hybrid_melting_temp(
    spacer_seq: str,
    na_molar: float = 0.15,
    strand_conc_molar: float = 1e-6,
) -> float:
    """
    Estimate RNA:DNA hybrid melting temperature T_m in degrees Celsius with salt correction.

    Uses Sugimoto nearest-neighbor parameters and SantaLucia monovalent cation correction:
    1/T_m(Na+) = 1/T_m(1M) + (4.29 * f_GC - 3.95) * 1e-5 * ln[Na+] + 9.40e-6 * ln^2[Na+]

    Args:
        spacer_seq: DNA-alphabet spacer sequence (5'->3').
        na_molar: Effective monovalent salt concentration in mol/L (physiological ~0.15 M).
        strand_conc_molar: Total effective RNP-target concentration in mol/L.

    Returns:
        Melting temperature in Celsius.
    """
    upper = spacer_seq.upper()
    dh, ds = compute_duplex_enthalpy_entropy(
        upper, SUGIMOTO_RNA_DNA_NN, RNA_DNA_INIT[0], RNA_DNA_INIT[1]
    )
    # Convert deltaS from cal/(mol*K) to kcal/(mol*K)
    ds_kcal = ds / 1000.0
    denom = ds_kcal + GAS_CONSTANT_KCAL * math.log(max(1e-12, strand_conc_molar / 4.0))
    if abs(denom) < 1e-9:
        return 65.0
    tm_1m_k = dh / denom

    # Salt correction
    f_gc = (upper.count("G") + upper.count("C")) / max(1, len(upper))
    ln_na = math.log(max(1e-4, na_molar))
    inv_tm = (
        (1.0 / tm_1m_k)
        + (4.29 * f_gc - 3.95) * 1e-5 * ln_na
        + 9.40e-6 * (ln_na ** 2)
    )
    tm_k = 1.0 / max(1e-4, inv_tm)
    return round(tm_k - 273.15, 2)


def compute_step_rloop_energies(
    spacer_seq: str,
    target_sense_seq: str,
    profile: NucleaseProfile,
    params: Optional[KineticParameters] = None,
) -> Tuple[float, float, List[float], float]:
    """
    Compute per-step net R-loop free energy changes in PAM-to-distal zipping order.

    At each step k along the R-loop zipping trajectory:
        ddG_step(k) = dG_RNA:DNA(k) - dG_DNA:DNA(k) + protein_stabilization + ddG_mismatch(k)

    Args:
        spacer_seq: 5'->3' guide RNA spacer sequence (DNA alphabet).
        target_sense_seq: 5'->3' target protospacer sense-strand DNA sequence (same length).
        profile: NucleaseProfile defining PAM side, seed range, and initiation penalty.
        params: Optional KineticParameters.

    Returns:
        Tuple ``(dg_rna_dna, dg_dna_dna, step_ddg_list, ddg_mismatch_total)``.
    """
    if params is None:
        params = KineticParameters()
    if len(spacer_seq) != len(target_sense_seq):
        raise ValueError(
            f"Spacer length {len(spacer_seq)} != target length {len(target_sense_seq)}"
        )

    sp = spacer_seq.upper()
    tg = target_sense_seq.upper()
    temp_k = params.temperature_k

    # Total RNA:DNA hybrid dG at temperature T
    dh_rd, ds_rd = compute_duplex_enthalpy_entropy(
        sp, SUGIMOTO_RNA_DNA_NN, RNA_DNA_INIT[0], RNA_DNA_INIT[1]
    )
    dg_rna_dna = dh_rd - temp_k * (ds_rd / 1000.0)

    # Total DNA:DNA duplex dG for the target protospacer
    end_gc = tg[0] in ("G", "C") or tg[-1] in ("G", "C")
    init_dna = DNA_DNA_INIT_GC if end_gc else DNA_DNA_INIT_AT
    dh_dd, ds_dd = compute_duplex_enthalpy_entropy(
        tg, SANTALUCIA_DNA_DNA_NN, init_dna[0], init_dna[1]
    )
    dg_dna_dna = dh_dd - temp_k * (ds_dd / 1000.0)

    zip_order = pam_zipping_order(profile)
    step_ddg: List[float] = []
    total_mm_penalty = 0.0
    seed_lo, seed_hi = profile.seed_range

    for step_idx, pos0 in enumerate(zip_order):
        pos1 = pos0 + 1
        # Local dinucleotide contribution around pos0
        if pos0 < len(sp) - 1:
            dinuc_r = sp[pos0 : pos0 + 2]
            dinuc_d = tg[pos0 : pos0 + 2]
        else:
            dinuc_r = sp[pos0 - 1 : pos0 + 1]
            dinuc_d = tg[pos0 - 1 : pos0 + 1]

        r_dh, r_ds = SUGIMOTO_RNA_DNA_NN.get(dinuc_r, (-8.0, -22.0))
        d_dh, d_ds = SANTALUCIA_DNA_DNA_NN.get(dinuc_d, (-8.0, -22.0))
        local_rd = r_dh - temp_k * (r_ds / 1000.0)
        local_dd = d_dh - temp_k * (d_ds / 1000.0)

        # Differential strand-exchange free energy + Cas RNP protein stabilization
        base_ddg = (local_rd - local_dd) + params.protein_stabilization_per_nt
        # Add mild initiation barrier to the very first PAM-proximal step
        if step_idx == 0:
            base_ddg += profile.rloop_initiation_penalty * 0.45

        # Mismatch penalty if spacer base != target sense base
        mm_pen = 0.0
        if sp[pos0] != tg[pos0]:
            mm_pen = MISMATCH_PENALTY_KCAL.get((sp[pos0], tg[pos0]), 2.6)
            if seed_lo <= pos1 <= seed_hi:
                mm_pen *= params.seed_mismatch_multiplier
            total_mm_penalty += mm_pen

        step_ddg.append((base_ddg + mm_pen) * params.rloop_barrier_scale)

    return dg_rna_dna, dg_dna_dna, step_ddg, total_mm_penalty


def solve_rloop_master_equation(
    step_ddg: List[float],
    profile: NucleaseProfile,
    params: Optional[KineticParameters] = None,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Solve the 1D continuous-time Markov R-loop master equation for cleavage probability.

    State space: s in {0, 1, ..., L}, where s=0 is PAM-bound closed DNA and s=L is the
    catalytically active fully zippered R-loop.
    Transition rates:
        - Forward zipping:  k_f(s -> s+1) = k_zip * exp(-0.5 * beta * ddG_step[s])
        - Backward unzip:   k_b(s+1 -> s) = k_zip * exp(+0.5 * beta * ddG_step[s])
    Absorbing exit channels:
        - From state 0: PAM dissociation rate k_off = dissociation_rate_base
        - From state L: DNA cleavage rate k_cat = cleavage_rate_max
          (and slight pre-terminal cleavage leak from L-1 at 0.12 * k_cat)

    Under steady-flux initialization at state 0, the hitting/splitting probability vector
    and mean dwell occupancy satisfy the tridiagonal linear system:
        M * tau = -e_0,   P_cleave = k_cat * tau[L] + 0.12 * k_cat * tau[L-1]
    where tau[s] is mean residence time (seconds) in state s prior to absorption.

    Args:
        step_ddg: Per-step R-loop free energy increments in kcal/mol.
        profile: NucleaseProfile specifying k_cat and k_off.
        params: Optional KineticParameters.

    Returns:
        Tuple ``(normalized_state_occupancy, cleavage_probability, cumulative_dg_trajectory)``.
    """
    if params is None:
        params = KineticParameters()

    n_steps = len(step_ddg)
    n_states = n_steps + 1
    beta = 1.0 / (GAS_CONSTANT_KCAL * params.temperature_k)

    # Cumulative R-loop free energy trajectory starting at initiation barrier
    cum_dg: List[float] = [round(profile.rloop_initiation_penalty * 0.55, 4)]
    running = cum_dg[0]
    for d in step_ddg:
        running += d
        cum_dg.append(round(running, 4))

    # Build generator submatrix Q (size n_states x n_states) for transient states 0..L
    q_mat = np.zeros((n_states, n_states), dtype=np.float64)

    for s in range(n_steps):
        # Clamp exponent for numerical stability on extreme mismatch stacks
        exponent = max(-18.0, min(18.0, 0.5 * beta * step_ddg[s]))
        kf = params.k_zip_forward * math.exp(-exponent)
        kb = params.k_zip_forward * math.exp(exponent)

        q_mat[s, s + 1] += kf
        q_mat[s, s] -= kf
        q_mat[s + 1, s] += kb
        q_mat[s + 1, s + 1] -= kb

    # Exit (absorbing) channels from boundary states
    k_off = max(1e-4, profile.dissociation_rate_base)
    k_cat = max(1e-4, profile.cleavage_rate_max)
    k_leak = 0.12 * k_cat

    q_mat[0, 0] -= k_off
    q_mat[n_states - 1, n_states - 1] -= k_cat
    if n_states >= 2:
        q_mat[n_states - 2, n_states - 2] -= k_leak

    # Solve Q^T * tau = -e_0 for expected residence times tau[s]
    rhs = np.zeros(n_states, dtype=np.float64)
    rhs[0] = -1.0
    tau = solve(q_mat.T, rhs)
    tau = np.maximum(tau, 0.0)

    total_dwell = float(np.sum(tau))
    if total_dwell > 1e-12:
        occupancy = tau / total_dwell
    else:
        occupancy = np.zeros(n_states, dtype=np.float64)
        occupancy[0] = 1.0

    p_cleave = float(k_cat * tau[n_states - 1] + (k_leak * tau[n_states - 2] if n_states >= 2 else 0.0))
    p_cleave = max(0.0, min(1.0, p_cleave))
    return occupancy, p_cleave, cum_dg


def analyze_rloop_thermodynamics(
    spacer_seq: str,
    profile: NucleaseProfile,
    target_sense_seq: Optional[str] = None,
    params: Optional[KineticParameters] = None,
) -> ThermodynamicProfile:
    """
    Compute complete R-loop thermodynamic & kinetic profile for a guide-target pair.

    Args:
        spacer_seq: 5'->3' spacer sequence.
        profile: NucleaseProfile definition.
        target_sense_seq: Optional mismatched target sense-strand sequence (defaults to perfect match).
        params: Optional KineticParameters.

    Returns:
        Populated ThermodynamicProfile object.
    """
    if target_sense_seq is None:
        target_sense_seq = spacer_seq

    dg_rd, dg_dd, step_ddg, mm_pen = compute_step_rloop_energies(
        spacer_seq, target_sense_seq, profile, params
    )
    occupancy, p_cleave, cum_dg = solve_rloop_master_equation(step_ddg, profile, params)
    tm_c = compute_hybrid_melting_temp(spacer_seq)

    # Seed stability: sum of first 8 PAM-proximal step energies
    seed_len = profile.seed_range[1] - profile.seed_range[0] + 1
    seed_stab = sum(step_ddg[: max(1, min(seed_len, len(step_ddg)))])
    net_ddg = (dg_rd - dg_dd) + profile.rloop_initiation_penalty + mm_pen

    return ThermodynamicProfile(
        dg_rna_dna_kcal=round(dg_rd, 3),
        dg_dna_dna_kcal=round(dg_dd, 3),
        ddg_net_rloop_kcal=round(net_ddg, 3),
        tm_hybrid_celsius=tm_c,
        positional_dg_profile=tuple(cum_dg),
        state_occupancy=tuple(round(float(x), 6) for x in occupancy),
        kinetic_cleavage_prob=round(p_cleave, 6),
        seed_stability_kcal=round(seed_stab, 3),
    )
