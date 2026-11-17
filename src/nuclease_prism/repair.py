"""
Post-cleavage DNA double-strand break (DSB) repair kinetics and indel outcome simulator.

Models coupled kinetic competition among three canonical cellular repair pathways:
1. NHEJ (Non-Homologous End Joining): fast repair generating 1-bp templated insertions
   at staggered 5' overhangs (dictated by the -4 nucleotide 5' of the cleavage site,
   following Shen et al., Nature 2018 / inDelphi & Allen et al., Nature Biotech 2018 / FORECasT)
   and short 1-5 bp end-resection deletions.
2. MMEJ (Microhomology-Mediated End Joining): slower resection-dependent annealing of
   flanking direct microhomology repeats weighted by repeat length L, GC content, and
   intervening distance penalty exp(-lambda * deletion_size).
3. HDR (Homology-Directed Repair): Michaelis-Menten / Hill-type donor template concentration
   dependence competing with Ku70/80 NHEJ binding and CtIP/MRE11 resection.

The ODE kinetic system is integrated over [0, 24] hours using SciPy's RK45 solver with
strict probability mass conservation:
    U(t) + NHEJ(t) + MMEJ(t) + HDR(t) = 1.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .sequence import find_microhomologies, gc_content
from .types import NucleaseClass, NucleaseProfile, RepairAllele, RepairPathway, RepairSpectrum


@dataclass(frozen=True)
class RepairKineticRates:
    """
    Kinetic rate constants (hr^-1) and microhomology decay parameters for DSB repair.

    Attributes:
        k_nhej_base: Basal NHEJ ligation rate (hr^-1).
        k_mmej_base: Basal MMEJ resection+annealing scale (hr^-1).
        k_hdr_max: Maximum HDR rate under saturating donor DNA (hr^-1).
        donor_km_nm: Half-saturation donor concentration K_m (nanomolar).
        mmej_distance_decay: Exponential span decay constant lambda (bp^-1).
        overhang_ins_propensity: Baseline propensity for +1 templated insertion in blunt/1-nt staggered cuts.
    """

    k_nhej_base: float = 0.58
    k_mmej_base: float = 0.16
    k_hdr_max: float = 0.42
    donor_km_nm: float = 120.0
    mmej_distance_decay: float = 0.085
    overhang_ins_propensity: float = 0.42


# Empirical nucleotide-specific propensity multipliers for +1 templated insertion at -4 base
# (A and T at position -4 strongly favor staggered 5' 1-bp overhang fill-in; G disfavors)
_NHEJ_PLUS1_BASE_FACTOR: Dict[str, float] = {
    "A": 1.38,
    "T": 1.32,
    "C": 0.92,
    "G": 0.45,
}


def simulate_repair_time_course(
    mmej_drive: float,
    donor_conc_nm: float = 0.0,
    rates: Optional[RepairKineticRates] = None,
    horizon_hours: float = 24.0,
    n_points: int = 25,
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
    """
    Integrate the coupled 4-state DSB repair ODE system over [0, horizon_hours].

    State vector y = [U, N, M, H]:
        dU/dt = -(k_nhej + k_mmej + k_hdr) * U
        dN/dt = k_nhej * U
        dM/dt = k_mmej * U
        dH/dt = k_hdr * U

    Args:
        mmej_drive: Normalized microhomology availability drive in [0.1, 3.0].
        donor_conc_nm: Exogenous HDR repair template concentration in nanomolar (0 = NHEJ/MMEJ only).
        rates: Optional RepairKineticRates configuration.
        horizon_hours: Integration endpoint in hours.
        n_points: Number of uniformly spaced output time points.

    Returns:
        Tuple of tuples ``(times_h, unrepaired, nhej_cum, mmej_cum, hdr_cum)``.
    """
    if rates is None:
        rates = RepairKineticRates()

    k_nhej = max(1e-4, rates.k_nhej_base)
    k_mmej = max(1e-4, rates.k_mmej_base * mmej_drive)
    if donor_conc_nm > 0.0:
        # Hill coefficient ~1.2 for RAD51/ssODN strand-invasion cooperativity
        c_pow = donor_conc_nm ** 1.2
        km_pow = rates.donor_km_nm ** 1.2
        k_hdr = rates.k_hdr_max * (c_pow / (km_pow + c_pow))
    else:
        k_hdr = 0.0

    def _rhs(_t: float, y: np.ndarray) -> np.ndarray:
        u = y[0]
        return np.array([-(k_nhej + k_mmej + k_hdr) * u, k_nhej * u, k_mmej * u, k_hdr * u], dtype=np.float64)

    t_eval = np.linspace(0.0, horizon_hours, n_points)
    sol = solve_ivp(
        _rhs,
        t_span=(0.0, horizon_hours),
        y0=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )

    # Renormalize strictly to eliminate 1e-15 floating drift
    totals = np.sum(sol.y, axis=0)
    y_norm = np.clip(sol.y / totals[None, :], 0.0, 1.0)

    return (
        tuple(round(float(x), 3) for x in sol.t),
        tuple(round(float(x), 6) for x in y_norm[0]),
        tuple(round(float(x), 6) for x in y_norm[1]),
        tuple(round(float(x), 6) for x in y_norm[2]),
        tuple(round(float(x), 6) for x in y_norm[3]),
    )


def predict_repair_spectrum(
    context_seq: str,
    cut_index: int,
    profile: NucleaseProfile,
    donor_conc_nm: float = 0.0,
    hdr_edit_label: str = "HDR:precise_donor",
    rates: Optional[RepairKineticRates] = None,
) -> RepairSpectrum:
    """
    Predict full post-cleavage indel & HDR allele spectrum at a CRISPR cut site.

    Combines:
    - Flanking direct microhomology scan for MMEJ deletion weights:
      w_MH(d, L) = exp(0.65 * L + 0.45 * L * f_GC - lambda * d)
    - Staggered-overhang +1 insertion and short 1-5 bp NHEJ deletions
    - Coupled ODE competition between NHEJ, MMEJ, and HDR

    Args:
        context_seq: Local 5'->3' DNA sequence flanking the cut site.
        cut_index: 0-based index immediately 5' of the cleavage cut.
        profile: NucleaseProfile (used to account for staggered vs blunt cut geometry).
        donor_conc_nm: Exogenous HDR template concentration in nanomolar.
        hdr_edit_label: Label assigned to the precise HDR outcome allele.
        rates: Optional RepairKineticRates.

    Returns:
        Populated RepairSpectrum object with normalized allele frequencies.
    """
    if rates is None:
        rates = RepairKineticRates()

    seq = context_seq.upper()
    if cut_index < 2 or cut_index >= len(seq) - 2:
        raise ValueError(f"cut_index={cut_index} too close to boundary of context_seq (len={len(seq)})")

    # Non-DSB editors (Cas13d, Base/Prime editors) have minimal classical DSB indel profiles
    if profile.nuclease_class == NucleaseClass.RNA_CAS13D:
        allele = RepairAllele(
            allele_label="RNA:knockdown",
            pathway=RepairPathway.NHEJ,
            indel_size=0,
            probability=1.0,
            is_frameshift=False,
            microhomology_len=0,
            resulting_seq=seq,
        )
        return RepairSpectrum(
            pathway_fractions={"nhej": 1.0, "mmej": 0.0, "hdr": 0.0},
            frameshift_prob=0.0,
            inframe_prob=1.0,
            hdr_efficiency=0.0,
            indel_entropy_bits=0.0,
            precision_score=1.0,
            top_alleles=(allele,),
            kinetic_time_course={"time_h": (0.0, 24.0), "edited": (0.0, 1.0)},
        )

    # 1. Scan flanking microhomologies for MMEJ deletions
    mh_hits = find_microhomologies(seq, cut_index, min_len=2, max_len=8, max_deletion=22)
    mh_weights: List[Tuple[int, int, str, int, float]] = []
    for del_size, mh_len, motif, l_start in mh_hits:
        fgc = gc_content(motif)
        w = math.exp(0.62 * mh_len + 0.40 * mh_len * fgc - rates.mmej_distance_decay * del_size)
        mh_weights.append((del_size, mh_len, motif, l_start, w))

    total_mh_score = sum(item[4] for item in mh_weights)
    # Map total microhomology strength to kinetic MMEJ drive in [0.15, 2.8]
    mmej_drive = max(0.15, min(2.8, 0.25 + 0.35 * math.log1p(total_mh_score)))

    # 2. Solve coupled NHEJ/MMEJ/HDR kinetic ODEs
    times_h, unrep, nhej_curve, mmej_curve, hdr_curve = simulate_repair_time_course(
        mmej_drive=mmej_drive,
        donor_conc_nm=donor_conc_nm,
        rates=rates,
    )
    resolved_total = max(1e-9, nhej_curve[-1] + mmej_curve[-1] + hdr_curve[-1])
    f_nhej = nhej_curve[-1] / resolved_total
    f_mmej = mmej_curve[-1] / resolved_total
    f_hdr = hdr_curve[-1] / resolved_total

    alleles: List[RepairAllele] = []

    # 3. Allocate MMEJ alleles
    if mh_weights and f_mmej > 1e-6:
        for del_size, mh_len, motif, l_start, w in mh_weights[:10]:
            prob = f_mmej * (w / total_mh_score)
            res_seq = seq[:l_start] + seq[l_start + del_size :]
            alleles.append(
                RepairAllele(
                    allele_label=f"-{del_size}:MH({motif})",
                    pathway=RepairPathway.MMEJ,
                    indel_size=-del_size,
                    probability=prob,
                    is_frameshift=(del_size % 3 != 0),
                    microhomology_len=mh_len,
                    resulting_seq=res_seq,
                )
            )
    else:
        # Fallback if no microhomology >=2 bp exists: fold small MMEJ budget into NHEJ
        f_nhej += f_mmej
        f_mmej = 0.0

    # 4. Allocate NHEJ alleles (+1 templated insertion, +2 insertion, -1..-5 deletions)
    if f_nhej > 1e-6:
        template_base = seq[cut_index - 1]
        base_factor = _NHEJ_PLUS1_BASE_FACTOR.get(template_base, 1.0)
        # Cas12a staggered 5-bp overhang cuts strongly favor deletions over +1 templated insertions
        if profile.cleavage_overhang_bp >= 4:
            p1_raw = 0.08 * rates.overhang_ins_propensity
            del_weights = {1: 0.10, 2: 0.16, 3: 0.22, 4: 0.28, 5: 0.24}
        else:
            p1_raw = rates.overhang_ins_propensity * base_factor
            del_weights = {1: 0.36, 2: 0.26, 3: 0.18, 4: 0.12, 5: 0.08}

        raw_nhej_items: List[Tuple[str, int, float, str]] = []
        # +1 templated insertion
        raw_nhej_items.append(
            (f"+1:{template_base}", 1, p1_raw, seq[:cut_index] + template_base + seq[cut_index:])
        )
        # +2 minor insertion
        raw_nhej_items.append(
            (
                f"+2:{template_base}{template_base}",
                2,
                0.08 * p1_raw,
                seq[:cut_index] + template_base * 2 + seq[cut_index:],
            )
        )
        # -1 to -5 NHEJ deletions
        rem_del_budget = max(0.15, 1.0 - min(0.70, p1_raw * 0.7))
        for d_size, dw in del_weights.items():
            d_lo = max(0, cut_index - (d_size // 2))
            d_hi = d_lo + d_size
            res_seq = seq[:d_lo] + seq[d_hi:]
            raw_nhej_items.append((f"-{d_size}:NHEJ", -d_size, rem_del_budget * dw, res_seq))

        nhej_norm = sum(item[2] for item in raw_nhej_items)
        for label, sz, w, res_seq in raw_nhej_items:
            alleles.append(
                RepairAllele(
                    allele_label=label,
                    pathway=RepairPathway.NHEJ,
                    indel_size=sz,
                    probability=f_nhej * (w / nhej_norm),
                    is_frameshift=(abs(sz) % 3 != 0),
                    microhomology_len=0,
                    resulting_seq=res_seq,
                )
            )

    # 5. Allocate HDR allele if donor is active
    if f_hdr > 1e-6:
        alleles.append(
            RepairAllele(
                allele_label=hdr_edit_label,
                pathway=RepairPathway.HDR,
                indel_size=0,
                probability=f_hdr,
                is_frameshift=False,
                microhomology_len=0,
                resulting_seq=seq,
            )
        )

    # Strict renormalization across all emitted alleles
    total_p = sum(a.probability for a in alleles)
    norm_alleles: List[RepairAllele] = []
    for a in alleles:
        p_norm = a.probability / total_p
        norm_alleles.append(
            RepairAllele(
                allele_label=a.allele_label,
                pathway=a.pathway,
                indel_size=a.indel_size,
                probability=round(p_norm, 6),
                is_frameshift=a.is_frameshift,
                microhomology_len=a.microhomology_len,
                resulting_seq=a.resulting_seq,
            )
        )

    norm_alleles.sort(key=lambda a: -a.probability)

    # Compute summary statistics
    frameshift_p = sum(a.probability for a in norm_alleles if a.is_frameshift)
    inframe_p = sum(a.probability for a in norm_alleles if (not a.is_frameshift and a.pathway != RepairPathway.HDR))
    entropy = 0.0
    for a in norm_alleles:
        if a.probability > 1e-9:
            entropy -= a.probability * math.log2(a.probability)

    max_entropy = math.log2(max(2, len(norm_alleles)))
    precision = max(0.0, min(1.0, 1.0 - (entropy / max_entropy)))

    return RepairSpectrum(
        pathway_fractions={
            "nhej": round(f_nhej, 4),
            "mmej": round(f_mmej, 4),
            "hdr": round(f_hdr, 4),
        },
        frameshift_prob=round(min(1.0, frameshift_p), 4),
        inframe_prob=round(min(1.0, inframe_p), 4),
        hdr_efficiency=round(f_hdr, 4),
        indel_entropy_bits=round(entropy, 4),
        precision_score=round(precision, 4),
        top_alleles=tuple(norm_alleles[:12]),
        kinetic_time_course={
            "time_h": times_h,
            "unrepaired": unrep,
            "nhej": nhej_curve,
            "mmej": mmej_curve,
            "hdr": hdr_curve,
        },
    )
