"""
Multi-modal CRISPR guide scoring: on-target efficacy, CFD/MIT off-target specificity,
and Base/Prime editing window analytics.

Implements:
1. Azimuth/RuleSet2-inspired biophysical on-target scoring combining:
   - R-loop kinetic cleavage probability P_cleave from the Markov master equation
   - Positional nucleotide preferences (G20 stabilization, C16, PAM-proximal purines, avoiding T20)
   - GC content quadratic optimum (50-65% sweet spot) & local GC-clamp dispersion
   - Pol-III U6 terminator penalty & sgRNA hairpin MFE interference penalty
   - Chromatin accessibility modulation
2. Cutting Frequency Determination (CFD) multiplicative mismatch + PAM + bulge scoring
   (Doench et al., Nature Biotechnology 2016) & MIT/Hsu spatial clustering penalty.
3. Base Editor (ABE8e / CBE4max) bystander editing window profile and Prime Editor (PE3)
   Primer Binding Site (PBS) Tm + Reverse Transcriptase Template (RTT) optimization.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .indexing import GenomeFMIndex
from .nucleases import map_spacer_to_pam_distance
from .sequence import compute_rna_hairpin_mfe, gc_content, reverse_complement, sliding_gc_extremes
from .thermodynamics import MISMATCH_PENALTY_KCAL, compute_hybrid_melting_temp
from .types import (
    BulgeType,
    EditingWindowProfile,
    NucleaseClass,
    NucleaseProfile,
    OffTargetHit,
    ProtospacerCandidate,
    ThermodynamicProfile,
)


# Genomic annotation functional risk weights for genome-wide specificity aggregation
ANNOTATION_RISK_WEIGHTS: Dict[str, float] = {
    "exon": 1.0,
    "promoter": 0.75,
    "intron": 0.25,
    "intergenic": 0.12,
}

# Doench 2016 CFD-calibrated positional retention factors by PAM-proximal distance (1 = adjacent to PAM)
# Values represent survival fraction of nuclease cleavage with a non-wobble mismatch at distance d.
_CFD_DIST_SURVIVAL: Dict[int, float] = {
    1: 0.08,
    2: 0.11,
    3: 0.14,
    4: 0.17,
    5: 0.22,
    6: 0.28,
    7: 0.36,
    8: 0.44,
    9: 0.52,
    10: 0.60,
    11: 0.68,
    12: 0.74,
    13: 0.79,
    14: 0.83,
    15: 0.87,
    16: 0.90,
    17: 0.93,
    18: 0.95,
    19: 0.97,
    20: 0.98,
    21: 0.985,
    22: 0.99,
    23: 0.992,
}

# Hsu / MIT positional mismatch penalty weights w_d (higher = stronger disruption when mismatched)
_MIT_POS_WEIGHTS: Tuple[float, ...] = (
    0.000, 0.000, 0.014, 0.000, 0.000,
    0.395, 0.317, 0.000, 0.389, 0.079,
    0.445, 0.508, 0.613, 0.851, 0.732,
    0.828, 0.615, 0.804, 0.685, 0.583,
)


def compute_cfd_score(
    spacer_seq: str,
    aligned_spacer: str,
    aligned_target: str,
    pam_weight: float,
    bulge_type: BulgeType,
    profile: NucleaseProfile,
) -> Tuple[float, float]:
    """
    Compute CFD off-target cleavage likelihood and thermodynamic ddG penalty.

    Models:
    - Multiplicative per-mismatch retention factors modulated by PAM-proximal distance
      and wobble identity (rG:dT retains ~2.2x higher cleavage survival than purine clashes).
    - Bulge gap retention penalty (0.28 for 1-nt DNA bulge, 0.20 for 1-nt RNA bulge).
    - Multiplicative PAM licensing weight.

    Args:
        spacer_seq: Original ungapped spacer sequence.
        aligned_spacer: Aligned guide representation (may contain '-').
        aligned_target: Aligned genomic target representation (may contain '-').
        pam_weight: PAM licensing factor in (0, 1].
        bulge_type: Alignment bulge classification.
        profile: NucleaseProfile instance.

    Returns:
        Tuple ``(cfd_score_in_0_1, ddg_penalty_kcal)``.
    """
    score = float(pam_weight)
    ddg_total = 0.0

    if bulge_type == BulgeType.DNA_BULGE:
        score *= 0.28
        ddg_total += 2.35
    elif bulge_type == BulgeType.RNA_BULGE:
        score *= 0.20
        ddg_total += 2.75

    guide_pos = 0
    for g_ch, t_ch in zip(aligned_spacer, aligned_target):
        if g_ch == "-":
            continue
        guide_pos += 1
        if t_ch == "-":
            continue
        if g_ch != t_ch and guide_pos <= len(spacer_seq):
            pam_dist = map_spacer_to_pam_distance(guide_pos, profile)
            base_surv = _CFD_DIST_SURVIVAL.get(pam_dist, 0.99)
            # Wobble rG:dT (spacer G vs sense A) or rU:dG (spacer T vs sense G) is significantly tolerated
            if (g_ch == "G" and t_ch == "A") or (g_ch == "T" and t_ch == "G"):
                surv = min(0.98, base_surv * 1.85 + 0.10)
            else:
                surv = base_surv
            score *= surv
            ddg_total += MISMATCH_PENALTY_KCAL.get((g_ch, t_ch), 2.5)

    return round(max(0.0, min(1.0, score)), 6), round(ddg_total, 3)


def compute_mit_score(
    mismatch_positions: Tuple[int, ...],
    pam_weight: float,
    bulge_type: BulgeType,
    spacer_length: int,
) -> float:
    """
    Compute Hsu et al. (Nature Biotechnology 2013) MIT off-target score with clustering penalty.

    Combines:
    - Term 1: Product of (1 - w_i) across mismatched positions.
    - Term 2: Pairwise spatial clustering factor 1 / ((19 - d_mean)/19 * 4 + 1).
    - Term 3: Inverse quadratic mismatch burden 1 / n_mm^2.
    """
    if bulge_type != BulgeType.NONE:
        # Penalize bulges strongly in MIT metric
        return round(0.18 * pam_weight / max(1, (len(mismatch_positions) + 1) ** 2), 6)

    if not mismatch_positions:
        return round(float(pam_weight), 6)

    term1 = 1.0
    mapped_indices: List[int] = []
    for pos1 in mismatch_positions:
        # Map to 20-nt canonical MIT coordinate
        idx20 = min(19, max(0, int(round((pos1 - 1) * 19.0 / max(1, spacer_length - 1)))))
        mapped_indices.append(idx20)
        term1 *= max(0.01, 1.0 - _MIT_POS_WEIGHTS[idx20])

    n_mm = len(mapped_indices)
    if n_mm >= 2:
        diffs = [
            abs(mapped_indices[b] - mapped_indices[a])
            for a in range(n_mm)
            for b in range(a + 1, n_mm)
        ]
        d_mean = sum(diffs) / len(diffs)
        term2 = 1.0 / (((19.0 - d_mean) / 19.0) * 4.0 + 1.0)
    else:
        term2 = 1.0

    term3 = 1.0 / float(n_mm ** 2)
    return round(max(0.0, min(1.0, pam_weight * term1 * term2 * term3)), 6)


def score_on_target_efficacy(
    candidate: ProtospacerCandidate,
    thermo: ThermodynamicProfile,
    profile: NucleaseProfile,
) -> float:
    """
    Compute calibrated on-target cleavage efficacy score in [0, 100].

    Integrates:
    - R-loop Markov cleavage probability P_cleave (dominant mechanistic driver)
    - Sequence-specific positional nucleotide weights (G at -1 relative to PAM, avoiding T at -1/-2)
    - Optimal GC window (45%-65% optimal, steep penalty <30% or >78%) & local GC variance
    - Pol-III U6 terminator penalty & sgRNA hairpin MFE sequestration penalty
    - Chromatin accessibility & PAM licensing weight

    Args:
        candidate: Extracted ProtospacerCandidate.
        thermo: Computed ThermodynamicProfile.
        profile: NucleaseProfile definition.

    Returns:
        Score on [0.0, 100.0] scale.
    """
    sp = candidate.spacer_seq.upper()
    gc = candidate.gc_content
    min_gc5, max_gc5 = sliding_gc_extremes(sp, window=5)

    # 1. Logistic combination of mechanistic and empirical Azimuth-like features
    logit = 0.0

    # Kinetic R-loop cleavage probability contributes strongly
    p_cl = max(1e-4, min(0.999, thermo.kinetic_cleavage_prob))
    logit += 2.35 * (p_cl - 0.55)

    # Optimal net R-loop energy around [-14, -4] kcal/mol
    ddg = thermo.ddg_net_rloop_kcal
    logit -= 0.045 * ((ddg + 8.5) ** 2)

    # GC sweet spot centered near 0.54
    logit -= 14.0 * ((gc - 0.54) ** 2)
    # Extreme local AT breathing or GC-clamp jam penalty
    if min_gc5 < 0.15:
        logit -= 0.45
    if max_gc5 > 0.90:
        logit -= 0.35

    # Position-specific PAM-adjacent base preferences (for 3'-PAM Cas9/editors)
    if profile.pam_side == "3prime":
        last_base = sp[-1]
        penult_base = sp[-2]
        if last_base == "G":
            logit += 0.55
        elif last_base == "T":
            logit -= 0.65
        if penult_base == "C":
            logit += 0.25
        elif penult_base == "T":
            logit -= 0.30
        # Position 16 (spacer[-5]) C enrichment in RuleSet2
        if len(sp) >= 5 and sp[-5] == "C":
            logit += 0.28
        # PAM variable base NGG: CGG/TGG slightly preferred over AGG/GGG for loading
        if len(candidate.pam_seq) >= 1 and candidate.pam_seq[0] == "C":
            logit += 0.20
    else:
        # Cas12a 5'-PAM preferences: G/C enrichment in seed positions 1..6
        seed_gc = gc_content(sp[:6])
        logit += 0.65 * (seed_gc - 0.45)

    # Pol-III transcriptional termination & secondary structure folding penalties
    logit -= 2.8 * candidate.pol3_terminator_risk
    if candidate.hairpin_mfe_kcal < -1.5:
        logit -= 0.42 * abs(candidate.hairpin_mfe_kcal + 1.5)

    # Chromatin accessibility modulation
    logit += 0.85 * (candidate.accessibility - 0.70)

    prob = 1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, logit))))
    raw_score = 100.0 * prob * candidate.pam_weight
    return round(max(0.0, min(100.0, raw_score)), 2)


def evaluate_off_targets(
    candidate: ProtospacerCandidate,
    profile: NucleaseProfile,
    fm_index: GenomeFMIndex,
    max_mismatches: int = 3,
    include_bulges: bool = True,
) -> Tuple[float, float, Tuple[OffTargetHit, ...]]:
    """
    Query genome-wide FM-index for off-target hits and compute aggregate specificity.

    Specificity formula:
        S_spec = 100.0 / (1.0 + sum_j (CFD_j * annotation_weight_j))

    Args:
        candidate: ProtospacerCandidate to profile.
        profile: NucleaseProfile instance.
        fm_index: Built GenomeFMIndex over the reference genome.
        max_mismatches: Maximum Hamming mismatch cutoff.
        include_bulges: Whether to include 1-nt DNA/RNA bulge off-targets.

    Returns:
        Tuple ``(specificity_score_0_100, aggregate_cfd_burden, sorted_off_target_hits)``.
    """
    raw_hits = fm_index.search_pam_offtargets(
        spacer_seq=candidate.spacer_seq,
        profile=profile,
        max_mismatches=max_mismatches,
        include_bulges=include_bulges,
        exclude_locus=(candidate.chrom, candidate.start, candidate.strand),
    )

    hits: List[OffTargetHit] = []
    cfd_burden = 0.0

    for (
        chrom,
        pos,
        strand,
        aln_sp,
        aln_tg,
        pam_obs,
        bulge_type,
        mm_count,
        mm_pos,
        pam_w,
    ) in raw_hits:
        cfd, ddg_pen = compute_cfd_score(
            spacer_seq=candidate.spacer_seq,
            aligned_spacer=aln_sp,
            aligned_target=aln_tg,
            pam_weight=pam_w,
            bulge_type=bulge_type,
            profile=profile,
        )
        mit = compute_mit_score(
            mismatch_positions=mm_pos,
            pam_weight=pam_w,
            bulge_type=bulge_type,
            spacer_length=profile.spacer_length,
        )
        ann = fm_index.lookup_annotation(chrom, pos)
        ann_w = ANNOTATION_RISK_WEIGHTS.get(ann, 0.15)
        cfd_burden += cfd * ann_w

        hits.append(
            OffTargetHit(
                chrom=chrom,
                position=pos,
                strand=strand,
                aligned_spacer=aln_sp,
                aligned_target=aln_tg,
                pam_seq=pam_obs,
                pam_weight=round(pam_w, 4),
                mismatches=mm_count,
                bulge_type=bulge_type,
                bulge_size=0 if bulge_type == BulgeType.NONE else 1,
                mismatch_positions=mm_pos,
                cfd_score=cfd,
                mit_score=mit,
                ddg_penalty_kcal=ddg_pen,
                annotation=ann,
            )
        )

    hits.sort(key=lambda h: (-h.cfd_score, h.mismatches, h.chrom, h.position))
    spec_score = 100.0 / (1.0 + cfd_burden)
    return round(spec_score, 2), round(cfd_burden, 5), tuple(hits[:50])


def analyze_precision_editing_window(
    candidate: ProtospacerCandidate,
    profile: NucleaseProfile,
    on_target_score: float,
) -> Optional[EditingWindowProfile]:
    """
    Compute base-editor deamination window profile or prime-editor pegRNA PBS/RTT parameters.

    For ABE8e / CBE4max:
    Models Gaussian-shaped ssDNA R-loop exposure efficiency across the editing window
    centered at position 6 (sigma = 1.45 nt) and quantifies collateral bystander risk when
    multiple targetable adenines/cytosines reside inside the window.

    For PE3:
    Optimizes Primer Binding Site (PBS) length (10-15 nt) targeting T_m ~ 34-40 C and
    evaluates Reverse Transcriptase Template (RTT) folding risk.
    """
    if profile.nuclease_class == NucleaseClass.BASE_EDITOR and profile.editing_window is not None:
        win_lo, win_hi = profile.editing_window
        target_base = profile.target_base or "A"
        sp = candidate.spacer_seq.upper()
        center = 0.5 * (win_lo + win_hi)
        sigma = 1.55
        base_scale = on_target_score / 100.0

        positions: List[int] = []
        eff_map: Dict[int, float] = {}
        for pos1 in range(max(1, win_lo - 1), min(len(sp), win_hi + 1) + 1):
            if sp[pos1 - 1] == target_base:
                positions.append(pos1)
                gauss = math.exp(-0.5 * ((pos1 - center) / sigma) ** 2)
                # CBE exhibits TC motif preference; ABE slightly prefers YA
                motif_boost = 1.0
                if pos1 >= 2:
                    prev = sp[pos1 - 2]
                    if target_base == "C" and prev == "T":
                        motif_boost = 1.15
                    elif target_base == "C" and prev == "G":
                        motif_boost = 0.65
                    elif target_base == "A" and prev in ("T", "C"):
                        motif_boost = 1.08
                eff = min(0.95, max(0.02, base_scale * gauss * motif_boost * 0.88))
                eff_map[pos1] = round(eff, 4)

        if len(positions) <= 1:
            bystander_risk = 0.0
        else:
            sorted_eff = sorted(eff_map.values(), reverse=True)
            # Probability that at least one secondary base is co-edited alongside primary base
            bystander_risk = round(1.0 - math.prod(1.0 - e for e in sorted_eff[1:]), 4)

        return EditingWindowProfile(
            editor_name=profile.name,
            target_positions=tuple(positions),
            per_position_efficiency=eff_map,
            bystander_risk=bystander_risk,
        )

    if profile.nuclease_class == NucleaseClass.PRIME_EDITOR:
        ctx = candidate.context_seq
        cut_idx = candidate.cut_index_in_context
        # PBS hybridizes to the nicked non-target strand immediately 5' of the nick (up to cut_idx)
        best_pbs_tm = 37.0
        best_delta = 999.0
        for pbs_len in range(10, 16):
            if cut_idx - pbs_len >= 0:
                pbs_seq = ctx[cut_idx - pbs_len : cut_idx]
                tm_c = compute_hybrid_melting_temp(pbs_seq)
                if abs(tm_c - 38.0) < best_delta:
                    best_delta = abs(tm_c - 38.0)
                    best_pbs_tm = tm_c

        # Evaluate optimal RTT length minimizing 3' extension hairpin interference
        best_rtt = 14
        best_mfe = -99.0
        for rtt_len in (10, 12, 14, 16):
            if cut_idx + rtt_len <= len(ctx):
                rtt_frag = reverse_complement(ctx[cut_idx : cut_idx + rtt_len])
                mfe = compute_rna_hairpin_mfe(rtt_frag)
                if mfe > best_mfe:
                    best_mfe = mfe
                    best_rtt = rtt_len

        return EditingWindowProfile(
            editor_name=profile.name,
            target_positions=tuple(range(1, min(12, best_rtt) + 1)),
            per_position_efficiency={1: round(0.55 * (on_target_score / 100.0), 4)},
            bystander_risk=0.04,
            optimal_pbs_tm_celsius=round(best_pbs_tm, 2),
            optimal_rtt_length=best_rtt,
        )

    return None
