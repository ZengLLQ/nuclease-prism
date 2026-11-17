"""
Allele-specific SNV/haplotype CRISPR discrimination and multiplex guide set-cover solver.

Models:
1. Personal variant (SNV / multinucleotide substitution) impact on PAM creation/destruction
   and R-loop kinetic cleavage selectivity between REF and ALT haplotypes:
       selectivity_ratio = P_cleave(ALT) / max(1e-4, P_cleave(REF))
   Enabling allele-specific knockout of dominant gain-of-function disease mutations
   (e.g., Huntington HTT, RHO P23H, KRAS G12D) while sparing the wild-type allele.
2. Greedy weighted set-cover optimizer for selecting a minimal multiplex guide cocktail
   covering a panel of target isoforms, viral strains, or exon splice variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from .nucleases import evaluate_pam_weight, get_nuclease_profile
from .sequence import normalize_dna, reverse_complement, scan_protospacers
from .thermodynamics import KineticParameters, analyze_rloop_thermodynamics
from .types import NucleaseProfile, ProtospacerCandidate, Strand


@dataclass(frozen=True)
class GenomicVariant:
    """
    Single-nucleotide or short substitution variant on a reference locus.

    Attributes:
        chrom: Contig/chromosome identifier.
        pos: 0-based reference coordinate.
        ref: Reference nucleotide(s).
        alt: Alternate pathogenic/target nucleotide(s).
        variant_id: Identifier (e.g., 'rs12345' or 'KRAS_G12D').
    """

    chrom: str
    pos: int
    ref: str
    alt: str
    variant_id: str = "SNV_1"

    def __post_init__(self) -> None:
        if len(self.ref) != len(self.alt) or not self.ref:
            raise ValueError("GenomicVariant currently supports equal-length SNV/MNV substitutions.")


@dataclass(frozen=True)
class AlleleSpecificGuide:
    """
    Evaluation of a candidate guide's ability to discriminate ALT vs REF haplotypes.

    Attributes:
        guide_id: Candidate identifier on ALT haplotype.
        nuclease_name: Nuclease profile name.
        variant_id: Target GenomicVariant identifier.
        strand: Target genomic strand.
        spacer_seq: 5'->3' ALT-targeting spacer sequence.
        pam_alt: PAM sequence on the ALT allele.
        pam_ref: Corresponding sequence at the PAM coordinates on the REF allele.
        mechanism: 'PAM_CREATED', 'SEED_MISMATCH', or 'DISTAL_MISMATCH'.
        p_cleave_alt: Kinetic cleavage probability on ALT haplotype.
        p_cleave_ref: Kinetic cleavage probability on REF haplotype (0.0 if PAM destroyed).
        selectivity_ratio: Ratio P_cleave(ALT) / max(1e-4, P_cleave(REF)).
        discrimination_score: Normalized [0, 100] allele-discrimination score.
    """

    guide_id: str
    nuclease_name: str
    variant_id: str
    strand: Strand
    spacer_seq: str
    pam_alt: str
    pam_ref: str
    mechanism: str
    p_cleave_alt: float
    p_cleave_ref: float
    selectivity_ratio: float
    discrimination_score: float


def apply_variants_to_sequence(
    ref_seq: str,
    variants: Sequence[GenomicVariant],
) -> str:
    """
    Construct the alternate haplotype DNA sequence by applying equal-length variants.
    """
    seq_chars = list(normalize_dna(ref_seq))
    for var in variants:
        r_upper = var.ref.upper()
        a_upper = var.alt.upper()
        end = var.pos + len(r_upper)
        if var.pos < 0 or end > len(seq_chars):
            raise ValueError(f"Variant {var.variant_id} at {var.pos} out of bounds for len={len(seq_chars)}")
        observed = "".join(seq_chars[var.pos : end])
        if observed != r_upper:
            raise ValueError(f"Variant {var.variant_id} expected ref {r_upper!r}, observed {observed!r}")
        seq_chars[var.pos : end] = list(a_upper)
    return "".join(seq_chars)


def _extract_pam_on_other_allele(
    other_seq: str,
    cand: ProtospacerCandidate,
    profile: NucleaseProfile,
) -> str:
    """Extract the homologous PAM bases from ``other_seq`` at ``cand``'s PAM coordinates."""
    pam_len = len(profile.pam_pattern)
    if cand.strand == Strand.PLUS:
        if profile.pam_side == "3prime":
            return other_seq[cand.end : cand.end + pam_len]
        return other_seq[cand.start - pam_len : cand.start]
    if profile.pam_side == "3prime":
        return reverse_complement(other_seq[cand.start - pam_len : cand.start])
    return reverse_complement(other_seq[cand.end : cand.end + pam_len])


def _extract_spacer_on_other_allele(
    other_seq: str,
    cand: ProtospacerCandidate,
) -> str:
    """Extract the 5'->3' homologous target sense sequence from ``other_seq``."""
    raw = other_seq[cand.start : cand.end]
    return raw if cand.strand == Strand.PLUS else reverse_complement(raw)


def score_allele_specific_guides(
    ref_seq: str,
    variant: GenomicVariant,
    nuclease_name: str = "SpCas9",
    params: KineticParameters | None = None,
) -> List[AlleleSpecificGuide]:
    """
    Identify and rank ALT-selective CRISPR guides around a pathogenic SNV/MNV.

    Evaluates guides overlapping ``variant`` either within their PAM (PAM creation)
    or within their protospacer (seed or distal mismatch against REF), computing exact
    R-loop kinetic cleavage ratios between ALT and REF haplotypes.

    Args:
        ref_seq: Reference DNA sequence containing ``variant.ref`` at ``variant.pos``.
        variant: Target GenomicVariant.
        nuclease_name: Nuclease profile name.
        params: Optional KineticParameters.

    Returns:
        List of AlleleSpecificGuide records sorted descending by discrimination_score.
    """
    profile = get_nuclease_profile(nuclease_name)
    ref_norm = normalize_dna(ref_seq)
    alt_norm = apply_variants_to_sequence(ref_norm, [variant])

    alt_candidates = scan_protospacers(alt_norm, profile, chrom=variant.chrom)
    pam_len = len(profile.pam_pattern)
    var_lo = variant.pos
    var_hi = variant.pos + len(variant.ref)
    seed_lo, seed_hi = profile.seed_range

    results: List[AlleleSpecificGuide] = []
    for cand in alt_candidates:
        # Compute full footprint [fp_lo, fp_hi) covering spacer + PAM
        if cand.strand == Strand.PLUS:
            fp_lo = cand.start if profile.pam_side == "3prime" else cand.start - pam_len
            fp_hi = cand.end + pam_len if profile.pam_side == "3prime" else cand.end
        else:
            fp_lo = cand.start - pam_len if profile.pam_side == "3prime" else cand.start
            fp_hi = cand.end if profile.pam_side == "3prime" else cand.end + pam_len

        # Skip guides that do not overlap the variant at all
        if var_hi <= fp_lo or var_lo >= fp_hi:
            continue

        pam_ref = _extract_pam_on_other_allele(ref_norm, cand, profile)
        pam_ref_w = evaluate_pam_weight(pam_ref, profile)
        ref_target_sp = _extract_spacer_on_other_allele(ref_norm, cand)

        thermo_alt = analyze_rloop_thermodynamics(cand.spacer_seq, profile, params=params)
        p_alt = thermo_alt.kinetic_cleavage_prob

        if pam_ref_w == 0.0:
            mechanism = "PAM_CREATED"
            p_ref = 0.0
        else:
            thermo_ref = analyze_rloop_thermodynamics(
                cand.spacer_seq, profile, target_sense_seq=ref_target_sp, params=params
            )
            p_ref = pam_ref_w * thermo_ref.kinetic_cleavage_prob
            mm_positions = [i + 1 for i in range(len(cand.spacer_seq)) if cand.spacer_seq[i] != ref_target_sp[i]]
            if any(seed_lo <= p <= seed_hi for p in mm_positions):
                mechanism = "SEED_MISMATCH"
            else:
                mechanism = "DISTAL_MISMATCH"

        ratio = p_alt / max(1e-4, p_ref)
        disc_score = 100.0 * p_alt * (1.0 - min(1.0, p_ref / max(1e-6, p_alt)))
        results.append(
            AlleleSpecificGuide(
                guide_id=cand.guide_id,
                nuclease_name=profile.name,
                variant_id=variant.variant_id,
                strand=cand.strand,
                spacer_seq=cand.spacer_seq,
                pam_alt=cand.pam_seq,
                pam_ref=pam_ref,
                mechanism=mechanism,
                p_cleave_alt=round(p_alt, 5),
                p_cleave_ref=round(p_ref, 5),
                selectivity_ratio=round(ratio, 3),
                discrimination_score=round(max(0.0, min(100.0, disc_score)), 2),
            )
        )

    results.sort(key=lambda r: (-r.discrimination_score, -r.selectivity_ratio, r.guide_id))
    return results


def select_multiplex_guide_cover(
    target_sequences: Dict[str, str],
    nuclease_name: str = "SpCas9",
    max_guides: int = 5,
) -> Tuple[List[Tuple[str, Set[str]]], float]:
    """
    Select a minimal multiplex guide cocktail covering a panel of target sequences/isoforms.

    Solves the unweighted/weighted greedy Set Cover problem over exact PAM-licensed
    protospacers across all entries in ``target_sequences``.

    Args:
        target_sequences: Mapping from isoform/strain name to DNA sequence.
        nuclease_name: Nuclease profile name.
        max_guides: Maximum number of guides in the multiplex cocktail.

    Returns:
        Tuple ``(selected_list_of_(spacer_seq, covered_target_names), coverage_fraction)``.
    """
    if not target_sequences:
        return [], 0.0
    profile = get_nuclease_profile(nuclease_name)

    spacer_to_targets: Dict[str, Set[str]] = {}
    for target_id, raw_seq in target_sequences.items():
        for cand in scan_protospacers(raw_seq, profile, chrom=target_id):
            spacer_to_targets.setdefault(cand.spacer_seq, set()).add(target_id)

    uncovered: Set[str] = set(target_sequences.keys())
    selected: List[Tuple[str, Set[str]]] = []

    while uncovered and len(selected) < max_guides:
        best_spacer: str | None = None
        best_gain: Set[str] = set()
        for sp, cov in spacer_to_targets.items():
            gain = cov & uncovered
            if len(gain) > len(best_gain) or (
                len(gain) == len(best_gain) and best_spacer is not None and sp < best_spacer
            ):
                best_spacer = sp
                best_gain = gain
        if not best_spacer or not best_gain:
            break
        selected.append((best_spacer, set(spacer_to_targets[best_spacer])))
        uncovered -= best_gain

    total_n = len(target_sequences)
    cov_frac = (total_n - len(uncovered)) / total_n
    return selected, round(cov_frac, 4)
