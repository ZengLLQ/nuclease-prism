"""
DNA/RNA sequence mechanics, sgRNA hairpin MFE folding, microhomology detection, and FASTA parsing.

Implements:
- Strict DNA validation and reverse-complement transforms.
- Pol-III U6 terminator (TTTT) and homopolymer run risk scoring.
- Nussinov/Turner-inspired Zuker dynamic-programming RNA minimum free energy (MFE) folding
  including Watson-Crick + G:U wobble pairing and hairpin loop initiation penalties, used to
  quantify self-complementary sgRNA spacer-scaffold interference.
- Direct-repeat microhomology scanning around double-strand break sites.
- Protospacer extraction across both forward and reverse strands for any NucleaseProfile.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .nucleases import evaluate_pam_weight
from .types import GenomicInterval, NucleaseProfile, ProtospacerCandidate, Strand


COMPLEMENT_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")

# Standard SpCas9 sgRNA 5' handle + first stem-loop scaffold fragment (used for self-folding interference)
DEFAULT_TRACR_SEED = "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCG"

# Approximate RNA base-pair stacking bonuses (kcal/mol at 37 C) for DP folding
_RNA_PAIR_STACK_ENERGY: Dict[Tuple[str, str], float] = {
    ("G", "C"): -2.4,
    ("C", "G"): -2.4,
    ("A", "U"): -1.3,
    ("U", "A"): -1.3,
    ("G", "U"): -0.8,
    ("U", "G"): -0.8,
}

# Loop initiation destabilization penalty deltaG_loop(len) in kcal/mol (Turner-style logarithmic)
_MIN_HAIRPIN_LOOP = 3


def normalize_dna(seq: str) -> str:
    """
    Normalize a DNA/RNA sequence string to uppercase DNA ('U' -> 'T') and validate alphabet.

    Args:
        seq: Input nucleotide sequence.

    Returns:
        Uppercase DNA string containing only A, C, G, T, N.

    Raises:
        ValueError: If sequence is empty or contains non-nucleotide characters.
    """
    cleaned = "".join(seq.split()).upper().replace("U", "T")
    if not cleaned:
        raise ValueError("Sequence cannot be empty.")
    invalid = set(cleaned) - {"A", "C", "G", "T", "N"}
    if invalid:
        raise ValueError(f"Invalid nucleotide symbols encountered: {sorted(invalid)}")
    return cleaned


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(COMPLEMENT_TABLE)[::-1]


def dna_to_rna(seq: str) -> str:
    """Convert a DNA sequence ('T') to its RNA representation ('U')."""
    return seq.upper().replace("T", "U")


def gc_content(seq: str) -> float:
    """Compute fraction of G/C bases in ``seq``."""
    if not seq:
        return 0.0
    upper = seq.upper()
    gc = upper.count("G") + upper.count("C")
    return gc / len(upper)


def sliding_gc_extremes(seq: str, window: int = 5) -> Tuple[float, float]:
    """
    Compute minimum and maximum GC fraction across sliding sub-windows.

    Local high-GC clamps or AT-rich breathing tracts strongly modulate R-loop kinetics.
    """
    if len(seq) < window:
        val = gc_content(seq)
        return val, val
    vals = [gc_content(seq[i : i + window]) for i in range(len(seq) - window + 1)]
    return min(vals), max(vals)


def pol3_terminator_penalty(spacer_seq: str) -> float:
    """
    Compute Pol-III (U6/H1 promoter) premature transcriptional termination risk in [0, 1].

    RNA Polymerase III terminates efficiently at >= 4 consecutive thymidines ('TTTT'),
    and shows partial attenuation at 'TTT' near the 3' end of the spacer.
    """
    upper = spacer_seq.upper()
    if "TTTTT" in upper:
        return 1.0
    if "TTTT" in upper:
        return 0.85
    if upper.endswith("TTT"):
        return 0.45
    if "TTT" in upper:
        return 0.20
    return 0.0


def max_homopolymer_run(seq: str) -> int:
    """Return the length of the longest single-base homopolymer run in ``seq``."""
    if not seq:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best


def compute_rna_hairpin_mfe(spacer_dna: str, scaffold_dna: str = DEFAULT_TRACR_SEED) -> float:
    """
    Estimate minimum free energy (MFE, kcal/mol) of sgRNA spacer + scaffold self-folding.

    Uses a dynamic-programming Nussinov/Turner hybrid energy recursion over the concatenated
    RNA sequence ``rna_spacer + rna_scaffold``. More negative values indicate stable hairpins
    that sequester the spacer and impair Cas ribonucleoprotein target DNA hybridization.

    Args:
        spacer_dna: 5'->3' spacer DNA sequence.
        scaffold_dna: 5'->3' tracrRNA repeat scaffold sequence (optional).

    Returns:
        Estimated MFE in kcal/mol (<= 0.0).
    """
    rna = dna_to_rna(spacer_dna + scaffold_dna[:24])
    n = len(rna)
    if n < _MIN_HAIRPIN_LOOP + 2:
        return 0.0

    # DP table dp[i][j] stores minimum free energy on subsequence rna[i..j]
    dp: List[List[float]] = [[0.0] * n for _ in range(n)]

    for span in range(_MIN_HAIRPIN_LOOP + 1, n):
        for i in range(n - span):
            j = i + span
            best = min(dp[i + 1][j], dp[i][j - 1])

            # Check bifurcation
            for k in range(i + 1, j):
                split_val = dp[i][k] + dp[k + 1][j]
                if split_val < best:
                    best = split_val

            pair_e = _RNA_PAIR_STACK_ENERGY.get((rna[i], rna[j]))
            if pair_e is not None:
                inner_e = dp[i + 1][j - 1]
                # Penalize hairpin initiation if interior has zero base pairs
                if inner_e == 0.0:
                    loop_len = j - i - 1
                    loop_pen = 4.1 + 0.65 * math.log(max(1, loop_len))
                    cand = pair_e + loop_pen
                else:
                    # Coaxial / stacked stem stabilization bonus
                    cand = inner_e + pair_e * 0.85
                if cand < best:
                    best = cand

            dp[i][j] = min(0.0, best)

    # Subtract baseline scaffold-only fold energy so we isolate spacer-induced interference
    return round(dp[0][n - 1] - (-2.8), 3) if dp[0][n - 1] < -2.8 else 0.0


def find_microhomologies(
    context_seq: str,
    cut_index: int,
    min_len: int = 2,
    max_len: int = 8,
    max_deletion: int = 25,
) -> List[Tuple[int, int, str, int]]:
    """
    Detect flanking direct microhomology repeats spanning a double-strand break cut site.

    In MMEJ (microhomology-mediated end joining), 5'->3' resection exposes single-stranded
    overhangs on both sides of ``cut_index``; annealing of an identical motif of length L
    at left offset ``l_start`` and right offset ``r_start`` produces a deletion of size
    ``r_start - l_start`` removing one repeat copy and the intervening sequence.

    Args:
        context_seq: Local DNA context surrounding the DSB.
        cut_index: 0-based cleavage index separating left half [0:cut_index] and right [cut_index:].
        min_len: Minimum microhomology length in bp (default 2).
        max_len: Maximum microhomology length in bp (default 8).
        max_deletion: Maximum deletion distance scanned in bp.

    Returns:
        List of tuples ``(deletion_size, microhomology_len, repeat_seq, left_start_idx)``.
    """
    seq = context_seq.upper()
    n = len(seq)
    hits: List[Tuple[int, int, str, int]] = []
    seen_deletions: Dict[int, Tuple[int, int, str, int]] = {}

    left_limit = max(0, cut_index - max_deletion)
    right_limit = min(n, cut_index + max_deletion)

    for length in range(max_len, min_len - 1, -1):
        for l_start in range(left_limit, cut_index - length + 1):
            motif = seq[l_start : l_start + length]
            if "N" in motif:
                continue
            # Right repeat must sit at or past the cut site
            for r_start in range(max(cut_index, l_start + length), right_limit - length + 1):
                if seq[r_start : r_start + length] == motif:
                    del_size = r_start - l_start
                    if 1 <= del_size <= max_deletion:
                        # Keep the longest microhomology for each distinct deletion size
                        if del_size not in seen_deletions or length > seen_deletions[del_size][1]:
                            seen_deletions[del_size] = (del_size, length, motif, l_start)

    hits = sorted(seen_deletions.values(), key=lambda x: (-x[1], x[0]))
    return hits


def parse_fasta_text(fasta_text: str) -> Dict[str, str]:
    """
    Parse a multi-record FASTA text string into normalized uppercase DNA sequences.

    Args:
        fasta_text: Raw FASTA file content.

    Returns:
        Dictionary mapping contig header names to normalized DNA strings.
    """
    records: Dict[str, List[str]] = {}
    current_name: Optional[str] = None
    for raw_line in fasta_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            current_name = line[1:].strip().split()[0] or "contig_1"
            records.setdefault(current_name, [])
        else:
            if current_name is None:
                current_name = "contig_1"
                records.setdefault(current_name, [])
            records[current_name].append(line)
    return {name: normalize_dna("".join(chunks)) for name, chunks in records.items()}


def scan_protospacers(
    sequence: str,
    profile: NucleaseProfile,
    chrom: str = "chr1",
    intervals: Optional[List[GenomicInterval]] = None,
    allow_non_canonical_pam: bool = False,
    context_flank: int = 22,
) -> List[ProtospacerCandidate]:
    """
    Scan both strands of ``sequence`` for valid PAM-licensed protospacer candidates.

    Args:
        sequence: Target genomic DNA sequence (5'->3' on plus strand).
        profile: NucleaseProfile defining PAM motif, spacer length, and cut offset.
        chrom: Chromosome or contig name.
        intervals: Optional list of annotated GenomicIntervals for accessibility & exon tagging.
        allow_non_canonical_pam: If True, also retain non-canonical PAMs with weight > 0.
        context_flank: Flanking bp included on each side of the cut site for repair modeling.

    Returns:
        List of extracted ProtospacerCandidate objects ordered by genomic coordinate.
    """
    dna = normalize_dna(sequence)
    n = len(dna)
    pam_len = len(profile.pam_pattern)
    spacer_len = profile.spacer_length
    candidates: List[ProtospacerCandidate] = []

    def _lookup_annotation(pos: int) -> Tuple[str, float]:
        if not intervals:
            return ("exon", 1.0)
        for iv in intervals:
            if iv.chrom == chrom and iv.start <= pos < iv.end:
                return (iv.annotation, iv.accessibility)
        return ("intergenic", 0.45)

    def _extract_context(cut_pos: int, strand: Strand) -> Tuple[str, int]:
        lo = max(0, cut_pos - context_flank)
        hi = min(n, cut_pos + context_flank)
        raw_ctx = dna[lo:hi]
        if strand == Strand.PLUS:
            return raw_ctx, cut_pos - lo
        rc_ctx = reverse_complement(raw_ctx)
        # In reverse complement, cut_pos - lo flips to len(raw_ctx) - (cut_pos - lo)
        return rc_ctx, len(raw_ctx) - (cut_pos - lo)

    # Scan PLUS strand
    for i in range(n - spacer_len - pam_len + 1):
        if profile.pam_side == "3prime":
            spacer = dna[i : i + spacer_len]
            pam = dna[i + spacer_len : i + spacer_len + pam_len]
            sp_start, sp_end = i, i + spacer_len
            cut_pos = i + spacer_len + profile.cut_offset_from_pam
        else:
            pam = dna[i : i + pam_len]
            spacer = dna[i + pam_len : i + pam_len + spacer_len]
            sp_start, sp_end = i + pam_len, i + pam_len + spacer_len
            cut_pos = i + pam_len + profile.cut_offset_from_pam

        if "N" in spacer or "N" in pam:
            continue
        weight = evaluate_pam_weight(pam, profile)
        if weight == 0.0 or (not allow_non_canonical_pam and weight < 1.0):
            continue
        if cut_pos <= 2 or cut_pos >= n - 2:
            continue

        ctx_seq, ctx_cut = _extract_context(cut_pos, Strand.PLUS)
        ann, acc = _lookup_annotation((sp_start + sp_end) // 2)
        guide_id = f"{profile.name}_{chrom}_{sp_start}_{sp_end}_+"
        candidates.append(
            ProtospacerCandidate(
                guide_id=guide_id,
                nuclease_name=profile.name,
                chrom=chrom,
                start=sp_start,
                end=sp_end,
                strand=Strand.PLUS,
                spacer_seq=spacer,
                pam_seq=pam,
                pam_weight=weight,
                context_seq=ctx_seq,
                cut_index_in_context=ctx_cut,
                gc_content=round(gc_content(spacer), 4),
                pol3_terminator_risk=round(pol3_terminator_penalty(spacer), 4),
                hairpin_mfe_kcal=compute_rna_hairpin_mfe(spacer),
                accessibility=acc,
                annotation=ann,
            )
        )

    # Scan MINUS strand by evaluating reverse complement coordinates
    rc_dna = reverse_complement(dna)
    for j in range(n - spacer_len - pam_len + 1):
        if profile.pam_side == "3prime":
            spacer = rc_dna[j : j + spacer_len]
            pam = rc_dna[j + spacer_len : j + spacer_len + pam_len]
            rc_sp_start, rc_sp_end = j, j + spacer_len
            rc_cut_pos = j + spacer_len + profile.cut_offset_from_pam
        else:
            pam = rc_dna[j : j + pam_len]
            spacer = rc_dna[j + pam_len : j + pam_len + spacer_len]
            rc_sp_start, rc_sp_end = j + pam_len, j + pam_len + spacer_len
            rc_cut_pos = j + pam_len + profile.cut_offset_from_pam

        if "N" in spacer or "N" in pam:
            continue
        weight = evaluate_pam_weight(pam, profile)
        if weight == 0.0 or (not allow_non_canonical_pam and weight < 1.0):
            continue

        fwd_sp_start = n - rc_sp_end
        fwd_sp_end = n - rc_sp_start
        fwd_cut_pos = n - rc_cut_pos
        if fwd_cut_pos <= 2 or fwd_cut_pos >= n - 2:
            continue

        ctx_seq, ctx_cut = _extract_context(fwd_cut_pos, Strand.MINUS)
        ann, acc = _lookup_annotation((fwd_sp_start + fwd_sp_end) // 2)
        guide_id = f"{profile.name}_{chrom}_{fwd_sp_start}_{fwd_sp_end}_-"
        candidates.append(
            ProtospacerCandidate(
                guide_id=guide_id,
                nuclease_name=profile.name,
                chrom=chrom,
                start=fwd_sp_start,
                end=fwd_sp_end,
                strand=Strand.MINUS,
                spacer_seq=spacer,
                pam_seq=pam,
                pam_weight=weight,
                context_seq=ctx_seq,
                cut_index_in_context=ctx_cut,
                gc_content=round(gc_content(spacer), 4),
                pol3_terminator_risk=round(pol3_terminator_penalty(spacer), 4),
                hairpin_mfe_kcal=compute_rna_hairpin_mfe(spacer),
                accessibility=acc,
                annotation=ann,
            )
        )

    candidates.sort(key=lambda c: (c.start, c.strand.value))
    return candidates
