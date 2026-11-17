"""
Genome-wide BWT/FM-index exact/mismatch search and bulge-tolerant alignment engine.

Provides:
1. Full Burrows-Wheeler Transform (BWT) construction over multi-contig reference genomes
   with LF-mapping ``C`` array and sampled ``Occ`` checkpoint matrix for O(m) pattern query.
2. Branch-and-bound k-mismatch backtracking on the FM-index with immediate PAM licensing.
3. 2-bit packed DNA Myers/XOR bit-parallel Hamming distance evaluator for fast bulk rescoring.
4. Banded Needleman-Wunsch / Myers 1-nt DNA and RNA bulge aligner for CRISPR off-target discovery.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .nucleases import evaluate_pam_weight
from .sequence import normalize_dna, reverse_complement
from .types import BulgeType, GenomicInterval, NucleaseProfile, Strand


ALPHABET = ("$", "A", "C", "G", "T")
BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode_2bit(seq: str) -> int:
    """
    Pack a DNA string (A/C/G/T) into a compact Python integer using 2 bits per nucleotide.

    Args:
        seq: DNA string containing A, C, G, T.

    Returns:
        2-bit packed integer representation.
    """
    val = 0
    for ch in seq.upper():
        val = (val << 2) | BASE_TO_INT.get(ch, 0)
    return val


def bit_parallel_hamming(code_a: int, code_b: int, length: int) -> int:
    """
    Compute Hamming mismatch count between two 2-bit packed DNA integers in O(1) bit ops.

    Args:
        code_a: 2-bit packed integer for sequence A.
        code_b: 2-bit packed integer for sequence B.
        length: Sequence length in nucleotides.

    Returns:
        Exact number of mismatched nucleotide positions.
    """
    diff = code_a ^ code_b
    # Collapse each 2-bit pair (b1, b0) -> (b1 | b0) at the even bit position
    even_bits = (diff | (diff >> 1)) & int("5" * ((length + 1) // 2 + 1), 16)
    mask = (1 << (2 * length)) - 1
    return (even_bits & mask).bit_count()


@dataclass
class _ContigSpan:
    name: str
    start: int
    end: int
    sequence: str


class GenomeFMIndex:
    """
    Compressed Burrows-Wheeler FM-index supporting exact, k-mismatch, and 1-nt bulge searches.

    Concatenates multiple contigs with '#' sentinel separators and constructs the suffix array,
    BWT string, cumulative character counts ``C``, and sampled occurrence checkpoints ``Occ``.
    """

    def __init__(
        self,
        contigs: Dict[str, str],
        checkpoint_interval: int = 32,
        intervals: Optional[List[GenomicInterval]] = None,
    ) -> None:
        if not contigs:
            raise ValueError("Reference contigs dictionary cannot be empty.")
        self.checkpoint_interval = max(4, checkpoint_interval)
        self.intervals = intervals or []
        self._spans: List[_ContigSpan] = []
        self._contig_starts: List[int] = []
        self._contigs: Dict[str, str] = {}

        joined_parts: List[str] = []
        cursor = 0
        for name, raw_seq in contigs.items():
            seq = normalize_dna(raw_seq).replace("N", "A")
            self._contigs[name] = seq
            self._spans.append(_ContigSpan(name=name, start=cursor, end=cursor + len(seq), sequence=seq))
            self._contig_starts.append(cursor)
            joined_parts.append(seq)
            joined_parts.append("#")
            cursor += len(seq) + 1

        # Replace final separator with lexicographically smallest '$' sentinel
        joined_parts[-1] = "$"
        self.text: str = "".join(joined_parts)
        self.length: int = len(self.text)
        self.total_bp: int = sum(len(s) for s in self._contigs.values())

        # Suffix array construction (doubling / prefix-doubling O(n log n) algorithm)
        self.sa: List[int] = self._build_suffix_array(self.text)
        self.bwt: str = "".join(self.text[(idx - 1) % self.length] for idx in self.sa)

        # Build C table and sampled Occ checkpoints for DNA symbols + sentinels
        self._symbols = sorted(set(self.bwt))
        counts: Dict[str, int] = {ch: 0 for ch in self._symbols}
        for ch in self.bwt:
            counts[ch] += 1

        self.c_table: Dict[str, int] = {}
        running = 0
        for ch in self._symbols:
            self.c_table[ch] = running
            running += counts[ch]

        self._occ: Dict[str, List[int]] = {ch: [] for ch in ("A", "C", "G", "T")}
        running_occ = {"A": 0, "C": 0, "G": 0, "T": 0}
        for i, ch in enumerate(self.bwt):
            if i % self.checkpoint_interval == 0:
                for base in ("A", "C", "G", "T"):
                    self._occ[base].append(running_occ[base])
            if ch in running_occ:
                running_occ[ch] += 1
        # Final sentinel checkpoint
        for base in ("A", "C", "G", "T"):
            self._occ[base].append(running_occ[base])

    @staticmethod
    def _build_suffix_array(text: str) -> List[int]:
        """Construct suffix array via O(n log n) prefix-doubling ranking."""
        n = len(text)
        rank = [ord(c) for c in text]
        sa = list(range(n))
        k = 1
        tmp = [0] * n
        while k < n:
            sa.sort(key=lambda idx: (rank[idx], rank[(idx + k) % n]))
            tmp[sa[0]] = 0
            for i in range(1, n):
                prev_i, cur_i = sa[i - 1], sa[i]
                prev_key = (rank[prev_i], rank[(prev_i + k) % n])
                cur_key = (rank[cur_i], rank[(cur_i + k) % n])
                tmp[cur_i] = tmp[prev_i] + (1 if cur_key != prev_key else 0)
            rank[:] = tmp
            if rank[sa[-1]] == n - 1:
                break
            k <<= 1
        return sa

    def occ(self, base: str, idx: int) -> int:
        """
        Return count of ``base`` in ``bwt[0:idx]`` using sampled checkpoints.

        Args:
            base: DNA symbol ('A', 'C', 'G', or 'T').
            idx: Exclusive upper index in [0, len(bwt)].

        Returns:
            Exact occurrence count.
        """
        if idx <= 0 or base not in self._occ:
            return 0
        if idx > self.length:
            idx = self.length
        block = idx // self.checkpoint_interval
        base_count = self._occ[base][block]
        start = block * self.checkpoint_interval
        for pos in range(start, idx):
            if self.bwt[pos] == base:
                base_count += 1
        return base_count

    def exact_range(self, pattern: str) -> Tuple[int, int]:
        """
        Compute suffix array interval [lo, hi) for exact occurrences of ``pattern``.

        Args:
            pattern: Uppercase DNA string.

        Returns:
            Half-open SA interval ``(lo, hi)``; empty if ``lo >= hi``.
        """
        lo, hi = 0, self.length
        for ch in reversed(pattern.upper()):
            if ch not in self.c_table:
                return (0, 0)
            c_base = self.c_table[ch]
            lo = c_base + self.occ(ch, lo)
            hi = c_base + self.occ(ch, hi)
            if lo >= hi:
                return (0, 0)
        return (lo, hi)

    def exact_locate(self, pattern: str) -> List[Tuple[str, int]]:
        """Locate all exact forward-strand occurrences ``(contig_name, 0based_pos)``."""
        lo, hi = self.exact_range(pattern)
        results: List[Tuple[str, int]] = []
        pat_len = len(pattern)
        for sa_idx in range(lo, hi):
            global_pos = self.sa[sa_idx]
            mapped = self._map_global_pos(global_pos, pat_len)
            if mapped is not None:
                results.append(mapped)
        return results

    def _map_global_pos(self, global_pos: int, span_len: int) -> Optional[Tuple[str, int]]:
        """Translate global concatenated index back to ``(contig_name, local_start)``."""
        idx = bisect.bisect_right(self._contig_starts, global_pos) - 1
        if idx < 0:
            return None
        span = self._spans[idx]
        if global_pos + span_len <= span.end:
            return (span.name, global_pos - span.start)
        return None

    def lookup_annotation(self, chrom: str, pos: int) -> str:
        """Return functional genomic annotation ('exon', 'promoter', 'intron', 'intergenic')."""
        for iv in self.intervals:
            if iv.chrom == chrom and iv.start <= pos < iv.end:
                return iv.annotation
        return "intergenic"

    def search_k_mismatches(
        self,
        pattern: str,
        max_mismatches: int = 3,
        max_hits: int = 500,
    ) -> List[Tuple[str, int, str, int]]:
        """
        Backtracking FM-index search returning all matches with <= ``max_mismatches``.

        Args:
            pattern: Uppercase query DNA sequence.
            max_mismatches: Maximum allowed Hamming mismatches.
            max_hits: Safety cap on returned SA positions.

        Returns:
            List of ``(contig, pos, matched_seq, mismatch_count)``.
        """
        pat = pattern.upper()
        m = len(pat)
        collected: Dict[int, Tuple[str, int]] = {}

        def _dfs(pos_idx: int, lo: int, hi: int, rem_mm: int, acc_chars: List[str]) -> None:
            if len(collected) >= max_hits:
                return
            if pos_idx < 0:
                matched_str = "".join(reversed(acc_chars))
                used_mm = max_mismatches - rem_mm
                for sa_i in range(lo, hi):
                    gpos = self.sa[sa_i]
                    if gpos not in collected or used_mm < collected[gpos][1]:
                        collected[gpos] = (matched_str, used_mm)
                    if len(collected) >= max_hits:
                        return
                return

            target_ch = pat[pos_idx]
            # Prioritize exact match branch first
            ordered_bases = [target_ch] + [b for b in ("A", "C", "G", "T") if b != target_ch]
            for base in ordered_bases:
                cost = 0 if base == target_ch else 1
                if cost > rem_mm or base not in self.c_table:
                    continue
                c_base = self.c_table[base]
                new_lo = c_base + self.occ(base, lo)
                new_hi = c_base + self.occ(base, hi)
                if new_lo < new_hi:
                    acc_chars.append(base)
                    _dfs(pos_idx - 1, new_lo, new_hi, rem_mm - cost, acc_chars)
                    acc_chars.pop()

        _dfs(m - 1, 0, self.length, max_mismatches, [])

        hits: List[Tuple[str, int, str, int]] = []
        for gpos, (matched_str, mm_count) in collected.items():
            mapped = self._map_global_pos(gpos, m)
            if mapped is not None:
                hits.append((mapped[0], mapped[1], matched_str, mm_count))
        hits.sort(key=lambda x: (x[3], x[0], x[1]))
        return hits

    def search_pam_offtargets(
        self,
        spacer_seq: str,
        profile: NucleaseProfile,
        max_mismatches: int = 3,
        include_bulges: bool = True,
        max_bulge_mismatches: int = 2,
        exclude_locus: Optional[Tuple[str, int, Strand]] = None,
    ) -> List[Tuple[str, int, Strand, str, str, str, BulgeType, int, Tuple[int, ...], float]]:
        """
        Enumerate genome-wide PAM-licensed off-target sites across both strands.

        Combines:
        1. FM-index k-mismatch search over the spacer with PAM-adjacent verification.
        2. Seed-anchored 1-nt DNA bulge (target longer by 1 bp) and 1-nt RNA bulge
           (target shorter by 1 bp) alignment discovery via FM-index seed anchors.

        Args:
            spacer_seq: 5'->3' guide spacer DNA sequence.
            profile: NucleaseProfile specifying PAM rules and spacer length.
            max_mismatches: Maximum mismatch count for ungapped hits.
            include_bulges: Whether to run seed-anchored 1-nt DNA/RNA bulge search.
            max_bulge_mismatches: Maximum additional mismatches alongside a 1-nt bulge.
            exclude_locus: Optional ``(chrom, start, strand)`` of the intended on-target site.

        Returns:
            List of hit tuples:
            ``(chrom, start, strand, aligned_spacer, aligned_target, pam_seq, bulge_type, mm, mm_pos, pam_weight)``.
        """
        sp = spacer_seq.upper()
        rc_sp = reverse_complement(sp)
        pam_len = len(profile.pam_pattern)
        spacer_len = len(sp)
        results: List[Tuple[str, int, Strand, str, str, str, BulgeType, int, Tuple[int, ...], float]] = []
        seen_keys: set[Tuple[str, int, str]] = set()

        def _is_excluded(chrom: str, start: int, strand: Strand) -> bool:
            if exclude_locus is None:
                return False
            return chrom == exclude_locus[0] and start == exclude_locus[1] and strand == exclude_locus[2]

        def _check_pam_and_record(
            chrom: str,
            sp_start: int,
            strand: Strand,
            raw_target_5to3: str,
            bulge_type: BulgeType,
            aln_sp: str,
            aln_tg: str,
            mm_count: int,
            mm_positions: Tuple[int, ...],
        ) -> None:
            contig = self._contigs[chrom]
            t_len = len(raw_target_5to3)
            if strand == Strand.PLUS:
                if profile.pam_side == "3prime":
                    pam_lo, pam_hi = sp_start + t_len, sp_start + t_len + pam_len
                else:
                    pam_lo, pam_hi = sp_start - pam_len, sp_start
                if pam_lo < 0 or pam_hi > len(contig):
                    return
                pam_obs = contig[pam_lo:pam_hi]
            else:
                if profile.pam_side == "3prime":
                    pam_lo, pam_hi = sp_start - pam_len, sp_start
                else:
                    pam_lo, pam_hi = sp_start + t_len, sp_start + t_len + pam_len
                if pam_lo < 0 or pam_hi > len(contig):
                    return
                pam_obs = reverse_complement(contig[pam_lo:pam_hi])

            pam_w = evaluate_pam_weight(pam_obs, profile)
            if pam_w <= 0.0:
                return
            if _is_excluded(chrom, sp_start, strand):
                return
            key = (chrom, sp_start, strand.value)
            if key in seen_keys:
                return
            seen_keys.add(key)
            results.append(
                (chrom, sp_start, strand, aln_sp, aln_tg, pam_obs, bulge_type, mm_count, mm_positions, pam_w)
            )

        # 1) Ungapped FM-index k-mismatch search on both strands
        for chrom, pos, matched_seq, _ in self.search_k_mismatches(sp, max_mismatches=max_mismatches):
            mm_pos = tuple(i + 1 for i in range(spacer_len) if sp[i] != matched_seq[i])
            _check_pam_and_record(
                chrom, pos, Strand.PLUS, matched_seq, BulgeType.NONE, sp, matched_seq, len(mm_pos), mm_pos
            )

        for chrom, pos, matched_rc, _ in self.search_k_mismatches(rc_sp, max_mismatches=max_mismatches):
            target_5to3 = reverse_complement(matched_rc)
            mm_pos = tuple(i + 1 for i in range(spacer_len) if sp[i] != target_5to3[i])
            _check_pam_and_record(
                chrom, pos, Strand.MINUS, target_5to3, BulgeType.NONE, sp, target_5to3, len(mm_pos), mm_pos
            )

        # 2) Seed-anchored 1-nt DNA/RNA bulge search
        if include_bulges and spacer_len >= 14:
            self._search_bulge_candidates(
                sp=sp,
                profile=profile,
                max_bulge_mismatches=max_bulge_mismatches,
                callback=_check_pam_and_record,
                seen_keys=seen_keys,
            )

        return results

    def _search_bulge_candidates(
        self,
        sp: str,
        profile: NucleaseProfile,
        max_bulge_mismatches: int,
        callback: object,
        seen_keys: set[Tuple[str, int, str]],
    ) -> None:
        """
        Use PAM-proximal 8-nt seed anchors in the FM-index to detect 1-nt DNA & RNA bulges.
        """
        spacer_len = len(sp)
        seed_len = 8
        # Extract PAM-proximal 8-mer anchor
        if profile.pam_side == "3prime":
            anchor_plus = sp[-seed_len:]
            anchor_offset = spacer_len - seed_len
        else:
            anchor_plus = sp[:seed_len]
            anchor_offset = 0
        anchor_minus = reverse_complement(anchor_plus)

        bulge_specs = (
            (BulgeType.DNA_BULGE, spacer_len + 1),
            (BulgeType.RNA_BULGE, spacer_len - 1),
        )

        # Search PLUS strand anchors
        for chrom, anchor_pos in self.exact_locate(anchor_plus):
            contig = self._contigs[chrom]
            for bulge_type, target_len in bulge_specs:
                if profile.pam_side == "3prime":
                    t_start = anchor_pos + seed_len - target_len
                else:
                    t_start = anchor_pos
                if t_start < 0 or t_start + target_len > len(contig):
                    continue
                if (chrom, t_start, Strand.PLUS.value) in seen_keys:
                    continue
                raw_target = contig[t_start : t_start + target_len]
                aligned = align_single_bulge(sp, raw_target, bulge_type)
                if aligned is not None:
                    aln_sp, aln_tg, mm_count, mm_pos = aligned
                    if mm_count <= max_bulge_mismatches:
                        callback(  # type: ignore[operator]
                            chrom, t_start, Strand.PLUS, raw_target, bulge_type, aln_sp, aln_tg, mm_count, mm_pos
                        )

        # Search MINUS strand anchors (anchor_offset unused directly since RC flips geometry)
        _ = anchor_offset
        for chrom, anchor_pos in self.exact_locate(anchor_minus):
            contig = self._contigs[chrom]
            for bulge_type, target_len in bulge_specs:
                if profile.pam_side == "3prime":
                    # On minus strand, 3' PAM sits upstream of forward coordinates
                    t_start = anchor_pos
                else:
                    t_start = anchor_pos + seed_len - target_len
                if t_start < 0 or t_start + target_len > len(contig):
                    continue
                if (chrom, t_start, Strand.MINUS.value) in seen_keys:
                    continue
                raw_target = reverse_complement(contig[t_start : t_start + target_len])
                aligned = align_single_bulge(sp, raw_target, bulge_type)
                if aligned is not None:
                    aln_sp, aln_tg, mm_count, mm_pos = aligned
                    if mm_count <= max_bulge_mismatches:
                        callback(  # type: ignore[operator]
                            chrom, t_start, Strand.MINUS, raw_target, bulge_type, aln_sp, aln_tg, mm_count, mm_pos
                        )


def align_single_bulge(
    spacer_seq: str,
    target_seq: str,
    bulge_type: BulgeType,
) -> Optional[Tuple[str, str, int, Tuple[int, ...]]]:
    """
    Find optimal 1-nucleotide bulge alignment between ``spacer_seq`` and ``target_seq``.

    - ``DNA_BULGE``: ``len(target_seq) == len(spacer_seq) + 1``; deletes 1 target base as '-'.
    - ``RNA_BULGE``: ``len(target_seq) == len(spacer_seq) - 1``; inserts 1 '-' gap in target.

    Avoids placing bulges at terminal boundary bases (position 1 or L) which represent shifts.

    Returns:
        Tuple ``(aligned_spacer, aligned_target, mismatch_count, 1based_mismatch_positions)``
        or None if length constraints do not match.
    """
    sp = spacer_seq.upper()
    tg = target_seq.upper()
    m = len(sp)

    best_mm = m + 1
    best_result: Optional[Tuple[str, str, int, Tuple[int, ...]]] = None

    if bulge_type == BulgeType.DNA_BULGE:
        if len(tg) != m + 1:
            return None
        for gap_idx in range(1, len(tg) - 1):
            tg_nogap = tg[:gap_idx] + tg[gap_idx + 1 :]
            mm_pos = tuple(i + 1 for i in range(m) if sp[i] != tg_nogap[i])
            if len(mm_pos) < best_mm:
                best_mm = len(mm_pos)
                aln_sp = sp[:gap_idx] + "-" + sp[gap_idx:]
                best_result = (aln_sp, tg, best_mm, mm_pos)
    elif bulge_type == BulgeType.RNA_BULGE:
        if len(tg) != m - 1:
            return None
        for gap_idx in range(1, m - 1):
            aln_tg = tg[:gap_idx] + "-" + tg[gap_idx:]
            mm_pos = tuple(i + 1 for i in range(m) if i != gap_idx and sp[i] != aln_tg[i])
            if len(mm_pos) < best_mm:
                best_mm = len(mm_pos)
                best_result = (sp, aln_tg, best_mm, mm_pos)

    return best_result


def brute_force_k_mismatch(
    contigs: Dict[str, str],
    pattern: str,
    max_mismatches: int,
) -> List[Tuple[str, int, int]]:
    """
    Reference brute-force 2-bit bit-parallel matcher used for verification and small loci.

    Returns:
        List of ``(contig, pos, mismatch_count)``.
    """
    pat = pattern.upper()
    m = len(pat)
    pat_code = encode_2bit(pat)
    mask = (1 << (2 * m)) - 1
    hits: List[Tuple[str, int, int]] = []
    for name, raw in contigs.items():
        seq = normalize_dna(raw).replace("N", "A")
        if len(seq) < m:
            continue
        rolling = encode_2bit(seq[:m])
        for i in range(len(seq) - m + 1):
            if i > 0:
                rolling = ((rolling << 2) | BASE_TO_INT[seq[i + m - 1]]) & mask
            mm = bit_parallel_hamming(rolling, pat_code, m)
            if mm <= max_mismatches:
                hits.append((name, i, mm))
    return hits
