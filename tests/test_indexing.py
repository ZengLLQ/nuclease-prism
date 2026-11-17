"""Tests for BWT/FM-index construction, k-mismatch oracle parity, 2-bit Myers distance, and bulges."""

from __future__ import annotations

import numpy as np

from nuclease_prism.indexing import (
    GenomeFMIndex,
    align_single_bulge,
    bit_parallel_hamming,
    brute_force_k_mismatch,
    encode_2bit,
)
from nuclease_prism.nucleases import get_nuclease_profile
from nuclease_prism.types import BulgeType, Strand


def test_2bit_encoding_and_bit_parallel_hamming() -> None:
    a = "ACGTACGTACGTACGTACGT"
    b = "ACGTACATACGTACGAACGT"  # 2 mismatches
    c = "TTTTTTTTTTTTTTTTTTTT"
    assert bit_parallel_hamming(encode_2bit(a), encode_2bit(a), len(a)) == 0
    assert bit_parallel_hamming(encode_2bit(a), encode_2bit(b), len(a)) == 2
    assert bit_parallel_hamming(encode_2bit(a), encode_2bit(c), len(a)) == 15


def test_fm_index_exact_and_k_mismatch_matches_brute_force_oracle() -> None:
    rng = np.random.default_rng(7)
    bases = np.array(["A", "C", "G", "T"])
    contigs = {
        "c1": "".join(rng.choice(bases, size=350)),
        "c2": "".join(rng.choice(bases, size=280)),
    }
    query = contigs["c1"][45:61]
    # Implant a 2-mismatch copy in c2
    mut = list(query)
    mut[2] = "A" if mut[2] != "A" else "G"
    mut[10] = "T" if mut[10] != "T" else "C"
    contigs["c2"] = contigs["c2"][:100] + "".join(mut) + contigs["c2"][116:]

    fm = GenomeFMIndex(contigs, checkpoint_interval=8)
    exact_hits = fm.exact_locate(query)
    assert ("c1", 45) in exact_hits

    for k in (0, 1, 2, 3):
        fm_hits = {(c, p, mm) for c, p, _, mm in fm.search_k_mismatches(query, max_mismatches=k)}
        bf_hits = set(brute_force_k_mismatch(contigs, query, max_mismatches=k))
        assert fm_hits == bf_hits


def test_single_bulge_alignment_dna_and_rna() -> None:
    spacer = "GCTAGCTAGCATCGATCGAT"
    # DNA bulge: extra 'A' inserted in target at index 9
    target_dna_bulge = spacer[:9] + "A" + spacer[9:]
    res_dna = align_single_bulge(spacer, target_dna_bulge, BulgeType.DNA_BULGE)
    assert res_dna is not None
    aln_sp, aln_tg, mm_c, mm_pos = res_dna
    assert mm_c == 0
    assert "-" in aln_sp
    assert len(aln_sp) == len(aln_tg) == 21
    assert mm_pos == ()

    # RNA bulge: 1 base deleted from target at index 10
    target_rna_bulge = spacer[:10] + spacer[11:]
    res_rna = align_single_bulge(spacer, target_rna_bulge, BulgeType.RNA_BULGE)
    assert res_rna is not None
    aln_sp2, aln_tg2, mm_c2, _ = res_rna
    assert mm_c2 == 0
    assert "-" in aln_tg2
    assert len(aln_sp2) == len(aln_tg2) == 20


def test_fm_index_pam_offtarget_detects_mismatches_and_bulges() -> None:
    sp = get_nuclease_profile("SpCas9")
    spacer = "ACGTACGTACGTATCGATCG"
    # On-target on chr1, 1-mm on chr2 (plus), 1-nt DNA bulge on chr2 (minus/plus)
    on_site = spacer + "AGG"
    mm1_site = "TCGTACGTACGTATCGATCG" + "TGG"
    dna_bulge_site = spacer[:8] + "C" + spacer[8:] + "CGG"

    contigs = {
        "chr1": "AAAAAAAAAAAA" + on_site + "TTTTTTTTTTTT",
        "chr2": "CCCCCCCCCC" + mm1_site + "GGGGGGGGGG" + dna_bulge_site + "AAAAAAAAAA",
    }
    fm = GenomeFMIndex(contigs)
    hits = fm.search_pam_offtargets(
        spacer_seq=spacer,
        profile=sp,
        max_mismatches=2,
        include_bulges=True,
        exclude_locus=("chr1", 12, Strand.PLUS),
    )
    chrom_pos_types = {(h[0], h[1], h[6]) for h in hits}
    assert ("chr1", 12, BulgeType.NONE) not in chrom_pos_types
    assert ("chr2", 10, BulgeType.NONE) in chrom_pos_types
    assert any(h[6] == BulgeType.DNA_BULGE for h in hits)
