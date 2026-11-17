"""Unit tests for nuclease profiles, IUPAC PAM rules, sequence folding, and protospacer scanning."""

from __future__ import annotations

import pytest

from nuclease_prism.nucleases import (
    evaluate_pam_weight,
    get_nuclease_profile,
    list_supported_nucleases,
    map_spacer_to_pam_distance,
    matches_iupac,
    pam_zipping_order,
)
from nuclease_prism.sequence import (
    compute_rna_hairpin_mfe,
    find_microhomologies,
    gc_content,
    normalize_dna,
    parse_fasta_text,
    pol3_terminator_penalty,
    reverse_complement,
    scan_protospacers,
)
from nuclease_prism.types import Strand


def test_nuclease_registry_coverage_and_lookup() -> None:
    names = list_supported_nucleases()
    assert {"SpCas9", "SaCas9", "AsCas12a", "RfxCas13d", "ABE8e", "CBE4max", "PE3"} <= set(names)
    sp = get_nuclease_profile("spcas9")
    assert sp.name == "SpCas9"
    assert sp.spacer_length == 20
    assert sp.pam_side == "3prime"
    with pytest.raises(KeyError):
        get_nuclease_profile("NonExistentCas99")


def test_iupac_and_pam_licensing() -> None:
    assert matches_iupac("AGG", "NGG")
    assert matches_iupac("CGG", "NGG")
    assert not matches_iupac("AGA", "NGG")
    assert matches_iupac("TTTA", "TTTV")
    assert not matches_iupac("TTTT", "TTTV")
    assert matches_iupac("AAGGAT", "NNGRRT")

    sp = get_nuclease_profile("SpCas9")
    assert evaluate_pam_weight("TGG", sp) == 1.0
    assert 0.2 < evaluate_pam_weight("CAG", sp) < 0.3
    assert evaluate_pam_weight("TTT", sp) == 0.0


def test_pam_distance_and_zipping_order() -> None:
    sp = get_nuclease_profile("SpCas9")
    cas12a = get_nuclease_profile("AsCas12a")
    assert map_spacer_to_pam_distance(20, sp) == 1
    assert map_spacer_to_pam_distance(1, sp) == 20
    assert map_spacer_to_pam_distance(1, cas12a) == 1
    assert map_spacer_to_pam_distance(23, cas12a) == 23

    assert pam_zipping_order(sp)[0] == 19
    assert pam_zipping_order(sp)[-1] == 0
    assert pam_zipping_order(cas12a)[0] == 0
    assert pam_zipping_order(cas12a)[-1] == 22


def test_sequence_normalization_and_revcomp() -> None:
    assert normalize_dna(" aug cgt \n") == "ATGCGT"
    assert reverse_complement("ATGC") == "GCAT"
    assert gc_content("GGCCAA") == pytest.approx(4 / 6)
    with pytest.raises(ValueError):
        normalize_dna("ATGX")
    with pytest.raises(ValueError):
        normalize_dna("   ")


def test_pol3_terminator_and_hairpin_mfe() -> None:
    assert pol3_terminator_penalty("ACGTACGTACGTACGTACGT") == 0.0
    assert pol3_terminator_penalty("ACGTACGTACGTACGTTTTT") == 1.0
    assert pol3_terminator_penalty("ACGTACGTACGTACGTATTT") == 0.45

    # Strong self-complementary stem should yield a negative MFE
    palindromic = "GGGGCCCCAAAAGGGGCCCC"
    flat = "ACAAACAAACAAACAAACAA"
    mfe_pal = compute_rna_hairpin_mfe(palindromic)
    mfe_flat = compute_rna_hairpin_mfe(flat)
    assert mfe_pal <= mfe_flat <= 0.0


def test_microhomology_detection_and_fasta_parse(microhomology_context: tuple[str, int]) -> None:
    seq, cut_idx = microhomology_context
    hits = find_microhomologies(seq, cut_idx, min_len=3, max_len=8, max_deletion=20)
    motifs = [h[2] for h in hits]
    assert "GTCAG" in motifs

    fasta = ">chr1 desc\nATCGatcg\n>chr2\nGGGGCCCC\n"
    parsed = parse_fasta_text(fasta)
    assert parsed == {"chr1": "ATCGATCG", "chr2": "GGGGCCCC"}


def test_scan_protospacers_both_strands() -> None:
    sp = get_nuclease_profile("SpCas9")
    cas12a = get_nuclease_profile("AsCas12a")
    # Forward NGG at pos 30..33, plus reverse NGG (CCN) at pos 15..18
    dna = "AAAAAAAAAA" + "CCAGCATCGATCGATCGATCGAT" + "GCTAGCTAGCTAGCTAGCTA" + "AGG" + "TTTTTTTTTT"
    cands_sp = scan_protospacers(dna, sp, chrom="test_chr")
    strands = {c.strand for c in cands_sp}
    assert Strand.PLUS in strands
    for c in cands_sp:
        assert len(c.spacer_seq) == 20
        assert c.pam_weight == 1.0
        assert 0 < c.cut_index_in_context < len(c.context_seq)

    # Test Cas12a TTTV scanning
    dna_12a = "AAAAAA" + "TTTC" + "ACGTACGTACGTACGTACGTACG" + "GGGGGG"
    cands_12a = scan_protospacers(dna_12a, cas12a, chrom="test_chr")
    assert len(cands_12a) >= 1
    assert cands_12a[0].pam_seq == "TTTC"
    assert len(cands_12a[0].spacer_seq) == 23


@pytest.mark.parametrize(
    "nuclease_name,expected_len",
    [
        ("SpCas9", 20),
        ("SaCas9", 21),
        ("AsCas12a", 23),
        ("RfxCas13d", 23),
        ("ABE8e", 20),
        ("CBE4max", 20),
        ("PE3", 20),
    ],
)
def test_all_nuclease_spacer_lengths_and_zipping_bijection(nuclease_name: str, expected_len: int) -> None:
    prof = get_nuclease_profile(nuclease_name)
    assert prof.spacer_length == expected_len
    order = pam_zipping_order(prof)
    assert sorted(order) == list(range(expected_len))


@pytest.mark.parametrize(
    "pam,expected_positive",
    [
        ("AGG", True),
        ("TGG", True),
        ("CGG", True),
        ("GGG", True),
        ("NAG", False),
        ("AAA", False),
        ("CC", False),
    ],
)
def test_spcas9_pam_iupac_matrix(pam: str, expected_positive: bool) -> None:
    assert matches_iupac(pam, "NGG") is expected_positive
