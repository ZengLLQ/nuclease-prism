"""Unit and integration tests for allele-specific SNV discrimination and multiplex set cover."""

from __future__ import annotations

import pytest

from nuclease_prism.haplotype import (
    GenomicVariant,
    apply_variants_to_sequence,
    score_allele_specific_guides,
    select_multiplex_guide_cover,
)


def test_apply_variant_validation() -> None:
    ref = "AAAAAAAAAA" + "GCTAGCTAGCATCGATCGAT" + "AGA" + "TTTTTTTTTT"
    var = GenomicVariant(chrom="chr1", pos=32, ref="A", alt="G", variant_id="PAM_CREATE")
    alt = apply_variants_to_sequence(ref, [var])
    assert alt[30:33] == "AGG"

    with pytest.raises(ValueError):
        apply_variants_to_sequence(ref, [GenomicVariant(chrom="chr1", pos=32, ref="C", alt="G")])
    with pytest.raises(ValueError):
        GenomicVariant(chrom="chr1", pos=0, ref="AA", alt="G")


def test_pam_creating_snv_achieves_complete_ref_sparing() -> None:
    # REF has 'AGC' (unlicensed by SpCas9); SNV at pos 32 C->G creates canonical 'AGG' PAM
    ref = "CCCCCCCCCC" + "GCTAGCTAGCATCGATCGAT" + "AGC" + "CCCCCCCCCC"
    var = GenomicVariant(chrom="chr1", pos=32, ref="C", alt="G", variant_id="SNV_PAM")
    guides = score_allele_specific_guides(ref, var, nuclease_name="SpCas9")

    pam_created = [g for g in guides if g.mechanism == "PAM_CREATED"]
    assert len(pam_created) >= 1
    top = pam_created[0]
    assert top.p_cleave_ref == 0.0
    assert top.p_cleave_alt > 0.3
    assert top.discrimination_score > 30.0
    assert top.pam_alt == "AGG"
    assert top.pam_ref == "AGC"


def test_seed_mismatch_outperforms_distal_mismatch_in_discrimination() -> None:
    # Base locus with canonical AGG PAM at 30..33
    ref = "AAAAAAAAAA" + "GCTAGCTAGCATCGATCGAT" + "AGG" + "TTTTTTTTTT"
    # Variant at index 28 (spacer pos 19, inside SpCas9 seed) vs index 11 (spacer pos 2, distal)
    var_seed = GenomicVariant(chrom="chr1", pos=28, ref="A", alt="C", variant_id="SEED_SNV")
    var_distal = GenomicVariant(chrom="chr1", pos=11, ref="C", alt="A", variant_id="DIST_SNV")

    g_seed = [
        g for g in score_allele_specific_guides(ref, var_seed, "SpCas9")
        if g.strand.value == "+" and g.pam_alt == "AGG"
    ][0]
    g_distal = [
        g for g in score_allele_specific_guides(ref, var_distal, "SpCas9")
        if g.strand.value == "+" and g.pam_alt == "AGG"
    ][0]

    assert g_seed.mechanism == "SEED_MISMATCH"
    assert g_distal.mechanism == "DISTAL_MISMATCH"
    assert g_seed.selectivity_ratio > g_distal.selectivity_ratio
    assert g_seed.discrimination_score > g_distal.discrimination_score


def test_multiplex_guide_set_cover_covers_diverse_isoforms() -> None:
    shared_site = "GCTAGCTAGCATCGATCGATAGG"
    alt_site = "ATCGATCGATCGATCGATCGAGG"
    targets = {
        "isoform_1": "AAAAAAAAAA" + shared_site + "TTTTTTTTTT",
        "isoform_2": "CCCCCCCCCC" + shared_site + "GGGGGGGGGG",
        "isoform_3": "TTTTTTTTTT" + alt_site + "AAAAAAAAAA",
    }
    selected, cov_frac = select_multiplex_guide_cover(targets, nuclease_name="SpCas9", max_guides=4)
    assert cov_frac == 1.0
    assert len(selected) == 2
    covered_union = set().union(*(cov for _, cov in selected))
    assert covered_union == {"isoform_1", "isoform_2", "isoform_3"}

    empty_sel, empty_cov = select_multiplex_guide_cover({}, nuclease_name="SpCas9")
    assert empty_sel == [] and empty_cov == 0.0
