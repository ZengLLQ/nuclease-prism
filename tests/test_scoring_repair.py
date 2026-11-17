"""Tests for CFD/MIT off-target scoring, on-target efficacy, base/prime windows, and DNA repair."""

from __future__ import annotations

import pytest

from nuclease_prism.nucleases import get_nuclease_profile
from nuclease_prism.repair import predict_repair_spectrum, simulate_repair_time_course
from nuclease_prism.scoring import (
    analyze_precision_editing_window,
    compute_cfd_score,
    compute_mit_score,
    score_on_target_efficacy,
)
from nuclease_prism.sequence import scan_protospacers
from nuclease_prism.thermodynamics import analyze_rloop_thermodynamics
from nuclease_prism.types import BulgeType, RepairPathway


def test_cfd_and_mit_scoring_monotonicity() -> None:
    sp = get_nuclease_profile("SpCas9")
    spacer = "GCTAGCTAGCATCGATCGAT"
    # 1 distal mismatch vs 1 PAM-proximal seed mismatch vs 2 mismatches
    distal_t = "ACTAGCTAGCATCGATCGAT"
    seed_t = "GCTAGCTAGCATCGATCGAC"
    two_mm_t = "ACTAGCTAGCATCGATCGAC"

    cfd_dist, ddg_dist = compute_cfd_score(spacer, spacer, distal_t, 1.0, BulgeType.NONE, sp)
    cfd_seed, ddg_seed = compute_cfd_score(spacer, spacer, seed_t, 1.0, BulgeType.NONE, sp)
    cfd_two, _ = compute_cfd_score(spacer, spacer, two_mm_t, 1.0, BulgeType.NONE, sp)

    assert 0.0 < cfd_two < cfd_dist <= 1.0
    assert cfd_dist > cfd_seed
    assert ddg_dist > 0.0 and ddg_seed > 0.0

    mit_1 = compute_mit_score((1,), 1.0, BulgeType.NONE, 20)
    mit_2 = compute_mit_score((1, 18), 1.0, BulgeType.NONE, 20)
    mit_bulge = compute_mit_score((), 1.0, BulgeType.DNA_BULGE, 20)
    assert mit_1 > mit_2 > 0.0
    assert 0.0 < mit_bulge < 0.25


def test_on_target_penalizes_pol3_terminator_and_extreme_gc() -> None:
    sp = get_nuclease_profile("SpCas9")
    good_seq = "AAAAAAAAAA" + "GCTAGCATGCATGCTAGCCG" + "AGG" + "TTTTTTTTTT"
    bad_seq = "AAAAAAAAAA" + "ATTTTTATCGATTTTTATGT" + "TGG" + "TTTTTTTTTT"

    cand_good = [c for c in scan_protospacers(good_seq, sp) if c.spacer_seq == "GCTAGCATGCATGCTAGCCG"][0]
    cand_bad = [c for c in scan_protospacers(bad_seq, sp) if c.spacer_seq == "ATTTTTATCGATTTTTATGT"][0]

    s_good = score_on_target_efficacy(cand_good, analyze_rloop_thermodynamics(cand_good.spacer_seq, sp), sp)
    s_bad = score_on_target_efficacy(cand_bad, analyze_rloop_thermodynamics(cand_bad.spacer_seq, sp), sp)
    assert 0.0 <= s_bad < s_good <= 100.0


def test_base_and_prime_editing_window_profiles() -> None:
    abe = get_nuclease_profile("ABE8e")
    pe3 = get_nuclease_profile("PE3")
    seq = "CCCCCCCCCC" + "GCTAAACGATCGATCGATCG" + "AGG" + "GGGGGGGGGGGGGGGGGGGG"

    cand_abe = [c for c in scan_protospacers(seq, abe) if c.spacer_seq == "GCTAAACGATCGATCGATCG"][0]
    ed_abe = analyze_precision_editing_window(cand_abe, abe, on_target_score=75.0)
    assert ed_abe is not None
    assert 4 in ed_abe.target_positions and 5 in ed_abe.target_positions
    assert ed_abe.bystander_risk > 0.0

    cand_pe3 = [c for c in scan_protospacers(seq, pe3) if c.spacer_seq == "GCTAAACGATCGATCGATCG"][0]
    ed_pe3 = analyze_precision_editing_window(cand_pe3, pe3, on_target_score=70.0)
    assert ed_pe3 is not None
    assert ed_pe3.optimal_pbs_tm_celsius is not None
    assert ed_pe3.optimal_rtt_length in (10, 12, 14, 16)


def test_repair_ode_conservation_and_hdr_titration(microhomology_context: tuple[str, int]) -> None:
    seq, cut_idx = microhomology_context
    sp = get_nuclease_profile("SpCas9")

    t, unrep, nhej, mmej, hdr = simulate_repair_time_course(mmej_drive=1.2, donor_conc_nm=150.0)
    assert len(t) == 25
    for idx in range(len(t)):
        total = unrep[idx] + nhej[idx] + mmej[idx] + hdr[idx]
        assert total == pytest.approx(1.0, abs=1e-4)

    spec_no_donor = predict_repair_spectrum(seq, cut_idx, sp, donor_conc_nm=0.0)
    spec_high_donor = predict_repair_spectrum(seq, cut_idx, sp, donor_conc_nm=400.0)
    assert sum(spec_no_donor.pathway_fractions.values()) == pytest.approx(1.0, abs=1e-3)
    assert spec_no_donor.hdr_efficiency == 0.0
    assert spec_high_donor.hdr_efficiency > 0.25
    assert any(a.pathway == RepairPathway.MMEJ for a in spec_no_donor.top_alleles)
    assert any(a.pathway == RepairPathway.HDR for a in spec_high_donor.top_alleles)
    assert 0.0 <= spec_no_donor.frameshift_prob <= 1.0
    assert spec_no_donor.indel_entropy_bits >= 0.0


def test_repair_cas13d_and_invalid_cut_index() -> None:
    cas13 = get_nuclease_profile("RfxCas13d")
    spec = predict_repair_spectrum("ATCGATCGATCGATCGATCGATCGATCG", 14, cas13)
    assert spec.top_alleles[0].allele_label == "RNA:knockdown"

    sp = get_nuclease_profile("SpCas9")
    with pytest.raises(ValueError):
        predict_repair_spectrum("ATCGATCGATCG", 1, sp)
