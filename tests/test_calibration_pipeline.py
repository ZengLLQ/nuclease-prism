"""Tests for kinetic parameter calibration, bootstrap UQ, Sobol sensitivity, and GuideDesignPipeline."""

from __future__ import annotations

import json

from nuclease_prism.calibration import (
    bootstrap_confidence_interval,
    calibrate_rloop_kinetics,
    generate_synthetic_cleavage_dataset,
    sobol_rloop_sensitivity,
)
from nuclease_prism.nucleases import get_nuclease_profile
from nuclease_prism.pipeline import (
    GuideDesignPipeline,
    compute_pareto_ranks,
    report_to_csv,
    report_to_json,
)
from nuclease_prism.thermodynamics import KineticParameters
from nuclease_prism.types import GenomicInterval


def test_rloop_calibration_reduces_rmse() -> None:
    sp = get_nuclease_profile("SpCas9")
    truth = KineticParameters(k_zip_forward=160.0, protein_stabilization_per_nt=-0.40, seed_mismatch_multiplier=1.65)
    dataset = generate_synthetic_cleavage_dataset(sp, truth, n_samples=15, noise_std=0.01, seed=11)
    bad_init = KineticParameters(k_zip_forward=55.0, protein_stabilization_per_nt=-0.15, seed_mismatch_multiplier=1.05)

    cal = calibrate_rloop_kinetics(dataset, sp, initial_params=bad_init, maxiter=35)
    assert cal.fitted_rmse < cal.initial_rmse
    assert cal.r_squared > 0.75


def test_bootstrap_and_sobol_sensitivity_invariants() -> None:
    mean_v, lo, hi = bootstrap_confidence_interval([0.4, 0.5, 0.6, 0.55, 0.45, 0.65], seed=42)
    assert lo <= mean_v <= hi

    sp = get_nuclease_profile("SpCas9")
    spacer = "GCTAGCTAGCATCGATCGAT"
    target = "ACTAGCTAGCATCGATCGAT"
    sobol = sobol_rloop_sensitivity(spacer, target, sp, n_samples=24, seed=42)
    assert set(sobol.keys()) == {
        "k_zip_forward",
        "protein_stabilization_per_nt",
        "seed_mismatch_multiplier",
        "temperature_k",
    }
    assert all(0.0 <= v <= 1.0 for v in sobol.values())
    assert sum(sobol.values()) <= 1.0001


def test_pipeline_end_to_end_pareto_and_serialization(
    benchmark_genome: tuple[dict[str, str], list[GenomicInterval]],
) -> None:
    contigs, intervals = benchmark_genome
    pipeline = GuideDesignPipeline(reference_contigs=contigs, intervals=intervals)
    report = pipeline.design_for_locus(
        locus_name="TP53_EXON5",
        nuclease_names=("SpCas9", "AsCas12a"),
        donor_conc_nm=40.0,
        top_k=15,
    )
    assert report.total_candidates > 0
    assert len(report.evaluations) > 0
    assert len(report.pareto_frontier_ids) >= 1
    assert report.evaluations[0].pareto_rank == 1

    ranks = compute_pareto_ranks(report.evaluations)
    assert all(r >= 1 for r in ranks)

    json_str = report_to_json(report)
    parsed = json.loads(json_str)
    assert parsed["locus_name"] == "TP53_EXON5"
    assert len(parsed["evaluations"]) == len(report.evaluations)

    csv_str = report_to_csv(report)
    csv_rows = csv_str.strip().splitlines()
    assert len(csv_rows) == len(report.evaluations) + 1
