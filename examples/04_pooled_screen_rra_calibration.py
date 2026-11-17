"""
Example 04: Pooled CRISPR screen RRA hit calling, Bliss synergy, and R-loop calibration.

Demonstrates:
- Median-of-ratios normalization + Negative Binomial dispersion estimation
- Robust Rank Aggregation (RRA) gene depletion & resistance hit detection with BH FDR
- Dual-guide Bliss synthetic lethality scoring
- R-loop kinetic parameter calibration + Sobol first-order sensitivity indices
"""

from __future__ import annotations

import numpy as np

from nuclease_prism import (
    KineticParameters,
    analyze_pooled_screen,
    calibrate_rloop_kinetics,
    compute_bliss_synergy,
    generate_synthetic_cleavage_dataset,
    get_nuclease_profile,
    sobol_rloop_sensitivity,
)
from nuclease_prism.visualizer import assert_strict_ascii, render_screen_report


def main() -> None:
    rng = np.random.default_rng(2026)
    n_genes = 60
    guides_per_gene = 4
    genes = []
    essential = set()
    non_essential = set()
    counts = np.zeros((n_genes * guides_per_gene, 4), dtype=np.float64)

    for g_idx in range(n_genes):
        if g_idx < 6:
            name = f"ESSENTIAL_{g_idx+1}"
            essential.add(name)
            mult = 0.14
        elif g_idx < 10:
            name = f"TUMOR_SUPP_{g_idx-5}"
            mult = 3.1
        else:
            name = f"PASSENGER_{g_idx+1}"
            if g_idx < 35:
                non_essential.add(name)
            mult = 1.0
        for _ in range(guides_per_gene):
            row = len(genes)
            genes.append(name)
            mu0 = float(rng.uniform(400.0, 950.0))
            counts[row, 0] = rng.negative_binomial(30.0, 30.0 / (30.0 + mu0))
            counts[row, 1] = rng.negative_binomial(30.0, 30.0 / (30.0 + mu0 * 1.02))
            counts[row, 2] = rng.negative_binomial(20.0, 20.0 / (20.0 + mu0 * mult))
            counts[row, 3] = rng.negative_binomial(20.0, 20.0 / (20.0 + mu0 * mult * 0.98))

    screen_report = analyze_pooled_screen(
        counts=counts,
        guide_genes=genes,
        control_cols=[0, 1],
        treatment_cols=[2, 3],
        essential_genes=essential,
        non_essential_genes=non_essential,
    )
    print(render_screen_report(screen_report, top_n=12))

    # Bliss synergy check for BRCA1 + PARP1 dual knockout
    exp_lfc, delta, cls = compute_bliss_synergy(lfc_a=-0.35, lfc_b=-0.40, lfc_ab=-2.15)
    print(assert_strict_ascii(f"\nBliss Interaction (BRCA1+PARP1): expected={exp_lfc}, delta={delta} -> {cls}"))

    # R-loop parameter calibration + Sobol sensitivity
    profile = get_nuclease_profile("SpCas9")
    truth = KineticParameters(k_zip_forward=155.0, protein_stabilization_per_nt=-0.38, seed_mismatch_multiplier=1.6)
    obs = generate_synthetic_cleavage_dataset(profile, truth, n_samples=18, seed=99)
    guess = KineticParameters(k_zip_forward=95.0, protein_stabilization_per_nt=-0.24, seed_mismatch_multiplier=1.25)
    cal = calibrate_rloop_kinetics(obs, profile, initial_params=guess, maxiter=30)
    sobol = sobol_rloop_sensitivity(obs[0].spacer_seq, obs[0].target_sense_seq, profile, n_samples=32, seed=99)
    print(
        assert_strict_ascii(
            f"Calibration RMSE: {cal.initial_rmse:.4f} -> {cal.fitted_rmse:.4f} (R^2={cal.r_squared:.4f})\n"
            f"Sobol S_i: {sobol}"
        )
    )


if __name__ == "__main__":
    main()
