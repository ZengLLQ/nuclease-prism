"""Unit & statistical calibration tests for pooled CRISPR screen NB/RRA analysis."""

from __future__ import annotations

import numpy as np
import pytest

from nuclease_prism.screens import (
    analyze_pooled_screen,
    benjamini_hochberg,
    compute_auroc,
    compute_bliss_synergy,
    compute_gini_index,
    compute_rra_pvalue,
    median_ratio_normalize,
)


def test_gini_index_and_median_ratio_normalization() -> None:
    uniform = np.full(100, 500.0)
    skewed = np.array([1.0] * 95 + [10000.0] * 5)
    assert compute_gini_index(uniform) < 0.01
    assert compute_gini_index(skewed) > 0.70

    # Sample 1 has 2x sequencing depth of Sample 0
    counts = np.array([[100, 200], [200, 400], [500, 1000], [50, 100]], dtype=np.float64)
    norm, factors = median_ratio_normalize(counts)
    assert factors[1] / factors[0] == pytest.approx(2.0, rel=1e-2)
    assert np.allclose(norm[:, 0], norm[:, 1], rtol=1e-2)


def test_rra_order_statistics_and_bh_fdr() -> None:
    # Strongly depleted guides clustered near rank 0.02 should yield tiny RRA p-value
    p_hit = compute_rra_pvalue([0.005, 0.012, 0.025, 0.04])
    # Uniformly scattered guides should yield non-significant RRA p-value
    p_null = compute_rra_pvalue([0.20, 0.45, 0.72, 0.91])
    assert p_hit < 1e-4
    assert p_null > 0.25

    q_vals = benjamini_hochberg([1e-5, 0.01, 0.04, 0.50])
    assert q_vals[0] <= q_vals[1] <= q_vals[2] <= q_vals[3] <= 1.0


def test_pooled_screen_separates_essential_and_resistance_genes() -> None:
    rng = np.random.default_rng(1234)
    n_genes = 50
    guides_per_gene = 4
    genes = []
    essential = set()
    non_essential = set()
    counts = np.zeros((n_genes * guides_per_gene, 4), dtype=np.float64)

    for g in range(n_genes):
        if g < 5:
            name = f"ESS_{g}"
            essential.add(name)
            mult = 0.12
        elif g < 8:
            name = f"RES_{g}"
            mult = 3.2
        else:
            name = f"NEUT_{g}"
            if g < 30:
                non_essential.add(name)
            mult = 1.0
        for _ in range(guides_per_gene):
            r = len(genes)
            genes.append(name)
            mu = float(rng.uniform(400.0, 800.0))
            counts[r, 0] = rng.negative_binomial(30.0, 30.0 / (30.0 + mu))
            counts[r, 1] = rng.negative_binomial(30.0, 30.0 / (30.0 + mu))
            counts[r, 2] = rng.negative_binomial(25.0, 25.0 / (25.0 + mu * mult))
            counts[r, 3] = rng.negative_binomial(25.0, 25.0 / (25.0 + mu * mult))

    report = analyze_pooled_screen(
        counts=counts,
        guide_genes=genes,
        control_cols=[0, 1],
        treatment_cols=[2, 3],
        essential_genes=essential,
        non_essential_genes=non_essential,
    )
    assert report.n_genes == n_genes
    assert report.n_guides == n_genes * guides_per_gene
    assert report.gini_index_control < 0.25
    assert report.essential_auroc is not None and report.essential_auroc >= 0.95

    top_genes = {res.gene for res in report.gene_results[:8]}
    assert essential <= top_genes
    ess_fdrs = [res.fdr_depletion for res in report.gene_results if res.gene in essential]
    assert all(f < 0.05 for f in ess_fdrs)


def test_auroc_and_bliss_synergy() -> None:
    assert compute_auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    exp, delta, label = compute_bliss_synergy(-0.2, -0.3, -1.8)
    assert exp == pytest.approx(-0.5)
    assert delta == pytest.approx(-1.3)
    assert label == "synthetic_lethal"
