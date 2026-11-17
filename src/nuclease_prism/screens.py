"""
Pooled CRISPR functional screen statistical analyzer (MAGeCK-inspired RRA & Negative Binomial).

Implements:
1. Median-of-ratios size-factor normalization across control and treatment replicates.
2. Negative Binomial mean-variance dispersion trend estimation:
       Var(Y) = mu + (alpha + beta / mu) * mu^2 = mu + beta * mu + alpha * mu^2
   fitted via robust weighted regression on replicate sample variances.
3. Guide-level NB Wald z-scores, log2 fold changes (LFC), and normal/NB p-values.
4. Robust Rank Aggregation (RRA, Kolde et al. 2012 / Li et al. Genome Biology 2014):
   exact beta-distribution order-statistic p-values ``Beta(k, n - k + 1)`` for gene-level
   negative (depletion) and positive (enrichment) selection, with Bonferroni/BH FDR control.
5. Screen Quality Control metrics (Gini index, zero-dropout rate, essential vs non-essential AUROC)
   and dual-guide genetic interaction / Bliss independence synergy estimation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from .types import GeneScreenResult, ScreenAnalysisReport


def compute_gini_index(values: np.ndarray) -> float:
    """
    Compute the Gini inequality coefficient of a non-negative count vector.

    Healthy baseline plasmid/control pooled CRISPR libraries typically achieve Gini < 0.15.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0 or np.all(arr == 0.0):
        return 0.0
    arr = np.sort(np.maximum(arr, 0.0))
    n = arr.size
    index = np.arange(1, n + 1, dtype=np.float64)
    total = float(np.sum(arr))
    if total <= 0.0:
        return 0.0
    gini = float((2.0 * np.sum(index * arr)) / (n * total) - (n + 1.0) / n)
    return round(max(0.0, min(1.0, gini)), 5)


def median_ratio_normalize(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform DESeq2/MAGeCK median-of-ratios library normalization on a (G x S) count matrix.

    Args:
        counts: 2D array of shape (n_guides, n_samples) with non-negative counts.

    Returns:
        Tuple ``(normalized_counts, size_factors)``.
    """
    mat = np.asarray(counts, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] < 2 or mat.shape[1] < 2:
        raise ValueError("Count matrix must be 2D with at least 2 guides and 2 samples.")

    # Compute geometric mean per guide across samples (masking zeros via pseudocount)
    safe = mat + 1.0
    geo_means = np.exp(np.mean(np.log(safe), axis=1))
    valid = geo_means > 1.0
    if not np.any(valid):
        factors = np.ones(mat.shape[1], dtype=np.float64)
        return mat.copy(), factors

    ratios = safe[valid] / geo_means[valid, None]
    factors = np.median(ratios, axis=0)
    # Normalize geometric mean of size factors to 1.0
    gm_f = float(np.exp(np.mean(np.log(np.maximum(factors, 1e-6)))))
    factors = np.maximum(factors / gm_f, 1e-6)
    return mat / factors[None, :], factors


def fit_nb_dispersion(
    norm_counts: np.ndarray,
    group_indices: Sequence[int],
) -> Tuple[float, float]:
    """
    Fit Negative Binomial mean-variance dispersion trend across replicates.

    Model:
        excess_var / mu^2 = alpha + beta / mu
    where ``excess_var = max(0, sample_var - mu)``.

    Args:
        norm_counts: Normalized count matrix (n_guides x n_samples).
        group_indices: Column indices belonging to replicates of one condition.

    Returns:
        Tuple ``(alpha_asymptotic_dispersion, beta_low_count_slope)``.
    """
    sub = norm_counts[:, list(group_indices)]
    if sub.shape[1] < 2:
        return (0.04, 1.5)

    mu = np.mean(sub, axis=1)
    var = np.var(sub, axis=1, ddof=1)
    mask = mu >= 5.0
    if np.sum(mask) < 5:
        return (0.05, 1.5)

    mu_m = mu[mask]
    excess = np.maximum(1e-4, var[mask] - mu_m)
    y_rel = excess / (mu_m ** 2)
    x_inv = 1.0 / mu_m

    # Weighted least-squares fit: y_rel = alpha + beta * x_inv
    design = np.column_stack([np.ones_like(x_inv), x_inv])
    coefs, _, _, _ = np.linalg.lstsq(design, y_rel, rcond=None)
    alpha = float(max(0.005, min(1.5, coefs[0])))
    beta = float(max(0.0, min(25.0, coefs[1])))
    return round(alpha, 5), round(beta, 5)


def benjamini_hochberg(pvalues: Sequence[float]) -> List[float]:
    """
    Compute Benjamini-Hochberg False Discovery Rate (FDR) adjusted q-values.

    Preserves input ordering and enforces monotonicity from largest to smallest rank.
    """
    m = len(pvalues)
    if m == 0:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda pair: pair[1])
    q_vals = [0.0] * m
    running_min = 1.0
    for rev_rank in range(m - 1, -1, -1):
        orig_idx, p_val = indexed[rev_rank]
        rank_1 = rev_rank + 1
        candidate = min(1.0, (p_val * m) / rank_1)
        running_min = min(running_min, candidate)
        q_vals[orig_idx] = round(running_min, 8)
    return q_vals


def compute_rra_pvalue(norm_ranks: Sequence[float]) -> float:
    """
    Compute Robust Rank Aggregation (RRA) exact beta order-statistic p-value.

    Given ``k`` normalized uniform ranks ``u_(1) <= u_(2) <= ... <= u_(k)`` in (0, 1],
    each order statistic ``u_(i)`` follows ``Beta(i, k - i + 1)`` under the null hypothesis
    of uniform random ranks. The RRA rho score is ``rho = min_i P(Beta(i, k - i + 1) <= u_(i))``,
    Bonferroni-corrected across the ``k`` order statistics: ``min(1.0, k * rho)``.

    Args:
        norm_ranks: Normalized percentile ranks in (0, 1] for guides targeting one gene.

    Returns:
        Exact RRA p-value in (0, 1].
    """
    u_sorted = sorted(max(1e-12, min(1.0, float(u))) for u in norm_ranks)
    k = len(u_sorted)
    if k == 0:
        return 1.0
    min_p = 1.0
    for i_idx, u_val in enumerate(u_sorted, start=1):
        p_order = float(beta_dist.cdf(u_val, i_idx, k - i_idx + 1))
        if p_order < min_p:
            min_p = p_order
    # Bonferroni bound over k order statistics gives conservative uniform null calibration
    return max(1e-15, min(1.0, k * min_p))


def compute_auroc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """
    Compute exact Mann-Whitney / trapezoidal AUROC for binary classification labels.
    """
    if len(scores) != len(labels) or len(scores) < 2:
        return None
    pos = [s for s, lbl in zip(scores, labels) if lbl]
    neg = [s for s, lbl in zip(scores, labels) if not lbl]
    if not pos or not neg:
        return None
    wins = 0.0
    for p_s in pos:
        for n_s in neg:
            if p_s > n_s:
                wins += 1.0
            elif p_s == n_s:
                wins += 0.5
    return round(wins / (len(pos) * len(neg)), 4)


def analyze_pooled_screen(
    counts: np.ndarray,
    guide_genes: Sequence[str],
    control_cols: Sequence[int],
    treatment_cols: Sequence[int],
    essential_genes: Optional[set[str]] = None,
    non_essential_genes: Optional[set[str]] = None,
) -> ScreenAnalysisReport:
    """
    Run full MAGeCK-style pooled CRISPR screen differential abundance & RRA gene ranking.

    Args:
        counts: Raw integer/float guide count matrix (n_guides x n_samples).
        guide_genes: Gene symbol corresponding to each guide row (len == n_guides).
        control_cols: Column indices for baseline / plasmid / day-0 samples.
        treatment_cols: Column indices for post-selection / drug-treated samples.
        essential_genes: Optional reference set of positive-control essential genes for AUROC QC.
        non_essential_genes: Optional reference set of negative-control genes for AUROC QC.

    Returns:
        Complete ScreenAnalysisReport with QC metrics and FDR-corrected gene rankings.
    """
    mat = np.asarray(counts, dtype=np.float64)
    n_guides = mat.shape[0]
    if len(guide_genes) != n_guides:
        raise ValueError(f"guide_genes length {len(guide_genes)} != count rows {n_guides}")
    if not control_cols or not treatment_cols:
        raise ValueError("Both control_cols and treatment_cols must be non-empty.")

    norm_counts, _ = median_ratio_normalize(mat)
    ctrl_mat = norm_counts[:, list(control_cols)]
    trt_mat = norm_counts[:, list(treatment_cols)]

    ctrl_mean = np.mean(ctrl_mat, axis=1)
    trt_mean = np.mean(trt_mat, axis=1)

    gini_ctrl = compute_gini_index(np.mean(mat[:, list(control_cols)], axis=1))
    zero_dropout = float(np.mean(np.mean(mat[:, list(control_cols)], axis=1) <= 0.0))

    alpha, beta_slope = fit_nb_dispersion(norm_counts, control_cols)

    # Guide-level NB variance + Wald z-score + log2 fold change
    n_ctrl = len(control_cols)
    n_trt = len(treatment_cols)
    pseudocount = 0.5
    lfc = np.log2((trt_mean + pseudocount) / (ctrl_mean + pseudocount))

    mu_pool = np.maximum(1.0, 0.5 * (ctrl_mean + trt_mean))
    nb_var = mu_pool + beta_slope * mu_pool + alpha * (mu_pool ** 2)
    se_diff = np.sqrt(nb_var * (1.0 / n_ctrl + 1.0 / n_trt))
    z_scores = (trt_mean - ctrl_mean) / np.maximum(se_diff, 1e-6)

    # One-sided guide p-values for depletion (left tail) and enrichment (right tail)
    p_dep = norm.cdf(z_scores)
    p_enr = 1.0 - p_dep

    # Rank guides across the full library in (0, 1]
    order_dep = np.argsort(p_dep, kind="mergesort")
    ranks_dep = np.empty(n_guides, dtype=np.float64)
    ranks_dep[order_dep] = (np.arange(1, n_guides + 1, dtype=np.float64)) / n_guides

    order_enr = np.argsort(p_enr, kind="mergesort")
    ranks_enr = np.empty(n_guides, dtype=np.float64)
    ranks_enr[order_enr] = (np.arange(1, n_guides + 1, dtype=np.float64)) / n_guides

    # Group guide indices by gene
    gene_map: Dict[str, List[int]] = {}
    for idx, g in enumerate(guide_genes):
        gene_map.setdefault(g, []).append(idx)

    genes_sorted = sorted(gene_map.keys())
    raw_dep_p: List[float] = []
    raw_enr_p: List[float] = []
    gene_stats: List[Tuple[str, int, float, float]] = []

    for gene in genes_sorted:
        idxs = gene_map[gene]
        g_lfc = lfc[idxs]
        p_d = compute_rra_pvalue(ranks_dep[idxs])
        p_e = compute_rra_pvalue(ranks_enr[idxs])
        raw_dep_p.append(p_d)
        raw_enr_p.append(p_e)
        gene_stats.append((gene, len(idxs), float(np.mean(g_lfc)), float(np.median(g_lfc))))

    fdr_dep = benjamini_hochberg(raw_dep_p)
    fdr_enr = benjamini_hochberg(raw_enr_p)

    unsorted_results: List[GeneScreenResult] = []
    for i, (gene, ng, m_lfc, med_lfc) in enumerate(gene_stats):
        best_p = min(raw_dep_p[i], raw_enr_p[i])
        rra_score = round(-math.log10(max(1e-15, best_p)), 4)
        unsorted_results.append(
            GeneScreenResult(
                gene=gene,
                n_guides=ng,
                mean_lfc=round(m_lfc, 4),
                median_lfc=round(med_lfc, 4),
                rra_depletion_pvalue=raw_dep_p[i],
                rra_enrichment_pvalue=raw_enr_p[i],
                fdr_depletion=fdr_dep[i],
                fdr_enrichment=fdr_enr[i],
                rra_score=rra_score,
                rank=0,
            )
        )

    # Sort descending by RRA significance score
    unsorted_results.sort(key=lambda r: (-r.rra_score, r.mean_lfc))
    ranked_results = [
        GeneScreenResult(
            gene=r.gene,
            n_guides=r.n_guides,
            mean_lfc=r.mean_lfc,
            median_lfc=r.median_lfc,
            rra_depletion_pvalue=r.rra_depletion_pvalue,
            rra_enrichment_pvalue=r.rra_enrichment_pvalue,
            fdr_depletion=r.fdr_depletion,
            fdr_enrichment=r.fdr_enrichment,
            rra_score=r.rra_score,
            rank=idx,
        )
        for idx, r in enumerate(unsorted_results, start=1)
    ]

    auroc: Optional[float] = None
    if essential_genes and non_essential_genes:
        eval_scores: List[float] = []
        eval_labels: List[bool] = []
        for r in ranked_results:
            if r.gene in essential_genes:
                eval_scores.append(-r.mean_lfc)
                eval_labels.append(True)
            elif r.gene in non_essential_genes:
                eval_scores.append(-r.mean_lfc)
                eval_labels.append(False)
        auroc = compute_auroc(eval_scores, eval_labels)

    return ScreenAnalysisReport(
        n_guides=n_guides,
        n_genes=len(ranked_results),
        gini_index_control=gini_ctrl,
        zero_dropout_fraction=round(zero_dropout, 4),
        nb_dispersion_alpha=alpha,
        nb_dispersion_beta=beta_slope,
        essential_auroc=auroc,
        gene_results=ranked_results,
    )


def compute_bliss_synergy(
    lfc_a: float,
    lfc_b: float,
    lfc_ab: float,
) -> Tuple[float, float, str]:
    """
    Compute dual-guide synthetic lethality / genetic interaction score under Bliss additivity.

    In log2-fold-change viability coordinates, independent additive fitness effects satisfy
    ``E[LFC_AB] = LFC_A + LFC_B``. Negative interaction residual ``delta_LFC = LFC_AB - (LFC_A + LFC_B)``
    indicates synthetic lethality; positive residual indicates genetic buffering / epistasis.

    Returns:
        Tuple ``(expected_lfc, interaction_delta, classification)``.
    """
    expected = lfc_a + lfc_b
    delta = lfc_ab - expected
    if delta <= -0.75:
        label = "synthetic_lethal"
    elif delta >= 0.75:
        label = "buffering_epistasis"
    else:
        label = "additive"
    return round(expected, 4), round(delta, 4), label
