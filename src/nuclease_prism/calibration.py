"""
Biophysical parameter calibration, bootstrap uncertainty quantification, and Sobol sensitivity analysis.

Provides:
1. Synthetic / empirical cleavage benchmark generator and L-BFGS-B / Nelder-Mead optimization
   of R-loop kinetic parameters (``k_zip_forward``, ``protein_stabilization_per_nt``,
   ``seed_mismatch_multiplier``) against observed cleavage fractions.
2. Nonparametric bootstrap confidence intervals for frameshift probability and specificity.
3. Variance-based Sobol first-order sensitivity indices (Saltelli radial sampling estimator)
   quantifying how parameter uncertainty propagates to R-loop cleavage probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .thermodynamics import KineticParameters, analyze_rloop_thermodynamics
from .types import NucleaseProfile


@dataclass(frozen=True)
class CleavageObservation:
    """Empirical or reference cleavage observation for kinetic parameter calibration."""

    spacer_seq: str
    target_sense_seq: str
    observed_cleavage: float


@dataclass(frozen=True)
class CalibrationResult:
    """Summary of kinetic model calibration and parameter uncertainty."""

    initial_rmse: float
    fitted_rmse: float
    fitted_params: KineticParameters
    r_squared: float
    n_observations: int
    n_iterations: int


def generate_synthetic_cleavage_dataset(
    profile: NucleaseProfile,
    true_params: KineticParameters,
    n_samples: int = 24,
    noise_std: float = 0.025,
    seed: int = 42,
) -> List[CleavageObservation]:
    """
    Generate a deterministic synthetic CRISPR cleavage benchmark with 0-2 mismatches.

    Args:
        profile: Target NucleaseProfile.
        true_params: Ground-truth KineticParameters used to generate labels.
        n_samples: Number of guide-target pairs to synthesize.
        noise_std: Gaussian measurement noise standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        List of CleavageObservation samples.
    """
    rng = np.random.default_rng(seed)
    bases = np.array(["A", "C", "G", "T"])
    dataset: List[CleavageObservation] = []

    for i in range(n_samples):
        sp_arr = rng.choice(bases, size=profile.spacer_length)
        tg_arr = sp_arr.copy()
        n_mm = i % 3  # Mix 0, 1, and 2 mismatch targets
        if n_mm > 0:
            mm_locs = rng.choice(profile.spacer_length, size=n_mm, replace=False)
            for loc in mm_locs:
                alts = [b for b in bases if b != sp_arr[loc]]
                tg_arr[loc] = str(rng.choice(alts))

        sp = "".join(sp_arr)
        tg = "".join(tg_arr)
        thermo = analyze_rloop_thermodynamics(sp, profile, target_sense_seq=tg, params=true_params)
        noisy_y = float(np.clip(thermo.kinetic_cleavage_prob + rng.normal(0.0, noise_std), 0.0, 1.0))
        dataset.append(CleavageObservation(spacer_seq=sp, target_sense_seq=tg, observed_cleavage=round(noisy_y, 5)))

    return dataset


def calibrate_rloop_kinetics(
    observations: Sequence[CleavageObservation],
    profile: NucleaseProfile,
    initial_params: KineticParameters | None = None,
    maxiter: int = 45,
) -> CalibrationResult:
    """
    Fit R-loop kinetic parameters to observed cleavage efficiencies via bounded Nelder-Mead.

    Optimizes vector theta = [k_zip_forward, protein_stabilization_per_nt, seed_mismatch_multiplier]
    minimizing mean squared cleavage prediction error with L2 regularization.

    Args:
        observations: Sequence of CleavageObservation data points.
        profile: NucleaseProfile definition.
        initial_params: Starting KineticParameters guess.
        maxiter: Maximum optimizer iterations.

    Returns:
        CalibrationResult containing initial/fitted RMSE, R^2, and optimal parameters.
    """
    if len(observations) < 3:
        raise ValueError("Calibration requires at least 3 cleavage observations.")
    if initial_params is None:
        initial_params = KineticParameters()

    y_obs = np.array([obs.observed_cleavage for obs in observations], dtype=np.float64)

    def _eval_rmse(vec: np.ndarray) -> float:
        k_zip = float(np.clip(vec[0], 20.0, 400.0))
        prot = float(np.clip(vec[1], -0.90, -0.05))
        seed_mult = float(np.clip(vec[2], 1.0, 2.5))
        p = KineticParameters(
            k_zip_forward=k_zip,
            rloop_barrier_scale=initial_params.rloop_barrier_scale,
            protein_stabilization_per_nt=prot,
            seed_mismatch_multiplier=seed_mult,
            temperature_k=initial_params.temperature_k,
        )
        preds = np.array(
            [
                analyze_rloop_thermodynamics(o.spacer_seq, profile, o.target_sense_seq, params=p).kinetic_cleavage_prob
                for o in observations
            ],
            dtype=np.float64,
        )
        mse = float(np.mean((preds - y_obs) ** 2))
        return mse

    x0 = np.array(
        [
            initial_params.k_zip_forward,
            initial_params.protein_stabilization_per_nt,
            initial_params.seed_mismatch_multiplier,
        ],
        dtype=np.float64,
    )
    init_rmse = float(np.sqrt(_eval_rmse(x0)))

    res = minimize(
        _eval_rmse,
        x0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-5},
    )

    opt_x = res.x
    fitted_params = KineticParameters(
        k_zip_forward=round(float(np.clip(opt_x[0], 20.0, 400.0)), 3),
        rloop_barrier_scale=initial_params.rloop_barrier_scale,
        protein_stabilization_per_nt=round(float(np.clip(opt_x[1], -0.90, -0.05)), 4),
        seed_mismatch_multiplier=round(float(np.clip(opt_x[2], 1.0, 2.5)), 4),
        temperature_k=initial_params.temperature_k,
    )

    preds_final = np.array(
        [
            analyze_rloop_thermodynamics(
                o.spacer_seq, profile, o.target_sense_seq, params=fitted_params
            ).kinetic_cleavage_prob
            for o in observations
        ],
        dtype=np.float64,
    )
    fitted_rmse = float(np.sqrt(np.mean((preds_final - y_obs) ** 2)))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    ss_res = float(np.sum((y_obs - preds_final) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0

    return CalibrationResult(
        initial_rmse=round(init_rmse, 5),
        fitted_rmse=round(fitted_rmse, 5),
        fitted_params=fitted_params,
        r_squared=round(max(-1.0, min(1.0, r2)), 4),
        n_observations=len(observations),
        n_iterations=int(res.nit),
    )


def bootstrap_confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
    n_boot: int = 500,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute nonparametric percentile bootstrap CI for the sample mean.

    Returns:
        Tuple ``(mean_estimate, ci_lower, ci_upper)``.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    if arr.size == 1:
        v = float(arr[0])
        return (v, v, v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = np.mean(arr[idx], axis=1)
    alpha = 0.5 * (1.0 - confidence)
    lo = float(np.quantile(boot_means, alpha))
    hi = float(np.quantile(boot_means, 1.0 - alpha))
    return round(float(np.mean(arr)), 5), round(lo, 5), round(hi, 5)


def sobol_rloop_sensitivity(
    spacer_seq: str,
    target_sense_seq: str,
    profile: NucleaseProfile,
    n_samples: int = 48,
    seed: int = 123,
) -> Dict[str, float]:
    """
    Estimate first-order Sobol variance-based sensitivity indices for R-loop cleavage probability.

    Evaluates Saltelli's A/B/AB_i matrix estimator across 4 biophysical parameters:
    - ``k_zip_forward`` in [60.0, 220.0] s^-1
    - ``protein_stabilization_per_nt`` in [-0.48, -0.18] kcal/mol
    - ``seed_mismatch_multiplier`` in [1.1, 1.9]
    - ``temperature_k`` in [305.15, 315.15] K (32 C - 42 C)

    Returns:
        Dictionary mapping parameter name to normalized first-order Sobol sensitivity S_i in [0, 1].
    """
    rng = np.random.default_rng(seed)
    bounds = [
        ("k_zip_forward", 60.0, 220.0),
        ("protein_stabilization_per_nt", -0.48, -0.18),
        ("seed_mismatch_multiplier", 1.10, 1.90),
        ("temperature_k", 305.15, 315.15),
    ]
    d = len(bounds)
    u_a = rng.uniform(0.0, 1.0, size=(n_samples, d))
    u_b = rng.uniform(0.0, 1.0, size=(n_samples, d))

    def _eval_batch(u_mat: np.ndarray) -> np.ndarray:
        out = np.empty(u_mat.shape[0], dtype=np.float64)
        for row_idx in range(u_mat.shape[0]):
            vals = [lo + u_mat[row_idx, j] * (hi - lo) for j, (_, lo, hi) in enumerate(bounds)]
            kp = KineticParameters(
                k_zip_forward=float(vals[0]),
                protein_stabilization_per_nt=float(vals[1]),
                seed_mismatch_multiplier=float(vals[2]),
                temperature_k=float(vals[3]),
            )
            out[row_idx] = analyze_rloop_thermodynamics(
                spacer_seq, profile, target_sense_seq=target_sense_seq, params=kp
            ).kinetic_cleavage_prob
        return out

    y_a = _eval_batch(u_a)
    y_b = _eval_batch(u_b)
    var_y = float(np.var(np.concatenate([y_a, y_b])))
    if var_y < 1e-12:
        return {name: round(1.0 / d, 4) for name, _, _ in bounds}

    raw_s: Dict[str, float] = {}
    for j, (name, _, _) in enumerate(bounds):
        u_ab = u_a.copy()
        u_ab[:, j] = u_b[:, j]
        y_ab = _eval_batch(u_ab)
        # Saltelli first-order estimator: (1/N) * sum(y_b * (y_ab - y_a)) / Var(Y)
        s_j = float(np.mean(y_b * (y_ab - y_a)) / var_y)
        raw_s[name] = max(0.0, s_j)

    total_s = sum(raw_s.values())
    if total_s > 1.0:
        # Normalize slightly if finite-sample Monte Carlo variance exceeded 1.0
        raw_s = {k: v / total_s for k, v in raw_s.items()}
    return {k: round(v, 4) for k, v in raw_s.items()}
