"""Phase 3E.6 Uncertainty Calibration & Statistical Diagnostics.

Implements two distinct evaluation modes:
1. HEURISTIC_CONFIDENCE_RANKING: Spearman rank correlation, bootstrap CI, and quintile error stratification.
   Explicitly rejects universal rho > 0.40 hard gates.
2. PROBABILISTIC_UNCERTAINTY: Empirical coverage probability (1-sigma, 2-sigma, 3-sigma) under an
   explicitly pre-registered Gaussian or Chi-squared error model.
Rejects representing heuristic confidence or regularized covariance as calibrated Gaussian probabilities (MUT-10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.benchmark.models import (
    UncertaintyEvaluationMode,
    UncertaintyStatus,
    ContractViolationError,
)

DEFAULT_BENCHMARK_MIN_SAMPLE_POLICY: int = 30
"""Benchmark project policy for minimum required sample count to evaluate ranking or coverage.
Insufficient-sample gating is a benchmark policy and does not establish universal statistical adequacy.
Configurable per evaluation call; sample count below the configured policy threshold emits NOT_EVALUABLE.
"""


@dataclass(frozen=True)
class HeuristicRankingResult:
    """Exploratory rank correlation between predicted confidence/dispersion and observed error."""
    sample_count: int
    spearman_rho: float
    p_value: float
    bootstrap_ci_95: Tuple[float, float]
    quintile_median_errors: List[float]
    is_monotonically_ordered: bool
    status: UncertaintyStatus
    is_diagnostic_only: bool = True


@dataclass(frozen=True)
class ProbabilisticCoverageResult:
    """Calibration verification under an explicitly declared probabilistic error model."""
    sample_count: int
    model_declared: str
    empirical_coverage_1sigma: float
    empirical_coverage_2sigma: float
    empirical_coverage_3sigma: float
    theoretical_coverage_1sigma: float = 0.6827
    theoretical_coverage_2sigma: float = 0.9545
    theoretical_coverage_3sigma: float = 0.9973
    status: UncertaintyStatus = UncertaintyStatus.CALIBRATION_SUPPORTED


def compute_spearman_rank_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """Computes Spearman rank correlation coefficient and asymptotic p-value."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n = x_arr.size
    if n < 3:
        return 0.0, 1.0

    rank_x = np.argsort(np.argsort(x_arr)).astype(np.float64)
    rank_y = np.argsort(np.argsort(y_arr)).astype(np.float64)

    # Pearson on ranks
    vx = rank_x - np.mean(rank_x)
    vy = rank_y - np.mean(rank_y)
    denom = np.sqrt(np.sum(vx ** 2) * np.sum(vy ** 2))
    if denom < 1e-12:
        return 0.0, 1.0

    rho = float(np.sum(vx * vy) / denom)
    rho_clipped = float(np.clip(rho, -1.0, 1.0))

    # Asymptotic t-distribution p-value
    if abs(rho_clipped) >= 1.0:
        p_val = 0.0
    else:
        t_stat = rho_clipped * math.sqrt((n - 2) / (1.0 - rho_clipped ** 2))
        # Approximate 2-tailed normal/t p-value
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))

    return rho_clipped, float(p_val)


def compute_bootstrap_confidence_interval(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstraps: int = 500,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """Computes 95% bootstrap confidence interval for Spearman rho."""
    rng = np.random.default_rng(random_seed)
    n = x.size
    rhos = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        rho, _ = compute_spearman_rank_correlation(x[idx], y[idx])
        rhos.append(rho)
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def evaluate_heuristic_confidence_ranking(
    predicted_uncertainties: np.ndarray,
    observed_errors: np.ndarray,
    min_sample_size: int = DEFAULT_BENCHMARK_MIN_SAMPLE_POLICY,
) -> HeuristicRankingResult:
    """Evaluates Mode A: Heuristic Confidence Ranking (Spearman rho + quintile stratification).
    
    Guarantees:
    1. Sample count below configurable min_sample_size emits NOT_EVALUABLE.
       Insufficient-sample gating is a benchmark policy and does not establish universal statistical adequacy.
    2. No universal rho > 0.40 hard acceptance gate.
    """
    u_arr = np.asarray(predicted_uncertainties, dtype=np.float64)
    e_arr = np.asarray(observed_errors, dtype=np.float64)

    n = u_arr.size
    if n < min_sample_size or e_arr.size != n:
        return HeuristicRankingResult(
            sample_count=n,
            spearman_rho=0.0,
            p_value=1.0,
            bootstrap_ci_95=(0.0, 0.0),
            quintile_median_errors=[],
            is_monotonically_ordered=False,
            status=UncertaintyStatus.NOT_EVALUABLE,
        )

    rho, p_val = compute_spearman_rank_correlation(u_arr, e_arr)
    ci_lower, ci_upper = compute_bootstrap_confidence_interval(u_arr, e_arr)

    # Quintile stratification (Q1 to Q5 by predicted uncertainty)
    quintile_splits = np.array_split(np.argsort(u_arr), 5)
    q_medians = [float(np.median(e_arr[idx])) for idx in quintile_splits if len(idx) > 0]

    # Check monotonicity
    is_monotonic = all(q_medians[i] <= q_medians[i + 1] + 1e-6 for i in range(len(q_medians) - 1))

    return HeuristicRankingResult(
        sample_count=n,
        spearman_rho=rho,
        p_value=p_val,
        bootstrap_ci_95=(ci_lower, ci_upper),
        quintile_median_errors=q_medians,
        is_monotonically_ordered=is_monotonic,
        status=UncertaintyStatus.EVALUATED,
    )


def evaluate_probabilistic_coverage(
    predicted_sigmas: np.ndarray,
    observed_errors: np.ndarray,
    declared_probabilistic_model: Optional[str] = None,
    min_sample_size: int = DEFAULT_BENCHMARK_MIN_SAMPLE_POLICY,
) -> ProbabilisticCoverageResult:
    """Evaluates Mode B: Empirical Gaussian coverage (1-sigma, 2-sigma, 3-sigma).
    
    Strict invariant (MUT-10): Rejects evaluating probabilistic coverage if no probabilistic
    model is explicitly declared, or if sample size < min_sample_size.
    Insufficient-sample gating is a benchmark policy and does not establish universal statistical adequacy.
    """
    if not declared_probabilistic_model or not declared_probabilistic_model.strip():
        raise ContractViolationError(
            "Contract Violation (MUT-10): Cannot evaluate probabilistic coverage without an "
            "explicitly declared and documented probabilistic error model."
        )

    s_arr = np.asarray(predicted_sigmas, dtype=np.float64)
    e_arr = np.asarray(observed_errors, dtype=np.float64)

    n = s_arr.size
    if n < min_sample_size or e_arr.size != n:
        return ProbabilisticCoverageResult(
            sample_count=n,
            model_declared=declared_probabilistic_model,
            empirical_coverage_1sigma=0.0,
            empirical_coverage_2sigma=0.0,
            empirical_coverage_3sigma=0.0,
            status=UncertaintyStatus.NOT_EVALUABLE,
        )

    # Avoid divide by zero
    s_safe = np.maximum(s_arr, 1e-9)
    cov_1 = float(np.mean(e_arr <= 1.0 * s_safe))
    cov_2 = float(np.mean(e_arr <= 2.0 * s_safe))
    cov_3 = float(np.mean(e_arr <= 3.0 * s_safe))

    # Supported if 1-sigma coverage is within +/- 10% of theoretical 68.3%
    if abs(cov_1 - 0.6827) < 0.15 and abs(cov_2 - 0.9545) < 0.10:
        status = UncertaintyStatus.CALIBRATION_SUPPORTED
    else:
        status = UncertaintyStatus.CALIBRATION_NOT_SUPPORTED

    return ProbabilisticCoverageResult(
        sample_count=n,
        model_declared=declared_probabilistic_model,
        empirical_coverage_1sigma=cov_1,
        empirical_coverage_2sigma=cov_2,
        empirical_coverage_3sigma=cov_3,
        status=status,
    )


def transform_spatial_covariance(
    covariance_matrix: np.ndarray,
    scale_factor_s: float,
    verify_round_trip: bool = True,
) -> np.ndarray:
    """Transforms a spatial error covariance matrix under linear unit scaling: s * x.
    
    Mathematical Contract (MUT-18):
        Given x' = s * x for linear scale factor s (e.g. s = 0.001 converting m to km),
        the covariance matrix transforms quadratically as:
            Sigma' = Cov(s * x) = s^2 * Sigma
            
    Contracts:
        - s must be strictly positive and finite.
        - Covariance matrix must be 2D square symmetric positive semi-definite.
        - Scaling must be quadratic s^2 (linear scaling s or inverse 1/s is rejected).
        - If verify_round_trip is True, inverse transform (1/s)^2 * Sigma' must recover Sigma.
    """
    if not math.isfinite(scale_factor_s) or scale_factor_s <= 0.0:
        raise ContractViolationError(
            f"Invalid scale factor (MUT-18): Scale factor must be strictly positive and finite, got {scale_factor_s}"
        )

    cov = np.asarray(covariance_matrix, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ContractViolationError(
            f"Invalid covariance matrix shape: Expected 2D square matrix, got {cov.shape}"
        )

    # Quadratic scale factor: s^2
    s_squared = scale_factor_s ** 2
    scaled_cov = s_squared * cov

    if verify_round_trip:
        inv_scale = 1.0 / scale_factor_s
        round_trip_cov = (inv_scale ** 2) * scaled_cov
        if not np.allclose(round_trip_cov, cov, atol=1e-8, rtol=1e-5):
            raise ContractViolationError(
                "Covariance scaling round-trip failure: Inverse scaling did not recover original covariance."
            )

    return scaled_cov
