"""Scale observability and geometric trajectory degeneracy verification.

Checks:
1. Dimensionless Normalized Trajectory Dispersion:
       D_rel = 0.0 if D_max == 0 else RMS(||C_i - C_bar||) / D_max
   Rejects D_max == 0 or D_rel < tau_disp_dimless (1e-6) without dimensional epsilons.

2. Dimensionless Collinearity Eigenvalue Ratio:
       lambda_0 <= lambda_1 <= lambda_2
   Rejects lambda_1 / lambda_2 < tau_collinear (1e-4).

3. Physical Geospatial Metric Baseline Span:
       B_gnss = max_{i, j} ||z_i - z_j||_2
   Rejects B_gnss < tau_min_baseline_m (10.0m in physical ENU).

4. Minimum Inlier Count:
   Rejects M_inliers < 4.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import List, Tuple, Optional
import numpy as np


class FullSim3ObservabilityStatus(str, Enum):
    """Observability state of the full 7-DoF Sim(3) transformation."""
    FULL_SIM3_OBSERVABLE = "FULL_SIM3_OBSERVABLE"
    FULL_SIM3_NOT_OBSERVABLE_COLLINEAR = "FULL_SIM3_NOT_OBSERVABLE_COLLINEAR"
    FULL_SIM3_NOT_OBSERVABLE_STATIONARY = "FULL_SIM3_NOT_OBSERVABLE_STATIONARY"


@dataclass
class ScaleObservabilityReport:
    """Rigorous audit report of trajectory geometry and scale observability."""

    scale_observable: bool
    full_sim3_observability: FullSim3ObservabilityStatus
    dispersion_rel: float
    d_max: float
    collinearity_ratio: float
    eigenvalues: Tuple[float, float, float]
    gnss_baseline_span_m: float
    inlier_count: int
    is_collinear: bool = False
    scale_failure_reasons: List[str] = field(default_factory=list)
    pose_warnings: List[str] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)

    @property
    def is_observable(self) -> bool:
        """Backward-compatible alias for scale_observable."""
        return self.scale_observable


def check_scale_observability(
    camera_centers_rec: np.ndarray,
    gnss_positions_enu: np.ndarray,
    tau_disp_dimless: float = 1e-6,
    tau_collinear: float = 1e-4,
    tau_min_baseline_m: float = 10.0,
    min_inlier_count: int = 4,
) -> ScaleObservabilityReport:
    """Evaluate geometric observability of metric scale from camera trajectory and GNSS.
    
    Args:
        camera_centers_rec: (M, 3) camera optical centers in reconstruction units.
        gnss_positions_enu: (M, 3) corresponding GNSS antenna positions in local ENU meters.
        tau_disp_dimless: Dimensionless cutoff on D_rel (default 1e-6).
        tau_collinear: Dimensionless cutoff on lambda_1 / lambda_2 (default 1e-4).
        tau_min_baseline_m: Metric cutoff on GNSS baseline span (default 10.0m).
        min_inlier_count: Minimum required inlier points (default 4).
        
    Returns:
        ScaleObservabilityReport with all evaluated quantities and pass/fail verdict.
    """
    c_rec = np.asarray(camera_centers_rec, dtype=np.float64)
    z_enu = np.asarray(gnss_positions_enu, dtype=np.float64)

    m = c_rec.shape[0]
    scale_failure_reasons: List[str] = []
    pose_warnings: List[str] = []
    is_collinear = False

    # 1. Point count check
    if m < min_inlier_count:
        scale_failure_reasons.append(f"INSUFFICIENT_INLIERS: {m} < {min_inlier_count}")

    if m == 0:
        return ScaleObservabilityReport(
            scale_observable=False,
            full_sim3_observability=FullSim3ObservabilityStatus.FULL_SIM3_NOT_OBSERVABLE_STATIONARY,
            dispersion_rel=0.0,
            d_max=0.0,
            collinearity_ratio=0.0,
            eigenvalues=(0.0, 0.0, 0.0),
            gnss_baseline_span_m=0.0,
            inlier_count=0,
            is_collinear=False,
            scale_failure_reasons=scale_failure_reasons,
            pose_warnings=pose_warnings,
            failure_reasons=scale_failure_reasons,
        )

    # 2. Dimensionless Normalized Trajectory Dispersion
    c_bar = np.mean(c_rec, axis=0)
    # Compute maximum pairwise distance D_max
    if m > 1:
        diffs = c_rec[:, np.newaxis, :] - c_rec[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        d_max = float(np.max(dists))
    else:
        d_max = 0.0

    rms_dist = float(np.sqrt(np.mean(np.sum((c_rec - c_bar) ** 2, axis=-1))))

    # Exact piecewise dimensionless definition
    if d_max == 0.0:
        d_rel = 0.0
    else:
        d_rel = rms_dist / d_max

    if d_max == 0.0 or d_rel < tau_disp_dimless:
        scale_failure_reasons.append(
            f"SCALE_NOT_OBSERVABLE_STATIONARY: D_max={d_max:.2e}, D_rel={d_rel:.2e} < {tau_disp_dimless:.2e}"
        )

    # 3. Dimensionless Collinearity Eigenvalue Ratio (Affects Full Sim(3) Pose, NOT Scale)
    if m >= 3:
        cov = np.cov(c_rec, rowvar=False, bias=True)
        evals = np.linalg.eigvalsh(cov)
        evals_sorted = np.sort(np.maximum(0.0, evals))
        l0, l1, l2 = float(evals_sorted[0]), float(evals_sorted[1]), float(evals_sorted[2])

        if l2 > 0:
            collinearity_ratio = l1 / l2
        else:
            collinearity_ratio = 0.0

        if collinearity_ratio < tau_collinear:
            is_collinear = True
            pose_warnings.append(
                f"FULL_SIM3_NOT_OBSERVABLE_COLLINEAR: lambda_1/lambda_2={collinearity_ratio:.2e} < {tau_collinear:.2e}; "
                "rotation around trajectory axis is underconstrained from camera positions alone."
            )
    else:
        l0, l1, l2 = 0.0, 0.0, 0.0
        collinearity_ratio = 0.0
        if m < 3:
            is_collinear = True
            pose_warnings.append("FULL_SIM3_NOT_OBSERVABLE_COLLINEAR: Fewer than 3 cameras")

    # 4. Physical Geospatial Metric Baseline Span
    if z_enu.shape[0] > 1:
        z_diffs = z_enu[:, np.newaxis, :] - z_enu[np.newaxis, :, :]
        z_dists = np.linalg.norm(z_diffs, axis=-1)
        gnss_baseline_span_m = float(np.max(z_dists))
    else:
        gnss_baseline_span_m = 0.0

    if gnss_baseline_span_m < tau_min_baseline_m:
        scale_failure_reasons.append(
            f"INSUFFICIENT_PHYSICAL_BASELINE: B_gnss={gnss_baseline_span_m:.2f}m < {tau_min_baseline_m:.2f}m"
        )

    scale_observable = len(scale_failure_reasons) == 0

    if not scale_observable:
        full_sim3_status = FullSim3ObservabilityStatus.FULL_SIM3_NOT_OBSERVABLE_STATIONARY
    elif is_collinear:
        full_sim3_status = FullSim3ObservabilityStatus.FULL_SIM3_NOT_OBSERVABLE_COLLINEAR
    else:
        full_sim3_status = FullSim3ObservabilityStatus.FULL_SIM3_OBSERVABLE

    return ScaleObservabilityReport(
        scale_observable=scale_observable,
        full_sim3_observability=full_sim3_status,
        dispersion_rel=d_rel,
        d_max=d_max,
        collinearity_ratio=collinearity_ratio,
        eigenvalues=(l0, l1, l2),
        gnss_baseline_span_m=gnss_baseline_span_m,
        inlier_count=m,
        is_collinear=is_collinear,
        scale_failure_reasons=scale_failure_reasons,
        pose_warnings=pose_warnings,
        failure_reasons=scale_failure_reasons,
    )
