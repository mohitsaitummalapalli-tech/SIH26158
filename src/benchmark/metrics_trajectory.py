"""Phase 3E.6 Camera Trajectory Validation Metrics.

Implements three strictly separated trajectory evaluation protocols:
1. Raw / Metric-Frame Absolute Trajectory Error (ATE) - preserves scale and georeferencing.
2. Sim(3)-Aligned Trajectory Error - relative shape diagnostic only (mandates 7 DoF logging + disclaimer).
3. Relative Pose Error (RPE) - localized translational and rotational drift per delta interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.benchmark.models import (
    StatisticalSummary,
    ContractViolationError,
)
from src.benchmark.metrics_geometry import compute_statistical_summary

DISCLAIMER_SIM3_ALIGNMENT = (
    "Similarity alignment removes global similarity degrees of freedom (7 DoF: scale, 3D rotation, "
    "3D translation) and therefore must not be interpreted as absolute metric/geospatial accuracy."
)


def _rotation_angle_deg(r_diff: np.ndarray) -> float:
    """Computes geodesic angle in degrees from a 3x3 rotation matrix."""
    tr = float(np.trace(r_diff))
    val = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(val) * (180.0 / np.pi))


@dataclass(frozen=True)
class RawTrajectoryResult:
    """Evaluation in the true metric/geospatial frame without any alignment fitting."""
    frame_count: int
    ate_translation_rmse_m: float
    ate_translation_summary_m: StatisticalSummary
    ate_rotation_rmse_deg: float
    ate_rotation_summary_deg: StatisticalSummary
    alignment_applied: str = "NONE"
    preserves_absolute_scale_and_georeference: bool = True


@dataclass(frozen=True)
class Sim3AlignedTrajectoryResult:
    """Trajectory shape consistency diagnostic with global Sim(3) parameters removed."""
    frame_count: int
    aligned_ate_rmse_m: float
    aligned_ate_summary_m: StatisticalSummary
    scale_removed: float
    rotation_removed: List[List[float]]
    translation_removed: List[float]
    alignment_removed_dofs: int = 7
    alignment_type: str = "Sim3"
    disclaimer: str = DISCLAIMER_SIM3_ALIGNMENT


@dataclass(frozen=True)
class RpeDriftResult:
    """Relative pose error drift over temporal delta interval."""
    delta_interval_frames: int
    pair_count: int
    translational_drift_per_delta_rmse: float
    translational_drift_summary: StatisticalSummary
    rotational_drift_per_delta_rmse_deg: float
    rotational_drift_summary_deg: StatisticalSummary


def evaluate_raw_trajectory_ate(
    camera_centers_est: np.ndarray,
    camera_rotations_est: np.ndarray,
    camera_centers_ref: np.ndarray,
    camera_rotations_ref: np.ndarray,
) -> RawTrajectoryResult:
    """Evaluates raw trajectory ATE without any alignment. Preserves true scale and georeference."""
    c_est = np.asarray(camera_centers_est, dtype=np.float64)
    r_est = np.asarray(camera_rotations_est, dtype=np.float64)
    c_ref = np.asarray(camera_centers_ref, dtype=np.float64)
    r_ref = np.asarray(camera_rotations_ref, dtype=np.float64)

    n = c_est.shape[0]
    if n == 0 or c_ref.shape[0] != n:
        raise ValueError(f"Mismatched or empty trajectory lengths: est={n}, ref={c_ref.shape[0]}")

    trans_errors = np.linalg.norm(c_est - c_ref, axis=1)

    rot_errors = []
    for i in range(n):
        r_diff = r_est[i].T @ r_ref[i]
        rot_errors.append(_rotation_angle_deg(r_diff))
    rot_errors_arr = np.array(rot_errors, dtype=np.float64)

    t_summary = compute_statistical_summary(trans_errors, unit="meters")
    r_summary = compute_statistical_summary(rot_errors_arr, unit="degrees")

    return RawTrajectoryResult(
        frame_count=n,
        ate_translation_rmse_m=t_summary.rmse,
        ate_translation_summary_m=t_summary,
        ate_rotation_rmse_deg=r_summary.rmse,
        ate_rotation_summary_deg=r_summary,
    )


def solve_umeyama_sim3(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Solves best-fit Sim(3) transformation dst ~ s * R @ src + t using Umeyama algorithm."""
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    n, m = src.shape

    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = np.sum(src_c ** 2) / n

    cov = (dst_c.T @ src_c) / n

    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[-1, -1] = -1

    r_opt = u @ s_mat @ vt
    scale_opt = (1.0 / var_src) * np.sum(d * np.diag(s_mat))
    t_opt = mu_dst - scale_opt * (r_opt @ mu_src)

    return float(scale_opt), r_opt, t_opt


def evaluate_sim3_aligned_trajectory_ate(
    camera_centers_est: np.ndarray,
    camera_centers_ref: np.ndarray,
    suppress_disclaimer: bool = False,
) -> Sim3AlignedTrajectoryResult:
    """Evaluates relative trajectory shape error using Umeyama Sim(3) alignment.
    
    Mandates recording scale_removed, rotation_removed, translation_removed, and the disclaimer.
    """
    if suppress_disclaimer:
        # Detected adversarial attempt to hide alignment (MUT-09)
        raise ContractViolationError(
            "Adversarial mutation detected (MUT-09): Attempting to suppress Sim(3) alignment disclaimer!"
        )

    c_est = np.asarray(camera_centers_est, dtype=np.float64)
    c_ref = np.asarray(camera_centers_ref, dtype=np.float64)

    n = c_est.shape[0]
    if n < 3:
        raise ValueError("Sim(3) trajectory alignment requires at least 3 non-collinear camera centers.")

    scale_opt, r_opt, t_opt = solve_umeyama_sim3(c_est, c_ref)

    # Transform estimated centers
    c_aligned = (scale_opt * (r_opt @ c_est.T)).T + t_opt

    aligned_errors = np.linalg.norm(c_aligned - c_ref, axis=1)
    stat_summary = compute_statistical_summary(aligned_errors, unit="meters")

    return Sim3AlignedTrajectoryResult(
        frame_count=n,
        aligned_ate_rmse_m=stat_summary.rmse,
        aligned_ate_summary_m=stat_summary,
        scale_removed=scale_opt,
        rotation_removed=r_opt.tolist(),
        translation_removed=t_opt.tolist(),
        disclaimer=DISCLAIMER_SIM3_ALIGNMENT,
    )


def evaluate_rpe_drift(
    camera_centers_est: np.ndarray,
    camera_rotations_est: np.ndarray,
    camera_centers_ref: np.ndarray,
    camera_rotations_ref: np.ndarray,
    delta_interval_frames: int = 1,
) -> RpeDriftResult:
    """Evaluates localized Relative Pose Error (RPE) drift over temporal delta interval."""
    c_est = np.asarray(camera_centers_est, dtype=np.float64)
    r_est = np.asarray(camera_rotations_est, dtype=np.float64)
    c_ref = np.asarray(camera_centers_ref, dtype=np.float64)
    r_ref = np.asarray(camera_rotations_ref, dtype=np.float64)

    n = c_est.shape[0]
    step = delta_interval_frames
    if step <= 0 or step >= n:
        raise ValueError(f"Delta interval ({step}) must be in range [1, {n - 1}]")

    trans_drift = []
    rot_drift = []

    for i in range(n - step):
        j = i + step

        # Relative transformation reference: T_ref_i^{-1} * T_ref_j
        rel_c_ref = c_ref[j] - c_ref[i]
        rel_r_ref = r_ref[i].T @ r_ref[j]

        # Relative transformation estimated: T_est_i^{-1} * T_est_j
        rel_c_est = c_est[j] - c_est[i]
        rel_r_est = r_est[i].T @ r_est[j]

        # Discrepancy
        d_trans = np.linalg.norm(rel_c_est - rel_c_ref)
        d_rot = _rotation_angle_deg(rel_r_est.T @ rel_r_ref)

        trans_drift.append(d_trans)
        rot_drift.append(d_rot)

    arr_trans = np.array(trans_drift, dtype=np.float64)
    arr_rot = np.array(rot_drift, dtype=np.float64)

    stat_trans = compute_statistical_summary(arr_trans, unit="meters/delta")
    stat_rot = compute_statistical_summary(arr_rot, unit="degrees/delta")

    return RpeDriftResult(
        delta_interval_frames=step,
        pair_count=len(trans_drift),
        translational_drift_per_delta_rmse=stat_trans.rmse,
        translational_drift_summary=stat_trans,
        rotational_drift_per_delta_rmse_deg=stat_rot.rmse,
        rotational_drift_summary_deg=stat_rot,
    )
