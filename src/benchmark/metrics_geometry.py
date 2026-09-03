"""Phase 3E.6 Geometric Metrology Metrics.

Implements rigorous point-to-point, point-to-plane, bidirectional Chamfer,
Hausdorff (max and 95th percentile), precision/recall/F1 at tau, and normal angular deviation.
Always reports five-point statistical summaries (MAE, RMSE, median, P95, maximum).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.benchmark.models import (
    BenchmarkStatus,
    StatisticalSummary,
    ContractViolationError,
)
from src.benchmark.claim_policy import ClaimPolicyEngine


def compute_statistical_summary(
    errors: np.ndarray,
    unit: str = "meters",
) -> StatisticalSummary:
    """Computes full five-point statistical summary. Never reports only a mean."""
    arr = np.asarray(errors, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("Cannot compute statistical summary on an empty error array.")
    
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError("Non-finite values encountered in error distribution.")
    
    if np.any(arr < -1e-9):
        raise ValueError(f"Negative error encountered in distance metric: min={np.min(arr)}")

    arr_clean = np.maximum(arr, 0.0)
    mae = float(np.mean(arr_clean))
    rmse = float(np.sqrt(np.mean(arr_clean ** 2)))
    median = float(np.median(arr_clean))
    p95 = float(np.percentile(arr_clean, 95.0))
    maximum = float(np.max(arr_clean))

    return StatisticalSummary(
        mae=mae,
        rmse=rmse,
        median=median,
        p95=p95,
        maximum=maximum,
        sample_count=int(arr.size),
        unit=unit,
    )


def _filter_roi(
    points: np.ndarray,
    roi_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
) -> np.ndarray:
    """Filters points to those within a 3D bounding box ROI: (min_xyz, max_xyz)."""
    if roi_bounds is None:
        return points
    (min_x, min_y, min_z), (max_x, max_y, max_z) = roi_bounds
    mask = (
        (points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
        (points[:, 1] >= min_y) & (points[:, 1] <= max_y) &
        (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
    )
    return points[mask]


def _nearest_neighbors(
    queries: np.ndarray,
    references: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds nearest neighbor Euclidean distances and indices for queries into references.
    
    Uses scipy.spatial.cKDTree if available; falls back to vectorized numpy broadcasting.
    """
    if queries.size == 0 or references.size == 0:
        raise ValueError("Queries or references set is empty.")

    try:
        import scipy.spatial as spatial
        tree = spatial.KDTree(references)
        dists, indices = tree.query(queries, k=1)
        return np.asarray(dists, dtype=np.float64), np.asarray(indices, dtype=np.int64)
    except (ImportError, AttributeError):
        # Vectorized chunked fallback
        n_queries = queries.shape[0]
        chunk_size = 500
        all_dists = []
        all_indices = []
        for i in range(0, n_queries, chunk_size):
            q_chunk = queries[i:i + chunk_size]
            diff = q_chunk[:, np.newaxis, :] - references[np.newaxis, :, :]
            d2 = np.sum(diff ** 2, axis=-1)
            idx = np.argmin(d2, axis=1)
            dist = np.sqrt(d2[np.arange(len(q_chunk)), idx])
            all_dists.append(dist)
            all_indices.append(idx)
        return np.concatenate(all_dists), np.concatenate(all_indices)


def compute_point_to_point_distances(
    points_est: np.ndarray,
    points_gt: np.ndarray,
    roi_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
    est_hash: Optional[str] = None,
    gt_hash: Optional[str] = None,
) -> StatisticalSummary:
    """Computes point-to-point Euclidean distances from estimated points to reference."""
    ClaimPolicyEngine.verify_no_self_evaluation(est_hash or "", gt_hash or "", points_est, points_gt)
    
    pts_est = _filter_roi(np.asarray(points_est, dtype=np.float64), roi_bounds)
    pts_gt = _filter_roi(np.asarray(points_gt, dtype=np.float64), roi_bounds)

    if pts_est.shape[0] == 0 or pts_gt.shape[0] == 0:
        raise ValueError("Estimated or reference cloud has zero points inside the evaluation ROI.")

    dists, _ = _nearest_neighbors(pts_est, pts_gt)
    return compute_statistical_summary(dists, unit="meters")


def compute_point_to_plane_distances(
    points_est: np.ndarray,
    points_gt: np.ndarray,
    normals_gt: np.ndarray,
    roi_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
    est_hash: Optional[str] = None,
    gt_hash: Optional[str] = None,
) -> StatisticalSummary:
    """Computes point-to-plane distances |(p - q*) . n*| using certified reference normals."""
    ClaimPolicyEngine.verify_no_self_evaluation(est_hash or "", gt_hash or "", points_est, points_gt)

    pts_est = _filter_roi(np.asarray(points_est, dtype=np.float64), roi_bounds)
    pts_gt = np.asarray(points_gt, dtype=np.float64)
    norms_gt = np.asarray(normals_gt, dtype=np.float64)

    if pts_est.shape[0] == 0 or pts_gt.shape[0] == 0:
        raise ValueError("Estimated or reference cloud is empty.")

    if pts_gt.shape[0] != norms_gt.shape[0]:
        raise ValueError("Number of ground-truth points must match number of ground-truth normals.")

    dists, nn_idx = _nearest_neighbors(pts_est, pts_gt)
    q_stars = pts_gt[nn_idx]
    n_stars = norms_gt[nn_idx]

    # Normalize normals to unit length
    n_lens = np.linalg.norm(n_stars, axis=1, keepdims=True)
    n_stars_norm = np.where(n_lens > 1e-12, n_stars / n_lens, n_stars)

    p2pl = np.abs(np.sum((pts_est - q_stars) * n_stars_norm, axis=1))
    return compute_statistical_summary(p2pl, unit="meters")


def compute_bidirectional_chamfer(
    points_est: np.ndarray,
    points_gt: np.ndarray,
    roi_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
    est_hash: Optional[str] = None,
    gt_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Computes Bidirectional Chamfer distance.
    
    Strict invariant: Must evaluate both forward (est -> gt) and backward (gt -> est).
    """
    ClaimPolicyEngine.verify_no_self_evaluation(est_hash or "", gt_hash or "", points_est, points_gt)

    pts_est = _filter_roi(np.asarray(points_est, dtype=np.float64), roi_bounds)
    pts_gt = _filter_roi(np.asarray(points_gt, dtype=np.float64), roi_bounds)

    if pts_est.shape[0] == 0 or pts_gt.shape[0] == 0:
        raise ValueError("Empty point cloud in bidirectional Chamfer evaluation.")

    forward_dists, _ = _nearest_neighbors(pts_est, pts_gt)
    backward_dists, _ = _nearest_neighbors(pts_gt, pts_est)

    forward_mean = float(np.mean(forward_dists))
    backward_mean = float(np.mean(backward_dists))
    bidirectional_chamfer = 0.5 * (forward_mean + backward_mean)

    all_dists = np.concatenate([forward_dists, backward_dists])

    return {
        "chamfer_distance": bidirectional_chamfer,
        "forward_summary": compute_statistical_summary(forward_dists, unit="meters"),
        "backward_summary": compute_statistical_summary(backward_dists, unit="meters"),
        "combined_summary": compute_statistical_summary(all_dists, unit="meters"),
        "is_bidirectional": True,
    }


def compute_hausdorff_distances(
    points_est: np.ndarray,
    points_gt: np.ndarray,
    roi_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
) -> Dict[str, float]:
    """Computes directed, bidirectional maximum, and 95th percentile Hausdorff distances."""
    pts_est = _filter_roi(np.asarray(points_est, dtype=np.float64), roi_bounds)
    pts_gt = _filter_roi(np.asarray(points_gt, dtype=np.float64), roi_bounds)

    if pts_est.shape[0] == 0 or pts_gt.shape[0] == 0:
        raise ValueError("Empty point cloud in Hausdorff distance computation.")

    d_est_to_gt, _ = _nearest_neighbors(pts_est, pts_gt)
    d_gt_to_est, _ = _nearest_neighbors(pts_gt, pts_est)

    dir_est_to_gt = float(np.max(d_est_to_gt))
    dir_gt_to_est = float(np.max(d_gt_to_est))
    hausdorff_max = max(dir_est_to_gt, dir_gt_to_est)

    p95_est_to_gt = float(np.percentile(d_est_to_gt, 95.0))
    p95_gt_to_est = float(np.percentile(d_gt_to_est, 95.0))
    hausdorff_95 = max(p95_est_to_gt, p95_gt_to_est)

    return {
        "hausdorff_directed_est_to_gt": dir_est_to_gt,
        "hausdorff_directed_gt_to_est": dir_gt_to_est,
        "hausdorff_max": hausdorff_max,
        "hausdorff_95": hausdorff_95,
    }


def compute_f_score_at_tau(
    points_est: np.ndarray,
    points_gt: np.ndarray,
    tau_meters: float,
    roi_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
) -> Dict[str, float]:
    """Computes Precision(tau), Recall(tau), and F1(tau) in the defined ROI."""
    if tau_meters <= 0.0 or math.isnan(tau_meters):
        raise ValueError(f"Distance threshold tau ({tau_meters}) must be positive and finite.")

    pts_est = _filter_roi(np.asarray(points_est, dtype=np.float64), roi_bounds)
    pts_gt = _filter_roi(np.asarray(points_gt, dtype=np.float64), roi_bounds)

    if pts_est.shape[0] == 0 or pts_gt.shape[0] == 0:
        raise ValueError("Empty point cloud in F-score evaluation.")

    d_est_to_gt, _ = _nearest_neighbors(pts_est, pts_gt)
    d_gt_to_est, _ = _nearest_neighbors(pts_gt, pts_est)

    precision = float(np.mean(d_est_to_gt <= tau_meters))
    recall = float(np.mean(d_gt_to_est <= tau_meters))

    if precision + recall > 1e-12:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {
        "tau_meters": tau_meters,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "precision_pct": precision * 100.0,
        "recall_pct": recall * 100.0,
        "f1_score_pct": f1 * 100.0,
    }


def compute_normal_angular_deviation(
    points_est: np.ndarray,
    normals_est: np.ndarray,
    points_gt: np.ndarray,
    normals_gt: np.ndarray,
    roi_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
) -> StatisticalSummary:
    """Computes surface normal angular deviation theta in degrees."""
    pts_est = np.asarray(points_est, dtype=np.float64)
    norms_est = np.asarray(normals_est, dtype=np.float64)
    pts_gt = np.asarray(points_gt, dtype=np.float64)
    norms_gt = np.asarray(normals_gt, dtype=np.float64)

    if pts_est.shape[0] == 0 or pts_gt.shape[0] == 0:
        raise ValueError("Empty point cloud in normal angular deviation evaluation.")

    _, nn_idx = _nearest_neighbors(pts_est, pts_gt)
    n_gt_stars = norms_gt[nn_idx]

    # Normalize vectors
    len_est = np.linalg.norm(norms_est, axis=1, keepdims=True)
    len_gt = np.linalg.norm(n_gt_stars, axis=1, keepdims=True)
    n_est_unit = np.where(len_est > 1e-12, norms_est / len_est, norms_est)
    n_gt_unit = np.where(len_gt > 1e-12, n_gt_stars / len_gt, n_gt_stars)

    # Dot products clipped to [0, 1] (unoriented normals: absolute value)
    dots = np.abs(np.sum(n_est_unit * n_gt_unit, axis=1))
    clipped = np.clip(dots, 0.0, 1.0)
    angles_deg = np.arccos(clipped) * (180.0 / np.pi)

    return compute_statistical_summary(angles_deg, unit="degrees")
