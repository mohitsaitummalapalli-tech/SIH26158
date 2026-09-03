"""Phase 3E.6 Completeness Metrology & Evidence-Gated Visibility Taxonomy.

Evaluates bounded Region of Interest (ROI) precision and recall/completeness,
surface-area coverage, and 5-state visibility evidence tagging.
Rejects ungrounded occlusion classification without optical ray-tracing evidence (MUT-14)
and unbounded completeness claims without explicit reference ROI (MUT-15).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.benchmark.models import (
    VisibilityState,
    ContractViolationError,
)
from src.benchmark.metrics_geometry import _nearest_neighbors, _filter_roi


@dataclass(frozen=True)
class CompletenessEvaluationResult:
    """Completeness and coverage results within pre-registered evaluation ROI."""
    roi_bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    precision_at_tau: float
    recall_completeness_at_tau: float
    f1_score_at_tau: float
    surface_area_completeness: Optional[float]
    estimated_point_count_in_roi: int
    reference_point_count_in_roi: int
    tau_meters: float
    visibility_state_distribution: Dict[str, int]


def classify_visibility_evidence(
    has_optical_ray_intersection: bool,
    ray_intersection_angle_deg: float,
    in_camera_frustum: bool,
    ray_hits_foreground: bool,
    has_ray_tracing_evidence: bool,
    is_reconstructed: bool,
) -> VisibilityState:
    """Classifies visibility of a scene point under the 5-state evidence taxonomy.
    
    Strict invariant (MUT-14): PHYSICALLY_OCCLUDED may ONLY be emitted when explicit
    multi-view ray-tracing or depth map evidence is present. Otherwise, missing points
    are classified as UNDETERMINED.
    """
    if is_reconstructed:
        if has_optical_ray_intersection and ray_intersection_angle_deg >= 5.0:
            return VisibilityState.OBSERVED
        return VisibilityState.OBSERVED

    # Point is NOT reconstructed:
    if not in_camera_frustum:
        return VisibilityState.UNOBSERVED

    # Point was in camera frustum but not reconstructed:
    if not has_ray_tracing_evidence:
        # Without ray-tracing proof, it is mathematically forbidden to claim occlusion!
        return VisibilityState.UNDETERMINED

    if ray_hits_foreground:
        return VisibilityState.PHYSICALLY_OCCLUDED
    else:
        return VisibilityState.RECONSTRUCTION_MISSING


def evaluate_roi_completeness(
    points_est: np.ndarray,
    points_gt: np.ndarray,
    tau_meters: float,
    roi_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
    surface_area_est_roi: Optional[float] = None,
    surface_area_ref_roi: Optional[float] = None,
) -> CompletenessEvaluationResult:
    """Evaluates geometric completeness within a strictly bounded Region of Interest (ROI).
    
    Guarantees:
    1. Rejects unbounded completeness claims without explicit reference ROI (MUT-15).
    """
    if roi_bounds is None:
        raise ContractViolationError(
            "Contract Violation (MUT-15): Completeness cannot be evaluated on an unbounded scene. "
            "An explicit, bounded Region of Interest (ROI) is mandatory."
        )

    pts_est = _filter_roi(np.asarray(points_est, dtype=np.float64), roi_bounds)
    pts_gt = _filter_roi(np.asarray(points_gt, dtype=np.float64), roi_bounds)

    n_est = pts_est.shape[0]
    n_gt = pts_gt.shape[0]

    if n_gt == 0:
        raise ValueError("Zero reference points found inside the evaluation ROI.")

    if n_est == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        d_est_to_gt, _ = _nearest_neighbors(pts_est, pts_gt)
        d_gt_to_est, _ = _nearest_neighbors(pts_gt, pts_est)
        precision = float(np.mean(d_est_to_gt <= tau_meters))
        recall = float(np.mean(d_gt_to_est <= tau_meters))
        f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall > 1e-12 else 0.0

    # Surface-area completeness if CAD/continuous surfaces provided
    area_comp = None
    if surface_area_est_roi is not None and surface_area_ref_roi is not None:
        if surface_area_ref_roi > 0.0:
            area_comp = float(np.clip(surface_area_est_roi / surface_area_ref_roi, 0.0, 1.0))

    return CompletenessEvaluationResult(
        roi_bounds=roi_bounds,
        precision_at_tau=precision,
        recall_completeness_at_tau=recall,
        f1_score_at_tau=f1,
        surface_area_completeness=area_comp,
        estimated_point_count_in_roi=n_est,
        reference_point_count_in_roi=n_gt,
        tau_meters=tau_meters,
        visibility_state_distribution={},
    )
