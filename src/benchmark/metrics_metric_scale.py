"""Phase 3E.6 Metric Scale Validation Metrics.

Evaluates independent segment length accuracy and relative scale error.
Guarantees scale validation is strictly distinct from geospatial registration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.benchmark.models import (
    StatisticalSummary,
    ContractViolationError,
)
from src.benchmark.metrics_geometry import compute_statistical_summary


@dataclass(frozen=True)
class ValidationSegment:
    """An independent reference baseline segment between two physical markers."""
    segment_id: str
    point_a_id: str
    point_b_id: str
    reference_distance: float
    unit: str = "meters"

    def __post_init__(self) -> None:
        if self.reference_distance <= 0.0 or math.isnan(self.reference_distance):
            raise ValueError(f"Reference distance for segment {self.segment_id} must be positive, got {self.reference_distance}")


def compute_relative_scale_error(
    estimated_distance: float,
    reference_distance: float,
) -> float:
    """Computes relative scale error: |D_est - D_ref| / D_ref.
    
    Strict invariant (MUT-08): Denominator MUST be reference_distance D_ref.
    Dividing by estimated_distance is mathematically invalid and rejected.
    """
    if reference_distance <= 0.0 or math.isnan(reference_distance):
        raise ValueError(f"Reference distance must be strictly positive, got {reference_distance}")
    if estimated_distance < 0.0 or math.isnan(estimated_distance):
        raise ValueError(f"Estimated distance must be non-negative, got {estimated_distance}")

    return abs(estimated_distance - reference_distance) / reference_distance


def check_segment_collinearity(
    segments_endpoints: List[Tuple[np.ndarray, np.ndarray]],
) -> bool:
    """Checks whether the collection of validation segments spans >= 2 dimensions (non-collinear)."""
    if len(segments_endpoints) < 2:
        return False
    directions = []
    for a, b in segments_endpoints:
        vec = np.asarray(b) - np.asarray(a)
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            directions.append(vec / norm)
    if len(directions) < 2:
        return False
    # SVD on direction vectors
    d_mat = np.array(directions)
    _, s, _ = np.linalg.svd(d_mat - np.mean(d_mat, axis=0))
    # If second singular value is near zero, segments are essentially collinear
    return bool(len(s) >= 2 and s[1] > 1e-3)


def evaluate_metric_scale(
    segments: List[ValidationSegment],
    marker_positions_est: Dict[str, np.ndarray],
    scale_factor_to_metric: float = 1.0,
    enforce_non_collinear: bool = True,
) -> Dict[str, Any]:
    """Evaluates multi-segment metric scale accuracy.
    
    Parameters:
        segments: List of independent validation segments.
        marker_positions_est: Reconstructed 3D positions of markers.
        scale_factor_to_metric: Conversion factor from reconstruction units to reference units (e.g. meters).
        enforce_non_collinear: Whether to verify that segments are non-collinear.
    """
    if len(segments) < 3:
        raise ValueError(f"Metric scale validation requires at least 3 independent segments, got {len(segments)}")

    relative_errors = []
    absolute_errors = []
    segment_endpoints = []
    results = []

    for seg in segments:
        if seg.point_a_id not in marker_positions_est or seg.point_b_id not in marker_positions_est:
            raise KeyError(f"Missing reconstructed marker for segment {seg.segment_id}: ({seg.point_a_id}, {seg.point_b_id})")

        pt_a = np.asarray(marker_positions_est[seg.point_a_id], dtype=np.float64)
        pt_b = np.asarray(marker_positions_est[seg.point_b_id], dtype=np.float64)

        # Distance in reconstruction units converted to metric
        d_est = float(np.linalg.norm(pt_a - pt_b) * scale_factor_to_metric)
        d_ref = seg.reference_distance

        rel_err = compute_relative_scale_error(d_est, d_ref)
        abs_err = abs(d_est - d_ref)

        relative_errors.append(rel_err)
        absolute_errors.append(abs_err)
        segment_endpoints.append((pt_a, pt_b))

        results.append({
            "segment_id": seg.segment_id,
            "d_est": d_est,
            "d_ref": d_ref,
            "abs_error_m": abs_err,
            "relative_error": rel_err,
            "relative_error_pct": rel_err * 100.0,
        })

    is_non_collinear = check_segment_collinearity(segment_endpoints)
    if enforce_non_collinear and not is_non_collinear:
        raise ContractViolationError(
            "Metric scale validation requires non-collinear reference segments to evaluate full 3D isotropic scale."
        )

    rel_arr = np.array(relative_errors, dtype=np.float64)
    abs_arr = np.array(absolute_errors, dtype=np.float64)

    rel_summary = compute_statistical_summary(rel_arr, unit="fraction")
    abs_summary = compute_statistical_summary(abs_arr, unit="meters")

    return {
        "segment_count": len(segments),
        "is_non_collinear": is_non_collinear,
        "relative_scale_error_summary": rel_summary,
        "absolute_error_summary": abs_summary,
        "median_relative_error_pct": rel_summary.median * 100.0,
        "rmse_relative_error_pct": rel_summary.rmse * 100.0,
        "max_relative_error_pct": rel_summary.maximum * 100.0,
        "segment_details": results,
    }
