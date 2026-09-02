"""Uncertainty module: Spatial covariance and confidence estimation contracts."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class UncertaintyField:
    """Per-vertex / per-point uncertainty and observation statistics."""
    num_observations: int
    mean_reprojection_error_pixels: float
    triangulation_angle_deg: float
    spatial_covariance_trace: float  # Trace of 3x3 covariance matrix
    neural_confidence_score: float  # [0.0 (extrapolated) -> 1.0 (highly confident)]
    is_observed_geometry: bool = True  # False if hallucinated/inpainted/extrapolated


@dataclass
class UncertaintySummary:
    """Scene-level uncertainty distribution summary."""
    total_points: int
    observed_points_count: int
    extrapolated_points_count: int
    mean_spatial_std_meters: float
    high_confidence_ratio: float  # Ratio of points with confidence > 0.8
