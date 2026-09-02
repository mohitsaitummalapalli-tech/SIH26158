"""Fusion module: Multi-view pointmap alignment and graph optimization contracts."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class FusionNode:
    """Node in the multi-view reconstruction graph representing a keyframe."""
    frame_index: int
    estimated_pose_id: int
    telemetry_prior_id: Optional[int] = None
    pointmap_count: int = 0


@dataclass
class FusionEdge:
    """Edge in the multi-view graph representing a pairwise geometric constraint."""
    source_frame_idx: int
    target_frame_idx: int
    relative_scale: float
    confidence_weight: float
    num_inlier_matches: int


@dataclass
class FusionGraphConfig:
    """Configuration for global pointmap and pose graph optimization."""
    subgraph_window_size: int = 20
    overlap_frames: int = 5
    max_optimization_iterations: int = 100
    regularize_with_gnss: bool = True
    gnss_weight: float = 1.0
