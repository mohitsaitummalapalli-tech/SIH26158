"""Reconstruction module: Dense point clouds, surface meshes, and texture projection."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class PointCloudMetadata:
    """Metadata describing a dense 3D point cloud asset."""
    num_points: int
    has_colors: bool = True
    has_normals: bool = False
    has_uncertainty: bool = False
    bounding_box_min: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bounding_box_max: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    coordinate_frame: str = "local_model"  # "local_model" or "utm_metric"
    point_density_per_m2: Optional[float] = None


@dataclass
class MeshMetadata:
    """Metadata describing a 3D surface mesh asset."""
    num_vertices: int
    num_faces: int
    has_texture: bool = False
    texture_resolution: Optional[List[int]] = None
    is_watertight: bool = False
    format: str = "obj"  # "obj", "gltf", "glb", "ply"


@dataclass
class ReconstructionConfig:
    """Configuration for dense reconstruction and meshing pipeline."""
    target_engine: str = "fusion"  # "baseline_sfm", "dust3r", "vggt", "fusion"
    meshing_algorithm: str = "poisson"  # "poisson", "tsdf", "alpha_shape"
    poisson_depth: int = 10
    enable_texture_projection: bool = True
    texture_size: int = 4096
    outlier_removal_k_neighbors: int = 20
    outlier_std_ratio: float = 2.0
