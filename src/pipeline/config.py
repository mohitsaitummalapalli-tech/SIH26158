"""Phase 3F: Comprehensive Pipeline Configuration & Validation.

Aggregates sub-stage configurations, stage enable/disable gating, determinism
controls, and enforces valid dependency combinations without inventing new API
parameters for locked subsystem components.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

from src.pipeline.errors import ContractViolationError

# Import existing locked subsystem configs where available
from src.quality.keyframe_selection import KeyframeSelectionConfig
from src.geometry.contracts import CameraIntrinsics
from src.geometry.features import FeatureConfig
from src.geometry.two_view import TwoViewConfig
from src.geometry.sfm import SfMConfig
from src.geometry.bundle_adjustment import BundleAdjustmentConfig
from src.geometry.dense_stereo import DenseStereoConfig
from src.geometry.dense_point_generation import DensePointGeneratorConfig
from src.geometry.dense_fusion import DenseFusionConfig, SingleViewRetentionPolicy
from src.geometry.surface_reconstruction import SurfaceReconstructionConfig
from src.geometry.texture_association import TextureAssociationConfig
from src.geometry.texture_reconstruction import TextureReconstructionConfig


def _default_texture_reconstruction_config() -> TextureReconstructionConfig:
    return TextureReconstructionConfig(
        atlas_width=128,
        atlas_height=128,
        target_observation_count=2,
        min_confidence_observed=0.10,
    )


def _default_dense_fusion_config() -> DenseFusionConfig:
    return DenseFusionConfig(
        spatial_distance_threshold=0.6,
        voxel_grid_resolution=0.6,
        max_cluster_diameter=1.2,
        single_view_policy=SingleViewRetentionPolicy.RETAIN_AS_OBSERVED,
        min_distinct_view_support=1,
    )


def _default_surface_config() -> SurfaceReconstructionConfig:
    return SurfaceReconstructionConfig(
        alpha_radius=15.0,
        alpha_edge=15.0,
        min_distinct_views=1,
    )


@dataclass
class PipelineConfig:
    """Master pipeline configuration for Phase 3F reconstruction."""

    # Stage enable/disable flags
    enable_dense_stereo: bool = True
    enable_dense_point_generation: bool = True
    enable_dense_fusion: bool = True
    enable_surface_meshing: bool = True
    enable_texturing: bool = True
    enable_geospatial: bool = True

    # Determinism & ordering
    random_seed: int = 42
    ordering_policy: str = "CANONICAL_PTS"

    # Subsystem configurations
    keyframe_config: KeyframeSelectionConfig = field(default_factory=KeyframeSelectionConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    two_view_config: TwoViewConfig = field(default_factory=TwoViewConfig)
    sfm_config: SfMConfig = field(default_factory=SfMConfig)
    ba_config: BundleAdjustmentConfig = field(default_factory=BundleAdjustmentConfig)
    dense_stereo_config: DenseStereoConfig = field(default_factory=DenseStereoConfig)
    dense_point_config: DensePointGeneratorConfig = field(default_factory=DensePointGeneratorConfig)
    dense_fusion_config: DenseFusionConfig = field(default_factory=_default_dense_fusion_config)
    surface_config: SurfaceReconstructionConfig = field(default_factory=_default_surface_config)
    texture_association_config: TextureAssociationConfig = field(default_factory=TextureAssociationConfig)
    texture_reconstruction_config: TextureReconstructionConfig = field(default_factory=_default_texture_reconstruction_config)
    default_intrinsics: Optional[CameraIntrinsics] = None

    # Output directory & diagnostics
    output_dir: str = "data/processed/"
    strict_immutability_checks: bool = True

    def validate(self) -> None:
        """Validate logical dependencies between enabled stages.

        Raises ContractViolationError if an invalid combination is configured.
        """
        # Rule 1: Texturing requires a reconstructed surface mesh
        if self.enable_texturing and not self.enable_surface_meshing:
            raise ContractViolationError(
                "Invalid stage dependency: enable_texturing=True requires enable_surface_meshing=True."
            )

        # Rule 2: Dense point generation requires dense stereo disparity/depth
        if self.enable_dense_point_generation and not self.enable_dense_stereo:
            raise ContractViolationError(
                "Invalid stage dependency: enable_dense_point_generation=True requires enable_dense_stereo=True."
            )

        # Rule 3: Dense multi-view fusion requires dense point generation
        if self.enable_dense_fusion and not self.enable_dense_point_generation:
            raise ContractViolationError(
                "Invalid stage dependency: enable_dense_fusion=True requires enable_dense_point_generation=True."
            )

        # Rule 4: Surface meshing from dense fusion requires dense fusion
        # Note: If dense fusion is disabled, surface meshing cannot proceed unless another point source is declared
        if self.enable_surface_meshing and not self.enable_dense_fusion:
            raise ContractViolationError(
                "Invalid stage dependency: enable_surface_meshing=True requires enable_dense_fusion=True."
            )

    def compute_hash(self) -> str:
        """Computes a deterministic SHA-256 fingerprint of the configuration."""
        data = {
            "enable_dense_stereo": self.enable_dense_stereo,
            "enable_dense_point_generation": self.enable_dense_point_generation,
            "enable_dense_fusion": self.enable_dense_fusion,
            "enable_surface_meshing": self.enable_surface_meshing,
            "enable_texturing": self.enable_texturing,
            "enable_geospatial": self.enable_geospatial,
            "random_seed": self.random_seed,
            "ordering_policy": self.ordering_policy,
            "output_dir": self.output_dir,
            "strict_immutability_checks": self.strict_immutability_checks,
        }
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
