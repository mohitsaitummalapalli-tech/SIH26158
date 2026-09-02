"""Quality module: Sharpness, exposure, luminance, frame quality, temporal motion, photometric, dynamic scene, redundancy, and keyframe selection diagnostics."""

from dataclasses import dataclass
from typing import Optional

from src.quality.assessment import (
    QualityStatus,
    QualityAssessmentConfig,
    LuminanceStatistics,
    ClippingStatistics,
    ContrastStatistics,
    SharpnessStatistics,
    SpatialTileQuality,
    FrameQualityReport,
    FrameQualityAnalyzer,
)

from src.quality.temporal_motion import (
    MotionCategory,
    TemporalMotionConfig,
    SpatialMotionTile,
    TemporalMotionBlurReport,
    TemporalMotionAnalyzer,
)

from src.quality.photometric import (
    SpatialIlluminationPattern,
    PhotometricChangeCategory,
    PhotometricConfig,
    SpatialIlluminationTile,
    ColorStatistics,
    ExtendedLuminanceStatistics,
    DynamicRangeStatistics,
    TemporalPhotometricChange,
    PhotometricStabilityReport,
    PhotometricAnalyzer,
)

from src.quality.dynamic_scene import (
    DynamicEvidenceCategory,
    RegionMaskReference,
    CandidateDynamicRegion,
    DynamicSceneConfig,
    DynamicRegionProvider,
    SyntheticDynamicRegionProvider,
    DynamicSceneReport,
    DynamicSceneAnalyzer,
)

from src.quality.redundancy_viewpoint import (
    FrameRedundancyConfig,
    FramePairRelation,
    FrameRedundancyReport,
    FrameRedundancyViewpointAnalyzer,
)

from src.quality.keyframe_selection import (
    SelectionReason,
    RejectionReason,
    SelectedKeyframe,
    DeprioritizedCandidate,
    KeyframeSelectionConfig,
    KeyframeSelectionResult,
    CoverageAwareKeyframeSelector,
)


@dataclass(frozen=True)
class FrameQualityScore:
    """Quantitative quality metrics evaluated on a single video frame."""
    frame_index: int
    laplacian_variance: float       # High-frequency edge sharpness
    exposure_balance_score: float   # Under/over-exposure distribution [0, 1]
    motion_blur_metric: float       # Motion blur estimate [0 (sharp) -> 1 (blurred)]
    composite_quality_score: float  # Overall normalized quality index [0, 100]
    passed_filter: bool = True
    rejection_reason: Optional[str] = None


@dataclass
class QualityFilterConfig:
    """Threshold configuration for frame quality filtering."""
    min_laplacian_variance: float = 80.0
    min_composite_score: float = 40.0
    max_motion_blur_ratio: float = 0.6
    reject_degraded_frames: bool = True


__all__ = [
    "QualityStatus",
    "QualityAssessmentConfig",
    "LuminanceStatistics",
    "ClippingStatistics",
    "ContrastStatistics",
    "SharpnessStatistics",
    "SpatialTileQuality",
    "FrameQualityReport",
    "FrameQualityAnalyzer",
    "MotionCategory",
    "TemporalMotionConfig",
    "SpatialMotionTile",
    "TemporalMotionBlurReport",
    "TemporalMotionAnalyzer",
    "SpatialIlluminationPattern",
    "PhotometricChangeCategory",
    "PhotometricConfig",
    "SpatialIlluminationTile",
    "ColorStatistics",
    "ExtendedLuminanceStatistics",
    "DynamicRangeStatistics",
    "TemporalPhotometricChange",
    "PhotometricStabilityReport",
    "PhotometricAnalyzer",
    "DynamicEvidenceCategory",
    "RegionMaskReference",
    "CandidateDynamicRegion",
    "DynamicSceneConfig",
    "DynamicRegionProvider",
    "SyntheticDynamicRegionProvider",
    "DynamicSceneReport",
    "DynamicSceneAnalyzer",
    "FrameRedundancyConfig",
    "FramePairRelation",
    "FrameRedundancyReport",
    "FrameRedundancyViewpointAnalyzer",
    "SelectionReason",
    "RejectionReason",
    "SelectedKeyframe",
    "DeprioritizedCandidate",
    "KeyframeSelectionConfig",
    "KeyframeSelectionResult",
    "CoverageAwareKeyframeSelector",
    "FrameQualityScore",
    "QualityFilterConfig",
]
