"""Phase 3F: End-to-End Reconstruction Pipeline Package.

Exports typed pipeline models, configuration contracts, artifact interfaces,
and the master ReconstructionPipeline orchestrator.
"""

from src.pipeline.models import (
    PipelineStageType,
    StageClassification,
    StageStatus,
    PipelineStatus,
    ReconstructionUnitType,
    MetricScaleStatus,
    StageExecutionRecord,
    PipelineRunMetadata,
    PipelineResult,
)

from src.pipeline.errors import (
    PipelineError,
    ContractViolationError,
    StageExecutionError,
    InsufficientInputError,
    DataLeakageError,
)

from src.pipeline.config import (
    PipelineConfig,
)

from src.pipeline.artifacts import (
    ArtifactDomain,
    PipelineArtifact,
    compute_canonical_payload_hash,
    VideoArtifact,
    CanonicalTimelineArtifact,
    DecodedFramesArtifact,
    FrameQualityArtifact,
    KeyframeSetArtifact,
    CorrespondenceArtifact,
    TwoViewGeometryArtifact,
    SfMArtifact,
    BundleAdjustmentArtifact,
    DenseStereoArtifact,
    DensePointArtifact,
    DenseFusionArtifact,
    SurfaceArtifact,
    TextureAssociationArtifact,
    TexturedSurfaceArtifact,
    GeospatialArtifact,
    ValidationArtifact,
    FinalReconstructionArtifact,
)

from src.pipeline.orchestrator import (
    ReconstructionPipeline,
)

__all__ = [
    # Models
    "PipelineStageType",
    "StageClassification",
    "StageStatus",
    "PipelineStatus",
    "ReconstructionUnitType",
    "MetricScaleStatus",
    "StageExecutionRecord",
    "PipelineRunMetadata",
    "PipelineResult",
    # Errors
    "PipelineError",
    "ContractViolationError",
    "StageExecutionError",
    "InsufficientInputError",
    "DataLeakageError",
    # Config
    "PipelineConfig",
    # Artifacts
    "ArtifactDomain",
    "PipelineArtifact",
    "compute_canonical_payload_hash",
    "VideoArtifact",
    "CanonicalTimelineArtifact",
    "DecodedFramesArtifact",
    "FrameQualityArtifact",
    "KeyframeSetArtifact",
    "CorrespondenceArtifact",
    "TwoViewGeometryArtifact",
    "SfMArtifact",
    "BundleAdjustmentArtifact",
    "DenseStereoArtifact",
    "DensePointArtifact",
    "DenseFusionArtifact",
    "SurfaceArtifact",
    "TextureAssociationArtifact",
    "TexturedSurfaceArtifact",
    "GeospatialArtifact",
    "ValidationArtifact",
    "FinalReconstructionArtifact",
    # Orchestrator
    "ReconstructionPipeline",
]
