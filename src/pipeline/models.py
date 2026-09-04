"""Phase 3F: End-to-End Pipeline Domain Models & Execution Contracts.

Defines stage classifications, lifecycle status taxonomies, provenance records,
and final reconstruction results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PipelineStageType(str, Enum):
    """Authoritative enumeration of all pipeline execution stages in sequential DAG order."""
    INGESTION = "INGESTION"
    DECODING = "DECODING"
    FRAME_INTELLIGENCE = "FRAME_INTELLIGENCE"
    KEYFRAME_SELECTION = "KEYFRAME_SELECTION"
    CORRESPONDENCE = "CORRESPONDENCE"
    TWO_VIEW_GEOMETRY = "TWO_VIEW_GEOMETRY"
    INCREMENTAL_SFM = "INCREMENTAL_SFM"
    BUNDLE_ADJUSTMENT = "BUNDLE_ADJUSTMENT"
    DENSE_STEREO = "DENSE_STEREO"
    DENSE_POINT_GENERATION = "DENSE_POINT_GENERATION"
    DENSE_FUSION = "DENSE_FUSION"
    SURFACE_RECONSTRUCTION = "SURFACE_RECONSTRUCTION"
    TEXTURE_ASSOCIATION = "TEXTURE_ASSOCIATION"
    TEXTURE_RECONSTRUCTION = "TEXTURE_RECONSTRUCTION"
    GEOSPATIAL_TRANSFORM = "GEOSPATIAL_TRANSFORM"
    FINAL_VALIDATION = "FINAL_VALIDATION"
    FINALIZATION = "FINALIZATION"


class StageClassification(str, Enum):
    """Execution policy classification for pipeline stages."""
    MANDATORY = "MANDATORY"      # Critical geometric dependency; failure halts pipeline
    CONDITIONAL = "CONDITIONAL"  # Executed if enabled and input data present
    OPTIONAL = "OPTIONAL"        # Aesthetic/enhancement; failure leaves geometric model intact


class StageStatus(str, Enum):
    """Execution outcome status for an individual pipeline stage."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    NOT_EVALUABLE = "NOT_EVALUABLE"
    INSUFFICIENT_INPUT = "INSUFFICIENT_INPUT"
    CONTRACT_VIOLATION = "CONTRACT_VIOLATION"


class PipelineStatus(str, Enum):
    """Terminal outcome of an entire pipeline execution."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    NOT_EVALUABLE = "NOT_EVALUABLE"
    CONTRACT_VIOLATION = "CONTRACT_VIOLATION"


class ReconstructionUnitType(str, Enum):
    """Spatial coordinate units for geometric reconstruction."""
    RECONSTRUCTION_UNITS = "RECONSTRUCTION_UNITS"  # Arbitrary monocular scale gauge (||t_10|| = 1.0)
    METRIC_UNITS = "METRES"                        # Certified physical SI metres


class MetricScaleStatus(str, Enum):
    """Metric scale status of the reconstruction output."""
    SCALE_AMBIGUOUS = "SCALE_AMBIGUOUS"            # Gauge ambiguity unscaled
    METRICALLY_SCALED = "METRICALLY_SCALED"        # Scaled via verified metric telemetry or GCPs
    NOT_EVALUABLE = "NOT_EVALUABLE"                # Missing evidence prevents metric evaluation


@dataclass(frozen=True)
class StageExecutionRecord:
    """Immutable audit record of a single pipeline stage execution."""
    stage_id: str
    stage_type: PipelineStageType
    classification: StageClassification
    status: StageStatus
    input_artifact_ids: List[str] = field(default_factory=list)
    output_artifact_ids: List[str] = field(default_factory=list)
    configuration_hash: str = ""
    software_commit: str = "019deb2"
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "stage_type": self.stage_type.value,
            "classification": self.classification.value,
            "status": self.status.value,
            "input_artifact_ids": list(self.input_artifact_ids),
            "output_artifact_ids": list(self.output_artifact_ids),
            "configuration_hash": self.configuration_hash,
            "software_commit": self.software_commit,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class PipelineRunMetadata:
    """Provenance metadata uniquely identifying a pipeline execution run."""
    run_id: str
    git_commit: str
    config_hash: str
    random_seed: int
    ordering_policy: str
    started_at: str
    finished_at: Optional[str] = None
    total_duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "random_seed": self.random_seed,
            "ordering_policy": self.ordering_policy,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_duration_seconds": self.total_duration_seconds,
        }


@dataclass
class PipelineResult:
    """Final comprehensive deliverable of the Phase 3F Reconstruction Pipeline."""
    run_id: str
    pipeline_status: PipelineStatus
    metric_scale_status: MetricScaleStatus
    reconstruction_units: ReconstructionUnitType
    stage_records: List[StageExecutionRecord] = field(default_factory=list)
    output_artifacts: Dict[str, str] = field(default_factory=dict)  # artifact_type -> artifact_id
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "pipeline_status": self.pipeline_status.value,
            "metric_scale_status": self.metric_scale_status.value,
            "reconstruction_units": self.reconstruction_units.value,
            "stage_records": [rec.to_dict() for rec in self.stage_records],
            "output_artifacts": dict(self.output_artifacts),
            "diagnostics": dict(self.diagnostics),
            "failure_reason": self.failure_reason,
        }
