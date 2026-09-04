"""Phase 3F: Strongly Typed Artifact System & Anti-Leakage Boundary.

Implements immutable, SHA-256 fingerprinted artifacts for all inter-stage handoffs.
Enforces executable payload integrity verification and hard isolation between
reconstruction-domain artifacts and evaluation-truth artifacts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import numpy as np

from src.pipeline.errors import ContractViolationError, DataLeakageError


class ArtifactDomain(str, Enum):
    """Architectural domain classification for pipeline artifacts."""
    RECONSTRUCTION_INPUT = "RECONSTRUCTION_INPUT"    # Raw images, video, intrinsics, telemetry
    RECONSTRUCTION_OUTPUT = "RECONSTRUCTION_OUTPUT"  # Intermediate and final reconstructed models
    EVALUATION_TRUTH = "EVALUATION_TRUTH"            # CAD models, true poses, true depth (HIDDEN)
    VALIDATION_ONLY = "VALIDATION_ONLY"              # Diagnostics consumed only by validation stage


# Prohibited privileged evaluation keys that must NEVER enter reconstruction domain
FORBIDDEN_EVALUATION_KEYS: Set[str] = {
    "true_camera_poses",
    "true_camera_pose",
    "true_depth_maps",
    "true_depth",
    "true_mesh",
    "true_landmarks",
    "true_normals",
    "true_visibility_masks",
    "true_visibility_mask",
    "true_visibility",
    "true_textures",
    "true_texture",
    "cad_mesh",
    "cad_points",
    "ground_truth_poses",
    "ground_truth_pose",
    "ground_truth_depth",
    "ground_truth_points",
    "ground_truth_mesh",
    "ground_truth_texture",
    "ground_truth_textures",
    "ground_truth_visibility",
    "ground_truth_scale",
    "hidden_camera_pose",
    "hidden_camera_poses",
    "hidden_gt_texture",
    "hidden_texture",
    "hidden_visibility_mask",
    "hidden_metric_scale",
    "true_scale",
    "validation_reference",
    "validation_checkpoint",
    "validation_checkpoints",
}


def compute_canonical_payload_hash(payload: Any) -> str:
    """Computes a deterministic, cryptographic SHA-256 hash of an arbitrary payload.

    Guarantees:
    - Never hashes Python memory addresses or object IDs.
    - Explicitly serializes numpy arrays by contiguous bytes, dtype, and shape.
    - Deterministically serializes nested dicts/lists using canonical JSON.
    """
    hasher = hashlib.sha256()

    if isinstance(payload, np.ndarray):
        hasher.update(b"numpy_ndarray:")
        hasher.update(str(payload.dtype).encode("utf-8"))
        hasher.update(str(payload.shape).encode("utf-8"))
        hasher.update(np.ascontiguousarray(payload).tobytes())
    elif isinstance(payload, (dict, list)):
        try:
            # Attempt canonical JSON serialization
            serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
            hasher.update(b"json:")
            hasher.update(serialized.encode("utf-8"))
        except Exception:
            hasher.update(b"repr:")
            hasher.update(repr(payload).encode("utf-8"))
    elif hasattr(payload, "to_dict"):
        d = payload.to_dict()
        serialized = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
        hasher.update(b"to_dict:")
        hasher.update(serialized.encode("utf-8"))
    elif isinstance(payload, (bytes, bytearray)):
        hasher.update(b"raw_bytes:")
        hasher.update(payload)
    elif payload is None:
        hasher.update(b"null_payload")
    else:
        # Fallback to deterministic string representation
        hasher.update(b"string_repr:")
        hasher.update(str(payload).encode("utf-8"))

    return hasher.hexdigest()


@dataclass
class PipelineArtifact:
    """Base typed artifact representing an immutable handoff between pipeline stages."""
    artifact_id: str
    artifact_type: str
    domain: ArtifactDomain
    producer_stage: str
    input_artifact_ids: List[str]
    units: str
    coordinate_frame: str
    payload: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        # 1. Compute canonical payload hash if not provided
        if not self.content_hash:
            self.content_hash = compute_canonical_payload_hash(self.payload)

        # 2. Strict Anti-Leakage Boundary Enforcement
        if self.domain in (ArtifactDomain.RECONSTRUCTION_INPUT, ArtifactDomain.RECONSTRUCTION_OUTPUT):
            # Inspect metadata keys
            for key in self.metadata:
                if key.lower() in FORBIDDEN_EVALUATION_KEYS:
                    raise DataLeakageError(
                        f"Contract Violation: Privileged evaluation key '{key}' detected in "
                        f"reconstruction-domain artifact '{self.artifact_id}' ({self.artifact_type}).",
                        stage=self.producer_stage,
                    )
            # Inspect payload if dict
            if isinstance(self.payload, dict):
                for key in self.payload:
                    if str(key).lower() in FORBIDDEN_EVALUATION_KEYS:
                        raise DataLeakageError(
                            f"Contract Violation: Privileged evaluation key '{key}' detected in "
                            f"payload of reconstruction artifact '{self.artifact_id}'.",
                            stage=self.producer_stage,
                        )

        # 3. Monocular Scale Ambiguity Guard: Cannot claim METRES without certified metric scale
        if self.domain in (ArtifactDomain.RECONSTRUCTION_INPUT, ArtifactDomain.RECONSTRUCTION_OUTPUT):
            if self.units.upper() in ("METRES", "METERS", "M"):
                if isinstance(self.payload, dict):
                    if self.payload.get("has_monocular_scale_ambiguity", False) or not self.payload.get("is_metric_scale", True):
                        raise ContractViolationError(
                            f"Contract Violation: Cannot declare metric units '{self.units}' when artifact "
                            f"'{self.artifact_id}' has monocular scale ambiguity or is_metric_scale=False.",
                            stage=self.producer_stage,
                        )
                elif hasattr(self.payload, "has_monocular_scale_ambiguity") and getattr(self.payload, "has_monocular_scale_ambiguity", False):
                    raise ContractViolationError(
                        f"Contract Violation: Cannot declare metric units '{self.units}' when artifact "
                        f"'{self.artifact_id}' has monocular scale ambiguity.",
                        stage=self.producer_stage,
                    )
                elif hasattr(self.payload, "is_metric_scale") and not getattr(self.payload, "is_metric_scale", True):
                    raise ContractViolationError(
                        f"Contract Violation: Cannot declare metric units '{self.units}' when artifact "
                        f"'{self.artifact_id}' has is_metric_scale=False.",
                        stage=self.producer_stage,
                    )

    def verify_integrity(self) -> None:
        """Cryptographically verifies that the artifact payload has not been tampered with.

        Raises ContractViolationError if recomputed SHA-256 does not match content_hash.
        """
        current_hash = compute_canonical_payload_hash(self.payload)
        if current_hash != self.content_hash:
            raise ContractViolationError(
                f"Artifact tampering detected: Payload content_hash mismatch in artifact '{self.artifact_id}'. "
                f"Recorded: {self.content_hash}, Recomputed: {current_hash}.",
                stage=self.producer_stage,
                diagnostics={
                    "artifact_id": self.artifact_id,
                    "artifact_type": self.artifact_type,
                    "expected_hash": self.content_hash,
                    "actual_hash": current_hash,
                },
            )


# ---------------------------------------------------------------------------
# Typed Artifact Subclasses
# ---------------------------------------------------------------------------

@dataclass
class VideoArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "INGESTION", **kwargs: Any) -> None:
        super().__init__(
            artifact_id=artifact_id,
            artifact_type="VideoArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_INPUT,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units="SECONDS",
            coordinate_frame="TEMPORAL",
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )


@dataclass
class CanonicalTimelineArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "INGESTION", **kwargs: Any) -> None:
        super().__init__(
            artifact_id=artifact_id,
            artifact_type="CanonicalTimelineArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_INPUT,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units="MILLISECONDS",
            coordinate_frame="TEMPORAL_PTS",
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )


@dataclass
class DecodedFramesArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "DECODING", **kwargs: Any) -> None:
        super().__init__(
            artifact_id=artifact_id,
            artifact_type="DecodedFramesArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_INPUT,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units="PIXELS",
            coordinate_frame="RASTER_RGB",
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )


@dataclass
class FrameQualityArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "FRAME_INTELLIGENCE", **kwargs: Any) -> None:
        super().__init__(
            artifact_id=artifact_id,
            artifact_type="FrameQualityArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_OUTPUT,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units="DIMENSIONLESS",
            coordinate_frame="FRAME_LOCAL",
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )


@dataclass
class KeyframeSetArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "KEYFRAME_SELECTION", **kwargs: Any) -> None:
        super().__init__(
            artifact_id=artifact_id,
            artifact_type="KeyframeSetArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_OUTPUT,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units="INDEX",
            coordinate_frame="TEMPORAL",
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )


@dataclass
class CorrespondenceArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "CORRESPONDENCE", **kwargs: Any) -> None:
        super().__init__(
            artifact_id=artifact_id,
            artifact_type="CorrespondenceArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_OUTPUT,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units="PIXELS",
            coordinate_frame="IMAGE_SPACE",
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )


@dataclass
class TwoViewGeometryArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "TWO_VIEW_GEOMETRY", **kwargs: Any) -> None:
        super().__init__(
            artifact_id=artifact_id,
            artifact_type="TwoViewGeometryArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_OUTPUT,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units="RECONSTRUCTION_UNITS",
            coordinate_frame="MONOCULAR_GAUGE",
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )


@dataclass
class SfMArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "INCREMENTAL_SFM", **kwargs: Any) -> None:
        super().__init__(
            artifact_id=artifact_id,
            artifact_type="SfMArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_OUTPUT,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units="RECONSTRUCTION_UNITS",
            coordinate_frame="MONOCULAR_GAUGE",
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )


@dataclass
class BundleAdjustmentArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "BUNDLE_ADJUSTMENT", **kwargs: Any) -> None:
        super().__init__(
            artifact_id=artifact_id,
            artifact_type="BundleAdjustmentArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_OUTPUT,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units="RECONSTRUCTION_UNITS",
            coordinate_frame="MONOCULAR_GAUGE",
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )


@dataclass
class DenseStereoArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "DENSE_STEREO", **kwargs: Any) -> None:
        super().__init__(
            artifact_id=artifact_id,
            artifact_type="DenseStereoArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_OUTPUT,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units=kwargs.get("units", "RECONSTRUCTION_UNITS"),
            coordinate_frame="CAMERA_OPTICAL",
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )


@dataclass
class DensePointArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "DENSE_POINT_GENERATION", **kwargs: Any) -> None:
        super().__init__(
            artifact_id=artifact_id,
            artifact_type="DensePointArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_OUTPUT,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units=kwargs.get("units", "RECONSTRUCTION_UNITS"),
            coordinate_frame="MONOCULAR_GAUGE",
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )


@dataclass
class DenseFusionArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "DENSE_FUSION", **kwargs: Any) -> None:
        super().__init__(
            artifact_id=artifact_id,
            artifact_type="DenseFusionArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_OUTPUT,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units=kwargs.get("units", "RECONSTRUCTION_UNITS"),
            coordinate_frame="MONOCULAR_GAUGE",
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )


@dataclass
class SurfaceArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "SURFACE_RECONSTRUCTION", **kwargs: Any) -> None:
        super().__init__(
            artifact_id=artifact_id,
            artifact_type="SurfaceArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_OUTPUT,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units=kwargs.get("units", "RECONSTRUCTION_UNITS"),
            coordinate_frame="MONOCULAR_GAUGE",
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )


@dataclass
class TextureAssociationArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "TEXTURE_ASSOCIATION", **kwargs: Any) -> None:
        if isinstance(payload, dict):
            if payload.get("hard_backface_culling", False) or payload.get("cull_backface_decision") or "hard_backface_culling" in kwargs.get("metadata", {}):
                raise ContractViolationError(
                    "Contract Violation: Hard back-face culling is strictly prohibited under the normal-sign independent texture visibility contract.",
                    stage=producer_stage,
                )
        if kwargs.get("metadata", {}).get("hard_backface_culling", False):
            raise ContractViolationError(
                "Contract Violation: Hard back-face culling is strictly prohibited under the normal-sign independent texture visibility contract.",
                stage=producer_stage,
            )
        super().__init__(
            artifact_id=artifact_id,
            artifact_type="TextureAssociationArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_OUTPUT,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units=kwargs.get("units", "DIMENSIONLESS"),
            coordinate_frame=kwargs.get("coordinate_frame", "SURFACE_LOCAL"),
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )


@dataclass
class TexturedSurfaceArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "TEXTURE_RECONSTRUCTION", **kwargs: Any) -> None:
        if isinstance(payload, dict):
            if payload.get("unobserved_marked_observed_without_evidence", False):
                raise ContractViolationError(
                    "Contract Violation: Unobserved texel marked as OBSERVED without source evidence.",
                    stage=producer_stage,
                )
        elif hasattr(payload, "observed_texel_ratio") and getattr(payload, "observed_texel_ratio") > 0:
            texel_prov = getattr(payload, "texel_provenance", {})
            if len(texel_prov) == 0:
                raise ContractViolationError(
                    "Contract Violation: Texture atlas reports observed texels but contains zero texel provenance records.",
                    stage=producer_stage,
                )
            for p in texel_prov.values():
                p_state = getattr(p, "state", None)
                state_str = getattr(p_state, "value", str(p_state))
                if state_str == "OBSERVED_TEXTURE" and len(getattr(p, "contributing_frames", [])) == 0:
                    raise ContractViolationError(
                        "Contract Violation: Texel marked as OBSERVED_TEXTURE without contributing source frames.",
                        stage=producer_stage,
                    )
        super().__init__(
            artifact_id=artifact_id,
            artifact_type="TexturedSurfaceArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_OUTPUT,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units=kwargs.get("units", "RECONSTRUCTION_UNITS"),
            coordinate_frame=kwargs.get("coordinate_frame", "MONOCULAR_GAUGE"),
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )


@dataclass
class TelemetryArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "INGESTION", **kwargs: Any) -> None:
        super().__init__(
            artifact_id=artifact_id,
            artifact_type="TelemetryArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_INPUT,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units=kwargs.get("units", "WGS84_METRIC"),
            coordinate_frame=kwargs.get("coordinate_frame", "GEODETIC_WGS84"),
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )


@dataclass
class GeospatialArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "GEOSPATIAL_TRANSFORM", **kwargs: Any) -> None:
        is_metric = False
        status_val = None
        if isinstance(payload, dict):
            is_metric = payload.get("is_metric_scale", False)
            status_val = payload.get("status") or payload.get("metric_scale_status")
            if payload.get("claim_metric_accuracy_when_not_evaluable", False):
                raise ContractViolationError(
                    "Contract Violation: Cannot claim metric accuracy when geospatial status is NOT_EVALUABLE.",
                    stage=producer_stage,
                )
        elif hasattr(payload, "is_metric_scale"):
            is_metric = getattr(payload, "is_metric_scale", False)
            status_val = getattr(payload, "metric_scale_status", None)

        requested_units = kwargs.get("units")
        status_str = getattr(status_val, "value", str(status_val)) if status_val is not None else ""
        if requested_units in ("METRES", "METERS", "M") and (not is_metric or status_str in ("NOT_EVALUABLE", "SCALE_AMBIGUOUS")):
            raise ContractViolationError(
                f"Contract Violation: Cannot declare metric units '{requested_units}' when geospatial status is not metric.",
                stage=producer_stage,
            )

        default_units = "METRES" if is_metric else "RECONSTRUCTION_UNITS"
        default_frame = "TOPOCENTRIC_ENU" if is_metric else "MONOCULAR_GAUGE"

        super().__init__(
            artifact_id=artifact_id,
            artifact_type="GeospatialArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_OUTPUT,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units=kwargs.get("units", default_units),
            coordinate_frame=kwargs.get("coordinate_frame", default_frame),
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )


@dataclass
class ValidationArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "FINAL_VALIDATION", **kwargs: Any) -> None:
        if isinstance(payload, dict):
            if payload.get("claim_radiometric_without_calibration", False):
                raise ContractViolationError(
                    "Contract Violation: Cannot claim radiometric/colorimetric accuracy without radiometric calibration.",
                    stage=producer_stage,
                )
            if payload.get("claim_metric_accuracy_when_not_evaluable", False):
                raise ContractViolationError(
                    "Contract Violation: Cannot claim metric accuracy when geospatial status is NOT_EVALUABLE.",
                    stage=producer_stage,
                )
        super().__init__(
            artifact_id=artifact_id,
            artifact_type="ValidationArtifact",
            domain=ArtifactDomain.VALIDATION_ONLY,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units="DIMENSIONLESS",
            coordinate_frame="EVALUATION_REPORT",
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )


@dataclass
class FinalReconstructionArtifact(PipelineArtifact):
    def __init__(self, artifact_id: str, payload: Any, producer_stage: str = "FINALIZATION", **kwargs: Any) -> None:
        super().__init__(
            artifact_id=artifact_id,
            artifact_type="FinalReconstructionArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_OUTPUT,
            producer_stage=producer_stage,
            input_artifact_ids=kwargs.get("input_artifact_ids", []),
            units=kwargs.get("units", "RECONSTRUCTION_UNITS"),
            coordinate_frame=kwargs.get("coordinate_frame", "MONOCULAR_GAUGE"),
            payload=payload,
            metadata=kwargs.get("metadata", {}),
            content_hash=kwargs.get("content_hash", ""),
        )
