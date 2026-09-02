"""Canonical Flight Dataset: Integrated, immutable, provenance-preserving flight observation artifact."""

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Iterator

from src.ingestion.canonical_timeline import CanonicalFrame, CanonicalTimeline, VideoProvenance, VideoMetadata
from src.ingestion.canonical_telemetry import (
    TelemetryPosition,
    TelemetryOrientation,
    TelemetryVelocity,
    TelemetryQuality,
    TelemetryProvenance,
    TelemetryRecord,
    CanonicalTelemetryStream,
    AltitudeReference,
    PositionReference,
    TimestampSemantics,
    FixType,
)
from src.ingestion.synchronization import (
    SyncStatus,
    ClockOffsetStatus,
    ClockOffsetModel,
    IdentityClockModel,
    ConstantOffsetClockModel,
    SynchronizationConfig,
    SynchronizedFrameObservation,
    SynchronizedTrajectory,
    TemporalSynchronizationEngine,
)
from src.geospatial.normalization import (
    ECEFCoordinates,
    ENUCoordinates,
    GeodeticCoordinates,
    OriginPolicy,
    GeodeticOrigin,
    NormalizedTelemetryRecord,
    NormalizedTelemetryStream,
    CoordinateNormalizer,
    geodetic_to_ecef,
    ecef_to_enu,
)


class DatasetStatus(str, Enum):
    """Integrity and validity status of the assembled CanonicalFlightDataset."""
    VALID = "VALID"                       # Video and telemetry aligned, majority of frames synchronized
    PARTIALLY_VALID = "PARTIALLY_VALID"   # Minor gaps, missing optional sensors (e.g. IMU), or degraded coverage
    INVALID = "INVALID"                   # Severe flaws (missing stream, zero synchronized frames, corrupt metadata)


@dataclass(frozen=True)
class DatasetValidationIssue:
    """Formal diagnostic issue detected during dataset assembly."""
    severity: str  # "ERROR", "WARNING", "INFO"
    code: str
    message: str
    affected_frame_indices: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class DatasetProvenance:
    """End-to-end lineage and cryptographic/system identity for the flight dataset."""
    dataset_id: str
    creation_timestamp_utc: str
    software_version: str
    video_source_path: str
    video_checksum_sha256: Optional[str]
    telemetry_source_path: str
    telemetry_checksum_sha256: Optional[str]
    adapter_name: str
    clock_offset_model: str
    clock_offset_seconds: float
    clock_offset_status: str
    normalization_origin: Optional[Dict[str, Any]] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalFrameObservation:
    """Unified per-frame association combining video timing, telemetry, synchronization, and metric coordinates."""
    frame_id: str
    frame_index: int
    video_timestamp_seconds: float
    is_keyframe: bool
    sync_status: SyncStatus
    is_synchronized: bool
    original_position: Optional[TelemetryPosition] = None
    ecef_position: Optional[ECEFCoordinates] = None
    enu_position: Optional[ENUCoordinates] = None
    orientation: Optional[TelemetryOrientation] = None
    velocity: Optional[TelemetryVelocity] = None
    quality: Optional[TelemetryQuality] = None
    bracketing_interval_seconds: Optional[float] = None
    interpolation_fraction: Optional[float] = None
    source_record_indices: List[int] = field(default_factory=list)
    source_provenance: Optional[TelemetryProvenance] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalFlightDataset:
    """Integrated, immutable, provenance-preserving canonical flight representation."""
    dataset_id: str
    video_metadata: VideoMetadata
    timeline: CanonicalTimeline
    telemetry_stream: CanonicalTelemetryStream
    synchronized_trajectory: SynchronizedTrajectory
    normalized_telemetry: Optional[NormalizedTelemetryStream]
    frame_observations: List[CanonicalFrameObservation]
    origin: Optional[GeodeticOrigin]
    status: DatasetStatus
    validation_issues: List[DatasetValidationIssue]
    provenance: DatasetProvenance

    def __len__(self) -> int:
        return len(self.frame_observations)

    def __getitem__(self, idx: int) -> CanonicalFrameObservation:
        return self.frame_observations[idx]

    def __iter__(self) -> Iterator[CanonicalFrameObservation]:
        return iter(self.frame_observations)

    def get_frame_by_id(self, frame_id: str) -> Optional[CanonicalFrameObservation]:
        """Deterministic lookup by frame_id."""
        for obs in self.frame_observations:
            if obs.frame_id == frame_id:
                return obs
        return None

    def get_synchronized_frames(self) -> List[CanonicalFrameObservation]:
        """Return only observations that have valid synchronized spatial coordinates."""
        return [obs for obs in self.frame_observations if obs.is_synchronized]

    @property
    def synchronized_count(self) -> int:
        return sum(1 for obs in self.frame_observations if obs.is_synchronized)

    @property
    def total_frames(self) -> int:
        return len(self.frame_observations)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete dataset metadata and manifest without raw pixel binaries."""
        return {
            "dataset_id": self.dataset_id,
            "status": self.status.value,
            "provenance": asdict(self.provenance),
            "video_metadata": asdict(self.video_metadata),
            "origin": asdict(self.origin) if self.origin else None,
            "validation_issues": [asdict(issue) for issue in self.validation_issues],
            "timeline_summary": {
                "video_id": self.timeline.video_id,
                "total_frames": self.timeline.total_frames,
                "duration_seconds": self.timeline.duration_seconds,
                "nominal_fps": self.timeline.nominal_fps,
            },
            "telemetry_summary": {
                "stream_id": self.telemetry_stream.stream_id,
                "total_records": len(self.telemetry_stream),
            },
            "frame_observations": [
                {
                    "frame_id": obs.frame_id,
                    "frame_index": obs.frame_index,
                    "video_timestamp_seconds": obs.video_timestamp_seconds,
                    "is_keyframe": obs.is_keyframe,
                    "sync_status": obs.sync_status.value,
                    "is_synchronized": obs.is_synchronized,
                    "original_position": asdict(obs.original_position) if obs.original_position else None,
                    "ecef_position": asdict(obs.ecef_position) if obs.ecef_position else None,
                    "enu_position": asdict(obs.enu_position) if obs.enu_position else None,
                    "orientation": asdict(obs.orientation) if obs.orientation else None,
                    "velocity": asdict(obs.velocity) if obs.velocity else None,
                    "quality": asdict(obs.quality) if obs.quality else None,
                    "bracketing_interval_seconds": obs.bracketing_interval_seconds,
                    "interpolation_fraction": obs.interpolation_fraction,
                    "source_record_indices": obs.source_record_indices,
                    "source_provenance": asdict(obs.source_provenance) if obs.source_provenance else None,
                    "extra_metadata": obs.extra_metadata,
                }
                for obs in self.frame_observations
            ],
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize dataset metadata to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class CanonicalDatasetBuilder:
    """Builder pipeline constructing a CanonicalFlightDataset with validation and provenance tracking."""

    SOFTWARE_VERSION = "SIH26158_v1.0_Phase1B.5"

    @classmethod
    def build(
        cls,
        video_metadata: VideoMetadata,
        timeline: CanonicalTimeline,
        telemetry_stream: CanonicalTelemetryStream,
        sync_config: Optional[SynchronizationConfig] = None,
        origin_policy: OriginPolicy = OriginPolicy.FIRST_VALID_POSITION,
        explicit_origin: Optional[GeodeticOrigin] = None,
    ) -> CanonicalFlightDataset:
        """Assembles CanonicalFlightDataset from raw components via synchronization and normalization."""
        validation_issues: List[DatasetValidationIssue] = []
        cfg = sync_config or SynchronizationConfig()

        # 1. Basic Structural Validation
        if len(timeline) == 0:
            validation_issues.append(
                DatasetValidationIssue(severity="ERROR", code="EMPTY_TIMELINE", message="Video timeline contains zero frames.")
            )

        if len(telemetry_stream) == 0:
            validation_issues.append(
                DatasetValidationIssue(severity="ERROR", code="EMPTY_TELEMETRY", message="Telemetry stream contains zero records.")
            )

        # 2. Check for optional sensor fields across telemetry
        has_orientation = any(r.orientation is not None for r in telemetry_stream)
        has_velocity = any(r.velocity is not None for r in telemetry_stream)
        has_quality = any(r.quality is not None for r in telemetry_stream)

        if not has_orientation:
            validation_issues.append(
                DatasetValidationIssue(
                    severity="INFO",
                    code="MISSING_OPTIONAL_IMU",
                    message="Telemetry stream contains no IMU orientation (heading/pitch/roll) observations."
                )
            )
        if not has_velocity:
            validation_issues.append(
                DatasetValidationIssue(
                    severity="INFO",
                    code="MISSING_OPTIONAL_VELOCITY",
                    message="Telemetry stream contains no velocity observations."
                )
            )

        # 3. Check for non-ellipsoidal altitude references
        non_ellipsoidal_refs = {
            r.position.altitude_reference.value
            for r in telemetry_stream
            if r.position.altitude_reference != AltitudeReference.ELLIPSOIDAL
        }
        if non_ellipsoidal_refs:
            validation_issues.append(
                DatasetValidationIssue(
                    severity="WARNING",
                    code="NON_ELLIPSOIDAL_ALTITUDE",
                    message=f"Telemetry uses non-ellipsoidal vertical datums: {sorted(non_ellipsoidal_refs)}."
                )
            )

        # 4. Execute Temporal Synchronization
        synchronized_trajectory = TemporalSynchronizationEngine.synchronize(
            timeline, telemetry_stream, config=cfg
        )

        # 5. Execute Coordinate Normalization
        normalized_telemetry: Optional[NormalizedTelemetryStream] = None
        origin: Optional[GeodeticOrigin] = None

        if len(telemetry_stream) > 0:
            try:
                origin = CoordinateNormalizer.compute_origin(
                    telemetry_stream, policy=origin_policy, explicit_origin=explicit_origin
                )
                normalized_telemetry = CoordinateNormalizer.normalize_stream(
                    telemetry_stream, policy=origin_policy, explicit_origin=origin
                )
            except Exception as e:
                validation_issues.append(
                    DatasetValidationIssue(
                        severity="WARNING",
                        code="NORMALIZATION_ORIGIN_FAILURE",
                        message=f"Could not compute local ENU origin: {str(e)}"
                    )
                )

        # 6. Assemble Frame Observations with ENU Projection
        frame_observations: List[CanonicalFrameObservation] = []
        unsync_indices: List[int] = []

        for idx, (frame, sync_obs) in enumerate(zip(timeline, synchronized_trajectory)):
            pos = sync_obs.position
            ecef: Optional[ECEFCoordinates] = None
            enu: Optional[ENUCoordinates] = None

            if pos is not None:
                try:
                    ecef = geodetic_to_ecef(pos.latitude_deg, pos.longitude_deg, pos.altitude_meters)
                    if origin is not None:
                        enu = ecef_to_enu(ecef, origin)
                except Exception:
                    ecef = None
                    enu = None

            if not sync_obs.is_synchronized:
                unsync_indices.append(idx)

            obs = CanonicalFrameObservation(
                frame_id=frame.frame_id,
                frame_index=frame.frame_index,
                video_timestamp_seconds=frame.timestamp_seconds,
                is_keyframe=frame.is_keyframe,
                sync_status=sync_obs.status,
                is_synchronized=sync_obs.is_synchronized and ecef is not None,
                original_position=pos,
                ecef_position=ecef,
                enu_position=enu,
                orientation=sync_obs.orientation,
                velocity=sync_obs.velocity,
                quality=sync_obs.quality,
                bracketing_interval_seconds=sync_obs.bracketing_interval_seconds,
                interpolation_fraction=sync_obs.interpolation_fraction,
                source_record_indices=sync_obs.source_record_indices,
                source_provenance=sync_obs.source_provenance,
                extra_metadata=sync_obs.extra_metadata,
            )
            frame_observations.append(obs)

        if unsync_indices:
            validation_issues.append(
                DatasetValidationIssue(
                    severity="WARNING",
                    code="UNSYNCHRONIZED_FRAMES",
                    message=f"{len(unsync_indices)} frames out of {len(timeline)} could not be synchronized with telemetry.",
                    affected_frame_indices=unsync_indices,
                )
            )

        # 7. Determine Overall Dataset Status
        sync_count = sum(1 for obs in frame_observations if obs.is_synchronized)
        total_count = len(frame_observations)

        has_errors = any(issue.severity == "ERROR" for issue in validation_issues)
        if has_errors or total_count == 0 or sync_count == 0:
            dataset_status = DatasetStatus.INVALID
        elif sync_count < total_count:
            dataset_status = DatasetStatus.PARTIALLY_VALID
        else:
            dataset_status = DatasetStatus.VALID

        # 8. Build Provenance Record
        now_utc = datetime.now(timezone.utc).isoformat()
        dataset_id = f"ds_{timeline.video_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        provenance = DatasetProvenance(
            dataset_id=dataset_id,
            creation_timestamp_utc=now_utc,
            software_version=cls.SOFTWARE_VERSION,
            video_source_path=timeline.source_path,
            video_checksum_sha256=timeline.provenance.sha256_checksum if timeline.provenance else None,
            telemetry_source_path=telemetry_stream.provenance.source_identifier if telemetry_stream.provenance else "",
            telemetry_checksum_sha256=telemetry_stream.provenance.source_checksum if telemetry_stream.provenance else None,
            adapter_name=telemetry_stream.provenance.source_type if telemetry_stream.provenance else "unknown",
            clock_offset_model=type(cfg.clock_model).__name__,
            clock_offset_seconds=cfg.clock_model.get_offset_seconds(),
            clock_offset_status=cfg.clock_model.get_offset_status().value,
            normalization_origin=asdict(origin) if origin else None,
            extra_metadata={
                "total_frames": total_count,
                "synchronized_frames": sync_count,
                "origin_policy": origin_policy.value,
            },
        )

        return CanonicalFlightDataset(
            dataset_id=dataset_id,
            video_metadata=video_metadata,
            timeline=timeline,
            telemetry_stream=telemetry_stream,
            synchronized_trajectory=synchronized_trajectory,
            normalized_telemetry=normalized_telemetry,
            frame_observations=frame_observations,
            origin=origin,
            status=dataset_status,
            validation_issues=validation_issues,
            provenance=provenance,
        )
