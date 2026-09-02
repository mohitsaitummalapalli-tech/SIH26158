"""Ingestion module: Video demuxing, metadata validation, and canonical telemetry contracts."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from src.ingestion.exceptions import (
    IngestionError,
    VideoNotFoundError,
    UnsupportedVideoFormatError,
    UnsupportedFragmentedMP4Error,
    CorruptVideoError,
    InvalidVideoMetadataError,
    InvalidTelemetryDataError,
)
from src.ingestion.canonical_timeline import (
    CanonicalFrame,
    CanonicalTimeline,
    VideoProvenance,
    VideoMetadata,
)
from src.ingestion.canonical_telemetry import (
    AltitudeReference,
    PositionReference,
    TimestampSemantics,
    FixType,
    TelemetryPosition,
    TelemetryOrientation,
    TelemetryVelocity,
    TelemetryQuality,
    TelemetryProvenance,
    TelemetryRecord,
    CanonicalTelemetryStream,
)
from src.ingestion.adapters import (
    TelemetryAdapter,
    ParsedTelemetryRecord,
    RecordStatus,
    DJISRTAdapter,
    GenericCSVAdapter,
    CSVColumnMapping,
    KLVAdapterInterface,
    KLVPacket,
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
    geodetic_to_ecef,
    ecef_to_geodetic,
)
from src.ingestion.dataset import (
    DatasetStatus,
    DatasetValidationIssue,
    DatasetProvenance,
    CanonicalFrameObservation,
    CanonicalFlightDataset,
    CanonicalDatasetBuilder,
)
from src.ingestion.video_source import (
    VideoSource,
    ISOBMFFVideoSource,
)


@dataclass
class IngestionConfig:
    """Configuration parameters for video stream demuxing and telemetry extraction."""
    target_fps: float = 2.0
    max_resolution_width: int = 1920
    max_resolution_height: int = 1080
    telemetry_file_override: Optional[str] = None
    time_offset_seconds: float = 0.0  # Constant time sync bias between video and GPS logs


__all__ = [
    "IngestionError",
    "VideoNotFoundError",
    "UnsupportedVideoFormatError",
    "UnsupportedFragmentedMP4Error",
    "CorruptVideoError",
    "InvalidVideoMetadataError",
    "InvalidTelemetryDataError",
    "CanonicalFrame",
    "CanonicalTimeline",
    "VideoProvenance",
    "AltitudeReference",
    "PositionReference",
    "TimestampSemantics",
    "FixType",
    "TelemetryPosition",
    "TelemetryOrientation",
    "TelemetryVelocity",
    "TelemetryQuality",
    "TelemetryProvenance",
    "TelemetryRecord",
    "CanonicalTelemetryStream",
    "TelemetryAdapter",
    "ParsedTelemetryRecord",
    "RecordStatus",
    "DJISRTAdapter",
    "GenericCSVAdapter",
    "CSVColumnMapping",
    "KLVAdapterInterface",
    "KLVPacket",
    "SyncStatus",
    "ClockOffsetStatus",
    "ClockOffsetModel",
    "IdentityClockModel",
    "ConstantOffsetClockModel",
    "SynchronizationConfig",
    "SynchronizedFrameObservation",
    "SynchronizedTrajectory",
    "TemporalSynchronizationEngine",
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "DatasetStatus",
    "DatasetValidationIssue",
    "DatasetProvenance",
    "CanonicalFrameObservation",
    "CanonicalFlightDataset",
    "CanonicalDatasetBuilder",
    "VideoSource",
    "ISOBMFFVideoSource",
    "VideoMetadata",
    "IngestionConfig",
]
