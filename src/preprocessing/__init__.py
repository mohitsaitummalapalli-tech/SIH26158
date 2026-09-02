"""Preprocessing module: Frame decoding, extraction, undistortion, and keyframe selection."""

from dataclasses import dataclass, field
from typing import Optional, List, Any
from src.ingestion import TelemetryRecord

from src.preprocessing.decoder import (
    DecodeStatus,
    DecodeConfig,
    DecodedFrame,
    FrameDecoder,
    OpenCVFrameDecoder,
    SyntheticFrameDecoder,
    DatasetFrameDecoder,
)


@dataclass
class FrameData:
    """Individual extracted frame representation with associated telemetry."""
    frame_index: int
    timestamp_seconds: float
    image_path: Optional[str] = None
    width: int = 0
    height: int = 0
    telemetry: Optional[TelemetryRecord] = None
    is_keyframe: bool = False
    quality_score: Optional[float] = None


@dataclass
class KeyframeSelectionConfig:
    """Configuration for keyframe baseline spacing and optical overlap."""
    min_parallax_pixels: float = 30.0
    min_spatial_baseline_meters: float = 2.0
    max_spatial_baseline_meters: float = 15.0
    min_temporal_gap_seconds: float = 0.5
    min_quality_threshold: float = 50.0


__all__ = [
    "DecodeStatus",
    "DecodeConfig",
    "DecodedFrame",
    "FrameDecoder",
    "OpenCVFrameDecoder",
    "SyntheticFrameDecoder",
    "DatasetFrameDecoder",
    "FrameData",
    "KeyframeSelectionConfig",
]
