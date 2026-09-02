"""Canonical frame and timeline representations for video ingestion."""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator
from src.ingestion.exceptions import InvalidVideoMetadataError


@dataclass(frozen=True)
class VideoProvenance:
    """Cryptographic and operational provenance metadata for an ingested video file."""
    source_file_path: str
    file_size_bytes: int
    sha256_checksum: str
    ingestion_timestamp_utc: str
    metadata_extractor: str  # e.g., "ISOBMFFParser_v1.0"
    timestamp_source: str    # e.g., "container_ctts_pts", "container_stts_dts", "nominal_fps_fallback"


@dataclass(frozen=True)
class VideoMetadata:
    """Container for video container, codec, and temporal properties."""
    filepath: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str = "unknown"
    has_audio: bool = False
    telemetry_stream_type: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalFrame:
    """Canonical representation of an individual video frame in the timeline.
    
    Timestamp Semantics:
    - timestamp_seconds: Exact presentation timestamp (PTS) in fractional seconds from video start (t=0.0).
      Calculated as: pts / timescale.
    - pts: Integer Presentation TimeStamp in stream timescale units.
    - timescale: Integer ticks per second of the media stream.
    - frame_index: 0-based sequential frame index in decoding/display order.
    - frame_id: Unique deterministic frame identifier string: f"{video_id}_{frame_index:06d}".
    - is_keyframe: True if this frame is an IDR/Sync frame with zero inter-frame dependency.
    """
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    pts: int
    timescale: int
    source_video: str
    width: int
    height: int
    is_keyframe: bool = False
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise InvalidVideoMetadataError(f"CanonicalFrame 'frame_index' cannot be negative ({self.frame_index}).")
        if math.isnan(self.timestamp_seconds) or self.timestamp_seconds < 0.0:
            raise InvalidVideoMetadataError(f"CanonicalFrame 'timestamp_seconds' must be >= 0.0, got {self.timestamp_seconds}.")
        if self.width <= 0 or self.height <= 0:
            raise InvalidVideoMetadataError(f"CanonicalFrame dimensions must be positive, got {self.width}x{self.height}.")
        if self.timescale <= 0:
            raise InvalidVideoMetadataError(f"CanonicalFrame 'timescale' must be positive, got {self.timescale}.")


@dataclass
class CanonicalTimeline:
    """Discrete, ordered timeline of all canonical frames extracted from a single video track."""
    video_id: str
    source_path: str
    total_frames: int
    duration_seconds: float
    nominal_fps: float
    width: int
    height: int
    frames: List[CanonicalFrame]
    provenance: VideoProvenance

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate logical consistency, non-empty frames, and monotonic timestamp ordering."""
        if not self.frames:
            raise InvalidVideoMetadataError("CanonicalTimeline must contain at least one frame.")

        if self.total_frames != len(self.frames):
            raise InvalidVideoMetadataError(
                f"CanonicalTimeline total_frames count ({self.total_frames}) does not match frame list length ({len(self.frames)})."
            )

        prev_timestamp = -1.0
        for i, frame in enumerate(self.frames):
            if frame.frame_index != i:
                raise InvalidVideoMetadataError(
                    f"Frame index discontinuity at position {i}: expected frame_index {i}, got {frame.frame_index}."
                )
            if frame.timestamp_seconds < prev_timestamp:
                raise InvalidVideoMetadataError(
                    f"Non-monotonic timestamp detected at frame {i}: timestamp {frame.timestamp_seconds} < previous {prev_timestamp}."
                )
            prev_timestamp = frame.timestamp_seconds

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> CanonicalFrame:
        return self.frames[idx]

    def __iter__(self) -> Iterator[CanonicalFrame]:
        return iter(self.frames)

    def get_frame(self, frame_index: int) -> CanonicalFrame:
        """Retrieve canonical frame by 0-based index."""
        if 0 <= frame_index < len(self.frames):
            return self.frames[frame_index]
        raise IndexError(f"Frame index {frame_index} out of range for timeline with {len(self.frames)} frames.")

    def get_keyframes(self) -> List[CanonicalFrame]:
        """Return subset of frames marked as keyframes / sync frames."""
        return [f for f in self.frames if f.is_keyframe]

    def get_frame_at_timestamp(self, timestamp_seconds: float, tolerance_seconds: float = 0.05) -> Optional[CanonicalFrame]:
        """Find the closest frame to target timestamp within tolerance."""
        if not self.frames:
            return None
        # Binary search for closest timestamp
        left, right = 0, len(self.frames) - 1
        best_frame = self.frames[0]
        min_diff = abs(best_frame.timestamp_seconds - timestamp_seconds)

        while left <= right:
            mid = (left + right) // 2
            curr = self.frames[mid]
            diff = abs(curr.timestamp_seconds - timestamp_seconds)
            if diff < min_diff:
                min_diff = diff
                best_frame = curr

            if curr.timestamp_seconds < timestamp_seconds:
                left = mid + 1
            elif curr.timestamp_seconds > timestamp_seconds:
                right = mid - 1
            else:
                return curr

        if min_diff <= tolerance_seconds:
            return best_frame
        return None
