"""VideoSource abstract interface and ISOBMFF / MOV implementation."""

import os
import hashlib
import pathlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Optional, Set

from src.ingestion.exceptions import (
    VideoNotFoundError,
    UnsupportedVideoFormatError,
    CorruptVideoError,
    InvalidVideoMetadataError,
    UnsupportedFragmentedMP4Error,
)
from src.ingestion.isobmff_parser import ISOBMFFParser, ParsedContainerInfo, VideoTrackInfo
from src.ingestion.canonical_timeline import CanonicalFrame, CanonicalTimeline, VideoProvenance


class VideoSource(ABC):
    """Abstract base class for all video ingestion sources."""

    SUPPORTED_EXTENSIONS: Set[str] = {".mp4", ".mov"}

    def __init__(self, filepath: str) -> None:
        self.filepath = os.path.abspath(filepath)
        self._validate_file_path()

    def _validate_file_path(self) -> None:
        """Validate that file exists and extension is supported."""
        if not os.path.exists(self.filepath):
            raise VideoNotFoundError(f"Video file not found at path: '{self.filepath}'")

        if not os.path.isfile(self.filepath):
            raise VideoNotFoundError(f"Path is not a regular file: '{self.filepath}'")

        ext = pathlib.Path(self.filepath).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedVideoFormatError(
                f"Unsupported video file extension '{ext}'. Supported formats: {sorted(list(self.SUPPORTED_EXTENSIONS))}"
            )

    @classmethod
    def from_file(cls, filepath: str) -> "VideoSource":
        """Factory method returning the appropriate VideoSource implementation."""
        ext = pathlib.Path(filepath).suffix.lower()
        if ext in {".mp4", ".mov"}:
            return ISOBMFFVideoSource(filepath)
        raise UnsupportedVideoFormatError(
            f"Unsupported video file extension '{ext}'. Supported formats: {sorted(list(cls.SUPPORTED_EXTENSIONS))}"
        )

    @abstractmethod
    def build_canonical_timeline(self) -> CanonicalTimeline:
        """Construct the canonical timeline of presentation-timestamped frames."""
        pass


class ISOBMFFVideoSource(VideoSource):
    """Production video source for MP4 and QuickTime MOV containers."""

    def __init__(self, filepath: str) -> None:
        super().__init__(filepath)
        self._provenance: Optional[VideoProvenance] = None
        self._parsed_container: Optional[ParsedContainerInfo] = None

    def _compute_sha256(self) -> str:
        """Calculate SHA256 checksum of source video for provenance verification."""
        hasher = hashlib.sha256()
        with open(self.filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _ensure_parsed(self) -> ParsedContainerInfo:
        if self._parsed_container is None:
            self._parsed_container = ISOBMFFParser.parse_file(self.filepath)
        return self._parsed_container

    def get_provenance(self) -> VideoProvenance:
        """Return cryptographic provenance metadata for this video source."""
        if self._provenance is None:
            file_stat = os.stat(self.filepath)
            checksum = self._compute_sha256()
            now_iso = datetime.now(timezone.utc).isoformat()
            self._provenance = VideoProvenance(
                source_file_path=self.filepath,
                file_size_bytes=file_stat.st_size,
                sha256_checksum=checksum,
                ingestion_timestamp_utc=now_iso,
                metadata_extractor="ISOBMFFParser_v1.1",
                timestamp_source="container_pts",
            )
        return self._provenance

    def _select_primary_video_track(self, tracks: List[VideoTrackInfo]) -> VideoTrackInfo:
        """Select the primary video track (highest resolution)."""
        if not tracks:
            raise InvalidVideoMetadataError(f"No video tracks found in '{self.filepath}'.")
        if len(tracks) == 1:
            return tracks[0]
        return max(tracks, key=lambda t: t.width * t.height)

    def build_canonical_timeline(self) -> CanonicalTimeline:
        """Extract metadata and build the discrete CanonicalTimeline in Presentation Order (PTS)."""
        container = self._ensure_parsed()
        provenance = self.get_provenance()

        vtrack = self._select_primary_video_track(container.video_tracks)

        if vtrack.width <= 0 or vtrack.height <= 0:
            raise InvalidVideoMetadataError(
                f"Invalid video dimensions: {vtrack.width}x{vtrack.height} in '{self.filepath}'."
            )

        if vtrack.timescale <= 0:
            raise InvalidVideoMetadataError(
                f"Invalid stream timescale ({vtrack.timescale}) in '{self.filepath}'."
            )

        duration_sec = vtrack.duration_ticks / float(vtrack.timescale)
        if duration_sec <= 0.0:
            raise InvalidVideoMetadataError(
                f"Invalid zero or negative duration ({duration_sec}s) in '{self.filepath}'."
            )

        total_frames = vtrack.sample_count
        if total_frames <= 0:
            raise InvalidVideoMetadataError(
                f"No frames/samples found in video track for '{self.filepath}'."
            )

        nominal_fps = float(total_frames) / duration_sec if duration_sec > 0 else 0.0

        # Check for Edit List (elst) start offset
        start_offset_sec = 0.0
        if vtrack.edit_list_entries:
            first_edit = vtrack.edit_list_entries[0]
            if first_edit.media_time == -1:
                start_offset_sec = first_edit.segment_duration / float(container.movie_timescale)

        keyframe_set = set(vtrack.keyframe_indices)
        pts_list = vtrack.pts_list
        dts_list = vtrack.dts_list
        timescale = vtrack.timescale
        video_id = pathlib.Path(self.filepath).stem

        # Build raw frame items with (dts_index, raw_pts)
        raw_frames = []
        for i in range(total_frames):
            raw_pts = pts_list[i] if i < len(pts_list) else i * (vtrack.duration_ticks // total_frames)
            raw_dts = dts_list[i] if i < len(dts_list) else raw_pts
            ts_sec = (float(raw_pts) / float(timescale)) + start_offset_sec
            is_key = i in keyframe_set
            raw_frames.append({
                "sample_index": i,
                "pts": raw_pts,
                "dts": raw_dts,
                "timestamp_seconds": ts_sec,
                "is_keyframe": is_key
            })

        # When B-frames are present, samples in container are in DTS order.
        # Canonical Timeline orders frames in true presentation order (PTS).
        raw_frames.sort(key=lambda item: (item["pts"], item["sample_index"]))

        frames: List[CanonicalFrame] = []
        for display_idx, item in enumerate(raw_frames):
            frame = CanonicalFrame(
                frame_id=f"{video_id}_{display_idx:06d}",
                frame_index=display_idx,
                timestamp_seconds=item["timestamp_seconds"],
                pts=item["pts"],
                timescale=timescale,
                source_video=self.filepath,
                width=vtrack.width,
                height=vtrack.height,
                is_keyframe=item["is_keyframe"],
                extra_metadata={
                    "codec": vtrack.codec_fourcc,
                    "compressor_name": vtrack.compressor_name,
                    "sample_index": item["sample_index"],
                    "dts": item["dts"],
                    "is_vfr": vtrack.is_variable_frame_rate,
                    "has_b_frames": vtrack.has_b_frames,
                    "track_id": vtrack.track_id,
                }
            )
            frames.append(frame)

        return CanonicalTimeline(
            video_id=video_id,
            source_path=self.filepath,
            total_frames=total_frames,
            duration_seconds=duration_sec + start_offset_sec,
            nominal_fps=nominal_fps,
            width=vtrack.width,
            height=vtrack.height,
            frames=frames,
            provenance=provenance,
        )
