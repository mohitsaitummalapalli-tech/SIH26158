"""Frame decoding subsystem: Converts video streams and dataset frame references into canonical RGB image representations.

Canon:
- Channel Order: RGB (OpenCV BGR is explicitly converted to RGB on ingestion)
- Data Type: uint8 (Values in [0, 255])
- Memory: Sequential generator streaming to prevent high VRAM/RAM allocation on long drone flights
- Timestamp Source: Canonical Phase 1A Timeline PTS (not derived from frame_index / FPS)
"""

import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Iterator, Tuple

from src.ingestion.canonical_timeline import CanonicalFrame, CanonicalTimeline
from src.ingestion.dataset import CanonicalFlightDataset, CanonicalFrameObservation


class DecodeStatus(str, Enum):
    """Execution status for an individual frame decode operation."""
    SUCCESS = "SUCCESS"
    CORRUPTED = "CORRUPTED"
    DECODER_ERROR = "DECODER_ERROR"
    INDEX_OUT_OF_BOUNDS = "INDEX_OUT_OF_BOUNDS"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"


@dataclass(frozen=True)
class DecodeConfig:
    """Configuration parameters for video frame decoding."""
    target_width: Optional[int] = None           # Optional resize width (None = original resolution)
    target_height: Optional[int] = None          # Optional resize height (None = original resolution)
    channel_order: str = "RGB"                   # Canonical output format: "RGB" or "GRAY"
    dtype: str = "uint8"                         # Canonical pixel dtype: "uint8" (0-255)
    interpolation: int = cv2.INTER_LINEAR        # Interpolation flag if resized


@dataclass
class DecodedFrame:
    """Canonical, typed in-memory image representation linked to canonical timeline."""
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    width: int
    height: int
    channels: int
    channel_layout: str
    dtype: str
    data: Optional[np.ndarray]
    source_video: str
    decode_status: DecodeStatus
    timestamp_source: str = "canonical_timeline_pts"
    is_resized: bool = False
    original_width: Optional[int] = None
    original_height: Optional[int] = None
    decoder_backend: str = "OpenCVFrameDecoder"
    provenance: Optional[Dict[str, Any]] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Verify image data integrity, dimensional consistency, and dtype."""
        if self.decode_status == DecodeStatus.SUCCESS:
            if self.data is None:
                raise ValueError(f"DecodedFrame '{self.frame_id}' has SUCCESS status but data is None.")
            if not isinstance(self.data, np.ndarray):
                raise TypeError(f"DecodedFrame data must be a numpy.ndarray, got {type(self.data)}.")
            if self.data.dtype != np.dtype(self.dtype):
                raise ValueError(f"DecodedFrame data dtype {self.data.dtype} does not match declared dtype {self.dtype}.")
            if self.channels == 3:
                expected_shape = (self.height, self.width, 3)
                if self.data.shape != expected_shape:
                    raise ValueError(f"DecodedFrame shape mismatch: expected {expected_shape}, got {self.data.shape}.")
            elif self.channels == 1:
                expected_shape = (self.height, self.width)
                if self.data.shape != expected_shape and self.data.shape != (self.height, self.width, 1):
                    raise ValueError(f"DecodedFrame shape mismatch: expected {expected_shape}, got {self.data.shape}.")
            if not np.all(np.isfinite(self.data)):
                raise ValueError(f"DecodedFrame '{self.frame_id}' contains NaN or Infinite values.")

    @property
    def is_valid(self) -> bool:
        return self.decode_status == DecodeStatus.SUCCESS and self.data is not None


class FrameDecoder(ABC):
    """Abstract interface for video frame decoders."""

    @abstractmethod
    def decode_frame(self, frame_index: int) -> DecodedFrame:
        """Decode a single frame by 0-based presentation frame index (random access)."""
        pass

    @abstractmethod
    def decode_observation(self, obs: CanonicalFrameObservation) -> DecodedFrame:
        """Decode frame directly from a CanonicalFrameObservation."""
        pass

    @abstractmethod
    def iter_frames(
        self, start_index: int = 0, stop_index: Optional[int] = None, step: int = 1
    ) -> Iterator[DecodedFrame]:
        """Memory-safe sequential generator yielding decoded frames one at a time."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release underlying media handles."""
        pass

    def __enter__(self) -> "FrameDecoder":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class OpenCVFrameDecoder(FrameDecoder):
    """Production frame decoder backed by OpenCV VideoCapture with RGB conversion.
    
    LIMITATION NOTE ON RANDOM-ACCESS SEEKING:
    OpenCV's CAP_PROP_POS_FRAMES interacts with platform demuxers (FFMPEG, MSMF, GStreamer).
    On complex Variable Frame Rate (VFR) streams or long GOP B-frame encodings, seeking by index
    can land on keyframes or decode DTS frames rather than exact PTS presentation frames.
    Sequential streaming (iter_frames()) starting from frame 0 is the authoritative scientific
    reference path for deterministic presentation-order decoding.
    """

    def __init__(
        self,
        video_path: str,
        timeline: Optional[CanonicalTimeline] = None,
        config: Optional[DecodeConfig] = None,
    ) -> None:
        self.video_path = video_path
        self.timeline = timeline
        self.config = config or DecodeConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._total_frames: int = 0
        self._native_width: int = 0
        self._native_height: int = 0
        self._fps: float = 0.0

        if not os.path.exists(video_path):
            self._file_exists = False
        else:
            self._file_exists = True
            self._init_capture()

    def _init_capture(self) -> None:
        """Initialize OpenCV VideoCapture."""
        self._cap = cv2.VideoCapture(self.video_path)
        if self._cap.isOpened():
            self._native_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._native_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        else:
            self._cap = None

    def _get_timestamp_and_source(self, frame_index: int) -> Tuple[float, str]:
        """Retrieve timestamp from canonical timeline if available, otherwise nominal fps."""
        if self.timeline and 0 <= frame_index < len(self.timeline):
            return self.timeline[frame_index].timestamp_seconds, "canonical_timeline_pts"
        if self._fps > 0:
            return frame_index / self._fps, "inferred_nominal_fps"
        return 0.0, "unknown"

    def _get_frame_id_for_index(self, frame_index: int) -> str:
        """Retrieve canonical frame_id from timeline if available."""
        if self.timeline and 0 <= frame_index < len(self.timeline):
            return self.timeline[frame_index].frame_id
        base = os.path.basename(self.video_path)
        return f"{base}_{frame_index:06d}"

    def decode_frame(self, frame_index: int) -> DecodedFrame:
        """Random-access frame decode by index."""
        ts, ts_src = self._get_timestamp_and_source(frame_index)
        f_id = self._get_frame_id_for_index(frame_index)

        if not self._file_exists:
            return DecodedFrame(
                frame_id=f_id,
                frame_index=frame_index,
                timestamp_seconds=ts,
                width=0,
                height=0,
                channels=0,
                channel_layout="NONE",
                dtype="uint8",
                data=None,
                source_video=self.video_path,
                decode_status=DecodeStatus.FILE_NOT_FOUND,
                timestamp_source=ts_src,
            )

        if self._cap is None or not self._cap.isOpened():
            return DecodedFrame(
                frame_id=f_id,
                frame_index=frame_index,
                timestamp_seconds=ts,
                width=0,
                height=0,
                channels=0,
                channel_layout="NONE",
                dtype="uint8",
                data=None,
                source_video=self.video_path,
                decode_status=DecodeStatus.DECODER_ERROR,
                timestamp_source=ts_src,
            )

        if frame_index < 0 or (self._total_frames > 0 and frame_index >= self._total_frames):
            return DecodedFrame(
                frame_id=f_id,
                frame_index=frame_index,
                timestamp_seconds=ts,
                width=0,
                height=0,
                channels=0,
                channel_layout="NONE",
                dtype="uint8",
                data=None,
                source_video=self.video_path,
                decode_status=DecodeStatus.INDEX_OUT_OF_BOUNDS,
                timestamp_source=ts_src,
            )

        # Seek to frame position
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame_bgr = self._cap.read()

        if not ret or frame_bgr is None:
            return DecodedFrame(
                frame_id=f_id,
                frame_index=frame_index,
                timestamp_seconds=ts,
                width=0,
                height=0,
                channels=0,
                channel_layout="NONE",
                dtype="uint8",
                data=None,
                source_video=self.video_path,
                decode_status=DecodeStatus.CORRUPTED,
                timestamp_source=ts_src,
            )

        # Canonical Conversion: BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = frame_rgb.shape[:2]
        is_resized = False

        if (
            self.config.target_width is not None
            and self.config.target_height is not None
            and (orig_w != self.config.target_width or orig_h != self.config.target_height)
        ):
            frame_rgb = cv2.resize(
                frame_rgb,
                (self.config.target_width, self.config.target_height),
                interpolation=self.config.interpolation,
            )
            is_resized = True

        h, w = frame_rgb.shape[:2]

        decoded = DecodedFrame(
            frame_id=f_id,
            frame_index=frame_index,
            timestamp_seconds=ts,
            width=w,
            height=h,
            channels=3,
            channel_layout=self.config.channel_order,
            dtype=self.config.dtype,
            data=frame_rgb,
            source_video=self.video_path,
            decode_status=DecodeStatus.SUCCESS,
            timestamp_source=ts_src,
            is_resized=is_resized,
            original_width=orig_w,
            original_height=orig_h,
            decoder_backend="OpenCVFrameDecoder",
            provenance={"source_path": self.video_path, "seek_index": frame_index, "access_mode": "random_access"},
        )
        decoded.validate()
        return decoded

    def decode_observation(self, obs: CanonicalFrameObservation) -> DecodedFrame:
        """Decode frame directly from a CanonicalFrameObservation."""
        return self.decode_frame(obs.frame_index)

    def iter_frames(
        self, start_index: int = 0, stop_index: Optional[int] = None, step: int = 1
    ) -> Iterator[DecodedFrame]:
        """Sequential memory-safe frame stream generator in presentation order."""
        if not self._file_exists or self._cap is None:
            return

        max_frames = stop_index if stop_index is not None else self._total_frames
        curr_idx = start_index

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, curr_idx)

        while curr_idx < max_frames:
            ret, frame_bgr = self._cap.read()
            if not ret or frame_bgr is None:
                # Return corrupted frame marker without crashing generator
                ts, ts_src = self._get_timestamp_and_source(curr_idx)
                f_id = self._get_frame_id_for_index(curr_idx)
                yield DecodedFrame(
                    frame_id=f_id,
                    frame_index=curr_idx,
                    timestamp_seconds=ts,
                    width=0,
                    height=0,
                    channels=0,
                    channel_layout="NONE",
                    dtype="uint8",
                    data=None,
                    source_video=self.video_path,
                    decode_status=DecodeStatus.CORRUPTED,
                    timestamp_source=ts_src,
                )
                break

            if (curr_idx - start_index) % step == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = frame_rgb.shape[:2]
                is_resized = False

                if (
                    self.config.target_width is not None
                    and self.config.target_height is not None
                    and (orig_w != self.config.target_width or orig_h != self.config.target_height)
                ):
                    frame_rgb = cv2.resize(
                        frame_rgb,
                        (self.config.target_width, self.config.target_height),
                        interpolation=self.config.interpolation,
                    )
                    is_resized = True

                h, w = frame_rgb.shape[:2]
                ts, ts_src = self._get_timestamp_and_source(curr_idx)
                f_id = self._get_frame_id_for_index(curr_idx)

                decoded = DecodedFrame(
                    frame_id=f_id,
                    frame_index=curr_idx,
                    timestamp_seconds=ts,
                    width=w,
                    height=h,
                    channels=3,
                    channel_layout=self.config.channel_order,
                    dtype=self.config.dtype,
                    data=frame_rgb,
                    source_video=self.video_path,
                    decode_status=DecodeStatus.SUCCESS,
                    timestamp_source=ts_src,
                    is_resized=is_resized,
                    original_width=orig_w,
                    original_height=orig_h,
                    decoder_backend="OpenCVFrameDecoder",
                    provenance={"source_path": self.video_path, "stream_index": curr_idx, "access_mode": "sequential"},
                )
                decoded.validate()
                yield decoded

            curr_idx += 1

    def close(self) -> None:
        """Release VideoCapture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class SyntheticFrameDecoder(FrameDecoder):
    """Deterministic synthetic test decoder generating test card pattern frames in memory (TEST DATA)."""

    def __init__(
        self,
        total_frames: int = 10,
        width: int = 640,
        height: int = 480,
        fps: float = 2.0,
        source_name: str = "synthetic_flight.mp4",
        timeline: Optional[CanonicalTimeline] = None,
        config: Optional[DecodeConfig] = None,
        corrupt_indices: Optional[List[int]] = None,
    ) -> None:
        self.total_frames = total_frames
        self.width = width
        self.height = height
        self.fps = fps
        self.source_name = source_name
        self.timeline = timeline
        self.config = config or DecodeConfig()
        self.corrupt_indices = set(corrupt_indices or [])

    def _generate_synthetic_image(self, frame_index: int) -> np.ndarray:
        """Generate deterministic asymmetric RGB test pattern for frame."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Asymmetric color blocks: Top-Left=Red, Top-Right=Green, Bottom-Left=Blue, Bottom-Right=Yellow
        half_h, half_w = self.height // 2, self.width // 2
        img[0:half_h, 0:half_w] = [255, 0, 0]      # Top-Left: Pure RED
        img[0:half_h, half_w:] = [0, 255, 0]       # Top-Right: Pure GREEN
        img[half_h:, 0:half_w] = [0, 0, 255]       # Bottom-Left: Pure BLUE
        img[half_h:, half_w:] = [255, 255, 0]      # Bottom-Right: Pure YELLOW

        # Frame index identifier band in center
        img[half_h-5:half_h+5, :] = (frame_index * 20) % 255
        return img

    def decode_frame(self, frame_index: int) -> DecodedFrame:
        if frame_index < 0 or frame_index >= self.total_frames:
            ts = self.timeline[frame_index].timestamp_seconds if (self.timeline and 0 <= frame_index < len(self.timeline)) else frame_index / self.fps
            return DecodedFrame(
                frame_id=f"frame_{frame_index:06d}",
                frame_index=frame_index,
                timestamp_seconds=ts,
                width=0,
                height=0,
                channels=0,
                channel_layout="NONE",
                dtype="uint8",
                data=None,
                source_video=self.source_name,
                decode_status=DecodeStatus.INDEX_OUT_OF_BOUNDS,
                timestamp_source="canonical_timeline_pts" if self.timeline else "inferred_nominal_fps",
            )

        ts = self.timeline[frame_index].timestamp_seconds if self.timeline else frame_index / self.fps
        f_id = self.timeline[frame_index].frame_id if self.timeline else f"frame_{frame_index:06d}"
        ts_src = "canonical_timeline_pts" if self.timeline else "inferred_nominal_fps"

        if frame_index in self.corrupt_indices:
            return DecodedFrame(
                frame_id=f_id,
                frame_index=frame_index,
                timestamp_seconds=ts,
                width=0,
                height=0,
                channels=0,
                channel_layout="NONE",
                dtype="uint8",
                data=None,
                source_video=self.source_name,
                decode_status=DecodeStatus.CORRUPTED,
                timestamp_source=ts_src,
            )

        img_rgb = self._generate_synthetic_image(frame_index)
        orig_h, orig_w = img_rgb.shape[:2]
        is_resized = False

        if (
            self.config.target_width is not None
            and self.config.target_height is not None
            and (orig_w != self.config.target_width or orig_h != self.config.target_height)
        ):
            img_rgb = cv2.resize(
                img_rgb,
                (self.config.target_width, self.config.target_height),
                interpolation=self.config.interpolation,
            )
            is_resized = True

        h, w = img_rgb.shape[:2]

        decoded = DecodedFrame(
            frame_id=f_id,
            frame_index=frame_index,
            timestamp_seconds=ts,
            width=w,
            height=h,
            channels=3,
            channel_layout=self.config.channel_order,
            dtype=self.config.dtype,
            data=img_rgb,
            source_video=self.source_name,
            decode_status=DecodeStatus.SUCCESS,
            timestamp_source=ts_src,
            is_resized=is_resized,
            original_width=orig_w,
            original_height=orig_h,
            decoder_backend="SyntheticFrameDecoder",
            provenance={"source_path": self.source_name, "is_synthetic": True},
        )
        decoded.validate()
        return decoded

    def decode_observation(self, obs: CanonicalFrameObservation) -> DecodedFrame:
        return self.decode_frame(obs.frame_index)

    def iter_frames(
        self, start_index: int = 0, stop_index: Optional[int] = None, step: int = 1
    ) -> Iterator[DecodedFrame]:
        max_idx = stop_index if stop_index is not None else self.total_frames
        for idx in range(start_index, max_idx, step):
            yield self.decode_frame(idx)

    def close(self) -> None:
        pass


class DatasetFrameDecoder:
    """High-level decoder wrapper associating CanonicalFlightDataset with an underlying FrameDecoder backend."""

    def __init__(
        self,
        dataset: CanonicalFlightDataset,
        decoder: Optional[FrameDecoder] = None,
        config: Optional[DecodeConfig] = None,
    ) -> None:
        self.dataset = dataset
        self.config = config or DecodeConfig()
        if decoder is not None:
            self.decoder = decoder
        else:
            # Default to OpenCV decoder using dataset video path and timeline
            self.decoder = OpenCVFrameDecoder(
                video_path=dataset.video_metadata.filepath,
                timeline=dataset.timeline,
                config=self.config,
            )

    def decode_frame(self, frame_index: int) -> DecodedFrame:
        """Decode frame matching frame_index in dataset."""
        obs = self.dataset[frame_index]
        return self.decoder.decode_observation(obs)

    def decode_frame_by_id(self, frame_id: str) -> Optional[DecodedFrame]:
        """Decode frame matching frame_id in dataset."""
        obs = self.dataset.get_frame_by_id(frame_id)
        if obs is None:
            return None
        return self.decoder.decode_observation(obs)

    def iter_frames(
        self, start_index: int = 0, stop_index: Optional[int] = None, step: int = 1
    ) -> Iterator[Tuple[CanonicalFrameObservation, DecodedFrame]]:
        """Memory-safe sequential generator yielding (CanonicalFrameObservation, DecodedFrame) tuples."""
        max_idx = stop_index if stop_index is not None else len(self.dataset)
        for idx in range(start_index, max_idx, step):
            obs = self.dataset[idx]
            frame = self.decoder.decode_observation(obs)
            yield (obs, frame)

    def close(self) -> None:
        self.decoder.close()

    def __enter__(self) -> "DatasetFrameDecoder":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
