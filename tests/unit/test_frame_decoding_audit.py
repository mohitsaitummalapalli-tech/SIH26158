"""Focused audit tests for Phase 2A.1 Frame Decoder Correctness & Timestamp Integrity.

DISCLAIMER:
ALL FIXTURES IN THIS MODULE ARE SYNTHETIC TEST DATA GENERATED SOLELY FOR
CORRECTNESS AND TIMESTAMP AUDITING. THEY DO NOT REPRESENT A REAL DRONE FLIGHT.
"""

import pytest
import os
import cv2
import numpy as np

from src.preprocessing import (
    DecodeStatus,
    DecodeConfig,
    DecodedFrame,
    FrameDecoder,
    OpenCVFrameDecoder,
    SyntheticFrameDecoder,
    DatasetFrameDecoder,
)
from src.ingestion import (
    VideoMetadata,
    CanonicalFrame,
    CanonicalTimeline,
    VideoProvenance,
    TelemetryPosition,
    CanonicalTelemetryStream,
    TelemetryRecord,
    CanonicalDatasetBuilder,
)


@pytest.fixture
def vfr_timeline() -> CanonicalTimeline:
    """Synthetic Variable Frame Rate (VFR) timeline with non-uniform time steps (TEST DATA)."""
    vfr_timestamps = [0.0, 0.33, 0.75, 1.40, 2.10, 2.50, 3.20]
    frames = [
        CanonicalFrame(
            frame_id=f"vfr_frame_{i:04d}",
            frame_index=i,
            timestamp_seconds=ts,
            pts=int(ts * 1000),
            timescale=1000,
            source_video="vfr_test.mp4",
            width=640,
            height=480,
            is_keyframe=(i == 0 or i == 4),
        )
        for i, ts in enumerate(vfr_timestamps)
    ]
    prov = VideoProvenance(
        source_file_path="vfr_test.mp4",
        file_size_bytes=4096,
        sha256_checksum="v" * 64,
        ingestion_timestamp_utc="2026-09-02T00:00:00Z",
        metadata_extractor="ISOBMFFParser_v1.1",
        timestamp_source="container_ctts_pts",
    )
    return CanonicalTimeline(
        video_id="vfr_flight",
        source_path="vfr_test.mp4",
        total_frames=len(frames),
        duration_seconds=3.20,
        nominal_fps=2.1875,
        width=640,
        height=480,
        frames=frames,
        provenance=prov,
    )


@pytest.fixture
def bframe_timeline() -> CanonicalTimeline:
    """Synthetic B-frame timeline where composition time (PTS) differs from decode time (DTS) (TEST DATA)."""
    # Presentation order 0, 1, 2, 3, 4 with PTS = [0.0, 0.5, 1.0, 1.5, 2.0]
    frames = [
        CanonicalFrame(
            frame_id=f"bframe_{i:04d}",
            frame_index=i,
            timestamp_seconds=i * 0.5,
            pts=i * 500,
            timescale=1000,
            source_video="bframe_test.mp4",
            width=640,
            height=480,
            is_keyframe=(i == 0),
        )
        for i in range(5)
    ]
    prov = VideoProvenance(
        source_file_path="bframe_test.mp4",
        file_size_bytes=2048,
        sha256_checksum="b" * 64,
        ingestion_timestamp_utc="2026-09-02T00:00:00Z",
        metadata_extractor="ISOBMFFParser_v1.1",
        timestamp_source="container_ctts_pts",
    )
    return CanonicalTimeline(
        video_id="bframe_flight",
        source_path="bframe_test.mp4",
        total_frames=5,
        duration_seconds=2.0,
        nominal_fps=2.0,
        width=640,
        height=480,
        frames=frames,
        provenance=prov,
    )


# 1. Audit Test: VFR Timestamp Preservation
def test_vfr_timestamp_preservation(vfr_timeline):
    decoder = SyntheticFrameDecoder(
        total_frames=len(vfr_timeline), width=640, height=480, timeline=vfr_timeline
    )

    expected_pts = [0.0, 0.33, 0.75, 1.40, 2.10, 2.50, 3.20]
    for i, exp_ts in enumerate(expected_pts):
        frame = decoder.decode_frame(i)
        assert frame.decode_status == DecodeStatus.SUCCESS
        assert frame.timestamp_seconds == exp_ts
        assert frame.timestamp_source == "canonical_timeline_pts"
        # Confirm that timestamp does NOT equal naive index / nominal_fps
        if i in [1, 2, 3, 4]:
            assert frame.timestamp_seconds != i / vfr_timeline.nominal_fps


# 2. Audit Test: B-Frame Presentation Order & PTS Preservation
def test_bframe_pts_preservation(bframe_timeline):
    decoder = SyntheticFrameDecoder(
        total_frames=len(bframe_timeline), width=640, height=480, timeline=bframe_timeline
    )

    for i in range(len(bframe_timeline)):
        frame = decoder.decode_frame(i)
        assert frame.decode_status == DecodeStatus.SUCCESS
        assert frame.frame_index == i
        assert frame.frame_id == f"bframe_{i:04d}"
        assert frame.timestamp_seconds == i * 0.5


# 3. Audit Test: Sequential Ordering Consistency
def test_sequential_ordering_consistency(vfr_timeline):
    decoder = SyntheticFrameDecoder(
        total_frames=len(vfr_timeline), width=640, height=480, timeline=vfr_timeline
    )

    stream_frames = list(decoder.iter_frames())
    assert len(stream_frames) == len(vfr_timeline)

    prev_ts = -1.0
    for i, frame in enumerate(stream_frames):
        assert frame.frame_index == i
        assert frame.timestamp_seconds > prev_ts
        assert frame.frame_id == vfr_timeline[i].frame_id
        prev_ts = frame.timestamp_seconds


# 4. Audit Test: Random Access vs Sequential Reference Equivalence on Deterministic Media
def test_random_access_vs_sequential_consistency(vfr_timeline):
    decoder = SyntheticFrameDecoder(
        total_frames=len(vfr_timeline), width=640, height=480, timeline=vfr_timeline
    )

    sequential_frames = list(decoder.iter_frames())

    for i in range(len(vfr_timeline)):
        random_frame = decoder.decode_frame(i)
        seq_frame = sequential_frames[i]

        assert random_frame.frame_id == seq_frame.frame_id
        assert random_frame.timestamp_seconds == seq_frame.timestamp_seconds
        assert random_frame.data is not None
        assert seq_frame.data is not None
        assert np.array_equal(random_frame.data, seq_frame.data)


# 5. Audit Test: Mid-Stream Decode Failure Handling
def test_midstream_decode_failure(vfr_timeline):
    # Corrupt frame index 3
    decoder = SyntheticFrameDecoder(
        total_frames=len(vfr_timeline),
        width=640,
        height=480,
        timeline=vfr_timeline,
        corrupt_indices=[3],
    )

    # Frame 3 random access
    frame_3 = decoder.decode_frame(3)
    assert frame_3.decode_status == DecodeStatus.CORRUPTED
    assert frame_3.data is None
    assert frame_3.frame_index == 3
    assert frame_3.timestamp_seconds == vfr_timeline[3].timestamp_seconds # Metadata preserved, no fake substitute

    # Frame 2 and 4 are unaffected
    assert decoder.decode_frame(2).decode_status == DecodeStatus.SUCCESS
    assert decoder.decode_frame(4).decode_status == DecodeStatus.SUCCESS


# 6. Audit Test: Asymmetric Color Correctness (Channel Swap Detection)
def test_asymmetric_color_channel_order():
    decoder = SyntheticFrameDecoder(total_frames=1, width=640, height=480)
    frame = decoder.decode_frame(0)

    assert frame.channel_layout == "RGB"
    assert frame.data is not None
    # Top-Left quadrant (y: 0..240, x: 0..320) must be Pure RED: [255, 0, 0]
    tl_pixel = frame.data[50, 50]
    assert tl_pixel[0] == 255 # Red
    assert tl_pixel[1] == 0   # Green
    assert tl_pixel[2] == 0   # Blue

    # Bottom-Left quadrant (y: 240..480, x: 0..320) must be Pure BLUE: [0, 0, 255]
    bl_pixel = frame.data[350, 50]
    assert bl_pixel[0] == 0   # Red
    assert bl_pixel[1] == 0   # Green
    assert bl_pixel[2] == 255 # Blue


# 7. Audit Test: Explicit Resizing Configuration Preservation
def test_explicit_resize_preservation(vfr_timeline):
    config = DecodeConfig(target_width=320, target_height=240)
    decoder = SyntheticFrameDecoder(
        total_frames=len(vfr_timeline), width=640, height=480, timeline=vfr_timeline, config=config
    )
    frame = decoder.decode_frame(2)

    assert frame.is_resized is True
    assert frame.width == 320
    assert frame.height == 240
    assert frame.original_width == 640
    assert frame.original_height == 480
    assert frame.timestamp_seconds == vfr_timeline[2].timestamp_seconds
    assert frame.data is not None
    assert frame.data.shape == (240, 320, 3)
