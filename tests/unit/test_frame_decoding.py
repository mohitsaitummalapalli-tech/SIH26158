"""Deterministic unit tests for Phase 2A Frame Decoding & Canonical Image Representation.

DISCLAIMER:
ALL FIXTURES IN THIS MODULE ARE SYNTHETIC TEST DATA GENERATED SOLELY FOR
UNIT TESTING OF FRAME DECODING PIPELINES. THEY DO NOT REPRESENT A REAL DRONE FLIGHT.
"""

import pytest
import os
import cv2
import numpy as np
import tempfile

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
def synthetic_timeline() -> CanonicalTimeline:
    """Synthetic 10-frame timeline (TEST DATA)."""
    frames = [
        CanonicalFrame(
            frame_id=f"synth_flight_{i:06d}",
            frame_index=i,
            timestamp_seconds=i * 0.5,
            pts=int(i * 500),
            timescale=1000,
            source_video="synthetic_flight.mp4",
            width=640,
            height=480,
            is_keyframe=(i % 5 == 0),
        )
        for i in range(10)
    ]
    prov = VideoProvenance(
        source_file_path="synthetic_flight.mp4",
        file_size_bytes=1024,
        sha256_checksum="a" * 64,
        ingestion_timestamp_utc="2026-09-02T00:00:00Z",
        metadata_extractor="ISOBMFFParser_v1.1",
        timestamp_source="container_pts",
    )
    return CanonicalTimeline(
        video_id="synth_flight",
        source_path="synthetic_flight.mp4",
        total_frames=10,
        duration_seconds=4.5,
        nominal_fps=2.0,
        width=640,
        height=480,
        frames=frames,
        provenance=prov,
    )


@pytest.fixture
def real_synthetic_mp4(tmp_path) -> str:
    """Generate a real 10-frame synthetic MP4 file on disk for OpenCV decoding tests (TEST DATA)."""
    video_path = str(tmp_path / "synthetic_test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 2.0, (640, 480))

    for i in range(10):
        # Generate known BGR frame (e.g. Red bar at x = i*30 in RGB -> Blue in BGR)
        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_bgr[:, :, 0] = (i * 25) % 255 # Blue channel
        frame_bgr[:, :, 2] = 200            # Red channel
        out.write(frame_bgr)

    out.release()
    return video_path


# Test 1: Single-frame decode (random access via index)
def test_single_frame_decode(synthetic_timeline):
    decoder = SyntheticFrameDecoder(total_frames=10, width=640, height=480, timeline=synthetic_timeline)
    frame = decoder.decode_frame(3)

    assert frame.decode_status == DecodeStatus.SUCCESS
    assert frame.frame_index == 3
    assert frame.frame_id == "synth_flight_000003"
    assert frame.timestamp_seconds == 1.5
    assert frame.width == 640
    assert frame.height == 480
    assert frame.channels == 3
    assert frame.data is not None
    assert frame.data.shape == (480, 640, 3)


# Test 2: Sequential decode (streaming iteration)
def test_sequential_decode(synthetic_timeline):
    decoder = SyntheticFrameDecoder(total_frames=10, width=640, height=480, timeline=synthetic_timeline)
    frames = list(decoder.iter_frames(start_index=0, stop_index=5, step=1))

    assert len(frames) == 5
    for i, frame in enumerate(frames):
        assert frame.frame_index == i
        assert frame.decode_status == DecodeStatus.SUCCESS
        assert frame.timestamp_seconds == i * 0.5


# Test 3: Frame identity preservation
def test_frame_identity_preservation(synthetic_timeline):
    decoder = SyntheticFrameDecoder(total_frames=10, width=640, height=480, timeline=synthetic_timeline)
    for i in range(10):
        frame = decoder.decode_frame(i)
        assert frame.frame_id == synthetic_timeline[i].frame_id
        assert frame.frame_index == synthetic_timeline[i].frame_index


# Test 4: Timestamp preservation
def test_timestamp_preservation(synthetic_timeline):
    decoder = SyntheticFrameDecoder(total_frames=10, width=640, height=480, timeline=synthetic_timeline)
    for i in range(10):
        frame = decoder.decode_frame(i)
        assert frame.timestamp_seconds == synthetic_timeline[i].timestamp_seconds


# Test 5: RGB / BGR convention enforcement
def test_rgb_bgr_convention(real_synthetic_mp4, synthetic_timeline):
    decoder = OpenCVFrameDecoder(video_path=real_synthetic_mp4, timeline=synthetic_timeline)
    frame = decoder.decode_frame(0)

    assert frame.decode_status == DecodeStatus.SUCCESS
    assert frame.channel_layout == "RGB"
    assert frame.channels == 3
    # Check that data is in RGB format (Red channel at index 2 in BGR is converted to index 0 in RGB)
    assert frame.data[:, :, 0].mean() > 100 # Red channel in RGB
    decoder.close()


# Test 6: Dimensions and shape validation
def test_dimensions_validation(synthetic_timeline):
    decoder = SyntheticFrameDecoder(total_frames=10, width=640, height=480, timeline=synthetic_timeline)
    frame = decoder.decode_frame(0)

    assert frame.width == 640
    assert frame.height == 480
    assert frame.data.shape == (480, 640, 3)
    frame.validate() # Must pass validation without error


# Test 7: Dtype consistency
def test_dtype_consistency(synthetic_timeline):
    decoder = SyntheticFrameDecoder(total_frames=10, width=640, height=480, timeline=synthetic_timeline)
    frame = decoder.decode_frame(0)

    assert frame.dtype == "uint8"
    assert frame.data.dtype == np.uint8
    assert frame.data.min() >= 0
    assert frame.data.max() <= 255


# Test 8: Invalid frame index handling
def test_invalid_frame_index(synthetic_timeline):
    decoder = SyntheticFrameDecoder(total_frames=10, width=640, height=480, timeline=synthetic_timeline)
    frame_neg = decoder.decode_frame(-1)
    assert frame_neg.decode_status == DecodeStatus.INDEX_OUT_OF_BOUNDS
    assert frame_neg.data is None

    frame_overflow = decoder.decode_frame(999)
    assert frame_overflow.decode_status == DecodeStatus.INDEX_OUT_OF_BOUNDS
    assert frame_overflow.data is None


# Test 9: Missing video handling
def test_missing_video_handling(synthetic_timeline):
    decoder = OpenCVFrameDecoder(video_path="non_existent_file_path.mp4", timeline=synthetic_timeline)
    frame = decoder.decode_frame(0)

    assert frame.decode_status == DecodeStatus.FILE_NOT_FOUND
    assert frame.data is None
    decoder.close()


# Test 10: Corrupted video handling
def test_corrupted_video_handling(tmp_path, synthetic_timeline):
    corrupt_file = str(tmp_path / "corrupt.mp4")
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_A_VALID_MP4_HEADER_GARBAGE")

    decoder = OpenCVFrameDecoder(video_path=corrupt_file, timeline=synthetic_timeline)
    frame = decoder.decode_frame(0)

    assert frame.decode_status in {DecodeStatus.DECODER_ERROR, DecodeStatus.CORRUPTED}
    assert frame.data is None
    decoder.close()


# Test 11: Memory-safe sequential iteration
def test_memory_safe_sequential_iteration(real_synthetic_mp4, synthetic_timeline):
    decoder = OpenCVFrameDecoder(video_path=real_synthetic_mp4, timeline=synthetic_timeline)
    count = 0
    for frame in decoder.iter_frames(start_index=0, stop_index=10, step=2):
        assert frame.is_valid is True
        assert frame.frame_index == count * 2
        count += 1
    assert count == 5
    decoder.close()


# Test 12: Provenance preservation
def test_provenance_preservation(synthetic_timeline):
    decoder = SyntheticFrameDecoder(total_frames=10, width=640, height=480, timeline=synthetic_timeline)
    frame = decoder.decode_frame(4)

    assert frame.decoder_backend == "SyntheticFrameDecoder"
    assert frame.provenance is not None
    assert frame.provenance.get("is_synthetic") is True


# Test 13: Optional resize is explicit
def test_optional_resize_explicit(synthetic_timeline):
    config = DecodeConfig(target_width=320, target_height=240)
    decoder = SyntheticFrameDecoder(
        total_frames=10, width=640, height=480, timeline=synthetic_timeline, config=config
    )
    frame = decoder.decode_frame(0)

    assert frame.is_resized is True
    assert frame.width == 320
    assert frame.height == 240
    assert frame.original_width == 640
    assert frame.original_height == 480
    assert frame.data.shape == (240, 320, 3)


# Test 14: Deterministic repeated decode
def test_deterministic_repeated_decode(synthetic_timeline):
    decoder = SyntheticFrameDecoder(total_frames=10, width=640, height=480, timeline=synthetic_timeline)
    frame_a = decoder.decode_frame(2)
    frame_b = decoder.decode_frame(2)

    assert np.array_equal(frame_a.data, frame_b.data)
    assert frame_a.frame_id == frame_b.frame_id
    assert frame_a.timestamp_seconds == frame_b.timestamp_seconds


# Test 15: DatasetFrameDecoder High-Level Integration
def test_dataset_frame_decoder_integration(synthetic_timeline):
    telemetry_records = [
        TelemetryRecord(
            timestamp=float(i * 0.5),
            position=TelemetryPosition(latitude_deg=18.5, longitude_deg=73.8, altitude_meters=500.0),
        )
        for i in range(10)
    ]
    stream = CanonicalTelemetryStream(stream_id="test_stream", records=telemetry_records)
    vmeta = VideoMetadata(
        filepath="synthetic_flight.mp4", width=640, height=480, fps=2.0, total_frames=10, duration_seconds=4.5
    )
    dataset = CanonicalDatasetBuilder.build(vmeta, synthetic_timeline, stream)

    synth_decoder = SyntheticFrameDecoder(total_frames=10, width=640, height=480, timeline=synthetic_timeline)

    with DatasetFrameDecoder(dataset=dataset, decoder=synth_decoder) as ds_decoder:
        frame = ds_decoder.decode_frame(1)
        assert frame.is_valid is True
        assert frame.frame_id == dataset[1].frame_id
        assert frame.timestamp_seconds == dataset[1].video_timestamp_seconds

        frame_by_id = ds_decoder.decode_frame_by_id("synth_flight_000005")
        assert frame_by_id is not None
        assert frame_by_id.frame_index == 5
