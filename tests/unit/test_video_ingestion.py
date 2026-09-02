"""Deterministic unit tests for Phase 1A & Phase 1A.1 Video Ingestion & Compatibility."""

import os
import pytest
import tempfile
from tests.helpers.synthetic_video import create_synthetic_mp4, create_synthetic_fragmented_mp4
from src.ingestion import (
    VideoSource,
    ISOBMFFVideoSource,
    CanonicalTimeline,
    CanonicalFrame,
    VideoProvenance,
    VideoNotFoundError,
    UnsupportedVideoFormatError,
    UnsupportedFragmentedMP4Error,
    CorruptVideoError,
    InvalidVideoMetadataError,
)


@pytest.fixture
def synthetic_mp4_path(tmp_path):
    """Fixture producing a valid synthetic MP4 test video binary (TEST DATA)."""
    video_path = str(tmp_path / "synthetic_test_flight.mp4")
    create_synthetic_mp4(
        filepath=video_path,
        width=1920,
        height=1080,
        fps=30,
        num_frames=60,
        timescale=1000,
        keyframe_indices=[0, 30],
        codec_fourcc=b"avc1"
    )
    return video_path


# Test 1: Valid ordinary MP4 (H.264)
def test_valid_synthetic_video_ingestion(synthetic_mp4_path):
    source = VideoSource.from_file(synthetic_mp4_path)
    assert isinstance(source, ISOBMFFVideoSource)

    timeline = source.build_canonical_timeline()
    assert isinstance(timeline, CanonicalTimeline)

    assert timeline.width == 1920
    assert timeline.height == 1080
    assert timeline.total_frames == 60
    assert len(timeline) == 60
    assert pytest.approx(timeline.duration_seconds, 0.05) == 1.98
    assert pytest.approx(timeline.nominal_fps, 0.5) == 30.0
    assert timeline.source_path == os.path.abspath(synthetic_mp4_path)


# Test 2: QuickTime MOV file (.mov)
def test_mov_container_ingestion(tmp_path):
    mov_path = str(tmp_path / "cinematic_drone.mov")
    create_synthetic_mp4(
        filepath=mov_path,
        width=3840,
        height=2160,
        fps=24,
        num_frames=48,
        timescale=24000,
        codec_fourcc=b"apcn",  # ProRes 422
        is_mov=True
    )
    source = VideoSource.from_file(mov_path)
    timeline = source.build_canonical_timeline()

    assert timeline.width == 3840
    assert timeline.height == 2160
    assert timeline.total_frames == 48
    assert timeline.frames[0].extra_metadata["codec"] == "apcn"


# Test 3: H.265 / HEVC codec in MP4 (hvc1)
def test_hevc_codec_ingestion(tmp_path):
    hevc_path = str(tmp_path / "drone_4k_hevc.mp4")
    create_synthetic_mp4(
        filepath=hevc_path,
        width=3840,
        height=2160,
        fps=60,
        num_frames=120,
        timescale=60000,
        codec_fourcc=b"hvc1"
    )
    source = VideoSource.from_file(hevc_path)
    timeline = source.build_canonical_timeline()

    assert timeline.width == 3840
    assert timeline.frames[0].extra_metadata["codec"] == "hvc1"


# Test 4: B-frames with nontrivial composition offsets (ctts)
def test_bframes_composition_offsets(tmp_path):
    bframe_path = str(tmp_path / "bframes_video.mp4")
    # Decoding order: DTS=[0, 33, 66, 99], CTS_offset=[66, 0, 33, 0] -> PTS=[66, 33, 99, 99]
    # Presentation order after sorting by PTS -> Sample 1 (PTS=33), Sample 0 (PTS=66), Sample 2 (PTS=99), Sample 3 (PTS=99)
    ctts_offsets = [66, 0, 33, 0]
    create_synthetic_mp4(
        filepath=bframe_path,
        width=1280,
        height=720,
        fps=30,
        num_frames=4,
        timescale=1000,
        ctts_offsets=ctts_offsets
    )
    source = VideoSource.from_file(bframe_path)
    timeline = source.build_canonical_timeline()

    assert timeline.total_frames == 4
    assert timeline.frames[0].extra_metadata["has_b_frames"] is True
    # Verify presentation order: first frame in timeline is sample 1 with PTS=33
    assert timeline.frames[0].pts == 33
    assert timeline.frames[0].timestamp_seconds == 0.033
    assert timeline.frames[0].extra_metadata["sample_index"] == 1

    # Second frame in timeline is sample 0 with PTS=66
    assert timeline.frames[1].pts == 66
    assert timeline.frames[1].timestamp_seconds == 0.066
    assert timeline.frames[1].extra_metadata["sample_index"] == 0


# Test 5: Variable Frame Rate (VFR)
def test_variable_frame_rate_ingestion(tmp_path):
    vfr_path = str(tmp_path / "vfr_stream.mp4")
    # 10 frames with delta 33ms (~30fps), 10 frames with delta 16ms (~60fps)
    vfr_deltas = [(10, 33), (10, 16)]
    create_synthetic_mp4(
        filepath=vfr_path,
        width=1920,
        height=1080,
        timescale=1000,
        vfr_deltas=vfr_deltas
    )
    source = VideoSource.from_file(vfr_path)
    timeline = source.build_canonical_timeline()

    assert timeline.total_frames == 20
    assert timeline.frames[0].extra_metadata["is_vfr"] is True
    # Verify timestamps reflect variable delta
    assert timeline.frames[10].timestamp_seconds == pytest.approx(0.33, 0.01)  # 10 * 33ms
    assert timeline.frames[11].timestamp_seconds == pytest.approx(0.346, 0.01) # 330ms + 16ms


# Test 6: Edit list delay (elst start offset)
def test_edit_list_start_delay(tmp_path):
    elst_path = str(tmp_path / "elst_delayed.mp4")
    create_synthetic_mp4(
        filepath=elst_path,
        width=1920,
        height=1080,
        fps=30,
        num_frames=30,
        timescale=1000,
        edit_list_delay_ms=500  # 500ms initial dwell
    )
    source = VideoSource.from_file(elst_path)
    timeline = source.build_canonical_timeline()

    # Frame 0 timestamp should start at 0.5s due to 500ms edit list delay
    assert timeline.frames[0].timestamp_seconds == 0.500
    assert timeline.duration_seconds == pytest.approx(1.49, 0.05)  # 500ms + 30*33ms


# Test 7: Fragmented MP4 detection (moof/mvex)
def test_fragmented_mp4_rejection(tmp_path):
    fmp4_path = str(tmp_path / "fragmented_stream.mp4")
    create_synthetic_fragmented_mp4(fmp4_path)

    source = VideoSource.from_file(fmp4_path)
    with pytest.raises(UnsupportedFragmentedMP4Error, match="Fragmented MP4 detected"):
        source.build_canonical_timeline()


# Test 8: 64-bit largesize box handling
def test_64bit_largesize_box_handling(tmp_path):
    large_path = str(tmp_path / "large_box.mp4")
    create_synthetic_mp4(
        filepath=large_path,
        width=1920,
        height=1080,
        fps=30,
        num_frames=10,
        use_64bit_box=True
    )
    source = VideoSource.from_file(large_path)
    timeline = source.build_canonical_timeline()
    assert timeline.total_frames == 10


# Test 9: Multi-track video container (selects highest resolution)
def test_multi_track_primary_selection(tmp_path):
    multi_path = str(tmp_path / "multi_track.mp4")
    create_synthetic_mp4(
        filepath=multi_path,
        width=1920,
        height=1080,
        fps=30,
        num_frames=15,
        include_secondary_video_track=True  # Has 1920x1080 and 640x360 tracks
    )
    source = VideoSource.from_file(multi_path)
    timeline = source.build_canonical_timeline()

    # Must select 1920x1080 as primary
    assert timeline.width == 1920
    assert timeline.height == 1080


# Test 10: Unknown extension box skipping
def test_unknown_box_safe_skipping(tmp_path):
    unknown_path = str(tmp_path / "unknown_box.mp4")
    create_synthetic_mp4(
        filepath=unknown_path,
        width=1920,
        height=1080,
        fps=30,
        num_frames=10,
        include_unknown_box=True
    )
    source = VideoSource.from_file(unknown_path)
    timeline = source.build_canonical_timeline()
    assert timeline.total_frames == 10


# Test 11: Nonexistent file rejection
def test_nonexistent_file_rejection(tmp_path):
    missing_path = str(tmp_path / "nonexistent_flight.mp4")
    with pytest.raises(VideoNotFoundError, match="Video file not found"):
        VideoSource.from_file(missing_path)


# Test 12: Unsupported file extension rejection
def test_unsupported_extension_rejection(tmp_path):
    avi_path = str(tmp_path / "sample_flight.avi")
    with open(avi_path, "wb") as f:
        f.write(b"RIFF....AVI ")

    with pytest.raises(UnsupportedVideoFormatError, match="Unsupported video file extension"):
        VideoSource.from_file(avi_path)


# Test 13: Corrupt/empty video rejection
def test_empty_video_rejection(tmp_path):
    empty_mp4 = str(tmp_path / "empty.mp4")
    with open(empty_mp4, "wb") as f:
        pass  # 0 bytes

    source = VideoSource.from_file(empty_mp4)
    with pytest.raises(CorruptVideoError, match="is empty"):
        source.build_canonical_timeline()


def test_corrupt_header_rejection(tmp_path):
    corrupt_mp4 = str(tmp_path / "corrupt.mp4")
    with open(corrupt_mp4, "wb") as f:
        f.write(b"\x00\x00\x00\x20ftypmp42\x00\x00\x00\x00corrupt garbage bytes")

    source = VideoSource.from_file(corrupt_mp4)
    with pytest.raises(CorruptVideoError):
        source.build_canonical_timeline()


# Test 14: Timestamp ordering & monotonicity
def test_timestamp_monotonicity(synthetic_mp4_path):
    source = VideoSource.from_file(synthetic_mp4_path)
    timeline = source.build_canonical_timeline()

    prev_pts = -1
    prev_ts = -1.0
    for frame in timeline:
        assert frame.pts >= prev_pts
        assert frame.timestamp_seconds >= prev_ts
        assert frame.timestamp_seconds == pytest.approx(float(frame.pts) / float(frame.timescale))
        prev_pts = frame.pts
        prev_ts = frame.timestamp_seconds


# Test 15: Provenance preservation
def test_provenance_preservation(synthetic_mp4_path):
    source = VideoSource.from_file(synthetic_mp4_path)
    timeline = source.build_canonical_timeline()

    provenance = timeline.provenance
    assert isinstance(provenance, VideoProvenance)
    assert provenance.source_file_path == os.path.abspath(synthetic_mp4_path)
    assert provenance.file_size_bytes > 0
    assert len(provenance.sha256_checksum) == 64
    assert provenance.metadata_extractor == "ISOBMFFParser_v1.1"
    assert provenance.timestamp_source == "container_pts"
    assert provenance.ingestion_timestamp_utc is not None
