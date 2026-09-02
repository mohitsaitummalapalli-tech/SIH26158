"""Integration tests for Phase 1B.5 Canonical Flight Dataset.

DISCLAIMER:
ALL FIXTURES IN THIS MODULE ARE SYNTHETIC TEST DATA GENERATED SOLELY FOR
INTEGRATION TESTING OF THE CANONICAL FLIGHT DATASET SUBSYSTEM. THEY DO NOT REPRESENT A REAL DRONE FLIGHT.
"""

import pytest
import json
import math
from dataclasses import asdict

from src.ingestion import (
    VideoMetadata,
    CanonicalFrame,
    CanonicalTimeline,
    VideoProvenance,
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
    SyncStatus,
    ClockOffsetStatus,
    SynchronizationConfig,
    DatasetStatus,
    DatasetValidationIssue,
    DatasetProvenance,
    CanonicalFrameObservation,
    CanonicalFlightDataset,
    CanonicalDatasetBuilder,
)
from src.geospatial import (
    OriginPolicy,
    GeodeticOrigin,
)


@pytest.fixture
def synthetic_video_metadata() -> VideoMetadata:
    return VideoMetadata(
        filepath="synthetic_flight.mp4",
        width=1920,
        height=1080,
        fps=2.0,
        total_frames=10,
        duration_seconds=4.5,
    )


@pytest.fixture
def synthetic_timeline(synthetic_video_metadata) -> CanonicalTimeline:
    """Synthetic 10-frame timeline at 2Hz (TEST DATA)."""
    frames = []
    for i in range(10):
        t = i * 0.5
        frames.append(
            CanonicalFrame(
                frame_id=f"frame_{i:04d}",
                frame_index=i,
                timestamp_seconds=t,
                pts=int(t * 1000),
                timescale=1000,
                source_video="synthetic_flight.mp4",
                width=1920,
                height=1080,
                is_keyframe=(i % 5 == 0),
            )
        )
    prov = VideoProvenance(
        source_file_path="synthetic_flight.mp4",
        file_size_bytes=2048,
        sha256_checksum="1" * 64,
        ingestion_timestamp_utc="2026-09-02T00:00:00Z",
        metadata_extractor="ISOBMFFParser_v1.1",
        timestamp_source="container_pts",
    )
    return CanonicalTimeline(
        video_id="synth_vid_001",
        source_path="synthetic_flight.mp4",
        total_frames=10,
        duration_seconds=4.5,
        nominal_fps=2.0,
        width=1920,
        height=1080,
        frames=frames,
        provenance=prov,
    )


@pytest.fixture
def synthetic_telemetry_stream() -> CanonicalTelemetryStream:
    """Synthetic telemetry stream covering t=0.0 to 5.0 at 1Hz (TEST DATA)."""
    records = []
    for i in range(6):
        t = float(i)
        pos = TelemetryPosition(
            latitude_deg=18.5200 + i * 0.0010,
            longitude_deg=73.8500 + i * 0.0010,
            altitude_meters=550.0 + i * 5.0,
            altitude_reference=AltitudeReference.ELLIPSOIDAL,
        )
        ori = TelemetryOrientation(
            heading_deg=(i * 45.0) % 360.0,
            pitch_deg=-2.0,
            roll_deg=0.5,
            gimbal_pitch_deg=-45.0,
            gimbal_roll_deg=0.0,
            gimbal_yaw_deg=0.0,
        )
        vel = TelemetryVelocity(speed_mps=6.0, climb_rate_mps=0.5)
        qual = TelemetryQuality(fix_type=FixType.RTK_FIXED, satellites_visible=20)
        prov = TelemetryProvenance(source_type="csv_adapter", source_identifier="synth_telemetry.csv", record_index=i)

        records.append(
            TelemetryRecord(
                timestamp=t,
                position=pos,
                orientation=ori,
                velocity=vel,
                quality=qual,
                provenance=prov,
            )
        )

    return CanonicalTelemetryStream(
        stream_id="synth_tel_001",
        records=records,
        provenance=TelemetryProvenance(
            source_type="csv_adapter",
            source_identifier="synth_telemetry.csv",
            source_checksum="2" * 64,
        ),
    )


# Test 1: Valid End-to-End Dataset Construction
def test_valid_end_to_end_dataset_construction(synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream):
    dataset = CanonicalDatasetBuilder.build(
        video_metadata=synthetic_video_metadata,
        timeline=synthetic_timeline,
        telemetry_stream=synthetic_telemetry_stream,
        origin_policy=OriginPolicy.FIRST_VALID_POSITION,
    )

    assert dataset.status == DatasetStatus.VALID
    assert dataset.total_frames == 10
    assert dataset.synchronized_count == 10
    assert dataset.origin is not None
    assert dataset.origin.latitude_deg == 18.5200


# Test 2: Video + Telemetry Association
def test_video_telemetry_association(synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream):
    dataset = CanonicalDatasetBuilder.build(
        synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream
    )

    obs0 = dataset[0]
    assert obs0.frame_id == "frame_0000"
    assert obs0.video_timestamp_seconds == 0.0
    assert obs0.sync_status == SyncStatus.EXACT
    assert obs0.original_position.latitude_deg == 18.5200


# Test 3: Deterministic Frame Lookup
def test_frame_lookup_methods(synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream):
    dataset = CanonicalDatasetBuilder.build(
        synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream
    )

    obs = dataset.get_frame_by_id("frame_0004")
    assert obs is not None
    assert obs.frame_index == 4
    assert obs.video_timestamp_seconds == 2.0

    non_existent = dataset.get_frame_by_id("non_existent_frame")
    assert non_existent is None


# Test 4: Synchronized Telemetry Retrieval
def test_synchronized_telemetry_retrieval(synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream):
    dataset = CanonicalDatasetBuilder.build(
        synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream
    )

    sync_frames = dataset.get_synchronized_frames()
    assert len(sync_frames) == 10
    for frame in sync_frames:
        assert frame.is_synchronized is True
        assert frame.original_position is not None


# Test 5: ECEF Coordinates Propagation
def test_ecef_propagation(synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream):
    dataset = CanonicalDatasetBuilder.build(
        synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream
    )

    for obs in dataset:
        assert obs.ecef_position is not None
        assert not math.isnan(obs.ecef_position.x_meters)
        assert not math.isnan(obs.ecef_position.y_meters)
        assert not math.isnan(obs.ecef_position.z_meters)


# Test 6: ENU Coordinates Propagation Relative to Origin
def test_enu_propagation(synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream):
    dataset = CanonicalDatasetBuilder.build(
        synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream
    )

    obs0 = dataset[0]
    assert obs0.enu_position is not None
    assert obs0.enu_position.east_meters == pytest.approx(0.0, abs=1e-4)
    assert obs0.enu_position.north_meters == pytest.approx(0.0, abs=1e-4)
    assert obs0.enu_position.up_meters == pytest.approx(0.0, abs=1e-4)

    obs1 = dataset[1] # t=0.5
    assert obs1.enu_position is not None
    assert obs1.enu_position.east_meters > 0.0 # increasing longitude
    assert obs1.enu_position.north_meters > 0.0 # increasing latitude


# Test 7: Provenance Propagation
def test_provenance_propagation(synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream):
    dataset = CanonicalDatasetBuilder.build(
        synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream
    )

    prov = dataset.provenance
    assert prov.video_source_path == "synthetic_flight.mp4"
    assert prov.video_checksum_sha256 == "1" * 64
    assert prov.telemetry_source_path == "synth_telemetry.csv"
    assert prov.telemetry_checksum_sha256 == "2" * 64
    assert prov.software_version.startswith("SIH26158")


# Test 8: Missing Optional IMU Handling
def test_missing_optional_imu(synthetic_video_metadata, synthetic_timeline):
    records = [
        TelemetryRecord(
            timestamp=float(i),
            position=TelemetryPosition(latitude_deg=18.52 + i * 0.001, longitude_deg=73.85 + i * 0.001, altitude_meters=500.0, altitude_reference=AltitudeReference.ELLIPSOIDAL),
            orientation=None, # Explicitly missing IMU
            velocity=None,
        )
        for i in range(6)
    ]
    stream = CanonicalTelemetryStream(stream_id="no_imu_stream", records=records)
    dataset = CanonicalDatasetBuilder.build(synthetic_video_metadata, synthetic_timeline, stream)

    # Must be valid, not crashed
    assert dataset.status in {DatasetStatus.VALID, DatasetStatus.PARTIALLY_VALID}
    issue_codes = [issue.code for issue in dataset.validation_issues]
    assert "MISSING_OPTIONAL_IMU" in issue_codes
    for obs in dataset:
        assert obs.orientation is None # No fake zero orientations fabricated!


# Test 9: Unsynchronized Time Ranges Flagged
def test_unsynchronized_time_ranges_flagged(synthetic_video_metadata, synthetic_timeline):
    # Telemetry only covers t=0.0 to 1.0; frames go up to t=4.5
    records = [
        TelemetryRecord(
            timestamp=float(i),
            position=TelemetryPosition(latitude_deg=18.52 + i * 0.001, longitude_deg=73.85 + i * 0.001, altitude_meters=500.0, altitude_reference=AltitudeReference.ELLIPSOIDAL),
        )
        for i in range(2)
    ]
    stream = CanonicalTelemetryStream(stream_id="short_stream", records=records)
    dataset = CanonicalDatasetBuilder.build(synthetic_video_metadata, synthetic_timeline, stream)

    assert dataset.status == DatasetStatus.PARTIALLY_VALID
    issue_codes = [issue.code for issue in dataset.validation_issues]
    assert "UNSYNCHRONIZED_FRAMES" in issue_codes

    # Frames beyond t=1.0 are out of range
    assert dataset[9].is_synchronized is False
    assert dataset[9].sync_status == SyncStatus.OUT_OF_RANGE


# Test 10: Incompatible / Non-Ellipsoidal Altitude Warning
def test_incompatible_altitude_reference_warning(synthetic_video_metadata, synthetic_timeline):
    records = [
        TelemetryRecord(
            timestamp=float(i),
            position=TelemetryPosition(latitude_deg=18.52, longitude_deg=73.85, altitude_meters=500.0, altitude_reference=AltitudeReference.MSL),
        )
        for i in range(6)
    ]
    stream = CanonicalTelemetryStream(stream_id="msl_stream", records=records)
    dataset = CanonicalDatasetBuilder.build(synthetic_video_metadata, synthetic_timeline, stream)

    issue_codes = [issue.code for issue in dataset.validation_issues]
    assert "NON_ELLIPSOIDAL_ALTITUDE" in issue_codes


# Test 11: Invalid Source Data Rejection
def test_invalid_source_data_handling(synthetic_video_metadata, synthetic_timeline):
    empty_stream = CanonicalTelemetryStream(stream_id="empty_tel", records=[])

    dataset = CanonicalDatasetBuilder.build(synthetic_video_metadata, synthetic_timeline, empty_stream)
    assert dataset.status == DatasetStatus.INVALID
    assert dataset.synchronized_count == 0


# Test 12: Dataset Status Transitions
def test_dataset_status_transitions(synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream):
    # Full coverage -> VALID
    ds_valid = CanonicalDatasetBuilder.build(synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream)
    assert ds_valid.status == DatasetStatus.VALID


# Test 13: Serialization Round-Trip
def test_serialization_round_trip(synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream):
    dataset = CanonicalDatasetBuilder.build(synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream)

    d = dataset.to_dict()
    assert d["dataset_id"] == dataset.dataset_id
    assert d["status"] == "VALID"
    assert len(d["frame_observations"]) == 10
    assert "video_metadata" in d
    assert "origin" in d

    json_str = dataset.to_json(indent=2)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["dataset_id"] == dataset.dataset_id


# Test 14: Deterministic Reconstruction of Dataset Metadata
def test_deterministic_reconstruction_metadata(synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream):
    dataset = CanonicalDatasetBuilder.build(synthetic_video_metadata, synthetic_timeline, synthetic_telemetry_stream)

    json_str = dataset.to_json()
    parsed = json.loads(json_str)

    assert parsed["provenance"]["software_version"].startswith("SIH26158")
    assert parsed["frame_observations"][0]["frame_id"] == "frame_0000"
    assert parsed["frame_observations"][0]["enu_position"]["east_meters"] == pytest.approx(0.0, abs=1e-4)
