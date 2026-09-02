"""Deterministic unit tests for Phase 1B.3.1 Temporal Synchronization Scientific Corrections.

DISCLAIMER:
ALL FIXTURES IN THIS MODULE ARE SYNTHETIC TEST DATA GENERATED SOLELY FOR
UNIT TESTING TEMPORAL SYNCHRONIZATION ALGORITHMS. THEY DO NOT REPRESENT A REAL DRONE FLIGHT.
"""

import pytest
import math
from typing import List

from src.ingestion import (
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
    IdentityClockModel,
    ConstantOffsetClockModel,
    SynchronizedFrameObservation,
    SynchronizedTrajectory,
    TemporalSynchronizationEngine,
    geodetic_to_ecef,
    ecef_to_geodetic,
)


@pytest.fixture
def dummy_video_provenance() -> VideoProvenance:
    return VideoProvenance(
        source_file_path="synthetic_video.mp4",
        file_size_bytes=1024,
        sha256_checksum="a" * 64,
        ingestion_timestamp_utc="2026-09-02T00:00:00Z",
        metadata_extractor="ISOBMFFParser_v1.1",
        timestamp_source="container_pts"
    )


@pytest.fixture
def synthetic_timeline(dummy_video_provenance) -> CanonicalTimeline:
    """Synthetic 10-frame timeline at 2Hz (t=0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5) (TEST DATA)."""
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
                source_video="synthetic_video.mp4",
                width=1920,
                height=1080,
                is_keyframe=(i % 5 == 0)
            )
        )
    return CanonicalTimeline(
        video_id="synthetic_flight",
        source_path="synthetic_video.mp4",
        total_frames=10,
        duration_seconds=4.5,
        nominal_fps=2.0,
        width=1920,
        height=1080,
        frames=frames,
        provenance=dummy_video_provenance
    )


@pytest.fixture
def synthetic_telemetry_stream() -> CanonicalTelemetryStream:
    """Synthetic 1Hz telemetry stream covering t=0.0, 1.0, 2.0, 3.0, 4.0 (TEST DATA)."""
    records = []
    for i in range(5):
        t = float(i)
        pos = TelemetryPosition(
            latitude_deg=18.5000 + i * 0.0010,
            longitude_deg=73.8000 + i * 0.0010,
            altitude_meters=500.0 + i * 10.0,
            altitude_reference=AltitudeReference.MSL
        )
        ori = TelemetryOrientation(
            heading_deg=(i * 90.0) % 360.0,
            pitch_deg=-5.0,
            roll_deg=0.0,
            gimbal_pitch_deg=-45.0,
            gimbal_roll_deg=0.0,
            gimbal_yaw_deg=0.0
        )
        vel = TelemetryVelocity(speed_mps=5.0 + i * 1.0, climb_rate_mps=1.0)
        qual = TelemetryQuality(fix_type=FixType.RTK_FIXED, satellites_visible=18)
        prov = TelemetryProvenance(source_type="test_fixture", source_identifier="synth_log.csv", record_index=i)

        rec = TelemetryRecord(
            timestamp=t,
            position=pos,
            orientation=ori,
            velocity=vel,
            quality=qual,
            provenance=prov
        )
        records.append(rec)

    return CanonicalTelemetryStream(
        stream_id="synth_telemetry",
        records=records,
        provenance=TelemetryProvenance(source_type="test_stream", source_identifier="synth_log.csv")
    )


# Test 1: Exact timestamp match
def test_exact_timestamp_synchronization(synthetic_timeline, synthetic_telemetry_stream):
    config = SynchronizationConfig()
    trajectory = TemporalSynchronizationEngine.synchronize(
        synthetic_timeline, synthetic_telemetry_stream, config
    )

    obs0 = trajectory[0]
    assert obs0.status == SyncStatus.EXACT
    assert obs0.video_timestamp_seconds == 0.0
    assert obs0.telemetry_timestamp_seconds == 0.0
    assert obs0.position.latitude_deg == 18.5000
    assert obs0.position.altitude_meters == 500.0
    assert obs0.source_record_indices == [0]
    assert obs0.interpolation_fraction == 0.0
    assert obs0.bracketing_interval_seconds == 0.0


# Test 2: Linear position interpolation in ECEF space
def test_linear_position_interpolation(synthetic_timeline, synthetic_telemetry_stream):
    config = SynchronizationConfig()
    trajectory = TemporalSynchronizationEngine.synchronize(
        synthetic_timeline, synthetic_telemetry_stream, config
    )

    obs1 = trajectory[1]
    assert obs1.status == SyncStatus.INTERPOLATED
    assert obs1.video_timestamp_seconds == 0.5
    assert obs1.telemetry_timestamp_seconds == 0.5
    assert obs1.position.latitude_deg == pytest.approx(18.5005, 1e-4)
    assert obs1.position.longitude_deg == pytest.approx(73.8005, 1e-4)
    assert obs1.position.altitude_meters == pytest.approx(505.0, 1e-2)
    assert obs1.position.altitude_reference == AltitudeReference.MSL
    assert obs1.source_record_indices == [0, 1]
    assert obs1.interpolation_fraction == pytest.approx(0.5, 1e-4)
    assert obs1.bracketing_interval_seconds == pytest.approx(1.0, 1e-4)


# Test 3: Longitude wraparound across Antimeridian (+179.9 to -179.9)
def test_longitude_antimeridian_wraparound(dummy_video_provenance):
    frame = CanonicalFrame(
        frame_id="f0", frame_index=0, timestamp_seconds=0.5, pts=500, timescale=1000,
        source_video="test.mp4", width=1920, height=1080
    )
    timeline = CanonicalTimeline(
        video_id="wrap_test", source_path="test.mp4", total_frames=1, duration_seconds=1.0,
        nominal_fps=1.0, width=1920, height=1080, frames=[frame], provenance=dummy_video_provenance
    )

    # Telemetry: t=0.0 at lon=+179.9, t=1.0 at lon=-179.9 (total distance is 0.2 deg across 180 deg)
    p0 = TelemetryPosition(latitude_deg=0.0, longitude_deg=179.9, altitude_meters=100.0, altitude_reference=AltitudeReference.ELLIPSOIDAL)
    p1 = TelemetryPosition(latitude_deg=0.0, longitude_deg=-179.9, altitude_meters=100.0, altitude_reference=AltitudeReference.ELLIPSOIDAL)
    r0 = TelemetryRecord(timestamp=0.0, position=p0)
    r1 = TelemetryRecord(timestamp=1.0, position=p1)
    stream = CanonicalTelemetryStream(stream_id="wrap_stream", records=[r0, r1])

    # ECEF interpolation
    config = SynchronizationConfig(use_ecef_interpolation=True)
    trajectory = TemporalSynchronizationEngine.synchronize(timeline, stream, config)

    obs = trajectory[0]
    assert obs.status == SyncStatus.INTERPOLATED
    # At t=0.5, lon must be 180.0 (or -180.0), NOT 0.0 (Prime Meridian)!
    assert abs(abs(obs.position.longitude_deg) - 180.0) < 1e-3
    assert obs.position.latitude_deg == pytest.approx(0.0, 1e-4)


# Test 4: Incompatible altitude references rejected
def test_incompatible_altitude_references_rejected(dummy_video_provenance):
    frame = CanonicalFrame(
        frame_id="f0", frame_index=0, timestamp_seconds=0.5, pts=500, timescale=1000,
        source_video="test.mp4", width=1920, height=1080
    )
    timeline = CanonicalTimeline(
        video_id="incompat_test", source_path="test.mp4", total_frames=1, duration_seconds=1.0,
        nominal_fps=1.0, width=1920, height=1080, frames=[frame], provenance=dummy_video_provenance
    )

    p0 = TelemetryPosition(latitude_deg=18.0, longitude_deg=73.0, altitude_meters=500.0, altitude_reference=AltitudeReference.ELLIPSOIDAL)
    p1 = TelemetryPosition(latitude_deg=18.0, longitude_deg=73.0, altitude_meters=450.0, altitude_reference=AltitudeReference.MSL)
    r0 = TelemetryRecord(timestamp=0.0, position=p0)
    r1 = TelemetryRecord(timestamp=1.0, position=p1)
    stream = CanonicalTelemetryStream(stream_id="incompat_stream", records=[r0, r1])

    trajectory = TemporalSynchronizationEngine.synchronize(timeline, stream)
    obs = trajectory[0]
    assert obs.status == SyncStatus.INCOMPATIBLE_REFERENCE
    assert obs.position is None
    assert obs.is_synchronized is False


# Test 5: Compatible altitude references interpolated
def test_compatible_altitude_references():
    p0 = TelemetryPosition(latitude_deg=18.0, longitude_deg=73.0, altitude_meters=500.0, altitude_reference=AltitudeReference.MSL)
    p1 = TelemetryPosition(latitude_deg=18.0, longitude_deg=73.0, altitude_meters=600.0, altitude_reference=AltitudeReference.MSL)
    assert TemporalSynchronizationEngine.are_altitude_references_compatible(p0.altitude_reference, p1.altitude_reference) is True


# Test 6: Unknown altitude reference preserved
def test_unknown_altitude_reference_preserved(dummy_video_provenance):
    frame = CanonicalFrame(
        frame_id="f0", frame_index=0, timestamp_seconds=0.5, pts=500, timescale=1000,
        source_video="test.mp4", width=1920, height=1080
    )
    timeline = CanonicalTimeline(
        video_id="unk_test", source_path="test.mp4", total_frames=1, duration_seconds=1.0,
        nominal_fps=1.0, width=1920, height=1080, frames=[frame], provenance=dummy_video_provenance
    )

    p0 = TelemetryPosition(latitude_deg=18.0, longitude_deg=73.0, altitude_meters=500.0, altitude_reference=AltitudeReference.UNKNOWN)
    p1 = TelemetryPosition(latitude_deg=18.0, longitude_deg=73.0, altitude_meters=600.0, altitude_reference=AltitudeReference.UNKNOWN)
    r0 = TelemetryRecord(timestamp=0.0, position=p0)
    r1 = TelemetryRecord(timestamp=1.0, position=p1)
    stream = CanonicalTelemetryStream(stream_id="unk_stream", records=[r0, r1])

    trajectory = TemporalSynchronizationEngine.synchronize(timeline, stream)
    obs = trajectory[0]
    assert obs.status == SyncStatus.INTERPOLATED
    assert obs.position.altitude_reference == AltitudeReference.UNKNOWN


# Test 7: Bracketing interval separate from timebase uncertainty
def test_bracketing_interval_separated(synthetic_timeline, synthetic_telemetry_stream):
    config = SynchronizationConfig()
    trajectory = TemporalSynchronizationEngine.synchronize(
        synthetic_timeline, synthetic_telemetry_stream, config
    )

    obs1 = trajectory[1]
    assert obs1.bracketing_interval_seconds == 1.0
    assert obs1.timebase_uncertainty_seconds is None
    # Deprecated compatibility alias works
    assert obs1.temporal_uncertainty_seconds == 0.5


# Test 8: Configured offset marked as applied, not validated
def test_configured_offset_marked_applied(synthetic_timeline, synthetic_telemetry_stream):
    config = SynchronizationConfig(
        clock_model=ConstantOffsetClockModel(offset_seconds=1.0, status=ClockOffsetStatus.KNOWN_APPLIED)
    )
    trajectory = TemporalSynchronizationEngine.synchronize(
        synthetic_timeline, synthetic_telemetry_stream, config
    )

    obs0 = trajectory[0]
    assert obs0.status == SyncStatus.OFFSET_APPLIED
    assert obs0.clock_offset_status == ClockOffsetStatus.KNOWN_APPLIED
    assert obs0.clock_offset_seconds == 1.0


# Test 9: Rotation SLERP interpolation
def test_rotation_slerp_interpolation(synthetic_timeline, synthetic_telemetry_stream):
    config = SynchronizationConfig(interpolate_orientation=True)
    trajectory = TemporalSynchronizationEngine.synchronize(
        synthetic_timeline, synthetic_telemetry_stream, config
    )

    obs1 = trajectory[1]
    assert obs1.orientation is not None
    assert obs1.orientation.heading_deg == pytest.approx(45.0, 1e-2)
    assert obs1.orientation.pitch_deg == pytest.approx(-5.0, 1e-2)
    assert obs1.orientation.gimbal_pitch_deg == pytest.approx(-45.0, 1e-2)


# Test 10: Telemetry gap exceeding max_interpolation_gap -> UNSYNCHRONIZED
def test_telemetry_gap_rejection(synthetic_timeline):
    pos0 = TelemetryPosition(latitude_deg=10.0, longitude_deg=20.0, altitude_meters=100.0)
    pos1 = TelemetryPosition(latitude_deg=10.1, longitude_deg=20.1, altitude_meters=110.0)
    r0 = TelemetryRecord(timestamp=0.0, position=pos0)
    r1 = TelemetryRecord(timestamp=5.0, position=pos1)
    stream = CanonicalTelemetryStream(stream_id="gap_test", records=[r0, r1])

    config = SynchronizationConfig(max_interpolation_gap_seconds=2.0)
    trajectory = TemporalSynchronizationEngine.synchronize(synthetic_timeline, stream, config)

    obs1 = trajectory[1]
    assert obs1.status == SyncStatus.UNSYNCHRONIZED
    assert obs1.position is None
    assert obs1.is_synchronized is False
    assert obs1.bracketing_interval_seconds == 5.0


# Test 11: Out-of-range rejection & extrapolation
def test_out_of_range_rejection(synthetic_timeline, synthetic_telemetry_stream):
    config = SynchronizationConfig(allow_extrapolation=False)
    trajectory = TemporalSynchronizationEngine.synchronize(
        synthetic_timeline, synthetic_telemetry_stream, config
    )

    obs_last = trajectory[9]  # t=4.5
    assert obs_last.video_timestamp_seconds == 4.5
    assert obs_last.status == SyncStatus.OUT_OF_RANGE
    assert obs_last.position is None
    assert obs_last.is_synchronized is False


def test_explicit_extrapolation_enabled(synthetic_timeline, synthetic_telemetry_stream):
    config = SynchronizationConfig(allow_extrapolation=True)
    trajectory = TemporalSynchronizationEngine.synchronize(
        synthetic_timeline, synthetic_telemetry_stream, config
    )

    obs_last = trajectory[9]  # t=4.5
    assert obs_last.status == SyncStatus.EXTRAPOLATED
    assert obs_last.position is not None
    assert obs_last.position.latitude_deg == pytest.approx(18.5040, 1e-4)


# Test 12: Duplicate timestamps handled deterministically
def test_duplicate_telemetry_timestamps_handled(synthetic_timeline):
    pos0 = TelemetryPosition(latitude_deg=10.0, longitude_deg=20.0, altitude_meters=100.0)
    pos1_a = TelemetryPosition(latitude_deg=10.1, longitude_deg=20.1, altitude_meters=110.0)
    pos1_b = TelemetryPosition(latitude_deg=10.1, longitude_deg=20.1, altitude_meters=110.0)
    pos2 = TelemetryPosition(latitude_deg=10.2, longitude_deg=20.2, altitude_meters=120.0)

    r0 = TelemetryRecord(timestamp=0.0, position=pos0)
    r1_a = TelemetryRecord(timestamp=1.0, position=pos1_a)
    r1_b = TelemetryRecord(timestamp=1.0, position=pos1_b)
    r2 = TelemetryRecord(timestamp=2.0, position=pos2)

    stream = CanonicalTelemetryStream(stream_id="dup_stream", records=[r0, r1_a, r1_b, r2])
    config = SynchronizationConfig()
    trajectory = TemporalSynchronizationEngine.synchronize(synthetic_timeline, stream, config)

    obs2 = trajectory[2]
    assert obs2.status == SyncStatus.EXACT
    assert obs2.position.latitude_deg == 10.1


# Test 13: Numerical validation (finite values, no NaN/Inf)
def test_numerical_validation_finite():
    with pytest.raises(ValueError, match="cannot be NaN or Infinite"):
        ConstantOffsetClockModel(offset_seconds=float("nan"))

    with pytest.raises(ValueError, match="cannot be NaN or Infinite"):
        ConstantOffsetClockModel(offset_seconds=float("inf"))

    x, y, z = geodetic_to_ecef(0.0, 0.0, 0.0)
    assert not math.isnan(x) and not math.isinf(x)
    lat, lon, alt = ecef_to_geodetic(x, y, z)
    assert pytest.approx(lat, 1e-6) == 0.0
    assert pytest.approx(lon, 1e-6) == 0.0
    assert pytest.approx(alt, 1e-3) == 0.0
