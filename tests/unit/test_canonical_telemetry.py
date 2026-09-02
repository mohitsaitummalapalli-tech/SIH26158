"""Unit tests for Phase 1B.1 Canonical Telemetry Contracts and Validation."""

import math
import pytest
from src.ingestion import (
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
    InvalidTelemetryDataError,
)


# Test 1: Valid telemetry record creation
def test_valid_telemetry_record():
    pos = TelemetryPosition(
        latitude_deg=18.5204,
        longitude_deg=73.8567,
        altitude_meters=560.5,
        altitude_reference=AltitudeReference.ELLIPSOIDAL,
        position_reference=PositionReference.WGS84_GEODETIC
    )
    ori = TelemetryOrientation(
        heading_deg=145.2,
        pitch_deg=-5.0,
        roll_deg=1.2,
        gimbal_pitch_deg=-45.0,
        gimbal_roll_deg=0.0,
        gimbal_yaw_deg=145.2
    )
    vel = TelemetryVelocity(
        speed_mps=8.5,
        north_velocity_mps=6.0,
        east_velocity_mps=6.0,
        down_velocity_mps=0.2,
        climb_rate_mps=-0.2
    )
    qual = TelemetryQuality(
        fix_type=FixType.RTK_FIXED,
        satellites_visible=18,
        hdop=0.8,
        horizontal_accuracy_meters=0.02,
        vertical_accuracy_meters=0.04
    )
    prov = TelemetryProvenance(
        source_type="dji_srt",
        source_identifier="flight_001.SRT",
        record_index=0
    )

    record = TelemetryRecord(
        timestamp=10.5,
        position=pos,
        timestamp_semantics=TimestampSemantics.VIDEO_RELATIVE,
        orientation=ori,
        velocity=vel,
        quality=qual,
        provenance=prov
    )

    assert record.timestamp == 10.5
    assert record.latitude_deg == 18.5204
    assert record.longitude_deg == 73.8567
    assert record.altitude_meters == 560.5
    assert record.position.altitude_reference == AltitudeReference.ELLIPSOIDAL
    assert record.position.position_reference == PositionReference.WGS84_GEODETIC
    assert record.orientation.heading_deg == 145.2
    assert record.velocity.speed_mps == 8.5
    assert record.quality.fix_type == FixType.RTK_FIXED
    assert record.is_rtk_fixed is True


# Test 2: Invalid latitude (out of bounds >90 or <-90)
def test_invalid_latitude_bounds():
    with pytest.raises(InvalidTelemetryDataError, match="latitude out of valid range"):
        TelemetryPosition(latitude_deg=90.001, longitude_deg=0.0, altitude_meters=100.0)

    with pytest.raises(InvalidTelemetryDataError, match="latitude out of valid range"):
        TelemetryPosition(latitude_deg=-95.0, longitude_deg=0.0, altitude_meters=100.0)


# Test 3: Invalid longitude (out of bounds >180 or <-180)
def test_invalid_longitude_bounds():
    with pytest.raises(InvalidTelemetryDataError, match="longitude out of valid range"):
        TelemetryPosition(latitude_deg=0.0, longitude_deg=180.5, altitude_meters=100.0)

    with pytest.raises(InvalidTelemetryDataError, match="longitude out of valid range"):
        TelemetryPosition(latitude_deg=0.0, longitude_deg=-185.0, altitude_meters=100.0)


# Test 4: NaN latitude rejection
def test_nan_latitude_rejection():
    with pytest.raises(InvalidTelemetryDataError, match="cannot be NaN"):
        TelemetryPosition(latitude_deg=float("nan"), longitude_deg=0.0, altitude_meters=100.0)


# Test 5: NaN longitude rejection
def test_nan_longitude_rejection():
    with pytest.raises(InvalidTelemetryDataError, match="cannot be NaN"):
        TelemetryPosition(latitude_deg=0.0, longitude_deg=float("nan"), altitude_meters=100.0)


# Test 6: Invalid timestamp (negative or Infinite)
def test_invalid_timestamp_rejection():
    pos = TelemetryPosition(latitude_deg=10.0, longitude_deg=20.0, altitude_meters=50.0)
    with pytest.raises(InvalidTelemetryDataError, match="timestamp must be non-negative"):
        TelemetryRecord(timestamp=-1.0, position=pos)

    with pytest.raises(InvalidTelemetryDataError, match="timestamp must be non-negative"):
        TelemetryRecord(timestamp=float("inf"), position=pos)


# Test 7: Missing timestamp rejection
def test_missing_timestamp_rejection():
    pos = TelemetryPosition(latitude_deg=10.0, longitude_deg=20.0, altitude_meters=50.0)
    with pytest.raises(InvalidTelemetryDataError, match="requires a valid, non-null timestamp"):
        TelemetryRecord(timestamp=None, position=pos)


# Test 8: Optional orientation absent
def test_optional_orientation_absent():
    pos = TelemetryPosition(latitude_deg=12.0, longitude_deg=77.0, altitude_meters=900.0)
    record = TelemetryRecord(timestamp=0.0, position=pos)

    assert record.orientation is None
    assert record.gimbal_pitch_deg is None
    assert record.drone_heading_deg is None


# Test 9: Optional velocity absent
def test_optional_velocity_absent():
    pos = TelemetryPosition(latitude_deg=12.0, longitude_deg=77.0, altitude_meters=900.0)
    record = TelemetryRecord(timestamp=0.0, position=pos)

    assert record.velocity is None
    assert record.speed_mps is None


# Test 10: Altitude reference explicitly represented
def test_altitude_reference_explicit():
    pos_msl = TelemetryPosition(
        latitude_deg=0.0, longitude_deg=0.0, altitude_meters=100.0, altitude_reference=AltitudeReference.MSL
    )
    pos_agl = TelemetryPosition(
        latitude_deg=0.0, longitude_deg=0.0, altitude_meters=50.0, altitude_reference=AltitudeReference.AGL
    )

    assert pos_msl.altitude_reference == AltitudeReference.MSL
    assert pos_agl.altitude_reference == AltitudeReference.AGL


# Test 11: Unknown altitude reference preserved
def test_unknown_altitude_reference_preserved():
    pos = TelemetryPosition(latitude_deg=0.0, longitude_deg=0.0, altitude_meters=100.0)
    assert pos.altitude_reference == AltitudeReference.UNKNOWN


# Test 12: Duplicate timestamps supported in stream
def test_duplicate_timestamps_supported():
    pos1 = TelemetryPosition(latitude_deg=10.0, longitude_deg=20.0, altitude_meters=50.0)
    pos2 = TelemetryPosition(latitude_deg=10.0001, longitude_deg=20.0001, altitude_meters=50.2)

    r1 = TelemetryRecord(timestamp=1.0, position=pos1)
    r2 = TelemetryRecord(timestamp=1.0, position=pos2)

    stream = CanonicalTelemetryStream(stream_id="test_stream", records=[r1, r2])
    assert len(stream) == 2
    assert stream.has_duplicate_timestamps() is True
    assert stream[0].timestamp == stream[1].timestamp


# Test 13: Source order & provenance preserved
def test_source_order_and_provenance_preserved():
    pos1 = TelemetryPosition(latitude_deg=10.0, longitude_deg=20.0, altitude_meters=50.0)
    pos2 = TelemetryPosition(latitude_deg=10.0, longitude_deg=20.0, altitude_meters=51.0)

    # Inverted timestamps to test order preservation
    p1 = TelemetryProvenance(source_type="ardupilot_csv", record_index=0)
    p2 = TelemetryProvenance(source_type="ardupilot_csv", record_index=1)
    r1 = TelemetryRecord(timestamp=2.0, position=pos1, provenance=p1)
    r2 = TelemetryRecord(timestamp=1.0, position=pos2, provenance=p2)

    stream = CanonicalTelemetryStream(stream_id="order_test", records=[r1, r2])
    # Source order must be preserved in stream.records
    assert stream[0].provenance.record_index == 0
    assert stream[0].timestamp == 2.0
    assert stream[1].provenance.record_index == 1
    assert stream[1].timestamp == 1.0

    # Sorted chronological view without modifying underlying records
    chrono = stream.get_chronological_records()
    assert chrono[0].timestamp == 1.0
    assert chrono[1].timestamp == 2.0
    assert stream[0].timestamp == 2.0  # Original unaltered


# Test 14: Units explicitly represented
def test_units_explicit():
    pos = TelemetryPosition(latitude_deg=45.0, longitude_deg=-122.0, altitude_meters=150.0)
    ori = TelemetryOrientation(heading_deg=270.0, pitch_deg=-10.0, roll_deg=5.0)
    vel = TelemetryVelocity(speed_mps=12.4, climb_rate_mps=1.5)

    assert pos.latitude_deg == 45.0
    assert pos.longitude_deg == -122.0
    assert pos.altitude_meters == 150.0
    assert ori.heading_deg == 270.0
    assert vel.speed_mps == 12.4


# Test 15: Coordinate reference explicitly represented
def test_coordinate_reference_explicit():
    pos = TelemetryPosition(
        latitude_deg=18.0,
        longitude_deg=73.0,
        altitude_meters=500.0,
        position_reference=PositionReference.WGS84_GEODETIC
    )
    assert pos.position_reference == PositionReference.WGS84_GEODETIC


# Test 16: Serialization round-trip
def test_serialization_round_trip():
    pos = TelemetryPosition(
        latitude_deg=28.6139,
        longitude_deg=77.2090,
        altitude_meters=216.0,
        altitude_reference=AltitudeReference.MSL,
        position_reference=PositionReference.WGS84_GEODETIC
    )
    ori = TelemetryOrientation(
        heading_deg=90.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        gimbal_pitch_deg=-90.0
    )
    vel = TelemetryVelocity(speed_mps=5.0, north_velocity_mps=0.0, east_velocity_mps=5.0)
    qual = TelemetryQuality(fix_type=FixType.RTK_FIXED, horizontal_accuracy_meters=0.015)
    prov = TelemetryProvenance(source_type="dji_srt", source_identifier="DJI_0042.SRT", record_index=42)

    original = TelemetryRecord(
        timestamp=4.2,
        position=pos,
        timestamp_semantics=TimestampSemantics.VIDEO_RELATIVE,
        orientation=ori,
        velocity=vel,
        quality=qual,
        provenance=prov,
        extra_metadata={"custom_flag": True}
    )

    serialized = original.to_dict()
    assert isinstance(serialized, dict)
    assert serialized["position"]["latitude_deg"] == 28.6139
    assert serialized["quality"]["fix_type"] == "RTK_FIXED"

    reconstructed = TelemetryRecord.from_dict(serialized)
    assert reconstructed.timestamp == original.timestamp
    assert reconstructed.latitude_deg == original.latitude_deg
    assert reconstructed.longitude_deg == original.longitude_deg
    assert reconstructed.altitude_meters == original.altitude_meters
    assert reconstructed.position.altitude_reference == AltitudeReference.MSL
    assert reconstructed.orientation.gimbal_pitch_deg == -90.0
    assert reconstructed.velocity.speed_mps == 5.0
    assert reconstructed.quality.fix_type == FixType.RTK_FIXED
    assert reconstructed.provenance.record_index == 42
    assert reconstructed.extra_metadata["custom_flag"] is True
