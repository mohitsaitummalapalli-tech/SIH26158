"""Deterministic unit tests for Phase 1B.2 Telemetry Adapters.

DISCLAIMER:
ALL TEST FIXTURES IN THIS MODULE ARE SYNTHETIC TEST DATA GENERATED SOLELY FOR
UNIT TESTING TELEMETRY PARSERS. THEY ARE NOT REAL FLIGHT DATASETS.
"""

import os
import pytest
from src.ingestion import (
    DJISRTAdapter,
    GenericCSVAdapter,
    CSVColumnMapping,
    RecordStatus,
    AltitudeReference,
    PositionReference,
    TimestampSemantics,
    FixType,
)


# Fixture 1: Synthetic Valid DJI SRT File (TEST DATA)
@pytest.fixture
def synthetic_dji_srt(tmp_path):
    srt_content = """1
00:00:00,000 --> 00:00:00,033
HOME(73.8567,18.5204) 2023.08.15 14:30:12
GPS(73.8567,18.5204,18) D 15.20m, H 120.50m, H.S 5.20m/s, V.S 0.10m/s
[focal_len: 240] [dzoom: 100] [latitude: 18.5204] [longitude: 73.8567] [rel_alt: 120.500 abs_alt: 560.200]
[gb_pitch: -45.0 gb_roll: 0.0 gb_yaw: 145.2] [drone_pitch: -5.0 drone_roll: 1.2 drone_yaw: 145.2]

2
00:00:00,033 --> 00:00:00,066
HOME(73.8567,18.5204) 2023.08.15 14:30:12
GPS(73.8567,18.5204,18) D 15.40m, H 120.60m, H.S 5.25m/s, V.S 0.10m/s
[focal_len: 240] [dzoom: 100] [latitude: 18.52045] [longitude: 73.85675] [rel_alt: 120.600 abs_alt: 560.300]
[gb_pitch: -45.0 gb_roll: 0.0 gb_yaw: 145.2] [drone_pitch: -5.0 drone_roll: 1.2 drone_yaw: 145.2]
"""
    file_path = str(tmp_path / "valid_flight.SRT")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    return file_path


# Fixture 2: Synthetic Generic CSV (TEST DATA)
@pytest.fixture
def synthetic_generic_csv(tmp_path):
    csv_content = """time_sec,lat_deg,lon_deg,alt_m,yaw_deg,pitch_deg,roll_deg,speed_ms,sat_count
0.0,18.5204,73.8567,540.2,145.2,-5.0,1.2,5.2,18
0.1,18.5205,73.8568,540.4,145.3,-5.1,1.1,5.3,18
0.2,18.5206,73.8569,540.6,145.4,-5.0,1.2,5.1,19
"""
    file_path = str(tmp_path / "flight_telemetry.csv")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(csv_content)
    return file_path


# Test 1: Valid DJI SRT Parsing
def test_valid_dji_srt_parsing(synthetic_dji_srt):
    adapter = DJISRTAdapter(synthetic_dji_srt)
    records = adapter.parse_records()

    assert len(records) == 2
    assert records[0].status == RecordStatus.VALID
    rec = records[0].record
    assert rec is not None
    assert rec.timestamp == 0.0
    assert rec.latitude_deg == 18.5204
    assert rec.longitude_deg == 73.8567
    assert rec.altitude_meters == 560.2
    assert rec.position.altitude_reference == AltitudeReference.MSL
    assert rec.orientation.gimbal_pitch_deg == -45.0
    assert rec.orientation.heading_deg == 145.2
    assert rec.velocity.speed_mps == 5.2
    assert rec.quality.satellites_visible == 18
    assert rec.extra_metadata["focal_len"] == 240.0


# Test 2: Missing optional fields in SRT
def test_dji_srt_missing_optional_fields(tmp_path):
    # Minimal block with only timecode and coordinates (no gimbal/drone attitude, no speed)
    srt_content = """1
00:00:01,000 --> 00:00:01,033
[latitude: 18.5204] [longitude: 73.8567] [rel_alt: 100.0]
"""
    file_path = str(tmp_path / "minimal.SRT")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    adapter = DJISRTAdapter(file_path)
    records = adapter.parse_records()

    assert len(records) == 1
    assert records[0].status == RecordStatus.PARTIALLY_VALID
    rec = records[0].record
    assert rec.latitude_deg == 18.5204
    assert rec.orientation is None  # Not fabricated
    assert rec.velocity is None     # Not fabricated
    assert rec.position.altitude_reference == AltitudeReference.RELATIVE_TO_TAKEOFF


# Test 3: Unknown extra fields captured in extra_metadata
def test_dji_srt_unknown_extra_fields(tmp_path):
    srt_content = """1
00:00:00,000 --> 00:00:00,033
[latitude: 18.5204] [longitude: 73.8567] [altitude: 500.0] [custom_lens_id: ZEISS_50] [battery_temp: 34.5]
"""
    file_path = str(tmp_path / "custom_tags.SRT")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    adapter = DJISRTAdapter(file_path)
    records = adapter.parse_records()

    assert records[0].status == RecordStatus.PARTIALLY_VALID
    rec = records[0].record
    assert rec.extra_metadata["custom_lens_id"] == "ZEISS_50"
    assert rec.extra_metadata["battery_temp"] == 34.5


# Test 4: Malformed subtitle block rejection
def test_dji_srt_malformed_block(tmp_path):
    srt_content = """1
00:00:00,000 --> 00:00:00,033
Completely unparseable gibberish without any coordinates or tags
"""
    file_path = str(tmp_path / "malformed.SRT")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    adapter = DJISRTAdapter(file_path)
    records = adapter.parse_records()

    assert len(records) == 1
    assert records[0].status == RecordStatus.INVALID
    assert records[0].record is None
    assert "Missing essential spatial coordinates" in records[0].rejection_reason


# Test 5: Invalid coordinate rejection (out of bounds)
def test_dji_srt_invalid_coordinates(tmp_path):
    srt_content = """1
00:00:00,000 --> 00:00:00,033
[latitude: 195.0] [longitude: 73.8567] [altitude: 500.0]
"""
    file_path = str(tmp_path / "invalid_lat.SRT")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    adapter = DJISRTAdapter(file_path)
    records = adapter.parse_records()

    assert len(records) == 1
    assert records[0].status == RecordStatus.INVALID
    assert "Position validation failed" in records[0].rejection_reason


# Test 6: Duplicate timestamps handled cleanly
def test_dji_srt_duplicate_timestamps(tmp_path):
    srt_content = """1
00:00:01,000 --> 00:00:01,033
[latitude: 18.5204] [longitude: 73.8567] [altitude: 500.0]

2
00:00:01,000 --> 00:00:01,033
[latitude: 18.5205] [longitude: 73.8568] [altitude: 500.1]
"""
    file_path = str(tmp_path / "dup_time.SRT")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    adapter = DJISRTAdapter(file_path)
    stream = adapter.parse_stream()

    assert len(stream) == 2
    assert stream.has_duplicate_timestamps() is True
    assert stream[0].timestamp == stream[1].timestamp


# Test 7: Non-monotonic source timestamps preserved in source order
def test_dji_srt_non_monotonic_timestamps(tmp_path):
    srt_content = """1
00:00:02,000 --> 00:00:02,033
[latitude: 18.5204] [longitude: 73.8567] [altitude: 500.0]

2
00:00:01,000 --> 00:00:01,033
[latitude: 18.5205] [longitude: 73.8568] [altitude: 500.1]
"""
    file_path = str(tmp_path / "non_monotonic.SRT")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    adapter = DJISRTAdapter(file_path)
    stream = adapter.parse_stream()

    # Source order preserved
    assert stream[0].timestamp == 2.0
    assert stream[1].timestamp == 1.0

    # Chronological accessor sorts without modifying source order
    chrono = stream.get_chronological_records()
    assert chrono[0].timestamp == 1.0
    assert chrono[1].timestamp == 2.0
    assert stream[0].timestamp == 2.0


# Test 8: Valid generic CSV with explicit mapping
def test_valid_generic_csv(synthetic_generic_csv):
    mapping = CSVColumnMapping(
        timestamp_col="time_sec",
        latitude_col="lat_deg",
        longitude_col="lon_deg",
        altitude_col="alt_m",
        heading_col="yaw_deg",
        pitch_col="pitch_deg",
        roll_col="roll_deg",
        speed_col="speed_ms",
        satellites_col="sat_count",
        altitude_reference=AltitudeReference.ELLIPSOIDAL,
    )
    adapter = GenericCSVAdapter(synthetic_generic_csv, mapping)
    records = adapter.parse_records()

    assert len(records) == 3
    assert records[0].status == RecordStatus.VALID
    rec = records[0].record
    assert rec.timestamp == 0.0
    assert rec.latitude_deg == 18.5204
    assert rec.orientation.heading_deg == 145.2
    assert rec.velocity.speed_mps == 5.2
    assert rec.quality.satellites_visible == 18
    assert rec.position.altitude_reference == AltitudeReference.ELLIPSOIDAL


# Test 9: CSV missing required mapping
def test_csv_missing_required_mapping():
    with pytest.raises(ValueError, match="requires a non-empty 'timestamp_col'"):
        CSVColumnMapping(
            timestamp_col="",
            latitude_col="lat",
            longitude_col="lon",
            altitude_col="alt",
        )


# Test 10: Ambiguous/unsupported unit mapping
def test_csv_ambiguous_unit_mapping():
    with pytest.raises(ValueError, match="Ambiguous or unsupported angle_unit"):
        CSVColumnMapping(
            timestamp_col="time",
            latitude_col="lat",
            longitude_col="lon",
            altitude_col="alt",
            angle_unit="gradians",  # Invalid
        )


# Test 11: Provenance preservation
def test_adapter_provenance_preservation(synthetic_dji_srt):
    adapter = DJISRTAdapter(synthetic_dji_srt)
    records = adapter.parse_records()

    prov = records[0].record.provenance
    assert prov.source_type == "dji_srt"
    assert prov.source_identifier == os.path.abspath(synthetic_dji_srt)
    assert prov.record_index == 0
    assert prov.extraction_method == "DJISRTAdapter_v1.0"
    assert len(prov.source_checksum) == 64


# Test 12: Serialization round-trip
def test_adapter_record_serialization_roundtrip(synthetic_dji_srt):
    adapter = DJISRTAdapter(synthetic_dji_srt)
    records = adapter.parse_records()
    original_record = records[0].record

    serialized = original_record.to_dict()
    reconstructed = original_record.__class__.from_dict(serialized)

    assert reconstructed.timestamp == original_record.timestamp
    assert reconstructed.latitude_deg == original_record.latitude_deg
    assert reconstructed.longitude_deg == original_record.longitude_deg
    assert reconstructed.altitude_meters == original_record.altitude_meters
    assert reconstructed.orientation.gimbal_pitch_deg == original_record.orientation.gimbal_pitch_deg
    assert reconstructed.provenance.source_checksum == original_record.provenance.source_checksum
