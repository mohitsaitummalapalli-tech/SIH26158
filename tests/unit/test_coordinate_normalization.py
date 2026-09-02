"""Deterministic unit tests for Phase 1B.4 & 1B.4.1 Coordinate Normalization (WGS84 -> ECEF -> Local ENU).

DISCLAIMER:
ALL FIXTURES IN THIS MODULE ARE SYNTHETIC TEST DATA GENERATED SOLELY FOR
UNIT TESTING COORDINATE NORMALIZATION ALGORITHMS. THEY DO NOT REPRESENT A REAL DRONE FLIGHT.
"""

import pytest
import math
from dataclasses import asdict

from src.geospatial import (
    WGS84_A,
    WGS84_B,
    WGS84_F,
    ECEFCoordinates,
    ENUCoordinates,
    GeodeticCoordinates,
    OriginPolicy,
    GeodeticOrigin,
    NormalizedTelemetryRecord,
    NormalizedTelemetryStream,
    CoordinateNormalizer,
    geodetic_to_ecef,
    ecef_to_geodetic,
    ecef_to_enu,
    enu_to_ecef,
)
from src.ingestion import (
    TelemetryPosition,
    TelemetryRecord,
    CanonicalTelemetryStream,
    AltitudeReference,
    PositionReference,
    TelemetryProvenance,
)


# Test 1: Known WGS84 -> ECEF transformation examples & Global ECEF axis verification
def test_known_wgs84_to_ecef_examples():
    """Verify Global ECEF Frame (EPSG:4978):
    - +X axis passes through Equator & WGS84 Reference Meridian (0 lat, 0 lon)
    - +Y axis passes through Equator & 90 deg East Meridian (0 lat, +90 lon)
    - +Z axis aligns with Terrestrial Reference Pole (North Pole, +90 lat)
    """
    # 1. Equator at Prime Meridian (0 deg lat, 0 deg lon, 0 m alt) -> (+a, 0, 0)
    p_x = geodetic_to_ecef(0.0, 0.0, 0.0)
    assert p_x.x_meters == pytest.approx(WGS84_A, abs=1e-3)
    assert p_x.y_meters == pytest.approx(0.0, abs=1e-3)
    assert p_x.z_meters == pytest.approx(0.0, abs=1e-3)

    # 2. Equator at 90 deg East (0 deg lat, +90 deg lon, 0 m alt) -> (0, +a, 0)
    p_y = geodetic_to_ecef(0.0, 90.0, 0.0)
    assert p_y.x_meters == pytest.approx(0.0, abs=1e-3)
    assert p_y.y_meters == pytest.approx(WGS84_A, abs=1e-3)
    assert p_y.z_meters == pytest.approx(0.0, abs=1e-3)

    # 3. North Pole (90 deg lat, 0 deg lon, 0 m alt) -> (0, 0, +b)
    p_z = geodetic_to_ecef(90.0, 0.0, 0.0)
    assert p_z.x_meters == pytest.approx(0.0, abs=1e-3)
    assert p_z.y_meters == pytest.approx(0.0, abs=1e-3)
    assert p_z.z_meters == pytest.approx(WGS84_B, abs=1e-3)

    # 4. South Pole (-90 deg lat, 0 deg lon, 0 m alt) -> (0, 0, -b)
    p_sp = geodetic_to_ecef(-90.0, 0.0, 0.0)
    assert p_sp.x_meters == pytest.approx(0.0, abs=1e-3)
    assert p_sp.y_meters == pytest.approx(0.0, abs=1e-3)
    assert p_sp.z_meters == pytest.approx(-WGS84_B, abs=1e-3)


# Test 2: ECEF -> WGS84 round-trip precision
def test_ecef_to_wgs84_round_trip():
    test_coords = [
        (0.0, 0.0, 0.0),
        (45.0, 45.0, 1000.0),
        (-33.8688, 151.2093, 50.0), # Sydney
        (51.5074, -0.1278, 15.0),   # London
        (35.6762, 139.6503, 40.0),  # Tokyo
    ]
    for lat, lon, alt in test_coords:
        ecef = geodetic_to_ecef(lat, lon, alt)
        geo = ecef_to_geodetic(ecef.x_meters, ecef.y_meters, ecef.z_meters)

        assert geo.latitude_deg == pytest.approx(lat, abs=1e-7)
        assert geo.longitude_deg == pytest.approx(lon, abs=1e-7)
        assert geo.altitude_meters == pytest.approx(alt, abs=1e-3)


# Test 3: Known local ENU transformation
def test_known_enu_transformation():
    origin = GeodeticOrigin.from_geodetic(18.5204, 73.8567, 560.0)
    # Convert origin to ENU -> must be (0, 0, 0)
    enu_zero = ecef_to_enu(origin.ecef, origin)
    assert enu_zero.east_meters == pytest.approx(0.0, abs=1e-4)
    assert enu_zero.north_meters == pytest.approx(0.0, abs=1e-4)
    assert enu_zero.up_meters == pytest.approx(0.0, abs=1e-4)

    # Displace by (100m East, 200m North, 50m Up)
    target_enu = ENUCoordinates(east_meters=100.0, north_meters=200.0, up_meters=50.0)
    target_ecef = enu_to_ecef(target_enu, origin)
    recovered_enu = ecef_to_enu(target_ecef, origin)

    assert recovered_enu.east_meters == pytest.approx(100.0, abs=1e-4)
    assert recovered_enu.north_meters == pytest.approx(200.0, abs=1e-4)
    assert recovered_enu.up_meters == pytest.approx(50.0, abs=1e-4)


# Test 4: ENU axis orientation & right-handed orthogonality
def test_enu_axis_orthogonality():
    origin = GeodeticOrigin.from_geodetic(30.0, 45.0, 100.0)
    # Construct unit basis vectors in ENU
    e_unit = enu_to_ecef(ENUCoordinates(1.0, 0.0, 0.0), origin)
    n_unit = enu_to_ecef(ENUCoordinates(0.0, 1.0, 0.0), origin)
    u_unit = enu_to_ecef(ENUCoordinates(0.0, 0.0, 1.0), origin)

    # Vector relative to origin
    de = (e_unit.x_meters - origin.ecef.x_meters, e_unit.y_meters - origin.ecef.y_meters, e_unit.z_meters - origin.ecef.z_meters)
    dn = (n_unit.x_meters - origin.ecef.x_meters, n_unit.y_meters - origin.ecef.y_meters, n_unit.z_meters - origin.ecef.z_meters)
    du = (u_unit.x_meters - origin.ecef.x_meters, u_unit.y_meters - origin.ecef.y_meters, u_unit.z_meters - origin.ecef.z_meters)

    # 1. Unit Length
    assert math.sqrt(de[0]**2 + de[1]**2 + de[2]**2) == pytest.approx(1.0, abs=1e-7)
    assert math.sqrt(dn[0]**2 + dn[1]**2 + dn[2]**2) == pytest.approx(1.0, abs=1e-7)
    assert math.sqrt(du[0]**2 + du[1]**2 + du[2]**2) == pytest.approx(1.0, abs=1e-7)

    # 2. Dot products must be 0 (Mutual Orthogonality)
    dot_en = de[0]*dn[0] + de[1]*dn[1] + de[2]*dn[2]
    dot_nu = dn[0]*du[0] + dn[1]*du[1] + dn[2]*du[2]
    dot_eu = de[0]*du[0] + de[1]*du[1] + de[2]*du[2]
    assert dot_en == pytest.approx(0.0, abs=1e-7)
    assert dot_nu == pytest.approx(0.0, abs=1e-7)
    assert dot_eu == pytest.approx(0.0, abs=1e-7)

    # 3. Cross product East x North = Up (Right-handed System)
    cross_en = (
        de[1]*dn[2] - de[2]*dn[1],
        de[2]*dn[0] - de[0]*dn[2],
        de[0]*dn[1] - de[1]*dn[0]
    )
    assert cross_en[0] == pytest.approx(du[0], abs=1e-7)
    assert cross_en[1] == pytest.approx(du[1], abs=1e-7)
    assert cross_en[2] == pytest.approx(du[2], abs=1e-7)


# Test 4b: Physical displacement mapping to +E, +N, +U directions
def test_enu_physical_displacement_directions():
    origin = GeodeticOrigin.from_geodetic(0.0, 0.0, 0.0) # Equator / Prime Meridian

    # 1. Small displacement toward increasing longitude (+0.001 deg lon) -> +East
    p_east = geodetic_to_ecef(0.0, 0.001, 0.0)
    enu_east = ecef_to_enu(p_east, origin)
    assert enu_east.east_meters > 100.0
    assert abs(enu_east.north_meters) < 1.0
    assert abs(enu_east.up_meters) < 1.0

    # 2. Small displacement toward increasing latitude (+0.001 deg lat) -> +North
    p_north = geodetic_to_ecef(0.001, 0.0, 0.0)
    enu_north = ecef_to_enu(p_north, origin)
    assert enu_north.north_meters > 100.0
    assert abs(enu_east.north_meters) < 1.0
    assert abs(enu_north.east_meters) < 1.0

    # 3. Positive vertical displacement (+50m height) -> +Up
    p_up = geodetic_to_ecef(0.0, 0.0, 50.0)
    enu_up = ecef_to_enu(p_up, origin)
    assert enu_up.up_meters == pytest.approx(50.0, abs=1e-3)
    assert abs(enu_up.east_meters) < 1e-4
    assert abs(enu_up.north_meters) < 1e-4


# Test 5: Antimeridian continuity
def test_antimeridian_continuity():
    p1 = geodetic_to_ecef(0.0, 179.9, 0.0)
    p2 = geodetic_to_ecef(0.0, -179.9, 0.0)

    # Distance in ECEF space across 0.2 deg gap at equator
    dx = p2.x_meters - p1.x_meters
    dy = p2.y_meters - p1.y_meters
    dz = p2.z_meters - p1.z_meters
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)

    # Expected chord distance ~ 2 * a * sin(0.1 deg) ~ 22.26 km
    expected_dist = 2.0 * WGS84_A * math.sin(math.radians(0.1))
    assert dist == pytest.approx(expected_dist, rel=1e-3)
    assert dist < 25000.0  # Must NOT be 20,000 km across prime meridian


# Test 6: Negative height handling
def test_negative_height_roundtrip():
    lat, lon, alt = 31.5, 35.5, -420.0  # Dead Sea depression
    ecef = geodetic_to_ecef(lat, lon, alt)
    geo = ecef_to_geodetic(ecef.x_meters, ecef.y_meters, ecef.z_meters)

    assert geo.latitude_deg == pytest.approx(lat, abs=1e-7)
    assert geo.longitude_deg == pytest.approx(lon, abs=1e-7)
    assert geo.altitude_meters == pytest.approx(alt, abs=1e-3)


# Test 7: High-latitude robustness
def test_high_latitude_robustness():
    # Near North Pole 89.9999 deg
    lat, lon, alt = 89.9999, 45.0, 100.0
    ecef = geodetic_to_ecef(lat, lon, alt)
    geo = ecef_to_geodetic(ecef.x_meters, ecef.y_meters, ecef.z_meters)

    assert geo.latitude_deg == pytest.approx(lat, abs=1e-6)
    assert geo.longitude_deg == pytest.approx(lon, abs=1e-4)
    assert geo.altitude_meters == pytest.approx(alt, abs=1e-3)


# Test 8: Explicit origin policy
def test_explicit_origin_policy():
    records = [
        TelemetryRecord(
            timestamp=0.0,
            position=TelemetryPosition(latitude_deg=10.0, longitude_deg=20.0, altitude_meters=100.0, altitude_reference=AltitudeReference.ELLIPSOIDAL)
        )
    ]
    stream = CanonicalTelemetryStream(stream_id="test_exp", records=records)
    exp_orig = GeodeticOrigin.from_geodetic(12.0, 22.0, 150.0, policy=OriginPolicy.EXPLICIT_ORIGIN)

    norm_stream = CoordinateNormalizer.normalize_stream(stream, policy=OriginPolicy.EXPLICIT_ORIGIN, explicit_origin=exp_orig)
    assert norm_stream.origin.latitude_deg == 12.0
    assert norm_stream.origin.longitude_deg == 22.0


# Test 9: First-valid origin policy
def test_first_valid_origin_policy():
    records = [
        TelemetryRecord(
            timestamp=0.0,
            position=TelemetryPosition(latitude_deg=10.0, longitude_deg=20.0, altitude_meters=100.0, altitude_reference=AltitudeReference.ELLIPSOIDAL)
        ),
        TelemetryRecord(
            timestamp=1.0,
            position=TelemetryPosition(latitude_deg=11.0, longitude_deg=21.0, altitude_meters=110.0, altitude_reference=AltitudeReference.ELLIPSOIDAL)
        ),
    ]
    stream = CanonicalTelemetryStream(stream_id="test_first", records=records)
    norm_stream = CoordinateNormalizer.normalize_stream(stream, policy=OriginPolicy.FIRST_VALID_POSITION)
    assert norm_stream.origin.latitude_deg == 10.0
    assert norm_stream.origin.longitude_deg == 20.0
    assert norm_stream.origin.policy == OriginPolicy.FIRST_VALID_POSITION


# Test 10: Median ECEF-derived origin policy
def test_median_origin_policy():
    records = [
        TelemetryRecord(timestamp=0.0, position=TelemetryPosition(latitude_deg=10.0, longitude_deg=20.0, altitude_meters=100.0, altitude_reference=AltitudeReference.ELLIPSOIDAL)),
        TelemetryRecord(timestamp=1.0, position=TelemetryPosition(latitude_deg=10.2, longitude_deg=20.2, altitude_meters=120.0, altitude_reference=AltitudeReference.ELLIPSOIDAL)),
        TelemetryRecord(timestamp=2.0, position=TelemetryPosition(latitude_deg=10.4, longitude_deg=20.4, altitude_meters=140.0, altitude_reference=AltitudeReference.ELLIPSOIDAL)),
    ]
    stream = CanonicalTelemetryStream(stream_id="test_med", records=records)
    norm_stream = CoordinateNormalizer.normalize_stream(stream, policy=OriginPolicy.MEDIAN_POSITION)
    assert norm_stream.origin.latitude_deg == pytest.approx(10.2, abs=1e-3)
    assert norm_stream.origin.longitude_deg == pytest.approx(20.2, abs=1e-3)


# Test 11: Immutable origin
def test_immutable_origin():
    origin = GeodeticOrigin.from_geodetic(18.0, 73.0, 500.0)
    with pytest.raises(Exception):
        origin.latitude_deg = 20.0  # Frozen dataclass mutation must fail


# Test 12: ECEF unaffected by ENU origin change
def test_ecef_unaffected_by_enu_origin_change():
    pos = TelemetryPosition(latitude_deg=18.5, longitude_deg=73.8, altitude_meters=550.0, altitude_reference=AltitudeReference.ELLIPSOIDAL)
    ecef1 = geodetic_to_ecef(pos.latitude_deg, pos.longitude_deg, pos.altitude_meters)

    orig_a = GeodeticOrigin.from_geodetic(18.0, 73.0, 500.0)
    orig_b = GeodeticOrigin.from_geodetic(20.0, 75.0, 600.0)

    enu_a = ecef_to_enu(ecef1, orig_a)
    enu_b = ecef_to_enu(ecef1, orig_b)

    # ENU coordinates differ under different origins
    assert enu_a.east_meters != enu_b.east_meters

    # But converting back to ECEF recovers the exact same ECEF point
    rec_ecef_a = enu_to_ecef(enu_a, orig_a)
    rec_ecef_b = enu_to_ecef(enu_b, orig_b)

    assert rec_ecef_a.x_meters == pytest.approx(ecef1.x_meters, abs=1e-4)
    assert rec_ecef_b.x_meters == pytest.approx(ecef1.x_meters, abs=1e-4)


# Test 13: Invalid coordinate rejection
def test_invalid_coordinate_rejection():
    with pytest.raises(ValueError, match="Invalid latitude"):
        geodetic_to_ecef(95.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="Invalid longitude"):
        geodetic_to_ecef(0.0, 200.0, 0.0)

    with pytest.raises(ValueError, match="Invalid altitude"):
        geodetic_to_ecef(0.0, 0.0, float("nan"))


# Test 14: Unsupported / Non-ellipsoidal altitude reference handling
def test_unsupported_altitude_reference_handling():
    records = [
        TelemetryRecord(
            timestamp=0.0,
            position=TelemetryPosition(latitude_deg=10.0, longitude_deg=20.0, altitude_meters=100.0, altitude_reference=AltitudeReference.MSL)
        )
    ]
    stream = CanonicalTelemetryStream(stream_id="test_msl", records=records)
    norm_stream = CoordinateNormalizer.normalize_stream(stream, policy=OriginPolicy.FIRST_VALID_POSITION)

    rec0 = norm_stream[0]
    assert rec0.datum_status == "UNVERIFIED_MSL"


# Test 15: Provenance preservation
def test_normalization_provenance_preservation():
    prov = TelemetryProvenance(source_type="flight_log", source_identifier="log_001.csv", record_index=42)
    records = [
        TelemetryRecord(
            timestamp=0.0,
            position=TelemetryPosition(latitude_deg=10.0, longitude_deg=20.0, altitude_meters=100.0, altitude_reference=AltitudeReference.ELLIPSOIDAL),
            provenance=prov
        )
    ]
    stream = CanonicalTelemetryStream(stream_id="test_prov", records=records, provenance=prov)
    norm_stream = CoordinateNormalizer.normalize_stream(stream)

    assert norm_stream.provenance.source_identifier == "log_001.csv"
    assert norm_stream[0].source_provenance.record_index == 42


# Test 16: Normalization serialization round-trip
def test_normalization_serialization_roundtrip():
    records = [
        TelemetryRecord(
            timestamp=0.0,
            position=TelemetryPosition(latitude_deg=18.5, longitude_deg=73.8, altitude_meters=550.0, altitude_reference=AltitudeReference.ELLIPSOIDAL)
        )
    ]
    stream = CanonicalTelemetryStream(stream_id="test_ser", records=records)
    norm_stream = CoordinateNormalizer.normalize_stream(stream)

    rec_dict = asdict(norm_stream[0])
    assert rec_dict["record_id"] == "test_ser_rec_000000"
    assert "ecef_position" in rec_dict
    assert "enu_position" in rec_dict
    assert rec_dict["ecef_position"]["x_meters"] == pytest.approx(norm_stream[0].ecef_position.x_meters, abs=1e-4)
