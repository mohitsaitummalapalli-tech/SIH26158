"""Coordinate Normalization module: WGS84 Geodetic -> ECEF -> Local Topocentric ENU.

Authoritative WGS84 Constants (NGA / DoD Standard WGS 84, NIMA TR8350.2):
- Semi-major axis (a): 6378137.0 meters
- Reciprocal flattening (1/f): 298.257223563
- Semi-minor axis (b): a * (1 - f) = 6356752.314245179 meters
- First eccentricity squared (e^2): 2f - f^2 = (a^2 - b^2) / a^2 = 0.006694379990141316
- Second eccentricity squared (e'^2): (a^2 - b^2) / b^2 = e^2 / (1 - e^2) = 0.006739496742276434
"""

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Iterator

from src.ingestion.canonical_telemetry import (
    TelemetryPosition,
    TelemetryRecord,
    CanonicalTelemetryStream,
    AltitudeReference,
    PositionReference,
    TelemetryProvenance,
)


# --- WGS84 Ellipsoid Standard Constants ---
WGS84_A = 6378137.0                          # Semi-major axis in meters
WGS84_INV_F = 298.257223563                  # Reciprocal flattening
WGS84_F = 1.0 / WGS84_INV_F                  # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)          # Semi-minor axis in meters (6356752.314245179)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)         # First eccentricity squared (~0.00669437999014)
WGS84_EP2 = (WGS84_A**2 - WGS84_B**2) / (WGS84_B**2) # Second eccentricity squared (~0.00673949674228)


@dataclass(frozen=True)
class ECEFCoordinates:
    """Earth-Centered, Earth-Fixed (ECEF) 3D Cartesian coordinates in meters (EPSG:4978).
    
    Origin: Earth center of mass.
    X-axis: Intersection of Prime Meridian (0 deg lon) and Equator (0 deg lat).
    Y-axis: Intersection of 90 deg East longitude and Equator.
    Z-axis: Conventional Terrestrial Pole (True North).
    """
    x_meters: float
    y_meters: float
    z_meters: float

    def __post_init__(self) -> None:
        for name, val in [("x_meters", self.x_meters), ("y_meters", self.y_meters), ("z_meters", self.z_meters)]:
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"ECEF coordinate '{name}' must be a finite float, got {val}.")

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x_meters, self.y_meters, self.z_meters)


@dataclass(frozen=True)
class ENUCoordinates:
    """Local Topocentric East-North-Up (ENU) 3D Cartesian coordinates in meters.
    
    Origin: Tangent point defined by GeodeticOrigin.
    East (+X): Tangent to parallel of latitude in direction of increasing longitude.
    North (+Y): Tangent to meridian in direction of increasing latitude (True North).
    Up (+Z): Normal to WGS84 ellipsoid outward (away from Earth center).
    Convention: Right-handed orthogonal system (East x North = Up).
    """
    east_meters: float
    north_meters: float
    up_meters: float

    def __post_init__(self) -> None:
        for name, val in [("east_meters", self.east_meters), ("north_meters", self.north_meters), ("up_meters", self.up_meters)]:
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"ENU coordinate '{name}' must be a finite float, got {val}.")

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.east_meters, self.north_meters, self.up_meters)


@dataclass(frozen=True)
class GeodeticCoordinates:
    """WGS84 Geodetic coordinates (EPSG:4326)."""
    latitude_deg: float
    longitude_deg: float
    altitude_meters: float
    altitude_reference: AltitudeReference = AltitudeReference.ELLIPSOIDAL

    def __post_init__(self) -> None:
        if math.isnan(self.latitude_deg) or math.isinf(self.latitude_deg):
            raise ValueError(f"Latitude cannot be NaN or Infinite (got {self.latitude_deg}).")
        if self.latitude_deg < -90.0 or self.latitude_deg > 90.0:
            raise ValueError(f"Latitude out of bounds [-90.0, 90.0]: {self.latitude_deg}")

        if math.isnan(self.longitude_deg) or math.isinf(self.longitude_deg):
            raise ValueError(f"Longitude cannot be NaN or Infinite (got {self.longitude_deg}).")
        if self.longitude_deg < -180.0 or self.longitude_deg > 180.0:
            raise ValueError(f"Longitude out of bounds [-180.0, 180.0]: {self.longitude_deg}")

        if math.isnan(self.altitude_meters) or math.isinf(self.altitude_meters):
            raise ValueError(f"Altitude cannot be NaN or Infinite (got {self.altitude_meters}).")


class OriginPolicy(str, Enum):
    """Strategy for selecting the local Euclidean ENU tangent origin."""
    FIRST_VALID_POSITION = "FIRST_VALID_POSITION"  # First valid observation with ellipsoidal datum
    MEDIAN_POSITION = "MEDIAN_POSITION"            # Median position in metric 3D ECEF space
    EXPLICIT_ORIGIN = "EXPLICIT_ORIGIN"            # Manually configured anchor origin


@dataclass(frozen=True)
class GeodeticOrigin:
    """Immutable geodetic and ECEF anchor for local topocentric ENU projection."""
    latitude_deg: float
    longitude_deg: float
    altitude_meters: float
    altitude_reference: AltitudeReference
    ecef: ECEFCoordinates
    policy: OriginPolicy
    name: str = "local_enu_origin"

    @classmethod
    def from_geodetic(
        cls,
        latitude_deg: float,
        longitude_deg: float,
        altitude_meters: float,
        altitude_reference: AltitudeReference = AltitudeReference.ELLIPSOIDAL,
        policy: OriginPolicy = OriginPolicy.EXPLICIT_ORIGIN,
        name: str = "explicit_origin",
    ) -> "GeodeticOrigin":
        ecef = geodetic_to_ecef(latitude_deg, longitude_deg, altitude_meters)
        return cls(
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            altitude_meters=altitude_meters,
            altitude_reference=altitude_reference,
            ecef=ecef,
            policy=policy,
            name=name,
        )


# --- Core Transformation Mathematics ---

def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> ECEFCoordinates:
    """Transform WGS84 geodetic coordinates to Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates."""
    if math.isnan(lat_deg) or math.isinf(lat_deg) or lat_deg < -90.0 or lat_deg > 90.0:
        raise ValueError(f"Invalid latitude {lat_deg} deg.")
    if math.isnan(lon_deg) or math.isinf(lon_deg) or lon_deg < -180.0 or lon_deg > 180.0:
        raise ValueError(f"Invalid longitude {lon_deg} deg.")
    if math.isnan(alt_m) or math.isinf(alt_m):
        raise ValueError(f"Invalid altitude {alt_m} m.")

    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)

    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_lam = math.sin(lam)
    cos_lam = math.cos(lam)

    # Prime vertical radius of curvature
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)

    x = (n + alt_m) * cos_phi * cos_lam
    y = (n + alt_m) * cos_phi * sin_lam
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_phi

    return ECEFCoordinates(x_meters=x, y_meters=y, z_meters=z)


def ecef_to_geodetic(x: float, y: float, z: float) -> GeodeticCoordinates:
    """Transform ECEF Cartesian coordinates to WGS84 geodetic coordinates using Bowring's closed-form inversion."""
    if any(math.isnan(v) or math.isinf(v) for v in (x, y, z)):
        raise ValueError(f"ECEF coordinates cannot be NaN or Infinite, got ({x}, {y}, {z}).")

    p = math.sqrt(x * x + y * y)
    if p < 1e-6:
        # Exact polar singularity
        lat_deg = 90.0 if z >= 0 else -90.0
        lon_deg = 0.0
        alt_m = abs(z) - WGS84_B
        return GeodeticCoordinates(
            latitude_deg=lat_deg,
            longitude_deg=lon_deg,
            altitude_meters=alt_m,
            altitude_reference=AltitudeReference.ELLIPSOIDAL,
        )

    # Parametric angle theta
    theta = math.atan2(z * WGS84_A, p * WGS84_B)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    # Bowring geodetic latitude
    phi = math.atan2(
        z + WGS84_EP2 * WGS84_B * (sin_theta**3),
        p - WGS84_E2 * WGS84_A * (cos_theta**3)
    )
    lam = math.atan2(y, x)

    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)

    lat_deg = math.degrees(phi)
    lon_deg = math.degrees(lam)

    # Height calculation with high-latitude conditioning
    if abs(cos_phi) > 1e-4:
        alt_m = (p / cos_phi) - n
    else:
        alt_m = (z / sin_phi) - n * (1.0 - WGS84_E2)

    return GeodeticCoordinates(
        latitude_deg=lat_deg,
        longitude_deg=lon_deg,
        altitude_meters=alt_m,
        altitude_reference=AltitudeReference.ELLIPSOIDAL,
    )


def ecef_to_enu(point: ECEFCoordinates, origin: GeodeticOrigin) -> ENUCoordinates:
    """Transform ECEF Cartesian coordinates to local topocentric ENU coordinates relative to origin."""
    dx = point.x_meters - origin.ecef.x_meters
    dy = point.y_meters - origin.ecef.y_meters
    dz = point.z_meters - origin.ecef.z_meters

    phi_0 = math.radians(origin.latitude_deg)
    lam_0 = math.radians(origin.longitude_deg)

    sin_phi = math.sin(phi_0)
    cos_phi = math.cos(phi_0)
    sin_lam = math.sin(lam_0)
    cos_lam = math.cos(lam_0)

    # Standard orthonormal transformation matrix R_ecef_to_enu
    # East  = -sin(lam)*dx + cos(lam)*dy
    # North = -sin(phi)*cos(lam)*dx - sin(phi)*sin(lam)*dy + cos(phi)*dz
    # Up    =  cos(phi)*cos(lam)*dx + cos(phi)*sin(lam)*dy + sin(phi)*dz
    east = -sin_lam * dx + cos_lam * dy
    north = -sin_phi * cos_lam * dx - sin_phi * sin_lam * dy + cos_phi * dz
    up = cos_phi * cos_lam * dx + cos_phi * sin_lam * dy + sin_phi * dz

    return ENUCoordinates(east_meters=east, north_meters=north, up_meters=up)


def enu_to_ecef(point: ENUCoordinates, origin: GeodeticOrigin) -> ECEFCoordinates:
    """Transform local topocentric ENU coordinates to global ECEF coordinates relative to origin."""
    e = point.east_meters
    n = point.north_meters
    u = point.up_meters

    phi_0 = math.radians(origin.latitude_deg)
    lam_0 = math.radians(origin.longitude_deg)

    sin_phi = math.sin(phi_0)
    cos_phi = math.cos(phi_0)
    sin_lam = math.sin(lam_0)
    cos_lam = math.cos(lam_0)

    # Inverse rotation R^T
    dx = -sin_lam * e - sin_phi * cos_lam * n + cos_phi * cos_lam * u
    dy = cos_lam * e - sin_phi * sin_lam * n + cos_phi * sin_lam * u
    dz = cos_phi * n + sin_phi * u

    return ECEFCoordinates(
        x_meters=origin.ecef.x_meters + dx,
        y_meters=origin.ecef.y_meters + dy,
        z_meters=origin.ecef.z_meters + dz,
    )


# --- Normalized Stream Contracts & Normalizer Engine ---

@dataclass(frozen=True)
class NormalizedTelemetryRecord:
    """Normalized spatial telemetry observation containing both global and local metric coordinates."""
    record_id: str
    timestamp_seconds: float
    original_position: TelemetryPosition
    ecef_position: ECEFCoordinates
    enu_position: Optional[ENUCoordinates]
    is_valid_enu: bool
    datum_status: str  # "VALID_ELLIPSOIDAL", "INCOMPATIBLE_DATUM", "UNKNOWN_DATUM"
    source_provenance: Optional[TelemetryProvenance] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedTelemetryStream:
    """Collection of normalized telemetry observations referenced to an immutable local ENU origin."""
    stream_id: str
    origin: GeodeticOrigin
    records: List[NormalizedTelemetryRecord]
    provenance: Optional[TelemetryProvenance] = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> NormalizedTelemetryRecord:
        return self.records[idx]

    def __iter__(self) -> Iterator[NormalizedTelemetryRecord]:
        return iter(self.records)

    @property
    def valid_enu_count(self) -> int:
        return sum(1 for r in self.records if r.is_valid_enu)


class CoordinateNormalizer:
    """Engine for converting CanonicalTelemetryStream into NormalizedTelemetryStream."""

    @classmethod
    def compute_origin(
        cls,
        stream: CanonicalTelemetryStream,
        policy: OriginPolicy = OriginPolicy.FIRST_VALID_POSITION,
        explicit_origin: Optional[GeodeticOrigin] = None,
    ) -> GeodeticOrigin:
        """Deterministically establish the immutable local ENU origin according to policy."""
        if policy == OriginPolicy.EXPLICIT_ORIGIN:
            if explicit_origin is None:
                raise ValueError("OriginPolicy.EXPLICIT_ORIGIN requires an explicit_origin argument.")
            return explicit_origin

        valid_positions = [
            r.position for r in stream if r.position.altitude_reference == AltitudeReference.ELLIPSOIDAL
        ]
        if not valid_positions:
            # Fallback to any valid position if ellipsoidal not explicitly declared
            valid_positions = [r.position for r in stream]

        if not valid_positions:
            raise ValueError(f"Cannot compute origin: telemetry stream '{stream.stream_id}' has no valid positions.")

        if policy == OriginPolicy.FIRST_VALID_POSITION:
            pos0 = valid_positions[0]
            return GeodeticOrigin.from_geodetic(
                latitude_deg=pos0.latitude_deg,
                longitude_deg=pos0.longitude_deg,
                altitude_meters=pos0.altitude_meters,
                altitude_reference=pos0.altitude_reference,
                policy=policy,
                name=f"{stream.stream_id}_first_valid_origin",
            )

        elif policy == OriginPolicy.MEDIAN_POSITION:
            # Convert all positions to ECEF first to avoid spherical averaging errors
            ecef_points = [
                geodetic_to_ecef(p.latitude_deg, p.longitude_deg, p.altitude_meters)
                for p in valid_positions
            ]
            med_x = sorted(pt.x_meters for pt in ecef_points)[len(ecef_points) // 2]
            med_y = sorted(pt.y_meters for pt in ecef_points)[len(ecef_points) // 2]
            med_z = sorted(pt.z_meters for pt in ecef_points)[len(ecef_points) // 2]

            med_geo = ecef_to_geodetic(med_x, med_y, med_z)
            return GeodeticOrigin(
                latitude_deg=med_geo.latitude_deg,
                longitude_deg=med_geo.longitude_deg,
                altitude_meters=med_geo.altitude_meters,
                altitude_reference=valid_positions[0].altitude_reference,
                ecef=ECEFCoordinates(med_x, med_y, med_z),
                policy=policy,
                name=f"{stream.stream_id}_median_origin",
            )

        raise ValueError(f"Unknown OriginPolicy: {policy}")

    @classmethod
    def normalize_stream(
        cls,
        stream: CanonicalTelemetryStream,
        policy: OriginPolicy = OriginPolicy.FIRST_VALID_POSITION,
        explicit_origin: Optional[GeodeticOrigin] = None,
    ) -> NormalizedTelemetryStream:
        """Normalize all records in stream to ECEF and local topocentric ENU."""
        origin = cls.compute_origin(stream, policy=policy, explicit_origin=explicit_origin)
        normalized_records: List[NormalizedTelemetryRecord] = []

        for idx, rec in enumerate(stream):
            pos = rec.position
            # ECEF conversion is always computed directly
            ecef = geodetic_to_ecef(pos.latitude_deg, pos.longitude_deg, pos.altitude_meters)

            # Check datum compatibility for ENU
            is_ellipsoidal = pos.altitude_reference == AltitudeReference.ELLIPSOIDAL
            is_valid_enu = is_ellipsoidal or (origin.altitude_reference == pos.altitude_reference)

            enu: Optional[ENUCoordinates] = None
            if is_valid_enu:
                enu = ecef_to_enu(ecef, origin)

            datum_status = "VALID_ELLIPSOIDAL" if is_ellipsoidal else f"UNVERIFIED_{pos.altitude_reference.value}"

            norm_rec = NormalizedTelemetryRecord(
                record_id=f"{stream.stream_id}_rec_{idx:06d}",
                timestamp_seconds=rec.timestamp,
                original_position=pos,
                ecef_position=ecef,
                enu_position=enu,
                is_valid_enu=is_valid_enu,
                datum_status=datum_status,
                source_provenance=rec.provenance,
                extra_metadata=rec.extra_metadata,
            )
            normalized_records.append(norm_rec)

        return NormalizedTelemetryStream(
            stream_id=stream.stream_id,
            origin=origin,
            records=normalized_records,
            provenance=stream.provenance,
        )
