"""Geospatial coordinate frame handling and transformations.

Converts between:
- WGS84 Geodetic (latitude_deg, longitude_deg, altitude_m)
- Earth-Centered, Earth-Fixed ECEF (x, y, z in meters, EPSG:4978)
- Local Topocentric East-North-Up ENU (east, north, up in meters)

Follows NIMA TR8350.2 WGS84 ellipsoid constants and Bowring's closed-form inversion.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple, Union
import numpy as np

from src.ingestion.canonical_telemetry import AltitudeReference
from src.geospatial.normalization import (
    WGS84_A,
    WGS84_B,
    WGS84_F,
    WGS84_E2,
    WGS84_EP2,
    ECEFCoordinates,
    ENUCoordinates,
    GeodeticCoordinates,
    geodetic_to_ecef as norm_geodetic_to_ecef,
    ecef_to_geodetic as norm_ecef_to_geodetic,
    ecef_to_enu as norm_ecef_to_enu,
    enu_to_ecef as norm_enu_to_ecef,
    GeodeticOrigin,
    OriginPolicy as NormOriginPolicy,
)


class AltitudeReferenceType(str, Enum):
    """Altitude reference datum classification."""
    ELLIPSOIDAL_WGS84 = "ELLIPSOIDAL_WGS84"
    ORTHOMETRIC_MSL = "ORTHOMETRIC_MSL"
    RELATIVE_TAKEOFF = "RELATIVE_TAKEOFF"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class GeospatialAnchorOrigin:
    """Local Topocentric ENU Anchor Datum."""
    lat_deg: float
    lon_deg: float
    ellipsoidal_height_m: float
    altitude_reference: AltitudeReferenceType = AltitudeReferenceType.ELLIPSOIDAL_WGS84
    origin_policy: str = "FIRST_VALID_POSITION"

    def __post_init__(self) -> None:
        if math.isnan(self.lat_deg) or math.isinf(self.lat_deg):
            raise ValueError(f"Anchor latitude must be finite, got {self.lat_deg}")
        if self.lat_deg < -90.0 or self.lat_deg > 90.0:
            raise ValueError(f"Anchor latitude out of bounds [-90, 90]: {self.lat_deg}")
        if math.isnan(self.lon_deg) or math.isinf(self.lon_deg):
            raise ValueError(f"Anchor longitude must be finite, got {self.lon_deg}")
        if self.lon_deg < -180.0 or self.lon_deg > 180.0:
            raise ValueError(f"Anchor longitude out of bounds [-180, 180]: {self.lon_deg}")
        if math.isnan(self.ellipsoidal_height_m) or math.isinf(self.ellipsoidal_height_m):
            raise ValueError(f"Anchor height must be finite, got {self.ellipsoidal_height_m}")

    @property
    def ecef_anchor(self) -> Tuple[float, float, float]:
        """Convert anchor to ECEF coordinates."""
        return wgs84_to_ecef(self.lat_deg, self.lon_deg, self.ellipsoidal_height_m)

    def to_normalization_origin(self) -> GeodeticOrigin:
        """Convert to normalization.py GeodeticOrigin."""
        policy = NormOriginPolicy.FIRST_VALID_POSITION
        if self.origin_policy == "MEDIAN_POSITION":
            policy = NormOriginPolicy.MEDIAN_POSITION
        elif self.origin_policy == "EXPLICIT_ORIGIN":
            policy = NormOriginPolicy.EXPLICIT_ORIGIN

        alt_ref = AltitudeReference.ELLIPSOIDAL
        if self.altitude_reference == AltitudeReferenceType.ORTHOMETRIC_MSL:
            alt_ref = AltitudeReference.MSL
        elif self.altitude_reference == AltitudeReferenceType.RELATIVE_TAKEOFF:
            alt_ref = AltitudeReference.RELATIVE_TO_TAKEOFF

        return GeodeticOrigin.from_geodetic(
            latitude_deg=self.lat_deg,
            longitude_deg=self.lon_deg,
            altitude_meters=self.ellipsoidal_height_m,
            altitude_reference=alt_ref,
            policy=policy,
        )


def wgs84_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> Tuple[float, float, float]:
    """Convert WGS84 geodetic coordinates to ECEF Cartesian coordinates in meters.
    
    Args:
        lat_deg: Latitude in degrees in [-90.0, 90.0].
        lon_deg: Longitude in degrees in [-180.0, 180.0].
        alt_m: Ellipsoidal height in meters.
        
    Returns:
        (X, Y, Z) in meters.
    """
    if math.isnan(lat_deg) or abs(lat_deg) > 90.0:
        raise ValueError(f"Invalid latitude: {lat_deg}")
    if math.isnan(lon_deg) or abs(lon_deg) > 180.0:
        raise ValueError(f"Invalid longitude: {lon_deg}")
    if math.isnan(alt_m) or math.isinf(alt_m):
        raise ValueError(f"Invalid altitude: {alt_m}")
    ecef_coords = norm_geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    return ecef_coords.as_tuple()


def ecef_to_wgs84(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert ECEF Cartesian coordinates to WGS84 geodetic coordinates via Bowring's method.
    
    Args:
        x, y, z: ECEF coordinates in meters.
        
    Returns:
        (lat_deg, lon_deg, alt_m)
    """
    for val, name in [(x, "x"), (y, "y"), (z, "z")]:
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Invalid ECEF coordinate '{name}': {val}")
    geo_coords = norm_ecef_to_geodetic(x, y, z)
    return (geo_coords.latitude_deg, geo_coords.longitude_deg, geo_coords.altitude_meters)


def ecef_to_enu(
    x: float, y: float, z: float,
    origin_lat_deg: float, origin_lon_deg: float, origin_alt_m: float,
) -> Tuple[float, float, float]:
    """Convert ECEF Cartesian coordinates to local topocentric East-North-Up (ENU).
    
    Args:
        x, y, z: Target ECEF coordinates in meters.
        origin_lat_deg, origin_lon_deg, origin_alt_m: Anchor origin geodetic position.
        
    Returns:
        (east, north, up) in meters.
    """
    origin = GeodeticOrigin.from_geodetic(
        latitude_deg=origin_lat_deg,
        longitude_deg=origin_lon_deg,
        altitude_meters=origin_alt_m,
    )
    enu_coords = norm_ecef_to_enu(ECEFCoordinates(x_meters=x, y_meters=y, z_meters=z), origin)
    return enu_coords.as_tuple()


def enu_to_ecef(
    east: float, north: float, up: float,
    origin_lat_deg: float, origin_lon_deg: float, origin_alt_m: float,
) -> Tuple[float, float, float]:
    """Convert local topocentric East-North-Up (ENU) to ECEF Cartesian coordinates.
    
    Args:
        east, north, up: ENU coordinates in meters.
        origin_lat_deg, origin_lon_deg, origin_alt_m: Anchor origin geodetic position.
        
    Returns:
        (x, y, z) in meters.
    """
    origin = GeodeticOrigin.from_geodetic(
        latitude_deg=origin_lat_deg,
        longitude_deg=origin_lon_deg,
        altitude_meters=origin_alt_m,
    )
    ecef_coords = norm_enu_to_ecef(ENUCoordinates(east_meters=east, north_meters=north, up_meters=up), origin)
    return ecef_coords.as_tuple()


def wgs84_to_enu(
    lat_deg: float, lon_deg: float, alt_m: float,
    anchor: GeospatialAnchorOrigin,
) -> Tuple[float, float, float]:
    """Convert WGS84 geodetic coordinates directly to local ENU tangent frame.
    
    Optimization Rule: All Euclidean optimization occurs in ENU. Never optimize directly in degrees.
    """
    x, y, z = wgs84_to_ecef(lat_deg, lon_deg, alt_m)
    return ecef_to_enu(x, y, z, anchor.lat_deg, anchor.lon_deg, anchor.ellipsoidal_height_m)


def enu_to_wgs84(
    east: float, north: float, up: float,
    anchor: GeospatialAnchorOrigin,
) -> Tuple[float, float, float]:
    """Convert local ENU coordinates directly to WGS84 geodetic coordinates."""
    x, y, z = enu_to_ecef(east, north, up, anchor.lat_deg, anchor.lon_deg, anchor.ellipsoidal_height_m)
    return ecef_to_wgs84(x, y, z)


def build_rotation_enu_from_ecef(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Construct 3x3 orthonormal rotation matrix from ECEF to local ENU.
    
    R_ECEF->ENU = [
        [-sin(lon),               cos(lon),              0],
        [-sin(lat)*cos(lon),     -sin(lat)*sin(lon),    cos(lat)],
        [ cos(lat)*cos(lon),      cos(lat)*sin(lon),    sin(lat)]
    ]
    """
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_lam = math.sin(lam)
    cos_lam = math.cos(lam)

    return np.array([
        [-sin_lam, cos_lam, 0.0],
        [-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi],
        [cos_phi * cos_lam, cos_phi * sin_lam, sin_phi],
    ], dtype=np.float64)
