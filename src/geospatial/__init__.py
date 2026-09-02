"""Geospatial module: Coordinate normalization (WGS84 -> ECEF -> ENU), CRS contracts, and metadata."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.geospatial.normalization import (
    WGS84_A,
    WGS84_INV_F,
    WGS84_F,
    WGS84_B,
    WGS84_E2,
    WGS84_EP2,
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


@dataclass(frozen=True)
class CRSInfo:
    """Coordinate Reference System specification.
    
    Standards:
    - EPSG:4326: WGS84 Geodetic 3D (Latitude deg, Longitude deg, Ellipsoidal Height m).
    - EPSG:4978: WGS84 ECEF 3D Cartesian (X m, Y m, Z m).
    - EPSG:32601-32660: WGS84 UTM North Zones (Easting m, Northing m, Height m).
    - EPSG:32701-32760: WGS84 UTM South Zones (Easting m, Northing m, Height m).
    """
    auth_name: str = "EPSG"
    code: int = 4326
    is_projected: bool = False
    utm_zone: Optional[int] = None
    utm_hemisphere: Optional[str] = None  # "N" or "S"


@dataclass
class Sim3Transform:
    """7-DoF Similarity Transformation mapping relative reconstruction frame to local Euclidean ENU.
    
    Formulation:
    X_enu = scale * R * X_model + translation
    
    Conventions:
    - scale: Positive scalar metric factor (meters per model unit).
    - rotation_matrix: 3x3 orthonormal SO(3) matrix (det = +1).
    - translation_vector: 3D vector [East, North, Up] in meters.
    - target_frame: Local Topocentric ENU (+X East, +Y North, +Z Up).
    """
    scale: float = 1.0
    rotation_matrix: List[List[float]] = field(default_factory=lambda: [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    translation_vector: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    source_crs: str = "local_model"
    target_crs: str = "local_topocentric_enu"
    residual_rmse_meters: Optional[float] = None
    num_anchors_used: int = 0


@dataclass
class GeoreferenceMetadata:
    """Complete georeferencing manifest attached to exported 3D spatial models.
    
    Contains the transformation parameters from model coordinates to global Projected CRS.
    """
    target_crs: str  # e.g., "EPSG:32643" (WGS84 UTM Zone 43N)
    origin_latitude_deg: float
    origin_longitude_deg: float
    origin_altitude_meters: float
    sim3_transform: Sim3Transform
    bounds_utm_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (Easting, Northing, Height)
    bounds_utm_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (Easting, Northing, Height)
    geoid_model: str = "EGM96"  # Geoid model for orthometric height conversion


__all__ = [
    "WGS84_A",
    "WGS84_INV_F",
    "WGS84_F",
    "WGS84_B",
    "WGS84_E2",
    "WGS84_EP2",
    "ECEFCoordinates",
    "ENUCoordinates",
    "GeodeticCoordinates",
    "OriginPolicy",
    "GeodeticOrigin",
    "NormalizedTelemetryRecord",
    "NormalizedTelemetryStream",
    "CoordinateNormalizer",
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "ecef_to_enu",
    "enu_to_ecef",
    "CRSInfo",
    "Sim3Transform",
    "GeoreferenceMetadata",
]
