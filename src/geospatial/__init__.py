"""Geospatial module: Coordinate normalization, Sim(3) metric estimation, and georeferencing."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

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

from src.geospatial.coordinates import (
    GeospatialAnchorOrigin,
    AltitudeReferenceType,
    wgs84_to_ecef,
    ecef_to_wgs84,
    wgs84_to_enu,
    enu_to_wgs84,
    build_rotation_enu_from_ecef,
)

from src.geospatial.sim3 import (
    Sim3,
    UncertaintyType,
    Sim3TransformContract,
    solve_sim3_umeyama,
)

from src.geospatial.lever_arm import (
    LeverArm,
    LeverArmStatus,
)

from src.geospatial.telemetry_observation import (
    TelemetryObservation,
    ObservationClassification,
    GnssAccuracyInterpretation,
    construct_gnss_covariance,
)

from src.geospatial.synchronization import (
    RawTelemetryRecord,
    TelemetrySynchronizer,
)

from src.geospatial.observability import (
    ScaleObservabilityReport,
    FullSim3ObservabilityStatus,
    check_scale_observability,
)

from src.geospatial.robust_estimation import (
    RobustSim3Estimator,
    RobustSim3Result,
    EstimationDiagnostics,
    compute_isoperimetric_quotient,
)

from src.geospatial.metric_state import (
    MetricScaleStatus,
    MetricStateMachine,
    MetricStateTransition,
)

from src.geospatial.validation import (
    GroundControlPoint,
    MetricValidator,
    ValidationReport,
    CheckpointResidual,
)

from src.geospatial.uncertainty import (
    UncertaintyPropagator,
    Sim3UncertaintyReport,
)

from src.geospatial.pipeline import (
    GeospatialMetricReconstructor,
    GeospatialMetricReconstructionResult,
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
    translation_enu: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale_uncertainty_1sigma: float = 0.0
    uncertainty_type: str = "UNAVAILABLE"

    def __post_init__(self) -> None:
        if self.translation_vector and self.translation_enu == (0.0, 0.0, 0.0) and len(self.translation_vector) == 3:
            self.translation_enu = (self.translation_vector[0], self.translation_vector[1], self.translation_vector[2])
        elif self.translation_enu != (0.0, 0.0, 0.0) and self.translation_vector == [0.0, 0.0, 0.0]:
            self.translation_vector = [self.translation_enu[0], self.translation_enu[1], self.translation_enu[2]]


@dataclass
class GeoreferenceMetadata:
    """Complete georeferencing manifest attached to exported 3D spatial models."""
    target_crs: str  # e.g., "EPSG:32643" (WGS84 UTM Zone 43N)
    origin_latitude_deg: float
    origin_longitude_deg: float
    origin_altitude_meters: float
    sim3_transform: Sim3Transform
    bounds_utm_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_utm_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    geoid_model: str = "EGM96"


__all__ = [
    # WGS84 Constants & Normalization
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
    # Phase 3E.5 Components
    "GeospatialAnchorOrigin",
    "AltitudeReferenceType",
    "wgs84_to_ecef",
    "ecef_to_wgs84",
    "wgs84_to_enu",
    "enu_to_wgs84",
    "build_rotation_enu_from_ecef",
    "Sim3",
    "UncertaintyType",
    "Sim3TransformContract",
    "solve_sim3_umeyama",
    "LeverArm",
    "LeverArmStatus",
    "TelemetryObservation",
    "ObservationClassification",
    "GnssAccuracyInterpretation",
    "construct_gnss_covariance",
    "RawTelemetryRecord",
    "TelemetrySynchronizer",
    "ScaleObservabilityReport",
    "FullSim3ObservabilityStatus",
    "check_scale_observability",
    "RobustSim3Estimator",
    "RobustSim3Result",
    "EstimationDiagnostics",
    "compute_isoperimetric_quotient",
    "MetricScaleStatus",
    "MetricStateMachine",
    "MetricStateTransition",
    "GroundControlPoint",
    "MetricValidator",
    "ValidationReport",
    "CheckpointResidual",
    "UncertaintyPropagator",
    "Sim3UncertaintyReport",
    "GeospatialMetricReconstructor",
    "GeospatialMetricReconstructionResult",
]
