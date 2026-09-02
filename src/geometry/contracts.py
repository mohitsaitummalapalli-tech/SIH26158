"""Phase 3 Classical Geometry Contracts, Interfaces, and Mathematical Models.

Defines strict type annotations, coordinate conventions, failure taxonomy,
measurement labeling, and mathematical verification contracts for:
- Feature extraction & matching
- Two-view epipolar geometry (Fundamental Matrix F vs Essential Matrix E)
- Triangulation & cheirality validation
- Incremental Structure-from-Motion (SfM)
- Bundle Adjustment (BA) gauge fixing policies & nonlinear optimization
- Lens distortion models (Rectified vs Brown-Conrady vs Fisheye)
- Dense Multi-View Stereo (MVS) contracts & completeness metrics
- Three-level scientific evaluation hierarchy

SCIENTIFIC EVALUATION HIERARCHY:
- Level 1: Image-Space Geometric Consistency (reprojection errors, track lengths, epipolar inliers).
- Level 2: Relative / Scale-Aligned 3D Geometry (pointmaps, relative mesh surfaces, Sim(3)/SE(3) Chamfer).
- Level 3: Absolute Metric / Geospatial Accuracy (physical meters, WGS84 coordinates, Ground Control Points).

MONOCULAR SCALE AMBIGUITY:
- Monocular visual reconstructions possess an inherent 7-DoF similarity gauge ambiguity (Sim(3)).
- Scale-alignment (e.g. Procrustes / Umeyama) is an evaluation diagnostic, NOT proof of intrinsic metric correctness.
- Absolute metric accuracy requires independent metric reference (calibrated stereo baseline, survey GCPs).

GROUND TRUTH PROVENANCE:
- Onboard GNSS/navigation telemetry is a sensor measurement (TRAJECTORY_PROXY).
- Absolute Trajectory Error (ATE RMSE) is strictly GROUND_TRUTH_DEPENDENT and requires an independent certified reference.

COORDINATE FRAME CONVENTIONS:
1. Pixel Raster Space (u, v):
   - Origin (0, 0) at top-left pixel center / corner.
   - +u: image column axis (Right) in [0, width - 1].
   - +v: image row axis (Down) in [0, height - 1].
2. Normalized Camera Space (x_n, y_n, 1):
   - x_n = (u - cx) / fx, y_n = (v - cy) / fy.
   - Dimensionless ray direction vector.
3. Camera Optical Coordinate Frame (X_c, Y_c, Z_c):
   - Right-handed OpenCV optical standard:
     +X_c: Right
     +Y_c: Down
     +Z_c: Forward (Principal optical axis)
4. Local Tangent World Frame (E, N, U):
   - Metric local East-North-Up (ENU) Cartesian frame:
     +E: East (m)
     +N: North (m)
     +U: Up (m, perpendicular to WGS84 ellipsoid surface)
5. Extrinsics Transformation:
   - Camera pose transforms world points to camera coordinates:
     X_c = R_cw * X_w + t_cw
   - Camera optical center in world frame:
     C_w = -R_cw^T * t_cw
"""

import json
import math
import numpy as np
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Union


class EvaluationLevel(str, Enum):
    """Explicit three-tier scientific evaluation hierarchy."""
    LEVEL_1_IMAGE_SPACE_CONSISTENCY = "LEVEL_1_IMAGE_SPACE_CONSISTENCY"
    LEVEL_2_RELATIVE_SCALE_ALIGNED_3D = "LEVEL_2_RELATIVE_SCALE_ALIGNED_3D"
    LEVEL_3_ABSOLUTE_METRIC_GEOSPATIAL = "LEVEL_3_ABSOLUTE_METRIC_GEOSPATIAL"


class PipelineStageStatus(str, Enum):
    """Explicit execution status per reconstruction stage."""
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    EMPTY = "EMPTY"


class MeasurementType(str, Enum):
    """Explicit scientific classification of data values and diagnostic metrics."""
    DIRECTLY_OBSERVED = "DIRECTLY_OBSERVED"       # Directly measured in sensor raster (e.g. keypoint pixel coordinates)
    ESTIMATED = "ESTIMATED"                       # Solved via numerical estimation / optimization (e.g. relative pose, 3D points)
    TRAJECTORY_PROXY = "TRAJECTORY_PROXY"         # Indirect surrogate for uncalibrated physical state (e.g. GNSS telemetry baseline)
    PROXY = "PROXY"                               # Backward-compatibility alias for TRAJECTORY_PROXY
    HEURISTIC = "HEURISTIC"                       # Engineering rule or heuristic threshold (e.g. RANSAC inlier threshold)
    GROUND_TRUTH_DEPENDENT = "GROUND_TRUTH_DEPENDENT"  # Requires external ground truth benchmark to evaluate


class CompletenessMetricType(str, Enum):
    """Explicit scientific classification of 3D geometric completeness calculation."""
    REFERENCE_POINT_COMPLETENESS = "REFERENCE_POINT_COMPLETENESS"  # Fraction of sampled reference points within distance tolerance tau (assumes uniform surface point density)
    SURFACE_AREA_COMPLETENESS = "SURFACE_AREA_COMPLETENESS"        # Area-weighted surface coverage on continuous mesh/CAD surface within tolerance tau


class DistortionModel(str, Enum):
    """Explicit camera lens distortion model classification."""
    NONE_RECTIFIED = "NONE_RECTIFIED"                                  # Image raster already undistorted/rectified
    BROWN_CONRADY_RADIAL_TANGENTIAL = "BROWN_CONRADY_RADIAL_TANGENTIAL"  # Standard pinhole (k1, k2, p1, p2, k3)
    FISHEYE_EQUIDISTANT = "FISHEYE_EQUIDISTANT"                        # Wide-angle equidistant (k1, k2, k3, k4)
    UNSUPPORTED_UNKNOWN = "UNSUPPORTED_UNKNOWN"                        # Unrecognized or unmodeled distortion


class DistortionStatus(str, Enum):
    """Status of lens distortion calibration."""
    RECTIFIED_ZERO_DISTORTION = "RECTIFIED_ZERO_DISTORTION"
    EXPLICIT_MODEL_PRESENT = "EXPLICIT_MODEL_PRESENT"
    CALIBRATION_UNAVAILABLE = "CALIBRATION_UNAVAILABLE"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"


class GaugeFixingPolicy(str, Enum):
    """Policy for fixing 7-DoF similarity gauge freedom in monocular bundle adjustment.
    
    Fixes numerical coordinates in relative reconstruction space; does NOT establish metric scale.
    """
    FIX_FIRST_CAMERA_AND_UNIT_BASELINE = "FIX_FIRST_CAMERA_AND_UNIT_BASELINE"  # Fix Camera 0 to origin [I | 0] and set ||t_10|| = 1.0
    FIX_TWO_CAMERAS = "FIX_TWO_CAMERAS"                                        # Fix 2 camera optical centers in relative frame
    GAUGE_PRIOR_COVARIANCE = "GAUGE_PRIOR_COVARIANCE"                          # Soft gauge prior via covariance
    UNCONSTRAINED_FREE_GAUGE = "UNCONSTRAINED_FREE_GAUGE"                      # Gauge unconstrained (requires Levenberg-Marquardt damping)


class GeometryFailureReason(str, Enum):
    """Explicit, non-silent failure taxonomy for classical geometry pipeline."""
    INSUFFICIENT_FEATURES = "INSUFFICIENT_FEATURES"
    INSUFFICIENT_MATCHES = "INSUFFICIENT_MATCHES"
    GEOMETRIC_VERIFICATION_FAILED = "GEOMETRIC_VERIFICATION_FAILED"
    DEGENERATE_GEOMETRY = "DEGENERATE_GEOMETRY"
    PURE_ROTATION_RISK = "PURE_ROTATION_RISK"
    WEAK_BASELINE = "WEAK_BASELINE"
    CALIBRATION_UNAVAILABLE = "CALIBRATION_UNAVAILABLE"
    CAMERA_REGISTRATION_FAILED = "CAMERA_REGISTRATION_FAILED"
    INSUFFICIENT_2D_3D_CORRESPONDENCES = "INSUFFICIENT_2D_3D_CORRESPONDENCES"
    TRACK_CONFLICT = "TRACK_CONFLICT"
    INVALID_CAMERA_POSE = "INVALID_CAMERA_POSE"
    RECONSTRUCTION_STALLED = "RECONSTRUCTION_STALLED"
    TRIANGULATION_FAILED = "TRIANGULATION_FAILED"
    CHEIRALITY_VIOLATION = "CHEIRALITY_VIOLATION"
    SPARSE_RECONSTRUCTION_INSUFFICIENT = "SPARSE_RECONSTRUCTION_INSUFFICIENT"
    BUNDLE_ADJUSTMENT_DIVERGED = "BUNDLE_ADJUSTMENT_DIVERGED"
    MVS_DEPTH_ESTIMATION_FAILED = "MVS_DEPTH_ESTIMATION_FAILED"


@dataclass(frozen=True)
class GeometryThresholdConfig:
    """Configurable heuristic engineering defaults (HEURISTIC_DEFAULT) for geometry checks.
    
    These values are empirical operational parameters, NOT universal mathematical truths.
    """
    min_feature_count: int = 100                      # HEURISTIC_DEFAULT
    min_candidate_matches: int = 30                   # HEURISTIC_DEFAULT
    min_inlier_ratio: float = 0.20                    # HEURISTIC_DEFAULT
    weak_baseline_parallax_deg: float = 1.0           # HEURISTIC_DEFAULT
    min_sparse_points: int = 50                       # HEURISTIC_DEFAULT
    min_registered_cameras: int = 3                   # HEURISTIC_DEFAULT
    max_reprojection_rmse_px: float = 2.0             # HEURISTIC_DEFAULT
    config_version: str = "GeometryThresholds_v1.0"


@dataclass(frozen=True)
class TrajectoryEvaluationProvenance:
    """Explicit provenance record for trajectory and metric accuracy evaluation."""
    evaluation_level: EvaluationLevel = EvaluationLevel.LEVEL_3_ABSOLUTE_METRIC_GEOSPATIAL
    reference_trajectory_source: str = "SURVEYED_GROUND_TRUTH"  # Must NOT be unverified sensor telemetry
    is_ground_truth_certified: bool = False
    measurement_type: MeasurementType = MeasurementType.GROUND_TRUTH_DEPENDENT
    notes: str = ""


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters and Brown-Conrady distortion coefficients.
    
    Conventions:
    - fx, fy: Focal lengths in pixels.
    - cx, cy: Principal point offset from top-left pixel (0, 0) in pixels.
    - width, height: Sensor raster dimensions in pixels.
    - k1, k2, p1, p2, k3: Standard radial and tangential distortion coefficients.
    - distortion_model: Explicit distortion classification.
    - distortion_status: Calibration readiness status.
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    distortion_model: DistortionModel = DistortionModel.NONE_RECTIFIED
    distortion_status: DistortionStatus = DistortionStatus.RECTIFIED_ZERO_DISTORTION

    @property
    def matrix_3x3(self) -> List[List[float]]:
        return [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ]

    @property
    def is_calibrated(self) -> bool:
        """True if focal length and dimensions are positive and distortion status is ready."""
        return (
            self.fx > 0 and self.fy > 0 and self.width > 0 and self.height > 0
            and self.distortion_status != DistortionStatus.CALIBRATION_UNAVAILABLE
            and self.distortion_status != DistortionStatus.UNSUPPORTED_MODEL
        )


@dataclass
class ExtrinsicPose:
    """Rigid 6-DoF transformation representing camera pose in world/model space.
    
    Conventions:
    - rotation_matrix: 3x3 SO(3) matrix transforming vectors from camera optical frame to world frame.
    - translation_vector: 3D vector [X, Y, Z] representing camera optical center in world frame.
    - coordinate_convention: "opencv_optical" (Right: +X, Down: +Y, Forward: +Z).
    - scale_factor: Metric scale ratio (1.0 = true metric scale in meters).
    - is_metric: True if translation_vector is in true physical meters.
    """
    rotation_matrix: List[List[float]] = field(default_factory=lambda: [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    translation_vector: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    frame_index: int = 0
    timestamp_seconds: float = 0.0
    coordinate_convention: str = "opencv_optical"
    scale_factor: float = 1.0
    is_metric: bool = False


@dataclass
class PointmapData:
    """Pixel-aligned 3D pointmap output from foundation AI models (e.g. DUSt3R/VGGT).
    
    Conventions:
    - Shape: (height, width, 3) where each pixel (v, u) stores 3D point coordinates (X, Y, Z).
    - Coordinate Frame: Expressed in reference frame camera optical coordinates (+X right, +Y down, +Z forward).
    - Confidence Map: (height, width) scalar field in range [0.0, 1.0].
    """
    reference_frame_idx: int
    target_frame_idx: int
    width: int
    height: int
    coordinate_frame: str = "camera_optical"
    has_pointmap: bool = False
    has_confidence_map: bool = False
    mean_confidence: float = 0.0


@dataclass(frozen=True)
class FeatureKeypoint:
    """A detected 2D interest point in pixel coordinates."""
    x: float                     # Horizontal pixel coordinate u [0, width - 1]
    y: float                     # Vertical pixel coordinate v [0, height - 1]
    octave: int = 0              # Scale octave level
    response: float = 0.0        # Corner / feature response strength
    angle: float = 0.0           # Feature orientation angle in degrees [-180, 180]
    measurement_type: MeasurementType = MeasurementType.DIRECTLY_OBSERVED


@dataclass
class FeatureCorrespondences:
    """Pair-wise 2D feature correspondences between two frames."""
    frame_a_id: str
    frame_b_id: str
    points_a: np.ndarray         # Shape (N, 2) float64 pixel coordinates in frame A
    points_b: np.ndarray         # Shape (N, 2) float64 pixel coordinates in frame B
    descriptor_distances: np.ndarray  # Shape (N,) float64 descriptor matching distances
    match_count: int
    descriptor_type: str = "ORB_256BIT"
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TwoViewGeometryResult:
    """Result of two-view geometric verification and relative pose estimation.
    
    Epipolar Models:
    - fundamental_matrix: Operates on uncalibrated pixel raster coordinates: x_2_px^T * F * x_1_px = 0.
    - essential_matrix: Operates on calibrated normalized coordinates: x_2_norm^T * E * x_1_norm = 0.
    """
    frame_a_id: str
    frame_b_id: str
    fundamental_matrix: Optional[np.ndarray] = None   # Shape (3, 3) rank-2 matrix in pixel coordinates
    essential_matrix: Optional[np.ndarray] = None     # Shape (3, 3) essential matrix E = [t]_x R (requires calibration)
    relative_rotation: Optional[np.ndarray] = None    # Shape (3, 3) SO(3) rotation from frame A to frame B
    relative_translation: Optional[np.ndarray] = None # Shape (3,) unit translation vector (direction only)
    has_calibrated_intrinsics: bool = True            # False if intrinsics were unavailable, restricting to F only
    input_correspondence_count: int = 0
    inlier_mask: Optional[np.ndarray] = None          # Shape (N,) boolean mask
    inlier_count: int = 0
    inlier_ratio: float = 0.0                         # inlier_count / total_matches
    mean_reprojection_error_px: float = 0.0
    mean_epipolar_residual: float = 0.0
    median_epipolar_residual: float = 0.0
    cheirality_passed_count: int = 0                  # Points triangulating with positive depth in both cameras
    cheirality_ratio: float = 0.0
    median_parallax_deg: float = 0.0
    is_degenerate: bool = False
    model_used: str = "UNSPECIFIED"
    f_status: str = "NOT_ATTEMPTED"
    e_status: str = "NOT_ATTEMPTED"
    scale_status: str = "SCALE_AMBIGUOUS"
    translation_magnitude_status: str = "UNOBSERVABLE"
    relative_rotation_measurement: MeasurementType = MeasurementType.ESTIMATED
    relative_translation_measurement: MeasurementType = MeasurementType.ESTIMATED
    failure_reason: Optional[GeometryFailureReason] = None
    diagnostics: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dict."""
        return {
            "frame_a_id": self.frame_a_id,
            "frame_b_id": self.frame_b_id,
            "has_fundamental_matrix": self.fundamental_matrix is not None,
            "has_essential_matrix": self.essential_matrix is not None,
            "has_calibrated_intrinsics": self.has_calibrated_intrinsics,
            "model_used": self.model_used,
            "input_correspondence_count": self.input_correspondence_count,
            "inlier_count": self.inlier_count,
            "inlier_ratio": self.inlier_ratio,
            "mean_reprojection_error_px": self.mean_reprojection_error_px,
            "mean_epipolar_residual": self.mean_epipolar_residual,
            "median_epipolar_residual": self.median_epipolar_residual,
            "cheirality_passed_count": self.cheirality_passed_count,
            "cheirality_ratio": self.cheirality_ratio,
            "median_parallax_deg": self.median_parallax_deg,
            "is_degenerate": self.is_degenerate,
            "scale_status": self.scale_status,
            "translation_magnitude_status": self.translation_magnitude_status,
            "f_status": self.f_status,
            "e_status": self.e_status,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "diagnostics": self.diagnostics,
            "provenance": self.provenance,
        }


@dataclass
class TriangulatedTrack:
    """A reconstructed 3D landmark track observed across multiple views."""
    track_id: int
    world_point: np.ndarray                           # Shape (3,) [E, N, U] or arbitrary local coordinate (m)
    observations: Dict[str, Tuple[float, float]]      # frame_id -> (u, v) pixel observation
    reprojection_errors: Dict[str, float]             # frame_id -> reprojection error (px)
    cheirality_valid: bool = True                     # Positive optical depth in all observing views
    triangulation_angle_deg: float = 0.0              # Maximum parallax angle subtended by rays
    measurement_type: MeasurementType = MeasurementType.ESTIMATED


@dataclass
class SparseReconstructionResult:
    """Complete output of incremental Structure-from-Motion and Bundle Adjustment."""
    camera_poses: Dict[str, ExtrinsicPose]            # frame_id -> calibrated / estimated camera pose
    intrinsics: Dict[str, CameraIntrinsics]           # frame_id -> camera intrinsics
    points3d: Dict[int, TriangulatedTrack]            # track_id -> 3D point landmark track
    mean_reprojection_rmse_px: float
    percentile_90_reprojection_error_px: float
    total_registered_cameras: int
    total_triangulated_points: int
    mean_track_length: float
    is_metric_scale: bool = False                     # True if absolute GNSS metric scale was aligned
    evaluation_level: EvaluationLevel = EvaluationLevel.LEVEL_1_IMAGE_SPACE_CONSISTENCY
    has_monocular_scale_ambiguity: bool = True        # Scale requires external calibration
    gauge_policy: GaugeFixingPolicy = GaugeFixingPolicy.FIX_FIRST_CAMERA_AND_UNIT_BASELINE
    registered_frame_ids: List[str] = field(default_factory=list)
    unregistered_frame_ids: List[str] = field(default_factory=list)
    failed_frame_ids: List[str] = field(default_factory=list)
    camera_centers: Dict[str, List[float]] = field(default_factory=dict)
    status: PipelineStageStatus = PipelineStageStatus.SUCCESS
    failure_reason: Optional[GeometryFailureReason] = None
    diagnostics: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize reconstruction summary to dictionary."""
        return {
            "total_registered_cameras": self.total_registered_cameras,
            "total_triangulated_points": self.total_triangulated_points,
            "mean_reprojection_rmse_px": self.mean_reprojection_rmse_px,
            "percentile_90_reprojection_error_px": self.percentile_90_reprojection_error_px,
            "mean_track_length": self.mean_track_length,
            "is_metric_scale": self.is_metric_scale,
            "evaluation_level": self.evaluation_level.value,
            "has_monocular_scale_ambiguity": self.has_monocular_scale_ambiguity,
            "gauge_policy": self.gauge_policy.value,
            "status": self.status.value,
            "registered_frame_ids": self.registered_frame_ids,
            "unregistered_frame_ids": self.unregistered_frame_ids,
            "failed_frame_ids": self.failed_frame_ids,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "diagnostics": self.diagnostics,
            "provenance": self.provenance,
        }


@dataclass
class DenseMVSInput:
    """Input contract for Multi-View Stereo dense depth estimation."""
    sparse_reconstruction: SparseReconstructionResult
    selected_frame_ids: List[str]
    max_depth_meters: float = 200.0                   # HEURISTIC_DEFAULT
    min_depth_meters: float = 1.0                     # HEURISTIC_DEFAULT
    patch_match_window_size: int = 7                  # HEURISTIC_DEFAULT


@dataclass
class DenseMVSOutput:
    """Output contract for Multi-View Stereo dense point cloud."""
    sparse_sfm_status: PipelineStageStatus            # SfM status
    camera_registration_status: PipelineStageStatus   # Camera registration status
    dense_depth_status: PipelineStageStatus           # Depth estimation status
    point_cloud_fusion_status: PipelineStageStatus    # Point cloud fusion status
    depth_maps: Dict[str, np.ndarray]                 # frame_id -> (H, W) float32 depth map (m)
    confidence_maps: Dict[str, np.ndarray]            # frame_id -> (H, W) float32 confidence map [0, 1]
    fused_point_count: int                            # Descriptive density statistic (NOT quality proxy)
    reference_point_completeness_ratio: Optional[float] = None  # GROUND_TRUTH_DEPENDENT reference sample coverage in [0, 1]
    completeness_metric_type: CompletenessMetricType = CompletenessMetricType.REFERENCE_POINT_COMPLETENESS
    dense_cloud_file_path: Optional[str] = None
    evaluation_level: EvaluationLevel = EvaluationLevel.LEVEL_2_RELATIVE_SCALE_ALIGNED_3D
    has_monocular_scale_ambiguity: bool = True
    failure_reason: Optional[GeometryFailureReason] = None
    diagnostics: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert MVS output metadata to dictionary."""
        return {
            "sparse_sfm_status": self.sparse_sfm_status.value,
            "camera_registration_status": self.camera_registration_status.value,
            "dense_depth_status": self.dense_depth_status.value,
            "point_cloud_fusion_status": self.point_cloud_fusion_status.value,
            "fused_point_count": self.fused_point_count,
            "reference_point_completeness_ratio": self.reference_point_completeness_ratio,
            "completeness_metric_type": self.completeness_metric_type.value,
            "evaluation_level": self.evaluation_level.value,
            "has_monocular_scale_ambiguity": self.has_monocular_scale_ambiguity,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "diagnostics": self.diagnostics,
            "provenance": self.provenance,
        }


# ==============================================================================
# MATHEMATICAL VERIFICATION FUNCTIONS (CONTRACT HELPERS)
# ==============================================================================

class GeometryMathContracts:
    """Deterministic mathematical verification helper routines for geometry contracts."""

    @staticmethod
    def project_point(
        point_world: np.ndarray,
        r_matrix: np.ndarray,
        t_vec: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> Tuple[float, float, float]:
        """Project 3D world point into pinhole camera coordinates (u, v, Z_c).
        
        X_c = R * X_w + t
        u = fx * (X_c / Z_c) + cx
        v = fy * (Y_c / Z_c) + cy
        
        Returns:
            (u, v, Z_c) where Z_c is optical depth along camera forward axis.
        """
        pt_w = np.asarray(point_world, dtype=np.float64).reshape((3, 1))
        r_mat = np.asarray(r_matrix, dtype=np.float64).reshape((3, 3))
        t_arr = np.asarray(t_vec, dtype=np.float64).reshape((3, 1))

        pt_c = r_mat @ pt_w + t_arr
        xc, yc, zc = float(pt_c[0, 0]), float(pt_c[1, 0]), float(pt_c[2, 0])

        if abs(zc) < 1e-9:
            return (float("nan"), float("nan"), zc)

        u = fx * (xc / zc) + cx
        v = fy * (yc / zc) + cy
        return (u, v, zc)

    @staticmethod
    def compute_reprojection_error(
        point_world: np.ndarray,
        observed_pixel: Tuple[float, float],
        r_matrix: np.ndarray,
        t_vec: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> float:
        """Compute Euclidean reprojection error in pixel raster space."""
        u_proj, v_proj, zc = GeometryMathContracts.project_point(
            point_world, r_matrix, t_vec, fx, fy, cx, cy
        )
        if math.isnan(u_proj) or zc <= 0:
            return float("inf")
        du = u_proj - observed_pixel[0]
        dv = v_proj - observed_pixel[1]
        return math.sqrt(du * du + dv * dv)

    @staticmethod
    def verify_essential_matrix_constraint(
        point_a_norm: Tuple[float, float],
        point_b_norm: Tuple[float, float],
        essential_matrix: np.ndarray,
        tolerance: float = 1e-4,
    ) -> float:
        """Verify coplanarity algebraic error in normalized camera space: x_b_norm^T * E * x_a_norm ≈ 0."""
        xa = np.array([point_a_norm[0], point_a_norm[1], 1.0], dtype=np.float64).reshape((3, 1))
        xb = np.array([point_b_norm[0], point_b_norm[1], 1.0], dtype=np.float64).reshape((3, 1))
        e_mat = np.asarray(essential_matrix, dtype=np.float64).reshape((3, 3))
        res = xb.T @ e_mat @ xa
        val = float(res[0, 0])
        return abs(val)

    @staticmethod
    def verify_fundamental_matrix_constraint(
        point_a_pixel: Tuple[float, float],
        point_b_pixel: Tuple[float, float],
        fundamental_matrix: np.ndarray,
        tolerance: float = 1e-4,
    ) -> float:
        """Verify epipolar algebraic error in uncalibrated pixel raster space: x_b_px^T * F * x_a_px ≈ 0."""
        xa = np.array([point_a_pixel[0], point_a_pixel[1], 1.0], dtype=np.float64).reshape((3, 1))
        xb = np.array([point_b_pixel[0], point_b_pixel[1], 1.0], dtype=np.float64).reshape((3, 1))
        f_mat = np.asarray(fundamental_matrix, dtype=np.float64).reshape((3, 3))
        res = xb.T @ f_mat @ xa
        val = float(res[0, 0])
        return abs(val)

    @staticmethod
    def verify_epipolar_constraint(
        point_a_norm: Tuple[float, float],
        point_b_norm: Tuple[float, float],
        essential_matrix: np.ndarray,
        tolerance: float = 1e-4,
    ) -> float:
        """Backward-compatible alias for verify_essential_matrix_constraint."""
        return GeometryMathContracts.verify_essential_matrix_constraint(
            point_a_norm, point_b_norm, essential_matrix, tolerance
        )

    @staticmethod
    def check_cheirality(point_world: np.ndarray, r_matrix: np.ndarray, t_vec: np.ndarray) -> bool:
        """Verify that triangulated 3D point lies strictly in front of the camera optical plane (Z_c > 0)."""
        pt_w = np.asarray(point_world, dtype=np.float64).reshape((3, 1))
        r_mat = np.asarray(r_matrix, dtype=np.float64).reshape((3, 3))
        t_arr = np.asarray(t_vec, dtype=np.float64).reshape((3, 1))
        pt_c = r_mat @ pt_w + t_arr
        return float(pt_c[2, 0]) > 1e-6

    @staticmethod
    def compute_reference_point_completeness(
        reconstructed_points: np.ndarray,
        reference_sample_points: np.ndarray,
        distance_tolerance: float = 0.05,
    ) -> float:
        """Compute reference point completeness ratio: fraction of reference sample points with a reconstructed neighbor within tolerance tau.
        
        Completeness = |{p in S_ref : min_{q in S_rec} ||p - q|| <= tau}| / |S_ref|
        
        ASSUMPTION: Valid as a surface coverage proxy ONLY when reference_sample_points
        are sampled uniformly across the ground-truth surface.
        """
        rec = np.asarray(reconstructed_points, dtype=np.float64)
        ref = np.asarray(reference_sample_points, dtype=np.float64)

        if len(ref) == 0:
            return 1.0
        if len(rec) == 0:
            return 0.0

        covered_count = 0
        for p in ref:
            dists = np.linalg.norm(rec - p, axis=1)
            if np.min(dists) <= distance_tolerance:
                covered_count += 1

        return float(covered_count / len(ref))

    @staticmethod
    def compute_surface_completeness(
        reconstructed_points: np.ndarray,
        reference_surface_points: np.ndarray,
        distance_tolerance: float = 0.05,
    ) -> float:
        """Backward-compatible alias for compute_reference_point_completeness."""
        return GeometryMathContracts.compute_reference_point_completeness(
            reconstructed_points, reference_surface_points, distance_tolerance
        )
