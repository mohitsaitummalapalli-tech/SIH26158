"""Phase 3E.2: Dense 3D Point Generation & Geometric Validation.

Converts validated dense stereo depth observations (Phase 3E.1) into dense 3D points
in world/reconstruction coordinates while preserving exact camera conventions, poses,
source pixel identities, disparities, confidences, visibility states, and provenance.

KEY SCIENTIFIC & MATHEMATICAL INVARIANTS:
1. Camera Coordinate Convention:
   X_c = R_cw * X_w + t_cw
   C_w = -R_cw^T * t_cw
   t_cw = -R_cw * C_w
2. World Coordinate Transformation:
   X_w = R_cw^T * (X_c - t_cw) = C_w + R_cw^T * X_c
   (translation_vector is t_cw, NOT camera center C_w).
3. Rectification Rotation Consistency:
   Stereo depth Z_rect is optical depth along the RECTIFIED camera principal axis.
   Rectified camera coordinates:
       X_rect = [ Z_rect * (u_rect - cx_rect) / fx_rect ]
                [ Z_rect * (v_rect - cy_rect) / fy_rect ]
                [ Z_rect                                ]
   To recover original camera coordinates, the rectification rotation R1 must be inverted:
       X_c_orig = R1^T * X_rect
   Then to world coordinates:
       X_w = R_cw^T * (X_c_orig - t_cw) = R_cw^T * (R1^T * X_rect - t_cw)
4. Strict Geometric Validation:
   Every generated point undergoes rigorous multi-criteria validation:
   - Finite coordinate check (no NaN, +/-Inf)
   - Cheirality constraint (Z_c > 0 in original and rectified frames)
   - Depth range bounds [min_depth, max_depth] in RECONSTRUCTION_UNITS
   - Reprojection consistency check against originating pixel raster
   - Confidence thresholding
5. Reconstruction Units Only:
   All output coordinates are in RECONSTRUCTION_UNITS with is_metric_scale=False.
   Absolute metric scale is NOT claimed without independent certified ground truth.
6. Heuristic Confidence Preservation:
   Phase 3E.1 confidence scores are preserved as HEURISTIC_SCORE without claiming
   Bayesian probability, posterior distribution, or physical measurement uncertainty.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Optional, List, Dict, Any, Tuple, Set

import numpy as np

from src.geometry.contracts import (
    CameraIntrinsics,
    ExtrinsicPose,
)

from src.geometry.mvs import (
    DepthUnit,
    PointVisibilityState,
    PointValidationStatus,
    DepthMap,
    DepthConfidenceMap,
    DensePointObservation,
    DensePointCloud,
)

from src.geometry.dense_stereo import (
    DenseStereoResult,
    StereoRectificationResult,
)


class PointRejectionReason(str, Enum):
    """Explicit rejection taxonomy for dense 3D point generation."""
    INVALID_DEPTH_VALUE = "INVALID_DEPTH_VALUE"
    NON_POSITIVE_DEPTH = "NON_POSITIVE_DEPTH"
    OUT_OF_DEPTH_BOUNDS = "OUT_OF_DEPTH_BOUNDS"
    NON_FINITE_COORDINATES = "NON_FINITE_COORDINATES"
    CHEIRALITY_VIOLATION = "CHEIRALITY_VIOLATION"
    REPROJECTION_ERROR_EXCEEDED = "REPROJECTION_ERROR_EXCEEDED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    OCCLUDED_OR_INCONSISTENT = "OCCLUDED_OR_INCONSISTENT"
    INVALID_DISPARITY = "INVALID_DISPARITY"
    MASKED_OUT = "MASKED_OUT"


@dataclass(frozen=True)
class DensePointGeneratorConfig:
    """Configurable engineering defaults (HEURISTIC_DEFAULT) for dense 3D point generation."""
    min_depth_units: float = 0.5                        # HEURISTIC_DEFAULT: Minimum optical depth in reconstruction units
    max_depth_units: float = 100.0                      # HEURISTIC_DEFAULT: Maximum optical depth in reconstruction units
    min_confidence: float = 0.20                        # HEURISTIC_DEFAULT: Minimum stereo confidence [0, 1]
    max_reprojection_error_px: float = 2.0              # HEURISTIC_DEFAULT: Maximum allowable reprojection error in pixels
    reprojection_check_enabled: bool = True             # HEURISTIC_DEFAULT: Enable reprojection consistency validation
    require_positive_disparity: bool = True             # HEURISTIC_DEFAULT: Require subpixel disparity > 0
    require_cheirality: bool = True                     # HEURISTIC_DEFAULT: Require points strictly in front of camera
    config_version: str = "DensePointGeneratorConfig_v1.0"


@dataclass
class ValidatedDensePoint:
    """Typed container for a single validated dense 3D point observation."""
    world_point: np.ndarray                             # (3,) float64 in RECONSTRUCTION_UNITS
    camera_point_rect: np.ndarray                       # (3,) float64 in rectified camera frame
    camera_point_orig: np.ndarray                       # (3,) float64 in original unrectified camera frame
    reference_frame_id: str
    source_frame_id: str
    pixel_coord_rect: Tuple[float, float]               # (u, v) continuous coordinates in rectified raster
    depth: float                                        # Optical depth Z_rect in reconstruction units
    disparity: float                                    # Subpixel disparity in pixels
    stereo_confidence: float                            # Phase 3E.1 confidence score [0, 1], HEURISTIC_SCORE
    reprojection_error_px: Optional[float]              # Image-space reprojection error in pixels
    visibility_state: PointVisibilityState
    validation_status: PointValidationStatus
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_observation(self) -> DensePointObservation:
        """Convert to Phase 3E.0 DensePointObservation contract."""
        return DensePointObservation(
            world_point=self.world_point.copy(),
            reference_frame_id=self.reference_frame_id,
            pixel_coord=self.pixel_coord_rect,
            depth=self.depth,
            confidence=self.stereo_confidence,
            visibility_state=self.visibility_state,
            validation_status=self.validation_status,
            source_view_support_count=2,
            provenance=self.provenance.copy(),
        )


@dataclass
class DensePointGenerationResult:
    """Typed container for dense point generation and geometric validation output."""
    reference_frame_id: str
    source_frame_id: str
    total_pixels_evaluated: int
    valid_points_count: int
    rejected_points_count: int
    rejection_breakdown: Dict[str, int]
    validated_points: List[ValidatedDensePoint]
    observations: List[DensePointObservation]
    point_cloud: Optional[DensePointCloud]
    mean_reprojection_error_px: Optional[float]
    mean_confidence: float
    provenance: Dict[str, Any] = field(default_factory=dict)


class DensePointBackprojector:
    """Mathematically rigorous 3D backprojector from rectified stereo raster to world coordinates."""

    @staticmethod
    def backproject_rectified_pixel(
        u_rect: float,
        v_rect: float,
        depth_z: float,
        P1: np.ndarray,
        R1: np.ndarray,
        ref_pose: ExtrinsicPose,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
        """Backproject a rectified raster pixel (u_rect, v_rect) with optical depth Z_rect to world space.
        
        Transformation Chain:
            1. Rectified pixel + Z_rect -> Rectified camera space:
               X_rect = [ Z_rect * (u_rect - cx_rect) / fx_rect ]
                        [ Z_rect * (v_rect - cy_rect) / fy_rect ]
                        [ Z_rect                                ]
            2. Rectified camera space -> Original reference camera space:
               X_c_orig = R1^T * X_rect
            3. Original reference camera space -> World space:
               X_w = R_ref^T * (X_c_orig - t_ref)
        
        Returns:
            (X_w, X_c_orig, X_rect, is_valid)
        """
        if not (math.isfinite(depth_z) and depth_z > 1e-6 and math.isfinite(u_rect) and math.isfinite(v_rect)):
            return None, None, None, False

        # Extract rectified camera intrinsics from P1
        rect_fx = float(P1[0, 0])
        rect_fy = float(P1[1, 1])
        rect_cx = float(P1[0, 2])
        rect_cy = float(P1[1, 2])

        if rect_fx <= 1e-6 or rect_fy <= 1e-6:
            return None, None, None, False

        # 1. Camera coordinates in rectified frame
        x_rect = depth_z * (u_rect - rect_cx) / rect_fx
        y_rect = depth_z * (v_rect - rect_cy) / rect_fy
        z_rect = depth_z
        X_rect = np.array([x_rect, y_rect, z_rect], dtype=np.float64)

        # 2. Invert rectification rotation R1 to recover original reference camera coordinates
        #    OpenCV stereoRectify defines: X_rect = R1 * X_c_orig ==> X_c_orig = R1^T * X_rect
        X_c_orig = R1.T @ X_rect

        # 3. Transform original camera coordinates to world coordinates
        #    X_c_orig = R_ref * X_w + t_ref ==> X_w = R_ref^T * (X_c_orig - t_ref)
        R_ref = np.array(ref_pose.rotation_matrix, dtype=np.float64)
        t_ref = np.array(ref_pose.translation_vector, dtype=np.float64)
        X_w = R_ref.T @ (X_c_orig - t_ref)

        if not np.all(np.isfinite(X_w)):
            return None, None, None, False

        return X_w, X_c_orig, X_rect, True

    @staticmethod
    def project_world_to_rectified_pixel(
        X_w: np.ndarray,
        P1: np.ndarray,
        R1: np.ndarray,
        ref_pose: ExtrinsicPose,
    ) -> Tuple[Optional[Tuple[float, float]], float, bool]:
        """Forward-project 3D world point to rectified reference raster (u_rect, v_rect).
        
        Returns:
            (pixel_coord, rectified_optical_depth, is_valid)
        """
        R_ref = np.array(ref_pose.rotation_matrix, dtype=np.float64)
        t_ref = np.array(ref_pose.translation_vector, dtype=np.float64)

        # World -> Original camera
        X_c_orig = R_ref @ X_w + t_ref
        if float(X_c_orig[2]) <= 1e-6:
            return None, float(X_c_orig[2]), False

        # Original camera -> Rectified camera
        X_rect = R1 @ X_c_orig
        z_rect = float(X_rect[2])
        if z_rect <= 1e-6:
            return None, z_rect, False

        rect_fx = float(P1[0, 0])
        rect_fy = float(P1[1, 1])
        rect_cx = float(P1[0, 2])
        rect_cy = float(P1[1, 2])

        u_rect = rect_fx * (float(X_rect[0]) / z_rect) + rect_cx
        v_rect = rect_fy * (float(X_rect[1]) / z_rect) + rect_cy

        if not (math.isfinite(u_rect) and math.isfinite(v_rect)):
            return None, z_rect, False

        return (u_rect, v_rect), z_rect, True


class DensePointGeometricValidator:
    """Validates 3D points against physical cheirality, depth bounds, and reprojection consistency."""

    @staticmethod
    def validate(
        X_w: np.ndarray,
        X_c_orig: np.ndarray,
        X_rect: np.ndarray,
        origin_u_rect: float,
        origin_v_rect: float,
        depth_z: float,
        confidence: float,
        visibility_state_in: PointVisibilityState,
        P1: np.ndarray,
        R1: np.ndarray,
        ref_pose: ExtrinsicPose,
        config: DensePointGeneratorConfig,
    ) -> Tuple[PointValidationStatus, PointVisibilityState, Optional[float], Optional[PointRejectionReason]]:
        """Validate backprojected point against geometric constraints.
        
        Returns:
            (validation_status, visibility_state, reprojection_error_px, rejection_reason)
        """
        # 1. Finite coordinate check
        if not (np.all(np.isfinite(X_w)) and np.all(np.isfinite(X_c_orig)) and np.all(np.isfinite(X_rect))):
            return PointValidationStatus.REJECTED, PointVisibilityState.INVALID_DEPTH, None, PointRejectionReason.NON_FINITE_COORDINATES

        # 2. Cheirality check (Z > 0 in original and rectified camera frames)
        if config.require_cheirality:
            if X_c_orig[2] <= 1e-6 or X_rect[2] <= 1e-6:
                return PointValidationStatus.REJECTED, PointVisibilityState.INVALID_DEPTH, None, PointRejectionReason.CHEIRALITY_VIOLATION

        # 3. Depth range bounds check
        if depth_z < config.min_depth_units or depth_z > config.max_depth_units:
            return PointValidationStatus.REJECTED, PointVisibilityState.INVALID_DEPTH, None, PointRejectionReason.OUT_OF_DEPTH_BOUNDS

        # 4. Confidence threshold check
        if confidence < config.min_confidence:
            return PointValidationStatus.REJECTED, PointVisibilityState.LOW_CONFIDENCE, None, PointRejectionReason.LOW_CONFIDENCE

        # 5. Existing visibility state check
        if visibility_state_in in (PointVisibilityState.INCONSISTENT, PointVisibilityState.OCCLUDED, PointVisibilityState.INVALID_DEPTH):
            return PointValidationStatus.REJECTED, visibility_state_in, None, PointRejectionReason.OCCLUDED_OR_INCONSISTENT

        # 6. Reprojection consistency check
        reproj_error: Optional[float] = None
        if config.reprojection_check_enabled:
            proj_pixel, proj_z, ok = DensePointBackprojector.project_world_to_rectified_pixel(
                X_w=X_w, P1=P1, R1=R1, ref_pose=ref_pose
            )
            if not ok or proj_pixel is None:
                return PointValidationStatus.REJECTED, PointVisibilityState.INCONSISTENT, None, PointRejectionReason.REPROJECTION_ERROR_EXCEEDED

            du = proj_pixel[0] - origin_u_rect
            dv = proj_pixel[1] - origin_v_rect
            reproj_error = float(math.sqrt(du * du + dv * dv))

            if reproj_error > config.max_reprojection_error_px:
                return PointValidationStatus.REJECTED, PointVisibilityState.INCONSISTENT, reproj_error, PointRejectionReason.REPROJECTION_ERROR_EXCEEDED

        return PointValidationStatus.VALIDATED, PointVisibilityState.VALID, reproj_error, None


class DensePointGenerator:
    """Generates and geometrically validates dense 3D point clouds from Phase 3E.1 stereo results."""

    def __init__(self, config: Optional[DensePointGeneratorConfig] = None):
        self._config = config or DensePointGeneratorConfig()

    @property
    def config(self) -> DensePointGeneratorConfig:
        return self._config

    def generate_points(
        self,
        stereo_result: DenseStereoResult,
        ref_pose: ExtrinsicPose,
        ref_intrinsics: CameraIntrinsics,
    ) -> DensePointGenerationResult:
        """Convert validated dense stereo result into validated 3D point observations."""
        H, W = stereo_result.depth_map.height, stereo_result.depth_map.width
        P1 = stereo_result.rectification.P1
        R1 = stereo_result.rectification.R1

        depth_arr = stereo_result.depth_map.depth_array
        disp_arr = stereo_result.disparity_map
        valid_mask = stereo_result.depth_map.valid_mask
        conf_arr = stereo_result.confidence_map.overall_confidence
        vis_arr = stereo_result.confidence_map.visibility_state

        validated_points: List[ValidatedDensePoint] = []
        observations: List[DensePointObservation] = []
        rejection_breakdown: Dict[str, int] = {reason.value: 0 for reason in PointRejectionReason}

        reproj_errors: List[float] = []
        confidences: List[float] = []

        total_pixels = H * W

        for row in range(H):
            for col in range(W):
                u_rect = float(col)
                v_rect = float(row)

                if not valid_mask[row, col]:
                    rejection_breakdown[PointRejectionReason.MASKED_OUT.value] += 1
                    continue

                z_val = float(depth_arr[row, col])
                disp_val = float(disp_arr[row, col]) if disp_arr is not None else 0.0
                conf_val = float(conf_arr[row, col])

                # Get visibility state enum
                raw_vis = vis_arr[row, col]
                try:
                    vis_state = PointVisibilityState(raw_vis) if isinstance(raw_vis, str) else PointVisibilityState.VALID
                except ValueError:
                    vis_state = PointVisibilityState.VALID

                # Check disparity validity
                if self._config.require_positive_disparity and disp_val <= 0.0:
                    rejection_breakdown[PointRejectionReason.INVALID_DISPARITY.value] += 1
                    continue

                # 1. Backproject pixel + depth to 3D world point
                X_w, X_c_orig, X_rect, ok = DensePointBackprojector.backproject_rectified_pixel(
                    u_rect=u_rect,
                    v_rect=v_rect,
                    depth_z=z_val,
                    P1=P1,
                    R1=R1,
                    ref_pose=ref_pose,
                )

                if not ok or X_w is None or X_c_orig is None or X_rect is None:
                    rejection_breakdown[PointRejectionReason.INVALID_DEPTH_VALUE.value] += 1
                    continue

                # 2. Geometric validation
                status, out_vis, reproj_err, rej_reason = DensePointGeometricValidator.validate(
                    X_w=X_w,
                    X_c_orig=X_c_orig,
                    X_rect=X_rect,
                    origin_u_rect=u_rect,
                    origin_v_rect=v_rect,
                    depth_z=z_val,
                    confidence=conf_val,
                    visibility_state_in=vis_state,
                    P1=P1,
                    R1=R1,
                    ref_pose=ref_pose,
                    config=self._config,
                )

                if rej_reason is not None:
                    rejection_breakdown[rej_reason.value] += 1
                    continue

                point_obj = ValidatedDensePoint(
                    world_point=X_w,
                    camera_point_rect=X_rect,
                    camera_point_orig=X_c_orig,
                    reference_frame_id=stereo_result.reference_frame_id,
                    source_frame_id=stereo_result.source_frame_id,
                    pixel_coord_rect=(u_rect, v_rect),
                    depth=z_val,
                    disparity=disp_val,
                    stereo_confidence=conf_val,
                    reprojection_error_px=reproj_err,
                    visibility_state=out_vis,
                    validation_status=status,
                    provenance={
                        "depth_unit": DepthUnit.RECONSTRUCTION_UNITS.value,
                        "is_metric": False,
                        "rectified_fx": float(P1[0, 0]),
                        "pair_swapped": stereo_result.provenance.get("pair_swapped", False),
                    },
                )

                validated_points.append(point_obj)
                observations.append(point_obj.to_observation())
                confidences.append(conf_val)
                if reproj_err is not None:
                    reproj_errors.append(reproj_err)

        valid_count = len(validated_points)
        rejected_count = total_pixels - valid_count
        mean_reproj = float(np.mean(reproj_errors)) if len(reproj_errors) > 0 else None
        mean_conf = float(np.mean(confidences)) if len(confidences) > 0 else 0.0

        # Construct DensePointCloud container
        point_cloud: Optional[DensePointCloud] = None
        if valid_count > 0:
            pts_matrix = np.array([pt.world_point for pt in validated_points], dtype=np.float64)
            conf_vector = np.array([pt.stereo_confidence for pt in validated_points], dtype=np.float32)
            support_vector = np.full(valid_count, 2, dtype=np.int32)
            src_frame_lists = [[stereo_result.reference_frame_id, stereo_result.source_frame_id] for _ in range(valid_count)]
            vis_list = [pt.visibility_state for pt in validated_points]
            val_list = [pt.validation_status for pt in validated_points]

            point_cloud = DensePointCloud(
                points=pts_matrix,
                confidences=conf_vector,
                support_counts=support_vector,
                source_frame_ids=src_frame_lists,
                visibility_states=vis_list,
                validation_statuses=val_list,
                total_fused_points=valid_count,
                mean_confidence=mean_conf,
                depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
                is_metric_scale=False,
                provenance={
                    "stage": "Phase_3E.2_DensePointGeneration",
                    "reference_frame_id": stereo_result.reference_frame_id,
                    "source_frame_id": stereo_result.source_frame_id,
                    "mean_reprojection_error_px": mean_reproj,
                    "unit": DepthUnit.RECONSTRUCTION_UNITS.value,
                    "is_metric": False,
                },
            )

        return DensePointGenerationResult(
            reference_frame_id=stereo_result.reference_frame_id,
            source_frame_id=stereo_result.source_frame_id,
            total_pixels_evaluated=total_pixels,
            valid_points_count=valid_count,
            rejected_points_count=rejected_count,
            rejection_breakdown=rejection_breakdown,
            validated_points=validated_points,
            observations=observations,
            point_cloud=point_cloud,
            mean_reprojection_error_px=mean_reproj,
            mean_confidence=mean_conf,
            provenance={
                "config_version": self._config.config_version,
                "reprojection_check_enabled": self._config.reprojection_check_enabled,
                "max_reprojection_error_px": self._config.max_reprojection_error_px,
                "min_confidence": self._config.min_confidence,
                "unit": DepthUnit.RECONSTRUCTION_UNITS.value,
                "is_metric": False,
            },
        )
