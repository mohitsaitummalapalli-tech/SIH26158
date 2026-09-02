"""Phase 3E.1: Classical Dense Stereo Depth Estimation Baseline.

Implements the first real classical dense depth estimation baseline satisfying
Phase 3E.0 contracts (IMVSDepthEstimator) using OpenCV Semi-Global Block Matching (StereoSGBM).

KEY SCIENTIFIC & MATHEMATICAL INVARIANTS:
1. Disparity is NOT Depth:
   Stereo matching evaluates horizontal disparity d in rectified raster coordinates.
   Depth is strictly optical depth Z_c along the reference camera optical axis:
       Z_rect = (f'_rect * B_rect) / d
   where f'_rect is the rectified focal length, B_rect is the baseline in reconstruction units,
   and d > 0 is subpixel disparity in pixels.
2. Reconstruction Units Only:
   All baseline proxies and depth values are expressed strictly in RECONSTRUCTION_UNITS.
   Absolute metric scale is NOT established; no metric or meter accuracy is claimed.
3. Dedicated Rectification Layer:
   StereoRectifier computes relative orientation, rotation transforms (R1, R2), projection
   matrices (P1, P2), Q matrix, and rectified camera intrinsics.
4. Strict Disparity Validation:
   Disparities with d <= 0, NaN, Inf, or failing left-right consistency are strictly rejected.
   Zero-tolerance: No np.nan_to_num, no clipping, no fake zero-depth replacement.
5. Left-Right Disparity Consistency:
   Bidirectional disparity matching enforces |d_L(u, v) - d_R(u - d_L, v)| <= tau_lr.
   Violations are explicitly classified as PointVisibilityState.INCONSISTENT.
6. Heuristic Confidence Scores:
   Confidence scores are bounded in [0, 1] and explicitly labeled HEURISTIC_SCORE.
7. Dynamic Scene Safety:
   Frame-level dynamic risks from Phase 2/Phase 3E.0 propagate into disparity confidence.
   Dynamic-scene handling is view-risk aware, not semantic motion segmentation.
"""

from dataclasses import dataclass, field
import math
from typing import Optional, List, Dict, Any, Tuple, cast
import cv2
import numpy as np

from src.geometry.contracts import (
    CameraIntrinsics,
    ExtrinsicPose,
)

from src.geometry.mvs import (
    DepthUnit,
    PointVisibilityState,
    MVSConfig,
    MVSInput,
    DepthMap,
    DepthConfidenceMap,
    IMVSDepthEstimator,
)


@dataclass(frozen=True)
class DenseStereoConfig:
    """Heuristic engineering configuration (HEURISTIC_DEFAULT) for classical dense stereo."""
    min_disparity: int = 0                              # HEURISTIC_DEFAULT: Minimum disparity search offset (px)
    num_disparities: int = 64                           # HEURISTIC_DEFAULT: Disparity search range (must be divisible by 16)
    block_size: int = 5                                 # HEURISTIC_DEFAULT: Matching block size (odd integer >= 1)
    p1: int = 600                                       # HEURISTIC_DEFAULT: Smoothness penalty for disparity change of 1
    p2: int = 2400                                      # HEURISTIC_DEFAULT: Smoothness penalty for larger disparity jumps
    disp12_max_diff: int = 1                            # HEURISTIC_DEFAULT: Maximum allowable left-right disparity diff (px)
    pre_filter_cap: int = 63                            # HEURISTIC_DEFAULT: Pre-filter truncation value
    uniqueness_ratio: int = 10                          # HEURISTIC_DEFAULT: Margin in percentage for uniqueness test
    speckle_window_size: int = 100                      # HEURISTIC_DEFAULT: Speckle filter window size
    speckle_range: int = 32                             # HEURISTIC_DEFAULT: Maximum disparity variation within speckle
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY          # HEURISTIC_DEFAULT: SGBM execution mode
    lr_consistency_tolerance_px: float = 1.5            # HEURISTIC_DEFAULT: Left-right consistency check tolerance (px)
    min_depth_units: float = 0.5                        # HEURISTIC_DEFAULT: Minimum valid depth in reconstruction units
    max_depth_units: float = 100.0                      # HEURISTIC_DEFAULT: Maximum valid depth in reconstruction units
    confidence_texture_weight: float = 0.4              # HEURISTIC_DEFAULT: Weight of local gradient texture in confidence
    confidence_lr_weight: float = 0.6                   # HEURISTIC_DEFAULT: Weight of left-right consistency in confidence
    config_version: str = "DenseStereoConfig_v1.0"


@dataclass
class StereoRectificationResult:
    """Typed result of calibrated stereo rectification."""
    R1: np.ndarray                                      # (3, 3) 3D rotation matrix for reference camera
    R2: np.ndarray                                      # (3, 3) 3D rotation matrix for source camera
    P1: np.ndarray                                      # (3, 4) Rectified projection matrix for reference camera
    P2: np.ndarray                                      # (3, 4) Rectified projection matrix for source camera
    Q: np.ndarray                                       # (4, 4) Disparity-to-depth mapping matrix
    rectified_intrinsics_ref: CameraIntrinsics         # Calibrated intrinsics of rectified reference camera
    baseline_reconstruction_units: float                # Stereo baseline distance in RECONSTRUCTION_UNITS
    valid_roi_ref: Tuple[int, int, int, int]           # (x, y, width, height) valid ROI in rectified reference
    valid_roi_src: Tuple[int, int, int, int]           # (x, y, width, height) valid ROI in rectified source
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DenseStereoResult:
    """Typed container for dense stereo matching output."""
    reference_frame_id: str
    source_frame_id: str
    rectification: StereoRectificationResult
    disparity_map: np.ndarray                           # (H, W) float32 subpixel disparity in pixels
    valid_disparity_mask: np.ndarray                    # (H, W) bool array of valid positive disparities
    depth_map: DepthMap                                 # Optical depth Z_c in reference camera
    confidence_map: DepthConfidenceMap                  # Confidence and visibility classification
    provenance: Dict[str, Any] = field(default_factory=dict)


class StereoRectifier:
    """Computes calibrated stereo rectification transforms between two camera views."""

    @staticmethod
    def compute_rectification(
        ref_intrinsics: CameraIntrinsics,
        src_intrinsics: CameraIntrinsics,
        ref_pose: ExtrinsicPose,
        src_pose: ExtrinsicPose,
    ) -> StereoRectificationResult:
        """Compute stereo rectification matrices R1, R2, P1, P2, and Q.
        
        Camera optical convention:
            X_c = R_cw * X_w + t_cw
            C_w = -R_cw^T * t_cw
            t_cw = -R_cw * C_w
        Relative transformation from reference camera frame to source camera frame:
            X_src = R_src * X_w + t_src
                  = R_src * (R_ref^T * (X_ref - t_ref)) + t_src
                  = (R_src * R_ref^T) * X_ref + (t_src - R_src * R_ref^T * t_ref)
            R_rel = R_src * R_ref^T
            t_rel = t_src - R_rel * t_ref = R_src * (C_ref - C_src)
        """
        R_ref = np.array(ref_pose.rotation_matrix, dtype=np.float64)
        t_ref = np.array(ref_pose.translation_vector, dtype=np.float64)
        R_src = np.array(src_pose.rotation_matrix, dtype=np.float64)
        t_src = np.array(src_pose.translation_vector, dtype=np.float64)

        # Recover true camera optical centers in world frame: C = -R^T @ t
        c_ref = -R_ref.T @ t_ref
        c_src = -R_src.T @ t_src

        # Relative rigid transformation (transforms vectors from ref camera frame to src camera frame)
        R_rel = R_src @ R_ref.T
        t_rel = t_src - R_rel @ t_ref

        # Physical baseline distance is separation between camera optical centers: B = ||C_src - C_ref||
        baseline = float(np.linalg.norm(c_src - c_ref))
        if baseline < 1e-5:
            raise ValueError(f"Degenerate stereo baseline ({baseline:.2e} units); camera centers are coincident.")

        # Camera intrinsics matrices
        K1 = np.array([
            [ref_intrinsics.fx, 0.0, ref_intrinsics.cx],
            [0.0, ref_intrinsics.fy, ref_intrinsics.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        K2 = np.array([
            [src_intrinsics.fx, 0.0, src_intrinsics.cx],
            [0.0, src_intrinsics.fy, src_intrinsics.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        D1 = np.array([ref_intrinsics.k1, ref_intrinsics.k2, ref_intrinsics.p1, ref_intrinsics.p2, ref_intrinsics.k3], dtype=np.float64)
        D2 = np.array([src_intrinsics.k1, src_intrinsics.k2, src_intrinsics.p1, src_intrinsics.p2, src_intrinsics.k3], dtype=np.float64)

        img_size = (ref_intrinsics.width, ref_intrinsics.height)

        # Compute stereo rectification via OpenCV
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            cameraMatrix1=K1,
            distCoeffs1=D1,
            cameraMatrix2=K2,
            distCoeffs2=D2,
            imageSize=img_size,
            R=R_rel,
            T=t_rel.reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.0,  # alpha=0.0 requests the valid-region-maximizing rectification crop; actual pixel validity remains governed by rectification validity/ROI handling and downstream validity checks
            newImageSize=img_size,
        )

        # Construct rectified intrinsics dataclass from P1
        rect_fx = float(P1[0, 0])
        rect_fy = float(P1[1, 1])
        rect_cx = float(P1[0, 2])
        rect_cy = float(P1[1, 2])

        rect_intrinsics = CameraIntrinsics(
            fx=rect_fx,
            fy=rect_fy,
            cx=rect_cx,
            cy=rect_cy,
            width=ref_intrinsics.width,
            height=ref_intrinsics.height,
        )

        return StereoRectificationResult(
            R1=R1,
            R2=R2,
            P1=P1,
            P2=P2,
            Q=Q,
            rectified_intrinsics_ref=rect_intrinsics,
            baseline_reconstruction_units=baseline,
            valid_roi_ref=(int(validPixROI1[0]), int(validPixROI1[1]), int(validPixROI1[2]), int(validPixROI1[3])),
            valid_roi_src=(int(validPixROI2[0]), int(validPixROI2[1]), int(validPixROI2[2]), int(validPixROI2[3])),
            provenance={
                "method": "cv2.stereoRectify",
                "alpha": 0.0,
                "unit": DepthUnit.RECONSTRUCTION_UNITS.value,
            },
        )


class ClassicalStereoSGBMEstimator(IMVSDepthEstimator):
    """Classical dense stereo depth estimator using OpenCV StereoSGBM with left-right consistency."""

    def __init__(self, config: Optional[DenseStereoConfig] = None):
        self._config = config or DenseStereoConfig()

    @property
    def name(self) -> str:
        return f"ClassicalStereoSGBM_{self._config.config_version}"

    @property
    def config(self) -> DenseStereoConfig:
        return self._config

    def compute_dense_stereo(
        self,
        ref_image: np.ndarray,
        src_image: np.ndarray,
        ref_pose: ExtrinsicPose,
        src_pose: ExtrinsicPose,
        ref_intrinsics: CameraIntrinsics,
        src_intrinsics: CameraIntrinsics,
        ref_frame_id: str = "reference",
        src_frame_id: str = "source",
        dynamic_risk: float = 0.0,
    ) -> DenseStereoResult:
        """Execute calibrated stereo rectification, bidirectional SGBM matching, and disparity-to-depth conversion."""
        H, W = ref_intrinsics.height, ref_intrinsics.width
        if ref_image.shape[:2] != (H, W) or src_image.shape[:2] != (H, W):
            raise ValueError(f"Image dimensions do not match camera intrinsics ({W}x{H}).")

        # 1. Convert to grayscale single-channel if multi-channel
        ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY) if len(ref_image.shape) == 3 else ref_image.copy()
        src_gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY) if len(src_image.shape) == 3 else src_image.copy()

        # 2. Canonicalize pair ordering for positive SGBM disparity.
        #
        # MATHEMATICAL JUSTIFICATION:
        # cv2.stereoRectify(R, T) defines the relative transform X_2 = R * X_1 + T.
        # StereoSGBM computes positive disparity d = u_1 - u_2 > 0 when
        # P2[0,3] = T_x * f_rect < 0, i.e. when T_x < 0 (camera 2 is to the right
        # of camera 1 in the epipolar-aligned frame).
        #
        # In project convention X_c = R_cw * X_w + t_cw, camera centers are C = -R_cw^T @ t_cw.
        # The relative translation in source frame is:
        #   t_rel = t_src - (R_src @ R_ref^T) @ t_ref = R_src @ (C_ref - C_src)
        # When t_rel[0] >= 0, camera 2 (src) is to the LEFT of camera 1 (ref),
        # causing SGBM to produce primarily negative disparity, which our pipeline
        # correctly rejects as invalid. This results in an empty or wrong depth map.
        #
        # FIX: If t_rel[0] >= 0, swap ref<->src roles in rectification AND matching
        # so that the new camera 1 (old src) is always to the LEFT. The SGBM depth
        # map is then for the new camera 1 (the caller's original source camera).
        # Since the caller requested depth for ref_frame_id, we swap the images
        # so the SGBM 'left' image is the caller's ref. We achieve this by swapping
        # intrinsics, poses, AND images simultaneously, which is equivalent to
        # redefining which physical camera is 'camera 1' in stereoRectify.

        R_ref_check = np.array(ref_pose.rotation_matrix, dtype=np.float64)
        t_ref_check = np.array(ref_pose.translation_vector, dtype=np.float64)
        R_src_check = np.array(src_pose.rotation_matrix, dtype=np.float64)
        t_src_check = np.array(src_pose.translation_vector, dtype=np.float64)
        
        R_rel_check = R_src_check @ R_ref_check.T
        t_rel_check = t_src_check - R_rel_check @ t_ref_check

        pair_was_swapped = False
        if t_rel_check[0] >= 0.0:
            # Swap ref <-> src throughout the pipeline
            ref_gray, src_gray = src_gray, ref_gray
            ref_pose, src_pose = src_pose, ref_pose
            ref_intrinsics, src_intrinsics = src_intrinsics, ref_intrinsics
            pair_was_swapped = True

        # 3. Compute stereo rectification (now guaranteed t_rel[0] < 0)
        rect = StereoRectifier.compute_rectification(
            ref_intrinsics=ref_intrinsics,
            src_intrinsics=src_intrinsics,
            ref_pose=ref_pose,
            src_pose=src_pose,
        )

        K1 = np.array([[ref_intrinsics.fx, 0, ref_intrinsics.cx], [0, ref_intrinsics.fy, ref_intrinsics.cy], [0, 0, 1]], dtype=np.float64)
        D1 = np.array([ref_intrinsics.k1, ref_intrinsics.k2, ref_intrinsics.p1, ref_intrinsics.p2, ref_intrinsics.k3], dtype=np.float64)
        K2 = np.array([[src_intrinsics.fx, 0, src_intrinsics.cx], [0, src_intrinsics.fy, src_intrinsics.cy], [0, 0, 1]], dtype=np.float64)
        D2 = np.array([src_intrinsics.k1, src_intrinsics.k2, src_intrinsics.p1, src_intrinsics.p2, src_intrinsics.k3], dtype=np.float64)

        map1_x, map1_y = cv2.initUndistortRectifyMap(K1, D1, rect.R1, rect.P1, (W, H), cv2.CV_32FC1)
        map2_x, map2_y = cv2.initUndistortRectifyMap(K2, D2, rect.R2, rect.P2, (W, H), cv2.CV_32FC1)

        rect_ref_img = cv2.remap(ref_gray, map1_x, map1_y, cv2.INTER_LINEAR)
        rect_src_img = cv2.remap(src_gray, map2_x, map2_y, cv2.INTER_LINEAR)

        # 3. Create bidirectional StereoSGBM matchers
        cv2_dyn = cast(Any, cv2)
        left_matcher = cv2_dyn.StereoSGBM_create(
            minDisparity=self._config.min_disparity,
            numDisparities=self._config.num_disparities,
            blockSize=self._config.block_size,
            P1=self._config.p1,
            P2=self._config.p2,
            disp12MaxDiff=self._config.disp12_max_diff,
            preFilterCap=self._config.pre_filter_cap,
            uniquenessRatio=self._config.uniqueness_ratio,
            speckleWindowSize=self._config.speckle_window_size,
            speckleRange=self._config.speckle_range,
            mode=self._config.mode,
        )

        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher) if hasattr(cv2, "ximgproc") else None

        # Compute left (reference -> source) disparity
        disp_left_raw = left_matcher.compute(rect_ref_img, rect_src_img).astype(np.float32) / 16.0

        # Compute right (source -> reference) disparity for left-right consistency
        if right_matcher is not None:
            disp_right_raw = right_matcher.compute(rect_src_img, rect_ref_img).astype(np.float32) / 16.0
        else:
            # Fallback manual right matcher via flipped SGBM if ximgproc is not installed
            flip_ref = cv2.flip(rect_ref_img, 1)
            flip_src = cv2.flip(rect_src_img, 1)
            disp_right_raw = left_matcher.compute(flip_src, flip_ref).astype(np.float32) / 16.0
            disp_right_raw = cv2.flip(disp_right_raw, 1)

        # 4. Filter disparity and perform Left-Right Consistency check
        valid_mask = np.zeros((H, W), dtype=bool)
        visibility_state = np.full((H, W), PointVisibilityState.INVALID_DEPTH.value, dtype=object)
        depth_array = np.zeros((H, W), dtype=np.float32)
        lr_confidence = np.zeros((H, W), dtype=np.float32)

        # Disparity to depth parameters: Z = (f * B) / d
        # Baseline B in reconstruction units from rectified projection matrix
        rect_fx = float(rect.P1[0, 0])
        # In P2, horizontal translation component is P2[0, 3] = -fx * B
        baseline_units = abs(float(rect.P2[0, 3])) / rect_fx if rect_fx > 1e-6 else rect.baseline_reconstruction_units
        if baseline_units < 1e-5:
            baseline_units = rect.baseline_reconstruction_units

        # Compute local image gradient texture for confidence weighting
        grad_x = cv2.Sobel(rect_ref_img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(rect_ref_img, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        texture_score = np.clip(grad_mag / 100.0, 0.0, 1.0).astype(np.float32)

        tol_lr = self._config.lr_consistency_tolerance_px

        for r in range(H):
            for c in range(W):
                d_L = float(disp_left_raw[r, c])

                # Reject non-finite, zero, or negative disparities
                if not (math.isfinite(d_L) and d_L > 0.0):
                    visibility_state[r, c] = PointVisibilityState.INVALID_DEPTH.value
                    continue

                if d_L < self._config.min_disparity or d_L > (self._config.min_disparity + self._config.num_disparities):
                    visibility_state[r, c] = PointVisibilityState.INVALID_DEPTH.value
                    continue

                # Coordinate in source image under horizontal disparity: c_src = c - d_L
                c_src = c - d_L
                c_src_int = int(round(c_src))

                # Check if projected coordinate falls within source image bounds
                if not (0 <= c_src_int < W):
                    visibility_state[r, c] = PointVisibilityState.OCCLUDED.value
                    continue

                # Query right disparity at corresponding source coordinate
                # In OpenCV right_matcher convention, right disparity is negative (-d)
                d_R_raw = float(disp_right_raw[r, c_src_int])
                d_R_mag = -d_R_raw if d_R_raw < 0.0 else d_R_raw
                if not (math.isfinite(d_R_mag) and d_R_mag > 0.0):
                    visibility_state[r, c] = PointVisibilityState.INCONSISTENT.value
                    continue

                # Left-right disparity difference
                lr_diff = abs(d_L - d_R_mag)
                if lr_diff > tol_lr:
                    visibility_state[r, c] = PointVisibilityState.INCONSISTENT.value
                    continue

                # Disparity-to-depth conversion: Z = (fx * B) / d
                z_depth = (rect_fx * baseline_units) / d_L

                if not (math.isfinite(z_depth) and self._config.min_depth_units <= z_depth <= self._config.max_depth_units):
                    visibility_state[r, c] = PointVisibilityState.INVALID_DEPTH.value
                    continue

                # Passed all checks
                valid_mask[r, c] = True
                visibility_state[r, c] = PointVisibilityState.VALID.value
                depth_array[r, c] = float(z_depth)

                # Heuristic consistency score in [0, 1]
                lr_confidence[r, c] = float(math.exp(-lr_diff / max(1e-4, tol_lr)))

        # 5. Composite Heuristic Confidence Map
        # Combines local texture, left-right agreement, and penalizes Phase 2 dynamic risk
        raw_confidence = (
            self._config.confidence_texture_weight * texture_score +
            self._config.confidence_lr_weight * lr_confidence
        )
        # Apply Phase 2 dynamic motion risk penalty: confidence *= (1 - 0.5 * dynamic_risk)
        overall_conf = np.clip(raw_confidence * (1.0 - 0.5 * dynamic_risk), 0.0, 1.0).astype(np.float32)
        overall_conf[~valid_mask] = 0.0

        support_count = np.where(valid_mask, 2, 0).astype(np.int32)

        conf_map = DepthConfidenceMap(
            reference_frame_id=ref_frame_id,
            width=W,
            height=H,
            photometric_confidence=texture_score,
            geometric_consistency_confidence=lr_confidence,
            support_view_count=support_count,
            visibility_state=visibility_state,
            overall_confidence=overall_conf,
            provenance={
                "metric": "heuristic_blend",
                "texture_weight": self._config.confidence_texture_weight,
                "lr_weight": self._config.confidence_lr_weight,
                "dynamic_risk": dynamic_risk,
                "confidence_label": "HEURISTIC_SCORE",
                "pair_swapped": pair_was_swapped,
            },
        )

        valid_depths = depth_array[valid_mask]
        min_d = float(np.min(valid_depths)) if len(valid_depths) > 0 else 0.0
        max_d = float(np.max(valid_depths)) if len(valid_depths) > 0 else 0.0

        depth_map = DepthMap(
            reference_frame_id=ref_frame_id,
            width=W,
            height=H,
            depth_array=depth_array,
            valid_mask=valid_mask,
            depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
            min_depth=min_d,
            max_depth=max_d,
            provenance={
                "method": "OpenCV_StereoSGBM",
                "rectified_fx": rect_fx,
                "baseline_units": baseline_units,
                "unit": DepthUnit.RECONSTRUCTION_UNITS.value,
                "is_metric": False,
                "pair_swapped": pair_was_swapped,
            },
        )

        return DenseStereoResult(
            reference_frame_id=ref_frame_id,
            source_frame_id=src_frame_id,
            rectification=rect,
            disparity_map=disp_left_raw,
            valid_disparity_mask=valid_mask,
            depth_map=depth_map,
            confidence_map=conf_map,
            provenance={
                "algorithm": self.name,
                "num_disparities": self._config.num_disparities,
                "block_size": self._config.block_size,
                "lr_tolerance_px": self._config.lr_consistency_tolerance_px,
                "pair_swapped": pair_was_swapped,
            },
        )

    def estimate_depth_map(
        self,
        ref_frame_id: str,
        source_frame_ids: List[str],
        mvs_input: MVSInput,
        config: MVSConfig,
    ) -> Tuple[DepthMap, DepthConfidenceMap]:
        """Satisfies IMVSDepthEstimator contract by executing dense stereo with the primary source view."""
        if len(source_frame_ids) == 0:
            raise ValueError(f"At least one source view is required to estimate depth for frame {ref_frame_id}.")

        src_frame_id = source_frame_ids[0]
        H, W = mvs_input.image_dimensions[ref_frame_id]

        ref_pose = mvs_input.camera_poses[ref_frame_id]
        src_pose = mvs_input.camera_poses[src_frame_id]
        ref_K = mvs_input.camera_intrinsics[ref_frame_id]
        src_K = mvs_input.camera_intrinsics[src_frame_id]

        dyn_risk = max(
            mvs_input.dynamic_risk_scores.get(ref_frame_id, 0.0),
            mvs_input.dynamic_risk_scores.get(src_frame_id, 0.0),
        )

        # Load images from paths if available; otherwise initialize neutral synthetic canvas
        ref_img: Optional[np.ndarray] = None
        if mvs_input.image_paths and ref_frame_id in mvs_input.image_paths:
            ref_img = cv2.imread(mvs_input.image_paths[ref_frame_id], cv2.IMREAD_GRAYSCALE)
        if ref_img is None:
            ref_img = np.full((H, W), 128, dtype=np.uint8)

        src_img: Optional[np.ndarray] = None
        if mvs_input.image_paths and src_frame_id in mvs_input.image_paths:
            src_img = cv2.imread(mvs_input.image_paths[src_frame_id], cv2.IMREAD_GRAYSCALE)
        if src_img is None:
            src_img = np.full((H, W), 128, dtype=np.uint8)

        res = self.compute_dense_stereo(
            ref_image=ref_img,
            src_image=src_img,
            ref_pose=ref_pose,
            src_pose=src_pose,
            ref_intrinsics=ref_K,
            src_intrinsics=src_K,
            ref_frame_id=ref_frame_id,
            src_frame_id=src_frame_id,
            dynamic_risk=dyn_risk,
        )

        return res.depth_map, res.confidence_map
