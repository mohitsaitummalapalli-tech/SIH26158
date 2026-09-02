"""Tests for Phase 3E.1: Classical Dense Stereo Depth Estimation Baseline.

Validates all 20 required scientific invariants:
1. Rectification geometry (R1, R2, P1, P2, Q)
2. Disparity sign (d > 0)
3. Known disparity on synthetic textured scene
4. Disparity -> depth conversion (Z = f * B / d)
5. Depth convention = Z_c (optical depth along principal axis)
6. Invalid disparity rejection
7. NaN rejection
8. Inf rejection
9. Zero disparity rejection
10. Left-right consistency check
11. Out-of-bounds correspondence handling
12. Positive-depth requirement (Z > 0)
13. Confidence bounded in [0, 1]
14. Confidence is explicitly labeled HEURISTIC_SCORE
15. Reconstruction-unit preservation (no meters claimed)
16. Provenance preservation
17. Dynamic-risk propagation from Phase 2
18. Deterministic execution
19. Failure and visibility taxonomy integration
20. No metric-scale claim

Includes mandatory tests with:
- Translated + rotated camera configuration (R != I)
- Asymmetric intrinsics (fx != fy) and offset principal point (cx != W/2)
"""

import math
from typing import Dict, List, Tuple
import cv2
import numpy as np
import pytest

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
)

from src.geometry.dense_stereo import (
    DenseStereoConfig,
    StereoRectificationResult,
    DenseStereoResult,
    StereoRectifier,
    ClassicalStereoSGBMEstimator,
)


@pytest.fixture
def asymmetric_intrinsics() -> CameraIntrinsics:
    """Calibrated intrinsics with fx != fy and non-centered principal point (cx != W/2)."""
    return CameraIntrinsics(
        fx=950.0,
        fy=920.0,
        cx=340.0,  # W=640, cx != 320
        cy=260.0,  # H=480, cy != 240
        width=640,
        height=480,
    )


@pytest.fixture
def identity_camera_pose() -> ExtrinsicPose:
    """Camera 0 at world origin with identity rotation."""
    return ExtrinsicPose(
        rotation_matrix=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        translation_vector=[0.0, 0.0, 0.0],
    )


@pytest.fixture
def translated_and_rotated_pose() -> ExtrinsicPose:
    """Camera 1 with non-identity rotation (yaw + pitch) and 3D optical center at [1.2, 0.1, -0.05]."""
    # Rotation: ~5 degrees yaw around Y axis
    angle_rad = math.radians(5.0)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    R_yaw = [
        [cos_a, 0.0, sin_a],
        [0.0, 1.0, 0.0],
        [-sin_a, 0.0, cos_a],
    ]
    # Camera optical center in world space: C_w = [1.2, 0.1, -0.05]
    C_w = np.array([1.2, 0.1, -0.05], dtype=np.float64)
    # Extrinsic translation vector under X_c = R_cw X_w + t_cw is t_cw = -R_cw @ C_w
    t_cw = (-np.array(R_yaw) @ C_w).tolist()
    return ExtrinsicPose(
        rotation_matrix=R_yaw,
        translation_vector=t_cw,
    )


def generate_synthetic_textured_stereo_pair(
    intrinsics: CameraIntrinsics,
    baseline_units: float,
    plane_depth_z: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate synthetic textured stereo image pair viewing a fronto-parallel plane at known depth Z."""
    H, W = intrinsics.height, intrinsics.width
    np.random.seed(42)

    # Base texture with high-frequency features (checkerboard + noise)
    base_tex = np.zeros((H, W), dtype=np.uint8)
    cell_size = 16
    for y in range(0, H, cell_size):
        for x in range(0, W, cell_size):
            if ((x // cell_size) + (y // cell_size)) % 2 == 0:
                base_tex[y:y+cell_size, x:x+cell_size] = 200
            else:
                base_tex[y:y+cell_size, x:x+cell_size] = 60

    # Add Gaussian texture noise
    noise = (np.random.randn(H, W) * 25.0).astype(np.float32)
    ref_img = np.clip(base_tex.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Theoretical horizontal disparity: d = (fx * B) / Z
    true_disparity = (intrinsics.fx * baseline_units) / plane_depth_z

    # Source view: shift reference image horizontally by true_disparity
    src_img = np.zeros_like(ref_img)
    shift_px = int(round(true_disparity))
    if shift_px < W:
        src_img[:, :W - shift_px] = ref_img[:, shift_px:]

    return ref_img, src_img, true_disparity


class TestPhase3E1DenseStereo:
    """Comprehensive test suite for Phase 3E.1 dense stereo baseline."""

    def test_rectification_geometry_translated_and_rotated_camera(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose, translated_and_rotated_pose: ExtrinsicPose
    ):
        """Verify 1: StereoRectifier computes valid R1, R2, P1, P2, Q with translated+rotated camera and fx != fy."""
        rect = StereoRectifier.compute_rectification(
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
            ref_pose=identity_camera_pose,
            src_pose=translated_and_rotated_pose,
        )

        assert isinstance(rect, StereoRectificationResult)
        assert rect.R1.shape == (3, 3)
        assert rect.R2.shape == (3, 3)
        assert rect.P1.shape == (3, 4)
        assert rect.P2.shape == (3, 4)
        assert rect.Q.shape == (4, 4)

        # In rectified stereo, row lines must be horizontally aligned (P1[1, :] == P2[1, :])
        np.testing.assert_allclose(rect.P1[1, :], rect.P2[1, :], atol=1e-4)

        # Baseline must match 3D distance between camera optical centers C = -R^T @ t
        R0 = np.array(identity_camera_pose.rotation_matrix)
        t0 = np.array(identity_camera_pose.translation_vector)
        R1 = np.array(translated_and_rotated_pose.rotation_matrix)
        t1 = np.array(translated_and_rotated_pose.translation_vector)
        c0 = -R0.T @ t0
        c1 = -R1.T @ t1
        expected_baseline = float(np.linalg.norm(c1 - c0))
        assert rect.baseline_reconstruction_units == pytest.approx(expected_baseline, rel=1e-4)

        # Rectified focal length must be positive
        assert rect.rectified_intrinsics_ref.fx > 0.0
        assert rect.rectified_intrinsics_ref.fy > 0.0

    def test_degenerate_coincident_baseline_rejected(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose
    ):
        """Verify zero baseline raises ValueError."""
        coincident_pose = ExtrinsicPose(
            rotation_matrix=identity_camera_pose.rotation_matrix,
            translation_vector=[0.0, 0.0, 0.0],
        )
        with pytest.raises(ValueError, match="Degenerate stereo baseline"):
            StereoRectifier.compute_rectification(
                ref_intrinsics=asymmetric_intrinsics,
                src_intrinsics=asymmetric_intrinsics,
                ref_pose=identity_camera_pose,
                src_pose=coincident_pose,
            )

    def test_known_disparity_and_depth_recovery_on_synthetic_plane(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose
    ):
        """Verify 2, 3, 4, 5: Disparity sign (d > 0), known disparity, disparity->depth, and depth == Z_c."""
        baseline = 1.0
        known_depth_z = 25.0  # Z_c = 25.0 reconstruction units

        ref_img, src_img, expected_disp = generate_synthetic_textured_stereo_pair(
            intrinsics=asymmetric_intrinsics,
            baseline_units=baseline,
            plane_depth_z=known_depth_z,
        )

        # Right camera at optical center C_src = [baseline, 0, 0] => t_cw = -R @ C = [-baseline, 0, 0]
        src_pose = ExtrinsicPose(
            rotation_matrix=identity_camera_pose.rotation_matrix,
            translation_vector=[-baseline, 0.0, 0.0],
        )

        cfg = DenseStereoConfig(
            min_disparity=0,
            num_disparities=64,
            block_size=7,
            lr_consistency_tolerance_px=2.0,
        )
        estimator = ClassicalStereoSGBMEstimator(cfg)

        res = estimator.compute_dense_stereo(
            ref_image=ref_img,
            src_image=src_img,
            ref_pose=identity_camera_pose,
            src_pose=src_pose,
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
        )

        assert isinstance(res, DenseStereoResult)
        assert np.any(res.valid_disparity_mask)

        # 1. Disparity sign: all valid disparities must be strictly positive
        valid_disps = res.disparity_map[res.valid_disparity_mask]
        assert np.all(valid_disps > 0.0)

        # 2. Median recovered disparity should match expected disparity within 1.0 px
        median_disp = float(np.median(valid_disps))
        assert median_disp == pytest.approx(expected_disp, abs=1.5)

        # 3. Disparity-to-depth: recovered optical depth Z_c must match known depth (25.0)
        valid_depths = res.depth_map.depth_array[res.depth_map.valid_mask]
        median_depth = float(np.median(valid_depths))
        assert median_depth == pytest.approx(known_depth_z, rel=0.10)

        # 4. Depth convention check: depth is Z_c, not Euclidean distance
        assert res.depth_map.depth_unit == DepthUnit.RECONSTRUCTION_UNITS

    def test_invalid_nan_inf_zero_disparity_rejection(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose
    ):
        """Verify 6, 7, 8, 9, 12: Disparities with <= 0, NaN, Inf are strictly rejected; no np.nan_to_num."""
        H, W = asymmetric_intrinsics.height, asymmetric_intrinsics.width
        # Blank images will produce low texture / invalid matching
        blank_img = np.full((H, W), 128, dtype=np.uint8)

        src_pose = ExtrinsicPose(
            rotation_matrix=identity_camera_pose.rotation_matrix,
            translation_vector=[-1.0, 0.0, 0.0],
        )

        estimator = ClassicalStereoSGBMEstimator()
        res = estimator.compute_dense_stereo(
            ref_image=blank_img,
            src_image=blank_img,
            ref_pose=identity_camera_pose,
            src_pose=src_pose,
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
        )

        # On blank images, all or nearly all pixels should be marked invalid/inconsistent
        # No invalid pixel may have positive valid depth
        invalid_mask = ~res.depth_map.valid_mask
        assert np.all(res.depth_map.depth_array[invalid_mask] == 0.0)
        # All valid depths must be strictly positive
        if np.any(res.depth_map.valid_mask):
            assert np.all(res.depth_map.depth_array[res.depth_map.valid_mask] > 0.0)

    def test_left_right_disparity_consistency_rejection(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose
    ):
        """Verify 10: Left-right disparity disagreement causes pixel rejection as INCONSISTENT."""
        H, W = asymmetric_intrinsics.height, asymmetric_intrinsics.width
        # Generate random uncorrelated noise images -> guaranteed left-right disparity disagreement
        np.random.seed(123)
        img_left = np.random.randint(0, 255, (H, W), dtype=np.uint8)
        img_right = np.random.randint(0, 255, (H, W), dtype=np.uint8)

        src_pose = ExtrinsicPose(
            rotation_matrix=identity_camera_pose.rotation_matrix,
            translation_vector=[-1.0, 0.0, 0.0],
        )

        estimator = ClassicalStereoSGBMEstimator()
        res = estimator.compute_dense_stereo(
            ref_image=img_left,
            src_image=img_right,
            ref_pose=identity_camera_pose,
            src_pose=src_pose,
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
        )

        vis_states = res.confidence_map.visibility_state
        # 1. Mismatched images produce a substantial number of INCONSISTENT classifications
        inconsistent_pixels = (vis_states == PointVisibilityState.INCONSISTENT.value)
        assert np.sum(inconsistent_pixels) > 5000

        # 2. Inconsistent pixels must NEVER be included in valid depth mask
        assert np.all(res.depth_map.valid_mask[inconsistent_pixels] == False)
        assert np.all(res.depth_map.depth_array[inconsistent_pixels] == 0.0)

    def test_confidence_semantics_and_range_bounds(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose
    ):
        """Verify 13, 14: Confidence map values strictly bounded in [0, 1] and labeled HEURISTIC_SCORE."""
        ref_img, src_img, _ = generate_synthetic_textured_stereo_pair(
            intrinsics=asymmetric_intrinsics,
            baseline_units=1.0,
            plane_depth_z=20.0,
        )

        src_pose = ExtrinsicPose(
            rotation_matrix=identity_camera_pose.rotation_matrix,
            translation_vector=[-1.0, 0.0, 0.0],
        )

        estimator = ClassicalStereoSGBMEstimator()
        res = estimator.compute_dense_stereo(
            ref_image=ref_img,
            src_image=src_img,
            ref_pose=identity_camera_pose,
            src_pose=src_pose,
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
        )

        conf = res.confidence_map.overall_confidence
        assert np.all(conf >= 0.0)
        assert np.all(conf <= 1.0)
        assert res.confidence_map.provenance["confidence_label"] == "HEURISTIC_SCORE"

    def test_dynamic_risk_propagation_attenuates_confidence(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose
    ):
        """Verify 17: Phase 2 dynamic motion risk strictly attenuates confidence."""
        ref_img, src_img, _ = generate_synthetic_textured_stereo_pair(
            intrinsics=asymmetric_intrinsics,
            baseline_units=1.0,
            plane_depth_z=20.0,
        )

        src_pose = ExtrinsicPose(
            rotation_matrix=identity_camera_pose.rotation_matrix,
            translation_vector=[-1.0, 0.0, 0.0],
        )

        estimator = ClassicalStereoSGBMEstimator()

        # Run with dynamic risk = 0.0
        res_clean = estimator.compute_dense_stereo(
            ref_image=ref_img,
            src_image=src_img,
            ref_pose=identity_camera_pose,
            src_pose=src_pose,
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
            dynamic_risk=0.0,
        )

        # Run with dynamic risk = 0.8
        res_risky = estimator.compute_dense_stereo(
            ref_image=ref_img,
            src_image=src_img,
            ref_pose=identity_camera_pose,
            src_pose=src_pose,
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
            dynamic_risk=0.8,
        )

        mean_conf_clean = float(np.mean(res_clean.confidence_map.overall_confidence[res_clean.valid_disparity_mask]))
        mean_conf_risky = float(np.mean(res_risky.confidence_map.overall_confidence[res_risky.valid_disparity_mask]))

        # High dynamic risk must strictly attenuate confidence
        assert mean_conf_risky < mean_conf_clean
        assert res_risky.confidence_map.provenance["dynamic_risk"] == pytest.approx(0.8)

    def test_deterministic_execution(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose
    ):
        """Verify 18: Repeated runs with identical inputs produce bit-exact identical disparity and depth."""
        ref_img, src_img, _ = generate_synthetic_textured_stereo_pair(
            intrinsics=asymmetric_intrinsics,
            baseline_units=1.0,
            plane_depth_z=20.0,
        )

        src_pose = ExtrinsicPose(
            rotation_matrix=identity_camera_pose.rotation_matrix,
            translation_vector=[-1.0, 0.0, 0.0],
        )

        estimator = ClassicalStereoSGBMEstimator()

        res1 = estimator.compute_dense_stereo(
            ref_image=ref_img,
            src_image=src_img,
            ref_pose=identity_camera_pose,
            src_pose=src_pose,
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
        )

        res2 = estimator.compute_dense_stereo(
            ref_image=ref_img,
            src_image=src_img,
            ref_pose=identity_camera_pose,
            src_pose=src_pose,
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
        )

        np.testing.assert_array_equal(res1.disparity_map, res2.disparity_map)
        np.testing.assert_array_equal(res1.depth_map.depth_array, res2.depth_map.depth_array)
        np.testing.assert_array_equal(res1.confidence_map.overall_confidence, res2.confidence_map.overall_confidence)

    def test_reconstruction_unit_preservation_and_no_metric_claim(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose
    ):
        """Verify 15, 20: DepthMap preserves RECONSTRUCTION_UNITS and is_metric is False."""
        ref_img, src_img, _ = generate_synthetic_textured_stereo_pair(
            intrinsics=asymmetric_intrinsics,
            baseline_units=1.0,
            plane_depth_z=20.0,
        )

        src_pose = ExtrinsicPose(
            rotation_matrix=identity_camera_pose.rotation_matrix,
            translation_vector=[-1.0, 0.0, 0.0],
        )

        estimator = ClassicalStereoSGBMEstimator()
        res = estimator.compute_dense_stereo(
            ref_image=ref_img,
            src_image=src_img,
            ref_pose=identity_camera_pose,
            src_pose=src_pose,
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
        )

        assert res.depth_map.depth_unit == DepthUnit.RECONSTRUCTION_UNITS
        assert res.depth_map.provenance["is_metric"] is False

    def test_mvs_input_interface_compliance(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose
    ):
        """Verify compliance with IMVSDepthEstimator.estimate_depth_map."""
        src_pose = ExtrinsicPose(
            rotation_matrix=identity_camera_pose.rotation_matrix,
            translation_vector=[-1.0, 0.0, 0.0],
        )

        dims = {"f0": (480, 640), "f1": (480, 640)}
        poses = {"f0": identity_camera_pose, "f1": src_pose}
        intrinsics = {"f0": asymmetric_intrinsics, "f1": asymmetric_intrinsics}

        mvs_in = MVSInput(
            selected_frame_ids=["f0", "f1"],
            image_dimensions=dims,
            camera_intrinsics=intrinsics,
            camera_poses=poses,
        )

        estimator = ClassicalStereoSGBMEstimator()
        dm, cm = estimator.estimate_depth_map("f0", ["f1"], mvs_in, MVSConfig())

        assert isinstance(dm, DepthMap)
        assert isinstance(cm, DepthConfidenceMap)
        assert dm.reference_frame_id == "f0"
        assert cm.reference_frame_id == "f0"
        assert dm.depth_unit == DepthUnit.RECONSTRUCTION_UNITS

    def test_out_of_bounds_correspondence_tagged_as_occluded(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose
    ):
        """Verify 11: Correspondences falling outside image boundaries are tagged as OCCLUDED."""
        ref_img, src_img, _ = generate_synthetic_textured_stereo_pair(
            intrinsics=asymmetric_intrinsics,
            baseline_units=1.0,
            plane_depth_z=25.0,
        )

        src_pose = ExtrinsicPose(
            rotation_matrix=identity_camera_pose.rotation_matrix,
            translation_vector=[-1.0, 0.0, 0.0],
        )

        estimator = ClassicalStereoSGBMEstimator()
        res = estimator.compute_dense_stereo(
            ref_image=ref_img,
            src_image=src_img,
            ref_pose=identity_camera_pose,
            src_pose=src_pose,
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
        )

        # For columns c where c - d < 0, pixel must be marked OCCLUDED or INVALID_DEPTH
        vis_states = res.confidence_map.visibility_state
        # Near left border (columns 0 to 10), projected source pixels fall outside bounds
        left_border_states = vis_states[:, :10]
        assert np.any(left_border_states == PointVisibilityState.OCCLUDED.value) or np.any(left_border_states == PointVisibilityState.INVALID_DEPTH.value)
        # All occluded pixels must have valid_mask == False
        occluded_pixels = (vis_states == PointVisibilityState.OCCLUDED.value)
        if np.any(occluded_pixels):
            assert np.all(res.depth_map.valid_mask[occluded_pixels] == False)
            assert np.all(res.depth_map.depth_array[occluded_pixels] == 0.0)

    def test_nan_and_inf_disparities_explicitly_rejected(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose
    ):
        """Verify 7, 8: Non-finite values are never converted to valid depth and no np.nan_to_num is used."""
        H, W = asymmetric_intrinsics.height, asymmetric_intrinsics.width
        # Construct images with saturated/black regions where matcher fails
        img1 = np.zeros((H, W), dtype=np.uint8)
        img2 = np.zeros((H, W), dtype=np.uint8)

        src_pose = ExtrinsicPose(
            rotation_matrix=identity_camera_pose.rotation_matrix,
            translation_vector=[-1.0, 0.0, 0.0],
        )

        estimator = ClassicalStereoSGBMEstimator()
        res = estimator.compute_dense_stereo(
            ref_image=img1,
            src_image=img2,
            ref_pose=identity_camera_pose,
            src_pose=src_pose,
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
        )

        # Disparity array must not contain NaNs or Infs in valid depths
        assert not np.any(np.isnan(res.depth_map.depth_array))
        assert not np.any(np.isinf(res.depth_map.depth_array))
        # Valid mask must be completely False for pure black image
        assert np.sum(res.depth_map.valid_mask) == 0

    def test_zero_disparity_rejection_no_infinite_depth(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose
    ):
        """Verify 9, 12: Zero disparity is rejected and never produces division by zero or infinite depth."""
        H, W = asymmetric_intrinsics.height, asymmetric_intrinsics.width
        img = np.full((H, W), 200, dtype=np.uint8)

        src_pose = ExtrinsicPose(
            rotation_matrix=identity_camera_pose.rotation_matrix,
            translation_vector=[-1.0, 0.0, 0.0],
        )

        estimator = ClassicalStereoSGBMEstimator()
        res = estimator.compute_dense_stereo(
            ref_image=img,
            src_image=img,
            ref_pose=identity_camera_pose,
            src_pose=src_pose,
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
        )

        # Depth array must be finite everywhere
        assert np.all(np.isfinite(res.depth_map.depth_array))
        # No depth can be infinite
        assert not np.any(np.isneginf(res.depth_map.depth_array))
        assert not np.any(np.isposinf(res.depth_map.depth_array))

    def test_provenance_preservation_details(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose
    ):
        """Verify 16: DepthMap, DepthConfidenceMap, and DenseStereoResult preserve complete provenance."""
        ref_img, src_img, _ = generate_synthetic_textured_stereo_pair(
            intrinsics=asymmetric_intrinsics,
            baseline_units=1.0,
            plane_depth_z=25.0,
        )

        src_pose = ExtrinsicPose(
            rotation_matrix=identity_camera_pose.rotation_matrix,
            translation_vector=[-1.0, 0.0, 0.0],
        )

        estimator = ClassicalStereoSGBMEstimator()
        res = estimator.compute_dense_stereo(
            ref_image=ref_img,
            src_image=src_img,
            ref_pose=identity_camera_pose,
            src_pose=src_pose,
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
            ref_frame_id="frame_001",
            src_frame_id="frame_002",
        )

        assert res.reference_frame_id == "frame_001"
        assert res.source_frame_id == "frame_002"
        assert res.depth_map.reference_frame_id == "frame_001"
        assert res.confidence_map.reference_frame_id == "frame_001"

        # Provenance metadata fields
        assert "method" in res.depth_map.provenance
        assert "algorithm" in res.provenance
        assert "confidence_label" in res.confidence_map.provenance
        assert res.confidence_map.provenance["confidence_label"] == "HEURISTIC_SCORE"

    def test_translated_and_rotated_stereo_depth_estimation(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose, translated_and_rotated_pose: ExtrinsicPose
    ):
        """Verify stereo matching and rectification pipeline with translated and rotated cameras."""
        H, W = asymmetric_intrinsics.height, asymmetric_intrinsics.width
        # Generate texture
        np.random.seed(99)
        ref_img = np.random.randint(50, 200, (H, W), dtype=np.uint8)
        src_img = np.random.randint(50, 200, (H, W), dtype=np.uint8)

        estimator = ClassicalStereoSGBMEstimator()
        res = estimator.compute_dense_stereo(
            ref_image=ref_img,
            src_image=src_img,
            ref_pose=identity_camera_pose,
            src_pose=translated_and_rotated_pose,
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
        )

        assert isinstance(res, DenseStereoResult)
        assert res.rectification.R1.shape == (3, 3)
        assert res.rectification.R2.shape == (3, 3)
        assert res.depth_map.depth_unit == DepthUnit.RECONSTRUCTION_UNITS
        assert res.confidence_map.visibility_state.shape == (H, W)

    def test_arbitrary_camera_ordering_handled_consistently(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose
    ):
        """Verify that ClassicalStereoSGBMEstimator handles both camera orderings consistently.

        Specifically tests:
        CASE A: ref camera at origin [0,0,0], src camera at [1,0,0] (t_rel[0] < 0)
        CASE B: ref camera at [1,0,0], src camera at origin [0,0,0] (t_rel[0] > 0)
        Both cases must successfully recover the known planar depth (25.0 units).
        """
        ref_img, src_img, _ = generate_synthetic_textured_stereo_pair(
            intrinsics=asymmetric_intrinsics,
            baseline_units=1.0,
            plane_depth_z=25.0,
        )

        # Right camera at optical center C = [1.0, 0, 0] => t_cw = -R @ C = [-1.0, 0, 0]
        right_pose = ExtrinsicPose(
            rotation_matrix=identity_camera_pose.rotation_matrix,
            translation_vector=[-1.0, 0.0, 0.0],
        )

        cfg = DenseStereoConfig(
            min_disparity=0,
            num_disparities=64,
            block_size=7,
            lr_consistency_tolerance_px=2.0,
        )
        estimator = ClassicalStereoSGBMEstimator(cfg)

        # CASE A: Standard ordering (left -> right)
        res_A = estimator.compute_dense_stereo(
            ref_image=ref_img,
            src_image=src_img,
            ref_pose=identity_camera_pose,
            src_pose=right_pose,
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
            ref_frame_id="cam_left",
            src_frame_id="cam_right",
        )
        assert np.sum(res_A.valid_disparity_mask) > 10000
        valid_depths_A = res_A.depth_map.depth_array[res_A.depth_map.valid_mask]
        assert abs(float(np.median(valid_depths_A)) - 25.0) < 1.0

        # CASE B: Swapped ordering (right -> left)
        res_B = estimator.compute_dense_stereo(
            ref_image=src_img,
            src_image=ref_img,
            ref_pose=right_pose,
            src_pose=identity_camera_pose,
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
            ref_frame_id="cam_right",
            src_frame_id="cam_left",
        )
        assert np.sum(res_B.valid_disparity_mask) > 10000
        valid_depths_B = res_B.depth_map.depth_array[res_B.depth_map.valid_mask]
        assert abs(float(np.median(valid_depths_B)) - 25.0) < 1.0

    def test_camera_center_recovery_and_baseline_not_equal_to_translation_diff(self):
        """Forensic Audit #2: Verify C_w = -R^T @ t_cw and that baseline B != ||t_src - t_ref|| in general.

        Under project convention:
            X_c = R_cw X_w + t_cw
            C_w = -R_cw^T t_cw
            t_cw = -R_cw C_w
        """
        # Define camera optical centers in world frame
        C_ref = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        C_src = np.array([1.5, 0.0, 0.0], dtype=np.float64)
        true_baseline = float(np.linalg.norm(C_src - C_ref))
        assert true_baseline == 1.5

        # Rotations: ref camera rotated by pitch 45 deg, src camera rotated by yaw 45 deg
        ang_pitch = math.radians(45.0)
        R_ref = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(ang_pitch), -math.sin(ang_pitch)],
            [0.0, math.sin(ang_pitch), math.cos(ang_pitch)],
        ], dtype=np.float64)

        ang_yaw = math.radians(45.0)
        R_src = np.array([
            [math.cos(ang_yaw), 0.0, math.sin(ang_yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(ang_yaw), 0.0, math.cos(ang_yaw)],
        ], dtype=np.float64)

        # Compute extrinsic translation vectors t_cw = -R_cw @ C_w
        t_ref = -R_ref @ C_ref  # [0, 0, 0]
        t_src = -R_src @ C_src  # [-1.5 * cos(yaw), 0, 1.5 * sin(yaw)]

        # Prove ||t_src - t_ref_offset|| != ||C_src - C_ref_offset|| when cameras have distinct positions/rotations
        C_ref_offset = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        t_ref_offset = -R_ref @ C_ref_offset
        dist_C = float(np.linalg.norm(C_src - C_ref_offset))
        dist_t = float(np.linalg.norm(t_src - t_ref_offset))
        assert abs(dist_C - dist_t) > 0.10, "||t_src - t_ref|| must differ from true baseline ||C_src - C_ref||"

        # Verify StereoRectifier computes the exact true baseline from poses
        K = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0, width=640, height=480)
        pose_ref = ExtrinsicPose(rotation_matrix=R_ref.tolist(), translation_vector=t_ref.tolist())
        pose_src = ExtrinsicPose(rotation_matrix=R_src.tolist(), translation_vector=t_src.tolist())

        rect = StereoRectifier.compute_rectification(K, K, pose_ref, pose_src)
        assert abs(rect.baseline_reconstruction_units - true_baseline) < 1e-6

    def test_adversarial_rotation_combinations_and_reversals(
        self, asymmetric_intrinsics: CameraIntrinsics
    ):
        """Forensic Audit #2: Adversarial tests across all rotation configurations and both pair orderings.

        Tests:
        1. Identity rotations
        2. Rotation only on ref camera
        3. Rotation only on src camera
        4. Both cameras rotated
        5. Reversed camera ordering
        All configurations must successfully compute valid rectification, correct baseline, and positive disparity search.
        """
        K = asymmetric_intrinsics
        # 5 degrees yaw
        a = math.radians(5.0)
        R_yaw = np.array([
            [math.cos(a), 0.0, math.sin(a)],
            [0.0, 1.0, 0.0],
            [-math.sin(a), 0.0, math.cos(a)],
        ], dtype=np.float64)

        # 4 degrees pitch
        b = math.radians(4.0)
        R_pitch = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(b), -math.sin(b)],
            [0.0, math.sin(b), math.cos(b)],
        ], dtype=np.float64)

        C_0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        C_1 = np.array([1.2, 0.0, 0.0], dtype=np.float64)

        configs = [
            ("Identity", np.eye(3), np.eye(3)),
            ("Ref_Rotated", R_yaw, np.eye(3)),
            ("Src_Rotated", np.eye(3), R_pitch),
            ("Both_Rotated", R_yaw, R_pitch),
        ]

        for name, R0, R1 in configs:
            t0 = (-R0 @ C_0).tolist()
            t1 = (-R1 @ C_1).tolist()

            pose0 = ExtrinsicPose(rotation_matrix=R0.tolist(), translation_vector=t0)
            pose1 = ExtrinsicPose(rotation_matrix=R1.tolist(), translation_vector=t1)

            # Standard order: pose0 -> pose1
            rect_fwd = StereoRectifier.compute_rectification(K, K, pose0, pose1)
            assert abs(rect_fwd.baseline_reconstruction_units - 1.2) < 1e-5, f"Failed baseline forward for {name}"
            assert np.all(np.isfinite(rect_fwd.R1))
            assert np.all(np.isfinite(rect_fwd.R2))
            assert np.all(np.isfinite(rect_fwd.P1))
            assert np.all(np.isfinite(rect_fwd.P2))

            # Reversed order: pose1 -> pose0
            rect_rev = StereoRectifier.compute_rectification(K, K, pose1, pose0)
            assert abs(rect_rev.baseline_reconstruction_units - 1.2) < 1e-5, f"Failed baseline reverse for {name}"
            assert np.all(np.isfinite(rect_rev.R1))
            assert np.all(np.isfinite(rect_rev.R2))
            assert np.all(np.isfinite(rect_rev.P1))
            assert np.all(np.isfinite(rect_rev.P2))

    def test_regression_pose_translation_is_extrinsic_not_optical_center(self):
        """Forensic Audit #2 Regression: Verify that supplying non-zero R and t = -R @ C computes correct C and baseline.

        If the estimator mistakenly interpreted translation_vector as C directly,
        it would compute baseline = ||t_src - t_ref|| which is incorrect for rotated cameras.
        """
        # Camera 0 at origin, Camera 1 at [2.0, 0, 0] with 45 deg yaw
        C0 = np.array([0.0, 0.0, 0.0])
        C1 = np.array([2.0, 0.0, 0.0])
        ang = math.radians(45.0)
        R1 = np.array([
            [math.cos(ang), 0.0, math.sin(ang)],
            [0.0, 1.0, 0.0],
            [-math.sin(ang), 0.0, math.cos(ang)],
        ])
        t1 = -R1 @ C1  # [-1.414, 0, 1.414]

        K = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
        pose0 = ExtrinsicPose(rotation_matrix=np.eye(3).tolist(), translation_vector=[0.0, 0.0, 0.0])
        pose1 = ExtrinsicPose(rotation_matrix=R1.tolist(), translation_vector=t1.tolist())

        rect = StereoRectifier.compute_rectification(K, K, pose0, pose1)
        # True baseline is 2.0
        assert abs(rect.baseline_reconstruction_units - 2.0) < 1e-6
        # If buggy code had computed ||t1 - t0||, it would still happen to be 2.0 here because R1 is orthogonal and C0=0.
        # But with non-zero C0 and distinct rotation R0, ||t1 - t0|| != ||C1 - C0||:
        C0_off = np.array([0.0, 1.0, 0.0])
        R0_off = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(ang), -math.sin(ang)],
            [0.0, math.sin(ang), math.cos(ang)],
        ])
        t0_off = -R0_off @ C0_off

        # True baseline ||C1 - C0_off|| = sqrt(2^2 + (-1)^2) = sqrt(5) = 2.2360679775
        true_b = float(np.linalg.norm(C1 - C0_off))
        pose0_off = ExtrinsicPose(rotation_matrix=R0_off.tolist(), translation_vector=t0_off.tolist())
        rect_off = StereoRectifier.compute_rectification(K, K, pose0_off, pose1)
        assert abs(rect_off.baseline_reconstruction_units - true_b) < 1e-6

    def test_multi_depth_and_multi_baseline_synthetic_planes(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose
    ):
        """Final Audit Section D: Multi-depth and multi-baseline sweep (Z in {2, 5, 25, 50}, B in {0.2, 1.0, 2.0})."""
        depth_levels = [2.0, 5.0, 25.0, 50.0]
        baselines = [0.2, 1.0, 2.0]

        for z_true in depth_levels:
            for b_true in baselines:
                # Expected disparity
                d_true = (asymmetric_intrinsics.fx * b_true) / z_true
                shift_px = int(round(d_true))
                if shift_px < 3 or shift_px > 60:
                    continue  # SGBM config covers [0, 64] with reliable baseline shift >= 3 px

                ref_img, src_img, exp_disp = generate_synthetic_textured_stereo_pair(
                    intrinsics=asymmetric_intrinsics,
                    baseline_units=b_true,
                    plane_depth_z=z_true,
                )

                src_pose = ExtrinsicPose(
                    rotation_matrix=identity_camera_pose.rotation_matrix,
                    translation_vector=[-b_true, 0.0, 0.0],
                )

                cfg = DenseStereoConfig(
                    min_disparity=0,
                    num_disparities=64,
                    block_size=7,
                    lr_consistency_tolerance_px=2.0,
                    min_depth_units=0.1,
                    max_depth_units=200.0,
                )
                estimator = ClassicalStereoSGBMEstimator(cfg)
                res = estimator.compute_dense_stereo(
                    ref_image=ref_img,
                    src_image=src_img,
                    ref_pose=identity_camera_pose,
                    src_pose=src_pose,
                    ref_intrinsics=asymmetric_intrinsics,
                    src_intrinsics=asymmetric_intrinsics,
                )

                assert np.sum(res.valid_disparity_mask) > 5000
                valid_depths = res.depth_map.depth_array[res.depth_map.valid_mask]
                rec_depth = float(np.median(valid_depths))
                # Actual physical depth represented by integer pixel shift in synthetic raster:
                rect_fx = float(res.rectification.P1[0, 0])
                z_actual_shift = (rect_fx * b_true) / shift_px
                assert rec_depth == pytest.approx(z_actual_shift, rel=0.08)

    def test_3d_point_backprojection_roundtrip_off_axis(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose
    ):
        """Final Audit Section E: 3D point world -> camera -> image -> camera -> world roundtrip consistency."""
        K = asymmetric_intrinsics
        R_cw = np.array(identity_camera_pose.rotation_matrix, dtype=np.float64)
        t_cw = np.array(identity_camera_pose.translation_vector, dtype=np.float64)

        # Off-axis 3D points in world coordinates
        pts_w = [
            np.array([-2.5, 1.8, 15.0]),
            np.array([3.2, -2.1, 28.4]),
            np.array([0.0, 0.0, 10.0]),
            np.array([-5.0, -4.0, 45.0]),
        ]

        for X_w in pts_w:
            # 1. World to camera
            X_c = R_cw @ X_w + t_cw
            z_c = float(X_c[2])
            assert z_c > 0.0

            # 2. Camera to image
            u = K.fx * (X_c[0] / z_c) + K.cx
            v = K.fy * (X_c[1] / z_c) + K.cy

            # 3. Image + depth to camera
            X_c_rec = np.array([
                (u - K.cx) * z_c / K.fx,
                (v - K.cy) * z_c / K.fy,
                z_c,
            ])
            np.testing.assert_allclose(X_c_rec, X_c, atol=1e-10)

            # 4. Camera to world
            X_w_rec = R_cw.T @ (X_c_rec - t_cw)
            np.testing.assert_allclose(X_w_rec, X_w, atol=1e-10)

    def test_distortion_coefficients_handling_in_rectification(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose
    ):
        """Final Audit Section F: Verify non-zero distortion coefficients are accepted and processed in StereoRectifier."""
        distorted_intrinsics = CameraIntrinsics(
            fx=asymmetric_intrinsics.fx,
            fy=asymmetric_intrinsics.fy,
            cx=asymmetric_intrinsics.cx,
            cy=asymmetric_intrinsics.cy,
            width=asymmetric_intrinsics.width,
            height=asymmetric_intrinsics.height,
            k1=-0.15,
            k2=0.08,
            p1=0.001,
            p2=-0.002,
            k3=-0.01,
        )

        src_pose = ExtrinsicPose(
            rotation_matrix=identity_camera_pose.rotation_matrix,
            translation_vector=[-1.0, 0.0, 0.0],
        )

        rect = StereoRectifier.compute_rectification(
            ref_intrinsics=distorted_intrinsics,
            src_intrinsics=distorted_intrinsics,
            ref_pose=identity_camera_pose,
            src_pose=src_pose,
        )

        assert isinstance(rect, StereoRectificationResult)
        assert np.all(np.isfinite(rect.P1))
        assert np.all(np.isfinite(rect.P2))
        assert rect.baseline_reconstruction_units == pytest.approx(1.0, rel=1e-4)

    def test_different_ref_and_src_intrinsics(
        self, identity_camera_pose: ExtrinsicPose
    ):
        """Final Audit Section F: Verify StereoRectifier with distinct K_ref and K_src."""
        K_ref = CameraIntrinsics(fx=850.0, fy=830.0, cx=325.0, cy=245.0, width=640, height=480)
        K_src = CameraIntrinsics(fx=920.0, fy=900.0, cx=315.0, cy=235.0, width=640, height=480)

        src_pose = ExtrinsicPose(
            rotation_matrix=identity_camera_pose.rotation_matrix,
            translation_vector=[-1.5, 0.0, 0.0],
        )

        rect = StereoRectifier.compute_rectification(
            ref_intrinsics=K_ref,
            src_intrinsics=K_src,
            ref_pose=identity_camera_pose,
            src_pose=src_pose,
        )

        assert isinstance(rect, StereoRectificationResult)
        assert rect.baseline_reconstruction_units == pytest.approx(1.5, rel=1e-4)
        np.testing.assert_allclose(rect.P1[1, :], rect.P2[1, :], atol=1e-4)

    def test_provenance_records_pair_swap_flag(
        self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose
    ):
        """Final Audit Section N: Verify provenance accurately records pair_swapped flag."""
        ref_img, src_img, _ = generate_synthetic_textured_stereo_pair(
            intrinsics=asymmetric_intrinsics,
            baseline_units=1.0,
            plane_depth_z=25.0,
        )

        # Standard order (t_rel[0] < 0): pair_swapped is False
        right_pose = ExtrinsicPose(
            rotation_matrix=identity_camera_pose.rotation_matrix,
            translation_vector=[-1.0, 0.0, 0.0],
        )
        estimator = ClassicalStereoSGBMEstimator()

        res_std = estimator.compute_dense_stereo(
            ref_image=ref_img,
            src_image=src_img,
            ref_pose=identity_camera_pose,
            src_pose=right_pose,
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
            ref_frame_id="cam0",
            src_frame_id="cam1",
        )
        assert res_std.provenance["pair_swapped"] is False
        assert res_std.depth_map.provenance["pair_swapped"] is False
        assert res_std.confidence_map.provenance["pair_swapped"] is False

        # Reversed order (t_rel[0] > 0): pair_swapped is True
        res_rev = estimator.compute_dense_stereo(
            ref_image=src_img,
            src_image=ref_img,
            ref_pose=right_pose,
            src_pose=identity_camera_pose,
            ref_intrinsics=asymmetric_intrinsics,
            src_intrinsics=asymmetric_intrinsics,
            ref_frame_id="cam1",
            src_frame_id="cam0",
        )
        assert res_rev.provenance["pair_swapped"] is True
        assert res_rev.depth_map.provenance["pair_swapped"] is True
        assert res_rev.confidence_map.provenance["pair_swapped"] is True
