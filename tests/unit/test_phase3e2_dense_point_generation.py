import math
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pytest

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
    DenseStereoConfig,
    StereoRectificationResult,
    DenseStereoResult,
    StereoRectifier,
    ClassicalStereoSGBMEstimator,
)

from src.geometry.dense_point_generation import (
    PointRejectionReason,
    DensePointGeneratorConfig,
    ValidatedDensePoint,
    DensePointGenerationResult,
    DensePointBackprojector,
    DensePointGeometricValidator,
    DensePointGenerator,
)


@pytest.fixture
def asymmetric_intrinsics() -> CameraIntrinsics:
    """Asymmetric pinhole camera intrinsics fixture."""
    return CameraIntrinsics(
        fx=950.0,
        fy=920.0,
        cx=340.0,
        cy=260.0,
        width=640,
        height=480,
    )


@pytest.fixture
def identity_camera_pose() -> ExtrinsicPose:
    """Canonical world origin camera pose [I | 0]."""
    return ExtrinsicPose(
        rotation_matrix=np.eye(3).tolist(),
        translation_vector=[0.0, 0.0, 0.0],
        frame_index=0,
        timestamp_seconds=0.0,
        coordinate_convention="opencv_optical",
        scale_factor=1.0,
        is_metric=False,
    )


def generate_synthetic_dense_stereo_result(
    intrinsics: CameraIntrinsics,
    baseline_units: float = 1.0,
    plane_depth_z: float = 20.0,
    ref_pose: Optional[ExtrinsicPose] = None,
    src_pose: Optional[ExtrinsicPose] = None,
) -> Tuple[DenseStereoResult, ExtrinsicPose, ExtrinsicPose]:
    """Helper to generate a fully consistent synthetic DenseStereoResult for testing."""
    H, W = intrinsics.height, intrinsics.width
    pose_ref = ref_pose or ExtrinsicPose(rotation_matrix=np.eye(3).tolist(), translation_vector=[0.0, 0.0, 0.0])
    pose_src = src_pose or ExtrinsicPose(rotation_matrix=np.eye(3).tolist(), translation_vector=[-baseline_units, 0.0, 0.0])

    # Rectification
    rect = StereoRectifier.compute_rectification(intrinsics, intrinsics, pose_ref, pose_src)
    rect_fx = float(rect.P1[0, 0])
    exp_disp = (rect_fx * baseline_units) / plane_depth_z

    # Build depth map and confidence map
    depth_arr = np.full((H, W), plane_depth_z, dtype=np.float32)
    disp_arr = np.full((H, W), exp_disp, dtype=np.float32)
    valid_mask = np.ones((H, W), dtype=bool)

    # Invalidate 20px borders
    valid_mask[:20, :] = False
    valid_mask[-20:, :] = False
    valid_mask[:, :20] = False
    valid_mask[:, -20:] = False

    conf_arr = np.full((H, W), 0.90, dtype=np.float32)
    vis_arr = np.full((H, W), PointVisibilityState.VALID.value, dtype=object)

    depth_map = DepthMap(
        reference_frame_id="ref_cam",
        width=W,
        height=H,
        depth_array=depth_arr,
        valid_mask=valid_mask,
        depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
        min_depth=plane_depth_z,
        max_depth=plane_depth_z,
        provenance={"method": "synthetic_test", "is_metric": False},
    )

    conf_map = DepthConfidenceMap(
        reference_frame_id="ref_cam",
        width=W,
        height=H,
        photometric_confidence=np.full((H, W), 0.9, dtype=np.float32),
        geometric_consistency_confidence=np.full((H, W), 0.95, dtype=np.float32),
        support_view_count=np.full((H, W), 2, dtype=np.int32),
        visibility_state=vis_arr,
        overall_confidence=conf_arr,
        provenance={"metric": "synthetic", "confidence_label": "HEURISTIC_SCORE"},
    )

    res = DenseStereoResult(
        reference_frame_id="ref_cam",
        source_frame_id="src_cam",
        rectification=rect,
        disparity_map=disp_arr,
        valid_disparity_mask=valid_mask,
        depth_map=depth_map,
        confidence_map=conf_map,
        provenance={"pair_swapped": False},
    )
    return res, pose_ref, pose_src


class TestPhase3E2DensePointGeneration:
    """Scientific test suite validating Phase 3E.2 mathematical invariants and failure modes."""

    def test_identity_camera_roundtrip(self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose):
        """Test A: Identity camera backprojection and forward projection round-trip."""
        K = asymmetric_intrinsics
        P1 = np.array([
            [K.fx, 0.0, K.cx, 0.0],
            [0.0, K.fy, K.cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float64)
        R1 = np.eye(3, dtype=np.float64)

        u_orig, v_orig, depth_true = 300.0, 220.0, 15.0
        X_w, X_c_orig, X_rect, ok = DensePointBackprojector.backproject_rectified_pixel(
            u_rect=u_orig, v_rect=v_orig, depth_z=depth_true, P1=P1, R1=R1, ref_pose=identity_camera_pose
        )
        assert ok and X_w is not None and X_c_orig is not None and X_rect is not None
        np.testing.assert_allclose(X_w, X_c_orig, atol=1e-12)
        np.testing.assert_allclose(X_w, X_rect, atol=1e-12)
        assert X_w[2] == pytest.approx(depth_true, rel=1e-6)

        # Forward projection
        proj_pixel, proj_z, ok_proj = DensePointBackprojector.project_world_to_rectified_pixel(
            X_w=X_w, P1=P1, R1=R1, ref_pose=identity_camera_pose
        )
        assert ok_proj and proj_pixel is not None
        assert proj_pixel[0] == pytest.approx(u_orig, abs=1e-6)
        assert proj_pixel[1] == pytest.approx(v_orig, abs=1e-6)
        assert proj_z == pytest.approx(depth_true, abs=1e-6)

    def test_rotated_and_translated_camera_roundtrip(self, asymmetric_intrinsics: CameraIntrinsics):
        """Test B, C, D: Rotated and translated camera backprojection round-trip."""
        K = asymmetric_intrinsics
        P1 = np.array([
            [K.fx, 0.0, K.cx, 0.0],
            [0.0, K.fy, K.cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float64)
        R1 = np.eye(3, dtype=np.float64)

        # 30 deg yaw rotation and [3.0, -1.5, 0.5] camera center
        yaw = math.radians(30.0)
        R_cw = np.array([
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ], dtype=np.float64)
        C_w = np.array([3.0, -1.5, 0.5], dtype=np.float64)
        t_cw = -R_cw @ C_w

        pose = ExtrinsicPose(rotation_matrix=R_cw.tolist(), translation_vector=t_cw.tolist())

        u_orig, v_orig, depth_true = 380.0, 190.0, 25.0
        X_w, X_c_orig, X_rect, ok = DensePointBackprojector.backproject_rectified_pixel(
            u_rect=u_orig, v_rect=v_orig, depth_z=depth_true, P1=P1, R1=R1, ref_pose=pose
        )
        assert ok and X_w is not None and X_c_orig is not None and X_rect is not None

        # Verify X_c = R X_w + t matches X_c_orig
        X_c_check = R_cw @ X_w + t_cw
        np.testing.assert_allclose(X_c_check, X_c_orig, atol=1e-10)

        # Verify forward projection
        proj_pixel, proj_z, ok_proj = DensePointBackprojector.project_world_to_rectified_pixel(
            X_w=X_w, P1=P1, R1=R1, ref_pose=pose
        )
        assert ok_proj and proj_pixel is not None
        assert proj_pixel[0] == pytest.approx(u_orig, abs=1e-5)
        assert proj_pixel[1] == pytest.approx(v_orig, abs=1e-5)
        assert proj_z == pytest.approx(depth_true, abs=1e-5)

    def test_rectification_rotation_consistency_and_falsification(self, asymmetric_intrinsics: CameraIntrinsics):
        """Test Q & Step 4: Verify that non-identity R1 is correctly inverted, and omitting R1^T produces falsification."""
        K = asymmetric_intrinsics
        P1 = np.array([
            [800.0, 0.0, 320.0, 0.0],
            [0.0, 800.0, 240.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float64)

        # Non-identity rectification rotation R1 (e.g. 15 deg tilt)
        tilt = math.radians(15.0)
        R1 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(tilt), -math.sin(tilt)],
            [0.0, math.sin(tilt), math.cos(tilt)],
        ], dtype=np.float64)

        pose = ExtrinsicPose(rotation_matrix=np.eye(3).tolist(), translation_vector=[0.0, 0.0, 0.0])

        u_orig, v_orig, depth_true = 320.0, 240.0, 10.0
        X_w_correct, X_c_orig, X_rect, ok = DensePointBackprojector.backproject_rectified_pixel(
            u_rect=u_orig, v_rect=v_orig, depth_z=depth_true, P1=P1, R1=R1, ref_pose=pose
        )
        assert ok and X_w_correct is not None and X_c_orig is not None and X_rect is not None

        # In rectified frame, optical axis is along [0, 0, 10]
        np.testing.assert_allclose(X_rect, np.array([0.0, 0.0, 10.0]), atol=1e-6)

        # In original camera frame, X_c_orig = R1^T @ X_rect
        expected_X_c = R1.T @ np.array([0.0, 0.0, 10.0])
        np.testing.assert_allclose(X_c_orig, expected_X_c, atol=1e-6)

        # BUGGY FALSIFICATION: If code had omitted R1^T (i.e. used X_w = X_rect directly):
        X_w_buggy = np.array([0.0, 0.0, 10.0])
        err = float(np.linalg.norm(X_w_correct - X_w_buggy))
        # 10 * sin(15 deg) = 2.588 units discrepancy!
        assert err > 2.5, "Omitting R1^T must cause a significant 3D position error."

    def test_regression_proving_tcw_is_extrinsic_translation_not_optical_center(self):
        """Test W & Step 11: Deliberate regression test proving X_w = R^T(X_c - t) and NOT R^T X_c + t."""
        # Camera at C_w = [10.0, 5.0, -2.0] with 45 deg yaw
        ang = math.radians(45.0)
        R_cw = np.array([
            [math.cos(ang), 0.0, math.sin(ang)],
            [0.0, 1.0, 0.0],
            [-math.sin(ang), 0.0, math.cos(ang)],
        ], dtype=np.float64)
        C_w = np.array([10.0, 5.0, -2.0], dtype=np.float64)
        t_cw = -R_cw @ C_w

        pose = ExtrinsicPose(rotation_matrix=R_cw.tolist(), translation_vector=t_cw.tolist())
        P1 = np.array([[500.0, 0, 320.0, 0], [0, 500.0, 240.0, 0], [0, 0, 1, 0]], dtype=np.float64)
        R1 = np.eye(3, dtype=np.float64)

        # Optical center point in camera space X_c = [0, 0, 0]
        # (simulated with depth -> 0, or by direct check)
        # Point along optical axis at Z_c = 10
        X_w_correct, _, _, ok = DensePointBackprojector.backproject_rectified_pixel(
            u_rect=320.0, v_rect=240.0, depth_z=10.0, P1=P1, R1=R1, ref_pose=pose
        )
        assert ok and X_w_correct is not None

        # Mathematical ground truth: X_w = C_w + R_cw^T @ [0, 0, 10]
        X_w_expected = C_w + R_cw.T @ np.array([0.0, 0.0, 10.0])
        np.testing.assert_allclose(X_w_correct, X_w_expected, atol=1e-10)

        # BUGGY IMPLEMENTATION 1: X_w = R^T X_c + t
        X_w_buggy1 = R_cw.T @ np.array([0.0, 0.0, 10.0]) + t_cw
        assert np.linalg.norm(X_w_correct - X_w_buggy1) > 5.0

        # BUGGY IMPLEMENTATION 2: Treating t_cw as C_w directly (X_w = t_cw + R^T X_c)
        X_w_buggy2 = t_cw + R_cw.T @ np.array([0.0, 0.0, 10.0])
        assert np.linalg.norm(X_w_correct - X_w_buggy2) > 5.0

    def test_off_axis_points_sweep(self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose):
        """Test E: Sweep off-axis points (X != 0, Y != 0, Z != const) across image plane."""
        K = asymmetric_intrinsics
        P1 = np.array([[K.fx, 0, K.cx, 0], [0, K.fy, K.cy, 0], [0, 0, 1, 0]], dtype=np.float64)
        R1 = np.eye(3, dtype=np.float64)

        test_pixels = [
            (50.0, 50.0, 5.0),
            (590.0, 50.0, 12.0),
            (50.0, 430.0, 22.5),
            (590.0, 430.0, 48.0),
            (340.0, 260.0, 1.0),
        ]

        for u, v, z in test_pixels:
            X_w, _, _, ok = DensePointBackprojector.backproject_rectified_pixel(
                u_rect=u, v_rect=v, depth_z=z, P1=P1, R1=R1, ref_pose=identity_camera_pose
            )
            assert ok and X_w is not None
            proj_pixel, proj_z, ok_proj = DensePointBackprojector.project_world_to_rectified_pixel(
                X_w=X_w, P1=P1, R1=R1, ref_pose=identity_camera_pose
            )
            assert ok_proj and proj_pixel is not None
            assert proj_pixel[0] == pytest.approx(u, abs=1e-5)
            assert proj_pixel[1] == pytest.approx(v, abs=1e-5)
            assert proj_z == pytest.approx(z, abs=1e-5)

    def test_geometric_validator_rejection_modes(self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose):
        """Test H, I, J, K, L, M, N, O, P: Explicit rejection modes in DensePointGeometricValidator."""
        cfg = DensePointGeneratorConfig(
            min_depth_units=1.0,
            max_depth_units=50.0,
            min_confidence=0.30,
            max_reprojection_error_px=1.5,
        )
        K = asymmetric_intrinsics
        P1 = np.array([[K.fx, 0, K.cx, 0], [0, K.fy, K.cy, 0], [0, 0, 1, 0]], dtype=np.float64)
        R1 = np.eye(3, dtype=np.float64)

        # 1. Valid point
        X_w = np.array([1.0, 0.5, 10.0])
        status, vis, err, rej = DensePointGeometricValidator.validate(
            X_w=X_w, X_c_orig=X_w, X_rect=X_w, origin_u_rect=K.cx + 1.0 * K.fx / 10.0, origin_v_rect=K.cy + 0.5 * K.fy / 10.0,
            depth_z=10.0, confidence=0.85, visibility_state_in=PointVisibilityState.VALID, P1=P1, R1=R1, ref_pose=identity_camera_pose, config=cfg
        )
        assert status == PointValidationStatus.VALIDATED
        assert vis == PointVisibilityState.VALID
        assert rej is None

        # 2. Non-positive depth (zero depth)
        status, vis, err, rej = DensePointGeometricValidator.validate(
            X_w=np.array([0, 0, 0]), X_c_orig=np.array([0, 0, 0]), X_rect=np.array([0, 0, 0]), origin_u_rect=K.cx, origin_v_rect=K.cy,
            depth_z=0.0, confidence=0.85, visibility_state_in=PointVisibilityState.VALID, P1=P1, R1=R1, ref_pose=identity_camera_pose, config=cfg
        )
        assert status == PointValidationStatus.REJECTED
        assert rej in (PointRejectionReason.CHEIRALITY_VIOLATION, PointRejectionReason.OUT_OF_DEPTH_BOUNDS)

        # 3. Depth out of bounds (> max_depth_units)
        status, vis, err, rej = DensePointGeometricValidator.validate(
            X_w=np.array([0, 0, 100]), X_c_orig=np.array([0, 0, 100]), X_rect=np.array([0, 0, 100]), origin_u_rect=K.cx, origin_v_rect=K.cy,
            depth_z=100.0, confidence=0.85, visibility_state_in=PointVisibilityState.VALID, P1=P1, R1=R1, ref_pose=identity_camera_pose, config=cfg
        )
        assert status == PointValidationStatus.REJECTED
        assert rej == PointRejectionReason.OUT_OF_DEPTH_BOUNDS

        # 4. Low confidence (< min_confidence)
        status, vis, err, rej = DensePointGeometricValidator.validate(
            X_w=X_w, X_c_orig=X_w, X_rect=X_w, origin_u_rect=K.cx + 1.0 * K.fx / 10.0, origin_v_rect=K.cy + 0.5 * K.fy / 10.0,
            depth_z=10.0, confidence=0.15, visibility_state_in=PointVisibilityState.VALID, P1=P1, R1=R1, ref_pose=identity_camera_pose, config=cfg
        )
        assert status == PointValidationStatus.REJECTED
        assert rej == PointRejectionReason.LOW_CONFIDENCE

        # 5. Reprojection error exceeded (shifted origin pixel)
        status, vis, err, rej = DensePointGeometricValidator.validate(
            X_w=X_w, X_c_orig=X_w, X_rect=X_w, origin_u_rect=K.cx + 50.0, origin_v_rect=K.cy,
            depth_z=10.0, confidence=0.85, visibility_state_in=PointVisibilityState.VALID, P1=P1, R1=R1, ref_pose=identity_camera_pose, config=cfg
        )
        assert status == PointValidationStatus.REJECTED
        assert rej == PointRejectionReason.REPROJECTION_ERROR_EXCEEDED
        assert err is not None and err > 2.0

        # 6. Non-finite coordinates (NaN/Inf)
        status, vis, err, rej = DensePointGeometricValidator.validate(
            X_w=np.array([np.nan, 0.0, 10.0]), X_c_orig=X_w, X_rect=X_w, origin_u_rect=K.cx, origin_v_rect=K.cy,
            depth_z=10.0, confidence=0.85, visibility_state_in=PointVisibilityState.VALID, P1=P1, R1=R1, ref_pose=identity_camera_pose, config=cfg
        )
        assert status == PointValidationStatus.REJECTED
        assert rej == PointRejectionReason.NON_FINITE_COORDINATES

    def test_full_dense_point_generator_pipeline(self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose):
        """Test complete DensePointGenerator execution from DenseStereoResult."""
        res, ref_pose, _ = generate_synthetic_dense_stereo_result(
            intrinsics=asymmetric_intrinsics,
            baseline_units=1.2,
            plane_depth_z=20.0,
            ref_pose=identity_camera_pose,
        )

        generator = DensePointGenerator()
        point_gen_result = generator.generate_points(
            stereo_result=res,
            ref_pose=ref_pose,
            ref_intrinsics=asymmetric_intrinsics,
        )

        assert isinstance(point_gen_result, DensePointGenerationResult)
        assert point_gen_result.valid_points_count > 10000
        assert len(point_gen_result.validated_points) == point_gen_result.valid_points_count
        assert len(point_gen_result.observations) == point_gen_result.valid_points_count
        assert point_gen_result.point_cloud is not None
        assert point_gen_result.point_cloud.total_fused_points == point_gen_result.valid_points_count

        # Check mean depth on plane is ~20.0
        z_vals = [pt.world_point[2] for pt in point_gen_result.validated_points]
        assert np.median(z_vals) == pytest.approx(20.0, abs=1e-4)

        # Check reprojection error is near zero (< 0.01 px on synthetic plane)
        assert point_gen_result.mean_reprojection_error_px is not None
        assert point_gen_result.mean_reprojection_error_px < 0.05

        # Check scale units contract
        assert point_gen_result.point_cloud.depth_unit == DepthUnit.RECONSTRUCTION_UNITS
        assert point_gen_result.point_cloud.is_metric_scale is False

    def test_deterministic_repeated_execution(self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose):
        """Test T: Repeated execution in same environment yields bit-exact outputs."""
        res, ref_pose, _ = generate_synthetic_dense_stereo_result(
            intrinsics=asymmetric_intrinsics, baseline_units=1.0, plane_depth_z=15.0
        )
        gen = DensePointGenerator()

        res1 = gen.generate_points(res, ref_pose, asymmetric_intrinsics)
        res2 = gen.generate_points(res, ref_pose, asymmetric_intrinsics)

        assert res1.valid_points_count == res2.valid_points_count
        assert res1.point_cloud is not None and res2.point_cloud is not None
        np.testing.assert_array_equal(res1.point_cloud.points, res2.point_cloud.points)
        np.testing.assert_array_equal(res1.point_cloud.confidences, res2.point_cloud.confidences)

    def test_provenance_and_observation_traceability(self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose):
        """Test R & Step 8: Per-point provenance and traceability."""
        res, ref_pose, _ = generate_synthetic_dense_stereo_result(
            intrinsics=asymmetric_intrinsics, baseline_units=1.0, plane_depth_z=15.0
        )
        gen = DensePointGenerator()
        gen_res = gen.generate_points(res, ref_pose, asymmetric_intrinsics)

        sample_pt = gen_res.validated_points[0]
        assert sample_pt.reference_frame_id == "ref_cam"
        assert sample_pt.source_frame_id == "src_cam"
        assert sample_pt.depth == pytest.approx(15.0, abs=1e-4)
        assert sample_pt.provenance["depth_unit"] == DepthUnit.RECONSTRUCTION_UNITS.value
        assert sample_pt.provenance["is_metric"] is False

        obs = sample_pt.to_observation()
        assert isinstance(obs, DensePointObservation)
        assert obs.reference_frame_id == "ref_cam"
        assert obs.depth == pytest.approx(15.0, abs=1e-4)
        assert obs.confidence == sample_pt.stereo_confidence

    def test_critical_rectification_adversarial_mutants(self, asymmetric_intrinsics: CameraIntrinsics):
        """Audit Section C: Intentionally execute and falsify WRONG 1, WRONG 2, WRONG 3, WRONG 4 alternatives."""
        K = asymmetric_intrinsics
        # Non-identity rectification P1 and R1
        P1 = np.array([
            [900.0, 0.0, 320.0, 0.0],
            [0.0, 900.0, 240.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float64)

        tilt = math.radians(20.0)
        R1 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(tilt), -math.sin(tilt)],
            [0.0, math.sin(tilt), math.cos(tilt)],
        ], dtype=np.float64)

        # Non-identity camera pose: 30 deg yaw, translation [-2.0, 1.0, 0.5]
        yaw = math.radians(30.0)
        R_cw = np.array([
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ], dtype=np.float64)
        t_cw = np.array([-2.0, 1.0, 0.5], dtype=np.float64)
        pose = ExtrinsicPose(rotation_matrix=R_cw.tolist(), translation_vector=t_cw.tolist())

        # Ground truth world point X_w_true
        X_w_true = np.array([1.5, -0.8, 12.0], dtype=np.float64)

        # 1. Forward project X_w_true -> original camera -> rectified camera -> rectified pixel + depth
        X_c_orig_true = R_cw @ X_w_true + t_cw
        X_rect_true = R1 @ X_c_orig_true
        z_rect_true = float(X_rect_true[2])
        assert z_rect_true > 0.0

        u_rect = float(P1[0, 0] * (X_rect_true[0] / z_rect_true) + P1[0, 2])
        v_rect = float(P1[1, 1] * (X_rect_true[1] / z_rect_true) + P1[1, 2])

        # 2. Reconstruct using 3E.2 implementation
        X_w_rec, X_c_rec, X_rect_rec, ok = DensePointBackprojector.backproject_rectified_pixel(
            u_rect=u_rect, v_rect=v_rect, depth_z=z_rect_true, P1=P1, R1=R1, ref_pose=pose
        )
        assert ok and X_w_rec is not None and X_c_rec is not None and X_rect_rec is not None
        np.testing.assert_allclose(X_w_rec, X_w_true, atol=1e-10)

        # 3. Falsify WRONG 1: X_orig = X_rect (omits R1^T)
        X_orig_wrong1 = X_rect_rec.copy()
        X_w_wrong1 = R_cw.T @ (X_orig_wrong1 - t_cw)
        assert np.linalg.norm(X_w_rec - X_w_wrong1) > 2.0, "WRONG 1 must fail"

        # 4. Falsify WRONG 2: X_orig = R1 @ X_rect (applies R1 forward instead of transpose R1^T)
        X_orig_wrong2 = R1 @ X_rect_rec
        X_w_wrong2 = R_cw.T @ (X_orig_wrong2 - t_cw)
        assert np.linalg.norm(X_w_rec - X_w_wrong2) > 4.0, "WRONG 2 must fail"

        # 5. Falsify WRONG 3: X_world = R_cw^T @ X_orig + t_cw (adds t_cw instead of subtracting)
        X_w_wrong3 = R_cw.T @ X_c_rec + t_cw
        assert np.linalg.norm(X_w_rec - X_w_wrong3) > 1.5, "WRONG 3 must fail"

        # 6. Falsify WRONG 4: X_world = R_cw^T @ X_orig (omits translation t_cw)
        X_w_wrong4 = R_cw.T @ X_c_rec
        assert np.linalg.norm(X_w_rec - X_w_wrong4) > 1.0, "WRONG 4 must fail"

    def test_unrectified_k_versus_rectified_p1_discrepancy(self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose):
        """Audit Section D: Verify that using original unrectified K instead of rectified P1 produces significant error."""
        K_orig = asymmetric_intrinsics  # fx=950, fy=920, cx=340, cy=260
        # Rectification altered focal length to 850 and principal point to 320, 240
        P1 = np.array([
            [850.0, 0.0, 320.0, 0.0],
            [0.0, 850.0, 240.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float64)
        R1 = np.eye(3, dtype=np.float64)

        u_rect, v_rect, z_rect = 450.0, 300.0, 10.0

        # Correct unprojection with P1
        X_w_correct, _, _, ok = DensePointBackprojector.backproject_rectified_pixel(
            u_rect=u_rect, v_rect=v_rect, depth_z=z_rect, P1=P1, R1=R1, ref_pose=identity_camera_pose
        )
        assert ok and X_w_correct is not None

        # Buggy unprojection using K_orig instead of P1
        x_wrong = z_rect * (u_rect - K_orig.cx) / K_orig.fx
        y_wrong = z_rect * (v_rect - K_orig.cy) / K_orig.fy
        X_w_wrong = np.array([x_wrong, y_wrong, z_rect], dtype=np.float64)

        # Check discrepancy
        err = float(np.linalg.norm(X_w_correct - X_w_wrong))
        assert err > 0.35, "Using K_orig instead of P1 must produce a measurable error."

    def test_optical_depth_versus_euclidean_range_distinction(self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose):
        """Audit Section I: Verify that depth is optical depth Z_c and strictly differs from Euclidean range ||X_c||."""
        K = asymmetric_intrinsics
        P1 = np.array([[K.fx, 0, K.cx, 0], [0, K.fy, K.cy, 0], [0, 0, 1, 0]], dtype=np.float64)
        R1 = np.eye(3, dtype=np.float64)

        # Off-axis corner pixel: u=600, v=440
        u, v, z_opt = 600.0, 440.0, 20.0
        X_w, _, _, ok = DensePointBackprojector.backproject_rectified_pixel(
            u_rect=u, v_rect=v, depth_z=z_opt, P1=P1, R1=R1, ref_pose=identity_camera_pose
        )
        assert ok and X_w is not None

        assert X_w[2] == pytest.approx(z_opt, abs=1e-10)
        euclidean_range = float(np.linalg.norm(X_w))
        # Range is strictly greater than optical depth for off-axis points (21.10 vs 20.0)
        assert euclidean_range > z_opt + 1.0
        assert euclidean_range == pytest.approx(21.10149587, abs=1e-5)

    def test_gauge_scale_invariance(self, asymmetric_intrinsics: CameraIntrinsics):
        """Audit Section N: Verify that scaling baseline and depth by factor s scales 3D world coordinates by s."""
        K = asymmetric_intrinsics
        P1 = np.array([[K.fx, 0, K.cx, 0], [0, K.fy, K.cy, 0], [0, 0, 1, 0]], dtype=np.float64)
        R1 = np.eye(3, dtype=np.float64)

        scale_s = 2.5
        R_cw = np.eye(3, dtype=np.float64)
        t_cw_1 = np.array([1.0, 0.5, -0.2], dtype=np.float64)
        t_cw_s = t_cw_1 * scale_s

        pose_1 = ExtrinsicPose(rotation_matrix=R_cw.tolist(), translation_vector=t_cw_1.tolist())
        pose_s = ExtrinsicPose(rotation_matrix=R_cw.tolist(), translation_vector=t_cw_s.tolist())

        u, v = 380.0, 290.0
        z_1 = 10.0
        z_s = z_1 * scale_s

        X_w_1, _, _, ok1 = DensePointBackprojector.backproject_rectified_pixel(u, v, z_1, P1, R1, pose_1)
        X_w_s, _, _, ok2 = DensePointBackprojector.backproject_rectified_pixel(u, v, z_s, P1, R1, pose_s)

        assert ok1 and ok2 and X_w_1 is not None and X_w_s is not None
        np.testing.assert_allclose(X_w_s, X_w_1 * scale_s, atol=1e-10)

    def test_numerical_stability_under_extreme_values(self, asymmetric_intrinsics: CameraIntrinsics, identity_camera_pose: ExtrinsicPose):
        """Audit Section O: Verify rejection and stability under extreme/singular inputs."""
        K = asymmetric_intrinsics
        R1 = np.eye(3, dtype=np.float64)

        # 1. Zero focal length P1 -> rejected
        P1_zero_fx = np.array([[0.0, 0, 320, 0], [0, 800, 240, 0], [0, 0, 1, 0]], dtype=np.float64)
        X_w, _, _, ok = DensePointBackprojector.backproject_rectified_pixel(320, 240, 10.0, P1_zero_fx, R1, identity_camera_pose)
        assert not ok and X_w is None

        # 2. Negative depth -> rejected
        P1_normal = np.array([[800, 0, 320, 0], [0, 800, 240, 0], [0, 0, 1, 0]], dtype=np.float64)
        X_w_neg, _, _, ok_neg = DensePointBackprojector.backproject_rectified_pixel(320, 240, -5.0, P1_normal, R1, identity_camera_pose)
        assert not ok_neg and X_w_neg is None

        # 3. NaN/Inf depth -> rejected
        X_w_nan, _, _, ok_nan = DensePointBackprojector.backproject_rectified_pixel(320, 240, np.nan, P1_normal, R1, identity_camera_pose)
        assert not ok_nan and X_w_nan is None
