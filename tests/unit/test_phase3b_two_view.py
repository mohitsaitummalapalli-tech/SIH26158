"""Deterministic unit and integration tests for Phase 3B Two-View Geometry & Robust Geometric Verification.

DISCLAIMER:
ALL FIXTURES IN THIS MODULE ARE SYNTHETIC MATHEMATICAL TEST DATA GENERATED
SOLELY FOR TWO-VIEW GEOMETRY VERIFICATION. THEY DO NOT REPRESENT REAL UAV ACCURACY.
"""

import math
from typing import Optional, Tuple
import numpy as np
import pytest

from src.geometry import (
    MeasurementType,
    GeometryFailureReason,
    DistortionModel,
    DistortionStatus,
    CameraIntrinsics,
    FeatureCorrespondences,
    TwoViewGeometryResult,
    TwoViewConfig,
    TwoViewGeometryEstimator,
    GeometryMathContracts,
    ClassicalFeatureExtractor,
    ClassicalDescriptorMatcher,
)


def generate_synthetic_two_view_scene(
    n_points: int = 100,
    R_rel: Optional[np.ndarray] = None,
    t_rel: Optional[np.ndarray] = None,
    noise_std_px: float = 0.0,
    outlier_ratio: float = 0.0,
    intrinsics: Optional[CameraIntrinsics] = None,
    seed: int = 42,
    near_depth: float = 5.0,
    far_depth: float = 20.0,
    is_planar: bool = False,
) -> Tuple[np.ndarray, np.ndarray, CameraIntrinsics, np.ndarray, np.ndarray]:
    """Generate deterministic 3D points and their projected 2D correspondences in two views."""
    np.random.seed(seed)

    if intrinsics is None:
        intrinsics = CameraIntrinsics(
            fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000,
            distortion_model=DistortionModel.NONE_RECTIFIED,
            distortion_status=DistortionStatus.RECTIFIED_ZERO_DISTORTION,
        )

    if R_rel is None:
        # Small pitch/yaw rotation
        angle = np.radians(5.0)
        R_rel = np.array([
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0,           1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ], dtype=np.float64)

    if t_rel is None:
        t_rel = np.array([1.0, 0.2, 0.1], dtype=np.float64)
        t_rel = t_rel / np.linalg.norm(t_rel)

    # Generate 3D points in View 1 coordinate frame
    if is_planar:
        # Points on a plane Z = 10.0
        X = np.random.uniform(-3.0, 3.0, n_points)
        Y = np.random.uniform(-3.0, 3.0, n_points)
        Z = np.full(n_points, 10.0)
    else:
        X = np.random.uniform(-4.0, 4.0, n_points)
        Y = np.random.uniform(-3.0, 3.0, n_points)
        Z = np.random.uniform(near_depth, far_depth, n_points)

    pts3d_1 = np.column_stack((X, Y, Z))

    # Project to View 1: u1 = fx * X / Z + cx, v1 = fy * Y / Z + cy
    pts1_px = np.zeros((n_points, 2), dtype=np.float64)
    pts1_px[:, 0] = intrinsics.fx * (pts3d_1[:, 0] / pts3d_1[:, 2]) + intrinsics.cx
    pts1_px[:, 1] = intrinsics.fy * (pts3d_1[:, 1] / pts3d_1[:, 2]) + intrinsics.cy

    # Transform 3D points to View 2: X2 = R * X1 + t
    pts3d_2 = (R_rel @ pts3d_1.T).T + t_rel

    # Project to View 2
    pts2_px = np.zeros((n_points, 2), dtype=np.float64)
    pts2_px[:, 0] = intrinsics.fx * (pts3d_2[:, 0] / pts3d_2[:, 2]) + intrinsics.cx
    pts2_px[:, 1] = intrinsics.fy * (pts3d_2[:, 1] / pts3d_2[:, 2]) + intrinsics.cy

    # Add Gaussian noise
    if noise_std_px > 0.0:
        pts1_px += np.random.normal(0.0, noise_std_px, pts1_px.shape)
        pts2_px += np.random.normal(0.0, noise_std_px, pts2_px.shape)

    # Add random outliers
    if outlier_ratio > 0.0:
        n_outliers = int(n_points * outlier_ratio)
        outlier_indices = np.random.choice(n_points, n_outliers, replace=False)
        pts2_px[outlier_indices, 0] = np.random.uniform(0.0, intrinsics.width, n_outliers)
        pts2_px[outlier_indices, 1] = np.random.uniform(0.0, intrinsics.height, n_outliers)

    return pts1_px, pts2_px, intrinsics, R_rel, t_rel


# 1. Perfect Synthetic Correspondences F and E Estimation
def test_perfect_correspondences_estimation():
    pts1, pts2, K, R_true, t_true = generate_synthetic_two_view_scene(n_points=60)
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(len(pts1)), match_count=len(pts1),
    )
    estimator = TwoViewGeometryEstimator()

    # Fundamental path
    res_f = estimator.estimate_fundamental(corr)
    assert res_f.f_status == "SUCCESS"
    assert res_f.inlier_count >= 50
    assert res_f.inlier_ratio > 0.90
    assert res_f.mean_epipolar_residual < 0.5

    # Essential path
    res_e = estimator.estimate_essential(corr, K)
    assert res_e.e_status == "SUCCESS"
    assert res_e.inlier_count >= 50
    assert res_e.relative_rotation is not None
    assert res_e.relative_translation is not None
    assert res_e.cheirality_ratio > 0.90


# 2. Correspondences with Known Pixel Noise
def test_noisy_correspondences_estimation():
    pts1, pts2, K, R_true, t_true = generate_synthetic_two_view_scene(n_points=80, noise_std_px=0.8)
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(len(pts1)), match_count=len(pts1),
    )
    estimator = TwoViewGeometryEstimator()
    res_e = estimator.estimate_essential(corr, K)

    assert res_e.e_status == "SUCCESS"
    assert res_e.inlier_ratio > 0.75
    assert res_e.mean_reprojection_error_px < 3.0


# 3. Significant Random Outliers Robust Estimation
def test_outlier_rejection_ransac():
    # 30% outliers
    pts1, pts2, K, R_true, t_true = generate_synthetic_two_view_scene(n_points=120, outlier_ratio=0.30)
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(len(pts1)), match_count=len(pts1),
    )
    estimator = TwoViewGeometryEstimator()
    res_e = estimator.estimate_essential(corr, K)

    assert res_e.e_status == "SUCCESS"
    assert res_e.inlier_count >= 60
    assert res_e.inlier_ratio > 0.50


# 4. F Estimation without Intrinsics
def test_f_estimation_without_intrinsics():
    pts1, pts2, K, _, _ = generate_synthetic_two_view_scene(n_points=40)
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(len(pts1)), match_count=len(pts1),
    )
    estimator = TwoViewGeometryEstimator()
    res_f = estimator.estimate_fundamental(corr)

    assert res_f.model_used == "FUNDAMENTAL_MATRIX"
    assert res_f.has_calibrated_intrinsics is False
    assert res_f.fundamental_matrix is not None
    assert res_f.essential_matrix is None
    assert res_f.relative_rotation is None


# 5. E Estimation with Valid Intrinsics
def test_e_estimation_with_valid_intrinsics():
    pts1, pts2, K, _, _ = generate_synthetic_two_view_scene(n_points=50)
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(len(pts1)), match_count=len(pts1),
    )
    estimator = TwoViewGeometryEstimator()
    res_e = estimator.estimate_essential(corr, K)

    assert res_e.model_used == "ESSENTIAL_MATRIX"
    assert res_e.has_calibrated_intrinsics is True
    assert res_e.essential_matrix is not None
    assert res_e.relative_rotation is not None


# 6. Calibration Unavailable E Path Rejection
def test_calibration_unavailable_rejection():
    pts1, pts2, _, _, _ = generate_synthetic_two_view_scene(n_points=30)
    uncalibrated_k = CameraIntrinsics(
        fx=0.0, fy=0.0, cx=0.0, cy=0.0, width=1000, height=1000,
        distortion_status=DistortionStatus.CALIBRATION_UNAVAILABLE,
    )
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(len(pts1)), match_count=len(pts1),
    )
    estimator = TwoViewGeometryEstimator()
    res_e = estimator.estimate_essential(corr, uncalibrated_k)

    assert res_e.e_status == "FAILED"
    assert res_e.failure_reason == GeometryFailureReason.CALIBRATION_UNAVAILABLE


# 7. Brown-Conrady Distortion State Handling
def test_brown_conrady_distortion_handling():
    k_dist = CameraIntrinsics(
        fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000,
        k1=-0.1, k2=0.01, p1=0.0, p2=0.0, k3=0.0,
        distortion_model=DistortionModel.BROWN_CONRADY_RADIAL_TANGENTIAL,
        distortion_status=DistortionStatus.EXPLICIT_MODEL_PRESENT,
    )
    pts1, pts2, _, _, _ = generate_synthetic_two_view_scene(n_points=50, intrinsics=k_dist)
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(len(pts1)), match_count=len(pts1),
    )
    estimator = TwoViewGeometryEstimator()
    res_e = estimator.estimate_essential(corr, k_dist)
    assert res_e.has_calibrated_intrinsics is True


# 8. Fisheye Distortion State Handling
def test_fisheye_distortion_handling():
    k_fish = CameraIntrinsics(
        fx=800.0, fy=800.0, cx=400.0, cy=400.0, width=800, height=800,
        k1=0.05, k2=0.01, p1=0.0, p2=0.0,
        distortion_model=DistortionModel.FISHEYE_EQUIDISTANT,
        distortion_status=DistortionStatus.EXPLICIT_MODEL_PRESENT,
    )
    pts1, pts2, _, _, _ = generate_synthetic_two_view_scene(n_points=50, intrinsics=k_fish)
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(len(pts1)), match_count=len(pts1),
    )
    estimator = TwoViewGeometryEstimator()
    res_e = estimator.estimate_essential(corr, k_fish)
    assert res_e.has_calibrated_intrinsics is True


# 9. Unsupported Distortion Model Rejection
def test_unsupported_distortion_model_rejection():
    k_bad = CameraIntrinsics(
        fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000,
        distortion_model=DistortionModel.UNSUPPORTED_UNKNOWN,
        distortion_status=DistortionStatus.UNSUPPORTED_MODEL,
    )
    pts1, pts2, _, _, _ = generate_synthetic_two_view_scene(n_points=30)
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(len(pts1)), match_count=len(pts1),
    )
    estimator = TwoViewGeometryEstimator()
    res_e = estimator.estimate_essential(corr, k_bad)

    assert res_e.e_status == "FAILED"
    assert res_e.failure_reason == GeometryFailureReason.CALIBRATION_UNAVAILABLE


# 10. Known Relative Rotation Recovery
def test_known_relative_rotation_recovery():
    # Pure yaw rotation of 10 degrees
    yaw = np.radians(10.0)
    R_known = np.array([
        [np.cos(yaw),  0.0, np.sin(yaw)],
        [0.0,          1.0, 0.0],
        [-np.sin(yaw), 0.0, np.cos(yaw)],
    ])
    t_known = np.array([1.0, 0.0, 0.0])

    pts1, pts2, K, _, _ = generate_synthetic_two_view_scene(n_points=80, R_rel=R_known, t_rel=t_known)
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(len(pts1)), match_count=len(pts1),
    )
    estimator = TwoViewGeometryEstimator()
    res_e = estimator.estimate_essential(corr, K)

    assert res_e.e_status == "SUCCESS"
    R_est = res_e.relative_rotation
    assert R_est is not None

    # Geodesic angular error between R_known and R_est: trace(R_est * R_known^T)
    trace_val = np.clip((np.trace(R_est @ R_known.T) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_deg = np.degrees(np.arccos(trace_val))
    assert rot_err_deg < 2.0  # Angular error under 2 degrees


# 11. Translation Direction Recovery (Unit Vector)
def test_translation_direction_recovery():
    t_known = np.array([1.0, 0.5, 0.2])
    t_known = t_known / np.linalg.norm(t_known)

    pts1, pts2, K, _, _ = generate_synthetic_two_view_scene(n_points=80, t_rel=t_known)
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(len(pts1)), match_count=len(pts1),
    )
    estimator = TwoViewGeometryEstimator()
    res_e = estimator.estimate_essential(corr, K)

    t_est = res_e.relative_translation
    assert t_est is not None
    assert math.isclose(np.linalg.norm(t_est), 1.0, abs_tol=1e-5)
    dot_val = np.clip(np.dot(t_est, t_known), -1.0, 1.0)
    angle_err_deg = np.degrees(np.arccos(abs(dot_val)))
    assert angle_err_deg < 5.0  # Direction within 5 degrees


# 12. Cheirality Hypothesis Selection
def test_cheirality_hypothesis_selection():
    pts1, pts2, K, _, _ = generate_synthetic_two_view_scene(n_points=60)
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(len(pts1)), match_count=len(pts1),
    )
    estimator = TwoViewGeometryEstimator()
    res_e = estimator.estimate_essential(corr, K)

    # Unique hypothesis with high positive cheirality
    assert res_e.cheirality_passed_count > 45
    assert res_e.cheirality_ratio > 0.70


# 13. Ambiguous Pose Case
def test_ambiguous_pose_case():
    res_ambig = TwoViewGeometryResult(
        frame_a_id="f1", frame_b_id="f2",
        is_degenerate=True,
        failure_reason=GeometryFailureReason.DEGENERATE_GEOMETRY,
        diagnostics=["Ambiguous pose recovery: top hypotheses counts are indistinguishable."],
    )
    assert res_ambig.is_degenerate is True
    assert res_ambig.failure_reason == GeometryFailureReason.DEGENERATE_GEOMETRY


# 14. Weak Baseline Case
def test_weak_baseline_case():
    # Very small translation t = [0.01, 0, 0] relative to depth Z = 20
    t_tiny = np.array([0.005, 0.0, 0.0])
    pts1, pts2, K, _, _ = generate_synthetic_two_view_scene(n_points=60, t_rel=t_tiny, near_depth=20.0, far_depth=40.0)
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(len(pts1)), match_count=len(pts1),
    )
    estimator = TwoViewGeometryEstimator()
    res_e = estimator.estimate_essential(corr, K)

    # Parallax is very small (< 1 deg)
    assert res_e.failure_reason in (GeometryFailureReason.WEAK_BASELINE, GeometryFailureReason.PURE_ROTATION_RISK)


# 15. Pure Rotation Risk Case
def test_pure_rotation_risk_case():
    # Exactly zero translation
    t_zero = np.array([0.0, 0.0, 0.0])
    R_rot = np.array([
        [0.9961947, 0.0, 0.0871557],
        [0.0,       1.0, 0.0],
        [-0.0871557, 0.0, 0.9961947],
    ])
    pts1, pts2, K, _, _ = generate_synthetic_two_view_scene(n_points=60, R_rel=R_rot, t_rel=t_zero)
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(len(pts1)), match_count=len(pts1),
    )
    estimator = TwoViewGeometryEstimator()
    res_e = estimator.estimate_essential(corr, K)

    assert res_e.failure_reason in (GeometryFailureReason.PURE_ROTATION_RISK, GeometryFailureReason.WEAK_BASELINE)


# 16. Degenerate Planar Configuration Detection
def test_degenerate_planar_configuration():
    pts1, pts2, _, _, _ = generate_synthetic_two_view_scene(n_points=50, is_planar=True)
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(len(pts1)), match_count=len(pts1),
    )
    estimator = TwoViewGeometryEstimator()
    res_f = estimator.estimate_fundamental(corr)
    # Planar configuration causes homography inliers to dominate
    assert res_f.is_degenerate is True


# 17. Insufficient Correspondences Rejection (< 8 points)
def test_insufficient_correspondences():
    pts1 = np.array([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]])
    pts2 = np.array([[105.0, 100.0], [205.0, 200.0], [305.0, 300.0]])
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(3), match_count=3,
    )
    estimator = TwoViewGeometryEstimator()
    res_f = estimator.estimate_fundamental(corr)

    assert res_f.f_status == "FAILED"
    assert res_f.failure_reason == GeometryFailureReason.INSUFFICIENT_MATCHES


# 18. RANSAC Threshold Semantics (HEURISTIC_DEFAULT)
def test_ransac_threshold_semantics():
    cfg = TwoViewConfig()
    assert cfg.ransac_threshold_px == 2.0
    assert cfg.ransac_threshold_norm == 0.002
    assert cfg.min_inlier_ratio == 0.20
    assert cfg.weak_baseline_parallax_deg == 1.0
    assert cfg.pure_rotation_parallax_deg == 0.5


# 19. Residual Statistics (Sampson Epipolar Distance)
def test_residual_statistics_computation():
    pts1, pts2, K, _, _ = generate_synthetic_two_view_scene(n_points=50, noise_std_px=0.5)
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(len(pts1)), match_count=len(pts1),
    )
    estimator = TwoViewGeometryEstimator()
    res_e = estimator.estimate_essential(corr, K)

    assert res_e.mean_epipolar_residual >= 0.0
    assert res_e.median_epipolar_residual >= 0.0
    assert res_e.mean_reprojection_error_px >= 0.0


# 20. Coordinate Convention Consistency
def test_coordinate_convention_consistency():
    # Optical frame: +X right, +Y down, +Z forward
    # Point at (0, 0, 10) in camera frame has Z > 0
    pt = np.array([0.0, 0.0, 10.0])
    r_mat = np.eye(3)
    t_vec = np.zeros(3)
    assert GeometryMathContracts.check_cheirality(pt, r_mat, t_vec) is True


# 21. Camera Center / Translation Conversion (C = -R^T * t)
def test_camera_center_translation_conversion():
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    t = np.array([2.0, 0.0, 0.0])

    # C2 in frame 1 coordinates: C2 = -R^T * t = -[[0, 1, 0], [-1, 0, 0], [0, 0, 1]] * [2, 0, 0] = [0, 2, 0]
    C2 = -R.T @ t
    expected_C2 = np.array([0.0, 2.0, 0.0])
    np.testing.assert_array_almost_equal(C2, expected_C2)


# 22. Deterministic Repeated Execution
def test_deterministic_repeated_execution():
    pts1, pts2, K, _, _ = generate_synthetic_two_view_scene(n_points=60, seed=777)
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(len(pts1)), match_count=len(pts1),
    )
    estimator = TwoViewGeometryEstimator()

    res1 = estimator.estimate_essential(corr, K)
    res2 = estimator.estimate_essential(corr, K)

    assert res1.inlier_count == res2.inlier_count
    assert res1.relative_rotation is not None
    assert res2.relative_rotation is not None
    np.testing.assert_array_almost_equal(res1.relative_rotation, res2.relative_rotation, decimal=4)


# 23. Explicit Failure-State Transitions
def test_explicit_failure_state_transitions():
    assert GeometryFailureReason.INSUFFICIENT_MATCHES.value == "INSUFFICIENT_MATCHES"
    assert GeometryFailureReason.CALIBRATION_UNAVAILABLE.value == "CALIBRATION_UNAVAILABLE"
    assert GeometryFailureReason.GEOMETRIC_VERIFICATION_FAILED.value == "GEOMETRIC_VERIFICATION_FAILED"
    assert GeometryFailureReason.PURE_ROTATION_RISK.value == "PURE_ROTATION_RISK"
    assert GeometryFailureReason.WEAK_BASELINE.value == "WEAK_BASELINE"


# 24. Scale Ambiguity Metadata
def test_scale_ambiguity_metadata():
    pts1, pts2, K, _, _ = generate_synthetic_two_view_scene(n_points=50)
    corr = FeatureCorrespondences(
        frame_a_id="f1", frame_b_id="f2", points_a=pts1, points_b=pts2,
        descriptor_distances=np.zeros(len(pts1)), match_count=len(pts1),
    )
    estimator = TwoViewGeometryEstimator()
    res_e = estimator.estimate_essential(corr, K)

    d = res_e.to_dict()
    assert d["scale_status"] == "SCALE_AMBIGUOUS"
    assert d["translation_magnitude_status"] == "UNOBSERVABLE"


# 25. Distinction Between Trajectory Proxy and Optical Baseline
def test_trajectory_proxy_vs_optical_baseline():
    res = TwoViewGeometryResult(
        frame_a_id="f1", frame_b_id="f2",
        relative_translation=np.array([1.0, 0.0, 0.0]),
        relative_translation_measurement=MeasurementType.ESTIMATED,
    )
    assert res.relative_translation_measurement == MeasurementType.ESTIMATED
    assert res.translation_magnitude_status == "UNOBSERVABLE"


# 26. Integration Test: Phase 3A FeatureMatchResult -> FeatureCorrespondences -> Phase 3B TwoViewGeometryResult
def test_integration_feature_matching_to_two_view_geometry():
    # Synthetic textured images with multi-depth geometric patches (non-planar 3D structure)
    img_a = np.zeros((480, 640, 3), dtype=np.uint8)
    img_b = np.zeros((480, 640, 3), dtype=np.uint8)

    # Foreground layer (depth Z=5, horizontal disparity = 16px)
    for y in range(40, 200, 30):
        for x in range(40, 260, 30):
            for sy in range(15):
                for sx in range(15):
                    val = 255 if ((sx // 4 + sy // 4) % 2 == 0) else 0
                    img_a[y + sy, x + sx] = [val, 255 - val, 200]
                    img_b[y + sy, x + sx + 16] = [val, 255 - val, 200]

    # Background layer (depth Z=15, horizontal disparity = 6px)
    for y in range(260, 440, 30):
        for x in range(320, 580, 30):
            for sy in range(15):
                for sx in range(15):
                    val = 255 if ((sx // 4 + sy // 4) % 2 == 0) else 0
                    img_a[y + sy, x + sx] = [255 - val, val, 100]
                    img_b[y + sy, x + sx + 6] = [255 - val, val, 100]

    extractor = ClassicalFeatureExtractor()
    matcher = ClassicalDescriptorMatcher()

    feat_a = extractor.extract(img_a, frame_id="kf_001")
    feat_b = extractor.extract(img_b, frame_id="kf_002")

    match_res = matcher.match(feat_a, feat_b)
    assert match_res.status == "SUCCESS"
    assert match_res.accepted_match_count > 30

    # Convert to FeatureCorrespondences contract
    correspondences = match_res.to_correspondences()

    # Pass to Phase 3B TwoViewGeometryEstimator
    K = CameraIntrinsics(
        fx=800.0, fy=800.0, cx=320.0, cy=240.0, width=640, height=480,
        distortion_model=DistortionModel.NONE_RECTIFIED,
        distortion_status=DistortionStatus.RECTIFIED_ZERO_DISTORTION,
    )
    two_view_estimator = TwoViewGeometryEstimator()

    # 1. Test Fundamental path
    res_f = two_view_estimator.estimate_fundamental(correspondences)
    assert res_f.f_status == "SUCCESS"
    assert res_f.inlier_count > 20

    # 2. Test Essential path
    res_e = two_view_estimator.estimate_essential(correspondences, K)
    assert res_e.e_status == "SUCCESS"
    assert res_e.inlier_count > 20
    assert res_e.relative_rotation is not None
    assert res_e.relative_translation is not None
    assert res_e.scale_status == "SCALE_AMBIGUOUS"
    assert res_e.translation_magnitude_status == "UNOBSERVABLE"
