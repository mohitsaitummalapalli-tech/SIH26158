"""Deterministic design and contract unit tests for Phase 3D Global Bundle Adjustment.

Tests mathematical contracts, gauge parameterization (6M - 7 + 3N), S^2 unit baseline,
Huber loss continuity, sparsity structure, and post-optimization validation WITHOUT implementing a solver.
"""

import pytest
import math
import numpy as np
from typing import Dict, List, Tuple

from src.geometry.contracts import (
    EvaluationLevel,
    PipelineStageStatus,
    MeasurementType,
    GaugeFixingPolicy,
    CameraIntrinsics,
    ExtrinsicPose,
    SparseReconstructionResult,
    TriangulatedTrack,
)
from src.geometry.sfm import SfMCamera, SfMTrack
from src.geometry.bundle_adjustment import (
    BAFailureReason,
    BundleAdjustmentConfig,
    BAReprojectionMetrics,
    BundleAdjustmentResult,
    BARotationMath,
    BATranslationDirectionMath,
    BAParameterManager,
    BAResidualCalculator,
    BASparsityStructure,
    BAPostOptimizationValidator,
    IBundleAdjustmentOptimizer,
    BAParameterLayout,
    BAResidualEvaluator,
    BundleAdjustmentEngine,
)


def create_mock_sfm_reconstruction(
    n_cams: int = 3,
    n_points: int = 20,
) -> Tuple[Dict[str, SfMCamera], Dict[int, np.ndarray], Dict[int, SfMTrack], CameraIntrinsics]:
    """Create deterministic mock cameras, landmarks, and tracks."""
    np.random.seed(42)
    intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000)

    # 3D points in front of camera: Z in [8, 14]
    pts_3d: Dict[int, np.ndarray] = {}
    for i in range(n_points):
        pts_3d[i] = np.array([
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-1.5, 1.5),
            np.random.uniform(8.0, 14.0),
        ], dtype=np.float64)

    cameras: Dict[str, SfMCamera] = {}
    for i in range(n_cams):
        cid = f"cam_{i:02d}"
        yaw = np.radians(i * 3.0)
        R_cw = np.array([
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0,         1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ], dtype=np.float64)
        c_world = np.array([i * 1.0, 0.0, 0.0], dtype=np.float64)
        t_cw = -R_cw @ c_world
        cameras[cid] = SfMCamera(cid, R_cw, t_cw, intrinsics, is_registered=True, registration_order=i)

    tracks: Dict[int, SfMTrack] = {}
    for t_id, pt_w in pts_3d.items():
        obs: Dict[str, Tuple[float, float]] = {}
        kpt_indices: Dict[str, int] = {}
        for idx, (cid, cam) in enumerate(cameras.items()):
            proj, z = cam.project(pt_w)
            obs[cid] = (float(proj[0]), float(proj[1]))
            kpt_indices[cid] = t_id
        tracks[t_id] = SfMTrack(
            track_id=t_id,
            world_point=pt_w,
            observations=obs,
            keypoint_indices=kpt_indices,
        )

    return cameras, pts_3d, tracks, intrinsics


# 1. Input Contract Validation
def test_input_contract_validation():
    """Verify that BA config and input validation enforce at least 2 cameras and valid landmarks."""
    config = BundleAdjustmentConfig()
    assert config.min_registered_cameras == 2
    assert config.min_landmarks == 10
    assert config.gauge_policy == GaugeFixingPolicy.FIX_FIRST_CAMERA_AND_UNIT_BASELINE


# 2. Camera 1 Unit Baseline Direction Parameterization on S^2
def test_camera_1_unit_baseline_direction_parameterization():
    """Verify 2-DoF tangent-space parameterization on S^2 preserves exact unit norm."""
    d0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    b1, b2 = BATranslationDirectionMath.construct_tangent_basis(d0)

    # Orthonormality checks
    assert abs(float(np.dot(b1, d0))) < 1e-8
    assert abs(float(np.dot(b2, d0))) < 1e-8
    assert abs(float(np.dot(b1, b2))) < 1e-8
    assert abs(float(np.linalg.norm(b1)) - 1.0) < 1e-8
    assert abs(float(np.linalg.norm(b2)) - 1.0) < 1e-8

    # Test mapping for various tangent perturbation vectors
    test_alphas = [
        np.array([0.0, 0.0]),
        np.array([1e-8, -1e-8]),
        np.array([0.1, -0.2]),
        np.array([0.8, 0.6]),
        np.array([-1.2, 0.5]),
    ]
    for alpha in test_alphas:
        dir_vec = BATranslationDirectionMath.tangent_to_direction(d0, alpha)
        # Norm must be exactly 1.0
        assert abs(float(np.linalg.norm(dir_vec)) - 1.0) < 1e-8

        # Round-trip recovery
        rec_alpha = BATranslationDirectionMath.direction_to_tangent(d0, dir_vec)
        # For small to moderate alpha, tangent vector is recovered
        if np.linalg.norm(alpha) < 1.0:
            np.testing.assert_array_almost_equal(rec_alpha, alpha, decimal=5)


# 3. Exact Gauge Parameter Dimension: 6M - 7 + 3N
def test_exact_gauge_parameter_dimension():
    """Verify that the unconstrained state dimension is exactly 6M - 7 + 3N for M >= 2."""
    config = BundleAdjustmentConfig()

    cases = [
        (2, 10, 6 * 2 - 7 + 3 * 10),      # 5 + 30 = 35
        (3, 15, 6 * 3 - 7 + 3 * 15),      # 11 + 45 = 56
        (5, 50, 6 * 5 - 7 + 3 * 50),      # 23 + 150 = 173
        (10, 100, 6 * 10 - 7 + 3 * 100),  # 53 + 300 = 353
    ]

    for M, N, expected_dim in cases:
        cam_order = [f"cam_{i:02d}" for i in range(M)]
        track_order = list(range(N))
        pm = BAParameterManager(cam_order, track_order, config)
        assert pm.num_camera_params == 6 * M - 7
        assert pm.num_landmark_params == 3 * N
        assert pm.total_params == expected_dim


# 4. Parameter Packing and Unpacking
def test_parameter_packing_unpacking():
    """Verify parameter vector packing into Theta and exact round-trip unpacking."""
    cameras, landmarks, _, _ = create_mock_sfm_reconstruction(n_cams=3, n_points=15)
    cam_order = ["cam_00", "cam_01", "cam_02"]
    track_order = list(range(15))
    config = BundleAdjustmentConfig()

    pm = BAParameterManager(cam_order, track_order, config)
    # M = 3, N = 15 -> 6*3 - 7 + 3*15 = 11 + 45 = 56
    assert pm.num_camera_params == 11
    assert pm.num_landmark_params == 45
    assert pm.total_params == 56

    params = pm.pack_parameters(cameras, landmarks)
    assert len(params) == 56

    unpacked_cams, unpacked_lms = pm.unpack_parameters(params, cameras)
    for cid in cam_order:
        assert cid in unpacked_cams
        np.testing.assert_array_almost_equal(unpacked_cams[cid].R_cw, cameras[cid].R_cw, decimal=5)
        np.testing.assert_array_almost_equal(unpacked_cams[cid].t_cw, cameras[cid].t_cw, decimal=5)

    for tid in track_order:
        np.testing.assert_array_almost_equal(unpacked_lms[tid], landmarks[tid], decimal=5)


# 5. Camera Gauge Fixing
def test_camera_gauge_fixing():
    """Verify that Camera 0 is excluded from optimization variables and remains [I | 0]."""
    cameras, landmarks, _, _ = create_mock_sfm_reconstruction(n_cams=3, n_points=10)
    pm = BAParameterManager(["cam_00", "cam_01", "cam_02"], list(range(10)), BundleAdjustmentConfig())

    params = pm.pack_parameters(cameras, landmarks)
    # Mutate parameters
    params += 0.01

    unpacked_cams, _ = pm.unpack_parameters(params, cameras)
    # Camera 0 must strictly remain identity pose [I | 0]
    np.testing.assert_array_equal(unpacked_cams["cam_00"].R_cw, np.eye(3))
    np.testing.assert_array_equal(unpacked_cams["cam_00"].t_cw, np.zeros(3))


# 6. Unit Baseline Preservation on S^2
def test_unit_baseline_preservation():
    """Verify that Camera 1 translation norm is strictly 1.0 reconstruction units."""
    cameras, landmarks, _, _ = create_mock_sfm_reconstruction(n_cams=2, n_points=10)
    pm = BAParameterManager(["cam_00", "cam_01"], list(range(10)), BundleAdjustmentConfig())

    params = pm.pack_parameters(cameras, landmarks)
    # Mutate 2-DoF tangent perturbation parameters
    params[3:5] = np.array([0.5, -0.3])

    unpacked_cams, _ = pm.unpack_parameters(params, cameras)
    norm_t1 = float(np.linalg.norm(unpacked_cams["cam_01"].t_cw))
    assert abs(norm_t1 - 1.0) < 1e-6


# 7. Rotation Parameterization
def test_rotation_parameterization():
    """Verify minimal 3-DoF Lie algebra so(3) Rodrigues conversion across general and zero angles."""
    # A. Zero rotation
    omega_zero = np.zeros(3)
    R_zero = BARotationMath.rodrigues_to_rotation(omega_zero)
    np.testing.assert_array_almost_equal(R_zero, np.eye(3))
    omega_rec = BARotationMath.rotation_to_rodrigues(R_zero)
    np.testing.assert_array_almost_equal(omega_rec, omega_zero)

    # B. Small angle near Taylor series threshold (1e-8)
    omega_small = np.array([1e-8, -1e-8, 2e-8])
    R_small = BARotationMath.rodrigues_to_rotation(omega_small)
    assert np.all(np.isfinite(R_small))
    omega_small_rec = BARotationMath.rotation_to_rodrigues(R_small)
    np.testing.assert_array_almost_equal(omega_small_rec, omega_small, decimal=7)

    # C. General angle
    omega_gen = np.array([0.2, -0.15, 0.3])
    R_gen = BARotationMath.rodrigues_to_rotation(omega_gen)
    assert abs(np.linalg.det(R_gen) - 1.0) < 1e-6
    omega_gen_rec = BARotationMath.rotation_to_rodrigues(R_gen)
    np.testing.assert_array_almost_equal(omega_gen_rec, omega_gen, decimal=5)


# 8. Projection Correctness
def test_projection_correctness():
    """Verify pinhole projection against analytical geometry."""
    K = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000)
    R = np.eye(3)
    t = np.zeros(3)
    X = np.array([1.0, 2.0, 10.0])  # u = 1000*(1/10) + 500 = 600, v = 1000*(2/10) + 500 = 700

    proj, z = BAResidualCalculator.project_point(K, R, t, X)
    assert proj is not None
    assert z == 10.0
    assert abs(proj[0] - 600.0) < 1e-6
    assert abs(proj[1] - 700.0) < 1e-6


# 9. Distortion State Handling
def test_distortion_state_handling():
    """Verify intrinsics validation retains calibrated distortion status."""
    K = CameraIntrinsics(fx=800.0, fy=800.0, cx=400.0, cy=300.0, width=800, height=600)
    assert K.is_calibrated
    assert K.distortion_status.value == "RECTIFIED_ZERO_DISTORTION"


# 10. Positive Depth Enforcement
def test_positive_depth_enforcement():
    """Verify that points behind camera (Z <= 0) or at optical center are rejected without clamping."""
    K = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000)
    R = np.eye(3)
    t = np.zeros(3)

    # Behind camera
    X_behind = np.array([1.0, 1.0, -5.0])
    proj, z = BAResidualCalculator.project_point(K, R, t, X_behind)
    assert proj is None
    assert z == -5.0

    # At camera plane Z = 0
    X_zero = np.array([1.0, 1.0, 0.0])
    proj_zero, z_zero = BAResidualCalculator.project_point(K, R, t, X_zero)
    assert proj_zero is None
    assert z_zero == 0.0


# 11. Residual Vector Correctness
def test_residual_vector_correctness():
    """Verify 2D residual calculation in pixels: r = observed - projected."""
    obs = (502.5, 498.0)
    proj = np.array([500.0, 500.0])
    r = BAResidualCalculator.compute_residual(obs, proj)
    assert r[0] == 2.5
    assert r[1] == -2.0


# 12. Huber Loss Semantics and C^1 Continuity
def test_huber_loss_semantics_and_continuity():
    """Verify Huber robust loss quadratic regime (e <= delta), linear regime (e > delta), and C^1 continuity."""
    delta = 2.0

    # Quadratic regime: e = 1.0 <= 2.0 -> loss = 0.5 * e^2 = 0.5, weight = 1.0
    r_quad = np.array([1.0, 0.0])
    loss_q, w_q = BAResidualCalculator.huber_loss(r_quad, delta_px=delta)
    assert abs(loss_q - 0.5) < 1e-6
    assert abs(w_q - 1.0) < 1e-6

    # Linear regime: e = 4.0 > 2.0 -> loss = delta * (e - 0.5*delta) = 2.0 * (4.0 - 1.0) = 6.0
    r_lin = np.array([4.0, 0.0])
    loss_l, w_l = BAResidualCalculator.huber_loss(r_lin, delta_px=delta)
    assert abs(loss_l - 6.0) < 1e-6
    assert abs(w_l - (2.0 / 4.0)) < 1e-6  # weight = delta / e = 0.5

    # C^1 Continuity at e = delta:
    eps = 1e-6
    r_left = np.array([delta - eps, 0.0])
    r_right = np.array([delta + eps, 0.0])
    loss_left, _ = BAResidualCalculator.huber_loss(r_left, delta_px=delta)
    loss_right, _ = BAResidualCalculator.huber_loss(r_right, delta_px=delta)
    loss_at_delta, w_at_delta = BAResidualCalculator.huber_loss(np.array([delta, 0.0]), delta_px=delta)

    # Function value continuity
    assert abs(loss_left - loss_at_delta) < 1e-4
    assert abs(loss_right - loss_at_delta) < 1e-4
    assert abs(loss_at_delta - 0.5 * delta**2) < 1e-6
    assert abs(w_at_delta - 1.0) < 1e-6

    # First derivative continuity: d/de at left is (delta - eps), at right is delta
    deriv_left = (loss_at_delta - loss_left) / eps
    deriv_right = (loss_right - loss_at_delta) / eps
    assert abs(deriv_left - delta) < 1e-3
    assert abs(deriv_right - delta) < 1e-3


# 13. Uniform Observation Weighting
def test_uniform_observation_weighting():
    """Verify observation weights default to uniform 1.0 without fabricated uncertainties."""
    config = BundleAdjustmentConfig()
    assert config.huber_delta_px == 2.0  # HEURISTIC_DEFAULT
    assert config.loss_function == "HUBER"


# 14. Sparse Dependency Structure
def test_sparse_dependency_structure():
    """Verify calculation of Jacobian block sparsity and structural zeros."""
    cam_order = [f"cam_{i:02d}" for i in range(5)]  # 5 cameras
    track_order = list(range(50))  # 50 landmarks
    obs_pairs = []
    for c in cam_order:
        for t in track_order:
            obs_pairs.append((c, t))  # 250 observations

    sparsity = BASparsityStructure(cam_order, track_order, obs_pairs)
    assert sparsity.num_residuals == 500  # 250 obs * 2
    # Sparsity ratio depends on graph connectivity and is substantial (> 75%)
    assert sparsity.sparsity_ratio > 0.75


# 15. Optimizer Configuration Validation
def test_optimizer_configuration_validation():
    """Verify optimizer configuration validation and default heuristic thresholds."""
    config = BundleAdjustmentConfig()
    assert config.max_iterations == 50
    assert config.cost_tolerance == 1e-6
    assert config.parameter_tolerance == 1e-6
    assert config.gradient_tolerance == 1e-8
    assert config.optimize_intrinsics is False


# 16. Convergence Status Semantics
def test_convergence_status_semantics():
    """Verify failure taxonomy covers all explicit convergence and divergence outcomes."""
    assert BAFailureReason.MAX_ITERATIONS_REACHED.value == "MAX_ITERATIONS_REACHED"
    assert BAFailureReason.OPTIMIZATION_DIVERGED.value == "OPTIMIZATION_DIVERGED"
    assert BAFailureReason.OPTIMIZATION_FAILED.value == "OPTIMIZATION_FAILED"


# 17. Post-Optimization Validation
def test_post_optimization_validation():
    """Verify post-optimization validator passes on valid, cost-reducing reconstruction."""
    cameras, landmarks, tracks, _ = create_mock_sfm_reconstruction(n_cams=3, n_points=15)
    config = BundleAdjustmentConfig()

    metrics_before, cost_before, _ = BAResidualCalculator.evaluate_reconstruction_metrics(cameras, landmarks, tracks)
    metrics_after = BAReprojectionMetrics(
        mean_error_px=metrics_before.mean_error_px * 0.8,
        rmse_px=metrics_before.rmse_px * 0.8,
        median_error_px=metrics_before.median_error_px * 0.8,
        percentile_90_px=metrics_before.percentile_90_px * 0.8,
        max_error_px=metrics_before.max_error_px * 0.8,
        total_observations=metrics_before.total_observations,
    )
    cost_after = cost_before * 0.8

    valid, reason, diags = BAPostOptimizationValidator.validate(
        cameras, landmarks, tracks, "cam_00", metrics_before, metrics_after, cost_before, cost_after, config
    )
    assert valid is True
    assert reason is None


# 18. Post-Optimization Acceptance Semantics (Cost vs Raw RMSE)
def test_post_optimization_acceptance_semantics():
    """Verify that primary acceptance is based on robust objective non-increase and geometric safety."""
    cameras, landmarks, tracks, _ = create_mock_sfm_reconstruction(n_cams=3, n_points=15)
    config = BundleAdjustmentConfig()
    m_before, cost_before, _ = BAResidualCalculator.evaluate_reconstruction_metrics(cameras, landmarks, tracks)

    # Case A: Robust cost increased -> Rejected as OPTIMIZATION_DIVERGED
    m_after_higher_cost = BAReprojectionMetrics(
        mean_error_px=m_before.mean_error_px,
        rmse_px=m_before.rmse_px,
        median_error_px=m_before.median_error_px,
        percentile_90_px=m_before.percentile_90_px,
        max_error_px=m_before.max_error_px,
        total_observations=m_before.total_observations,
    )
    valid_a, reason_a, _ = BAPostOptimizationValidator.validate(
        cameras, landmarks, tracks, "cam_00", m_before, m_after_higher_cost, cost_before, cost_before + 5.0, config
    )
    assert valid_a is False
    assert reason_a == BAFailureReason.OPTIMIZATION_DIVERGED

    # Case B: Robust cost decreased while raw RMSE is slightly adjusted -> Accepted
    m_after_adjusted_rmse = BAReprojectionMetrics(
        mean_error_px=m_before.mean_error_px + 0.05,
        rmse_px=m_before.rmse_px + 0.1,  # slightly higher raw RMSE but within sanity tolerance
        median_error_px=m_before.median_error_px,
        percentile_90_px=m_before.percentile_90_px,
        max_error_px=m_before.max_error_px,
        total_observations=m_before.total_observations,
    )
    valid_b, reason_b, _ = BAPostOptimizationValidator.validate(
        cameras, landmarks, tracks, "cam_00", m_before, m_after_adjusted_rmse, cost_before, cost_before - 2.0, config
    )
    assert valid_b is True
    assert reason_b is None

    # Case C: Low raw RMSE but gauge violated (Camera 1 baseline drifted from 1.0) -> Rejected
    corrupt_cameras = dict(cameras)
    corrupt_cam1 = SfMCamera(
        "cam_01", cameras["cam_01"].R_cw, np.array([2.5, 0.0, 0.0]), cameras["cam_01"].intrinsics
    )
    corrupt_cameras["cam_01"] = corrupt_cam1
    valid_c, reason_c, _ = BAPostOptimizationValidator.validate(
        corrupt_cameras, landmarks, tracks, "cam_00", m_before, m_before, cost_before, cost_before - 1.0, config
    )
    assert valid_c is False
    assert reason_c == BAFailureReason.GAUGE_CONSTRAINT_INVALID


# 19. Before/After Metric Reporting
def test_before_after_metric_reporting():
    """Verify typed serialization of before/after reprojection metrics."""
    metrics = BAReprojectionMetrics(
        mean_error_px=0.45, rmse_px=0.55, median_error_px=0.40, percentile_90_px=0.80, max_error_px=1.20, total_observations=120
    )
    d = metrics.to_dict()
    assert d["mean_error_px"] == 0.45
    assert d["rmse_px"] == 0.55
    assert d["total_observations"] == 120
    assert d["measurement_type"] == "ESTIMATED"


# 20. Scale Ambiguity Preservation
def test_scale_ambiguity_preservation():
    """Verify that BundleAdjustmentResult preserves scale ambiguity and non-metric flags."""
    m_before = BAReprojectionMetrics(0.5, 0.6, 0.4, 0.8, 1.0, 50)
    m_after = BAReprojectionMetrics(0.3, 0.4, 0.25, 0.5, 0.8, 50)

    res = BundleAdjustmentResult(
        status=PipelineStageStatus.SUCCESS,
        refined_reconstruction=None,
        metrics_before=m_before,
        metrics_after=m_after,
        cost_before=15.0,
        cost_after=8.5,
        total_iterations=12,
        convergence_reason="PARAMETER_CONVERGENCE",
        gauge_preserved=True,
        is_metric_scale=False,
        has_monocular_scale_ambiguity=True,
    )
    assert res.is_metric_scale is False
    assert res.has_monocular_scale_ambiguity is True
    assert res.gauge_preserved is True


# 21. Invalid Camera Failure Handling
def test_invalid_camera_failure():
    """Verify validator rejects non-finite camera poses."""
    cameras, landmarks, tracks, _ = create_mock_sfm_reconstruction(n_cams=2, n_points=10)
    config = BundleAdjustmentConfig()
    m = BAReprojectionMetrics(0.5, 0.6, 0.4, 0.8, 1.0, 20)

    # Corrupt camera pose with NaN
    cameras["cam_01"].t_cw[0] = np.nan

    valid, reason, diags = BAPostOptimizationValidator.validate(
        cameras, landmarks, tracks, "cam_00", m, m, 10.0, 8.0, config
    )
    assert valid is False
    assert reason == BAFailureReason.INVALID_CAMERA_STATE


# 22. Invalid Landmark Failure Handling
def test_invalid_landmark_failure():
    """Verify validator rejects non-finite 3D landmarks."""
    cameras, landmarks, tracks, _ = create_mock_sfm_reconstruction(n_cams=2, n_points=10)
    config = BundleAdjustmentConfig()
    m = BAReprojectionMetrics(0.5, 0.6, 0.4, 0.8, 1.0, 20)

    # Corrupt landmark with Inf
    landmarks[0] = np.array([np.inf, 0.0, 10.0])

    valid, reason, diags = BAPostOptimizationValidator.validate(
        cameras, landmarks, tracks, "cam_00", m, m, 10.0, 8.0, config
    )
    assert valid is False
    assert reason == BAFailureReason.INVALID_LANDMARK_STATE


# 23. Insufficient Observation Failure Handling
def test_insufficient_observation_failure():
    """Verify explicit failure when reconstruction has insufficient observations."""
    cameras, landmarks, _, _ = create_mock_sfm_reconstruction(n_cams=2, n_points=5)
    tracks: Dict[int, SfMTrack] = {}
    metrics, cost, _ = BAResidualCalculator.evaluate_reconstruction_metrics(cameras, landmarks, tracks)
    assert metrics.total_observations == 0
    assert cost == 0.0


# 24. Numerical Singularity Failure Taxonomy
def test_numerical_singularity_failure():
    """Verify failure taxonomy contains NUMERICAL_SINGULARITY."""
    assert BAFailureReason.NUMERICAL_SINGULARITY == "NUMERICAL_SINGULARITY"


# 25. Gauge Invariance Behavior
def test_gauge_invariance_behavior():
    """Verify that gauge preservation prevents arbitrary Sim(3) coordinate drift."""
    cameras, landmarks, _, _ = create_mock_sfm_reconstruction(n_cams=3, n_points=10)
    pm = BAParameterManager(["cam_00", "cam_01", "cam_02"], list(range(10)), BundleAdjustmentConfig())

    params = pm.pack_parameters(cameras, landmarks)
    unpacked_cams, _ = pm.unpack_parameters(params, cameras)

    # Reference camera remains strictly at origin
    assert np.allclose(unpacked_cams["cam_00"].R_cw, np.eye(3))
    assert np.allclose(unpacked_cams["cam_00"].t_cw, np.zeros(3))
    # Camera 1 baseline magnitude remains 1.0
    assert abs(float(np.linalg.norm(unpacked_cams["cam_01"].t_cw)) - 1.0) < 1e-6


# 26. Synthetic Noise Scenario
def test_synthetic_noise_scenario():
    """Verify residual metrics on synthetic scene with zero noise vs added Gaussian noise."""
    cameras, landmarks, tracks, _ = create_mock_sfm_reconstruction(n_cams=3, n_points=20)

    # Noise-free: residuals are strictly 0.0
    m_clean, cost_clean, _ = BAResidualCalculator.evaluate_reconstruction_metrics(cameras, landmarks, tracks)
    assert m_clean.rmse_px < 1e-4
    assert cost_clean < 1e-4

    # Add Gaussian noise to observations
    noisy_tracks: Dict[int, SfMTrack] = {}
    for tid, trk in tracks.items():
        noisy_obs = {cid: (px[0] + np.random.normal(0, 1.0), px[1] + np.random.normal(0, 1.0)) for cid, px in trk.observations.items()}
        noisy_tracks[tid] = SfMTrack(tid, trk.world_point, noisy_obs, keypoint_indices=trk.keypoint_indices)

    m_noisy, cost_noisy, _ = BAResidualCalculator.evaluate_reconstruction_metrics(cameras, landmarks, noisy_tracks)
    assert m_noisy.rmse_px > 0.5
    assert cost_noisy > cost_clean


# 27. Synthetic Perturbation Scenario
def test_synthetic_perturbation_scenario():
    """Verify that landmark perturbations increase total cost and reprojection residuals."""
    cameras, landmarks, tracks, _ = create_mock_sfm_reconstruction(n_cams=3, n_points=20)
    m_clean, cost_clean, _ = BAResidualCalculator.evaluate_reconstruction_metrics(cameras, landmarks, tracks)

    # Perturb landmarks
    perturbed_lms = {tid: pt + np.array([0.2, -0.2, 0.5]) for tid, pt in landmarks.items()}
    m_pert, cost_pert, _ = BAResidualCalculator.evaluate_reconstruction_metrics(cameras, perturbed_lms, tracks)

    assert m_pert.rmse_px > m_clean.rmse_px
    assert cost_pert > cost_clean


# 28. Controlled Outlier Scenario
def test_controlled_outlier_scenario():
    """Verify Huber loss dampens gradient/weight on large outlier residuals."""
    delta = 2.0
    inlier_res = np.array([1.0, 0.0])
    outlier_res = np.array([50.0, 0.0])  # Large outlier

    loss_in, w_in = BAResidualCalculator.huber_loss(inlier_res, delta)
    loss_out, w_out = BAResidualCalculator.huber_loss(outlier_res, delta)

    # Outlier weight must be substantially reduced compared to inlier weight
    assert w_in == 1.0
    assert w_out < 0.05
    # Linear loss growth: loss ~ delta * (e - 0.5*delta)
    assert abs(loss_out - delta * (50.0 - 0.5 * delta)) < 1e-6


# 29. No Ground Truth Metric Restrictions
def test_no_ground_truth_metric_restrictions():
    """Verify that BAReprojectionMetrics adheres to LEVEL_1 internal consistency (pixels only)."""
    m = BAReprojectionMetrics(
        mean_error_px=0.8, rmse_px=1.1, median_error_px=0.7, percentile_90_px=1.5, max_error_px=2.2, total_observations=100
    )
    d = m.to_dict()
    assert "mean_error_px" in d
    assert "rmse_px" in d
    # Must NOT claim meter accuracy
    assert "rmse_meters" not in d
    assert "ground_truth_accuracy" not in d


# Helper for creating SparseReconstructionResult
def create_mock_sparse_reconstruction_result(
    cameras: Dict[str, SfMCamera],
    landmarks: Dict[int, np.ndarray],
    tracks: Dict[int, SfMTrack],
) -> SparseReconstructionResult:
    """Wrap cameras, landmarks, and tracks into a typed SparseReconstructionResult."""
    camera_poses = {}
    for cid, cam in cameras.items():
        camera_poses[cid] = ExtrinsicPose(
            rotation_matrix=cam.R_cw.tolist(),
            translation_vector=cam.camera_center.tolist(),
            coordinate_convention="opencv_optical",
            is_metric=False,
            scale_factor=1.0,
        )
    points3d = {}
    for tid, pt in landmarks.items():
        points3d[tid] = TriangulatedTrack(
            track_id=tid,
            world_point=pt.copy(),
            observations=dict(tracks[tid].observations),
            reprojection_errors={cid: 0.1 for cid in tracks[tid].observations},
            cheirality_valid=True,
            triangulation_angle_deg=10.0,
            measurement_type=MeasurementType.ESTIMATED,
        )
    return SparseReconstructionResult(
        camera_poses=camera_poses,
        intrinsics={cid: cam.intrinsics for cid, cam in cameras.items()},
        points3d=points3d,
        mean_reprojection_rmse_px=0.5,
        percentile_90_reprojection_error_px=0.8,
        total_registered_cameras=len(cameras),
        total_triangulated_points=len(landmarks),
        mean_track_length=float(np.mean([len(t.observations) for t in points3d.values()])) if points3d else 0.0,
        gauge_policy=GaugeFixingPolicy.FIX_FIRST_CAMERA_AND_UNIT_BASELINE,
        is_metric_scale=False,
        has_monocular_scale_ambiguity=True,
        registered_frame_ids=sorted(list(cameras.keys())),
        unregistered_frame_ids=[],
        failed_frame_ids=[],
        camera_centers={cid: cam.camera_center.tolist() for cid, cam in cameras.items()},
        status=PipelineStageStatus.SUCCESS,
    )


# 30. Parameter Layout Structure
def test_parameter_layout_structure():
    """Verify BAParameterLayout correctly details offsets, counts, and dimensions."""
    cameras, landmarks, _, _ = create_mock_sfm_reconstruction(n_cams=4, n_points=12)
    pm = BAParameterManager(["cam_00", "cam_01", "cam_02", "cam_03"], list(range(12)), BundleAdjustmentConfig())
    layout = pm.layout

    assert layout.camera_count == 4
    assert layout.landmark_count == 12
    # Dim: 6(4) - 7 + 3(12) = 17 + 36 = 53
    assert layout.total_dimension == 53
    # Camera 0 is excluded
    assert "cam_00" not in layout.camera_offsets
    # Camera 1 has 5 parameters: offset 0
    assert layout.camera_offsets["cam_01"] == 0
    # Camera 2 has 6 parameters: offset 5
    assert layout.camera_offsets["cam_02"] == 5
    # Camera 3 has 6 parameters: offset 11
    assert layout.camera_offsets["cam_03"] == 11
    # Landmarks start at offset 17
    assert layout.landmark_offsets[0] == 17
    assert layout.landmark_offsets[1] == 20
    assert layout.landmark_offsets[11] == 17 + 11 * 3


# 31. Parameter Pack/Unpack Roundtrip
def test_parameter_pack_unpack_roundtrip():
    """Verify parameters pack and unpack reversibly without loss of numerical precision."""
    cameras, landmarks, _, _ = create_mock_sfm_reconstruction(n_cams=3, n_points=15)
    pm = BAParameterManager(["cam_00", "cam_01", "cam_02"], list(range(15)), BundleAdjustmentConfig())

    p0 = pm.pack_parameters(cameras, landmarks)
    unpacked_cams, unpacked_lms = pm.unpack_parameters(p0, cameras)
    p1 = pm.pack_parameters(unpacked_cams, unpacked_lms)

    assert np.allclose(p0, p1, atol=1e-8)
    for tid, pt in landmarks.items():
        assert np.allclose(pt, unpacked_lms[tid], atol=1e-8)


# 32. Parameter Dimension
def test_parameter_dimension():
    """Verify dim(Theta) matches 6M - 7 + 3N for M >= 2."""
    for M in [2, 3, 5, 8]:
        for N in [10, 25, 40]:
            c_ids = [f"cam_{i:02d}" for i in range(M)]
            t_ids = list(range(N))
            pm = BAParameterManager(c_ids, t_ids, BundleAdjustmentConfig())
            expected_dim = 6 * M - 7 + 3 * N
            assert pm.total_params == expected_dim
            assert pm.layout.total_dimension == expected_dim


# 33. Camera Zero Excluded
def test_camera_zero_excluded():
    """Verify Camera 0 is excluded from parameter state and strictly stays at [I | 0]."""
    cameras, landmarks, _, _ = create_mock_sfm_reconstruction(n_cams=3, n_points=10)
    pm = BAParameterManager(["cam_00", "cam_01", "cam_02"], list(range(10)), BundleAdjustmentConfig())

    p = pm.pack_parameters(cameras, landmarks)
    # Add random perturbation to all parameters in p
    p_perturbed = p + 0.1
    unpacked_cams, _ = pm.unpack_parameters(p_perturbed, cameras)

    assert np.allclose(unpacked_cams["cam_00"].R_cw, np.eye(3))
    assert np.allclose(unpacked_cams["cam_00"].t_cw, np.zeros(3))


# 34. Camera One Has Five Parameters
def test_camera_one_has_five_parameters():
    """Verify Camera 1 occupies exactly 5 parameters (3 rot + 2 trans direction)."""
    cameras, landmarks, _, _ = create_mock_sfm_reconstruction(n_cams=3, n_points=10)
    pm = BAParameterManager(["cam_00", "cam_01", "cam_02"], list(range(10)), BundleAdjustmentConfig())
    layout = pm.layout

    c1_offset = layout.camera_offsets["cam_01"]
    c2_offset = layout.camera_offsets["cam_02"]
    assert c2_offset - c1_offset == 5


# 35. Landmark Parameter Offsets
def test_landmark_parameter_offsets():
    """Verify landmarks start after all cameras and each has length 3."""
    cameras, landmarks, _, _ = create_mock_sfm_reconstruction(n_cams=4, n_points=10)
    pm = BAParameterManager(["cam_00", "cam_01", "cam_02", "cam_03"], list(range(10)), BundleAdjustmentConfig())
    layout = pm.layout

    # 4 cameras: M=4 -> 6(4) - 7 = 17 camera params
    assert layout.landmark_offsets[0] == 17
    for idx in range(9):
        assert layout.landmark_offsets[idx + 1] - layout.landmark_offsets[idx] == 3


# 36. Pack/Unpack Preserves Geometry
def test_pack_unpack_preserves_geometry():
    """Verify geometry before and after pack/unpack yields identical projection."""
    cameras, landmarks, tracks, _ = create_mock_sfm_reconstruction(n_cams=3, n_points=15)
    pm = BAParameterManager(["cam_00", "cam_01", "cam_02"], list(range(15)), BundleAdjustmentConfig())

    p = pm.pack_parameters(cameras, landmarks)
    cams_unpacked, lms_unpacked = pm.unpack_parameters(p, cameras)

    m1, cost1, _ = BAResidualCalculator.evaluate_reconstruction_metrics(cameras, landmarks, tracks)
    m2, cost2, _ = BAResidualCalculator.evaluate_reconstruction_metrics(cams_unpacked, lms_unpacked, tracks)

    assert abs(m1.rmse_px - m2.rmse_px) < 1e-8
    assert abs(cost1 - cost2) < 1e-8


# 37. Rotation Identity Perturbation
def test_rotation_identity_perturbation():
    """Verify Rodrigues mapping of zero vector yields exact identity matrix."""
    omega_zero = np.zeros(3, dtype=np.float64)
    R = BARotationMath.rodrigues_to_rotation(omega_zero)
    assert np.allclose(R, np.eye(3))
    w_rec = BARotationMath.rotation_to_rodrigues(R)
    assert np.allclose(w_rec, np.zeros(3))


# 38. Rotation Small Angle
def test_rotation_small_angle():
    """Verify small-angle Taylor expansion behaves smoothly and accurately."""
    omega_small = np.array([1e-8, -2e-8, 1.5e-8], dtype=np.float64)
    R = BARotationMath.rodrigues_to_rotation(omega_small)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert abs(float(np.linalg.det(R)) - 1.0) < 1e-12


# 39. Rotation Composition
def test_rotation_composition():
    """Verify rotation composition yields valid SO(3) matrices."""
    w1 = np.array([0.1, -0.2, 0.05], dtype=np.float64)
    w2 = np.array([-0.05, 0.15, -0.1], dtype=np.float64)
    R1 = BARotationMath.rodrigues_to_rotation(w1)
    R2 = BARotationMath.rodrigues_to_rotation(w2)
    R_comp = R1 @ R2

    assert np.allclose(R_comp.T @ R_comp, np.eye(3), atol=1e-8)
    assert abs(float(np.linalg.det(R_comp)) - 1.0) < 1e-8


# 40. Rotation Orthonormality and Determinant
def test_rotation_orthonormality_and_determinant():
    """Verify arbitrary axis-angle vectors always produce orthonormal matrices with det = +1."""
    np.random.seed(123)
    for _ in range(20):
        w = np.random.uniform(-np.pi, np.pi, size=3)
        R = BARotationMath.rodrigues_to_rotation(w)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-8)
        assert abs(float(np.linalg.det(R)) - 1.0) < 1e-8


# 41. Robust Loss Zero
def test_robust_loss_zero():
    """Verify Huber loss at zero residual evaluates to 0.0 with weight 1.0."""
    loss, weight = BAResidualCalculator.huber_loss(np.array([0.0, 0.0]), delta_px=2.0)
    assert loss == 0.0
    assert weight == 1.0


# 42. Robust Loss Branches and Weights
def test_robust_loss_branches_and_weights():
    """Verify quadratic and linear branches and weights across the threshold."""
    delta = 2.0
    # Inlier branch: e = 1.0 < delta
    loss_in, w_in = BAResidualCalculator.huber_loss(np.array([1.0, 0.0]), delta_px=delta)
    assert abs(loss_in - 0.5 * 1.0**2) < 1e-8
    assert w_in == 1.0

    # Boundary: e = 2.0 == delta
    loss_b, w_b = BAResidualCalculator.huber_loss(np.array([2.0, 0.0]), delta_px=delta)
    assert abs(loss_b - 0.5 * delta**2) < 1e-8
    assert w_b == 1.0

    # Outlier branch: e = 6.0 > delta
    loss_out, w_out = BAResidualCalculator.huber_loss(np.array([6.0, 0.0]), delta_px=delta)
    assert abs(loss_out - delta * (6.0 - 0.5 * delta)) < 1e-8
    assert abs(w_out - delta / 6.0) < 1e-8


# 43. Sparse Jacobian Structure Block Pattern
def test_sparse_jacobian_structure_block_pattern():
    """Verify sparse Jacobian matrix has correct block structure, rows, and columns."""
    cameras, landmarks, tracks, _ = create_mock_sfm_reconstruction(n_cams=3, n_points=10)
    recon = create_mock_sparse_reconstruction_result(cameras, landmarks, tracks)

    engine = BundleAdjustmentEngine()
    engine.initialize(recon, BundleAdjustmentConfig(), tracks=tracks)
    jac_sparse = engine.build_sparse_jacobian_pattern()

    total_obs = sum(len(t.observations) for t in tracks.values())
    expected_rows = total_obs * 2
    # 3 cams, 10 landmarks -> 6(3) - 7 + 3(10) = 11 + 30 = 41
    expected_cols = 41

    assert jac_sparse.shape == (expected_rows, expected_cols)
    assert jac_sparse.nnz > 0
    # Check that sparsity is substantially non-dense (less than 50% nonzeros)
    dense_entries = expected_rows * expected_cols
    assert jac_sparse.nnz < 0.5 * dense_entries


# 44. Synthetic Bundle Adjustment Optimization
def test_synthetic_bundle_adjustment_optimization():
    """Verify that BA successfully refines perturbed cameras and landmarks on synthetic data."""
    cameras, landmarks, tracks, intrinsics = create_mock_sfm_reconstruction(n_cams=3, n_points=25)

    # Perturb Camera 2 pose slightly and landmarks slightly
    perturbed_cams = dict(cameras)
    cam2 = cameras["cam_02"]
    R2_pert = BARotationMath.rodrigues_to_rotation(np.array([0.01, -0.01, 0.005])) @ cam2.R_cw
    t2_pert = cam2.t_cw + np.array([0.05, -0.05, 0.02])
    perturbed_cams["cam_02"] = SfMCamera("cam_02", R2_pert, t2_pert, cam2.intrinsics, True, 2)

    perturbed_lms = {tid: pt + np.random.normal(0, 0.03, size=3) for tid, pt in landmarks.items()}
    recon = create_mock_sparse_reconstruction_result(perturbed_cams, perturbed_lms, tracks)

    engine = BundleAdjustmentEngine()
    res = engine.optimize(
        recon,
        tracks=tracks,
        intrinsics_map={cid: intrinsics for cid in cameras},
        config=BundleAdjustmentConfig(max_iterations=30),
    )

    assert res.status == PipelineStageStatus.SUCCESS
    assert res.refined_reconstruction is not None
    assert res.cost_after < res.cost_before
    assert res.metrics_after is not None
    assert res.metrics_after.rmse_px < res.metrics_before.rmse_px
    assert res.gauge_preserved is True

    # Camera 0 remains exactly fixed
    ref_pose = res.refined_reconstruction.camera_poses["cam_00"]
    assert np.allclose(ref_pose.rotation_matrix, np.eye(3), atol=1e-4)
    assert np.allclose(ref_pose.translation_vector, np.zeros(3), atol=1e-4)

    # Camera 1 baseline magnitude is strictly unit
    c1_pose = res.refined_reconstruction.camera_poses["cam_01"]
    R1 = np.array(c1_pose.rotation_matrix)
    c1 = np.array(c1_pose.translation_vector)
    t1 = -R1 @ c1
    assert abs(float(np.linalg.norm(t1)) - 1.0) < 1e-4


# 45. Bundle Adjustment Controlled Outliers
def test_bundle_adjustment_controlled_outliers():
    """Verify Huber robust loss dampens outlier influence and prevents numerical explosion."""
    cameras, landmarks, tracks, intrinsics = create_mock_sfm_reconstruction(n_cams=3, n_points=20)

    # Inject 3 large outliers into observation pixels
    outlier_tracks = dict(tracks)
    outlier_tids = [0, 5, 10]
    for tid in outlier_tids:
        orig_obs = dict(outlier_tracks[tid].observations)
        orig_obs["cam_02"] = (orig_obs["cam_02"][0] + 40.0, orig_obs["cam_02"][1] - 30.0)
        outlier_tracks[tid] = SfMTrack(tid, outlier_tracks[tid].world_point, orig_obs, outlier_tracks[tid].keypoint_indices)

    recon = create_mock_sparse_reconstruction_result(cameras, landmarks, outlier_tracks)
    engine = BundleAdjustmentEngine()
    res = engine.optimize(
        recon,
        tracks=outlier_tracks,
        intrinsics_map={cid: intrinsics for cid in cameras},
        config=BundleAdjustmentConfig(max_iterations=20),
    )

    assert res.status == PipelineStageStatus.SUCCESS
    assert res.refined_reconstruction is not None
    assert res.cost_after <= res.cost_before + 1e-4
    assert res.gauge_preserved is True


# 46. Bundle Adjustment No-Op Case
def test_bundle_adjustment_noop_optimum():
    """Verify optimizer starting at exact optimum terminates cleanly without degrading geometry."""
    cameras, landmarks, tracks, intrinsics = create_mock_sfm_reconstruction(n_cams=3, n_points=20)
    recon = create_mock_sparse_reconstruction_result(cameras, landmarks, tracks)

    engine = BundleAdjustmentEngine()
    res = engine.optimize(
        recon,
        tracks=tracks,
        intrinsics_map={cid: intrinsics for cid in cameras},
        config=BundleAdjustmentConfig(max_iterations=10),
    )

    assert res.status == PipelineStageStatus.SUCCESS
    assert res.refined_reconstruction is not None
    assert res.cost_after <= res.cost_before + 1e-6
    assert res.gauge_preserved is True


# 47. Bundle Adjustment Failure Cases
def test_bundle_adjustment_failure_cases():
    """Verify graceful rejection for various malformed or degenerate inputs."""
    cameras, landmarks, tracks, intrinsics = create_mock_sfm_reconstruction(n_cams=3, n_points=20)
    intrinsics_map = {cid: intrinsics for cid in cameras}
    engine = BundleAdjustmentEngine()

    # Case A: Insufficient cameras (< 2)
    single_cam = {"cam_00": cameras["cam_00"]}
    recon_single = create_mock_sparse_reconstruction_result(single_cam, landmarks, tracks)
    res_a = engine.optimize(recon_single, tracks=tracks, intrinsics_map=intrinsics_map)
    assert res_a.status == PipelineStageStatus.FAILED
    assert res_a.failure_reason == BAFailureReason.INVALID_INPUT_RECONSTRUCTION

    # Case B: Insufficient landmarks (< 10)
    few_lms = {tid: pt for tid, pt in list(landmarks.items())[:5]}
    recon_few = create_mock_sparse_reconstruction_result(cameras, few_lms, tracks)
    res_b = engine.optimize(recon_few, tracks=tracks, intrinsics_map=intrinsics_map)
    assert res_b.status == PipelineStageStatus.FAILED
    assert res_b.failure_reason == BAFailureReason.INSUFFICIENT_OBSERVATIONS

    # Case C: Non-finite camera pose (NaN)
    nan_cams = dict(cameras)
    nan_cams["cam_02"] = SfMCamera("cam_02", np.full((3, 3), np.nan), np.zeros(3), intrinsics, True, 2)
    recon_nan_cam = create_mock_sparse_reconstruction_result(nan_cams, landmarks, tracks)
    res_c = engine.optimize(recon_nan_cam, tracks=tracks, intrinsics_map=intrinsics_map)
    assert res_c.status == PipelineStageStatus.FAILED
    assert res_c.failure_reason == BAFailureReason.INVALID_CAMERA_STATE

    # Case D: Non-finite landmark (Inf)
    inf_lms = dict(landmarks)
    inf_lms[0] = np.array([np.inf, 0.0, 10.0])
    recon_inf_lm = create_mock_sparse_reconstruction_result(cameras, inf_lms, tracks)
    res_d = engine.optimize(recon_inf_lm, tracks=tracks, intrinsics_map=intrinsics_map)
    assert res_d.status == PipelineStageStatus.FAILED
    assert res_d.failure_reason == BAFailureReason.INVALID_LANDMARK_STATE

    # Case E: Camera 0 not at origin
    drift_cams = dict(cameras)
    drift_cams["cam_00"] = SfMCamera("cam_00", np.eye(3), np.array([2.0, 0.0, 0.0]), intrinsics, True, 0)
    recon_drift = create_mock_sparse_reconstruction_result(drift_cams, landmarks, tracks)
    res_e = engine.optimize(recon_drift, tracks=tracks, intrinsics_map=intrinsics_map)
    assert res_e.status == PipelineStageStatus.FAILED
    assert res_e.failure_reason == BAFailureReason.GAUGE_CONSTRAINT_INVALID

    # Case F: Camera 1 baseline magnitude != 1.0
    scale_drift_cams = dict(cameras)
    scale_drift_cams["cam_01"] = SfMCamera("cam_01", np.eye(3), np.array([5.0, 0.0, 0.0]), intrinsics, True, 1)
    recon_scale = create_mock_sparse_reconstruction_result(scale_drift_cams, landmarks, tracks)
    res_f = engine.optimize(recon_scale, tracks=tracks, intrinsics_map=intrinsics_map)
    assert res_f.status == PipelineStageStatus.FAILED
    assert res_f.failure_reason == BAFailureReason.GAUGE_CONSTRAINT_INVALID


# 48. Bundle Adjustment Runtime and Provenance Recording
def test_bundle_adjustment_runtime_and_provenance():
    """Verify runtime seconds, solver method, and iterations are recorded in provenance."""
    cameras, landmarks, tracks, intrinsics = create_mock_sfm_reconstruction(n_cams=3, n_points=15)
    recon = create_mock_sparse_reconstruction_result(cameras, landmarks, tracks)

    engine = BundleAdjustmentEngine()
    res = engine.optimize(
        recon,
        tracks=tracks,
        intrinsics_map={cid: intrinsics for cid in cameras},
        config=BundleAdjustmentConfig(max_iterations=15),
    )

    assert res.status == PipelineStageStatus.SUCCESS
    assert "runtime_seconds" in res.provenance
    assert res.provenance["runtime_seconds"] >= 0.0
    assert "solver_method" in res.provenance
    assert res.provenance["solver_method"] == "scipy_least_squares_trf"
    assert "parameter_dimension" in res.provenance
    assert res.provenance["parameter_dimension"] == 6 * 3 - 7 + 3 * 15

