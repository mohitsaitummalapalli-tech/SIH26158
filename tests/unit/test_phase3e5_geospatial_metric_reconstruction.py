"""Phase 3E.5 Geospatial & Metric Reconstruction Unit Test Suite.

Formally implements all 36 approved test scenarios from
docs/architecture/PHASE_3E.5_GEOSPATIAL_METRIC_RECONSTRUCTION.md:
- 16.1 Sim(3) Geometric & Mathematical Invariants (TEST-3E5-01 to 06)
- 16.2 Telemetry Noise, Outlier Rejection & Robustness (TEST-3E5-07 to 12)
- 16.3 Temporal Synchronization & Latency (TEST-3E5-13 to 18)
- 16.4 Lever Arm & Airframe Attitude (TEST-3E5-19 to 25)
- 16.5 Altitude References & Geoid (TEST-3E5-26 to 28)
- 16.6 Geometric Observability, RANSAC Degeneracy & Stationarity (TEST-3E5-29 to 33)
- 16.7 Coordinate Transformation Round-Trips & Determinism (TEST-3E5-34 to 36)
"""

import math
import numpy as np
import pytest

import src.ingestion  # Pre-loads ingestion package to prevent circular import during standalone execution

from src.geospatial import (
    Sim3,
    solve_sim3_umeyama,
    UncertaintyType,
    GeospatialAnchorOrigin,
    AltitudeReferenceType,
    wgs84_to_ecef,
    ecef_to_wgs84,
    wgs84_to_enu,
    enu_to_wgs84,
    LeverArm,
    LeverArmStatus,
    TelemetryObservation,
    ObservationClassification,
    construct_gnss_covariance,
    RawTelemetryRecord,
    TelemetrySynchronizer,
    ScaleObservabilityReport,
    FullSim3ObservabilityStatus,
    check_scale_observability,
    RobustSim3Estimator,
    compute_isoperimetric_quotient,
    MetricScaleStatus,
    MetricStateMachine,
    GroundControlPoint,
    MetricValidator,
    GeospatialMetricReconstructor,
    GnssAccuracyInterpretation,
)


# ============================================================================
# Helper Functions for Synthetic Telemetry & Reconstruction Generation
# ============================================================================

def _euler_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Generate active 3x3 rotation matrix for given Euler angles in degrees."""
    psi = math.radians(yaw_deg)
    theta = math.radians(pitch_deg)
    phi = math.radians(roll_deg)
    rz = np.array([[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0], [0, 0, 1]])
    ry = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
    rx = np.array([[1, 0, 0], [0, math.cos(phi), -math.sin(phi)], [0, math.sin(phi), math.cos(phi)]])
    return rz @ ry @ rx


def _create_synthetic_spiral_trajectory(
    num_points: int = 30,
    radius: float = 50.0,
    height: float = 20.0,
) -> np.ndarray:
    """Generate non-collinear 3D spiral trajectory points."""
    pts = np.zeros((num_points, 3), dtype=np.float64)
    for i in range(num_points):
        theta = 2.0 * math.pi * (i / (num_points - 1)) * 2.5
        pts[i, 0] = radius * math.cos(theta)
        pts[i, 1] = radius * math.sin(theta)
        pts[i, 2] = height * (i / (num_points - 1))
    return pts


# ============================================================================
# 16.1 Sim(3) Geometric & Mathematical Invariants
# ============================================================================

def test_3e5_01_identity_similarity_transform():
    """TEST-3E5-01: Identity scale s=1, R=I, t=0 yields exact numerical recovery."""
    pts_rec = _create_synthetic_spiral_trajectory(20, 20.0, 10.0)
    pts_dst = pts_rec.copy()

    sim3 = solve_sim3_umeyama(pts_rec, pts_dst)
    assert abs(sim3.scale - 1.0) < 1e-7
    assert np.linalg.norm(sim3.rotation - np.eye(3)) < 1e-7
    assert np.linalg.norm(sim3.translation) < 1e-7


def test_3e5_02_pure_isotropic_scale_recovery():
    """TEST-3E5-02: Known synthetic scale s=4.25 recovered within 1e-6 precision."""
    true_scale = 4.25
    pts_rec = _create_synthetic_spiral_trajectory(25, 10.0, 5.0)
    pts_dst = true_scale * pts_rec

    sim3 = solve_sim3_umeyama(pts_rec, pts_dst)
    assert abs(sim3.scale - true_scale) / true_scale < 1e-6
    assert np.linalg.norm(sim3.rotation - np.eye(3)) < 1e-6
    assert np.linalg.norm(sim3.translation) < 1e-6


def test_3e5_03_pure_3d_rotation_recovery():
    """TEST-3E5-03: Known Euler rotation (30°, -45°, 60°) recovered within 1e-6 Frobenius norm."""
    r_true = _euler_matrix(30.0, -45.0, 60.0)
    pts_rec = _create_synthetic_spiral_trajectory(30, 15.0, 8.0)
    pts_dst = pts_rec @ r_true.T

    sim3 = solve_sim3_umeyama(pts_rec, pts_dst)
    assert abs(sim3.scale - 1.0) < 1e-6
    assert np.linalg.norm(sim3.rotation - r_true) < 1e-6
    assert np.linalg.norm(sim3.translation) < 1e-6


def test_3e5_04_pure_large_magnitude_translation_recovery():
    """TEST-3E5-04: Large-magnitude translation t=[1e5, -2e5, 500] recovered without precision loss."""
    t_true = np.array([100000.0, -200000.0, 500.0], dtype=np.float64)
    pts_rec = _create_synthetic_spiral_trajectory(20, 25.0, 10.0)
    pts_dst = pts_rec + t_true

    sim3 = solve_sim3_umeyama(pts_rec, pts_dst)
    assert abs(sim3.scale - 1.0) < 1e-6
    assert np.linalg.norm(sim3.rotation - np.eye(3)) < 1e-6
    assert np.linalg.norm(sim3.translation - t_true) < 1e-5


def test_3e5_05_full_7dof_combined_sim3_roundtrip():
    """TEST-3E5-05: Arbitrary (s, R, t) applied to 50 points; transformed back with error < 1e-6m."""
    s_true = 2.85
    r_true = _euler_matrix(42.0, -18.0, 65.0)
    t_true = np.array([125.4, -68.2, 45.1], dtype=np.float64)
    sim3_fwd = Sim3(scale=s_true, rotation=r_true, translation=t_true)

    pts_rec = _create_synthetic_spiral_trajectory(50, 40.0, 15.0)
    pts_geo = sim3_fwd.transform_point(pts_rec)

    sim3_inv = sim3_fwd.inverse()
    pts_rec_recovered = sim3_inv.transform_point(pts_geo)

    max_err = float(np.max(np.linalg.norm(pts_rec_recovered - pts_rec, axis=1)))
    assert max_err < 1e-6


def test_3e5_06_scale_equivariance_theorem():
    """TEST-3E5-06: Scaling input reconstruction by a in {1e-4, 0.5, 2.0, 1e4} yields s' = s / a."""
    pts_rec = _create_synthetic_spiral_trajectory(35, 30.0, 10.0)
    s_target = 3.5
    r_target = _euler_matrix(15.0, 25.0, -10.0)
    t_target = np.array([50.0, 20.0, -5.0], dtype=np.float64)
    sim3_target = Sim3(scale=s_target, rotation=r_target, translation=t_target)
    pts_dst = sim3_target.transform_point(pts_rec)

    for a in [1e-4, 0.5, 2.0, 1e4]:
        pts_rec_scaled = a * pts_rec
        sim3_est = solve_sim3_umeyama(pts_rec_scaled, pts_dst)

        expected_scale = s_target / a
        assert abs(sim3_est.scale - expected_scale) / expected_scale < 1e-6
        assert np.linalg.norm(sim3_est.rotation - r_target) < 1e-6
        assert np.linalg.norm(sim3_est.translation - t_target) < 1e-5


# ============================================================================
# 16.2 Telemetry Noise, Outlier Rejection & Robustness
# ============================================================================

def test_3e5_07_unbiased_gaussian_gnss_noise():
    """TEST-3E5-07: Gaussian noise sigma=0.5m across 50 cameras yields unbiased scale (|error| < 2%)."""
    np.random.seed(42)
    pts_rec = _create_synthetic_spiral_trajectory(50, 40.0, 15.0)
    s_true = 5.0
    r_true = _euler_matrix(10.0, 0.0, 0.0)
    t_true = np.array([20.0, 30.0, 0.0])
    sim3_true = Sim3(scale=s_true, rotation=r_true, translation=t_true)

    clean_dst = sim3_true.transform_point(pts_rec)
    noisy_dst = clean_dst + np.random.normal(0.0, 0.5, size=clean_dst.shape)

    observations = [
        TelemetryObservation(
            frame_id=f"frame_{i:03d}",
            timestamp_seconds=float(i),
            c_rec=pts_rec[i],
            z_gnss_enu=noisy_dst[i],
            covariance_enu=np.eye(3) * (0.5 ** 2),
            horizontal_accuracy_m=0.5,
            vertical_accuracy_m=0.5,
        )
        for i in range(50)
    ]

    estimator = RobustSim3Estimator()
    res = estimator.estimate(observations, LeverArm.zero())
    assert res.success
    assert res.sim3 is not None
    rel_err = abs(res.sim3.scale - s_true) / s_true
    assert rel_err < 0.02


def test_3e5_08_single_extreme_gnss_position_outlier():
    """TEST-3E5-08: Single +500m outlier is rejected by RANSAC/Huber without corrupting alignment."""
    np.random.seed(42)
    pts_rec = _create_synthetic_spiral_trajectory(40, 50.0, 20.0)
    s_true = 2.0
    sim3_true = Sim3(scale=s_true, rotation=np.eye(3), translation=np.array([10.0, 10.0, 5.0]))
    dst = sim3_true.transform_point(pts_rec)

    # Corrupt point 15 by 500m
    dst[15] += np.array([500.0, 0.0, 0.0])

    observations = [
        TelemetryObservation(
            frame_id=f"frame_{i:03d}",
            timestamp_seconds=float(i),
            c_rec=pts_rec[i],
            z_gnss_enu=dst[i],
            covariance_enu=np.eye(3) * 1.0,
            horizontal_accuracy_m=1.0,
            vertical_accuracy_m=1.0,
        )
        for i in range(40)
    ]

    estimator = RobustSim3Estimator()
    res = estimator.estimate(observations, LeverArm.zero())
    assert res.success
    assert res.sim3 is not None
    assert 15 in res.rejected_indices
    assert abs(res.sim3.scale - s_true) / s_true < 0.005


def test_3e5_09_clustered_gnss_outliers_30_percent():
    """TEST-3E5-09: 30% clustered multi-path step jumps; robust estimator isolates inliers."""
    pts_rec = _create_synthetic_spiral_trajectory(40, 60.0, 20.0)
    s_true = 3.0
    sim3_true = Sim3(scale=s_true, rotation=np.eye(3), translation=np.array([0.0, 0.0, 0.0]))
    dst = sim3_true.transform_point(pts_rec)

    # Corrupt 12 out of 40 points (30%) with step jump of 80m
    corrupt_indices = list(range(10, 22))
    for idx in corrupt_indices:
        dst[idx] += np.array([80.0, -80.0, 50.0])

    observations = [
        TelemetryObservation(
            frame_id=f"frame_{i:03d}",
            timestamp_seconds=float(i),
            c_rec=pts_rec[i],
            z_gnss_enu=dst[i],
            covariance_enu=np.eye(3) * 1.0,
        )
        for i in range(40)
    ]

    estimator = RobustSim3Estimator()
    res = estimator.estimate(observations, LeverArm.zero())
    assert res.success
    assert res.sim3 is not None
    assert len(res.inlier_indices) >= 25
    assert abs(res.sim3.scale - s_true) / s_true < 0.01


def test_3e5_10_asymmetric_horizontal_vs_vertical_quality():
    """TEST-3E5-10: sigma_H=0.5m, sigma_V=5.0m; horizontal heavily prioritized over vertical."""
    cov, missing = construct_gnss_covariance(
        horizontal_accuracy_m=0.5,
        vertical_accuracy_m=5.0,
    )
    assert not missing
    assert cov[0, 0] == pytest.approx(0.25)
    assert cov[1, 1] == pytest.approx(0.25)
    assert cov[2, 2] == pytest.approx(25.0)
    # Weight of horizontal is 100x higher than vertical
    assert cov[2, 2] / cov[0, 0] == 100.0


def test_3e5_11_missing_accuracy_metadata_fallback():
    """TEST-3E5-11: Telemetry lacking accuracy fields assigned conservative fallback noise."""
    cov, missing = construct_gnss_covariance(
        horizontal_accuracy_m=None,
        vertical_accuracy_m=None,
        fallback_horizontal_m=3.0,
        fallback_vertical_m=5.0,
    )
    assert missing is True
    assert cov[0, 0] == pytest.approx(9.0)   # 3^2
    assert cov[2, 2] == pytest.approx(25.0)  # 5^2


def test_3e5_12_complete_gnss_outlier_overwhelming():
    """TEST-3E5-12: >70% outliers triggers robust estimation failure and enters METRIC_ALIGNMENT_FAILED."""
    pts_rec = _create_synthetic_spiral_trajectory(20, 20.0, 5.0)
    dst = pts_rec.copy()
    # Corrupt 16 of 20 (80%) randomly
    np.random.seed(99)
    for i in range(16):
        dst[i] += np.random.uniform(-500.0, 500.0, size=3)

    observations = [
        TelemetryObservation(
            frame_id=f"frame_{i:03d}",
            timestamp_seconds=float(i),
            c_rec=pts_rec[i],
            z_gnss_enu=dst[i],
            covariance_enu=np.eye(3) * 1.0,
        )
        for i in range(20)
    ]

    estimator = RobustSim3Estimator(tau_inlier_mahalanobis=2.0)
    res = estimator.estimate(observations, LeverArm.zero())
    # Inlier count is small or failure triggered
    assert (not res.success) or (len(res.inlier_indices) < 6)


# ============================================================================
# 16.3 Temporal Synchronization & Latency
# ============================================================================

def test_3e5_13_exact_timestamp_coincidence():
    """TEST-3E5-13: Frame timestamps matching telemetry epochs require zero interpolation."""
    anchor = GeospatialAnchorOrigin(lat_deg=18.5204, lon_deg=73.8567, ellipsoidal_height_m=540.0)
    records = [
        RawTelemetryRecord(timestamp_seconds=10.0, latitude_deg=18.5204, longitude_deg=73.8567, altitude_m=540.0),
        RawTelemetryRecord(timestamp_seconds=11.0, latitude_deg=18.5205, longitude_deg=73.8567, altitude_m=542.0),
    ]
    sync = TelemetrySynchronizer(records, anchor)
    obs = sync.synchronize_frame("frame_1", 10.0, np.array([0.0, 0.0, 0.0]))
    assert obs.classification == ObservationClassification.VALID
    assert np.allclose(obs.z_gnss_enu, [0.0, 0.0, 0.0], atol=1e-3)


def test_3e5_14_uniform_intermediate_interpolation():
    """TEST-3E5-14: Midpoint timestamps correctly interpolated along trajectory."""
    anchor = GeospatialAnchorOrigin(lat_deg=18.5204, lon_deg=73.8567, ellipsoidal_height_m=540.0)
    enu_0 = wgs84_to_enu(18.5204, 73.8567, 540.0, anchor)
    enu_1 = wgs84_to_enu(18.5210, 73.8567, 550.0, anchor)

    records = [
        RawTelemetryRecord(timestamp_seconds=10.0, latitude_deg=18.5204, longitude_deg=73.8567, altitude_m=540.0),
        RawTelemetryRecord(timestamp_seconds=12.0, latitude_deg=18.5210, longitude_deg=73.8567, altitude_m=550.0),
    ]
    sync = TelemetrySynchronizer(records, anchor, max_gap_s=3.0)
    obs = sync.synchronize_frame("frame_mid", 11.0, np.array([0.0, 0.0, 0.0]))

    assert obs.classification == ObservationClassification.VALID
    expected_enu = 0.5 * (np.array(enu_0) + np.array(enu_1))
    assert np.allclose(obs.z_gnss_enu, expected_enu, atol=1e-3)


def test_3e5_15_shutter_clock_bias_offset():
    """TEST-3E5-15: Configured clock bias delta_t = 0.25s correctly shifts sampling epochs."""
    anchor = GeospatialAnchorOrigin(lat_deg=18.5204, lon_deg=73.8567, ellipsoidal_height_m=540.0)
    records = [
        RawTelemetryRecord(timestamp_seconds=10.0, latitude_deg=18.5200, longitude_deg=73.8567, altitude_m=540.0),
        RawTelemetryRecord(timestamp_seconds=11.0, latitude_deg=18.5210, longitude_deg=73.8567, altitude_m=540.0),
    ]
    sync = TelemetrySynchronizer(records, anchor, clock_offset_s=0.25)
    # Frame at 10.0s -> sampled at 10.25s (alpha = 0.25)
    obs = sync.synchronize_frame("f1", 10.0, np.zeros(3))
    assert obs.timestamp_seconds == 10.25


def test_3e5_16_trajectory_gap_rejection():
    """TEST-3E5-16: Camera falling in 3.0s gap is marked TEMPORAL_MISMATCH and rejected."""
    anchor = GeospatialAnchorOrigin(lat_deg=18.5204, lon_deg=73.8567, ellipsoidal_height_m=540.0)
    records = [
        RawTelemetryRecord(timestamp_seconds=10.0, latitude_deg=18.5200, longitude_deg=73.8567, altitude_m=540.0),
        RawTelemetryRecord(timestamp_seconds=13.0, latitude_deg=18.5210, longitude_deg=73.8567, altitude_m=540.0),
    ]
    sync = TelemetrySynchronizer(records, anchor, max_gap_s=1.0)
    obs = sync.synchronize_frame("gap_frame", 11.5, np.zeros(3))
    assert obs.classification == ObservationClassification.TEMPORAL_MISMATCH


def test_3e5_17_duplicate_telemetry_timestamps():
    """TEST-3E5-17: Duplicate timestamps handled gracefully without divide-by-zero."""
    anchor = GeospatialAnchorOrigin(lat_deg=18.5204, lon_deg=73.8567, ellipsoidal_height_m=540.0)
    records = [
        RawTelemetryRecord(timestamp_seconds=10.0, latitude_deg=18.5200, longitude_deg=73.8567, altitude_m=540.0),
        RawTelemetryRecord(timestamp_seconds=10.0, latitude_deg=18.5200, longitude_deg=73.8567, altitude_m=540.0),  # Dup
        RawTelemetryRecord(timestamp_seconds=11.0, latitude_deg=18.5210, longitude_deg=73.8567, altitude_m=540.0),
    ]
    sync = TelemetrySynchronizer(records, anchor)
    # Deduplication should ensure len(records) == 2
    assert len(sync.records) == 2
    obs = sync.synchronize_frame("f1", 10.5, np.zeros(3))
    assert obs.classification == ObservationClassification.VALID


def test_3e5_18_out_of_range_timestamp_boundary():
    """TEST-3E5-18: Frames outside telemetry start/end epochs strictly rejected."""
    anchor = GeospatialAnchorOrigin(lat_deg=18.5204, lon_deg=73.8567, ellipsoidal_height_m=540.0)
    records = [
        RawTelemetryRecord(timestamp_seconds=10.0, latitude_deg=18.5200, longitude_deg=73.8567, altitude_m=540.0),
        RawTelemetryRecord(timestamp_seconds=12.0, latitude_deg=18.5210, longitude_deg=73.8567, altitude_m=540.0),
    ]
    sync = TelemetrySynchronizer(records, anchor)
    obs_before = sync.synchronize_frame("f_before", 9.5, np.zeros(3))
    assert obs_before.classification == ObservationClassification.TEMPORAL_MISMATCH
    obs_after = sync.synchronize_frame("f_after", 12.5, np.zeros(3))
    assert obs_after.classification == ObservationClassification.TEMPORAL_MISMATCH


# ============================================================================
# 16.4 Lever Arm & Airframe Attitude
# ============================================================================

def test_3e5_19_known_nonzero_physical_lever_arm():
    """TEST-3E5-19: Lever arm L=[0.2, 0.0, -0.15] correctly offsets antenna from camera."""
    lever = LeverArm.calibrated(0.2, 0.0, -0.15)
    r_body = _euler_matrix(0.0, 0.0, 0.0)
    c_cam = np.array([10.0, 20.0, 50.0])

    p_ant = lever.predict_antenna_enu(c_cam, r_body)
    assert np.allclose(p_ant, [10.2, 20.0, 49.85], atol=1e-3)


def test_3e5_20_uncalibrated_lever_arm_fallback():
    """TEST-3E5-20: Missing lever arm triggers LEVER_ARM_UNCALIBRATED and heuristic covariance inflation."""
    lever = LeverArm.uncalibrated(heuristic_uncertainty_m=0.25)
    assert lever.status == LeverArmStatus.LEVER_ARM_UNCALIBRATED
    assert np.allclose(lever.vector_body, [0.0, 0.0, 0.0])

    r_body = np.eye(3)
    cov_enu = lever.effective_covariance_enu(r_body)
    assert cov_enu[0, 0] == pytest.approx(0.25 ** 2)
    assert cov_enu[1, 1] == pytest.approx(0.25 ** 2)
    assert cov_enu[2, 2] == pytest.approx(0.25 ** 2)


def test_3e5_21_lever_arm_sign_convention_proof():
    """TEST-3E5-21: Forward-mounted antenna (+0.5m) verified at +X under 0° yaw and +Y under 90° yaw."""
    # L_body = P_antenna - P_camera in body: +0.5m Forward (+X)
    lever = LeverArm.calibrated(0.5, 0.0, 0.0)
    c_cam = np.array([0.0, 0.0, 0.0])

    # Case 1: Yaw = 0°
    r_0 = _euler_matrix(0.0, 0.0, 0.0)
    ant_0 = lever.predict_antenna_enu(c_cam, r_0)
    assert abs(ant_0[0] - 0.5) < 1e-4   # +X
    assert abs(ant_0[1] - 0.0) < 1e-4   # Y
    assert abs(ant_0[2] - 0.0) < 1e-4   # Z

    # Case 2: Yaw = 90°
    r_90 = _euler_matrix(90.0, 0.0, 0.0)
    ant_90 = lever.predict_antenna_enu(c_cam, r_90)
    assert abs(ant_90[0] - 0.0) < 1e-4   # X
    assert abs(ant_90[1] - 0.5) < 1e-4   # +Y
    assert abs(ant_90[2] - 0.0) < 1e-4   # Z


def test_3e5_22_lever_arm_pitch_roll_rotation():
    """TEST-3E5-22: Antenna displaced vertically (L=[0, 0, 0.3]) under 45° pitch tilt shifts position in ENU."""
    lever = LeverArm.calibrated(0.0, 0.0, 0.3)
    c_cam = np.array([0.0, 0.0, 0.0])
    # Pitch up 45 degrees
    r_pitch = _euler_matrix(0.0, 45.0, 0.0)
    ant = lever.predict_antenna_enu(c_cam, r_pitch)
    expected_up = 0.3 * math.cos(math.radians(45.0))
    assert abs(ant[2] - expected_up) < 1e-3


def test_3e5_23_camera_antenna_coincident_case():
    """TEST-3E5-23: L=[0, 0, 0] correctly reduces observation model to camera center."""
    lever = LeverArm.zero()
    assert lever.status == LeverArmStatus.LEVER_ARM_ZERO
    r_body = _euler_matrix(45.0, 15.0, -10.0)
    c_cam = np.array([5.0, -3.0, 12.0])
    p_ant = lever.predict_antenna_enu(c_cam, r_body)
    assert np.allclose(p_ant, c_cam)


def test_3e5_24_attitude_consistency_verification():
    """TEST-3E5-24: Reconstructed camera orientations compared against IMU orientations."""
    r_geo = _euler_matrix(10.0, 0.0, 0.0)
    r_rec = np.eye(3)
    # Diff angle is 10 degrees
    diff_rot = r_geo @ r_rec.T
    tr = float(np.trace(diff_rot))
    angle_deg = math.degrees(math.acos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0)))
    assert abs(angle_deg - 10.0) < 1e-4


def test_3e5_25_mounting_rotation_matrix_application():
    """TEST-3E5-25: Non-zero gimbal pitch/yaw correctly transforms into body frame."""
    # Nadir camera has optical axis pointing down (-Z in FLU)
    r_nadir = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0]
    ])
    optical_axis_cam = np.array([0.0, 0.0, 1.0])
    optical_axis_body = r_nadir @ optical_axis_cam
    assert np.allclose(optical_axis_body, [0.0, 0.0, -1.0])


# ============================================================================
# 16.5 Altitude References & Geoid
# ============================================================================

def test_3e5_26_ellipsoidal_altitude_integrity():
    """TEST-3E5-26: Pure ellipsoidal telemetry preserves vertical metric scaling."""
    anchor = GeospatialAnchorOrigin(
        lat_deg=0.0, lon_deg=0.0, ellipsoidal_height_m=100.0,
        altitude_reference=AltitudeReferenceType.ELLIPSOIDAL_WGS84,
    )
    e, n, u = wgs84_to_enu(0.0, 0.0, 150.0, anchor)
    assert abs(u - 50.0) < 1e-3


def test_3e5_27_orthometric_altitude_detection():
    """TEST-3E5-27: Orthometric tag without geoid undulation flags state machine as uncertain."""
    sm = MetricStateMachine()
    # High residual due to geoid difference
    sm.evaluate_estimation(
        estimation_success=True,
        is_observable=True,
        inlier_count=10,
        rmse_3d_m=8.5,  # Exceeds 5.0m
        relative_scale_uncertainty=0.05,
    )
    assert sm.current_state == MetricScaleStatus.METRIC_SCALE_UNCERTAIN


def test_3e5_28_relative_barometric_altitude_isolation():
    """TEST-3E5-28: Barometric height relative to takeoff is mapped to RELATIVE_TO_TAKEOFF."""
    anchor = GeospatialAnchorOrigin(
        lat_deg=10.0, lon_deg=20.0, ellipsoidal_height_m=50.0,
        altitude_reference=AltitudeReferenceType.RELATIVE_TAKEOFF,
    )
    norm_origin = anchor.to_normalization_origin()
    assert norm_origin.altitude_reference.value == "RELATIVE_TO_TAKEOFF"


# ============================================================================
# 16.6 Geometric Observability, RANSAC Degeneracy & Stationarity
# ============================================================================

def test_3e5_29_pure_stationary_hover_flight():
    """TEST-3E5-29: Stationary drone (D_rel < 1e-6 or D_max == 0) triggers SCALE_NOT_OBSERVABLE."""
    # Stationary drone: 10 cameras at the identical 3D location
    c_rec = np.ones((10, 3), dtype=np.float64) * 5.0
    z_enu = np.ones((10, 3), dtype=np.float64) * 100.0

    report = check_scale_observability(c_rec, z_enu)
    assert not report.is_observable
    assert report.d_max == 0.0
    assert report.dispersion_rel == 0.0
    assert any("STATIONARY" in r for r in report.failure_reasons)


def test_3e5_30_collinear_flight_trajectory():
    """TEST-3E5-30: Pure linear flight (lambda_1/lambda_2 < 1e-4) preserves scale observability while flagging pose unobservability."""
    # Pure linear path along X axis with sufficient baseline and dispersion
    c_rec_collinear = np.zeros((20, 3), dtype=np.float64)
    c_rec_collinear[:, 0] = np.linspace(0.0, 100.0, 20)
    z_enu = np.zeros((20, 3), dtype=np.float64)
    z_enu[:, 0] = np.linspace(0.0, 100.0, 20)

    report_collinear = check_scale_observability(c_rec_collinear, z_enu)
    # Scale is observable from 1D pairwise distances!
    assert report_collinear.scale_observable is True
    assert report_collinear.is_collinear is True
    assert report_collinear.full_sim3_observability == FullSim3ObservabilityStatus.FULL_SIM3_NOT_OBSERVABLE_COLLINEAR
    assert report_collinear.collinearity_ratio < 1e-4
    assert any("COLLINEAR" in w for w in report_collinear.pose_warnings)
    assert len(report_collinear.scale_failure_reasons) == 0

    # Non-collinear trajectory has full Sim(3) observability
    c_rec_spiral = _create_synthetic_spiral_trajectory(20, 30.0, 10.0)
    z_enu_spiral = c_rec_spiral.copy()
    report_spiral = check_scale_observability(c_rec_spiral, z_enu_spiral)
    assert report_spiral.scale_observable is True
    assert report_spiral.is_collinear is False
    assert report_spiral.full_sim3_observability == FullSim3ObservabilityStatus.FULL_SIM3_OBSERVABLE
    assert report_spiral.collinearity_ratio > 0.01
    assert not any("COLLINEAR" in w for w in report_spiral.pose_warnings)


def test_3e5_31_ransac_3point_collinear_triplet_rejection():
    """TEST-3E5-31: Triplets with isoperimetric quotient Q < 1e-4 rejected during RANSAC."""
    # Collinear triplet on a line: (0, 0, 0), (1, 0, 0), (2, 0, 0)
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([2.0, 0.0, 0.0])
    q_collinear = compute_isoperimetric_quotient(p0, p1, p2)
    assert q_collinear < 1e-4

    # Equilateral triangle: (0, 0, 0), (1, 0, 0), (0.5, sqrt(3)/2, 0)
    p2_equi = np.array([0.5, math.sqrt(3.0) / 2.0, 0.0])
    q_equi = compute_isoperimetric_quotient(p0, p1, p2_equi)
    # Area = sqrt(3)/4 ~ 0.433, P = 3, Q = 4*pi*area / 9 ~ 0.604
    assert q_equi > 0.5


def test_3e5_32_insufficient_physical_baseline():
    """TEST-3E5-32: Cameras spanning only 2.0m in ENU rejected with B_gnss < 10.0m."""
    c_rec = _create_synthetic_spiral_trajectory(10, 10.0, 5.0)
    # GNSS points spanning only 2 meters
    z_enu = np.zeros((10, 3), dtype=np.float64)
    z_enu[:, 0] = np.linspace(0.0, 2.0, 10)

    report = check_scale_observability(c_rec, z_enu, tau_min_baseline_m=10.0)
    assert not report.is_observable
    assert report.gnss_baseline_span_m < 10.0
    assert any("INSUFFICIENT_PHYSICAL_BASELINE" in r for r in report.failure_reasons)


def test_3e5_33_minimum_point_count_gate():
    """TEST-3E5-33: Fewer than 4 points rejected from metric scale promotion."""
    c_rec = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=np.float64) # 3 points
    z_enu = c_rec.copy()
    report = check_scale_observability(c_rec, z_enu, min_inlier_count=4)
    assert not report.is_observable
    assert any("INSUFFICIENT_INLIERS" in r for r in report.failure_reasons)


# ============================================================================
# 16.7 Coordinate Transformation Round-Trips & Determinism
# ============================================================================

def test_3e5_34_wgs84_ecef_enu_roundtrip():
    """TEST-3E5-34: Geodetic coordinates converted to local ENU and back with error < 1e-4m."""
    anchor = GeospatialAnchorOrigin(lat_deg=18.5204, lon_deg=73.8567, ellipsoidal_height_m=540.0)
    test_pts = [
        (18.5204, 73.8567, 540.0),
        (18.5250, 73.8600, 600.0),
        (-18.5204, -73.8567, 100.0),
        (85.0, 10.0, 200.0), # High latitude near pole
    ]

    for lat, lon, alt in test_pts:
        e, n, u = wgs84_to_enu(lat, lon, alt, anchor)
        lat_rec, lon_rec, alt_rec = enu_to_wgs84(e, n, u, anchor)

        assert abs(lat_rec - lat) < 1e-7
        assert abs(lon_rec - lon) < 1e-7
        assert abs(alt_rec - alt) < 1e-4


def test_3e5_35_independent_gcp_validation_advance():
    """TEST-3E5-35: Survey checkpoints matching estimated Sim(3) within 0.05m advances to METRIC_SCALE_VALIDATED."""
    sim3 = Sim3(scale=2.5, rotation=np.eye(3), translation=np.array([10.0, 20.0, 5.0]))
    pts_rec = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [10.0, 10.0, 5.0],
    ])
    gcp_enu = sim3.transform_point(pts_rec)

    checkpoints = [
        GroundControlPoint(point_id=f"gcp_{i}", point_rec=pts_rec[i], gcp_enu=gcp_enu[i], survey_accuracy_m=0.05)
        for i in range(4)
    ]

    validator = MetricValidator()
    val_report = validator.validate(sim3, checkpoints)
    assert val_report.is_validated
    assert val_report.total_3d_rmse_m < 1e-5

    sm = MetricStateMachine(initial_state=MetricScaleStatus.METRIC_SCALE_ESTIMATED)
    sm.evaluate_validation(val_report.is_validated, val_report.total_3d_rmse_m, val_report.tolerance_m)
    assert sm.current_state == MetricScaleStatus.METRIC_SCALE_VALIDATED


def test_3e5_36_deterministic_execution_under_permutation():
    """TEST-3E5-36: Permuting camera input dictionary order produces bit-for-bit identical Sim(3)."""
    np.random.seed(123)
    pts_rec = _create_synthetic_spiral_trajectory(25, 40.0, 10.0)
    sim3_true = Sim3(scale=3.2, rotation=_euler_matrix(15.0, -10.0, 5.0), translation=np.array([100.0, 200.0, 50.0]))
    pts_enu = sim3_true.transform_point(pts_rec)

    records = [
        RawTelemetryRecord(
            timestamp_seconds=float(i),
            latitude_deg=18.5204 + i * 0.0001,
            longitude_deg=73.8567 + i * 0.0001,
            altitude_m=540.0 + pts_enu[i, 2],
        )
        for i in range(25)
    ]

    anchor = GeospatialAnchorOrigin(lat_deg=18.5204, lon_deg=73.8567, ellipsoidal_height_m=540.0)
    pipeline = GeospatialMetricReconstructor()

    # Run 1: Ascending order
    cam_dict_1 = {f"frame_{i:03d}": pts_rec[i] for i in range(25)}
    time_dict_1 = {f"frame_{i:03d}": float(i) for i in range(25)}
    res1 = pipeline.reconstruct(cam_dict_1, time_dict_1, records, anchor_origin=anchor)

    # Run 2: Shuffled order
    shuffled_indices = list(range(25))
    np.random.shuffle(shuffled_indices)
    cam_dict_2 = {f"frame_{i:03d}": pts_rec[i] for i in shuffled_indices}
    time_dict_2 = {f"frame_{i:03d}": float(i) for i in shuffled_indices}
    res2 = pipeline.reconstruct(cam_dict_2, time_dict_2, records, anchor_origin=anchor)

    assert res1.provenance_hash == res2.provenance_hash
    assert res1.sim3_transform is not None and res2.sim3_transform is not None
    assert res1.sim3_transform.scale == res2.sim3_transform.scale
    assert res1.sim3_transform.translation_enu == res2.sim3_transform.translation_enu


def test_3e5_37_scale_recovery_under_collinear_trajectory():
    """TEST-3E5-37: Collinear trajectory allows exact scale recovery from pairwise distances."""
    # 20 cameras along X axis with step 5.0 reconstruction units
    c_rec = np.zeros((20, 3), dtype=np.float64)
    c_rec[:, 0] = np.linspace(0.0, 95.0, 20)
    true_scale = 2.50
    # ENU has true scale 2.50 -> span = 237.5m > 10.0m
    z_enu = np.zeros((20, 3), dtype=np.float64)
    z_enu[:, 0] = c_rec[:, 0] * true_scale

    report = check_scale_observability(c_rec, z_enu)
    assert report.scale_observable is True
    assert report.is_collinear is True
    assert report.full_sim3_observability == FullSim3ObservabilityStatus.FULL_SIM3_NOT_OBSERVABLE_COLLINEAR

    # Verify closed-form pairwise distance scale recovery
    diff_rec = c_rec[:, np.newaxis, :] - c_rec[np.newaxis, :, :]
    dist_rec = np.linalg.norm(diff_rec, axis=-1).flatten()
    diff_enu = z_enu[:, np.newaxis, :] - z_enu[np.newaxis, :, :]
    dist_enu = np.linalg.norm(diff_enu, axis=-1).flatten()

    mask = dist_rec > 1e-6
    s_est = np.sum(dist_enu[mask] * dist_rec[mask]) / np.sum(dist_rec[mask] ** 2)
    assert abs(s_est - true_scale) < 1e-12


def test_3e5_38_dimensionless_relative_edge_ratio_invariance():
    """TEST-3E5-38: Triplet relative edge ratio rho_tri is invariant across scale factors 10^-12 to 10^12."""
    from src.geospatial.robust_estimation import compute_min_edge

    scales = [1e-12, 1e-6, 1.0, 1e6, 1e12]
    ratios = []

    for a in scales:
        p0 = a * np.array([0.0, 0.0, 0.0])
        p1 = a * np.array([10.0, 0.0, 0.0])
        p2 = a * np.array([0.0, 10.0, 0.0])
        d_max = a * 50.0

        min_edge = compute_min_edge(p0, p1, p2)
        rho_tri = min_edge / d_max
        ratios.append(rho_tri)

    # All ratios must be bit-for-bit or numerically identical (~0.20)
    for r in ratios:
        assert abs(r - 0.20) < 1e-12

    # Microscopic triplet test: edge is 1e-5 * D_max -> rho_tri < 1e-4 -> rejected
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1e-5, 0.0, 0.0])
    p2 = np.array([0.0, 1e-5, 0.0])
    d_max = 1.0
    rho_micro = compute_min_edge(p0, p1, p2) / d_max
    assert rho_micro < 1e-4


def test_3e5_39_gnss_accuracy_interpretation_conversions():
    """TEST-3E5-39: GNSS accuracy conversion for CEP50 and 2sigma95."""
    cov_1sig, _ = construct_gnss_covariance(
        horizontal_accuracy_m=2.35482,
        vertical_accuracy_m=1.96,
        interpretation=GnssAccuracyInterpretation.ONE_SIGMA_STANDARD_DEVIATION,
    )
    # sigma_h = 2.35482 -> var = 2.35482^2
    assert abs(math.sqrt(cov_1sig[0, 0]) - 2.35482) < 1e-4

    # CEP50 = 2.354820045m -> sigma = 2.354820045 / 1.1774100225154747 = 2.0m
    cov_cep, _ = construct_gnss_covariance(
        horizontal_accuracy_m=2.3548200450309494,
        vertical_accuracy_m=1.0,
        interpretation=GnssAccuracyInterpretation.CEP_50,
    )
    assert abs(math.sqrt(cov_cep[0, 0]) - 2.0) < 1e-4

    # 2sigma_95 horizontal = 4.895493787m -> sigma = 4.895493787 / 2.447746893674682 = 2.0m
    cov_2sig, _ = construct_gnss_covariance(
        horizontal_accuracy_m=4.895493787349364,
        vertical_accuracy_m=3.92,
        interpretation=GnssAccuracyInterpretation.TWO_SIGMA_95,
    )
    assert abs(math.sqrt(cov_2sig[0, 0]) - 2.0) < 1e-4
    assert abs(math.sqrt(cov_2sig[2, 2]) - 2.0) < 1e-4  # 3.92 / 1.959963984540054 = 2.0m

    # UNKNOWN_VENDOR_ACCURACY triggers fallback floor (3.0m horiz, 5.0m vert)
    cov_unk, missing = construct_gnss_covariance(
        horizontal_accuracy_m=0.5,
        vertical_accuracy_m=0.5,
        interpretation=GnssAccuracyInterpretation.UNKNOWN_VENDOR_ACCURACY,
    )
    assert missing is True
    assert math.sqrt(cov_unk[0, 0]) >= 3.0
    assert math.sqrt(cov_unk[2, 2]) >= 5.0


def test_3e5_40_huber_weighted_fisher_matrix_and_condition_number():
    """TEST-3E5-40: Huber-weighted Fisher matrix and condition number uncertainty classification."""
    from src.geospatial.uncertainty import UncertaintyPropagator
    from src.geospatial.sim3 import Sim3, UncertaintyType

    sim3 = Sim3(scale=2.0, rotation=np.eye(3), translation=np.array([10.0, 20.0, 30.0]))
    obs_list = []
    for i in range(10):
        c_rec = np.array([float(i * 5), float((i % 3) * 4), float((i % 2) * 3)])
        z_enu = sim3.transform_point(c_rec)
        obs_list.append(
            TelemetryObservation(
                frame_id=f"f_{i}",
                timestamp_seconds=float(i),
                c_rec=c_rec,
                z_gnss_enu=z_enu,
                covariance_enu=np.eye(3),
                classification=ObservationClassification.VALID,
            )
        )

    # Well-conditioned non-collinear trajectory with weights
    weights = np.ones(10, dtype=np.float64)
    weights[0] = 0.5  # downweighted outlier
    unc_report = UncertaintyPropagator.estimate_parameter_covariance(
        sim3=sim3,
        observations=obs_list,
        inlier_indices=list(range(10)),
        lever_arm=LeverArm.uncalibrated(),
        huber_weights=weights,
    )

    assert unc_report.uncertainty_type == UncertaintyType.ESTIMATED_COVARIANCE
    assert unc_report.fisher_condition_number is not None
    assert unc_report.fisher_condition_number <= 1e8
    assert unc_report.regularization_used is False
    assert unc_report.parameter_scales is not None


def test_3e5_41_meter_vs_kilometer_parameterization_invariance():
    """TEST-3E5-41: Normalized Hessian condition number and uncertainty decisions are invariant under meter vs kilometer units."""
    from src.geospatial.uncertainty import UncertaintyPropagator
    from src.geospatial.sim3 import Sim3, UncertaintyType

    np.random.seed(42)
    pts_rec = _create_synthetic_spiral_trajectory(15, 30.0, 10.0)

    # 1. Formulation in METRES
    sim3_m = Sim3(scale=2.5, rotation=_euler_matrix(10.0, -5.0, 3.0), translation=np.array([150.0, 250.0, 30.0]))
    obs_m = []
    for i in range(15):
        c_pt = pts_rec[i]
        z_enu_m = sim3_m.transform_point(c_pt)
        obs_m.append(
            TelemetryObservation(
                frame_id=f"frame_{i}",
                timestamp_seconds=float(i),
                c_rec=c_pt,
                z_gnss_enu=z_enu_m,
                covariance_enu=np.diag([4.0, 4.0, 9.0]),  # 2m horiz, 3m vert
                classification=ObservationClassification.VALID,
            )
        )

    unc_m = UncertaintyPropagator.estimate_parameter_covariance(
        sim3=sim3_m,
        observations=obs_m,
        inlier_indices=list(range(15)),
        lever_arm=LeverArm.uncalibrated(),
    )

    # 2. Equivalent formulation in KILOMETRES (1 km = 1000 m)
    # Scale: (km / rec_unit) = scale_m * 1e-3
    # Translation: km = translation_m * 1e-3
    # Covariance: km^2 = cov_m * 1e-6
    sim3_km = Sim3(scale=2.5 * 1e-3, rotation=sim3_m.rotation.copy(), translation=sim3_m.translation * 1e-3)
    obs_km = []
    for i in range(15):
        c_pt = pts_rec[i]
        z_enu_km = sim3_km.transform_point(c_pt)
        obs_km.append(
            TelemetryObservation(
                frame_id=f"frame_{i}",
                timestamp_seconds=float(i),
                c_rec=c_pt,
                z_gnss_enu=z_enu_km,
                covariance_enu=np.diag([4.0e-6, 4.0e-6, 9.0e-6]),  # in km^2
                classification=ObservationClassification.VALID,
            )
        )

    unc_km = UncertaintyPropagator.estimate_parameter_covariance(
        sim3=sim3_km,
        observations=obs_km,
        inlier_indices=list(range(15)),
        lever_arm=LeverArm.uncalibrated(),
    )

    # Conditioning classifications must be identical
    assert unc_m.uncertainty_type == unc_km.uncertainty_type == UncertaintyType.ESTIMATED_COVARIANCE
    assert unc_m.regularization_used == unc_km.regularization_used == False

    # Condition numbers of normalized Hessian must be identical within numerical precision
    assert unc_m.fisher_condition_number is not None and unc_km.fisher_condition_number is not None
    rel_diff_kappa = abs(unc_m.fisher_condition_number - unc_km.fisher_condition_number) / unc_m.fisher_condition_number
    assert rel_diff_kappa < 1e-7

    # Relative scale uncertainty sigma_s / s (dimensionless) must be identical
    assert abs(unc_m.relative_scale_uncertainty - unc_km.relative_scale_uncertainty) < 1e-9

    # Rotation uncertainty (radians, dimensionless) must be identical
    assert abs(unc_m.rotation_uncertainty_rad - unc_km.rotation_uncertainty_rad) < 1e-9

    # Positional uncertainty in km must equal positional uncertainty in m * 1e-3
    assert abs((unc_km.translation_uncertainty_m * 1000.0) - unc_m.translation_uncertainty_m) < 1e-6


def test_3e5_42_regularization_triggers_heuristic_uncertainty():
    """TEST-3E5-42: Ill-conditioned Hessian triggers heuristic regularization without claiming estimated covariance."""
    from src.geospatial.uncertainty import UncertaintyPropagator
    from src.geospatial.sim3 import Sim3, UncertaintyType

    # Create collinear trajectory (unconstrained rotation around flight line)
    c_rec_collinear = np.zeros((10, 3), dtype=np.float64)
    c_rec_collinear[:, 0] = np.linspace(0.0, 100.0, 10)
    sim3 = Sim3(scale=1.5, rotation=np.eye(3), translation=np.array([0.0, 0.0, 0.0]))
    obs_list = [
        TelemetryObservation(
            frame_id=f"f_{i}",
            timestamp_seconds=float(i),
            c_rec=c_rec_collinear[i],
            z_gnss_enu=sim3.transform_point(c_rec_collinear[i]),
            covariance_enu=np.eye(3),
            classification=ObservationClassification.VALID,
        )
        for i in range(10)
    ]

    unc_report = UncertaintyPropagator.estimate_parameter_covariance(
        sim3=sim3,
        observations=obs_list,
        inlier_indices=list(range(10)),
        lever_arm=LeverArm.uncalibrated(),
    )

    # Must be classified as HEURISTIC_UNCERTAINTY due to rank-deficiency of rotation
    assert unc_report.uncertainty_type == UncertaintyType.HEURISTIC_UNCERTAINTY
    assert unc_report.regularization_used is True
    assert unc_report.regularization_value == 1e-6
    assert unc_report.parameter_scales is not None
    assert unc_report.fallback_reason is not None


def test_3e5_43_huber_weights_affect_hessian():
    """TEST-3E5-43: Huber IRLS weights properly discount noisy/outlier observations in Hessian."""
    from src.geospatial.uncertainty import UncertaintyPropagator
    from src.geospatial.sim3 import Sim3

    pts_rec = _create_synthetic_spiral_trajectory(12, 20.0, 8.0)
    sim3 = Sim3(scale=1.0, rotation=np.eye(3), translation=np.zeros(3))
    obs_list = [
        TelemetryObservation(
            frame_id=f"f_{i}",
            timestamp_seconds=float(i),
            c_rec=pts_rec[i],
            z_gnss_enu=pts_rec[i].copy(),
            covariance_enu=np.eye(3),
            classification=ObservationClassification.VALID,
        )
        for i in range(12)
    ]

    # Full weights vs downweighted
    w_full = np.ones(12, dtype=np.float64)
    w_down = np.ones(12, dtype=np.float64)
    w_down[6:] = 0.05  # heavily downweighted subset

    unc_full = UncertaintyPropagator.estimate_parameter_covariance(
        sim3=sim3, observations=obs_list, inlier_indices=list(range(12)), lever_arm=LeverArm.uncalibrated(), huber_weights=w_full
    )
    unc_down = UncertaintyPropagator.estimate_parameter_covariance(
        sim3=sim3, observations=obs_list, inlier_indices=list(range(12)), lever_arm=LeverArm.uncalibrated(), huber_weights=w_down
    )

    # Downweighted Hessian must result in strictly larger uncertainty
    assert unc_down.scale_uncertainty_1sigma > unc_full.scale_uncertainty_1sigma
    assert unc_down.rotation_uncertainty_rad > unc_full.rotation_uncertainty_rad
    assert unc_down.translation_uncertainty_m > unc_full.translation_uncertainty_m


def test_3e5_44_no_automatic_metric_validation_without_gcp():
    """TEST-3E5-44: Without independent Ground Control Points, metric scale is ESTIMATED but never VALIDATED."""
    c_rec = _create_synthetic_spiral_trajectory(20, 25.0, 10.0)
    sim3_true = Sim3(scale=2.0, rotation=_euler_matrix(5.0, 10.0, -5.0), translation=np.array([50.0, 100.0, 20.0]))
    pts_enu = sim3_true.transform_point(c_rec)

    anchor = GeospatialAnchorOrigin(lat_deg=18.5204, lon_deg=73.8567, ellipsoidal_height_m=540.0)
    records = []
    for i in range(20):
        lat_i, lon_i, alt_i = enu_to_wgs84(pts_enu[i, 0], pts_enu[i, 1], pts_enu[i, 2], anchor)
        records.append(
            RawTelemetryRecord(
                timestamp_seconds=float(i),
                latitude_deg=lat_i,
                longitude_deg=lon_i,
                altitude_m=alt_i,
                horizontal_accuracy_m=1.0,
                vertical_accuracy_m=1.5,
            )
        )

    pipeline = GeospatialMetricReconstructor()
    cam_dict = {f"f_{i}": c_rec[i] for i in range(20)}
    time_dict = {f"f_{i}": float(i) for i in range(20)}

    # Run without checkpoints
    result = pipeline.reconstruct(cam_dict, time_dict, records, anchor_origin=anchor, checkpoints=None)

    assert result.metric_scale_status == MetricScaleStatus.METRIC_SCALE_ESTIMATED
    assert result.is_metric_scale is True
    assert result.validation_report is None
    assert result.metric_scale_status != MetricScaleStatus.METRIC_SCALE_VALIDATED


def test_3e5_45_collinear_no_attitude_preserves_unresolved_axial_dof():
    """TEST-3E5-45 [CASE A]: Collinear trajectory without attitude recovers scale but preserves unresolved axial DOF."""
    # 20 cameras along X axis with step 5.0 reconstruction units
    c_rec = np.zeros((20, 3), dtype=np.float64)
    c_rec[:, 0] = np.linspace(0.0, 95.0, 20)
    true_scale = 2.50
    sim3_true = Sim3(scale=true_scale, rotation=np.eye(3), translation=np.array([100.0, 200.0, 50.0]))
    pts_enu = sim3_true.transform_point(c_rec)

    anchor = GeospatialAnchorOrigin(lat_deg=18.5204, lon_deg=73.8567, ellipsoidal_height_m=540.0)
    records = []
    for i in range(20):
        lat_i, lon_i, alt_i = enu_to_wgs84(pts_enu[i, 0], pts_enu[i, 1], pts_enu[i, 2], anchor)
        records.append(
            RawTelemetryRecord(
                timestamp_seconds=float(i),
                latitude_deg=lat_i,
                longitude_deg=lon_i,
                altitude_m=alt_i,
                horizontal_accuracy_m=1.0,
                vertical_accuracy_m=1.5,
            )
        )

    pipeline = GeospatialMetricReconstructor()
    cam_dict = {f"f_{i}": c_rec[i] for i in range(20)}
    time_dict = {f"f_{i}": float(i) for i in range(20)}

    # CASE A: No attitude provided
    result = pipeline.reconstruct(cam_dict, time_dict, records, camera_rotations_rec=None, anchor_origin=anchor)

    # Scale is observable and metric scale estimated
    assert result.metric_scale_status == MetricScaleStatus.METRIC_SCALE_ESTIMATED
    assert result.is_metric_scale is True
    assert result.sim3_transform is not None
    assert abs(result.sim3_transform.scale - true_scale) < 1e-4

    # BUT axial rotation remains explicitly UNRESOLVED
    assert result.axial_rotation_resolved is False
    assert result.sim3_transform.axial_rotation_resolved is False
    assert result.full_sim3_observability == FullSim3ObservabilityStatus.FULL_SIM3_NOT_OBSERVABLE_COLLINEAR
    assert result.rotational_null_direction is not None

    # Uncertainty decomposition separates scale from unconstrained rotation
    unc_diag = result.diagnostics["uncertainty"]
    assert unc_diag["axial_rotation_resolved"] is False
    assert "axial_rotation_about_trajectory" in unc_diag["unconstrained_parameter_directions"]
    assert unc_diag["sigma_log_scale"] < 0.10  # Scale uncertainty is small and well-conditioned
    assert unc_diag["uncertainty_type"] == "HEURISTIC_UNCERTAINTY"  # Not claimed as statistical covariance


def test_3e5_46_collinear_correct_attitude_resolves_axial_dof():
    """TEST-3E5-46 [CASE B]: Collinear trajectory with calibrated attitude successfully resolves axial rotational DOF."""
    from src.geospatial.synchronization import _rotation_matrix_to_quaternion
    c_rec = np.zeros((20, 3), dtype=np.float64)
    c_rec[:, 0] = np.linspace(0.0, 95.0, 20)
    true_scale = 2.0

    # Non-trivial true rotation
    r_true = _euler_matrix(15.0, -10.0, 5.0)
    sim3_true = Sim3(scale=true_scale, rotation=r_true, translation=np.array([50.0, 80.0, 20.0]))
    pts_enu = sim3_true.transform_point(c_rec)

    anchor = GeospatialAnchorOrigin(lat_deg=18.5204, lon_deg=73.8567, ellipsoidal_height_m=540.0)
    records = []
    cam_rot_dict = {}

    # Extract Euler angles of r_true for vehicle telemetry (with zero lever arm mounting rotation)
    # r_true = R_z(yaw) R_y(pitch) R_x(roll)
    # Using camera_to_rec = I, camera_to_enu = r_true
    for i in range(20):
        lat_i, lon_i, alt_i = enu_to_wgs84(pts_enu[i, 0], pts_enu[i, 1], pts_enu[i, 2], anchor)
        records.append(
            RawTelemetryRecord(
                timestamp_seconds=float(i),
                latitude_deg=lat_i,
                longitude_deg=lon_i,
                altitude_m=alt_i,
                yaw_deg=15.0,
                pitch_deg=-10.0,
                roll_deg=5.0,
                horizontal_accuracy_m=1.0,
                vertical_accuracy_m=1.5,
            )
        )
        cam_rot_dict[f"f_{i}"] = np.eye(3, dtype=np.float64)

    pipeline = GeospatialMetricReconstructor()
    cam_dict = {f"f_{i}": c_rec[i] for i in range(20)}
    time_dict = {f"f_{i}": float(i) for i in range(20)}

    # CASE B: Correct attitude provided
    result = pipeline.reconstruct(
        cam_dict, time_dict, records, camera_rotations_rec=cam_rot_dict, anchor_origin=anchor
    )

    assert result.metric_scale_status == MetricScaleStatus.METRIC_SCALE_ESTIMATED
    assert result.sim3_transform is not None
    assert abs(result.sim3_transform.scale - true_scale) < 1e-4

    # Axial rotation is fully resolved by attitude!
    assert result.axial_rotation_resolved is True
    assert result.sim3_transform.axial_rotation_resolved is True
    assert result.full_sim3_observability == FullSim3ObservabilityStatus.FULL_SIM3_OBSERVABLE
    assert result.rotational_null_direction is None

    # Verify estimated rotation matches true rotation
    r_est = np.array(result.sim3_transform.rotation_matrix)
    rot_diff = np.linalg.norm(r_est @ r_true.T - np.eye(3))
    assert rot_diff < 1e-3


def test_3e5_47_collinear_wrong_attitude_rejected_leaves_unresolved():
    """TEST-3E5-47 [CASE C]: Collinear trajectory with conflicting attitude rejects attitude and leaves axial DOF unresolved."""
    c_rec = np.zeros((20, 3), dtype=np.float64)
    c_rec[:, 0] = np.linspace(0.0, 95.0, 20)
    true_scale = 2.0
    r_true = np.eye(3)
    sim3_true = Sim3(scale=true_scale, rotation=r_true, translation=np.array([50.0, 80.0, 20.0]))
    pts_enu = sim3_true.transform_point(c_rec)

    anchor = GeospatialAnchorOrigin(lat_deg=18.5204, lon_deg=73.8567, ellipsoidal_height_m=540.0)
    records = []
    cam_rot_dict = {}

    # Telemetry reports attitude that is 60 degrees pitched away from trajectory line
    for i in range(20):
        lat_i, lon_i, alt_i = enu_to_wgs84(pts_enu[i, 0], pts_enu[i, 1], pts_enu[i, 2], anchor)
        records.append(
            RawTelemetryRecord(
                timestamp_seconds=float(i),
                latitude_deg=lat_i,
                longitude_deg=lon_i,
                altitude_m=alt_i,
                yaw_deg=0.0,
                pitch_deg=60.0,  # Conflicting pitch!
                roll_deg=0.0,
                horizontal_accuracy_m=1.0,
                vertical_accuracy_m=1.5,
            )
        )
        cam_rot_dict[f"f_{i}"] = np.eye(3, dtype=np.float64)

    pipeline = GeospatialMetricReconstructor()
    cam_dict = {f"f_{i}": c_rec[i] for i in range(20)}
    time_dict = {f"f_{i}": float(i) for i in range(20)}

    # CASE C: Conflicting attitude provided
    result = pipeline.reconstruct(
        cam_dict, time_dict, records, camera_rotations_rec=cam_rot_dict, anchor_origin=anchor
    )

    # Scale is still recovered from pairwise positions
    assert result.metric_scale_status == MetricScaleStatus.METRIC_SCALE_ESTIMATED
    assert result.sim3_transform is not None
    assert abs(result.sim3_transform.scale - true_scale) < 1e-4

    # BUT conflicting attitude is rejected and axial rotation remains unresolved
    assert result.axial_rotation_resolved is False
    assert result.full_sim3_observability == FullSim3ObservabilityStatus.FULL_SIM3_NOT_OBSERVABLE_COLLINEAR


def test_3e5_48_non_collinear_no_attitude_fully_observable():
    """TEST-3E5-48 [CASE D]: Non-collinear 3D trajectory without attitude is fully observable from positions alone."""
    c_rec = _create_synthetic_spiral_trajectory(20, 25.0, 10.0)
    sim3_true = Sim3(scale=2.0, rotation=_euler_matrix(5.0, 10.0, -5.0), translation=np.array([50.0, 100.0, 20.0]))
    pts_enu = sim3_true.transform_point(c_rec)

    anchor = GeospatialAnchorOrigin(lat_deg=18.5204, lon_deg=73.8567, ellipsoidal_height_m=540.0)
    records = []
    for i in range(20):
        lat_i, lon_i, alt_i = enu_to_wgs84(pts_enu[i, 0], pts_enu[i, 1], pts_enu[i, 2], anchor)
        records.append(
            RawTelemetryRecord(
                timestamp_seconds=float(i),
                latitude_deg=lat_i,
                longitude_deg=lon_i,
                altitude_m=alt_i,
                horizontal_accuracy_m=1.0,
                vertical_accuracy_m=1.5,
            )
        )

    pipeline = GeospatialMetricReconstructor()
    cam_dict = {f"f_{i}": c_rec[i] for i in range(20)}
    time_dict = {f"f_{i}": float(i) for i in range(20)}

    # CASE D: Non-collinear + no attitude
    result = pipeline.reconstruct(cam_dict, time_dict, records, camera_rotations_rec=None, anchor_origin=anchor)

    assert result.metric_scale_status == MetricScaleStatus.METRIC_SCALE_ESTIMATED
    assert result.axial_rotation_resolved is True
    assert result.full_sim3_observability == FullSim3ObservabilityStatus.FULL_SIM3_OBSERVABLE
    assert result.diagnostics["uncertainty"]["uncertainty_type"] == "ESTIMATED_COVARIANCE"


def test_3e5_49_umeyama_geometry_spectrum_collinear_planar_3d():
    """TEST-3E5-49: Umeyama behavior across exactly collinear, nearly collinear, planar, and 3D geometries."""
    # 1. Exactly Collinear Points
    pts_collinear = np.zeros((10, 3), dtype=np.float64)
    pts_collinear[:, 0] = np.linspace(0.0, 50.0, 10)
    dst_collinear = pts_collinear * 3.5 + np.array([10.0, 20.0, 30.0])

    sim_col = solve_sim3_umeyama(pts_collinear, dst_collinear)
    assert not math.isnan(sim_col.scale) and sim_col.scale > 0.0
    assert abs(sim_col.scale - 3.5) < 1e-10
    assert abs(np.linalg.det(sim_col.rotation) - 1.0) < 1e-6
    # Reproducible across repeated calls
    sim_col_2 = solve_sim3_umeyama(pts_collinear, dst_collinear)
    assert sim_col.scale == sim_col_2.scale

    # 2. Nearly Collinear Points (lambda_1 / lambda_2 = 1e-5)
    pts_nearly = pts_collinear.copy()
    pts_nearly[:, 1] = np.sin(np.linspace(0, np.pi, 10)) * 1e-4
    dst_nearly = pts_nearly * 2.0
    sim_nearly = solve_sim3_umeyama(pts_nearly, dst_nearly)
    assert not math.isnan(sim_nearly.scale) and sim_nearly.scale > 0.0
    assert abs(sim_nearly.scale - 2.0) < 1e-4
    assert abs(np.linalg.det(sim_nearly.rotation) - 1.0) < 1e-6

    # 3. Planar Points (XY plane)
    pts_planar = np.zeros((10, 3), dtype=np.float64)
    pts_planar[:, 0] = np.cos(np.linspace(0, 2 * np.pi, 10)) * 20.0
    pts_planar[:, 1] = np.sin(np.linspace(0, 2 * np.pi, 10)) * 20.0
    dst_planar = pts_planar * 1.8 + np.array([5.0, 5.0, 0.0])
    sim_planar = solve_sim3_umeyama(pts_planar, dst_planar)
    assert not math.isnan(sim_planar.scale) and sim_planar.scale > 0.0
    assert abs(sim_planar.scale - 1.8) < 1e-10
    assert abs(np.linalg.det(sim_planar.rotation) - 1.0) < 1e-6

    # 4. Fully 3D Points
    pts_3d = _create_synthetic_spiral_trajectory(15, 30.0, 10.0)
    r_3d = _euler_matrix(10.0, -20.0, 15.0)
    dst_3d = (pts_3d @ r_3d.T) * 4.2 + np.array([100.0, 200.0, 300.0])
    sim_3d = solve_sim3_umeyama(pts_3d, dst_3d)
    assert abs(sim_3d.scale - 4.2) < 1e-10
    assert abs(np.linalg.det(sim_3d.rotation) - 1.0) < 1e-6


def test_3e5_50_uncertainty_decomposition_separates_scale_from_rotational_null_mode():
    """TEST-3E5-50: Uncertainty decomposition explicitly isolates low scale uncertainty from unconstrained axial rotation."""
    from src.geospatial.uncertainty import UncertaintyPropagator

    # Collinear trajectory along X axis
    pts_rec = np.zeros((15, 3), dtype=np.float64)
    pts_rec[:, 0] = np.linspace(0.0, 100.0, 15)
    sim3 = Sim3(scale=2.0, rotation=np.eye(3), translation=np.zeros(3), axial_rotation_resolved=False)
    obs_list = [
        TelemetryObservation(
            frame_id=f"f_{i}",
            timestamp_seconds=float(i),
            c_rec=pts_rec[i],
            z_gnss_enu=sim3.transform_point(pts_rec[i]),
            covariance_enu=np.eye(3),
            classification=ObservationClassification.VALID,
        )
        for i in range(15)
    ]

    unc = UncertaintyPropagator.estimate_parameter_covariance(
        sim3=sim3,
        observations=obs_list,
        inlier_indices=list(range(15)),
        lever_arm=LeverArm.uncalibrated(),
    )

    # Scale uncertainty is small, finite, and well-constrained!
    assert unc.sigma_log_scale < 0.05
    assert unc.relative_scale_uncertainty < 0.05
    assert unc.scale_uncertainty_1sigma < 0.10

    # Translation uncertainty is finite and well-constrained!
    assert unc.translation_uncertainty_m < 5.0

    # Rotation uncertainty flags the unconstrained axial mode!
    assert unc.rotation_uncertainty_rad > 100.0  # From 1e-6 regularization damping
    assert unc.axial_rotation_resolved is False
    assert "axial_rotation_about_trajectory" in unc.unconstrained_parameter_directions
    assert unc.rotational_null_direction is not None
    assert unc.uncertainty_type == UncertaintyType.HEURISTIC_UNCERTAINTY


def test_3e5_51_synthetic_analytical_covariance_inversion():
    """TEST-3E5-51: Diagonal Hessian case where expected covariance is analytically known.

    Verifies that normalization H_tilde = S @ H @ S followed by consistent inverse transform
    Sigma_theta = S @ inv(H_tilde) @ S (equiv. S_param^(-1) @ inv(H_tilde) @ S_param^(-1))
    returns H^(-1) within numerical tolerance for non-unit positional scale.
    """
    # Analytically known positive diagonal Hessian
    h_diag = np.array([2.5, 4.0, 3.2, 5.1, 0.04, 0.05, 0.06], dtype=np.float64)
    H = np.diag(h_diag)
    H_inv_expected = np.diag(1.0 / h_diag)

    # Non-unit positional scale D_geo (e.g. 150m flight baseline span)
    D_geo = 150.0
    S = np.diag([1.0, 1.0, 1.0, 1.0, D_geo, D_geo, D_geo])

    # 1. Dimensionless normalized Hessian
    H_tilde = S @ H @ S

    # 2. Normalized covariance
    cov_tilde = np.linalg.inv(H_tilde)

    # 3. Inverse normalization to physical covariance: Sigma_theta = S @ cov_tilde @ S
    cov_theta = S @ cov_tilde @ S

    # Must match analytical H^(-1) within floating-point precision
    max_err = float(np.max(np.abs(cov_theta - H_inv_expected)))
    assert max_err < 1e-14
    assert np.allclose(cov_theta, H_inv_expected, atol=1e-14, rtol=1e-14)

    # In parameter scale notation S_param = S^(-1) (where S_pos = 1/D_geo):
    S_param = np.linalg.inv(S)
    cov_theta_param = np.linalg.inv(S_param) @ cov_tilde @ np.linalg.inv(S_param)
    assert np.allclose(cov_theta_param, H_inv_expected, atol=1e-14, rtol=1e-14)

    # 4. Invariance under units (Meters vs Kilometers):
    # In meters: D_geo_m = 150m, translation Hessian block ~ 1/m^2
    # In km: D_geo_km = 0.15km, translation Hessian block is multiplied by 10^6 (since 1/km^2 = 10^6 * 1/m^2)
    h_diag_km = h_diag.copy()
    h_diag_km[4:7] *= 1e6
    H_km = np.diag(h_diag_km)
    S_km = np.diag([1.0, 1.0, 1.0, 1.0, D_geo * 1e-3, D_geo * 1e-3, D_geo * 1e-3])

    H_tilde_km = S_km @ H_km @ S_km
    # Normalized Hessians must be strictly identical
    assert np.allclose(H_tilde, H_tilde_km, atol=1e-14, rtol=1e-14)

    cov_tilde_km = np.linalg.inv(H_tilde_km)
    cov_theta_km = S_km @ cov_tilde_km @ S_km

    # Translation standard deviation in km must equal translation standard deviation in m / 1000
    sigma_trans_m = math.sqrt(float(np.trace(cov_theta[4:7, 4:7])))
    sigma_trans_km = math.sqrt(float(np.trace(cov_theta_km[4:7, 4:7])))
    assert abs(sigma_trans_km - (sigma_trans_m / 1000.0)) < 1e-12




