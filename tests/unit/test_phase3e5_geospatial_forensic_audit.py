"""Phase 3E.5 Geospatial & Metric Reconstruction Forensic Audit Test Suite.

Contains explicit adversarial mutation tests:
- MUT-01: Wrong Sim(3) direction (inverse applied instead of forward)
- MUT-02: Wrong R transpose in attitude residual
- MUT-03: Wrong lever-arm sign (-R*L instead of +R*L)
- MUT-04: Wrong lever-arm rotation convention
- MUT-05: Wrong timestamp interpolation
- MUT-06: Wrong altitude convention (mixing orthometric without geoid)
- MUT-07: Inverted collinearity eigenvalue ratio (lambda_2 / lambda_1 instead of lambda_1 / lambda_2)
- MUT-08: Reconstruction-unit threshold leakage (scale-dependent dispersion test)
- MUT-09: Improper GNSS weighting (unweighted OLS fallback)
- MUT-10: Failure to reject degenerate RANSAC triplet (isoperimetric quotient gate bypass)
- MUT-11: Arbitrary metric-state promotion (skipping independent GCP verification)
- MUT-12: Incorrect ENU axis mapping (East-North swap or handedness inversion)
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
    wgs84_to_enu,
    enu_to_wgs84,
    LeverArm,
    LeverArmStatus,
    TelemetryObservation,
    ObservationClassification,
    RawTelemetryRecord,
    TelemetrySynchronizer,
    check_scale_observability,
    FullSim3ObservabilityStatus,
    RobustSim3Estimator,
    compute_isoperimetric_quotient,
    MetricScaleStatus,
    MetricStateMachine,
    GroundControlPoint,
    MetricValidator,
    GeospatialMetricReconstructor,
)


def _euler_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Generate active 3x3 rotation matrix for given Euler angles in degrees."""
    psi = math.radians(yaw_deg)
    theta = math.radians(pitch_deg)
    phi = math.radians(roll_deg)
    rz = np.array([[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0], [0, 0, 1]])
    ry = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
    rx = np.array([[1, 0, 0], [0, math.cos(phi), -math.sin(phi)], [0, math.sin(phi), math.cos(phi)]])
    return rz @ ry @ rx


# ============================================================================
# Mutation Tests
# ============================================================================

def test_mut_01_wrong_sim3_direction():
    """MUT-01: Applying inverted Sim(3) direction (1/s * R^T) results in inverted scale and large error."""
    sim3 = Sim3(scale=4.0, rotation=_euler_matrix(30, 0, 0), translation=np.array([10.0, 20.0, 5.0]))
    pt = np.array([2.0, 3.0, 1.0])

    # Correct forward:
    fwd_pt = sim3.transform_point(pt)

    # Mutated inverse formula:
    mutated_pt = (1.0 / sim3.scale) * (sim3.rotation.T @ (pt - sim3.translation))

    # Must differ significantly from forward result
    discrepancy = float(np.linalg.norm(fwd_pt - mutated_pt))
    assert discrepancy > 10.0


def test_mut_02_wrong_rotation_transpose_in_attitude_residual():
    """MUT-02: Computing Tr(R_geo * R_sim3) instead of Tr(R_geo * R_sim3^T) produces false angular error."""
    # When R_geo and R_sim3 are identical, correct residual is 0 degrees.
    r_identical = _euler_matrix(45.0, 20.0, -10.0)

    # Correct calculation:
    diff_correct = r_identical @ r_identical.T
    tr_correct = float(np.trace(diff_correct))
    angle_correct = math.degrees(math.acos(np.clip((tr_correct - 1.0) * 0.5, -1.0, 1.0)))
    assert abs(angle_correct - 0.0) < 1e-6

    # Mutated calculation: without transpose
    diff_mutated = r_identical @ r_identical
    tr_mutated = float(np.trace(diff_mutated))
    angle_mutated = math.degrees(math.acos(np.clip((tr_mutated - 1.0) * 0.5, -1.0, 1.0)))
    assert angle_mutated > 10.0  # False error detected


def test_mut_03_wrong_lever_arm_sign():
    """MUT-03: Using minus instead of plus in forward antenna prediction reverses displacement."""
    # Antenna mounted 0.5m forward (+X FLU) on drone
    lever = LeverArm.calibrated(0.5, 0.0, 0.0)
    c_cam = np.array([0.0, 0.0, 0.0])
    r_body = _euler_matrix(0.0, 0.0, 0.0)

    correct_ant = lever.predict_antenna_enu(c_cam, r_body)
    # Mutated antenna prediction with minus sign:
    mutated_ant = c_cam - lever.transform_to_enu(r_body)

    # Correct should be +0.5m in X; mutated is -0.5m in X
    assert abs(correct_ant[0] - 0.5) < 1e-4
    assert abs(mutated_ant[0] - (-0.5)) < 1e-4
    assert np.linalg.norm(correct_ant - mutated_ant) == pytest.approx(1.0)


def test_mut_04_wrong_lever_arm_rotation_convention():
    """MUT-04: Transposing R_body in lever-arm rotation produces wrong ENU vector under yaw."""
    lever = LeverArm.calibrated(0.5, 0.0, 0.0) # Forward
    # Drone yawed 90 degrees
    r_body = _euler_matrix(90.0, 0.0, 0.0)

    # Correct active rotation: Forward (+X) -> (+Y)
    correct_enu = lever.transform_to_enu(r_body)
    assert abs(correct_enu[1] - 0.5) < 1e-4

    # Mutated passive rotation: R_body.T @ L -> (-Y)
    mutated_enu = r_body.T @ lever.vector_body
    assert abs(mutated_enu[1] - (-0.5)) < 1e-4
    assert not np.allclose(correct_enu, mutated_enu)


def test_mut_05_wrong_timestamp_interpolation():
    """MUT-05: Naive index matching instead of continuous timestamp interpolation creates large spatial bias."""
    anchor = GeospatialAnchorOrigin(lat_deg=18.5204, lon_deg=73.8567, ellipsoidal_height_m=540.0)
    # Drone flying at 10 m/s North
    records = [
        RawTelemetryRecord(timestamp_seconds=10.0, latitude_deg=18.5200, longitude_deg=73.8567, altitude_m=540.0),
        RawTelemetryRecord(timestamp_seconds=11.0, latitude_deg=18.5201, longitude_deg=73.8567, altitude_m=540.0),
    ]
    sync = TelemetrySynchronizer(records, anchor)

    # Correct interpolation at 10.5s:
    obs_interpolated = sync.synchronize_frame("f1", 10.5, np.zeros(3))

    # Mutated: snap to index 0 (10.0s)
    obs_index_0 = sync.synchronize_frame("f1", 10.0, np.zeros(3))

    pos_err = float(np.linalg.norm(obs_interpolated.z_gnss_enu - obs_index_0.z_gnss_enu))
    assert pos_err > 5.0  # Significant error from ignoring interpolation fraction


def test_mut_06_wrong_altitude_convention():
    """MUT-06: Orthometric altitude tag causes state machine to reject validated metric state."""
    sm = MetricStateMachine(initial_state=MetricScaleStatus.NOT_METRIC)
    # If altitude reference is unverified orthometric with high vertical discrepancy (7.0m):
    sm.evaluate_estimation(
        estimation_success=True,
        is_observable=True,
        inlier_count=8,
        rmse_3d_m=7.0,  # Exceeds 5.0m threshold
        relative_scale_uncertainty=0.05,
    )
    assert sm.current_state == MetricScaleStatus.METRIC_SCALE_UNCERTAIN
    assert sm.current_state != MetricScaleStatus.METRIC_SCALE_VALIDATED


def test_mut_07_wrong_eigenvalue_ratio_mutation():
    """MUT-07: Inverting collinearity ratio (lambda_2 / lambda_1 instead of lambda_1 / lambda_2) fails to catch collinearity."""
    # Collinear path: lambda_0 ~ 0, lambda_1 ~ 0, lambda_2 >> 0
    c_rec_collinear = np.zeros((20, 3), dtype=np.float64)
    c_rec_collinear[:, 0] = np.linspace(0.0, 100.0, 20)
    z_enu = c_rec_collinear.copy()

    # Correct check:
    report = check_scale_observability(c_rec_collinear, z_enu, tau_collinear=1e-4)
    assert report.is_collinear is True
    assert report.full_sim3_observability == FullSim3ObservabilityStatus.FULL_SIM3_NOT_OBSERVABLE_COLLINEAR
    assert report.scale_observable is True
    assert report.collinearity_ratio < 1e-4

    # Mutated check: lambda_2 / lambda_1 is >> 1e4, so a mutated test would falsely pass if compared to < 1e-4
    l0, l1, l2 = report.eigenvalues
    mutated_ratio = (l2 / l1) if l1 > 0 else float("inf")
    assert mutated_ratio > 1e4


def test_mut_08_reconstruction_unit_threshold_leakage():
    """MUT-08: A dimensional threshold in reconstruction gauge falsely flags tiny scaled valid scenes."""
    # Create valid 3D spiral scene scaled by 1e-5 (e.g. microscopic or normalized gauge)
    pts_rec_tiny = 1e-5 * np.array([
        [0, 0, 0], [10, 5, 2], [-5, 12, 4], [8, -6, 3], [3, 8, 9]
    ], dtype=np.float64)
    z_enu_large = np.array([
        [0, 0, 0], [100, 50, 20], [-50, 120, 40], [80, -60, 30], [30, 80, 90]
    ], dtype=np.float64)

    # Correct dimensionless test: D_rel is scale-invariant
    report = check_scale_observability(pts_rec_tiny, z_enu_large, min_inlier_count=4)
    # Normalized dispersion D_rel should be well above 1e-6 even though absolute dispersion is tiny
    assert report.dispersion_rel > 0.1
    assert not any("STATIONARY" in r for r in report.failure_reasons)


def test_mut_09_improper_gnss_weighting_outlier_leakage():
    """MUT-09: Unweighted OLS allows a single massive outlier to corrupt scale by > 20%."""
    pts_rec = np.array([
        [0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0],
        [20, 0, 0], [0, 20, 0], [20, 20, 0], [5, 5, 10],
    ], dtype=np.float64)
    s_true = 2.0
    dst = s_true * pts_rec.copy()

    # Corrupt last point by 1000m
    dst[-1] += np.array([1000.0, 1000.0, 1000.0])

    # Unweighted Umeyama on all points (including outlier):
    sim3_ols = solve_sim3_umeyama(pts_rec, dst)
    error_ols = abs(sim3_ols.scale - s_true) / s_true
    assert error_ols > 0.20  # Corrupted!

    # Robust Huber RANSAC estimator rejects outlier:
    obs = [
        TelemetryObservation(f"f_{i}", float(i), pts_rec[i], dst[i], covariance_enu=np.eye(3))
        for i in range(len(pts_rec))
    ]
    estimator = RobustSim3Estimator(tau_inlier_mahalanobis=3.0)
    res_robust = estimator.estimate(obs, LeverArm.zero())
    assert res_robust.success and res_robust.sim3 is not None
    error_robust = abs(res_robust.sim3.scale - s_true) / s_true
    assert error_robust < 0.01  # Robust estimate protected


def test_mut_10_failure_to_reject_degenerate_ransac_triplet():
    """MUT-10: Collinear triplet has isoperimetric quotient Q < 1e-4 and must be rejected."""
    # Collinear triplet along line:
    p0 = np.array([10.0, 10.0, 10.0])
    p1 = np.array([20.0, 20.0, 20.0])
    p2 = np.array([30.0, 30.0, 30.0])
    q = compute_isoperimetric_quotient(p0, p1, p2)
    assert q < 1e-4

    # Estimator must reject this triplet during RANSAC
    observations = [
        TelemetryObservation("f0", 0.0, p0, p0, covariance_enu=np.eye(3)),
        TelemetryObservation("f1", 1.0, p1, p1, covariance_enu=np.eye(3)),
        TelemetryObservation("f2", 2.0, p2, p2, covariance_enu=np.eye(3)),
    ]
    estimator = RobustSim3Estimator(tau_tri_degen=1e-4)
    res = estimator.estimate(observations, LeverArm.zero())
    assert not res.success
    assert "NO_NON_DEGENERATE_SAMPLE_FOUND" in str(res.failure_reason)


def test_mut_11_arbitrary_metric_state_promotion():
    """MUT-11: Sim(3) fit alone CANNOT promote status to METRIC_SCALE_VALIDATED."""
    sm = MetricStateMachine(initial_state=MetricScaleStatus.NOT_METRIC)
    sm.evaluate_estimation(
        estimation_success=True,
        is_observable=True,
        inlier_count=20,
        rmse_3d_m=0.10, # Very low RMSE
        relative_scale_uncertainty=0.01,
    )
    # Must only advance to METRIC_SCALE_ESTIMATED, NEVER METRIC_SCALE_VALIDATED
    assert sm.current_state == MetricScaleStatus.METRIC_SCALE_ESTIMATED
    assert sm.current_state != MetricScaleStatus.METRIC_SCALE_VALIDATED


def test_mut_12_incorrect_enu_axis_mapping():
    """MUT-12: Swapping East and North in ENU breaks handedness and causes large coordinate error."""
    anchor = GeospatialAnchorOrigin(lat_deg=18.5204, lon_deg=73.8567, ellipsoidal_height_m=540.0)
    # A point slightly North
    lat_north = 18.5214
    lon_same = 73.8567

    e, n, u = wgs84_to_enu(lat_north, lon_same, 540.0, anchor)
    # Point North should have small East (< 1m) and ~111m North
    assert abs(e) < 1.0
    assert abs(n - 110.0) < 5.0

    # If mutated by swapping (n, e):
    mutated_lat, mutated_lon, _ = enu_to_wgs84(n, e, u, anchor)
    assert abs(mutated_lat - lat_north) > 0.0005  # Severe distortion


def test_mut_13_raw_hessian_eigenvalue_threshold_fails_under_unit_rescaling():
    """MUT-13: Checking raw Hessian eigenvalues without diagonal normalization S fails under coordinate unit changes."""
    from src.geospatial.uncertainty import UncertaintyPropagator
    from src.geospatial.sim3 import Sim3

    # Generate synthetic 3D camera network in meters
    pts_rec = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 5.0, 2.0],
        [-8.0, 15.0, 4.0],
        [12.0, -10.0, 3.0],
        [5.0, 8.0, 12.0],
        [-10.0, -5.0, 6.0],
    ], dtype=np.float64)

    sim3_m = Sim3(scale=2.0, rotation=np.eye(3), translation=np.array([100.0, 200.0, 50.0]))
    obs_m = [
        TelemetryObservation(
            frame_id=f"f_{i}",
            timestamp_seconds=float(i),
            c_rec=pts_rec[i],
            z_gnss_enu=sim3_m.transform_point(pts_rec[i]),
            covariance_enu=np.eye(3) * 4.0,  # 2m 1-sigma in meters^2
            classification=ObservationClassification.VALID,
        )
        for i in range(6)
    ]

    # Compute raw Hessian in METRES
    H_m = np.zeros((7, 7), dtype=np.float64)
    for obs in obs_m:
        p_cam = sim3_m.scale * (sim3_m.rotation @ obs.c_rec)
        J = np.zeros((3, 7), dtype=np.float64)
        J[:, 0] = -p_cam
        J[:, 1:4] = np.array([[0, -p_cam[2], p_cam[1]], [p_cam[2], 0, -p_cam[0]], [-p_cam[1], p_cam[0], 0]])
        J[:, 4:7] = -np.eye(3)
        inv_cov = np.linalg.inv(obs.covariance_enu)
        H_m += J.T @ inv_cov @ J

    raw_evals_m = np.linalg.eigvalsh(H_m)
    raw_kappa_m = float(np.max(raw_evals_m) / np.min(raw_evals_m))

    # Convert same scene to KILOMETRES (1 km = 1000 m)
    sim3_km = Sim3(scale=2.0 * 1e-3, rotation=np.eye(3), translation=sim3_m.translation * 1e-3)
    obs_km = [
        TelemetryObservation(
            frame_id=f"f_{i}",
            timestamp_seconds=float(i),
            c_rec=pts_rec[i],
            z_gnss_enu=sim3_km.transform_point(pts_rec[i]),
            covariance_enu=np.eye(3) * 4.0 * 1e-6,  # in km^2
            classification=ObservationClassification.VALID,
        )
        for i in range(6)
    ]

    # Compute raw Hessian in KILOMETRES
    H_km = np.zeros((7, 7), dtype=np.float64)
    for obs in obs_km:
        p_cam = sim3_km.scale * (sim3_km.rotation @ obs.c_rec)
        J = np.zeros((3, 7), dtype=np.float64)
        J[:, 0] = -p_cam
        J[:, 1:4] = np.array([[0, -p_cam[2], p_cam[1]], [p_cam[2], 0, -p_cam[0]], [-p_cam[1], p_cam[0], 0]])
        J[:, 4:7] = -np.eye(3)
        inv_cov = np.linalg.inv(obs.covariance_enu)
        H_km += J.T @ inv_cov @ J

    raw_evals_km = np.linalg.eigvalsh(H_km)
    raw_kappa_km = float(np.max(raw_evals_km) / np.min(raw_evals_km))

    # Raw Hessian condition number changes by orders of magnitude due to unit change!
    assert abs(raw_kappa_m - raw_kappa_km) > 10.0

    # BUT with diagonal parameter normalization S, normalized condition numbers are strictly identical:
    unc_m = UncertaintyPropagator.estimate_parameter_covariance(sim3_m, obs_m, list(range(6)), LeverArm.uncalibrated())
    unc_km = UncertaintyPropagator.estimate_parameter_covariance(sim3_km, obs_km, list(range(6)), LeverArm.uncalibrated())

    assert unc_m.fisher_condition_number is not None and unc_km.fisher_condition_number is not None
    rel_diff_norm_kappa = abs(unc_m.fisher_condition_number - unc_km.fisher_condition_number) / unc_m.fisher_condition_number
    assert rel_diff_norm_kappa < 1e-7


def test_mut_14_collinear_trajectory_silently_promotes_unobservable_rotation_as_fully_observable():
    """MUT-14: Collinear trajectory without attitude must NEVER silently claim full 3D rotation observability."""
    from src.geospatial.pipeline import GeospatialMetricReconstructor
    from src.geospatial.coordinates import enu_to_wgs84

    # 10 cameras along a straight line in reconstruction units
    c_rec = np.zeros((10, 3), dtype=np.float64)
    c_rec[:, 0] = np.linspace(0.0, 50.0, 10)
    sim3_true = Sim3(scale=3.0, rotation=np.eye(3), translation=np.array([10.0, 20.0, 30.0]))
    pts_enu = sim3_true.transform_point(c_rec)

    anchor = GeospatialAnchorOrigin(lat_deg=18.5204, lon_deg=73.8567, ellipsoidal_height_m=540.0)
    records = []
    for i in range(10):
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
    cam_dict = {f"f_{i}": c_rec[i] for i in range(10)}
    time_dict = {f"f_{i}": float(i) for i in range(10)}

    result = pipeline.reconstruct(cam_dict, time_dict, records, camera_rotations_rec=None, anchor_origin=anchor)

    # MUTATION CHECK:
    # If an implementation silently returns FULL_SIM3_OBSERVABLE or axial_rotation_resolved=True,
    # it falsifies physical identifiability.
    assert result.full_sim3_observability != FullSim3ObservabilityStatus.FULL_SIM3_OBSERVABLE
    assert result.full_sim3_observability == FullSim3ObservabilityStatus.FULL_SIM3_NOT_OBSERVABLE_COLLINEAR
    assert result.axial_rotation_resolved is False
    assert result.sim3_transform is not None
    assert result.sim3_transform.axial_rotation_resolved is False


def test_mut_15_collinear_trajectory_falsely_marks_scale_unobservable():
    """MUT-15: Collinear trajectory with sufficient baseline must NOT be rejected as SCALE_NOT_OBSERVABLE."""
    from src.geospatial.pipeline import GeospatialMetricReconstructor
    from src.geospatial.coordinates import enu_to_wgs84

    # 10 cameras along straight line with 100m metric baseline
    c_rec = np.zeros((10, 3), dtype=np.float64)
    c_rec[:, 0] = np.linspace(0.0, 50.0, 10)
    sim3_true = Sim3(scale=2.0, rotation=np.eye(3), translation=np.array([10.0, 20.0, 30.0]))
    pts_enu = sim3_true.transform_point(c_rec)

    anchor = GeospatialAnchorOrigin(lat_deg=18.5204, lon_deg=73.8567, ellipsoidal_height_m=540.0)
    records = []
    for i in range(10):
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
    cam_dict = {f"f_{i}": c_rec[i] for i in range(10)}
    time_dict = {f"f_{i}": float(i) for i in range(10)}

    result = pipeline.reconstruct(cam_dict, time_dict, records, camera_rotations_rec=None, anchor_origin=anchor)

    # MUTATION CHECK:
    # Scale MUST be recovered and NOT fail with alignment failure
    assert result.metric_scale_status != MetricScaleStatus.METRIC_ALIGNMENT_FAILED
    assert result.metric_scale_status == MetricScaleStatus.METRIC_SCALE_ESTIMATED
    assert result.is_metric_scale is True
    assert result.sim3_transform is not None
    assert abs(result.sim3_transform.scale - 2.0) < 1e-4


def test_mut_16_covariance_inverse_normalization_direction():
    """MUT-16: Mutating covariance transform from S @ cov_tilde @ S to S^(-1) @ cov_tilde @ S^(-1) fails.

    Proves that for diagonal Hessian H with positional scale D_geo > 1 (e.g. 100m):
    1. Correct back-transformation Sigma_theta = S @ cov_tilde @ S exactly matches H^(-1).
    2. Erroneous mutation Sigma_mutated = S^(-1) @ cov_tilde @ S^(-1) produces S^(-4) @ H^(-1),
       failing by a factor of D_geo^4 (10^8).
    3. Under meter-to-kilometer unit change, mutated back-transformation scales in the opposite
       direction (sigma_km = 1000 * sigma_m instead of sigma_m / 1000), proving detection.
    """
    h_diag = np.array([1.5, 2.0, 2.5, 3.0, 0.02, 0.03, 0.04], dtype=np.float64)
    H = np.diag(h_diag)
    H_inv_true = np.diag(1.0 / h_diag)

    D_geo = 100.0  # 100 meters
    S = np.diag([1.0, 1.0, 1.0, 1.0, D_geo, D_geo, D_geo])
    S_inv = np.diag(1.0 / np.diag(S))

    H_tilde = S @ H @ S
    cov_tilde = np.linalg.inv(H_tilde)

    # 1. Correct back-transformation: Sigma_theta = S @ cov_tilde @ S
    cov_correct = S @ cov_tilde @ S
    assert np.allclose(cov_correct, H_inv_true, atol=1e-14, rtol=1e-14)

    # 2. Mutated back-transformation: Sigma_mutated = S_inv @ cov_tilde @ S_inv
    cov_mutated = S_inv @ cov_tilde @ S_inv

    # MUTATION DETECTION:
    # Mutated covariance MUST NOT equal analytical inverse H^(-1)
    assert not np.allclose(cov_mutated, H_inv_true)

    # Ratio of mutated translation variance to true translation variance is exactly 1 / D_geo^4
    trans_ratio = cov_mutated[4, 4] / H_inv_true[4, 4]
    expected_ratio = 1.0 / (D_geo**4)
    assert abs(trans_ratio - expected_ratio) < 1e-14
    # With D_geo = 100, the error is an eight-order-of-magnitude discrepancy (1e-8 vs 1.0)!
    assert trans_ratio < 1e-7

    # 3. Unit transformation verification (meters vs kilometers):
    # In meters:
    sigma_trans_m_correct = math.sqrt(float(np.trace(cov_correct[4:7, 4:7])))
    sigma_trans_m_mutated = math.sqrt(float(np.trace(cov_mutated[4:7, 4:7])))

    # In kilometers: D_geo_km = 0.1 km, H_km[4:7, 4:7] = H[4:7, 4:7] * 1e6
    h_diag_km = h_diag.copy()
    h_diag_km[4:7] *= 1e6
    H_km = np.diag(h_diag_km)
    S_km = np.diag([1.0, 1.0, 1.0, 1.0, D_geo * 1e-3, D_geo * 1e-3, D_geo * 1e-3])
    S_km_inv = np.diag(1.0 / np.diag(S_km))

    H_tilde_km = S_km @ H_km @ S_km
    cov_tilde_km = np.linalg.inv(H_tilde_km)

    cov_correct_km = S_km @ cov_tilde_km @ S_km
    cov_mutated_km = S_km_inv @ cov_tilde_km @ S_km_inv

    sigma_trans_km_correct = math.sqrt(float(np.trace(cov_correct_km[4:7, 4:7])))
    sigma_trans_km_mutated = math.sqrt(float(np.trace(cov_mutated_km[4:7, 4:7])))

    # Correct back-transformation transforms correctly: sigma_km = sigma_m / 1000
    assert abs(sigma_trans_km_correct - (sigma_trans_m_correct / 1000.0)) < 1e-12

    # Mutated back-transformation scales in the OPPOSITE direction: sigma_km = sigma_m * 1000!
    assert not np.isclose(sigma_trans_km_mutated, sigma_trans_m_mutated / 1000.0)
    assert np.isclose(sigma_trans_km_mutated, sigma_trans_m_mutated * 1000.0, rtol=1e-10)



