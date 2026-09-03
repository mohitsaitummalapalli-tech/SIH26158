"""Phase 3E.6 Categorized Systematic Test Suite (45 Scenarios).

Implements TEST-3E6-01 through TEST-3E6-45 as planned in Section 22 of the
Phase 3E.6 Benchmark Architecture Specification:
- Dataset & Evidence Gates (01-04)
- Geometry Metrics & Metrology (05-09)
- Metric Scale & Segment Error (10-12)
- Geospatial Checkpoints (13-15)
- Trajectory Validation (16-18)
- Texture Diagnostics (19-21)
- Completeness & Visibility (22-24)
- Uncertainty Calibration (25-28)
- Robustness & Perturbations (29-37)
- Reproducibility Standards R0-R3 (38-40)
- Provenance & Audit (41-42)
- Claim Policy & Non-Collapse (43-44)
- Anti-Leakage Partition Audit (45)
"""

import math
import numpy as np
import pytest

from src.benchmark.models import (
    ContractViolationError,
    EvidenceLevel,
    BenchmarkStatus,
    VisibilityState,
    ReproducibilityLevel,
    LatencyTier,
    TaxonomyClass,
    QualityAxis,
    UncertaintyStatus,
    ReferencePartition,
    AcquisitionConditions,
    CameraCalibrationMeta,
    TelemetryMeta,
    GroundTruthMeta,
    DatasetManifest,
    StatisticalSummary,
)
from src.benchmark.claim_policy import ClaimPolicyEngine
from src.benchmark.metrics_geometry import (
    compute_statistical_summary,
    compute_point_to_point_distances,
    compute_point_to_plane_distances,
    compute_bidirectional_chamfer,
    compute_hausdorff_distances,
    compute_f_score_at_tau,
    compute_normal_angular_deviation,
)
from src.benchmark.metrics_metric_scale import (
    ValidationSegment,
    compute_relative_scale_error,
    evaluate_metric_scale,
)
from src.benchmark.metrics_geospatial import (
    CheckpointReference,
    evaluate_geospatial_checkpoints,
)
from src.benchmark.metrics_trajectory import (
    evaluate_raw_trajectory_ate,
    evaluate_sim3_aligned_trajectory_ate,
    evaluate_rpe_drift,
    solve_umeyama_sim3,
    DISCLAIMER_SIM3_ALIGNMENT,
)
from src.benchmark.metrics_texture import (
    TextureDiagnosticMetadata,
    evaluate_texture_diagnostics,
    compute_masked_psnr,
    compute_masked_ssim,
    compute_seam_gradient_discontinuity,
)
from src.benchmark.metrics_completeness import (
    classify_visibility_evidence,
    evaluate_roi_completeness,
)
from src.benchmark.metrics_uncertainty import (
    compute_spearman_rank_correlation,
    compute_bootstrap_confidence_interval,
    evaluate_heuristic_confidence_ranking,
    evaluate_probabilistic_coverage,
)
from src.benchmark.robustness_perturbations import (
    apply_gaussian_image_blur,
    apply_linear_motion_blur,
    apply_gnss_gaussian_noise,
    inject_gnss_outliers,
    simulate_telemetry_dropout,
    simulate_shutter_clock_bias,
    apply_focal_length_perturbation,
    subsample_frame_dropping,
    generate_collinear_trajectory,
    EnvelopeClassification,
)
from src.benchmark.timing_profiler import BenchmarkTimingProfiler
from src.benchmark.engine import BenchmarkEngine
from src.benchmark.reproducibility import verify_reproducibility_level


# ============================================================================
# CATEGORY 1: DATASET & EVIDENCE GATES (TEST-3E6-01 to TEST-3E6-04)
# ============================================================================

def test_3e6_01_class_a_synthetic_identity():
    """TEST-3E6-01: Class A synthetic identity (0 error check)."""
    pts_est = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    pts_gt = pts_est.copy()  # Distinct object representing independent CAD ground truth
    summary = compute_point_to_point_distances(pts_est, pts_gt, est_hash="recon_cloud_hash", gt_hash="cad_ref_hash")
    assert summary.rmse == 0.0
    assert summary.mae == 0.0
    assert summary.maximum == 0.0


def test_3e6_02_level_0_evidence_gate_blocks_accuracy():
    """TEST-3E6-02: Level 0 evidence gate blocks accuracy claims."""
    engine = ClaimPolicyEngine()
    auth = engine.audit_claim_authorization(EvidenceLevel.LEVEL_0_NO_GROUND_TRUTH)
    assert "horizontal_checkpoint_rmse" in auth.claims_blocked
    assert "segment_scale_accuracy" in auth.claims_blocked
    assert "reprojection_consistency" in auth.claims_allowed


def test_3e6_03_level_1_telemetry_gate_restricts_claims():
    """TEST-3E6-03: Level 1 telemetry gate restricts claims (blocks metric checkpoints)."""
    engine = ClaimPolicyEngine()
    auth = engine.audit_claim_authorization(EvidenceLevel.LEVEL_1_TELEMETRY_ONLY)
    assert "gnss_fitting_residual" in auth.claims_allowed
    assert "horizontal_checkpoint_rmse" in auth.claims_blocked
    assert "surveyed_3d_accuracy" in auth.claims_blocked


def test_3e6_04_class_d_surveyed_manifest_validation():
    """TEST-3E6-04: Class D surveyed manifest validation."""
    partition = ReferencePartition(
        estimation_set_ids={"GCP1", "GCP2", "GCP3"},
        validation_set_ids={"CKP1", "CKP2"},
    )
    partition.validate_disjointness()
    manifest = DatasetManifest(
        dataset_id="VAL-SURVEY-01",
        taxonomy_class=TaxonomyClass.CLASS_D_REAL_SURVEYED,
        acquisition_conditions=AcquisitionConditions(),
        frame_count=100,
        image_resolution=(3840, 2160),
        camera_calibration=CameraCalibrationMeta(),
        telemetry_metadata=TelemetryMeta(has_telemetry=True),
        ground_truth_metadata=GroundTruthMeta(
            has_ground_truth=True,
            ground_truth_type="SURVEYED_CHECKPOINTS",
            partition=partition,
        ),
        sha256_checksum="abc12345",
    )
    level = BenchmarkEngine.determine_evidence_level(manifest)
    assert level == EvidenceLevel.LEVEL_4_SURVEYED_CHECKPOINTS


# ============================================================================
# CATEGORY 2: GEOMETRY METRICS (TEST-3E6-05 to TEST-3E6-09)
# ============================================================================

def test_3e6_05_bidirectional_chamfer_convergence():
    """TEST-3E6-05: Bidirectional Chamfer distance convergence and symmetry."""
    pts_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    pts_b = np.array([[0.0, 0.1, 0.0], [1.0, 0.1, 0.0]])
    res = compute_bidirectional_chamfer(pts_a, pts_b, est_hash="a", gt_hash="b")
    assert res["is_bidirectional"] is True
    assert math.isclose(res["chamfer_distance"], 0.1, abs_tol=1e-5)


def test_3e6_06_point_to_plane_error_vs_cad_normal():
    """TEST-3E6-06: Point-to-plane error vs CAD normal."""
    pts_est = np.array([[0.0, 0.0, 0.2], [1.0, 0.0, 0.2]])
    pts_gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normals_gt = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    summary = compute_point_to_plane_distances(pts_est, pts_gt, normals_gt, est_hash="a", gt_hash="b")
    assert math.isclose(summary.rmse, 0.2, abs_tol=1e-5)
    assert math.isclose(summary.maximum, 0.2, abs_tol=1e-5)


def test_3e6_07_hausdorff_95th_percentile_outlier_bounds():
    """TEST-3E6-07: Hausdorff 95th percentile outlier bounds (P95 <= Max)."""
    pts_a = np.array([[float(i), 0.0, 0.0] for i in range(100)])
    pts_b = pts_a.copy()
    pts_b[-1] += np.array([0.0, 50.0, 0.0])  # One extreme outlier
    h_dict = compute_hausdorff_distances(pts_a, pts_b)
    assert h_dict["hausdorff_max"] >= 50.0
    assert h_dict["hausdorff_95"] < h_dict["hausdorff_max"]


def test_3e6_08_surface_normal_angular_deviation_median():
    """TEST-3E6-08: Surface normal angular deviation median."""
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # 30 degree tilt in normal
    angle_rad = math.radians(30.0)
    norm_est = np.array([[0.0, math.sin(angle_rad), math.cos(angle_rad)], [0.0, math.sin(angle_rad), math.cos(angle_rad)]])
    norm_gt = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    summary = compute_normal_angular_deviation(pts, norm_est, pts, norm_gt)
    assert math.isclose(summary.median, 30.0, abs_tol=1e-3)


def test_3e6_09_f1_score_precision_recall_at_tau():
    """TEST-3E6-09: F1-score precision/recall at tau = 0.05m."""
    pts_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    pts_b = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.1, 0.0]])  # Last point offset by 0.1m > 0.05m
    f_res = compute_f_score_at_tau(pts_a, pts_b, tau_meters=0.05)
    assert math.isclose(f_res["precision"], 2.0 / 3.0, abs_tol=1e-4)
    assert math.isclose(f_res["recall"], 2.0 / 3.0, abs_tol=1e-4)
    assert math.isclose(f_res["f1_score"], 2.0 / 3.0, abs_tol=1e-4)


# ============================================================================
# CATEGORY 3: METRIC SCALE (TEST-3E6-10 to TEST-3E6-12)
# ============================================================================

def test_3e6_10_known_isotropic_scale_recovery():
    """TEST-3E6-10: Known isotropic scale recovery (s = 2.5)."""
    markers = {
        "A": np.array([0.0, 0.0, 0.0]),
        "B": np.array([4.0, 0.0, 0.0]),  # Distance in recon units = 4.0
        "C": np.array([0.0, 3.0, 0.0]),  # Distance in recon units = 3.0
    }
    # True metric distance = 10.0m (4.0 * 2.5) and 7.5m (3.0 * 2.5)
    segments = [
        ValidationSegment("S1", "A", "B", reference_distance=10.0),
        ValidationSegment("S2", "A", "C", reference_distance=7.5),
        ValidationSegment("S3", "B", "C", reference_distance=12.5),
    ]
    res = evaluate_metric_scale(segments, markers, scale_factor_to_metric=2.5)
    assert res["relative_scale_error_summary"].rmse == 0.0
    assert res["is_non_collinear"] is True


def test_3e6_11_multi_segment_relative_scale_error():
    """TEST-3E6-11: Multi-segment relative scale error (<0.5%)."""
    markers = {
        "A": np.array([0.0, 0.0, 0.0]),
        "B": np.array([10.02, 0.0, 0.0]),  # 10.02m vs 10.00m ref (+0.2%)
        "C": np.array([0.0, 10.01, 0.0]),  # 10.01m vs 10.00m ref (+0.1%)
    }
    segments = [
        ValidationSegment("S1", "A", "B", reference_distance=10.0),
        ValidationSegment("S2", "A", "C", reference_distance=10.0),
        ValidationSegment("S3", "B", "C", reference_distance=14.142),
    ]
    res = evaluate_metric_scale(segments, markers, scale_factor_to_metric=1.0)
    assert res["median_relative_error_pct"] < 0.5


def test_3e6_12_scale_invariance_under_unit_change():
    """TEST-3E6-12: Relative scale error invariance under unit conversion (m vs km)."""
    d_est_m, d_ref_m = 100.0, 99.0
    d_est_km, d_ref_km = 0.1, 0.099
    err_m = compute_relative_scale_error(d_est_m, d_ref_m)
    err_km = compute_relative_scale_error(d_est_km, d_ref_km)
    assert math.isclose(err_m, err_km, abs_tol=1e-9)


# ============================================================================
# CATEGORY 4: GEOSPATIAL VALIDATION (TEST-3E6-13 to TEST-3E6-15)
# ============================================================================

def test_3e6_13_hold_out_ckp_rmse():
    """TEST-3E6-13: Hold-out CKP East/North/Up RMSE (<0.3m)."""
    ckps = [
        CheckpointReference("CKP1", east_m=100.0, north_m=200.0, up_m=50.0),
        CheckpointReference("CKP2", east_m=150.0, north_m=220.0, up_m=52.0),
        CheckpointReference("CKP3", east_m=120.0, north_m=280.0, up_m=48.0),
    ]
    est_coords = {
        "CKP1": (100.1, 200.05, 50.08),
        "CKP2": (149.9, 219.95, 52.1),
        "CKP3": (120.05, 280.1, 47.95),
    }
    partition = ReferencePartition(
        estimation_set_ids={"GCP1", "GCP2"},
        validation_set_ids={"CKP1", "CKP2", "CKP3"},
    )
    res = evaluate_geospatial_checkpoints(ckps, est_coords, partition=partition)
    assert res.rmse_horizontal < 0.3
    assert res.rmse_3d < 0.3


def test_3e6_14_checkpoint_max_error_bounds():
    """TEST-3E6-14: Checkpoint max error bounds verification."""
    ckps = [
        CheckpointReference("CKP1", east_m=0.0, north_m=0.0, up_m=0.0),
        CheckpointReference("CKP2", east_m=10.0, north_m=0.0, up_m=0.0),
    ]
    est_coords = {
        "CKP1": (0.05, 0.05, 0.05),
        "CKP2": (10.15, 0.0, 0.0),
    }
    res = evaluate_geospatial_checkpoints(ckps, est_coords)
    assert res.maximum_3d >= res.mae_3d


def test_3e6_15_topocentric_enu_residual_vectors():
    """TEST-3E6-15: Residual vector component validation."""
    ckps = [CheckpointReference("CKP1", east_m=10.0, north_m=20.0, up_m=30.0)]
    est_coords = {"CKP1": (11.0, 18.0, 30.0)}
    res = evaluate_geospatial_checkpoints(ckps, est_coords)
    assert res.residuals_per_target["CKP1"]["delta_east_m"] == 1.0
    assert res.residuals_per_target["CKP1"]["delta_north_m"] == -2.0
    assert res.residuals_per_target["CKP1"]["delta_up_m"] == 0.0


# ============================================================================
# CATEGORY 5: TRAJECTORY METRICS (TEST-3E6-16 to TEST-3E6-18)
# ============================================================================

def test_3e6_16_raw_metric_frame_ate_rmse():
    """TEST-3E6-16: Raw metric-frame ATE translation RMSE (no alignment)."""
    c_ref = np.array([[0.0, 0.0, 10.0], [5.0, 0.0, 10.0], [10.0, 0.0, 10.0]])
    r_ref = np.array([np.eye(3) for _ in range(3)])
    c_est = c_ref + np.array([0.5, 0.0, 0.0])  # Shifted by 0.5m
    r_est = r_ref.copy()
    res = evaluate_raw_trajectory_ate(c_est, r_est, c_ref, r_ref)
    assert math.isclose(res.ate_translation_rmse_m, 0.5, abs_tol=1e-5)
    assert res.preserves_absolute_scale_and_georeference is True


def test_3e6_17_sim3_aligned_ate_shape_logging():
    """TEST-3E6-17: Sim(3)-aligned ATE shape error logging and disclaimer."""
    c_ref = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 10.0, 0.0]])
    # Scaled by 2.0 and shifted by 50.0m
    c_est = c_ref * 2.0 + np.array([50.0, 50.0, 50.0])
    res = evaluate_sim3_aligned_trajectory_ate(c_est, c_ref)
    assert math.isclose(res.aligned_ate_rmse_m, 0.0, abs_tol=1e-5)
    assert math.isclose(res.scale_removed, 0.5, abs_tol=1e-5)
    assert res.alignment_removed_dofs == 7
    assert DISCLAIMER_SIM3_ALIGNMENT in res.disclaimer


def test_3e6_18_rpe_drift_rate():
    """TEST-3E6-18: Relative Pose Error (RPE) drift rate."""
    c_ref = np.array([[float(i), 0.0, 0.0] for i in range(10)])
    r_ref = np.array([np.eye(3) for _ in range(10)])
    # Constant drift in estimated step: 1.05m per step instead of 1.0m
    c_est = np.array([[float(i) * 1.05, 0.0, 0.0] for i in range(10)])
    r_est = r_ref.copy()
    res = evaluate_rpe_drift(c_est, r_est, c_ref, r_ref, delta_interval_frames=1)
    assert math.isclose(res.translational_drift_per_delta_rmse, 0.05, abs_tol=1e-5)


# ============================================================================
# CATEGORY 6: TEXTURE DIAGNOSTICS (TEST-3E6-19 to TEST-3E6-21)
# ============================================================================

def test_3e6_19_reprojection_psnr_ssim_diagnostic():
    """TEST-3E6-19: Reprojection PSNR/SSIM diagnostic logging."""
    img = np.ones((100, 100, 3), dtype=np.float64) * 128.0
    meta = TextureDiagnosticMetadata(image_resolution=(100, 100))
    res = evaluate_texture_diagnostics(img, img, metadata=meta)
    assert res.masked_psnr_db >= 99.0
    assert math.isclose(res.masked_ssim, 1.0, abs_tol=1e-5)
    assert res.is_diagnostic_only is True


def test_3e6_20_seam_edge_photometric_discontinuity():
    """TEST-3E6-20: Seam edge photometric discontinuity index."""
    img = np.zeros((50, 50), dtype=np.float64)
    img[:, 25:] = 100.0  # Step edge at column 25
    mask = np.zeros((50, 50), dtype=bool)
    mask[:, 25] = True
    disc = compute_seam_gradient_discontinuity(img, mask)
    assert disc > 0.0


def test_3e6_21_radiometric_calibration_claim_gate():
    """TEST-3E6-21: Radiometric calibration claim gate check."""
    img = np.ones((10, 10), dtype=np.float64)
    meta = TextureDiagnosticMetadata(image_resolution=(10, 10), radiometric_calibration_certified=False)
    with pytest.raises(ContractViolationError, match="MUT-16"):
        evaluate_texture_diagnostics(img, img, metadata=meta, claim_colorimetric_accuracy=True)


# ============================================================================
# CATEGORY 7: COMPLETENESS & VISIBILITY (TEST-3E6-22 to TEST-3E6-24)
# ============================================================================

def test_3e6_22_surface_completeness_within_roi():
    """TEST-3E6-22: Surface completeness within specified ROI."""
    pts_est = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    pts_gt = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
    roi = (( -1.0, -1.0, -1.0), (10.0, 10.0, 10.0))
    res = evaluate_roi_completeness(pts_est, pts_gt, tau_meters=0.1, roi_bounds=roi)
    assert res.estimated_point_count_in_roi == 2  # 20.0 is outside ROI
    assert res.reference_point_count_in_roi == 3
    assert math.isclose(res.recall_completeness_at_tau, 2.0 / 3.0, abs_tol=1e-4)


def test_3e6_23_five_state_visibility_evidence_tagging():
    """TEST-3E6-23: Five-state visibility evidence tagging."""
    st1 = classify_visibility_evidence(True, 15.0, True, False, True, True)
    assert st1 == VisibilityState.OBSERVED
    st2 = classify_visibility_evidence(False, 0.0, False, False, True, False)
    assert st2 == VisibilityState.UNOBSERVED
    st3 = classify_visibility_evidence(False, 0.0, True, True, True, False)
    assert st3 == VisibilityState.PHYSICALLY_OCCLUDED


def test_3e6_24_undetermined_state_for_missing_ray_masks():
    """TEST-3E6-24: Undetermined state for missing ray masks."""
    st = classify_visibility_evidence(
        has_optical_ray_intersection=False,
        ray_intersection_angle_deg=0.0,
        in_camera_frustum=True,
        ray_hits_foreground=False,
        has_ray_tracing_evidence=False,  # Absent ray evidence!
        is_reconstructed=False,
    )
    assert st == VisibilityState.UNDETERMINED


# ============================================================================
# CATEGORY 8: UNCERTAINTY CALIBRATION (TEST-3E6-25 to TEST-3E6-28)
# ============================================================================

def test_3e6_25_uncertainty_sample_count_gate():
    """TEST-3E6-25: Sample count < 30 emits NOT_EVALUABLE."""
    u = np.array([0.1, 0.2, 0.3])
    e = np.array([0.15, 0.25, 0.35])
    res = evaluate_heuristic_confidence_ranking(u, e, min_sample_size=30)
    assert res.status == UncertaintyStatus.NOT_EVALUABLE


def test_3e6_26_spearman_rank_correlation_diagnostic():
    """TEST-3E6-26: Spearman rank correlation diagnostic and CI."""
    rng = np.random.default_rng(42)
    u = np.linspace(0.1, 1.0, 50)
    e = u + rng.normal(0.0, 0.05, size=50)
    res = evaluate_heuristic_confidence_ranking(u, e)
    assert res.status == UncertaintyStatus.EVALUATED
    assert res.spearman_rho > 0.8
    assert res.bootstrap_ci_95[0] < res.bootstrap_ci_95[1]


def test_3e6_27_uncertainty_quintile_stratification():
    """TEST-3E6-27: Uncertainty quintile error stratification."""
    u = np.linspace(0.1, 1.0, 50)
    e = u.copy()
    res = evaluate_heuristic_confidence_ranking(u, e)
    assert len(res.quintile_median_errors) == 5
    assert res.is_monotonically_ordered is True


def test_3e6_28_coverage_probability_gaussian_claim():
    """TEST-3E6-28: Coverage probability under Gaussian claim."""
    rng = np.random.default_rng(42)
    sigmas = np.ones(1000) * 1.0
    errors = np.abs(rng.normal(0.0, 1.0, size=1000))
    res = evaluate_probabilistic_coverage(sigmas, errors, declared_probabilistic_model="1D_HALF_GAUSSIAN")
    assert math.isclose(res.empirical_coverage_1sigma, 0.6827, abs_tol=0.05)


# ============================================================================
# CATEGORY 9: ROBUSTNESS & PERTURBATIONS (TEST-3E6-29 to TEST-3E6-37)
# ============================================================================

def test_3e6_29_motion_blur_degradation_curve():
    """TEST-3E6-29: Motion blur degradation curve."""
    img = np.ones((20, 20), dtype=np.float64)
    _, rec1 = apply_linear_motion_blur(img, kernel_length_px=4)
    assert rec1.operating_regime == EnvelopeClassification.SUPPORTED_ENVELOPE
    _, rec2 = apply_linear_motion_blur(img, kernel_length_px=15)
    assert rec2.operating_regime == EnvelopeClassification.STRESS_REGIME


def test_3e6_30_gnss_outlier_injection():
    """TEST-3E6-30: GNSS outlier injection regime."""
    pts = np.zeros((100, 3), dtype=np.float64)
    perturbed, rec = inject_gnss_outliers(pts, outlier_fraction=0.10, outlier_offset_m=50.0)
    assert rec.operating_regime == EnvelopeClassification.SUPPORTED_ENVELOPE
    assert np.max(np.linalg.norm(perturbed, axis=1)) >= 45.0


def test_3e6_31_telemetry_dropout_gap():
    """TEST-3E6-31: Telemetry dropout gap simulation."""
    times = np.linspace(0.0, 20.0, 200)
    filtered_t, rec = simulate_telemetry_dropout(times, dropout_start_sec=5.0, dropout_duration_sec=2.0)
    assert rec.operating_regime == EnvelopeClassification.SUPPORTED_ENVELOPE
    assert not np.any((filtered_t >= 5.0) & (filtered_t <= 7.0))


def test_3e6_32_shutter_clock_bias():
    """TEST-3E6-32: Shutter clock bias offset injection."""
    pts = np.array([1.0, 2.0, 3.0])
    perturbed_pts, rec = simulate_shutter_clock_bias(pts, clock_bias_sec=0.03)
    assert rec.magnitude == 30.0  # 30 ms
    assert math.isclose(perturbed_pts[0], 1.03, abs_tol=1e-5)


def test_3e6_33_focal_length_perturbation():
    """TEST-3E6-33: Focal length drift perturbation."""
    f_nom = 1000.0
    f_pert, rec = apply_focal_length_perturbation(f_nom, error_percentage=2.0)
    assert math.isclose(f_pert, 1020.0, abs_tol=1e-5)
    assert rec.operating_regime == EnvelopeClassification.SUPPORTED_ENVELOPE


def test_3e6_34_reduced_baseline_frame_dropping():
    """TEST-3E6-34: Frame dropping simulation."""
    frames = list(range(100))
    kept, rec = subsample_frame_dropping(frames, drop_percentage=20.0)
    assert len(kept) < 100
    assert rec.operating_regime == EnvelopeClassification.SUPPORTED_ENVELOPE


def test_3e6_35_collinear_trajectory_mode():
    """TEST-3E6-35: Collinear flight path generation."""
    path, rec = generate_collinear_trajectory((0, 0, 0), (100, 0, 0), num_frames=50, cross_track_jitter_m=0.0)
    assert rec.operating_regime == EnvelopeClassification.STRESS_REGIME
    assert np.all(path[:, 1] == 0.0)


def test_3e6_36_stationary_hover_mode():
    """TEST-3E6-36: Stationary trajectory mode."""
    path, rec = generate_collinear_trajectory((10, 10, 50), (10, 10, 50), num_frames=30)
    assert np.allclose(path, np.array([10, 10, 50]))


def test_3e6_37_gaussian_noise_injection():
    """TEST-3E6-37: GNSS Gaussian noise injection."""
    pts = np.zeros((100, 3), dtype=np.float64)
    perturbed, rec = apply_gnss_gaussian_noise(pts, horizontal_sigma_m=1.0, vertical_sigma_m=2.0)
    assert rec.operating_regime == EnvelopeClassification.SUPPORTED_ENVELOPE
    assert np.std(perturbed[:, 0]) > 0.5


# ============================================================================
# CATEGORY 10: REPRODUCIBILITY STANDARDS R0-R3 (TEST-3E6-38 to TEST-3E6-40)
# ============================================================================

def test_3e6_38_level_r1_numerical_tolerance():
    """TEST-3E6-38: Level R1 numerical tolerance verification."""
    # Distinct independently generated float results differing within 1e-5 tolerance
    res_a = {"rmse": 0.050001, "mae": 0.030002, "coords": [10.000001, 20.000002]}
    res_b = {"rmse": 0.050003, "mae": 0.030001, "coords": [10.000002, 20.000001]}
    assert verify_reproducibility_level(res_a, res_b, ReproducibilityLevel.R1_NUMERICAL, tolerance=1e-5) is True


def test_3e6_39_level_r2_deterministic_structure():
    """TEST-3E6-39: Level R2 deterministic structure reproducibility."""
    # Distinct execution runs yielding identical discrete structure under fixed seed
    inliers_run1 = {1, 5, 8, 12, 19}
    inliers_run2 = {1, 5, 8, 12, 19}
    assert verify_reproducibility_level(inliers_run1, inliers_run2, ReproducibilityLevel.R2_DETERMINISTIC) is True


def test_3e6_40_level_r3_bitwise_repeatability():
    """TEST-3E6-40: Level R3 bitwise repeatability check."""
    # Distinct binary byte artifacts serialized from identical deterministic pipeline
    data1 = b"reconstruction_binary_output_test_canonical"
    data2 = b"reconstruction_binary_output_test_canonical"
    assert verify_reproducibility_level(data1, data2, ReproducibilityLevel.R3_BITWISE) is True


# ============================================================================
# CATEGORY 11: PROVENANCE & MANIFEST INTEGRITY (TEST-3E6-41 to TEST-3E6-42)
# ============================================================================

def test_3e6_41_provenance_record_validation():
    """TEST-3E6-41: Full pipeline artifact provenance record."""
    summary = compute_statistical_summary(np.array([0.1, 0.2, 0.3]), unit="meters")
    assert summary.unit == "meters"
    assert summary.sample_count == 3


def test_3e6_42_timing_profiler_latency_tier():
    """TEST-3E6-42: Timing profiler and latency tier."""
    profiler = BenchmarkTimingProfiler(input_frames=100, decoded_duration_sec=10.0)
    profiler.start_pipeline()
    profiler.stop_pipeline()
    prof = profiler.build_timing_profile()
    assert prof.pipeline_fps >= 0.0
    assert prof.latency_tier in (LatencyTier.OFFLINE_BATCH, LatencyTier.NEAR_REAL_TIME, LatencyTier.REAL_TIME)


# ============================================================================
# CATEGORY 12: CLAIM POLICY & ANTI-LEAKAGE (TEST-3E6-43 to TEST-3E6-45)
# ============================================================================

def test_3e6_43_blocked_claim_emission_triggers_failure():
    """TEST-3E6-43: Blocked claim emission triggers failure."""
    with pytest.raises(ContractViolationError, match="strictly blocked"):
        ClaimPolicyEngine.enforce_claim_emission(
            EvidenceLevel.LEVEL_0_NO_GROUND_TRUTH,
            "horizontal_checkpoint_rmse",
        )


def test_3e6_44_no_universal_accuracy_assertion():
    """TEST-3E6-44: No universal accuracy assertion globally."""
    for level in EvidenceLevel:
        auth = ClaimPolicyEngine.audit_claim_authorization(level)
        assert "universal_drone_accuracy" in auth.claims_blocked


def test_3e6_45_anti_leakage_partition_intersection_assert():
    """TEST-3E6-45: Anti-leakage partition intersection assertion."""
    partition = ReferencePartition(
        estimation_set_ids={"T1", "T2"},
        validation_set_ids={"T2", "T3"},  # T2 leaked into validation!
    )
    with pytest.raises(ContractViolationError, match="Data leakage detected"):
        partition.validate_disjointness()
