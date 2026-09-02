"""Deterministic unit tests for Phase 3 Classical Geometry Contracts & Mathematical Models.

DISCLAIMER:
ALL FIXTURES IN THIS MODULE ARE SYNTHETIC TEST DATA GENERATED SOLELY FOR
GEOMETRIC CONTRACTS AUDITING. THEY DO NOT REPRESENT A REAL RECONSTRUCTION.
"""

import math
import numpy as np
import pytest

from src.geometry import (
    EvaluationLevel,
    PipelineStageStatus,
    MeasurementType,
    CompletenessMetricType,
    DistortionModel,
    DistortionStatus,
    GaugeFixingPolicy,
    GeometryFailureReason,
    GeometryThresholdConfig,
    TrajectoryEvaluationProvenance,
    CameraIntrinsics,
    ExtrinsicPose,
    FeatureKeypoint,
    FeatureCorrespondences,
    TwoViewGeometryResult,
    TriangulatedTrack,
    SparseReconstructionResult,
    DenseMVSInput,
    DenseMVSOutput,
    GeometryMathContracts,
)


# 1. Pinhole Camera Model Projection Math
def test_pinhole_projection_math():
    fx, fy, cx, cy = 1000.0, 1000.0, 500.0, 500.0
    r_mat = np.eye(3)
    t_vec = np.array([0.0, 0.0, 0.0])

    # 3D Point along optical axis at (0, 0, 10) -> projects exactly to principal point (500, 500)
    pt_w = np.array([0.0, 0.0, 10.0])
    u, v, zc = GeometryMathContracts.project_point(pt_w, r_mat, t_vec, fx, fy, cx, cy)

    assert math.isclose(u, 500.0, abs_tol=1e-6)
    assert math.isclose(v, 500.0, abs_tol=1e-6)
    assert math.isclose(zc, 10.0, abs_tol=1e-6)


# 2. Off-Center Projection & Optical Depth
def test_off_center_projection():
    fx, fy, cx, cy = 1000.0, 1000.0, 500.0, 500.0
    r_mat = np.eye(3)
    t_vec = np.array([0.0, 0.0, 0.0])

    # 3D Point at (1, 2, 10) -> u = 1000 * (1/10) + 500 = 600, v = 1000 * (2/10) + 500 = 700
    pt_w = np.array([1.0, 2.0, 10.0])
    u, v, zc = GeometryMathContracts.project_point(pt_w, r_mat, t_vec, fx, fy, cx, cy)

    assert math.isclose(u, 600.0, abs_tol=1e-6)
    assert math.isclose(v, 700.0, abs_tol=1e-6)
    assert math.isclose(zc, 10.0, abs_tol=1e-6)


# 3. Reprojection Error Exact Zero on True Projection
def test_reprojection_error_exact_zero():
    fx, fy, cx, cy = 800.0, 800.0, 400.0, 300.0
    r_mat = np.eye(3)
    t_vec = np.array([0.0, 0.0, 0.0])
    pt_w = np.array([2.0, 1.0, 5.0])

    u, v, zc = GeometryMathContracts.project_point(pt_w, r_mat, t_vec, fx, fy, cx, cy)
    err = GeometryMathContracts.compute_reprojection_error(
        pt_w, (u, v), r_mat, t_vec, fx, fy, cx, cy
    )

    assert math.isclose(err, 0.0, abs_tol=1e-9)


# 4. Reprojection Error with Pixel Perturbation
def test_reprojection_error_perturbed():
    fx, fy, cx, cy = 800.0, 800.0, 400.0, 300.0
    r_mat = np.eye(3)
    t_vec = np.array([0.0, 0.0, 0.0])
    pt_w = np.array([2.0, 1.0, 5.0])

    u, v, _ = GeometryMathContracts.project_point(pt_w, r_mat, t_vec, fx, fy, cx, cy)
    # Observed point shifted by 3px horizontally and 4px vertically (expected Euclidean error = 5.0px)
    perturbed_obs = (u + 3.0, v + 4.0)
    err = GeometryMathContracts.compute_reprojection_error(
        pt_w, perturbed_obs, r_mat, t_vec, fx, fy, cx, cy
    )

    assert math.isclose(err, 5.0, abs_tol=1e-6)


# 5. Epipolar Constraint Verification (x_2^T * E * x_1 = 0 vs x_2^T * F * x_1 = 0)
def test_epipolar_constraint_verification():
    # Pure horizontal translation t = [1, 0, 0], R = I
    # Essential matrix E = [t]_x R = [[0, 0, 0], [0, 0, -1], [0, 1, 0]]
    E = np.array([
        [0.0,  0.0,  0.0],
        [0.0,  0.0, -1.0],
        [0.0,  1.0,  0.0],
    ])

    # 3D point at (0, 0, 10) in frame 1: x_1 = (0, 0, 1)
    # In frame 2 (translated by +1 in X): point is (-1, 0, 10), so x_2 = (-0.1, 0, 1)
    x1_norm = (0.0, 0.0)
    x2_norm = (-0.1, 0.0)

    err = GeometryMathContracts.verify_essential_matrix_constraint(x1_norm, x2_norm, E)
    assert math.isclose(err, 0.0, abs_tol=1e-6)


# 6. Cheirality Positive vs Negative Depth Validation
def test_cheirality_validation():
    r_mat = np.eye(3)
    t_vec = np.array([0.0, 0.0, 0.0])

    # Point in front of camera (Z_c = +5.0) -> Valid cheirality
    pt_front = np.array([0.0, 0.0, 5.0])
    assert GeometryMathContracts.check_cheirality(pt_front, r_mat, t_vec) is True

    # Point behind camera (Z_c = -2.0) -> Invalid cheirality
    pt_behind = np.array([0.0, 0.0, -2.0])
    assert GeometryMathContracts.check_cheirality(pt_behind, r_mat, t_vec) is False


# 7. Degenerate Geometry: Zero-Baseline Pure Rotation
def test_zero_baseline_pure_rotation_degenerate():
    # When translation t = 0, E = [0]_x R = 0 (rank 0 matrix)
    E_zero = np.zeros((3, 3))
    rank = np.linalg.matrix_rank(E_zero)
    assert rank == 0


# 8. MeasurementType Scientific Classification
def test_measurement_type_classification():
    kp = FeatureKeypoint(x=120.5, y=340.2, measurement_type=MeasurementType.DIRECTLY_OBSERVED)
    assert kp.measurement_type == MeasurementType.DIRECTLY_OBSERVED

    track = TriangulatedTrack(
        track_id=1,
        world_point=np.array([10.0, 20.0, 5.0]),
        observations={"f_0": (120.0, 340.0), "f_1": (125.0, 340.0)},
        reprojection_errors={"f_0": 0.5, "f_1": 0.4},
        measurement_type=MeasurementType.ESTIMATED,
    )
    assert track.measurement_type == MeasurementType.ESTIMATED


# 9. GeometryFailureReason Enumeration & Serialization
def test_failure_reason_serialization():
    res = TwoViewGeometryResult(
        frame_a_id="f_0",
        frame_b_id="f_1",
        is_degenerate=True,
        failure_reason=GeometryFailureReason.PURE_ROTATION_RISK,
        diagnostics=["Baseline too small relative to scene depth."],
    )

    d = res.to_dict()
    assert d["is_degenerate"] is True
    assert d["failure_reason"] == "PURE_ROTATION_RISK"
    assert len(d["diagnostics"]) == 1


# 10. SparseReconstructionResult Contract
def test_sparse_reconstruction_result_contract():
    intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000)
    pose0 = ExtrinsicPose(frame_index=0, timestamp_seconds=0.0)
    pose1 = ExtrinsicPose(frame_index=1, timestamp_seconds=0.5, translation_vector=[2.0, 0.0, 0.0])

    recon = SparseReconstructionResult(
        camera_poses={"f_0": pose0, "f_1": pose1},
        intrinsics={"f_0": intrinsics, "f_1": intrinsics},
        points3d={},
        mean_reprojection_rmse_px=0.65,
        percentile_90_reprojection_error_px=1.10,
        total_registered_cameras=2,
        total_triangulated_points=150,
        mean_track_length=2.8,
        is_metric_scale=False,
        failure_reason=None,
        gauge_policy=GaugeFixingPolicy.FIX_FIRST_CAMERA_AND_UNIT_BASELINE,
        provenance={"algorithm": "IncrementalSfM_v1.0"},
    )

    d = recon.to_dict()
    assert d["total_registered_cameras"] == 2
    assert d["mean_reprojection_rmse_px"] == 0.65
    assert d["evaluation_level"] == "LEVEL_1_IMAGE_SPACE_CONSISTENCY"
    assert d["has_monocular_scale_ambiguity"] is True
    assert d["gauge_policy"] == "FIX_FIRST_CAMERA_AND_UNIT_BASELINE"


# 11. DenseMVSInput & DenseMVSOutput Contracts
def test_dense_mvs_contracts():
    recon = SparseReconstructionResult(
        camera_poses={},
        intrinsics={},
        points3d={},
        mean_reprojection_rmse_px=0.0,
        percentile_90_reprojection_error_px=0.0,
        total_registered_cameras=0,
        total_triangulated_points=0,
        mean_track_length=0.0,
    )

    mvs_in = DenseMVSInput(
        sparse_reconstruction=recon,
        selected_frame_ids=["f_0", "f_1"],
    )
    assert len(mvs_in.selected_frame_ids) == 2

    mvs_out = DenseMVSOutput(
        sparse_sfm_status=PipelineStageStatus.SUCCESS,
        camera_registration_status=PipelineStageStatus.SUCCESS,
        dense_depth_status=PipelineStageStatus.SUCCESS,
        point_cloud_fusion_status=PipelineStageStatus.SUCCESS,
        depth_maps={"f_0": np.ones((100, 100), dtype=np.float32)},
        confidence_maps={"f_0": np.ones((100, 100), dtype=np.float32)},
        fused_point_count=10000,
        completeness_metric_type=CompletenessMetricType.REFERENCE_POINT_COMPLETENESS,
    )
    assert mvs_out.fused_point_count == 10000
    assert mvs_out.dense_depth_status == PipelineStageStatus.SUCCESS
    assert mvs_out.completeness_metric_type == CompletenessMetricType.REFERENCE_POINT_COMPLETENESS
    assert "f_0" in mvs_out.depth_maps


# 12. Heuristic Threshold Configuration Semantics (HEURISTIC_DEFAULT)
def test_heuristic_threshold_config_semantics():
    cfg = GeometryThresholdConfig()
    assert cfg.min_feature_count == 100
    assert cfg.min_candidate_matches == 30
    assert cfg.min_inlier_ratio == 0.20
    assert cfg.weak_baseline_parallax_deg == 1.0
    assert cfg.min_sparse_points == 50
    assert cfg.min_registered_cameras == 3
    assert cfg.max_reprojection_rmse_px == 2.0


# 13. Evaluation Level Hierarchy Separation
def test_evaluation_level_hierarchy_contracts():
    assert EvaluationLevel.LEVEL_1_IMAGE_SPACE_CONSISTENCY.value == "LEVEL_1_IMAGE_SPACE_CONSISTENCY"
    assert EvaluationLevel.LEVEL_2_RELATIVE_SCALE_ALIGNED_3D.value == "LEVEL_2_RELATIVE_SCALE_ALIGNED_3D"
    assert EvaluationLevel.LEVEL_3_ABSOLUTE_METRIC_GEOSPATIAL.value == "LEVEL_3_ABSOLUTE_METRIC_GEOSPATIAL"


# 14. Monocular Scale Ambiguity Flag
def test_monocular_scale_ambiguity_flag():
    recon = SparseReconstructionResult(
        camera_poses={},
        intrinsics={},
        points3d={},
        mean_reprojection_rmse_px=0.5,
        percentile_90_reprojection_error_px=0.8,
        total_registered_cameras=2,
        total_triangulated_points=100,
        mean_track_length=2.5,
        is_metric_scale=False,
        has_monocular_scale_ambiguity=True,
    )
    d = recon.to_dict()
    assert d["has_monocular_scale_ambiguity"] is True
    assert d["is_metric_scale"] is False


# 15. Surface Completeness Calculation (Reference Point Sampling Semantics)
def test_surface_completeness_calculation():
    # Reference surface: 4 points on a grid [ (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0) ]
    ref_surface = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])

    # Reconstructed cloud has 100 noisy points clustered ONLY near (0, 0, 0)
    # High point count, but low reference point completeness (1 out of 4 reference regions = 25%)
    clustered_rec = np.random.normal(loc=0.0, scale=0.01, size=(100, 3))

    comp_clustered = GeometryMathContracts.compute_reference_point_completeness(
        clustered_rec, ref_surface, distance_tolerance=0.05
    )
    assert comp_clustered == 0.25

    # Well-distributed cloud with only 4 points covering all reference locations (100% completeness)
    distributed_rec = ref_surface + 0.01

    comp_distributed = GeometryMathContracts.compute_reference_point_completeness(
        distributed_rec, ref_surface, distance_tolerance=0.05
    )
    assert comp_distributed == 1.0


# 16. Sparse SfM Distinct from Dense MVS Status
def test_sparse_sfm_distinct_from_dense_mvs_status():
    # Scenario: Sparse SfM succeeded, but Dense Depth estimation failed
    mvs_out = DenseMVSOutput(
        sparse_sfm_status=PipelineStageStatus.SUCCESS,
        camera_registration_status=PipelineStageStatus.SUCCESS,
        dense_depth_status=PipelineStageStatus.FAILED,
        point_cloud_fusion_status=PipelineStageStatus.EMPTY,
        depth_maps={},
        confidence_maps={},
        fused_point_count=0,
        failure_reason=GeometryFailureReason.MVS_DEPTH_ESTIMATION_FAILED,
    )

    d = mvs_out.to_dict()
    assert d["sparse_sfm_status"] == "SUCCESS"
    assert d["dense_depth_status"] == "FAILED"
    assert d["failure_reason"] == "MVS_DEPTH_ESTIMATION_FAILED"


# 17. Full Measurement Provenance Taxonomy
def test_full_measurement_provenance_taxonomy():
    assert MeasurementType.DIRECTLY_OBSERVED.value == "DIRECTLY_OBSERVED"
    assert MeasurementType.ESTIMATED.value == "ESTIMATED"
    assert MeasurementType.TRAJECTORY_PROXY.value == "TRAJECTORY_PROXY"
    assert MeasurementType.HEURISTIC.value == "HEURISTIC"
    assert MeasurementType.GROUND_TRUTH_DEPENDENT.value == "GROUND_TRUTH_DEPENDENT"


# 18. Ground-Truth Trajectory Provenance Requirement for ATE Evaluation
def test_ground_truth_trajectory_provenance():
    # Sensor telemetry is a proxy, NOT ground truth
    telemetry_prov = TrajectoryEvaluationProvenance(
        reference_trajectory_source="ONBOARD_DJI_TELEMETRY",
        is_ground_truth_certified=False,
        measurement_type=MeasurementType.TRAJECTORY_PROXY,
    )
    assert telemetry_prov.is_ground_truth_certified is False
    assert telemetry_prov.measurement_type == MeasurementType.TRAJECTORY_PROXY

    # Certified survey trajectory is ground truth
    gt_prov = TrajectoryEvaluationProvenance(
        reference_trajectory_source="TOTAL_STATION_SURVEY_RTK",
        is_ground_truth_certified=True,
        measurement_type=MeasurementType.GROUND_TRUTH_DEPENDENT,
    )
    assert gt_prov.is_ground_truth_certified is True
    assert gt_prov.measurement_type == MeasurementType.GROUND_TRUTH_DEPENDENT


# 19. Fundamental Matrix (Pixel Space) vs Essential Matrix (Normalized Space) Calibration Requirement
def test_fundamental_vs_essential_calibration_requirement():
    # Pixel raster epipolar constraint: x_2_px^T * F * x_1_px = 0
    F = np.array([
        [0.0,  0.0,  0.0],
        [0.0,  0.0, -0.001],
        [0.0,  0.001, 0.0],
    ])
    p1_px = (500.0, 500.0)
    p2_px = (400.0, 500.0)
    err_f = GeometryMathContracts.verify_fundamental_matrix_constraint(p1_px, p2_px, F)
    assert math.isclose(err_f, 0.0, abs_tol=1e-6)

    # If camera calibration is unavailable, Essential Matrix cannot be estimated
    uncalibrated_result = TwoViewGeometryResult(
        frame_a_id="f_0",
        frame_b_id="f_1",
        fundamental_matrix=F,
        essential_matrix=None,
        has_calibrated_intrinsics=False,
        failure_reason=GeometryFailureReason.CALIBRATION_UNAVAILABLE,
        diagnostics=["Intrinsics unavailable; cannot estimate Essential Matrix E = [t]_x R."],
    )
    d = uncalibrated_result.to_dict()
    assert d["has_calibrated_intrinsics"] is False
    assert d["has_essential_matrix"] is False
    assert d["has_fundamental_matrix"] is True
    assert d["failure_reason"] == "CALIBRATION_UNAVAILABLE"


# 20. Lens Distortion Model Semantics
def test_lens_distortion_model_semantics():
    # 1. Calibrated with explicit Brown-Conrady model
    intrinsics_dist = CameraIntrinsics(
        fx=1200.0, fy=1200.0, cx=640.0, cy=360.0, width=1280, height=720,
        k1=-0.15, k2=0.02, p1=0.001, p2=0.0,
        distortion_model=DistortionModel.BROWN_CONRADY_RADIAL_TANGENTIAL,
        distortion_status=DistortionStatus.EXPLICIT_MODEL_PRESENT,
    )
    assert intrinsics_dist.is_calibrated is True
    assert intrinsics_dist.distortion_model == DistortionModel.BROWN_CONRADY_RADIAL_TANGENTIAL

    # 2. Calibrated and explicitly rectified (zero distortion)
    intrinsics_rect = CameraIntrinsics(
        fx=1200.0, fy=1200.0, cx=640.0, cy=360.0, width=1280, height=720,
        distortion_model=DistortionModel.NONE_RECTIFIED,
        distortion_status=DistortionStatus.RECTIFIED_ZERO_DISTORTION,
    )
    assert intrinsics_rect.is_calibrated is True

    # 3. Calibration unavailable
    intrinsics_uncal = CameraIntrinsics(
        fx=0.0, fy=0.0, cx=0.0, cy=0.0, width=1280, height=720,
        distortion_status=DistortionStatus.CALIBRATION_UNAVAILABLE,
    )
    assert intrinsics_uncal.is_calibrated is False

    # 4. Unsupported distortion model
    intrinsics_unknown = CameraIntrinsics(
        fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000,
        distortion_model=DistortionModel.UNSUPPORTED_UNKNOWN,
        distortion_status=DistortionStatus.UNSUPPORTED_MODEL,
    )
    assert intrinsics_unknown.is_calibrated is False


# 21. Bundle Adjustment Gauge Fixing Policy Contracts
def test_bundle_adjustment_gauge_fixing_policy():
    recon_unit = SparseReconstructionResult(
        camera_poses={},
        intrinsics={},
        points3d={},
        mean_reprojection_rmse_px=0.4,
        percentile_90_reprojection_error_px=0.7,
        total_registered_cameras=5,
        total_triangulated_points=500,
        mean_track_length=3.2,
        gauge_policy=GaugeFixingPolicy.FIX_FIRST_CAMERA_AND_UNIT_BASELINE,
    )
    assert recon_unit.gauge_policy == GaugeFixingPolicy.FIX_FIRST_CAMERA_AND_UNIT_BASELINE
    assert recon_unit.is_metric_scale is False  # Gauge fixing fixes relative gauge, NOT physical metric scale
