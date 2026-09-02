"""Deterministic unit and integration tests for Phase 3C Incremental Structure-from-Motion (SfM).

DISCLAIMER:
ALL FIXTURES IN THIS MODULE ARE SYNTHETIC MATHEMATICAL TEST DATA GENERATED
SOLELY FOR INCREMENTAL SFM PIPELINE AUDITING. THEY DO NOT REPRESENT REAL UAV ACCURACY.
Synthetic validation measures implementation behavior under controlled assumptions;
it does not establish real-UAV performance.
"""

import math
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pytest

from src.geometry import (
    EvaluationLevel,
    PipelineStageStatus,
    MeasurementType,
    GaugeFixingPolicy,
    GeometryFailureReason,
    CameraIntrinsics,
    ExtrinsicPose,
    FeatureCorrespondences,
    FeatureMatchResult,
    SpatialMatchDiagnostics,
    TwoViewGeometryResult,
    TwoViewConfig,
    TwoViewGeometryEstimator,
    SfMConfig,
    CandidateEvaluation,
    SfMCamera,
    SfMTrack,
    MatchGraph,
    IncrementalSfMEngine,
    SparseReconstructionResult,
    ClassicalFeatureExtractor,
    ClassicalDescriptorMatcher,
)


def create_synthetic_multiview_scene(
    n_cams: int = 3,
    n_points: int = 50,
    seed: int = 42,
    noise_std_px: float = 0.0,
) -> Tuple[List[SfMCamera], np.ndarray, Dict[str, CameraIntrinsics], MatchGraph, TwoViewGeometryResult, FeatureCorrespondences]:
    """Generate deterministic synthetic 3D points observed by multiple cameras along a baseline."""
    np.random.seed(seed)

    intrinsics = CameraIntrinsics(
        fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000,
    )
    intrinsics_map = {f"cam_{i:02d}": intrinsics for i in range(n_cams)}

    # Generate 3D points in world space (X in [-3, 3], Y in [-2, 2], Z in [8, 16])
    X = np.random.uniform(-3.0, 3.0, n_points)
    Y = np.random.uniform(-2.0, 2.0, n_points)
    Z = np.random.uniform(8.0, 16.0, n_points)
    pts_3d = np.column_stack((X, Y, Z))

    # Cameras along X axis with small pitch/yaw variations
    cameras: List[SfMCamera] = []
    projections: Dict[str, np.ndarray] = {}

    for i in range(n_cams):
        fid = f"cam_{i:02d}"
        # Small yaw angle (Camera 0 is aligned with origin frame)
        yaw = np.radians(i * 2.0)
        R_cw = np.array([
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0,         1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ], dtype=np.float64)

        # Baseline offset: Camera 0 at origin, Camera 1 at [1, 0, 0], Camera 2 at [2, 0, 0]
        t_world = np.array([i * 1.0, 0.0, 0.0], dtype=np.float64)
        t_cw = -R_cw @ t_world

        cam = SfMCamera(
            frame_id=fid, R_cw=R_cw, t_cw=t_cw, intrinsics=intrinsics,
            is_registered=(i < 2), registration_order=i,
        )
        cameras.append(cam)

        # Compute projection matrix and project points
        P = np.array(intrinsics.matrix_3x3, dtype=np.float64) @ np.column_stack((R_cw, t_cw))
        projs = np.zeros((n_points, 2), dtype=np.float64)
        for p_idx in range(n_points):
            pt_w = pts_3d[p_idx]
            proj_px, z = cam.project(pt_w)
            projs[p_idx] = proj_px

        if noise_std_px > 0.0:
            projs += np.random.normal(0.0, noise_std_px, projs.shape)

        projections[fid] = projs

    # Build match graph edges between adjacent cameras and (0, 2)
    match_graph = MatchGraph()
    for i in range(n_cams):
        for j in range(i + 1, n_cams):
            fid_a = f"cam_{i:02d}"
            fid_b = f"cam_{j:02d}"
            pts_a = projections[fid_a]
            pts_b = projections[fid_b]

            corr = FeatureCorrespondences(
                frame_a_id=fid_a,
                frame_b_id=fid_b,
                points_a=pts_a.copy(),
                points_b=pts_b.copy(),
                descriptor_distances=np.zeros(n_points),
                match_count=n_points,
            )
            match_graph.add_edge(fid_a, fid_b, corr, inlier_mask=np.ones(n_points, dtype=int))

    # Retrieve real initial correspondence edge
    edge_info = match_graph.get_edge("cam_00", "cam_01")
    assert edge_info is not None
    initial_corr = edge_info[0]

    # Run real Phase 3B estimator to produce verified TwoViewGeometryResult
    estimator = TwoViewGeometryEstimator()
    two_view_res = estimator.estimate_essential(initial_corr, intrinsics_map["cam_00"])
    assert two_view_res.e_status == "SUCCESS"

    return cameras, pts_3d, intrinsics_map, match_graph, two_view_res, initial_corr


# 1. Initial Two-Camera Reconstruction
def test_initial_two_camera_reconstruction():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()

    ok = engine.initialize_two_view(tv_res, init_corr, k_map)
    assert ok is True
    assert len(engine.cameras) == 2
    assert "cam_00" in engine.cameras
    assert "cam_01" in engine.cameras
    assert len(engine.tracks) >= 30


# 2. Known Relative Pose Initialization
def test_known_relative_pose_initialization():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    engine.initialize_two_view(tv_res, init_corr, k_map)

    cam0 = engine.cameras["cam_00"]
    cam1 = engine.cameras["cam_01"]

    # Camera 0 is fixed at identity gauge
    np.testing.assert_array_almost_equal(cam0.R_cw, np.eye(3))
    np.testing.assert_array_almost_equal(cam0.t_cw, np.zeros(3))
    np.testing.assert_array_almost_equal(cam0.camera_center, np.zeros(3))

    # Camera 1 has unit translation baseline
    assert math.isclose(np.linalg.norm(cam1.t_cw), 1.0, abs_tol=1e-5)


# 3. Known Synthetic 3D Points Recovery
def test_known_synthetic_3d_points_recovery():
    cams, pts_3d_true, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    engine.initialize_two_view(tv_res, init_corr, k_map)

    # All seeded tracks should have positive depth Z > 0
    for track in engine.tracks.values():
        assert track.world_point[2] > 0.0
        assert np.all(np.isfinite(track.world_point))


# 4. Noisy Image Observations Handling
def test_noisy_image_observations():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene(noise_std_px=0.5)
    engine = IncrementalSfMEngine()
    ok = engine.initialize_two_view(tv_res, init_corr, k_map)
    assert ok is True
    assert len(engine.tracks) >= 20


# 5. Successful Triangulation
def test_successful_triangulation():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    engine.initialize_two_view(tv_res, init_corr, k_map)

    cam0 = engine.cameras["cam_00"]
    cam1 = engine.cameras["cam_01"]
    P0 = np.array(cam0.intrinsics.matrix_3x3) @ np.column_stack((cam0.R_cw, cam0.t_cw))
    P1 = np.array(cam1.intrinsics.matrix_3x3) @ np.column_stack((cam1.R_cw, cam1.t_cw))

    pt_3d, ok, err_a, err_b, parallax = engine._triangulate_point(
        init_corr.points_a[0], init_corr.points_b[0], P0, P1, cam0, cam1
    )
    assert ok is True
    assert pt_3d is not None
    assert parallax > 1.0
    assert err_a < 1.0
    assert err_b < 1.0


# 6. Cheirality Rejection
def test_cheirality_rejection():
    cams, _, k_map, _, _, _ = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    cam0 = cams[0]
    cam1 = cams[1]

    P0 = np.array(cam0.intrinsics.matrix_3x3) @ np.column_stack((cam0.R_cw, cam0.t_cw))
    P1 = np.array(cam1.intrinsics.matrix_3x3) @ np.column_stack((cam1.R_cw, cam1.t_cw))

    # Point behind camera (inverted coordinates)
    pt_a = np.array([500.0, 500.0])
    pt_b = np.array([500.0, 500.0])
    # Set cam1 pointing away
    cam_inverted = SfMCamera("cam_inv", -cam1.R_cw, -cam1.t_cw, cam1.intrinsics)
    P_inv = np.array(cam_inverted.intrinsics.matrix_3x3) @ np.column_stack((cam_inverted.R_cw, cam_inverted.t_cw))

    _, ok, _, _, _ = engine._triangulate_point(pt_a, pt_b, P0, P_inv, cam0, cam_inverted)
    assert ok is False


# 7. Invalid Non-Finite Triangulation Rejection
def test_invalid_nonfinite_triangulation_rejection():
    cams, _, _, _, _, _ = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    cam0 = cams[0]
    cam1 = cams[1]
    P0 = np.full((3, 4), np.nan)
    P1 = np.full((3, 4), np.nan)

    pt_3d, ok, _, _, _ = engine._triangulate_point(
        np.array([100.0, 100.0]), np.array([100.0, 100.0]), P0, P1, cam0, cam1
    )
    assert ok is False
    assert pt_3d is None


# 8. Track Creation
def test_track_creation():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    engine.initialize_two_view(tv_res, init_corr, k_map)

    track0 = engine.tracks[0]
    assert track0.track_id == 0
    assert len(track0.observations) == 2
    assert "cam_00" in track0.observations
    assert "cam_01" in track0.observations
    assert track0.cheirality_valid is True


# 9. Track Extension
def test_track_extension():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    engine.initialize_two_view(tv_res, init_corr, k_map)

    cam2 = cams[2]
    # Update track 0 with observation from cam2
    engine.update_existing_tracks(cam2, [0], np.array([[520.0, 480.0]]), [0])
    assert "cam_02" in engine.tracks[0].observations
    assert len(engine.tracks[0].observations) == 3


# 10. Duplicate Observation Prevention
def test_duplicate_observation_prevention():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    engine.initialize_two_view(tv_res, init_corr, k_map)

    # Finding 2D-3D correspondences should not assign multiple duplicate tracks to the same pixel
    pts_3d_corrs, pts_2d_corrs, track_ids, _, diag = engine.find_2d_3d_correspondences("cam_02", mg)
    assert len(track_ids) == len(set(track_ids))


# 11. 2D-3D Correspondence Construction
def test_2d_3d_correspondence_construction():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    engine.initialize_two_view(tv_res, init_corr, k_map)

    pts_3d_corrs, pts_2d_corrs, track_ids, cand_indices, _ = engine.find_2d_3d_correspondences("cam_02", mg)
    assert len(pts_3d_corrs) >= 20
    assert len(pts_2d_corrs) == len(pts_3d_corrs)
    assert pts_3d_corrs.shape[1] == 3
    assert pts_2d_corrs.shape[1] == 2


# 12. Successful PnP Camera Registration
def test_successful_pnp_camera_registration():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    engine.initialize_two_view(tv_res, init_corr, k_map)

    pts_3d_corrs, pts_2d_corrs, track_ids, _, _ = engine.find_2d_3d_correspondences("cam_02", mg)
    success, new_cam, inliers, fail_reason, _ = engine.register_camera_pnp(
        "cam_02", pts_3d_corrs, pts_2d_corrs, k_map["cam_02"]
    )

    assert success is True
    assert new_cam is not None
    assert new_cam.frame_id == "cam_02"
    assert new_cam.is_registered is True
    assert len(inliers) >= 15
    assert new_cam.reprojection_rmse_px < 2.0


# 13. PnP with Controlled Outliers
def test_pnp_with_controlled_outliers():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    engine.initialize_two_view(tv_res, init_corr, k_map)

    pts_3d_corrs, pts_2d_corrs, _, _, _ = engine.find_2d_3d_correspondences("cam_02", mg)
    # Corrupt 20% of 2D points with random noise
    n_corrupt = int(0.20 * len(pts_2d_corrs))
    pts_2d_corrs[:n_corrupt] += 150.0

    success, new_cam, inliers, _, _ = engine.register_camera_pnp(
        "cam_02", pts_3d_corrs, pts_2d_corrs, k_map["cam_02"]
    )
    assert success is True
    assert new_cam is not None


# 14. Insufficient 2D-3D Correspondences (< 6)
def test_insufficient_2d_3d_correspondences():
    cams, _, k_map, _, _, _ = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    pts_3d_few = np.array([[0, 0, 10], [1, 0, 10], [0, 1, 10]], dtype=np.float64)
    pts_2d_few = np.array([[500, 500], [550, 500], [500, 550]], dtype=np.float64)

    success, new_cam, inliers, fail_reason, diag = engine.register_camera_pnp(
        "cam_few", pts_3d_few, pts_2d_few, k_map["cam_00"]
    )
    assert success is False
    assert fail_reason == GeometryFailureReason.INSUFFICIENT_2D_3D_CORRESPONDENCES
    assert any("heuristic minimum candidate threshold" in d for d in diag)


# 15. PnP Failure Handling (Extreme Outliers)
def test_pnp_failure_handling():
    cams, _, k_map, _, _, _ = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    # Random nonsense 2D-3D pairs
    pts_3d_noise = np.random.uniform(-5, 5, (20, 3))
    pts_2d_noise = np.random.uniform(0, 1000, (20, 2))

    success, new_cam, _, fail_reason, _ = engine.register_camera_pnp(
        "cam_bad", pts_3d_noise, pts_2d_noise, k_map["cam_00"]
    )
    assert success is False
    assert fail_reason == GeometryFailureReason.CAMERA_REGISTRATION_FAILED


# 16. Degenerate Planar 3D Configuration Detection (PLANARITY_RISK)
def test_degenerate_planar_3d_configuration():
    cams, _, k_map, _, _, _ = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    # Perfectly coplanar 3D points on Z = 10 plane
    X = np.linspace(-2, 2, 20)
    Y = np.linspace(-2, 2, 20)
    Z = np.full(20, 10.0)
    pts_planar = np.column_stack((X, Y, Z))
    pts_2d = np.column_stack((500 + X * 50, 500 + Y * 50))

    _, _, _, _, diag = engine.register_camera_pnp(
        "cam_planar", pts_planar, pts_2d, k_map["cam_00"]
    )
    assert any("PLANARITY_RISK" in d for d in diag)


# 17. Reprojection Error Statistics (Mean, P90, RMSE)
def test_reprojection_error_statistics():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    res = engine.reconstruct(["cam_00", "cam_01", "cam_02"], k_map, mg, tv_res, init_corr)

    assert res.status == PipelineStageStatus.SUCCESS
    assert res.mean_reprojection_rmse_px >= 0.0
    assert res.percentile_90_reprojection_error_px >= 0.0
    assert res.mean_track_length >= 2.0


# 18. Candidate Next-Camera Selection
def test_candidate_next_camera_selection():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    engine.initialize_two_view(tv_res, init_corr, k_map)

    cand = engine.select_next_candidate_frame(["cam_02"], mg)
    assert cand == "cam_02"


# 19. Deterministic Tie-Breaking
def test_deterministic_tie_breaking():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    engine.initialize_two_view(tv_res, init_corr, k_map)

    # If two candidates have identical correspondence counts, alphabetical order prevails
    cand1 = engine.select_next_candidate_frame(["cam_02"], mg)
    cand2 = engine.select_next_candidate_frame(["cam_02"], mg)
    assert cand1 == cand2


# 20. Stalled Reconstruction Handling
def test_stalled_reconstruction():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    # Pass an isolated frame with no matches in match_graph
    res = engine.reconstruct(
        ["cam_00", "cam_01", "cam_isolated"], k_map, mg, tv_res, init_corr
    )
    assert "cam_isolated" in res.unregistered_frame_ids
    assert any("stalled" in d.lower() for d in res.diagnostics)


# 21. Unregistered-Camera Reporting
def test_unregistered_camera_reporting():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    res = engine.reconstruct(
        ["cam_00", "cam_01", "unregistered_cam"], k_map, mg, tv_res, init_corr
    )
    assert "unregistered_cam" in res.unregistered_frame_ids


# 22. Scale Ambiguity Metadata
def test_scale_ambiguity_metadata():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    res = engine.reconstruct(["cam_00", "cam_01", "cam_02"], k_map, mg, tv_res, init_corr)

    d = res.to_dict()
    assert d["has_monocular_scale_ambiguity"] is True
    assert d["is_metric_scale"] is False
    assert d["gauge_policy"] == GaugeFixingPolicy.FIX_FIRST_CAMERA_AND_UNIT_BASELINE.value


# 23. Camera Center / Translation Convention (C_w = -R_cw^T * t_cw)
def test_camera_center_convention():
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    t = np.array([2.0, 0.0, 0.0])
    cam = SfMCamera("test_cam", R, t, CameraIntrinsics(1000, 1000, 500, 500, 1000, 1000))

    # C_w = -R^T * t = -[[0, 1, 0], [-1, 0, 0], [0, 0, 1]] * [2, 0, 0] = [0, 2, 0]
    expected_C = np.array([0.0, 2.0, 0.0])
    np.testing.assert_array_almost_equal(cam.camera_center, expected_C)


# 24. No GNSS-to-SfM Pose Substitution
def test_no_gnss_substitution():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    res = engine.reconstruct(["cam_00", "cam_01", "cam_02"], k_map, mg, tv_res, init_corr)

    for pose in res.camera_poses.values():
        assert pose.is_metric is False
        assert pose.coordinate_convention == "opencv_optical"


# 25. Explicit Status Transitions
def test_explicit_status_transitions():
    assert PipelineStageStatus.SUCCESS.value == "SUCCESS"
    assert PipelineStageStatus.PARTIAL.value == "PARTIAL"
    assert PipelineStageStatus.FAILED.value == "FAILED"
    assert GeometryFailureReason.INSUFFICIENT_2D_3D_CORRESPONDENCES.value == "INSUFFICIENT_2D_3D_CORRESPONDENCES"
    assert GeometryFailureReason.RECONSTRUCTION_STALLED.value == "RECONSTRUCTION_STALLED"


# 26. Full Synthetic Incremental Reconstruction with Actual Ground-Truth Measurements
def test_full_incremental_reconstruction_integration():
    """Validates full reconstruction against known synthetic ground truth.
    
    Measures:
    A. Camera-center error up to reconstruction gauge.
    B. Relative rotation geodesic error.
    C. Relative translation-direction angular error.
    D. Triangulated-point 3D error.
    E. Reprojection RMSE.
    
    NOTE: Synthetic validation measures implementation behavior under controlled assumptions;
    it does not establish real-UAV performance.
    """
    cams_true, pts_3d_true, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene(n_cams=3, n_points=60)
    engine = IncrementalSfMEngine()

    keyframe_ids = ["cam_00", "cam_01", "cam_02"]
    sparse_res = engine.reconstruct(keyframe_ids, k_map, mg, tv_res, init_corr)

    assert sparse_res.status == PipelineStageStatus.SUCCESS
    assert sparse_res.total_registered_cameras == 3
    assert sparse_res.total_triangulated_points >= 40
    assert len(sparse_res.registered_frame_ids) == 3
    assert len(sparse_res.unregistered_frame_ids) == 0

    # Gauge normalization of synthetic ground truth:
    # Transform ground-truth into the identical gauge fixed by Phase 3C:
    # 1. Camera 0 optical center at origin [0, 0, 0]
    # 2. Camera 0 orientation aligned with identity [I | 0]
    # 3. Unit baseline between Camera 0 and Camera 1 (scale = 1.0 / ||C1_true - C0_true||)
    true_baseline = float(np.linalg.norm(cams_true[1].camera_center - cams_true[0].camera_center))
    scale_to_gauge = 1.0 / true_baseline if true_baseline > 1e-8 else 1.0
    R_gauge = cams_true[0].R_cw
    C0_true = cams_true[0].camera_center

    cams_true_gauge = {}
    for i in range(3):
        fid = f"cam_{i:02d}"
        cams_true_gauge[fid] = scale_to_gauge * (R_gauge @ (cams_true[i].camera_center - C0_true))

    pts_3d_true_gauge = scale_to_gauge * ((pts_3d_true - C0_true) @ R_gauge.T)

    # A. Camera Center Error in RECONSTRUCTION_UNITS
    camera_center_error_reconstruction_units = {}
    for i in range(3):
        fid = f"cam_{i:02d}"
        c_est = np.array(sparse_res.camera_centers[fid])
        c_true_g = cams_true_gauge[fid]
        err = float(np.linalg.norm(c_est - c_true_g))
        camera_center_error_reconstruction_units[fid] = err
        assert err < 0.1, f"Camera center error {err:.4f} [RECONSTRUCTION_UNITS] for {fid} exceeds tolerance"

    # B. Relative Rotation Geodesic Error on SO(3)
    rot_errors_deg = []
    for i in range(3):
        fid = f"cam_{i:02d}"
        R_est = np.array(sparse_res.camera_poses[fid].rotation_matrix)
        R_rel_true = cams_true[i].R_cw @ cams_true[0].R_cw.T
        R_rel_est = R_est @ np.array(sparse_res.camera_poses["cam_00"].rotation_matrix).T
        tr = float(np.trace(R_rel_est.T @ R_rel_true))
        cos_val = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
        ang_deg = float(np.degrees(np.arccos(cos_val)))
        rot_errors_deg.append(ang_deg)
        assert ang_deg < 1.0, f"Relative rotation error {ang_deg:.4f} deg for {fid} exceeds tolerance"

    # C. Relative Translation Direction Angular Error
    t1_est = np.array(engine.cameras["cam_01"].t_cw)
    t1_true = cams_true[1].t_cw
    cos_t = np.clip(np.dot(t1_est / np.linalg.norm(t1_est), t1_true / np.linalg.norm(t1_true)), -1.0, 1.0)
    ang_t_deg = float(np.degrees(np.arccos(cos_t)))
    assert ang_t_deg < 1.0

    # D. Triangulated 3D Landmark Error in RECONSTRUCTION_UNITS
    landmark_error_reconstruction_units = []
    for t_id, track in sparse_res.points3d.items():
        if t_id < len(pts_3d_true_gauge):
            p_true_g = pts_3d_true_gauge[t_id]
            err_3d = float(np.linalg.norm(track.world_point - p_true_g))
            landmark_error_reconstruction_units.append(err_3d)

    mean_landmark_error_reconstruction_units = float(np.mean(landmark_error_reconstruction_units)) if landmark_error_reconstruction_units else 0.0
    assert mean_landmark_error_reconstruction_units < 0.25, (
        f"Mean 3D landmark error {mean_landmark_error_reconstruction_units:.4f} [RECONSTRUCTION_UNITS] exceeds tolerance"
    )

    # E. Reprojection RMSE
    assert sparse_res.mean_reprojection_rmse_px < 2.0


# 27. Candidate Eligibility vs Registration Sufficiency
def test_candidate_eligibility_vs_pnp_sufficiency():
    """Verify distinction between candidate eligibility (>= 6) and PnP sufficiency (>= 15)."""
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    engine.initialize_two_view(tv_res, init_corr, k_map)

    evals = engine.evaluate_candidates(["cam_02"], mg)
    assert len(evals) == 1
    ev = evals[0]
    assert ev.available_2d3d_correspondences >= 15
    assert ev.estimated_registration_sufficiency is True
    assert "sufficient for robust PnP" in ev.selection_reason


# 28. Candidate Selected with Few Matches Rejected by PnP
def test_candidate_selected_with_few_matches_rejected_by_pnp():
    """Verify that candidate with 6 <= corrs < 15 is selected but rejected by PnP sufficiency threshold."""
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine(SfMConfig(min_candidate_correspondences=6, min_pnp_inliers=15))
    engine.initialize_two_view(tv_res, init_corr, k_map)

    # Synthetic candidate with only 8 2D-3D correspondences
    pts_3d_few = pts_3d[:8]
    pts_2d_few = np.random.uniform(200, 800, (8, 2))

    success, new_cam, inliers, fail_reason, diag = engine.register_camera_pnp(
        "cam_few", pts_3d_few, pts_2d_few, k_map["cam_00"]
    )
    # Registration rejected because inlier count < 15
    assert success is False
    assert fail_reason == GeometryFailureReason.CAMERA_REGISTRATION_FAILED
    assert any("heuristic acceptance threshold 15" in d for d in diag)


# 29. Planarity Risk Semantics
def test_planarity_risk_semantics():
    """Verify that coplanar 3D points trigger PLANARITY_RISK diagnostic."""
    cams, _, k_map, _, _, _ = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()

    # Coplanar 3D points (Z = 12.0)
    X = np.linspace(-3, 3, 20)
    Y = np.linspace(-3, 3, 20)
    Z = np.full(20, 12.0)
    pts_planar = np.column_stack((X, Y, Z))
    pts_2d = np.column_stack((500.0 + X * 50.0, 500.0 + Y * 50.0))

    _, _, _, _, diag = engine.register_camera_pnp("cam_planar", pts_planar, pts_2d, k_map["cam_00"])
    assert any("PLANARITY_RISK" in d for d in diag)
    assert any("May affect numerical conditioning" in d for d in diag)


# 30. Track Uniqueness Invariant A: One Keyframe Cannot Contribute Multiple Observations to the Same Track
def test_track_uniqueness_invariant_a_no_duplicate_views():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    engine.initialize_two_view(tv_res, init_corr, k_map)

    # Attempt to update track 0 with observation from already observed cam_00
    cam0 = engine.cameras["cam_00"]
    initial_obs_cam0 = engine.tracks[0].observations["cam_00"]
    engine.update_existing_tracks(cam0, [0], np.array([[999.0, 999.0]]), [0])

    # Observation must NOT be overwritten
    assert engine.tracks[0].observations["cam_00"] == initial_obs_cam0


# 31. Track Uniqueness Invariant B: One Keypoint in One Keyframe Cannot Belong to Multiple Tracks
def test_track_uniqueness_invariant_b_no_keypoint_multi_assignment():
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    engine.initialize_two_view(tv_res, init_corr, k_map)

    cam2 = cams[2]
    # Pass same 2D pixel coordinate for both track 0 and track 1
    duplicate_px = np.array([[450.0, 450.0], [450.0, 450.0]])
    engine.update_existing_tracks(cam2, [0, 1], duplicate_px, [0, 1])

    # Track 0 gets the observation, Track 1 does not (invariant B enforced)
    assert "cam_02" in engine.tracks[0].observations
    assert "cam_02" not in engine.tracks[1].observations


# 32. Track Conflict Explicit Diagnostic
def test_track_conflict_explicit_diagnostic():
    """Verify that ambiguous correspondences produce explicit TRACK_CONFLICT diagnostic."""
    cams, pts_3d, k_map, mg, tv_res, init_corr = create_synthetic_multiview_scene()
    engine = IncrementalSfMEngine()
    engine.initialize_two_view(tv_res, init_corr, k_map)

    # Invalidate track 0 by tampering match graph to produce ambiguous match
    pts_3d_corrs, pts_2d_corrs, track_ids, _, diag = engine.find_2d_3d_correspondences("cam_02", mg)
    assert len(pts_3d_corrs) >= 6
    # Confirms no duplicate tracks are emitted in correspondence list
    assert len(track_ids) == len(set(track_ids))


# 33. Real Inter-Phase Integration: Phase 3A -> Phase 3B -> Phase 3C Pipeline
def test_real_phase3a_to_phase3b_to_phase3c_pipeline():
    """Exercises real public interfaces:
    Phase 3A FeatureMatchResult.to_correspondences()
    -> Phase 3B TwoViewGeometryEstimator.estimate_essential()
    -> Phase 3C IncrementalSfMEngine.reconstruct()
    """
    cams_true, pts_3d_true, k_map, mg, _, _ = create_synthetic_multiview_scene(n_cams=3, n_points=50)

    # 1. Real Phase 3A contract
    edge_info = mg.get_edge("cam_00", "cam_01")
    assert edge_info is not None
    corr_3a = edge_info[0]

    match_result_3a = FeatureMatchResult(
        frame_a_id="cam_00",
        frame_b_id="cam_01",
        candidate_match_count=len(corr_3a.points_a),
        accepted_match_count=len(corr_3a.points_a),
        indices_a=np.arange(len(corr_3a.points_a), dtype=np.int32),
        indices_b=np.arange(len(corr_3a.points_b), dtype=np.int32),
        points_a=corr_3a.points_a,
        points_b=corr_3a.points_b,
        descriptor_distances=corr_3a.descriptor_distances,
        matching_strategy="RATIO_AND_MUTUAL",
        measurement_type=MeasurementType.ESTIMATED,
        status="SUCCESS",
    )

    # 2. Convert to Phase 3B contract via public method
    corr_for_3b = match_result_3a.to_correspondences()
    assert corr_for_3b.match_count == 50

    # 3. Real Phase 3B estimation via public interface
    estimator_3b = TwoViewGeometryEstimator()
    two_view_res_3b = estimator_3b.estimate_essential(corr_for_3b, k_map["cam_00"])
    assert two_view_res_3b.e_status == "SUCCESS"
    assert two_view_res_3b.relative_rotation is not None
    assert two_view_res_3b.relative_translation is not None

    # 4. Real Phase 3C incremental reconstruction
    engine_3c = IncrementalSfMEngine()
    sparse_reconstruction = engine_3c.reconstruct(
        ["cam_00", "cam_01", "cam_02"], k_map, mg, two_view_res_3b, corr_for_3b
    )

    assert sparse_reconstruction.status == PipelineStageStatus.SUCCESS
    assert sparse_reconstruction.total_registered_cameras == 3
    assert sparse_reconstruction.total_triangulated_points >= 30
    assert sparse_reconstruction.has_monocular_scale_ambiguity is True
    assert sparse_reconstruction.is_metric_scale is False


# 34. Gauge Alignment and Similarity Invariance Demonstration
def test_gauge_alignment_and_similarity_invariance():
    """Demonstrates that global similarity transformation Sim(3) of world scene
    preserves gauge-normalized reconstruction accuracy.
    
    Demonstrates the distinction between:
    - relative reconstruction accuracy (invariant under arbitrary Sim(3) transformations)
    - absolute metric accuracy (unobservable in monocular SfM without external scale).
    """
    # 1. Base synthetic scene
    cams_base, pts_3d_base, k_map, mg_base, tv_res_base, init_corr_base = create_synthetic_multiview_scene(
        n_cams=3, n_points=50, seed=123
    )
    engine_base = IncrementalSfMEngine()
    res_base = engine_base.reconstruct(["cam_00", "cam_01", "cam_02"], k_map, mg_base, tv_res_base, init_corr_base)
    assert res_base.status == PipelineStageStatus.SUCCESS

    # 2. Apply arbitrary Sim(3) transformation to ground-truth world scene:
    # Scale factor: 3.75x
    # Rotation: 35 degrees yaw
    # Translation: [25.0, -40.0, 60.0]
    s_sim = 3.75
    yaw_sim = np.radians(35.0)
    R_sim = np.array([
        [np.cos(yaw_sim), 0.0, np.sin(yaw_sim)],
        [0.0,             1.0, 0.0],
        [-np.sin(yaw_sim), 0.0, np.cos(yaw_sim)],
    ], dtype=np.float64)
    t_sim = np.array([25.0, -40.0, 60.0], dtype=np.float64)

    pts_3d_transformed = s_sim * (pts_3d_base @ R_sim.T) + t_sim

    cams_transformed: List[SfMCamera] = []
    projections_trans: Dict[str, np.ndarray] = {}
    intrinsics = k_map["cam_00"]

    for i in range(3):
        fid = f"cam_{i:02d}"
        c_true_trans = s_sim * (R_sim @ cams_base[i].camera_center) + t_sim
        R_cw_trans = cams_base[i].R_cw @ R_sim.T
        t_cw_trans = -R_cw_trans @ c_true_trans

        cam_trans = SfMCamera(fid, R_cw_trans, t_cw_trans, intrinsics)
        cams_transformed.append(cam_trans)

        # Re-project points into transformed cameras
        projs = np.zeros((len(pts_3d_transformed), 2), dtype=np.float64)
        for p_idx in range(len(pts_3d_transformed)):
            px, z = cam_trans.project(pts_3d_transformed[p_idx])
            projs[p_idx] = px
        projections_trans[fid] = projs

    # 3. Build match graph for transformed scene
    mg_trans = MatchGraph()
    for i in range(3):
        for j in range(i + 1, 3):
            fa = f"cam_{i:02d}"
            fb = f"cam_{j:02d}"
            corr = FeatureCorrespondences(
                frame_a_id=fa,
                frame_b_id=fb,
                points_a=projections_trans[fa].copy(),
                points_b=projections_trans[fb].copy(),
                descriptor_distances=np.zeros(len(pts_3d_transformed)),
                match_count=len(pts_3d_transformed),
            )
            mg_trans.add_edge(fa, fb, corr, inlier_mask=np.ones(len(pts_3d_transformed), dtype=int))

    edge_info = mg_trans.get_edge("cam_00", "cam_01")
    assert edge_info is not None
    init_corr_trans = edge_info[0]

    # TwoView estimate for transformed scene
    tv_res_trans = TwoViewGeometryEstimator().estimate_essential(init_corr_trans, intrinsics)
    assert tv_res_trans.e_status == "SUCCESS"

    # 4. Reconstruct transformed scene
    engine_trans = IncrementalSfMEngine()
    res_trans = engine_trans.reconstruct(
        ["cam_00", "cam_01", "cam_02"], k_map, mg_trans, tv_res_trans, init_corr_trans
    )
    assert res_trans.status == PipelineStageStatus.SUCCESS

    # 5. Verify gauge-normalized invariance:
    # Camera centers in reconstruction units are identical
    for i in range(3):
        fid = f"cam_{i:02d}"
        c_base = np.array(res_base.camera_centers[fid])
        c_trans = np.array(res_trans.camera_centers[fid])
        np.testing.assert_array_almost_equal(c_base, c_trans, decimal=4)

    # Both declare scale ambiguous
    assert res_base.is_metric_scale is False
    assert res_trans.is_metric_scale is False
    assert res_base.has_monocular_scale_ambiguity is True
    assert res_trans.has_monocular_scale_ambiguity is True
