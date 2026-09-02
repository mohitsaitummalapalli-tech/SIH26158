"""Tests for Phase 3E.0: Dense Multi-View Stereo (MVS) Architecture & Contracts.

Verifies:
A. Known camera + known depth -> exact backprojection
B. Multiple views -> consistent 3D point
C. Depth inconsistency -> rejected
D. Invalid depth -> rejected
E. Occlusion -> explicitly represented
F. Multiple source views -> support count preserved
G. Duplicate observations -> deterministic fusion behavior
H. Camera coordinate convention -> round-trip projection/backprojection
I. Camera 0 / Camera 1 gauge compatibility -> no accidental scale normalization
J. Reconstruction-unit preservation -> no meters claim
K. Dynamic-risk propagation -> metadata survives into MVS contracts
"""

from typing import List, Tuple
import numpy as np
import pytest

from src.geometry.contracts import (
    CameraIntrinsics,
    ExtrinsicPose,
)

from src.geometry.mvs import (
    MVSFailureReason,
    PointVisibilityState,
    PointValidationStatus,
    DepthUnit,
    MVSConfig,
    MVSInput,
    MVSViewGraph,
    DepthMap,
    DepthConfidenceMap,
    DensePointObservation,
    DensePointCloud,
    MVSGeometryMath,
    depth_to_world_points,
    HeuristicViewPairSelector,
    GeometricDepthConsistencyChecker,
    VoxelGridDensePointFusion,
    MVSValidator,
    IMVSDepthEstimator,
)


@pytest.fixture
def standard_intrinsics() -> CameraIntrinsics:
    """Calibrated pinhole intrinsics: 1920x1080, focal length 1200.0."""
    return CameraIntrinsics(
        fx=1200.0,
        fy=1200.0,
        cx=960.0,
        cy=540.0,
        width=1920,
        height=1080,
    )


@pytest.fixture
def identity_pose() -> ExtrinsicPose:
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
def baseline_pose() -> ExtrinsicPose:
    """Camera 1 translated by 1.0 unit along X axis."""
    return ExtrinsicPose(
        rotation_matrix=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        translation_vector=[1.0, 0.0, 0.0],
    )


class TestMVSContracts:
    """Synthetic mathematical verification of Phase 3E.0 MVS contracts."""

    def test_camera_convention_projection_backprojection_roundtrip(
        self, standard_intrinsics: CameraIntrinsics, baseline_pose: ExtrinsicPose
    ):
        """Verify contract H: X_c = R_cw X_w + t_cw roundtrip with depth backprojection."""
        R_cw = np.array(baseline_pose.rotation_matrix, dtype=np.float64)
        c_w = np.array(baseline_pose.translation_vector, dtype=np.float64)
        t_cw = -R_cw @ c_w

        # Known 3D world point
        X_w_orig = np.array([1.5, -0.3, 8.0], dtype=np.float64)

        # 1. Project into camera
        px, z_c, ok = MVSGeometryMath.project_world_point(X_w_orig, standard_intrinsics, R_cw, t_cw)
        assert ok is True
        assert px is not None
        assert z_c > 0.0

        u, v = px
        # 2. Backproject with exact optical depth z_c
        X_w_reconstructed, ok_back = MVSGeometryMath.backproject_pixel(u, v, z_c, standard_intrinsics, R_cw, t_cw)
        assert ok_back is True
        assert X_w_reconstructed is not None

        # Check exact roundtrip recovery
        np.testing.assert_allclose(X_w_reconstructed, X_w_orig, atol=1e-8)

    def test_known_camera_known_depth_exact_backprojection(
        self, standard_intrinsics: CameraIntrinsics, identity_pose: ExtrinsicPose
    ):
        """Verify contract A: Known camera + known depth -> exact backprojection."""
        R_cw = np.array(identity_pose.rotation_matrix, dtype=np.float64)
        c_w = np.array(identity_pose.translation_vector, dtype=np.float64)
        t_cw = -R_cw @ c_w

        # Optical center pixel (cx, cy) with depth 10.0 should be at (0, 0, 10.0)
        u, v, depth_z = standard_intrinsics.cx, standard_intrinsics.cy, 10.0
        X_w, ok = MVSGeometryMath.backproject_pixel(u, v, depth_z, standard_intrinsics, R_cw, t_cw)
        assert ok is True
        assert X_w is not None
        np.testing.assert_allclose(X_w, [0.0, 0.0, 10.0], atol=1e-10)

        # Off-center pixel: u = cx + fx = 2160, v = cy = 540 -> x_c = 10.0 * 1.0 = 10.0
        X_w_off, ok_off = MVSGeometryMath.backproject_pixel(standard_intrinsics.cx + standard_intrinsics.fx, standard_intrinsics.cy, 10.0, standard_intrinsics, R_cw, t_cw)
        assert ok_off is True
        assert X_w_off is not None
        np.testing.assert_allclose(X_w_off, [10.0, 0.0, 10.0], atol=1e-10)

    def test_invalid_and_non_positive_depth_rejection(
        self, standard_intrinsics: CameraIntrinsics, identity_pose: ExtrinsicPose
    ):
        """Verify contract D: Non-positive, zero, NaN, and Inf depths are rejected."""
        R_cw = np.array(identity_pose.rotation_matrix, dtype=np.float64)
        t_cw = np.zeros(3, dtype=np.float64)

        for invalid_z in [0.0, -5.0, -1e-8, float("nan"), float("inf"), float("-inf")]:
            X_w, ok = MVSGeometryMath.backproject_pixel(960.0, 540.0, invalid_z, standard_intrinsics, R_cw, t_cw)
            assert ok is False
            assert X_w is None

    def test_depth_map_query_invariants(self):
        """Verify DepthMap contract and continuous depth query."""
        H, W = 100, 150
        depth_data = np.full((H, W), 5.0, dtype=np.float32)
        valid_mask = np.ones((H, W), dtype=bool)
        valid_mask[50, 50] = False  # hole

        dm = DepthMap(
            reference_frame_id="frame_0",
            width=W,
            height=H,
            depth_array=depth_data,
            valid_mask=valid_mask,
            depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
            min_depth=1.0,
            max_depth=20.0,
        )

        # Valid point query
        assert dm.get_depth_at(10.0, 20.0) == pytest.approx(5.0, abs=1e-5)
        # Invalid pixel query
        assert dm.get_depth_at(50.0, 50.0) is None
        # Out of bounds query
        assert dm.get_depth_at(-1.0, 10.0) is None
        assert dm.get_depth_at(200.0, 10.0) is None

    def test_view_pair_selector_multi_source_and_ordering(
        self, standard_intrinsics: CameraIntrinsics
    ):
        """Verify contracts E & F: Deterministic view-pair selection with multi-source ranking."""
        cfg = MVSConfig(max_source_views=2, min_overlap_ratio=0.2)
        poses = {
            "f0": ExtrinsicPose(rotation_matrix=np.eye(3).tolist(), translation_vector=[0.0, 0.0, 0.0]),
            "f1": ExtrinsicPose(rotation_matrix=np.eye(3).tolist(), translation_vector=[1.0, 0.0, 0.0]),
            "f2": ExtrinsicPose(rotation_matrix=np.eye(3).tolist(), translation_vector=[2.0, 0.0, 0.0]),
            "f3": ExtrinsicPose(rotation_matrix=np.eye(3).tolist(), translation_vector=[3.0, 0.0, 0.0]),
        }
        dims = {fid: (1080, 1920) for fid in poses}
        intrinsics = {fid: standard_intrinsics for fid in poses}

        mvs_in = MVSInput(
            selected_frame_ids=["f0", "f1", "f2", "f3"],
            image_dimensions=dims,
            camera_intrinsics=intrinsics,
            camera_poses=poses,
        )

        selector = HeuristicViewPairSelector()
        graph: MVSViewGraph = selector.select_pairs(mvs_in, cfg)

        assert len(graph.frame_nodes) == 4
        assert len(graph.selected_edges) > 0

        # For f0, source views should be selected up to max_source_views (2)
        f0_sources = graph.get_source_views("f0")
        assert len(f0_sources) <= cfg.max_source_views
        # Confirm deterministic ordering (descending score)
        scores = [s.viewpoint_suitability_score for s in f0_sources]
        assert scores == sorted(scores, reverse=True)

    def test_view_pair_selector_rejection_rules(
        self, standard_intrinsics: CameraIntrinsics
    ):
        """Verify candidate rejection for zero baseline and extreme angles."""
        cfg = MVSConfig(max_triangulation_angle_deg=30.0)

        # Pose f1 coincident with f0 -> zero baseline
        # Pose f2 rotated by 60 degrees -> viewing angle too steep
        R_60 = [
            [0.5, 0.0, 0.866025],
            [0.0, 1.0, 0.0],
            [-0.866025, 0.0, 0.5],
        ]
        poses = {
            "f0": ExtrinsicPose(rotation_matrix=np.eye(3).tolist(), translation_vector=[0.0, 0.0, 0.0]),
            "f1": ExtrinsicPose(rotation_matrix=np.eye(3).tolist(), translation_vector=[0.0, 0.0, 0.0]),
            "f2": ExtrinsicPose(rotation_matrix=R_60, translation_vector=[1.0, 0.0, 0.0]),
        }
        dims = {fid: (1080, 1920) for fid in poses}
        intrinsics = {fid: standard_intrinsics for fid in poses}

        mvs_in = MVSInput(
            selected_frame_ids=["f0", "f1", "f2"],
            image_dimensions=dims,
            camera_intrinsics=intrinsics,
            camera_poses=poses,
        )

        selector = HeuristicViewPairSelector()
        graph = selector.select_pairs(mvs_in, cfg)

        # (f0, f1) must be rejected due to zero baseline
        assert ("f0", "f1") in graph.rejection_reasons
        assert "Coincident" in graph.rejection_reasons[("f0", "f1")]

        # (f0, f2) must be rejected due to steep angle
        assert ("f0", "f2") in graph.rejection_reasons
        assert "Viewing angle too steep" in graph.rejection_reasons[("f0", "f2")]

    def test_dynamic_risk_propagation_into_view_pairs(
        self, standard_intrinsics: CameraIntrinsics
    ):
        """Verify contract K: Dynamic motion risk survives into MVS view-pair scores."""
        cfg = MVSConfig()
        poses = {
            "f0": ExtrinsicPose(rotation_matrix=np.eye(3).tolist(), translation_vector=[0.0, 0.0, 0.0]),
            "f1": ExtrinsicPose(rotation_matrix=np.eye(3).tolist(), translation_vector=[1.0, 0.0, 0.0]),
        }
        dims = {fid: (1080, 1920) for fid in poses}
        intrinsics = {fid: standard_intrinsics for fid in poses}

        # Case 1: zero dynamic risk
        mvs_clean = MVSInput(
            selected_frame_ids=["f0", "f1"],
            image_dimensions=dims,
            camera_intrinsics=intrinsics,
            camera_poses=poses,
            dynamic_risk_scores={"f0": 0.0, "f1": 0.0},
        )
        selector = HeuristicViewPairSelector()
        g_clean = selector.select_pairs(mvs_clean, cfg)
        score_clean = g_clean.selected_edges[0].viewpoint_suitability_score

        # Case 2: high dynamic risk on f1
        mvs_risky = MVSInput(
            selected_frame_ids=["f0", "f1"],
            image_dimensions=dims,
            camera_intrinsics=intrinsics,
            camera_poses=poses,
            dynamic_risk_scores={"f0": 0.0, "f1": 0.8},
        )
        g_risky = selector.select_pairs(mvs_risky, cfg)
        score_risky = g_risky.selected_edges[0].viewpoint_suitability_score

        # High dynamic risk must strictly penalize suitability score
        assert score_risky < score_clean
        assert g_risky.selected_edges[0].dynamic_risk == pytest.approx(0.8)

    def test_cross_view_geometric_depth_consistency_acceptance(
        self, standard_intrinsics: CameraIntrinsics, identity_pose: ExtrinsicPose, baseline_pose: ExtrinsicPose
    ):
        """Verify contract B: Consistent synthetic 3D point across two views passes consistency."""
        H, W = standard_intrinsics.height, standard_intrinsics.width
        ref_depth_arr = np.zeros((H, W), dtype=np.float32)
        src_depth_arr = np.zeros((H, W), dtype=np.float32)
        ref_mask = np.zeros((H, W), dtype=bool)
        src_mask = np.zeros((H, W), dtype=bool)

        # True 3D point in world
        X_w = np.array([0.5, 0.0, 6.0], dtype=np.float64)

        # Project into reference camera
        R_ref = np.array(identity_pose.rotation_matrix, dtype=np.float64)
        c_ref = np.array(identity_pose.translation_vector, dtype=np.float64)
        t_ref = -R_ref @ c_ref
        px_ref, z_ref, ok_ref = MVSGeometryMath.project_world_point(X_w, standard_intrinsics, R_ref, t_ref)
        assert ok_ref is True and px_ref is not None

        # Project into source camera
        R_src = np.array(baseline_pose.rotation_matrix, dtype=np.float64)
        c_src = np.array(baseline_pose.translation_vector, dtype=np.float64)
        t_src = -R_src @ c_src
        px_src, z_src, ok_src = MVSGeometryMath.project_world_point(X_w, standard_intrinsics, R_src, t_src)
        assert ok_src is True and px_src is not None

        # Create maps with exact matching depth at projected locations
        r_row, r_col = int(round(px_ref[1])), int(round(px_ref[0]))
        s_row, s_col = int(round(px_src[1])), int(round(px_src[0]))

        ref_depth_arr[r_row, r_col] = float(z_ref)
        ref_mask[r_row, r_col] = True

        src_depth_arr[s_row, s_col] = float(z_src)
        src_mask[s_row, s_col] = True

        ref_depth = DepthMap("f0", W, H, ref_depth_arr, ref_mask)
        src_depth = DepthMap("f1", W, H, src_depth_arr, src_mask)

        checker = GeometricDepthConsistencyChecker()
        cfg = MVSConfig()
        c_mask, v_state = checker.check_consistency(
            ref_depth, src_depth, identity_pose, baseline_pose, standard_intrinsics, standard_intrinsics, cfg
        )

        assert bool(c_mask[r_row, r_col]) is True
        assert v_state[r_row, r_col] == PointVisibilityState.VALID.value

    def test_cross_view_geometric_depth_inconsistency_rejection(
        self, standard_intrinsics: CameraIntrinsics, identity_pose: ExtrinsicPose, baseline_pose: ExtrinsicPose
    ):
        """Verify contract C: Depth disagreement exceeding tolerance is rejected as INCONSISTENT."""
        H, W = standard_intrinsics.height, standard_intrinsics.width
        ref_depth_arr = np.zeros((H, W), dtype=np.float32)
        src_depth_arr = np.zeros((H, W), dtype=np.float32)
        ref_mask = np.zeros((H, W), dtype=bool)
        src_mask = np.zeros((H, W), dtype=bool)

        X_w = np.array([0.5, 0.0, 6.0], dtype=np.float64)

        R_ref = np.array(identity_pose.rotation_matrix, dtype=np.float64)
        c_ref = np.array(identity_pose.translation_vector, dtype=np.float64)
        t_ref = -R_ref @ c_ref
        px_ref, z_ref, _ = MVSGeometryMath.project_world_point(X_w, standard_intrinsics, R_ref, t_ref)

        R_src = np.array(baseline_pose.rotation_matrix, dtype=np.float64)
        c_src = np.array(baseline_pose.translation_vector, dtype=np.float64)
        t_src = -R_src @ c_src
        px_src, z_src, _ = MVSGeometryMath.project_world_point(X_w, standard_intrinsics, R_src, t_src)

        assert px_ref is not None and px_src is not None
        r_row, r_col = int(round(px_ref[1])), int(round(px_ref[0]))
        s_row, s_col = int(round(px_src[1])), int(round(px_src[0]))

        ref_depth_arr[r_row, r_col] = float(z_ref)
        ref_mask[r_row, r_col] = True

        # Inconsistent source depth (30% disagreement, tolerance is 5%)
        src_depth_arr[s_row, s_col] = float(z_src) * 1.30
        src_mask[s_row, s_col] = True

        ref_depth = DepthMap("f0", W, H, ref_depth_arr, ref_mask)
        src_depth = DepthMap("f1", W, H, src_depth_arr, src_mask)

        checker = GeometricDepthConsistencyChecker()
        cfg = MVSConfig()
        c_mask, v_state = checker.check_consistency(
            ref_depth, src_depth, identity_pose, baseline_pose, standard_intrinsics, standard_intrinsics, cfg
        )

        assert bool(c_mask[r_row, r_col]) is False
        assert v_state[r_row, r_col] == PointVisibilityState.INCONSISTENT.value

    def test_cross_view_occlusion_and_bounds_rejection(
        self, standard_intrinsics: CameraIntrinsics, identity_pose: ExtrinsicPose, baseline_pose: ExtrinsicPose
    ):
        """Verify contract E: Points outside source bounds or occluded are tagged OCCLUDED."""
        H, W = standard_intrinsics.height, standard_intrinsics.width
        ref_depth_arr = np.zeros((H, W), dtype=np.float32)
        ref_mask = np.zeros((H, W), dtype=bool)

        # Point at boundary that projects outside source camera raster
        ref_depth_arr[0, 0] = 2.0
        ref_mask[0, 0] = True

        ref_depth = DepthMap("f0", W, H, ref_depth_arr, ref_mask)
        src_depth = DepthMap("f1", W, H, np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=bool))

        checker = GeometricDepthConsistencyChecker()
        cfg = MVSConfig()
        c_mask, v_state = checker.check_consistency(
            ref_depth, src_depth, identity_pose, baseline_pose, standard_intrinsics, standard_intrinsics, cfg
        )

        assert bool(c_mask[0, 0]) is False
        assert v_state[0, 0] in (PointVisibilityState.OCCLUDED.value, PointVisibilityState.INCONSISTENT.value)

    def test_depth_to_world_points_backprojection_filtering(
        self, standard_intrinsics: CameraIntrinsics, identity_pose: ExtrinsicPose
    ):
        """Verify depth_to_world_points backprojects only valid pixels."""
        H, W = 10, 10
        depth_arr = np.full((H, W), 5.0, dtype=np.float32)
        valid_mask = np.ones((H, W), dtype=bool)
        valid_mask[0, 0] = False  # Masked hole
        depth_arr[1, 1] = 0.1     # Below min_depth_units (0.5)

        conf_arr = np.full((H, W), 0.8, dtype=np.float32)
        supp_arr = np.full((H, W), 3, dtype=np.int32)
        vis_arr = np.full((H, W), PointVisibilityState.VALID.value, dtype=object)

        d_map = DepthMap("f0", W, H, depth_arr, valid_mask)
        c_map = DepthConfidenceMap("f0", W, H, conf_arr, conf_arr, supp_arr, vis_arr, conf_arr)

        cfg = MVSConfig(min_depth_units=0.5)
        obs = depth_to_world_points(d_map, c_map, standard_intrinsics, identity_pose, cfg)

        # Expected points: 100 - 1 (masked) - 1 (too close) = 98
        assert len(obs) == 98
        assert all(o.visibility_state == PointVisibilityState.VALID for o in obs)
        assert all(o.validation_status == PointValidationStatus.VALIDATED for o in obs)

    def test_dense_point_fusion_spatial_deduplication_and_support_count(self):
        """Verify contract G: Duplicate observations in same voxel fuse into centroid and retain support."""
        # 3 observations of essentially the same physical point from 3 different frames
        obs = [
            DensePointObservation(
                world_point=np.array([1.000, 2.000, 3.000]),
                reference_frame_id="f0",
                pixel_coord=(100.0, 100.0),
                depth=3.0,
                confidence=0.8,
                source_view_support_count=2,
            ),
            DensePointObservation(
                world_point=np.array([1.005, 2.002, 3.001]),
                reference_frame_id="f1",
                pixel_coord=(95.0, 100.0),
                depth=3.0,
                confidence=0.9,
                source_view_support_count=2,
            ),
            DensePointObservation(
                world_point=np.array([1.002, 2.001, 3.004]),
                reference_frame_id="f2",
                pixel_coord=(90.0, 100.0),
                depth=3.0,
                confidence=0.7,
                source_view_support_count=2,
            ),
        ]

        cfg = MVSConfig(voxel_grid_resolution=0.05, min_consistent_views=2)
        fusion = VoxelGridDensePointFusion()
        cloud = fusion.fuse(obs, cfg)

        # All 3 points fall within the 0.05 resolution voxel -> 1 consolidated fused point
        assert cloud.total_fused_points == 1
        assert cloud.support_counts[0] == 3
        assert set(cloud.source_frame_ids[0]) == {"f0", "f1", "f2"}
        # Check centroid coordinate is near [1.002, 2.001, 3.002]
        np.testing.assert_allclose(cloud.points[0], [1.002, 2.001, 3.002], atol=1e-2)

    def test_dense_point_fusion_filters_insufficient_multi_view_support(self):
        """Verify contract F: Points supported by only 1 view are filtered out when min_consistent_views=2."""
        obs = [
            DensePointObservation(
                world_point=np.array([5.0, 5.0, 5.0]),
                reference_frame_id="f0",
                pixel_coord=(50.0, 50.0),
                depth=5.0,
                confidence=0.8,
                source_view_support_count=1,
            )
        ]

        cfg = MVSConfig(min_consistent_views=2)
        fusion = VoxelGridDensePointFusion()
        cloud = fusion.fuse(obs, cfg)

        assert cloud.total_fused_points == 0

    def test_reconstruction_unit_preservation_no_meters_claim(self):
        """Verify contract J: Dense point cloud preserves RECONSTRUCTION_UNITS and is_metric_scale=False."""
        cloud = DensePointCloud(
            points=np.array([[1.0, 2.0, 3.0]]),
            confidences=np.array([0.9], dtype=np.float32),
            support_counts=np.array([3], dtype=np.int32),
            source_frame_ids=[["f0", "f1", "f2"]],
            visibility_states=[PointVisibilityState.VALID],
            validation_statuses=[PointValidationStatus.VALIDATED],
            total_fused_points=1,
            mean_confidence=0.9,
            depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
            is_metric_scale=False,
        )

        cfg = MVSConfig()
        ok, reason, diags = MVSValidator.validate_point_cloud(cloud, cfg)
        assert ok is True
        assert reason is None
        assert cloud.depth_unit == DepthUnit.RECONSTRUCTION_UNITS
        assert cloud.is_metric_scale is False

    def test_validator_rejects_premature_metric_scale_claim(self):
        """Verify validator flags any illegal claim of metric scale in Phase 3E."""
        illegal_cloud = DensePointCloud(
            points=np.array([[1.0, 2.0, 3.0]]),
            confidences=np.array([0.9], dtype=np.float32),
            support_counts=np.array([3], dtype=np.int32),
            source_frame_ids=[["f0", "f1", "f2"]],
            visibility_states=[PointVisibilityState.VALID],
            validation_statuses=[PointValidationStatus.VALIDATED],
            total_fused_points=1,
            mean_confidence=0.9,
            depth_unit=DepthUnit.METRIC_METERS,
            is_metric_scale=True,  # Illegal without georeferencing
        )

        cfg = MVSConfig()
        ok, reason, diags = MVSValidator.validate_point_cloud(illegal_cloud, cfg)
        assert ok is False
        assert reason == MVSFailureReason.DENSE_FUSION_FAILED
        assert any("claims metric scale" in d for d in diags)

    def test_validator_input_and_depth_map_checks(self, standard_intrinsics: CameraIntrinsics):
        """Verify MVSValidator catches missing poses and dimension mismatches."""
        cfg = MVSConfig(min_source_views=2)

        # Incomplete input (only 1 frame)
        bad_input = MVSInput(
            selected_frame_ids=["f0"],
            image_dimensions={"f0": (1080, 1920)},
            camera_intrinsics={"f0": standard_intrinsics},
            camera_poses={},
        )
        ok_in, reason_in, _ = MVSValidator.validate_mvs_input(bad_input, cfg)
        assert ok_in is False
        assert reason_in == MVSFailureReason.INSUFFICIENT_VALID_VIEWS

        # Dimension mismatch in depth map
        bad_map = DepthMap(
            reference_frame_id="f0",
            width=100,
            height=80,
            depth_array=np.zeros((50, 50), dtype=np.float32),  # Shape mismatch
            valid_mask=np.zeros((50, 50), dtype=bool),
        )
        ok_dm, reason_dm, _ = MVSValidator.validate_depth_map(bad_map, cfg)
        assert ok_dm is False
        assert reason_dm == MVSFailureReason.INCOMPATIBLE_IMAGE_DIMENSIONS

    def test_abstract_depth_estimator_interface_compliance(self, standard_intrinsics: CameraIntrinsics):
        """Verify contract 16: IMVSDepthEstimator satisfies plugin architecture without algorithm lock-in."""
        class MockMVSPlugin(IMVSDepthEstimator):
            @property
            def name(self) -> str:
                return "MockClassicalStereoPlugin_v1"

            def estimate_depth_map(
                self,
                ref_frame_id: str,
                source_frame_ids: List[str],
                mvs_input: MVSInput,
                config: MVSConfig,
            ) -> Tuple[DepthMap, DepthConfidenceMap]:
                H, W = 10, 10
                d_arr = np.full((H, W), 4.0, dtype=np.float32)
                v_mask = np.ones((H, W), dtype=bool)
                c_arr = np.full((H, W), 0.75, dtype=np.float32)
                s_arr = np.full((H, W), len(source_frame_ids), dtype=np.int32)
                vis_arr = np.full((H, W), PointVisibilityState.VALID.value, dtype=object)

                dm = DepthMap(ref_frame_id, W, H, d_arr, v_mask)
                cm = DepthConfidenceMap(ref_frame_id, W, H, c_arr, c_arr, s_arr, vis_arr, c_arr)
                return dm, cm

        plugin = MockMVSPlugin()
        assert plugin.name == "MockClassicalStereoPlugin_v1"

        poses = {
            "f0": ExtrinsicPose(rotation_matrix=np.eye(3).tolist(), translation_vector=[0.0, 0.0, 0.0]),
            "f1": ExtrinsicPose(rotation_matrix=np.eye(3).tolist(), translation_vector=[1.0, 0.0, 0.0]),
        }
        mvs_in = MVSInput(
            selected_frame_ids=["f0", "f1"],
            image_dimensions={"f0": (10, 10), "f1": (10, 10)},
            camera_intrinsics={"f0": standard_intrinsics, "f1": standard_intrinsics},
            camera_poses=poses,
        )
        dm, cm = plugin.estimate_depth_map("f0", ["f1"], mvs_in, MVSConfig())
        assert dm.reference_frame_id == "f0"
        assert cm.support_view_count[0, 0] == 1
        assert dm.get_depth_at(5.0, 5.0) == pytest.approx(4.0)

    def test_failure_and_visibility_taxonomies_completeness(self):
        """Verify full enumeration of failure and visibility taxonomies."""
        assert len(MVSFailureReason) == 13
        assert len(PointVisibilityState) == 7
        assert len(PointValidationStatus) == 4
        assert len(DepthUnit) == 2

        # Check critical failure members exist
        assert MVSFailureReason.INSUFFICIENT_VALID_VIEWS.value == "INSUFFICIENT_VALID_VIEWS"
        assert MVSFailureReason.INSUFFICIENT_CONSISTENCY.value == "INSUFFICIENT_CONSISTENCY"
        assert MVSFailureReason.DENSE_FUSION_FAILED.value == "DENSE_FUSION_FAILED"

        # Check visibility members
        assert PointVisibilityState.VISIBLE.value == "VISIBLE"
        assert PointVisibilityState.OCCLUDED.value == "OCCLUDED"
        assert PointVisibilityState.INCONSISTENT.value == "INCONSISTENT"
        assert PointVisibilityState.VALID.value == "VALID"
