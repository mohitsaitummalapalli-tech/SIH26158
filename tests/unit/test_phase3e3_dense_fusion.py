"""Phase 3E.3: Multi-View Dense Point Fusion Unit & Adversarial Tests.

Validates:
1. Spatial clustering and multi-view consensus filtering
2. Distinct-view support count enforcement (same-frame observations do not inflate support)
3. Explicit geometric compatibility (no naive proximity merging)
4. Anti-chaining bounding diameter constraints
5. Confidence weighting and heuristic score semantics
6. Input-order invariance and bit-exact determinism
7. Scale equivariance in RECONSTRUCTION_UNITS
8. Comprehensive rejection taxonomy accounting
"""

import math
import random
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
    DensePointObservation,
    DensePointCloud,
)

from src.geometry.dense_fusion import (
    FusionRejectionReason,
    FusionWeightingScheme,
    SingleViewRetentionPolicy,
    DenseFusionConfig,
    FusedDensePoint,
    DensePointCluster,
    DenseFusionResult,
    DensePointFusionEngine,
)


@pytest.fixture
def base_config() -> DenseFusionConfig:
    return DenseFusionConfig(
        spatial_distance_threshold=0.05,
        voxel_grid_resolution=0.05,
        min_distinct_view_support=2,
        min_observation_confidence=0.20,
        max_cluster_diameter=0.07,
        weighting_scheme=FusionWeightingScheme.CONFIDENCE_WEIGHTED,
        single_view_policy=SingleViewRetentionPolicy.REJECT_SINGLE_VIEW,
    )


class TestPhase3E3DenseFusion:
    """Rigorous first-principles unit and adversarial test suite for Phase 3E.3."""

    def test_two_observations_same_point_merge(self, base_config: DenseFusionConfig):
        """Test A: Two observations of the same surface point from different frames merge."""
        obs1 = DensePointObservation(
            world_point=np.array([1.0, 2.0, 10.0], dtype=np.float64),
            reference_frame_id="frame_001",
            pixel_coord=(320.0, 240.0),
            depth=10.0,
            confidence=0.85,
            visibility_state=PointVisibilityState.VALID,
            validation_status=PointValidationStatus.VALIDATED,
        )
        obs2 = DensePointObservation(
            world_point=np.array([1.01, 2.01, 10.01], dtype=np.float64),
            reference_frame_id="frame_002",
            pixel_coord=(325.0, 242.0),
            depth=10.01,
            confidence=0.75,
            visibility_state=PointVisibilityState.VALID,
            validation_status=PointValidationStatus.VALIDATED,
        )

        engine = DensePointFusionEngine(base_config)
        res = engine.fuse_observations([obs1, obs2])

        assert res.total_fused_points == 1
        fused_pt = res.fused_points[0]
        assert fused_pt.distinct_view_count == 2
        assert fused_pt.total_observation_count == 2
        assert fused_pt.contributing_frame_ids == ["frame_001", "frame_002"]
        assert fused_pt.validation_status == PointValidationStatus.VALIDATED
        assert fused_pt.visibility_state == PointVisibilityState.VALID
        np.testing.assert_allclose(fused_pt.world_point, [1.0046875, 2.0046875, 10.0046875], atol=1e-5)

    def test_three_view_support(self, base_config: DenseFusionConfig):
        """Test B: Three observations from three distinct frames merge with support_count = 3."""
        obs1 = DensePointObservation(
            world_point=np.array([0.0, 0.0, 5.0], dtype=np.float64),
            reference_frame_id="cam_A",
            pixel_coord=(100.0, 100.0),
            depth=5.0,
            confidence=0.9,
        )
        obs2 = DensePointObservation(
            world_point=np.array([0.01, 0.0, 5.0], dtype=np.float64),
            reference_frame_id="cam_B",
            pixel_coord=(102.0, 100.0),
            depth=5.0,
            confidence=0.8,
        )
        obs3 = DensePointObservation(
            world_point=np.array([-0.01, 0.0, 5.0], dtype=np.float64),
            reference_frame_id="cam_C",
            pixel_coord=(98.0, 100.0),
            depth=5.0,
            confidence=0.7,
        )

        engine = DensePointFusionEngine(base_config)
        res = engine.fuse_observations([obs1, obs2, obs3])

        assert res.total_fused_points == 1
        assert res.fused_points[0].distinct_view_count == 3
        assert res.fused_points[0].contributing_frame_ids == ["cam_A", "cam_B", "cam_C"]

    def test_minimum_distinct_view_support_rejection_and_retention(self):
        """Test C: Single-view observation behavior under REJECT_SINGLE_VIEW vs RETAIN_AS_OBSERVED."""
        obs = DensePointObservation(
            world_point=np.array([3.0, 4.0, 15.0], dtype=np.float64),
            reference_frame_id="cam_solo",
            pixel_coord=(200.0, 150.0),
            depth=15.0,
            confidence=0.8,
        )

        # 1. Policy: REJECT_SINGLE_VIEW
        cfg_reject = DenseFusionConfig(min_distinct_view_support=2, single_view_policy=SingleViewRetentionPolicy.REJECT_SINGLE_VIEW)
        engine_reject = DensePointFusionEngine(cfg_reject)
        res_reject = engine_reject.fuse_observations([obs])
        assert res_reject.total_fused_points == 0
        assert res_reject.rejection_breakdown[FusionRejectionReason.INSUFFICIENT_DISTINCT_VIEWS.value] == 1

        # 2. Policy: RETAIN_AS_OBSERVED
        cfg_retain = DenseFusionConfig(min_distinct_view_support=2, single_view_policy=SingleViewRetentionPolicy.RETAIN_AS_OBSERVED)
        engine_retain = DensePointFusionEngine(cfg_retain)
        res_retain = engine_retain.fuse_observations([obs])
        assert res_retain.total_fused_points == 1
        assert res_retain.fused_points[0].distinct_view_count == 1
        assert res_retain.fused_points[0].validation_status == PointValidationStatus.OBSERVED

    def test_same_frame_duplicate_does_not_inflate_support(self, base_config: DenseFusionConfig):
        """Test D: Multiple observations from the SAME frame merge but distinct_view_count remains 1."""
        obs1 = DensePointObservation(
            world_point=np.array([1.0, 1.0, 8.0], dtype=np.float64),
            reference_frame_id="same_frame",
            pixel_coord=(100.0, 100.0),
            depth=8.0,
            confidence=0.8,
        )
        obs2 = DensePointObservation(
            world_point=np.array([1.01, 1.01, 8.01], dtype=np.float64),
            reference_frame_id="same_frame",
            pixel_coord=(101.0, 101.0),
            depth=8.01,
            confidence=0.8,
        )

        # Under min_distinct_view_support=2, same-frame duplicates must NOT satisfy multi-view support!
        engine = DensePointFusionEngine(base_config)
        res = engine.fuse_observations([obs1, obs2])
        assert res.total_fused_points == 0
        assert res.rejection_breakdown[FusionRejectionReason.INSUFFICIENT_DISTINCT_VIEWS.value] == 2

    def test_nearby_distinct_surfaces_remain_separate(self, base_config: DenseFusionConfig):
        """Test E: Surfaces separated by > spatial_distance_threshold remain separate clusters."""
        # Surface 1
        obs1_a = DensePointObservation(world_point=np.array([0.0, 0.0, 10.0], dtype=np.float64), reference_frame_id="f1", pixel_coord=(0, 0), depth=10.0, confidence=0.8)
        obs1_b = DensePointObservation(world_point=np.array([0.01, 0.0, 10.0], dtype=np.float64), reference_frame_id="f2", pixel_coord=(0, 0), depth=10.0, confidence=0.8)

        # Surface 2: 0.20 units away (> 0.05 threshold)
        obs2_a = DensePointObservation(world_point=np.array([0.20, 0.0, 10.0], dtype=np.float64), reference_frame_id="f1", pixel_coord=(50, 0), depth=10.0, confidence=0.8)
        obs2_b = DensePointObservation(world_point=np.array([0.21, 0.0, 10.0], dtype=np.float64), reference_frame_id="f2", pixel_coord=(50, 0), depth=10.0, confidence=0.8)

        engine = DensePointFusionEngine(base_config)
        res = engine.fuse_observations([obs1_a, obs1_b, obs2_a, obs2_b])

        assert res.total_fused_points == 2
        assert abs(res.fused_points[0].world_point[0] - 0.005) < 1e-4
        assert abs(res.fused_points[1].world_point[0] - 0.205) < 1e-4

    def test_voxel_boundary_search_merges_adjacent_points(self):
        """Test F: Observations straddling a voxel boundary (e.g. at x=0.049 and x=0.051 for voxel_res=0.05) merge."""
        cfg = DenseFusionConfig(spatial_distance_threshold=0.05, voxel_grid_resolution=0.05, min_distinct_view_support=2)
        # obs1 is in voxel 0 (x=0.049), obs2 is in voxel 1 (x=0.051). Distance = 0.002 <= 0.05.
        obs1 = DensePointObservation(world_point=np.array([0.049, 1.0, 5.0], dtype=np.float64), reference_frame_id="f1", pixel_coord=(0, 0), depth=5.0, confidence=0.8)
        obs2 = DensePointObservation(world_point=np.array([0.051, 1.0, 5.0], dtype=np.float64), reference_frame_id="f2", pixel_coord=(0, 0), depth=5.0, confidence=0.8)

        engine = DensePointFusionEngine(cfg)
        res = engine.fuse_observations([obs1, obs2])

        assert res.total_fused_points == 1
        assert res.fused_points[0].distinct_view_count == 2
        assert res.fused_points[0].world_point[0] == pytest.approx(0.050, abs=1e-5)

    def test_confidence_weighting_pulls_centroid_towards_higher_confidence(self, base_config: DenseFusionConfig):
        """Test G: Fused point coordinate centroid is weighted proportionally to heuristic confidence."""
        # Point 1 at x=0.0 with low confidence 0.2
        obs1 = DensePointObservation(world_point=np.array([0.0, 0.0, 10.0], dtype=np.float64), reference_frame_id="f1", pixel_coord=(0, 0), depth=10.0, confidence=0.20)
        # Point 2 at x=0.04 with high confidence 0.8
        obs2 = DensePointObservation(world_point=np.array([0.04, 0.0, 10.0], dtype=np.float64), reference_frame_id="f2", pixel_coord=(0, 0), depth=10.0, confidence=0.80)

        engine = DensePointFusionEngine(base_config)
        res = engine.fuse_observations([obs1, obs2])

        assert res.total_fused_points == 1
        # Weighted x = (0.0 * 0.2 + 0.04 * 0.8) / (0.2 + 0.8) = 0.032
        assert res.fused_points[0].world_point[0] == pytest.approx(0.032, abs=1e-5)

    def test_low_and_invalid_confidence_rejection(self, base_config: DenseFusionConfig):
        """Test H & J: Observations with low, negative, NaN, Inf, or >1.0 confidence are rejected."""
        obs_low = DensePointObservation(world_point=np.array([1, 1, 1], dtype=np.float64), reference_frame_id="f1", pixel_coord=(0, 0), depth=1.0, confidence=0.10)
        obs_nan = DensePointObservation(world_point=np.array([1, 1, 1], dtype=np.float64), reference_frame_id="f1", pixel_coord=(0, 0), depth=1.0, confidence=float("nan"))
        obs_inf = DensePointObservation(world_point=np.array([1, 1, 1], dtype=np.float64), reference_frame_id="f1", pixel_coord=(0, 0), depth=1.0, confidence=float("inf"))
        obs_neg = DensePointObservation(world_point=np.array([1, 1, 1], dtype=np.float64), reference_frame_id="f1", pixel_coord=(0, 0), depth=1.0, confidence=-0.5)
        obs_over = DensePointObservation(world_point=np.array([1, 1, 1], dtype=np.float64), reference_frame_id="f1", pixel_coord=(0, 0), depth=1.0, confidence=1.5)

        engine = DensePointFusionEngine(base_config)
        res = engine.fuse_observations([obs_low, obs_nan, obs_inf, obs_neg, obs_over])

        assert res.total_fused_points == 0
        assert res.rejection_breakdown[FusionRejectionReason.LOW_CONFIDENCE.value] == 1
        assert res.rejection_breakdown[FusionRejectionReason.NON_FINITE_CONFIDENCE.value] == 2
        assert res.rejection_breakdown[FusionRejectionReason.OUT_OF_BOUNDS_CONFIDENCE.value] == 2

    def test_invalid_xyz_rejection(self, base_config: DenseFusionConfig):
        """Test I: Observations with NaN or Inf coordinates are safely rejected."""
        obs_nan = DensePointObservation(world_point=np.array([np.nan, 0.0, 5.0], dtype=np.float64), reference_frame_id="f1", pixel_coord=(0, 0), depth=5.0, confidence=0.8)
        obs_inf = DensePointObservation(world_point=np.array([0.0, np.inf, 5.0], dtype=np.float64), reference_frame_id="f1", pixel_coord=(0, 0), depth=5.0, confidence=0.8)

        engine = DensePointFusionEngine(base_config)
        res = engine.fuse_observations([obs_nan, obs_inf])

        assert res.total_fused_points == 0
        assert res.rejection_breakdown[FusionRejectionReason.NON_FINITE_COORDINATES.value] == 2

    def test_deterministic_repeated_execution_and_input_order_invariance(self, base_config: DenseFusionConfig):
        """Test K & L: Repeated execution and arbitrary input permutations produce bit-exact identical outputs."""
        obs_list = [
            DensePointObservation(world_point=np.array([1.0, 2.0, 10.0], dtype=np.float64), reference_frame_id="frame_01", pixel_coord=(100, 100), depth=10.0, confidence=0.8),
            DensePointObservation(world_point=np.array([1.02, 2.01, 10.01], dtype=np.float64), reference_frame_id="frame_02", pixel_coord=(105, 102), depth=10.01, confidence=0.75),
            DensePointObservation(world_point=np.array([5.0, 5.0, 12.0], dtype=np.float64), reference_frame_id="frame_01", pixel_coord=(300, 300), depth=12.0, confidence=0.9),
            DensePointObservation(world_point=np.array([5.01, 5.02, 12.0], dtype=np.float64), reference_frame_id="frame_03", pixel_coord=(295, 301), depth=12.0, confidence=0.85),
        ]

        engine = DensePointFusionEngine(base_config)
        res_original = engine.fuse_observations(obs_list)

        # 1. Reverse order
        res_reversed = engine.fuse_observations(list(reversed(obs_list)))

        # 2. Random permutation
        shuffled = list(obs_list)
        random.seed(42)
        random.shuffle(shuffled)
        res_shuffled = engine.fuse_observations(shuffled)

        assert res_original.total_fused_points == 2
        assert res_reversed.total_fused_points == 2
        assert res_shuffled.total_fused_points == 2

        np.testing.assert_array_equal(res_original.point_cloud.points, res_reversed.point_cloud.points)
        np.testing.assert_array_equal(res_original.point_cloud.points, res_shuffled.point_cloud.points)
        np.testing.assert_array_equal(res_original.point_cloud.confidences, res_reversed.point_cloud.confidences)

    def test_provenance_preservation(self, base_config: DenseFusionConfig):
        """Test M: Every fused point preserves full traceability to all contributing observations."""
        obs1 = DensePointObservation(world_point=np.array([0, 0, 5], dtype=np.float64), reference_frame_id="f_alpha", pixel_coord=(10, 20), depth=5.0, confidence=0.8)
        obs2 = DensePointObservation(world_point=np.array([0.01, 0, 5], dtype=np.float64), reference_frame_id="f_beta", pixel_coord=(12, 22), depth=5.01, confidence=0.9)

        engine = DensePointFusionEngine(base_config)
        res = engine.fuse_observations([obs1, obs2])

        fused_pt = res.fused_points[0]
        assert fused_pt.contributing_frame_ids == ["f_alpha", "f_beta"]
        assert fused_pt.contributing_pixel_coords == [(10, 20), (12, 22)]
        assert fused_pt.contributing_depths == [5.0, 5.01]
        assert fused_pt.contributing_confidences == [0.8, 0.9]
        assert fused_pt.provenance["depth_unit"] == DepthUnit.RECONSTRUCTION_UNITS.value
        assert fused_pt.provenance["is_metric_scale"] is False

    def test_scale_equivariance(self):
        """Test N: Scaling input coordinates and thresholds by s scales fused output coordinates by s."""
        scale_s = 3.5

        # Unscaled inputs
        cfg_1 = DenseFusionConfig(spatial_distance_threshold=0.05, voxel_grid_resolution=0.05, max_cluster_diameter=0.07, min_distinct_view_support=2)
        obs1_1 = DensePointObservation(world_point=np.array([1.0, 2.0, 10.0], dtype=np.float64), reference_frame_id="f1", pixel_coord=(0, 0), depth=10.0, confidence=0.8)
        obs2_1 = DensePointObservation(world_point=np.array([1.02, 2.01, 10.01], dtype=np.float64), reference_frame_id="f2", pixel_coord=(0, 0), depth=10.01, confidence=0.6)

        # Scaled inputs
        cfg_s = DenseFusionConfig(spatial_distance_threshold=0.05 * scale_s, voxel_grid_resolution=0.05 * scale_s, max_cluster_diameter=0.07 * scale_s, min_distinct_view_support=2)
        obs1_s = DensePointObservation(world_point=np.array([1.0, 2.0, 10.0], dtype=np.float64) * scale_s, reference_frame_id="f1", pixel_coord=(0, 0), depth=10.0 * scale_s, confidence=0.8)
        obs2_s = DensePointObservation(world_point=np.array([1.02, 2.01, 10.01], dtype=np.float64) * scale_s, reference_frame_id="f2", pixel_coord=(0, 0), depth=10.01 * scale_s, confidence=0.6)

        res_1 = DensePointFusionEngine(cfg_1).fuse_observations([obs1_1, obs2_1])
        res_s = DensePointFusionEngine(cfg_s).fuse_observations([obs1_s, obs2_s])

        assert res_1.total_fused_points == 1 and res_s.total_fused_points == 1
        np.testing.assert_allclose(res_s.fused_points[0].world_point, res_1.fused_points[0].world_point * scale_s, atol=1e-10)

    def test_dynamic_scene_risk_filtering(self):
        """Test Q: Observations with high dynamic-scene risk in provenance are rejected."""
        cfg = DenseFusionConfig(max_dynamic_risk=0.50, min_distinct_view_support=2)
        obs1 = DensePointObservation(world_point=np.array([1, 1, 10], dtype=np.float64), reference_frame_id="f1", pixel_coord=(0, 0), depth=10.0, confidence=0.8, provenance={"dynamic_risk": 0.85})
        obs2 = DensePointObservation(world_point=np.array([1.01, 1, 10], dtype=np.float64), reference_frame_id="f2", pixel_coord=(0, 0), depth=10.0, confidence=0.8, provenance={"dynamic_risk": 0.10})

        engine = DensePointFusionEngine(cfg)
        res = engine.fuse_observations([obs1, obs2])

        assert res.rejection_breakdown[FusionRejectionReason.DYNAMIC_RISK_EXCEEDED.value] == 1
        assert res.total_fused_points == 0  # obs2 alone fails min_distinct_view_support=2

    def test_adversarial_cluster_chain_rejection(self):
        """Test S (Critical Adversarial): Prevent naive transitive chaining from creating elongated false clusters.
        
        Point A at [0.0, 0.0, 10.0]
        Point B at [0.04, 0.0, 10.0] (dist A-B = 0.04 <= spatial_thresh 0.05)
        Point C at [0.08, 0.0, 10.0] (dist B-C = 0.04 <= spatial_thresh 0.05, but dist A-C = 0.08 > max_cluster_diameter 0.06)
        """
        cfg = DenseFusionConfig(
            spatial_distance_threshold=0.05,
            max_cluster_diameter=0.06,  # Strict diameter limit
            min_distinct_view_support=1,
            single_view_policy=SingleViewRetentionPolicy.RETAIN_AS_OBSERVED,
        )

        obs_A = DensePointObservation(world_point=np.array([0.00, 0.0, 10.0], dtype=np.float64), reference_frame_id="fA", pixel_coord=(0, 0), depth=10.0, confidence=0.8)
        obs_B = DensePointObservation(world_point=np.array([0.04, 0.0, 10.0], dtype=np.float64), reference_frame_id="fB", pixel_coord=(0, 0), depth=10.0, confidence=0.8)
        obs_C = DensePointObservation(world_point=np.array([0.08, 0.0, 10.0], dtype=np.float64), reference_frame_id="fC", pixel_coord=(0, 0), depth=10.0, confidence=0.8)

        engine = DensePointFusionEngine(cfg)
        res = engine.fuse_observations([obs_A, obs_B, obs_C])

        # A and B merge into Cluster 1 (diameter 0.04 <= 0.06). C cannot join because dist(A, C) = 0.08 > 0.06.
        # So exactly 2 fused points must be formed!
        assert res.total_fused_points == 2
        assert res.fused_points[0].total_observation_count == 2
        assert res.fused_points[1].total_observation_count == 1

    def test_adversarial_distinct_surfaces_within_same_voxel_do_not_merge(self):
        """Test: Two distinct surfaces inside the same voxel bucket that exceed spatial threshold do NOT merge."""
        # Voxel resolution is 0.10. Both points fall into voxel (0, 0, 100).
        # Point 1 is at [0.01, 0.01, 10.01].
        # Point 2 is at [0.08, 0.08, 10.08].
        # Euclidean distance = sqrt(0.07^2 + 0.07^2 + 0.07^2) = 0.1212 > spatial_thresh (0.05).
        cfg = DenseFusionConfig(
            voxel_grid_resolution=0.10,
            spatial_distance_threshold=0.05,
            min_distinct_view_support=1,
            single_view_policy=SingleViewRetentionPolicy.RETAIN_AS_OBSERVED,
        )

        obs1 = DensePointObservation(world_point=np.array([0.01, 0.01, 10.01], dtype=np.float64), reference_frame_id="f1", pixel_coord=(0, 0), depth=10.01, confidence=0.8)
        obs2 = DensePointObservation(world_point=np.array([0.08, 0.08, 10.08], dtype=np.float64), reference_frame_id="f2", pixel_coord=(0, 0), depth=10.08, confidence=0.8)

        engine = DensePointFusionEngine(cfg)
        res = engine.fuse_observations([obs1, obs2])

        # Even though both are in voxel (0, 0, 100), they MUST form 2 separate points!
        assert res.total_fused_points == 2

    def test_mvs_contract_interface_compatibility(self, base_config: DenseFusionConfig):
        """Test: DensePointFusionEngine conforms to abstract DensePointFusion.fuse() interface."""
        obs1 = DensePointObservation(world_point=np.array([1, 1, 5], dtype=np.float64), reference_frame_id="f1", pixel_coord=(0, 0), depth=5.0, confidence=0.8)
        obs2 = DensePointObservation(world_point=np.array([1.01, 1, 5], dtype=np.float64), reference_frame_id="f2", pixel_coord=(0, 0), depth=5.0, confidence=0.8)

        engine = DensePointFusionEngine(base_config)
        cloud = engine.fuse([obs1, obs2])

        assert isinstance(cloud, DensePointCloud)
        assert cloud.total_fused_points == 1
        assert cloud.depth_unit == DepthUnit.RECONSTRUCTION_UNITS
        assert cloud.is_metric_scale is False
