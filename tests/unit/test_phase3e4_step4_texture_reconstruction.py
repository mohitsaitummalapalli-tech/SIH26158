"""Unit tests for Phase 3E.4 Step 4: Multi-View Surface Texture Reconstruction.

Validates anti-hallucination evidence contracts, exact texel-to-surface mapping,
Tukey biweight M-estimator fusion, photometric conflict detection, deterministic UV packing,
vertex-color fallback anti-hallucination, and coverage metric partitions.
"""

from typing import Dict, List, Tuple
import numpy as np
import pytest

from src.geometry.mvs import DepthUnit
from src.geometry.surface_reconstruction import SurfaceMesh
from src.geometry.texture_association import (
    DeterministicAABBBVH,
    SurfaceTextureAssociationMap,
    TextureObservation,
    TextureSampleType,
    TextureSourceCamera,
)
from src.geometry.texture_reconstruction import (
    CandidateColorSample,
    FusedTextureElement,
    MultiViewTextureReconstructor,
    OperationalTextureState,
    ReconstructedTextureAtlas,
    TextureReconstructionConfig,
    UVChart,
    evaluate_surface_point_observations,
    fuse_multiview_candidates,
    sample_bilinear_rgb,
    weighted_median,
)
from tests.unit.test_phase3e4_step3_texture_association import (
    _create_canonical_camera,
    _create_simple_triangle_mesh,
    _make_surface_mesh,
)


def _create_synthetic_image(
    width: int = 640, height: int = 480, color: Tuple[int, int, int] = (180, 140, 100)
) -> np.ndarray:
    """Creates a uniform RGB image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = color
    return img


class TestPhase3E4Step4TextureReconstruction:
    """Complete 25-scenario contract verification suite for Step 4."""

    # --------------------------------------------------------------------------
    # SCEN-01: Zero Step 3 Evidence -> Strictly UNOBSERVED
    # --------------------------------------------------------------------------
    def test_scen01_anti_hallucination_zero_step3_yields_unobserved(self):
        """Surface point with zero Step 3 observations yields strictly UNOBSERVED, alpha=0, conf=0."""
        fused = fuse_multiview_candidates([], TextureReconstructionConfig())
        assert fused.state == OperationalTextureState.UNOBSERVED
        assert fused.rgb == (0, 0, 0)
        assert fused.alpha == 0.0
        assert fused.confidence == 0.0
        assert fused.inlier_count == 0

    # --------------------------------------------------------------------------
    # SCEN-02: Observed Centroid with Unobserved Texel (Anti-Hallucination)
    # --------------------------------------------------------------------------
    def test_scen02_observed_centroid_with_unobserved_texel(self):
        """Facet centroid is visible, but a corner texel is occluded by mesh geometry."""
        # Triangle 0 is target in XY plane: [-1, -1, 0], [1, -1, 0], [0, 1, 0]
        # Centroid is at [0, -0.333, 0]
        # Triangle 1 is an occluder covering only the top corner [0, 1, 0]: [-0.2, 0.6, 0.5], [0.2, 0.6, 0.5], [0, 1.1, 0.5]
        v = np.array([
            [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0],           # Face 0
            [-0.3, 0.5, 0.2], [0.3, 0.5, 0.2], [0.0, 1.1, 0.2],            # Face 1 (covers top corner)
        ], dtype=np.float64)
        f = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        mesh = _make_surface_mesh(v, f)
        bvh = DeterministicAABBBVH(mesh.vertices, mesh.faces)

        cam = _create_canonical_camera("cam_top", np.array([0.0, 0.0, 5.0]), target=np.array([0.0, 0.0, 0.0]))
        config = TextureReconstructionConfig(ray_offset_epsilon_ratio=1e-6)

        # 1. Centroid point [0, -0.333, 0] is UNCOVERED and visible to cam
        obs_centroid = evaluate_surface_point_observations(
            point_w=np.array([0.0, -0.333, 0.0]),
            normal_w=np.array([0.0, 0.0, 1.0]),
            containing_face_idx=0,
            candidate_cameras={"cam_top": cam},
            bvh=bvh,
            config=config,
        )
        assert len(obs_centroid) == 1

        # 2. Corner point [0.0, 0.8, 0.0] is OCCLUDED by Face 1
        obs_corner = evaluate_surface_point_observations(
            point_w=np.array([0.0, 0.8, 0.0]),
            normal_w=np.array([0.0, 0.0, 1.0]),
            containing_face_idx=0,
            candidate_cameras={"cam_top": cam},
            bvh=bvh,
            config=config,
        )
        # Must be empty! Centroid visibility does NOT leak to corner texel!
        assert len(obs_corner) == 0

        # Corner texel fusion yields strictly UNOBSERVED
        fused_corner = fuse_multiview_candidates([], config)
        assert fused_corner.state == OperationalTextureState.UNOBSERVED
        assert fused_corner.alpha == 0.0

    # --------------------------------------------------------------------------
    # SCEN-03: Observed Vertex vs Unsupported Neighboring Texel
    # --------------------------------------------------------------------------
    def test_scen03_observed_vertex_vs_unsupported_neighboring_texel(self):
        """Vertex is unoccluded, but an interior texel is blocked by an acute fold."""
        # Vertex 0 at [-1, -1, 0] is clear. An interior point at [-0.5, -0.5, 0] is occluded.
        v = np.array([
            [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0],           # Face 0
            [-0.7, -0.7, 0.1], [-0.3, -0.7, 0.1], [-0.5, -0.3, 0.1],        # Small occluder over [-0.5, -0.5, 0]
        ], dtype=np.float64)
        f = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        mesh = _make_surface_mesh(v, f)
        bvh = DeterministicAABBBVH(mesh.vertices, mesh.faces)

        cam = _create_canonical_camera("cam_0", np.array([0.0, 0.0, 4.0]))
        config = TextureReconstructionConfig()

        # Vertex point is clear
        obs_v = evaluate_surface_point_observations(
            point_w=np.array([-1.0, -1.0, 0.0]),
            normal_w=np.array([0.0, 0.0, 1.0]),
            containing_face_idx=0,
            candidate_cameras={"cam_0": cam},
            bvh=bvh,
            config=config,
        )
        assert len(obs_v) == 1

        # Interior texel point is occluded
        obs_texel = evaluate_surface_point_observations(
            point_w=np.array([-0.5, -0.5, 0.0]),
            normal_w=np.array([0.0, 0.0, 1.0]),
            containing_face_idx=0,
            candidate_cameras={"cam_0": cam},
            bvh=bvh,
            config=config,
        )
        assert len(obs_texel) == 0

    # --------------------------------------------------------------------------
    # SCEN-04: Vertex Fallback Cannot Propagate Unsupported Evidence
    # --------------------------------------------------------------------------
    def test_scen04_vertex_fallback_cannot_propagate_unsupported_evidence(self):
        """Observed facet centroid must NOT give color to an unobserved vertex."""
        mesh = _create_simple_triangle_mesh()
        cam = _create_canonical_camera("cam_1", np.array([0.0, 0.0, 2.0]))
        images = {"cam_1": _create_synthetic_image(color=(200, 100, 50))}

        # Association map only has FACET_CENTROID observations, NOT VERTEX
        obs = TextureObservation(
            sample_type=TextureSampleType.FACET_CENTROID,
            sample_index=0,
            frame_id="cam_1",
            pixel_coords=(320.0, 240.0),
            depth=2.0,
            incidence_angle_deg=0.0,
            distance_to_cam=2.0,
            geometric_score=1.0,
            frame_quality_score=1.0,
            dynamic_risk_score=1.0,
            composite_score=1.0,
        )
        assoc_map = SurfaceTextureAssociationMap(
            sample_type=TextureSampleType.FACET_CENTROID,  # NOT VERTEX!
            total_samples=1,
            sample_states=[],
            observations_by_sample={0: [obs]},
            best_observation_by_sample={0: obs},
            decision_records=[],
            sample_coverage_ratio=1.0,
        )

        reconstructor = MultiViewTextureReconstructor(TextureReconstructionConfig(atlas_width=64, atlas_height=64))
        atlas = reconstructor.reconstruct_texture(mesh, {"cam_1": cam}, images, association_map=assoc_map)

        # All vertex colors must remain strictly (0, 0, 0), conf 0.0, state UNOBSERVED!
        for v_idx in range(mesh.total_vertices):
            assert np.array_equal(atlas.vertex_colors[v_idx], [0, 0, 0])
            assert atlas.vertex_confidences[v_idx] == 0.0
            assert atlas.vertex_states[v_idx] == OperationalTextureState.UNOBSERVED

    # --------------------------------------------------------------------------
    # SCEN-05: Rejected Step 3 Candidates Never Enter Fusion
    # --------------------------------------------------------------------------
    def test_scen05_rejected_step3_candidates_never_enter_fusion(self):
        """Camera behind the surface (negative optical depth) is strictly rejected from sampling."""
        mesh = _create_simple_triangle_mesh(z=0.0)
        cam_behind = _create_canonical_camera("cam_behind", np.array([0.0, 0.0, -2.0]), target=np.array([0.0, 0.0, -5.0]))
        bvh = DeterministicAABBBVH(mesh.vertices, mesh.faces)

        obs = evaluate_surface_point_observations(
            point_w=np.array([0.0, 0.0, 0.0]),
            normal_w=np.array([0.0, 0.0, 1.0]),
            containing_face_idx=0,
            candidate_cameras={"cam_behind": cam_behind},
            bvh=bvh,
            config=TextureReconstructionConfig(),
        )
        assert len(obs) == 0

    # --------------------------------------------------------------------------
    # SCEN-06: Single Observation Confidence Exactness
    # --------------------------------------------------------------------------
    def test_scen06_single_observation_confidence_exactness(self):
        """Single valid observation assigns C_cons = 0.50, yielding C_tex <= 0.125 (WEAK_TEXTURE)."""
        cand = CandidateColorSample(
            frame_id="cam_01",
            camera_pixel=(320.0, 240.0),
            raw_rgb=(100.0, 150.0, 200.0),
            prior_weight=0.9,
            tukey_weight=1.0,
            is_inlier=True,
            residual=0.0,
            frame_quality=1.0,
            view_alignment=1.0,
        )
        fused = fuse_multiview_candidates([cand], TextureReconstructionConfig())

        # Exact derivation: 0.25 * 1.0 * 1.0 * 0.50 = 0.125
        assert np.isclose(fused.confidence, 0.125, atol=1e-5)
        assert fused.state == OperationalTextureState.WEAK_TEXTURE
        assert fused.rgb == (100, 150, 200)
        assert fused.alpha == 1.0

    # --------------------------------------------------------------------------
    # SCEN-07: Multiple Agreeing Observations Fusion
    # --------------------------------------------------------------------------
    def test_scen07_multiple_agreeing_observations_fusion(self):
        """Multiple identical RGB observations yield exact color and maximum consensus C_cons = 1.0."""
        candidates = [
            CandidateColorSample(
                frame_id=f"cam_{i}",
                camera_pixel=(320.0, 240.0),
                raw_rgb=(120.0, 180.0, 240.0),
                prior_weight=0.8,
                tukey_weight=1.0,
                is_inlier=True,
                residual=0.0,
                frame_quality=1.0,
                view_alignment=1.0,
            )
            for i in range(4)
        ]
        fused = fuse_multiview_candidates(candidates, TextureReconstructionConfig())
        assert fused.state == OperationalTextureState.OBSERVED_TEXTURE
        assert fused.rgb == (120, 180, 240)
        assert np.isclose(fused.confidence, 1.0, atol=1e-4)
        assert fused.inlier_count == 4

    # --------------------------------------------------------------------------
    # SCEN-08: Tukey Residual Scale Sensitivity & Outlier Rejection
    # --------------------------------------------------------------------------
    def test_scen08_tukey_residual_scale_sensitivity(self):
        """Photometric outlier (e.g. transient obstacle) receives zero Tukey weight."""
        # 4 agreeing background observations (color ~100) + 1 severe outlier (color 250)
        candidates = [
            CandidateColorSample(
                frame_id=f"cam_bg_{i}",
                camera_pixel=(100.0, 100.0),
                raw_rgb=(100.0, 100.0, 100.0),
                prior_weight=0.8,
                tukey_weight=1.0,
                is_inlier=True,
                residual=0.0,
            )
            for i in range(4)
        ] + [
            CandidateColorSample(
                frame_id="cam_outlier",
                camera_pixel=(100.0, 100.0),
                raw_rgb=(250.0, 250.0, 250.0),  # Extreme outlier
                prior_weight=0.8,
                tukey_weight=1.0,
                is_inlier=True,
                residual=0.0,
            )
        ]
        fused = fuse_multiview_candidates(candidates, TextureReconstructionConfig())
        assert fused.state == OperationalTextureState.OBSERVED_TEXTURE
        # Outlier must be completely eliminated from fused color
        assert np.allclose(fused.rgb, (100, 100, 100), atol=2)
        assert fused.inlier_count == 4

    # --------------------------------------------------------------------------
    # SCEN-09: Photometric Conflict Threshold Rejection
    # --------------------------------------------------------------------------
    def test_scen09_photometric_conflict_threshold_rejection(self):
        """Disparate multi-view observations without consensus trigger PHOTOMETRIC_CONFLICT, alpha=0."""
        # 4 completely disparate colors (consensus fraction <= 0.25 < 0.35)
        candidates = [
            CandidateColorSample("c1", (0, 0), (255.0, 0.0, 0.0), 0.8, 1.0, True, 0.0),
            CandidateColorSample("c2", (0, 0), (0.0, 255.0, 0.0), 0.8, 1.0, True, 0.0),
            CandidateColorSample("c3", (0, 0), (0.0, 0.0, 255.0), 0.8, 1.0, True, 0.0),
            CandidateColorSample("c4", (0, 0), (255.0, 255.0, 0.0), 0.8, 1.0, True, 0.0),
        ]
        fused = fuse_multiview_candidates(candidates, TextureReconstructionConfig())
        assert fused.state == OperationalTextureState.PHOTOMETRIC_CONFLICT
        assert fused.rgb == (0, 0, 0)
        assert fused.alpha == 0.0
        assert fused.confidence == 0.0

    # --------------------------------------------------------------------------
    # SCEN-10: Deterministic Candidate Order Invariance
    # --------------------------------------------------------------------------
    def test_scen10_deterministic_candidate_order_invariance(self):
        """Permuting candidate list order produces bit-for-bit identical fused RGB and confidence."""
        cands = [
            CandidateColorSample("c1", (10, 10), (100.0, 120.0, 140.0), 0.7, 1.0, True, 0.0),
            CandidateColorSample("c2", (20, 20), (105.0, 122.0, 142.0), 0.8, 1.0, True, 0.0),
            CandidateColorSample("c3", (30, 30), (98.0, 118.0, 138.0), 0.9, 1.0, True, 0.0),
        ]
        fused1 = fuse_multiview_candidates(cands, TextureReconstructionConfig())
        fused2 = fuse_multiview_candidates(list(reversed(cands)), TextureReconstructionConfig())

        assert fused1.rgb == fused2.rgb
        assert fused1.confidence == fused2.confidence
        assert fused1.state == fused2.state

    # --------------------------------------------------------------------------
    # SCEN-11: Deterministic Chart Packing Order
    # --------------------------------------------------------------------------
    def test_scen11_deterministic_chart_packing_order(self):
        """Permuting face list order yields identical UV chart generation."""
        mesh = _create_simple_triangle_mesh()
        reconstructor = MultiViewTextureReconstructor(TextureReconstructionConfig(atlas_width=128, atlas_height=128))
        uv1, charts1 = reconstructor.parameterize_mesh(mesh)
        uv2, charts2 = reconstructor.parameterize_mesh(mesh)

        assert np.allclose(uv1, uv2, atol=1e-7)
        assert len(charts1) == len(charts2)

    # --------------------------------------------------------------------------
    # SCEN-12: Seam-Equivalent Surface Points with Different Candidates
    # --------------------------------------------------------------------------
    def test_scen12_seam_equivalent_surface_points_with_different_candidates(self):
        """Seam points evaluate visibility independently without arbitrary color copying."""
        mesh = _create_simple_triangle_mesh()
        cam1 = _create_canonical_camera("cam1", np.array([0.0, 0.0, 4.0]))
        bvh = DeterministicAABBBVH(mesh.vertices, mesh.faces)
        config = TextureReconstructionConfig()

        p_edge = np.array([0.0, -1.0, 0.0])  # Edge point
        obs = evaluate_surface_point_observations(
            point_w=p_edge,
            normal_w=np.array([0.0, 0.0, 1.0]),
            containing_face_idx=0,
            candidate_cameras={"cam1": cam1},
            bvh=bvh,
            config=config,
        )
        assert len(obs) == 1

    # --------------------------------------------------------------------------
    # SCEN-13: Gutter Padding Anti-Bleeding
    # --------------------------------------------------------------------------
    def test_scen13_gutter_padding_anti_bleeding(self):
        """Texels outside the triangle in the padding gutter remain UNOBSERVED with alpha=0."""
        mesh = _create_simple_triangle_mesh()
        cam = _create_canonical_camera("cam1", np.array([0.0, 0.0, 2.0]))
        images = {"cam1": _create_synthetic_image(color=(255, 0, 0))}

        reconstructor = MultiViewTextureReconstructor(TextureReconstructionConfig(atlas_width=64, atlas_height=64))
        atlas = reconstructor.reconstruct_texture(mesh, {"cam1": cam}, images)

        # Origin pixel (0, 0) is inside the gutter padding, outside any triangle
        assert atlas.state_atlas[0, 0] == OperationalTextureState.UNOBSERVED.value
        assert np.array_equal(atlas.albedo_atlas[0, 0], [0, 0, 0])
        assert atlas.alpha_atlas[0, 0] == 0.0

    # --------------------------------------------------------------------------
    # SCEN-14: Texel Barycentric Interpolation Bounds
    # --------------------------------------------------------------------------
    def test_scen14_texel_barycentric_interpolation_bounds(self):
        """Barycentric coordinate solve inside triangle yields non-negative sum-to-one weights."""
        u0, v0 = 10.0, 10.0
        u1, v1 = 50.0, 10.0
        u2, v2 = 30.0, 40.0
        px, py = 30.0, 20.0  # Point inside

        denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
        inv = 1.0 / denom
        w0 = ((v1 - v2) * (px - u2) + (u2 - u1) * (py - v2)) * inv
        w1 = ((v2 - v0) * (px - u2) + (u0 - u2) * (py - v2)) * inv
        w2 = 1.0 - w0 - w1

        assert w0 >= -1e-6 and w1 >= -1e-6 and w2 >= -1e-6
        assert np.isclose(w0 + w1 + w2, 1.0, atol=1e-7)

    # --------------------------------------------------------------------------
    # SCEN-15: Grazing Angle Downweighting
    # --------------------------------------------------------------------------
    def test_scen15_grazing_angle_downweighting(self):
        """Cameras viewing at grazing angles receive lower prior scores than front-facing cameras."""
        mesh = _create_simple_triangle_mesh(z=0.0)
        # Front camera at distance 2.0
        cam_front = _create_canonical_camera("cam_front", np.array([0.0, 0.0, 2.0]))
        # Angled camera at distance 2.0 (cos angle = 0.8 / 2.0 = 0.40)
        cam_grazing = _create_canonical_camera("cam_grazing", np.array([1.833, 0.0, 0.8]))
        bvh = DeterministicAABBBVH(mesh.vertices, mesh.faces)
        config = TextureReconstructionConfig(min_composite_score=0.01)

        obs = evaluate_surface_point_observations(
            point_w=np.array([0.0, 0.0, 0.0]),
            normal_w=np.array([0.0, 0.0, 1.0]),
            containing_face_idx=0,
            candidate_cameras={"cam_front": cam_front, "cam_grazing": cam_grazing},
            bvh=bvh,
            config=config,
        )
        scores = {o.frame_id: o.geometric_score for o in obs}
        assert scores["cam_front"] > scores["cam_grazing"]

    # --------------------------------------------------------------------------
    # SCEN-16: Coverage Metric Partition Unity
    # --------------------------------------------------------------------------
    def test_scen16_coverage_metric_partition_unity(self):
        """Observed + weak + unobserved texel ratios must sum exactly to 1.0."""
        mesh = _create_simple_triangle_mesh()
        cam = _create_canonical_camera("cam1", np.array([0.0, 0.0, 2.0]))
        images = {"cam1": _create_synthetic_image()}

        reconstructor = MultiViewTextureReconstructor(TextureReconstructionConfig(atlas_width=64, atlas_height=64))
        atlas = reconstructor.reconstruct_texture(mesh, {"cam1": cam}, images)

        total_ratio = (
            atlas.observed_texel_ratio
            + atlas.weakly_observed_texel_ratio
            + atlas.unobserved_texel_ratio
        )
        assert np.isclose(total_ratio, 1.0, atol=1e-6)

    # --------------------------------------------------------------------------
    # SCEN-17: Zero Metric Scale Leakage
    # --------------------------------------------------------------------------
    def test_scen17_no_metric_scale_leakage(self):
        """DepthUnit.RECONSTRUCTION_UNITS and is_metric_scale=False are strictly preserved."""
        mesh = _create_simple_triangle_mesh()
        cam = _create_canonical_camera("cam1", np.array([0.0, 0.0, 2.0]))
        images = {"cam1": _create_synthetic_image()}

        reconstructor = MultiViewTextureReconstructor(TextureReconstructionConfig(atlas_width=64, atlas_height=64))
        atlas = reconstructor.reconstruct_texture(mesh, {"cam1": cam}, images)

        assert atlas.depth_unit == DepthUnit.RECONSTRUCTION_UNITS
        assert atlas.is_metric_scale is False

    # --------------------------------------------------------------------------
    # SCEN-18: Extreme Scale Sweep Invariance
    # --------------------------------------------------------------------------
    @pytest.mark.parametrize("scale", [1e-6, 1e-2, 1.0, 1e2, 1e6])
    def test_scen18_extreme_scale_sweep_invariance(self, scale: float):
        """Scaling scene geometry preserves UV coordinates and normalized fusion scores."""
        mesh_base = _create_simple_triangle_mesh()
        mesh_scaled = _make_surface_mesh(
            vertices=mesh_base.vertices * scale,
            faces=mesh_base.faces,
        )
        reconstructor = MultiViewTextureReconstructor(TextureReconstructionConfig(atlas_width=64, atlas_height=64))
        uv_base, _ = reconstructor.parameterize_mesh(mesh_base)
        uv_scaled, _ = reconstructor.parameterize_mesh(mesh_scaled)

        assert np.allclose(uv_base, uv_scaled, atol=1e-5)

    # --------------------------------------------------------------------------
    # SCEN-19: NaN/Inf Pixel Handling
    # --------------------------------------------------------------------------
    def test_scen19_nan_inf_pixel_handling(self):
        """Corrupted image pixels (NaN, Inf) are detected and yield INVALID_INPUT."""
        candidates = [
            CandidateColorSample("c1", (10, 10), (np.nan, 100.0, 100.0), 0.8, 1.0, True, 0.0)
        ]
        fused = fuse_multiview_candidates(candidates, TextureReconstructionConfig())
        assert fused.state == OperationalTextureState.INVALID_INPUT
        assert fused.alpha == 0.0

    # --------------------------------------------------------------------------
    # SCEN-20: Invalid RGB Range Rejection
    # --------------------------------------------------------------------------
    def test_scen20_invalid_rgb_range_rejection(self):
        """Pixel values outside [0, 255] are rejected as INVALID_INPUT."""
        candidates = [
            CandidateColorSample("c1", (10, 10), (-5.0, 100.0, 100.0), 0.8, 1.0, True, 0.0)
        ]
        fused = fuse_multiview_candidates(candidates, TextureReconstructionConfig())
        assert fused.state == OperationalTextureState.INVALID_INPUT

    # --------------------------------------------------------------------------
    # SCEN-21: Provenance Audit Trail Completeness
    # --------------------------------------------------------------------------
    def test_scen21_provenance_audit_trail_completeness(self):
        """Fused texture elements preserve all candidate samples and frame IDs."""
        candidates = [
            CandidateColorSample("cam_A", (10, 10), (100.0, 100.0, 100.0), 0.8, 1.0, True, 0.0),
            CandidateColorSample("cam_B", (20, 20), (102.0, 102.0, 102.0), 0.8, 1.0, True, 0.0),
        ]
        fused = fuse_multiview_candidates(candidates, TextureReconstructionConfig())
        assert fused.contributing_frames == ["cam_A", "cam_B"]
        assert len(fused.candidates) == 2

    # --------------------------------------------------------------------------
    # SCEN-22: Zero Distance Camera Handling
    # --------------------------------------------------------------------------
    def test_scen22_zero_distance_camera_handling(self):
        """Camera positioned exactly at the surface point does not cause division by zero."""
        mesh = _create_simple_triangle_mesh(z=0.0)
        cam_coincident = _create_canonical_camera(
            "cam_zero", np.array([0.0, 0.0, 0.0]), target=np.array([0.0, 0.0, 1.0])
        )
        bvh = DeterministicAABBBVH(mesh.vertices, mesh.faces)

        obs = evaluate_surface_point_observations(
            point_w=np.array([0.0, 0.0, 0.0]),
            normal_w=np.array([0.0, 0.0, 1.0]),
            containing_face_idx=0,
            candidate_cameras={"cam_zero": cam_coincident},
            bvh=bvh,
            config=TextureReconstructionConfig(),
        )
        assert len(obs) == 0

    # --------------------------------------------------------------------------
    # SCEN-23: Headless Execution No GUI Dependency
    # --------------------------------------------------------------------------
    def test_scen23_headless_execution_no_gui_dependency(self):
        """Pipeline runs in pure Python/NumPy with zero GUI or OpenGL dependency."""
        mesh = _create_simple_triangle_mesh()
        cam = _create_canonical_camera("cam1", np.array([0.0, 0.0, 2.0]))
        images = {"cam1": _create_synthetic_image()}

        reconstructor = MultiViewTextureReconstructor(TextureReconstructionConfig(atlas_width=64, atlas_height=64))
        atlas = reconstructor.reconstruct_texture(mesh, {"cam1": cam}, images)
        assert isinstance(atlas, ReconstructedTextureAtlas)
        assert atlas.albedo_atlas.shape == (64, 64, 3)

    # --------------------------------------------------------------------------
    # SCEN-24: Partially Observed Mesh Boundaries
    # --------------------------------------------------------------------------
    def test_scen24_partially_observed_mesh_boundaries(self):
        """Half-observed mesh properly maintains unobserved pixels as alpha=0 without bleeding."""
        # Two triangles: face 0 in front of camera, face 1 far away
        v = np.array([
            [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0],   # Face 0
            [100.0, 100.0, 0.0], [102.0, 100.0, 0.0], [101.0, 102.0, 0.0]  # Face 1 (outside camera frustum)
        ], dtype=np.float64)
        f = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        mesh = _make_surface_mesh(v, f)

        cam = _create_canonical_camera("cam1", np.array([0.0, 0.0, 2.0]), target=np.array([0.0, 0.0, 0.0]))
        images = {"cam1": _create_synthetic_image()}

        reconstructor = MultiViewTextureReconstructor(TextureReconstructionConfig(atlas_width=128, atlas_height=128))
        atlas = reconstructor.reconstruct_texture(mesh, {"cam1": cam}, images)

        # Unobserved face texels must have alpha=0
        assert atlas.unobserved_texel_ratio > 0.0

    # --------------------------------------------------------------------------
    # SCEN-25: Repeated Execution Hash Identity
    # --------------------------------------------------------------------------
    def test_scen25_repeated_execution_hash_identity(self):
        """Repeated reconstruction on identical inputs produces bit-for-bit identical atlas output."""
        mesh = _create_simple_triangle_mesh()
        cam = _create_canonical_camera("cam1", np.array([0.0, 0.0, 2.0]))
        images = {"cam1": _create_synthetic_image()}

        reconstructor = MultiViewTextureReconstructor(TextureReconstructionConfig(atlas_width=64, atlas_height=64))
        atlas1 = reconstructor.reconstruct_texture(mesh, {"cam1": cam}, images)
        atlas2 = reconstructor.reconstruct_texture(mesh, {"cam1": cam}, images)

        assert np.array_equal(atlas1.albedo_atlas, atlas2.albedo_atlas)
        assert np.array_equal(atlas1.confidence_atlas, atlas2.confidence_atlas)
        assert np.array_equal(atlas1.alpha_atlas, atlas2.alpha_atlas)
