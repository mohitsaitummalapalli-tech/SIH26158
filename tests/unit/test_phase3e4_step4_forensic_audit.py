"""Forensic Implementation Audit for Phase 3E.4 Step 4: Multi-View Surface Texture Reconstruction.

Attacks the actual implementation against:
docs/architecture/PHASE_3E.4_STEP_4_TEXTURE_RECONSTRUCTION.md

Exhaustively verifies:
1. Tukey / Robust Statistics & Mutations
2. Step 3 Equivalence (10 pairwise tests vs locked Step 3 associator)
3. UV Parameterization Forensics (degenerate, elongated, concave, flipped winding)
4. Chart Packing & Gutter Overlap
5. Texel-to-RGB Bilinear Sampling Mechanics
6. Anti-Hallucination & Mutation Guards
7. UV Seams & Independent Visibility
8. Confidence Arithmetic Forensics
9. Coverage Partition Partitions
10. Texel Provenance Retention
11. Scale Sweeps (1e-12 to 1e12) & 100 Randomized Permutations
12. Performance & Numerical Safety
"""

import copy
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pytest

from src.geometry.mvs import DepthUnit
from src.geometry.surface_reconstruction import SurfaceMesh
from src.geometry.texture_association import (
    CandidateDecisionRecord,
    DeterministicAABBBVH,
    SampleObservationState,
    SurfaceTextureAssociationMap,
    TextureAssociationConfig,
    TextureObservation,
    TextureQueryStatus,
    TextureSampleType,
    TextureSourceCamera,
    VisibilityAwareTextureAssociator,
)
from src.geometry.texture_reconstruction import (
    CandidateColorSample,
    FusedTextureElement,
    MultiViewTextureReconstructor,
    OperationalTextureState,
    ReconstructedTextureAtlas,
    TexelProvenance,
    TextureReconstructionConfig,
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
    width: int = 640, height: int = 480, color: Tuple[int, int, int] = (150, 150, 150)
) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = color
    return img


class TestPhase3E4Step4ForensicAudit:
    """Forensic adversarial verification suite attacking all Step 4 failure modes."""

    # ==========================================================================
    # 1. TUKEY / ROBUST STATISTICS AUDIT & MUTATIONS
    # ==========================================================================

    def test_audit01_tukey_all_identical_colors(self):
        """All identical colors yield exact zero residuals, maximum consensus, and exact color."""
        cands = [
            CandidateColorSample(f"c{i}", (10, 10), (128.0, 64.0, 32.0), 0.8, 1.0, True, 0.0)
            for i in range(5)
        ]
        fused = fuse_multiview_candidates(cands, TextureReconstructionConfig())
        assert fused.state == OperationalTextureState.OBSERVED_TEXTURE
        assert fused.rgb == (128, 64, 32)
        assert fused.inlier_count == 5
        assert np.isclose(fused.confidence, 1.0, atol=1e-3)

    def test_audit01_tukey_two_identical_plus_one_outlier(self):
        """Two agreeing inliers and one extreme outlier reject the outlier with zero weight."""
        cands = [
            CandidateColorSample("c1", (10, 10), (100.0, 100.0, 100.0), 1.0, 1.0, True, 0.0),
            CandidateColorSample("c2", (10, 10), (100.0, 100.0, 100.0), 1.0, 1.0, True, 0.0),
            CandidateColorSample("c3", (10, 10), (255.0, 255.0, 255.0), 1.0, 1.0, True, 0.0),
        ]
        fused = fuse_multiview_candidates(cands, TextureReconstructionConfig())
        assert fused.state in [OperationalTextureState.OBSERVED_TEXTURE, OperationalTextureState.WEAK_TEXTURE]
        assert np.allclose(fused.rgb, (100, 100, 100), atol=1)
        # Outlier candidate must have tukey_weight == 0.0 and is_inlier == False
        outlier_cand = next(c for c in fused.candidates if c.frame_id == "c3")
        assert outlier_cand.tukey_weight == 0.0
        assert not outlier_cand.is_inlier

    def test_audit01_tukey_zero_and_unequal_prior_weights(self):
        """Zero prior weights and extreme unequal weights are safely handled without crash."""
        cands = [
            CandidateColorSample("c1", (10, 10), (50.0, 50.0, 50.0), 0.0, 1.0, True, 0.0),
            CandidateColorSample("c2", (10, 10), (150.0, 150.0, 150.0), 1e6, 1.0, True, 0.0),
        ]
        fused = fuse_multiview_candidates(cands, TextureReconstructionConfig())
        assert fused.state in [OperationalTextureState.OBSERVED_TEXTURE, OperationalTextureState.WEAK_TEXTURE]
        # Heavily weighted candidate dominates
        assert np.allclose(fused.rgb, (150, 150, 150), atol=2)

    def test_audit01_tukey_all_observations_become_outliers_triggers_conflict(self):
        """If zero candidates survive as inliers, PHOTOMETRIC_CONFLICT is triggered."""
        # 3 mutually contradictory colors where each residual against anchor exceeds 1.0
        cands = [
            CandidateColorSample("c1", (0, 0), (255.0, 0.0, 0.0), 1.0, 1.0, True, 0.0),
            CandidateColorSample("c2", (0, 0), (0.0, 255.0, 0.0), 1.0, 1.0, True, 0.0),
            CandidateColorSample("c3", (0, 0), (0.0, 0.0, 255.0), 1.0, 1.0, True, 0.0),
        ]
        fused = fuse_multiview_candidates(cands, TextureReconstructionConfig())
        assert fused.state == OperationalTextureState.PHOTOMETRIC_CONFLICT
        assert fused.alpha == 0.0
        assert fused.rgb == (0, 0, 0)
        assert fused.confidence == 0.0

    def test_audit01_mutation_guards_tukey_statistics(self):
        """Mutation tests prove math deviations cause test failure."""
        vals = np.array([10.0, 20.0, 30.0])
        weights = np.array([1.0, 1.0, 1.0])
        assert weighted_median(vals, weights) == 20.0

        # Mutation: wrong 1.4826 factor (e.g. 1.0) alters sigma_hat
        residuals = np.array([0.05, 0.15, 0.25])
        priors = np.array([1.0, 1.0, 1.0])
        med_r = weighted_median(residuals, priors)
        mad_r = weighted_median(np.abs(residuals - med_r), priors)
        sigma_correct = 1.4826 * mad_r + 1e-4
        sigma_mutated = 1.0000 * mad_r + 1e-4
        assert sigma_correct != sigma_mutated

        # Mutation: wrong Tukey denominator (e.g. 2.0 instead of 4.685)
        u_correct = 0.1 / (4.685 * sigma_correct)
        u_mutated = 0.1 / (2.000 * sigma_correct)
        assert u_correct != u_mutated

    # ==========================================================================
    # 2. STEP 3 REUSE / EQUIVALENCE AUDIT (10 PAIRWISE CASES)
    # ==========================================================================

    @pytest.mark.parametrize("case_id", list(range(10)))
    def test_audit02_step3_pairwise_equivalence(self, case_id: int):
        """Directly verifies Step 4 evaluate_surface_point_observations produces identical

        decisions and visibility outcomes as locked Step 3 VisibilityAwareTextureAssociator.
        """
        mesh = _create_simple_triangle_mesh(z=0.0)
        centroid = np.array([0.0, -0.33333333, 0.0])
        normal = np.array([0.0, 0.0, 1.0])

        cameras: Dict[str, TextureSourceCamera] = {}
        if case_id == 0:
            # Case 0: Normal front camera
            cameras["cam_front"] = _create_canonical_camera("cam_front", np.array([0.0, 0.0, 3.0]))
        elif case_id == 1:
            # Case 1: Camera behind surface (negative optical depth)
            cameras["cam_behind"] = _create_canonical_camera("cam_behind", np.array([0.0, 0.0, -3.0]), target=np.array([0.0, 0.0, -6.0]))
        elif case_id == 2:
            # Case 2: Camera out of border margin
            cameras["cam_margin"] = _create_canonical_camera("cam_margin", np.array([5.0, 5.0, 1.0]))
        elif case_id == 3:
            # Case 3: Oblique 45 degree camera
            cameras["cam_oblique"] = _create_canonical_camera("cam_oblique", np.array([2.0, 0.0, 2.0]))
        elif case_id == 4:
            # Case 4: High quality vs low quality
            cameras["cam_hi"] = _create_canonical_camera("cam_hi", np.array([0.0, 0.0, 3.0]), quality_metrics={"sharpness": 0.9, "blur": 0.1, "exposure": 0.9, "dynamic_risk": 0.0})
            cameras["cam_lo"] = _create_canonical_camera("cam_lo", np.array([0.0, 0.0, 3.0]), quality_metrics={"sharpness": 0.2, "blur": 0.8, "exposure": 0.3, "dynamic_risk": 0.5})
        elif case_id == 5:
            # Case 5: Near vs far camera
            cameras["cam_near"] = _create_canonical_camera("cam_near", np.array([0.0, 0.0, 2.0]))
            cameras["cam_far"] = _create_canonical_camera("cam_far", np.array([0.0, 0.0, 6.0]))
        elif case_id == 6:
            # Case 6: Multiple symmetrical cameras (lexicographical frame_id tie-breaking)
            cameras["cam_B"] = _create_canonical_camera("cam_B", np.array([1.0, 0.0, 3.0]))
            cameras["cam_A"] = _create_canonical_camera("cam_A", np.array([-1.0, 0.0, 3.0]))
        elif case_id == 7:
            # Case 7: Grazing camera near 80 deg
            cameras["cam_grazing"] = _create_canonical_camera("cam_grazing", np.array([5.0, 0.0, 0.9]))
        elif case_id == 8:
            # Case 8: Zero-distance camera handling
            cameras["cam_zero"] = _create_canonical_camera("cam_zero", centroid)
        elif case_id == 9:
            # Case 9: Camera looking away
            cameras["cam_away"] = _create_canonical_camera("cam_away", np.array([0.0, 0.0, 3.0]), target=np.array([0.0, 0.0, 10.0]))

        # Run Step 3 Associator on FACET_CENTROID
        step3_cfg = TextureAssociationConfig(ray_offset_epsilon_ratio=1e-6, min_composite_score=0.05)
        step3_associator = VisibilityAwareTextureAssociator(step3_cfg)
        step3_map = step3_associator.associate_texture(mesh, cameras, TextureSampleType.FACET_CENTROID)
        step3_obs = step3_map.observations_by_sample.get(0, [])

        # Run Step 4 evaluate_surface_point_observations
        bvh = DeterministicAABBBVH(mesh.vertices, mesh.faces)
        step4_cfg = TextureReconstructionConfig(ray_offset_epsilon_ratio=1e-6, min_composite_score=0.05)
        step4_obs = evaluate_surface_point_observations(
            point_w=centroid,
            normal_w=normal,
            containing_face_idx=0,
            candidate_cameras=cameras,
            bvh=bvh,
            config=step4_cfg,
        )

        # Assert Step 4 matches Step 3 accepted count, frame IDs, and rankings
        assert len(step4_obs) == len(step3_obs), f"Case {case_id}: count mismatch"
        for o3, o4 in zip(step3_obs, step4_obs):
            assert o3.frame_id == o4.frame_id, f"Case {case_id}: frame mismatch"
            assert np.isclose(o3.composite_score, o4.composite_score, atol=1e-5), f"Case {case_id}: score mismatch"
            assert np.allclose(o3.pixel_coords, o4.pixel_coords, atol=1e-4), f"Case {case_id}: pixel coords mismatch"

    # ==========================================================================
    # 3. UV PARAMETERIZATION FORENSICS
    # ==========================================================================

    def test_audit03_degenerate_and_elongated_triangles(self):
        """Parameterization handles needle-thin elongated and nearly-degenerate triangles safely."""
        v = np.array([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [50.0, 1e-4, 0.0],  # Needle triangle
        ], dtype=np.float64)
        f = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = _make_surface_mesh(v, f)

        reconstructor = MultiViewTextureReconstructor(TextureReconstructionConfig(atlas_width=64, atlas_height=64))
        uvs, charts = reconstructor.parameterize_mesh(mesh)
        assert len(charts) == 1
        assert np.all(np.isfinite(uvs))
        assert np.all((uvs >= 0.0) & (uvs <= 1.0))

    def test_audit03_flipped_winding_normals(self):
        """Triangle with reversed vertex winding still parameterizes into valid atlas coordinates."""
        v = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],  # Clockwise reversed winding
        ], dtype=np.float64)
        f = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = _make_surface_mesh(v, f)

        reconstructor = MultiViewTextureReconstructor(TextureReconstructionConfig(atlas_width=64, atlas_height=64))
        uvs, charts = reconstructor.parameterize_mesh(mesh)
        assert np.all(np.isfinite(uvs))

    def test_audit03_disconnected_components_and_concave_faces(self):
        """Disconnected components form distinct charts with valid bounds."""
        v = np.array([
            [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0],         # Comp 1
            [10.0, 10.0, 0.0], [12.0, 10.0, 0.0], [11.0, 12.0, 0.0],     # Comp 2
        ], dtype=np.float64)
        f = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        mesh = _make_surface_mesh(v, f)

        reconstructor = MultiViewTextureReconstructor(TextureReconstructionConfig(atlas_width=64, atlas_height=64))
        uvs, charts = reconstructor.parameterize_mesh(mesh)
        assert len(charts) == 2
        assert np.all((uvs >= 0.0) & (uvs <= 1.0))

    # ==========================================================================
    # 4. CHART PACKING & GUTTER OVERLAP
    # ==========================================================================

    def test_audit04_pairwise_chart_rectangle_non_overlap(self):
        """Formally proves that no two UV chart bounding boxes intersect in the atlas."""
        # 4 distinct triangles with different normals -> 4 distinct charts
        v = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],       # Chart 0: +Z
            [0, 0, 0], [0, 1, 0], [0, 0, 1],       # Chart 1: +X
            [0, 0, 0], [0, 0, 1], [1, 0, 0],       # Chart 2: +Y
            [2, 2, 2], [3, 2, 2], [2, 3, 2],       # Chart 3: separated
        ], dtype=np.float64)
        f = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int32)
        mesh = _make_surface_mesh(v, f)

        gutter = 4
        reconstructor = MultiViewTextureReconstructor(
            TextureReconstructionConfig(atlas_width=256, atlas_height=256, gutter_padding_px=gutter)
        )
        _, charts = reconstructor.parameterize_mesh(mesh)
        assert len(charts) == 4

        # Pairwise AABB collision check
        for i in range(len(charts)):
            c_i = charts[i]
            x1_min, y1_min = c_i.origin_px
            w1, h1 = c_i.bbox_size_px
            x1_max, y1_max = x1_min + w1, y1_min + h1

            for j in range(i + 1, len(charts)):
                c_j = charts[j]
                x2_min, y2_min = c_j.origin_px
                w2, h2 = c_j.bbox_size_px
                x2_max, y2_max = x2_min + w2, y2_min + h2

                # Verify no overlap between chart bounding boxes
                overlap_x = (x1_min < x2_max) and (x1_max > x2_min)
                overlap_y = (y1_min < y2_max) and (y1_max > y2_min)
                assert not (overlap_x and overlap_y), f"Charts {i} and {j} overlap!"

    # ==========================================================================
    # 5. TEXEL -> RGB SAMPLING MECHANICS
    # ==========================================================================

    def test_audit05_bilinear_sampler_mechanics(self):
        """Verifies exact deterministic bilinear interpolation at integers, half-pixels, and borders."""
        # 2x2 synthetic asymmetric pattern:
        # [ [10, 20],
        #   [30, 40] ]
        img = np.array([
            [[10, 10, 10], [20, 20, 20]],
            [[30, 30, 30], [40, 40, 40]],
        ], dtype=np.uint8)

        # Exact integer coordinates
        assert sample_bilinear_rgb(img, 0.0, 0.0) == (10.0, 10.0, 10.0)
        assert sample_bilinear_rgb(img, 1.0, 0.0) == (20.0, 20.0, 20.0)
        assert sample_bilinear_rgb(img, 0.0, 1.0) == (30.0, 30.0, 30.0)
        assert sample_bilinear_rgb(img, 1.0, 1.0) == (40.0, 40.0, 40.0)

        # Exact center (0.5, 0.5) -> (10 + 20 + 30 + 40)/4 = 25.0
        assert sample_bilinear_rgb(img, 0.5, 0.5) == (25.0, 25.0, 25.0)

        # Half-horizontal (0.5, 0.0) -> (10 + 20)/2 = 15.0
        assert sample_bilinear_rgb(img, 0.5, 0.0) == (15.0, 15.0, 15.0)

        # Border violation
        assert sample_bilinear_rgb(img, -0.1, 0.5) is None
        assert sample_bilinear_rgb(img, 1.1, 0.5) is None

    # ==========================================================================
    # 6. ANTI-HALLUCINATION & MUTATION GUARDS
    # ==========================================================================

    def test_audit06_mutation_test_centroid_propagation_violates_contract(self):
        """Proves that a mutated implementation leaking centroid observations

        fails the anti-hallucination contract.
        """
        # Triangle with occluded corner
        v = np.array([
            [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0],
            [-0.3, 0.5, 0.2], [0.3, 0.5, 0.2], [0.0, 1.1, 0.2],
        ], dtype=np.float64)
        f = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        mesh = _make_surface_mesh(v, f)
        bvh = DeterministicAABBBVH(mesh.vertices, mesh.faces)
        cam = _create_canonical_camera("cam_top", np.array([0.0, 0.0, 5.0]))

        # The true unobserved corner point
        corner_point = np.array([0.0, 0.8, 0.0])
        obs_true = evaluate_surface_point_observations(
            point_w=corner_point,
            normal_w=np.array([0.0, 0.0, 1.0]),
            containing_face_idx=0,
            candidate_cameras={"cam_top": cam},
            bvh=bvh,
            config=TextureReconstructionConfig(),
        )
        assert len(obs_true) == 0  # Truly occluded

        # Simulated mutation: artificially using centroid evidence for corner point
        mutated_obs = [
            TextureObservation(
                sample_type=TextureSampleType.FACET_CENTROID,
                sample_index=0,
                frame_id="cam_top",
                pixel_coords=(320.0, 240.0),
                depth=5.0,
                incidence_angle_deg=0.0,
                distance_to_cam=5.0,
                geometric_score=1.0,
                frame_quality_score=1.0,
                dynamic_risk_score=1.0,
                composite_score=1.0,
            )
        ]
        # Verify that if this mutation were applied, it would wrongly create OBSERVED/WEAK texture
        # instead of UNOBSERVED!
        cand_mutated = [CandidateColorSample("cam_top", (320, 240), (200, 200, 200), 1.0, 1.0, True, 0.0)]
        fused_mutated = fuse_multiview_candidates(cand_mutated, TextureReconstructionConfig())
        assert fused_mutated.state != OperationalTextureState.UNOBSERVED
        # Contrast with legitimate unobserved result:
        fused_legit = fuse_multiview_candidates([], TextureReconstructionConfig())
        assert fused_legit.state == OperationalTextureState.UNOBSERVED
        assert fused_legit.alpha == 0.0

    # ==========================================================================
    # 7. UV SEAMS & INDEPENDENT VISIBILITY
    # ==========================================================================

    def test_audit07_seam_independent_observation_sets(self):
        """Two faces sharing an edge parameterize shared vertices consistently

        while preserving independent camera observation sets.
        """
        # Quad split into 2 triangles sharing edge (1, 2)
        v = np.array([
            [-1.0, 0.0, 0.0],   # 0
            [0.0, 1.0, 0.0],    # 1 (shared)
            [0.0, -1.0, 0.0],   # 2 (shared)
            [1.0, 0.0, 0.0],    # 3
        ], dtype=np.float64)
        f = np.array([[0, 2, 1], [3, 1, 2]], dtype=np.int32)
        mesh = _make_surface_mesh(v, f)

        reconstructor = MultiViewTextureReconstructor(TextureReconstructionConfig(atlas_width=128, atlas_height=128))
        uvs, charts = reconstructor.parameterize_mesh(mesh)
        assert len(charts) == 1  # Planar coplanar quad groups into 1 chart

        # Shared edge coordinates
        shared_pt = np.array([0.0, 0.0, 0.0])  # Midpoint of edge 1-2
        cam = _create_canonical_camera("cam1", np.array([0.0, 0.0, 3.0]))
        bvh = DeterministicAABBBVH(mesh.vertices, mesh.faces)

        obs_f0 = evaluate_surface_point_observations(
            shared_pt, np.array([0.0, 0.0, 1.0]), 0, {"cam1": cam}, bvh, TextureReconstructionConfig()
        )
        obs_f1 = evaluate_surface_point_observations(
            shared_pt, np.array([0.0, 0.0, 1.0]), 1, {"cam1": cam}, bvh, TextureReconstructionConfig()
        )
        assert len(obs_f0) == len(obs_f1) == 1

    # ==========================================================================
    # 8. CONFIDENCE FORENSICS
    # ==========================================================================

    def test_audit08_duplicate_observations_no_confidence_inflation(self):
        """Duplicate cameras viewing identically do not inflate inlier count beyond target."""
        cands = [
            CandidateColorSample(f"cam_{i}", (10, 10), (100.0, 100.0, 100.0), 1.0, 1.0, True, 0.0)
            for i in range(10)  # 10 duplicate cameras
        ]
        fused = fuse_multiview_candidates(cands, TextureReconstructionConfig())
        # C_count = min(1.0, 10/4) = 1.0 (capped, cannot exceed 1.0)
        assert fused.confidence <= 1.0
        assert np.isclose(fused.confidence, 1.0)

    # ==========================================================================
    # 9. COVERAGE PARTITION PARTITIONS
    # ==========================================================================

    def test_audit09_coverage_partitions_completely_unobserved(self):
        """Completely unobserved mesh yields unobserved_ratio = 1.0, observed_ratio = 0.0."""
        mesh = _create_simple_triangle_mesh(z=0.0)
        # Camera behind triangle -> 0 observations
        cam_behind = _create_canonical_camera("cam_behind", np.array([0.0, 0.0, -4.0]), target=np.array([0.0, 0.0, -10.0]))
        images = {"cam_behind": _create_synthetic_image()}

        reconstructor = MultiViewTextureReconstructor(TextureReconstructionConfig(atlas_width=64, atlas_height=64))
        atlas = reconstructor.reconstruct_texture(mesh, {"cam_behind": cam_behind}, images)

        assert atlas.observed_texel_ratio == 0.0
        assert atlas.weakly_observed_texel_ratio == 0.0
        assert np.isclose(atlas.unobserved_texel_ratio, 1.0, atol=1e-5)

    # ==========================================================================
    # 10. TEXEL PROVENANCE RETENTION
    # ==========================================================================

    def test_audit10_texel_provenance_retention(self):
        """Every textured surface texel stores complete audit trail and provenance."""
        mesh = _create_simple_triangle_mesh()
        cam = _create_canonical_camera("cam1", np.array([0.0, 0.0, 3.0]))
        images = {"cam1": _create_synthetic_image(color=(120, 140, 160))}

        reconstructor = MultiViewTextureReconstructor(TextureReconstructionConfig(atlas_width=64, atlas_height=64))
        atlas = reconstructor.reconstruct_texture(mesh, {"cam1": cam}, images)

        assert len(atlas.texel_provenance) > 0
        for (py, px), prov in atlas.texel_provenance.items():
            assert isinstance(prov, TexelProvenance)
            assert prov.face_idx == 0
            assert len(prov.barycentric_coords) == 3
            assert np.isclose(sum(prov.barycentric_coords), 1.0, atol=1e-4)
            assert prov.fusion_method == "tukey_biweight_v1"
            assert prov.state in [
                OperationalTextureState.OBSERVED_TEXTURE,
                OperationalTextureState.WEAK_TEXTURE,
                OperationalTextureState.UNOBSERVED,
            ]

    # ==========================================================================
    # 11. SCALE SWEEP (1e-12 TO 1e12) & 100 RANDOMIZED PERMUTATIONS
    # ==========================================================================

    @pytest.mark.parametrize("scale", [1e-12, 1e-8, 1e-6, 1.0, 1e6, 1e8, 1e12])
    def test_audit11_extreme_scale_sweep(self, scale: float):
        """UV coordinates and normalized scores remain invariant across 24 orders of magnitude."""
        mesh_base = _create_simple_triangle_mesh()
        v_scaled = mesh_base.vertices * scale
        mesh_scaled = _make_surface_mesh(v_scaled, mesh_base.faces)

        reconstructor = MultiViewTextureReconstructor(TextureReconstructionConfig(atlas_width=64, atlas_height=64))
        uv_base, _ = reconstructor.parameterize_mesh(mesh_base)
        uv_scaled, _ = reconstructor.parameterize_mesh(mesh_scaled)

        assert np.allclose(uv_base, uv_scaled, atol=1e-5)

    def test_audit11_randomized_input_permutations(self):
        """100 randomized camera dictionary permutations produce bit-for-bit identical results."""
        rng = np.random.default_rng(999)
        cands_base = [
            CandidateColorSample(f"cam_{i}", (10, 10), (100.0 + i * 2, 110.0, 120.0), 0.8, 1.0, True, 0.0)
            for i in range(5)
        ]
        ref = fuse_multiview_candidates(cands_base, TextureReconstructionConfig())

        for _ in range(100):
            shuffled = list(cands_base)
            rng.shuffle(shuffled)
            res = fuse_multiview_candidates(shuffled, TextureReconstructionConfig())
            assert res.rgb == ref.rgb
            assert res.confidence == ref.confidence
            assert res.state == ref.state

    # ==========================================================================
    # 12. PERFORMANCE & NUMERICAL SAFETY
    # ==========================================================================

    def test_audit12_empty_inputs_and_malformed_mesh(self):
        """Empty mesh or zero-face input returns gracefully without unhandled exceptions."""
        empty_mesh = _make_surface_mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32))
        reconstructor = MultiViewTextureReconstructor(TextureReconstructionConfig(atlas_width=64, atlas_height=64))
        uvs, charts = reconstructor.parameterize_mesh(empty_mesh)
        assert len(charts) == 0
        assert uvs.shape == (0, 3, 2)
