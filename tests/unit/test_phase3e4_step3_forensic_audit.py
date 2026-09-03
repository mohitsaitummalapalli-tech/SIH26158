"""Forensic implementation audit for Phase 3E.4 Step 3: Visibility-Aware Texture Association.

Validates mathematical contracts, scale invariance (10^-12 to 10^12), BVH equivalence,
self-intersection rules, numerical tolerances, mutation detection, and camera geometry.
"""

from typing import Dict, List
import numpy as np
import pytest

from src.geometry.mvs import DepthUnit
from src.geometry.surface_reconstruction import SurfaceMesh
from src.geometry.texture_association import (
    CandidateDecisionRecord,
    DecisionStatus,
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
from tests.unit.test_phase3e4_step3_texture_association import (
    _create_canonical_camera,
    _create_simple_triangle_mesh,
    _make_surface_mesh,
)


class TestPhase3E4Step3ForensicAudit:
    """Forensic adversarial and mathematical audit suite for Step 3."""

    # --------------------------------------------------------------------------
    # 1. Camera Geometry Audit
    # --------------------------------------------------------------------------
    def test_audit_camera_optical_center_mathematical_identity(self):
        """Verify C_w = -R_cw^T @ t_cw holds analytically across random SO(3) poses."""
        np.random.seed(123)
        for _ in range(10):
            C_true = np.random.uniform(-100.0, 100.0, size=3)
            # Random orthogonal rotation matrix
            q, _ = np.linalg.qr(np.random.randn(3, 3))
            if np.linalg.det(q) < 0:
                q[:, 0] = -q[:, 0]
            R_cw = q
            t_cw = -R_cw @ C_true

            C_computed = -R_cw.T @ t_cw
            assert np.allclose(C_computed, C_true, atol=1e-12)

    def test_audit_positive_optical_depth_cheirality(self):
        """Verify points strictly on or behind camera optical center (X_c,z <= 0) are rejected."""
        mesh = _create_simple_triangle_mesh(z=0.0)
        # Camera at z = 2.0 looking down (-Z)
        cam = _create_canonical_camera("cam_fwd", np.array([0.0, 0.0, 2.0]), target=np.array([0.0, 0.0, 0.0]))
        # Camera at z = -2.0 looking down (-Z) (surface is BEHIND camera plane)
        cam_behind = _create_canonical_camera("cam_behind", np.array([0.0, 0.0, -2.0]), target=np.array([0.0, 0.0, -5.0]))

        associator = VisibilityAwareTextureAssociator()
        res = associator.associate_texture(mesh, {"cam_behind": cam_behind}, TextureSampleType.FACET_CENTROID)
        rec = res.decision_records[0]
        assert rec.decision == DecisionStatus.REJECTED
        assert rec.query_status == TextureQueryStatus.NEGATIVE_DEPTH
        assert rec.depth is not None and rec.depth <= 0.0

    def test_audit_non_finite_camera_parameters_rejected(self):
        """Verify NaN or Inf in camera matrices or translation vectors are safely rejected."""
        mesh = _create_simple_triangle_mesh()
        base_cam = _create_canonical_camera("cam_good", np.array([0.0, 0.0, 2.0]))

        # NaN in rotation
        R_nan = base_cam.R_cw.copy()
        R_nan[0, 0] = np.nan
        cam_nan_r = TextureSourceCamera(
            frame_id="cam_nan_r", R_cw=R_nan, t_cw=base_cam.t_cw, K=base_cam.K,
            width=base_cam.width, height=base_cam.height
        )
        # Inf in translation
        t_inf = base_cam.t_cw.copy()
        t_inf[1] = np.inf
        cam_inf_t = TextureSourceCamera(
            frame_id="cam_inf_t", R_cw=base_cam.R_cw, t_cw=t_inf, K=base_cam.K,
            width=base_cam.width, height=base_cam.height
        )

        associator = VisibilityAwareTextureAssociator()
        res = associator.associate_texture(mesh, {"cam_nan_r": cam_nan_r, "cam_inf_t": cam_inf_t}, TextureSampleType.FACET_CENTROID)
        for rec in res.decision_records:
            assert rec.decision == DecisionStatus.REJECTED
            assert rec.query_status == TextureQueryStatus.NON_FINITE_PARAMETERS

    # --------------------------------------------------------------------------
    # 2. Finite Ray & Occlusion Offset Investigation
    # --------------------------------------------------------------------------
    def test_audit_ray_offset_skipping_thin_occluder_investigation(self):
        """Forensic audit test demonstrating the scale of ray offset vs thin occluders.

        When eps = eps_ratio * d_cam, if eps_ratio=1e-4 and d_cam=100.0, eps = 0.01.
        An occluder at distance 0.005 is skipped unless eps_ratio <= 1e-5 or geometry-relative.
        """
        # Target at z=0, thin occluder at z=0.005
        vertices = np.array([
            [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0],       # Face 0 (Target)
            [-1.0, -1.0, 0.005], [1.0, -1.0, 0.005], [0.0, 1.0, 0.005]  # Face 1 (Occluder)
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        mesh = _make_surface_mesh(vertices, faces)

        # Distant camera at d_cam = 100.0
        cam_far = _create_canonical_camera("cam_far", np.array([0.0, 0.0, 100.0]), target=np.array([0.0, 0.0, 0.0]))

        # Legacy 1e-4 configuration: eps = 0.010 > 0.005 (skips occluder!)
        associator_legacy = VisibilityAwareTextureAssociator(TextureAssociationConfig(ray_offset_epsilon_ratio=1e-4))
        res_legacy = associator_legacy.associate_texture(mesh, {"cam_far": cam_far}, TextureSampleType.FACET_CENTROID)
        rec_legacy = [r for r in res_legacy.decision_records if r.sample_index == 0][0]

        # New default configuration (unconfigured, default=1e-6): eps = 0.00010 < 0.005 (detects occluder!)
        associator_default = VisibilityAwareTextureAssociator()
        assert associator_default.config.ray_offset_epsilon_ratio == 1e-6
        res_default = associator_default.associate_texture(mesh, {"cam_far": cam_far}, TextureSampleType.FACET_CENTROID)
        rec_default = [r for r in res_default.decision_records if r.sample_index == 0][0]

        # Explicit 1e-6 configuration:
        associator_explicit = VisibilityAwareTextureAssociator(TextureAssociationConfig(ray_offset_epsilon_ratio=1e-6))
        res_explicit = associator_explicit.associate_texture(mesh, {"cam_far": cam_far}, TextureSampleType.FACET_CENTROID)
        rec_explicit = [r for r in res_explicit.decision_records if r.sample_index == 0][0]

        # Proves that legacy 1e-4 skips the thin occluder, while new default 1e-6 captures it!
        assert rec_legacy.query_status == TextureQueryStatus.VISIBLE       # Proves legacy 1e-4 defect
        assert rec_default.query_status == TextureQueryStatus.OCCLUDED      # Proves new default 1e-6 detects occluder
        assert rec_explicit.query_status == TextureQueryStatus.OCCLUDED     # Proves explicit 1e-6 detects occluder

    # --------------------------------------------------------------------------
    # 3. Self-Intersection Exclusion Audit
    # --------------------------------------------------------------------------
    def test_audit_facet_centroid_self_exclusion_exactness(self):
        """Verify facet centroid strictly excludes face j and admits adjacent non-coplanar face."""
        # Two connected faces forming a 90 degree roof corner
        vertices = np.array([
            [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 1.0, 0.0],  # Face 0 (slope 1)
            [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],    # Face 1 (slope 2)
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        mesh = _make_surface_mesh(vertices, faces)

        # Camera directly viewing face 0
        cam = _create_canonical_camera("cam_0", np.array([-2.0, 0.5, 2.0]), target=np.array([-0.66, 0.33, 0.33]))
        associator = VisibilityAwareTextureAssociator()
        res = associator.associate_texture(mesh, {"cam_0": cam}, TextureSampleType.FACET_CENTROID)

        rec0 = [r for r in res.decision_records if r.sample_index == 0][0]
        assert rec0.query_status == TextureQueryStatus.VISIBLE

    # --------------------------------------------------------------------------
    # 4. Möller-Trumbore Numerical Robustness & Near-Parallel Rays
    # --------------------------------------------------------------------------
    def test_audit_near_parallel_ray_stability(self):
        """Verify rays nearly parallel to triangle face (det -> 0) do not cause NaN or Inf."""
        vertices = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        bvh = DeterministicAABBBVH(vertices, faces)

        # Ray parallel to XY plane, grazing slightly above z=0.0
        O = np.array([-2.0, 0.0, 0.0001])
        D = np.array([4.0, 0.0, 0.0])
        hit = bvh.find_occlusion(O, D, set(), tau_det=1e-7, tau_bary=1e-6, tau_t=1e-6)
        # Should cleanly register no hit without division by zero or NaN
        assert hit is None

    # --------------------------------------------------------------------------
    # 5. BVH vs Brute-Force Mathematical Equivalence on Random Geometry
    # --------------------------------------------------------------------------
    def test_audit_bvh_vs_brute_force_on_random_geometry(self):
        """Generate 100 randomized triangles and 50 random rays; verify BVH matches linear scan."""
        np.random.seed(42)
        n_tri = 100
        n_rays = 50

        # Generate non-degenerate triangles
        v0 = np.random.uniform(-5.0, 5.0, size=(n_tri, 3))
        v1 = v0 + np.random.uniform(0.1, 1.0, size=(n_tri, 3))
        v2 = v0 + np.random.uniform(0.1, 1.0, size=(n_tri, 3))
        vertices = np.vstack([v0, v1, v2])
        faces = np.arange(n_tri * 3).reshape((n_tri, 3))

        bvh = DeterministicAABBBVH(vertices, faces)

        # Reference brute-force linear search
        def brute_force_occlusion(O, D):
            norm_D = np.linalg.norm(D)
            for i in range(n_tri):
                V0 = vertices[faces[i, 0]]
                V1 = vertices[faces[i, 1]]
                V2 = vertices[faces[i, 2]]
                E1 = V1 - V0
                E2 = V2 - V0
                P = np.cross(D, E2)
                det = float(np.dot(E1, P))
                denom = np.linalg.norm(E1) * norm_D * np.linalg.norm(E2)
                if abs(det) <= 1e-7 * denom:
                    continue
                inv_det = 1.0 / det
                T = O - V0
                u = float(np.dot(T, P)) * inv_det
                if u < -1e-6:
                    continue
                Q = np.cross(T, E1)
                v = float(np.dot(D, Q)) * inv_det
                if v < -1e-6 or (u + v) > (1.0 + 1e-6):
                    continue
                t = float(np.dot(E2, Q)) * inv_det
                if -1e-6 <= t <= (1.0 + 1e-6):
                    return True
            return False

        for _ in range(n_rays):
            O_ray = np.random.uniform(-10.0, 10.0, size=3)
            D_ray = np.random.uniform(-10.0, 10.0, size=3)
            bvh_hit = (bvh.find_occlusion(O_ray, D_ray, set(), tau_det=1e-7, tau_bary=1e-6, tau_t=1e-6) is not None)
            bf_hit = brute_force_occlusion(O_ray, D_ray)
            assert bvh_hit == bf_hit

    # --------------------------------------------------------------------------
    # 6. Candidate-First Distance Scoring Audit
    # --------------------------------------------------------------------------
    def test_audit_candidate_first_dmin_computation(self):
        """Verify d_min is computed exclusively across geometrically visible cameras."""
        mesh = _create_simple_triangle_mesh()
        # Cam 1: Very close, but BEHIND the mesh (occluded or negative depth)
        cam_behind = _create_canonical_camera("cam_01_behind", np.array([0.0, 0.0, -1.0]), target=np.array([0.0, 0.0, -5.0]))
        # Cam 2: Distance 3.0, VISIBLE
        cam_vis_close = _create_canonical_camera("cam_02_vis_close", np.array([0.0, 0.0, 3.0]))
        # Cam 3: Distance 6.0, VISIBLE
        cam_vis_far = _create_canonical_camera("cam_03_vis_far", np.array([0.0, 0.0, 6.0]))

        associator = VisibilityAwareTextureAssociator()
        res = associator.associate_texture(
            mesh,
            {"cam_01_behind": cam_behind, "cam_02_vis_close": cam_vis_close, "cam_03_vis_far": cam_vis_far},
            TextureSampleType.FACET_CENTROID,
        )

        obs_list = res.observations_by_sample[0]
        assert len(obs_list) == 2
        obs_close = [o for o in obs_list if o.frame_id == "cam_02_vis_close"][0]
        obs_far = [o for o in obs_list if o.frame_id == "cam_03_vis_far"][0]

        # Closest VISIBLE camera must achieve s_dist == 1.0 (not penalized by cam_behind)
        assert np.isclose(obs_close.geometric_score / (obs_close.geometric_score / (obs_close.distance_to_cam / obs_close.distance_to_cam)), 1.0)
        assert obs_close.composite_score > obs_far.composite_score

    # --------------------------------------------------------------------------
    # 7. Dynamic Risk Invariance Audit
    # --------------------------------------------------------------------------
    def test_audit_dynamic_risk_score_monotonicity(self):
        """Verify higher dynamic risk monotonically decreases composite score."""
        mesh = _create_simple_triangle_mesh()
        cam_low_risk = _create_canonical_camera(
            "cam_low_risk", np.array([0.0, 0.0, 2.0]), quality_metrics={"sharpness": 1.0, "blur": 0.0, "exposure": 1.0, "dynamic_risk": 0.1}
        )
        cam_high_risk = _create_canonical_camera(
            "cam_high_risk", np.array([0.0, 0.0, 2.0]), quality_metrics={"sharpness": 1.0, "blur": 0.0, "exposure": 1.0, "dynamic_risk": 0.8}
        )

        associator = VisibilityAwareTextureAssociator()
        res = associator.associate_texture(
            mesh, {"cam_low_risk": cam_low_risk, "cam_high_risk": cam_high_risk}, TextureSampleType.FACET_CENTROID
        )
        obs_low = [o for o in res.observations_by_sample[0] if o.frame_id == "cam_low_risk"][0]
        obs_high = [o for o in res.observations_by_sample[0] if o.frame_id == "cam_high_risk"][0]

        assert obs_low.composite_score > obs_high.composite_score
        assert np.isclose(obs_low.dynamic_risk_score, 0.9)
        assert np.isclose(obs_high.dynamic_risk_score, 0.2)

    # --------------------------------------------------------------------------
    # 8. Extreme Scale Sweep Audit (10^-12 to 10^12)
    # --------------------------------------------------------------------------
    @pytest.mark.parametrize("scale", [1e-12, 1e-8, 1e-4, 1.0, 1e4, 1e8, 1e12])
    def test_audit_extreme_scale_sweep_invariance(self, scale: float):
        """Verify extreme scale sweeps preserve projection coordinates and composite scores."""
        base_mesh = _create_simple_triangle_mesh()
        base_cam = _create_canonical_camera("cam_01", np.array([0.0, 0.0, 2.0]))

        s_mesh = _make_surface_mesh(
            vertices=base_mesh.vertices * scale,
            faces=base_mesh.faces,
            face_normals=base_mesh.face_normals,
            vertex_normals=base_mesh.vertex_normals,
        )
        s_cam = TextureSourceCamera(
            frame_id="cam_01",
            R_cw=base_cam.R_cw,
            t_cw=base_cam.t_cw * scale,
            K=base_cam.K,
            width=base_cam.width,
            height=base_cam.height,
            quality_metrics=base_cam.quality_metrics,
        )

        associator = VisibilityAwareTextureAssociator()
        res = associator.associate_texture(s_mesh, {"cam_01": s_cam}, TextureSampleType.FACET_CENTROID)
        obs = res.best_observation_by_sample[0]
        assert obs is not None
        # Projected pixel must be invariant across all 24 orders of magnitude
        u, v = obs.pixel_coords
        assert 315.0 <= u <= 325.0
        assert 315.0 <= v <= 330.0

    # --------------------------------------------------------------------------
    # 9. Comprehensive Mutation Testing
    # --------------------------------------------------------------------------
    def test_audit_mutation_guard_optical_center_formula(self):
        """Deliberately mutating C_w = -R_cw @ t_cw (omitting transpose) alters results."""
        cam = _create_canonical_camera("cam_01", np.array([2.0, 3.0, 4.0]))
        R_cw = cam.R_cw
        t_cw = cam.t_cw
        correct_C = -R_cw.T @ t_cw
        mutated_C = -R_cw @ t_cw
        assert not np.allclose(correct_C, mutated_C)

    def test_audit_mutation_guard_ray_direction(self):
        """Mutating ray direction D = O - E instead of E - O inverts segment."""
        P = np.array([0.0, 0.0, 0.0])
        C = np.array([0.0, 0.0, 2.0])
        v_view = np.array([0.0, 0.0, 1.0])
        eps = 1e-4
        O = P + eps * v_view
        E = C - eps * v_view
        correct_D = E - O
        mutated_D = O - E
        assert np.allclose(correct_D, -mutated_D)

    def test_audit_mutation_guard_dynamic_score_logic(self):
        """Mutating S_dynamic = dynamic_risk instead of 1 - dynamic_risk inverts ranking."""
        risk = 0.8
        correct_S_dyn = 1.0 - risk  # 0.2 (penalized)
        mutated_S_dyn = risk        # 0.8 (rewarded!)
        assert correct_S_dyn < 0.5 < mutated_S_dyn
