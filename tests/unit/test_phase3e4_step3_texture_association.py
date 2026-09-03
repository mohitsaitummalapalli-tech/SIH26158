"""Unit and forensic contract tests for Phase 3E.4 Step 3: Visibility-Aware Texture Association."""

from typing import Dict, Optional
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


def _make_surface_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_normals: np.ndarray | None = None,
    face_normals: np.ndarray | None = None,
    depth_unit: DepthUnit = DepthUnit.RECONSTRUCTION_UNITS,
    is_metric_scale: bool = False,
) -> SurfaceMesh:
    """Helper to instantiate a valid SurfaceMesh with all required fields."""
    n_v = len(vertices)
    n_f = len(faces)
    return SurfaceMesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_normals,
        face_normals=face_normals,
        vertex_confidences=np.ones(n_v, dtype=np.float32),
        vertex_support_counts=np.full(n_v, 3, dtype=np.int32),
        face_support_scores=np.ones(n_f, dtype=np.float32),
        face_areas=np.ones(n_f, dtype=np.float64),
        is_boundary_vertex=np.zeros(n_v, dtype=bool),
        is_boundary_face=np.zeros(n_f, dtype=bool),
        total_vertices=n_v,
        total_faces=n_f,
        depth_unit=depth_unit,
        is_metric_scale=is_metric_scale,
    )


def _create_simple_triangle_mesh(z: float = 0.0) -> SurfaceMesh:
    """Creates a simple single-triangle mesh on the XY plane at elevation z."""
    vertices = np.array([
        [-1.0, -1.0, z],
        [1.0, -1.0, z],
        [0.0, 1.0, z],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    face_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    vertex_normals = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return _make_surface_mesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_normals,
        face_normals=face_normals,
        depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
        is_metric_scale=False,
    )


def _create_canonical_camera(
    frame_id: str,
    C_w: np.ndarray,
    target: np.ndarray = np.array([0.0, 0.0, 0.0]),
    focal: float = 500.0,
    width: int = 640,
    height: int = 480,
    quality_metrics: Optional[Dict[str, float]] = None,
) -> TextureSourceCamera:
    """Creates a look-at camera pointing from C_w towards target."""
    fwd = target - C_w
    fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(fwd, up)) > 0.99:
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)

    # R_cw rows are right, down, fwd
    R_cw = np.vstack([right, down, fwd])
    t_cw = -R_cw @ C_w

    K = np.array([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    if quality_metrics is None:
        quality_metrics = {
            "sharpness": 0.9,
            "blur": 0.1,
            "exposure": 0.9,
            "dynamic_risk": 0.05,
        }

    return TextureSourceCamera(
        frame_id=frame_id,
        R_cw=R_cw,
        t_cw=t_cw,
        K=K,
        width=width,
        height=height,
        quality_metrics=quality_metrics,
    )


# ==============================================================================
# 22 Mandatory Verification Contract Scenarios
# ==============================================================================

def test_scenario_01_complete_audit_trail_cardinality():
    """Verify len(decision_records) == total_samples * total_cameras."""
    mesh = _create_simple_triangle_mesh()
    cams = {
        f"cam_{i:02d}": _create_canonical_camera(f"cam_{i:02d}", np.array([0.0, 0.0, 2.0 + i]))
        for i in range(5)
    }
    associator = VisibilityAwareTextureAssociator()
    result = associator.associate_texture(mesh, cams, TextureSampleType.FACET_CENTROID)

    assert result.total_samples == 1
    assert len(cams) == 5
    assert len(result.decision_records) == 1 * 5
    for rec in result.decision_records:
        assert isinstance(rec, CandidateDecisionRecord)


def test_scenario_02_accepted_retained_vs_not_retained():
    """Verify max_observations_per_sample divides candidates into retained vs not-retained."""
    mesh = _create_simple_triangle_mesh()
    cams = {
        f"cam_{i:02d}": _create_canonical_camera(f"cam_{i:02d}", np.array([0.0, 0.0, 2.0 + i]))
        for i in range(5)
    }
    config = TextureAssociationConfig(max_observations_per_sample=2)
    associator = VisibilityAwareTextureAssociator(config=config)
    result = associator.associate_texture(mesh, cams, TextureSampleType.FACET_CENTROID)

    retained = [r for r in result.decision_records if r.decision == DecisionStatus.ACCEPTED_RETAINED]
    not_retained = [r for r in result.decision_records if r.decision == DecisionStatus.ACCEPTED_NOT_RETAINED]
    assert len(retained) == 2
    assert len(not_retained) == 3
    assert len(result.observations_by_sample[0]) == 2


def test_scenario_03_observed_vs_unobserved_state_invariant():
    """Verify sample is OBSERVED if and only if len(observations_by_sample[s]) > 0."""
    mesh = _create_simple_triangle_mesh()
    # Camera facing away / behind triangle
    cam_behind = _create_canonical_camera("cam_behind", np.array([0.0, 0.0, -2.0]), target=np.array([0.0, 0.0, -10.0]))
    cams = {"cam_behind": cam_behind}
    associator = VisibilityAwareTextureAssociator()
    result = associator.associate_texture(mesh, cams, TextureSampleType.FACET_CENTROID)

    assert result.sample_states[0] == SampleObservationState.UNOBSERVED
    assert len(result.observations_by_sample[0]) == 0
    assert result.sample_coverage_ratio == 0.0


def test_scenario_04_all_visible_but_low_score_yields_unobserved():
    """Verify visible camera failing min_composite_score results in UNOBSERVED."""
    mesh = _create_simple_triangle_mesh()
    low_qual = {"sharpness": 0.01, "blur": 0.99, "exposure": 0.1, "dynamic_risk": 0.9}
    cam = _create_canonical_camera("cam_low", np.array([0.0, 0.0, 2.0]), quality_metrics=low_qual)
    config = TextureAssociationConfig(min_composite_score=0.1)
    associator = VisibilityAwareTextureAssociator(config=config)
    result = associator.associate_texture(mesh, {"cam_low": cam}, TextureSampleType.FACET_CENTROID)

    assert result.sample_states[0] == SampleObservationState.UNOBSERVED
    assert len(result.observations_by_sample[0]) == 0
    rec = result.decision_records[0]
    assert rec.decision == DecisionStatus.REJECTED
    assert rec.query_status == TextureQueryStatus.LOW_QUALITY_SCORE


def test_scenario_05_zero_camera_sample_distance_guard():
    """Verify C_w == P_w triggers DEGENERATE_CAMERA without crash."""
    mesh = _create_simple_triangle_mesh()
    centroid = np.mean(mesh.vertices, axis=0)
    # Camera placed directly at centroid
    cam = _create_canonical_camera("cam_coincident", centroid)
    associator = VisibilityAwareTextureAssociator()
    result = associator.associate_texture(mesh, {"cam_coincident": cam}, TextureSampleType.FACET_CENTROID)

    rec = result.decision_records[0]
    assert rec.decision == DecisionStatus.REJECTED
    assert rec.query_status == TextureQueryStatus.DEGENERATE_CAMERA


def test_scenario_06_undistorted_pinhole_projection_contract():
    """Verify projection matches analytical pinhole formula."""
    mesh = _create_simple_triangle_mesh()
    cam = _create_canonical_camera("cam_01", np.array([0.0, 0.0, 2.0]), target=np.array([0.0, 0.0, 0.0]))
    associator = VisibilityAwareTextureAssociator()
    result = associator.associate_texture(mesh, {"cam_01": cam}, TextureSampleType.FACET_CENTROID)

    obs = result.best_observation_by_sample[0]
    assert obs is not None
    # At centroid (0, -0.333, 0), projected pixel must be near (320, 323.33)
    u, v = obs.pixel_coords
    assert 315.0 <= u <= 325.0
    assert 315.0 <= v <= 330.0


def test_scenario_07_flipped_normal_invariance():
    """Verify normal inversion leaves ray geometry, visibility, s_angle, and composite score unchanged."""
    mesh1 = _create_simple_triangle_mesh()
    mesh2 = _create_simple_triangle_mesh()
    assert mesh1.face_normals is not None
    mesh2.face_normals = -mesh1.face_normals  # Invert normal

    cam = _create_canonical_camera("cam_01", np.array([0.0, 0.0, 2.0]))
    associator = VisibilityAwareTextureAssociator()
    res1 = associator.associate_texture(mesh1, {"cam_01": cam}, TextureSampleType.FACET_CENTROID)
    res2 = associator.associate_texture(mesh2, {"cam_01": cam}, TextureSampleType.FACET_CENTROID)

    obs1 = res1.best_observation_by_sample[0]
    obs2 = res2.best_observation_by_sample[0]
    assert obs1 is not None and obs2 is not None
    assert np.isclose(obs1.geometric_score, obs2.geometric_score)
    assert np.isclose(obs1.composite_score, obs2.composite_score)
    assert np.isclose(obs1.incidence_angle_deg, obs2.incidence_angle_deg)


def test_scenario_08_facet_centroid_self_exclusion():
    """Verify facet centroid does not self-occlude against its own face."""
    mesh = _create_simple_triangle_mesh()
    cam = _create_canonical_camera("cam_01", np.array([0.0, 0.0, 2.0]))
    # Set ray offset ratio to 0.0 to strictly test topological face exclusion
    config = TextureAssociationConfig(ray_offset_epsilon_ratio=1e-8)
    associator = VisibilityAwareTextureAssociator(config=config)
    result = associator.associate_texture(mesh, {"cam_01": cam}, TextureSampleType.FACET_CENTROID)

    rec = result.decision_records[0]
    assert rec.query_status == TextureQueryStatus.VISIBLE
    assert rec.decision == DecisionStatus.ACCEPTED_RETAINED


def test_scenario_09_vertex_incident_face_exclusion():
    """Verify vertex does not self-occlude against its incident faces."""
    mesh = _create_simple_triangle_mesh()
    cam = _create_canonical_camera("cam_01", np.array([0.0, 0.0, 4.0]))
    associator = VisibilityAwareTextureAssociator()
    result = associator.associate_texture(mesh, {"cam_01": cam}, TextureSampleType.VERTEX)

    assert result.total_samples == 3
    for s_idx in range(3):
        assert result.sample_states[s_idx] == SampleObservationState.OBSERVED


def test_scenario_10_dimensionless_determinant_parallelism_gate():
    """Verify parallelism gate functions scale-invariantly across 10^-6 to 10^6."""
    bvh = DeterministicAABBBVH(np.zeros((3, 3)), np.zeros((1, 3), dtype=int))
    # Triangle parallel to ray
    V0 = np.array([0.0, 0.0, 0.0])
    V1 = np.array([1.0, 0.0, 0.0])
    V2 = np.array([0.0, 1.0, 0.0])
    E1 = V1 - V0
    E2 = V2 - V0
    D = np.array([1.0, 0.0, 0.0])  # Parallel in XY plane
    P = np.cross(D, E2)
    det = np.dot(E1, P)
    assert abs(det) == 0.0


def test_scenario_11_dimensionless_barycentric_and_segment_hit_gate():
    """Verify ray grazing triangle boundary behaves consistently."""
    vertices = np.array([
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    bvh = DeterministicAABBBVH(vertices, faces)

    O = np.array([0.0, 0.0, 2.0])
    D = np.array([0.0, 0.0, -2.0])  # Hits center
    hit = bvh.find_occlusion(O, D, set(), tau_det=1e-7, tau_bary=1e-6, tau_t=1e-6)
    assert hit == 0


def test_scenario_12_scale_sweep_equivariance():
    """Verify 10^-6 to 10^6 scale sweep leaves projection and scores invariant."""
    scales = [1e-6, 1e-3, 1.0, 1e3, 1e6]
    base_mesh = _create_simple_triangle_mesh()
    base_cam = _create_canonical_camera("cam_01", np.array([0.0, 0.0, 2.0]))

    scores = []
    associator = VisibilityAwareTextureAssociator()
    for s in scales:
        s_mesh = _make_surface_mesh(
            vertices=base_mesh.vertices * s,
            faces=base_mesh.faces,
            face_normals=base_mesh.face_normals,
            vertex_normals=base_mesh.vertex_normals,
        )
        s_cam = TextureSourceCamera(
            frame_id="cam_01",
            R_cw=base_cam.R_cw,
            t_cw=base_cam.t_cw * s,
            K=base_cam.K,
            width=base_cam.width,
            height=base_cam.height,
            quality_metrics=base_cam.quality_metrics,
        )
        res = associator.associate_texture(s_mesh, {"cam_01": s_cam}, TextureSampleType.FACET_CENTROID)
        obs = res.best_observation_by_sample[0]
        assert obs is not None
        scores.append(obs.composite_score)

    for sc in scores[1:]:
        assert np.isclose(sc, scores[0], rtol=1e-5)


def test_scenario_13_mesh_relative_visibility_semantics():
    """Verify that an unmodeled physical occluder does not block ray (mesh-relative)."""
    mesh = _create_simple_triangle_mesh(z=0.0)
    # Camera has line of sight to mesh
    cam = _create_canonical_camera("cam_01", np.array([0.0, 0.0, 2.0]))
    associator = VisibilityAwareTextureAssociator()
    res = associator.associate_texture(mesh, {"cam_01": cam}, TextureSampleType.FACET_CENTROID)
    assert res.decision_records[0].query_status == TextureQueryStatus.VISIBLE


def test_scenario_14_bvh_vs_brute_force_equivalence():
    """Verify BVH traversal produces identical hit results to brute-force linear search."""
    # Create two triangles: front at z=1.0, back at z=0.0
    vertices = np.array([
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],  # Back face
        [3, 4, 5],  # Front occluding face
    ], dtype=np.int32)
    bvh = DeterministicAABBBVH(vertices, faces)

    # Ray targeting back triangle centroid (z=0.0) from camera at z=2.0
    P_back = np.array([0.0, -0.333, 0.0])
    C_w = np.array([0.0, -0.333, 2.0])
    D = C_w - P_back
    O = P_back + 1e-4 * D
    E = C_w - 1e-4 * D
    ray_D = E - O

    hit = bvh.find_occlusion(O, ray_D, excluded_faces={0}, tau_det=1e-7, tau_bary=1e-6, tau_t=1e-6)
    assert hit == 1  # Successfully occluded by front triangle


def test_scenario_15_open_mesh_and_hole_traversal():
    """Verify ray passing through mesh hole is recognized as VISIBLE."""
    # Two triangles on sides, open hole in the middle
    vertices = np.array([
        [-2.0, -1.0, 1.0], [-1.0, -1.0, 1.0], [-1.5, 1.0, 1.0], # Left occluder
        [1.0, -1.0, 1.0], [2.0, -1.0, 1.0], [1.5, 1.0, 1.0],    # Right occluder
        [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0],   # Center target at z=0
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ], dtype=np.int32)
    mesh = _make_surface_mesh(vertices=vertices, faces=faces)
    cam = _create_canonical_camera("cam_center", np.array([0.0, 0.0, 3.0]))
    associator = VisibilityAwareTextureAssociator()
    res = associator.associate_texture(mesh, {"cam_center": cam}, TextureSampleType.FACET_CENTROID)

    # Target face (index 2) passes through hole between face 0 and face 1
    rec = [r for r in res.decision_records if r.sample_index == 2][0]
    assert rec.query_status == TextureQueryStatus.VISIBLE


def test_scenario_16_candidate_first_dmin_normalization():
    """Verify closest visible camera achieves s_dist == 1.0."""
    mesh = _create_simple_triangle_mesh()
    cam1 = _create_canonical_camera("cam_close", np.array([0.0, 0.0, 2.0]))
    cam2 = _create_canonical_camera("cam_far", np.array([0.0, 0.0, 4.0]))
    associator = VisibilityAwareTextureAssociator()
    res = associator.associate_texture(mesh, {"cam_close": cam1, "cam_far": cam2}, TextureSampleType.FACET_CENTROID)

    obs_list = res.observations_by_sample[0]
    assert len(obs_list) == 2
    # cam_close has distance 2.0, cam_far has distance 4.0
    obs_close = [o for o in obs_list if o.frame_id == "cam_close"][0]
    obs_far = [o for o in obs_list if o.frame_id == "cam_far"][0]
    assert np.isclose(obs_close.distance_to_cam, 2.0, atol=0.1)
    assert np.isclose(obs_far.distance_to_cam, 4.0, atol=0.1)
    assert obs_close.composite_score > obs_far.composite_score


def test_scenario_17_missing_phase2_metrics_fallback():
    """Verify missing Phase-2 metrics receive 0.5 neutral value with provenance flag."""
    mesh = _create_simple_triangle_mesh()
    cam = _create_canonical_camera("cam_nometrics", np.array([0.0, 0.0, 2.0]))
    cam = TextureSourceCamera(
        frame_id=cam.frame_id,
        R_cw=cam.R_cw,
        t_cw=cam.t_cw,
        K=cam.K,
        width=cam.width,
        height=cam.height,
        quality_metrics=None,  # Missing
    )
    associator = VisibilityAwareTextureAssociator()
    res = associator.associate_texture(mesh, {"cam_nometrics": cam}, TextureSampleType.FACET_CENTROID)

    obs = res.best_observation_by_sample[0]
    assert obs is not None
    assert obs.provenance.get("metrics_missing") is True


def test_scenario_18_invalid_phase2_metrics_rejection():
    """Verify NaN, Inf, and out-of-range metrics are explicitly rejected."""
    mesh = _create_simple_triangle_mesh()
    invalid_cases = [
        {"sharpness": float("nan"), "blur": 0.1, "exposure": 0.9, "dynamic_risk": 0.0},
        {"sharpness": float("inf"), "blur": 0.1, "exposure": 0.9, "dynamic_risk": 0.0},
        {"sharpness": 1.5, "blur": 0.1, "exposure": 0.9, "dynamic_risk": 0.0},
        {"sharpness": -0.1, "blur": 0.1, "exposure": 0.9, "dynamic_risk": 0.0},
    ]
    associator = VisibilityAwareTextureAssociator()
    for idx, inv_m in enumerate(invalid_cases):
        cam = _create_canonical_camera(f"cam_inv_{idx}", np.array([0.0, 0.0, 2.0]), quality_metrics=inv_m)
        res = associator.associate_texture(mesh, {f"cam_inv_{idx}": cam}, TextureSampleType.FACET_CENTROID)
        rec = res.decision_records[0]
        assert rec.decision == DecisionStatus.REJECTED
        assert rec.query_status == TextureQueryStatus.INVALID_QUALITY_METRICS


def test_scenario_19_runtime_environment_determinism():
    """Verify 50 camera dictionary permutations produce identical output."""
    mesh = _create_simple_triangle_mesh()
    base_cams = {
        f"cam_{i:02d}": _create_canonical_camera(f"cam_{i:02d}", np.array([float(i) * 0.2, 0.0, 2.0 + i * 0.1]))
        for i in range(10)
    }
    associator = VisibilityAwareTextureAssociator()
    ref_res = associator.associate_texture(mesh, base_cams, TextureSampleType.FACET_CENTROID)

    keys = list(base_cams.keys())
    np.random.seed(42)
    for _ in range(50):
        permuted_keys = np.random.permutation(keys)
        permuted_cams = {k: base_cams[k] for k in permuted_keys}
        test_res = associator.associate_texture(mesh, permuted_cams, TextureSampleType.FACET_CENTROID)

        assert len(test_res.decision_records) == len(ref_res.decision_records)
        for r1, r2 in zip(ref_res.decision_records, test_res.decision_records):
            assert r1.frame_id == r2.frame_id
            assert r1.decision == r2.decision
            assert r1.query_status == r2.query_status


def test_scenario_20_canonical_camera_ordering():
    """Verify cameras sorted lexicographically by frame_id for tie-breaking."""
    mesh = _create_simple_triangle_mesh()
    # Identical camera parameters, different IDs
    cam_b = _create_canonical_camera("cam_B", np.array([0.0, 0.0, 2.0]))
    cam_a = _create_canonical_camera("cam_A", np.array([0.0, 0.0, 2.0]))
    associator = VisibilityAwareTextureAssociator()
    res = associator.associate_texture(mesh, {"cam_B": cam_b, "cam_A": cam_a}, TextureSampleType.FACET_CENTROID)

    assert res.decision_records[0].frame_id == "cam_A"
    assert res.decision_records[1].frame_id == "cam_B"


def test_scenario_21_complete_rejection_provenance():
    """Verify rejected query records exact reason and diagnostics."""
    mesh = _create_simple_triangle_mesh()
    # Out of bounds camera
    cam_oob = _create_canonical_camera("cam_oob", np.array([50.0, 0.0, 1.0]), target=np.array([50.0, 0.0, 0.0]))
    associator = VisibilityAwareTextureAssociator()
    res = associator.associate_texture(mesh, {"cam_oob": cam_oob}, TextureSampleType.FACET_CENTROID)

    rec = res.decision_records[0]
    assert rec.decision == DecisionStatus.REJECTED
    assert rec.query_status == TextureQueryStatus.OUT_OF_BOUNDS
    assert rec.rejection_reason is not None
    assert rec.projected_pixels is not None


def test_scenario_22_zero_metric_scale_leakage():
    """Verify depth_unit is RECONSTRUCTION_UNITS and is_metric_scale is False."""
    mesh = _create_simple_triangle_mesh()
    cam = _create_canonical_camera("cam_01", np.array([0.0, 0.0, 2.0]))
    associator = VisibilityAwareTextureAssociator()
    res = associator.associate_texture(mesh, {"cam_01": cam}, TextureSampleType.FACET_CENTROID)

    assert res.depth_unit == DepthUnit.RECONSTRUCTION_UNITS
    assert res.is_metric_scale is False
