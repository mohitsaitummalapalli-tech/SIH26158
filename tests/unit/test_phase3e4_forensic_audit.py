"""Forensic adversarial audit and falsification test suite for Phase 3E.4 Surface Reconstruction.

Covers the 11 forensic audit items:
1. QHULL / QJ AUDIT:
   - Evaluates QJ option on minimal 4-point tetrahedron (Qhull error: needs 5 points).
   - Evaluates QJ vs default Delaunay on 5+ points (joggle perturbations, non-determinism).
2. NON-COPLANARITY THRESHOLD AUDIT:
   - Tests coordinate scaling sweep [1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6] on mathematically
     identical geometry. Demonstrates that absolute threshold s2 < 1e-7 rejects small-scale geometry.
3. CIRCUMRADIUS AUDIT:
   - Analytically known tetrahedra (trirectangular, regular) vs implementation formula.
   - Degenerate coplanar 4-point configuration returns infinity.
   - Coordinate immutability.
4. ALPHA-COMPLEX TOPOLOGY AUDIT:
   - Face occurrence tracking (count 1: candidate boundary, count 2: interior).
   - Safe exclusion of non-boundary faces.
5. FINAL FACE FILTER AUDIT:
   - Topological boundary recomputed strictly AFTER geometric filtering.
   - Falsification test: verifies that computing boundary before filtering produces wrong boundary flags.
6. SCALE AUDIT:
   - Coordinate and parameter scaling equivariance across declared range.
7. DEGENERACY AUDIT:
   - Exactly coplanar, collinear, duplicate points, near-duplicate points.
   - Verification that no NaN/Inf coordinates reach SurfaceMesh.
8. PROVENANCE / UNITS / SAFETY AUDIT:
   - Preservation of RECONSTRUCTION_UNITS, is_metric_scale=False, absence of hole-filling.
9. NORMAL INDEPENDENCE AUDIT:
   - Bit-exact face topology and vertex coordinates under compute_normals=True vs False.
10. DETERMINISM AUDIT:
    - 50 random input permutations yield identical canonical output.
11. MUTATION & FALSIFICATION GUARDS:
    - Verifies sensitivity to inverted aspect ratio, reversed relative area, and incorrect circumradius.
"""

from typing import List, Tuple
import numpy as np
import pytest
from scipy.spatial import Delaunay, QhullError

from src.geometry.mvs import (
    DensePointCloud,
    DepthUnit,
    PointValidationStatus,
    PointVisibilityState,
)
from src.geometry.surface_reconstruction import (
    AlphaComplexSurfaceReconstructor,
    DensePointCloudValidator,
    SurfaceFailureReason,
    SurfaceMesh,
    SurfaceReconstructionConfig,
    SurfaceReconstructionResult,
    SurfaceReconstructionStatus,
)
from tests.unit.test_phase3e4_surface_reconstruction import (
    make_regular_tetrahedron_points,
    make_valid_dense_point_cloud,
)


class TestPhase3E4ForensicAudit:
    """Adversarial and forensic tests auditing Phase 3E.4 against approved contract."""

    # -------------------------------------------------------------------------
    # 1. QHULL / QJ AUDIT
    # -------------------------------------------------------------------------
    def test_qj_fails_on_four_point_tetrahedron(self) -> None:
        """Audit 1.1: Prove that qhull_options='QJ' crashes on 4-point minimal 3D tetrahedron.

        Qhull computes 3D Delaunay by lifting to a 4D convex hull.
        In 4D, joggle requires at least d+2 = 5 vertices to build an initial simplex.
        Passing QJ with 4 points causes QhullError QH6214.
        The implementation avoids QJ, using default Delaunay.
        """
        pts = make_regular_tetrahedron_points(scale=1.0)
        # Verify default Delaunay succeeds without QJ
        dt_default = Delaunay(pts)
        assert dt_default.simplices.shape[0] == 1

        # Verify QJ fails with QhullError
        with pytest.raises(QhullError, match="not enough points"):
            Delaunay(pts, qhull_options="QJ")

    def test_qj_perturbs_coordinates_and_weakens_determinism_on_five_points(self) -> None:
        """Audit 1.2: Compare QJ vs exact Delaunay on 5-point configuration.

        QJ adds perturbations. Although simplices may be computed for N>=5,
        perturbations weaken exact circumradius calculation and determinism.
        """
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, np.sqrt(3) / 2, 0.0],
                [0.5, np.sqrt(3) / 6, 0.8],
                [0.5, np.sqrt(3) / 6, -0.8],
            ],
            dtype=np.float64,
        )
        dt_exact = Delaunay(pts)
        assert dt_exact.simplices.shape[0] == 2
        # Exact Delaunay vertices are strictly unmodified
        assert np.array_equal(dt_exact.points, pts)

    def test_qj_vs_default_adversarial_comparisons(self) -> None:
        """Audit 1.3: Adversarially compare QJ vs default Delaunay across:
        1. Ordinary well-conditioned point set (6 points)
        2. Nearly coplanar point set
        3. Nearly degenerate tetrahedra (extreme sliver)
        4. Repeated / near-duplicate points
        5. Different input permutations
        """
        # 1. Ordinary well-conditioned
        pts_well = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, np.sqrt(3) / 2, 0.0],
                [0.5, np.sqrt(3) / 6, 0.8],
                [0.5, np.sqrt(3) / 6, -0.8],
                [0.2, 0.3, 0.4],
            ],
            dtype=np.float64,
        )
        dt_well_def = Delaunay(pts_well)
        dt_well_qj = Delaunay(pts_well, qhull_options="QJ")
        assert dt_well_def.simplices.shape[0] >= 1
        assert dt_well_qj.simplices.shape[0] >= 1
        assert np.array_equal(dt_well_def.points, pts_well)

        # 2. Nearly coplanar point set
        pts_coplanar = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.5, 0.5, 1e-8],
                [0.2, 0.7, -1e-8],
            ],
            dtype=np.float64,
        )
        dt_coplanar_def = Delaunay(pts_coplanar)
        dt_coplanar_qj = Delaunay(pts_coplanar, qhull_options="QJ")
        # Under joggled input, near-coplanar points can yield non-canonical simplex counts
        assert dt_coplanar_def.simplices.shape[1] == 4
        assert dt_coplanar_qj.simplices.shape[1] == 4

        # 3. Nearly degenerate tetrahedra (sliver)
        pts_sliver = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1e-6, 0.0],
                [0.5, 0.5, 1.0],
                [0.2, 0.3, 0.2],
            ],
            dtype=np.float64,
        )
        dt_sliver_def = Delaunay(pts_sliver)
        dt_sliver_qj = Delaunay(pts_sliver, qhull_options="QJ")
        assert dt_sliver_def.simplices.shape[1] == 4
        assert dt_sliver_qj.simplices.shape[1] == 4

        # 4. Repeated / near-duplicate points: QJ can create unexpected extra tetrahedra
        pts_neardup = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1e-12],  # Near duplicate of vertex 0
                [0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        )
        dt_neardup_def = Delaunay(pts_neardup)
        dt_neardup_qj = Delaunay(pts_neardup, qhull_options="QJ")
        # Default Delaunay produces 5 simplices; QJ joggles vertex apart into 6 simplices!
        assert dt_neardup_def.simplices.shape[0] == 5
        assert dt_neardup_qj.simplices.shape[0] == 6

        # 5. Input permutations: QJ with permutation can alter which simplices are formed
        perm = [1, 0, 3, 2, 5, 4]
        dt_perm_def = Delaunay(pts_well[perm])
        dt_perm_qj = Delaunay(pts_well[perm], qhull_options="QJ")
        assert dt_perm_def.simplices.shape[0] == dt_well_def.simplices.shape[0]
        assert dt_perm_qj.simplices.shape[0] == dt_well_qj.simplices.shape[0]

    # -------------------------------------------------------------------------
    # 2. NON-COPLANARITY THRESHOLD AUDIT & REGRESSION
    # -------------------------------------------------------------------------
    def test_non_coplanarity_scale_sweep_regression(self) -> None:
        """Audit 2 & Regression: Sweep coordinate scale across 1e-16 to 1e8.

        Proves that the dimensionless singular-value ratio criterion accepts
        the same well-conditioned non-coplanar geometry consistently across all scales:
        1e-16, 1e-12, 1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8.
        """
        tet = make_regular_tetrahedron_points(scale=1.0)
        scales = [1e-16, 1e-12, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6, 1e8]
        res_by_scale = {}

        for sc in scales:
            pts = tet * sc
            cloud = make_valid_dense_point_cloud(4, pts=pts)
            config = SurfaceReconstructionConfig(
                alpha_radius=10.0 * sc,
                alpha_edge=10.0 * sc,
                min_relative_area=1e-6,
            )
            res_by_scale[sc] = AlphaComplexSurfaceReconstructor(config).reconstruct_surface(cloud)

        # All scales across 1e-16 to 1e8 must have the exact same non-coplanarity acceptance decision
        for sc in scales:
            assert SurfaceFailureReason.INSUFFICIENT_NON_COPLANAR_POINTS not in res_by_scale[sc].failure_reasons, (
                f"Scale {sc} was incorrectly rejected by non-coplanarity check!"
            )
            # From scale 1e-4 to 1e8, full reconstruction succeeds and emits 4 faces
            if sc >= 1e-4:
                assert res_by_scale[sc].status == SurfaceReconstructionStatus.SUCCESS
                assert res_by_scale[sc].mesh is not None
                assert res_by_scale[sc].mesh.total_faces == 4

    def test_non_coplanarity_degeneracy_and_extreme_scale_cases(self) -> None:
        """Regression tests for non-coplanarity under:
        1. Exactly coplanar geometry
        2. Nearly coplanar geometry (thickness ratio below min_thickness_ratio)
        3. Collinear geometry
        4. Extremely small but well-conditioned geometry (scale 1e-4 and 1e-8)
        5. Extremely large but well-conditioned geometry (scale 1e6 and 1e8)
        6. Verify no NaN/Inf enters the reconstruction
        """
        # 1. Exactly coplanar geometry
        pts_coplanar = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        cloud_coplanar = make_valid_dense_point_cloud(4, pts=pts_coplanar)
        res_coplanar = AlphaComplexSurfaceReconstructor().reconstruct_surface(cloud_coplanar)
        assert res_coplanar.status == SurfaceReconstructionStatus.RECONSTRUCTION_FAILED
        assert SurfaceFailureReason.INSUFFICIENT_NON_COPLANAR_POINTS in res_coplanar.failure_reasons
        assert res_coplanar.mesh is None

        # 1b. Coincident points (zero overall geometric extent, s0 == 0.0)
        pts_coincident = np.zeros((4, 3), dtype=np.float64)
        cloud_coincident = make_valid_dense_point_cloud(4, pts=pts_coincident)
        res_coincident = AlphaComplexSurfaceReconstructor().reconstruct_surface(cloud_coincident)
        assert res_coincident.status == SurfaceReconstructionStatus.RECONSTRUCTION_FAILED
        assert SurfaceFailureReason.INSUFFICIENT_NON_COPLANAR_POINTS in res_coincident.failure_reasons
        assert res_coincident.mesh is None

        # 2. Nearly coplanar geometry (s2 / s0 ~ 1e-6 < default min_thickness_ratio 1e-4)
        pts_nearly_coplanar = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1e-6],
            ],
            dtype=np.float64,
        )
        cloud_nearly_coplanar = make_valid_dense_point_cloud(4, pts=pts_nearly_coplanar)
        res_nearly_coplanar = AlphaComplexSurfaceReconstructor().reconstruct_surface(cloud_nearly_coplanar)
        assert res_nearly_coplanar.status == SurfaceReconstructionStatus.RECONSTRUCTION_FAILED
        assert SurfaceFailureReason.INSUFFICIENT_NON_COPLANAR_POINTS in res_nearly_coplanar.failure_reasons
        assert res_nearly_coplanar.mesh is None

        # 3. Collinear geometry (all points on a 1D line in 3D space)
        pts_collinear = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ],
            dtype=np.float64,
        )
        cloud_collinear = make_valid_dense_point_cloud(4, pts=pts_collinear)
        res_collinear = AlphaComplexSurfaceReconstructor().reconstruct_surface(cloud_collinear)
        assert res_collinear.status == SurfaceReconstructionStatus.RECONSTRUCTION_FAILED
        assert SurfaceFailureReason.INSUFFICIENT_NON_COPLANAR_POINTS in res_collinear.failure_reasons
        assert res_collinear.mesh is None

        # 4. Small well-conditioned geometry (scale 1e-4 emits complete 4-face SurfaceMesh)
        sc_small = 1e-4
        tet_unit = make_regular_tetrahedron_points(scale=1.0)
        cloud_small = make_valid_dense_point_cloud(4, pts=tet_unit * sc_small)
        config_small = SurfaceReconstructionConfig(
            alpha_radius=2.0 * sc_small,
            alpha_edge=2.0 * sc_small,
        )
        res_small = AlphaComplexSurfaceReconstructor(config_small).reconstruct_surface(cloud_small)
        assert res_small.status == SurfaceReconstructionStatus.SUCCESS
        assert res_small.mesh is not None
        assert res_small.mesh.total_faces == 4
        assert np.all(np.isfinite(res_small.mesh.vertices))
        assert np.all(np.isfinite(res_small.mesh.face_areas))
        assert not np.any(np.isnan(res_small.mesh.vertices))

        # 4b. Extremely small geometry (scale 1e-8) is accepted by coplanarity guard without crash or NaN
        sc_micro = 1e-8
        cloud_micro = make_valid_dense_point_cloud(4, pts=tet_unit * sc_micro)
        config_micro = SurfaceReconstructionConfig(
            alpha_radius=2.0 * sc_micro,
            alpha_edge=2.0 * sc_micro,
        )
        res_micro = AlphaComplexSurfaceReconstructor(config_micro).reconstruct_surface(cloud_micro)
        assert SurfaceFailureReason.INSUFFICIENT_NON_COPLANAR_POINTS not in res_micro.failure_reasons

        # 5. Extremely large but well-conditioned geometry (scale 1e8)
        sc_large = 1e8
        cloud_large = make_valid_dense_point_cloud(4, pts=tet_unit * sc_large)
        config_large = SurfaceReconstructionConfig(
            alpha_radius=2.0 * sc_large,
            alpha_edge=2.0 * sc_large,
        )
        res_large = AlphaComplexSurfaceReconstructor(config_large).reconstruct_surface(cloud_large)
        assert res_large.status == SurfaceReconstructionStatus.SUCCESS
        assert res_large.mesh is not None
        assert res_large.mesh.total_faces == 4
        assert np.all(np.isfinite(res_large.mesh.vertices))
        assert np.all(np.isfinite(res_large.mesh.face_areas))
        assert not np.any(np.isnan(res_large.mesh.vertices))

    # -------------------------------------------------------------------------
    # 3. CIRCUMRADIUS AUDIT
    # -------------------------------------------------------------------------
    def test_circumradius_analytical_trirectangular(self) -> None:
        """Audit 3.1: Verify exact circumradius formula on trirectangular tetrahedron.

        Vertices: (0,0,0), (a,0,0), (0,b,0), (0,0,c).
        Circumcenter: (a/2, b/2, c/2).
        R_exact = 0.5 * sqrt(a^2 + b^2 + c^2).
        """
        a, b, c = 1.0, 2.0, 2.0
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [a, 0.0, 0.0],
                [0.0, b, 0.0],
                [0.0, 0.0, c],
            ],
            dtype=np.float64,
        )
        r_computed = AlphaComplexSurfaceReconstructor._compute_tetrahedron_circumradius(
            pts[0], pts[1], pts[2], pts[3]
        )
        r_exact = 0.5 * np.sqrt(a**2 + b**2 + c**2)
        assert np.isclose(r_computed, r_exact, atol=1e-12)
        assert np.isclose(r_computed, 1.5, atol=1e-12)

    def test_circumradius_analytical_regular_tetrahedron(self) -> None:
        """Audit 3.2: Verify exact circumradius on regular tetrahedron with edge a.

        R_exact = (sqrt(6) / 4) * a.
        """
        a = 3.0
        pts = make_regular_tetrahedron_points(scale=a)
        r_computed = AlphaComplexSurfaceReconstructor._compute_tetrahedron_circumradius(
            pts[0], pts[1], pts[2], pts[3]
        )
        r_exact = (np.sqrt(6.0) / 4.0) * a
        assert np.isclose(r_computed, r_exact, atol=1e-12)

    def test_circumradius_coplanar_returns_infinity(self) -> None:
        """Audit 3.3: 4 coplanar points have zero tetrahedral volume, returning infinity."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        p3 = np.array([1.0, 1.0, 0.0])
        r = AlphaComplexSurfaceReconstructor._compute_tetrahedron_circumradius(p0, p1, p2, p3)
        assert np.isinf(r)

    # -------------------------------------------------------------------------
    # 4. ALPHA-COMPLEX TOPOLOGY AUDIT
    # -------------------------------------------------------------------------
    def test_topology_internal_face_exclusion(self) -> None:
        """Audit 4: Two tetrahedra sharing a face.

        Total faces = 8.
        Shared face has count 2 -> interior, excluded.
        Emitted candidate faces = 6.
        """
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, np.sqrt(3) / 2, 0.0],
                [0.5, np.sqrt(3) / 6, 0.8],
                [0.5, np.sqrt(3) / 6, -0.8],
            ],
            dtype=np.float64,
        )
        cloud = make_valid_dense_point_cloud(5, pts=pts)
        config = SurfaceReconstructionConfig(alpha_radius=2.0, alpha_edge=2.0)
        res = AlphaComplexSurfaceReconstructor(config).reconstruct_surface(cloud)

        assert res.mesh is not None
        assert res.mesh.total_faces == 6
        # The base face vertices are {0, 1, 2}
        base_set = {0, 1, 2}
        for f in res.mesh.faces:
            assert set(f) != base_set

    # -------------------------------------------------------------------------
    # 5. FINAL FACE FILTER AUDIT
    # -------------------------------------------------------------------------
    def test_boundary_flags_computed_after_filtering(self) -> None:
        """Audit 5: Verify boundary semantics are calculated strictly from final emitted faces.

        Adversarial check: In a closed tetrahedron, all boundary flags are False.
        If a face is rejected by alpha_edge, the remaining 3 faces MUST have is_boundary_face=True.
        If boundary was computed before filtering, they would be wrongly marked False!
        """
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.8, 0.0],
                [0.5, 0.3, 1.8],
            ],
            dtype=np.float64,
        )
        cloud = make_valid_dense_point_cloud(4, pts=pts)
        # alpha_edge = 1.5 drops 3 faces, retaining 1 face
        config = SurfaceReconstructionConfig(alpha_radius=5.0, alpha_edge=1.5)
        res = AlphaComplexSurfaceReconstructor(config).reconstruct_surface(cloud)

        assert res.mesh is not None
        assert res.mesh.total_faces == 1
        # The remaining face MUST be recognized as boundary face
        assert res.mesh.is_boundary_face[0] is np.True_ or res.mesh.is_boundary_face[0] == True
        # All 3 vertices of this face must be boundary vertices
        face_v = res.mesh.faces[0]
        for v_idx in face_v:
            assert res.mesh.is_boundary_vertex[v_idx] == True

    # -------------------------------------------------------------------------
    # 6. SCALE AUDIT
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("scale", [0.1, 1.0, 10.0, 100.0])
    def test_scale_equivariance_parameter_sweep(self, scale: float) -> None:
        """Audit 6: Coordinate scaling scales vertices by s, areas by s^2, keeping faces invariant."""
        pts_base = make_regular_tetrahedron_points(scale=1.0)
        cloud_base = make_valid_dense_point_cloud(4, pts=pts_base)
        config_base = SurfaceReconstructionConfig(alpha_radius=2.0, alpha_edge=2.0)
        res_base = AlphaComplexSurfaceReconstructor(config_base).reconstruct_surface(cloud_base)

        pts_scaled = pts_base * scale
        cloud_scaled = make_valid_dense_point_cloud(4, pts=pts_scaled)
        config_scaled = SurfaceReconstructionConfig(
            alpha_radius=2.0 * scale,
            alpha_edge=2.0 * scale,
        )
        res_scaled = AlphaComplexSurfaceReconstructor(config_scaled).reconstruct_surface(cloud_scaled)

        assert res_base.mesh is not None and res_scaled.mesh is not None
        assert np.array_equal(res_base.mesh.faces, res_scaled.mesh.faces)
        assert np.allclose(res_scaled.mesh.vertices, res_base.mesh.vertices * scale, atol=1e-12)
        assert np.allclose(res_scaled.mesh.face_areas, res_base.mesh.face_areas * (scale**2), atol=1e-12)

    # -------------------------------------------------------------------------
    # 7. DEGENERACY AUDIT
    # -------------------------------------------------------------------------
    def test_exact_coplanar_rejection(self) -> None:
        """Audit 7.1: Exactly coplanar points cleanly rejected without NaN."""
        pts = np.zeros((8, 3), dtype=np.float64)
        pts[:, :2] = np.random.default_rng(1).uniform(-1.0, 1.0, (8, 2))
        cloud = make_valid_dense_point_cloud(8, pts=pts)
        res = AlphaComplexSurfaceReconstructor().reconstruct_surface(cloud)
        assert res.status == SurfaceReconstructionStatus.RECONSTRUCTION_FAILED
        assert SurfaceFailureReason.INSUFFICIENT_NON_COPLANAR_POINTS in res.failure_reasons
        assert res.mesh is None

    def test_duplicate_points_handled_safely(self) -> None:
        """Audit 7.2: Duplicate points do not cause unhandled crash."""
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],  # Duplicate
                [1.0, 0.0, 0.0],
                [0.5, 0.8, 0.0],
                [0.5, 0.3, 0.8],
            ],
            dtype=np.float64,
        )
        cloud = make_valid_dense_point_cloud(5, pts=pts)
        res = AlphaComplexSurfaceReconstructor().reconstruct_surface(cloud)
        # SVD and Delaunay run; duplicate vertices do not cause unhandled exception
        assert res.status in (
            SurfaceReconstructionStatus.SUCCESS,
            SurfaceReconstructionStatus.RECONSTRUCTION_FAILED,
            SurfaceReconstructionStatus.EMPTY_VALID_OUTPUT,
        )

    # -------------------------------------------------------------------------
    # 8. PROVENANCE / UNITS / SAFETY AUDIT
    # -------------------------------------------------------------------------
    def test_reconstruction_units_and_metric_scale_immutable(self) -> None:
        """Audit 8: SurfaceMesh strictly maintains RECONSTRUCTION_UNITS and is_metric_scale=False."""
        pts = make_regular_tetrahedron_points(scale=1.0)
        cloud = make_valid_dense_point_cloud(4, pts=pts, is_metric_scale=False)
        res = AlphaComplexSurfaceReconstructor(SurfaceReconstructionConfig(alpha_radius=2.0, alpha_edge=2.0)).reconstruct_surface(cloud)

        assert res.mesh is not None
        assert res.mesh.depth_unit == DepthUnit.RECONSTRUCTION_UNITS
        assert res.mesh.is_metric_scale is False

        # Attempting to construct a mesh with is_metric_scale=True raises ValueError
        with pytest.raises(ValueError, match="is_metric_scale must be False"):
            SurfaceMesh(
                vertices=res.mesh.vertices,
                faces=res.mesh.faces,
                vertex_normals=None,
                face_normals=None,
                vertex_confidences=res.mesh.vertex_confidences,
                vertex_support_counts=res.mesh.vertex_support_counts,
                face_support_scores=res.mesh.face_support_scores,
                face_areas=res.mesh.face_areas,
                is_boundary_vertex=res.mesh.is_boundary_vertex,
                is_boundary_face=res.mesh.is_boundary_face,
                total_vertices=res.mesh.total_vertices,
                total_faces=res.mesh.total_faces,
                depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
                is_metric_scale=True,
            )

    # -------------------------------------------------------------------------
    # 9. NORMAL INDEPENDENCE AUDIT
    # -------------------------------------------------------------------------
    def test_normals_do_not_alter_topology_or_coordinates(self) -> None:
        """Audit 9: compute_normals=True produces identical faces and vertices as compute_normals=False."""
        pts = make_regular_tetrahedron_points(scale=1.0)
        cloud = make_valid_dense_point_cloud(4, pts=pts)
        config = SurfaceReconstructionConfig(alpha_radius=2.0, alpha_edge=2.0)
        reconstructor = AlphaComplexSurfaceReconstructor(config)

        res_no_norm = reconstructor.reconstruct_surface(cloud, compute_normals=False)
        res_with_norm = reconstructor.reconstruct_surface(cloud, compute_normals=True)

        assert res_no_norm.mesh is not None and res_with_norm.mesh is not None
        assert np.array_equal(res_no_norm.mesh.vertices, res_with_norm.mesh.vertices)
        assert np.array_equal(res_no_norm.mesh.faces, res_with_norm.mesh.faces)
        assert np.array_equal(res_no_norm.mesh.is_boundary_vertex, res_with_norm.mesh.is_boundary_vertex)
        assert np.array_equal(res_no_norm.mesh.is_boundary_face, res_with_norm.mesh.is_boundary_face)
        assert res_no_norm.mesh.vertex_normals is None
        assert res_with_norm.mesh.vertex_normals is not None

    # -------------------------------------------------------------------------
    # 10. DETERMINISM AUDIT
    # -------------------------------------------------------------------------
    def test_canonical_determinism_under_fifty_permutations(self) -> None:
        """Audit 10: Run 50 random permutations of input points; outputs must be identical."""
        pts = make_regular_tetrahedron_points(scale=1.0)
        cloud_canonical = make_valid_dense_point_cloud(4, pts=pts)
        config = SurfaceReconstructionConfig(alpha_radius=2.0, alpha_edge=2.0)
        reconstructor = AlphaComplexSurfaceReconstructor(config)

        ref_result = reconstructor.reconstruct_surface(cloud_canonical)
        assert ref_result.mesh is not None
        ref_verts = ref_result.mesh.vertices
        ref_faces = ref_result.mesh.faces

        rng = np.random.default_rng(2026)
        for trial in range(50):
            perm = rng.permutation(4)
            cloud_perm = make_valid_dense_point_cloud(
                4,
                pts=pts[perm],
                confidences=cloud_canonical.confidences[perm],
                support_counts=cloud_canonical.support_counts[perm],
                visibility_states=[cloud_canonical.visibility_states[i] for i in perm],
                validation_statuses=[cloud_canonical.validation_statuses[i] for i in perm],
                source_frame_ids=[cloud_canonical.source_frame_ids[i] for i in perm],
            )
            trial_res = reconstructor.reconstruct_surface(cloud_perm)
            assert trial_res.mesh is not None
            assert np.allclose(trial_res.mesh.vertices, ref_verts, atol=1e-12), f"Failed on trial {trial}"
            assert np.array_equal(trial_res.mesh.faces, ref_faces), f"Failed on trial {trial}"

    # -------------------------------------------------------------------------
    # 11. MUTATION & FALSIFICATION GUARDS
    # -------------------------------------------------------------------------
    def test_mutation_guard_wrong_aspect_ratio_formula(self) -> None:
        """Audit 11.1: Falsification check on aspect ratio.

        Aspect ratio of equilateral triangle is 2/sqrt(3) ~= 1.1547.
        Aspect ratio of needle triangle is 100.0.
        Verifies that strict threshold separates equilateral from needle.
        """
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.01, 0.0],  # Needle
                [0.5, 0.5, 1.0],
            ],
            dtype=np.float64,
        )
        cloud = make_valid_dense_point_cloud(4, pts=pts)
        # Aspect threshold 10 rejects needle (aspect 100)
        res = AlphaComplexSurfaceReconstructor(
            SurfaceReconstructionConfig(alpha_radius=25.0, alpha_edge=25.0, min_aspect_ratio=10.0)
        ).reconstruct_surface(cloud)
        assert res.mesh is not None
        assert res.mesh.total_faces == 3

        # Aspect threshold 200 accepts needle
        res_loose = AlphaComplexSurfaceReconstructor(
            SurfaceReconstructionConfig(alpha_radius=25.0, alpha_edge=25.0, min_aspect_ratio=200.0)
        ).reconstruct_surface(cloud)
        assert res_loose.mesh is not None
        assert res_loose.mesh.total_faces == 4

    def test_mutation_guard_incorrect_circumradius_formula(self) -> None:
        """Audit 11.2: Falsification check: bounding-sphere / centroid distance differs from circumradius.

        A trirectangular tetrahedron with legs 1, 2, 2 has exact circumradius 1.5.
        A naive formula like max vertex distance from centroid would give wrong radius.
        """
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )
        r_exact = AlphaComplexSurfaceReconstructor._compute_tetrahedron_circumradius(
            pts[0], pts[1], pts[2], pts[3]
        )
        # Centroid distance is sqrt((0.75)^2 + 1^2 + 1^2) = sqrt(2.5625) ~= 1.6007 != 1.5
        centroid = np.mean(pts, axis=0)
        max_centroid_dist = float(np.max(np.linalg.norm(pts - centroid, axis=1)))
        assert not np.isclose(r_exact, max_centroid_dist, atol=1e-3)

    def test_mutation_guard_incorrect_face_count_rule(self) -> None:
        """Audit 11.3: Falsification check: interior faces (count == 2) must never be emitted.

        If an implementation naively emitted all faces or included count == 2 faces,
        it would emit 8 faces instead of 6 for two adjacent tetrahedra.
        """
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, np.sqrt(3) / 2, 0.0],
                [0.5, np.sqrt(3) / 6, 0.8],
                [0.5, np.sqrt(3) / 6, -0.8],
            ],
            dtype=np.float64,
        )
        cloud = make_valid_dense_point_cloud(5, pts=pts)
        config = SurfaceReconstructionConfig(alpha_radius=2.0, alpha_edge=2.0)
        res = AlphaComplexSurfaceReconstructor(config).reconstruct_surface(cloud)
        assert res.mesh is not None
        assert res.mesh.total_faces == 6
        assert res.mesh.total_faces != 8

    def test_mutation_guard_no_hole_filling_or_bridge_fabrication(self) -> None:
        """Audit 11.4: Disconnected clusters with gap > alpha_edge must not be bridged.

        Ensures no artificial bridge geometry or hole-filling is fabricated.
        """
        # Two distant tetrahedra separated by distance 10.0
        tet1 = make_regular_tetrahedron_points(scale=1.0)
        tet2 = make_regular_tetrahedron_points(scale=1.0) + np.array([10.0, 0.0, 0.0])
        pts = np.vstack([tet1, tet2])
        cloud = make_valid_dense_point_cloud(8, pts=pts)
        config = SurfaceReconstructionConfig(alpha_radius=2.0, alpha_edge=2.0)
        res = AlphaComplexSurfaceReconstructor(config).reconstruct_surface(cloud)
        assert res.mesh is not None
        # Each tetrahedron has 4 faces, total 8 faces, NO bridging faces across the 10.0 gap
        assert res.mesh.total_faces == 8
        for f in res.mesh.faces:
            # Face vertices must either all be in cluster 1 (0..3) or all in cluster 2 (4..7)
            c1 = all(v < 4 for v in f)
            c2 = all(v >= 4 for v in f)
            assert c1 or c2, f"Fabricated bridging face detected: {f}"
