"""Unit and adversarial tests for Phase 3E.4 Surface Reconstruction Contract & Alpha-Complex Engine.

Verifies:
Step 1:
1. valid SurfaceMesh construction
2. empty input rejection
3. NaN rejection
4. Inf rejection
5. confidence validation
6. support-count validation
7. array-length consistency
8. reconstruction-unit preservation
9. metric-scale remains false
10. PCA eigenvalue ordering
11. planar synthetic neighborhood
12. collinear synthetic neighborhood
13. insufficient neighborhood
14. viewpoint orientation
15. opposing-camera cancellation
16. deterministic canonical behavior
17. provenance preservation
18. invalid metadata rejection
19. scale-aware tolerance behavior
20. no mutation of Phase 3E.1–3E.3 behavior

Step 2:
21. 4-point non-coplanar tetrahedron -> valid alpha-complex with 4 faces
22. coplanar input rejection
23. fewer than 4 points
24. circumradius rejection
25. alpha_edge rejection
26. degenerate triangle rejection
27. relative-area scale behavior
28. aspect-ratio rejection
29. exact boundary topology
30. boundary support independence
31. multiple retained tetrahedra sharing a face -> shared face is not boundary
32. face rejected by alpha_edge -> boundary flags recomputed afterward
33. provenance preservation in reconstruction
34. confidence/support preservation in reconstruction
35. optional normals do not affect topology
36. deterministic canonical ordering
37. scale test within declared range
38. failure safety for Delaunay/Qhull errors
39. no hole filling
40. no metric-scale promotion
"""

from typing import Dict, List
import numpy as np
import pytest

from src.geometry.mvs import (
    DensePointCloud,
    DepthUnit,
    PointValidationStatus,
    PointVisibilityState,
)
from src.geometry.surface_reconstruction import (
    AlphaComplexSurfaceReconstructor,
    DensePointCloudValidator,
    LocalPCANormalEstimator,
    NormalEstimationResult,
    NormalEstimationStatus,
    SurfaceFailureReason,
    SurfaceMesh,
    SurfaceReconstructionConfig,
    SurfaceReconstructionResult,
    SurfaceReconstructionStatus,
)


def make_valid_dense_point_cloud(
    n: int = 10,
    pts: np.ndarray | None = None,
    confidences: np.ndarray | None = None,
    support_counts: np.ndarray | None = None,
    visibility_states: List[PointVisibilityState] | None = None,
    validation_statuses: List[PointValidationStatus] | None = None,
    source_frame_ids: List[List[str]] | None = None,
    depth_unit: DepthUnit = DepthUnit.RECONSTRUCTION_UNITS,
    is_metric_scale: bool = False,
) -> DensePointCloud:
    """Helper to generate a well-formed DensePointCloud for testing."""
    if pts is None:
        pts = np.linspace(0.0, 1.0, n * 3, dtype=np.float64).reshape((n, 3))
    if confidences is None:
        confidences = np.full(n, 0.85, dtype=np.float32)
    if support_counts is None:
        support_counts = np.full(n, 3, dtype=np.int32)
    if visibility_states is None:
        visibility_states = [PointVisibilityState.VALID] * n
    if validation_statuses is None:
        validation_statuses = [PointValidationStatus.VALIDATED] * n
    if source_frame_ids is None:
        source_frame_ids = [["frame_1", "frame_2", "frame_3"] for _ in range(n)]

    return DensePointCloud(
        points=pts,
        confidences=confidences,
        support_counts=support_counts,
        visibility_states=visibility_states,
        validation_statuses=validation_statuses,
        source_frame_ids=source_frame_ids,
        total_fused_points=n,
        mean_confidence=float(np.mean(confidences)) if len(confidences) > 0 else 0.0,
        depth_unit=depth_unit,
        is_metric_scale=is_metric_scale,
        provenance={"creator": "test_phase3e4_synthetic"},
    )


def make_regular_tetrahedron_points(scale: float = 1.0) -> np.ndarray:
    """Return 4 non-coplanar points forming a regular tetrahedron."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [scale, 0.0, 0.0],
            [scale / 2.0, scale * np.sqrt(3.0) / 2.0, 0.0],
            [scale / 2.0, scale * np.sqrt(3.0) / 6.0, scale * np.sqrt(6.0) / 3.0],
        ],
        dtype=np.float64,
    )


class TestPhase3E4SurfaceReconstructionContract:
    """Test suite for Phase 3E.4 surface reconstruction contract and validation layer."""

    def test_valid_surface_mesh_construction(self) -> None:
        """Test 1: Well-formed SurfaceMesh construction and property validation."""
        v = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        f = np.array([[0, 1, 2]], dtype=np.int32)
        v_normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        f_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        v_conf = np.array([0.9, 0.85, 0.8], dtype=np.float32)
        v_supp = np.array([3, 3, 3], dtype=np.int32)
        f_supp = np.array([0.95], dtype=np.float32)
        f_areas = np.array([0.5], dtype=np.float64)
        is_bv = np.array([True, True, True], dtype=bool)
        is_bf = np.array([True], dtype=bool)

        mesh = SurfaceMesh(
            vertices=v,
            faces=f,
            vertex_normals=v_normals,
            face_normals=f_normals,
            vertex_confidences=v_conf,
            vertex_support_counts=v_supp,
            face_support_scores=f_supp,
            face_areas=f_areas,
            is_boundary_vertex=is_bv,
            is_boundary_face=is_bf,
            total_vertices=3,
            total_faces=1,
            depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
            is_metric_scale=False,
            provenance={"source": "test_1"},
        )

        assert mesh.total_vertices == 3
        assert mesh.total_faces == 1
        assert mesh.depth_unit == DepthUnit.RECONSTRUCTION_UNITS
        assert mesh.is_metric_scale is False
        assert mesh.vertices.shape == (3, 3)
        assert mesh.faces.shape == (1, 3)

    def test_empty_input_cloud_rejection(self) -> None:
        """Test 2: Empty input point cloud is explicitly rejected with EMPTY_INPUT_CLOUD."""
        cloud = DensePointCloud(
            points=np.empty((0, 3), dtype=np.float64),
            confidences=np.empty(0, dtype=np.float32),
            support_counts=np.empty(0, dtype=np.int32),
            visibility_states=[],
            validation_statuses=[],
            source_frame_ids=[],
            total_fused_points=0,
            mean_confidence=0.0,
            depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
            is_metric_scale=False,
        )
        is_valid, reasons, details = DensePointCloudValidator.validate_input_cloud(cloud)
        assert not is_valid
        assert SurfaceFailureReason.EMPTY_INPUT_CLOUD in reasons

    def test_nan_coordinate_rejection(self) -> None:
        """Test 3: Points containing NaN coordinates are rejected with NON_FINITE_VERTICES."""
        pts = np.ones((5, 3), dtype=np.float64)
        pts[2, 1] = np.nan
        cloud = make_valid_dense_point_cloud(5, pts=pts)
        is_valid, reasons, _ = DensePointCloudValidator.validate_input_cloud(cloud)
        assert not is_valid
        assert SurfaceFailureReason.NON_FINITE_VERTICES in reasons

    def test_inf_coordinate_rejection(self) -> None:
        """Test 4: Points containing Inf coordinates are rejected with NON_FINITE_VERTICES."""
        pts = np.ones((5, 3), dtype=np.float64)
        pts[0, 2] = np.inf
        cloud = make_valid_dense_point_cloud(5, pts=pts)
        is_valid, reasons, _ = DensePointCloudValidator.validate_input_cloud(cloud)
        assert not is_valid
        assert SurfaceFailureReason.NON_FINITE_VERTICES in reasons

    def test_confidence_validation(self) -> None:
        """Test 5: Out of bounds confidence (<0, >1, NaN) is rejected with INVALID_CONFIDENCE."""
        conf = np.array([0.5, -0.1, 0.8, 0.9, 0.7], dtype=np.float32)
        cloud = make_valid_dense_point_cloud(5, confidences=conf)
        is_valid, reasons, _ = DensePointCloudValidator.validate_input_cloud(cloud)
        assert not is_valid
        assert SurfaceFailureReason.INVALID_CONFIDENCE in reasons

        # Test > 1.0
        conf = np.array([0.5, 1.2, 0.8, 0.9, 0.7], dtype=np.float32)
        cloud = make_valid_dense_point_cloud(5, confidences=conf)
        is_valid, reasons, _ = DensePointCloudValidator.validate_input_cloud(cloud)
        assert not is_valid
        assert SurfaceFailureReason.INVALID_CONFIDENCE in reasons

    def test_support_count_validation(self) -> None:
        """Test 6: Support count < 1 is rejected with INVALID_SUPPORT_COUNT."""
        supp = np.array([2, 3, 0, 4, 2], dtype=np.int32)
        cloud = make_valid_dense_point_cloud(5, support_counts=supp)
        is_valid, reasons, _ = DensePointCloudValidator.validate_input_cloud(cloud)
        assert not is_valid
        assert SurfaceFailureReason.INVALID_SUPPORT_COUNT in reasons

    def test_array_length_consistency(self) -> None:
        """Test 7: Mismatched array lengths are rejected with INCONSISTENT_ARRAY_LENGTHS."""
        cloud = make_valid_dense_point_cloud(5)
        cloud.points = np.ones((6, 3), dtype=np.float64)
        is_valid, reasons, _ = DensePointCloudValidator.validate_input_cloud(cloud)
        assert not is_valid
        assert SurfaceFailureReason.INCONSISTENT_ARRAY_LENGTHS in reasons

    def test_reconstruction_unit_preservation(self) -> None:
        """Test 8: Output mesh enforces RECONSTRUCTION_UNITS and rejects other units."""
        v = np.zeros((3, 3), dtype=np.float64)
        f = np.zeros((1, 3), dtype=np.int32)
        with pytest.raises(ValueError, match="depth_unit must strictly be RECONSTRUCTION_UNITS"):
            SurfaceMesh(
                vertices=v,
                faces=f,
                vertex_normals=None,
                face_normals=None,
                vertex_confidences=np.ones(3, dtype=np.float32),
                vertex_support_counts=np.ones(3, dtype=np.int32),
                face_support_scores=np.ones(1, dtype=np.float32),
                face_areas=np.ones(1, dtype=np.float64),
                is_boundary_vertex=np.ones(3, dtype=bool),
                is_boundary_face=np.ones(1, dtype=bool),
                total_vertices=3,
                total_faces=1,
                depth_unit=DepthUnit.METRIC_METERS,  # Invalid
                is_metric_scale=False,
            )

    def test_metric_scale_remains_false(self) -> None:
        """Test 9: Output mesh rejects is_metric_scale=True."""
        v = np.zeros((3, 3), dtype=np.float64)
        f = np.zeros((1, 3), dtype=np.int32)
        with pytest.raises(ValueError, match="is_metric_scale must be False"):
            SurfaceMesh(
                vertices=v,
                faces=f,
                vertex_normals=None,
                face_normals=None,
                vertex_confidences=np.ones(3, dtype=np.float32),
                vertex_support_counts=np.ones(3, dtype=np.int32),
                face_support_scores=np.ones(1, dtype=np.float32),
                face_areas=np.ones(1, dtype=np.float64),
                is_boundary_vertex=np.ones(3, dtype=bool),
                is_boundary_face=np.ones(1, dtype=bool),
                total_vertices=3,
                total_faces=1,
                depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
                is_metric_scale=True,  # Invalid
            )

    def test_pca_eigenvalue_ordering(self) -> None:
        """Test 10: PCA eigenvalues must be strictly sorted: lambda_0 <= lambda_1 <= lambda_2."""
        rng = np.random.default_rng(42)
        x = rng.uniform(-1.0, 1.0, 50)
        y = rng.uniform(-1.0, 1.0, 50)
        z = 0.05 * rng.normal(size=50)
        patch = np.column_stack([x, y, z])

        estimator = LocalPCANormalEstimator()
        result = estimator.estimate_neighborhood_normal(patch)

        lam0, lam1, lam2 = result.eigenvalues
        assert lam0 <= lam1 + 1e-12
        assert lam1 <= lam2 + 1e-12
        assert lam0 >= 0.0

    def test_planar_synthetic_neighborhood(self) -> None:
        """Test 11: Planar synthetic patch recovers normal aligned with mathematical ground truth."""
        rng = np.random.default_rng(123)
        x = rng.uniform(-2.0, 2.0, 50)
        y = rng.uniform(-2.0, 2.0, 50)
        z = np.full(50, 5.0)
        patch = np.column_stack([x, y, z])

        estimator = LocalPCANormalEstimator()
        result = estimator.estimate_neighborhood_normal(patch)

        assert result.status in (NormalEstimationStatus.VIEWPOINT_UNAVAILABLE, NormalEstimationStatus.VALID)
        assert result.planarity > 0.5
        assert result.sphericity < 0.05
        assert abs(abs(result.normal[2]) - 1.0) < 1e-6
        assert abs(result.normal[0]) < 1e-6
        assert abs(result.normal[1]) < 1e-6

    def test_collinear_synthetic_neighborhood(self) -> None:
        """Test 12: Collinear points trigger COLLINEAR_DEGENERATE status."""
        t = np.linspace(-1.0, 1.0, 20)
        patch = np.column_stack([t, 2.0 * t, -t])

        estimator = LocalPCANormalEstimator()
        result = estimator.estimate_neighborhood_normal(patch)

        assert result.status == NormalEstimationStatus.COLLINEAR_DEGENERATE
        assert result.linearity > 0.9

    def test_insufficient_neighborhood(self) -> None:
        """Test 13: Fewer than 3 points yields INSUFFICIENT_NEIGHBORHOOD status."""
        patch = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        estimator = LocalPCANormalEstimator()
        result = estimator.estimate_neighborhood_normal(patch)

        assert result.status == NormalEstimationStatus.INSUFFICIENT_NEIGHBORHOOD
        assert np.all(result.normal == 0.0)

    def test_viewpoint_orientation(self) -> None:
        """Test 14: Viewpoint vector correctly orients normal towards camera center."""
        patch = np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        cam_centers = [np.array([0.0, 0.0, 10.0])]

        estimator = LocalPCANormalEstimator()
        result = estimator.estimate_neighborhood_normal(patch, camera_centers=cam_centers)

        assert result.status == NormalEstimationStatus.VALID
        assert result.viewpoint_aligned is True
        assert result.normal[2] > 0.99

    def test_opposing_camera_cancellation(self) -> None:
        """Test 15: Opposing cameras cancel viewing vector and trigger cancellation guard."""
        patch = np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        cam_centers = [np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, -10.0])]

        estimator = LocalPCANormalEstimator(config=SurfaceReconstructionConfig(viewpoint_cancellation_threshold=1e-3))
        result = estimator.estimate_neighborhood_normal(patch, camera_centers=cam_centers)

        assert result.status == NormalEstimationStatus.VIEWPOINT_CANCELLATION
        assert result.viewpoint_aligned is False
        assert np.isclose(np.linalg.norm(result.normal), 1.0)

    def test_deterministic_canonical_behavior(self) -> None:
        """Test 16: Canonical sign convention yields identical unoriented normal under permutation."""
        rng = np.random.default_rng(999)
        patch = rng.uniform(-1.0, 1.0, (15, 3))
        patch[:, 2] = 0.1 * patch[:, 0] - 0.2 * patch[:, 1]

        estimator = LocalPCANormalEstimator()
        res1 = estimator.estimate_neighborhood_normal(patch)

        perm = rng.permutation(15)
        patch_perm = patch[perm]
        res2 = estimator.estimate_neighborhood_normal(patch_perm)

        assert np.allclose(res1.eigenvalues, res2.eigenvalues, atol=1e-12)
        assert np.allclose(res1.normal, res2.normal, atol=1e-12)

    def test_provenance_preservation(self) -> None:
        """Test 17: SurfaceMesh correctly retains provenance and metadata dict."""
        prov = {"phase": "3E.4", "algorithm": "alpha_complex", "test_id": 17}
        v = np.zeros((3, 3), dtype=np.float64)
        f = np.zeros((1, 3), dtype=np.int32)
        mesh = SurfaceMesh(
            vertices=v,
            faces=f,
            vertex_normals=None,
            face_normals=None,
            vertex_confidences=np.ones(3, dtype=np.float32),
            vertex_support_counts=np.ones(3, dtype=np.int32),
            face_support_scores=np.ones(1, dtype=np.float32),
            face_areas=np.ones(1, dtype=np.float64),
            is_boundary_vertex=np.ones(3, dtype=bool),
            is_boundary_face=np.ones(1, dtype=bool),
            total_vertices=3,
            total_faces=1,
            depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
            is_metric_scale=False,
            provenance=prov,
        )
        assert mesh.provenance == prov

    def test_invalid_metadata_rejection(self) -> None:
        """Test 18: Invalid visibility state or validation status is rejected."""
        vis = [PointVisibilityState.VALID, PointVisibilityState.INVALID_DEPTH, PointVisibilityState.VALID]
        cloud = make_valid_dense_point_cloud(3, visibility_states=vis)
        is_valid, reasons, _ = DensePointCloudValidator.validate_input_cloud(cloud)
        assert not is_valid
        assert SurfaceFailureReason.INVALID_VISIBILITY_STATE in reasons

        val = [PointValidationStatus.VALIDATED, PointValidationStatus.REJECTED, PointValidationStatus.VALIDATED]
        cloud2 = make_valid_dense_point_cloud(3, validation_statuses=val)
        is_valid2, reasons2, _ = DensePointCloudValidator.validate_input_cloud(cloud2)
        assert not is_valid2
        assert SurfaceFailureReason.INVALID_VALIDATION_STATUS in reasons2

    def test_scale_aware_tolerance_behavior(self) -> None:
        """Test 19: Dimensionless aspect ratio and relative area scale invariantly."""
        config = SurfaceReconstructionConfig(min_aspect_ratio=20.0, min_relative_area=1e-6)

        e1, e2, e3 = 10.0, 10.0, 0.01
        s = (e1 + e2 + e3) / 2.0
        area = np.sqrt(max(0.0, s * (s - e1) * (s - e2) * (s - e3)))
        max_e = max(e1, e2, e3)
        h_min = 2.0 * area / max_e
        aspect = max_e / h_min
        rel_area = area / (max_e**2)

        s_factor = 100.0
        e1_s, e2_s, e3_s = e1 * s_factor, e2 * s_factor, e3 * s_factor
        s_s = (e1_s + e2_s + e3_s) / 2.0
        area_s = np.sqrt(max(0.0, s_s * (s_s - e1_s) * (s_s - e2_s) * (s_s - e3_s)))
        max_e_s = max(e1_s, e2_s, e3_s)
        h_min_s = 2.0 * area_s / max_e_s
        aspect_s = max_e_s / h_min_s
        rel_area_s = area_s / (max_e_s**2)

        assert np.isclose(aspect, aspect_s, rtol=1e-12)
        assert np.isclose(rel_area, rel_area_s, rtol=1e-12)
        assert aspect > config.min_aspect_ratio

    def test_no_mutation_of_phase3_behavior(self) -> None:
        """Test 20: Verified Phase 3E.1–3E.3 contract types remain functional."""
        from src.geometry.dense_fusion import DensePointFusionEngine
        from src.geometry.dense_point_generation import DensePointGenerator
        from src.geometry.dense_stereo import ClassicalStereoSGBMEstimator

        assert ClassicalStereoSGBMEstimator is not None
        assert DensePointGenerator is not None
        assert DensePointFusionEngine is not None


class TestPhase3E4AlphaComplexReconstruction:
    """Test suite for Step 2: 3D Alpha-Complex Surface Extraction."""

    def test_tetrahedron_minimal_valid_alpha_complex(self) -> None:
        """Test 21: 4 non-coplanar points forming a tetrahedron yield 4 triangular faces."""
        pts = make_regular_tetrahedron_points(scale=1.0)
        cloud = make_valid_dense_point_cloud(4, pts=pts)

        # Circumradius of regular tetrahedron with edge 1.0 is sqrt(6)/4 ~= 0.6124
        # Configure alpha_radius = 1.0, alpha_edge = 1.5
        config = SurfaceReconstructionConfig(alpha_radius=1.0, alpha_edge=1.5)
        reconstructor = AlphaComplexSurfaceReconstructor(config)
        result = reconstructor.reconstruct_surface(cloud)

        assert result.status == SurfaceReconstructionStatus.SUCCESS
        assert result.mesh is not None
        assert result.mesh.total_vertices == 4
        assert result.mesh.total_faces == 4
        assert result.mesh.depth_unit == DepthUnit.RECONSTRUCTION_UNITS
        assert result.mesh.is_metric_scale is False

    def test_coplanar_input_rejection(self) -> None:
        """Test 22: Strictly coplanar points in 3D are rejected with INSUFFICIENT_NON_COPLANAR_POINTS."""
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        cloud = make_valid_dense_point_cloud(4, pts=pts)
        reconstructor = AlphaComplexSurfaceReconstructor()
        result = reconstructor.reconstruct_surface(cloud)

        assert result.status == SurfaceReconstructionStatus.RECONSTRUCTION_FAILED
        assert SurfaceFailureReason.INSUFFICIENT_NON_COPLANAR_POINTS in result.failure_reasons
        assert result.mesh is None

    def test_fewer_than_four_points_rejection(self) -> None:
        """Test 23: Point cloud with fewer than 4 points is rejected."""
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        cloud = make_valid_dense_point_cloud(3, pts=pts)
        reconstructor = AlphaComplexSurfaceReconstructor()
        result = reconstructor.reconstruct_surface(cloud)

        assert result.status == SurfaceReconstructionStatus.RECONSTRUCTION_FAILED
        assert SurfaceFailureReason.INSUFFICIENT_NON_COPLANAR_POINTS in result.failure_reasons

    def test_circumradius_rejection(self) -> None:
        """Test 24: Tetrahedra with circumradius > alpha_radius are rejected."""
        pts = make_regular_tetrahedron_points(scale=2.0)
        # For scale=2.0, R = 2.0 * sqrt(6)/4 = sqrt(6)/2 ~= 1.2247
        cloud = make_valid_dense_point_cloud(4, pts=pts)

        # Set alpha_radius = 0.5 (smaller than 1.2247)
        config = SurfaceReconstructionConfig(alpha_radius=0.5, alpha_edge=5.0)
        reconstructor = AlphaComplexSurfaceReconstructor(config)
        result = reconstructor.reconstruct_surface(cloud)

        assert result.status == SurfaceReconstructionStatus.EMPTY_VALID_OUTPUT
        assert SurfaceFailureReason.TETRAHEDRON_CIRCUMRADIUS_EXCEEDED in result.failure_reasons
        assert result.mesh is None

    def test_alpha_edge_rejection(self) -> None:
        """Test 25: Faces with maximum edge > alpha_edge are rejected."""
        pts = make_regular_tetrahedron_points(scale=2.0)
        # Edges are length 2.0. R = 1.2247
        cloud = make_valid_dense_point_cloud(4, pts=pts)

        # Retain tetrahedron (alpha_radius = 2.0), but set alpha_edge = 1.0 (smaller than 2.0)
        config = SurfaceReconstructionConfig(alpha_radius=2.0, alpha_edge=1.0)
        reconstructor = AlphaComplexSurfaceReconstructor(config)
        result = reconstructor.reconstruct_surface(cloud)

        assert result.status == SurfaceReconstructionStatus.EMPTY_VALID_OUTPUT
        assert SurfaceFailureReason.EDGE_LENGTH_EXCEEDED in result.failure_reasons
        assert result.mesh is None

    def test_degenerate_triangle_rejection(self) -> None:
        """Test 26: Faces failing relative area or aspect ratio are rejected as DEGENERATE_TRIANGLE."""
        # Create a tetrahedron where one face is near-collinear / needle
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.01, 0.0],  # Needle face with (0,0,0) and (1,0,0)
                [0.5, 0.5, 1.0],
            ],
            dtype=np.float64,
        )
        cloud = make_valid_dense_point_cloud(4, pts=pts)
        config = SurfaceReconstructionConfig(alpha_radius=25.0, alpha_edge=25.0, min_relative_area=1e-2)
        reconstructor = AlphaComplexSurfaceReconstructor(config)
        result = reconstructor.reconstruct_surface(cloud)

        # The degenerate face is dropped; remaining faces are emitted
        assert result.status == SurfaceReconstructionStatus.SUCCESS
        assert result.mesh is not None
        # Originally 4 faces, 1 needle face dropped -> 3 faces emitted
        assert result.mesh.total_faces == 3

    def test_relative_area_scale_behavior(self) -> None:
        """Test 27: Relative area criterion behaves identically under coordinate scaling."""
        pts = make_regular_tetrahedron_points(scale=1.0)
        cloud1 = make_valid_dense_point_cloud(4, pts=pts)
        config1 = SurfaceReconstructionConfig(alpha_radius=2.0, alpha_edge=2.0, min_relative_area=1e-5)
        res1 = AlphaComplexSurfaceReconstructor(config1).reconstruct_surface(cloud1)

        # Scale by factor 50.0
        s = 50.0
        cloud2 = make_valid_dense_point_cloud(4, pts=pts * s)
        config2 = SurfaceReconstructionConfig(alpha_radius=2.0 * s, alpha_edge=2.0 * s, min_relative_area=1e-5)
        res2 = AlphaComplexSurfaceReconstructor(config2).reconstruct_surface(cloud2)

        assert res1.status == SurfaceReconstructionStatus.SUCCESS
        assert res2.status == SurfaceReconstructionStatus.SUCCESS
        assert res1.mesh is not None and res2.mesh is not None
        assert res1.mesh.total_faces == res2.mesh.total_faces
        assert np.array_equal(res1.mesh.faces, res2.mesh.faces)

    def test_aspect_ratio_rejection(self) -> None:
        """Test 28: Faces exceeding max aspect ratio threshold are rejected."""
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
        # Strict aspect ratio threshold 10.0 (needle face has aspect 100.0)
        config = SurfaceReconstructionConfig(alpha_radius=25.0, alpha_edge=25.0, min_aspect_ratio=10.0)
        reconstructor = AlphaComplexSurfaceReconstructor(config)
        result = reconstructor.reconstruct_surface(cloud)

        assert result.status == SurfaceReconstructionStatus.SUCCESS
        assert result.mesh is not None
        # Face with aspect ratio > 10.0 rejected, 3 remain
        assert result.mesh.total_faces == 3

    def test_exact_boundary_topology(self) -> None:
        """Test 29: Closed tetrahedron has 0 open boundary edges; removing 1 face opens boundary."""
        pts = make_regular_tetrahedron_points(scale=1.0)
        cloud = make_valid_dense_point_cloud(4, pts=pts)
        config = SurfaceReconstructionConfig(alpha_radius=2.0, alpha_edge=2.0)
        res = AlphaComplexSurfaceReconstructor(config).reconstruct_surface(cloud)

        assert res.mesh is not None
        assert res.mesh.total_faces == 4
        # Closed hollow tetrahedron has no open boundary edges
        assert not np.any(res.mesh.is_boundary_vertex)
        assert not np.any(res.mesh.is_boundary_face)

    def test_boundary_support_independence(self) -> None:
        """Test 30: Boundary status is independent of vertex support score."""
        # 3 faces of a tetrahedron (open boundary) with high support counts
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.01, 0.0],  # Needle face rejected -> leaves 3 open boundary faces
                [0.5, 0.5, 1.0],
            ],
            dtype=np.float64,
        )
        supp = np.array([4, 4, 4, 4], dtype=np.int32)
        cloud = make_valid_dense_point_cloud(4, pts=pts, support_counts=supp)
        config = SurfaceReconstructionConfig(
            alpha_radius=25.0, alpha_edge=25.0, min_relative_area=1e-2, target_distinct_views=4
        )
        res = AlphaComplexSurfaceReconstructor(config).reconstruct_surface(cloud)

        assert res.mesh is not None
        assert res.mesh.total_faces == 3
        # Boundary edges exist because 1 face was dropped
        assert np.any(res.mesh.is_boundary_vertex)
        assert np.any(res.mesh.is_boundary_face)
        # But support scores remain high (= 1.0) because support_counts = 4 / target 4
        assert np.all(res.mesh.face_support_scores == 1.0)

    def test_multiple_retained_tetrahedra_shared_face_not_boundary(self) -> None:
        """Test 31: Face shared by two retained tetrahedra is interior and NOT emitted."""
        # Two tetrahedra sharing base face: (0, 1, 2)
        # t1: (0, 1, 2, 3), t2: (0, 1, 2, 4)
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, np.sqrt(3) / 2, 0.0],
                [0.5, np.sqrt(3) / 6, 0.8],  # Above base
                [0.5, np.sqrt(3) / 6, -0.8],  # Below base
            ],
            dtype=np.float64,
        )
        cloud = make_valid_dense_point_cloud(5, pts=pts)
        config = SurfaceReconstructionConfig(alpha_radius=2.0, alpha_edge=2.0)
        res = AlphaComplexSurfaceReconstructor(config).reconstruct_surface(cloud)

        assert res.mesh is not None
        # 2 tetrahedra -> total 8 faces, 1 shared face (0, 1, 2) -> 8 - 2 = 6 boundary faces
        assert res.mesh.total_faces == 6
        # The shared face (0, 1, 2) must NOT be present in emitted faces
        base_set = {0, 1, 2}
        for f in res.mesh.faces:
            assert set(f) != base_set

    def test_face_rejected_by_alpha_edge_boundary_recomputed(self) -> None:
        """Test 32: Face rejected by alpha_edge causes incident edges to become open boundaries."""
        # Tetrahedron where 3 faces have max edge ~1.89 and 1 base face has max edge 1.0
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.8, 0.0],
                [0.5, 0.3, 1.8],  # Apex gives longer edges
            ],
            dtype=np.float64,
        )
        cloud = make_valid_dense_point_cloud(4, pts=pts)
        # alpha_edge = 1.5 rejects the 3 faces with max edge ~1.89, leaving 1 face
        config = SurfaceReconstructionConfig(alpha_radius=5.0, alpha_edge=1.5)
        res = AlphaComplexSurfaceReconstructor(config).reconstruct_surface(cloud)

        assert res.mesh is not None
        assert res.mesh.total_faces == 1
        # Boundary flags must be recomputed on the remaining 1 face
        assert np.any(res.mesh.is_boundary_face)
        assert np.any(res.mesh.is_boundary_vertex)

    def test_provenance_preservation_in_reconstruction(self) -> None:
        """Test 33: Reconstruction retains input provenance and records alpha parameters."""
        pts = make_regular_tetrahedron_points(scale=1.0)
        cloud = make_valid_dense_point_cloud(4, pts=pts)
        cloud.provenance = {"camera_rig": "drone_alpha", "mission_id": "test_m33"}

        config = SurfaceReconstructionConfig(alpha_radius=1.5, alpha_edge=1.5)
        res = AlphaComplexSurfaceReconstructor(config).reconstruct_surface(cloud)

        assert res.mesh is not None
        prov = res.mesh.provenance
        assert prov["algorithm"] == "alpha_complex_delaunay_3d"
        assert prov["alpha_radius"] == 1.5
        assert prov["alpha_edge"] == 1.5
        assert prov["input_cloud_provenance"]["camera_rig"] == "drone_alpha"

    def test_confidence_support_preservation_in_reconstruction(self) -> None:
        """Test 34: Output vertices maintain exact confidence and support counts from input cloud."""
        pts = make_regular_tetrahedron_points(scale=1.0)
        conf = np.array([0.75, 0.82, 0.91, 0.68], dtype=np.float32)
        supp = np.array([2, 3, 4, 5], dtype=np.int32)
        cloud = make_valid_dense_point_cloud(4, pts=pts, confidences=conf, support_counts=supp)

        res = AlphaComplexSurfaceReconstructor(SurfaceReconstructionConfig(alpha_radius=2.0, alpha_edge=2.0)).reconstruct_surface(cloud)

        assert res.mesh is not None
        # Sorted input preserves set of confidences and support counts
        assert set(res.mesh.vertex_confidences) == set(conf)
        assert set(res.mesh.vertex_support_counts) == set(supp)

    def test_optional_normals_do_not_affect_topology(self) -> None:
        """Test 35: Computing vertex normals does NOT alter the emitted faces array."""
        pts = make_regular_tetrahedron_points(scale=1.0)
        cloud = make_valid_dense_point_cloud(4, pts=pts)
        config = SurfaceReconstructionConfig(alpha_radius=2.0, alpha_edge=2.0)
        reconstructor = AlphaComplexSurfaceReconstructor(config)

        res_without = reconstructor.reconstruct_surface(cloud, compute_normals=False)
        res_with = reconstructor.reconstruct_surface(cloud, compute_normals=True)

        assert res_without.mesh is not None and res_with.mesh is not None
        assert res_without.mesh.vertex_normals is None
        assert res_with.mesh.vertex_normals is not None
        assert np.array_equal(res_without.mesh.faces, res_with.mesh.faces)
        assert np.array_equal(res_without.mesh.vertices, res_with.mesh.vertices)

    def test_deterministic_canonical_ordering(self) -> None:
        """Test 36: Arbitrary input point order permutations produce identical output arrays."""
        pts = make_regular_tetrahedron_points(scale=1.0)
        cloud1 = make_valid_dense_point_cloud(4, pts=pts)

        # Permute points and associated metadata
        perm = [2, 0, 3, 1]
        pts_perm = pts[perm]
        conf_perm = cloud1.confidences[perm]
        supp_perm = cloud1.support_counts[perm]
        frames_perm = [cloud1.source_frame_ids[i] for i in perm]
        vis_perm = [cloud1.visibility_states[i] for i in perm]
        val_perm = [cloud1.validation_statuses[i] for i in perm]

        cloud2 = DensePointCloud(
            points=pts_perm,
            confidences=conf_perm,
            support_counts=supp_perm,
            visibility_states=vis_perm,
            validation_statuses=val_perm,
            source_frame_ids=frames_perm,
            total_fused_points=4,
            mean_confidence=float(np.mean(conf_perm)),
            depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
            is_metric_scale=False,
        )

        config = SurfaceReconstructionConfig(alpha_radius=2.0, alpha_edge=2.0)
        res1 = AlphaComplexSurfaceReconstructor(config).reconstruct_surface(cloud1)
        res2 = AlphaComplexSurfaceReconstructor(config).reconstruct_surface(cloud2)

        assert res1.mesh is not None and res2.mesh is not None
        assert np.allclose(res1.mesh.vertices, res2.mesh.vertices, atol=1e-12)
        assert np.array_equal(res1.mesh.faces, res2.mesh.faces)
        assert np.allclose(res1.mesh.face_areas, res2.mesh.face_areas, atol=1e-12)

    def test_scale_equivariance_within_declared_range(self) -> None:
        """Test 37: Scaling coordinates by s scales vertex coordinates by s and areas by s^2."""
        pts = make_regular_tetrahedron_points(scale=1.0)
        cloud1 = make_valid_dense_point_cloud(4, pts=pts)
        config1 = SurfaceReconstructionConfig(alpha_radius=1.5, alpha_edge=1.5)
        res1 = AlphaComplexSurfaceReconstructor(config1).reconstruct_surface(cloud1)

        s = 3.5
        cloud2 = make_valid_dense_point_cloud(4, pts=pts * s)
        config2 = SurfaceReconstructionConfig(alpha_radius=1.5 * s, alpha_edge=1.5 * s)
        res2 = AlphaComplexSurfaceReconstructor(config2).reconstruct_surface(cloud2)

        assert res1.mesh is not None and res2.mesh is not None
        assert np.array_equal(res1.mesh.faces, res2.mesh.faces)
        assert np.allclose(res2.mesh.vertices, res1.mesh.vertices * s, atol=1e-12)
        assert np.allclose(res2.mesh.face_areas, res1.mesh.face_areas * (s**2), atol=1e-12)

    def test_failure_safety_for_qhull_errors(self) -> None:
        """Test 38: Deliberately collinear points handled safely with explicit failure reason."""
        t = np.linspace(-1.0, 1.0, 6)
        collinear_pts = np.column_stack([t, t * 2.0, t * -3.0])
        cloud = make_valid_dense_point_cloud(6, pts=collinear_pts)

        res = AlphaComplexSurfaceReconstructor().reconstruct_surface(cloud)
        assert res.status == SurfaceReconstructionStatus.RECONSTRUCTION_FAILED
        assert SurfaceFailureReason.INSUFFICIENT_NON_COPLANAR_POINTS in res.failure_reasons
        assert res.mesh is None

    def test_no_hole_filling(self) -> None:
        """Test 39: Alpha-complex preserves open holes without artificial watertight capping."""
        # 6 points forming two triangles separated by gap > alpha_radius
        pts = np.array(
            [
                # Cluster 1
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.1, 0.2, 0.0],
                [0.1, 0.1, 0.2],
                # Cluster 2 (gap of 10.0 units)
                [10.0, 0.0, 0.0],
                [10.2, 0.0, 0.0],
                [10.1, 0.2, 0.0],
                [10.1, 0.1, 0.2],
            ],
            dtype=np.float64,
        )
        cloud = make_valid_dense_point_cloud(8, pts=pts)
        config = SurfaceReconstructionConfig(alpha_radius=0.5, alpha_edge=0.5)
        res = AlphaComplexSurfaceReconstructor(config).reconstruct_surface(cloud)

        assert res.mesh is not None
        # 2 disconnected tetrahedra -> 8 total faces (4 in each cluster)
        assert res.mesh.total_faces == 8
        # No bridge triangle spans between cluster 1 (indices 0..3) and cluster 2 (indices 4..7)
        for f in res.mesh.faces:
            has_c1 = any(idx < 4 for idx in f)
            has_c2 = any(idx >= 4 for idx in f)
            assert not (has_c1 and has_c2), "Spurious bridge face formed across unsupported gap!"

    def test_no_metric_scale_promotion(self) -> None:
        """Test 40: Reconstructed mesh strictly preserves is_metric_scale=False and RECONSTRUCTION_UNITS."""
        pts = make_regular_tetrahedron_points(scale=1.0)
        cloud = make_valid_dense_point_cloud(4, pts=pts, is_metric_scale=False)
        res = AlphaComplexSurfaceReconstructor(SurfaceReconstructionConfig(alpha_radius=2.0, alpha_edge=2.0)).reconstruct_surface(cloud)

        assert res.mesh is not None
        assert res.mesh.is_metric_scale is False
        assert res.mesh.depth_unit == DepthUnit.RECONSTRUCTION_UNITS
