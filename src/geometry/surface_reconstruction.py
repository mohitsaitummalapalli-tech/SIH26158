"""Surface reconstruction types, validation layer, normal estimation, and 3D Alpha-Complex engine.

Phase 3E.4 — Step 2: 3D Alpha-Complex Surface Extraction.

This module implements the complete 3D alpha-complex surface reconstruction pipeline
from multi-view fused dense point clouds (Phase 3E.3).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial import Delaunay, KDTree, QhullError

from src.geometry.mvs import (
    DensePointCloud,
    DepthUnit,
    PointValidationStatus,
    PointVisibilityState,
)


class SurfaceFailureReason(str, Enum):
    """Explicit taxonomy of surface reconstruction rejection and failure modes."""

    EMPTY_INPUT_CLOUD = "EMPTY_INPUT_CLOUD"
    NON_FINITE_VERTICES = "NON_FINITE_VERTICES"
    INVALID_CONFIDENCE = "INVALID_CONFIDENCE"
    INVALID_SUPPORT_COUNT = "INVALID_SUPPORT_COUNT"
    INVALID_VISIBILITY_STATE = "INVALID_VISIBILITY_STATE"
    INVALID_VALIDATION_STATUS = "INVALID_VALIDATION_STATUS"
    INCONSISTENT_ARRAY_LENGTHS = "INCONSISTENT_ARRAY_LENGTHS"
    INSUFFICIENT_NON_COPLANAR_POINTS = "INSUFFICIENT_NON_COPLANAR_POINTS"
    DEGENERATE_TRIANGLE = "DEGENERATE_TRIANGLE"
    EDGE_LENGTH_EXCEEDED = "EDGE_LENGTH_EXCEEDED"
    TETRAHEDRON_CIRCUMRADIUS_EXCEEDED = "TETRAHEDRON_CIRCUMRADIUS_EXCEEDED"
    DEGENERATE_NORMAL_COVARIANCE = "DEGENERATE_NORMAL_COVARIANCE"
    INSUFFICIENT_NEIGHBORHOOD = "INSUFFICIENT_NEIGHBORHOOD"
    VIEWPOINT_CANCELLATION = "VIEWPOINT_CANCELLATION"


class NormalEstimationStatus(str, Enum):
    """Status classification for local PCA surface normal estimation."""

    VALID = "VALID"
    UNCERTAIN = "UNCERTAIN"
    COLLINEAR_DEGENERATE = "COLLINEAR_DEGENERATE"
    SPHERICAL_DEGENERATE = "SPHERICAL_DEGENERATE"
    INSUFFICIENT_NEIGHBORHOOD = "INSUFFICIENT_NEIGHBORHOOD"
    VIEWPOINT_UNAVAILABLE = "VIEWPOINT_UNAVAILABLE"
    VIEWPOINT_CANCELLATION = "VIEWPOINT_CANCELLATION"


class SurfaceReconstructionStatus(str, Enum):
    """High-level outcome status for surface reconstruction attempts."""

    SUCCESS = "SUCCESS"
    EMPTY_VALID_OUTPUT = "EMPTY_VALID_OUTPUT"
    INPUT_REJECTED = "INPUT_REJECTED"
    RECONSTRUCTION_FAILED = "RECONSTRUCTION_FAILED"


class SurfaceReconstructionConfig(BaseModel):
    """Configuration parameters for surface reconstruction and normal estimation."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    alpha_radius: float = Field(
        default=0.15,
        gt=0.0,
        description="Circumradius threshold for 3D Delaunay tetrahedra in RECONSTRUCTION_UNITS.",
    )
    alpha_edge: float = Field(
        default=0.15,
        gt=0.0,
        description="Maximum admissible edge length for boundary triangle facets in RECONSTRUCTION_UNITS.",
    )
    min_distinct_views: int = Field(
        default=2,
        ge=1,
        description="Minimum distinct camera views required for a vertex to be considered multi-view supported.",
    )
    target_distinct_views: int = Field(
        default=4,
        ge=1,
        description="Target number of distinct views for full multi-view support score normalization.",
    )
    min_aspect_ratio: float = Field(
        default=20.0,
        gt=1.0,
        description="Maximum admissible triangle aspect ratio (longest edge / shortest altitude).",
    )
    min_relative_area: float = Field(
        default=1e-6,
        gt=0.0,
        description="Dimensionless relative area threshold (Area / max_edge^2).",
    )
    min_thickness_ratio: float = Field(
        default=1e-4,
        gt=0.0,
        lt=1.0,
        description="Minimum dimensionless singular value ratio (s[2] / s[0]) for non-coplanarity.",
    )
    normal_k_neighbors: int = Field(
        default=12,
        ge=3,
        description="Number of nearest neighbors used for local covariance PCA.",
    )
    normal_min_planarity: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum planarity metric (lambda_1 - lambda_0) / lambda_2 for reliable normal.",
    )
    normal_max_sphericity: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Maximum sphericity metric lambda_0 / lambda_2 for reliable normal.",
    )
    viewpoint_cancellation_threshold: float = Field(
        default=1e-3,
        gt=0.0,
        description="Norm threshold for aggregate viewing vector below which viewpoints cancel out.",
    )


@dataclass(frozen=True)
class NormalEstimationResult:
    """Result of local PCA normal estimation for a single point."""

    normal: np.ndarray
    eigenvalues: Tuple[float, float, float]  # lambda_0 <= lambda_1 <= lambda_2
    planarity: float
    linearity: float
    sphericity: float
    status: NormalEstimationStatus
    viewpoint_aligned: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SurfaceMesh:
    """Consolidated surface mesh representation in relative reconstruction coordinates."""

    vertices: np.ndarray  # (V, 3) float64 in RECONSTRUCTION_UNITS
    faces: np.ndarray  # (F, 3) int32 triangle vertex indices
    vertex_normals: Optional[np.ndarray]  # (V, 3) float64 unit normal vectors or None
    face_normals: Optional[np.ndarray]  # (F, 3) float64 unit face normal vectors or None
    vertex_confidences: np.ndarray  # (V,) float32 in [0, 1], HEURISTIC_SCORE
    vertex_support_counts: np.ndarray  # (V,) int32 distinct frame counts
    face_support_scores: np.ndarray  # (F,) float32 aggregated heuristic support score [0, 1]
    face_areas: np.ndarray  # (F,) float64 face area in RECONSTRUCTION_UNITS^2
    is_boundary_vertex: np.ndarray  # (V,) bool, True if vertex incident to open boundary
    is_boundary_face: np.ndarray  # (F,) bool, True if face has open boundary edge
    total_vertices: int
    total_faces: int
    depth_unit: DepthUnit = DepthUnit.RECONSTRUCTION_UNITS
    is_metric_scale: bool = False
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate internal dimensional consistency of SurfaceMesh."""
        if not isinstance(self.vertices, np.ndarray) or self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (V, 3) ndarray, got {getattr(self.vertices, 'shape', None)}")
        if not isinstance(self.faces, np.ndarray) or self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must be (F, 3) ndarray, got {getattr(self.faces, 'shape', None)}")
        if self.vertices.shape[0] != self.total_vertices:
            raise ValueError(f"total_vertices {self.total_vertices} != vertices length {self.vertices.shape[0]}")
        if self.faces.shape[0] != self.total_faces:
            raise ValueError(f"total_faces {self.total_faces} != faces length {self.faces.shape[0]}")
        if self.vertex_confidences.shape[0] != self.total_vertices:
            raise ValueError("vertex_confidences length != total_vertices")
        if self.vertex_support_counts.shape[0] != self.total_vertices:
            raise ValueError("vertex_support_counts length != total_vertices")
        if self.is_boundary_vertex.shape[0] != self.total_vertices:
            raise ValueError("is_boundary_vertex length != total_vertices")
        if self.face_support_scores.shape[0] != self.total_faces:
            raise ValueError("face_support_scores length != total_faces")
        if self.face_areas.shape[0] != self.total_faces:
            raise ValueError("face_areas length != total_faces")
        if self.is_boundary_face.shape[0] != self.total_faces:
            raise ValueError("is_boundary_face length != total_faces")
        if self.vertex_normals is not None and self.vertex_normals.shape != (self.total_vertices, 3):
            raise ValueError("vertex_normals must be (V, 3) matching total_vertices")
        if self.face_normals is not None and self.face_normals.shape != (self.total_faces, 3):
            raise ValueError("face_normals must be (F, 3) matching total_faces")
        if self.depth_unit != DepthUnit.RECONSTRUCTION_UNITS:
            raise ValueError("depth_unit must strictly be RECONSTRUCTION_UNITS")
        if self.is_metric_scale:
            raise ValueError("is_metric_scale must be False unless explicitly calibrated externally")


@dataclass
class SurfaceReconstructionResult:
    """Consolidated outcome of surface reconstruction and validation."""

    mesh: Optional[SurfaceMesh]
    status: SurfaceReconstructionStatus
    failure_reasons: List[SurfaceFailureReason] = field(default_factory=list)
    rejection_details: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class DensePointCloudValidator:
    """Validates DensePointCloud against contract rules before surface reconstruction."""

    @staticmethod
    def validate_input_cloud(
        cloud: DensePointCloud,
    ) -> Tuple[bool, List[SurfaceFailureReason], Dict[str, Any]]:
        """Perform strict mathematical and schema validation of DensePointCloud.

        Parameters
        ----------
        cloud : DensePointCloud
            Fused dense point cloud from Phase 3E.3.

        Returns
        -------
        Tuple[bool, List[SurfaceFailureReason], Dict[str, Any]]
            (is_valid, failure_reasons, diagnostic_details)
        """
        reasons: List[SurfaceFailureReason] = []
        details: Dict[str, Any] = {}

        if cloud.total_fused_points == 0 or cloud.points.size == 0:
            reasons.append(SurfaceFailureReason.EMPTY_INPUT_CLOUD)
            details["total_fused_points"] = cloud.total_fused_points
            return False, reasons, details

        n = cloud.total_fused_points
        # Array length consistency
        if (
            cloud.points.shape[0] != n
            or cloud.confidences.shape[0] != n
            or cloud.support_counts.shape[0] != n
            or len(cloud.source_frame_ids) != n
            or len(cloud.visibility_states) != n
            or len(cloud.validation_statuses) != n
        ):
            reasons.append(SurfaceFailureReason.INCONSISTENT_ARRAY_LENGTHS)
            details["lengths"] = {
                "total_fused_points": n,
                "points": cloud.points.shape[0],
                "confidences": cloud.confidences.shape[0],
                "support_counts": cloud.support_counts.shape[0],
                "source_frame_ids": len(cloud.source_frame_ids),
                "visibility_states": len(cloud.visibility_states),
                "validation_statuses": len(cloud.validation_statuses),
            }
            return False, reasons, details

        # 1. Finite XYZ check
        if not np.all(np.isfinite(cloud.points)):
            reasons.append(SurfaceFailureReason.NON_FINITE_VERTICES)
            details["non_finite_points_count"] = int(np.sum(~np.isfinite(cloud.points)))

        # 2. Confidence validation [0, 1] and finite
        if not np.all(np.isfinite(cloud.confidences)) or np.any(cloud.confidences < 0.0) or np.any(cloud.confidences > 1.0):
            reasons.append(SurfaceFailureReason.INVALID_CONFIDENCE)
            details["invalid_confidences"] = True

        # 3. Support count validation >= 1 and finite
        if not np.all(np.isfinite(cloud.support_counts)) or np.any(cloud.support_counts < 1):
            reasons.append(SurfaceFailureReason.INVALID_SUPPORT_COUNT)
            details["invalid_support_counts"] = True

        # 4. Visibility state check (must not be INVALID_DEPTH or INCONSISTENT)
        if any(v in (PointVisibilityState.INVALID_DEPTH, PointVisibilityState.INCONSISTENT) for v in cloud.visibility_states):
            reasons.append(SurfaceFailureReason.INVALID_VISIBILITY_STATE)
            details["invalid_visibility_count"] = sum(
                1 for v in cloud.visibility_states if v in (PointVisibilityState.INVALID_DEPTH, PointVisibilityState.INCONSISTENT)
            )

        # 5. Validation status check (must be VALIDATED or OBSERVED)
        if any(s not in (PointValidationStatus.VALIDATED, PointValidationStatus.OBSERVED) for s in cloud.validation_statuses):
            reasons.append(SurfaceFailureReason.INVALID_VALIDATION_STATUS)
            details["invalid_validation_status_count"] = sum(
                1 for s in cloud.validation_statuses if s not in (PointValidationStatus.VALIDATED, PointValidationStatus.OBSERVED)
            )

        # 6. Unit & metric scale check
        if cloud.depth_unit != DepthUnit.RECONSTRUCTION_UNITS:
            details["depth_unit_violation"] = str(cloud.depth_unit)
        if cloud.is_metric_scale:
            details["is_metric_scale_violation"] = True

        is_valid = len(reasons) == 0
        return is_valid, reasons, details


class LocalPCANormalEstimator:
    """Estimates surface normals from local point neighborhoods using covariance PCA."""

    def __init__(self, config: Optional[SurfaceReconstructionConfig] = None) -> None:
        self.config = config or SurfaceReconstructionConfig()

    def estimate_neighborhood_normal(
        self,
        neighborhood: np.ndarray,
        query_point: Optional[np.ndarray] = None,
        camera_centers: Optional[List[np.ndarray]] = None,
    ) -> NormalEstimationResult:
        """Compute local surface normal for a point neighborhood.

        Parameters
        ----------
        neighborhood : np.ndarray
            (K, 3) float64 array of neighbor points (K >= 3).
        query_point : Optional[np.ndarray]
            (3,) float64 query point coordinate. If None, uses mean of neighborhood.
        camera_centers : Optional[List[np.ndarray]]
            List of (3,) camera centers from contributing frames for viewpoint orientation.

        Returns
        -------
        NormalEstimationResult
            Detailed normal estimation outcome including eigenvalues and diagnostics.
        """
        if not isinstance(neighborhood, np.ndarray) or neighborhood.ndim != 2 or neighborhood.shape[1] != 3:
            raise ValueError(f"neighborhood must be (K, 3) ndarray, got {getattr(neighborhood, 'shape', None)}")

        k = neighborhood.shape[0]
        if k < 3:
            return NormalEstimationResult(
                normal=np.zeros(3, dtype=np.float64),
                eigenvalues=(0.0, 0.0, 0.0),
                planarity=0.0,
                linearity=0.0,
                sphericity=0.0,
                status=NormalEstimationStatus.INSUFFICIENT_NEIGHBORHOOD,
                viewpoint_aligned=False,
                diagnostics={"k_neighbors": k},
            )

        if not np.all(np.isfinite(neighborhood)):
            return NormalEstimationResult(
                normal=np.zeros(3, dtype=np.float64),
                eigenvalues=(0.0, 0.0, 0.0),
                planarity=0.0,
                linearity=0.0,
                sphericity=0.0,
                status=NormalEstimationStatus.UNCERTAIN,
                viewpoint_aligned=False,
                diagnostics={"error": "non_finite_neighborhood"},
            )

        centroid = np.mean(neighborhood, axis=0)
        centered = neighborhood - centroid
        cov = (centered.T @ centered) / float(k)

        # Eigen decomposition of real symmetric covariance matrix
        eigvals, eigvecs = np.linalg.eigh(cov)
        # np.linalg.eigh returns eigenvalues in ascending order: lambda_0 <= lambda_1 <= lambda_2
        # Clamp tiny negative eigenvalues from floating point roundoff
        eigvals = np.maximum(eigvals, 0.0)
        lam0, lam1, lam2 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])

        # Eigenvector corresponding to minimal eigenvalue lambda_0 is unoriented normal
        unoriented_normal = eigvecs[:, 0].copy()
        norm_val = np.linalg.norm(unoriented_normal)
        if norm_val > 1e-12:
            unoriented_normal /= norm_val
        else:
            unoriented_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # Canonicalize unoriented normal sign (first non-zero component positive)
        unoriented_normal = self._canonicalize_normal_sign(unoriented_normal)

        # Planarity, Linearity, Sphericity metrics
        if lam2 > 1e-14:
            planarity = float((lam1 - lam0) / lam2)
            linearity = float((lam2 - lam1) / lam2)
            sphericity = float(lam0 / lam2)
        else:
            # Completely degenerate (coincident points)
            planarity = 0.0
            linearity = 0.0
            sphericity = 0.0

        # Status determination
        if lam2 <= 1e-14 or (lam1 <= 1e-14 and lam0 <= 1e-14):
            status = NormalEstimationStatus.COLLINEAR_DEGENERATE
        elif planarity < self.config.normal_min_planarity and linearity > 0.7:
            status = NormalEstimationStatus.COLLINEAR_DEGENERATE
        elif sphericity > self.config.normal_max_sphericity:
            status = NormalEstimationStatus.SPHERICAL_DEGENERATE
        elif planarity < self.config.normal_min_planarity:
            status = NormalEstimationStatus.UNCERTAIN
        else:
            status = NormalEstimationStatus.VALID

        # Viewpoint orientation
        oriented_normal = unoriented_normal.copy()
        viewpoint_aligned = False
        pt = query_point if query_point is not None else centroid

        if camera_centers and len(camera_centers) > 0:
            view_dirs = []
            for c_cam in camera_centers:
                v = c_cam - pt
                v_norm = float(np.linalg.norm(v))
                if v_norm > 1e-12:
                    view_dirs.append(v / v_norm)

            if len(view_dirs) > 0:
                v_agg = np.mean(view_dirs, axis=0)
                v_agg_norm = float(np.linalg.norm(v_agg))

                # Vector cancellation guard
                if v_agg_norm < self.config.viewpoint_cancellation_threshold:
                    status = NormalEstimationStatus.VIEWPOINT_CANCELLATION
                    # Keep canonical unoriented sign
                else:
                    v_agg_unit = v_agg / v_agg_norm
                    dot_prod = float(np.dot(unoriented_normal, v_agg_unit))
                    if dot_prod < 0.0:
                        oriented_normal = -unoriented_normal
                    else:
                        oriented_normal = unoriented_normal
                    viewpoint_aligned = True
            else:
                if status == NormalEstimationStatus.VALID:
                    status = NormalEstimationStatus.VIEWPOINT_UNAVAILABLE
        else:
            if status == NormalEstimationStatus.VALID:
                status = NormalEstimationStatus.VIEWPOINT_UNAVAILABLE

        return NormalEstimationResult(
            normal=oriented_normal,
            eigenvalues=(lam0, lam1, lam2),
            planarity=planarity,
            linearity=linearity,
            sphericity=sphericity,
            status=status,
            viewpoint_aligned=viewpoint_aligned,
            diagnostics={"k_neighbors": k},
        )

    @staticmethod
    def _canonicalize_normal_sign(n: np.ndarray) -> np.ndarray:
        """Establish deterministic canonical sign convention for unoriented normal."""
        for val in n:
            if abs(val) > 1e-9:
                if val < 0.0:
                    return -n
                return n
        return n


class AlphaComplexSurfaceReconstructor:
    """3D Alpha-Complex surface reconstructor based on Delaunay tetrahedralization.

    Implements the full Phase 3E.4 surface reconstruction contract:
    DensePointCloud
    -> Validate schema and numerical bounds
    -> Require N >= 4 non-coplanar points
    -> Canonical input ordering for reproducible execution
    -> 3D Delaunay tetrahedralization via scipy.spatial.Delaunay
    -> Compute exact circumradius for each tetrahedron
    -> Retain tetrahedra with circumradius <= alpha_radius
    -> Extract triangular faces belonging to exactly one retained tetrahedron
    -> Apply scale-aware geometric constraints (alpha_edge, relative area, aspect ratio)
    -> Recompute topological boundary semantics on the FINAL emitted face set
    -> Construct and return SurfaceMesh in RECONSTRUCTION_UNITS with is_metric_scale=False.
    """

    def __init__(self, config: Optional[SurfaceReconstructionConfig] = None) -> None:
        self.config = config or SurfaceReconstructionConfig()
        self.normal_estimator = LocalPCANormalEstimator(self.config)

    @staticmethod
    def _sorted_face_key(a: int, b: int, c: int) -> Tuple[int, int, int]:
        """Return canonical sorted 3-tuple key for a face."""
        if a > b:
            a, b = b, a
        if b > c:
            b, c = c, b
        if a > b:
            a, b = b, a
        return (a, b, c)

    @staticmethod
    def _sorted_edge_key(a: int, b: int) -> Tuple[int, int]:
        """Return canonical sorted 2-tuple key for an undirected edge."""
        return (a, b) if a <= b else (b, a)

    def reconstruct_surface(
        self,
        cloud: DensePointCloud,
        compute_normals: bool = False,
        camera_centers: Optional[Dict[str, np.ndarray]] = None,
    ) -> SurfaceReconstructionResult:
        """Reconstruct 3D surface mesh from fused dense point cloud.

        Parameters
        ----------
        cloud : DensePointCloud
            Consolidated point cloud from Phase 3E.3.
        compute_normals : bool
            Whether to estimate vertex normals via local covariance PCA.
            Normals are optional and will never alter the reconstructed mesh topology.
        camera_centers : Optional[Dict[str, np.ndarray]]
            Map of frame_id -> (3,) camera center in world coordinates for viewpoint orientation.

        Returns
        -------
        SurfaceReconstructionResult
            Outcome containing SurfaceMesh on success, or explicit failure taxonomy.
        """
        # 1. Input Validation
        is_valid, reasons, details = DensePointCloudValidator.validate_input_cloud(cloud)
        if not is_valid:
            return SurfaceReconstructionResult(
                mesh=None,
                status=SurfaceReconstructionStatus.INPUT_REJECTED,
                failure_reasons=reasons,
                rejection_details=details,
            )

        n_pts = cloud.total_fused_points
        # 2. Check N >= 4 points
        if n_pts < 4:
            return SurfaceReconstructionResult(
                mesh=None,
                status=SurfaceReconstructionStatus.RECONSTRUCTION_FAILED,
                failure_reasons=[SurfaceFailureReason.INSUFFICIENT_NON_COPLANAR_POINTS],
                rejection_details={"total_points": n_pts, "required": 4},
            )

        # 3. Canonical input ordering for deterministic execution in same environment
        sort_keys = []
        for i in range(n_pts):
            frames_str = ",".join(sorted(cloud.source_frame_ids[i]))
            sort_keys.append(
                (
                    cloud.points[i, 0],
                    cloud.points[i, 1],
                    cloud.points[i, 2],
                    float(cloud.confidences[i]),
                    int(cloud.support_counts[i]),
                    frames_str,
                )
            )
        sort_order = sorted(range(n_pts), key=lambda idx: sort_keys[idx])
        sort_order_arr = np.array(sort_order, dtype=np.intp)

        points = cloud.points[sort_order_arr].copy()
        confidences = cloud.confidences[sort_order_arr].copy()
        support_counts = cloud.support_counts[sort_order_arr].copy()
        source_frame_ids = [cloud.source_frame_ids[idx] for idx in sort_order]

        # 4. Check for Coplanarity and 3D Degeneracy (Degenerate 3D Point Set)
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            if (
                len(s) < 3
                or not np.all(np.isfinite(s))
                or s[0] <= 0.0
                or (s[2] / s[0]) < self.config.min_thickness_ratio
            ):
                return SurfaceReconstructionResult(
                    mesh=None,
                    status=SurfaceReconstructionStatus.RECONSTRUCTION_FAILED,
                    failure_reasons=[SurfaceFailureReason.INSUFFICIENT_NON_COPLANAR_POINTS],
                    rejection_details={"singular_values": [float(val) for val in s]},
                )
        except np.linalg.LinAlgError as err:
            return SurfaceReconstructionResult(
                mesh=None,
                status=SurfaceReconstructionStatus.RECONSTRUCTION_FAILED,
                failure_reasons=[SurfaceFailureReason.INSUFFICIENT_NON_COPLANAR_POINTS],
                rejection_details={"svd_error": str(err)},
            )

        # 5. 3D Delaunay Tetrahedralization
        try:
            dt = Delaunay(points)
        except (QhullError, ValueError) as err:
            return SurfaceReconstructionResult(
                mesh=None,
                status=SurfaceReconstructionStatus.RECONSTRUCTION_FAILED,
                failure_reasons=[SurfaceFailureReason.INSUFFICIENT_NON_COPLANAR_POINTS],
                rejection_details={"delaunay_error": str(err)},
            )

        simplices = dt.simplices
        if simplices.shape[0] == 0:
            return SurfaceReconstructionResult(
                mesh=None,
                status=SurfaceReconstructionStatus.RECONSTRUCTION_FAILED,
                failure_reasons=[SurfaceFailureReason.INSUFFICIENT_NON_COPLANAR_POINTS],
                rejection_details={"simplices_count": 0},
            )

        # 6. Compute Circumradius for Each Tetrahedron and Filter by alpha_radius
        retained_tetrahedra: List[Tuple[int, int, int, int]] = []
        for tet in simplices:
            r_circ = self._compute_tetrahedron_circumradius(
                points[tet[0]], points[tet[1]], points[tet[2]], points[tet[3]]
            )
            if np.isfinite(r_circ) and r_circ <= self.config.alpha_radius:
                retained_tetrahedra.append((int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])))

        if len(retained_tetrahedra) == 0:
            return SurfaceReconstructionResult(
                mesh=None,
                status=SurfaceReconstructionStatus.EMPTY_VALID_OUTPUT,
                failure_reasons=[SurfaceFailureReason.TETRAHEDRON_CIRCUMRADIUS_EXCEEDED],
                rejection_details={
                    "total_tetrahedra": len(simplices),
                    "retained_tetrahedra": 0,
                    "alpha_radius": self.config.alpha_radius,
                },
            )

        # 7. Extract Candidate Boundary Faces (Faces belonging to exactly ONE retained tetrahedron)
        face_occurrences: Dict[Tuple[int, int, int], List[Tuple[int, int, int, int]]] = {}
        for tet in retained_tetrahedra:
            # 4 triangular faces of tetrahedron with opposite vertex
            tet_faces = [
                (tet[0], tet[1], tet[2], tet[3]),
                (tet[0], tet[1], tet[3], tet[2]),
                (tet[0], tet[2], tet[3], tet[1]),
                (tet[1], tet[2], tet[3], tet[0]),
            ]
            for u, v, w, opp in tet_faces:
                key = self._sorted_face_key(u, v, w)
                if key not in face_occurrences:
                    face_occurrences[key] = []
                face_occurrences[key].append((u, v, w, opp))

        candidate_faces: List[Tuple[int, int, int]] = []
        for key, occs in face_occurrences.items():
            # Boundary facet of the 3D alpha complex must belong to exactly 1 retained tetrahedron
            if len(occs) == 1:
                u, v, w, opp = occs[0]
                # Orient face normal outward from the tetrahedron
                p_u, p_v, p_w = points[u], points[v], points[w]
                p_opp = points[opp]
                n_cross = np.cross(p_v - p_u, p_w - p_u)
                # Vector pointing away from opposite vertex
                v_out = p_u - p_opp
                if np.dot(n_cross, v_out) < 0.0:
                    candidate_faces.append((u, w, v))
                else:
                    candidate_faces.append((u, v, w))

        if len(candidate_faces) == 0:
            return SurfaceReconstructionResult(
                mesh=None,
                status=SurfaceReconstructionStatus.EMPTY_VALID_OUTPUT,
                failure_reasons=[SurfaceFailureReason.EDGE_LENGTH_EXCEEDED],
                rejection_details={"candidate_boundary_faces": 0},
            )

        # 8. Apply scale-aware geometric constraints (alpha_edge, relative area, aspect ratio)
        final_faces: List[Tuple[int, int, int]] = []
        filter_failures: List[SurfaceFailureReason] = []

        for f in candidate_faces:
            u, v, w = f
            p0, p1, p2 = points[u], points[v], points[w]
            e0 = float(np.linalg.norm(p1 - p0))
            e1 = float(np.linalg.norm(p2 - p1))
            e2 = float(np.linalg.norm(p0 - p2))
            l_max = max(e0, e1, e2)

            # Constraint 1: Maximum edge length cap (alpha_edge)
            if l_max > self.config.alpha_edge:
                if SurfaceFailureReason.EDGE_LENGTH_EXCEEDED not in filter_failures:
                    filter_failures.append(SurfaceFailureReason.EDGE_LENGTH_EXCEEDED)
                continue

            n_cross = np.cross(p1 - p0, p2 - p0)
            cross_norm = float(np.linalg.norm(n_cross))
            area = 0.5 * cross_norm

            # Constraint 2: Relative area check (dimensionless scale-aware)
            if l_max <= 1e-14 or (area / (l_max**2)) < self.config.min_relative_area:
                if SurfaceFailureReason.DEGENERATE_TRIANGLE not in filter_failures:
                    filter_failures.append(SurfaceFailureReason.DEGENERATE_TRIANGLE)
                continue

            # Constraint 3: Dimensionless aspect ratio check
            h_min = (2.0 * area) / l_max
            aspect = l_max / h_min
            if aspect > self.config.min_aspect_ratio:
                if SurfaceFailureReason.DEGENERATE_TRIANGLE not in filter_failures:
                    filter_failures.append(SurfaceFailureReason.DEGENERATE_TRIANGLE)
                continue

            final_faces.append(f)

        if len(final_faces) == 0:
            return SurfaceReconstructionResult(
                mesh=None,
                status=SurfaceReconstructionStatus.EMPTY_VALID_OUTPUT,
                failure_reasons=filter_failures or [SurfaceFailureReason.EDGE_LENGTH_EXCEEDED],
                rejection_details={
                    "candidate_boundary_faces": len(candidate_faces),
                    "final_emitted_faces": 0,
                },
            )

        # 9. Recompute Topological Boundary Semantics Strictly from FINAL Emitted Faces
        edge_counts: Counter[Tuple[int, int]] = Counter()
        for f in final_faces:
            e0 = self._sorted_edge_key(f[0], f[1])
            e1 = self._sorted_edge_key(f[1], f[2])
            e2 = self._sorted_edge_key(f[2], f[0])
            edge_counts[e0] += 1
            edge_counts[e1] += 1
            edge_counts[e2] += 1

        boundary_edges: Set[Tuple[int, int]] = {e for e, c in edge_counts.items() if c == 1}

        is_boundary_vertex = np.zeros(n_pts, dtype=bool)
        for u_e, v_e in boundary_edges:
            is_boundary_vertex[u_e] = True
            is_boundary_vertex[v_e] = True

        n_faces = len(final_faces)
        is_boundary_face = np.zeros(n_faces, dtype=bool)
        for j, f in enumerate(final_faces):
            e0 = self._sorted_edge_key(f[0], f[1])
            e1 = self._sorted_edge_key(f[1], f[2])
            e2 = self._sorted_edge_key(f[2], f[0])
            if e0 in boundary_edges or e1 in boundary_edges or e2 in boundary_edges:
                is_boundary_face[j] = True

        # 10. Face Metrics: Support Score, Areas, and Face Normals
        face_support_scores = np.zeros(n_faces, dtype=np.float32)
        face_areas = np.zeros(n_faces, dtype=np.float64)
        face_normals = np.zeros((n_faces, 3), dtype=np.float64)
        target_views = float(self.config.target_distinct_views)

        for j, f in enumerate(final_faces):
            # Face support score is explicitly a HEURISTIC_SUPPORT_SCORE [0, 1]
            supps = [min(1.0, float(support_counts[idx]) / target_views) for idx in f]
            face_support_scores[j] = float(np.mean(supps))

            p0, p1, p2 = points[f[0]], points[f[1]], points[f[2]]
            n_cross = np.cross(p1 - p0, p2 - p0)
            norm_val = float(np.linalg.norm(n_cross))
            face_areas[j] = 0.5 * norm_val
            if norm_val > 1e-12:
                face_normals[j] = n_cross / norm_val
            else:
                face_normals[j] = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # 11. Optional Vertex Normal Estimation via Local PCA
        vertex_normals: Optional[np.ndarray] = None
        if compute_normals:
            kdtree = KDTree(points)
            k_neighbors = min(self.config.normal_k_neighbors, n_pts)
            v_norms = np.zeros((n_pts, 3), dtype=np.float64)
            for i in range(n_pts):
                _, idxs = kdtree.query(points[i], k=k_neighbors)
                neighbor_pts = points[idxs]
                cam_list: Optional[List[np.ndarray]] = None
                if camera_centers:
                    fids = source_frame_ids[i]
                    cams = [camera_centers[fid] for fid in fids if fid in camera_centers]
                    if len(cams) > 0:
                        cam_list = cams

                res = self.normal_estimator.estimate_neighborhood_normal(
                    neighborhood=neighbor_pts,
                    query_point=points[i],
                    camera_centers=cam_list,
                )
                v_norms[i] = res.normal
            vertex_normals = v_norms

        # 12. Build and return SurfaceMesh
        mesh = SurfaceMesh(
            vertices=points,
            faces=np.array(final_faces, dtype=np.int32),
            vertex_normals=vertex_normals,
            face_normals=face_normals,
            vertex_confidences=confidences,
            vertex_support_counts=support_counts,
            face_support_scores=face_support_scores,
            face_areas=face_areas,
            is_boundary_vertex=is_boundary_vertex,
            is_boundary_face=is_boundary_face,
            total_vertices=n_pts,
            total_faces=n_faces,
            depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
            is_metric_scale=False,
            provenance={
                "algorithm": "alpha_complex_delaunay_3d",
                "alpha_radius": self.config.alpha_radius,
                "alpha_edge": self.config.alpha_edge,
                "total_tetrahedra": len(simplices),
                "retained_tetrahedra": len(retained_tetrahedra),
                "candidate_boundary_faces": len(candidate_faces),
                "emitted_faces": n_faces,
                "input_cloud_provenance": cloud.provenance,
            },
        )

        return SurfaceReconstructionResult(
            mesh=mesh,
            status=SurfaceReconstructionStatus.SUCCESS,
            diagnostics={
                "total_tetrahedra": len(simplices),
                "retained_tetrahedra": len(retained_tetrahedra),
                "candidate_boundary_faces": len(candidate_faces),
                "emitted_faces": n_faces,
            },
        )

    @staticmethod
    def _compute_tetrahedron_circumradius(
        a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
    ) -> float:
        """Compute circumsphere radius R of 3D tetrahedron using vector cross products."""
        A = a - d
        B = b - d
        C = c - d
        cross_bc = np.cross(B, C)
        cross_ca = np.cross(C, A)
        cross_ab = np.cross(A, B)
        denom = 2.0 * float(np.dot(A, cross_bc))
        if abs(denom) < 1e-14 or not np.isfinite(denom):
            return float("inf")

        num = (
            np.dot(A, A) * cross_bc
            + np.dot(B, B) * cross_ca
            + np.dot(C, C) * cross_ab
        )
        r = num / denom
        r_norm = float(np.linalg.norm(r))
        return r_norm if np.isfinite(r_norm) else float("inf")
