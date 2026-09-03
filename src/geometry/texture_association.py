"""Phase 3E.4 Step 3: Visibility-Aware Surface-to-Image Texture Association.

Associates reconstructed surface samples (facet centroids or vertices) with calibrated
multi-view camera frames using exact finite line segment raycasting, topological
self-intersection exclusion, deterministic AABB BVH traversal, and multi-view
heuristic scoring.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.geometry.mvs import DepthUnit
from src.geometry.surface_reconstruction import SurfaceMesh


class TextureSampleType(str, Enum):
    """Geometric primitive entity being textured."""
    VERTEX = "VERTEX"
    FACET_CENTROID = "FACET_CENTROID"


class SampleObservationState(str, Enum):
    """Sample-level observation status."""
    OBSERVED = "OBSERVED"
    UNOBSERVED = "UNOBSERVED"


class TextureQueryStatus(str, Enum):
    """Per-camera query evaluation status."""
    VISIBLE = "VISIBLE"
    NEGATIVE_DEPTH = "NEGATIVE_DEPTH"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    OCCLUDED = "OCCLUDED"
    LOW_QUALITY_SCORE = "LOW_QUALITY_SCORE"
    DEGENERATE_CAMERA = "DEGENERATE_CAMERA"
    NON_FINITE_PARAMETERS = "NON_FINITE_PARAMETERS"
    INVALID_QUALITY_METRICS = "INVALID_QUALITY_METRICS"


class DecisionStatus(str, Enum):
    """Auditable candidate inclusion decision classification."""
    ACCEPTED_RETAINED = "ACCEPTED_RETAINED"
    ACCEPTED_NOT_RETAINED = "ACCEPTED_NOT_RETAINED"
    REJECTED = "REJECTED"


class TextureAssociationConfig(BaseModel):
    """Configuration governing projection, occlusion, and scoring tolerances."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    image_border_margin_px: float = Field(
        default=4.0, ge=0.0,
        description="Sensor margin in pixels excluded to prevent lens edge artifacts."
    )
    margin_score_reference_px: float = Field(
        default=32.0, gt=0.0,
        description="Configurable pixel-domain transition distance for margin scoring."
    )
    ray_offset_epsilon_ratio: float = Field(
        default=1e-6, gt=0.0, lt=1e-1,
        description="Dimensionless ratio of ray length used to offset ray origin and target."
    )
    tau_det: float = Field(
        default=1e-7, gt=0.0,
        description="Dimensionless determinant tolerance ratio for ray-triangle parallelism."
    )
    tau_bary: float = Field(
        default=1e-6, ge=0.0,
        description="Dimensionless barycentric coordinate numerical tolerance."
    )
    tau_t: float = Field(
        default=1e-6, ge=0.0,
        description="Dimensionless ray segment parameter numerical tolerance."
    )
    min_composite_score: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description="Minimum composite heuristic score required to accept observation."
    )
    max_observations_per_sample: int = Field(
        default=8, ge=1,
        description="Maximum number of candidate observations preserved per sample (top-K)."
    )


@dataclass(frozen=True)
class TextureSourceCamera:
    """Calibrated camera frame input in undistorted pinhole domain."""
    frame_id: str
    R_cw: np.ndarray          # (3, 3) rotation matrix
    t_cw: np.ndarray          # (3,) translation vector
    K: np.ndarray             # (3, 3) intrinsic matrix
    width: int
    height: int
    quality_metrics: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class CandidateDecisionRecord:
    """Exact audit record for EVERY evaluated sample-camera pair."""
    sample_type: TextureSampleType
    sample_index: int
    frame_id: str
    decision: DecisionStatus
    query_status: TextureQueryStatus
    projected_pixels: Optional[Tuple[float, float]]
    depth: Optional[float]
    distance_to_cam: Optional[float]
    composite_score: Optional[float]
    rejection_reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextureObservation:
    """Individual accepted, unoccluded surface texture observation."""
    sample_type: TextureSampleType
    sample_index: int
    frame_id: str
    pixel_coords: Tuple[float, float]
    depth: float
    incidence_angle_deg: float
    distance_to_cam: float
    geometric_score: float
    frame_quality_score: float
    dynamic_risk_score: float
    composite_score: float
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SurfaceTextureAssociationMap:
    """Consolidated association output mapping samples to ranked observations."""
    sample_type: TextureSampleType
    total_samples: int
    sample_states: List[SampleObservationState]
    observations_by_sample: Dict[int, List[TextureObservation]]
    best_observation_by_sample: Dict[int, Optional[TextureObservation]]
    decision_records: List[CandidateDecisionRecord]
    sample_coverage_ratio: float
    depth_unit: DepthUnit = DepthUnit.RECONSTRUCTION_UNITS
    is_metric_scale: bool = False
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def rejection_records(self) -> List[CandidateDecisionRecord]:
        """Derived convenience view of all rejected candidates."""
        return [r for r in self.decision_records if r.decision == DecisionStatus.REJECTED]


# ==============================================================================
# Deterministic Axis-Aligned Bounding Box (AABB) BVH
# ==============================================================================

class _AABBNode:
    """Internal node of deterministic AABB Bounding Volume Hierarchy."""
    def __init__(
        self,
        min_bound: np.ndarray,
        max_bound: np.ndarray,
        triangles: Optional[List[int]] = None,
        left: Optional["_AABBNode"] = None,
        right: Optional["_AABBNode"] = None,
    ):
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.triangles = triangles
        self.left = left
        self.right = right
        self.is_leaf = triangles is not None


class DeterministicAABBBVH:
    """Deterministic AABB Bounding Volume Hierarchy over mesh faces."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        max_leaf_size: int = 4,
    ):
        self.vertices = vertices
        self.faces = faces
        self.max_leaf_size = max_leaf_size
        self.root: Optional[_AABBNode] = None

        if len(faces) > 0:
            self._tri_v0 = vertices[faces[:, 0]]
            self._tri_v1 = vertices[faces[:, 1]]
            self._tri_v2 = vertices[faces[:, 2]]
            self._tri_e1 = self._tri_v1 - self._tri_v0
            self._tri_e2 = self._tri_v2 - self._tri_v0
            self._tri_centroids = (self._tri_v0 + self._tri_v1 + self._tri_v2) / 3.0
            self._tri_min = np.minimum(np.minimum(self._tri_v0, self._tri_v1), self._tri_v2)
            self._tri_max = np.maximum(np.maximum(self._tri_v0, self._tri_v1), self._tri_v2)
            self.root = self._build_node(list(range(len(faces))))

    def _build_node(self, tri_indices: List[int]) -> _AABBNode:
        node_min = np.min(self._tri_min[tri_indices], axis=0)
        node_max = np.max(self._tri_max[tri_indices], axis=0)

        if len(tri_indices) <= self.max_leaf_size:
            return _AABBNode(node_min, node_max, triangles=tri_indices)

        # Centroid spread
        centroids = self._tri_centroids[tri_indices]
        c_min = np.min(centroids, axis=0)
        c_max = np.max(centroids, axis=0)
        spread = c_max - c_min
        split_axis = int(np.argmax(spread))

        if spread[split_axis] <= 1e-12:
            return _AABBNode(node_min, node_max, triangles=tri_indices)

        # Deterministic sorting by (centroid along split axis, tri_index)
        sorted_indices = sorted(
            tri_indices,
            key=lambda idx: (self._tri_centroids[idx, split_axis], idx)
        )
        mid = len(sorted_indices) // 2
        left_indices = sorted_indices[:mid]
        right_indices = sorted_indices[mid:]

        if len(left_indices) == 0 or len(right_indices) == 0:
            return _AABBNode(node_min, node_max, triangles=tri_indices)

        left_node = self._build_node(left_indices)
        right_node = self._build_node(right_indices)
        return _AABBNode(node_min, node_max, left=left_node, right=right_node)

    @staticmethod
    def _ray_box_intersect(
        O: np.ndarray,
        D: np.ndarray,
        min_b: np.ndarray,
        max_b: np.ndarray,
        tau_t: float,
    ) -> bool:
        """Kay-Kajiya slab method on finite segment t in [-tau_t, 1 + tau_t]."""
        t_min = -tau_t
        t_max = 1.0 + tau_t

        for i in range(3):
            d_i = D[i]
            o_i = O[i]
            if abs(d_i) < 1e-15:
                if o_i < min_b[i] or o_i > max_b[i]:
                    return False
            else:
                inv_d = 1.0 / d_i
                t1 = (min_b[i] - o_i) * inv_d
                t2 = (max_b[i] - o_i) * inv_d
                if t1 > t2:
                    t1, t2 = t2, t1
                if t1 > t_min:
                    t_min = t1
                if t2 < t_max:
                    t_max = t2
                if t_min > t_max:
                    return False
        return True

    def find_occlusion(
        self,
        O: np.ndarray,
        D: np.ndarray,
        excluded_faces: Set[int],
        tau_det: float,
        tau_bary: float,
        tau_t: float,
    ) -> Optional[int]:
        """Test for occlusion along segment R(t) = O + tD, t in [0, 1].

        Returns the hit triangle index if an occlusion is found, otherwise None.
        """
        if self.root is None:
            return None

        norm_D = np.linalg.norm(D)
        if norm_D <= 1e-15:
            return None

        stack = [self.root]
        while stack:
            node = stack.pop()
            if not self._ray_box_intersect(O, D, node.min_bound, node.max_bound, tau_t):
                continue

            if node.is_leaf and node.triangles is not None:
                for tri_idx in node.triangles:
                    if tri_idx in excluded_faces:
                        continue

                    E1 = self._tri_e1[tri_idx]
                    E2 = self._tri_e2[tri_idx]
                    P = np.cross(D, E2)
                    det = float(np.dot(E1, P))

                    # Dimensionless parallelism gate
                    norm_E1 = np.linalg.norm(E1)
                    norm_E2 = np.linalg.norm(E2)
                    denom = norm_E1 * norm_D * norm_E2
                    if denom <= 1e-15 or abs(det) <= tau_det * denom:
                        continue

                    inv_det = 1.0 / det
                    V0 = self._tri_v0[tri_idx]
                    T = O - V0
                    u_bary = float(np.dot(T, P)) * inv_det
                    if u_bary < -tau_bary:
                        continue

                    Q = np.cross(T, E1)
                    v_bary = float(np.dot(D, Q)) * inv_det
                    if v_bary < -tau_bary or (u_bary + v_bary) > (1.0 + tau_bary):
                        continue

                    t_hit = float(np.dot(E2, Q)) * inv_det
                    if -tau_t <= t_hit <= (1.0 + tau_t):
                        return tri_idx
            else:
                if node.right is not None:
                    stack.append(node.right)
                if node.left is not None:
                    stack.append(node.left)

        return None


# ==============================================================================
# Visibility-Aware Surface Texture Associator Engine
# ==============================================================================

class VisibilityAwareTextureAssociator:
    """Associates reconstructed surface geometry with multi-view camera frames."""

    def __init__(self, config: Optional[TextureAssociationConfig] = None):
        self.config = config or TextureAssociationConfig()

    def associate_texture(
        self,
        mesh: SurfaceMesh,
        cameras: Dict[str, Any],
        sample_type: TextureSampleType = TextureSampleType.FACET_CENTROID,
    ) -> SurfaceTextureAssociationMap:
        """Execute visibility-aware texture association on surface samples."""
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return SurfaceTextureAssociationMap(
                sample_type=sample_type,
                total_samples=0,
                sample_states=[],
                observations_by_sample={},
                best_observation_by_sample={},
                decision_records=[],
                sample_coverage_ratio=0.0,
                depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
                is_metric_scale=False,
                provenance={"status": "EMPTY_SURFACE_MESH"},
            )

        # 1. Normalize and canonically sort cameras by frame_id
        canonical_cameras: List[TextureSourceCamera] = []
        for frame_id in sorted(cameras.keys()):
            cam_val = cameras[frame_id]
            if isinstance(cam_val, TextureSourceCamera):
                canonical_cameras.append(cam_val)
            elif isinstance(cam_val, dict):
                quality = cam_val.get("quality_metrics", None)
                if quality is None:
                    quality = {
                        "sharpness": cam_val.get("sharpness", 0.5),
                        "blur": cam_val.get("blur", 0.0),
                        "exposure": cam_val.get("exposure", 1.0),
                        "dynamic_risk": cam_val.get("dynamic_risk", 0.0),
                    }
                canonical_cameras.append(
                    TextureSourceCamera(
                        frame_id=frame_id,
                        R_cw=np.asarray(cam_val["R_cw"], dtype=np.float64),
                        t_cw=np.asarray(cam_val["t_cw"], dtype=np.float64).reshape(-1),
                        K=np.asarray(cam_val["K"], dtype=np.float64),
                        width=int(cam_val.get("width", cam_val.get("W", 0))),
                        height=int(cam_val.get("height", cam_val.get("H", 0))),
                        quality_metrics=quality,
                    )
                )
            else:
                raise ValueError(f"Unsupported camera type for frame {frame_id}: {type(cam_val)}")

        # 2. Extract surface samples and topological exclusion sets
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)

        if sample_type == TextureSampleType.FACET_CENTROID:
            total_samples = len(faces)
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]
            sample_points = (v0 + v1 + v2) / 3.0
            if mesh.face_normals is not None and len(mesh.face_normals) == len(faces):
                sample_normals = np.asarray(mesh.face_normals, dtype=np.float64)
            else:
                cross = np.cross(v1 - v0, v2 - v0)
                norms = np.linalg.norm(cross, axis=1, keepdims=True)
                norms = np.where(norms > 1e-12, norms, 1.0)
                sample_normals = cross / norms
            incident_exclusions: List[Set[int]] = [{j} for j in range(total_samples)]
        else:  # VERTEX
            total_samples = len(vertices)
            sample_points = vertices
            if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(vertices):
                sample_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
            else:
                sample_normals = np.zeros_like(vertices)
                sample_normals[:, 2] = 1.0

            incident_exclusions = [set() for _ in range(total_samples)]
            for face_idx, face in enumerate(faces):
                for v_idx in face:
                    incident_exclusions[v_idx].add(face_idx)

        # 3. Construct deterministic AABB BVH
        bvh = DeterministicAABBBVH(vertices, faces)

        # 4. Association loop over samples and cameras
        decision_records: List[CandidateDecisionRecord] = []
        observations_by_sample: Dict[int, List[TextureObservation]] = {}
        best_observation_by_sample: Dict[int, Optional[TextureObservation]] = {}
        sample_states: List[SampleObservationState] = []

        m_border = self.config.image_border_margin_px
        ref_margin_px = self.config.margin_score_reference_px
        eps_ratio = self.config.ray_offset_epsilon_ratio
        tau_det = self.config.tau_det
        tau_bary = self.config.tau_bary
        tau_t = self.config.tau_t
        min_comp_score = self.config.min_composite_score
        max_obs = self.config.max_observations_per_sample

        for s_idx in range(total_samples):
            P_w = sample_points[s_idx]
            n_w = sample_normals[s_idx]
            norm_n = float(np.linalg.norm(n_w))
            if norm_n > 1e-12:
                n_w = n_w / norm_n
            else:
                n_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            excluded = incident_exclusions[s_idx]

            # Pass 1: Geometric Visibility & Discovery
            visible_candidates: List[Dict[str, Any]] = []

            for cam in canonical_cameras:
                frame_id = cam.frame_id

                # A. Numerical finiteness check
                if (
                    not np.all(np.isfinite(cam.R_cw))
                    or not np.all(np.isfinite(cam.t_cw))
                    or not np.all(np.isfinite(cam.K))
                    or cam.width <= 0
                    or cam.height <= 0
                ):
                    decision_records.append(
                        CandidateDecisionRecord(
                            sample_type=sample_type,
                            sample_index=s_idx,
                            frame_id=frame_id,
                            decision=DecisionStatus.REJECTED,
                            query_status=TextureQueryStatus.NON_FINITE_PARAMETERS,
                            projected_pixels=None,
                            depth=None,
                            distance_to_cam=None,
                            composite_score=None,
                            rejection_reason="Camera parameters are non-finite or dimensions non-positive.",
                        )
                    )
                    continue

                # B. Camera Center & Zero Distance Guard
                C_w = -cam.R_cw.T @ cam.t_cw
                cam_vec = C_w - P_w
                d_cam = float(np.linalg.norm(cam_vec))
                if d_cam <= 0.0 or not np.isfinite(d_cam):
                    decision_records.append(
                        CandidateDecisionRecord(
                            sample_type=sample_type,
                            sample_index=s_idx,
                            frame_id=frame_id,
                            decision=DecisionStatus.REJECTED,
                            query_status=TextureQueryStatus.DEGENERATE_CAMERA,
                            projected_pixels=None,
                            depth=0.0,
                            distance_to_cam=d_cam,
                            composite_score=None,
                            rejection_reason="Camera optical center coincides with surface sample (distance <= 0).",
                        )
                    )
                    continue

                # C. World to camera transform
                X_c = cam.R_cw @ P_w + cam.t_cw
                depth = float(X_c[2])

                # D. Cheirality (Positive Depth)
                if depth <= 0.0 or not np.isfinite(depth):
                    decision_records.append(
                        CandidateDecisionRecord(
                            sample_type=sample_type,
                            sample_index=s_idx,
                            frame_id=frame_id,
                            decision=DecisionStatus.REJECTED,
                            query_status=TextureQueryStatus.NEGATIVE_DEPTH,
                            projected_pixels=None,
                            depth=depth,
                            distance_to_cam=d_cam,
                            composite_score=None,
                            rejection_reason=f"Optical depth X_c,z = {depth} is non-positive.",
                        )
                    )
                    continue

                # E. Sensor projection (Undistorted Pinhole Domain)
                fx = cam.K[0, 0]
                fy = cam.K[1, 1]
                cx = cam.K[0, 2]
                cy = cam.K[1, 2]
                u = float(fx * (X_c[0] / depth) + cx)
                v = float(fy * (X_c[1] / depth) + cy)

                # F. Image Bounds & Sensor Margin
                if (
                    u < m_border
                    or u > (cam.width - 1.0 - m_border)
                    or v < m_border
                    or v > (cam.height - 1.0 - m_border)
                ):
                    decision_records.append(
                        CandidateDecisionRecord(
                            sample_type=sample_type,
                            sample_index=s_idx,
                            frame_id=frame_id,
                            decision=DecisionStatus.REJECTED,
                            query_status=TextureQueryStatus.OUT_OF_BOUNDS,
                            projected_pixels=(u, v),
                            depth=depth,
                            distance_to_cam=d_cam,
                            composite_score=None,
                            rejection_reason=f"Projected pixel ({u:.2f}, {v:.2f}) outside sensor bounds [margin={m_border}].",
                        )
                    )
                    continue

                # G. Finite Line Segment Formulation (Normal-Sign Independent)
                v_view = cam_vec / d_cam
                eps = eps_ratio * d_cam
                O = P_w + eps * v_view
                E = C_w - eps * v_view
                D = E - O

                # H. BVH Occlusion Query
                hit_tri = bvh.find_occlusion(O, D, excluded, tau_det, tau_bary, tau_t)
                if hit_tri is not None:
                    decision_records.append(
                        CandidateDecisionRecord(
                            sample_type=sample_type,
                            sample_index=s_idx,
                            frame_id=frame_id,
                            decision=DecisionStatus.REJECTED,
                            query_status=TextureQueryStatus.OCCLUDED,
                            projected_pixels=(u, v),
                            depth=depth,
                            distance_to_cam=d_cam,
                            composite_score=None,
                            rejection_reason=f"Line of sight occluded by mesh facet {hit_tri}.",
                            diagnostics={"hit_triangle_index": hit_tri},
                        )
                    )
                    continue

                # Survives geometric visibility!
                visible_candidates.append({
                    "camera": cam,
                    "frame_id": frame_id,
                    "depth": depth,
                    "u": u,
                    "v": v,
                    "d_cam": d_cam,
                    "v_view": v_view,
                })

            # End of Pass 1 for sample s_idx
            if len(visible_candidates) == 0:
                sample_states.append(SampleObservationState.UNOBSERVED)
                observations_by_sample[s_idx] = []
                best_observation_by_sample[s_idx] = None
                continue

            # Pass 2: Quality Scoring & Top-K Ranking
            d_min = min(c["d_cam"] for c in visible_candidates)
            scored_candidates: List[Dict[str, Any]] = []

            for cand in visible_candidates:
                cam = cand["camera"]
                frame_id = cand["frame_id"]
                d_cam = cand["d_cam"]
                u = cand["u"]
                v = cand["v"]
                depth = cand["depth"]
                v_view = cand["v_view"]

                # Phase-2 Quality metric validation
                qm = cam.quality_metrics or {}
                metrics_missing = False

                q_sharpness = qm.get("sharpness", None)
                p_blur = qm.get("blur", None)
                q_exposure = qm.get("exposure", None)
                dynamic_risk = qm.get("dynamic_risk", None)

                # Check for missing
                if q_sharpness is None or p_blur is None or q_exposure is None or dynamic_risk is None:
                    metrics_missing = True
                    if q_sharpness is None:
                        q_sharpness = 0.5
                    if p_blur is None:
                        p_blur = 0.5
                    if q_exposure is None:
                        q_exposure = 0.5
                    if dynamic_risk is None:
                        dynamic_risk = 0.5

                # Check for invalid metrics: NaN, Inf, or outside [0, 1]
                vals = [q_sharpness, p_blur, q_exposure, dynamic_risk]
                is_invalid = False
                for val in vals:
                    if not np.isfinite(val) or val < 0.0 or val > 1.0:
                        is_invalid = True
                        break

                if is_invalid:
                    decision_records.append(
                        CandidateDecisionRecord(
                            sample_type=sample_type,
                            sample_index=s_idx,
                            frame_id=frame_id,
                            decision=DecisionStatus.REJECTED,
                            query_status=TextureQueryStatus.INVALID_QUALITY_METRICS,
                            projected_pixels=(u, v),
                            depth=depth,
                            distance_to_cam=d_cam,
                            composite_score=None,
                            rejection_reason=f"Quality metric invalid (NaN/Inf or outside [0, 1]): {vals}.",
                        )
                    )
                    continue

                # Scoring formulation
                s_dist = d_min / d_cam  # in (0, 1]
                s_angle = float(abs(np.dot(n_w, v_view)))  # acute normal alignment
                angle_deg = float(np.degrees(np.arccos(np.clip(s_angle, 0.0, 1.0))))

                d_edge = min(u, cam.width - 1.0 - u, v, cam.height - 1.0 - v)
                s_margin = float(np.clip((d_edge - m_border) / ref_margin_px, 0.0, 1.0))

                s_geom = float(s_dist * s_angle * s_margin)
                s_frame = float(q_sharpness * (1.0 - p_blur) * q_exposure)
                s_dynamic = float(1.0 - np.clip(dynamic_risk, 0.0, 1.0))
                s_comp = float(s_geom * s_frame * s_dynamic)

                if s_comp < min_comp_score:
                    decision_records.append(
                        CandidateDecisionRecord(
                            sample_type=sample_type,
                            sample_index=s_idx,
                            frame_id=frame_id,
                            decision=DecisionStatus.REJECTED,
                            query_status=TextureQueryStatus.LOW_QUALITY_SCORE,
                            projected_pixels=(u, v),
                            depth=depth,
                            distance_to_cam=d_cam,
                            composite_score=s_comp,
                            rejection_reason=f"Composite score {s_comp:.4f} < threshold {min_comp_score}.",
                            diagnostics={
                                "s_geom": s_geom,
                                "s_frame": s_frame,
                                "s_dynamic": s_dynamic,
                                "metrics_missing": metrics_missing,
                            },
                        )
                    )
                    continue

                scored_candidates.append({
                    "sample_type": sample_type,
                    "sample_index": s_idx,
                    "frame_id": frame_id,
                    "pixel_coords": (u, v),
                    "depth": depth,
                    "incidence_angle_deg": angle_deg,
                    "distance_to_cam": d_cam,
                    "geometric_score": s_geom,
                    "frame_quality_score": s_frame,
                    "dynamic_risk_score": s_dynamic,
                    "composite_score": s_comp,
                    "metrics_missing": metrics_missing,
                })

            if len(scored_candidates) == 0:
                sample_states.append(SampleObservationState.UNOBSERVED)
                observations_by_sample[s_idx] = []
                best_observation_by_sample[s_idx] = None
                continue

            sample_states.append(SampleObservationState.OBSERVED)

            # Deterministic sorting: S_comp DESC, frame_id ASC
            scored_candidates.sort(key=lambda c: (-c["composite_score"], c["frame_id"]))

            sample_obs: List[TextureObservation] = []
            for rank, cand in enumerate(scored_candidates):
                frame_id = cand["frame_id"]
                s_comp = cand["composite_score"]
                u, v = cand["pixel_coords"]

                if rank < max_obs:
                    decision_records.append(
                        CandidateDecisionRecord(
                            sample_type=sample_type,
                            sample_index=s_idx,
                            frame_id=frame_id,
                            decision=DecisionStatus.ACCEPTED_RETAINED,
                            query_status=TextureQueryStatus.VISIBLE,
                            projected_pixels=(u, v),
                            depth=cand["depth"],
                            distance_to_cam=cand["distance_to_cam"],
                            composite_score=s_comp,
                            diagnostics={"metrics_missing": cand["metrics_missing"]},
                        )
                    )
                    sample_obs.append(
                        TextureObservation(
                            sample_type=sample_type,
                            sample_index=s_idx,
                            frame_id=frame_id,
                            pixel_coords=(u, v),
                            depth=cand["depth"],
                            incidence_angle_deg=cand["incidence_angle_deg"],
                            distance_to_cam=cand["distance_to_cam"],
                            geometric_score=cand["geometric_score"],
                            frame_quality_score=cand["frame_quality_score"],
                            dynamic_risk_score=cand["dynamic_risk_score"],
                            composite_score=s_comp,
                            provenance={
                                "metrics_missing": cand["metrics_missing"],
                                "rank": rank,
                            },
                        )
                    )
                else:
                    decision_records.append(
                        CandidateDecisionRecord(
                            sample_type=sample_type,
                            sample_index=s_idx,
                            frame_id=frame_id,
                            decision=DecisionStatus.ACCEPTED_NOT_RETAINED,
                            query_status=TextureQueryStatus.VISIBLE,
                            projected_pixels=(u, v),
                            depth=cand["depth"],
                            distance_to_cam=cand["distance_to_cam"],
                            composite_score=s_comp,
                            rejection_reason=f"Exceeded max_observations_per_sample ({max_obs}).",
                            diagnostics={"metrics_missing": cand["metrics_missing"]},
                        )
                    )

            observations_by_sample[s_idx] = sample_obs
            best_observation_by_sample[s_idx] = sample_obs[0] if sample_obs else None

        observed_count = sum(1 for st in sample_states if st == SampleObservationState.OBSERVED)
        coverage_ratio = float(observed_count / total_samples) if total_samples > 0 else 0.0

        return SurfaceTextureAssociationMap(
            sample_type=sample_type,
            total_samples=total_samples,
            sample_states=sample_states,
            observations_by_sample=observations_by_sample,
            best_observation_by_sample=best_observation_by_sample,
            decision_records=decision_records,
            sample_coverage_ratio=coverage_ratio,
            depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
            is_metric_scale=False,
            provenance={
                "algorithm": "visibility_aware_raycast_texture_association",
                "total_cameras": len(canonical_cameras),
                "total_samples": total_samples,
                "observed_samples": observed_count,
                "coverage_ratio": coverage_ratio,
            },
        )
