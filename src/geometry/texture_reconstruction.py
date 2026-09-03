"""Phase 3E.4 Step 4: Multi-View Surface Texture Reconstruction.

Consumes the locked Phase 3E.4 Step 2 SurfaceMesh and Step 3 visibility infrastructure
to reconstruct trustworthy, photometrically consistent surface textures without hallucination.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.geometry.mvs import DepthUnit
from src.geometry.surface_reconstruction import SurfaceMesh
from src.geometry.texture_association import (
    DeterministicAABBBVH,
    SurfaceTextureAssociationMap,
    TextureObservation,
    TextureSampleType,
    TextureSourceCamera,
)


class OperationalTextureState(str, Enum):
    """Mutually exclusive operational state for each textured element."""
    OBSERVED_TEXTURE = "OBSERVED_TEXTURE"
    WEAK_TEXTURE = "WEAK_TEXTURE"
    PHOTOMETRIC_CONFLICT = "PHOTOMETRIC_CONFLICT"
    UNOBSERVED = "UNOBSERVED"
    INVALID_INPUT = "INVALID_INPUT"


class TextureReconstructionConfig(BaseModel):
    """Configuration governing UV parameterization, M-estimator fusion, and confidence thresholds."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    atlas_width: int = Field(default=2048, ge=32, le=8192)
    atlas_height: int = Field(default=2048, ge=32, le=8192)
    gutter_padding_px: int = Field(default=4, ge=1, le=32)
    chart_max_normal_angle_deg: float = Field(default=45.0, gt=0.0, le=90.0)
    target_packing_efficiency: float = Field(default=0.70, gt=0.1, le=0.95)

    tukey_tuning_constant: float = Field(default=4.685, gt=0.0)
    max_m_estimator_iterations: int = Field(default=5, ge=1, le=20)
    convergence_threshold_rgb: float = Field(default=0.5, gt=0.0)

    min_confidence_observed: float = Field(default=0.20, ge=0.0, le=1.0)
    photometric_conflict_threshold: float = Field(default=0.20, gt=0.0, le=1.0)
    min_consensus_fraction: float = Field(default=0.35, gt=0.0, le=1.0)
    target_observation_count: int = Field(default=4, ge=1)

    epsilon_scale: float = Field(default=1e-4, gt=0.0)
    tau_det: float = Field(default=1e-7, gt=0.0)
    tau_bary: float = Field(default=1e-6, ge=0.0)
    tau_t: float = Field(default=1e-6, ge=0.0)
    ray_offset_epsilon_ratio: float = Field(default=1e-6, gt=0.0, lt=1e-1)
    image_border_margin_px: float = Field(default=4.0, ge=0.0)
    margin_score_reference_px: float = Field(default=32.0, gt=0.0)
    min_composite_score: float = Field(default=0.05, ge=0.0, le=1.0)


@dataclass(frozen=True)
class CandidateColorSample:
    """Sampled visual and geometric evidence from an accepted observation."""
    frame_id: str
    camera_pixel: Tuple[float, float]
    raw_rgb: Tuple[float, float, float]
    prior_weight: float
    tukey_weight: float
    is_inlier: bool
    residual: float
    frame_quality: float = 1.0
    view_alignment: float = 1.0


@dataclass(frozen=True)
class FusedTextureElement:
    """Robust photometric fusion result for an exact surface point."""
    state: OperationalTextureState
    rgb: Tuple[int, int, int]
    alpha: float
    confidence: float
    inlier_count: int
    total_candidate_count: int
    contributing_frames: List[str]
    candidates: List[CandidateColorSample]


@dataclass(frozen=True)
class TexelProvenance:
    """Exact audit record and provenance for a textured surface texel."""
    face_idx: int
    barycentric_coords: Tuple[float, float, float]
    state: OperationalTextureState
    contributing_frames: List[str]
    pixel_coords: Dict[str, Tuple[float, float]]
    observation_scores: Dict[str, float]
    photometric_residuals: Dict[str, float]
    tukey_weights: Dict[str, float]
    fusion_method: str = "tukey_biweight_v1"


@dataclass(frozen=True)
class UVChart:
    """Planar surface chart parameterization."""
    chart_id: int
    face_indices: List[int]
    origin_px: Tuple[int, int]
    bbox_size_px: Tuple[int, int]
    average_normal: np.ndarray
    basis_u: np.ndarray
    basis_v: np.ndarray


@dataclass(frozen=True)
class ReconstructedTextureAtlas:
    """Complete multi-view reconstructed texture atlas with diagnostics and provenance."""
    albedo_atlas: np.ndarray              # (H, W, 3) uint8 RGB
    alpha_atlas: np.ndarray               # (H, W) float32 in [0, 1]
    confidence_atlas: np.ndarray          # (H, W) float32 in [0, 1]
    state_atlas: np.ndarray               # (H, W) object string representation
    uv_coordinates: np.ndarray            # (F, 3, 2) float32 in [0, 1]
    vertex_colors: np.ndarray             # (V, 3) uint8 fallback colors
    vertex_confidences: np.ndarray        # (V,) float32 in [0, 1]
    vertex_states: List[OperationalTextureState]

    total_surface_texels: int
    observed_texel_ratio: float
    weakly_observed_texel_ratio: float
    unobserved_texel_ratio: float

    texel_provenance: Dict[Tuple[int, int], TexelProvenance] = field(default_factory=dict)
    depth_unit: DepthUnit = DepthUnit.RECONSTRUCTION_UNITS
    is_metric_scale: bool = False
    config: TextureReconstructionConfig = field(default_factory=TextureReconstructionConfig)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Deterministically computes the weighted median of 1D values."""
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cum_w = np.cumsum(w_sorted)
    if cum_w[-1] <= 0.0 or not np.isfinite(cum_w[-1]):
        mid = len(values) // 2
        return float(v_sorted[mid])
    cutoff = 0.5 * cum_w[-1]
    idx = int(np.searchsorted(cum_w, cutoff))
    return float(v_sorted[min(idx, len(values) - 1)])


def sample_bilinear_rgb(
    image: np.ndarray, u: float, v: float, margin: float = 0.0
) -> Optional[Tuple[float, float, float]]:
    """Samples RGB from an image using bilinear interpolation with strict bounds checking.

    Returns None if (u, v) is out-of-bounds or image data is non-finite or invalid.
    """
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return None
    h, w = image.shape[:2]
    if u < margin or u > (w - 1.0 - margin) or v < margin or v > (h - 1.0 - margin):
        return None

    x0 = int(math.floor(u))
    y0 = int(math.floor(v))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)

    dx = u - x0
    dy = v - y0

    p00 = image[y0, x0, :3].astype(np.float64)
    p01 = image[y0, x1, :3].astype(np.float64)
    p10 = image[y1, x0, :3].astype(np.float64)
    p11 = image[y1, x1, :3].astype(np.float64)

    # Reject non-finite or invalid values
    block = np.vstack([p00, p01, p10, p11])
    if not np.all(np.isfinite(block)):
        return None
    if np.any(block < 0.0) or np.any(block > 255.0):
        return None

    val = (
        (1.0 - dx) * (1.0 - dy) * p00
        + dx * (1.0 - dy) * p01
        + (1.0 - dx) * dy * p10
        + dx * dy * p11
    )
    return (float(val[0]), float(val[1]), float(val[2]))


def evaluate_surface_point_observations(
    point_w: np.ndarray,
    normal_w: np.ndarray,
    containing_face_idx: int,
    candidate_cameras: Dict[str, TextureSourceCamera],
    bvh: DeterministicAABBBVH,
    config: TextureReconstructionConfig,
) -> List[TextureObservation]:
    """Evaluates strict, visibility-qualified observations for an exact 3D surface point.

    Guarantees that centroid or vertex visibility is NEVER falsely attributed
    to an arbitrary surface point that may be locally occluded.
    """
    # 1. Parameter validation
    if not (np.all(np.isfinite(point_w)) and np.all(np.isfinite(normal_w))):
        return []
    norm_n = float(np.linalg.norm(normal_w))
    if norm_n < 1e-12:
        return []
    n_unit = normal_w / norm_n

    # Sort cameras deterministically by frame_id
    sorted_cameras = sorted(candidate_cameras.items(), key=lambda kv: kv[0])

    # Pass 1: Geometric Line-of-Sight & Sensor Qualification
    visible_candidates: List[Tuple[str, TextureSourceCamera, np.ndarray, float, float, float, float]] = []

    m_border = config.image_border_margin_px
    eps_ratio = config.ray_offset_epsilon_ratio

    for frame_id, cam in sorted_cameras:
        if not (
            np.all(np.isfinite(cam.R_cw))
            and np.all(np.isfinite(cam.t_cw))
            and np.all(np.isfinite(cam.K))
        ):
            continue

        C_w = -cam.R_cw.T @ cam.t_cw
        cam_vec = C_w - point_w
        d_cam = float(np.linalg.norm(cam_vec))
        if d_cam <= 0.0 or not np.isfinite(d_cam):
            continue

        X_c = cam.R_cw @ point_w + cam.t_cw
        depth = float(X_c[2])
        if depth <= 0.0 or not np.isfinite(depth):
            continue

        fx = float(cam.K[0, 0])
        fy = float(cam.K[1, 1])
        cx = float(cam.K[0, 2])
        cy = float(cam.K[1, 2])
        u = float(fx * (X_c[0] / depth) + cx)
        v = float(fy * (X_c[1] / depth) + cy)

        if (
            u < m_border
            or u > (cam.width - 1.0 - m_border)
            or v < m_border
            or v > (cam.height - 1.0 - m_border)
        ):
            continue

        v_view = cam_vec / d_cam
        eps = eps_ratio * d_cam
        O = point_w + eps * v_view
        E = C_w - eps * v_view
        D = E - O

        # BVH occlusion check excluding the containing face
        hit_tri = bvh.find_occlusion(
            O, D, {containing_face_idx}, config.tau_det, config.tau_bary, config.tau_t
        )
        if hit_tri is not None:
            continue

        visible_candidates.append((frame_id, cam, v_view, d_cam, u, v, depth))

    if not visible_candidates:
        return []

    # Pass 2: Scoring & Prior Weight Allocation
    d_min = min(c[3] for c in visible_candidates)
    scored_observations: List[Tuple[float, str, TextureObservation]] = []

    for frame_id, cam, v_view, d_cam, u, v, depth in visible_candidates:
        s_dist = d_min / d_cam
        s_angle = float(abs(np.dot(n_unit, v_view)))
        inc_angle_deg = float(math.degrees(math.acos(float(np.clip(s_angle, -1.0, 1.0)))))
        d_edge = min(
            u - m_border,
            cam.width - 1.0 - m_border - u,
            v - m_border,
            cam.height - 1.0 - m_border - v,
        )
        s_margin = min(1.0, max(0.0, d_edge / config.margin_score_reference_px))
        s_geom = s_dist * s_angle * s_margin

        # Phase 2 metrics
        qm = cam.quality_metrics or {}
        sh = qm.get("sharpness", 0.5)
        bl = qm.get("blur", 0.5)
        ex = qm.get("exposure", 0.5)
        dr = qm.get("dynamic_risk", 0.0)

        # Validate metrics
        vals = [sh, bl, ex, dr]
        is_invalid = False
        for val in vals:
            if not np.isfinite(val) or val < 0.0 or val > 1.0:
                is_invalid = True
                break
        if is_invalid:
            continue

        s_frame = float(sh * (1.0 - bl) * ex)
        s_dyn = float(1.0 - np.clip(dr, 0.0, 1.0))

        s_composite = float(s_geom * s_frame * s_dyn)
        if s_composite < config.min_composite_score:
            continue

        obs = TextureObservation(
            sample_type=TextureSampleType.FACET_CENTROID,
            sample_index=containing_face_idx,
            frame_id=frame_id,
            pixel_coords=(u, v),
            depth=depth,
            incidence_angle_deg=inc_angle_deg,
            distance_to_cam=d_cam,
            geometric_score=s_geom,
            frame_quality_score=s_frame,
            dynamic_risk_score=s_dyn,
            composite_score=s_composite,
            provenance={"point_w": point_w.tolist(), "normal_w": n_unit.tolist()},
        )
        scored_observations.append((s_composite, frame_id, obs))

    # Sort deterministically by (score DESC, frame_id ASC)
    scored_observations.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored_observations[:config.target_observation_count * 2]]


def fuse_multiview_candidates(
    candidates: List[CandidateColorSample],
    config: TextureReconstructionConfig,
) -> FusedTextureElement:
    """Fuses multi-view color evidence using Tukey biweight M-estimator."""
    if not candidates:
        return FusedTextureElement(
            state=OperationalTextureState.UNOBSERVED,
            rgb=(0, 0, 0),
            alpha=0.0,
            confidence=0.0,
            inlier_count=0,
            total_candidate_count=0,
            contributing_frames=[],
            candidates=[],
        )

    # Validate finite RGB and ranges
    for c in candidates:
        for ch in c.raw_rgb:
            if not np.isfinite(ch) or ch < 0.0 or ch > 255.0:
                return FusedTextureElement(
                    state=OperationalTextureState.INVALID_INPUT,
                    rgb=(0, 0, 0),
                    alpha=0.0,
                    confidence=0.0,
                    inlier_count=0,
                    total_candidate_count=len(candidates),
                    contributing_frames=[c.frame_id for c in candidates],
                    candidates=candidates,
                )

    M = len(candidates)
    frames = [c.frame_id for c in candidates]

    # Handle Single Observation Case (Mathematically Guaranteed WEAK_TEXTURE)
    if M == 1:
        c0 = candidates[0]
        c_count = 0.25  # 1.0 / target_observation_count (4)
        c_qual = c0.frame_quality
        c_geom = c0.view_alignment
        c_cons = 0.50   # Documented neutral consensus heuristic
        c_tex = c_count * c_qual * c_geom * c_cons

        rgb_int = (
            int(np.clip(np.round(c0.raw_rgb[0]), 0, 255)),
            int(np.clip(np.round(c0.raw_rgb[1]), 0, 255)),
            int(np.clip(np.round(c0.raw_rgb[2]), 0, 255)),
        )
        updated_cand = [
            CandidateColorSample(
                frame_id=c0.frame_id,
                camera_pixel=c0.camera_pixel,
                raw_rgb=c0.raw_rgb,
                prior_weight=c0.prior_weight,
                tukey_weight=1.0,
                is_inlier=True,
                residual=0.0,
                frame_quality=c0.frame_quality,
                view_alignment=c0.view_alignment,
            )
        ]
        return FusedTextureElement(
            state=OperationalTextureState.WEAK_TEXTURE,
            rgb=rgb_int,
            alpha=1.0,
            confidence=c_tex,
            inlier_count=1,
            total_candidate_count=1,
            contributing_frames=frames,
            candidates=updated_cand,
        )

    # Multi-observation case: Initialize robust color anchor via weighted median luminance
    rgb_arr = np.array([c.raw_rgb for c in candidates], dtype=np.float64)  # (M, 3)
    priors = np.array([c.prior_weight for c in candidates], dtype=np.float64)  # (M,)
    luminances = 0.2126 * rgb_arr[:, 0] + 0.7152 * rgb_arr[:, 1] + 0.0722 * rgb_arr[:, 2]

    # Weighted median luminance anchor
    anchor_lum = weighted_median(luminances, priors)
    diff_lum = np.abs(luminances - anchor_lum)
    best_init_idx = int(np.argmin(diff_lum))
    c_anchor = rgb_arr[best_init_idx].copy()

    # Iterative Tukey Biweight M-Estimator Loop
    psi_weights = np.ones(M, dtype=np.float64)
    residuals = np.zeros(M, dtype=np.float64)

    for _ in range(config.max_m_estimator_iterations):
        # 1. Residuals against anchor
        diff = rgb_arr - c_anchor
        residuals = np.linalg.norm(diff, axis=1) / (np.sqrt(3.0) * 255.0)

        # 2. Robust scale sigma_hat via weighted MAD of residuals
        med_r = weighted_median(residuals, priors)
        mad_r = weighted_median(np.abs(residuals - med_r), priors)
        sigma_hat = 1.4826 * mad_r + config.epsilon_scale

        # 3. Normalized residuals
        u_m = residuals / (config.tukey_tuning_constant * sigma_hat)

        # 4. Tukey biweight influence weight
        inlier_mask = np.abs(u_m) <= 1.0
        psi_weights = np.where(inlier_mask, (1.0 - u_m**2) ** 2, 0.0)

        # 5. Combined weights and reweighted anchor update
        tilde_w = priors * psi_weights
        sum_w = float(np.sum(tilde_w))
        if sum_w <= 1e-12:
            break

        c_new = np.sum(tilde_w[:, None] * rgb_arr, axis=0) / sum_w
        shift = float(np.linalg.norm(c_new - c_anchor))
        c_anchor = c_new
        if shift < config.convergence_threshold_rgb:
            break

    # Convergence diagnostics & Conflict Check
    inliers = np.where(psi_weights > 0.0)[0]
    m_inliers = len(inliers)
    sum_prior = float(np.sum(priors))
    consensus_fraction = (
        float(np.sum(priors[inliers] * psi_weights[inliers])) / sum_prior
        if sum_prior > 0.0
        else 0.0
    )
    mean_inlier_residual = (
        float(np.sum(priors[inliers] * residuals[inliers])) / float(np.sum(priors[inliers]))
        if m_inliers > 0
        else 1.0
    )

    is_conflict = (
        m_inliers < 1
        or consensus_fraction < config.min_consensus_fraction
        or mean_inlier_residual > config.photometric_conflict_threshold
    )

    updated_candidates = [
        CandidateColorSample(
            frame_id=candidates[i].frame_id,
            camera_pixel=candidates[i].camera_pixel,
            raw_rgb=candidates[i].raw_rgb,
            prior_weight=candidates[i].prior_weight,
            tukey_weight=float(psi_weights[i]),
            is_inlier=bool(psi_weights[i] > 0.0),
            residual=float(residuals[i]),
            frame_quality=candidates[i].frame_quality,
            view_alignment=candidates[i].view_alignment,
        )
        for i in range(M)
    ]

    if is_conflict:
        return FusedTextureElement(
            state=OperationalTextureState.PHOTOMETRIC_CONFLICT,
            rgb=(0, 0, 0),
            alpha=0.0,
            confidence=0.0,
            inlier_count=m_inliers,
            total_candidate_count=M,
            contributing_frames=frames,
            candidates=updated_candidates,
        )

    # Compute texture confidence
    c_count = min(1.0, float(m_inliers) / float(config.target_observation_count))
    c_qual = (
        float(np.sum(priors[inliers] * np.array([candidates[i].frame_quality for i in inliers])))
        / float(np.sum(priors[inliers]))
    )
    c_geom = float(np.max([candidates[i].view_alignment for i in inliers]))
    c_cons = max(0.0, 1.0 - (mean_inlier_residual / config.photometric_conflict_threshold))
    c_tex = float(c_count * c_qual * c_geom * c_cons)

    state = (
        OperationalTextureState.OBSERVED_TEXTURE
        if c_tex >= config.min_confidence_observed
        else OperationalTextureState.WEAK_TEXTURE
    )
    rgb_final = (
        int(np.clip(np.round(c_anchor[0]), 0, 255)),
        int(np.clip(np.round(c_anchor[1]), 0, 255)),
        int(np.clip(np.round(c_anchor[2]), 0, 255)),
    )
    return FusedTextureElement(
        state=state,
        rgb=rgb_final,
        alpha=1.0,
        confidence=c_tex,
        inlier_count=m_inliers,
        total_candidate_count=M,
        contributing_frames=frames,
        candidates=updated_candidates,
    )


class MultiViewTextureReconstructor:
    """Headless, deterministic multi-view texture reconstruction engine."""

    def __init__(self, config: Optional[TextureReconstructionConfig] = None) -> None:
        self.config = config or TextureReconstructionConfig()

    def parameterize_mesh(
        self, mesh: SurfaceMesh
    ) -> Tuple[np.ndarray, List[UVChart]]:
        """Deterministically generates planar UV charts and shelf-packs them into atlas."""
        n_faces = len(mesh.faces)
        if n_faces == 0:
            return np.zeros((0, 3, 2), dtype=np.float32), []

        # 1. Build adjacency graph
        edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
        for f_idx in range(n_faces):
            f = mesh.faces[f_idx]
            for i in range(3):
                edge = (min(f[i], f[(i + 1) % 3]), max(f[i], f[(i + 1) % 3]))
                edge_to_faces.setdefault(edge, []).append(f_idx)

        face_adj: List[List[int]] = [[] for _ in range(n_faces)]
        for shared in edge_to_faces.values():
            if len(shared) == 2:
                face_adj[shared[0]].append(shared[1])
                face_adj[shared[1]].append(shared[0])

        # Sort adjacencies for determinism
        for adj in face_adj:
            adj.sort()

        # Compute or use active face normals
        if mesh.face_normals is not None:
            active_face_normals = mesh.face_normals
        else:
            v0 = mesh.vertices[mesh.faces[:, 0]]
            v1 = mesh.vertices[mesh.faces[:, 1]]
            v2 = mesh.vertices[mesh.faces[:, 2]]
            cross = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(cross, axis=1, keepdims=True)
            active_face_normals = cross / np.maximum(norm, 1e-12)

        # 2. Cluster faces into charts based on normal continuity
        cos_thresh = math.cos(math.radians(self.config.chart_max_normal_angle_deg))
        visited = np.zeros(n_faces, dtype=bool)
        charts_raw: List[List[int]] = []

        for f_start in range(n_faces):
            if visited[f_start]:
                continue
            chart_faces = [f_start]
            visited[f_start] = True
            queue = [f_start]

            while queue:
                curr = queue.pop(0)
                curr_n = active_face_normals[curr]
                for nbr in face_adj[curr]:
                    if not visited[nbr]:
                        nbr_n = active_face_normals[nbr]
                        if float(np.dot(curr_n, nbr_n)) >= cos_thresh:
                            visited[nbr] = True
                            chart_faces.append(nbr)
                            queue.append(nbr)

            chart_faces.sort()
            charts_raw.append(chart_faces)

        # 3. Local Planar Projection for each chart
        v0 = mesh.vertices[mesh.faces[:, 0]]
        v1 = mesh.vertices[mesh.faces[:, 1]]
        v2 = mesh.vertices[mesh.faces[:, 2]]
        geom_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        total_mesh_area = float(np.sum(geom_areas))
        total_mesh_area = total_mesh_area if total_mesh_area > 0.0 else 1.0

        # Target texel density
        rho = math.sqrt(
            (self.config.atlas_width * self.config.atlas_height * self.config.target_packing_efficiency)
            / total_mesh_area
        )

        chart_polys: List[Dict[str, Any]] = []
        gutter = self.config.gutter_padding_px

        for c_id, f_indices in enumerate(charts_raw):
            # Compute area-weighted average normal
            weights = geom_areas[f_indices]
            if np.sum(weights) <= 0.0:
                avg_n = np.mean(active_face_normals[f_indices], axis=0)
            else:
                avg_n = np.sum(active_face_normals[f_indices] * weights[:, None], axis=0)
            norm_avg_n = float(np.linalg.norm(avg_n))
            if norm_avg_n < 1e-12:
                avg_n = np.array([0.0, 0.0, 1.0])
            else:
                avg_n = avg_n / norm_avg_n

            # Form orthonormal basis via PCA on chart vertices
            v_unique_idx = sorted(list(set(mesh.faces[f_indices].flatten())))
            v_pts = mesh.vertices[v_unique_idx]
            v_mean = np.mean(v_pts, axis=0)
            v_centered = v_pts - v_mean

            # Covariance matrix
            cov = v_centered.T @ v_centered
            eigvals, eigvecs = np.linalg.eigh(cov)
            # Find eigenvector most orthogonal to avg_n
            e1 = eigvecs[:, 2]
            e1 = e1 - np.dot(e1, avg_n) * avg_n
            norm_e1 = float(np.linalg.norm(e1))
            if norm_e1 < 1e-12:
                e1 = np.array([1.0, 0.0, 0.0])
                e1 = e1 - np.dot(e1, avg_n) * avg_n
                norm_e1 = float(np.linalg.norm(e1))
                if norm_e1 < 1e-12:
                    e1 = np.array([0.0, 1.0, 0.0])
                    e1 = e1 - np.dot(e1, avg_n) * avg_n
                    norm_e1 = float(np.linalg.norm(e1))
            e1 = e1 / max(norm_e1, 1e-12)
            e2 = np.cross(avg_n, e1)

            # Project all chart face vertices
            f_uvs_unscaled: Dict[int, np.ndarray] = {}
            min_u, max_u = np.inf, -np.inf
            min_v, max_v = np.inf, -np.inf

            for fi in f_indices:
                verts = mesh.vertices[mesh.faces[fi]]
                u_p = (verts - v_mean) @ e1 * rho
                v_p = (verts - v_mean) @ e2 * rho
                f_uvs_unscaled[fi] = np.stack([u_p, v_p], axis=1)  # (3, 2)
                min_u = min(min_u, float(np.min(u_p)))
                max_u = max(max_u, float(np.max(u_p)))
                min_v = min(min_v, float(np.min(v_p)))
                max_v = max(max_v, float(np.max(v_p)))

            w_px = int(math.ceil(max_u - min_u)) + 2 * gutter
            h_px = int(math.ceil(max_v - min_v)) + 2 * gutter

            chart_polys.append({
                "chart_id": c_id,
                "face_indices": f_indices,
                "avg_n": avg_n,
                "e1": e1,
                "e2": e2,
                "w_px": max(w_px, 1),
                "h_px": max(h_px, 1),
                "min_u": min_u,
                "min_v": min_v,
                "f_uvs_unscaled": f_uvs_unscaled,
            })

        # 4. Deterministic Shelf-Bin Packing
        # Primary key: height DESC, secondary: width DESC, tertiary: min face index ASC
        chart_polys.sort(key=lambda cp: (-cp["h_px"], -cp["w_px"], cp["face_indices"][0]))

        uv_coords = np.zeros((n_faces, 3, 2), dtype=np.float32)
        uv_charts: List[UVChart] = []

        curr_x = gutter
        curr_y = gutter
        shelf_h = 0
        w_atlas = self.config.atlas_width
        h_atlas = self.config.atlas_height

        placed_charts: List[Tuple[Dict[str, Any], int, int]] = []
        max_seen_x = 0
        max_seen_y = 0

        for cp in chart_polys:
            w_box = cp["w_px"]
            h_box = cp["h_px"]

            if curr_x + w_box > w_atlas - gutter and curr_x > gutter:
                # Start new shelf
                curr_x = gutter
                curr_y += shelf_h + gutter
                shelf_h = 0

            # Assign origin
            origin_x = curr_x
            origin_y = curr_y
            shelf_h = max(shelf_h, h_box)
            curr_x += w_box + gutter

            max_seen_x = max(max_seen_x, origin_x + w_box + gutter)
            max_seen_y = max(max_seen_y, origin_y + h_box + gutter)
            placed_charts.append((cp, origin_x, origin_y))

        # Check if packing exceeded atlas dimensions; scale down if necessary to preserve [0, 1] bounds
        scale_x = float(w_atlas) / float(max(max_seen_x, w_atlas))
        scale_y = float(h_atlas) / float(max(max_seen_y, h_atlas))
        fit_scale = min(scale_x, scale_y)

        for cp, origin_x, origin_y in placed_charts:
            ox = int(origin_x * fit_scale)
            oy = int(origin_y * fit_scale)
            wb = max(1, int(cp["w_px"] * fit_scale))
            hb = max(1, int(cp["h_px"] * fit_scale))

            # Offset face UVs into atlas coordinates
            for fi, uvs_raw in cp["f_uvs_unscaled"].items():
                u_atlas = ox + gutter + (uvs_raw[:, 0] - cp["min_u"]) * fit_scale
                v_atlas = oy + gutter + (uvs_raw[:, 1] - cp["min_v"]) * fit_scale
                # Normalize and clamp strictly to [0, 1]
                uv_coords[fi, :, 0] = np.clip((u_atlas + 0.5) / float(w_atlas), 0.0, 1.0)
                uv_coords[fi, :, 1] = np.clip((v_atlas + 0.5) / float(h_atlas), 0.0, 1.0)

            uv_charts.append(
                UVChart(
                    chart_id=cp["chart_id"],
                    face_indices=cp["face_indices"],
                    origin_px=(ox, oy),
                    bbox_size_px=(wb, hb),
                    average_normal=cp["avg_n"],
                    basis_u=cp["e1"],
                    basis_v=cp["e2"],
                )
            )

        return uv_coords, uv_charts

    def reconstruct_texture(
        self,
        mesh: SurfaceMesh,
        cameras: Dict[str, TextureSourceCamera],
        camera_images: Dict[str, np.ndarray],
        bvh: Optional[DeterministicAABBBVH] = None,
        association_map: Optional[SurfaceTextureAssociationMap] = None,
    ) -> ReconstructedTextureAtlas:
        """Executes full multi-view surface texture reconstruction."""
        # 1. Prepare BVH
        active_bvh = bvh or DeterministicAABBBVH(mesh.vertices, mesh.faces)

        # 2. Parameterize UVs
        uv_coords, uv_charts = self.parameterize_mesh(mesh)
        w_atlas = self.config.atlas_width
        h_atlas = self.config.atlas_height

        # Allocate atlas arrays
        albedo = np.zeros((h_atlas, w_atlas, 3), dtype=np.uint8)
        alpha = np.zeros((h_atlas, w_atlas), dtype=np.float32)
        confidence = np.zeros((h_atlas, w_atlas), dtype=np.float32)
        state_grid = np.full(
            (h_atlas, w_atlas), OperationalTextureState.UNOBSERVED.value, dtype=object
        )

        n_faces = len(mesh.faces)
        total_surface_texels = 0
        n_observed = 0
        n_weak = 0
        n_unobserved = 0
        texel_prov: Dict[Tuple[int, int], TexelProvenance] = {}

        # Compute or use active face normals
        if mesh.face_normals is not None:
            active_face_normals = mesh.face_normals
        else:
            v0 = mesh.vertices[mesh.faces[:, 0]]
            v1 = mesh.vertices[mesh.faces[:, 1]]
            v2 = mesh.vertices[mesh.faces[:, 2]]
            cross = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(cross, axis=1, keepdims=True)
            active_face_normals = cross / np.maximum(norm, 1e-12)

        # 3. Rasterize faces into atlas
        for f_idx in range(n_faces):
            face_verts = mesh.vertices[mesh.faces[f_idx]]
            face_normals = (
                mesh.vertex_normals[mesh.faces[f_idx]]
                if mesh.vertex_normals is not None
                else np.tile(active_face_normals[f_idx], (3, 1))
            )
            uv_f = uv_coords[f_idx]  # (3, 2) in [0, 1]
            uv_px = uv_f * np.array([w_atlas, h_atlas]) - 0.5

            # Bounding box of triangle in pixel space
            min_x = max(0, int(math.floor(np.min(uv_px[:, 0]))))
            max_x = min(w_atlas - 1, int(math.ceil(np.max(uv_px[:, 0]))))
            min_y = max(0, int(math.floor(np.min(uv_px[:, 1]))))
            max_y = min(h_atlas - 1, int(math.ceil(np.max(uv_px[:, 1]))))

            # Triangle 2D edge setup for barycentric solve
            u0, v0 = uv_px[0]
            u1, v1 = uv_px[1]
            u2, v2 = uv_px[2]
            denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
            if abs(denom) < 1e-12:
                continue

            inv_denom = 1.0 / denom

            for py in range(min_y, max_y + 1):
                for px in range(min_x, max_x + 1):
                    # Compute barycentric coordinates
                    w0 = ((v1 - v2) * (px - u2) + (u2 - u1) * (py - v2)) * inv_denom
                    w1 = ((v2 - v0) * (px - u2) + (u0 - u2) * (py - v2)) * inv_denom
                    w2 = 1.0 - w0 - w1

                    if w0 < -self.config.tau_bary or w1 < -self.config.tau_bary or w2 < -self.config.tau_bary:
                        continue

                    # Clamped non-negative barycentric weights
                    lam0 = max(0.0, w0)
                    lam1 = max(0.0, w1)
                    lam2 = max(0.0, w2)
                    sum_lam = lam0 + lam1 + lam2
                    lam0 /= sum_lam
                    lam1 /= sum_lam
                    lam2 /= sum_lam

                    total_surface_texels += 1

                    # Exact 3D surface point & normal
                    p_w = lam0 * face_verts[0] + lam1 * face_verts[1] + lam2 * face_verts[2]
                    n_w = lam0 * face_normals[0] + lam1 * face_normals[1] + lam2 * face_normals[2]

                    # Dedicated Step 3 evidence evaluation for exact surface point
                    obs_list = evaluate_surface_point_observations(
                        point_w=p_w,
                        normal_w=n_w,
                        containing_face_idx=f_idx,
                        candidate_cameras=cameras,
                        bvh=active_bvh,
                        config=self.config,
                    )

                    # Sample color from images
                    cand_samples: List[CandidateColorSample] = []
                    for obs in obs_list:
                        img = camera_images.get(obs.frame_id)
                        if img is None:
                            continue
                        color = sample_bilinear_rgb(img, obs.pixel_coords[0], obs.pixel_coords[1])
                        if color is None:
                            continue
                        cand_samples.append(
                            CandidateColorSample(
                                frame_id=obs.frame_id,
                                camera_pixel=obs.pixel_coords,
                                raw_rgb=color,
                                prior_weight=obs.composite_score,
                                tukey_weight=1.0,
                                is_inlier=True,
                                residual=0.0,
                                frame_quality=obs.frame_quality_score,
                                view_alignment=float(abs(np.dot(
                                    n_w / max(float(np.linalg.norm(n_w)), 1e-12),
                                    (cameras[obs.frame_id].R_cw.T @ -cameras[obs.frame_id].t_cw - p_w)
                                    / max(obs.distance_to_cam, 1e-12),
                                ))),
                            )
                        )

                    # Photometric M-estimator fusion
                    fused = fuse_multiview_candidates(cand_samples, self.config)

                    albedo[py, px] = fused.rgb
                    alpha[py, px] = fused.alpha
                    confidence[py, px] = fused.confidence
                    state_grid[py, px] = fused.state.value

                    texel_prov[(py, px)] = TexelProvenance(
                        face_idx=f_idx,
                        barycentric_coords=(float(lam0), float(lam1), float(lam2)),
                        state=fused.state,
                        contributing_frames=list(fused.contributing_frames),
                        pixel_coords={c.frame_id: c.camera_pixel for c in fused.candidates},
                        observation_scores={c.frame_id: c.prior_weight for c in fused.candidates},
                        photometric_residuals={c.frame_id: c.residual for c in fused.candidates},
                        tukey_weights={c.frame_id: c.tukey_weight for c in fused.candidates},
                        fusion_method="tukey_biweight_v1",
                    )

                    if fused.state == OperationalTextureState.OBSERVED_TEXTURE:
                        n_observed += 1
                    elif fused.state == OperationalTextureState.WEAK_TEXTURE:
                        n_weak += 1
                    else:
                        n_unobserved += 1

        # 4. Fallback Vertex Colors (Strictly evidence-driven, no interpolation from centroids)
        n_verts = len(mesh.vertices)
        v_colors = np.zeros((n_verts, 3), dtype=np.uint8)
        v_confidences = np.zeros(n_verts, dtype=np.float32)
        v_states: List[OperationalTextureState] = [
            OperationalTextureState.UNOBSERVED for _ in range(n_verts)
        ]

        if association_map is not None and association_map.sample_type == TextureSampleType.VERTEX:
            for v_idx in range(n_verts):
                v_obs = association_map.observations_by_sample.get(v_idx, [])
                if not v_obs:
                    continue
                v_cand_samples: List[CandidateColorSample] = []
                for obs in v_obs:
                    img = camera_images.get(obs.frame_id)
                    if img is None:
                        continue
                    color = sample_bilinear_rgb(img, obs.pixel_coords[0], obs.pixel_coords[1])
                    if color is None:
                        continue
                    v_cand_samples.append(
                        CandidateColorSample(
                            frame_id=obs.frame_id,
                            camera_pixel=obs.pixel_coords,
                            raw_rgb=color,
                            prior_weight=obs.composite_score,
                            tukey_weight=1.0,
                            is_inlier=True,
                            residual=0.0,
                            frame_quality=obs.frame_quality_score,
                        )
                    )
                v_fused = fuse_multiview_candidates(v_cand_samples, self.config)
                v_colors[v_idx] = v_fused.rgb
                v_confidences[v_idx] = v_fused.confidence
                v_states[v_idx] = v_fused.state

        # Compute coverage ratios over parameterized surface texels
        tot = max(total_surface_texels, 1)
        obs_ratio = n_observed / tot
        weak_ratio = n_weak / tot
        unobs_ratio = n_unobserved / tot

        return ReconstructedTextureAtlas(
            albedo_atlas=albedo,
            alpha_atlas=alpha,
            confidence_atlas=confidence,
            state_atlas=state_grid,
            uv_coordinates=uv_coords,
            vertex_colors=v_colors,
            vertex_confidences=v_confidences,
            vertex_states=v_states,
            total_surface_texels=total_surface_texels,
            observed_texel_ratio=obs_ratio,
            weakly_observed_texel_ratio=weak_ratio,
            unobserved_texel_ratio=unobs_ratio,
            texel_provenance=texel_prov,
            depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
            is_metric_scale=False,
            config=self.config,
            diagnostics={
                "chart_count": len(uv_charts),
                "n_observed_texels": n_observed,
                "n_weak_texels": n_weak,
                "n_unobserved_texels": n_unobserved,
            },
        )
