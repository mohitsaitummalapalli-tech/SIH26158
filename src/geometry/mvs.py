"""Phase 3E.0: Dense Multi-View Stereo (MVS) Architecture & Mathematical Contracts.

DESIGN AND CONTRACT DEFINITIONS ONLY.
Production depth estimation solvers and dense fusion algorithms are deferred to Phase 3E.1.

SCIENTIFIC OBJECTIVE:
Converts the gauge-constrained, optimized sparse reconstruction from Phase 3D.1 into
dense surface point geometry via Multi-View Stereo (MVS).

PIPELINE STAGES:
Optimized SfM (Phase 3D.1)
    ↓
MVS Candidate View Selection (MVSViewPairSelector)
    ↓
Reference / Source View Pairs (MVSViewGraph)
    ↓
Dense Correspondence / Depth Estimation Contract (IMVSDepthEstimator)
    ↓
Depth Map & Confidence (DepthMap, DepthConfidenceMap)
    ↓
Cross-View Geometric Consistency (DepthConsistencyChecker)
    ↓
Depth Filtering & Occlusion Tagging (PointVisibilityState)
    ↓
3D Backprojection (depth_to_world_points)
    ↓
Dense Point Fusion (DensePointFusion)
    ↓
Dense Point Cloud (DensePointCloud)

CRITICAL SCIENTIFIC CONSTRAINTS:
1. Reconstruction Units Only: Dense MVS operates strictly in the relative coordinate
   gauge established by SfM (Phase 3D.1). Output coordinates are in RECONSTRUCTION_UNITS.
   Absolute metric scale is NOT established here; no metric or meter accuracy is claimed.
2. Camera Coordinate Convention:
   X_c = R_cw * X_w + t_cw
   C_w = -R_cw^T * t_cw
   u = fx * (X_c / Z_c) + cx
   v = fy * (Y_c / Z_c) + cy
3. Pinhole Depth Definition: Depth strictly refers to optical depth Z_c along the
   principal axis in camera frame, not radial distance ||X_c||.
4. Occlusion & Consistency: Invisible or inconsistent depths are explicitly tagged
   (OCCLUDED, INCONSISTENT, INVALID_DEPTH) and must never be fabricated into fake 3D points.
5. Dynamic Scene Risk: Frame-level and pair-level dynamic motion risks from Phase 2
   propagate into view selection and confidence weighting.
6. Heuristic Defaults: All thresholds are explicitly labeled HEURISTIC_DEFAULT or HEURISTIC_SCORE.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Optional, List, Dict, Any, Tuple, Set

import numpy as np

from src.geometry.contracts import (
    EvaluationLevel,
    PipelineStageStatus,
    MeasurementType,
    CompletenessMetricType,
    GaugeFixingPolicy,
    GeometryFailureReason,
    CameraIntrinsics,
    ExtrinsicPose,
    TriangulatedTrack,
    SparseReconstructionResult,
    DenseMVSInput,
    DenseMVSOutput,
)


class MVSFailureReason(str, Enum):
    """Explicit failure taxonomy for Multi-View Stereo."""
    INSUFFICIENT_VALID_VIEWS = "INSUFFICIENT_VALID_VIEWS"
    INVALID_CAMERA_CALIBRATION = "INVALID_CAMERA_CALIBRATION"
    MISSING_IMAGE = "MISSING_IMAGE"
    INCOMPATIBLE_IMAGE_DIMENSIONS = "INCOMPATIBLE_IMAGE_DIMENSIONS"
    INVALID_POSE = "INVALID_POSE"
    INVALID_INTRINSICS = "INVALID_INTRINSICS"
    NO_USABLE_VIEW_PAIRS = "NO_USABLE_VIEW_PAIRS"
    INSUFFICIENT_OVERLAP = "INSUFFICIENT_OVERLAP"
    INSUFFICIENT_GEOMETRIC_BASELINE = "INSUFFICIENT_GEOMETRIC_BASELINE"
    DEPTH_ESTIMATION_FAILED = "DEPTH_ESTIMATION_FAILED"
    EXCESSIVE_OCCLUSION = "EXCESSIVE_OCCLUSION"
    INSUFFICIENT_CONSISTENCY = "INSUFFICIENT_CONSISTENCY"
    DENSE_FUSION_FAILED = "DENSE_FUSION_FAILED"


class PointVisibilityState(str, Enum):
    """Explicit visibility and consistency classification for dense depth observations."""
    VISIBLE = "VISIBLE"                         # Confirmed visible in reference and source views
    OCCLUDED = "OCCLUDED"                       # Occluded by foreground surface in source view
    INCONSISTENT = "INCONSISTENT"               # Failed cross-view reprojection or depth consistency
    INVALID_DEPTH = "INVALID_DEPTH"             # Depth is non-positive, NaN, Inf, or outside valid range
    INSUFFICIENT_SUPPORT = "INSUFFICIENT_SUPPORT" # Supported by fewer than required source views
    LOW_CONFIDENCE = "LOW_CONFIDENCE"           # Below minimum photometric or geometric confidence threshold
    VALID = "VALID"                             # Passed all validation, occlusion, and consistency checks


class PointValidationStatus(str, Enum):
    """Lifecycle validation status of dense 3D points."""
    OBSERVED = "OBSERVED"                       # Raw unverified backprojected depth observation
    VALIDATED = "VALIDATED"                     # Passed cross-view geometric consistency and bounds checks
    REJECTED = "REJECTED"                       # Failed consistency, cheirality, or confidence checks
    UNKNOWN = "UNKNOWN"                         # State cannot be determined from available views


class DepthUnit(str, Enum):
    """Coordinate measurement unit for depth and 3D points."""
    RECONSTRUCTION_UNITS = "RECONSTRUCTION_UNITS" # Relative SfM gauge units (scale ambiguous)
    METRIC_METERS = "METRIC_METERS"               # Certified absolute metric scale (requires external ground truth)


@dataclass(frozen=True)
class MVSConfig:
    """Heuristic engineering defaults (HEURISTIC_DEFAULT) for Multi-View Stereo.
    
    All numerical parameters in this configuration are engineering defaults for monocular
    UAV reconstruction and must not be construed as universal constants.
    """
    min_depth_units: float = 0.5                        # HEURISTIC_DEFAULT: Minimum optical depth Z_c
    max_depth_units: float = 100.0                      # HEURISTIC_DEFAULT: Maximum optical depth Z_c
    min_disparity_px: float = 1.0                       # HEURISTIC_DEFAULT: Minimum disparity in pixels
    max_disparity_px: float = 64.0                      # HEURISTIC_DEFAULT: Maximum disparity search range
    patch_match_window_size: int = 7                    # HEURISTIC_DEFAULT: Matching window size (odd integer)
    min_source_views: int = 2                           # HEURISTIC_DEFAULT: Minimum source views required per reference
    max_source_views: int = 4                           # HEURISTIC_DEFAULT: Maximum source views selected per reference
    min_overlap_ratio: float = 0.25                     # HEURISTIC_DEFAULT: Minimum estimated visual overlap
    min_triangulation_angle_deg: float = 2.0            # HEURISTIC_DEFAULT: Minimum baseline parallax angle
    max_triangulation_angle_deg: float = 40.0           # HEURISTIC_DEFAULT: Maximum baseline angle (avoids extreme distortion)
    reprojection_consistency_tolerance_px: float = 1.5  # HEURISTIC_DEFAULT: Maximum cross-view reprojection error
    relative_depth_consistency_tolerance: float = 0.05  # HEURISTIC_DEFAULT: Maximum relative depth disparity |Z1-Z2|/Z1
    min_consistent_views: int = 2                       # HEURISTIC_DEFAULT: Minimum consistent source views for fusion
    confidence_threshold: float = 0.5                   # HEURISTIC_DEFAULT: Minimum confidence score [0, 1]
    voxel_grid_resolution: float = 0.02                 # HEURISTIC_DEFAULT: Fusion spatial deduplication grid resolution
    config_version: str = "MVSConfig_v1.0"


@dataclass
class MVSInput:
    """Typed immutable input contract for Multi-View Stereo dense depth estimation."""
    selected_frame_ids: List[str]
    image_dimensions: Dict[str, Tuple[int, int]]       # frame_id -> (height, width)
    camera_intrinsics: Dict[str, CameraIntrinsics]      # frame_id -> calibrated intrinsics
    camera_poses: Dict[str, ExtrinsicPose]             # frame_id -> optimized extrinsics (from Phase 3D.1)
    image_paths: Optional[Dict[str, str]] = None       # frame_id -> image file path (streaming-compatible)
    sparse_landmarks: Optional[Dict[int, TriangulatedTrack]] = None # Sparse SfM priors
    dynamic_risk_scores: Dict[str, float] = field(default_factory=dict) # Phase 2 dynamic risk [0, 1]
    coordinate_convention: str = "opencv_optical"      # X_c = R_cw * X_w + t_cw
    depth_unit: DepthUnit = DepthUnit.RECONSTRUCTION_UNITS
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_frame_ids": self.selected_frame_ids,
            "total_frames": len(self.selected_frame_ids),
            "coordinate_convention": self.coordinate_convention,
            "depth_unit": self.depth_unit.value,
            "has_sparse_landmarks": self.sparse_landmarks is not None,
            "has_dynamic_risk": len(self.dynamic_risk_scores) > 0,
            "provenance": self.provenance,
        }


@dataclass
class MVSViewPair:
    """Typed contract for a selected reference-source stereo view pair."""
    reference_frame_id: str
    source_frame_id: str
    baseline_proxy: float                               # Distance between camera centers in reconstruction units
    relative_rotation_angle_deg: float                  # Geodesic angular distance in SO(3)
    relative_translation_direction: np.ndarray          # 3D unit direction vector from ref to src
    overlap_estimate: float                             # Estimated shared visual frustum overlap [0, 1]
    viewpoint_suitability_score: float                  # HEURISTIC_SCORE: Combined suitability metric [0, 1]
    selection_reason: str
    dynamic_risk: float = 0.0                           # Combined dynamic motion risk from Phase 2
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_frame_id": self.reference_frame_id,
            "source_frame_id": self.source_frame_id,
            "baseline_proxy": self.baseline_proxy,
            "relative_rotation_angle_deg": self.relative_rotation_angle_deg,
            "overlap_estimate": self.overlap_estimate,
            "viewpoint_suitability_score": self.viewpoint_suitability_score,
            "selection_reason": self.selection_reason,
            "dynamic_risk": self.dynamic_risk,
            "provenance": self.provenance,
        }


@dataclass
class MVSViewGraph:
    """Typed view graph containing selected stereo pairs and rejection audit logs."""
    frame_nodes: List[str]
    candidate_edges: List[Tuple[str, str]]
    selected_edges: List[MVSViewPair]
    per_edge_scores: Dict[Tuple[str, str], float]
    rejection_reasons: Dict[Tuple[str, str], str]
    provenance: Dict[str, Any] = field(default_factory=dict)

    def get_source_views(self, ref_id: str) -> List[MVSViewPair]:
        """Return all selected source views for a given reference view, in descending score order."""
        pairs = [p for p in self.selected_edges if p.reference_frame_id == ref_id]
        return sorted(pairs, key=lambda x: x.viewpoint_suitability_score, reverse=True)


@dataclass
class DepthMap:
    """Typed 2D dense depth map defined in reference camera coordinate frame."""
    reference_frame_id: str
    width: int
    height: int
    depth_array: np.ndarray                             # (H, W) float32 array: optical depth Z_c
    valid_mask: np.ndarray                              # (H, W) bool array: True if depth is finite and > 0
    depth_unit: DepthUnit = DepthUnit.RECONSTRUCTION_UNITS
    min_depth: float = 0.0
    max_depth: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)

    def get_depth_at(self, u: float, v: float) -> Optional[float]:
        """Query depth at continuous pixel coordinate (u, v) using bilinear or nearest lookup."""
        col = int(round(u))
        row = int(round(v))
        if 0 <= row < self.height and 0 <= col < self.width:
            if self.valid_mask[row, col]:
                val = float(self.depth_array[row, col])
                return val if math.isfinite(val) and val > 1e-6 else None
        return None


@dataclass
class DepthConfidenceMap:
    """Typed multi-criteria confidence and visibility classification map."""
    reference_frame_id: str
    width: int
    height: int
    photometric_confidence: np.ndarray                  # (H, W) float32 [0, 1], HEURISTIC_SCORE
    geometric_consistency_confidence: np.ndarray        # (H, W) float32 [0, 1], HEURISTIC_SCORE
    support_view_count: np.ndarray                      # (H, W) int32: number of consistent source views
    visibility_state: np.ndarray                        # (H, W) uint8 or string-equivalent PointVisibilityState
    overall_confidence: np.ndarray                      # (H, W) float32 [0, 1], HEURISTIC_SCORE
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DensePointObservation:
    """Individual 3D point observation backprojected from a depth map pixel."""
    world_point: np.ndarray                             # 3D coordinate in RECONSTRUCTION_UNITS
    reference_frame_id: str
    pixel_coord: Tuple[float, float]                    # (u, v) in reference raster
    depth: float                                        # Optical depth Z_c in reference camera
    confidence: float                                   # Combined confidence [0, 1], HEURISTIC_SCORE
    visibility_state: PointVisibilityState = PointVisibilityState.VISIBLE
    validation_status: PointValidationStatus = PointValidationStatus.OBSERVED
    source_view_support_count: int = 1
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DensePointCloud:
    """Complete fused dense 3D point cloud in relative reconstruction coordinates."""
    points: np.ndarray                                  # (N, 3) float64 in RECONSTRUCTION_UNITS
    confidences: np.ndarray                             # (N,) float32 in [0, 1], HEURISTIC_SCORE
    support_counts: np.ndarray                          # (N,) int32
    source_frame_ids: List[List[str]]                   # List of observing frame IDs per point
    visibility_states: List[PointVisibilityState]
    validation_statuses: List[PointValidationStatus]
    total_fused_points: int
    mean_confidence: float
    depth_unit: DepthUnit = DepthUnit.RECONSTRUCTION_UNITS
    is_metric_scale: bool = False                       # SCALE_AMBIGUOUS without certified external metric ground truth
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_fused_points": self.total_fused_points,
            "mean_confidence": self.mean_confidence,
            "depth_unit": self.depth_unit.value,
            "is_metric_scale": self.is_metric_scale,
            "provenance": self.provenance,
        }


# ==============================================================================
# ABSTRACT INTERFACES & ALGORITHM CONTRACTS
# ==============================================================================

class MVSViewPairSelector(ABC):
    """Abstract interface for selecting stereo view pairs based on geometric suitability."""

    @abstractmethod
    def select_pairs(
        self,
        mvs_input: MVSInput,
        config: MVSConfig,
    ) -> MVSViewGraph:
        """Evaluate all candidate view pairs and construct the MVSViewGraph."""
        pass


class DepthConsistencyChecker(ABC):
    """Abstract interface for cross-view geometric and depth consistency verification."""

    @abstractmethod
    def check_consistency(
        self,
        ref_depth: DepthMap,
        src_depth: DepthMap,
        ref_pose: ExtrinsicPose,
        src_pose: ExtrinsicPose,
        ref_K: CameraIntrinsics,
        src_K: CameraIntrinsics,
        config: MVSConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Verify cross-view depth and reprojection consistency.
        
        Returns:
            consistency_mask: (H, W) bool array, True if consistent within tolerances
            visibility_state: (H, W) array of PointVisibilityState
        """
        pass


class DensePointFusion(ABC):
    """Abstract interface for fusing multi-view depth observations into a consolidated point cloud."""

    @abstractmethod
    def fuse(
        self,
        observations: List[DensePointObservation],
        config: MVSConfig,
    ) -> DensePointCloud:
        """Fuse and deduplicate observations into a clean dense point cloud."""
        pass


class IMVSDepthEstimator(ABC):
    """Abstract interface for dense depth estimation implementations (Classical, Plane-Sweep, or Learned)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Descriptive identifier of the depth estimator."""
        pass

    @abstractmethod
    def estimate_depth_map(
        self,
        ref_frame_id: str,
        source_frame_ids: List[str],
        mvs_input: MVSInput,
        config: MVSConfig,
    ) -> Tuple[DepthMap, DepthConfidenceMap]:
        """Estimate 2D optical depth map and confidence map for a reference view."""
        pass


# ==============================================================================
# CANONICAL GEOMETRIC IMPLEMENTATIONS (PHASE 3E.0 REFERENCE CONTRACTS)
# ==============================================================================

class MVSGeometryMath:
    """Canonical geometric transformations, projections, and backprojections for MVS."""

    @staticmethod
    def backproject_pixel(
        u: float,
        v: float,
        depth_z: float,
        K: CameraIntrinsics,
        R_cw: np.ndarray,
        t_cw: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Backproject pixel (u, v) with optical depth Z_c into 3D world coordinates.
        
        X_c = Z_c * K^{-1} * [u, v, 1]^T
        X_w = R_cw^T * (X_c - t_cw)
        """
        if not (math.isfinite(depth_z) and depth_z > 1e-6):
            return None, False

        # Camera frame optical ray
        x_c = depth_z * (u - K.cx) / K.fx
        y_c = depth_z * (v - K.cy) / K.fy
        z_c = depth_z
        X_c = np.array([x_c, y_c, z_c], dtype=np.float64)

        # World coordinate
        X_w = R_cw.T @ (X_c - t_cw)
        return X_w, True

    @staticmethod
    def project_world_point(
        X_w: np.ndarray,
        K: CameraIntrinsics,
        R_cw: np.ndarray,
        t_cw: np.ndarray,
    ) -> Tuple[Optional[Tuple[float, float]], float, bool]:
        """Project 3D world point into camera raster.
        
        Returns:
            pixel_coord: (u, v) or None if non-finite or non-positive depth
            optical_depth: Z_c
            is_valid: True if Z_c > 1e-6
        """
        X_c = R_cw @ X_w + t_cw
        z_c = float(X_c[2])
        if z_c <= 1e-6 or not np.all(np.isfinite(X_c)):
            return None, z_c, False

        u = K.fx * (float(X_c[0]) / z_c) + K.cx
        v = K.fy * (float(X_c[1]) / z_c) + K.cy

        if not (math.isfinite(u) and math.isfinite(v)):
            return None, z_c, False

        return (u, v), z_c, True


def depth_to_world_points(
    depth_map: DepthMap,
    confidence_map: DepthConfidenceMap,
    intrinsics: CameraIntrinsics,
    extrinsic_pose: ExtrinsicPose,
    config: Optional[MVSConfig] = None,
) -> List[DensePointObservation]:
    """Deterministically backproject all valid pixels in DepthMap into 3D DensePointObservations."""
    cfg = config or MVSConfig()
    R_cw = np.array(extrinsic_pose.rotation_matrix, dtype=np.float64)
    c_w = np.array(extrinsic_pose.translation_vector, dtype=np.float64)
    t_cw = -R_cw @ c_w

    observations: List[DensePointObservation] = []
    H, W = depth_map.height, depth_map.width

    for row in range(H):
        for col in range(W):
            if not depth_map.valid_mask[row, col]:
                continue

            z = float(depth_map.depth_array[row, col])
            if z < cfg.min_depth_units or z > cfg.max_depth_units:
                continue

            u, v = float(col), float(row)
            X_w, ok = MVSGeometryMath.backproject_pixel(u, v, z, intrinsics, R_cw, t_cw)
            if not ok or X_w is None:
                continue

            conf = float(confidence_map.overall_confidence[row, col])
            supp = int(confidence_map.support_view_count[row, col])

            obs = DensePointObservation(
                world_point=X_w,
                reference_frame_id=depth_map.reference_frame_id,
                pixel_coord=(u, v),
                depth=z,
                confidence=conf,
                visibility_state=PointVisibilityState.VALID,
                validation_status=PointValidationStatus.VALIDATED,
                source_view_support_count=supp,
                provenance={"unit": DepthUnit.RECONSTRUCTION_UNITS.value},
            )
            observations.append(obs)

    return observations


class HeuristicViewPairSelector(MVSViewPairSelector):
    """Deterministic heuristic view-pair selector based on baseline proxy, angular separation, and overlap."""

    def select_pairs(
        self,
        mvs_input: MVSInput,
        config: MVSConfig,
    ) -> MVSViewGraph:
        frame_ids = sorted(list(mvs_input.selected_frame_ids))
        candidate_edges: List[Tuple[str, str]] = []
        selected_edges: List[MVSViewPair] = []
        per_edge_scores: Dict[Tuple[str, str], float] = {}
        rejection_reasons: Dict[Tuple[str, str], str] = {}

        # Precompute camera centers and rotations
        centers: Dict[str, np.ndarray] = {}
        rotations: Dict[str, np.ndarray] = {}
        for fid in frame_ids:
            p = mvs_input.camera_poses[fid]
            R = np.array(p.rotation_matrix, dtype=np.float64)
            c = np.array(p.translation_vector, dtype=np.float64)
            centers[fid] = c
            rotations[fid] = R

        for i, ref_id in enumerate(frame_ids):
            scored_candidates: List[MVSViewPair] = []

            for j, src_id in enumerate(frame_ids):
                if i == j:
                    continue
                edge = (ref_id, src_id)
                candidate_edges.append(edge)

                # Baseline proxy in reconstruction units
                c_diff = centers[src_id] - centers[ref_id]
                baseline = float(np.linalg.norm(c_diff))
                if baseline < 1e-4:
                    rejection_reasons[edge] = "Coincident camera centers (degenerate baseline)"
                    continue

                t_dir = c_diff / baseline

                # Relative rotation angle in degrees
                R_rel = rotations[src_id] @ rotations[ref_id].T
                cos_ang = np.clip((float(np.trace(R_rel)) - 1.0) / 2.0, -1.0, 1.0)
                rot_ang_deg = math.degrees(math.acos(cos_ang))

                if rot_ang_deg > config.max_triangulation_angle_deg:
                    rejection_reasons[edge] = f"Viewing angle too steep ({rot_ang_deg:.1f}° > {config.max_triangulation_angle_deg}°)"
                    continue

                # Heuristic overlap proxy based on angular separation and temporal sequence index
                idx_dist = abs(i - j)
                overlap_proxy = max(0.0, 1.0 - (rot_ang_deg / config.max_triangulation_angle_deg) * 0.7 - min(0.3, idx_dist * 0.05))

                if overlap_proxy < config.min_overlap_ratio:
                    rejection_reasons[edge] = f"Insufficient visual overlap ({overlap_proxy:.2f} < {config.min_overlap_ratio})"
                    continue

                # Suitability score: balance baseline parallax with visual overlap
                baseline_factor = min(1.0, baseline / 1.0)
                suitability = float(0.6 * overlap_proxy + 0.4 * baseline_factor)

                # Dynamic risk penalty
                dyn_risk = max(
                    mvs_input.dynamic_risk_scores.get(ref_id, 0.0),
                    mvs_input.dynamic_risk_scores.get(src_id, 0.0),
                )
                suitability *= (1.0 - 0.5 * dyn_risk)

                pair = MVSViewPair(
                    reference_frame_id=ref_id,
                    source_frame_id=src_id,
                    baseline_proxy=baseline,
                    relative_rotation_angle_deg=rot_ang_deg,
                    relative_translation_direction=t_dir,
                    overlap_estimate=overlap_proxy,
                    viewpoint_suitability_score=suitability,
                    selection_reason="Heuristic geometric baseline and overlap consensus",
                    dynamic_risk=dyn_risk,
                )
                scored_candidates.append(pair)
                per_edge_scores[edge] = suitability

            # Sort deterministically and select top source views
            scored_candidates.sort(key=lambda x: x.viewpoint_suitability_score, reverse=True)
            chosen = scored_candidates[:config.max_source_views]
            selected_edges.extend(chosen)

        return MVSViewGraph(
            frame_nodes=frame_ids,
            candidate_edges=candidate_edges,
            selected_edges=selected_edges,
            per_edge_scores=per_edge_scores,
            rejection_reasons=rejection_reasons,
        )


class GeometricDepthConsistencyChecker(DepthConsistencyChecker):
    """Reference implementation of cross-view geometric reprojection and depth consistency."""

    def check_consistency(
        self,
        ref_depth: DepthMap,
        src_depth: DepthMap,
        ref_pose: ExtrinsicPose,
        src_pose: ExtrinsicPose,
        ref_K: CameraIntrinsics,
        src_K: CameraIntrinsics,
        config: MVSConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
        H, W = ref_depth.height, ref_depth.width
        consistency_mask = np.zeros((H, W), dtype=bool)
        visibility_state = np.full((H, W), PointVisibilityState.INVALID_DEPTH.value, dtype=object)

        R_ref = np.array(ref_pose.rotation_matrix, dtype=np.float64)
        c_ref = np.array(ref_pose.translation_vector, dtype=np.float64)
        t_ref = -R_ref @ c_ref

        R_src = np.array(src_pose.rotation_matrix, dtype=np.float64)
        c_src = np.array(src_pose.translation_vector, dtype=np.float64)
        t_src = -R_src @ c_src

        for row in range(H):
            for col in range(W):
                if not ref_depth.valid_mask[row, col]:
                    continue

                z_ref = float(ref_depth.depth_array[row, col])
                u_ref, v_ref = float(col), float(row)

                # 1. Backproject reference pixel to 3D world
                X_w, ok = MVSGeometryMath.backproject_pixel(u_ref, v_ref, z_ref, ref_K, R_ref, t_ref)
                if not ok or X_w is None:
                    continue

                # 2. Project into source camera
                src_px, z_src_proj, ok_proj = MVSGeometryMath.project_world_point(X_w, src_K, R_src, t_src)
                if not ok_proj or src_px is None:
                    visibility_state[row, col] = PointVisibilityState.OCCLUDED.value
                    continue

                u_src, v_src = src_px
                # 3. Check source image bounds
                if not (0 <= u_src < src_depth.width and 0 <= v_src < src_depth.height):
                    visibility_state[row, col] = PointVisibilityState.OCCLUDED.value
                    continue

                # 4. Lookup source depth
                z_src_obs = src_depth.get_depth_at(u_src, v_src)
                if z_src_obs is None:
                    visibility_state[row, col] = PointVisibilityState.INCONSISTENT.value
                    continue

                # 5. Check relative depth consistency
                depth_diff = abs(z_src_proj - z_src_obs) / max(1e-6, z_src_proj)
                if depth_diff > config.relative_depth_consistency_tolerance:
                    visibility_state[row, col] = PointVisibilityState.INCONSISTENT.value
                    continue

                # 6. Reprojection back-check: backproject source point and project into reference camera
                X_w_src, ok_src = MVSGeometryMath.backproject_pixel(u_src, v_src, z_src_obs, src_K, R_src, t_src)
                if not ok_src or X_w_src is None:
                    visibility_state[row, col] = PointVisibilityState.INCONSISTENT.value
                    continue

                ref_px_back, _, ok_ref_back = MVSGeometryMath.project_world_point(X_w_src, ref_K, R_ref, t_ref)
                if not ok_ref_back or ref_px_back is None:
                    visibility_state[row, col] = PointVisibilityState.INCONSISTENT.value
                    continue

                reproj_dist = math.sqrt((ref_px_back[0] - u_ref)**2 + (ref_px_back[1] - v_ref)**2)
                if reproj_dist > config.reprojection_consistency_tolerance_px:
                    visibility_state[row, col] = PointVisibilityState.INCONSISTENT.value
                    continue

                # Passed both consistency checks
                consistency_mask[row, col] = True
                visibility_state[row, col] = PointVisibilityState.VALID.value

        return consistency_mask, visibility_state


class VoxelGridDensePointFusion(DensePointFusion):
    """Reference implementation of spatial voxel grid deduplication and multi-view fusion."""

    def fuse(
        self,
        observations: List[DensePointObservation],
        config: MVSConfig,
    ) -> DensePointCloud:
        if len(observations) == 0:
            return DensePointCloud(
                points=np.zeros((0, 3), dtype=np.float64),
                confidences=np.zeros((0,), dtype=np.float32),
                support_counts=np.zeros((0,), dtype=np.int32),
                source_frame_ids=[],
                visibility_states=[],
                validation_statuses=[],
                total_fused_points=0,
                mean_confidence=0.0,
                depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
                is_metric_scale=False,
            )

        voxel_res = max(1e-4, config.voxel_grid_resolution)
        grid: Dict[Tuple[int, int, int], List[DensePointObservation]] = {}

        # Spatial clustering via voxel grid
        for obs in observations:
            if obs.confidence < config.confidence_threshold:
                continue

            pt = obs.world_point
            key = (
                int(math.floor(pt[0] / voxel_res)),
                int(math.floor(pt[1] / voxel_res)),
                int(math.floor(pt[2] / voxel_res)),
            )
            if key not in grid:
                grid[key] = []
            grid[key].append(obs)

        fused_pts: List[np.ndarray] = []
        fused_confs: List[float] = []
        fused_supps: List[int] = []
        fused_frames: List[List[str]] = []
        fused_vis: List[PointVisibilityState] = []
        fused_vals: List[PointValidationStatus] = []

        # Deterministic sorting of grid keys
        sorted_keys = sorted(grid.keys())

        for key in sorted_keys:
            cluster = grid[key]
            total_supp = sum(c.source_view_support_count for c in cluster)
            unique_frames = sorted(list(set(c.reference_frame_id for c in cluster)))

            # Multi-view support threshold
            if len(unique_frames) < config.min_consistent_views:
                continue

            # Confidence-weighted coordinate centroid
            weights = np.array([c.confidence for c in cluster], dtype=np.float64)
            sum_w = float(np.sum(weights))
            if sum_w < 1e-8:
                mean_pt = np.mean(np.array([c.world_point for c in cluster]), axis=0)
                mean_conf = 0.0
            else:
                pts_arr = np.array([c.world_point for c in cluster], dtype=np.float64)
                mean_pt = np.sum(pts_arr * weights[:, None], axis=0) / sum_w
                mean_conf = float(sum_w / len(cluster))

            fused_pts.append(mean_pt)
            fused_confs.append(mean_conf)
            fused_supps.append(len(unique_frames))
            fused_frames.append(unique_frames)
            fused_vis.append(PointVisibilityState.VALID)
            fused_vals.append(PointValidationStatus.VALIDATED)

        n_pts = len(fused_pts)
        pts_arr = np.array(fused_pts, dtype=np.float64) if n_pts > 0 else np.zeros((0, 3), dtype=np.float64)
        confs_arr = np.array(fused_confs, dtype=np.float32) if n_pts > 0 else np.zeros((0,), dtype=np.float32)
        supps_arr = np.array(fused_supps, dtype=np.int32) if n_pts > 0 else np.zeros((0,), dtype=np.int32)
        mean_c = float(np.mean(confs_arr)) if n_pts > 0 else 0.0

        return DensePointCloud(
            points=pts_arr,
            confidences=confs_arr,
            support_counts=supps_arr,
            source_frame_ids=fused_frames,
            visibility_states=fused_vis,
            validation_statuses=fused_vals,
            total_fused_points=n_pts,
            mean_confidence=mean_c,
            depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
            is_metric_scale=False,
            provenance={"algorithm": "VoxelGridDensePointFusion", "voxel_resolution": voxel_res},
        )


class MVSValidator:
    """Validates MVS data structures against architectural and mathematical invariants."""

    @staticmethod
    def validate_mvs_input(
        mvs_input: MVSInput,
        config: MVSConfig,
    ) -> Tuple[bool, Optional[MVSFailureReason], List[str]]:
        """Validate MVSInput consistency and minimum requirements."""
        diags: List[str] = []

        if len(mvs_input.selected_frame_ids) < config.min_source_views + 1:
            diags.append(f"Insufficient frames: {len(mvs_input.selected_frame_ids)} < {config.min_source_views + 1}.")
            return False, MVSFailureReason.INSUFFICIENT_VALID_VIEWS, diags

        for fid in mvs_input.selected_frame_ids:
            if fid not in mvs_input.camera_intrinsics:
                diags.append(f"Missing intrinsics for frame {fid}.")
                return False, MVSFailureReason.INVALID_INTRINSICS, diags

            if fid not in mvs_input.camera_poses:
                diags.append(f"Missing camera pose for frame {fid}.")
                return False, MVSFailureReason.INVALID_POSE, diags

            p = mvs_input.camera_poses[fid]
            if not (np.all(np.isfinite(p.rotation_matrix)) and np.all(np.isfinite(p.translation_vector))):
                diags.append(f"Non-finite pose matrix or center for frame {fid}.")
                return False, MVSFailureReason.INVALID_POSE, diags

        return True, None, diags

    @staticmethod
    def validate_depth_map(
        depth_map: DepthMap,
        config: MVSConfig,
    ) -> Tuple[bool, Optional[MVSFailureReason], List[str]]:
        """Validate DepthMap dimensions, finite values, and valid depth range."""
        diags: List[str] = []

        if depth_map.width <= 0 or depth_map.height <= 0:
            diags.append(f"Invalid dimensions ({depth_map.width}x{depth_map.height}).")
            return False, MVSFailureReason.INCOMPATIBLE_IMAGE_DIMENSIONS, diags

        if depth_map.depth_array.shape != (depth_map.height, depth_map.width):
            diags.append("Depth array shape does not match map dimensions.")
            return False, MVSFailureReason.INCOMPATIBLE_IMAGE_DIMENSIONS, diags

        valid_depths = depth_map.depth_array[depth_map.valid_mask]
        if len(valid_depths) == 0:
            diags.append("No valid depths present in DepthMap.")
            return False, MVSFailureReason.DEPTH_ESTIMATION_FAILED, diags

        if np.any(valid_depths <= 0.0) or not np.all(np.isfinite(valid_depths)):
            diags.append("Non-positive or non-finite depth values marked as valid.")
            return False, MVSFailureReason.DEPTH_ESTIMATION_FAILED, diags

        return True, None, diags

    @staticmethod
    def validate_point_cloud(
        point_cloud: DensePointCloud,
        config: MVSConfig,
    ) -> Tuple[bool, Optional[MVSFailureReason], List[str]]:
        """Validate DensePointCloud integrity and coordinate invariants."""
        diags: List[str] = []

        if point_cloud.total_fused_points != len(point_cloud.points):
            diags.append("Point cloud count mismatch.")
            return False, MVSFailureReason.DENSE_FUSION_FAILED, diags

        if len(point_cloud.points) > 0 and not np.all(np.isfinite(point_cloud.points)):
            diags.append("Non-finite point coordinates in DensePointCloud.")
            return False, MVSFailureReason.DENSE_FUSION_FAILED, diags

        # Critical scientific invariant check: Dense MVS must not claim metric scale
        if point_cloud.is_metric_scale:
            diags.append("Violation of Phase 3E constraint: DensePointCloud claims metric scale without external calibration.")
            return False, MVSFailureReason.DENSE_FUSION_FAILED, diags

        return True, None, diags
