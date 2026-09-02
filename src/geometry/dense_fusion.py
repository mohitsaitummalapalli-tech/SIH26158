"""Phase 3E.3: Multi-View Dense Point Fusion Layer.

Fuses multi-view dense 3D point observations (from Phase 3E.2) into a consolidated,
multi-view validated dense point cloud (DensePointCloud) in relative reconstruction
coordinates (RECONSTRUCTION_UNITS).

KEY SCIENTIFIC & MATHEMATICAL INVARIANTS:
1. No Metric Scale Claim:
   All geometry remains strictly in RECONSTRUCTION_UNITS with is_metric_scale=False.
   Absolute metric scale is NOT established without independent certified ground truth.
2. Distinct-View Support Semantics:
   A point is only multi-view validated if it is supported by observations from at least
   `min_distinct_view_support` distinct reference/source camera frames. Observations
   from the same camera frame do NOT increase distinct-view support.
3. Explicit Geometric Compatibility (No Naive Proximity Merging):
   Observations merge only if:
   - Pairwise Euclidean distance <= spatial_distance_threshold in reconstruction units.
   - Total cluster bounding diameter <= max_cluster_diameter (prevents transitive chaining).
4. Deterministic Weighted Centroid:
   For an accepted cluster of observations {X_i} with heuristic confidences {c_i}:
       X_fused = sum(w_i * X_i) / sum(w_i)
   where w_i are heuristic weights (e.g. w_i = c_i).
5. Heuristic Confidence Semantics:
   Fused confidence is an aggregated HEURISTIC_SCORE in [0, 1], not a Bayesian probability,
   covariance matrix, or physical measurement uncertainty.
6. Full Provenance Preservation:
   Fused points retain full traceability back to contributing frame IDs, pixel coordinates,
   individual depth observations, and original confidence scores.
7. Determinism & Input-Order Invariance:
   Observations are canonically sorted before clustering, ensuring bit-exact identical outputs
   regardless of input observation list ordering.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Optional, List, Dict, Any, Tuple, Set

import numpy as np

from src.geometry.contracts import (
    CameraIntrinsics,
    ExtrinsicPose,
)

from src.geometry.mvs import (
    DepthUnit,
    PointVisibilityState,
    PointValidationStatus,
    DensePointObservation,
    DensePointCloud,
    DensePointFusion,
    MVSConfig,
)


class FusionRejectionReason(str, Enum):
    """Explicit rejection taxonomy for multi-view dense point fusion."""
    NON_FINITE_COORDINATES = "NON_FINITE_COORDINATES"
    NON_FINITE_CONFIDENCE = "NON_FINITE_CONFIDENCE"
    OUT_OF_BOUNDS_CONFIDENCE = "OUT_OF_BOUNDS_CONFIDENCE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    INSUFFICIENT_DISTINCT_VIEWS = "INSUFFICIENT_DISTINCT_VIEWS"
    ISOLATED_OBSERVATION = "ISOLATED_OBSERVATION"
    SPATIAL_CLUSTER_DIAMETER_EXCEEDED = "SPATIAL_CLUSTER_DIAMETER_EXCEEDED"
    DYNAMIC_RISK_EXCEEDED = "DYNAMIC_RISK_EXCEEDED"
    INVALID_VISIBILITY_STATE = "INVALID_VISIBILITY_STATE"
    INVALID_VALIDATION_STATUS = "INVALID_VALIDATION_STATUS"
    CLUSTER_MERGE_REJECTED = "CLUSTER_MERGE_REJECTED"


class FusionWeightingScheme(str, Enum):
    """Mathematical weighting scheme for cluster coordinate fusion."""
    CONFIDENCE_WEIGHTED = "CONFIDENCE_WEIGHTED"       # w_i = confidence_i (heuristic score)
    UNIFORM = "UNIFORM"                               # w_i = 1.0
    CONFIDENCE_SUPPORT_WEIGHTED = "CONFIDENCE_SUPPORT_WEIGHTED" # w_i = confidence_i * support_count_i


class SingleViewRetentionPolicy(str, Enum):
    """Policy for handling observations that lack multi-view support."""
    REJECT_SINGLE_VIEW = "REJECT_SINGLE_VIEW"         # Strictly require >= min_distinct_views
    RETAIN_AS_OBSERVED = "RETAIN_AS_OBSERVED"         # Retain single-view points tagged as OBSERVED


@dataclass(frozen=True)
class DenseFusionConfig:
    """Configurable engineering defaults (HEURISTIC_DEFAULT) for multi-view dense point fusion."""
    spatial_distance_threshold: float = 0.05            # HEURISTIC_DEFAULT: Max distance for merging in reconstruction units
    voxel_grid_resolution: float = 0.05                 # HEURISTIC_DEFAULT: Spatial hash bucket cell dimension
    min_distinct_view_support: int = 2                  # HEURISTIC_DEFAULT: Minimum distinct frames required for multi-view validation
    min_observation_confidence: float = 0.20            # HEURISTIC_DEFAULT: Minimum observation confidence threshold
    max_dynamic_risk: float = 0.80                      # HEURISTIC_DEFAULT: Maximum acceptable dynamic scene risk
    weighting_scheme: FusionWeightingScheme = FusionWeightingScheme.CONFIDENCE_WEIGHTED
    single_view_policy: SingleViewRetentionPolicy = SingleViewRetentionPolicy.REJECT_SINGLE_VIEW
    max_cluster_diameter: float = 0.10                  # HEURISTIC_DEFAULT: Max cluster bounding diameter (prevents transitive chaining)
    depth_unit: DepthUnit = DepthUnit.RECONSTRUCTION_UNITS
    is_metric_scale: bool = False
    config_version: str = "DenseFusionConfig_v1.0"


@dataclass
class FusedDensePoint:
    """Typed container for a single multi-view fused dense 3D point."""
    world_point: np.ndarray                             # (3,) float64 in RECONSTRUCTION_UNITS
    fused_confidence: float                             # HEURISTIC_SCORE in [0, 1]
    distinct_view_count: int                            # Number of distinct contributing frames
    total_observation_count: int                        # Total number of merged observations
    contributing_frame_ids: List[str]                   # Sorted unique frame IDs
    contributing_pixel_coords: List[Tuple[float, float]]# Pixel coordinates in each contributing frame
    contributing_depths: List[float]                    # Depths in contributing frames
    contributing_confidences: List[float]               # Confidences of contributing observations
    visibility_state: PointVisibilityState
    validation_status: PointValidationStatus
    cluster_spatial_std: float                          # Spatial standard deviation of observations in cluster
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "world_point": self.world_point.tolist(),
            "fused_confidence": self.fused_confidence,
            "distinct_view_count": self.distinct_view_count,
            "total_observation_count": self.total_observation_count,
            "contributing_frame_ids": self.contributing_frame_ids,
            "contributing_pixel_coords": self.contributing_pixel_coords,
            "contributing_depths": self.contributing_depths,
            "contributing_confidences": self.contributing_confidences,
            "visibility_state": self.visibility_state.value,
            "validation_status": self.validation_status.value,
            "cluster_spatial_std": self.cluster_spatial_std,
            "provenance": self.provenance,
        }


@dataclass
class DensePointCluster:
    """Spatial cluster of geometrically compatible dense point observations."""
    cluster_id: int
    observations: List[DensePointObservation] = field(default_factory=list)
    contributing_frames: Set[str] = field(default_factory=set)

    def can_accept(
        self,
        obs: DensePointObservation,
        spatial_thresh: float,
        max_cluster_diameter: float,
    ) -> bool:
        """Evaluate explicit geometric compatibility for adding an observation to this cluster.
        
        Requirements:
        1. Distance from observation to cluster centroid <= spatial_thresh.
        2. Distance from observation to every existing point in cluster <= max_cluster_diameter
           (strict guard against transitive chaining).
        """
        if len(self.observations) == 0:
            return True

        pt = obs.world_point
        # 1. Check distance to current centroid
        pts = np.array([o.world_point for o in self.observations], dtype=np.float64)
        centroid = np.mean(pts, axis=0)
        dist_to_centroid = float(np.linalg.norm(pt - centroid))
        if dist_to_centroid > spatial_thresh:
            return False

        # 2. Check maximum pairwise distance across cluster (prevent chaining)
        dists_to_members = np.linalg.norm(pts - pt, axis=1)
        if np.any(dists_to_members > max_cluster_diameter):
            return False

        return True

    def add_observation(self, obs: DensePointObservation) -> None:
        self.observations.append(obs)
        self.contributing_frames.add(obs.reference_frame_id)

    def compute_fused_point(
        self,
        config: DenseFusionConfig,
    ) -> Tuple[Optional[FusedDensePoint], Optional[FusionRejectionReason]]:
        """Compute the weighted fused 3D point and metadata for this cluster."""
        n_obs = len(self.observations)
        if n_obs == 0:
            return None, FusionRejectionReason.ISOLATED_OBSERVATION

        unique_frames = sorted(list(self.contributing_frames))
        distinct_count = len(unique_frames)

        # Check distinct view support
        if distinct_count < config.min_distinct_view_support:
            if config.single_view_policy == SingleViewRetentionPolicy.REJECT_SINGLE_VIEW:
                return None, FusionRejectionReason.INSUFFICIENT_DISTINCT_VIEWS
            status = PointValidationStatus.OBSERVED
        else:
            status = PointValidationStatus.VALIDATED

        pts = np.array([o.world_point for o in self.observations], dtype=np.float64)
        confs = [float(o.confidence) for o in self.observations]
        depths = [float(o.depth) for o in self.observations]
        pixels = [o.pixel_coord for o in self.observations]

        # Calculate heuristic fusion weights
        if config.weighting_scheme == FusionWeightingScheme.UNIFORM:
            weights = np.ones((n_obs,), dtype=np.float64)
        elif config.weighting_scheme == FusionWeightingScheme.CONFIDENCE_SUPPORT_WEIGHTED:
            weights = np.array(
                [max(1e-4, o.confidence * max(1, o.source_view_support_count)) for o in self.observations],
                dtype=np.float64,
            )
        else:  # CONFIDENCE_WEIGHTED
            weights = np.array([max(1e-4, o.confidence) for o in self.observations], dtype=np.float64)

        sum_w = float(np.sum(weights))
        if sum_w < 1e-8:
            weights = np.ones((n_obs,), dtype=np.float64)
            sum_w = float(n_obs)

        # Weighted centroid
        fused_xyz = np.sum(pts * weights[:, None], axis=0) / sum_w

        # Fused confidence: conservative weighted average (HEURISTIC_SCORE)
        fused_conf = float(np.sum(np.array(confs, dtype=np.float64) * weights) / sum_w)
        fused_conf = max(0.0, min(1.0, fused_conf))

        # Spatial dispersion within cluster
        if n_obs > 1:
            spatial_std = float(np.mean(np.std(pts, axis=0)))
        else:
            spatial_std = 0.0

        provenance = {
            "algorithm": "DensePointFusionEngine",
            "fusion_version": config.config_version,
            "weighting_scheme": config.weighting_scheme.value,
            "spatial_distance_threshold": config.spatial_distance_threshold,
            "distinct_view_count": distinct_count,
            "total_observations_merged": n_obs,
            "depth_unit": config.depth_unit.value,
            "is_metric_scale": config.is_metric_scale,
        }

        fused_pt = FusedDensePoint(
            world_point=fused_xyz,
            fused_confidence=fused_conf,
            distinct_view_count=distinct_count,
            total_observation_count=n_obs,
            contributing_frame_ids=unique_frames,
            contributing_pixel_coords=pixels,
            contributing_depths=depths,
            contributing_confidences=confs,
            visibility_state=PointVisibilityState.VALID,
            validation_status=status,
            cluster_spatial_std=spatial_std,
            provenance=provenance,
        )
        return fused_pt, None


@dataclass
class DenseFusionResult:
    """Comprehensive typed result from multi-view dense point fusion."""
    fused_points: List[FusedDensePoint]
    point_cloud: DensePointCloud
    total_input_observations: int
    total_fused_points: int
    total_rejected_observations: int
    rejection_breakdown: Dict[str, int]
    mean_fused_confidence: float
    mean_cluster_size: float
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_input_observations": self.total_input_observations,
            "total_fused_points": self.total_fused_points,
            "total_rejected_observations": self.total_rejected_observations,
            "rejection_breakdown": self.rejection_breakdown,
            "mean_fused_confidence": self.mean_fused_confidence,
            "mean_cluster_size": self.mean_cluster_size,
            "depth_unit": self.point_cloud.depth_unit.value,
            "is_metric_scale": self.point_cloud.is_metric_scale,
            "provenance": self.provenance,
        }


class DensePointFusionEngine(DensePointFusion):
    """Production multi-view dense point fusion engine implementing deterministic spatial clustering.
    
    Adheres strictly to the abstract DensePointFusion interface from Phase 3E.0 while providing
    complete mathematical clustering, distinct-view enforcement, and rich provenance.
    """

    def __init__(self, config: Optional[DenseFusionConfig] = None) -> None:
        self.config = config or DenseFusionConfig()

    def fuse(
        self,
        observations: List[DensePointObservation],
        config: Optional[MVSConfig] = None,
    ) -> DensePointCloud:
        """Interface compliance method matching Phase 3E.0 DensePointFusion contract."""
        if config is not None:
            fusion_cfg = DenseFusionConfig(
                spatial_distance_threshold=config.voxel_grid_resolution,
                voxel_grid_resolution=config.voxel_grid_resolution,
                min_distinct_view_support=config.min_consistent_views,
                min_observation_confidence=config.confidence_threshold,
                depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
                is_metric_scale=False,
            )
        else:
            fusion_cfg = self.config

        result = self.fuse_observations(observations, fusion_cfg)
        return result.point_cloud

    def fuse_observations(
        self,
        observations: List[DensePointObservation],
        config: Optional[DenseFusionConfig] = None,
    ) -> DenseFusionResult:
        """Execute deterministic multi-view dense fusion across all input observations."""
        cfg = config or self.config
        total_inputs = len(observations)
        rejection_breakdown: Dict[str, int] = {reason.value: 0 for reason in FusionRejectionReason}

        if total_inputs == 0:
            empty_cloud = DensePointCloud(
                points=np.zeros((0, 3), dtype=np.float64),
                confidences=np.zeros((0,), dtype=np.float32),
                support_counts=np.zeros((0,), dtype=np.int32),
                source_frame_ids=[],
                visibility_states=[],
                validation_statuses=[],
                total_fused_points=0,
                mean_confidence=0.0,
                depth_unit=cfg.depth_unit,
                is_metric_scale=cfg.is_metric_scale,
                provenance={"algorithm": "DensePointFusionEngine", "status": "EMPTY_INPUT"},
            )
            return DenseFusionResult(
                fused_points=[],
                point_cloud=empty_cloud,
                total_input_observations=0,
                total_fused_points=0,
                total_rejected_observations=0,
                rejection_breakdown=rejection_breakdown,
                mean_fused_confidence=0.0,
                mean_cluster_size=0.0,
                provenance={"algorithm": "DensePointFusionEngine", "status": "EMPTY_INPUT"},
            )

        # 1. Filter invalid observations
        valid_observations: List[DensePointObservation] = []
        for obs in observations:
            pt = obs.world_point
            # Check coordinate finiteness
            if pt is None or not (isinstance(pt, np.ndarray) and pt.shape == (3,)):
                rejection_breakdown[FusionRejectionReason.NON_FINITE_COORDINATES.value] += 1
                continue
            if not np.all(np.isfinite(pt)):
                rejection_breakdown[FusionRejectionReason.NON_FINITE_COORDINATES.value] += 1
                continue

            # Check confidence validity
            conf = obs.confidence
            if not (isinstance(conf, (int, float)) and math.isfinite(conf)):
                rejection_breakdown[FusionRejectionReason.NON_FINITE_CONFIDENCE.value] += 1
                continue
            if conf < 0.0 or conf > 1.0:
                rejection_breakdown[FusionRejectionReason.OUT_OF_BOUNDS_CONFIDENCE.value] += 1
                continue
            if conf < cfg.min_observation_confidence:
                rejection_breakdown[FusionRejectionReason.LOW_CONFIDENCE.value] += 1
                continue

            # Check dynamic risk in provenance if present
            dyn_risk = float(obs.provenance.get("dynamic_risk", 0.0))
            if dyn_risk > cfg.max_dynamic_risk:
                rejection_breakdown[FusionRejectionReason.DYNAMIC_RISK_EXCEEDED.value] += 1
                continue

            valid_observations.append(obs)

        # 2. Canonical deterministic sort of valid observations to ensure input-order invariance
        # Sort key: (ref_frame_id, pixel_u, pixel_v, X, Y, Z, confidence)
        def canonical_sort_key(o: DensePointObservation) -> Tuple[str, float, float, float, float, float, float]:
            u, v = o.pixel_coord if o.pixel_coord is not None else (0.0, 0.0)
            p = o.world_point
            return (
                str(o.reference_frame_id),
                round(float(u), 4),
                round(float(v), 4),
                round(float(p[0]), 6),
                round(float(p[1]), 6),
                round(float(p[2]), 6),
                round(float(o.confidence), 4),
            )

        sorted_observations = sorted(valid_observations, key=canonical_sort_key)

        # 3. Spatial Voxel Hashing & Geometric Clustering
        voxel_res = max(1e-4, cfg.voxel_grid_resolution)
        # voxel_index -> list of cluster_ids
        spatial_hash: Dict[Tuple[int, int, int], List[int]] = {}
        clusters: List[DensePointCluster] = []

        def get_voxel_coord(point: np.ndarray) -> Tuple[int, int, int]:
            return (
                int(math.floor(point[0] / voxel_res)),
                int(math.floor(point[1] / voxel_res)),
                int(math.floor(point[2] / voxel_res)),
            )

        for obs in sorted_observations:
            pt = obs.world_point
            v_coord = get_voxel_coord(pt)

            # Search current voxel and 26 adjacent 3D neighbor voxels
            matched_cluster: Optional[DensePointCluster] = None
            best_dist = float("inf")

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor_voxel = (v_coord[0] + dx, v_coord[1] + dy, v_coord[2] + dz)
                        if neighbor_voxel in spatial_hash:
                            for c_id in spatial_hash[neighbor_voxel]:
                                candidate_cluster = clusters[c_id]
                                if candidate_cluster.can_accept(
                                    obs=obs,
                                    spatial_thresh=cfg.spatial_distance_threshold,
                                    max_cluster_diameter=cfg.max_cluster_diameter,
                                ):
                                    # Distance to centroid
                                    c_pts = np.array([o.world_point for o in candidate_cluster.observations], dtype=np.float64)
                                    c_centroid = np.mean(c_pts, axis=0)
                                    dist = float(np.linalg.norm(pt - c_centroid))
                                    if dist < best_dist:
                                        best_dist = dist
                                        matched_cluster = candidate_cluster

            if matched_cluster is not None:
                matched_cluster.add_observation(obs)
                # Re-index cluster in spatial hash if it expanded into new voxel
                new_v = get_voxel_coord(pt)
                if new_v not in spatial_hash:
                    spatial_hash[new_v] = []
                if matched_cluster.cluster_id not in spatial_hash[new_v]:
                    spatial_hash[new_v].append(matched_cluster.cluster_id)
            else:
                # Create new cluster
                new_c_id = len(clusters)
                new_cluster = DensePointCluster(cluster_id=new_c_id)
                new_cluster.add_observation(obs)
                clusters.append(new_cluster)
                if v_coord not in spatial_hash:
                    spatial_hash[v_coord] = []
                spatial_hash[v_coord].append(new_c_id)

        # 4. Generate fused points from clusters
        fused_points_list: List[FusedDensePoint] = []
        cluster_sizes: List[int] = []

        for cluster in clusters:
            fused_pt, rej_reason = cluster.compute_fused_point(cfg)
            if rej_reason is not None:
                rejection_breakdown[rej_reason.value] += len(cluster.observations)
                continue
            if fused_pt is not None:
                fused_points_list.append(fused_pt)
                cluster_sizes.append(len(cluster.observations))

        # 5. Deterministic sorting of fused points
        fused_points_list.sort(
            key=lambda p: (
                round(float(p.world_point[0]), 6),
                round(float(p.world_point[1]), 6),
                round(float(p.world_point[2]), 6),
                p.contributing_frame_ids[0] if len(p.contributing_frame_ids) > 0 else "",
            )
        )

        n_fused = len(fused_points_list)
        if n_fused > 0:
            fused_xyz_arr = np.array([p.world_point for p in fused_points_list], dtype=np.float64)
            fused_confs_arr = np.array([p.fused_confidence for p in fused_points_list], dtype=np.float32)
            fused_supp_arr = np.array([p.distinct_view_count for p in fused_points_list], dtype=np.int32)
            fused_frame_lists = [p.contributing_frame_ids for p in fused_points_list]
            fused_vis_list = [p.visibility_state for p in fused_points_list]
            fused_val_list = [p.validation_status for p in fused_points_list]
            mean_conf = float(np.mean(fused_confs_arr))
            mean_c_size = float(np.mean(cluster_sizes))
        else:
            fused_xyz_arr = np.zeros((0, 3), dtype=np.float64)
            fused_confs_arr = np.zeros((0,), dtype=np.float32)
            fused_supp_arr = np.zeros((0,), dtype=np.int32)
            fused_frame_lists = []
            fused_vis_list = []
            fused_val_list = []
            mean_conf = 0.0
            mean_c_size = 0.0

        total_rejected = total_inputs - sum(cluster_sizes)

        point_cloud = DensePointCloud(
            points=fused_xyz_arr,
            confidences=fused_confs_arr,
            support_counts=fused_supp_arr,
            source_frame_ids=fused_frame_lists,
            visibility_states=fused_vis_list,
            validation_statuses=fused_val_list,
            total_fused_points=n_fused,
            mean_confidence=mean_conf,
            depth_unit=cfg.depth_unit,
            is_metric_scale=cfg.is_metric_scale,
            provenance={
                "algorithm": "DensePointFusionEngine",
                "fusion_config_version": cfg.config_version,
                "weighting_scheme": cfg.weighting_scheme.value,
                "spatial_distance_threshold": cfg.spatial_distance_threshold,
                "voxel_grid_resolution": cfg.voxel_grid_resolution,
                "min_distinct_view_support": cfg.min_distinct_view_support,
                "total_input_observations": total_inputs,
                "total_fused_points": n_fused,
            },
        )

        return DenseFusionResult(
            fused_points=fused_points_list,
            point_cloud=point_cloud,
            total_input_observations=total_inputs,
            total_fused_points=n_fused,
            total_rejected_observations=total_rejected,
            rejection_breakdown=rejection_breakdown,
            mean_fused_confidence=mean_conf,
            mean_cluster_size=mean_c_size,
            provenance={
                "algorithm": "DensePointFusionEngine",
                "config_version": cfg.config_version,
            },
        )
