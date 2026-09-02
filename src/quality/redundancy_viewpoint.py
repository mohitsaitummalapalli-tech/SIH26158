"""Frame Redundancy & Viewpoint Diversity Diagnostics Subsystem.

Computes explainable, deterministic pair-wise and sequence-local redundancy metrics,
visual appearance similarity, local feature overlap, spatial match distributions,
and trajectory-based baseline/orientation indicators.

SCIENTIFIC INTERPRETATION BOUNDARIES:
- VISUAL SIMILARITY != GEOMETRIC REDUNDANCY.
  Similar appearance can occur from different viewpoints, while distinct illumination
  can obscure identical geometry.
- FEATURE-MATCH COUNT != RECONSTRUCTION ACCURACY.
  Few feature matches may result from low surface texture or illumination shifts,
  NOT necessarily unsuitability for 3D reconstruction.
- GNSS TRAJECTORY BASELINE != OPTICAL CAMERA BASELINE.
  Without lever-arm and camera calibration, ENU coordinates represent aircraft navigation proxies.
- AIRCRAFT ATTITUDE != CAMERA VIEW ANGLE.
  Without gimbal angle calibration, aircraft orientation is a platform attitude proxy.
- This subsystem produces DIAGNOSTICS ONLY. It must NEVER drop frames or make keyframe decisions.
"""

import json
import math
import numpy as np
import cv2
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple

from src.preprocessing.decoder import DecodedFrame, DecodeStatus
from src.quality.assessment import FrameQualityAnalyzer
from src.quality.dynamic_scene import DynamicSceneReport


@dataclass(frozen=True)
class FrameRedundancyConfig:
    """Configurable heuristic thresholds and search window for redundancy diagnostics.
    
    All default thresholds are heuristic defaults (HEURISTIC_DEFAULT) requiring empirical
    calibration on representative flight validation datasets.
    """
    temporal_window_frames: int = 5              # HEURISTIC_DEFAULT: Number of prior neighbor frames to evaluate
    orb_max_features: int = 500                  # HEURISTIC_DEFAULT: Max ORB keypoints per frame
    orb_scale_factor: float = 1.2                # HEURISTIC_DEFAULT: ORB pyramid decimation ratio
    orb_n_levels: int = 4                        # HEURISTIC_DEFAULT: ORB pyramid levels
    visual_similarity_threshold: float = 0.90    # HEURISTIC_DEFAULT: Appearance similarity threshold for redundancy
    match_ratio_redundancy_threshold: float = 0.60 # HEURISTIC_DEFAULT: Match ratio threshold for redundancy
    trajectory_baseline_threshold_meters: float = 1.0 # HEURISTIC_DEFAULT: Trajectory distance threshold (m)
    orientation_change_threshold_degrees: float = 2.0 # HEURISTIC_DEFAULT: Trajectory orientation angle threshold (deg)
    temporal_redundancy_time_threshold_seconds: float = 0.5 # HEURISTIC_DEFAULT: Temporal threshold (s)
    grid_occupancy_divisions: int = 4            # HEURISTIC_DEFAULT: 4x4 spatial grid for match distribution
    config_version: str = "RedundancyViewpoint_v1.0"


@dataclass(frozen=True)
class FramePairRelation:
    """Diagnostic relationship and similarity metrics between a target frame and a candidate reference frame."""
    frame_a_id: str                              # Target frame ID
    frame_b_id: str                              # Reference candidate frame ID
    frame_a_timestamp: float
    frame_b_timestamp: float
    delta_t_seconds: float
    visual_similarity_score: float               # Normalized appearance cross-correlation in [0.0, 1.0]
    keypoints_a_count: int
    keypoints_b_count: int
    descriptor_match_count: int
    match_ratio: float                           # matches / min(keypoints_a, keypoints_b)
    spatial_coverage_convex_hull_ratio: float    # Area of match convex hull / image area
    match_grid_occupancy_ratio: float            # Fraction of grid cells containing >= 1 match
    trajectory_baseline_meters: Optional[float]  # Euclidean distance between ENU positions (m)
    trajectory_orientation_change_degrees: Optional[float] # Geodesic quaternion angle difference (deg)
    matches_inside_dynamic_regions_fraction: Optional[float] # Fraction of matches falling inside candidate dynamic boxes
    high_visual_similarity_indicator: bool
    low_feature_novelty_indicator: bool
    low_trajectory_baseline_indicator: bool
    low_orientation_change_indicator: bool
    high_temporal_redundancy_indicator: bool
    diagnostics: List[str] = field(default_factory=list)


@dataclass
class FrameRedundancyReport:
    """Comprehensive frame redundancy and viewpoint diversity diagnostic report for a single video frame."""
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    source_video: str
    pair_relations: List[FramePairRelation]
    mean_neighbor_match_ratio: float
    mean_neighbor_baseline_meters: Optional[float]
    diagnostics: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    config_version: str = "RedundancyViewpoint_v1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class FrameRedundancyViewpointAnalyzer:
    """Deterministic engine for computing pair-wise redundancy, feature overlap, and viewpoint diversity."""

    @classmethod
    def compute_visual_similarity(
        cls, gray_a: np.ndarray, gray_b: np.ndarray
    ) -> float:
        """Compute normalized zero-mean cross-correlation (ZNCC) in [0.0, 1.0]."""
        if gray_a.shape != gray_b.shape:
            return 0.0
        
        # Downscale for fast robust appearance correlation
        h, w = gray_a.shape
        if h > 120 or w > 120:
            ga = cv2.resize(gray_a, (120, 120), interpolation=cv2.INTER_AREA).astype(np.float64)
            gb = cv2.resize(gray_b, (120, 120), interpolation=cv2.INTER_AREA).astype(np.float64)
        else:
            ga = gray_a.astype(np.float64)
            gb = gray_b.astype(np.float64)

        ga_norm = ga - np.mean(ga)
        gb_norm = gb - np.mean(gb)
        std_a = np.std(ga_norm)
        std_b = np.std(gb_norm)

        if std_a < 1e-4 or std_b < 1e-4:
            # Uniform or flat image
            diff = float(np.mean(np.abs(ga - gb)))
            return max(0.0, min(1.0, 1.0 - (diff / 255.0)))

        zncc = float(np.mean(ga_norm * gb_norm) / (std_a * std_b))
        # Map [-1, 1] to [0, 1]
        similarity = max(0.0, min(1.0, 0.5 * (zncc + 1.0)))
        return round(similarity, 4)

    @classmethod
    def extract_orb_features(
        cls, gray: np.ndarray, max_features: int = 500
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Extract deterministic ORB keypoints and descriptors."""
        orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=4,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20,
        )
        kps, desc = orb.detectAndCompute(gray.astype(np.uint8), None)
        return kps, desc

    @classmethod
    def match_features(
        cls, desc_a: Optional[np.ndarray], desc_b: Optional[np.ndarray]
    ) -> List[cv2.DMatch]:
        """Perform mutual cross-check matching between binary descriptors."""
        if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
            return []
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc_a, desc_b)
        # Sort matches by Hamming distance
        matches = sorted(matches, key=lambda m: m.distance)
        return matches

    @classmethod
    def compute_spatial_match_distribution(
        cls,
        kps_a: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        image_shape: Tuple[int, int],
        grid_divisions: int = 4,
    ) -> Tuple[float, float]:
        """Compute convex hull coverage ratio and grid occupancy ratio of matches in image A."""
        h, w = image_shape
        total_area = max(1, h * w)
        if not matches or len(kps_a) == 0:
            return 0.0, 0.0

        pts = np.float32([kps_a[m.queryIdx].pt for m in matches])
        if len(pts) < 3:
            return 0.0, 0.0

        # 1. Convex Hull Coverage Ratio
        hull = cv2.convexHull(pts)
        hull_area = float(cv2.contourArea(hull))
        coverage_ratio = max(0.0, min(1.0, hull_area / total_area))

        # 2. Grid Occupancy Ratio
        divs = max(1, grid_divisions)
        grid_occupied = set()
        for pt in pts:
            gx = min(divs - 1, max(0, int(pt[0] / (w / divs))))
            gy = min(divs - 1, max(0, int(pt[1] / (h / divs))))
            grid_occupied.add((gy, gx))

        occupancy_ratio = len(grid_occupied) / float(divs * divs)

        return round(coverage_ratio, 4), round(occupancy_ratio, 4)

    @classmethod
    def compute_quaternion_geodesic_angle(
        cls, q_a: Tuple[float, float, float, float], q_b: Tuple[float, float, float, float]
    ) -> float:
        """Compute geodesic rotation angle (degrees) between two unit quaternions (w, x, y, z).
        
        theta = 2 * arccos(|q_a . q_b|)
        """
        dot = abs(q_a[0]*q_b[0] + q_a[1]*q_b[1] + q_a[2]*q_b[2] + q_a[3]*q_b[3])
        dot = min(1.0, max(-1.0, dot))
        angle_rad = 2.0 * math.acos(dot)
        return round(math.degrees(angle_rad), 4)

    @classmethod
    def evaluate_pair_relation(
        cls,
        frame_a: DecodedFrame,
        frame_b: DecodedFrame,
        enu_pos_a: Optional[Tuple[float, float, float]] = None,
        enu_pos_b: Optional[Tuple[float, float, float]] = None,
        quat_a: Optional[Tuple[float, float, float, float]] = None,
        quat_b: Optional[Tuple[float, float, float, float]] = None,
        dynamic_report_a: Optional[DynamicSceneReport] = None,
        config: Optional[FrameRedundancyConfig] = None,
    ) -> FramePairRelation:
        """Evaluate detailed pair-wise redundancy and viewpoint diversity metrics."""
        cfg = config or FrameRedundancyConfig()
        diagnostics: List[str] = []

        dt = abs(frame_a.timestamp_seconds - frame_b.timestamp_seconds)

        gray_a = FrameQualityAnalyzer.rgb_to_luminance(frame_a.data)
        gray_b = FrameQualityAnalyzer.rgb_to_luminance(frame_b.data)
        h, w = gray_a.shape

        # 1. Visual Appearance Similarity
        vis_sim = cls.compute_visual_similarity(gray_a, gray_b)

        # 2. Local Feature Extraction & Matching
        kps_a, desc_a = cls.extract_orb_features(gray_a, max_features=cfg.orb_max_features)
        kps_b, desc_b = cls.extract_orb_features(gray_b, max_features=cfg.orb_max_features)
        matches = cls.match_features(desc_a, desc_b)

        kp_a_count = len(kps_a)
        kp_b_count = len(kps_b)
        match_count = len(matches)

        denom = min(kp_a_count, kp_b_count) if min(kp_a_count, kp_b_count) > 0 else 1
        match_ratio = round(match_count / float(denom), 4) if min(kp_a_count, kp_b_count) > 0 else 0.0

        # Spatial Distribution
        cov_ratio, occ_ratio = cls.compute_spatial_match_distribution(kps_a, matches, (h, w), cfg.grid_occupancy_divisions)

        if kp_a_count < 15 or kp_b_count < 15:
            diagnostics.append("INSUFFICIENT_FEATURE_EVIDENCE: Low keypoint count in one or both frames.")

        # 3. Trajectory-Based Baseline
        traj_baseline: Optional[float] = None
        if enu_pos_a is not None and enu_pos_b is not None:
            dist = math.sqrt(
                (enu_pos_a[0] - enu_pos_b[0])**2 +
                (enu_pos_a[1] - enu_pos_b[1])**2 +
                (enu_pos_a[2] - enu_pos_b[2])**2
            )
            traj_baseline = round(dist, 4)

        # 4. Orientation Change
        orient_change: Optional[float] = None
        if quat_a is not None and quat_b is not None:
            orient_change = cls.compute_quaternion_geodesic_angle(quat_a, quat_b)

        # 5. Dynamic-Scene Integration
        dynamic_match_frac: Optional[float] = None
        if dynamic_report_a and dynamic_report_a.candidate_regions and matches:
            pts_a = [kps_a[m.queryIdx].pt for m in matches]
            inside_dynamic = 0
            for pt in pts_a:
                px, py = pt[0], pt[1]
                for reg in dynamic_report_a.candidate_regions:
                    y_min, x_min, y_max, x_max = reg.bbox
                    if x_min <= px <= x_max and y_min <= py <= y_max:
                        inside_dynamic += 1
                        break
            dynamic_match_frac = round(inside_dynamic / float(len(matches)), 4)

        # 6. Redundancy Indicators (Separated)
        high_vis = vis_sim >= cfg.visual_similarity_threshold
        low_feat = match_ratio >= cfg.match_ratio_redundancy_threshold
        low_base = (traj_baseline is not None) and (traj_baseline < cfg.trajectory_baseline_threshold_meters)
        low_orient = (orient_change is not None) and (orient_change < cfg.orientation_change_threshold_degrees)
        high_temp = dt < cfg.temporal_redundancy_time_threshold_seconds

        return FramePairRelation(
            frame_a_id=frame_a.frame_id,
            frame_b_id=frame_b.frame_id,
            frame_a_timestamp=frame_a.timestamp_seconds,
            frame_b_timestamp=frame_b.timestamp_seconds,
            delta_t_seconds=round(dt, 4),
            visual_similarity_score=vis_sim,
            keypoints_a_count=kp_a_count,
            keypoints_b_count=kp_b_count,
            descriptor_match_count=match_count,
            match_ratio=match_ratio,
            spatial_coverage_convex_hull_ratio=cov_ratio,
            match_grid_occupancy_ratio=occ_ratio,
            trajectory_baseline_meters=traj_baseline,
            trajectory_orientation_change_degrees=orient_change,
            matches_inside_dynamic_regions_fraction=dynamic_match_frac,
            high_visual_similarity_indicator=high_vis,
            low_feature_novelty_indicator=low_feat,
            low_trajectory_baseline_indicator=low_base,
            low_orientation_change_indicator=low_orient,
            high_temporal_redundancy_indicator=high_temp,
            diagnostics=diagnostics,
        )

    @classmethod
    def analyze_frame_redundancy(
        cls,
        target_frame: DecodedFrame,
        prior_frames: List[DecodedFrame],
        target_enu_pos: Optional[Tuple[float, float, float]] = None,
        prior_enu_positions: Optional[List[Tuple[float, float, float]]] = None,
        target_quat: Optional[Tuple[float, float, float, float]] = None,
        prior_quats: Optional[List[Tuple[float, float, float, float]]] = None,
        dynamic_report: Optional[DynamicSceneReport] = None,
        config: Optional[FrameRedundancyConfig] = None,
    ) -> FrameRedundancyReport:
        """Analyze redundancy and viewpoint diversity against candidate prior frames within a temporal window."""
        cfg = config or FrameRedundancyConfig()
        diagnostics: List[str] = []

        if target_frame.decode_status != DecodeStatus.SUCCESS or target_frame.data is None:
            raise ValueError(f"Cannot analyze frame '{target_frame.frame_id}' with invalid decode status.")

        # Windowed prior candidate frames (O(W) bounded complexity)
        window = prior_frames[-cfg.temporal_window_frames:] if cfg.temporal_window_frames > 0 else prior_frames
        enu_window = prior_enu_positions[-len(window):] if prior_enu_positions else [None] * len(window)
        quat_window = prior_quats[-len(window):] if prior_quats else [None] * len(window)

        pair_relations: List[FramePairRelation] = []
        match_ratios: List[float] = []
        baselines: List[float] = []

        for ref_frame, ref_enu, ref_quat in zip(window, enu_window, quat_window):
            if ref_frame.decode_status != DecodeStatus.SUCCESS or ref_frame.data is None:
                continue

            rel = cls.evaluate_pair_relation(
                frame_a=target_frame,
                frame_b=ref_frame,
                enu_pos_a=target_enu_pos,
                enu_pos_b=ref_enu,
                quat_a=target_quat,
                quat_b=ref_quat,
                dynamic_report_a=dynamic_report,
                config=cfg,
            )
            pair_relations.append(rel)
            match_ratios.append(rel.match_ratio)
            if rel.trajectory_baseline_meters is not None:
                baselines.append(rel.trajectory_baseline_meters)

        mean_match_ratio = round(float(np.mean(match_ratios)), 4) if match_ratios else 0.0
        mean_baseline = round(float(np.mean(baselines)), 4) if baselines else None

        if not pair_relations:
            diagnostics.append("First frame or no valid prior candidate frames in temporal window.")

        provenance = {
            "target_frame_id": target_frame.frame_id,
            "target_frame_index": target_frame.frame_index,
            "target_timestamp_seconds": target_frame.timestamp_seconds,
            "evaluated_pairs_count": len(pair_relations),
            "window_frames_limit": cfg.temporal_window_frames,
            "feature_detector": "OpenCV_ORB_v1.0",
        }

        return FrameRedundancyReport(
            frame_id=target_frame.frame_id,
            frame_index=target_frame.frame_index,
            timestamp_seconds=target_frame.timestamp_seconds,
            source_video=target_frame.source_video,
            pair_relations=pair_relations,
            mean_neighbor_match_ratio=mean_match_ratio,
            mean_neighbor_baseline_meters=mean_baseline,
            diagnostics=diagnostics,
            provenance=provenance,
            config_version=cfg.config_version,
        )
