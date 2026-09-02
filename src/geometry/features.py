"""Phase 3A: Classical Feature Extraction & Robust Descriptor Matching.

Implements a deterministic, explainable classical 2D feature extraction and
descriptor matching subsystem. Converts canonical RGB keyframes into detected
keypoints, binary ORB descriptors, candidate descriptor correspondences, and
spatial distribution diagnostics.

IMPORTANT SCIENTIFIC DISTINCTION:
- Descriptor similarity != Geometric correctness.
- This module outputs candidate descriptor matches only.
- Epipolar geometric verification and RANSAC filtering belong strictly to Phase 3B.
- All image coordinates are directly observed in pixel raster space: (u, v) in PIXEL_OBSERVED.
"""

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Union

import cv2
import numpy as np

from src.geometry.contracts import (
    MeasurementType,
    FeatureKeypoint,
    FeatureCorrespondences,
)


class FeatureDetectorType(str, Enum):
    """Supported 2D feature detector architectures."""
    ORB = "ORB"


class DescriptorMatcherType(str, Enum):
    """Supported descriptor matching algorithms."""
    BRUTE_FORCE_HAMMING = "BRUTE_FORCE_HAMMING"


class MatchingStrategy(str, Enum):
    """Candidate match filtering strategies."""
    RATIO_TEST = "RATIO_TEST"
    MUTUAL_CONSISTENCY = "MUTUAL_CONSISTENCY"
    RATIO_AND_MUTUAL = "RATIO_AND_MUTUAL"


class FeatureFailureReason(str, Enum):
    """Explicit, non-silent failure taxonomy for Phase 3A feature extraction and matching."""
    NO_FEATURES_DETECTED = "NO_FEATURES_DETECTED"
    INSUFFICIENT_FEATURES = "INSUFFICIENT_FEATURES"
    DESCRIPTOR_EXTRACTION_FAILED = "DESCRIPTOR_EXTRACTION_FAILED"
    NO_CANDIDATE_MATCHES = "NO_CANDIDATE_MATCHES"
    INSUFFICIENT_DESCRIPTOR_MATCHES = "INSUFFICIENT_DESCRIPTOR_MATCHES"
    INVALID_IMAGE = "INVALID_IMAGE"
    UNSUPPORTED_FEATURE_CONFIGURATION = "UNSUPPORTED_FEATURE_CONFIGURATION"


@dataclass(frozen=True)
class FeatureConfig:
    """Configurable heuristic defaults (HEURISTIC_DEFAULT) for feature extraction and matching.
    
    All numerical parameters are empirical engineering defaults, NOT universal mathematical truths.
    """
    detector_type: FeatureDetectorType = FeatureDetectorType.ORB
    max_features: int = 2000                          # HEURISTIC_DEFAULT
    scale_factor: float = 1.2                         # HEURISTIC_DEFAULT
    n_levels: int = 8                                 # HEURISTIC_DEFAULT
    edge_threshold: int = 31                          # HEURISTIC_DEFAULT
    first_level: int = 0                              # HEURISTIC_DEFAULT
    wta_k: int = 2                                    # HEURISTIC_DEFAULT
    score_type: str = "HARRIS_SCORE"                  # HEURISTIC_DEFAULT
    patch_size: int = 31                              # HEURISTIC_DEFAULT
    fast_threshold: int = 20                          # HEURISTIC_DEFAULT
    min_features_threshold: int = 100                 # HEURISTIC_DEFAULT
    matching_strategy: MatchingStrategy = MatchingStrategy.RATIO_AND_MUTUAL
    lowe_ratio: float = 0.75                          # HEURISTIC_DEFAULT
    min_accepted_matches: int = 30                    # HEURISTIC_DEFAULT
    max_descriptor_distance: float = 64.0             # HEURISTIC_DEFAULT (Hamming distance for 256-bit ORB)
    grid_rows: int = 8                                # HEURISTIC_DEFAULT
    grid_cols: int = 8                                # HEURISTIC_DEFAULT
    config_version: str = "FeatureConfig_v1.0"


@dataclass(frozen=True)
class SpatialMatchDiagnostics:
    """Deterministic spatial distribution metrics for matched keypoints."""
    grid_occupancy_ratio: float = 0.0                 # Occupied cells / total cells in grid
    occupied_cell_count: int = 0
    total_cell_count: int = 64
    convex_hull_area_fraction: float = 0.0            # Convex hull area / total image area in [0, 1]
    normalized_bounding_box: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0) # (min_u, min_v, max_u, max_v)
    edge_concentration_indicator: float = 0.0         # Fraction of matches in outer 10% border
    spatial_entropy: float = 0.0                      # Normalized Shannon entropy across grid bins [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureExtractionResult:
    """Complete output of 2D feature extraction for a single frame."""
    frame_id: str
    width: int
    height: int
    keypoint_count: int
    keypoints_xy: np.ndarray                          # Shape (N, 2) float64 pixel coordinates (u, v) in PIXEL_OBSERVED
    keypoint_scales: np.ndarray                       # Shape (N,) float64 diameter in pixels
    keypoint_angles: np.ndarray                       # Shape (N,) float64 orientation in degrees [-180, 180]
    keypoint_responses: np.ndarray                    # Shape (N,) float64 response strength
    keypoint_octaves: np.ndarray                      # Shape (N,) int32 pyramid octave
    descriptors: np.ndarray                           # Shape (N, 32) uint8 256-bit binary descriptors
    descriptor_dtype: str = "uint8"
    descriptor_dim: int = 32                          # 32 bytes = 256 bits for ORB
    detector_type: str = "ORB"
    preprocessing_status: str = "CONVERTED_RGB_TO_GRAYSCALE_BT601"
    measurement_type: MeasurementType = MeasurementType.DIRECTLY_OBSERVED
    status: str = "SUCCESS"
    failure_reason: Optional[FeatureFailureReason] = None
    diagnostics: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def get_keypoints(self) -> List[FeatureKeypoint]:
        """Convert extracted points to typed FeatureKeypoint contracts."""
        kps = []
        for i in range(self.keypoint_count):
            kps.append(FeatureKeypoint(
                x=float(self.keypoints_xy[i, 0]),
                y=float(self.keypoints_xy[i, 1]),
                octave=int(self.keypoint_octaves[i]),
                response=float(self.keypoint_responses[i]),
                angle=float(self.keypoint_angles[i]),
                measurement_type=MeasurementType.DIRECTLY_OBSERVED,
            ))
        return kps

    def to_dict(self) -> Dict[str, Any]:
        """Serialize extraction metadata to dictionary."""
        return {
            "frame_id": self.frame_id,
            "width": self.width,
            "height": self.height,
            "keypoint_count": self.keypoint_count,
            "descriptor_dtype": self.descriptor_dtype,
            "descriptor_dim": self.descriptor_dim,
            "detector_type": self.detector_type,
            "preprocessing_status": self.preprocessing_status,
            "measurement_type": self.measurement_type.value,
            "status": self.status,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "diagnostics": self.diagnostics,
            "provenance": self.provenance,
        }


@dataclass
class FeatureMatchResult:
    """Output of candidate descriptor matching between two frames.
    
    NOTE: Candidate descriptor matches are NOT geometrically verified inliers.
    """
    frame_a_id: str
    frame_b_id: str
    candidate_match_count: int
    accepted_match_count: int
    indices_a: np.ndarray                             # Shape (M,) int32 indices in frame A
    indices_b: np.ndarray                             # Shape (M,) int32 indices in frame B
    points_a: np.ndarray                              # Shape (M, 2) float64 pixel coordinates (u, v) in Frame A
    points_b: np.ndarray                              # Shape (M, 2) float64 pixel coordinates (u, v) in Frame B
    descriptor_distances: np.ndarray                  # Shape (M,) float64 Hamming distances
    min_distance: float = 0.0
    median_distance: float = 0.0
    mean_distance: float = 0.0
    percentile_90_distance: float = 0.0
    acceptance_ratio: float = 0.0                     # accepted_match_count / max(1, candidate_match_count)
    spatial_diagnostics_a: SpatialMatchDiagnostics = field(default_factory=SpatialMatchDiagnostics)
    spatial_diagnostics_b: SpatialMatchDiagnostics = field(default_factory=SpatialMatchDiagnostics)
    matching_strategy: str = "RATIO_AND_MUTUAL"
    measurement_type: MeasurementType = MeasurementType.ESTIMATED
    status: str = "SUCCESS"
    failure_reason: Optional[FeatureFailureReason] = None
    diagnostics: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_correspondences(self) -> FeatureCorrespondences:
        """Convert accepted matches into a FeatureCorrespondences contract for Phase 3B."""
        return FeatureCorrespondences(
            frame_a_id=self.frame_a_id,
            frame_b_id=self.frame_b_id,
            points_a=self.points_a,
            points_b=self.points_b,
            descriptor_distances=self.descriptor_distances,
            match_count=self.accepted_match_count,
            descriptor_type=f"ORB_256BIT_{self.matching_strategy}",
            provenance=self.provenance,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert match result metadata to dictionary."""
        return {
            "frame_a_id": self.frame_a_id,
            "frame_b_id": self.frame_b_id,
            "candidate_match_count": self.candidate_match_count,
            "accepted_match_count": self.accepted_match_count,
            "acceptance_ratio": self.acceptance_ratio,
            "min_distance": self.min_distance,
            "median_distance": self.median_distance,
            "mean_distance": self.mean_distance,
            "percentile_90_distance": self.percentile_90_distance,
            "spatial_occupancy_a": self.spatial_diagnostics_a.grid_occupancy_ratio,
            "spatial_occupancy_b": self.spatial_diagnostics_b.grid_occupancy_ratio,
            "matching_strategy": self.matching_strategy,
            "measurement_type": self.measurement_type.value,
            "status": self.status,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "diagnostics": self.diagnostics,
            "provenance": self.provenance,
        }


class SpatialDistributionCalculator:
    """Calculates deterministic 2D spatial distribution metrics for image coordinates."""

    @staticmethod
    def compute(
        points_xy: np.ndarray,
        width: int,
        height: int,
        grid_rows: int = 8,
        grid_cols: int = 8,
    ) -> SpatialMatchDiagnostics:
        """Compute grid occupancy, bounding box, edge concentration, and spatial entropy."""
        total_cells = grid_rows * grid_cols
        if len(points_xy) == 0 or width <= 0 or height <= 0:
            return SpatialMatchDiagnostics(
                grid_occupancy_ratio=0.0,
                occupied_cell_count=0,
                total_cell_count=total_cells,
                convex_hull_area_fraction=0.0,
                normalized_bounding_box=(0.0, 0.0, 0.0, 0.0),
                edge_concentration_indicator=0.0,
                spatial_entropy=0.0,
            )

        pts = np.asarray(points_xy, dtype=np.float64)
        u = pts[:, 0]
        v = pts[:, 1]

        # Normalized coordinates in [0, 1]
        u_norm = np.clip(u / float(width), 0.0, 1.0)
        v_norm = np.clip(v / float(height), 0.0, 1.0)

        # Bounding box
        min_u, max_u = float(np.min(u_norm)), float(np.max(u_norm))
        min_v, max_v = float(np.min(v_norm)), float(np.max(v_norm))
        bbox = (min_u, min_v, max_u, max_v)

        # Grid cell occupancy
        col_indices = np.clip((u_norm * grid_cols).astype(int), 0, grid_cols - 1)
        row_indices = np.clip((v_norm * grid_rows).astype(int), 0, grid_rows - 1)
        cell_keys = row_indices * grid_cols + col_indices
        unique_cells, counts = np.unique(cell_keys, return_counts=True)
        occupied_count = int(len(unique_cells))
        occupancy_ratio = float(occupied_count / total_cells)

        # Spatial entropy: - sum(p_i * log2(p_i)) / log2(total_cells)
        probs = counts / float(len(pts))
        entropy_raw = -float(np.sum(probs * np.log2(probs + 1e-12)))
        max_entropy = math.log2(total_cells) if total_cells > 1 else 1.0
        norm_entropy = float(np.clip(entropy_raw / max_entropy, 0.0, 1.0))

        # Edge concentration (points in outer 10% border)
        is_edge = (u_norm < 0.10) | (u_norm > 0.90) | (v_norm < 0.10) | (v_norm > 0.90)
        edge_concentration = float(np.mean(is_edge.astype(float)))

        # Convex hull area fraction
        hull_fraction = 0.0
        if len(pts) >= 3:
            pts_cv = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
            hull = cv2.convexHull(pts_cv)
            hull_area = cv2.contourArea(hull)
            total_img_area = float(width * height)
            hull_fraction = float(np.clip(hull_area / total_img_area, 0.0, 1.0))

        return SpatialMatchDiagnostics(
            grid_occupancy_ratio=occupancy_ratio,
            occupied_cell_count=occupied_count,
            total_cell_count=total_cells,
            convex_hull_area_fraction=hull_fraction,
            normalized_bounding_box=bbox,
            edge_concentration_indicator=edge_concentration,
            spatial_entropy=norm_entropy,
        )


class ClassicalFeatureExtractor:
    """Classical 2D feature extractor using open OpenCV ORB baseline."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        if self.config.detector_type != FeatureDetectorType.ORB:
            raise ValueError(f"Unsupported detector type: {self.config.detector_type}")

        score_type_cv = cv2.ORB_HARRIS_SCORE if self.config.score_type == "HARRIS_SCORE" else cv2.ORB_FAST_SCORE
        self._orb = cv2.ORB.create(
            nfeatures=self.config.max_features,
            scaleFactor=self.config.scale_factor,
            nlevels=self.config.n_levels,
            edgeThreshold=self.config.edge_threshold,
            firstLevel=self.config.first_level,
            WTA_K=self.config.wta_k,
            scoreType=score_type_cv,
            patchSize=self.config.patch_size,
            fastThreshold=self.config.fast_threshold,
        )

    def extract(self, frame_or_rgb: Any, frame_id: str = "unnamed") -> FeatureExtractionResult:
        """Extract 2D ORB features and binary descriptors from canonical RGB image."""
        # 1. Extract RGB array and validate
        rgb = None
        if isinstance(frame_or_rgb, np.ndarray):
            rgb = frame_or_rgb
        elif hasattr(frame_or_rgb, "data") and getattr(frame_or_rgb, "data") is not None:
            rgb = frame_or_rgb.data
            if hasattr(frame_or_rgb, "frame_id"):
                frame_id = getattr(frame_or_rgb, "frame_id", frame_id)
        elif hasattr(frame_or_rgb, "rgb_array") and getattr(frame_or_rgb, "rgb_array") is not None:
            rgb = frame_or_rgb.rgb_array
            if hasattr(frame_or_rgb, "frame_id"):
                frame_id = getattr(frame_or_rgb, "frame_id", frame_id)

        if rgb is None or not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            return FeatureExtractionResult(
                frame_id=frame_id,
                width=0,
                height=0,
                keypoint_count=0,
                keypoints_xy=np.empty((0, 2), dtype=np.float64),
                keypoint_scales=np.empty((0,), dtype=np.float64),
                keypoint_angles=np.empty((0,), dtype=np.float64),
                keypoint_responses=np.empty((0,), dtype=np.float64),
                keypoint_octaves=np.empty((0,), dtype=np.int32),
                descriptors=np.empty((0, 32), dtype=np.uint8),
                status="FAILED",
                failure_reason=FeatureFailureReason.INVALID_IMAGE,
                diagnostics=["Input must be a uint8 numpy array with shape (H, W, 3)."],
            )

        height, width, _ = rgb.shape

        # 2. Convert to grayscale explicitly via ITU-R BT.601 weights
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # 3. Detect and compute ORB keypoints and descriptors
        cv_keypoints, cv_descriptors = self._orb.detectAndCompute(gray, None)

        if cv_keypoints is None or len(cv_keypoints) == 0 or cv_descriptors is None:
            return FeatureExtractionResult(
                frame_id=frame_id,
                width=width,
                height=height,
                keypoint_count=0,
                keypoints_xy=np.empty((0, 2), dtype=np.float64),
                keypoint_scales=np.empty((0,), dtype=np.float64),
                keypoint_angles=np.empty((0,), dtype=np.float64),
                keypoint_responses=np.empty((0,), dtype=np.float64),
                keypoint_octaves=np.empty((0,), dtype=np.int32),
                descriptors=np.empty((0, 32), dtype=np.uint8),
                status="FAILED",
                failure_reason=FeatureFailureReason.NO_FEATURES_DETECTED,
                diagnostics=["ORB detector found 0 keypoints in the frame."],
                provenance={"detector": "OpenCV_ORB", "config_version": self.config.config_version},
            )

        count = len(cv_keypoints)
        pts_xy = np.empty((count, 2), dtype=np.float64)
        scales = np.empty((count,), dtype=np.float64)
        angles = np.empty((count,), dtype=np.float64)
        responses = np.empty((count,), dtype=np.float64)
        octaves = np.empty((count,), dtype=np.int32)

        for i, kp in enumerate(cv_keypoints):
            pts_xy[i, 0] = float(kp.pt[0])
            pts_xy[i, 1] = float(kp.pt[1])
            scales[i] = float(kp.size)
            angles[i] = float(kp.angle)
            responses[i] = float(kp.response)
            octaves[i] = int(kp.octave)

        desc_arr = np.asarray(cv_descriptors, dtype=np.uint8)

        # 4. Check sufficiency threshold
        status = "SUCCESS"
        failure_reason = None
        diagnostics = []
        if count < self.config.min_features_threshold:
            status = "DEGRADED"
            failure_reason = FeatureFailureReason.INSUFFICIENT_FEATURES
            diagnostics.append(
                f"Feature count {count} is below heuristic threshold {self.config.min_features_threshold}."
            )

        return FeatureExtractionResult(
            frame_id=frame_id,
            width=width,
            height=height,
            keypoint_count=count,
            keypoints_xy=pts_xy,
            keypoint_scales=scales,
            keypoint_angles=angles,
            keypoint_responses=responses,
            keypoint_octaves=octaves,
            descriptors=desc_arr,
            descriptor_dtype="uint8",
            descriptor_dim=desc_arr.shape[1] if desc_arr.ndim == 2 else 32,
            detector_type="ORB",
            preprocessing_status="CONVERTED_RGB_TO_GRAYSCALE_BT601",
            measurement_type=MeasurementType.DIRECTLY_OBSERVED,
            status=status,
            failure_reason=failure_reason,
            diagnostics=diagnostics,
            provenance={
                "detector": "OpenCV_ORB",
                "max_features": self.config.max_features,
                "scale_factor": self.config.scale_factor,
                "n_levels": self.config.n_levels,
                "config_version": self.config.config_version,
            },
        )


class ClassicalDescriptorMatcher:
    """Robust classical descriptor matcher using Hamming distance and Lowe ratio / mutual consistency filtering."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(
        self,
        features_a: FeatureExtractionResult,
        features_b: FeatureExtractionResult,
    ) -> FeatureMatchResult:
        """Match descriptors between frame A and frame B using configured strategy."""
        frame_a_id = features_a.frame_id
        frame_b_id = features_b.frame_id

        # 1. Validation checks
        if features_a.keypoint_count == 0 or features_b.keypoint_count == 0 or len(features_a.descriptors) == 0 or len(features_b.descriptors) == 0:
            return FeatureMatchResult(
                frame_a_id=frame_a_id,
                frame_b_id=frame_b_id,
                candidate_match_count=0,
                accepted_match_count=0,
                indices_a=np.empty((0,), dtype=np.int32),
                indices_b=np.empty((0,), dtype=np.int32),
                points_a=np.empty((0, 2), dtype=np.float64),
                points_b=np.empty((0, 2), dtype=np.float64),
                descriptor_distances=np.empty((0,), dtype=np.float64),
                status="FAILED",
                failure_reason=FeatureFailureReason.NO_CANDIDATE_MATCHES,
                diagnostics=["One or both frames contain 0 descriptors."],
                provenance={"matcher": "BFMatcher_Hamming", "config_version": self.config.config_version},
            )

        desc_a = features_a.descriptors
        desc_b = features_b.descriptors

        # 2. k=2 Nearest Neighbor matching A -> B
        if len(desc_b) < 2:
            return FeatureMatchResult(
                frame_a_id=frame_a_id,
                frame_b_id=frame_b_id,
                candidate_match_count=0,
                accepted_match_count=0,
                indices_a=np.empty((0,), dtype=np.int32),
                indices_b=np.empty((0,), dtype=np.int32),
                points_a=np.empty((0, 2), dtype=np.float64),
                points_b=np.empty((0, 2), dtype=np.float64),
                descriptor_distances=np.empty((0,), dtype=np.float64),
                status="FAILED",
                failure_reason=FeatureFailureReason.NO_CANDIDATE_MATCHES,
                diagnostics=["Target descriptor count < 2; cannot perform k=2 nearest neighbor search."],
                provenance={"matcher": "BFMatcher_Hamming", "config_version": self.config.config_version},
            )

        knn_matches_a2b = self._matcher.knnMatch(desc_a, desc_b, k=2)
        total_candidates = len(knn_matches_a2b)

        # 3. Apply Lowe ratio filtering (A -> B)
        ratio_passed_a2b: Dict[int, Tuple[int, float]] = {}  # idx_a -> (idx_b, distance)
        for pair in knn_matches_a2b:
            if len(pair) == 2:
                m, n = pair[0], pair[1]
                if m.distance <= self.config.lowe_ratio * n.distance and m.distance <= self.config.max_descriptor_distance:
                    ratio_passed_a2b[m.queryIdx] = (m.trainIdx, float(m.distance))

        # 4. Optional Mutual Nearest Neighbor check (B -> A)
        final_matches: List[Tuple[int, int, float]] = []  # (idx_a, idx_b, distance)
        if self.config.matching_strategy in (MatchingStrategy.MUTUAL_CONSISTENCY, MatchingStrategy.RATIO_AND_MUTUAL):
            if len(desc_a) >= 2:
                knn_matches_b2a = self._matcher.knnMatch(desc_b, desc_a, k=2)
                ratio_passed_b2a: Dict[int, int] = {}  # idx_b -> idx_a
                for pair in knn_matches_b2a:
                    if len(pair) >= 1:
                        m = pair[0]
                        # If ratio test is also active on reverse match
                        if self.config.matching_strategy == MatchingStrategy.RATIO_AND_MUTUAL:
                            if len(pair) == 2:
                                n = pair[1]
                                if m.distance <= self.config.lowe_ratio * n.distance and m.distance <= self.config.max_descriptor_distance:
                                    ratio_passed_b2a[m.queryIdx] = m.trainIdx
                        else:
                            if m.distance <= self.config.max_descriptor_distance:
                                ratio_passed_b2a[m.queryIdx] = m.trainIdx

                # Cross-check mutual consistency
                for idx_a, (idx_b, dist) in ratio_passed_a2b.items():
                    if ratio_passed_b2a.get(idx_b) == idx_a:
                        final_matches.append((idx_a, idx_b, dist))
            else:
                for idx_a, (idx_b, dist) in ratio_passed_a2b.items():
                    final_matches.append((idx_a, idx_b, dist))
        else:
            for idx_a, (idx_b, dist) in ratio_passed_a2b.items():
                final_matches.append((idx_a, idx_b, dist))

        accepted_count = len(final_matches)

        if accepted_count == 0:
            return FeatureMatchResult(
                frame_a_id=frame_a_id,
                frame_b_id=frame_b_id,
                candidate_match_count=total_candidates,
                accepted_match_count=0,
                indices_a=np.empty((0,), dtype=np.int32),
                indices_b=np.empty((0,), dtype=np.int32),
                points_a=np.empty((0, 2), dtype=np.float64),
                points_b=np.empty((0, 2), dtype=np.float64),
                descriptor_distances=np.empty((0,), dtype=np.float64),
                status="FAILED",
                failure_reason=FeatureFailureReason.NO_CANDIDATE_MATCHES,
                diagnostics=["0 descriptor matches survived Lowe ratio and mutual consistency filters."],
                provenance={"matcher": "BFMatcher_Hamming", "config_version": self.config.config_version},
            )

        # 5. Extract matched coordinates and arrays
        idx_a_arr = np.array([m[0] for m in final_matches], dtype=np.int32)
        idx_b_arr = np.array([m[1] for m in final_matches], dtype=np.int32)
        dist_arr = np.array([m[2] for m in final_matches], dtype=np.float64)

        pts_a = features_a.keypoints_xy[idx_a_arr]
        pts_b = features_b.keypoints_xy[idx_b_arr]

        # 6. Compute distance summary statistics
        min_d = float(np.min(dist_arr))
        med_d = float(np.median(dist_arr))
        mean_d = float(np.mean(dist_arr))
        p90_d = float(np.percentile(dist_arr, 90))
        acc_ratio = float(accepted_count / max(1, total_candidates))

        # 7. Compute spatial match diagnostics
        diag_a = SpatialDistributionCalculator.compute(
            pts_a, features_a.width, features_a.height, self.config.grid_rows, self.config.grid_cols
        )
        diag_b = SpatialDistributionCalculator.compute(
            pts_b, features_b.width, features_b.height, self.config.grid_rows, self.config.grid_cols
        )

        # 8. Check sufficiency threshold
        status = "SUCCESS"
        failure_reason = None
        diagnostics = []
        if accepted_count < self.config.min_accepted_matches:
            status = "DEGRADED"
            failure_reason = FeatureFailureReason.INSUFFICIENT_DESCRIPTOR_MATCHES
            diagnostics.append(
                f"Accepted descriptor matches {accepted_count} < heuristic threshold {self.config.min_accepted_matches}."
            )

        return FeatureMatchResult(
            frame_a_id=frame_a_id,
            frame_b_id=frame_b_id,
            candidate_match_count=total_candidates,
            accepted_match_count=accepted_count,
            indices_a=idx_a_arr,
            indices_b=idx_b_arr,
            points_a=pts_a,
            points_b=pts_b,
            descriptor_distances=dist_arr,
            min_distance=min_d,
            median_distance=med_d,
            mean_distance=mean_d,
            percentile_90_distance=p90_d,
            acceptance_ratio=acc_ratio,
            spatial_diagnostics_a=diag_a,
            spatial_diagnostics_b=diag_b,
            matching_strategy=self.config.matching_strategy.value,
            measurement_type=MeasurementType.ESTIMATED,
            status=status,
            failure_reason=failure_reason,
            diagnostics=diagnostics,
            provenance={
                "matcher": "BFMatcher_Hamming",
                "matching_strategy": self.config.matching_strategy.value,
                "lowe_ratio": self.config.lowe_ratio,
                "max_descriptor_distance": self.config.max_descriptor_distance,
                "config_version": self.config.config_version,
            },
        )
