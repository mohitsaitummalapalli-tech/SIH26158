"""Temporal Motion and Blur Diagnostic Subsystem.

Computes explainable, deterministic temporal motion diagnostics and motion-blur proxy indicators
using multi-frame sequence analysis on DecodedFrame sequences.

SCIENTIFIC INTERPRETATION BOUNDARIES:
- INTER-FRAME APPARENT MOTION (px/s) is the optical displacement between discrete presentation frames.
  It is NOT camera pose estimation or visual odometry.
- SINGLE-FRAME OPTICAL SHARPNESS is spatial edge energy (Laplacian variance).
- INTRA-EXPOSURE MOTION BLUR is physical optical integration during the shutter opening.
  The motion_blur_indicator is a diagnostic heuristic combining inter-frame velocity with
  sharpness; it is NOT a direct measurement of physical intra-exposure blur or a calibrated probability.
- SPATIAL MOTION HETEROGENEITY is a diagnostic pattern indicator, NOT moving-object segmentation.
- ALL CONFIGURABLE THRESHOLDS are heuristic defaults requiring empirical calibration.
- These diagnostics MUST NOT be used to drop or filter frames at this stage.
"""

import json
import math
import numpy as np
import cv2
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple

from src.preprocessing.decoder import DecodedFrame, DecodeStatus
from src.quality.assessment import FrameQualityAnalyzer, QualityAssessmentConfig, FrameQualityReport


class MotionCategory(str, Enum):
    """Categorical diagnostic hypothesis based on temporal displacement field patterns."""
    POTENTIAL_CAMERA_MOTION = "POTENTIAL_CAMERA_MOTION"
    POTENTIAL_LOCAL_MOTION = "POTENTIAL_LOCAL_MOTION"
    MIXED_MOTION = "MIXED_MOTION"
    LOW_APPARENT_MOTION = "LOW_APPARENT_MOTION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class TemporalMotionConfig:
    """Configurable heuristic thresholds for temporal motion and blur diagnostics.
    
    All default values are heuristic defaults (HEURISTIC_DEFAULT) requiring empirical
    calibration on representative flight validation datasets.
    """
    tile_grid_rows: int = 3                      # HEURISTIC_DEFAULT: vertical grid tiles
    tile_grid_cols: int = 3                      # HEURISTIC_DEFAULT: horizontal grid tiles
    max_temporal_gap_seconds: float = 2.0        # HEURISTIC_DEFAULT: max allowed inter-frame delta
    coherence_threshold: float = 0.70            # HEURISTIC_DEFAULT: directional coherence bound
    local_motion_ratio_threshold: float = 2.0    # HEURISTIC_DEFAULT: tile/median outlier ratio
    displacement_lower_bound: float = 0.5        # HEURISTIC_DEFAULT: minimum flow magnitude (px)
    low_texture_variance_threshold: float = 10.0 # HEURISTIC_DEFAULT: luma variance for low texture
    config_version: str = "TemporalMotion_v1.1_ScientificCorrection"


@dataclass(frozen=True)
class SpatialMotionTile:
    """Motion displacement statistics evaluated on a spatial sub-grid tile."""
    tile_row: int
    tile_col: int
    bbox: Tuple[int, int, int, int]              # (y_min, x_min, y_max, x_max)
    mean_displacement_pixels: float
    median_displacement_pixels: float
    mean_velocity_px_per_sec: float
    displacement_std: float
    valid_fraction: float


@dataclass
class TemporalMotionBlurReport:
    """Structured temporal motion and blur diagnostic report for a single video frame."""
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    source_video: str
    neighbor_frame_ids: List[str]
    neighbor_timestamps: List[float]
    time_deltas_seconds: List[float]
    motion_category: MotionCategory
    median_displacement_pixels: float
    mean_displacement_pixels: float
    displacement_p10_pixels: float
    displacement_p90_pixels: float
    global_velocity_px_per_sec: float
    temporal_residual_mean: float
    directional_coherence_score: float
    temporal_acceleration_px_per_sec2: Optional[float]
    valid_flow_fraction: float
    spatial_motion_heterogeneity: float
    low_texture_indicator: bool
    target_sharpness_laplacian: float
    neighbor_sharpness_laplacian_mean: Optional[float]
    relative_sharpness_ratio: Optional[float]
    motion_blur_indicator: float                 # Diagnostic proxy index in [0.0, 1.0]
    motion_blur_diagnostic: str
    spatial_tiles: List[SpatialMotionTile] = field(default_factory=list)
    future_exposure_metadata: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    config_version: str = "TemporalMotion_v1.1_ScientificCorrection"

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to serializable dictionary."""
        d = asdict(self)
        d["motion_category"] = self.motion_category.value
        return d

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class TemporalMotionAnalyzer:
    """Deterministic engine for multi-frame temporal displacement and motion-blur analysis."""

    @staticmethod
    def _to_gray_f32(data: np.ndarray) -> np.ndarray:
        """Convert canonical RGB uint8 to normalized float32 grayscale."""
        if data.ndim == 3 and data.shape[2] == 3:
            return cv2.cvtColor(data, cv2.COLOR_RGB2GRAY).astype(np.float32)
        elif data.ndim == 3 and data.shape[2] == 1:
            return data[:, :, 0].astype(np.float32)
        return data.astype(np.float32)

    @classmethod
    def compute_dense_optical_flow(
        cls, gray_prev: np.ndarray, gray_curr: np.ndarray
    ) -> np.ndarray:
        """Compute 2D dense optical flow displacement field (u, v) from prev to curr.
        
        Uses classical deterministic Farnebäck algorithm.
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev=gray_prev.astype(np.uint8),
            next=gray_curr.astype(np.uint8),
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        return flow  # Shape (H, W, 2) where [..., 0] is dx (u), [..., 1] is dy (v)

    @classmethod
    def analyze_temporal_motion(
        cls,
        target_frame: DecodedFrame,
        prev_frame: Optional[DecodedFrame] = None,
        next_frame: Optional[DecodedFrame] = None,
        quality_report: Optional[FrameQualityReport] = None,
        config: Optional[TemporalMotionConfig] = None,
    ) -> TemporalMotionBlurReport:
        """Analyze temporal motion and motion-blur heuristic indicator for a target frame in context.
        
        DOES NOT MODIFY any frame buffers.
        """
        cfg = config or TemporalMotionConfig()
        diagnostics: List[str] = []
        neighbor_ids: List[str] = []
        neighbor_timestamps: List[float] = []
        time_deltas: List[float] = []

        # Validate target frame
        if target_frame.decode_status != DecodeStatus.SUCCESS or target_frame.data is None:
            return TemporalMotionBlurReport(
                frame_id=target_frame.frame_id,
                frame_index=target_frame.frame_index,
                timestamp_seconds=target_frame.timestamp_seconds,
                source_video=target_frame.source_video,
                neighbor_frame_ids=[],
                neighbor_timestamps=[],
                time_deltas_seconds=[],
                motion_category=MotionCategory.INSUFFICIENT_EVIDENCE,
                median_displacement_pixels=0.0,
                mean_displacement_pixels=0.0,
                displacement_p10_pixels=0.0,
                displacement_p90_pixels=0.0,
                global_velocity_px_per_sec=0.0,
                temporal_residual_mean=0.0,
                directional_coherence_score=0.0,
                temporal_acceleration_px_per_sec2=None,
                valid_flow_fraction=0.0,
                spatial_motion_heterogeneity=0.0,
                low_texture_indicator=False,
                target_sharpness_laplacian=0.0,
                neighbor_sharpness_laplacian_mean=None,
                relative_sharpness_ratio=None,
                motion_blur_indicator=0.0,
                motion_blur_diagnostic="Target frame data is missing or corrupted.",
                spatial_tiles=[],
                diagnostics=["Target frame decode status is invalid."],
                provenance={"target_frame_id": target_frame.frame_id},
                config_version=cfg.config_version,
            )

        target_gray = cls._to_gray_f32(target_frame.data)
        h, w = target_gray.shape

        # Evaluate target static sharpness
        if quality_report and quality_report.sharpness:
            target_sharpness = quality_report.sharpness.laplacian_variance
        else:
            lap = cv2.Laplacian(target_gray, cv2.CV_32F)
            target_sharpness = float(np.var(lap))

        # Check low texture on target frame
        target_luma_var = float(np.var(target_gray))
        low_texture = target_luma_var < cfg.low_texture_variance_threshold
        if low_texture:
            diagnostics.append(f"Low surface texture detected (luma variance {target_luma_var:.2f} < {cfg.low_texture_variance_threshold}).")

        # Check neighbors
        has_prev = (
            prev_frame is not None
            and prev_frame.decode_status == DecodeStatus.SUCCESS
            and prev_frame.data is not None
        )
        has_next = (
            next_frame is not None
            and next_frame.decode_status == DecodeStatus.SUCCESS
            and next_frame.data is not None
        )

        flow_backward: Optional[np.ndarray] = None
        dt_prev: Optional[float] = None
        neighbor_sharpness_values: List[float] = []

        if has_prev:
            dt = target_frame.timestamp_seconds - prev_frame.timestamp_seconds
            if 0.0 < dt <= cfg.max_temporal_gap_seconds:
                prev_gray = cls._to_gray_f32(prev_frame.data)
                if prev_gray.shape == target_gray.shape:
                    flow_backward = cls.compute_dense_optical_flow(prev_gray, target_gray)
                    dt_prev = dt
                    neighbor_ids.append(prev_frame.frame_id)
                    neighbor_timestamps.append(prev_frame.timestamp_seconds)
                    time_deltas.append(round(dt, 4))
                    neighbor_sharpness_values.append(float(np.var(cv2.Laplacian(prev_gray, cv2.CV_32F))))
                else:
                    diagnostics.append("Previous frame shape mismatch.")
            else:
                diagnostics.append(f"Previous frame temporal gap ({dt:.3f}s) exceeds max threshold ({cfg.max_temporal_gap_seconds}s).")

        flow_forward: Optional[np.ndarray] = None
        dt_next: Optional[float] = None
        if has_next:
            dt = next_frame.timestamp_seconds - target_frame.timestamp_seconds
            if 0.0 < dt <= cfg.max_temporal_gap_seconds:
                next_gray = cls._to_gray_f32(next_frame.data)
                if next_gray.shape == target_gray.shape:
                    flow_forward = cls.compute_dense_optical_flow(target_gray, next_gray)
                    dt_next = dt
                    neighbor_ids.append(next_frame.frame_id)
                    neighbor_timestamps.append(next_frame.timestamp_seconds)
                    time_deltas.append(round(dt, 4))
                    neighbor_sharpness_values.append(float(np.var(cv2.Laplacian(next_gray, cv2.CV_32F))))
                else:
                    diagnostics.append("Next frame shape mismatch.")
            else:
                diagnostics.append(f"Next frame temporal gap ({dt:.3f}s) exceeds max threshold ({cfg.max_temporal_gap_seconds}s).")

        # Insufficient neighbors
        if flow_backward is None and flow_forward is None:
            return TemporalMotionBlurReport(
                frame_id=target_frame.frame_id,
                frame_index=target_frame.frame_index,
                timestamp_seconds=target_frame.timestamp_seconds,
                source_video=target_frame.source_video,
                neighbor_frame_ids=neighbor_ids,
                neighbor_timestamps=neighbor_timestamps,
                time_deltas_seconds=time_deltas,
                motion_category=MotionCategory.INSUFFICIENT_EVIDENCE,
                median_displacement_pixels=0.0,
                mean_displacement_pixels=0.0,
                displacement_p10_pixels=0.0,
                displacement_p90_pixels=0.0,
                global_velocity_px_per_sec=0.0,
                temporal_residual_mean=0.0,
                directional_coherence_score=0.0,
                temporal_acceleration_px_per_sec2=None,
                valid_flow_fraction=0.0,
                spatial_motion_heterogeneity=0.0,
                low_texture_indicator=low_texture,
                target_sharpness_laplacian=round(target_sharpness, 4),
                neighbor_sharpness_laplacian_mean=None,
                relative_sharpness_ratio=None,
                motion_blur_indicator=0.0,
                motion_blur_diagnostic="Insufficient adjacent temporal context available.",
                spatial_tiles=[],
                diagnostics=diagnostics or ["Boundary frame or missing temporal neighbors."],
                provenance={"target_frame_id": target_frame.frame_id},
                config_version=cfg.config_version,
            )

        # Primary flow displacement field
        if flow_backward is not None and flow_forward is not None and dt_prev and dt_next:
            v_back = flow_backward / dt_prev
            v_fwd = flow_forward / dt_next
            primary_flow = flow_backward
            effective_dt = dt_prev
            v_diff = np.sqrt((v_fwd[..., 0] - v_back[..., 0])**2 + (v_fwd[..., 1] - v_back[..., 1])**2)
            acceleration = float(np.mean(v_diff)) / ((dt_prev + dt_next) / 2.0)
        elif flow_backward is not None and dt_prev:
            primary_flow = flow_backward
            effective_dt = dt_prev
            acceleration = None
        else:
            primary_flow = flow_forward  # type: ignore
            effective_dt = dt_next  # type: ignore
            acceleration = None

        # Global displacement magnitudes
        u = primary_flow[..., 0]
        v = primary_flow[..., 1]
        mags = np.sqrt(u**2 + v**2)
        valid_flow_mask = np.isfinite(mags)
        valid_flow_fraction = float(np.sum(valid_flow_mask)) / mags.size

        valid_mags = mags[valid_flow_mask]
        if valid_mags.size > 0:
            mean_disp = float(np.mean(valid_mags))
            median_disp = float(np.median(valid_mags))
            p10, p90 = np.percentile(valid_mags, [10, 90])
        else:
            mean_disp = 0.0
            median_disp = 0.0
            p10, p90 = 0.0, 0.0

        velocity = mean_disp / effective_dt if effective_dt > 0 else 0.0

        # Temporal residual
        if has_prev and dt_prev:
            residual = float(np.mean(np.abs(target_gray - cls._to_gray_f32(prev_frame.data))))
        elif has_next and dt_next:
            residual = float(np.mean(np.abs(cls._to_gray_f32(next_frame.data) - target_gray)))
        else:
            residual = 0.0

        # Directional Coherence Score (Mean unit vector magnitude)
        unit_u = np.divide(u, mags, out=np.zeros_like(u), where=mags > 0.1)
        unit_v = np.divide(v, mags, out=np.zeros_like(v), where=mags > 0.1)
        mean_unit_u = float(np.mean(unit_u))
        mean_unit_v = float(np.mean(unit_v))
        directional_coherence = float(np.sqrt(mean_unit_u**2 + mean_unit_v**2))

        # Spatial Grid Tiles
        rows = max(1, cfg.tile_grid_rows)
        cols = max(1, cfg.tile_grid_cols)
        row_edges = np.linspace(0, h, rows + 1, dtype=int)
        col_edges = np.linspace(0, w, cols + 1, dtype=int)

        spatial_tiles: List[SpatialMotionTile] = []
        tile_displacements: List[float] = []

        for r in range(rows):
            for c in range(cols):
                y_min, y_max = row_edges[r], row_edges[r + 1]
                x_min, x_max = col_edges[c], col_edges[c + 1]

                tile_mags = mags[y_min:y_max, x_min:x_max]
                if tile_mags.size == 0:
                    continue

                t_mean = float(np.mean(tile_mags))
                t_med = float(np.median(tile_mags))
                t_std = float(np.std(tile_mags))
                t_vel = t_mean / effective_dt if effective_dt > 0 else 0.0

                tile_displacements.append(t_med)
                spatial_tiles.append(
                    SpatialMotionTile(
                        tile_row=r,
                        tile_col=c,
                        bbox=(int(y_min), int(x_min), int(y_max), int(x_max)),
                        mean_displacement_pixels=round(t_mean, 4),
                        median_displacement_pixels=round(t_med, 4),
                        mean_velocity_px_per_sec=round(t_vel, 4),
                        displacement_std=round(t_std, 4),
                        valid_fraction=1.0,
                    )
                )

        # Spatial Motion Heterogeneity: Standard deviation of tile median displacements
        spatial_heterogeneity = float(np.std(tile_displacements)) if tile_displacements else 0.0

        # Relative Sharpness Analysis
        if neighbor_sharpness_values:
            neighbor_sharpness_mean = float(np.mean(neighbor_sharpness_values))
            rel_sharpness_ratio = target_sharpness / max(neighbor_sharpness_mean, 1e-4)
            # Relative sharpness drop: how much target sharpness decreased relative to neighbors
            sharpness_drop = max(0.0, min(1.0, 1.0 - rel_sharpness_ratio))
        else:
            neighbor_sharpness_mean = None
            rel_sharpness_ratio = None
            # Fallback to normalized static sharpness scaling
            sharpness_drop = max(0.0, min(1.0, (100.0 - target_sharpness) / 100.0))

        # Categorize Motion Hypotheses
        max_tile_disp = max(tile_displacements) if tile_displacements else 0.0
        min_tile_disp = min(tile_displacements) if tile_displacements else 0.0

        has_local_outlier = (
            max_tile_disp > 1.5
            and max_tile_disp > cfg.local_motion_ratio_threshold * max(median_disp, cfg.displacement_lower_bound)
        )

        if has_local_outlier and directional_coherence < cfg.coherence_threshold:
            motion_cat = MotionCategory.POTENTIAL_LOCAL_MOTION
            diagnostics.append("Localized displacement heterogeneity detected in sub-grid tiles.")
        elif median_disp < cfg.displacement_lower_bound and not has_local_outlier:
            if low_texture:
                motion_cat = MotionCategory.INSUFFICIENT_EVIDENCE
                diagnostics.append("Low flow displacement in low-texture scene; evidence insufficient.")
            else:
                motion_cat = MotionCategory.LOW_APPARENT_MOTION
                diagnostics.append(f"Low apparent inter-frame displacement (< {cfg.displacement_lower_bound} px).")
        elif directional_coherence >= cfg.coherence_threshold and not has_local_outlier:
            motion_cat = MotionCategory.POTENTIAL_CAMERA_MOTION
        else:
            motion_cat = MotionCategory.MIXED_MOTION

        # Motion Blur Diagnostic Indicator [0.0, 1.0]
        # Combining inter-frame velocity with relative/static sharpness degradation
        norm_vel = min(1.0, velocity / 100.0)
        motion_blur_indicator = round(float(norm_vel * sharpness_drop), 4)

        if motion_blur_indicator > 0.6:
            blur_diag = "High motion-blur diagnostic indicator (high temporal velocity combined with relative sharpness drop)."
        elif motion_blur_indicator > 0.3:
            blur_diag = "Moderate motion-blur diagnostic indicator (detectable temporal velocity with intermediate sharpness)."
        elif velocity < 5.0 and target_sharpness < 30.0:
            blur_diag = "Low-motion degradation (sharpness low despite low velocity; potential defocus or low texture)."
        else:
            blur_diag = "Low motion-blur diagnostic indicator (sharp edges maintained or low temporal velocity)."

        provenance = {
            "target_frame_id": target_frame.frame_id,
            "target_frame_index": target_frame.frame_index,
            "target_timestamp_seconds": target_frame.timestamp_seconds,
            "neighbor_frame_ids": neighbor_ids,
            "effective_dt_seconds": effective_dt,
            "analysis_method": "DenseFarnebackOpticalFlow_v1.0",
            "threshold_coherence": cfg.coherence_threshold,
            "threshold_local_ratio": cfg.local_motion_ratio_threshold,
            "threshold_displacement_bound": cfg.displacement_lower_bound,
        }

        return TemporalMotionBlurReport(
            frame_id=target_frame.frame_id,
            frame_index=target_frame.frame_index,
            timestamp_seconds=target_frame.timestamp_seconds,
            source_video=target_frame.source_video,
            neighbor_frame_ids=neighbor_ids,
            neighbor_timestamps=neighbor_timestamps,
            time_deltas_seconds=time_deltas,
            motion_category=motion_cat,
            median_displacement_pixels=round(median_disp, 4),
            mean_displacement_pixels=round(mean_disp, 4),
            displacement_p10_pixels=round(float(p10), 4),
            displacement_p90_pixels=round(float(p90), 4),
            global_velocity_px_per_sec=round(velocity, 4),
            temporal_residual_mean=round(residual, 4),
            directional_coherence_score=round(directional_coherence, 4),
            temporal_acceleration_px_per_sec2=round(acceleration, 4) if acceleration is not None else None,
            valid_flow_fraction=round(valid_flow_fraction, 4),
            spatial_motion_heterogeneity=round(spatial_heterogeneity, 4),
            low_texture_indicator=low_texture,
            target_sharpness_laplacian=round(target_sharpness, 4),
            neighbor_sharpness_laplacian_mean=round(neighbor_sharpness_mean, 4) if neighbor_sharpness_mean is not None else None,
            relative_sharpness_ratio=round(rel_sharpness_ratio, 4) if rel_sharpness_ratio is not None else None,
            motion_blur_indicator=motion_blur_indicator,
            motion_blur_diagnostic=blur_diag,
            spatial_tiles=spatial_tiles,
            future_exposure_metadata={},
            diagnostics=diagnostics,
            provenance=provenance,
            config_version=cfg.config_version,
        )
