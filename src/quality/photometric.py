"""Photometric & Illumination Stability Diagnostics Subsystem.

Computes explainable, deterministic photometric diagnostics, spatial illumination uniformity,
dynamic range characterization, and frame-to-frame photometric stability metrics.

SCIENTIFIC INTERPRETATION BOUNDARIES:
- Image luminance statistics do NOT measure physical scene radiance or lux.
- Histogram distance is a distribution change indicator, NOT a definitive measurement of camera exposure change.
- Spatial illumination variance does NOT prove physical shadows or lens vignetting.
- Color statistics are appearance descriptors, NOT radiometric or white-balance calibrations.
- Photometric stability indicators do NOT directly guarantee 3D reconstruction accuracy.
- This subsystem produces DIAGNOSTICS ONLY. It must NEVER modify source frame buffers or filter frames.
"""

import json
import math
import numpy as np
import cv2
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple

from src.preprocessing.decoder import DecodedFrame, DecodeStatus
from src.quality.assessment import (
    FrameQualityAnalyzer,
    QualityAssessmentConfig,
    FrameQualityReport,
    LuminanceStatistics,
)


class SpatialIlluminationPattern(str, Enum):
    """Categorical diagnostic hypothesis for spatial illumination distribution across the frame."""
    UNIFORM = "UNIFORM"
    GRADIENT = "GRADIENT"
    LOCALIZED_BRIGHTNESS = "LOCALIZED_BRIGHTNESS"
    LOCALIZED_DARKNESS = "LOCALIZED_DARKNESS"
    MIXED = "MIXED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class PhotometricChangeCategory(str, Enum):
    """Categorical diagnostic hypothesis for frame-to-frame photometric transitions."""
    STABLE = "STABLE"
    POTENTIAL_EXPOSURE_TRANSITION = "POTENTIAL_EXPOSURE_TRANSITION"
    POTENTIAL_LOCAL_ILLUMINATION_CHANGE = "POTENTIAL_LOCAL_ILLUMINATION_CHANGE"
    MIXED_CHANGE = "MIXED_CHANGE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class PhotometricConfig:
    """Configurable parameters and heuristic thresholds for photometric analysis.
    
    All default thresholds are heuristic defaults (HEURISTIC_DEFAULT) requiring empirical
    calibration on representative flight validation datasets.
    """
    histogram_bins: int = 64                     # HEURISTIC_DEFAULT: Number of luminance histogram bins
    tile_grid_rows: int = 3                      # HEURISTIC_DEFAULT: Spatial grid rows
    tile_grid_cols: int = 3                      # HEURISTIC_DEFAULT: Spatial grid columns
    shadow_threshold: float = 5.0                # HEURISTIC_DEFAULT: Luminance <= shadow_threshold is clipped
    highlight_threshold: float = 250.0           # HEURISTIC_DEFAULT: Luminance >= highlight_threshold is clipped
    max_temporal_gap_seconds: float = 2.0        # HEURISTIC_DEFAULT: Max allowed inter-frame delta
    spatial_uniformity_std_thresh: float = 8.0   # HEURISTIC_DEFAULT: Tile std for UNIFORM pattern
    gradient_slope_threshold: float = 12.0       # HEURISTIC_DEFAULT: Row/col delta for GRADIENT pattern
    local_tile_ratio_threshold: float = 1.5      # HEURISTIC_DEFAULT: Local outlier brightness/darkness ratio
    bhattacharyya_change_threshold: float = 0.15 # HEURISTIC_DEFAULT: Histogram distance for transition
    config_version: str = "Photometric_v1.0"


@dataclass(frozen=True)
class SpatialIlluminationTile:
    """Photometric statistics evaluated on an individual spatial sub-grid tile."""
    tile_row: int
    tile_col: int
    bbox: Tuple[int, int, int, int]              # (y_min, x_min, y_max, x_max)
    mean_luminance: float
    median_luminance: float
    percentile_spread_90_10: float
    shadow_fraction: float
    highlight_fraction: float


@dataclass(frozen=True)
class ColorStatistics:
    """Channel-wise color distribution statistics in canonical RGB color space."""
    r_mean: float
    g_mean: float
    b_mean: float
    r_median: float
    g_median: float
    b_median: float
    chromatic_spread: float                      # Standard deviation of channel mean differences (R-G, G-B, B-R)


@dataclass(frozen=True)
class ExtendedLuminanceStatistics:
    """Extended luminance distribution statistics covering full percentile spectrum."""
    mean: float
    median: float
    std: float
    p1: float
    p5: float
    p10: float
    p50: float
    p90: float
    p95: float
    p99: float
    observed_min: float
    observed_max: float


@dataclass(frozen=True)
class DynamicRangeStatistics:
    """Dynamic range and clipping diagnostics of the decoded image."""
    observed_range: float                        # observed_max - observed_min
    percentile_dynamic_range: float              # p99 - p1
    effective_dynamic_range_fraction: float      # (p95 - p5) / 255.0
    shadow_fraction: float                       # Fraction of pixels <= shadow_threshold
    highlight_fraction: float                    # Fraction of pixels >= highlight_threshold


@dataclass(frozen=True)
class TemporalPhotometricChange:
    """Frame-to-frame photometric comparison metrics with a neighboring frame."""
    neighbor_frame_id: str
    neighbor_timestamp_seconds: float
    delta_t_seconds: float
    mean_luminance_change: float                 # target_mean - neighbor_mean
    median_luminance_change: float               # target_median - neighbor_median
    percentile_p90_change: float                 # target_p90 - neighbor_p90
    percentile_p10_change: float                 # target_p10 - neighbor_p10
    bhattacharyya_distance: float                # Classical Bhattacharyya distance between luma histograms
    bhattacharyya_coefficient: float             # Sum(sqrt(P_k * Q_k)) in [0.0, 1.0]
    change_category: PhotometricChangeCategory


@dataclass
class PhotometricStabilityReport:
    """Comprehensive, structured photometric diagnostic report for a single video frame."""
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    source_video: str
    luminance: ExtendedLuminanceStatistics
    dynamic_range: DynamicRangeStatistics
    spatial_pattern: SpatialIlluminationPattern
    spatial_nonuniformity_std: float             # Standard deviation of tile mean luminances
    spatial_tiles: List[SpatialIlluminationTile]
    color: Optional[ColorStatistics]
    temporal_change: Optional[TemporalPhotometricChange]
    diagnostics: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    config_version: str = "Photometric_v1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to serializable dictionary."""
        d = asdict(self)
        d["spatial_pattern"] = self.spatial_pattern.value
        if self.temporal_change:
            d["temporal_change"]["change_category"] = self.temporal_change.change_category.value
        return d

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class PhotometricAnalyzer:
    """Deterministic, explainable photometric and illumination analysis engine."""

    @classmethod
    def compute_normalized_histogram(
        cls, luma: np.ndarray, bins: int = 64
    ) -> np.ndarray:
        """Compute normalized 1D luminance histogram with unit sum."""
        hist, _ = np.histogram(luma, bins=bins, range=(0.0, 255.0))
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            return (hist / hist_sum).astype(np.float64)
        return np.zeros(bins, dtype=np.float64)

    @classmethod
    def compute_bhattacharyya_distance(
        cls, hist_p: np.ndarray, hist_q: np.ndarray
    ) -> Tuple[float, float]:
        """Compute Bhattacharyya Coefficient (BC) and Bhattacharyya Distance (DB) between two normalized histograms.
        
        BC(P, Q) = sum_{k} sqrt(P_k * Q_k)
        DB(P, Q) = -ln(BC(P, Q)) if BC > 0 else inf
        """
        bc = float(np.sum(np.sqrt(np.clip(hist_p, 0.0, 1.0) * np.clip(hist_q, 0.0, 1.0))))
        bc = max(0.0, min(1.0, bc))

        if bc > 1e-12:
            db = -float(np.log(bc))
        else:
            db = 10.0  # Safe finite upper bound for disjoint distributions

        return round(db, 4), round(bc, 4)

    @classmethod
    def analyze_frame(
        cls,
        target_frame: DecodedFrame,
        neighbor_frame: Optional[DecodedFrame] = None,
        quality_report: Optional[FrameQualityReport] = None,
        config: Optional[PhotometricConfig] = None,
    ) -> PhotometricStabilityReport:
        """Analyze photometric stability and spatial illumination distribution for a target frame.
        
        DOES NOT MODIFY frame.data.
        """
        cfg = config or PhotometricConfig()
        diagnostics: List[str] = []

        # Validate input frame
        if target_frame.decode_status != DecodeStatus.SUCCESS or target_frame.data is None:
            raise ValueError(f"Cannot perform photometric analysis on frame '{target_frame.frame_id}' with invalid decode status {target_frame.decode_status.value}.")

        if not np.all(np.isfinite(target_frame.data)):
            diagnostics.append("Image data contains non-finite values.")

        # 1. Extract Luminance Matrix (ITU-R BT.601)
        luma = FrameQualityAnalyzer.rgb_to_luminance(target_frame.data)
        h, w = luma.shape
        total_pixels = luma.size

        # 2. Extended Luminance Statistics
        mean_val = float(np.mean(luma))
        median_val = float(np.median(luma))
        std_val = float(np.std(luma))
        p1, p5, p10, p50, p90, p95, p99 = np.percentile(luma, [1, 5, 10, 50, 90, 95, 99])
        obs_min = float(np.min(luma))
        obs_max = float(np.max(luma))

        ext_luma = ExtendedLuminanceStatistics(
            mean=round(mean_val, 4),
            median=round(median_val, 4),
            std=round(std_val, 4),
            p1=round(float(p1), 4),
            p5=round(float(p5), 4),
            p10=round(float(p10), 4),
            p50=round(float(p50), 4),
            p90=round(float(p90), 4),
            p95=round(float(p95), 4),
            p99=round(float(p99), 4),
            observed_min=round(obs_min, 4),
            observed_max=round(obs_max, 4),
        )

        # 3. Dynamic Range & Clipping
        shadow_mask = luma <= cfg.shadow_threshold
        highlight_mask = luma >= cfg.highlight_threshold
        shadow_frac = float(np.sum(shadow_mask)) / total_pixels
        highlight_frac = float(np.sum(highlight_mask)) / total_pixels

        dyn_range = DynamicRangeStatistics(
            observed_range=round(obs_max - obs_min, 4),
            percentile_dynamic_range=round(float(p99 - p1), 4),
            effective_dynamic_range_fraction=round(float(p95 - p5) / 255.0, 4),
            shadow_fraction=round(shadow_frac, 4),
            highlight_fraction=round(highlight_frac, 4),
        )

        # 4. Spatial Grid Illumination Analysis
        rows = max(1, cfg.tile_grid_rows)
        cols = max(1, cfg.tile_grid_cols)
        row_edges = np.linspace(0, h, rows + 1, dtype=int)
        col_edges = np.linspace(0, w, cols + 1, dtype=int)

        spatial_tiles: List[SpatialIlluminationTile] = []
        tile_means: List[float] = []
        grid_matrix = np.zeros((rows, cols), dtype=np.float64)

        for r in range(rows):
            for c in range(cols):
                y_min, y_max = row_edges[r], row_edges[r + 1]
                x_min, x_max = col_edges[c], col_edges[c + 1]

                tile_data = luma[y_min:y_max, x_min:x_max]
                if tile_data.size == 0:
                    continue

                t_mean = float(np.mean(tile_data))
                t_med = float(np.median(tile_data))
                t_p10, t_p90 = np.percentile(tile_data, [10, 90])
                t_shadow = float(np.sum(tile_data <= cfg.shadow_threshold)) / tile_data.size
                t_highlight = float(np.sum(tile_data >= cfg.highlight_threshold)) / tile_data.size

                tile_means.append(t_mean)
                grid_matrix[r, c] = t_mean

                spatial_tiles.append(
                    SpatialIlluminationTile(
                        tile_row=r,
                        tile_col=c,
                        bbox=(int(y_min), int(x_min), int(y_max), int(x_max)),
                        mean_luminance=round(t_mean, 4),
                        median_luminance=round(t_med, 4),
                        percentile_spread_90_10=round(float(t_p90 - t_p10), 4),
                        shadow_fraction=round(t_shadow, 4),
                        highlight_fraction=round(t_highlight, 4),
                    )
                )

        spatial_nonuniformity_std = float(np.std(tile_means)) if tile_means else 0.0

        # Categorize Spatial Pattern
        if spatial_nonuniformity_std < cfg.spatial_uniformity_std_thresh:
            spatial_pat = SpatialIlluminationPattern.UNIFORM
        else:
            # Check for monotonic gradient across rows or columns
            col_means = np.mean(grid_matrix, axis=0)
            row_means = np.mean(grid_matrix, axis=1)
            col_delta = abs(col_means[-1] - col_means[0])
            row_delta = abs(row_means[-1] - row_means[0])

            is_col_monotonic = bool(
                (len(col_means) > 1) and (np.all(np.diff(col_means) > 1.0) or np.all(np.diff(col_means) < -1.0))
            )
            is_row_monotonic = bool(
                (len(row_means) > 1) and (np.all(np.diff(row_means) > 1.0) or np.all(np.diff(row_means) < -1.0))
            )

            med_tile = float(np.median(tile_means))
            max_tile = float(np.max(tile_means))
            min_tile = float(np.min(tile_means))

            has_bright_spot = max_tile > cfg.local_tile_ratio_threshold * max(med_tile, 10.0)
            has_dark_spot = min_tile < (1.0 / cfg.local_tile_ratio_threshold) * max(med_tile, 10.0)

            if (is_col_monotonic and col_delta > cfg.gradient_slope_threshold) or (is_row_monotonic and row_delta > cfg.gradient_slope_threshold):
                spatial_pat = SpatialIlluminationPattern.GRADIENT
                diagnostics.append(f"Spatial illumination gradient detected (delta across axes: X={col_delta:.1f}, Y={row_delta:.1f}).")
            elif has_bright_spot and not has_dark_spot:
                spatial_pat = SpatialIlluminationPattern.LOCALIZED_BRIGHTNESS
                diagnostics.append(f"Localized brightness anomaly detected (max tile {max_tile:.1f} vs median {med_tile:.1f}).")
            elif has_dark_spot and not has_bright_spot:
                spatial_pat = SpatialIlluminationPattern.LOCALIZED_DARKNESS
                diagnostics.append(f"Localized shadow anomaly detected (min tile {min_tile:.1f} vs median {med_tile:.1f}).")
            else:
                spatial_pat = SpatialIlluminationPattern.MIXED

        # 5. Optional Color Statistics (RGB channels)
        color_stats: Optional[ColorStatistics] = None
        if target_frame.data.ndim == 3 and target_frame.data.shape[2] == 3:
            r_ch = target_frame.data[:, :, 0]
            g_ch = target_frame.data[:, :, 1]
            b_ch = target_frame.data[:, :, 2]

            r_m, g_m, b_m = float(np.mean(r_ch)), float(np.mean(g_ch)), float(np.mean(b_ch))
            r_med, g_med, b_med = float(np.median(r_ch)), float(np.median(g_ch)), float(np.median(b_ch))
            chroma_spread = float(np.std([r_m - g_m, g_m - b_m, b_m - r_m]))

            color_stats = ColorStatistics(
                r_mean=round(r_m, 4),
                g_mean=round(g_m, 4),
                b_mean=round(b_m, 4),
                r_median=round(r_med, 4),
                g_median=round(g_med, 4),
                b_median=round(b_med, 4),
                chromatic_spread=round(chroma_spread, 4),
            )

        # 6. Frame-to-Frame Temporal Photometric Comparison
        temp_change: Optional[TemporalPhotometricChange] = None
        if (
            neighbor_frame is not None
            and neighbor_frame.decode_status == DecodeStatus.SUCCESS
            and neighbor_frame.data is not None
        ):
            dt = abs(target_frame.timestamp_seconds - neighbor_frame.timestamp_seconds)
            if dt <= cfg.max_temporal_gap_seconds:
                neighbor_luma = FrameQualityAnalyzer.rgb_to_luminance(neighbor_frame.data)
                n_mean = float(np.mean(neighbor_luma))
                n_med = float(np.median(neighbor_luma))
                n_p10, n_p90 = np.percentile(neighbor_luma, [10, 90])

                mean_diff = mean_val - n_mean
                med_diff = median_val - n_med
                p90_diff = float(p90) - float(n_p90)
                p10_diff = float(p10) - float(n_p10)

                hist_curr = cls.compute_normalized_histogram(luma, bins=cfg.histogram_bins)
                hist_prev = cls.compute_normalized_histogram(neighbor_luma, bins=cfg.histogram_bins)
                db, bc = cls.compute_bhattacharyya_distance(hist_curr, hist_prev)

                # Classify Temporal Photometric Transition
                if db < cfg.bhattacharyya_change_threshold and abs(mean_diff) < 5.0:
                    change_cat = PhotometricChangeCategory.STABLE
                elif abs(mean_diff) >= 15.0 or db >= cfg.bhattacharyya_change_threshold:
                    if spatial_pat in {SpatialIlluminationPattern.LOCALIZED_BRIGHTNESS, SpatialIlluminationPattern.LOCALIZED_DARKNESS}:
                        change_cat = PhotometricChangeCategory.POTENTIAL_LOCAL_ILLUMINATION_CHANGE
                    else:
                        change_cat = PhotometricChangeCategory.POTENTIAL_EXPOSURE_TRANSITION
                        diagnostics.append(f"Global luminance shift detected (mean diff: {mean_diff:+.1f}, Bhattacharyya distance: {db:.3f}).")
                else:
                    change_cat = PhotometricChangeCategory.MIXED_CHANGE

                temp_change = TemporalPhotometricChange(
                    neighbor_frame_id=neighbor_frame.frame_id,
                    neighbor_timestamp_seconds=neighbor_frame.timestamp_seconds,
                    delta_t_seconds=round(dt, 4),
                    mean_luminance_change=round(mean_diff, 4),
                    median_luminance_change=round(med_diff, 4),
                    percentile_p90_change=round(p90_diff, 4),
                    percentile_p10_change=round(p10_diff, 4),
                    bhattacharyya_distance=db,
                    bhattacharyya_coefficient=bc,
                    change_category=change_cat,
                )
            else:
                diagnostics.append(f"Neighbor temporal gap ({dt:.3f}s) exceeds max threshold ({cfg.max_temporal_gap_seconds}s).")

        provenance = {
            "source_frame_id": target_frame.frame_id,
            "source_frame_index": target_frame.frame_index,
            "source_timestamp_seconds": target_frame.timestamp_seconds,
            "source_video": target_frame.source_video,
            "histogram_bins": cfg.histogram_bins,
            "analysis_dimensions": [w, h],
        }

        return PhotometricStabilityReport(
            frame_id=target_frame.frame_id,
            frame_index=target_frame.frame_index,
            timestamp_seconds=target_frame.timestamp_seconds,
            source_video=target_frame.source_video,
            luminance=ext_luma,
            dynamic_range=dyn_range,
            spatial_pattern=spatial_pat,
            spatial_nonuniformity_std=round(spatial_nonuniformity_std, 4),
            spatial_tiles=spatial_tiles,
            color=color_stats,
            temporal_change=temp_change,
            diagnostics=diagnostics,
            provenance=provenance,
            config_version=cfg.config_version,
        )
