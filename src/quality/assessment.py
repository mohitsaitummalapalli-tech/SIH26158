"""Frame Quality Assessment Subsystem.

Computes explainable, deterministic mathematical image statistics on a DecodedFrame:
- Luminance distribution (ITU-R BT.601)
- Clipping & saturation (shadow / highlight fractions)
- Contrast (percentile spreads, standard deviations)
- Sharpness metrics (Laplacian variance, Tenengrad Sobel energy)
- Spatial grid diagnostics (tile-level sharpness and luminance)
- High-frequency residual indicator (compression proxy)

SCIENTIFIC INTEGRITY:
These metrics are diagnostic image statistics and proxies. They do NOT represent
ground-truth motion blur measurements or final 3D reconstruction accuracy.
"""

import json
import numpy as np
import cv2
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple

from src.preprocessing.decoder import DecodedFrame, DecodeStatus


class QualityStatus(str, Enum):
    """Categorical diagnostic status of frame image quality."""
    VALID = "VALID"
    DEGRADED = "DEGRADED"
    SEVERELY_DEGRADED = "SEVERELY_DEGRADED"
    ANALYSIS_ERROR = "ANALYSIS_ERROR"


@dataclass(frozen=True)
class QualityAssessmentConfig:
    """Configuration thresholds and parameters for frame quality evaluation."""
    shadow_threshold: float = 5.0                # Luminance <= threshold is shadow clipped
    highlight_threshold: float = 250.0           # Luminance >= threshold is highlight clipped
    tile_grid_rows: int = 3                      # Number of vertical spatial grid tiles
    tile_grid_cols: int = 3                      # Number of horizontal spatial grid tiles
    min_image_width: int = 32                    # Minimum allowable image width for analysis
    min_image_height: int = 32                   # Minimum allowable image height for analysis
    degraded_laplacian_threshold: float = 30.0   # Diagnostic boundary for blurred / degraded sharpness
    severely_degraded_laplacian_threshold: float = 10.0 # Diagnostic boundary for extreme blur
    degraded_clipping_fraction: float = 0.20     # >20% clipped pixels triggers DEGRADED warning
    severe_clipping_fraction: float = 0.40       # >40% clipped pixels triggers SEVERELY_DEGRADED
    config_version: str = "QualityAssessment_v1.0"


@dataclass(frozen=True)
class LuminanceStatistics:
    """Luminance distribution statistics computed via ITU-R BT.601 luma conversion."""
    mean: float
    median: float
    std: float
    p5: float
    p25: float
    p75: float
    p95: float


@dataclass(frozen=True)
class ClippingStatistics:
    """Shadow and highlight clipping saturation diagnostics."""
    shadow_clipping_fraction: float
    highlight_clipping_fraction: float
    channel_shadow_clipping: Dict[str, float]
    channel_highlight_clipping: Dict[str, float]


@dataclass(frozen=True)
class ContrastStatistics:
    """Deterministic contrast indicators from luminance distribution."""
    std_luminance: float
    percentile_spread_90_10: float
    michelson_contrast: float


@dataclass(frozen=True)
class SharpnessStatistics:
    """Edge-energy and high-frequency sharpness statistics."""
    laplacian_variance: float
    tenengrad_gradient_energy: float
    modified_laplacian: float


@dataclass(frozen=True)
class SpatialTileQuality:
    """Quality statistics evaluated on an individual spatial sub-grid tile."""
    tile_row: int
    tile_col: int
    bbox: Tuple[int, int, int, int]  # (y_min, x_min, y_max, x_max)
    mean_luminance: float
    contrast_std: float
    laplacian_variance: float


@dataclass
class FrameQualityReport:
    """Comprehensive, structured quality diagnostic report for a single video frame."""
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    source_video: str
    status: QualityStatus
    luminance: Optional[LuminanceStatistics]
    clipping: Optional[ClippingStatistics]
    contrast: Optional[ContrastStatistics]
    sharpness: Optional[SharpnessStatistics]
    spatial_tiles: List[SpatialTileQuality] = field(default_factory=list)
    compression_artifact_indicator: Optional[float] = None
    diagnostics: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    config_version: str = "QualityAssessment_v1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to standard serializable dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class FrameQualityAnalyzer:
    """Deterministic, explainable frame quality analysis engine."""

    @staticmethod
    def rgb_to_luminance(rgb_array: np.ndarray) -> np.ndarray:
        """Convert canonical RGB uint8 array to float64 luminance via ITU-R BT.601 standard:
        
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        """
        if rgb_array.ndim == 2:
            return rgb_array.astype(np.float64)
        elif rgb_array.ndim == 3 and rgb_array.shape[2] == 3:
            r = rgb_array[:, :, 0].astype(np.float64)
            g = rgb_array[:, :, 1].astype(np.float64)
            b = rgb_array[:, :, 2].astype(np.float64)
            return 0.299 * r + 0.587 * g + 0.114 * b
        elif rgb_array.ndim == 3 and rgb_array.shape[2] == 1:
            return rgb_array[:, :, 0].astype(np.float64)
        else:
            raise ValueError(f"Unsupported array shape for luminance conversion: {rgb_array.shape}")

    @classmethod
    def compute_luminance_statistics(cls, luma: np.ndarray) -> LuminanceStatistics:
        """Compute mean, median, standard deviation, and percentiles on luma matrix."""
        mean_val = float(np.mean(luma))
        median_val = float(np.median(luma))
        std_val = float(np.std(luma))
        p5, p25, p75, p95 = np.percentile(luma, [5, 25, 75, 95])
        return LuminanceStatistics(
            mean=round(mean_val, 4),
            median=round(median_val, 4),
            std=round(std_val, 4),
            p5=round(float(p5), 4),
            p25=round(float(p25), 4),
            p75=round(float(p75), 4),
            p95=round(float(p95), 4),
        )

    @classmethod
    def compute_clipping_statistics(
        cls, rgb_array: np.ndarray, luma: np.ndarray, config: QualityAssessmentConfig
    ) -> ClippingStatistics:
        """Measure fraction of pixels in extreme shadow (<= shadow_thresh) and highlight (>= highlight_thresh)."""
        total_pixels = luma.size
        shadow_mask = luma <= config.shadow_threshold
        highlight_mask = luma >= config.highlight_threshold

        shadow_fraction = float(np.sum(shadow_mask)) / total_pixels
        highlight_fraction = float(np.sum(highlight_mask)) / total_pixels

        channel_shadow = {}
        channel_highlight = {}
        channel_names = ["R", "G", "B"] if rgb_array.ndim == 3 and rgb_array.shape[2] == 3 else ["Y"]

        for idx, ch_name in enumerate(channel_names):
            ch_data = rgb_array[:, :, idx] if rgb_array.ndim == 3 else rgb_array
            ch_shadow = float(np.sum(ch_data <= config.shadow_threshold)) / total_pixels
            ch_highlight = float(np.sum(ch_data >= config.highlight_threshold)) / total_pixels
            channel_shadow[ch_name] = round(ch_shadow, 4)
            channel_highlight[ch_name] = round(ch_highlight, 4)

        return ClippingStatistics(
            shadow_clipping_fraction=round(shadow_fraction, 4),
            highlight_clipping_fraction=round(highlight_fraction, 4),
            channel_shadow_clipping=channel_shadow,
            channel_highlight_clipping=channel_highlight,
        )

    @classmethod
    def compute_contrast_statistics(cls, luma: np.ndarray) -> ContrastStatistics:
        """Compute contrast spread indicators."""
        std_val = float(np.std(luma))
        p10, p90, p5, p95 = np.percentile(luma, [10, 90, 5, 95])
        percentile_spread = float(p90 - p10)

        denom = float(p95 + p5 + 1e-6)
        michelson = float(p95 - p5) / denom if denom > 0 else 0.0

        return ContrastStatistics(
            std_luminance=round(std_val, 4),
            percentile_spread_90_10=round(percentile_spread, 4),
            michelson_contrast=round(michelson, 4),
        )

    @classmethod
    def compute_sharpness_statistics(cls, luma: np.ndarray) -> SharpnessStatistics:
        """Compute classical discrete edge energy and Laplacian sharpness statistics."""
        # 1. Variance of Laplacian: Var(∇² I)
        luma_f32 = luma.astype(np.float32)
        laplacian = cv2.Laplacian(luma_f32, cv2.CV_32F)
        lap_var = float(np.var(laplacian))

        # 2. Tenengrad Gradient Energy: (1/N) * sum(Gx² + Gy²)
        sobel_x = cv2.Sobel(luma_f32, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(luma_f32, cv2.CV_32F, 0, 1, ksize=3)
        tenengrad = float(np.mean(sobel_x**2 + sobel_y**2))

        # 3. Modified Laplacian: Sum of absolute 1D second derivatives
        # |2*I(x,y) - I(x-1,y) - I(x+1,y)| + |2*I(x,y) - I(x,y-1) - I(x,y+1)|
        kernel_x = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32)
        kernel_y = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32)
        m_lap_x = np.abs(cv2.filter2D(luma_f32, -1, kernel_x))
        m_lap_y = np.abs(cv2.filter2D(luma_f32, -1, kernel_y))
        mod_lap = float(np.mean(m_lap_x + m_lap_y))

        return SharpnessStatistics(
            laplacian_variance=round(lap_var, 4),
            tenengrad_gradient_energy=round(tenengrad, 4),
            modified_laplacian=round(mod_lap, 4),
        )

    @classmethod
    def compute_spatial_tiles(
        cls, luma: np.ndarray, config: QualityAssessmentConfig
    ) -> List[SpatialTileQuality]:
        """Evaluate localized statistics across a regular grid of spatial tiles."""
        h, w = luma.shape
        rows = max(1, config.tile_grid_rows)
        cols = max(1, config.tile_grid_cols)

        row_edges = np.linspace(0, h, rows + 1, dtype=int)
        col_edges = np.linspace(0, w, cols + 1, dtype=int)

        tiles: List[SpatialTileQuality] = []

        for r in range(rows):
            for c in range(cols):
                y_min, y_max = row_edges[r], row_edges[r + 1]
                x_min, x_max = col_edges[c], col_edges[c + 1]

                tile_data = luma[y_min:y_max, x_min:x_max]
                if tile_data.size < 4:
                    continue

                mean_val = float(np.mean(tile_data))
                std_val = float(np.std(tile_data))
                lap = cv2.Laplacian(tile_data.astype(np.float32), cv2.CV_32F)
                lap_var = float(np.var(lap))

                tiles.append(
                    SpatialTileQuality(
                        tile_row=r,
                        tile_col=c,
                        bbox=(int(y_min), int(x_min), int(y_max), int(x_max)),
                        mean_luminance=round(mean_val, 4),
                        contrast_std=round(std_val, 4),
                        laplacian_variance=round(lap_var, 4),
                    )
                )

        return tiles

    @classmethod
    def compute_compression_artifact_indicator(cls, luma: np.ndarray) -> float:
        """Compute high-frequency residual energy as a conservative proxy for compression noise."""
        # Gaussian blurred residual: (I - G(I)) high-frequency standard deviation
        blurred = cv2.GaussianBlur(luma.astype(np.float32), (5, 5), 1.0)
        residual = luma.astype(np.float32) - blurred
        return round(float(np.std(residual)), 4)

    @classmethod
    def analyze_frame(
        cls, frame: DecodedFrame, config: Optional[QualityAssessmentConfig] = None
    ) -> FrameQualityReport:
        """Analyze a DecodedFrame and return a structured FrameQualityReport.
        
        DOES NOT MODIFY frame.data.
        """
        cfg = config or QualityAssessmentConfig()
        diagnostics: List[str] = []

        # 1. Defensive validation of input frame
        if frame.decode_status != DecodeStatus.SUCCESS or frame.data is None:
            return FrameQualityReport(
                frame_id=frame.frame_id,
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                source_video=frame.source_video,
                status=QualityStatus.ANALYSIS_ERROR,
                luminance=None,
                clipping=None,
                contrast=None,
                sharpness=None,
                spatial_tiles=[],
                diagnostics=[f"Decode status is {frame.decode_status.value}; image data is unavailable."],
                provenance={"source_frame_id": frame.frame_id, "decoder_backend": frame.decoder_backend},
                config_version=cfg.config_version,
            )

        # Check dimensions
        h, w = frame.data.shape[:2]
        if h < cfg.min_image_height or w < cfg.min_image_width:
            return FrameQualityReport(
                frame_id=frame.frame_id,
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                source_video=frame.source_video,
                status=QualityStatus.ANALYSIS_ERROR,
                luminance=None,
                clipping=None,
                contrast=None,
                sharpness=None,
                spatial_tiles=[],
                diagnostics=[f"Image dimensions ({w}x{h}) are below minimum ({cfg.min_image_width}x{cfg.min_image_height})."],
                provenance={"source_frame_id": frame.frame_id, "decoder_backend": frame.decoder_backend},
                config_version=cfg.config_version,
            )

        # Check finite
        if not np.all(np.isfinite(frame.data)):
            return FrameQualityReport(
                frame_id=frame.frame_id,
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                source_video=frame.source_video,
                status=QualityStatus.ANALYSIS_ERROR,
                luminance=None,
                clipping=None,
                contrast=None,
                sharpness=None,
                spatial_tiles=[],
                diagnostics=["Image buffer contains non-finite (NaN / Inf) values."],
                provenance={"source_frame_id": frame.frame_id, "decoder_backend": frame.decoder_backend},
                config_version=cfg.config_version,
            )

        # 2. Extract Luminance Matrix
        luma = cls.rgb_to_luminance(frame.data)

        # 3. Compute Metrics
        luma_stats = cls.compute_luminance_statistics(luma)
        clipping_stats = cls.compute_clipping_statistics(frame.data, luma, cfg)
        contrast_stats = cls.compute_contrast_statistics(luma)
        sharpness_stats = cls.compute_sharpness_statistics(luma)
        spatial_tiles = cls.compute_spatial_tiles(luma, cfg)
        compression_indicator = cls.compute_compression_artifact_indicator(luma)

        # 4. Determine Quality Status Categorization
        total_clipping = clipping_stats.shadow_clipping_fraction + clipping_stats.highlight_clipping_fraction
        status = QualityStatus.VALID

        if (
            sharpness_stats.laplacian_variance < cfg.severely_degraded_laplacian_threshold
            or total_clipping >= cfg.severe_clipping_fraction
        ):
            status = QualityStatus.SEVERELY_DEGRADED
            if sharpness_stats.laplacian_variance < cfg.severely_degraded_laplacian_threshold:
                diagnostics.append(
                    f"Severe blur detected: Laplacian variance ({sharpness_stats.laplacian_variance}) < {cfg.severely_degraded_laplacian_threshold}."
                )
            if total_clipping >= cfg.severe_clipping_fraction:
                diagnostics.append(
                    f"Severe clipping detected: {round(total_clipping * 100, 2)}% pixels saturated."
                )
        elif (
            sharpness_stats.laplacian_variance < cfg.degraded_laplacian_threshold
            or total_clipping >= cfg.degraded_clipping_fraction
        ):
            status = QualityStatus.DEGRADED
            if sharpness_stats.laplacian_variance < cfg.degraded_laplacian_threshold:
                diagnostics.append(
                    f"Moderate blur detected: Laplacian variance ({sharpness_stats.laplacian_variance}) < {cfg.degraded_laplacian_threshold}."
                )
            if total_clipping >= cfg.degraded_clipping_fraction:
                diagnostics.append(
                    f"Moderate clipping detected: {round(total_clipping * 100, 2)}% pixels saturated."
                )

        provenance = {
            "source_frame_id": frame.frame_id,
            "source_frame_index": frame.frame_index,
            "source_timestamp_seconds": frame.timestamp_seconds,
            "source_video": frame.source_video,
            "decoder_backend": frame.decoder_backend,
            "analysis_dimensions": [w, h],
        }

        return FrameQualityReport(
            frame_id=frame.frame_id,
            frame_index=frame.frame_index,
            timestamp_seconds=frame.timestamp_seconds,
            source_video=frame.source_video,
            status=status,
            luminance=luma_stats,
            clipping=clipping_stats,
            contrast=contrast_stats,
            sharpness=sharpness_stats,
            spatial_tiles=spatial_tiles,
            compression_artifact_indicator=compression_indicator,
            diagnostics=diagnostics,
            provenance=provenance,
            config_version=cfg.config_version,
        )
