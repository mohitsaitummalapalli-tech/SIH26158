"""Deterministic unit tests for Phase 2B.1 Frame Quality Assessment.

DISCLAIMER:
ALL FIXTURES IN THIS MODULE ARE SYNTHETIC TEST DATA GENERATED SOLELY FOR
STATISTICAL AND QUALITY ENGINE AUDITING. THEY DO NOT REPRESENT A REAL DRONE FLIGHT.
"""

import pytest
import numpy as np
import cv2
import json

from src.preprocessing.decoder import DecodedFrame, DecodeStatus
from src.quality import (
    QualityStatus,
    QualityAssessmentConfig,
    LuminanceStatistics,
    ClippingStatistics,
    ContrastStatistics,
    SharpnessStatistics,
    SpatialTileQuality,
    FrameQualityReport,
    FrameQualityAnalyzer,
)


def create_decoded_frame(
    data: np.ndarray,
    frame_id: str = "test_frame_001",
    frame_index: int = 0,
    timestamp: float = 0.0,
) -> DecodedFrame:
    """Helper to wrap numpy array in DecodedFrame (TEST DATA)."""
    h, w = data.shape[:2]
    channels = data.shape[2] if data.ndim == 3 else 1
    return DecodedFrame(
        frame_id=frame_id,
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        width=w,
        height=h,
        channels=channels,
        channel_layout="RGB" if channels == 3 else "GRAY",
        dtype="uint8" if data.dtype == np.uint8 else str(data.dtype),
        data=data,
        source_video="test_synthetic_flight.mp4",
        decode_status=DecodeStatus.SUCCESS,
        decoder_backend="TestSyntheticBackend",
    )


# Test 1: Report Structure & Serialization
def test_report_structure():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    frame = create_decoded_frame(img)

    report = FrameQualityAnalyzer.analyze_frame(frame)

    assert isinstance(report, FrameQualityReport)
    assert report.frame_id == "test_frame_001"
    assert report.frame_index == 0
    assert report.timestamp_seconds == 0.0
    assert report.status in list(QualityStatus)
    assert report.luminance is not None
    assert report.clipping is not None
    assert report.contrast is not None
    assert report.sharpness is not None
    assert len(report.spatial_tiles) == 9 # 3x3 default

    # Verify JSON serialization
    json_str = report.to_json(indent=2)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["frame_id"] == "test_frame_001"
    assert parsed["status"] == report.status.value


# Test 2: Luminance Statistics on Uniform Gray & Gradient
def test_luminance_statistics():
    # Uniform gray 128
    img_gray = np.full((100, 100, 3), 128, dtype=np.uint8)
    report_gray = FrameQualityAnalyzer.analyze_frame(create_decoded_frame(img_gray))

    assert report_gray.luminance.mean == pytest.approx(128.0, abs=1e-2)
    assert report_gray.luminance.median == pytest.approx(128.0, abs=1e-2)
    assert report_gray.luminance.std == pytest.approx(0.0, abs=1e-2)

    # Linear gradient across columns from 0 to 255
    grad_1d = np.linspace(0, 255, 100, dtype=np.uint8)
    grad_2d = np.tile(grad_1d, (100, 1))
    img_grad = np.stack([grad_2d, grad_2d, grad_2d], axis=-1)

    report_grad = FrameQualityAnalyzer.analyze_frame(create_decoded_frame(img_grad))
    assert report_grad.luminance.mean == pytest.approx(127.5, abs=2.0)
    assert report_grad.luminance.std > 60.0
    assert report_grad.luminance.p5 < 20.0
    assert report_grad.luminance.p95 > 235.0


# Test 3: Clipping Detection (Shadow & Highlight)
def test_clipping_detection():
    # Mostly black image (<= 5)
    img_dark = np.zeros((100, 100, 3), dtype=np.uint8)
    img_dark[0:10, 0:10, :] = 100 # 1% non-clipped
    report_dark = FrameQualityAnalyzer.analyze_frame(create_decoded_frame(img_dark))

    assert report_dark.clipping.shadow_clipping_fraction >= 0.99
    assert report_dark.clipping.highlight_clipping_fraction == 0.0
    assert report_dark.status == QualityStatus.SEVERELY_DEGRADED

    # Mostly bright / saturated white (>= 250)
    img_bright = np.full((100, 100, 3), 255, dtype=np.uint8)
    report_bright = FrameQualityAnalyzer.analyze_frame(create_decoded_frame(img_bright))

    assert report_bright.clipping.highlight_clipping_fraction == 1.0
    assert report_bright.status == QualityStatus.SEVERELY_DEGRADED


# Test 4: Contrast Measurement
def test_contrast_measurement():
    # Low contrast uniform image
    img_low = np.full((100, 100, 3), 128, dtype=np.uint8)
    rep_low = FrameQualityAnalyzer.analyze_frame(create_decoded_frame(img_low))
    assert rep_low.contrast.std_luminance == pytest.approx(0.0, abs=1e-3)
    assert rep_low.contrast.percentile_spread_90_10 == pytest.approx(0.0, abs=1e-3)

    # High contrast checkerboard pattern
    img_high = np.zeros((100, 100, 3), dtype=np.uint8)
    img_high[::2, ::2, :] = 255
    img_high[1::2, 1::2, :] = 255
    rep_high = FrameQualityAnalyzer.analyze_frame(create_decoded_frame(img_high))
    assert rep_high.contrast.std_luminance > 100.0
    assert rep_high.contrast.percentile_spread_90_10 > 200.0


# Test 5: Laplacian Sharpness on Flat vs Structured
def test_laplacian_sharpness_behavior():
    # Flat image has zero Laplacian variance
    flat = np.full((100, 100, 3), 128, dtype=np.uint8)
    rep_flat = FrameQualityAnalyzer.analyze_frame(create_decoded_frame(flat))
    assert rep_flat.sharpness.laplacian_variance == pytest.approx(0.0, abs=1e-4)

    # Sharp binary grid
    grid = np.zeros((100, 100, 3), dtype=np.uint8)
    grid[::10, :, :] = 255
    grid[:, ::10, :] = 255
    rep_grid = FrameQualityAnalyzer.analyze_frame(create_decoded_frame(grid))
    assert rep_grid.sharpness.laplacian_variance > 500.0


# Test 6: Blur vs Sharp Relative Ordering
def test_blur_vs_sharp_relative_ordering():
    # Create sharp synthetic edge pattern
    sharp = np.zeros((120, 120, 3), dtype=np.uint8)
    sharp[30:90, 30:90, :] = 220

    # Apply Gaussian blur kernel (sigma=4.0)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 4.0)

    rep_sharp = FrameQualityAnalyzer.analyze_frame(create_decoded_frame(sharp))
    rep_blur = FrameQualityAnalyzer.analyze_frame(create_decoded_frame(blurred))

    # Sharpness metric of sharp image must strictly exceed blurred counterpart
    assert rep_sharp.sharpness.laplacian_variance > rep_blur.sharpness.laplacian_variance
    assert rep_sharp.sharpness.tenengrad_gradient_energy > rep_blur.sharpness.tenengrad_gradient_energy
    assert rep_sharp.sharpness.modified_laplacian > rep_blur.sharpness.modified_laplacian


# Test 7: Spatial Tile Metrics
def test_spatial_tile_metrics():
    # Image where center tile is sharp but edges are flat
    img = np.full((120, 120, 3), 128, dtype=np.uint8)
    # Center tile (row 1, col 1 in 3x3 grid: y:40..80, x:40..80)
    img[40:80:4, 40:80, :] = 255 # stripes in center

    cfg = QualityAssessmentConfig(tile_grid_rows=3, tile_grid_cols=3)
    rep = FrameQualityAnalyzer.analyze_frame(create_decoded_frame(img), config=cfg)

    assert len(rep.spatial_tiles) == 9

    center_tile = next(t for t in rep.spatial_tiles if t.tile_row == 1 and t.tile_col == 1)
    corner_tile = next(t for t in rep.spatial_tiles if t.tile_row == 0 and t.tile_col == 0)

    assert center_tile.laplacian_variance > 100.0
    assert corner_tile.laplacian_variance == pytest.approx(0.0, abs=1e-4)


# Test 8: RGB Channel Correctness (BT.601 Weightings)
def test_rgb_channel_weightings():
    # Pure Red [255, 0, 0] -> Y = 0.299 * 255 = 76.245
    pure_red = np.zeros((50, 50, 3), dtype=np.uint8)
    pure_red[:, :, 0] = 255
    rep_red = FrameQualityAnalyzer.analyze_frame(create_decoded_frame(pure_red))
    assert rep_red.luminance.mean == pytest.approx(76.245, abs=0.5)

    # Pure Green [0, 255, 0] -> Y = 0.587 * 255 = 149.685
    pure_green = np.zeros((50, 50, 3), dtype=np.uint8)
    pure_green[:, :, 1] = 255
    rep_green = FrameQualityAnalyzer.analyze_frame(create_decoded_frame(pure_green))
    assert rep_green.luminance.mean == pytest.approx(149.685, abs=0.5)

    # Pure Blue [0, 0, 255] -> Y = 0.114 * 255 = 29.07
    pure_blue = np.zeros((50, 50, 3), dtype=np.uint8)
    pure_blue[:, :, 2] = 255
    rep_blue = FrameQualityAnalyzer.analyze_frame(create_decoded_frame(pure_blue))
    assert rep_blue.luminance.mean == pytest.approx(29.07, abs=0.5)


# Test 9: Configuration Override
def test_configuration_override():
    # Moderately blurred image on mid-gray background (no clipping)
    sharp = np.full((100, 100, 3), 128, dtype=np.uint8)
    sharp[20:80, 20:80, :] = 200
    blurred = cv2.GaussianBlur(sharp, (9, 9), 2.0)
    frame = create_decoded_frame(blurred)

    # Strict config where degraded threshold is high
    strict_cfg = QualityAssessmentConfig(degraded_laplacian_threshold=500.0)
    rep_strict = FrameQualityAnalyzer.analyze_frame(frame, config=strict_cfg)
    assert rep_strict.status in {QualityStatus.DEGRADED, QualityStatus.SEVERELY_DEGRADED}

    # Lenient config where threshold is very low
    lenient_cfg = QualityAssessmentConfig(
        degraded_laplacian_threshold=0.01,
        severely_degraded_laplacian_threshold=0.001,
    )
    rep_lenient = FrameQualityAnalyzer.analyze_frame(frame, config=lenient_cfg)
    assert rep_lenient.status == QualityStatus.VALID


# Test 10: Original Frame Unchanged (Immutability)
def test_original_frame_unchanged():
    img_original = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    img_copy = img_original.copy()
    frame = create_decoded_frame(img_original)

    _ = FrameQualityAnalyzer.analyze_frame(frame)

    # Assert frame buffer has not been modified
    assert np.array_equal(frame.data, img_copy)


# Test 11: Provenance Preservation
def test_provenance_preservation():
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    frame = create_decoded_frame(img, frame_id="flight_f0042", frame_index=42, timestamp=21.0)

    rep = FrameQualityAnalyzer.analyze_frame(frame)

    assert rep.provenance["source_frame_id"] == "flight_f0042"
    assert rep.provenance["source_frame_index"] == 42
    assert rep.provenance["source_timestamp_seconds"] == 21.0
    assert rep.provenance["source_video"] == "test_synthetic_flight.mp4"
    assert rep.provenance["analysis_dimensions"] == [64, 64]


# Test 12: Deterministic Repeated Analysis
def test_deterministic_repeated_analysis():
    img = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
    frame = create_decoded_frame(img)

    rep1 = FrameQualityAnalyzer.analyze_frame(frame)
    rep2 = FrameQualityAnalyzer.analyze_frame(frame)

    assert rep1.luminance.mean == rep2.luminance.mean
    assert rep1.sharpness.laplacian_variance == rep2.sharpness.laplacian_variance
    assert rep1.clipping.shadow_clipping_fraction == rep2.clipping.shadow_clipping_fraction
    assert rep1.status == rep2.status


# Test 13: Invalid / Too-Small Image Handling
def test_too_small_image_handling():
    # 16x16 is below default 32x32 minimum
    tiny = np.zeros((16, 16, 3), dtype=np.uint8)
    frame = create_decoded_frame(tiny)

    rep = FrameQualityAnalyzer.analyze_frame(frame)

    assert rep.status == QualityStatus.ANALYSIS_ERROR
    assert len(rep.diagnostics) > 0
    assert "below minimum" in rep.diagnostics[0]


# Test 14: NaN / Inf Defensive Handling
def test_nan_inf_defensive_handling():
    # Corrupted float array with NaN
    corrupt_data = np.full((50, 50, 3), np.nan, dtype=np.float32)
    frame = DecodedFrame(
        frame_id="corrupt_frame",
        frame_index=0,
        timestamp_seconds=0.0,
        width=50,
        height=50,
        channels=3,
        channel_layout="RGB",
        dtype="float32",
        data=corrupt_data,
        source_video="corrupt.mp4",
        decode_status=DecodeStatus.SUCCESS,
    )

    rep = FrameQualityAnalyzer.analyze_frame(frame)
    assert rep.status == QualityStatus.ANALYSIS_ERROR
    assert "non-finite" in rep.diagnostics[0]
