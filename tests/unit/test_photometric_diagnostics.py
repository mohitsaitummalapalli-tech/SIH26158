"""Deterministic unit tests for Phase 2D.1 Illumination & Photometric Stability Diagnostics.

DISCLAIMER:
ALL FIXTURES IN THIS MODULE ARE SYNTHETIC TEST DATA GENERATED SOLELY FOR
PHOTOMETRIC STABILITY AND DIAGNOSTIC ENGINE AUDITING. THEY DO NOT REPRESENT A REAL DRONE FLIGHT.
"""

import pytest
import numpy as np
import cv2
import json

from src.preprocessing.decoder import DecodedFrame, DecodeStatus
from src.quality import (
    SpatialIlluminationPattern,
    PhotometricChangeCategory,
    PhotometricConfig,
    SpatialIlluminationTile,
    ColorStatistics,
    ExtendedLuminanceStatistics,
    DynamicRangeStatistics,
    TemporalPhotometricChange,
    PhotometricStabilityReport,
    PhotometricAnalyzer,
    FrameQualityAnalyzer,
)


def create_decoded_frame(
    data: np.ndarray,
    frame_id: str = "f_0001",
    frame_index: int = 1,
    timestamp: float = 0.5,
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
        dtype="uint8",
        data=data,
        source_video="synthetic_photometric_test.mp4",
        decode_status=DecodeStatus.SUCCESS,
        decoder_backend="TestSyntheticBackend",
    )


# 1. Report Contract & Serialization
def test_report_contract_and_serialization():
    # TEST DATA: Uniform gray image (120x120)
    data = np.full((120, 120, 3), 128, dtype=np.uint8)
    frame = create_decoded_frame(data)

    report = PhotometricAnalyzer.analyze_frame(frame)

    assert isinstance(report, PhotometricStabilityReport)
    assert report.frame_id == "f_0001"
    assert report.frame_index == 1
    assert report.timestamp_seconds == 0.5
    assert isinstance(report.luminance, ExtendedLuminanceStatistics)
    assert isinstance(report.dynamic_range, DynamicRangeStatistics)
    assert isinstance(report.spatial_pattern, SpatialIlluminationPattern)
    assert len(report.spatial_tiles) == 9 # 3x3 default
    assert report.spatial_pattern == SpatialIlluminationPattern.UNIFORM

    # Serialization
    json_str = report.to_json(indent=2)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["frame_id"] == "f_0001"
    assert parsed["spatial_pattern"] == "UNIFORM"


# 2. Luminance Reuse / Consistency with Phase 2B.1
def test_luminance_consistency_with_phase_2b1():
    # TEST DATA: Random RGB image
    np.random.seed(42)
    data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    frame = create_decoded_frame(data)

    quality_rep = FrameQualityAnalyzer.analyze_frame(frame)
    photo_rep = PhotometricAnalyzer.analyze_frame(frame, quality_report=quality_rep)

    expected_luma = FrameQualityAnalyzer.rgb_to_luminance(data)
    assert photo_rep.luminance.mean == pytest.approx(float(np.mean(expected_luma)), abs=1e-3)
    assert photo_rep.luminance.median == pytest.approx(float(np.median(expected_luma)), abs=1e-3)
    assert photo_rep.luminance.mean == pytest.approx(quality_rep.luminance.mean, abs=1e-3)


# 3. Dynamic-Range Statistics
def test_dynamic_range_statistics():
    # TEST DATA: Ramp image with values spanning 20 to 220
    ramp = np.linspace(20, 220, 10000, dtype=np.uint8).reshape((100, 100))
    data = np.stack([ramp, ramp, ramp], axis=-1)
    frame = create_decoded_frame(data)

    report = PhotometricAnalyzer.analyze_frame(frame)

    assert report.dynamic_range.observed_range == pytest.approx(200.0, abs=1.0)
    assert report.dynamic_range.percentile_dynamic_range > 180.0
    assert report.dynamic_range.effective_dynamic_range_fraction > 0.65


# 4. Shadow Fraction Detection
def test_shadow_fraction_detection():
    # TEST DATA: Half 0 (clipped black), half 128
    data = np.full((100, 100, 3), 128, dtype=np.uint8)
    data[:50, :] = 0 # 50% shadows
    frame = create_decoded_frame(data)

    report = PhotometricAnalyzer.analyze_frame(frame)

    assert report.dynamic_range.shadow_fraction == pytest.approx(0.50, abs=1e-3)
    assert report.dynamic_range.highlight_fraction == pytest.approx(0.0, abs=1e-3)


# 5. Highlight Fraction Detection
def test_highlight_fraction_detection():
    # TEST DATA: 30% saturated white (255)
    data = np.full((100, 100, 3), 100, dtype=np.uint8)
    data[:30, :] = 255
    frame = create_decoded_frame(data)

    report = PhotometricAnalyzer.analyze_frame(frame)

    assert report.dynamic_range.highlight_fraction == pytest.approx(0.30, abs=1e-3)
    assert report.dynamic_range.shadow_fraction == pytest.approx(0.0, abs=1e-3)


# 6. Spatial Illumination Metrics & Nonuniformity
def test_spatial_illumination_metrics():
    # TEST DATA: Uniform image -> low spatial nonuniformity
    data = np.full((90, 90, 3), 150, dtype=np.uint8)
    frame = create_decoded_frame(data)

    report = PhotometricAnalyzer.analyze_frame(frame)

    assert report.spatial_nonuniformity_std < 1.0
    for tile in report.spatial_tiles:
        assert tile.mean_luminance == pytest.approx(150.0, abs=1.0)
        assert tile.shadow_fraction == 0.0
        assert tile.highlight_fraction == 0.0


# 7. Histogram Normalization (Unit Sum)
def test_histogram_normalization():
    # TEST DATA: Random image
    data = np.random.randint(0, 256, (120, 120, 3), dtype=np.uint8)
    luma = FrameQualityAnalyzer.rgb_to_luminance(data)

    hist = PhotometricAnalyzer.compute_normalized_histogram(luma, bins=64)

    assert hist.shape == (64,)
    assert np.sum(hist) == pytest.approx(1.0, abs=1e-6)
    assert np.all(hist >= 0.0)


# 8. Histogram Distance Calculation (Bhattacharyya)
def test_histogram_distance_calculation():
    # Identical histograms -> DB = 0.0, BC = 1.0
    h1 = np.full(32, 1.0 / 32)
    h2 = np.full(32, 1.0 / 32)
    db, bc = PhotometricAnalyzer.compute_bhattacharyya_distance(h1, h2)
    assert db == pytest.approx(0.0, abs=1e-4)
    assert bc == pytest.approx(1.0, abs=1e-4)

    # Disjoint histograms -> DB > 5.0, BC = 0.0
    h_left = np.zeros(32)
    h_left[:16] = 1.0 / 16
    h_right = np.zeros(32)
    h_right[16:] = 1.0 / 16
    db_disjoint, bc_disjoint = PhotometricAnalyzer.compute_bhattacharyya_distance(h_left, h_right)
    assert db_disjoint > 5.0
    assert bc_disjoint == pytest.approx(0.0, abs=1e-4)


# 9. Unchanged Frame Pair Stability
def test_unchanged_frame_pair():
    # TEST DATA: Identical frames at t=0.0 and t=0.5
    data = np.full((100, 100, 3), 120, dtype=np.uint8)
    f0 = create_decoded_frame(data, "f_0", 0, 0.0)
    f1 = create_decoded_frame(data, "f_1", 1, 0.5)

    report = PhotometricAnalyzer.analyze_frame(f1, neighbor_frame=f0)

    assert report.temporal_change is not None
    assert report.temporal_change.mean_luminance_change == pytest.approx(0.0, abs=1e-4)
    assert report.temporal_change.bhattacharyya_distance == pytest.approx(0.0, abs=1e-4)
    assert report.temporal_change.change_category == PhotometricChangeCategory.STABLE


# 10. Global Brightness Shift Detection
def test_global_brightness_shift_detection():
    # TEST DATA: f0 is 100, f1 is 160 (+60 luma step)
    f0 = create_decoded_frame(np.full((100, 100, 3), 100, dtype=np.uint8), "f_0", 0, 0.0)
    f1 = create_decoded_frame(np.full((100, 100, 3), 160, dtype=np.uint8), "f_1", 1, 0.5)

    report = PhotometricAnalyzer.analyze_frame(f1, neighbor_frame=f0)

    assert report.temporal_change is not None
    assert report.temporal_change.mean_luminance_change == pytest.approx(60.0, abs=1.0)
    assert report.temporal_change.bhattacharyya_distance > 0.5
    assert report.temporal_change.change_category == PhotometricChangeCategory.POTENTIAL_EXPOSURE_TRANSITION


# 11. Localized Change / Nonuniform Gradient Detection
def test_localized_and_gradient_detection():
    # TEST DATA: Left-to-right gradient (10 -> 240)
    grad = np.tile(np.linspace(10, 240, 120, dtype=np.uint8), (120, 1))
    data_grad = np.stack([grad, grad, grad], axis=-1)
    frame_grad = create_decoded_frame(data_grad)

    rep_grad = PhotometricAnalyzer.analyze_frame(frame_grad)
    assert rep_grad.spatial_pattern == SpatialIlluminationPattern.GRADIENT

    # TEST DATA: Localized bright patch in center tile
    data_spot = np.full((120, 120, 3), 50, dtype=np.uint8)
    data_spot[40:80, 40:80] = 220 # center tile bright
    frame_spot = create_decoded_frame(data_spot)

    rep_spot = PhotometricAnalyzer.analyze_frame(frame_spot)
    assert rep_spot.spatial_pattern == SpatialIlluminationPattern.LOCALIZED_BRIGHTNESS


# 12. Unequal Timestamp Interval Handling
def test_unequal_timestamp_interval_handling():
    f0 = create_decoded_frame(np.full((100, 100, 3), 100, dtype=np.uint8), "f_0", 0, 0.0)
    f1 = create_decoded_frame(np.full((100, 100, 3), 110, dtype=np.uint8), "f_1", 1, 1.4) # dt = 1.4s

    report = PhotometricAnalyzer.analyze_frame(f1, neighbor_frame=f0)

    assert report.temporal_change is not None
    assert report.temporal_change.delta_t_seconds == pytest.approx(1.4, abs=1e-3)


# 13. Missing Neighbor Handling
def test_missing_neighbor_handling():
    f0 = create_decoded_frame(np.full((100, 100, 3), 128, dtype=np.uint8), "f_0", 0, 0.0)

    # Single-frame analysis without neighbor
    report = PhotometricAnalyzer.analyze_frame(f0)

    assert report.temporal_change is None
    assert report.spatial_pattern == SpatialIlluminationPattern.UNIFORM


# 14. Corrupted Neighbor Handling
def test_corrupted_neighbor_handling():
    f0_corrupt = DecodedFrame(
        frame_id="corrupt_f0",
        frame_index=0,
        timestamp_seconds=0.0,
        width=100,
        height=100,
        channels=3,
        channel_layout="RGB",
        dtype="uint8",
        data=None,
        source_video="test.mp4",
        decode_status=DecodeStatus.CORRUPTED,
    )
    f1 = create_decoded_frame(np.full((100, 100, 3), 128, dtype=np.uint8), "f_1", 1, 0.5)

    report = PhotometricAnalyzer.analyze_frame(f1, neighbor_frame=f0_corrupt)

    # Must produce valid report while safely omitting temporal comparison
    assert report.temporal_change is None
    assert report.luminance.mean == pytest.approx(128.0, abs=1e-3)


# 15. Provenance Preservation
def test_provenance_preservation():
    frame = create_decoded_frame(np.full((100, 100, 3), 100, dtype=np.uint8), "f_test", 5, 2.5)

    report = PhotometricAnalyzer.analyze_frame(frame)

    assert report.provenance["source_frame_id"] == "f_test"
    assert report.provenance["source_frame_index"] == 5
    assert report.provenance["source_timestamp_seconds"] == 2.5
    assert report.provenance["analysis_dimensions"] == [100, 100]


# 16. Immutability (Source Frames Untouched)
def test_immutability():
    data0 = np.full((100, 100, 3), 100, dtype=np.uint8)
    data1 = np.full((100, 100, 3), 150, dtype=np.uint8)
    copy0 = data0.copy()
    copy1 = data1.copy()

    f0 = create_decoded_frame(data0, "f_0", 0, 0.0)
    f1 = create_decoded_frame(data1, "f_1", 1, 0.5)

    _ = PhotometricAnalyzer.analyze_frame(f1, neighbor_frame=f0)

    assert np.array_equal(f0.data, copy0)
    assert np.array_equal(f1.data, copy1)


# 17. Deterministic Repeated Analysis
def test_deterministic_repeated_analysis():
    np.random.seed(123)
    data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    frame = create_decoded_frame(data)

    rep1 = PhotometricAnalyzer.analyze_frame(frame)
    rep2 = PhotometricAnalyzer.analyze_frame(frame)

    assert rep1.luminance.mean == rep2.luminance.mean
    assert rep1.luminance.p99 == rep2.luminance.p99
    assert rep1.dynamic_range.shadow_fraction == rep2.dynamic_range.shadow_fraction
    assert rep1.spatial_pattern == rep2.spatial_pattern


# 18. Configuration Override
def test_configuration_override():
    data = np.full((100, 100, 3), 10, dtype=np.uint8)
    frame = create_decoded_frame(data)

    # Default shadow threshold is 5.0 -> value 10 is NOT clipped shadow
    rep_default = PhotometricAnalyzer.analyze_frame(frame)
    assert rep_default.dynamic_range.shadow_fraction == 0.0

    # Custom shadow threshold of 15.0 -> value 10 IS clipped shadow
    custom_cfg = PhotometricConfig(shadow_threshold=15.0)
    rep_custom = PhotometricAnalyzer.analyze_frame(frame, config=custom_cfg)
    assert rep_custom.dynamic_range.shadow_fraction == 1.0
