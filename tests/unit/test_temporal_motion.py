"""Deterministic unit tests for Phase 2C.1.1 Temporal Motion and Blur Diagnostics Scientific Corrections.

DISCLAIMER:
ALL FIXTURES IN THIS MODULE ARE SYNTHETIC TEST DATA GENERATED SOLELY FOR
TEMPORAL MOTION AND DIAGNOSTIC ENGINE AUDITING. THEY DO NOT REPRESENT A REAL DRONE FLIGHT.
"""

import pytest
import numpy as np
import cv2
import json

from src.preprocessing.decoder import DecodedFrame, DecodeStatus
from src.quality import (
    MotionCategory,
    TemporalMotionConfig,
    SpatialMotionTile,
    TemporalMotionBlurReport,
    TemporalMotionAnalyzer,
    FrameQualityAnalyzer,
)


def create_decoded_frame(
    data: np.ndarray,
    frame_id: str,
    frame_index: int,
    timestamp: float,
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
        source_video="synthetic_motion_test.mp4",
        decode_status=DecodeStatus.SUCCESS,
        decoder_backend="TestSyntheticBackend",
    )


def generate_textured_pattern(shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    """Generate synthetic high-frequency texture shifted by (shift_x, shift_y) (TEST DATA)."""
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    for y in range(0, 120, 10):
        for x in range(0, 120, 10):
            if ((x // 10) + (y // 10)) % 2 == 0:
                img[y:y+10, x:x+10] = 200
            else:
                img[y:y+10, x:x+10] = 50

    if shift_x != 0 or shift_y != 0:
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, M, (120, 120), borderMode=cv2.BORDER_REFLECT)
    return img


# Test 1: Report Structure & Serialization
def test_report_structure():
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "frame_0000", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(2, 0), "frame_0001", 1, 0.5)
    f2 = create_decoded_frame(generate_textured_pattern(4, 0), "frame_0002", 2, 1.0)

    report = TemporalMotionAnalyzer.analyze_temporal_motion(f1, prev_frame=f0, next_frame=f2)

    assert isinstance(report, TemporalMotionBlurReport)
    assert report.frame_id == "frame_0001"
    assert report.frame_index == 1
    assert report.timestamp_seconds == 0.5
    assert len(report.neighbor_frame_ids) == 2
    assert report.motion_category in list(MotionCategory)
    assert len(report.spatial_tiles) == 9 # 3x3 default
    assert hasattr(report, "motion_blur_indicator")

    # JSON serialization
    json_str = report.to_json(indent=2)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["frame_id"] == "frame_0001"
    assert parsed["motion_category"] == report.motion_category.value


# Test 2: Timestamp-Aware Temporal Velocity Normalization
def test_timestamp_aware_temporal_delta():
    # 5 px translation over 0.5 s vs 5 px translation over 2.0 s
    f_prev = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f_curr_fast = create_decoded_frame(generate_textured_pattern(5, 0), "f_1", 1, 0.5) # dt = 0.5s -> ~10 px/s
    f_curr_slow = create_decoded_frame(generate_textured_pattern(5, 0), "f_1", 1, 2.0) # dt = 2.0s -> ~2.5 px/s

    rep_fast = TemporalMotionAnalyzer.analyze_temporal_motion(f_curr_fast, prev_frame=f_prev)
    rep_slow = TemporalMotionAnalyzer.analyze_temporal_motion(f_curr_slow, prev_frame=f_prev)

    # Displacements in pixels should be similar
    assert rep_fast.mean_displacement_pixels == pytest.approx(rep_slow.mean_displacement_pixels, abs=1.0)
    # Velocity (px/s) must scale inversely with dt
    assert rep_fast.global_velocity_px_per_sec > 2.5 * rep_slow.global_velocity_px_per_sec


# Test 3: Static Sequence Gives LOW_APPARENT_MOTION
def test_static_sequence_gives_low_motion():
    static_img = generate_textured_pattern(0, 0)
    f0 = create_decoded_frame(static_img, "f_0", 0, 0.0)
    f1 = create_decoded_frame(static_img, "f_1", 1, 0.5)
    f2 = create_decoded_frame(static_img, "f_2", 2, 1.0)

    report = TemporalMotionAnalyzer.analyze_temporal_motion(f1, prev_frame=f0, next_frame=f2)

    assert report.median_displacement_pixels < 0.5
    assert report.motion_category == MotionCategory.LOW_APPARENT_MOTION
    assert report.motion_blur_indicator == pytest.approx(0.0, abs=1e-3)


# Test 4: Coherent Motion Gives POTENTIAL_CAMERA_MOTION
def test_coherent_motion_gives_camera_motion():
    # Shift entire image consistently by 4 px along X axis
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(4, 0), "f_1", 1, 0.5)
    f2 = create_decoded_frame(generate_textured_pattern(8, 0), "f_2", 2, 1.0)

    report = TemporalMotionAnalyzer.analyze_temporal_motion(f1, prev_frame=f0, next_frame=f2)

    assert report.motion_category == MotionCategory.POTENTIAL_CAMERA_MOTION
    assert report.directional_coherence_score > 0.70
    assert report.median_displacement_pixels > 2.0


# Test 5: Local Motion Produces Spatial Heterogeneity
def test_local_motion_heterogeneity():
    # Static background with a small moving patch in the bottom-right quadrant
    bg = generate_textured_pattern(0, 0)
    f0_data = bg.copy()

    f1_data = bg.copy()
    # Move bottom-right patch (y: 80..120, x: 80..120) by 12 pixels
    patch = bg[80:120, 80:120].copy()
    M = np.float32([[1, 0, 12], [0, 1, 0]])
    f1_data[80:120, 80:120] = cv2.warpAffine(patch, M, (40, 40), borderMode=cv2.BORDER_REFLECT)

    f0 = create_decoded_frame(f0_data, "f_0", 0, 0.0)
    f1 = create_decoded_frame(f1_data, "f_1", 1, 0.5)

    report = TemporalMotionAnalyzer.analyze_temporal_motion(f1, prev_frame=f0)

    # Must detect local outlier tile
    br_tile = next(t for t in report.spatial_tiles if t.tile_row == 2 and t.tile_col == 2)
    tl_tile = next(t for t in report.spatial_tiles if t.tile_row == 0 and t.tile_col == 0)

    assert br_tile.mean_displacement_pixels > tl_tile.mean_displacement_pixels + 1.0
    assert report.motion_category in {MotionCategory.POTENTIAL_LOCAL_MOTION, MotionCategory.MIXED_MOTION}
    assert report.spatial_motion_heterogeneity > 0.5


# Test 6: Temporal Inconsistency / Acceleration Detection
def test_temporal_inconsistency_detection():
    # Sequence with abrupt motion reversal: 0 -> +6 px -> -6 px
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(6, 0), "f_1", 1, 0.5)
    f2 = create_decoded_frame(generate_textured_pattern(0, 0), "f_2", 2, 1.0) # sudden reversal

    report = TemporalMotionAnalyzer.analyze_temporal_motion(f1, prev_frame=f0, next_frame=f2)

    assert report.temporal_acceleration_px_per_sec2 is not None
    assert report.temporal_acceleration_px_per_sec2 > 5.0


# Test 7: Missing Neighbor Handling (First, Last, Isolated)
def test_missing_neighbor_handling():
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(3, 0), "f_1", 1, 0.5)

    # First frame (no prev)
    rep_first = TemporalMotionAnalyzer.analyze_temporal_motion(f0, next_frame=f1)
    assert len(rep_first.neighbor_frame_ids) == 1
    assert rep_first.neighbor_frame_ids[0] == "f_1"

    # Last frame (no next)
    rep_last = TemporalMotionAnalyzer.analyze_temporal_motion(f1, prev_frame=f0)
    assert len(rep_last.neighbor_frame_ids) == 1
    assert rep_last.neighbor_frame_ids[0] == "f_0"

    # Isolated frame (no neighbors)
    rep_iso = TemporalMotionAnalyzer.analyze_temporal_motion(f0)
    assert rep_iso.motion_category == MotionCategory.INSUFFICIENT_EVIDENCE
    assert len(rep_iso.neighbor_frame_ids) == 0


# Test 8: Unequal Timestamp Intervals
def test_unequal_timestamp_intervals():
    # dt_prev = 0.2s, dt_next = 0.8s
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(2, 0), "f_1", 1, 0.2)
    f2 = create_decoded_frame(generate_textured_pattern(10, 0), "f_2", 2, 1.0)

    report = TemporalMotionAnalyzer.analyze_temporal_motion(f1, prev_frame=f0, next_frame=f2)

    assert report.time_deltas_seconds == [0.2, 0.8]
    assert report.global_velocity_px_per_sec > 0.0


# Test 9: Corrupted Neighbor Handling
def test_corrupted_neighbor_handling():
    f0_corrupt = DecodedFrame(
        frame_id="corrupt_f0",
        frame_index=0,
        timestamp_seconds=0.0,
        width=120,
        height=120,
        channels=3,
        channel_layout="RGB",
        dtype="uint8",
        data=None,
        source_video="test.mp4",
        decode_status=DecodeStatus.CORRUPTED,
    )
    f1 = create_decoded_frame(generate_textured_pattern(3, 0), "f_1", 1, 0.5)
    f2 = create_decoded_frame(generate_textured_pattern(6, 0), "f_2", 2, 1.0)

    # Should gracefully skip corrupted f0 and analyze using f2
    report = TemporalMotionAnalyzer.analyze_temporal_motion(f1, prev_frame=f0_corrupt, next_frame=f2)

    assert len(report.neighbor_frame_ids) == 1
    assert report.neighbor_frame_ids[0] == "f_2"
    assert report.global_velocity_px_per_sec > 0.0


# Test 10: Provenance Preservation & Diagnostic Form
def test_provenance_preservation():
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(2, 0), "f_1", 1, 0.5)

    rep = TemporalMotionAnalyzer.analyze_temporal_motion(f1, prev_frame=f0)

    assert rep.provenance["target_frame_id"] == "f_1"
    assert rep.provenance["target_frame_index"] == 1
    assert rep.provenance["target_timestamp_seconds"] == 0.5
    assert rep.provenance["analysis_method"] == "DenseFarnebackOpticalFlow_v1.0"
    assert "threshold_coherence" in rep.provenance


# Test 11: Deterministic Repeated Analysis
def test_deterministic_repeated_analysis():
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(3, 1), "f_1", 1, 0.5)

    rep1 = TemporalMotionAnalyzer.analyze_temporal_motion(f1, prev_frame=f0)
    rep2 = TemporalMotionAnalyzer.analyze_temporal_motion(f1, prev_frame=f0)

    assert rep1.mean_displacement_pixels == rep2.mean_displacement_pixels
    assert rep1.global_velocity_px_per_sec == rep2.global_velocity_px_per_sec
    assert rep1.motion_category == rep2.motion_category
    assert rep1.motion_blur_indicator == rep2.motion_blur_indicator


# Test 12: Original Frames Unchanged (Immutability)
def test_original_frames_unchanged():
    img0 = generate_textured_pattern(0, 0)
    img1 = generate_textured_pattern(4, 0)
    img0_copy = img0.copy()
    img1_copy = img1.copy()

    f0 = create_decoded_frame(img0, "f_0", 0, 0.0)
    f1 = create_decoded_frame(img1, "f_1", 1, 0.5)

    _ = TemporalMotionAnalyzer.analyze_temporal_motion(f1, prev_frame=f0)

    assert np.array_equal(f0.data, img0_copy)
    assert np.array_equal(f1.data, img1_copy)


# Test 13: High-Motion Sharp Frame Has Low Blur Indicator
def test_high_motion_sharp_frame_has_low_blur_indicator():
    # Sequence with high apparent motion (10 px shift) but pristine sharp texture
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(10, 0), "f_1", 1, 0.5)
    f2 = create_decoded_frame(generate_textured_pattern(20, 0), "f_2", 2, 1.0)

    report = TemporalMotionAnalyzer.analyze_temporal_motion(f1, prev_frame=f0, next_frame=f2)

    # Sharp edges maintained throughout sequence -> relative sharpness ratio is ~1.0 -> sharpness drop ~0.0
    assert report.global_velocity_px_per_sec > 15.0 # High motion
    assert report.motion_blur_indicator < 0.05 # Low blur indicator (< 0.05)


# Test 14: Low-Texture Flat Image Gives INSUFFICIENT_EVIDENCE
def test_low_texture_insufficient_evidence():
    # Completely flat gray image (no texture)
    flat_img = np.full((120, 120, 3), 128, dtype=np.uint8)
    f0 = create_decoded_frame(flat_img, "f_0", 0, 0.0)
    f1 = create_decoded_frame(flat_img, "f_1", 1, 0.5)

    report = TemporalMotionAnalyzer.analyze_temporal_motion(f1, prev_frame=f0)

    assert report.low_texture_indicator is True
    assert report.motion_category == MotionCategory.INSUFFICIENT_EVIDENCE


# Test 15: Heuristic Threshold Configurability
def test_heuristic_threshold_configurability():
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(2, 0), "f_1", 1, 0.5)

    # Standard default config
    rep_default = TemporalMotionAnalyzer.analyze_temporal_motion(f1, prev_frame=f0)

    # Strict config with high displacement lower bound (10.0 px)
    strict_cfg = TemporalMotionConfig(displacement_lower_bound=10.0)
    rep_strict = TemporalMotionAnalyzer.analyze_temporal_motion(f1, prev_frame=f0, config=strict_cfg)

    # Raw measurements are identical
    assert rep_default.mean_displacement_pixels == rep_strict.mean_displacement_pixels
    # But classification changes due to modified threshold
    assert rep_strict.motion_category == MotionCategory.LOW_APPARENT_MOTION
