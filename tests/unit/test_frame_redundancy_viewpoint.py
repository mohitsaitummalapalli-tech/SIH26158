"""Deterministic unit tests for Phase 2F.1 Frame Redundancy & Viewpoint Diversity Diagnostics.

DISCLAIMER:
ALL FIXTURES IN THIS MODULE ARE SYNTHETIC TEST DATA GENERATED SOLELY FOR
FRAME REDUNDANCY AND VIEWPOINT DIVERSITY DIAGNOSTIC ENGINE AUDITING. THEY DO NOT REPRESENT A REAL DRONE FLIGHT.
"""

import pytest
import numpy as np
import cv2
import json

from src.preprocessing.decoder import DecodedFrame, DecodeStatus
from src.quality import (
    FrameRedundancyConfig,
    FramePairRelation,
    FrameRedundancyReport,
    FrameRedundancyViewpointAnalyzer,
    DynamicSceneReport,
    CandidateDynamicRegion,
    RegionMaskReference,
    DynamicEvidenceCategory,
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
        source_video="synthetic_redundancy_test.mp4",
        decode_status=DecodeStatus.SUCCESS,
        decoder_backend="TestSyntheticBackend",
    )


def generate_rich_texture(shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    """Generate deterministic rich synthetic texture pattern with corners/edges (TEST DATA)."""
    img = np.zeros((140, 140, 3), dtype=np.uint8)
    for y in range(0, 140, 14):
        for x in range(0, 140, 14):
            val = ((x * 17 + y * 23) % 255)
            img[y:y+14, x:x+14] = (val, (val * 2) % 255, (val * 3) % 255)
            # Add small high-contrast corner dot
            img[y+4:y+8, x+4:x+8] = 255

    if shift_x != 0 or shift_y != 0:
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, M, (140, 140), borderMode=cv2.BORDER_REFLECT)
    return img


# 1. Frame-Pair Contract & JSON Serialization
def test_frame_pair_contract_and_serialization():
    img0 = generate_rich_texture(0, 0)
    img1 = generate_rich_texture(2, 0)
    f0 = create_decoded_frame(img0, "f_0", 0, 0.0)
    f1 = create_decoded_frame(img1, "f_1", 1, 0.5)

    report = FrameRedundancyViewpointAnalyzer.analyze_frame_redundancy(
        target_frame=f1,
        prior_frames=[f0],
        target_enu_pos=(10.0, 20.0, 50.0),
        prior_enu_positions=[(10.0, 19.5, 50.0)],
        target_quat=(1.0, 0.0, 0.0, 0.0),
        prior_quats=[(1.0, 0.0, 0.0, 0.0)],
    )

    assert isinstance(report, FrameRedundancyReport)
    assert report.frame_id == "f_1"
    assert len(report.pair_relations) == 1

    rel = report.pair_relations[0]
    assert rel.frame_a_id == "f_1"
    assert rel.frame_b_id == "f_0"
    assert isinstance(rel.visual_similarity_score, float)
    assert isinstance(rel.match_ratio, float)

    # JSON serialization
    json_str = report.to_json(indent=2)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["frame_id"] == "f_1"
    assert len(parsed["pair_relations"]) == 1


# 2. Timestamp Separation (Canonical delta_t)
def test_timestamp_separation():
    img0 = generate_rich_texture(0, 0)
    f0 = create_decoded_frame(img0, "f_0", 0, 1.25)
    f1 = create_decoded_frame(img0, "f_1", 1, 2.75) # dt = 1.50s

    rel = FrameRedundancyViewpointAnalyzer.evaluate_pair_relation(f1, f0)
    assert rel.delta_t_seconds == pytest.approx(1.50, abs=1e-3)
    assert rel.high_temporal_redundancy_indicator is False # dt > 0.5s


# 3. Identical Frame Similarity (ZNCC = 1.0)
def test_identical_frame_similarity():
    img0 = generate_rich_texture(0, 0)
    f0 = create_decoded_frame(img0, "f_0", 0, 0.0)
    f1 = create_decoded_frame(img0, "f_1", 1, 0.1)

    rel = FrameRedundancyViewpointAnalyzer.evaluate_pair_relation(f1, f0)
    assert rel.visual_similarity_score == pytest.approx(1.0, abs=1e-2)
    assert rel.match_ratio > 0.90
    assert rel.high_visual_similarity_indicator is True
    assert rel.low_feature_novelty_indicator is True


# 4. Visual Similarity Change (Appearance Shift)
def test_visual_similarity_change():
    img0 = generate_rich_texture(0, 0)
    # Drastic translation + color shift
    img1 = generate_rich_texture(50, 40)
    f0 = create_decoded_frame(img0, "f_0", 0, 0.0)
    f1 = create_decoded_frame(img1, "f_1", 1, 0.5)

    rel = FrameRedundancyViewpointAnalyzer.evaluate_pair_relation(f1, f0)
    assert rel.visual_similarity_score < 0.85
    assert rel.high_visual_similarity_indicator is False


# 5. Feature Extraction
def test_feature_extraction():
    img = generate_rich_texture(0, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    kps, desc = FrameRedundancyViewpointAnalyzer.extract_orb_features(gray, max_features=200)
    assert len(kps) > 20
    assert desc is not None
    assert desc.shape[0] == len(kps)
    assert desc.shape[1] == 32 # 256-bit binary descriptors


# 6. Feature Matching & Match Ratio
def test_feature_matching_and_match_ratio():
    img0 = generate_rich_texture(0, 0)
    img1 = generate_rich_texture(4, 2)
    f0 = create_decoded_frame(img0, "f_0", 0, 0.0)
    f1 = create_decoded_frame(img1, "f_1", 1, 0.5)

    rel = FrameRedundancyViewpointAnalyzer.evaluate_pair_relation(f1, f0)
    assert rel.descriptor_match_count > 15
    assert 0.0 < rel.match_ratio <= 1.0


# 7. Spatial Match Distribution (Convex Hull & Grid Occupancy)
def test_spatial_match_distribution():
    img0 = generate_rich_texture(0, 0)
    img1 = generate_rich_texture(2, 0)
    f0 = create_decoded_frame(img0, "f_0", 0, 0.0)
    f1 = create_decoded_frame(img1, "f_1", 1, 0.5)

    rel = FrameRedundancyViewpointAnalyzer.evaluate_pair_relation(f1, f0)
    # Rich texture across full canvas -> convex hull coverage > 35% and high grid occupancy (> 70%)
    assert rel.spatial_coverage_convex_hull_ratio > 0.35
    assert rel.match_grid_occupancy_ratio > 0.70


# 8. Low-Texture Insufficient Evidence
def test_low_texture_insufficient_evidence():
    # Completely flat gray canvas (no corners/edges)
    flat_img = np.full((140, 140, 3), 128, dtype=np.uint8)
    f0 = create_decoded_frame(flat_img, "f_0", 0, 0.0)
    f1 = create_decoded_frame(flat_img, "f_1", 1, 0.5)

    rel = FrameRedundancyViewpointAnalyzer.evaluate_pair_relation(f1, f0)
    assert rel.keypoints_a_count < 15
    assert any("INSUFFICIENT_FEATURE_EVIDENCE" in d for d in rel.diagnostics)


# 9. Trajectory Baseline Calculation (ENU Distance)
def test_trajectory_baseline_calculation():
    img = generate_rich_texture(0, 0)
    f0 = create_decoded_frame(img, "f_0", 0, 0.0)
    f1 = create_decoded_frame(img, "f_1", 1, 0.5)

    # 3-4-5 triangle: delta_x = 3.0m, delta_y = 4.0m -> distance = 5.0m
    enu_a = (103.0, 204.0, 50.0)
    enu_b = (100.0, 200.0, 50.0)

    rel = FrameRedundancyViewpointAnalyzer.evaluate_pair_relation(f1, f0, enu_pos_a=enu_a, enu_pos_b=enu_b)
    assert rel.trajectory_baseline_meters == pytest.approx(5.0, abs=1e-3)
    assert rel.low_trajectory_baseline_indicator is False # baseline 5.0m > 1.0m threshold


# 10. Orientation Change (Quaternion Geodesic Angle)
def test_orientation_change_calculation():
    img = generate_rich_texture(0, 0)
    f0 = create_decoded_frame(img, "f_0", 0, 0.0)
    f1 = create_decoded_frame(img, "f_1", 1, 0.5)

    # 90-degree yaw rotation: q = (cos(pi/4), 0, 0, sin(pi/4)) = (sqrt(0.5), 0, 0, sqrt(0.5))
    q_a = (1.0, 0.0, 0.0, 0.0)
    q_b = (float(np.sqrt(0.5)), 0.0, 0.0, float(np.sqrt(0.5)))

    rel = FrameRedundancyViewpointAnalyzer.evaluate_pair_relation(f1, f0, quat_a=q_a, quat_b=q_b)
    assert rel.trajectory_orientation_change_degrees == pytest.approx(90.0, abs=1e-2)
    assert rel.low_orientation_change_indicator is False


# 11. Dynamic Region Integration
def test_dynamic_region_integration():
    img0 = generate_rich_texture(0, 0)
    img1 = generate_rich_texture(2, 0)
    f0 = create_decoded_frame(img0, "f_0", 0, 0.0)
    f1 = create_decoded_frame(img1, "f_1", 1, 0.5)

    # Synthetic dynamic report placing a dynamic box covering top-left quadrant
    dyn_report = DynamicSceneReport(
        frame_id="f_1",
        frame_index=1,
        timestamp_seconds=0.5,
        source_video="test.mp4",
        candidate_regions=[
            CandidateDynamicRegion(
                region_id="reg_0",
                bbox=(0, 0, 70, 70),
                mask_ref=RegionMaskReference(mask_type="BBOX_ONLY"),
                semantic_label="vehicle",
                semantic_confidence=0.9,
                provider_name="TestProvider",
                local_velocity_px_per_sec=10.0,
                relative_motion_discrepancy=8.0,
                temporal_persistence_count=3,
                dynamic_evidence_score=0.8,
                evidence_category=DynamicEvidenceCategory.DYNAMIC_EVIDENCE,
            )
        ],
        global_motion_velocity_px_per_sec=2.0,
        dominant_motion_vector=(2.0, 0.0),
        overall_scene_status=DynamicEvidenceCategory.DYNAMIC_EVIDENCE,
        static_scene_fraction=0.75,
    )

    rel = FrameRedundancyViewpointAnalyzer.evaluate_pair_relation(f1, f0, dynamic_report_a=dyn_report)
    assert rel.matches_inside_dynamic_regions_fraction is not None
    assert 0.0 <= rel.matches_inside_dynamic_regions_fraction <= 1.0


# 12. Provenance Preservation
def test_provenance_preservation():
    img = generate_rich_texture(0, 0)
    f0 = create_decoded_frame(img, "f_0", 0, 0.0)
    f1 = create_decoded_frame(img, "f_1", 1, 0.5)

    report = FrameRedundancyViewpointAnalyzer.analyze_frame_redundancy(f1, prior_frames=[f0])
    assert report.provenance["target_frame_id"] == "f_1"
    assert report.provenance["evaluated_pairs_count"] == 1
    assert report.provenance["feature_detector"] == "OpenCV_ORB_v1.0"


# 13. Deterministic Repeated Analysis
def test_deterministic_repeated_analysis():
    img0 = generate_rich_texture(0, 0)
    img1 = generate_rich_texture(3, 1)
    f0 = create_decoded_frame(img0, "f_0", 0, 0.0)
    f1 = create_decoded_frame(img1, "f_1", 1, 0.5)

    rep1 = FrameRedundancyViewpointAnalyzer.analyze_frame_redundancy(f1, prior_frames=[f0])
    rep2 = FrameRedundancyViewpointAnalyzer.analyze_frame_redundancy(f1, prior_frames=[f0])

    assert rep1.mean_neighbor_match_ratio == rep2.mean_neighbor_match_ratio
    assert rep1.pair_relations[0].visual_similarity_score == rep2.pair_relations[0].visual_similarity_score


# 14. Frame Immutability (Source Frames Untouched)
def test_frame_immutability():
    img0 = generate_rich_texture(0, 0)
    img1 = generate_rich_texture(4, 0)
    copy0 = img0.copy()
    copy1 = img1.copy()

    f0 = create_decoded_frame(img0, "f_0", 0, 0.0)
    f1 = create_decoded_frame(img1, "f_1", 1, 0.5)

    _ = FrameRedundancyViewpointAnalyzer.analyze_frame_redundancy(f1, prior_frames=[f0])

    assert np.array_equal(f0.data, copy0)
    assert np.array_equal(f1.data, copy1)


# 15. No Final Selection Decision (Diagnostic Separation)
def test_no_final_selection_decision():
    img0 = generate_rich_texture(0, 0)
    img1 = generate_rich_texture(2, 0)
    f0 = create_decoded_frame(img0, "f_0", 0, 0.0)
    f1 = create_decoded_frame(img1, "f_1", 1, 0.5)

    report = FrameRedundancyViewpointAnalyzer.analyze_frame_redundancy(f1, prior_frames=[f0])

    # Asserts report contains all raw evidence metrics and indicators separately without dropping frames
    assert hasattr(report, "pair_relations")
    rel = report.pair_relations[0]
    assert hasattr(rel, "high_visual_similarity_indicator")
    assert hasattr(rel, "low_feature_novelty_indicator")
    assert hasattr(rel, "low_trajectory_baseline_indicator")
    assert hasattr(rel, "low_orientation_change_indicator")
    assert hasattr(rel, "high_temporal_redundancy_indicator")
