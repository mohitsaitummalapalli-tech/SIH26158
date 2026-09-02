"""Deterministic unit tests for Phase 2G.1 Coverage-Aware Keyframe Selection.

DISCLAIMER:
ALL FIXTURES IN THIS MODULE ARE SYNTHETIC TEST DATA GENERATED SOLELY FOR
KEYFRAME SELECTION ENGINE AUDITING. THEY DO NOT REPRESENT A REAL DRONE FLIGHT.
"""

import pytest
import numpy as np
import cv2
import json

from src.preprocessing.decoder import DecodedFrame, DecodeStatus
from src.quality import (
    SelectionReason,
    RejectionReason,
    SelectedKeyframe,
    DeprioritizedCandidate,
    KeyframeSelectionConfig,
    KeyframeSelectionResult,
    CoverageAwareKeyframeSelector,
    FrameQualityReport,
    QualityStatus,
    DynamicSceneReport,
    DynamicEvidenceCategory,
    CandidateDynamicRegion,
    RegionMaskReference,
    FrameRedundancyReport,
    FramePairRelation,
)


def create_decoded_frame(
    data: np.ndarray,
    frame_id: str = "f_0001",
    frame_index: int = 1,
    timestamp: float = 0.5,
    decode_status: DecodeStatus = DecodeStatus.SUCCESS,
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
        source_video="synthetic_keyframe_test.mp4",
        decode_status=decode_status,
        decoder_backend="TestSyntheticBackend",
    )


def generate_pattern(val: int = 100) -> np.ndarray:
    """Generate synthetic image buffer (TEST DATA)."""
    img = np.full((100, 100, 3), val, dtype=np.uint8)
    img[20:40, 20:40] = 250
    return img


# 1. Selector Contract & JSON Serialization
def test_selector_contract_and_serialization():
    frames = [
        create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0),
        create_decoded_frame(generate_pattern(120), "f_1", 1, 0.5),
        create_decoded_frame(generate_pattern(140), "f_2", 2, 1.0),
    ]

    result = CoverageAwareKeyframeSelector.select_keyframes(frames)

    assert isinstance(result, KeyframeSelectionResult)
    assert result.total_input_frames == 3
    assert result.selected_count > 0
    assert len(result.selected_keyframe_ids) == result.selected_count
    assert isinstance(result.selected_keyframes[0], SelectedKeyframe)
    assert result.selected_keyframes[0].primary_reason in list(SelectionReason)

    # JSON Serialization
    json_str = result.to_json(indent=2)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["selected_count"] == result.selected_count
    assert len(parsed["selected_keyframes"]) == result.selected_count


# 2. Hard Safety Gate (Decode Failure & Severe Degradation Exclusions)
def test_hard_safety_gate_exclusions():
    f0 = create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0)
    f1_corrupt = create_decoded_frame(generate_pattern(100), "f_1", 1, 0.5, decode_status=DecodeStatus.CORRUPTED)
    f2 = create_decoded_frame(generate_pattern(120), "f_2", 2, 1.0)

    # Mock quality report with SEVERELY_DEGRADED for f2
    q_rep_f2 = FrameQualityReport(
        frame_id="f_2",
        frame_index=2,
        timestamp_seconds=1.0,
        source_video="test.mp4",
        status=QualityStatus.SEVERELY_DEGRADED,
        luminance=None,
        clipping=None,
        contrast=None,
        sharpness=None,
        spatial_tiles=[],
    )
    f3 = create_decoded_frame(generate_pattern(140), "f_3", 3, 1.5)

    result = CoverageAwareKeyframeSelector.select_keyframes(
        frames=[f0, f1_corrupt, f2, f3],
        quality_reports={"f_2": q_rep_f2},
    )

    dep_ids = [d.frame_id for d in result.deprioritized_candidates]
    assert "f_1" in dep_ids
    assert "f_2" in dep_ids
    assert "f_0" in result.selected_keyframe_ids
    assert "f_3" in result.selected_keyframe_ids


# 3. Temporal Coverage (Anchoring First & Last Frames)
def test_temporal_coverage_boundary_anchors():
    frames = [
        create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0),
        create_decoded_frame(generate_pattern(110), "f_1", 1, 0.4),
        create_decoded_frame(generate_pattern(120), "f_2", 2, 0.8),
        create_decoded_frame(generate_pattern(130), "f_3", 3, 1.2),
    ]

    result = CoverageAwareKeyframeSelector.select_keyframes(frames)

    assert result.selected_keyframe_ids[0] == "f_0"
    assert result.selected_keyframes[0].primary_reason == SelectionReason.INITIAL_ANCHOR
    assert result.selected_keyframe_ids[-1] == "f_3"
    assert result.selected_keyframes[-1].primary_reason == SelectionReason.FINAL_ANCHOR


# 4. Minimum Temporal Spacing Enforcement
def test_minimum_temporal_spacing():
    # Frames arriving at 0.0s, 0.1s, 0.2s, 0.6s (min spacing = 0.25s)
    frames = [
        create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0),
        create_decoded_frame(generate_pattern(105), "f_1", 1, 0.1), # dt=0.1s < 0.25s
        create_decoded_frame(generate_pattern(110), "f_2", 2, 0.2), # dt=0.2s < 0.25s
        create_decoded_frame(generate_pattern(130), "f_3", 3, 0.6), # dt=0.6s >= 0.25s
    ]

    cfg = KeyframeSelectionConfig(min_temporal_spacing_seconds=0.25)
    result = CoverageAwareKeyframeSelector.select_keyframes(frames, config=cfg)

    dep_ids = [d.frame_id for d in result.deprioritized_candidates if d.rejection_reason == RejectionReason.BELOW_MIN_TEMPORAL_SPACING]
    assert "f_1" in dep_ids
    assert "f_2" in dep_ids
    assert "f_3" in result.selected_keyframe_ids


# 5. Maximum Temporal Gap Constraint Enforcement
def test_maximum_temporal_gap_forced_selection():
    # Gap of 3.0s between f0 (0.0s) and f1 (3.0s), max_gap = 2.5s
    frames = [
        create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0),
        create_decoded_frame(generate_pattern(100), "f_1", 1, 3.0), # identical texture but gap >= 2.5s
    ]

    cfg = KeyframeSelectionConfig(max_temporal_gap_seconds=2.5)
    result = CoverageAwareKeyframeSelector.select_keyframes(frames, config=cfg)

    # f1 must be selected due to TEMPORAL_GAP_COVERAGE
    f1_key = next(k for k in result.selected_keyframes if k.frame_id == "f_1")
    assert f1_key.primary_reason in {SelectionReason.TEMPORAL_GAP_COVERAGE, SelectionReason.FINAL_ANCHOR}


# 6. Feature Novelty Contribution
def test_feature_novelty_selection():
    f0 = create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_pattern(150), "f_1", 1, 0.6)

    # Mock redundancy report with low match ratio (high novelty)
    red_rep = FrameRedundancyReport(
        frame_id="f_1",
        frame_index=1,
        timestamp_seconds=0.6,
        source_video="test.mp4",
        pair_relations=[
            FramePairRelation(
                frame_a_id="f_1",
                frame_b_id="f_0",
                frame_a_timestamp=0.6,
                frame_b_timestamp=0.0,
                delta_t_seconds=0.6,
                visual_similarity_score=0.4,
                keypoints_a_count=100,
                keypoints_b_count=100,
                descriptor_match_count=20,
                match_ratio=0.20, # Low match ratio -> High novelty
                spatial_coverage_convex_hull_ratio=0.5,
                match_grid_occupancy_ratio=0.6,
                trajectory_baseline_meters=None,
                trajectory_orientation_change_degrees=None,
                matches_inside_dynamic_regions_fraction=None,
                high_visual_similarity_indicator=False,
                low_feature_novelty_indicator=False,
                low_trajectory_baseline_indicator=False,
                low_orientation_change_indicator=False,
                high_temporal_redundancy_indicator=False,
            )
        ],
        mean_neighbor_match_ratio=0.20,
        mean_neighbor_baseline_meters=None,
    )

    result = CoverageAwareKeyframeSelector.select_keyframes(
        frames=[f0, f1],
        redundancy_reports={"f_1": red_rep},
    )

    f1_key = next(k for k in result.selected_keyframes if k.frame_id == "f_1")
    assert f1_key.marginal_gain_breakdown["feature_novelty_gain"] > 0.70


# 7. Spatial / Feature Coverage Contribution
def test_spatial_coverage_contribution():
    f0 = create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_pattern(120), "f_1", 1, 0.6)

    result = CoverageAwareKeyframeSelector.select_keyframes(frames=[f0, f1])
    assert len(result.selected_keyframes) == 2


# 8. Trajectory Diversity Contribution (ENU Distance Proxy)
def test_trajectory_diversity_contribution():
    f0 = create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_pattern(100), "f_1", 1, 0.5)

    enu_map = {
        "f_0": (10.0, 20.0, 50.0),
        "f_1": (15.0, 20.0, 50.0), # 5.0m baseline
    }

    result = CoverageAwareKeyframeSelector.select_keyframes(
        frames=[f0, f1],
        enu_positions=enu_map,
    )

    f1_key = next(k for k in result.selected_keyframes if k.frame_id == "f_1")
    assert f1_key.marginal_gain_breakdown["trajectory_diversity_gain"] > 0.80


# 9. Orientation Proxy Contribution (Attitude Quaternion)
def test_orientation_proxy_contribution():
    f0 = create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_pattern(100), "f_1", 1, 0.5)

    quat_map = {
        "f_0": (1.0, 0.0, 0.0, 0.0),
        "f_1": (float(np.sqrt(0.5)), 0.0, 0.0, float(np.sqrt(0.5))), # 90 deg rotation
    }

    result = CoverageAwareKeyframeSelector.select_keyframes(
        frames=[f0, f1],
        quaternions=quat_map,
    )

    f1_key = next(k for k in result.selected_keyframes if k.frame_id == "f_1")
    assert f1_key.marginal_gain_breakdown["orientation_diversity_gain"] > 0.80


# 10. Dynamic-Risk Integration (Extreme Contamination Exclusion)
def test_dynamic_risk_integration():
    f0 = create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_pattern(120), "f_1", 1, 0.5)
    f2 = create_decoded_frame(generate_pattern(140), "f_2", 2, 1.0)

    # f1 has 80% dynamic area (> 60% threshold)
    dyn_f1 = DynamicSceneReport(
        frame_id="f_1",
        frame_index=1,
        timestamp_seconds=0.5,
        source_video="test.mp4",
        candidate_regions=[],
        global_motion_velocity_px_per_sec=0.0,
        dominant_motion_vector=(0.0, 0.0),
        overall_scene_status=DynamicEvidenceCategory.DYNAMIC_EVIDENCE,
        static_scene_fraction=0.20, # 80% dynamic
    )

    cfg = KeyframeSelectionConfig(max_dynamic_contamination_fraction=0.60)
    result = CoverageAwareKeyframeSelector.select_keyframes(
        frames=[f0, f1, f2],
        dynamic_reports={"f_1": dyn_f1},
        config=cfg,
    )

    dep_ids = [d.frame_id for d in result.deprioritized_candidates if d.rejection_reason == RejectionReason.DYNAMIC_RISK_EXCLUSION]
    assert "f_1" in dep_ids
    assert "f_1" not in result.selected_keyframe_ids


# 11. Budget Enforcement (Target & Max Count)
def test_budget_enforcement():
    frames = [
        create_decoded_frame(generate_pattern(100 + i*5), f"f_{i}", i, float(i)*0.5)
        for i in range(10)
    ]

    cfg = KeyframeSelectionConfig(max_keyframe_count=3)
    result = CoverageAwareKeyframeSelector.select_keyframes(frames, config=cfg)

    assert result.selected_count <= 3
    assert len(result.selected_keyframe_ids) <= 3


# 12. Greedy Marginal Selection Determinism
def test_greedy_marginal_selection_determinism():
    frames = [
        create_decoded_frame(generate_pattern(100 + i*5), f"f_{i}", i, float(i)*0.4)
        for i in range(6)
    ]

    res1 = CoverageAwareKeyframeSelector.select_keyframes(frames)
    res2 = CoverageAwareKeyframeSelector.select_keyframes(frames)

    assert res1.selected_keyframe_ids == res2.selected_keyframe_ids
    assert res1.selected_timestamps_seconds == res2.selected_timestamps_seconds


# 13. Fallback Behavior (All Candidates Fail Gate)
def test_fallback_behavior():
    # Both frames have corrupted decode status
    f0 = create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0, decode_status=DecodeStatus.CORRUPTED)
    f1 = create_decoded_frame(generate_pattern(120), "f_1", 1, 0.5, decode_status=DecodeStatus.CORRUPTED)

    cfg = KeyframeSelectionConfig(enable_fallback_on_empty=True)
    result = CoverageAwareKeyframeSelector.select_keyframes([f0, f1], config=cfg)

    assert result.fallback_used is True
    assert result.selected_count > 0
    assert result.selected_keyframes[0].primary_reason == SelectionReason.FALLBACK


# 14. Selection Explanations Preservation
def test_selection_explanation_preservation():
    frames = [
        create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0),
        create_decoded_frame(generate_pattern(150), "f_1", 1, 0.5),
    ]

    result = CoverageAwareKeyframeSelector.select_keyframes(frames)

    for kf in result.selected_keyframes:
        assert isinstance(kf.primary_reason, SelectionReason)
        assert len(kf.detailed_reasons) > 0
        assert "temporal" in kf.marginal_gain_breakdown


# 15. Provenance Preservation
def test_provenance_preservation():
    frames = [
        create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0),
        create_decoded_frame(generate_pattern(120), "f_1", 1, 0.5),
    ]

    result = CoverageAwareKeyframeSelector.select_keyframes(frames)

    assert result.provenance["total_input_frames"] == 2
    assert result.provenance["selection_algorithm"] == "GreedyMarginalGain_v1.1"
    assert "min_spacing_seconds" in result.provenance


# 16. Original Frame / Video Immutability
def test_original_frame_immutability():
    img0 = generate_pattern(100)
    img1 = generate_pattern(120)
    copy0 = img0.copy()
    copy1 = img1.copy()

    f0 = create_decoded_frame(img0, "f_0", 0, 0.0)
    f1 = create_decoded_frame(img1, "f_1", 1, 0.5)

    _ = CoverageAwareKeyframeSelector.select_keyframes([f0, f1])

    assert np.array_equal(f0.data, copy0)
    assert np.array_equal(f1.data, copy1)


# 17. Repeated-Run Determinism
def test_repeated_run_determinism():
    frames = [
        create_decoded_frame(generate_pattern(50 + i*15), f"f_{i}", i, float(i)*0.5)
        for i in range(5)
    ]

    res1 = CoverageAwareKeyframeSelector.select_keyframes(frames)
    res2 = CoverageAwareKeyframeSelector.select_keyframes(frames)

    assert res1.selected_keyframe_ids == res2.selected_keyframe_ids
    assert res1.reduction_ratio == res2.reduction_ratio


# 18. Redundant Candidate Deprioritization
def test_redundant_candidate_deprioritization():
    # 5 identical frames at 0.0s, 0.3s, 0.6s, 0.9s, 1.2s without spatial motion
    frames = [
        create_decoded_frame(generate_pattern(100), f"f_{i}", i, float(i)*0.3)
        for i in range(5)
    ]

    result = CoverageAwareKeyframeSelector.select_keyframes(frames)
    assert result.selected_count < len(frames)
    assert len(result.deprioritized_candidates) > 0


# 19. Conflict-Resolution Behavior
def test_conflict_resolution_behavior():
    # Candidate with high geometric novelty but near temporal spacing limit
    f0 = create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_pattern(200), "f_1", 1, 0.26) # just above 0.25s

    result = CoverageAwareKeyframeSelector.select_keyframes([f0, f1])
    assert result.selected_count == 2


# 20. Unsafe Initial Boundary Anchor Replacement (INITIAL_BOUNDARY_FALLBACK)
def test_unsafe_initial_boundary_anchor_replacement():
    # Frame 0 is corrupt, Frame 1 is safe
    f0_bad = create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0, decode_status=DecodeStatus.CORRUPTED)
    f1_safe = create_decoded_frame(generate_pattern(120), "f_1", 1, 0.4)
    f2_safe = create_decoded_frame(generate_pattern(140), "f_2", 2, 0.8)

    result = CoverageAwareKeyframeSelector.select_keyframes([f0_bad, f1_safe, f2_safe])

    assert result.selected_keyframe_ids[0] == "f_1"
    assert result.selected_keyframes[0].primary_reason == SelectionReason.INITIAL_BOUNDARY_FALLBACK
    assert "First frame was unsafe" in result.selected_keyframes[0].detailed_reasons[0]


# 21. Unsafe Final Boundary Anchor Replacement (FINAL_BOUNDARY_FALLBACK)
def test_unsafe_final_boundary_anchor_replacement():
    # Frame 2 is corrupt, Frame 1 is safe
    f0_safe = create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0)
    f1_safe = create_decoded_frame(generate_pattern(120), "f_1", 1, 0.5)
    f2_bad = create_decoded_frame(generate_pattern(140), "f_2", 2, 1.0, decode_status=DecodeStatus.CORRUPTED)

    result = CoverageAwareKeyframeSelector.select_keyframes([f0_safe, f1_safe, f2_bad])

    assert result.selected_keyframe_ids[-1] == "f_1"
    assert result.selected_keyframes[-1].primary_reason == SelectionReason.FINAL_BOUNDARY_FALLBACK


# 22. Uncovered Boundary Reporting (BOUNDARY_UNCOVERED)
def test_uncovered_boundary_reporting():
    f0_bad = create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0, decode_status=DecodeStatus.CORRUPTED)
    f1_bad = create_decoded_frame(generate_pattern(120), "f_1", 1, 0.5, decode_status=DecodeStatus.CORRUPTED)

    cfg = KeyframeSelectionConfig(enable_fallback_on_empty=False)
    result = CoverageAwareKeyframeSelector.select_keyframes([f0_bad, f1_bad], config=cfg)

    assert result.selected_count == 0
    assert any("BOUNDARY_UNCOVERED" in d for d in result.diagnostics)


# 23. Speed-Aware Trajectory Displacement Diagnostic
def test_speed_aware_trajectory_displacement():
    f0 = create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_pattern(120), "f_1", 1, 0.5) # dt = 0.5s

    # 10.0 m/s ground speed -> expected displacement = 5.0m
    ground_speeds = {"f_0": 10.0, "f_1": 10.0}
    enu_map = {"f_0": (0.0, 0.0, 50.0), "f_1": (5.0, 0.0, 50.0)}

    result = CoverageAwareKeyframeSelector.select_keyframes(
        [f0, f1],
        ground_speeds=ground_speeds,
        enu_positions=enu_map,
    )

    f1_key = next(k for k in result.selected_keyframes if k.frame_id == "f_1")
    assert f1_key.expected_trajectory_displacement_meters == 5.0


# 24. Heuristic Threshold Labeling
def test_heuristic_threshold_labeling():
    cfg = KeyframeSelectionConfig()
    result = CoverageAwareKeyframeSelector.select_keyframes(
        [create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0)],
        config=cfg,
    )
    assert result.provenance.get("heuristic_defaults_active") is True


# 25. Quality vs Coverage Conflict (Novel Candidate Preferred over Sharp Redundant)
def test_quality_vs_novelty_conflict():
    f0 = create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0)
    f1_sharp_redundant = create_decoded_frame(generate_pattern(100), "f_1", 1, 0.3)
    f2_mod_novel = create_decoded_frame(generate_pattern(200), "f_2", 2, 0.6)

    # f2 has high feature novelty
    red_rep_f2 = FrameRedundancyReport(
        frame_id="f_2",
        frame_index=2,
        timestamp_seconds=0.6,
        source_video="test.mp4",
        pair_relations=[
            FramePairRelation(
                frame_a_id="f_2",
                frame_b_id="f_0",
                frame_a_timestamp=0.6,
                frame_b_timestamp=0.0,
                delta_t_seconds=0.6,
                visual_similarity_score=0.3,
                keypoints_a_count=100,
                keypoints_b_count=100,
                descriptor_match_count=10,
                match_ratio=0.10, # Very high novelty
                spatial_coverage_convex_hull_ratio=0.5,
                match_grid_occupancy_ratio=0.6,
                trajectory_baseline_meters=None,
                trajectory_orientation_change_degrees=None,
                matches_inside_dynamic_regions_fraction=None,
                high_visual_similarity_indicator=False,
                low_feature_novelty_indicator=False,
                low_trajectory_baseline_indicator=False,
                low_orientation_change_indicator=False,
                high_temporal_redundancy_indicator=False,
            )
        ],
        mean_neighbor_match_ratio=0.10,
        mean_neighbor_baseline_meters=None,
    )

    result = CoverageAwareKeyframeSelector.select_keyframes(
        [f0, f1_sharp_redundant, f2_mod_novel],
        redundancy_reports={"f_2": red_rep_f2},
    )

    assert "f_2" in result.selected_keyframe_ids
    # f1 was deprioritized due to redundancy despite being sharp
    dep_ids = [d.frame_id for d in result.deprioritized_candidates]
    assert "f_1" in dep_ids


# 26. Dynamic Area vs Static Feature Match Conflict
def test_dynamic_area_vs_static_feature_conflict():
    f0 = create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0)
    f1_dyn_area = create_decoded_frame(generate_pattern(150), "f_1", 1, 0.5)

    # 70% dynamic candidate area (exceeds 60% threshold)
    dyn_rep = DynamicSceneReport(
        frame_id="f_1",
        frame_index=1,
        timestamp_seconds=0.5,
        source_video="test.mp4",
        candidate_regions=[],
        global_motion_velocity_px_per_sec=0.0,
        dominant_motion_vector=(0.0, 0.0),
        overall_scene_status=DynamicEvidenceCategory.POSSIBLY_DYNAMIC,
        static_scene_fraction=0.30, # 70% dynamic area
    )

    # BUT feature matches show that only 5% of correspondences are inside dynamic region
    red_rep = FrameRedundancyReport(
        frame_id="f_1",
        frame_index=1,
        timestamp_seconds=0.5,
        source_video="test.mp4",
        pair_relations=[
            FramePairRelation(
                frame_a_id="f_1",
                frame_b_id="f_0",
                frame_a_timestamp=0.5,
                frame_b_timestamp=0.0,
                delta_t_seconds=0.5,
                visual_similarity_score=0.5,
                keypoints_a_count=100,
                keypoints_b_count=100,
                descriptor_match_count=50,
                match_ratio=0.50,
                spatial_coverage_convex_hull_ratio=0.5,
                match_grid_occupancy_ratio=0.6,
                trajectory_baseline_meters=None,
                trajectory_orientation_change_degrees=None,
                matches_inside_dynamic_regions_fraction=0.05, # only 5% dynamic
                high_visual_similarity_indicator=False,
                low_feature_novelty_indicator=False,
                low_trajectory_baseline_indicator=False,
                low_orientation_change_indicator=False,
                high_temporal_redundancy_indicator=False,
            )
        ],
        mean_neighbor_match_ratio=0.50,
        mean_neighbor_baseline_meters=None,
    )

    result = CoverageAwareKeyframeSelector.select_keyframes(
        [f0, f1_dyn_area],
        dynamic_reports={"f_1": dyn_rep},
        redundancy_reports={"f_1": red_rep},
    )

    # Frame is retained with a diagnostic warning
    assert "f_1" in result.selected_keyframe_ids
    assert any("Dynamic candidate area is high" in d for d in result.diagnostics)


# 27. Incompatible Budget and Coverage Constraints Diagnostic
def test_incompatible_budget_coverage_constraints():
    # 10 frames over 10.0s, max_temporal_gap = 2.0s -> requires >= 6 keyframes. Budget = 3
    frames = [
        create_decoded_frame(generate_pattern(100 + i*10), f"f_{i}", i, float(i)*1.0)
        for i in range(11) # 0.0s to 10.0s
    ]

    cfg = KeyframeSelectionConfig(max_keyframe_count=3, max_temporal_gap_seconds=2.0)
    result = CoverageAwareKeyframeSelector.select_keyframes(frames, config=cfg)

    assert result.selected_count <= 3
    assert any("BUDGET_INCOMPATIBLE_WITH_COVERAGE_CONSTRAINT" in d for d in result.diagnostics)


# 28. Deterministic Conflicting Constraint Resolution
def test_deterministic_conflicting_constraint_resolution():
    frames = [
        create_decoded_frame(generate_pattern(100 + i*20), f"f_{i}", i, float(i)*0.25)
        for i in range(8)
    ]

    res1 = CoverageAwareKeyframeSelector.select_keyframes(frames)
    res2 = CoverageAwareKeyframeSelector.select_keyframes(frames)

    assert res1.selected_keyframe_ids == res2.selected_keyframe_ids
    assert res1.selected_count == res2.selected_count


# 29. Complete Selection Explanation Preserved
def test_complete_selection_explanation():
    f0 = create_decoded_frame(generate_pattern(100), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_pattern(120), "f_1", 1, 0.5)

    result = CoverageAwareKeyframeSelector.select_keyframes([f0, f1])

    for kf in result.selected_keyframes:
        assert kf.greedy_heuristic_gain is not None
        assert isinstance(kf.marginal_gain_breakdown, dict)
        assert len(kf.detailed_reasons) > 0
