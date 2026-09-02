"""Deterministic unit tests for Phase 2E.1 Dynamic Scene Analysis Contract & Temporal Evidence.

DISCLAIMER:
ALL FIXTURES IN THIS MODULE ARE SYNTHETIC TEST DATA GENERATED SOLELY FOR
DYNAMIC SCENE DIAGNOSTIC ENGINE AUDITING. THEY DO NOT REPRESENT A REAL DRONE FLIGHT.
"""

import pytest
from typing import List, Tuple, Optional
import numpy as np
import cv2
import json

from src.preprocessing.decoder import DecodedFrame, DecodeStatus
from src.quality import (
    DynamicEvidenceCategory,
    RegionMaskReference,
    CandidateDynamicRegion,
    DynamicSceneConfig,
    DynamicRegionProvider,
    SyntheticDynamicRegionProvider,
    DynamicSceneReport,
    DynamicSceneAnalyzer,
    TemporalMotionAnalyzer,
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
        source_video="synthetic_dynamic_test.mp4",
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
        M = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
        img = cv2.warpAffine(img, M, (120, 120), borderMode=cv2.BORDER_REFLECT)
    return img


# 1. Report Contract & JSON Serialization
def test_report_contract_and_serialization():
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(2, 0), "f_1", 1, 0.5)

    provider = SyntheticDynamicRegionProvider([
        ((20, 20, 60, 60), "car", 0.92, RegionMaskReference(mask_type="BBOX_ONLY"))
    ])

    report = DynamicSceneAnalyzer.analyze_frame(f1, region_provider=provider, prev_frame=f0)

    assert isinstance(report, DynamicSceneReport)
    assert report.frame_id == "f_1"
    assert report.frame_index == 1
    assert len(report.candidate_regions) == 1
    assert report.candidate_regions[0].semantic_label == "car"
    assert report.candidate_regions[0].evidence_category in list(DynamicEvidenceCategory)

    # Serialization
    json_str = report.to_json(indent=2)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["frame_id"] == "f_1"
    assert len(parsed["candidate_regions"]) == 1


# 2. Detector Provider Interface
def test_detector_provider_interface():
    class CustomMockProvider(DynamicRegionProvider):
        def detect_candidate_regions(
            self, frame: DecodedFrame
        ) -> List[Tuple[Tuple[int, int, int, int], Optional[str], Optional[float], RegionMaskReference]]:
            return [((10, 10, 40, 40), "person", 0.85, RegionMaskReference(mask_type="BBOX_ONLY"))]

    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(0, 0), "f_1", 1, 0.5)

    provider = CustomMockProvider()
    report = DynamicSceneAnalyzer.analyze_frame(f1, region_provider=provider, prev_frame=f0)

    assert len(report.candidate_regions) == 1
    assert report.candidate_regions[0].provider_name == "CustomMockProvider"
    assert report.candidate_regions[0].semantic_label == "person"


# 3. Bounding Box Preservation
def test_bounding_box_preservation():
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(0, 0), "f_1", 1, 0.5)

    bbox_in = (15, 25, 75, 85)
    provider = SyntheticDynamicRegionProvider([
        (bbox_in, "signboard", 0.95, RegionMaskReference(mask_type="BBOX_ONLY"))
    ])

    report = DynamicSceneAnalyzer.analyze_frame(f1, region_provider=provider, prev_frame=f0)
    assert report.candidate_regions[0].bbox == bbox_in


# 4. Mask Reference Preservation
def test_mask_reference_preservation():
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(0, 0), "f_1", 1, 0.5)

    mask_ref = RegionMaskReference(mask_type="RLE", rle_counts="10 5 10 5", mask_uri="masks/f1_reg0.rle")
    provider = SyntheticDynamicRegionProvider([
        ((10, 10, 50, 50), "vehicle", 0.88, mask_ref)
    ])

    report = DynamicSceneAnalyzer.analyze_frame(f1, region_provider=provider, prev_frame=f0)
    assert report.candidate_regions[0].mask_ref.mask_type == "RLE"
    assert report.candidate_regions[0].mask_ref.mask_uri == "masks/f1_reg0.rle"


# 5. Semantic Label Provenance
def test_semantic_label_provenance():
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(0, 0), "f_1", 1, 0.5)

    provider = SyntheticDynamicRegionProvider([
        ((10, 10, 50, 50), "excavator", 0.77, RegionMaskReference(mask_type="BBOX_ONLY"))
    ])

    report = DynamicSceneAnalyzer.analyze_frame(f1, region_provider=provider, prev_frame=f0)
    assert report.candidate_regions[0].semantic_label == "excavator"
    assert report.candidate_regions[0].semantic_confidence == 0.77


# 6. Static Semantic Object Not Automatically Dynamic (Parked Car)
def test_static_semantic_object_not_automatically_dynamic():
    # Entire image is static, but provider returns "car" label
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(0, 0), "f_1", 1, 0.5)

    provider = SyntheticDynamicRegionProvider([
        ((20, 20, 80, 80), "truck", 0.99, RegionMaskReference(mask_type="BBOX_ONLY"))
    ])

    report = DynamicSceneAnalyzer.analyze_frame(f1, region_provider=provider, prev_frame=f0)

    # Even with high semantic confidence, zero motion discrepancy -> STATIC_EVIDENCE
    assert report.candidate_regions[0].evidence_category == DynamicEvidenceCategory.STATIC_EVIDENCE
    assert report.candidate_regions[0].dynamic_evidence_score == pytest.approx(0.0, abs=1e-3)


# 7. Local Motion Evidence (Independent Moving Object)
def test_local_motion_evidence():
    # Background is static, bottom-right patch moves by 12 px
    bg = generate_textured_pattern(0, 0)
    f0_data = bg.copy()
    f1_data = bg.copy()
    patch = bg[80:120, 80:120].copy()
    M = np.array([[1, 0, 12], [0, 1, 0]], dtype=np.float32)
    f1_data[80:120, 80:120] = cv2.warpAffine(patch, M, (40, 40), borderMode=cv2.BORDER_REFLECT)

    f0 = create_decoded_frame(f0_data, "f_0", 0, 0.0)
    f1 = create_decoded_frame(f1_data, "f_1", 1, 0.5)

    # Provider flags the bottom-right moving patch
    provider = SyntheticDynamicRegionProvider([
        ((80, 80, 120, 120), "drone", 0.90, RegionMaskReference(mask_type="BBOX_ONLY"))
    ])

    report = DynamicSceneAnalyzer.analyze_frame(
        f1,
        region_provider=provider,
        prev_frame=f0,
        historical_persistence={"drone_4_4": 3}, # Persistent across 3 frames
    )

    region = report.candidate_regions[0]
    assert region.relative_motion_discrepancy > 4.0
    assert region.evidence_category == DynamicEvidenceCategory.DYNAMIC_EVIDENCE
    assert region.dynamic_evidence_score > 0.40


# 8. Global Coherent Motion Not Automatically Dynamic
def test_global_coherent_motion_not_automatically_dynamic():
    # Entire frame translates coherently by 6 px (camera motion)
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(6, 0), "f_1", 1, 0.5)

    # Provider flags a static building in the center
    provider = SyntheticDynamicRegionProvider([
        ((40, 40, 80, 80), "building", 0.95, RegionMaskReference(mask_type="BBOX_ONLY"))
    ])

    report = DynamicSceneAnalyzer.analyze_frame(f1, region_provider=provider, prev_frame=f0)

    # Building moves with the global flow (zero discrepancy) -> STATIC_EVIDENCE
    region = report.candidate_regions[0]
    assert region.relative_motion_discrepancy < 2.0
    assert region.evidence_category in {DynamicEvidenceCategory.STATIC_EVIDENCE, DynamicEvidenceCategory.POSSIBLY_DYNAMIC}


# 9. Temporal Persistence Representation
def test_temporal_persistence_representation():
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(0, 0), "f_1", 1, 0.5)

    provider = SyntheticDynamicRegionProvider([
        ((10, 10, 50, 50), "car", 0.85, RegionMaskReference(mask_type="BBOX_ONLY"))
    ])

    history = {"car_0_0": 4} # 4 prior consecutive frames
    report = DynamicSceneAnalyzer.analyze_frame(
        f1, region_provider=provider, prev_frame=f0, historical_persistence=history
    )

    assert report.candidate_regions[0].temporal_persistence_count == 5


# 10. Transient Region Handling (Single Frame Anomaly)
def test_transient_region_handling():
    # Moving patch observed for the very first time (persistence count = 1)
    bg = generate_textured_pattern(0, 0)
    f0_data = bg.copy()
    f1_data = bg.copy()
    patch = bg[80:120, 80:120].copy()
    M = np.array([[1, 0, 12], [0, 1, 0]], dtype=np.float32)
    f1_data[80:120, 80:120] = cv2.warpAffine(patch, M, (40, 40), borderMode=cv2.BORDER_REFLECT)

    f0 = create_decoded_frame(f0_data, "f_0", 0, 0.0)
    f1 = create_decoded_frame(f1_data, "f_1", 1, 0.5)

    provider = SyntheticDynamicRegionProvider([
        ((80, 80, 120, 120), "unlabeled", None, RegionMaskReference(mask_type="BBOX_ONLY"))
    ])

    # No history -> single frame appearance
    report = DynamicSceneAnalyzer.analyze_frame(f1, region_provider=provider, prev_frame=f0)

    # Single-frame transient motion is flagged as POSSIBLY_DYNAMIC rather than definitive DYNAMIC_EVIDENCE
    assert report.candidate_regions[0].temporal_persistence_count == 1
    assert report.candidate_regions[0].evidence_category == DynamicEvidenceCategory.POSSIBLY_DYNAMIC


# 11. Missing Provider Output (Empty Candidate List)
def test_missing_provider_output():
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(2, 0), "f_1", 1, 0.5)

    # Provider returns empty candidate list
    provider = SyntheticDynamicRegionProvider([])
    report = DynamicSceneAnalyzer.analyze_frame(f1, region_provider=provider, prev_frame=f0)

    assert len(report.candidate_regions) == 0
    assert report.static_scene_fraction == 1.0
    assert report.overall_scene_status == DynamicEvidenceCategory.STATIC_EVIDENCE


# 12. Missing Neighboring Frame (Boundary Frame)
def test_missing_neighboring_frame():
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)

    provider = SyntheticDynamicRegionProvider([
        ((20, 20, 60, 60), "car", 0.90, RegionMaskReference(mask_type="BBOX_ONLY"))
    ])

    # Single isolated frame
    report = DynamicSceneAnalyzer.analyze_frame(f0, region_provider=provider)

    assert report.overall_scene_status == DynamicEvidenceCategory.INSUFFICIENT_EVIDENCE
    assert report.candidate_regions[0].evidence_category == DynamicEvidenceCategory.INSUFFICIENT_EVIDENCE


# 13. Insufficient Evidence Handling
def test_insufficient_evidence_handling():
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(0, 0), "f_1", 1, 5.0) # dt = 5.0s > max 2.0s

    provider = SyntheticDynamicRegionProvider([
        ((10, 10, 40, 40), "person", 0.8, RegionMaskReference(mask_type="BBOX_ONLY"))
    ])

    report = DynamicSceneAnalyzer.analyze_frame(f1, region_provider=provider, prev_frame=f0)
    assert report.overall_scene_status == DynamicEvidenceCategory.INSUFFICIENT_EVIDENCE


# 14. Provenance Preservation
def test_provenance_preservation():
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(2, 0), "f_1", 1, 0.5)

    provider = SyntheticDynamicRegionProvider([
        ((10, 10, 40, 40), "car", 0.8, RegionMaskReference(mask_type="BBOX_ONLY"))
    ])

    report = DynamicSceneAnalyzer.analyze_frame(f1, region_provider=provider, prev_frame=f0)

    assert report.provenance["target_frame_id"] == "f_1"
    assert report.provenance["candidate_count"] == 1
    assert "discrepancy_threshold_px_s" in report.provenance


# 15. Immutability (Source Frames Untouched)
def test_immutability():
    img0 = generate_textured_pattern(0, 0)
    img1 = generate_textured_pattern(4, 0)
    copy0 = img0.copy()
    copy1 = img1.copy()

    f0 = create_decoded_frame(img0, "f_0", 0, 0.0)
    f1 = create_decoded_frame(img1, "f_1", 1, 0.5)

    _ = DynamicSceneAnalyzer.analyze_frame(f1, prev_frame=f0)

    assert f0.data is not None
    assert f1.data is not None
    assert np.array_equal(f0.data, copy0)
    assert np.array_equal(f1.data, copy1)


# 16. Deterministic Repeated Analysis
def test_deterministic_repeated_analysis():
    f0 = create_decoded_frame(generate_textured_pattern(0, 0), "f_0", 0, 0.0)
    f1 = create_decoded_frame(generate_textured_pattern(3, 0), "f_1", 1, 0.5)

    provider = SyntheticDynamicRegionProvider([
        ((10, 10, 50, 50), "car", 0.85, RegionMaskReference(mask_type="BBOX_ONLY"))
    ])

    rep1 = DynamicSceneAnalyzer.analyze_frame(f1, region_provider=provider, prev_frame=f0)
    rep2 = DynamicSceneAnalyzer.analyze_frame(f1, region_provider=provider, prev_frame=f0)

    assert rep1.global_motion_velocity_px_per_sec == rep2.global_motion_velocity_px_per_sec
    assert rep1.candidate_regions[0].relative_motion_discrepancy == rep2.candidate_regions[0].relative_motion_discrepancy
    assert rep1.candidate_regions[0].dynamic_evidence_score == rep2.candidate_regions[0].dynamic_evidence_score


# 17. Heuristic Threshold Configurability
def test_heuristic_threshold_configurability():
    bg = generate_textured_pattern(0, 0)
    f0_data = bg.copy()
    f1_data = bg.copy()
    patch = bg[80:120, 80:120].copy()
    M = np.array([[1, 0, 8], [0, 1, 0]], dtype=np.float32)
    f1_data[80:120, 80:120] = cv2.warpAffine(patch, M, (40, 40), borderMode=cv2.BORDER_REFLECT)

    f0 = create_decoded_frame(f0_data, "f_0", 0, 0.0)
    f1 = create_decoded_frame(f1_data, "f_1", 1, 0.5)

    provider = SyntheticDynamicRegionProvider([
        ((80, 80, 120, 120), "drone", 0.90, RegionMaskReference(mask_type="BBOX_ONLY"))
    ])

    # Default config (discrepancy threshold = 4.0 px/s) -> discrepancy ~16 px/s triggers DYNAMIC_EVIDENCE with persistence
    rep_default = DynamicSceneAnalyzer.analyze_frame(
        f1, region_provider=provider, prev_frame=f0, historical_persistence={"drone_4_4": 3}
    )
    assert rep_default.candidate_regions[0].evidence_category == DynamicEvidenceCategory.DYNAMIC_EVIDENCE

    # Strict config with very high discrepancy threshold (50.0 px/s) -> reverts to POSSIBLY_DYNAMIC or STATIC
    strict_cfg = DynamicSceneConfig(motion_discrepancy_threshold_px_s=50.0)
    rep_strict = DynamicSceneAnalyzer.analyze_frame(
        f1, region_provider=provider, prev_frame=f0, historical_persistence={"drone_4_4": 3}, config=strict_cfg
    )
    assert rep_strict.candidate_regions[0].evidence_category == DynamicEvidenceCategory.POSSIBLY_DYNAMIC
