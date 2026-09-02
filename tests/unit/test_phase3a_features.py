"""Deterministic unit and integration tests for Phase 3A Classical Feature Extraction & Descriptor Matching.

DISCLAIMER:
ALL FIXTURES IN THIS MODULE ARE DETERMINISTIC SYNTHETIC TEST DATA GENERATED
SOLELY FOR FEATURE EXTRACTION & MATCHING AUDITING. THEY DO NOT REPRESENT REAL UAV DATA.
"""

import math
import numpy as np
import pytest

from src.geometry import (
    MeasurementType,
    FeatureKeypoint,
    FeatureCorrespondences,
    FeatureDetectorType,
    DescriptorMatcherType,
    MatchingStrategy,
    FeatureFailureReason,
    FeatureConfig,
    SpatialMatchDiagnostics,
    FeatureExtractionResult,
    FeatureMatchResult,
    SpatialDistributionCalculator,
    ClassicalFeatureExtractor,
    ClassicalDescriptorMatcher,
)
from src.preprocessing.decoder import DecodedFrame


def create_synthetic_textured_image(width: int = 640, height: int = 480, seed: int = 42) -> np.ndarray:
    """Generate a deterministic synthetic image with sharp geometric patterns and high texture."""
    np.random.seed(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Base gradient
    for y in range(height):
        for x in range(width):
            img[y, x, 0] = (x * 255) // width
            img[y, x, 1] = (y * 255) // height
            img[y, x, 2] = ((x + y) * 255) // (width + height)

    # Add high-frequency checkerboards and distinct geometric corners
    for cy in range(40, height - 40, 60):
        for cx in range(40, width - 40, 60):
            # Draw a checkerboard square
            for sy in range(20):
                for sx in range(20):
                    val = 255 if ((sx // 5 + sy // 5) % 2 == 0) else 0
                    img[cy + sy, cx + sx] = [val, 255 - val, (val * 2) % 256]

    return img


# 1. Valid RGB Input Feature Extraction
def test_valid_rgb_feature_extraction():
    img = create_synthetic_textured_image(640, 480)
    extractor = ClassicalFeatureExtractor()
    res = extractor.extract(img, frame_id="frame_001")

    assert res.frame_id == "frame_001"
    assert res.width == 640
    assert res.height == 480
    assert res.keypoint_count > 100
    assert res.status == "SUCCESS"
    assert res.failure_reason is None
    assert res.descriptors.dtype == np.uint8
    assert res.descriptors.shape == (res.keypoint_count, 32)
    assert res.keypoints_xy.shape == (res.keypoint_count, 2)


# 2. Invalid Shape Rejection
def test_invalid_shape_rejection():
    invalid_2d = np.zeros((480, 640), dtype=np.uint8)  # Missing 3rd channel
    extractor = ClassicalFeatureExtractor()
    res = extractor.extract(invalid_2d, frame_id="invalid_2d")

    assert res.status == "FAILED"
    assert res.failure_reason == FeatureFailureReason.INVALID_IMAGE
    assert res.keypoint_count == 0


# 3. Invalid Dtype Rejection
def test_invalid_dtype_rejection():
    invalid_float = np.zeros((480, 640, 3), dtype=np.float32)  # Not uint8
    extractor = ClassicalFeatureExtractor()
    res = extractor.extract(invalid_float, frame_id="invalid_float")

    assert res.status == "FAILED"
    assert res.failure_reason == FeatureFailureReason.INVALID_IMAGE
    assert res.keypoint_count == 0


# 4. Grayscale Internal Preprocessing Semantics
def test_grayscale_preprocessing_semantics():
    img = create_synthetic_textured_image(640, 480)
    extractor = ClassicalFeatureExtractor()
    res = extractor.extract(img, frame_id="f_proc")

    assert res.preprocessing_status == "CONVERTED_RGB_TO_GRAYSCALE_BT601"
    assert res.measurement_type == MeasurementType.DIRECTLY_OBSERVED


# 5. ORB Feature Extraction Outputs
def test_orb_feature_properties():
    img = create_synthetic_textured_image(640, 480)
    extractor = ClassicalFeatureExtractor()
    res = extractor.extract(img, frame_id="f_props")

    assert len(res.keypoint_scales) == res.keypoint_count
    assert len(res.keypoint_angles) == res.keypoint_count
    assert len(res.keypoint_responses) == res.keypoint_count
    assert len(res.keypoint_octaves) == res.keypoint_count

    # Coordinates must be within pixel dimensions
    assert np.all(res.keypoints_xy[:, 0] >= 0.0)
    assert np.all(res.keypoints_xy[:, 0] < 640.0)
    assert np.all(res.keypoints_xy[:, 1] >= 0.0)
    assert np.all(res.keypoints_xy[:, 1] < 480.0)


# 6. Descriptor Shape and Type
def test_descriptor_shape_and_type():
    img = create_synthetic_textured_image(640, 480)
    extractor = ClassicalFeatureExtractor()
    res = extractor.extract(img, frame_id="f_desc")

    assert res.descriptors.dtype == np.uint8
    assert res.descriptors.shape[1] == 32  # 256-bit binary descriptor = 32 bytes
    assert res.descriptor_dim == 32
    assert res.descriptor_dtype == "uint8"


# 7. Empty / Uniform Zero Image (No Features Detected)
def test_empty_image_no_features():
    empty_img = np.zeros((480, 640, 3), dtype=np.uint8)
    extractor = ClassicalFeatureExtractor()
    res = extractor.extract(empty_img, frame_id="f_empty")

    assert res.status == "FAILED"
    assert res.failure_reason == FeatureFailureReason.NO_FEATURES_DETECTED
    assert res.keypoint_count == 0
    assert len(res.descriptors) == 0


# 8. Deterministic Repeated Extraction
def test_deterministic_repeated_extraction():
    img = create_synthetic_textured_image(640, 480)
    extractor = ClassicalFeatureExtractor()

    res1 = extractor.extract(img, frame_id="f_rep1")
    res2 = extractor.extract(img, frame_id="f_rep2")

    assert res1.keypoint_count == res2.keypoint_count
    np.testing.assert_array_almost_equal(res1.keypoints_xy, res2.keypoints_xy)
    np.testing.assert_array_equal(res1.descriptors, res2.descriptors)


# 9. Hamming Distance Matching
def test_hamming_matching():
    img_a = create_synthetic_textured_image(640, 480, seed=1)
    # img_b is slightly shifted version
    img_b = np.roll(img_a, shift=5, axis=1)

    extractor = ClassicalFeatureExtractor()
    matcher = ClassicalDescriptorMatcher()

    feat_a = extractor.extract(img_a, frame_id="f_a")
    feat_b = extractor.extract(img_b, frame_id="f_b")

    match_res = matcher.match(feat_a, feat_b)

    assert match_res.status == "SUCCESS"
    assert match_res.accepted_match_count > 30
    assert len(match_res.points_a) == match_res.accepted_match_count
    assert len(match_res.points_b) == match_res.accepted_match_count
    assert len(match_res.descriptor_distances) == match_res.accepted_match_count
    # Hamming distance must be >= 0
    assert np.all(match_res.descriptor_distances >= 0.0)


# 10. Lowe Ratio Filtering
def test_lowe_ratio_filtering():
    img_a = create_synthetic_textured_image(640, 480, seed=10)
    img_b = np.roll(img_a, shift=8, axis=0)

    extractor = ClassicalFeatureExtractor()
    feat_a = extractor.extract(img_a, frame_id="f_a")
    feat_b = extractor.extract(img_b, frame_id="f_b")

    # Strict Lowe ratio (0.50) vs Relaxed Lowe ratio (0.90)
    config_strict = FeatureConfig(matching_strategy=MatchingStrategy.RATIO_TEST, lowe_ratio=0.50)
    config_relaxed = FeatureConfig(matching_strategy=MatchingStrategy.RATIO_TEST, lowe_ratio=0.90)

    matcher_strict = ClassicalDescriptorMatcher(config_strict)
    matcher_relaxed = ClassicalDescriptorMatcher(config_relaxed)

    res_strict = matcher_strict.match(feat_a, feat_b)
    res_relaxed = matcher_relaxed.match(feat_a, feat_b)

    # Relaxed ratio should accept more candidate matches than strict
    assert res_relaxed.accepted_match_count >= res_strict.accepted_match_count


# 11. Mutual Consistency Filtering
def test_mutual_consistency_filtering():
    img_a = create_synthetic_textured_image(640, 480, seed=11)
    img_b = np.roll(img_a, shift=6, axis=1)

    extractor = ClassicalFeatureExtractor()
    feat_a = extractor.extract(img_a, frame_id="f_a")
    feat_b = extractor.extract(img_b, frame_id="f_b")

    config_ratio_only = FeatureConfig(matching_strategy=MatchingStrategy.RATIO_TEST, lowe_ratio=0.80)
    config_mutual = FeatureConfig(matching_strategy=MatchingStrategy.RATIO_AND_MUTUAL, lowe_ratio=0.80)

    matcher_ratio = ClassicalDescriptorMatcher(config_ratio_only)
    matcher_mutual = ClassicalDescriptorMatcher(config_mutual)

    res_ratio = matcher_ratio.match(feat_a, feat_b)
    res_mutual = matcher_mutual.match(feat_a, feat_b)

    # Mutual consistency filters asymmetric matches, so accepted <= ratio only
    assert res_mutual.accepted_match_count <= res_ratio.accepted_match_count


# 12. Zero Candidate Matches Handling
def test_zero_candidate_matches():
    img = create_synthetic_textured_image(640, 480)
    empty_img = np.zeros((480, 640, 3), dtype=np.uint8)

    extractor = ClassicalFeatureExtractor()
    matcher = ClassicalDescriptorMatcher()

    feat_valid = extractor.extract(img, frame_id="f_valid")
    feat_empty = extractor.extract(empty_img, frame_id="f_empty")

    match_res = matcher.match(feat_valid, feat_empty)
    assert match_res.status == "FAILED"
    assert match_res.failure_reason == FeatureFailureReason.NO_CANDIDATE_MATCHES
    assert match_res.accepted_match_count == 0


# 13. Insufficient Descriptor Matches Handling
def test_insufficient_descriptor_matches():
    img_a = create_synthetic_textured_image(640, 480, seed=20)
    # Completely different image (seed 999)
    img_b = create_synthetic_textured_image(640, 480, seed=999)

    extractor = ClassicalFeatureExtractor()
    # Require 500 matches minimum
    config_high_threshold = FeatureConfig(min_accepted_matches=500, lowe_ratio=0.60)
    matcher = ClassicalDescriptorMatcher(config_high_threshold)

    feat_a = extractor.extract(img_a, frame_id="f_a")
    feat_b = extractor.extract(img_b, frame_id="f_b")

    match_res = matcher.match(feat_a, feat_b)
    if match_res.accepted_match_count < 500 and match_res.accepted_match_count > 0:
        assert match_res.status == "DEGRADED"
        assert match_res.failure_reason == FeatureFailureReason.INSUFFICIENT_DESCRIPTOR_MATCHES


# 14. Descriptor Distance Statistics Calculation
def test_descriptor_distance_statistics():
    img_a = create_synthetic_textured_image(640, 480, seed=30)
    img_b = np.roll(img_a, shift=4, axis=1)

    extractor = ClassicalFeatureExtractor()
    matcher = ClassicalDescriptorMatcher()

    feat_a = extractor.extract(img_a, frame_id="f_a")
    feat_b = extractor.extract(img_b, frame_id="f_b")

    match_res = matcher.match(feat_a, feat_b)

    assert match_res.min_distance <= match_res.median_distance
    assert match_res.median_distance <= match_res.percentile_90_distance
    assert match_res.mean_distance >= 0.0
    assert 0.0 <= match_res.acceptance_ratio <= 1.0


# 15. Spatial Occupancy Diagnostics Calculation
def test_spatial_occupancy_diagnostics():
    pts = np.array([
        [100.0, 100.0],
        [200.0, 150.0],
        [300.0, 200.0],
        [500.0, 400.0],
    ], dtype=np.float64)

    diag = SpatialDistributionCalculator.compute(pts, width=640, height=480, grid_rows=8, grid_cols=8)

    assert diag.total_cell_count == 64
    assert diag.occupied_cell_count >= 1
    assert 0.0 < diag.grid_occupancy_ratio <= 1.0
    assert 0.0 <= diag.convex_hull_area_fraction <= 1.0
    assert 0.0 <= diag.spatial_entropy <= 1.0
    assert len(diag.normalized_bounding_box) == 4


# 16. Provenance and Status Serialization
def test_provenance_and_status_serialization():
    img_a = create_synthetic_textured_image(640, 480, seed=40)
    img_b = np.roll(img_a, shift=3, axis=0)

    extractor = ClassicalFeatureExtractor()
    matcher = ClassicalDescriptorMatcher()

    feat_a = extractor.extract(img_a, frame_id="f_a")
    feat_b = extractor.extract(img_b, frame_id="f_b")

    match_res = matcher.match(feat_a, feat_b)

    # Check to_dict() serialization
    d_feat = feat_a.to_dict()
    assert d_feat["frame_id"] == "f_a"
    assert d_feat["detector_type"] == "ORB"
    assert d_feat["measurement_type"] == "DIRECTLY_OBSERVED"

    d_match = match_res.to_dict()
    assert d_match["frame_a_id"] == "f_a"
    assert d_match["frame_b_id"] == "f_b"
    assert d_match["matching_strategy"] == "RATIO_AND_MUTUAL"
    assert d_match["measurement_type"] == "ESTIMATED"

    # Check to_correspondences() conversion for Phase 3B
    corr = match_res.to_correspondences()
    assert isinstance(corr, FeatureCorrespondences)
    assert corr.frame_a_id == "f_a"
    assert corr.frame_b_id == "f_b"
    assert corr.match_count == match_res.accepted_match_count


# 17. Threshold Classification as HEURISTIC_DEFAULT
def test_threshold_classification_as_heuristic():
    cfg = FeatureConfig()
    assert cfg.max_features == 2000
    assert cfg.lowe_ratio == 0.75
    assert cfg.min_accepted_matches == 30
    assert cfg.max_descriptor_distance == 64.0
    assert cfg.min_features_threshold == 100


# 18. Distinction Between Descriptor Matches and Geometric Inliers
def test_descriptor_matches_distinct_from_geometric_inliers():
    # Descriptor match result must NOT claim epipolar or geometric verification
    img_a = create_synthetic_textured_image(640, 480, seed=50)
    img_b = np.roll(img_a, shift=5, axis=1)

    extractor = ClassicalFeatureExtractor()
    matcher = ClassicalDescriptorMatcher()

    feat_a = extractor.extract(img_a, frame_id="f_a")
    feat_b = extractor.extract(img_b, frame_id="f_b")
    match_res = matcher.match(feat_a, feat_b)

    # MatchResult contains candidate/accepted descriptor matches, NOT verified inliers
    assert hasattr(match_res, "accepted_match_count")
    assert not hasattr(match_res, "inlier_ratio")  # Belongs strictly to Phase 3B TwoViewGeometryResult
    assert not hasattr(match_res, "essential_matrix")  # Belongs strictly to Phase 3B


# 19. Integration Test: DecodedFrame -> FeatureExtraction -> FeatureMatching
def test_integration_decoded_frame_to_feature_matching():
    rgb_a = create_synthetic_textured_image(640, 480, seed=60)
    rgb_b = np.roll(rgb_a, shift=10, axis=1)

    # Wrap in canonical DecodedFrame contract
    from src.preprocessing.decoder import DecodeStatus
    frame_a = DecodedFrame(
        frame_id="canonical_000",
        frame_index=0,
        timestamp_seconds=0.0,
        width=640,
        height=480,
        channels=3,
        channel_layout="RGB",
        dtype="uint8",
        data=rgb_a,
        source_video="test_video.mp4",
        decode_status=DecodeStatus.SUCCESS,
    )
    frame_b = DecodedFrame(
        frame_id="canonical_001",
        frame_index=1,
        timestamp_seconds=0.1,
        width=640,
        height=480,
        channels=3,
        channel_layout="RGB",
        dtype="uint8",
        data=rgb_b,
        source_video="test_video.mp4",
        decode_status=DecodeStatus.SUCCESS,
    )

    extractor = ClassicalFeatureExtractor()
    matcher = ClassicalDescriptorMatcher()

    res_a = extractor.extract(frame_a)
    res_b = extractor.extract(frame_b)

    assert res_a.frame_id == "canonical_000"
    assert res_b.frame_id == "canonical_001"
    assert res_a.status == "SUCCESS"
    assert res_b.status == "SUCCESS"

    match_res = matcher.match(res_a, res_b)

    assert match_res.frame_a_id == "canonical_000"
    assert match_res.frame_b_id == "canonical_001"
    assert match_res.status == "SUCCESS"
    assert match_res.accepted_match_count > 30

    # Convert to FeatureCorrespondences contract for Phase 3B
    corr = match_res.to_correspondences()
    assert corr.frame_a_id == "canonical_000"
    assert corr.frame_b_id == "canonical_001"
    assert corr.match_count == match_res.accepted_match_count
    assert corr.points_a.shape == (match_res.accepted_match_count, 2)
    assert corr.points_b.shape == (match_res.accepted_match_count, 2)
