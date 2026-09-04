"""Phase 3F: Integration Test Suite for Steps 1–4.

Covers models, stage classifications, artifact immutability, cryptographic SHA-256
fingerprinting, anti-leakage isolation, configuration constraints, and orchestrator skeleton.
"""

from __future__ import annotations

import copy
import numpy as np
import pytest

from src.pipeline import (
    PipelineStageType,
    StageClassification,
    StageStatus,
    PipelineStatus,
    ReconstructionUnitType,
    MetricScaleStatus,
    StageExecutionRecord,
    PipelineConfig,
    PipelineArtifact,
    ArtifactDomain,
    compute_canonical_payload_hash,
    VideoArtifact,
    CanonicalTimelineArtifact,
    DecodedFramesArtifact,
    FrameQualityArtifact,
    KeyframeSetArtifact,
    CorrespondenceArtifact,
    TwoViewGeometryArtifact,
    SfMArtifact,
    DenseStereoArtifact,
    SurfaceArtifact,
    GeospatialArtifact,
    FinalReconstructionArtifact,
    ReconstructionPipeline,
    ContractViolationError,
    DataLeakageError,
)
from src.geometry.contracts import PipelineStageStatus


def test_3f_01_model_status_enums():
    """TEST-3F-01: Verify model and status enum values and definitions."""
    assert PipelineStageType.INGESTION.value == "INGESTION"
    assert PipelineStageType.FINALIZATION.value == "FINALIZATION"
    assert len(PipelineStageType) == 17

    assert StageStatus.SUCCESS.value == "SUCCESS"
    assert StageStatus.CONTRACT_VIOLATION.value == "CONTRACT_VIOLATION"
    assert StageStatus.INSUFFICIENT_INPUT.value == "INSUFFICIENT_INPUT"

    assert PipelineStatus.SUCCESS.value == "SUCCESS"
    assert PipelineStatus.CONTRACT_VIOLATION.value == "CONTRACT_VIOLATION"

    assert ReconstructionUnitType.RECONSTRUCTION_UNITS.value == "RECONSTRUCTION_UNITS"
    assert MetricScaleStatus.SCALE_AMBIGUOUS.value == "SCALE_AMBIGUOUS"


def test_3f_02_stage_classification_correctness():
    """TEST-3F-02: Verify authoritative stage classifications (Mandatory vs Conditional vs Optional)."""
    classifications = ReconstructionPipeline.STAGE_CLASSIFICATIONS

    # Core geometry stages are strictly MANDATORY
    assert classifications[PipelineStageType.INGESTION] == StageClassification.MANDATORY
    assert classifications[PipelineStageType.DECODING] == StageClassification.MANDATORY
    assert classifications[PipelineStageType.KEYFRAME_SELECTION] == StageClassification.MANDATORY
    assert classifications[PipelineStageType.CORRESPONDENCE] == StageClassification.MANDATORY
    assert classifications[PipelineStageType.TWO_VIEW_GEOMETRY] == StageClassification.MANDATORY
    assert classifications[PipelineStageType.INCREMENTAL_SFM] == StageClassification.MANDATORY
    assert classifications[PipelineStageType.BUNDLE_ADJUSTMENT] == StageClassification.MANDATORY

    # Dense & Geospatial stages are CONDITIONAL
    assert classifications[PipelineStageType.DENSE_STEREO] == StageClassification.CONDITIONAL
    assert classifications[PipelineStageType.DENSE_POINT_GENERATION] == StageClassification.CONDITIONAL
    assert classifications[PipelineStageType.DENSE_FUSION] == StageClassification.CONDITIONAL
    assert classifications[PipelineStageType.SURFACE_RECONSTRUCTION] == StageClassification.CONDITIONAL
    assert classifications[PipelineStageType.GEOSPATIAL_TRANSFORM] == StageClassification.CONDITIONAL

    # Texturing is OPTIONAL
    assert classifications[PipelineStageType.TEXTURE_ASSOCIATION] == StageClassification.OPTIONAL
    assert classifications[PipelineStageType.TEXTURE_RECONSTRUCTION] == StageClassification.OPTIONAL


def test_3f_03_artifact_sha256_deterministic_hashing():
    """TEST-3F-03: Verify deterministic SHA-256 fingerprinting for arrays, dicts, and nested structures."""
    arr1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    arr2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    hash1 = compute_canonical_payload_hash(arr1)
    hash2 = compute_canonical_payload_hash(arr2)

    assert isinstance(hash1, str)
    assert len(hash1) == 64
    assert hash1 == hash2

    # Dict hashing is order-independent due to sort_keys=True
    d1 = {"z": 1, "a": [1, 2, 3]}
    d2 = {"a": [1, 2, 3], "z": 1}
    assert compute_canonical_payload_hash(d1) == compute_canonical_payload_hash(d2)


def test_3f_04_artifact_tampering_detected():
    """TEST-3F-04: In-memory tampering of payload raises ContractViolationError on verify_integrity()."""
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    art = SfMArtifact(artifact_id="art_sfm_001", payload=pts)

    # Initial integrity check passes
    art.verify_integrity()

    # Tamper with payload in-place
    pts[0, 0] = 999.9

    # Re-verification must detect tampering
    with pytest.raises(ContractViolationError, match="Artifact tampering detected"):
        art.verify_integrity()


def test_3f_05_artifact_metadata_immutability_and_integrity():
    """TEST-3F-05: Verify artifact metadata structure and immutability properties."""
    art = VideoArtifact(
        artifact_id="art_vid_001",
        payload={"filepath": "flight.mp4", "duration": 10.0},
        metadata={"camera_model": "FC3582"},
    )
    assert art.artifact_type == "VideoArtifact"
    assert art.domain == ArtifactDomain.RECONSTRUCTION_INPUT
    assert art.units == "SECONDS"
    assert art.coordinate_frame == "TEMPORAL"
    assert art.metadata["camera_model"] == "FC3582"
    assert len(art.content_hash) == 64


def test_3f_06_artifact_domain_separation():
    """TEST-3F-06: Verify domain classification tags on artifacts."""
    input_art = VideoArtifact(artifact_id="art_in", payload={"f": 1})
    assert input_art.domain == ArtifactDomain.RECONSTRUCTION_INPUT

    sfm_art = SfMArtifact(artifact_id="art_sfm", payload={"pts": []})
    assert sfm_art.domain == ArtifactDomain.RECONSTRUCTION_OUTPUT


def test_3f_07_hidden_evaluation_artifact_rejected():
    """TEST-3F-07: Ingestion of EVALUATION_TRUTH artifact or forbidden keys into reconstruction domain is blocked."""
    # 1. Prohibited evaluation truth key in reconstruction-domain artifact metadata
    with pytest.raises(DataLeakageError, match="Privileged evaluation key 'true_camera_poses'"):
        SfMArtifact(
            artifact_id="art_leaked_01",
            payload={"points": []},
            metadata={"true_camera_poses": np.eye(4)},
        )

    # 2. Prohibited evaluation truth key in payload
    with pytest.raises(DataLeakageError, match="Privileged evaluation key 'true_depth_maps'"):
        DenseStereoArtifact(
            artifact_id="art_leaked_02",
            payload={"true_depth_maps": np.ones((10, 10))},
        )

    # 3. Passing EVALUATION_TRUTH domain artifact to orchestrator
    eval_truth = PipelineArtifact(
        artifact_id="art_eval_cad",
        artifact_type="CADTruthArtifact",
        domain=ArtifactDomain.EVALUATION_TRUTH,
        producer_stage="BENCHMARK_HARNESS",
        input_artifact_ids=[],
        units="METRES",
        coordinate_frame="CAD_LOCAL",
        payload={"mesh": "hemisphere"},
    )
    pipe = ReconstructionPipeline()
    with pytest.raises(DataLeakageError, match="domain EVALUATION_TRUTH"):
        pipe.run([eval_truth])


def test_3f_08_configuration_validation():
    """TEST-3F-08: Default configuration is valid and computes deterministic hash."""
    cfg = PipelineConfig()
    cfg.validate()
    h1 = cfg.compute_hash()
    h2 = cfg.compute_hash()
    assert h1 == h2
    assert len(h1) == 64


def test_3f_09_invalid_stage_dependency_rejected():
    """TEST-3F-09: Incompatible stage configurations are rejected with ContractViolationError."""
    # Texturing enabled without surface meshing
    cfg_invalid_1 = PipelineConfig(enable_texturing=True, enable_surface_meshing=False)
    with pytest.raises(ContractViolationError, match="enable_texturing=True requires enable_surface_meshing=True"):
        cfg_invalid_1.validate()

    # Dense point generation without dense stereo
    cfg_invalid_2 = PipelineConfig(enable_dense_point_generation=True, enable_dense_stereo=False)
    with pytest.raises(ContractViolationError, match="enable_dense_point_generation=True requires enable_dense_stereo=True"):
        cfg_invalid_2.validate()

    # Dense fusion without dense point generation
    cfg_invalid_3 = PipelineConfig(enable_dense_fusion=True, enable_dense_point_generation=False)
    with pytest.raises(ContractViolationError, match="enable_dense_fusion=True requires enable_dense_point_generation=True"):
        cfg_invalid_3.validate()


def test_3f_10_mandatory_stage_failure_propagation():
    """TEST-3F-10: Failing a mandatory stage terminates the pipeline with FAILED status."""
    pipe = ReconstructionPipeline()

    # Manually execute a mandatory stage with a simulated fatal exception
    class SimFatalError(Exception):
        pass

    # Injecting invalid artifact triggers stage failure and halts pipeline
    corrupted_art = VideoArtifact(artifact_id="art_corrupt", payload={"data": [1, 2]})
    corrupted_art.content_hash = "fake_invalid_hash_00000000000000000000000000000000000000000000000000000000"

    res = pipe.run([corrupted_art])
    assert res.pipeline_status in (PipelineStatus.FAILED, PipelineStatus.CONTRACT_VIOLATION)
    assert res.failure_reason is not None


def test_3f_11_conditional_stage_not_evaluable_behavior():
    """TEST-3F-11: Disabling a conditional stage emits NOT_EVALUABLE without failing the pipeline."""
    cfg = PipelineConfig(enable_geospatial=False)
    pipe = ReconstructionPipeline(config=cfg)

    record = pipe.execute_stage(PipelineStageType.GEOSPATIAL_TRANSFORM, [])
    assert record.status == StageStatus.NOT_EVALUABLE
    assert record.classification == StageClassification.CONDITIONAL
    assert "enable_geospatial is configured to False" in record.diagnostics.get("reason", "")


def test_3f_12_stage_provenance_recording():
    """TEST-3F-12: StageExecutionRecord records duration, software commit, configuration hash, and timestamps."""
    pipe = ReconstructionPipeline()
    record = pipe.execute_stage(PipelineStageType.INGESTION, [])

    assert record.stage_type == PipelineStageType.INGESTION
    assert record.classification == StageClassification.MANDATORY
    assert record.software_commit == "019deb2"
    assert len(record.configuration_hash) == 64
    assert record.duration_seconds >= 0.0
    assert record.started_at != ""
    assert record.finished_at != ""


def test_3f_13_pipeline_result_status_propagation():
    """TEST-3F-13: PipelineResult aggregates stage records and accurately reports execution summary."""
    pipe = ReconstructionPipeline()
    res = pipe.run([])

    assert res.run_id.startswith("run_")
    assert isinstance(res.stage_records, list)
    assert len(res.stage_records) > 0
    assert res.reconstruction_units == ReconstructionUnitType.RECONSTRUCTION_UNITS
    assert res.metric_scale_status == MetricScaleStatus.NOT_EVALUABLE


def test_3f_14_deterministic_hashes_under_same_canonical_payload():
    """TEST-3F-14: Two independent artifacts constructed from identical payload have identical content_hash."""
    payload_a = {"points": [1.0, 2.0, 3.0], "label": "test_anchor"}
    payload_b = {"points": [1.0, 2.0, 3.0], "label": "test_anchor"}

    art_a = SfMArtifact(artifact_id="art_a", payload=payload_a)
    art_b = SfMArtifact(artifact_id="art_b", payload=payload_b)

    assert art_a.content_hash == art_b.content_hash


def test_3f_15_changed_payload_changes_hash():
    """TEST-3F-15: Any modification in payload produces a completely different cryptographic SHA-256 hash."""
    payload_orig = {"points": [1.0, 2.0, 3.0]}
    payload_diff = {"points": [1.0, 2.0, 3.00001]}

    hash_orig = compute_canonical_payload_hash(payload_orig)
    hash_diff = compute_canonical_payload_hash(payload_diff)

    assert hash_orig != hash_diff


# ===========================================================================
# Steps 5–8 Integration Tests: Actual Image-Based Reconstruction Pipeline
# ===========================================================================

from tests.integration.synthetic_scene_fixture import generate_synthetic_multiview_dataset
from src.geometry.contracts import CameraIntrinsics, DistortionModel, DistortionStatus, FeatureCorrespondences, TwoViewGeometryResult
from src.geometry.features import ClassicalFeatureExtractor, ClassicalDescriptorMatcher
from src.geometry.two_view import TwoViewGeometryEstimator
from src.geometry.sfm import IncrementalSfMEngine, MatchGraph
from src.geometry.bundle_adjustment import BundleAdjustmentEngine


def test_3f_16_image_sequence_to_canonical_timeline():
    """TEST-3F-16: Actual image sequence produces a validated CanonicalTimeline with monotonic PTS."""
    imgs, K, _ = generate_synthetic_multiview_dataset(n_views=3)
    pipe = ReconstructionPipeline(config=PipelineConfig(default_intrinsics=K))

    in_art = VideoArtifact(artifact_id="art_vid_16", payload={"frames": imgs})
    record = pipe.execute_stage(PipelineStageType.INGESTION, [in_art])

    assert record.status == StageStatus.SUCCESS
    timeline_art = pipe.artifacts_by_type["CanonicalTimelineArtifact"]
    timeline = timeline_art.payload
    assert timeline.total_frames == 3
    assert [f.pts for f in timeline.frames] == [0, 500, 1000]
    assert timeline.frames[0].width == 640
    assert timeline.frames[0].height == 480


def test_3f_17_decoded_images_to_keyframe_selection():
    """TEST-3F-17: Decoded image frames pass through frame intelligence to keyframe selection."""
    imgs, K, _ = generate_synthetic_multiview_dataset(n_views=3)
    pipe = ReconstructionPipeline(config=PipelineConfig(default_intrinsics=K))

    in_art = VideoArtifact(artifact_id="art_vid_17", payload={"frames": imgs})
    pipe.execute_stage(PipelineStageType.INGESTION, [in_art])
    pipe.execute_stage(PipelineStageType.DECODING, [in_art])
    pipe.execute_stage(PipelineStageType.FRAME_INTELLIGENCE, [pipe.artifacts_by_type["DecodedFramesArtifact"]])
    rec_kf = pipe.execute_stage(PipelineStageType.KEYFRAME_SELECTION, [pipe.artifacts_by_type["DecodedFramesArtifact"]])

    assert rec_kf.status == StageStatus.SUCCESS
    kf_art = pipe.artifacts_by_type["KeyframeSetArtifact"]
    assert len(kf_art.payload.selected_keyframe_ids) >= 2


def test_3f_18_keyframes_to_actual_descriptors_and_matches():
    """TEST-3F-18: Keyframes produce actual 2D ORB descriptors and non-trivial matches across pairs."""
    imgs, K, _ = generate_synthetic_multiview_dataset(n_views=3)
    pipe = ReconstructionPipeline(config=PipelineConfig(default_intrinsics=K))

    in_art = VideoArtifact(artifact_id="art_vid_18", payload={"frames": imgs})
    pipe.execute_stage(PipelineStageType.INGESTION, [in_art])
    pipe.execute_stage(PipelineStageType.DECODING, [in_art])
    pipe.execute_stage(PipelineStageType.FRAME_INTELLIGENCE, [pipe.artifacts_by_type["DecodedFramesArtifact"]])
    pipe.execute_stage(PipelineStageType.KEYFRAME_SELECTION, [pipe.artifacts_by_type["DecodedFramesArtifact"]])
    rec_corr = pipe.execute_stage(PipelineStageType.CORRESPONDENCE, [pipe.artifacts_by_type["KeyframeSetArtifact"]])

    assert rec_corr.status == StageStatus.SUCCESS
    corr_art = pipe.artifacts_by_type["CorrespondenceArtifact"]
    pairwise_corr = corr_art.payload
    assert len(pairwise_corr) >= 1
    # Check that matches have real non-zero coordinates and descriptor distances
    first_pair = list(pairwise_corr.keys())[0]
    corr = pairwise_corr[first_pair]
    assert corr.match_count >= 15
    assert corr.points_a.shape[0] == corr.match_count
    assert corr.points_b.shape[0] == corr.match_count
    assert len(corr.descriptor_distances) == corr.match_count


def test_3f_19_actual_correspondences_to_two_view_geometry():
    """TEST-3F-19: Real correspondences yield a verified Essential matrix and relative pose."""
    imgs, K, _ = generate_synthetic_multiview_dataset(n_views=3)
    pipe = ReconstructionPipeline(config=PipelineConfig(default_intrinsics=K))

    in_art = VideoArtifact(artifact_id="art_vid_19", payload={"frames": imgs})
    pipe.execute_stage(PipelineStageType.INGESTION, [in_art])
    pipe.execute_stage(PipelineStageType.DECODING, [in_art])
    pipe.execute_stage(PipelineStageType.FRAME_INTELLIGENCE, [pipe.artifacts_by_type["DecodedFramesArtifact"]])
    pipe.execute_stage(PipelineStageType.KEYFRAME_SELECTION, [pipe.artifacts_by_type["DecodedFramesArtifact"]])
    pipe.execute_stage(PipelineStageType.CORRESPONDENCE, [pipe.artifacts_by_type["KeyframeSetArtifact"]])
    rec_tv = pipe.execute_stage(PipelineStageType.TWO_VIEW_GEOMETRY, [pipe.artifacts_by_type["CorrespondenceArtifact"]])

    assert rec_tv.status == StageStatus.SUCCESS
    tv_art = pipe.artifacts_by_type["TwoViewGeometryArtifact"]
    seed_res = tv_art.payload["seed_result"]
    assert seed_res.e_status == "SUCCESS"
    assert seed_res.inlier_count >= 15
    assert not seed_res.is_degenerate
    assert seed_res.relative_rotation is not None
    assert seed_res.relative_translation is not None


def test_3f_20_two_view_result_to_sfm_seed():
    """TEST-3F-20: Seed pair initializes gauge with Camera 0 at origin and Camera 1 with unit baseline."""
    imgs, K, _ = generate_synthetic_multiview_dataset(n_views=2)
    ext = ClassicalFeatureExtractor()
    f0 = ext.extract(imgs[0], frame_id="cam_00")
    f1 = ext.extract(imgs[1], frame_id="cam_01")
    mat = ClassicalDescriptorMatcher()
    m = mat.match(f0, f1)
    corr = m.to_correspondences()

    est = TwoViewGeometryEstimator()
    tv_res = est.estimate_essential(corr, K)
    assert tv_res.e_status == "SUCCESS"

    sfm = IncrementalSfMEngine()
    ok = sfm.initialize_two_view(tv_res, corr, {"cam_00": K, "cam_01": K})
    assert ok is True
    assert len(sfm.cameras) == 2

    cam0 = sfm.cameras["cam_00"]
    cam1 = sfm.cameras["cam_01"]
    # Camera 0 is at origin [I | 0]
    np.testing.assert_allclose(cam0.R_cw, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(cam0.t_cw, np.zeros(3), atol=1e-6)
    # Camera 1 has unit baseline ||t_cw|| = 1.0
    baseline = np.linalg.norm(cam1.t_cw)
    np.testing.assert_allclose(baseline, 1.0, atol=1e-5)


def test_3f_21_sfm_incremental_landmark_creation():
    """TEST-3F-21: Incremental SfM creates verified 3D landmarks with positive optical depth."""
    imgs, K, _ = generate_synthetic_multiview_dataset(n_views=3)
    pipe = ReconstructionPipeline(config=PipelineConfig(default_intrinsics=K))

    in_art = VideoArtifact(artifact_id="art_vid_21", payload={"frames": imgs})
    pipe.artifacts_by_id[in_art.artifact_id] = in_art
    pipe.artifacts_by_type[in_art.artifact_type] = in_art
    for stg in [
        PipelineStageType.INGESTION,
        PipelineStageType.DECODING,
        PipelineStageType.FRAME_INTELLIGENCE,
        PipelineStageType.KEYFRAME_SELECTION,
        PipelineStageType.CORRESPONDENCE,
        PipelineStageType.TWO_VIEW_GEOMETRY,
        PipelineStageType.INCREMENTAL_SFM,
    ]:
        rec = pipe.execute_stage(stg, list(pipe.artifacts_by_id.values()))
        assert rec.status == StageStatus.SUCCESS

    sfm_art = pipe.artifacts_by_type["SfMArtifact"]
    recon = sfm_art.payload
    assert recon.total_registered_cameras >= 2
    assert recon.total_triangulated_points >= 20
    assert recon.mean_reprojection_rmse_px > 0.0
    assert recon.is_metric_scale is False
    assert recon.has_monocular_scale_ambiguity is True


def test_3f_22_sfm_to_bundle_adjustment():
    """TEST-3F-22: Bundle adjustment refines sparse reconstruction without increasing reprojection cost."""
    imgs, K, _ = generate_synthetic_multiview_dataset(n_views=3)
    pipe = ReconstructionPipeline(config=PipelineConfig(default_intrinsics=K))

    in_art = VideoArtifact(artifact_id="art_vid_22", payload={"frames": imgs})
    pipe.artifacts_by_id[in_art.artifact_id] = in_art
    pipe.artifacts_by_type[in_art.artifact_type] = in_art
    for stg in [
        PipelineStageType.INGESTION,
        PipelineStageType.DECODING,
        PipelineStageType.FRAME_INTELLIGENCE,
        PipelineStageType.KEYFRAME_SELECTION,
        PipelineStageType.CORRESPONDENCE,
        PipelineStageType.TWO_VIEW_GEOMETRY,
        PipelineStageType.INCREMENTAL_SFM,
        PipelineStageType.BUNDLE_ADJUSTMENT,
    ]:
        rec = pipe.execute_stage(stg, list(pipe.artifacts_by_id.values()))
        assert rec.status == StageStatus.SUCCESS

    ba_art = pipe.artifacts_by_type["BundleAdjustmentArtifact"]
    ba_res = ba_art.payload
    assert ba_res.metrics_before is not None
    assert ba_res.metrics_after is not None
    assert ba_res.metrics_after.rmse_px <= ba_res.metrics_before.rmse_px + 1e-4
    assert ba_res.gauge_preserved is True


def test_3f_23_ba_preserves_gauge_semantics():
    """TEST-3F-23: Bundle adjustment strictly preserves unit baseline and monocular scale ambiguity."""
    imgs, K, _ = generate_synthetic_multiview_dataset(n_views=3)
    pipe = ReconstructionPipeline(config=PipelineConfig(default_intrinsics=K))

    in_art = VideoArtifact(artifact_id="art_vid_23", payload={"frames": imgs})
    res = pipe.run([in_art])
    assert res.pipeline_status == PipelineStatus.SUCCESS

    ba_art = pipe.artifacts_by_type["BundleAdjustmentArtifact"]
    ba_res = ba_art.payload
    assert ba_res.gauge_preserved is True
    assert ba_res.is_metric_scale is False
    assert ba_res.has_monocular_scale_ambiguity is True

    refined_recon = ba_res.refined_reconstruction
    assert refined_recon.is_metric_scale is False
    assert refined_recon.has_monocular_scale_ambiguity is True


def test_3f_24_insufficient_correspondence_failure():
    """TEST-3F-24: Featureless/blank image frames trigger InsufficientInputError and halt pipeline."""
    # Create blank black frames with 0 texture
    blank_imgs = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
    K = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0, width=640, height=480)
    pipe = ReconstructionPipeline(config=PipelineConfig(default_intrinsics=K))

    in_art = VideoArtifact(artifact_id="art_blank_24", payload={"frames": blank_imgs})
    res = pipe.run([in_art])

    # Pipeline must fail cleanly without inventing features or shortcuts
    assert res.pipeline_status == PipelineStatus.FAILED
    assert res.failure_reason is not None
    assert "CORRESPONDENCE" in res.failure_reason or "KEYFRAME_SELECTION" in res.failure_reason


def test_3f_25_degenerate_two_view_pair_rejection():
    """TEST-3F-25: Degenerate pure planar or zero-baseline configurations are rejected by two-view estimator."""
    K = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0, width=640, height=480)
    est = TwoViewGeometryEstimator()

    # Create degenerate planar homography points (pure affine transform in 2D with 0 parallax)
    pts_a = np.random.uniform(50, 400, size=(50, 2))
    pts_b = pts_a + 0.1  # Virtually zero parallax

    corr = FeatureCorrespondences(
        frame_a_id="frame_0",
        frame_b_id="frame_1",
        points_a=pts_a,
        points_b=pts_b,
        descriptor_distances=np.zeros(50),
        match_count=50,
        descriptor_type="ORB",
        provenance={"test": "degenerate_zero_parallax"},
    )

    res = est.estimate_essential(corr, K)
    assert res.e_status == "FAILED" or res.is_degenerate is True


def test_3f_26_provenance_propagation_through_sfm_and_ba():
    """TEST-3F-26: Stage records record exact input/output artifact IDs, software commit, and duration."""
    imgs, K, _ = generate_synthetic_multiview_dataset(n_views=3)
    pipe = ReconstructionPipeline(config=PipelineConfig(default_intrinsics=K))

    in_art = VideoArtifact(artifact_id="art_vid_26", payload={"frames": imgs})
    res = pipe.run([in_art])
    assert res.pipeline_status == PipelineStatus.SUCCESS

    sfm_records = [r for r in res.stage_records if r.stage_type == PipelineStageType.INCREMENTAL_SFM]
    ba_records = [r for r in res.stage_records if r.stage_type == PipelineStageType.BUNDLE_ADJUSTMENT]

    assert len(sfm_records) == 1
    assert len(ba_records) == 1

    sfm_rec = sfm_records[0]
    ba_rec = ba_records[0]

    assert sfm_rec.status == StageStatus.SUCCESS
    assert ba_rec.status == StageStatus.SUCCESS
    assert sfm_rec.software_commit == "019deb2"
    assert ba_rec.software_commit == "019deb2"
    assert len(sfm_rec.input_artifact_ids) > 0
    assert len(ba_rec.input_artifact_ids) > 0
    assert sfm_rec.duration_seconds > 0.0
    assert ba_rec.duration_seconds > 0.0


def test_3f_27_end_to_end_authenticity_and_anti_shortcut_assertions():
    """TEST-3F-27: Strict verification that reconstruction is computed purely from image pixels without GT leakage."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    pipe = ReconstructionPipeline(config=PipelineConfig(default_intrinsics=K))

    # 1. Verify RGB images are actual pipeline inputs
    assert all(isinstance(img, np.ndarray) and img.shape == (480, 640, 3) for img in imgs)
    in_art = VideoArtifact(artifact_id="art_vid_27", payload={"frames": imgs})
    res = pipe.run([in_art])
    assert res.pipeline_status == PipelineStatus.SUCCESS

    # 2. Hidden GT artifact IDs and hashes do NOT occur in reconstruction artifacts
    gt_id = "art_hidden_truth_scene_001"
    gt_hash = compute_canonical_payload_hash(hidden_gt["true_3d_facets"])
    for art in pipe.artifacts_by_id.values():
        assert art.artifact_id != gt_id
        assert art.content_hash != gt_hash

    # 3. Camera poses are produced by SfM and BA
    sfm_art = pipe.artifacts_by_type["SfMArtifact"]
    recon = sfm_art.payload
    assert len(recon.camera_poses) >= 2
    for cid, pose in recon.camera_poses.items():
        assert pose.rotation_matrix is not None
        assert pose.translation_vector is not None
        assert pose.is_metric is False  # Scale ambiguous

    # 4. Landmarks are produced from image correspondences, not copied from GT
    ba_art = pipe.artifacts_by_type["BundleAdjustmentArtifact"]
    ba_res = ba_art.payload
    refined_recon = ba_res.refined_reconstruction
    assert len(refined_recon.points3d) >= 20

    # Ensure landmark coordinates do not match hidden GT facet coordinates
    all_gt_pts = np.vstack([facet.corners_3d for facet in hidden_gt["true_3d_facets"]])
    for track_id, track in refined_recon.points3d.items():
        assert len(track.observations) >= 2  # Real 2D ray observations
        diffs = np.linalg.norm(all_gt_pts - track.world_point, axis=1)
        assert np.min(diffs) > 0.05  # Reconstructed coordinates are in relative SfM gauge, not GT world meters

    # 5. RECONSTRUCTION_UNITS remains scale ambiguous
    assert res.reconstruction_units == ReconstructionUnitType.RECONSTRUCTION_UNITS
    assert res.metric_scale_status in (MetricScaleStatus.SCALE_AMBIGUOUS, MetricScaleStatus.NOT_EVALUABLE)
    assert refined_recon.is_metric_scale is False
    assert refined_recon.has_monocular_scale_ambiguity is True

    # 6. BA preserves the gauge constraint
    assert ba_res.gauge_preserved is True
    cam0_center = refined_recon.camera_centers["frame_0000"]
    np.testing.assert_allclose(cam0_center, np.zeros(3), atol=1e-5)


def test_3f_28_insufficient_frames_failure():
    """TEST-3F-28: Providing insufficient frames (< 2) cleanly fails pipeline without false success."""
    imgs, K, _ = generate_synthetic_multiview_dataset(n_views=1)
    pipe = ReconstructionPipeline(config=PipelineConfig(default_intrinsics=K))

    in_art = VideoArtifact(artifact_id="art_single_frame", payload={"frames": imgs})
    res = pipe.run([in_art])

    assert res.pipeline_status == PipelineStatus.FAILED
    assert res.failure_reason is not None
    assert "KEYFRAME_SELECTION" in res.failure_reason or "Insufficient keyframes" in res.failure_reason


def test_3f_29_ba_failure_clean_handling():
    """TEST-3F-29: Solver divergence or failure in Bundle Adjustment fails cleanly without marking SUCCESS."""
    imgs, K, _ = generate_synthetic_multiview_dataset(n_views=3)
    # Configure impossible BA acceptance criteria (e.g. minimum 99 cameras required)
    from src.geometry.bundle_adjustment import BundleAdjustmentConfig
    impossible_ba_cfg = BundleAdjustmentConfig(min_registered_cameras=99)
    cfg = PipelineConfig(default_intrinsics=K, ba_config=impossible_ba_cfg)
    pipe = ReconstructionPipeline(config=cfg)

    in_art = VideoArtifact(artifact_id="art_vid_29", payload={"frames": imgs})
    res = pipe.run([in_art])

    assert res.pipeline_status == PipelineStatus.FAILED
    assert res.failure_reason is not None
    assert "BUNDLE_ADJUSTMENT" in res.failure_reason
    # No failed stage may be marked SUCCESS
    ba_record = [r for r in res.stage_records if r.stage_type == PipelineStageType.BUNDLE_ADJUSTMENT][0]
    assert ba_record.status == StageStatus.FAILED


def test_3f_30_keyframe_selector_retains_at_least_three_views():
    """TEST-3F-30: Keyframe selector retains >= 3 useful views for incremental SfM test."""
    imgs, K, _ = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_30", payload={"frames": imgs})
    res = pipe.run([in_art])

    kf_art = pipe.artifacts_by_type["KeyframeSetArtifact"]
    kf_result = kf_art.payload
    assert len(kf_result.selected_keyframe_ids) >= 3
    assert len(kf_result.selected_keyframes) >= 3
    for kf in kf_result.selected_keyframes:
        assert kf.primary_reason is not None


def test_3f_31_third_view_2d_3d_association_executes():
    """TEST-3F-31: Production 2D-3D association successfully matches candidate third frame to seed 3D tracks."""
    imgs, K, _ = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_31", payload={"frames": imgs})
    res = pipe.run([in_art])
    assert res.pipeline_status == PipelineStatus.SUCCESS

    sfm_engine = pipe._active_sfm_engine
    assert sfm_engine is not None
    # Verify candidate evaluation diagnostics recorded for candidate third frame
    assert len(sfm_engine.cameras) >= 3


def test_3f_32_third_camera_registered_by_production_pnp():
    """TEST-3F-32: Third camera is registered by production PnP with verified inliers and inlier ratio."""
    imgs, K, _ = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_32", payload={"frames": imgs})
    res = pipe.run([in_art])
    assert res.pipeline_status == PipelineStatus.SUCCESS

    sfm_engine = pipe._active_sfm_engine
    assert sfm_engine is not None
    assert "frame_0002" in sfm_engine.cameras
    cam2 = sfm_engine.cameras["frame_0002"]

    # Verify actual PnP execution statistics on third camera
    assert cam2.is_registered is True
    assert cam2.pnp_inlier_count >= 15
    assert cam2.pnp_inlier_ratio >= 0.25
    # Verify rotation is strictly SO(3)
    np.testing.assert_allclose(cam2.R_cw @ cam2.R_cw.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(cam2.R_cw), 1.0, atol=1e-5)


def test_3f_33_incremental_reconstruction_contains_at_least_three_cameras():
    """TEST-3F-33: Incremental SfM sparse reconstruction result contains >= 3 registered cameras."""
    imgs, K, _ = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_33", payload={"frames": imgs})
    res = pipe.run([in_art])
    assert res.pipeline_status == PipelineStatus.SUCCESS

    sfm_art = pipe.artifacts_by_type["SfMArtifact"]
    recon = sfm_art.payload
    assert recon.total_registered_cameras >= 3
    assert len(recon.camera_poses) >= 3
    assert recon.total_triangulated_points >= 20
    assert recon.status == PipelineStageStatus.SUCCESS


def test_3f_34_bundle_adjustment_includes_all_registered_cameras():
    """TEST-3F-34: Bundle adjustment optimizes all registered cameras (>= 3) while preserving gauge constraints."""
    imgs, K, _ = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_34", payload={"frames": imgs})
    res = pipe.run([in_art])
    assert res.pipeline_status == PipelineStatus.SUCCESS

    ba_art = pipe.artifacts_by_type["BundleAdjustmentArtifact"]
    ba_res = ba_art.payload
    assert ba_res.status == PipelineStageStatus.SUCCESS
    assert ba_res.gauge_preserved is True

    refined_recon = ba_res.refined_reconstruction
    assert len(refined_recon.camera_poses) >= 3

    # Verify gauge constraints preserved over the 3-camera network
    cam0 = refined_recon.camera_poses["frame_0000"]
    cam1 = refined_recon.camera_poses["frame_0001"]
    np.testing.assert_allclose(np.array(cam0.translation_vector), np.zeros(3), atol=1e-5)
    norm_t1 = float(np.linalg.norm(cam1.translation_vector))
    assert np.isclose(norm_t1, 1.0, atol=1e-4)


def test_3f_35_camera_count_growth_is_genuine_not_copied_from_gt():
    """TEST-3F-35: Camera count growth is genuine and camera poses are not copied from ground truth."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_35", payload={"frames": imgs})
    res = pipe.run([in_art])
    assert res.pipeline_status == PipelineStatus.SUCCESS

    sfm_art = pipe.artifacts_by_type["SfMArtifact"]
    recon = sfm_art.payload

    gt_poses = hidden_gt["true_camera_poses"]
    # Reconstructed translations are in arbitrary monocular gauge units, not GT world meters
    for cid in ("frame_0001", "frame_0002"):
        pose = recon.camera_poses[cid]
        assert pose.is_metric is False
        recon_t = np.array(pose.translation_vector)
        # Verify that reconstructed translations are NOT equal to ground-truth translations
        for gt_p in gt_poses:
            gt_t = gt_p["t_cw"]
            assert not np.allclose(recon_t, gt_t, atol=1e-2)


# ===========================================================================
# Phase 3F Steps 9–12 Dense Reconstruction Integration Tests
# ===========================================================================

def test_3f_36_sfm_cameras_to_valid_dense_stereo_pair():
    """TEST-3F-36: SfM camera poses form valid calibrated stereo pairs for dense rectification."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_36", payload={"frames": imgs})
    res = pipe.run([in_art])
    assert res.pipeline_status == PipelineStatus.SUCCESS

    stereo_art = pipe.artifacts_by_type.get("DenseStereoArtifact")
    assert stereo_art is not None
    assert isinstance(stereo_art.payload, dict)
    assert ("frame_0000", "frame_0001") in stereo_art.payload

    st_res = stereo_art.payload[("frame_0000", "frame_0001")]
    assert st_res.rectification.baseline_reconstruction_units > 0.0


def test_3f_37_actual_stereo_images_to_disparity_and_depth():
    """TEST-3F-37: Actual stereo images produce valid disparity and depth."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_37", payload={"frames": imgs})
    pipe.run([in_art])

    stereo_art = pipe.artifacts_by_type["DenseStereoArtifact"]
    st_res = stereo_art.payload[("frame_0000", "frame_0001")]

    valid_mask = st_res.depth_map.valid_mask
    valid_count = int(np.sum(valid_mask))
    assert valid_count > 1000

    depths = st_res.depth_map.depth_array[valid_mask]
    assert np.all(np.isfinite(depths))
    assert np.all(depths > 0.0)

    disp = st_res.disparity_map[valid_mask]
    assert np.all(disp > 0.0)


def test_3f_38_invalid_disparity_rejection():
    """TEST-3F-38: Invalid disparity values are rejected during dense point generation."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_38", payload={"frames": imgs})
    pipe.run([in_art])

    pts_art = pipe.artifacts_by_type["DensePointArtifact"]
    gen_res = pts_art.payload["generation_results"][("frame_0000", "frame_0001")]

    assert gen_res.rejected_points_count > 0
    assert sum(gen_res.rejection_breakdown.values()) == gen_res.rejected_points_count


def test_3f_39_dense_depth_to_world_points():
    """TEST-3F-39: Dense depth maps backproject to valid 3D world points."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_39", payload={"frames": imgs})
    pipe.run([in_art])

    pts_art = pipe.artifacts_by_type["DensePointArtifact"]
    validated_points = pts_art.payload["points"]
    assert len(validated_points) > 1000

    for pt in validated_points[:50]:
        assert pt.world_point.shape == (3,)
        assert np.all(np.isfinite(pt.world_point))
        assert pt.depth > 0.0  # Cheirality: optical Z > 0
        assert pt.camera_point_orig[2] > 0.0


def test_3f_40_dense_point_provenance_preservation():
    """TEST-3F-40: Dense point generation preserves provenance and coordinate contracts."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_40", payload={"frames": imgs})
    pipe.run([in_art])

    pts_art = pipe.artifacts_by_type["DensePointArtifact"]
    assert pts_art.units == "RECONSTRUCTION_UNITS"
    assert pts_art.coordinate_frame == "MONOCULAR_GAUGE"

    sample_obs = pts_art.payload["observations"][0]
    assert sample_obs.reference_frame_id in ("frame_0000", "frame_0001")
    assert sample_obs.depth > 0.0


def test_3f_41_multi_view_fusion_with_distinct_view_support():
    """TEST-3F-41: Multi-view fusion retains points and computes distinct-view support."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_41", payload={"frames": imgs})
    pipe.run([in_art])

    fusion_art = pipe.artifacts_by_type["DenseFusionArtifact"]
    fusion_res = fusion_art.payload
    assert fusion_res.total_fused_points > 0
    assert len(fusion_res.fused_points) == fusion_res.total_fused_points
    assert all(p.distinct_view_count >= 1 for p in fusion_res.fused_points)


def test_3f_42_same_frame_duplicates_do_not_inflate_support():
    """TEST-3F-42: Repeated observations from the same frame do not inflate distinct-view count."""
    from src.geometry.mvs import DensePointObservation, PointVisibilityState
    from src.geometry.dense_fusion import DensePointFusionEngine, DenseFusionConfig, SingleViewRetentionPolicy

    obs1 = DensePointObservation(
        reference_frame_id="frame_0000",
        pixel_coord=(100.0, 100.0),
        depth=10.0,
        world_point=np.array([1.0, 2.0, 10.0], dtype=np.float64),
        confidence=0.8,
        visibility_state=PointVisibilityState.VALID,
    )
    obs2 = DensePointObservation(
        reference_frame_id="frame_0000",  # Same reference frame!
        pixel_coord=(100.2, 100.2),
        depth=10.01,
        world_point=np.array([1.01, 2.01, 10.01], dtype=np.float64),
        confidence=0.85,
        visibility_state=PointVisibilityState.VALID,
    )

    fusion_cfg = DenseFusionConfig(
        spatial_distance_threshold=0.1,
        voxel_grid_resolution=0.1,
        single_view_policy=SingleViewRetentionPolicy.RETAIN_AS_OBSERVED,
        min_distinct_view_support=1,
    )
    engine = DensePointFusionEngine(config=fusion_cfg)
    res = engine.fuse_observations([obs1, obs2])

    assert res.total_fused_points == 1
    fused_pt = res.fused_points[0]
    assert fused_pt.total_observation_count == 2
    assert fused_pt.distinct_view_count == 1  # Crucial: NOT 2!


def test_3f_43_fusion_deterministic_ordering():
    """TEST-3F-43: Dense point fusion maintains deterministic canonical spatial ordering."""
    from src.geometry.mvs import DensePointObservation, PointVisibilityState
    from src.geometry.dense_fusion import DensePointFusionEngine, DenseFusionConfig, SingleViewRetentionPolicy

    coords = [
        [5.0, 1.0, 10.0],
        [1.0, 3.0, 12.0],
        [3.0, 2.0, 11.0],
        [2.0, 4.0, 15.0],
    ]
    obs_list = [
        DensePointObservation(
            reference_frame_id=f"frame_000{i % 2}",
            pixel_coord=(float(i * 10), float(i * 10)),
            depth=float(coords[i][2]),
            world_point=np.array(coords[i], dtype=np.float64),
            confidence=0.8,
            visibility_state=PointVisibilityState.VALID,
        )
        for i in range(len(coords))
    ]

    fusion_cfg = DenseFusionConfig(
        spatial_distance_threshold=0.01,
        single_view_policy=SingleViewRetentionPolicy.RETAIN_AS_OBSERVED,
        min_distinct_view_support=1,
    )
    engine = DensePointFusionEngine(config=fusion_cfg)

    res_1 = engine.fuse_observations(obs_list)
    res_2 = engine.fuse_observations(list(reversed(obs_list)))

    np.testing.assert_allclose(res_1.point_cloud.points, res_2.point_cloud.points, atol=1e-8)


def test_3f_44_fused_dense_cloud_to_valid_surface():
    """TEST-3F-44: Fused dense point cloud reconstructs into a topologically valid surface mesh."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    from src.geometry.surface_reconstruction import SurfaceReconstructionStatus
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_44", payload={"frames": imgs})
    pipe.run([in_art])

    surf_art = pipe.artifacts_by_type["SurfaceArtifact"]
    surf_res = surf_art.payload
    assert surf_res.status == SurfaceReconstructionStatus.SUCCESS
    assert surf_res.mesh is not None

    mesh = surf_res.mesh
    assert mesh.total_vertices > 0
    assert mesh.total_faces > 0
    assert mesh.faces.ndim == 2 and mesh.faces.shape[1] == 3
    assert np.all(mesh.faces < mesh.total_vertices)


def test_3f_45_surface_preserves_reconstruction_units():
    """TEST-3F-45: Surface mesh preserves RECONSTRUCTION_UNITS with is_metric_scale=False."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    from src.geometry.mvs import DepthUnit
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_45", payload={"frames": imgs})
    pipe.run([in_art])

    surf_art = pipe.artifacts_by_type["SurfaceArtifact"]
    assert surf_art.units == "RECONSTRUCTION_UNITS"
    assert surf_art.coordinate_frame == "MONOCULAR_GAUGE"

    mesh = surf_art.payload.mesh
    assert mesh.depth_unit == DepthUnit.RECONSTRUCTION_UNITS
    assert mesh.is_metric_scale is False


def test_3f_46_full_steps_9_to_12_artifact_chain():
    """TEST-3F-46: Full Steps 9–12 artifact chain executes and connects seamlessly."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_46", payload={"frames": imgs})
    res = pipe.run([in_art])
    assert res.pipeline_status == PipelineStatus.SUCCESS

    # Check all four dense reconstruction artifacts exist in registry
    assert "DenseStereoArtifact" in pipe.artifacts_by_type
    assert "DensePointArtifact" in pipe.artifacts_by_type
    assert "DenseFusionArtifact" in pipe.artifacts_by_type
    assert "SurfaceArtifact" in pipe.artifacts_by_type

    records_by_type = {rec.stage_type: rec.status for rec in pipe.stage_records}
    assert records_by_type[PipelineStageType.DENSE_STEREO] == StageStatus.SUCCESS
    assert records_by_type[PipelineStageType.DENSE_POINT_GENERATION] == StageStatus.SUCCESS
    assert records_by_type[PipelineStageType.DENSE_FUSION] == StageStatus.SUCCESS
    assert records_by_type[PipelineStageType.SURFACE_RECONSTRUCTION] == StageStatus.SUCCESS


def test_3f_47_failure_propagation_insufficient_valid_depth():
    """TEST-3F-47: Clean failure propagation when dense stereo produces zero valid depth."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_47", payload={"frames": imgs})

    # Run up to bundle adjustment first
    for stage_type in [
        PipelineStageType.INGESTION,
        PipelineStageType.DECODING,
        PipelineStageType.FRAME_INTELLIGENCE,
        PipelineStageType.KEYFRAME_SELECTION,
        PipelineStageType.CORRESPONDENCE,
        PipelineStageType.TWO_VIEW_GEOMETRY,
        PipelineStageType.INCREMENTAL_SFM,
        PipelineStageType.BUNDLE_ADJUSTMENT,
    ]:
        pipe.execute_stage(stage_type, [in_art])

    # Now simulate blank decoded frames where SGBM cannot find any disparity
    dec_art = pipe.artifacts_by_type["DecodedFramesArtifact"]
    blank_frames = []
    for f in dec_art.payload:
        f_copy = copy.copy(f)
        f_copy.data = np.zeros_like(f.data)
        blank_frames.append(f_copy)
    new_dec_art = DecodedFramesArtifact(
        artifact_id=dec_art.artifact_id,
        payload=blank_frames,
    )
    pipe.artifacts_by_id[dec_art.artifact_id] = new_dec_art
    pipe.artifacts_by_type["DecodedFramesArtifact"] = new_dec_art

    # Attempting to execute dense stereo with blank frames must fail cleanly
    rec = pipe.execute_stage(PipelineStageType.DENSE_STEREO, list(pipe.artifacts_by_id.values()))
    assert rec.status == StageStatus.FAILED


# ===========================================================================
# Steps 13–16 Integration Tests: Texture, Geospatial & Final Validation
# ===========================================================================

from src.geometry.texture_association import (
    SurfaceTextureAssociationMap,
    TextureSampleType,
    DecisionStatus,
    SampleObservationState,
)
from src.geometry.texture_reconstruction import (
    ReconstructedTextureAtlas,
    OperationalTextureState,
)
from src.geospatial.pipeline import GeospatialMetricReconstructionResult
from src.geospatial.synchronization import RawTelemetryRecord
from src.pipeline.artifacts import (
    TextureAssociationArtifact,
    TexturedSurfaceArtifact,
    TelemetryArtifact,
    GeospatialArtifact,
    ValidationArtifact,
)


def test_3f_48_surface_to_texture_association():
    """TEST-3F-48: Surface mesh and registered camera frames produce a valid TextureAssociationArtifact."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_48", payload={"frames": imgs})
    pipe.run([in_art])

    assert "TextureAssociationArtifact" in pipe.artifacts_by_type
    assoc_art = pipe.artifacts_by_type["TextureAssociationArtifact"]
    assert assoc_art.units == "DIMENSIONLESS"
    assert assoc_art.coordinate_frame == "SURFACE_LOCAL"

    assoc_map = assoc_art.payload
    assert isinstance(assoc_map, SurfaceTextureAssociationMap)
    assert assoc_map.total_samples > 0
    assert assoc_map.sample_coverage_ratio > 0.0
    assert len(assoc_map.observations_by_sample) > 0


def test_3f_49_visibility_gated_texture_observation():
    """TEST-3F-49: Visibility gating ensures texture observations have positive depth, valid bounds, and auditable decisions."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_49", payload={"frames": imgs})
    pipe.run([in_art])

    assoc_map: SurfaceTextureAssociationMap = pipe.artifacts_by_type["TextureAssociationArtifact"].payload
    assert len(assoc_map.decision_records) > 0

    # Verify decision audit records
    decision_types = {r.decision for r in assoc_map.decision_records}
    assert DecisionStatus.ACCEPTED_RETAINED in decision_types or DecisionStatus.ACCEPTED_NOT_RETAINED in decision_types

    # Verify all retained observations satisfy geometric bounds
    observed_count = 0
    for s_idx, obs_list in assoc_map.observations_by_sample.items():
        for obs in obs_list:
            observed_count += 1
            assert obs.depth > 0.0
            assert obs.frame_id in ("frame_0000", "frame_0001", "frame_0002", "frame_0", "frame_1", "frame_2")
            assert 0.0 <= obs.pixel_coords[0] <= 640.0
            assert 0.0 <= obs.pixel_coords[1] <= 480.0
            assert obs.composite_score >= 0.0
    assert observed_count > 0


def test_3f_50_surface_to_texture_atlas():
    """TEST-3F-50: Texture association produces a multi-view reconstructed texture atlas."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_50", payload={"frames": imgs})
    pipe.run([in_art])

    assert "TexturedSurfaceArtifact" in pipe.artifacts_by_type
    atlas_art = pipe.artifacts_by_type["TexturedSurfaceArtifact"]
    assert atlas_art.units == "RECONSTRUCTION_UNITS"
    assert atlas_art.coordinate_frame == "MONOCULAR_GAUGE"

    atlas = atlas_art.payload
    assert isinstance(atlas, ReconstructedTextureAtlas)
    assert atlas.total_surface_texels > 0
    assert atlas.observed_texel_ratio > 0.0
    assert atlas.albedo_atlas.ndim == 3 and atlas.albedo_atlas.shape[2] == 3
    assert atlas.alpha_atlas.shape == (cfg.texture_reconstruction_config.atlas_height, cfg.texture_reconstruction_config.atlas_width)


def test_3f_51_unobserved_texel_remains_unobserved():
    """TEST-3F-51: Unobserved regions are strictly not hallucinated; zero alpha and zero confidence."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_51", payload={"frames": imgs})
    pipe.run([in_art])

    atlas: ReconstructedTextureAtlas = pipe.artifacts_by_type["TexturedSurfaceArtifact"].payload
    unobserved_mask = (atlas.state_atlas == OperationalTextureState.UNOBSERVED.value)

    # In unobserved pixels, alpha and confidence must be zero
    if np.any(unobserved_mask):
        assert np.all(atlas.alpha_atlas[unobserved_mask] == 0.0)
        assert np.all(atlas.confidence_atlas[unobserved_mask] == 0.0)


def test_3f_52_texture_provenance_propagation():
    """TEST-3F-52: Observed texels contain complete mathematical provenance with contributing frame IDs."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_52", payload={"frames": imgs})
    pipe.run([in_art])

    atlas: ReconstructedTextureAtlas = pipe.artifacts_by_type["TexturedSurfaceArtifact"].payload
    assert len(atlas.texel_provenance) > 0

    # Inspect a sampled texel provenance
    sample_coord = next(iter(atlas.texel_provenance))
    prov = atlas.texel_provenance[sample_coord]
    assert prov.face_idx >= 0
    assert len(prov.barycentric_coords) == 3
    assert prov.fusion_method == "tukey_biweight_v1"
    for frame_id in prov.contributing_frames:
        assert frame_id in ("frame_0000", "frame_0001", "frame_0002", "frame_0", "frame_1", "frame_2")


def test_3f_53_no_telemetry_geospatial_not_evaluable():
    """TEST-3F-53: Absent telemetry causes geospatial stage to report NOT_EVALUABLE with monocular gauge units."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_53", payload={"frames": imgs})
    pipe.run([in_art])

    assert "GeospatialArtifact" in pipe.artifacts_by_type
    geo_art = pipe.artifacts_by_type["GeospatialArtifact"]
    assert geo_art.units == "RECONSTRUCTION_UNITS"
    assert geo_art.coordinate_frame == "MONOCULAR_GAUGE"

    geo_records = [r for r in pipe.stage_records if r.stage_type == PipelineStageType.GEOSPATIAL_TRANSFORM]
    assert len(geo_records) == 1
    assert geo_records[0].status == StageStatus.NOT_EVALUABLE


def test_3f_54_metric_scale_remains_ambiguous_without_evidence():
    """TEST-3F-54: Metric scale status strictly remains SCALE_AMBIGUOUS without independent physical evidence."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_54", payload={"frames": imgs})
    res = pipe.run([in_art])

    assert res.metric_scale_status == MetricScaleStatus.SCALE_AMBIGUOUS
    assert res.reconstruction_units == ReconstructionUnitType.RECONSTRUCTION_UNITS


def test_3f_55_valid_telemetry_geospatial_stage_execution():
    """TEST-3F-55: Providing independent telemetry records executes the geospatial stage."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_55", payload={"frames": imgs})

    # Create independent telemetry epoch records bracketing image triggers
    telemetry_records = [
        RawTelemetryRecord(
            timestamp_seconds=0.0,
            latitude_deg=37.7749,
            longitude_deg=-122.4194,
            altitude_m=100.0,
            horizontal_accuracy_m=0.5,
            vertical_accuracy_m=1.0,
        ),
        RawTelemetryRecord(
            timestamp_seconds=0.1,
            latitude_deg=37.7750,
            longitude_deg=-122.4193,
            altitude_m=101.0,
            horizontal_accuracy_m=0.5,
            vertical_accuracy_m=1.0,
        ),
        RawTelemetryRecord(
            timestamp_seconds=0.2,
            latitude_deg=37.7751,
            longitude_deg=-122.4192,
            altitude_m=102.0,
            horizontal_accuracy_m=0.5,
            vertical_accuracy_m=1.0,
        ),
        RawTelemetryRecord(
            timestamp_seconds=0.3,
            latitude_deg=37.7752,
            longitude_deg=-122.4191,
            altitude_m=103.0,
            horizontal_accuracy_m=0.5,
            vertical_accuracy_m=1.0,
        ),
    ]
    tel_art = TelemetryArtifact(artifact_id="art_tel_55", payload=telemetry_records)

    res = pipe.run([in_art, tel_art])
    assert res.pipeline_status == PipelineStatus.SUCCESS

    geo_records = [r for r in pipe.stage_records if r.stage_type == PipelineStageType.GEOSPATIAL_TRANSFORM]
    assert len(geo_records) == 1
    assert geo_records[0].status == StageStatus.SUCCESS

    geo_art = pipe.artifacts_by_type["GeospatialArtifact"]
    assert isinstance(geo_art.payload, GeospatialMetricReconstructionResult)
    assert geo_art.payload.anchor_origin is not None


def test_3f_56_final_validation_artifact():
    """TEST-3F-56: Final validation stage produces a certified ValidationArtifact with auditable claim matrix."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_56", payload={"frames": imgs})
    pipe.run([in_art])

    assert "ValidationArtifact" in pipe.artifacts_by_type
    val_art = pipe.artifacts_by_type["ValidationArtifact"]
    val_data = val_art.payload

    assert val_data["validation_status"] == "PASS"
    assert val_data["geometric_status"] == "SUCCESS"
    assert val_data["dense_status"] == "SUCCESS"
    assert val_data["surface_status"] == "SUCCESS"
    assert val_data["texture_status"] == "SUCCESS"
    assert val_data["evidence_level"] == "LEVEL_0_NO_GROUND_TRUTH"
    assert "reprojection_consistency" in val_data["evaluable_axes"]
    assert "universal_drone_accuracy" in val_data["blocked_claims"]
    assert "radiometric_color_accuracy" in val_data["blocked_claims"]


def test_3f_57_final_reconstruction_artifact_provenance():
    """TEST-3F-57: Final deliverable preserves separated statuses and does not collapse into a single accuracy score."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_57", payload={"frames": imgs})
    pipe.run([in_art])

    assert "FinalReconstructionArtifact" in pipe.artifacts_by_type
    final_art = pipe.artifacts_by_type["FinalReconstructionArtifact"]
    p = final_art.payload

    # Must contain distinct, uncollapsed status dimensions
    assert "geometric_status" in p
    assert "dense_status" in p
    assert "surface_status" in p
    assert "texture_status" in p
    assert "metric_scale_status" in p
    assert "geospatial_status" in p
    assert "validation_status" in p
    assert "evidence_level" in p
    assert "evaluable_axes" in p
    assert "blocked_claims" in p
    assert "provenance" in p
    assert "accuracy_score" not in p  # Strictly forbidden to collapse into one score


def test_3f_58_full_steps_1_to_16_orchestration():
    """TEST-3F-58: Full pipeline executes all 17 DAG stages from ingestion to final deliverable."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_58", payload={"frames": imgs})
    res = pipe.run([in_art])

    assert res.pipeline_status == PipelineStatus.SUCCESS
    assert len(res.stage_records) == 17

    executed_stages = [rec.stage_type for rec in res.stage_records]
    assert executed_stages == ReconstructionPipeline.DAG_SEQUENCE

    # Verify all expected artifacts exist in registry
    assert "CanonicalTimelineArtifact" in pipe.artifacts_by_type
    assert "DecodedFramesArtifact" in pipe.artifacts_by_type
    assert "FrameQualityArtifact" in pipe.artifacts_by_type
    assert "KeyframeSetArtifact" in pipe.artifacts_by_type
    assert "CorrespondenceArtifact" in pipe.artifacts_by_type
    assert "TwoViewGeometryArtifact" in pipe.artifacts_by_type
    assert "SfMArtifact" in pipe.artifacts_by_type
    assert "BundleAdjustmentArtifact" in pipe.artifacts_by_type
    assert "DenseStereoArtifact" in pipe.artifacts_by_type
    assert "DensePointArtifact" in pipe.artifacts_by_type
    assert "DenseFusionArtifact" in pipe.artifacts_by_type
    assert "SurfaceArtifact" in pipe.artifacts_by_type
    assert "TextureAssociationArtifact" in pipe.artifacts_by_type
    assert "TexturedSurfaceArtifact" in pipe.artifacts_by_type
    assert "GeospatialArtifact" in pipe.artifacts_by_type
    assert "ValidationArtifact" in pipe.artifacts_by_type
    assert "FinalReconstructionArtifact" in pipe.artifacts_by_type


def test_3f_59_conditional_stages_do_not_falsely_become_success():
    """TEST-3F-59: Disabled conditional stages remain NOT_EVALUABLE and do not report false SUCCESS."""
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
        enable_dense_stereo=False,
        enable_dense_point_generation=False,
        enable_dense_fusion=False,
        enable_surface_meshing=False,
        enable_texturing=False,
        enable_geospatial=False,
    )
    pipe = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_59", payload={"frames": imgs})
    res = pipe.run([in_art])

    status_by_stage = {rec.stage_type: rec.status for rec in res.stage_records}
    assert status_by_stage[PipelineStageType.DENSE_STEREO] == StageStatus.NOT_EVALUABLE
    assert status_by_stage[PipelineStageType.DENSE_POINT_GENERATION] == StageStatus.NOT_EVALUABLE
    assert status_by_stage[PipelineStageType.DENSE_FUSION] == StageStatus.NOT_EVALUABLE
    assert status_by_stage[PipelineStageType.SURFACE_RECONSTRUCTION] == StageStatus.NOT_EVALUABLE
    assert status_by_stage[PipelineStageType.TEXTURE_ASSOCIATION] == StageStatus.NOT_EVALUABLE
    assert status_by_stage[PipelineStageType.TEXTURE_RECONSTRUCTION] == StageStatus.NOT_EVALUABLE
    assert status_by_stage[PipelineStageType.GEOSPATIAL_TRANSFORM] == StageStatus.NOT_EVALUABLE


def test_3f_60_visible_triangle_reversed_normal():
    """TEST-3F-60: VISIBLE_TRIANGLE_REVERSED_NORMAL

    Verifies that when a triangle is unoccluded and geometrically within the sensor frustum,
    reversing its normal vector (such that n · v <= 0) does NOT trigger hard back-face culling.
    Ray visibility remains geometrically valid, observation is accepted, and provenance is recorded.
    """
    from src.geometry.surface_reconstruction import SurfaceMesh
    from src.geometry.mvs import DepthUnit
    from src.geometry.texture_association import (
        VisibilityAwareTextureAssociator,
        TextureSourceCamera,
        TextureSampleType,
        SampleObservationState,
        DecisionStatus,
        TextureQueryStatus,
    )
    from src.geometry.contracts import ExtrinsicPose
    from src.pipeline.artifacts import DecodedFramesArtifact, SurfaceArtifact, SfMArtifact
    from src.preprocessing.decoder import DecodedFrame
    from tests.unit.test_phase3e4_step3_texture_association import _create_canonical_camera

    # Construct a single triangle on XY plane at z = 0.0
    vertices = np.array([
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    # REVERSED NORMAL: Points downwards [0, 0, -1] away from camera at [0, 0, 2]
    # Viewing vector v_view from facet to camera points along +Z [0, 0, 1]
    # Hence n · v_view = -1.0 <= 0 (back-facing relative to camera)
    face_normals_reversed = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
    vertex_normals_reversed = np.array([
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)

    mesh_reversed = SurfaceMesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_normals_reversed,
        face_normals=face_normals_reversed,
        vertex_confidences=np.ones(3, dtype=np.float32),
        vertex_support_counts=np.full(3, 3, dtype=np.int32),
        face_support_scores=np.ones(1, dtype=np.float32),
        face_areas=np.ones(1, dtype=np.float64),
        is_boundary_vertex=np.zeros(3, dtype=bool),
        is_boundary_face=np.zeros(1, dtype=bool),
        total_vertices=3,
        total_faces=1,
        depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
        is_metric_scale=False,
    )

    # Camera at [0, 0, 2.0] looking down at [0, 0, 0]
    cam = _create_canonical_camera(
        frame_id="frame_0000",
        C_w=np.array([0.0, 0.0, 2.0]),
        target=np.array([0.0, 0.0, 0.0]),
        quality_metrics={"sharpness": 1.0, "blur": 0.0, "exposure": 1.0, "dynamic_risk": 0.0},
    )
    cameras = {"frame_0000": cam}

    # Evaluate texture association on reversed-normal mesh directly
    associator = VisibilityAwareTextureAssociator()
    assoc_map_reversed = associator.associate_texture(
        mesh=mesh_reversed,
        cameras=cameras,
        sample_type=TextureSampleType.FACET_CENTROID,
    )

    # 1. Ray visibility remains geometrically valid (not culled!)
    assert assoc_map_reversed.sample_states[0] == SampleObservationState.OBSERVED
    assert len(assoc_map_reversed.observations_by_sample[0]) == 1

    # 2. Decision status is ACCEPTED_RETAINED, query_status is VISIBLE (NOT REJECTED)
    decision = assoc_map_reversed.decision_records[0]
    assert decision.decision == DecisionStatus.ACCEPTED_RETAINED
    assert decision.query_status == TextureQueryStatus.VISIBLE
    assert decision.rejection_reason is None

    # 3. Provenance remains valid
    best_obs = assoc_map_reversed.best_observation_by_sample[0]
    assert best_obs is not None
    assert best_obs.frame_id == "frame_0000"
    assert best_obs.sample_type == TextureSampleType.FACET_CENTROID
    assert best_obs.sample_index == 0

    # 4. Also verify via ReconstructionPipeline orchestrator Stage 13 execution
    pipe = ReconstructionPipeline()
    from src.preprocessing.decoder import DecodeStatus
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dec_frame = DecodedFrame(
        frame_id="frame_0000",
        frame_index=0,
        timestamp_seconds=0.0,
        width=640,
        height=480,
        channels=3,
        channel_layout="RGB",
        dtype="uint8",
        data=dummy_img,
        source_video="synthetic",
        decode_status=DecodeStatus.SUCCESS,
    )
    pipe.artifacts_by_type["DecodedFramesArtifact"] = DecodedFramesArtifact("art_dec_test60", [dec_frame])
    pipe.artifacts_by_type["SurfaceArtifact"] = SurfaceArtifact("art_surf_test60", mesh_reversed)
    R_cw = cam.R_cw
    t_cw = cam.t_cw
    C_w = -R_cw.T @ t_cw
    pose = ExtrinsicPose(rotation_matrix=R_cw.T.tolist(), translation_vector=C_w.tolist())

    class MockSfM:
        camera_poses = {"frame_0000": pose}

    pipe.artifacts_by_type["SfMArtifact"] = SfMArtifact("art_sfm_test60", MockSfM())

    stg_status = pipe._execute_texture_association()
    assert stg_status == StageStatus.SUCCESS
    assoc_art = pipe.artifacts_by_type["TextureAssociationArtifact"]
    assert assoc_art.payload.sample_states[0] == SampleObservationState.OBSERVED
