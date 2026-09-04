"""Phase 3F: Adversarial Forensic Mutation Tests for Steps 1–4.

Validates that:
- MUT-3F-00: Mutating payload after artifact creation causes verify_integrity() failure.
- MUT-3F-04: A failed stage cannot be recorded as SUCCESS.
- MUT-3F-08: Manual artifact hash tampering is caught by production verification.
- MUT-3F-10: Evaluation-truth artifacts cannot cross into the reconstruction domain.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.models import (
    PipelineStageType,
    StageStatus,
    PipelineStatus,
)
from src.pipeline.errors import (
    ContractViolationError,
    DataLeakageError,
)
from src.pipeline.artifacts import (
    PipelineArtifact,
    ArtifactDomain,
    VideoArtifact,
    SfMArtifact,
    DenseStereoArtifact,
    SurfaceArtifact,
)
from src.pipeline.config import PipelineConfig
from src.pipeline.orchestrator import ReconstructionPipeline


def test_mut_3f_00_mutate_payload_after_creation_detected():
    """MUT-3F-00: Mutating payload data post-creation MUST trigger ContractViolationError on integrity check."""
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    artifact = SfMArtifact(artifact_id="art_sfm_mut00", payload=pts)

    # Clean verification
    artifact.verify_integrity()

    # Mutation: Cheat by modifying point coordinates in-place
    pts[0, 1] = 999.0

    # Production code must catch this modification
    with pytest.raises(ContractViolationError, match="Artifact tampering detected"):
        artifact.verify_integrity()


def test_mut_3f_04_failed_stage_cannot_be_marked_success():
    """MUT-3F-04: Production orchestrator must not allow a failed stage execution to report SUCCESS."""
    pipe = ReconstructionPipeline()

    # Corrupt artifact causes consumption check to raise ContractViolationError
    corrupted_art = SfMArtifact(artifact_id="art_corrupt", payload={"data": [10, 20]})
    corrupted_art.content_hash = "0000000000000000000000000000000000000000000000000000000000000000"

    res = pipe.run([corrupted_art])

    # The pipeline must NOT mark SUCCESS
    assert res.pipeline_status != PipelineStatus.SUCCESS
    assert res.pipeline_status in (PipelineStatus.CONTRACT_VIOLATION, PipelineStatus.FAILED)

    # Any failed stage record must have a non-SUCCESS status
    failed_records = [rec for rec in res.stage_records if rec.status != StageStatus.SUCCESS]
    assert len(failed_records) > 0
    for rec in failed_records:
        assert rec.status in (StageStatus.FAILED, StageStatus.CONTRACT_VIOLATION, StageStatus.NOT_EVALUABLE)


def test_mut_3f_08_artifact_hash_tampering_detected():
    """MUT-3F-08: Tampering with the recorded content_hash string is intercepted by verify_integrity()."""
    artifact = DenseStereoArtifact(
        artifact_id="art_stereo_mut08",
        payload={"disparity_shape": [100, 100], "min_disp": 0, "max_disp": 64},
    )
    artifact.verify_integrity()

    # Mutation: Alter the content_hash to forge an identity
    artifact.content_hash = "deadbeef" * 8

    # Production verify_integrity must reject the forged hash
    with pytest.raises(ContractViolationError, match="Artifact tampering detected"):
        artifact.verify_integrity()


def test_mut_3f_10_evaluation_artifact_cannot_cross_reconstruction_domain():
    """MUT-3F-10: Feeding an EVALUATION_TRUTH artifact into the pipeline is intercepted by DataLeakageError."""
    hidden_cad = PipelineArtifact(
        artifact_id="art_hidden_cad_mut10",
        artifact_type="CADTruthModel",
        domain=ArtifactDomain.EVALUATION_TRUTH,
        producer_stage="CAD_GROUND_TRUTH_GENERATOR",
        input_artifact_ids=[],
        units="METRES",
        coordinate_frame="CAD_LOCAL",
        payload={"mesh_vertices": np.zeros((100, 3)), "mesh_faces": np.zeros((50, 3))},
    )

    pipe = ReconstructionPipeline()
    # Orchestrator must reject this artifact at the reconstruction boundary
    with pytest.raises(DataLeakageError, match="Data Leakage Violation"):
        pipe.run([hidden_cad])


def test_mut_3f_01_inject_hidden_true_camera_pose_rejected():
    """MUT-3F-01: Injecting hidden true camera pose into SfM stage is intercepted by DataLeakageError."""
    hidden_pose_artifact = PipelineArtifact(
        artifact_id="art_true_poses_mut01",
        artifact_type="CameraTrajectoryTruth",
        domain=ArtifactDomain.EVALUATION_TRUTH,
        producer_stage="GROUND_TRUTH_TRAJECTORY_GENERATOR",
        input_artifact_ids=[],
        units="METRES",
        coordinate_frame="GROUND_TRUTH_WORLD",
        payload={"true_camera_poses": {"cam_00": np.eye(4), "cam_01": np.eye(4)}},
    )

    pipe = ReconstructionPipeline()
    with pytest.raises(DataLeakageError, match="Data Leakage Violation"):
        pipe.verify_input_artifact(hidden_pose_artifact, PipelineStageType.INCREMENTAL_SFM)


def test_mut_3f_02_inject_true_depth_or_landmarks_rejected():
    """MUT-3F-02: Injecting true depth or ground-truth 3D landmarks into reconstruction is rejected."""
    # Attempting to construct an artifact containing forbidden evaluation keys triggers anti-leakage check
    with pytest.raises(DataLeakageError, match="Privileged evaluation key"):
        PipelineArtifact(
            artifact_id="art_gt_landmarks_mut02",
            artifact_type="DenseDepthArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_OUTPUT,
            producer_stage="DENSE_STEREO",
            input_artifact_ids=[],
            units="METRES",
            coordinate_frame="CAMERA",
            payload={"depth_map": np.ones((100, 100)), "true_depth": np.ones((100, 100))},
        )


def test_mut_3f_05_replace_sfm_output_with_hidden_gt_landmarks_rejected():
    """MUT-3F-05: Replacing SfM output with hidden ground-truth landmarks fails provenance/domain check."""
    gt_sfm_artifact = PipelineArtifact(
        artifact_id="art_fake_sfm_mut05",
        artifact_type="SfMArtifact",
        domain=ArtifactDomain.EVALUATION_TRUTH,
        producer_stage="SYNTHETIC_SCENE_RENDERER",
        input_artifact_ids=[],
        units="METRES",
        coordinate_frame="WORLD_TRUE",
        payload={"synthetic_gt_landmarks": np.random.uniform(0, 10, size=(100, 3))},
    )

    pipe = ReconstructionPipeline()
    with pytest.raises(DataLeakageError, match="Data Leakage Violation"):
        pipe.verify_input_artifact(gt_sfm_artifact, PipelineStageType.BUNDLE_ADJUSTMENT)


def test_mut_3f_09_non_monotonic_frame_timestamps_rejected():
    """MUT-3F-09: Frame inputs with non-monotonic timestamps trigger temporal-order ContractViolationError."""
    from src.preprocessing.decoder import DecodedFrame, DecodeStatus
    from src.pipeline.artifacts import VideoArtifact

    frame_0 = DecodedFrame(
        frame_id="frame_0000",
        frame_index=0,
        timestamp_seconds=1.0,  # Later timestamp first
        width=640,
        height=480,
        channels=3,
        channel_layout="RGB",
        dtype="uint8",
        data=np.zeros((480, 640, 3), dtype=np.uint8),
        source_video="input",
        decode_status=DecodeStatus.SUCCESS,
    )
    frame_1 = DecodedFrame(
        frame_id="frame_0001",
        frame_index=1,
        timestamp_seconds=0.5,  # Retrograde timestamp
        width=640,
        height=480,
        channels=3,
        channel_layout="RGB",
        dtype="uint8",
        data=np.zeros((480, 640, 3), dtype=np.uint8),
        source_video="input",
        decode_status=DecodeStatus.SUCCESS,
    )

    in_art = VideoArtifact(artifact_id="art_non_monotonic_vid", payload={"frames": [frame_0, frame_1]})
    pipe = ReconstructionPipeline()

    with pytest.raises(ContractViolationError, match="Chronological PTS ordering violation"):
        pipe.execute_stage(PipelineStageType.INGESTION, [in_art])


def test_mut_3f_11_fake_sfm_artifact_with_evaluation_truth_rejected():
    """MUT-3F-11: An SfM artifact tagged with EVALUATION_TRUTH domain is rejected by Bundle Adjustment."""
    fake_sfm = PipelineArtifact(
        artifact_id="art_sfm_eval_truth_mut11",
        artifact_type="SfMArtifact",
        domain=ArtifactDomain.EVALUATION_TRUTH,
        producer_stage="INCREMENTAL_SFM",
        input_artifact_ids=[],
        units="METRES",
        coordinate_frame="MODEL",
        payload={"camera_poses": {}, "points3d": {}},
    )

    pipe = ReconstructionPipeline()
    with pytest.raises(DataLeakageError, match="Data Leakage Violation"):
        pipe.execute_stage(PipelineStageType.BUNDLE_ADJUSTMENT, [fake_sfm])


def test_mut_3f_03_inject_hidden_ground_truth_mesh_rejected():
    """MUT-3F-03: Injecting hidden ground-truth mesh/geometry into surface input boundary is rejected."""
    # Attempting to construct a reconstruction artifact containing privileged CAD/true mesh
    with pytest.raises(DataLeakageError, match="Privileged evaluation key 'true_mesh'"):
        PipelineArtifact(
            artifact_id="art_gt_mesh_mut03",
            artifact_type="SurfaceArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_OUTPUT,
            producer_stage="SURFACE_RECONSTRUCTION",
            input_artifact_ids=[],
            units="METRES",
            coordinate_frame="WORLD",
            payload={"vertices": np.zeros((10, 3)), "true_mesh": np.zeros((10, 3))},
        )

    # Also test injecting an EVALUATION_TRUTH CAD mesh into the orchestrator surface input
    cad_mesh_art = PipelineArtifact(
        artifact_id="art_cad_mesh_eval",
        artifact_type="SurfaceArtifact",
        domain=ArtifactDomain.EVALUATION_TRUTH,
        producer_stage="CAD_GENERATOR",
        input_artifact_ids=[],
        units="METRES",
        coordinate_frame="CAD_WORLD",
        payload={"mesh_vertices": np.zeros((20, 3))},
    )
    pipe = ReconstructionPipeline()
    with pytest.raises(DataLeakageError, match="Data Leakage Violation"):
        pipe.verify_input_artifact(cad_mesh_art, PipelineStageType.SURFACE_RECONSTRUCTION)


def test_mut_3f_06_convert_reconstruction_units_to_metres_without_evidence_rejected():
    """MUT-3F-06: Attempting to declare METRES on an artifact with monocular scale ambiguity triggers ContractViolationError."""
    # Scale-ambiguous SfM payload cannot be fraudulently tagged with metric units 'METRES'
    scale_ambiguous_payload = {
        "camera_poses": {},
        "points3d": {},
        "is_metric_scale": False,
        "has_monocular_scale_ambiguity": True,
    }

    with pytest.raises(ContractViolationError, match="Cannot declare metric units 'METRES'"):
        PipelineArtifact(
            artifact_id="art_fraudulent_metric_mut06",
            artifact_type="SfMArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_OUTPUT,
            producer_stage="INCREMENTAL_SFM",
            input_artifact_ids=[],
            units="METRES",  # Fraudulent metric claim
            coordinate_frame="MODEL",
            payload=scale_ambiguous_payload,
        )


def test_mut_3f_07_treat_gnss_residual_as_independent_accuracy_rejected():
    """MUT-3F-07: Attempting to claim surveyed/independent checkpoint accuracy from GNSS fitting residuals triggers ClaimPolicy ContractViolationError."""
    from src.benchmark.claim_policy import ClaimPolicyEngine
    from src.benchmark.models import EvidenceLevel

    # Under telemetry-only evidence (LEVEL_1), claiming surveyed horizontal checkpoint RMSE is strictly prohibited
    with pytest.raises(Exception, match="Claim Policy Violation"):
        ClaimPolicyEngine.enforce_claim_emission(
            evidence_level=EvidenceLevel.LEVEL_1_TELEMETRY_ONLY,
            claim_key="horizontal_checkpoint_rmse",
            independent_reference_available=False,
            holdout_enforced=False,
        )

    # Claiming surveyed 3D accuracy without surveyed ground truth is also strictly blocked
    with pytest.raises(Exception, match="Claim Policy Violation"):
        ClaimPolicyEngine.enforce_claim_emission(
            evidence_level=EvidenceLevel.LEVEL_1_TELEMETRY_ONLY,
            claim_key="surveyed_3d_accuracy",
            independent_reference_available=False,
        )


def test_mut_3f_12_inject_hidden_third_camera_pose_rejected():
    """MUT-3F-12: Attempting to inject hidden third-camera pose during camera registration triggers DataLeakageError."""
    from tests.integration.synthetic_scene_fixture import generate_synthetic_multiview_dataset
    imgs, K, hidden_gt = generate_synthetic_multiview_dataset(n_views=3)
    hidden_poses = hidden_gt["true_camera_poses"]

    # 1. Attempt to sneak true camera poses into reconstruction artifact payload
    with pytest.raises(DataLeakageError, match="Privileged evaluation key 'true_camera_poses'"):
        PipelineArtifact(
            artifact_id="art_leaked_cam2_pose",
            artifact_type="SfMArtifact",
            domain=ArtifactDomain.RECONSTRUCTION_OUTPUT,
            producer_stage="INCREMENTAL_SFM",
            input_artifact_ids=[],
            units="RECONSTRUCTION_UNITS",
            coordinate_frame="MODEL",
            payload={"true_camera_poses": hidden_poses},
        )

    # 2. Attempt to feed an EVALUATION_TRUTH artifact carrying true third camera pose into Incremental SfM
    gt_cam_art = PipelineArtifact(
        artifact_id="art_cam2_truth",
        artifact_type="GroundTruthPoseArtifact",
        domain=ArtifactDomain.EVALUATION_TRUTH,
        producer_stage="SIMULATOR",
        input_artifact_ids=[],
        units="METRES",
        coordinate_frame="WORLD",
        payload={"cam2_pose": hidden_poses[2]},
    )
    pipe = ReconstructionPipeline()
    with pytest.raises(DataLeakageError, match="Data Leakage Violation"):
        pipe.verify_input_artifact(gt_cam_art, PipelineStageType.INCREMENTAL_SFM)

    # 3. Verify hidden pose values are strictly absent from normal 3-view reconstruction artifacts
    from src.quality.keyframe_selection import KeyframeSelectionConfig
    cfg = PipelineConfig(
        default_intrinsics=K,
        keyframe_config=KeyframeSelectionConfig(max_temporal_gap_seconds=0.4),
    )
    pipe_clean = ReconstructionPipeline(config=cfg)
    in_art = VideoArtifact(artifact_id="art_vid_clean", payload={"frames": imgs})
    pipe_clean.run([in_art])

    for art_id, art in pipe_clean.artifacts_by_id.items():
        assert art.domain != ArtifactDomain.EVALUATION_TRUTH
        # Check payload does not contain true poses
        if isinstance(art.payload, dict):
            for k in art.payload.keys():
                k_str = str(k).lower()
                assert "true_camera_pose" not in k_str
                assert "ground_truth" not in k_str


# ===========================================================================
# Phase 3F Steps 9–12 Dense Reconstruction Forensic Mutations
# ===========================================================================

def test_mut_3f_13_inject_hidden_ground_truth_depth_rejected():
    """MUT-3F-13: Attempt to inject hidden ground-truth depth into dense stereo output must trigger DataLeakageError."""
    # Attempt via metadata
    with pytest.raises(DataLeakageError, match="Contract Violation: Privileged evaluation key"):
        DenseStereoArtifact(
            artifact_id="art_mut_stereo_gt1",
            payload={"disparity": np.zeros((10, 10))},
            metadata={"true_depth": np.ones((10, 10))},
        )

    # Attempt via payload dict key
    with pytest.raises(DataLeakageError, match="Contract Violation: Privileged evaluation key"):
        DenseStereoArtifact(
            artifact_id="art_mut_stereo_gt2",
            payload={"true_depth_maps": np.ones((10, 10))},
        )


def test_mut_3f_14_inject_gt_camera_pose_into_stereo_rectification_rejected():
    """MUT-3F-14: Attempt to inject GT camera poses into stereo rectification pipeline must trigger DataLeakageError."""
    with pytest.raises(DataLeakageError, match="Contract Violation: Privileged evaluation key"):
        DenseStereoArtifact(
            artifact_id="art_mut_stereo_pose",
            payload={"rectified_pairs": []},
            metadata={"ground_truth_poses": [{"R": np.eye(3), "t": np.zeros(3)}]},
        )


def test_mut_3f_15_inject_gt_dense_point_cloud_rejected():
    """MUT-3F-15: Attempt to inject GT dense points into DensePointArtifact must trigger DataLeakageError."""
    from src.pipeline.artifacts import DensePointArtifact
    with pytest.raises(DataLeakageError, match="Contract Violation: Privileged evaluation key"):
        DensePointArtifact(
            artifact_id="art_mut_points_gt",
            payload={"ground_truth_points": np.ones((100, 3))},
        )

    with pytest.raises(DataLeakageError, match="Contract Violation: Privileged evaluation key"):
        DensePointArtifact(
            artifact_id="art_mut_points_cad",
            payload={"cad_points": np.ones((100, 3))},
        )


def test_mut_3f_16_inject_gt_mesh_into_surface_reconstruction_rejected():
    """MUT-3F-16: Attempt to inject GT mesh into SurfaceArtifact must trigger DataLeakageError."""
    with pytest.raises(DataLeakageError, match="Contract Violation: Privileged evaluation key"):
        SurfaceArtifact(
            artifact_id="art_mut_surface_gt",
            payload={"ground_truth_mesh": {"vertices": np.zeros((10, 3)), "faces": np.zeros((5, 3))}},
        )

    with pytest.raises(DataLeakageError, match="Contract Violation: Privileged evaluation key"):
        SurfaceArtifact(
            artifact_id="art_mut_surface_true",
            payload={"true_mesh": "hidden_cad_mesh"},
        )


def test_mut_3f_17_declare_dense_output_as_metres_without_evidence_rejected():
    """MUT-3F-17: Declaring dense point or surface output as METRES without certified metric evidence must fail."""
    from src.pipeline.artifacts import DensePointArtifact
    from src.geometry.mvs import DepthUnit
    from src.geometry.surface_reconstruction import SurfaceMesh

    # Attempt to create DensePointArtifact declaring METRES with unscaled points
    with pytest.raises(ContractViolationError, match="Cannot declare metric units"):
        DensePointArtifact(
            artifact_id="art_mut_dense_meters",
            payload={
                "points": np.random.randn(50, 3),
                "is_metric_scale": False,
            },
            units="METRES",
        )

    # Attempt to create SurfaceArtifact declaring METRES when mesh has is_metric_scale=False
    V = 4
    F = 2
    mock_mesh = SurfaceMesh(
        vertices=np.zeros((V, 3), dtype=np.float64),
        faces=np.zeros((F, 3), dtype=np.int32),
        vertex_normals=None,
        face_normals=None,
        vertex_confidences=np.ones(V, dtype=np.float32),
        vertex_support_counts=np.ones(V, dtype=np.int32),
        face_support_scores=np.ones(F, dtype=np.float32),
        face_areas=np.ones(F, dtype=np.float64),
        is_boundary_vertex=np.zeros(V, dtype=bool),
        is_boundary_face=np.zeros(F, dtype=bool),
        total_vertices=V,
        total_faces=F,
        depth_unit=DepthUnit.RECONSTRUCTION_UNITS,
        is_metric_scale=False,
    )

    with pytest.raises(ContractViolationError, match="Cannot declare metric units"):
        SurfaceArtifact(
            artifact_id="art_mut_surface_meters",
            payload=mock_mesh,
            units="METRES",
        )


# ===========================================================================
# Steps 13–16 Forensic Mutation Tests: Texture, Geospatial & Final Validation
# ===========================================================================

from src.pipeline.artifacts import (
    TextureAssociationArtifact,
    TexturedSurfaceArtifact,
    TelemetryArtifact,
    GeospatialArtifact,
    ValidationArtifact,
    FinalReconstructionArtifact,
)
from src.benchmark.claim_policy import ClaimPolicyEngine
from src.benchmark.models import EvidenceLevel


def test_mut_3f_18_inject_hidden_gt_texture_rejected():
    """MUT-3F-18: Attempt to inject hidden ground-truth texture into texture reconstruction must trigger DataLeakageError."""
    with pytest.raises(DataLeakageError, match="Contract Violation: Privileged evaluation key"):
        TexturedSurfaceArtifact(
            artifact_id="art_mut_tex_gt1",
            payload={"ground_truth_texture": np.ones((64, 64, 3))},
        )

    with pytest.raises(DataLeakageError, match="Contract Violation: Privileged evaluation key"):
        TexturedSurfaceArtifact(
            artifact_id="art_mut_tex_gt2",
            payload={"atlas": np.zeros((64, 64, 3))},
            metadata={"hidden_gt_texture": "secret_texture_map.png"},
        )


def test_mut_3f_19_inject_hidden_visibility_mask_rejected():
    """MUT-3F-19: Attempt to inject hidden visibility mask into texture association/reconstruction must trigger DataLeakageError."""
    with pytest.raises(DataLeakageError, match="Contract Violation: Privileged evaluation key"):
        TextureAssociationArtifact(
            artifact_id="art_mut_vis_gt1",
            payload={"true_visibility_mask": np.ones((10, 10), dtype=bool)},
        )

    with pytest.raises(DataLeakageError, match="Contract Violation: Privileged evaluation key"):
        TextureAssociationArtifact(
            artifact_id="art_mut_vis_gt2",
            payload={"associations": []},
            metadata={"hidden_visibility_mask": np.ones((10, 10))},
        )


def test_mut_3f_20_mark_unobserved_texels_as_observed_without_evidence_rejected():
    """MUT-3F-20: Attempt to mark unobserved texels as OBSERVED without source evidence must trigger ContractViolationError."""
    with pytest.raises(ContractViolationError, match="Contract Violation"):
        TexturedSurfaceArtifact(
            artifact_id="art_mut_unobs_cheat",
            payload={"unobserved_marked_observed_without_evidence": True},
        )


def test_mut_3f_21_inject_hidden_camera_pose_into_geospatial_rejected():
    """MUT-3F-21: Attempt to inject hidden true camera poses into geospatial alignment stage must trigger DataLeakageError."""
    with pytest.raises(DataLeakageError, match="Contract Violation: Privileged evaluation key"):
        TelemetryArtifact(
            artifact_id="art_mut_tel_pose_gt",
            payload={"records": []},
            metadata={"true_camera_pose": np.eye(4)},
        )

    with pytest.raises(DataLeakageError, match="Contract Violation: Privileged evaluation key"):
        GeospatialArtifact(
            artifact_id="art_mut_geo_pose_gt",
            payload={"hidden_camera_pose": [0, 0, 0]},
        )


def test_mut_3f_22_inject_hidden_metric_scale_rejected():
    """MUT-3F-22: Attempt to inject hidden metric scale into geospatial or reconstruction artifact must trigger DataLeakageError / ContractViolationError."""
    with pytest.raises(DataLeakageError, match="Contract Violation: Privileged evaluation key"):
        GeospatialArtifact(
            artifact_id="art_mut_scale_gt",
            payload={"hidden_metric_scale": 12.34},
        )

    # Cannot declare METRES when is_metric_scale is False
    with pytest.raises(ContractViolationError, match="Cannot declare metric units"):
        GeospatialArtifact(
            artifact_id="art_mut_geo_fake_meters",
            payload={"status": "SCALE_AMBIGUOUS", "is_metric_scale": False},
            units="METRES",
        )


def test_mut_3f_23_use_validation_reference_for_sim3_fitting_rejected():
    """MUT-3F-23: Attempt to use validation reference checkpoints for Sim(3) fitting must trigger DataLeakageError."""
    with pytest.raises(DataLeakageError, match="Contract Violation: Privileged evaluation key"):
        GeospatialArtifact(
            artifact_id="art_mut_val_ref",
            payload={"validation_reference": "holdout_gps_station"},
        )


from src.benchmark.models import ContractViolationError as BenchmarkContractViolationError


def test_mut_3f_24_claim_metric_accuracy_when_geospatial_not_evaluable_rejected():
    """MUT-3F-24: Claiming metric accuracy when geospatial status is NOT_EVALUABLE must trigger ContractViolationError."""
    with pytest.raises(ContractViolationError, match="Claim Policy Violation|Cannot claim metric accuracy"):
        ValidationArtifact(
            artifact_id="art_mut_metric_claim",
            payload={"claim_metric_accuracy_when_not_evaluable": True},
        )

    with pytest.raises((ContractViolationError, BenchmarkContractViolationError), match="Claim Policy Violation"):
        ClaimPolicyEngine.enforce_claim_emission(
            evidence_level=EvidenceLevel.LEVEL_0_NO_GROUND_TRUTH,
            claim_key="surveyed_3d_accuracy",
            geospatial_reference_available=False,
        )


def test_mut_3f_25_claim_radiometric_accuracy_without_calibration_rejected():
    """MUT-3F-25: Claiming radiometric/colorimetric accuracy without radiometric calibration must trigger ContractViolationError."""
    with pytest.raises(ContractViolationError, match="radiometric"):
        ValidationArtifact(
            artifact_id="art_mut_radio_claim",
            payload={"claim_radiometric_without_calibration": True},
        )

    with pytest.raises((ContractViolationError, BenchmarkContractViolationError), match="Claim Policy Violation"):
        ClaimPolicyEngine.enforce_claim_emission(
            evidence_level=EvidenceLevel.LEVEL_0_NO_GROUND_TRUTH,
            claim_key="radiometric_color_accuracy",
            radiometric_calibration_available=False,
        )


def test_mut_3f_26_reintroduce_hard_backface_culling_rejected():
    """MUT-3F-26: Attempt to reintroduce hard back-face culling as an undocumented texture visibility gate must trigger ContractViolationError."""
    with pytest.raises(ContractViolationError, match="Hard back-face culling is strictly prohibited"):
        TextureAssociationArtifact(
            artifact_id="art_mut_bf_cull_meta",
            payload={"associations": []},
            metadata={"hard_backface_culling": True},
        )

    with pytest.raises(ContractViolationError, match="Hard back-face culling is strictly prohibited"):
        TextureAssociationArtifact(
            artifact_id="art_mut_bf_cull_payload",
            payload={"hard_backface_culling": True},
        )

    with pytest.raises(ContractViolationError, match="Hard back-face culling is strictly prohibited"):
        TextureAssociationArtifact(
            artifact_id="art_mut_bf_cull_decision",
            payload={"cull_backface_decision": "REJECTED_BACKFACE"},
        )
