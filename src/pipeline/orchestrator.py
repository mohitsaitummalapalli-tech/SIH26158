"""Phase 3F: Master Pipeline Orchestrator Implementation (Steps 5–8).

Executes the end-to-end photogrammetric reconstruction DAG:
INGESTION -> DECODING -> FRAME_INTELLIGENCE -> KEYFRAME_SELECTION ->
CORRESPONDENCE -> TWO_VIEW_GEOMETRY -> INCREMENTAL_SFM -> BUNDLE_ADJUSTMENT.

Enforces cryptographic artifact integrity, strict anti-leakage isolation,
authoritative PTS temporal monotonicity, and gauge-preserved relative geometry.
"""

from __future__ import annotations

import datetime
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.pipeline.models import (
    PipelineStageType,
    StageClassification,
    StageStatus,
    PipelineStatus,
    ReconstructionUnitType,
    MetricScaleStatus,
    StageExecutionRecord,
    PipelineRunMetadata,
    PipelineResult,
)
from src.pipeline.config import PipelineConfig
from src.pipeline.errors import (
    PipelineError,
    ContractViolationError,
    StageExecutionError,
    InsufficientInputError,
    DataLeakageError,
)
from src.pipeline.artifacts import (
    PipelineArtifact,
    ArtifactDomain,
    VideoArtifact,
    CanonicalTimelineArtifact,
    DecodedFramesArtifact,
    FrameQualityArtifact,
    KeyframeSetArtifact,
    CorrespondenceArtifact,
    TwoViewGeometryArtifact,
    SfMArtifact,
    BundleAdjustmentArtifact,
    DenseStereoArtifact,
    DensePointArtifact,
    DenseFusionArtifact,
    SurfaceArtifact,
    TextureAssociationArtifact,
    TexturedSurfaceArtifact,
    TelemetryArtifact,
    GeospatialArtifact,
    ValidationArtifact,
    FinalReconstructionArtifact,
)

# Subsystem integrations
from src.ingestion.canonical_timeline import CanonicalFrame, CanonicalTimeline, VideoProvenance
from src.preprocessing.decoder import DecodedFrame, DecodeStatus
from src.quality.assessment import FrameQualityAnalyzer, QualityAssessmentConfig, FrameQualityReport
from src.quality.keyframe_selection import CoverageAwareKeyframeSelector, KeyframeSelectionResult
from src.geometry.contracts import (
    CameraIntrinsics,
    DistortionModel,
    DistortionStatus,
    FeatureCorrespondences,
    TwoViewGeometryResult,
    SparseReconstructionResult,
    PipelineStageStatus,
    ExtrinsicPose,
)
from src.geometry.features import ClassicalFeatureExtractor, ClassicalDescriptorMatcher
from src.geometry.two_view import TwoViewGeometryEstimator
from src.geometry.sfm import IncrementalSfMEngine, MatchGraph
from src.geometry.bundle_adjustment import BundleAdjustmentEngine, BundleAdjustmentResult
from src.geometry.dense_stereo import (
    ClassicalStereoSGBMEstimator,
    DenseStereoResult,
    StereoRectificationResult,
)
from src.geometry.dense_point_generation import (
    DensePointGenerator,
    DensePointGenerationResult,
    ValidatedDensePoint,
)
from src.geometry.dense_fusion import (
    DensePointFusionEngine,
    DenseFusionResult,
)
from src.geometry.mvs import (
    DensePointCloud,
    DensePointObservation,
    DepthUnit,
)
from src.geometry.surface_reconstruction import (
    AlphaComplexSurfaceReconstructor,
    SurfaceReconstructionResult,
    SurfaceMesh,
    SurfaceReconstructionStatus,
)
from src.geometry.texture_association import (
    TextureAssociationConfig,
    VisibilityAwareTextureAssociator,
    TextureSourceCamera,
    TextureSampleType,
    SurfaceTextureAssociationMap,
    SampleObservationState,
)
from src.geometry.texture_reconstruction import (
    MultiViewTextureReconstructor,
    ReconstructedTextureAtlas,
    TextureReconstructionConfig,
    OperationalTextureState,
)
from src.geospatial.pipeline import (
    GeospatialMetricReconstructor,
    GeospatialMetricReconstructionResult,
)
from src.geospatial.synchronization import RawTelemetryRecord
from src.benchmark.claim_policy import ClaimPolicyEngine
from src.benchmark.models import EvidenceLevel


class ReconstructionPipeline:
    """Master orchestrator executing the Phase 3F end-to-end reconstruction DAG."""

    DAG_SEQUENCE: List[PipelineStageType] = [
        PipelineStageType.INGESTION,
        PipelineStageType.DECODING,
        PipelineStageType.FRAME_INTELLIGENCE,
        PipelineStageType.KEYFRAME_SELECTION,
        PipelineStageType.CORRESPONDENCE,
        PipelineStageType.TWO_VIEW_GEOMETRY,
        PipelineStageType.INCREMENTAL_SFM,
        PipelineStageType.BUNDLE_ADJUSTMENT,
        PipelineStageType.DENSE_STEREO,
        PipelineStageType.DENSE_POINT_GENERATION,
        PipelineStageType.DENSE_FUSION,
        PipelineStageType.SURFACE_RECONSTRUCTION,
        PipelineStageType.TEXTURE_ASSOCIATION,
        PipelineStageType.TEXTURE_RECONSTRUCTION,
        PipelineStageType.GEOSPATIAL_TRANSFORM,
        PipelineStageType.FINAL_VALIDATION,
        PipelineStageType.FINALIZATION,
    ]

    STAGE_CLASSIFICATIONS: Dict[PipelineStageType, StageClassification] = {
        PipelineStageType.INGESTION: StageClassification.MANDATORY,
        PipelineStageType.DECODING: StageClassification.MANDATORY,
        PipelineStageType.FRAME_INTELLIGENCE: StageClassification.MANDATORY,
        PipelineStageType.KEYFRAME_SELECTION: StageClassification.MANDATORY,
        PipelineStageType.CORRESPONDENCE: StageClassification.MANDATORY,
        PipelineStageType.TWO_VIEW_GEOMETRY: StageClassification.MANDATORY,
        PipelineStageType.INCREMENTAL_SFM: StageClassification.MANDATORY,
        PipelineStageType.BUNDLE_ADJUSTMENT: StageClassification.MANDATORY,
        PipelineStageType.DENSE_STEREO: StageClassification.CONDITIONAL,
        PipelineStageType.DENSE_POINT_GENERATION: StageClassification.CONDITIONAL,
        PipelineStageType.DENSE_FUSION: StageClassification.CONDITIONAL,
        PipelineStageType.SURFACE_RECONSTRUCTION: StageClassification.CONDITIONAL,
        PipelineStageType.TEXTURE_ASSOCIATION: StageClassification.OPTIONAL,
        PipelineStageType.TEXTURE_RECONSTRUCTION: StageClassification.OPTIONAL,
        PipelineStageType.GEOSPATIAL_TRANSFORM: StageClassification.CONDITIONAL,
        PipelineStageType.FINAL_VALIDATION: StageClassification.MANDATORY,
        PipelineStageType.FINALIZATION: StageClassification.MANDATORY,
    }

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.validate_config()
        self.artifacts_by_id: Dict[str, PipelineArtifact] = {}
        self.artifacts_by_type: Dict[str, PipelineArtifact] = {}
        self.stage_records: List[StageExecutionRecord] = []
        self._active_sfm_engine: Optional[IncrementalSfMEngine] = None
        self._active_intrinsics: Optional[CameraIntrinsics] = None

    def validate_config(self) -> None:
        """Validates configuration parameters and stage dependency combinations."""
        self.config.validate()

    def verify_input_artifact(
        self,
        artifact: PipelineArtifact,
        expected_stage: PipelineStageType,
    ) -> None:
        """Validates an input artifact prior to stage consumption.

        1. Enforces anti-leakage isolation: rejects EVALUATION_TRUTH artifacts.
        2. Verifies cryptographic content hash integrity.
        """
        if artifact.domain == ArtifactDomain.EVALUATION_TRUTH:
            raise DataLeakageError(
                f"Data Leakage Violation: Artifact '{artifact.artifact_id}' belongs to "
                f"EVALUATION_TRUTH and cannot be consumed by reconstruction stage '{expected_stage.value}'.",
                stage=expected_stage.value,
            )

        if self.config.strict_immutability_checks:
            artifact.verify_integrity()

    def record_stage(
        self,
        stage_type: PipelineStageType,
        status: StageStatus,
        started_at: str,
        duration_s: float,
        input_ids: Optional[List[str]] = None,
        output_ids: Optional[List[str]] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> StageExecutionRecord:
        """Constructs and appends an immutable StageExecutionRecord."""
        record = StageExecutionRecord(
            stage_id=f"stg_{stage_type.value.lower()}_{uuid.uuid4().hex[:6]}",
            stage_type=stage_type,
            classification=self.STAGE_CLASSIFICATIONS[stage_type],
            status=status,
            input_artifact_ids=input_ids or [],
            output_artifact_ids=output_ids or [],
            configuration_hash=self.config.compute_hash(),
            software_commit="019deb2",
            started_at=started_at,
            finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            duration_seconds=round(duration_s, 6),
            diagnostics=diagnostics or {},
        )
        self.stage_records.append(record)
        return record

    def _resolve_intrinsics(self, width: int, height: int) -> CameraIntrinsics:
        """Resolves or creates calibrated camera intrinsics."""
        if self._active_intrinsics is not None:
            return self._active_intrinsics
        if self.config.default_intrinsics is not None:
            self._active_intrinsics = self.config.default_intrinsics
            return self._active_intrinsics

        # Default pinhole model with 60 deg horizontal FoV
        fx = float(width) * 1.0
        fy = float(width) * 1.0
        cx = float(width) / 2.0
        cy = float(height) / 2.0
        self._active_intrinsics = CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            distortion_model=DistortionModel.NONE_RECTIFIED,
            distortion_status=DistortionStatus.RECTIFIED_ZERO_DISTORTION,
        )
        return self._active_intrinsics

    @staticmethod
    def _extract_camera_rt_and_center(pose: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts camera rotation R_cw, translation t_cw, and optical center C_w from pose."""
        if hasattr(pose, "R_cw") and hasattr(pose, "t_cw"):
            R_cw = np.asarray(pose.R_cw, dtype=np.float64)
            t_cw = np.asarray(pose.t_cw, dtype=np.float64).reshape(-1)
            C_w = -R_cw.T @ t_cw
        elif hasattr(pose, "rotation_matrix") and hasattr(pose, "translation_vector"):
            R_wc = np.asarray(pose.rotation_matrix, dtype=np.float64)
            C_w = np.asarray(pose.translation_vector, dtype=np.float64).reshape(-1)
            R_cw = R_wc.T
            t_cw = -R_cw @ C_w
        elif isinstance(pose, dict):
            if "R_cw" in pose and "t_cw" in pose:
                R_cw = np.asarray(pose["R_cw"], dtype=np.float64)
                t_cw = np.asarray(pose["t_cw"], dtype=np.float64).reshape(-1)
                C_w = -R_cw.T @ t_cw
            else:
                R_wc = np.asarray(pose.get("rotation_matrix", np.eye(3)), dtype=np.float64)
                C_w = np.asarray(pose.get("translation_vector", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
                R_cw = R_wc.T
                t_cw = -R_cw @ C_w
        else:
            raise ValueError(f"Unsupported camera pose representation: {type(pose)}")
        return R_cw, t_cw, C_w

    # -----------------------------------------------------------------------
    # Stage Execution Implementations
    # -----------------------------------------------------------------------

    def _execute_ingestion(self) -> StageStatus:
        """STG-01: Ingestion and Canonical Timeline assembly."""
        if "CanonicalTimelineArtifact" in self.artifacts_by_type:
            return StageStatus.SUCCESS

        # Check for VideoArtifact or DecodedFramesArtifact
        vid_art = self.artifacts_by_type.get("VideoArtifact")
        dec_art = self.artifacts_by_type.get("DecodedFramesArtifact")

        if vid_art is None and dec_art is None:
            return StageStatus.INSUFFICIENT_INPUT

        # Build timeline from provided frames or video payload
        frames: List[Any] = []
        if dec_art is not None:
            frames = dec_art.payload
        elif vid_art is not None and isinstance(vid_art.payload, dict) and "frames" in vid_art.payload:
            frames = vid_art.payload["frames"]

        if len(frames) == 0:
            raise InsufficientInputError(
                "Zero frames available for timeline construction.",
                stage=PipelineStageType.INGESTION.value,
            )

        # Enforce strict chronological order: timestamps must be monotonic non-decreasing
        timeline_frames: List[CanonicalFrame] = []
        prev_t = -1.0
        for i, f in enumerate(frames):
            t = getattr(f, "timestamp_seconds", i * 0.5)
            if t < prev_t:
                raise ContractViolationError(
                    f"Chronological PTS ordering violation: Frame {i} timestamp {t} < previous {prev_t}.",
                    stage=PipelineStageType.INGESTION.value,
                )
            prev_t = t
            fid = getattr(f, "frame_id", f"frame_{i:04d}")
            if hasattr(f, "width") and hasattr(f, "height"):
                fw, fh = int(f.width), int(f.height)
            elif isinstance(f, np.ndarray):
                fh, fw = int(f.shape[0]), int(f.shape[1])
            else:
                fw, fh = 640, 480

            timeline_frames.append(
                CanonicalFrame(
                    frame_id=fid,
                    frame_index=i,
                    timestamp_seconds=float(t),
                    pts=int(round(t * 1000)),
                    timescale=1000,
                    source_video="reconstruction_input",
                    width=fw,
                    height=fh,
                )
            )

        w = timeline_frames[0].width if timeline_frames else 640
        h = timeline_frames[0].height if timeline_frames else 480
        dur = float(timeline_frames[-1].timestamp_seconds) if timeline_frames else 0.0

        prov = VideoProvenance(
            source_file_path="reconstruction_input",
            file_size_bytes=1024,
            sha256_checksum="0" * 64,
            ingestion_timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            metadata_extractor="OrchestratorIngestion_v1.0",
            timestamp_source="container_ctts_pts",
        )

        timeline = CanonicalTimeline(
            video_id=f"vid_{uuid.uuid4().hex[:8]}",
            source_path="reconstruction_input",
            total_frames=len(timeline_frames),
            duration_seconds=dur,
            nominal_fps=30.0,
            width=w,
            height=h,
            frames=timeline_frames,
            provenance=prov,
        )
        art = CanonicalTimelineArtifact(
            artifact_id=f"art_timeline_{uuid.uuid4().hex[:8]}",
            payload=timeline,
        )
        self.artifacts_by_id[art.artifact_id] = art
        self.artifacts_by_type[art.artifact_type] = art
        return StageStatus.SUCCESS

    def _execute_decoding(self) -> StageStatus:
        """STG-02: Frame Decoding into Canonical RGB representations."""
        if "DecodedFramesArtifact" in self.artifacts_by_type:
            return StageStatus.SUCCESS

        vid_art = self.artifacts_by_type.get("VideoArtifact")
        if vid_art is None:
            raise InsufficientInputError(
                "Cannot decode frames: No VideoArtifact available.",
                stage=PipelineStageType.DECODING.value,
            )

        # Extract frames from VideoArtifact payload
        raw_frames = vid_art.payload.get("frames", []) if isinstance(vid_art.payload, dict) else []
        if len(raw_frames) == 0:
            raise InsufficientInputError(
                "Zero frames successfully decoded from video source.",
                stage=PipelineStageType.DECODING.value,
            )

        timeline_art = self.artifacts_by_type.get("CanonicalTimelineArtifact")
        timeline: Optional[CanonicalTimeline] = timeline_art.payload if timeline_art is not None else None

        decoded_list: List[DecodedFrame] = []
        for i, item in enumerate(raw_frames):
            if isinstance(item, DecodedFrame):
                decoded_list.append(item)
            elif isinstance(item, np.ndarray):
                h, w = item.shape[:2]
                t_sec = float(timeline.frames[i].timestamp_seconds) if (timeline and i < len(timeline.frames)) else i * 0.5
                fid = timeline.frames[i].frame_id if (timeline and i < len(timeline.frames)) else f"frame_{i:04d}"
                decoded_list.append(
                    DecodedFrame(
                        frame_id=fid,
                        frame_index=i,
                        timestamp_seconds=t_sec,
                        width=w,
                        height=h,
                        channels=item.shape[2] if item.ndim == 3 else 1,
                        channel_layout="RGB",
                        dtype="uint8",
                        data=item,
                        source_video="input",
                        decode_status=DecodeStatus.SUCCESS,
                    )
                )

        art = DecodedFramesArtifact(
            artifact_id=f"art_decoded_{uuid.uuid4().hex[:8]}",
            payload=decoded_list,
        )
        self.artifacts_by_id[art.artifact_id] = art
        self.artifacts_by_type[art.artifact_type] = art
        return StageStatus.SUCCESS

    def _execute_frame_intelligence(self) -> StageStatus:
        """STG-03: Frame Quality Assessment & Diagnostics."""
        dec_art = self.artifacts_by_type.get("DecodedFramesArtifact")
        if dec_art is None:
            raise InsufficientInputError(
                "Decoded frames unavailable for frame intelligence.",
                stage=PipelineStageType.FRAME_INTELLIGENCE.value,
            )

        frames: List[DecodedFrame] = dec_art.payload
        reports: Dict[str, FrameQualityReport] = {}
        cfg = QualityAssessmentConfig()

        for frame in frames:
            reports[frame.frame_id] = FrameQualityAnalyzer.analyze_frame(frame, cfg)

        art = FrameQualityArtifact(
            artifact_id=f"art_quality_{uuid.uuid4().hex[:8]}",
            payload=reports,
        )
        self.artifacts_by_id[art.artifact_id] = art
        self.artifacts_by_type[art.artifact_type] = art
        return StageStatus.SUCCESS

    def _execute_keyframe_selection(self) -> StageStatus:
        """STG-04: Keyframe Selection via greedy marginal gain."""
        dec_art = self.artifacts_by_type.get("DecodedFramesArtifact")
        qual_art = self.artifacts_by_type.get("FrameQualityArtifact")

        if dec_art is None:
            raise InsufficientInputError(
                "Decoded frames unavailable for keyframe selection.",
                stage=PipelineStageType.KEYFRAME_SELECTION.value,
            )

        frames: List[DecodedFrame] = dec_art.payload
        quality_reports = qual_art.payload if qual_art is not None else None

        kf_result: KeyframeSelectionResult = CoverageAwareKeyframeSelector.select_keyframes(
            frames=frames,
            quality_reports=quality_reports,
            config=self.config.keyframe_config,
        )

        if len(kf_result.selected_keyframe_ids) < 2:
            raise InsufficientInputError(
                f"Insufficient keyframes selected ({len(kf_result.selected_keyframe_ids)} < 2) for multi-view reconstruction.",
                stage=PipelineStageType.KEYFRAME_SELECTION.value,
            )

        art = KeyframeSetArtifact(
            artifact_id=f"art_keyframes_{uuid.uuid4().hex[:8]}",
            payload=kf_result,
        )
        self.artifacts_by_id[art.artifact_id] = art
        self.artifacts_by_type[art.artifact_type] = art
        return StageStatus.SUCCESS

    def _execute_correspondences(self) -> StageStatus:
        """STG-05: Real 2D Feature Extraction & Pairwise Correspondence Matching."""
        dec_art = self.artifacts_by_type.get("DecodedFramesArtifact")
        kf_art = self.artifacts_by_type.get("KeyframeSetArtifact")

        if dec_art is None or kf_art is None:
            raise InsufficientInputError(
                "Decoded frames or keyframe set unavailable for correspondence stage.",
                stage=PipelineStageType.CORRESPONDENCE.value,
            )

        frames_by_id: Dict[str, DecodedFrame] = {f.frame_id: f for f in dec_art.payload}
        kf_result: KeyframeSelectionResult = kf_art.payload
        selected_ids = kf_result.selected_keyframe_ids

        extractor = ClassicalFeatureExtractor(config=self.config.feature_config)
        matcher = ClassicalDescriptorMatcher(config=self.config.feature_config)

        # 1. Extract features on all selected keyframes
        features_by_id: Dict[str, Any] = {}
        for fid in selected_ids:
            frame = frames_by_id[fid]
            features_by_id[fid] = extractor.extract(frame, frame_id=fid)

        # 2. Match pairs of keyframes (consecutive pairs and skip pairs)
        pairwise_correspondences: Dict[Tuple[str, str], FeatureCorrespondences] = {}
        total_matches = 0

        for i in range(len(selected_ids)):
            for j in range(i + 1, min(i + 3, len(selected_ids))):
                id_a = selected_ids[i]
                id_b = selected_ids[j]
                feat_a = features_by_id[id_a]
                feat_b = features_by_id[id_b]

                match_res = matcher.match(feat_a, feat_b)
                if match_res.status == "SUCCESS" and match_res.accepted_match_count >= 8:
                    corr = match_res.to_correspondences()
                    pairwise_correspondences[(id_a, id_b)] = corr
                    total_matches += corr.match_count

        if len(pairwise_correspondences) == 0 or total_matches < 8:
            raise InsufficientInputError(
                f"Insufficient candidate feature correspondences (pairs: {len(pairwise_correspondences)}, total: {total_matches}).",
                stage=PipelineStageType.CORRESPONDENCE.value,
            )

        art = CorrespondenceArtifact(
            artifact_id=f"art_corr_{uuid.uuid4().hex[:8]}",
            payload=pairwise_correspondences,
        )
        self.artifacts_by_id[art.artifact_id] = art
        self.artifacts_by_type[art.artifact_type] = art
        return StageStatus.SUCCESS

    def _execute_two_view_geometry(self) -> StageStatus:
        """STG-06: Robust Two-View Essential Matrix Estimation & Seed Selection."""
        corr_art = self.artifacts_by_type.get("CorrespondenceArtifact")
        dec_art = self.artifacts_by_type.get("DecodedFramesArtifact")

        if corr_art is None or dec_art is None:
            raise InsufficientInputError(
                "Correspondence artifact unavailable for two-view geometry.",
                stage=PipelineStageType.TWO_VIEW_GEOMETRY.value,
            )

        pairwise_corr: Dict[Tuple[str, str], FeatureCorrespondences] = corr_art.payload
        first_frame: DecodedFrame = dec_art.payload[0]
        intrinsics = self._resolve_intrinsics(first_frame.width, first_frame.height)

        estimator = TwoViewGeometryEstimator(config=self.config.two_view_config)

        # Sort candidate pairs by match count descending
        sorted_pairs = sorted(
            pairwise_corr.keys(),
            key=lambda p: pairwise_corr[p].match_count,
            reverse=True,
        )

        seed_result: Optional[TwoViewGeometryResult] = None
        seed_pair: Optional[Tuple[str, str]] = None

        for pair in sorted_pairs:
            corr = pairwise_corr[pair]
            res = estimator.estimate_essential(corr, intrinsics)
            if res.e_status == "SUCCESS" and not res.is_degenerate and res.inlier_count >= self.config.two_view_config.min_inliers:
                seed_result = res
                seed_pair = pair
                break

        if seed_result is None:
            raise StageExecutionError(
                "Two-view geometry failed: No candidate keyframe pair satisfied epipolar and cheirality constraints.",
                stage=PipelineStageType.TWO_VIEW_GEOMETRY.value,
            )

        art = TwoViewGeometryArtifact(
            artifact_id=f"art_twoview_{uuid.uuid4().hex[:8]}",
            payload={"seed_result": seed_result, "seed_pair": seed_pair},
        )
        self.artifacts_by_id[art.artifact_id] = art
        self.artifacts_by_type[art.artifact_type] = art
        return StageStatus.SUCCESS

    def _execute_incremental_sfm(self) -> StageStatus:
        """STG-07: Incremental SfM Camera Registration & Landmark Triangulation."""
        tv_art = self.artifacts_by_type.get("TwoViewGeometryArtifact")
        corr_art = self.artifacts_by_type.get("CorrespondenceArtifact")
        dec_art = self.artifacts_by_type.get("DecodedFramesArtifact")

        if tv_art is None or corr_art is None or dec_art is None:
            raise InsufficientInputError(
                "Two-view geometry or correspondence artifacts unavailable for SfM.",
                stage=PipelineStageType.INCREMENTAL_SFM.value,
            )

        tv_payload = tv_art.payload
        seed_result: TwoViewGeometryResult = tv_payload["seed_result"]
        seed_pair: Tuple[str, str] = tv_payload["seed_pair"]
        pairwise_corr: Dict[Tuple[str, str], FeatureCorrespondences] = corr_art.payload

        first_frame: DecodedFrame = dec_art.payload[0]
        intrinsics = self._resolve_intrinsics(first_frame.width, first_frame.height)
        intrinsics_map = {f.frame_id: intrinsics for f in dec_art.payload}

        engine = IncrementalSfMEngine(config=self.config.sfm_config)
        seed_corr = pairwise_corr[seed_pair]

        # Build match graph
        match_graph = MatchGraph()
        for (fa, fb), corr in pairwise_corr.items():
            inlier_mask = seed_result.inlier_mask if (fa, fb) == seed_pair else None
            match_graph.add_edge(fa, fb, corr, inlier_mask=inlier_mask)

        all_keyframe_ids = sorted(list({fa for (fa, _) in pairwise_corr.keys()} | {fb for (_, fb) in pairwise_corr.keys()}))

        sparse_recon = engine.reconstruct(
            keyframe_ids=all_keyframe_ids,
            intrinsics_map=intrinsics_map,
            match_graph=match_graph,
            initial_two_view=seed_result,
            initial_correspondences=seed_corr,
        )

        if sparse_recon.status != PipelineStageStatus.SUCCESS or len(sparse_recon.camera_poses) < 2:
            raise StageExecutionError(
                f"Incremental SfM failed: {sparse_recon.diagnostics}",
                stage=PipelineStageType.INCREMENTAL_SFM.value,
            )

        self._active_sfm_engine = engine

        art = SfMArtifact(
            artifact_id=f"art_sfm_{uuid.uuid4().hex[:8]}",
            payload=sparse_recon,
        )
        self.artifacts_by_id[art.artifact_id] = art
        self.artifacts_by_type[art.artifact_type] = art
        return StageStatus.SUCCESS

    def _execute_bundle_adjustment(self) -> StageStatus:
        """STG-08: Gauge-Preserving Huber Global Bundle Adjustment."""
        sfm_art = self.artifacts_by_type.get("SfMArtifact")
        if sfm_art is None:
            raise InsufficientInputError(
                "SfM artifact unavailable for Bundle Adjustment.",
                stage=PipelineStageType.BUNDLE_ADJUSTMENT.value,
            )

        sparse_recon: SparseReconstructionResult = sfm_art.payload
        tracks = self._active_sfm_engine.tracks if self._active_sfm_engine is not None else None
        intrinsics_map = {cid: self._resolve_intrinsics(1000, 1000) for cid in sparse_recon.camera_poses}

        ba_engine = BundleAdjustmentEngine(config=self.config.ba_config)
        ba_result = ba_engine.optimize(
            reconstruction=sparse_recon,
            tracks=tracks,
            intrinsics_map=intrinsics_map,
        )

        if ba_result.status != PipelineStageStatus.SUCCESS:
            raise StageExecutionError(
                f"Bundle Adjustment failed: {ba_result.diagnostics}",
                stage=PipelineStageType.BUNDLE_ADJUSTMENT.value,
            )

        art = BundleAdjustmentArtifact(
            artifact_id=f"art_ba_{uuid.uuid4().hex[:8]}",
            payload=ba_result,
        )
        self.artifacts_by_id[art.artifact_id] = art
        self.artifacts_by_type[art.artifact_type] = art
        return StageStatus.SUCCESS

    # -----------------------------------------------------------------------
    # Step 9: Dense Stereo Rectification & Disparity/Depth Estimation
    # -----------------------------------------------------------------------

    def _execute_dense_stereo(self) -> StageStatus:
        """STG-09: Multi-View Stereo Rectification & Dense Depth Estimation."""
        dec_art = self.artifacts_by_type.get("DecodedFramesArtifact")
        ba_art = self.artifacts_by_type.get("BundleAdjustmentArtifact")
        sfm_art = self.artifacts_by_type.get("SfMArtifact")

        if dec_art is None or (ba_art is None and sfm_art is None):
            raise InsufficientInputError(
                "Decoded frames or SfM/BA cameras unavailable for dense stereo.",
                stage=PipelineStageType.DENSE_STEREO.value,
            )

        if ba_art is not None and getattr(ba_art.payload, "refined_reconstruction", None) is not None:
            camera_poses = ba_art.payload.refined_reconstruction.camera_poses
        elif sfm_art is not None:
            camera_poses = sfm_art.payload.camera_poses
        else:
            camera_poses = {}

        frames_by_id = {f.frame_id: f.data for f in dec_art.payload}

        cam_ids = sorted(list(camera_poses.keys()))
        if len(cam_ids) < 2:
            raise InsufficientInputError(
                f"Insufficient registered cameras ({len(cam_ids)} < 2) for dense stereo.",
                stage=PipelineStageType.DENSE_STEREO.value,
            )

        estimator = ClassicalStereoSGBMEstimator(config=self.config.dense_stereo_config)
        stereo_results: Dict[Tuple[str, str], DenseStereoResult] = {}
        total_valid_depth_pixels = 0

        for i in range(len(cam_ids) - 1):
            ref_id = cam_ids[i]
            src_id = cam_ids[i + 1]

            if ref_id not in frames_by_id or src_id not in frames_by_id:
                continue

            ref_pose = camera_poses[ref_id]
            src_pose = camera_poses[src_id]
            ref_img = frames_by_id[ref_id]
            src_img = frames_by_id[src_id]

            ref_intrinsics = self._resolve_intrinsics(ref_img.shape[1], ref_img.shape[0])
            src_intrinsics = self._resolve_intrinsics(src_img.shape[1], src_img.shape[0])

            st_res = estimator.compute_dense_stereo(
                ref_image=ref_img,
                src_image=src_img,
                ref_pose=ref_pose,
                src_pose=src_pose,
                ref_intrinsics=ref_intrinsics,
                src_intrinsics=src_intrinsics,
                ref_frame_id=ref_id,
                src_frame_id=src_id,
            )
            stereo_results[(ref_id, src_id)] = st_res
            valid_pixels = int(np.sum(st_res.depth_map.valid_mask))
            total_valid_depth_pixels += valid_pixels

        if len(stereo_results) == 0 or total_valid_depth_pixels == 0:
            return StageStatus.FAILED

        art = DenseStereoArtifact(
            artifact_id=f"art_stereo_{uuid.uuid4().hex[:8]}",
            payload=stereo_results,
        )
        self.artifacts_by_id[art.artifact_id] = art
        self.artifacts_by_type[art.artifact_type] = art
        return StageStatus.SUCCESS

    # -----------------------------------------------------------------------
    # Step 10: Dense 3D Point Generation & Backprojection
    # -----------------------------------------------------------------------

    def _execute_dense_point_generation(self) -> StageStatus:
        """STG-10: Dense 3D Point Cloud Backprojection & Geometric Validation."""
        stereo_art = self.artifacts_by_type.get("DenseStereoArtifact")
        ba_art = self.artifacts_by_type.get("BundleAdjustmentArtifact")
        sfm_art = self.artifacts_by_type.get("SfMArtifact")

        if stereo_art is None:
            raise InsufficientInputError(
                "Dense stereo artifact unavailable for dense point generation.",
                stage=PipelineStageType.DENSE_POINT_GENERATION.value,
            )

        if ba_art is not None and getattr(ba_art.payload, "refined_reconstruction", None) is not None:
            camera_poses = ba_art.payload.refined_reconstruction.camera_poses
        elif sfm_art is not None:
            camera_poses = sfm_art.payload.camera_poses
        else:
            camera_poses = {}

        stereo_results: Dict[Tuple[str, str], DenseStereoResult] = stereo_art.payload
        generator = DensePointGenerator(config=self.config.dense_point_config)
        all_observations: List[DensePointObservation] = []
        all_points: List[ValidatedDensePoint] = []
        gen_results: Dict[Tuple[str, str], DensePointGenerationResult] = {}

        for pair_key, st_res in stereo_results.items():
            ref_id = pair_key[0]
            ref_pose = camera_poses[ref_id]
            ref_intrinsics = self._resolve_intrinsics(st_res.depth_map.width, st_res.depth_map.height)

            pt_res = generator.generate_points(
                stereo_result=st_res,
                ref_pose=ref_pose,
                ref_intrinsics=ref_intrinsics,
            )
            gen_results[pair_key] = pt_res
            all_observations.extend(pt_res.observations)
            all_points.extend(pt_res.validated_points)

        if len(all_points) == 0:
            raise StageExecutionError(
                "Dense point generation yielded 0 geometrically valid 3D points.",
                stage=PipelineStageType.DENSE_POINT_GENERATION.value,
            )

        art = DensePointArtifact(
            artifact_id=f"art_densepoints_{uuid.uuid4().hex[:8]}",
            payload={
                "generation_results": gen_results,
                "observations": all_observations,
                "points": all_points,
            },
        )
        self.artifacts_by_id[art.artifact_id] = art
        self.artifacts_by_type[art.artifact_type] = art
        return StageStatus.SUCCESS

    # -----------------------------------------------------------------------
    # Step 11: Multi-View Spatial Fusion
    # -----------------------------------------------------------------------

    def _execute_dense_fusion(self) -> StageStatus:
        """STG-11: Multi-View Spatial Fusion & Distinct-View Support Consensus."""
        pts_art = self.artifacts_by_type.get("DensePointArtifact")
        if pts_art is None:
            raise InsufficientInputError(
                "Dense point artifact unavailable for multi-view fusion.",
                stage=PipelineStageType.DENSE_FUSION.value,
            )

        payload = pts_art.payload
        observations: List[DensePointObservation] = payload.get("observations", [])

        if len(observations) == 0:
            raise StageExecutionError(
                "Zero input point observations provided for dense fusion.",
                stage=PipelineStageType.DENSE_FUSION.value,
            )

        fusion_engine = DensePointFusionEngine(config=self.config.dense_fusion_config)
        fusion_result: DenseFusionResult = fusion_engine.fuse_observations(observations)

        if fusion_result.total_fused_points < 4:
            raise StageExecutionError(
                f"Multi-view fusion produced insufficient points ({fusion_result.total_fused_points} < 4) for surface reconstruction.",
                stage=PipelineStageType.DENSE_FUSION.value,
            )

        art = DenseFusionArtifact(
            artifact_id=f"art_fusion_{uuid.uuid4().hex[:8]}",
            payload=fusion_result,
        )
        self.artifacts_by_id[art.artifact_id] = art
        self.artifacts_by_type[art.artifact_type] = art
        return StageStatus.SUCCESS

    # -----------------------------------------------------------------------
    # Step 12: Alpha-Complex 3D Surface Reconstruction
    # -----------------------------------------------------------------------

    def _execute_surface_reconstruction(self) -> StageStatus:
        """STG-12: Alpha-Complex 3D Surface Reconstruction & Normal Estimation."""
        fusion_art = self.artifacts_by_type.get("DenseFusionArtifact")
        ba_art = self.artifacts_by_type.get("BundleAdjustmentArtifact")
        sfm_art = self.artifacts_by_type.get("SfMArtifact")

        if fusion_art is None:
            raise InsufficientInputError(
                "Dense fusion artifact unavailable for surface reconstruction.",
                stage=PipelineStageType.SURFACE_RECONSTRUCTION.value,
            )

        fusion_result: DenseFusionResult = fusion_art.payload
        point_cloud: DensePointCloud = fusion_result.point_cloud

        if ba_art is not None and getattr(ba_art.payload, "refined_reconstruction", None) is not None:
            camera_poses = ba_art.payload.refined_reconstruction.camera_poses
        elif sfm_art is not None:
            camera_poses = sfm_art.payload.camera_poses
        else:
            camera_poses = {}

        camera_centers = {
            cid: -np.array(pose.rotation_matrix).T @ np.array(pose.translation_vector)
            for cid, pose in camera_poses.items()
        }

        reconstructor = AlphaComplexSurfaceReconstructor(config=self.config.surface_config)
        surface_result: SurfaceReconstructionResult = reconstructor.reconstruct_surface(
            cloud=point_cloud,
            compute_normals=True,
            camera_centers=camera_centers,
        )

        if surface_result.status != SurfaceReconstructionStatus.SUCCESS or surface_result.mesh is None:
            raise StageExecutionError(
                f"Surface reconstruction failed: {surface_result.failure_reasons} - {surface_result.diagnostics}",
                stage=PipelineStageType.SURFACE_RECONSTRUCTION.value,
            )

        art = SurfaceArtifact(
            artifact_id=f"art_surface_{uuid.uuid4().hex[:8]}",
            payload=surface_result,
        )
        self.artifacts_by_id[art.artifact_id] = art
        self.artifacts_by_type[art.artifact_type] = art
        return StageStatus.SUCCESS

    # -----------------------------------------------------------------------
    # Step 13: Visibility-Aware Texture Association
    # -----------------------------------------------------------------------

    def _execute_texture_association(self) -> StageStatus:
        """STG-13: Visibility-Aware Surface Texture Association."""
        surface_art = self.artifacts_by_type.get("SurfaceArtifact")
        ba_art = self.artifacts_by_type.get("BundleAdjustmentArtifact")
        sfm_art = self.artifacts_by_type.get("SfMArtifact")
        dec_art = self.artifacts_by_type.get("DecodedFramesArtifact")

        if surface_art is None:
            raise InsufficientInputError(
                "Surface artifact unavailable for texture association.",
                stage=PipelineStageType.TEXTURE_ASSOCIATION.value,
            )

        if dec_art is None or (ba_art is None and sfm_art is None):
            raise InsufficientInputError(
                "Decoded frames or camera poses unavailable for texture association.",
                stage=PipelineStageType.TEXTURE_ASSOCIATION.value,
            )

        mesh = surface_art.payload.mesh if hasattr(surface_art.payload, "mesh") else surface_art.payload

        if ba_art is not None and getattr(ba_art.payload, "refined_reconstruction", None) is not None:
            camera_poses = ba_art.payload.refined_reconstruction.camera_poses
        elif sfm_art is not None:
            camera_poses = sfm_art.payload.camera_poses
        else:
            camera_poses = {}

        frames_by_id = {f.frame_id: f.data for f in dec_art.payload}

        quality_art = self.artifacts_by_type.get("FrameQualityArtifact")
        quality_by_id: Dict[str, Any] = {}
        if quality_art is not None and isinstance(quality_art.payload, list):
            for q in quality_art.payload:
                if hasattr(q, "frame_id"):
                    quality_by_id[q.frame_id] = q

        texture_cameras: Dict[str, TextureSourceCamera] = {}
        for cid, pose in camera_poses.items():
            if cid not in frames_by_id:
                continue
            img = frames_by_id[cid]
            h, w = img.shape[:2]
            intrinsics = self._resolve_intrinsics(w, h)
            q_rep = quality_by_id.get(cid)
            if q_rep is not None and getattr(q_rep, "sharpness", None) is not None:
                sharpness_val = float(getattr(q_rep.sharpness, "laplacian_variance", 50.0)) / 100.0
            else:
                sharpness_val = 1.0

            quality_metrics = {
                "sharpness": min(1.0, max(0.01, float(sharpness_val))),
                "blur": 0.0,
                "exposure": 1.0,
                "dynamic_risk": 0.0,
            }
            R_cw, t_cw, _ = self._extract_camera_rt_and_center(pose)
            texture_cameras[cid] = TextureSourceCamera(
                frame_id=cid,
                R_cw=R_cw,
                t_cw=t_cw,
                K=np.asarray(intrinsics.matrix_3x3, dtype=np.float64),
                width=w,
                height=h,
                quality_metrics=quality_metrics,
            )

        if len(texture_cameras) == 0:
            return StageStatus.FAILED

        associator = VisibilityAwareTextureAssociator(config=self.config.texture_association_config)
        assoc_map: SurfaceTextureAssociationMap = associator.associate_texture(
            mesh=mesh,
            cameras=texture_cameras,
            sample_type=TextureSampleType.FACET_CENTROID,
        )

        art = TextureAssociationArtifact(
            artifact_id=f"art_tex_assoc_{uuid.uuid4().hex[:8]}",
            payload=assoc_map,
            units="DIMENSIONLESS",
            coordinate_frame="SURFACE_LOCAL",
        )
        self.artifacts_by_id[art.artifact_id] = art
        self.artifacts_by_type[art.artifact_type] = art
        return StageStatus.SUCCESS

    # -----------------------------------------------------------------------
    # Step 14: Multi-View Texture Atlas Reconstruction
    # -----------------------------------------------------------------------

    def _execute_texture_reconstruction(self) -> StageStatus:
        """STG-14: Robust Multi-View Texture Atlas Reconstruction."""
        surface_art = self.artifacts_by_type.get("SurfaceArtifact")
        assoc_art = self.artifacts_by_type.get("TextureAssociationArtifact")
        dec_art = self.artifacts_by_type.get("DecodedFramesArtifact")
        ba_art = self.artifacts_by_type.get("BundleAdjustmentArtifact")
        sfm_art = self.artifacts_by_type.get("SfMArtifact")

        if surface_art is None or assoc_art is None or dec_art is None:
            raise InsufficientInputError(
                "Surface, association, or decoded frames unavailable for texture reconstruction.",
                stage=PipelineStageType.TEXTURE_RECONSTRUCTION.value,
            )

        mesh = surface_art.payload.mesh if hasattr(surface_art.payload, "mesh") else surface_art.payload
        assoc_map: SurfaceTextureAssociationMap = assoc_art.payload
        frames_by_id = {f.frame_id: f.data for f in dec_art.payload}

        if ba_art is not None and getattr(ba_art.payload, "refined_reconstruction", None) is not None:
            camera_poses = ba_art.payload.refined_reconstruction.camera_poses
        elif sfm_art is not None:
            camera_poses = sfm_art.payload.camera_poses
        else:
            camera_poses = {}

        quality_art = self.artifacts_by_type.get("FrameQualityArtifact")
        quality_by_id: Dict[str, Any] = {}
        if quality_art is not None and isinstance(quality_art.payload, list):
            for q in quality_art.payload:
                if hasattr(q, "frame_id"):
                    quality_by_id[q.frame_id] = q

        texture_cameras: Dict[str, TextureSourceCamera] = {}
        for cid, pose in camera_poses.items():
            if cid not in frames_by_id:
                continue
            img = frames_by_id[cid]
            h, w = img.shape[:2]
            intrinsics = self._resolve_intrinsics(w, h)
            q_rep = quality_by_id.get(cid)
            if q_rep is not None and getattr(q_rep, "sharpness", None) is not None:
                sharpness_val = float(getattr(q_rep.sharpness, "laplacian_variance", 50.0)) / 100.0
            else:
                sharpness_val = 1.0

            quality_metrics = {
                "sharpness": min(1.0, max(0.01, float(sharpness_val))),
                "blur": 0.0,
                "exposure": 1.0,
                "dynamic_risk": 0.0,
            }
            R_cw, t_cw, _ = self._extract_camera_rt_and_center(pose)
            texture_cameras[cid] = TextureSourceCamera(
                frame_id=cid,
                R_cw=R_cw,
                t_cw=t_cw,
                K=np.asarray(intrinsics.matrix_3x3, dtype=np.float64),
                width=w,
                height=h,
                quality_metrics=quality_metrics,
            )

        reconstructor = MultiViewTextureReconstructor(config=self.config.texture_reconstruction_config)
        atlas: ReconstructedTextureAtlas = reconstructor.reconstruct_texture(
            mesh=mesh,
            cameras=texture_cameras,
            camera_images=frames_by_id,
            association_map=assoc_map,
        )

        art = TexturedSurfaceArtifact(
            artifact_id=f"art_tex_atlas_{uuid.uuid4().hex[:8]}",
            payload=atlas,
            units="RECONSTRUCTION_UNITS",
            coordinate_frame="MONOCULAR_GAUGE",
        )
        self.artifacts_by_id[art.artifact_id] = art
        self.artifacts_by_type[art.artifact_type] = art
        return StageStatus.SUCCESS

    # -----------------------------------------------------------------------
    # Step 15: Conditional Geospatial Metric Reconstruction
    # -----------------------------------------------------------------------

    def _execute_geospatial(self) -> StageStatus:
        """STG-15: Conditional Geospatial Alignment & Metric Scale Estimation."""
        telemetry_art = self.artifacts_by_type.get("TelemetryArtifact")
        ba_art = self.artifacts_by_type.get("BundleAdjustmentArtifact")
        sfm_art = self.artifacts_by_type.get("SfMArtifact")
        dec_art = self.artifacts_by_type.get("DecodedFramesArtifact")

        # CASE A: No telemetry available
        if telemetry_art is None or not telemetry_art.payload:
            payload = {
                "status": "NOT_EVALUABLE",
                "metric_scale_status": "SCALE_AMBIGUOUS",
                "is_metric_scale": False,
                "depth_unit": "RECONSTRUCTION_UNITS",
                "reason": "Independent telemetry unavailable; metric scale remains ambiguous.",
            }
            art = GeospatialArtifact(
                artifact_id=f"art_geo_{uuid.uuid4().hex[:8]}",
                payload=payload,
                units="RECONSTRUCTION_UNITS",
                coordinate_frame="MONOCULAR_GAUGE",
            )
            self.artifacts_by_id[art.artifact_id] = art
            self.artifacts_by_type[art.artifact_type] = art
            return StageStatus.NOT_EVALUABLE

        # CASE B: Telemetry present
        if ba_art is not None and getattr(ba_art.payload, "refined_reconstruction", None) is not None:
            camera_poses = ba_art.payload.refined_reconstruction.camera_poses
        elif sfm_art is not None:
            camera_poses = sfm_art.payload.camera_poses
        else:
            camera_poses = {}

        if len(camera_poses) == 0:
            return StageStatus.FAILED

        camera_centers_rec: Dict[str, np.ndarray] = {}
        camera_rotations_rec: Dict[str, np.ndarray] = {}
        for cid, pose in camera_poses.items():
            R_cw, _, C_w = self._extract_camera_rt_and_center(pose)
            camera_rotations_rec[cid] = R_cw
            camera_centers_rec[cid] = C_w

        camera_timestamps_s: Dict[str, float] = {}
        if dec_art is not None:
            for f in dec_art.payload:
                camera_timestamps_s[f.frame_id] = getattr(f, "pts_seconds", 0.0)
        else:
            for idx, cid in enumerate(sorted(camera_centers_rec.keys())):
                camera_timestamps_s[cid] = float(idx) * 0.1

        telemetry_records = telemetry_art.payload
        reconstructor = GeospatialMetricReconstructor()
        geo_result: GeospatialMetricReconstructionResult = reconstructor.reconstruct(
            camera_centers_rec=camera_centers_rec,
            camera_timestamps_s=camera_timestamps_s,
            telemetry_records=telemetry_records,
            camera_rotations_rec=camera_rotations_rec,
        )

        is_metric = geo_result.is_metric_scale
        units = "METRES" if is_metric else "RECONSTRUCTION_UNITS"
        frame = "TOPOCENTRIC_ENU" if is_metric else "MONOCULAR_GAUGE"

        art = GeospatialArtifact(
            artifact_id=f"art_geo_{uuid.uuid4().hex[:8]}",
            payload=geo_result,
            units=units,
            coordinate_frame=frame,
        )
        self.artifacts_by_id[art.artifact_id] = art
        self.artifacts_by_type[art.artifact_type] = art
        return StageStatus.SUCCESS

    # -----------------------------------------------------------------------
    # Step 16: Final Validation & Finalization
    # -----------------------------------------------------------------------

    def _execute_final_validation(self) -> StageStatus:
        """STG-16: Final Claim-Policy Audit and Evidence Validation."""
        ba_art = self.artifacts_by_type.get("BundleAdjustmentArtifact")
        sfm_art = self.artifacts_by_type.get("SfMArtifact")
        geometric_status = "SUCCESS" if (ba_art is not None or sfm_art is not None) else "FAILED"

        fusion_art = self.artifacts_by_type.get("DenseFusionArtifact")
        dense_status = "SUCCESS" if fusion_art is not None else "NOT_EVALUABLE"

        surface_art = self.artifacts_by_type.get("SurfaceArtifact")
        surface_status = "SUCCESS" if surface_art is not None else "NOT_EVALUABLE"

        atlas_art = self.artifacts_by_type.get("TexturedSurfaceArtifact")
        texture_status = "SUCCESS" if atlas_art is not None else "NOT_EVALUABLE"

        geo_art = self.artifacts_by_type.get("GeospatialArtifact")
        is_metric = False
        if geo_art is not None:
            if isinstance(geo_art.payload, dict):
                is_metric = geo_art.payload.get("is_metric_scale", False)
                geospatial_status = "EVALUATED" if geo_art.payload.get("status") != "NOT_EVALUABLE" else "NOT_EVALUABLE"
            elif hasattr(geo_art.payload, "is_metric_scale"):
                is_metric = getattr(geo_art.payload, "is_metric_scale", False)
                geospatial_status = "EVALUATED"
            else:
                geospatial_status = "NOT_EVALUABLE"
        else:
            geospatial_status = "NOT_EVALUABLE"

        metric_scale_status = "METRICALLY_SCALED" if is_metric else "SCALE_AMBIGUOUS"

        evidence_level = (
            EvidenceLevel.LEVEL_1_TELEMETRY_ONLY
            if geospatial_status == "EVALUATED"
            else EvidenceLevel.LEVEL_0_NO_GROUND_TRUTH
        )

        auth = ClaimPolicyEngine.audit_claim_authorization(
            evidence_level=evidence_level,
            metric_scale_validated=is_metric,
            geospatial_reference_available=(geospatial_status == "EVALUATED"),
            surface_reference_available=(surface_status == "SUCCESS"),
            radiometric_calibration_available=False,
        )

        validation_payload = {
            "validation_status": "PASS",
            "geometric_status": geometric_status,
            "dense_status": dense_status,
            "surface_status": surface_status,
            "texture_status": texture_status,
            "metric_scale_status": metric_scale_status,
            "geospatial_status": geospatial_status,
            "evidence_level": evidence_level.name,
            "evaluable_axes": list(auth.claims_allowed),
            "blocked_claims": list(auth.claims_blocked),
            "provenance": {
                "software_commit": "019deb2",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "total_stages_recorded": len(self.stage_records),
            },
        }

        art = ValidationArtifact(
            artifact_id=f"art_val_{uuid.uuid4().hex[:8]}",
            payload=validation_payload,
        )
        self.artifacts_by_id[art.artifact_id] = art
        self.artifacts_by_type[art.artifact_type] = art
        return StageStatus.SUCCESS

    def _execute_finalization(self) -> StageStatus:
        """STG-17: Compilation of Immutable Final Reconstruction Deliverable."""
        val_art = self.artifacts_by_type.get("ValidationArtifact")
        if val_art is None:
            raise InsufficientInputError(
                "Validation artifact unavailable for finalization.",
                stage=PipelineStageType.FINALIZATION.value,
            )

        val_data = val_art.payload
        is_metric = val_data.get("metric_scale_status") == "METRICALLY_SCALED"
        final_units = "METRES" if is_metric else "RECONSTRUCTION_UNITS"
        final_frame = "TOPOCENTRIC_ENU" if is_metric else "MONOCULAR_GAUGE"

        final_payload = {
            "geometric_status": val_data.get("geometric_status", "UNKNOWN"),
            "dense_status": val_data.get("dense_status", "NOT_EVALUABLE"),
            "surface_status": val_data.get("surface_status", "NOT_EVALUABLE"),
            "texture_status": val_data.get("texture_status", "NOT_EVALUABLE"),
            "metric_scale_status": val_data.get("metric_scale_status", "SCALE_AMBIGUOUS"),
            "geospatial_status": val_data.get("geospatial_status", "NOT_EVALUABLE"),
            "validation_status": val_data.get("validation_status", "PASS"),
            "evidence_level": val_data.get("evidence_level", EvidenceLevel.LEVEL_0_NO_GROUND_TRUTH.name),
            "evaluable_axes": val_data.get("evaluable_axes", []),
            "blocked_claims": val_data.get("blocked_claims", []),
            "provenance": {
                "software_commit": "019deb2",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "total_artifacts": len(self.artifacts_by_id),
                "artifact_types": list(self.artifacts_by_type.keys()),
            },
        }

        art = FinalReconstructionArtifact(
            artifact_id=f"art_final_{uuid.uuid4().hex[:8]}",
            payload=final_payload,
            units=final_units,
            coordinate_frame=final_frame,
        )
        self.artifacts_by_id[art.artifact_id] = art
        self.artifacts_by_type[art.artifact_type] = art
        return StageStatus.SUCCESS

    # -----------------------------------------------------------------------
    # Main Execution Loop
    # -----------------------------------------------------------------------

    def execute_stage(
        self,
        stage_type: PipelineStageType,
        input_artifacts: List[PipelineArtifact],
    ) -> StageExecutionRecord:
        """Dispatches execution to the corresponding stage implementation."""
        t0 = time.perf_counter()
        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        classification = self.STAGE_CLASSIFICATIONS[stage_type]

        # 1. Verify inputs
        for art in input_artifacts:
            self.verify_input_artifact(art, stage_type)
            self.artifacts_by_id[art.artifact_id] = art
            self.artifacts_by_type[art.artifact_type] = art

        input_ids = [art.artifact_id for art in input_artifacts]

        # 2. Check conditional gating
        if stage_type == PipelineStageType.DENSE_STEREO and not self.config.enable_dense_stereo:
            return self.record_stage(stage_type, StageStatus.NOT_EVALUABLE, started_at, time.perf_counter() - t0, input_ids=input_ids, diagnostics={"reason": "enable_dense_stereo is configured to False"})

        if stage_type == PipelineStageType.DENSE_POINT_GENERATION and not self.config.enable_dense_point_generation:
            return self.record_stage(stage_type, StageStatus.NOT_EVALUABLE, started_at, time.perf_counter() - t0, input_ids=input_ids, diagnostics={"reason": "enable_dense_point_generation is configured to False"})

        if stage_type == PipelineStageType.DENSE_FUSION and not self.config.enable_dense_fusion:
            return self.record_stage(stage_type, StageStatus.NOT_EVALUABLE, started_at, time.perf_counter() - t0, input_ids=input_ids, diagnostics={"reason": "enable_dense_fusion is configured to False"})

        if stage_type == PipelineStageType.SURFACE_RECONSTRUCTION and not self.config.enable_surface_meshing:
            return self.record_stage(stage_type, StageStatus.NOT_EVALUABLE, started_at, time.perf_counter() - t0, input_ids=input_ids, diagnostics={"reason": "enable_surface_meshing is configured to False"})

        if stage_type in (PipelineStageType.TEXTURE_ASSOCIATION, PipelineStageType.TEXTURE_RECONSTRUCTION) and not self.config.enable_texturing:
            return self.record_stage(stage_type, StageStatus.NOT_EVALUABLE, started_at, time.perf_counter() - t0, input_ids=input_ids, diagnostics={"reason": "enable_texturing is configured to False"})

        if stage_type == PipelineStageType.GEOSPATIAL_TRANSFORM and not self.config.enable_geospatial:
            return self.record_stage(stage_type, StageStatus.NOT_EVALUABLE, started_at, time.perf_counter() - t0, input_ids=input_ids, diagnostics={"reason": "enable_geospatial is configured to False"})

        # 3. Dispatch implemented stages (Steps 1–16)
        status = StageStatus.SUCCESS
        diagnostics: Dict[str, Any] = {}

        try:
            if stage_type == PipelineStageType.INGESTION:
                status = self._execute_ingestion()
            elif stage_type == PipelineStageType.DECODING:
                status = self._execute_decoding()
            elif stage_type == PipelineStageType.FRAME_INTELLIGENCE:
                status = self._execute_frame_intelligence()
            elif stage_type == PipelineStageType.KEYFRAME_SELECTION:
                status = self._execute_keyframe_selection()
            elif stage_type == PipelineStageType.CORRESPONDENCE:
                status = self._execute_correspondences()
            elif stage_type == PipelineStageType.TWO_VIEW_GEOMETRY:
                status = self._execute_two_view_geometry()
            elif stage_type == PipelineStageType.INCREMENTAL_SFM:
                status = self._execute_incremental_sfm()
            elif stage_type == PipelineStageType.BUNDLE_ADJUSTMENT:
                status = self._execute_bundle_adjustment()
            elif stage_type == PipelineStageType.DENSE_STEREO:
                status = self._execute_dense_stereo()
            elif stage_type == PipelineStageType.DENSE_POINT_GENERATION:
                status = self._execute_dense_point_generation()
            elif stage_type == PipelineStageType.DENSE_FUSION:
                status = self._execute_dense_fusion()
            elif stage_type == PipelineStageType.SURFACE_RECONSTRUCTION:
                status = self._execute_surface_reconstruction()
            elif stage_type == PipelineStageType.TEXTURE_ASSOCIATION:
                status = self._execute_texture_association()
            elif stage_type == PipelineStageType.TEXTURE_RECONSTRUCTION:
                status = self._execute_texture_reconstruction()
            elif stage_type == PipelineStageType.GEOSPATIAL_TRANSFORM:
                status = self._execute_geospatial()
            elif stage_type == PipelineStageType.FINAL_VALIDATION:
                status = self._execute_final_validation()
            elif stage_type == PipelineStageType.FINALIZATION:
                status = self._execute_finalization()
            else:
                status = StageStatus.NOT_EVALUABLE
                diagnostics = {"step": f"Unknown stage {stage_type}"}
        except (DataLeakageError, ContractViolationError):
            raise
        except InsufficientInputError as e:
            status = StageStatus.INSUFFICIENT_INPUT
            diagnostics = {"error": str(e), "error_class": e.__class__.__name__}
        except Exception as e:
            status = StageStatus.FAILED
            diagnostics = {"error": str(e), "error_class": e.__class__.__name__}

        output_ids = [art.artifact_id for art in self.artifacts_by_id.values()]
        return self.record_stage(
            stage_type,
            status,
            started_at,
            time.perf_counter() - t0,
            input_ids=input_ids,
            output_ids=output_ids,
            diagnostics=diagnostics,
        )

    def run(self, input_artifacts: Optional[List[PipelineArtifact]] = None) -> PipelineResult:
        """Executes the complete pipeline sequence across all DAG stages."""
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        input_artifacts = input_artifacts or []

        # 1. Store input artifacts and check for evaluation truth leakage
        for art in input_artifacts:
            if art.domain == ArtifactDomain.EVALUATION_TRUTH:
                raise DataLeakageError(
                    f"Data Leakage Violation: Input artifact '{art.artifact_id}' has domain EVALUATION_TRUTH.",
                    stage="PIPELINE_INIT",
                )
            self.artifacts_by_id[art.artifact_id] = art
            self.artifacts_by_type[art.artifact_type] = art

        # 2. Iterate through DAG stages in strict order
        pipeline_status = PipelineStatus.SUCCESS
        failure_reason: Optional[str] = None

        for stage_type in self.DAG_SEQUENCE:
            classification = self.STAGE_CLASSIFICATIONS[stage_type]
            current_inputs = list(self.artifacts_by_id.values())

            try:
                record = self.execute_stage(stage_type, current_inputs)
            except Exception as e:
                record = self.record_stage(
                    stage_type,
                    StageStatus.CONTRACT_VIOLATION if isinstance(e, ContractViolationError) else StageStatus.FAILED,
                    datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    0.0,
                    input_ids=[art.artifact_id for art in current_inputs],
                    diagnostics={"error": str(e), "error_class": e.__class__.__name__},
                )
                if classification == StageClassification.MANDATORY:
                    pipeline_status = (
                        PipelineStatus.CONTRACT_VIOLATION
                        if isinstance(e, ContractViolationError)
                        else PipelineStatus.FAILED
                    )
                    failure_reason = f"Mandatory stage '{stage_type.value}' failed: {e}"
                    break

            if record.status in (StageStatus.FAILED, StageStatus.INSUFFICIENT_INPUT, StageStatus.CONTRACT_VIOLATION):
                if classification == StageClassification.MANDATORY:
                    if record.status == StageStatus.CONTRACT_VIOLATION:
                        pipeline_status = PipelineStatus.CONTRACT_VIOLATION
                    elif record.status == StageStatus.INSUFFICIENT_INPUT:
                        pipeline_status = PipelineStatus.FAILED
                    else:
                        pipeline_status = PipelineStatus.FAILED
                    failure_reason = f"Mandatory stage '{stage_type.value}' terminated with status {record.status.value}"
                    break

        return self.finalize(run_id, pipeline_status, failure_reason)

    def finalize(
        self,
        run_id: str,
        pipeline_status: PipelineStatus,
        failure_reason: Optional[str] = None,
    ) -> PipelineResult:
        """Packages the final PipelineResult."""
        final_art = self.artifacts_by_type.get("FinalReconstructionArtifact")
        geo_art = self.artifacts_by_type.get("GeospatialArtifact")

        is_metric = False
        if final_art is not None and isinstance(final_art.payload, dict):
            is_metric = (final_art.payload.get("metric_scale_status") == "METRICALLY_SCALED")
        elif geo_art is not None:
            if hasattr(geo_art.payload, "is_metric_scale"):
                is_metric = getattr(geo_art.payload, "is_metric_scale", False)
            elif isinstance(geo_art.payload, dict):
                is_metric = geo_art.payload.get("is_metric_scale", False)

        if any(rec.status in (StageStatus.FAILED, StageStatus.CONTRACT_VIOLATION) for rec in self.stage_records):
            if pipeline_status == PipelineStatus.SUCCESS:
                pipeline_status = PipelineStatus.FAILED

        if pipeline_status == PipelineStatus.SUCCESS:
            metric_status = (
                MetricScaleStatus.METRICALLY_SCALED
                if is_metric
                else MetricScaleStatus.SCALE_AMBIGUOUS
            )
            reconstruction_units = (
                ReconstructionUnitType.METRIC_UNITS
                if is_metric
                else ReconstructionUnitType.RECONSTRUCTION_UNITS
            )
        else:
            metric_status = MetricScaleStatus.NOT_EVALUABLE
            reconstruction_units = ReconstructionUnitType.RECONSTRUCTION_UNITS

        return PipelineResult(
            run_id=run_id,
            pipeline_status=pipeline_status,
            metric_scale_status=metric_status,
            reconstruction_units=reconstruction_units,
            stage_records=list(self.stage_records),
            output_artifacts={art_type: art.artifact_id for art_type, art in self.artifacts_by_type.items()},
            diagnostics={"total_stages_executed": len(self.stage_records)},
            failure_reason=failure_reason,
        )
