"""Dynamic Scene Analysis Contract & Temporal Evidence Subsystem.

Provides a detector-agnostic framework and deterministic temporal evidence engine
for candidate dynamic scene regions.

SCIENTIFIC INTERPRETATION BOUNDARIES:
- IMAGE MOTION != OBJECT MOTION != SEMANTIC OBJECT IDENTITY.
- A semantic label (e.g. 'car', 'person', 'boat') does NOT automatically mean the object is moving.
  Parked cars and static structures must be evaluated based on temporal motion evidence.
- A dynamic region does NOT require a semantic label (e.g. unknown moving objects or animals).
- Local motion discrepancy from the dominant global flow provides diagnostic evidence,
  NOT definitive proof of independent physical motion (parallax and depth discontinuities can mimic local flow).
- Dynamic evidence scores are heuristic diagnostic indices in [0.0, 1.0], NOT calibrated probabilities.
- All default thresholds are HEURISTIC_DEFAULT baselines requiring empirical validation.
- This subsystem produces DIAGNOSTICS ONLY. It must NEVER drop frames or modify source images.
"""

import json
import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple

from src.preprocessing.decoder import DecodedFrame, DecodeStatus
from src.quality.temporal_motion import (
    TemporalMotionAnalyzer,
    TemporalMotionConfig,
    TemporalMotionBlurReport,
    MotionCategory,
)


class DynamicEvidenceCategory(str, Enum):
    """Categorical diagnostic hypothesis for dynamic scene content."""
    STATIC_EVIDENCE = "STATIC_EVIDENCE"
    POSSIBLY_DYNAMIC = "POSSIBLY_DYNAMIC"
    DYNAMIC_EVIDENCE = "DYNAMIC_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class RegionMaskReference:
    """Lightweight reference to a region segmentation mask or bounding geometry."""
    mask_type: str                               # "BBOX_ONLY", "RLE", "POLYGON", "FILE_REF"
    mask_uri: Optional[str] = None               # URI or relative path if stored externally
    rle_counts: Optional[str] = None             # Run-length encoded representation if string
    polygon_points: Optional[List[Tuple[float, float]]] = None # Normalized polygon coordinates (x, y)


@dataclass(frozen=True)
class CandidateDynamicRegion:
    """A candidate spatial region evaluated for dynamic scene evidence."""
    region_id: str
    bbox: Tuple[int, int, int, int]              # (y_min, x_min, y_max, x_max) in pixel coordinates
    mask_ref: RegionMaskReference
    semantic_label: Optional[str]                # Raw label from external provider (e.g., "car", "person")
    semantic_confidence: Optional[float]         # Confidence score from external detector [0, 1]
    provider_name: str                           # Name of detector/provider source
    local_velocity_px_per_sec: float             # Mean apparent velocity within region
    relative_motion_discrepancy: float           # Norm of (v_region - v_global_dominant) [px/s]
    temporal_persistence_count: int              # Number of consecutive frames region pattern persisted
    dynamic_evidence_score: float                # Diagnostic heuristic index in [0.0, 1.0]
    evidence_category: DynamicEvidenceCategory


@dataclass(frozen=True)
class DynamicSceneConfig:
    """Configurable heuristic thresholds for dynamic scene analysis.
    
    All default values are heuristic defaults (HEURISTIC_DEFAULT) requiring empirical
    calibration on representative flight validation datasets.
    """
    motion_discrepancy_threshold_px_s: float = 4.0 # HEURISTIC_DEFAULT: Velocity discrepancy to flag local motion
    min_persistence_frames: int = 2               # HEURISTIC_DEFAULT: Consecutive frames to establish persistence
    dynamic_score_threshold: float = 0.50         # HEURISTIC_DEFAULT: Score bound for DYNAMIC_EVIDENCE
    max_temporal_gap_seconds: float = 2.0         # HEURISTIC_DEFAULT: Max allowed inter-frame delta
    config_version: str = "DynamicScene_v1.0"


class DynamicRegionProvider(ABC):
    """Detector-agnostic abstract provider interface for candidate dynamic regions."""

    @abstractmethod
    def detect_candidate_regions(
        self, frame: DecodedFrame
    ) -> List[Tuple[Tuple[int, int, int, int], Optional[str], Optional[float], RegionMaskReference]]:
        """Extract candidate bounding boxes and optional semantic annotations.
        
        Returns:
            List of tuples: (bbox, semantic_label, semantic_confidence, mask_ref)
        """
        pass


class SyntheticDynamicRegionProvider(DynamicRegionProvider):
    """Deterministic synthetic provider for testing and pipeline validation (TEST DATA)."""

    def __init__(
        self,
        canned_regions: Optional[
            List[Tuple[Tuple[int, int, int, int], Optional[str], Optional[float], RegionMaskReference]]
        ] = None,
    ):
        self.canned_regions = canned_regions or []

    def detect_candidate_regions(
        self, frame: DecodedFrame
    ) -> List[Tuple[Tuple[int, int, int, int], Optional[str], Optional[float], RegionMaskReference]]:
        return self.canned_regions


@dataclass
class DynamicSceneReport:
    """Structured dynamic scene diagnostic report for a single video frame."""
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    source_video: str
    candidate_regions: List[CandidateDynamicRegion]
    global_motion_velocity_px_per_sec: float
    dominant_motion_vector: Tuple[float, float]  # (u_global, v_global) in px/s
    overall_scene_status: DynamicEvidenceCategory
    static_scene_fraction: float                 # Estimated fraction of frame exhibiting static behavior [0, 1]
    diagnostics: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    config_version: str = "DynamicScene_v1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to serializable dictionary."""
        d = asdict(self)
        d["overall_scene_status"] = self.overall_scene_status.value
        for r in d["candidate_regions"]:
            r["evidence_category"] = r["evidence_category"].value
        return d

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class DynamicSceneAnalyzer:
    """Deterministic engine for evaluating dynamic scene evidence across temporal sequences."""

    @classmethod
    def analyze_frame(
        cls,
        target_frame: DecodedFrame,
        region_provider: Optional[DynamicRegionProvider] = None,
        prev_frame: Optional[DecodedFrame] = None,
        next_frame: Optional[DecodedFrame] = None,
        temporal_report: Optional[TemporalMotionBlurReport] = None,
        historical_persistence: Optional[Dict[str, int]] = None,
        config: Optional[DynamicSceneConfig] = None,
    ) -> DynamicSceneReport:
        """Analyze candidate regions and evaluate temporal dynamic evidence.
        
        DOES NOT MODIFY frame.data.
        """
        cfg = config or DynamicSceneConfig()
        diagnostics: List[str] = []
        historical = historical_persistence or {}

        # Validate target frame
        if target_frame.decode_status != DecodeStatus.SUCCESS or target_frame.data is None:
            raise ValueError(f"Cannot perform dynamic scene analysis on frame '{target_frame.frame_id}' with invalid decode status.")

        # Obtain temporal motion evidence
        if temporal_report is None:
            temporal_report = TemporalMotionAnalyzer.analyze_temporal_motion(
                target_frame=target_frame,
                prev_frame=prev_frame,
                next_frame=next_frame,
            )

        # Global dominant motion velocity
        global_vel = temporal_report.global_velocity_px_per_sec
        dt_eff = temporal_report.provenance.get("effective_dt_seconds", 1.0) or 1.0
        
        # Estimate dominant global vector from mean displacement
        u_global = (temporal_report.mean_displacement_pixels / dt_eff) * temporal_report.directional_coherence_score
        v_global = 0.0 # Standard projection
        dominant_vector = (round(u_global, 4), round(v_global, 4))

        # Query region provider if available
        raw_candidates = []
        provider_name = "None"
        if region_provider is not None:
            provider_name = region_provider.__class__.__name__
            raw_candidates = region_provider.detect_candidate_regions(target_frame)

        evaluated_regions: List[CandidateDynamicRegion] = []
        total_dynamic_area = 0
        h, w = target_frame.height, target_frame.width
        frame_area = max(1, h * w)

        # Evaluate optical flow displacement field if neighbors exist
        has_temporal_context = len(temporal_report.neighbor_frame_ids) > 0

        for i, (bbox, sem_label, sem_conf, mask_ref) in enumerate(raw_candidates):
            reg_id = f"{target_frame.frame_id}_reg_{i:03d}"
            y_min, x_min, y_max, x_max = bbox
            y_min = max(0, min(h, y_min))
            y_max = max(0, min(h, y_max))
            x_min = max(0, min(w, x_min))
            x_max = max(0, min(w, x_max))
            reg_area = max(1, (y_max - y_min) * (x_max - x_min))

            # Query tile-level spatial motion in the region's bounding box
            intersecting_tiles = [
                t for t in temporal_report.spatial_tiles
                if not (t.bbox[2] <= y_min or t.bbox[0] >= y_max or t.bbox[3] <= x_min or t.bbox[1] >= x_max)
            ]

            if intersecting_tiles and has_temporal_context:
                local_vel = float(np.mean([t.mean_velocity_px_per_sec for t in intersecting_tiles]))
                local_disp = float(np.mean([t.mean_displacement_pixels for t in intersecting_tiles]))
            else:
                local_vel = global_vel
                local_disp = temporal_report.median_displacement_pixels

            # Compute relative motion discrepancy from dominant global motion
            motion_discrepancy = abs(local_vel - global_vel)

            # Update temporal persistence
            persistence_key = f"{sem_label or 'unlabeled'}_{x_min // 20}_{y_min // 20}"
            persisted_frames = historical.get(persistence_key, 0) + 1

            # Compute Dynamic Evidence Score (Diagnostic Heuristic Index [0.0, 1.0])
            # High discrepancy + persistence = elevated dynamic evidence
            if not has_temporal_context:
                dyn_score = 0.0
                category = DynamicEvidenceCategory.INSUFFICIENT_EVIDENCE
            else:
                norm_discrepancy = min(1.0, motion_discrepancy / max(cfg.motion_discrepancy_threshold_px_s * 2.0, 1.0))
                norm_persistence = min(1.0, persisted_frames / max(cfg.min_persistence_frames, 1))

                # Weight discrepancy heavily over semantic label
                # Semantic label alone NEVER elevates dynamic score
                dyn_score = round(float(0.70 * norm_discrepancy + 0.30 * (norm_discrepancy * norm_persistence)), 4)

                if motion_discrepancy >= cfg.motion_discrepancy_threshold_px_s:
                    if persisted_frames >= cfg.min_persistence_frames:
                        category = DynamicEvidenceCategory.DYNAMIC_EVIDENCE
                        total_dynamic_area += reg_area
                    else:
                        category = DynamicEvidenceCategory.POSSIBLY_DYNAMIC
                elif motion_discrepancy < 1.0:
                    category = DynamicEvidenceCategory.STATIC_EVIDENCE
                else:
                    category = DynamicEvidenceCategory.POSSIBLY_DYNAMIC

            evaluated_regions.append(
                CandidateDynamicRegion(
                    region_id=reg_id,
                    bbox=(int(y_min), int(x_min), int(y_max), int(x_max)),
                    mask_ref=mask_ref,
                    semantic_label=sem_label,
                    semantic_confidence=sem_conf,
                    provider_name=provider_name,
                    local_velocity_px_per_sec=round(local_vel, 4),
                    relative_motion_discrepancy=round(motion_discrepancy, 4),
                    temporal_persistence_count=persisted_frames,
                    dynamic_evidence_score=dyn_score,
                    evidence_category=category,
                )
            )

        # Overall scene categorization
        if not has_temporal_context:
            overall_status = DynamicEvidenceCategory.INSUFFICIENT_EVIDENCE
            diagnostics.append("Insufficient adjacent temporal context to evaluate dynamic evidence.")
        elif any(r.evidence_category == DynamicEvidenceCategory.DYNAMIC_EVIDENCE for r in evaluated_regions):
            overall_status = DynamicEvidenceCategory.DYNAMIC_EVIDENCE
            diagnostics.append("Persistent independent local motion detected in candidate regions.")
        elif any(r.evidence_category == DynamicEvidenceCategory.POSSIBLY_DYNAMIC for r in evaluated_regions):
            overall_status = DynamicEvidenceCategory.POSSIBLY_DYNAMIC
        else:
            overall_status = DynamicEvidenceCategory.STATIC_EVIDENCE

        static_fraction = max(0.0, min(1.0, 1.0 - (total_dynamic_area / frame_area)))

        provenance = {
            "target_frame_id": target_frame.frame_id,
            "target_frame_index": target_frame.frame_index,
            "target_timestamp_seconds": target_frame.timestamp_seconds,
            "region_provider": provider_name,
            "candidate_count": len(evaluated_regions),
            "discrepancy_threshold_px_s": cfg.motion_discrepancy_threshold_px_s,
        }

        return DynamicSceneReport(
            frame_id=target_frame.frame_id,
            frame_index=target_frame.frame_index,
            timestamp_seconds=target_frame.timestamp_seconds,
            source_video=target_frame.source_video,
            candidate_regions=evaluated_regions,
            global_motion_velocity_px_per_sec=round(global_vel, 4),
            dominant_motion_vector=dominant_vector,
            overall_scene_status=overall_status,
            static_scene_fraction=round(static_fraction, 4),
            diagnostics=diagnostics,
            provenance=provenance,
            config_version=cfg.config_version,
        )
