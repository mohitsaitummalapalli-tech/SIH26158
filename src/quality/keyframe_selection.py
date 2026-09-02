"""Coverage-Aware Keyframe Selection Subsystem.

Provides deterministic, explainable keyframe selection based on two-stage filtering:
Stage A (Hard Safety Gate) followed by Stage B (Greedy Marginal-Gain Coverage Selection).

SCIENTIFIC INTERPRETATION BOUNDARIES:
- FRAME QUALITY != FRAME USEFULNESS.
  A pristine frame may be geometrically redundant, while a moderately textured frame
  may provide critical baseline coverage.
- VISUAL NOVELTY != GEOMETRIC NOVELTY.
  Visual similarity on planar surfaces != identical 3D camera geometry.
- GNSS TRAJECTORY DIVERSITY != CAMERA VIEWPOINT DIVERSITY.
  Without lever-arm and gimbal calibration, ENU coordinates and attitude quaternions
  are platform navigation proxies (TRAJECTORY_PROXY / ORIENTATION_PROXY).
- DYNAMIC AREA FRACTION != GEOMETRIC CONTAMINATION.
  A large dynamic candidate region with near-zero feature matches in the dynamic area
  does not corrupt static background correspondences.
- GREEDY HEURISTIC != GLOBALLY OPTIMAL SELECTION.
  Selection is an explainable engineering policy to be validated against downstream reconstruction outcomes.
- SELECTION DOES NOT PHYSICALLY DELETE FRAMES.
  It produces candidate reference indices for downstream geometry stages.
"""

import json
import math
import numpy as np
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple

from src.preprocessing.decoder import DecodedFrame, DecodeStatus
from src.quality.assessment import FrameQualityReport, QualityStatus
from src.quality.temporal_motion import TemporalMotionBlurReport
from src.quality.photometric import PhotometricStabilityReport
from src.quality.dynamic_scene import DynamicSceneReport, DynamicEvidenceCategory
from src.quality.redundancy_viewpoint import (
    FrameRedundancyReport,
    FramePairRelation,
    FrameRedundancyViewpointAnalyzer,
)


class SelectionReason(str, Enum):
    """Machine-readable explainable reason for selecting a keyframe."""
    INITIAL_ANCHOR = "INITIAL_ANCHOR"
    INITIAL_BOUNDARY_FALLBACK = "INITIAL_BOUNDARY_FALLBACK"
    FINAL_ANCHOR = "FINAL_ANCHOR"
    FINAL_BOUNDARY_FALLBACK = "FINAL_BOUNDARY_FALLBACK"
    BOUNDARY_UNCOVERED = "BOUNDARY_UNCOVERED"
    TEMPORAL_GAP_COVERAGE = "TEMPORAL_GAP_COVERAGE"
    FEATURE_NOVELTY = "FEATURE_NOVELTY"
    SPATIAL_COVERAGE = "SPATIAL_COVERAGE"
    TRAJECTORY_DIVERSITY = "TRAJECTORY_DIVERSITY"
    ORIENTATION_DIVERSITY = "ORIENTATION_DIVERSITY"
    QUALITY_TIE_BREAK = "QUALITY_TIE_BREAK"
    FALLBACK = "FALLBACK"


class RejectionReason(str, Enum):
    """Machine-readable reason for deprioritizing or rejecting a candidate frame."""
    UNSAFE_DECODE_FAILURE = "UNSAFE_DECODE_FAILURE"
    UNSAFE_SEVERELY_DEGRADED = "UNSAFE_SEVERELY_DEGRADED"
    BELOW_MIN_TEMPORAL_SPACING = "BELOW_MIN_TEMPORAL_SPACING"
    HIGH_REDUNDANCY = "HIGH_REDUNDANCY"
    DYNAMIC_RISK_EXCLUSION = "DYNAMIC_RISK_EXCLUSION"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class KeyframeSelectionConfig:
    """Configurable parameters and heuristic thresholds for keyframe selection.
    
    All default values are heuristic defaults (HEURISTIC_DEFAULT) requiring empirical
    validation on representative reconstruction benchmarks.
    """
    target_keyframe_count: Optional[int] = None       # Target budget limit
    max_keyframe_count: Optional[int] = None          # Hard upper budget limit
    min_temporal_spacing_seconds: float = 0.25        # HEURISTIC_DEFAULT: Minimum allowed time between keyframes (s)
    max_temporal_gap_seconds: float = 2.5             # HEURISTIC_DEFAULT: Max allowed gap before forced keyframe (s)
    min_trajectory_baseline_meters: float = 0.5       # HEURISTIC_DEFAULT: Minimum trajectory spatial movement (m)
    min_orientation_change_degrees: float = 2.0       # HEURISTIC_DEFAULT: Minimum attitude change (deg)
    max_dynamic_contamination_fraction: float = 0.60  # HEURISTIC_DEFAULT: Dynamic area fraction to reject
    force_first_and_last_frames: bool = True          # HEURISTIC_DEFAULT: Anchor boundaries when safe
    enable_fallback_on_empty: bool = True             # Preserves temporal coverage if diagnostics fail
    config_version: str = "KeyframeSelection_v1.1"


@dataclass(frozen=True)
class SelectedKeyframe:
    """A selected keyframe reference with explainable selection diagnostics."""
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    primary_reason: SelectionReason
    detailed_reasons: List[str]
    marginal_gain_breakdown: Dict[str, float]
    greedy_heuristic_gain: Optional[float] = None
    expected_trajectory_displacement_meters: Optional[float] = None  # TRAJECTORY_PROXY


@dataclass(frozen=True)
class DeprioritizedCandidate:
    """A candidate frame excluded or deprioritized by policy."""
    frame_id: str
    frame_index: int
    timestamp_seconds: float
    rejection_reason: RejectionReason
    explanation: str


@dataclass
class KeyframeSelectionResult:
    """Structured, provenance-preserving keyframe selection output."""
    selected_keyframe_ids: List[str]
    selected_frame_indices: List[int]
    selected_timestamps_seconds: List[float]
    selected_keyframes: List[SelectedKeyframe]
    deprioritized_candidates: List[DeprioritizedCandidate]
    total_input_frames: int
    selected_count: int
    reduction_ratio: float                            # (total - selected) / total
    fallback_used: bool
    diagnostics: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    config_version: str = "KeyframeSelection_v1.1"

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        d = asdict(self)
        for k in d["selected_keyframes"]:
            k["primary_reason"] = k["primary_reason"].value
        for c in d["deprioritized_candidates"]:
            c["rejection_reason"] = c["rejection_reason"].value
        return d

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class CoverageAwareKeyframeSelector:
    """Deterministic, explainable greedy marginal-gain keyframe selector."""

    @classmethod
    def select_keyframes(
        cls,
        frames: List[DecodedFrame],
        quality_reports: Optional[Dict[str, FrameQualityReport]] = None,
        motion_reports: Optional[Dict[str, TemporalMotionBlurReport]] = None,
        photometric_reports: Optional[Dict[str, PhotometricStabilityReport]] = None,
        dynamic_reports: Optional[Dict[str, DynamicSceneReport]] = None,
        redundancy_reports: Optional[Dict[str, FrameRedundancyReport]] = None,
        enu_positions: Optional[Dict[str, Tuple[float, float, float]]] = None,
        quaternions: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
        ground_speeds: Optional[Dict[str, float]] = None,
        config: Optional[KeyframeSelectionConfig] = None,
    ) -> KeyframeSelectionResult:
        """Execute two-stage coverage-aware keyframe selection over candidate frame sequence."""
        cfg = config or KeyframeSelectionConfig()
        diagnostics: List[str] = []
        deprioritized: List[DeprioritizedCandidate] = []
        q_map = quality_reports or {}
        dyn_map = dynamic_reports or {}
        red_map = redundancy_reports or {}
        enu_map = enu_positions or {}
        quat_map = quaternions or {}
        speed_map = ground_speeds or {}

        total_frames = len(frames)
        if total_frames == 0:
            return KeyframeSelectionResult(
                selected_keyframe_ids=[],
                selected_frame_indices=[],
                selected_timestamps_seconds=[],
                selected_keyframes=[],
                deprioritized_candidates=[],
                total_input_frames=0,
                selected_count=0,
                reduction_ratio=0.0,
                fallback_used=False,
                diagnostics=["Input frames list is empty."],
                provenance={"config_version": cfg.config_version},
                config_version=cfg.config_version,
            )

        # Check for Budget / Coverage Incompatibility
        if cfg.max_keyframe_count is not None and total_frames > 1:
            t_span = frames[-1].timestamp_seconds - frames[0].timestamp_seconds
            if t_span > 0:
                min_req_keyframes = math.ceil(t_span / max(cfg.max_temporal_gap_seconds, 0.1)) + 1
                if cfg.max_keyframe_count < min_req_keyframes:
                    diagnostics.append(
                        f"BUDGET_INCOMPATIBLE_WITH_COVERAGE_CONSTRAINT: Requested max budget ({cfg.max_keyframe_count}) "
                        f"is mathematically insufficient to guarantee max temporal gap ({cfg.max_temporal_gap_seconds:.2f}s) "
                        f"over sequence duration ({t_span:.2f}s, requires >= {min_req_keyframes} keyframes)."
                    )

        # STAGE A: HARD SAFETY GATE
        surviving_candidates: List[DecodedFrame] = []
        for f in frames:
            # 1. Decode failure
            if f.decode_status != DecodeStatus.SUCCESS or f.data is None:
                deprioritized.append(
                    DeprioritizedCandidate(
                        frame_id=f.frame_id,
                        frame_index=f.frame_index,
                        timestamp_seconds=f.timestamp_seconds,
                        rejection_reason=RejectionReason.UNSAFE_DECODE_FAILURE,
                        explanation="Decode status was not SUCCESS or pixel data missing.",
                    )
                )
                continue

            # 2. Severe Quality Degradation
            q_rep = q_map.get(f.frame_id)
            if q_rep and q_rep.status == QualityStatus.SEVERELY_DEGRADED:
                deprioritized.append(
                    DeprioritizedCandidate(
                        frame_id=f.frame_id,
                        frame_index=f.frame_index,
                        timestamp_seconds=f.timestamp_seconds,
                        rejection_reason=RejectionReason.UNSAFE_SEVERELY_DEGRADED,
                        explanation="Frame quality report marked SEVERELY_DEGRADED.",
                    )
                )
                continue

            # 3. Dynamic Contamination Evaluation
            dyn_rep = dyn_map.get(f.frame_id)
            red_rep = red_map.get(f.frame_id)
            if dyn_rep:
                dyn_area_frac = 1.0 - dyn_rep.static_scene_fraction
                # Check if feature matches are available to verify actual contamination
                dyn_match_frac = None
                if red_rep and red_rep.pair_relations:
                    dyn_match_fracs = [r.matches_inside_dynamic_regions_fraction for r in red_rep.pair_relations if r.matches_inside_dynamic_regions_fraction is not None]
                    if dyn_match_fracs:
                        dyn_match_frac = float(np.mean(dyn_match_fracs))

                if dyn_area_frac > cfg.max_dynamic_contamination_fraction:
                    # If feature matches show that static background is actually intact (match contamination < 15%), allow with warning
                    if dyn_match_frac is not None and dyn_match_frac < 0.15:
                        diagnostics.append(
                            f"Frame {f.frame_id}: Dynamic candidate area is high ({dyn_area_frac:.2f}) but static correspondence "
                            f"contamination is low ({dyn_match_frac:.2f}). Retaining candidate with diagnostic warning."
                        )
                    else:
                        deprioritized.append(
                            DeprioritizedCandidate(
                                frame_id=f.frame_id,
                                frame_index=f.frame_index,
                                timestamp_seconds=f.timestamp_seconds,
                                rejection_reason=RejectionReason.DYNAMIC_RISK_EXCLUSION,
                                explanation=f"Dynamic contamination ({dyn_area_frac:.2f}) exceeds threshold ({cfg.max_dynamic_contamination_fraction:.2f}).",
                            )
                        )
                        continue

            surviving_candidates.append(f)

        # Fallback check
        fallback_used = False
        if not surviving_candidates and cfg.enable_fallback_on_empty and frames:
            diagnostics.append("FALLBACK_USED: All candidates failed hard safety gate. Retaining first valid decoded frame as fallback.")
            fallback_used = True
            valid_f = next((f for f in frames if f.data is not None), frames[0])
            surviving_candidates = [valid_f]

        # STAGE B: GREEDY MARGINAL-GAIN COVERAGE SELECTION
        selected: List[SelectedKeyframe] = []

        if surviving_candidates:
            # Step 1: Initial Boundary Anchor
            first_safe = surviving_candidates[0]
            is_true_initial = (first_safe.frame_id == frames[0].frame_id)

            if fallback_used:
                initial_reason = SelectionReason.FALLBACK
                initial_detail = "Fallback anchor."
            elif is_true_initial:
                initial_reason = SelectionReason.INITIAL_ANCHOR
                initial_detail = "Sequence temporal origin anchor."
            else:
                initial_reason = SelectionReason.INITIAL_BOUNDARY_FALLBACK
                initial_detail = f"First frame was unsafe; anchored nearest safe candidate (frame {first_safe.frame_id})."

            speed_init = speed_map.get(first_safe.frame_id)
            selected.append(
                SelectedKeyframe(
                    frame_id=first_safe.frame_id,
                    frame_index=first_safe.frame_index,
                    timestamp_seconds=first_safe.timestamp_seconds,
                    primary_reason=initial_reason,
                    detailed_reasons=[initial_detail],
                    marginal_gain_breakdown={"temporal": 1.0, "feature": 1.0, "trajectory": 1.0},
                    greedy_heuristic_gain=1.0,
                    expected_trajectory_displacement_meters=0.0,
                )
            )

            # Step 2: Sequential Greedy Selection
            last_selected_time = first_safe.timestamp_seconds
            last_selected_enu = enu_map.get(first_safe.frame_id)
            last_selected_quat = quat_map.get(first_safe.frame_id)

            max_budget = cfg.max_keyframe_count or (cfg.target_keyframe_count or len(surviving_candidates))

            for cand in surviving_candidates[1:]:
                # Check budget limit
                if len(selected) >= max_budget:
                    deprioritized.append(
                        DeprioritizedCandidate(
                            frame_id=cand.frame_id,
                            frame_index=cand.frame_index,
                            timestamp_seconds=cand.timestamp_seconds,
                            rejection_reason=RejectionReason.BUDGET_EXCEEDED,
                            explanation="Maximum keyframe budget reached.",
                        )
                    )
                    continue

                dt = cand.timestamp_seconds - last_selected_time

                # Enforce Minimum Temporal Spacing
                if dt < cfg.min_temporal_spacing_seconds:
                    deprioritized.append(
                        DeprioritizedCandidate(
                            frame_id=cand.frame_id,
                            frame_index=cand.frame_index,
                            timestamp_seconds=cand.timestamp_seconds,
                            rejection_reason=RejectionReason.BELOW_MIN_TEMPORAL_SPACING,
                            explanation=f"Delta time ({dt:.3f}s) below min temporal spacing ({cfg.min_temporal_spacing_seconds:.3f}s).",
                        )
                    )
                    continue

                # Compute Marginal Utilities (HEURISTIC_DEFAULT)
                # 1. Temporal Gap Contribution
                gain_time = min(1.0, dt / max(cfg.max_temporal_gap_seconds, 0.1))

                # 2. Trajectory Distance Contribution (TRAJECTORY_PROXY)
                cand_enu = enu_map.get(cand.frame_id)
                gain_traj = 0.0
                traj_dist = 0.0
                if cand_enu is not None and last_selected_enu is not None:
                    traj_dist = math.sqrt(
                        (cand_enu[0] - last_selected_enu[0])**2 +
                        (cand_enu[1] - last_selected_enu[1])**2 +
                        (cand_enu[2] - last_selected_enu[2])**2
                    )
                    gain_traj = min(1.0, traj_dist / max(cfg.min_trajectory_baseline_meters * 2.0, 0.1))

                # Speed-aware displacement diagnostic (TRAJECTORY_PROXY)
                cand_speed = speed_map.get(cand.frame_id)
                expected_disp = None
                if cand_speed is not None:
                    expected_disp = round(float(cand_speed * dt), 3)

                # 3. Orientation Diversity Contribution (ORIENTATION_PROXY)
                cand_quat = quat_map.get(cand.frame_id)
                gain_orient = 0.0
                if cand_quat is not None and last_selected_quat is not None:
                    dot = abs(cand_quat[0]*last_selected_quat[0] + cand_quat[1]*last_selected_quat[1] + cand_quat[2]*last_selected_quat[2] + cand_quat[3]*last_selected_quat[3])
                    angle_deg = math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))
                    gain_orient = min(1.0, angle_deg / max(cfg.min_orientation_change_degrees * 2.0, 0.1))

                # 4. Feature Novelty Contribution
                red_rep = red_map.get(cand.frame_id)
                gain_feat = 0.0
                if red_rep and red_rep.pair_relations:
                    min_match_ratio = min(r.match_ratio for r in red_rep.pair_relations)
                    gain_feat = max(0.0, min(1.0, 1.0 - min_match_ratio))

                # 5. Dynamic Risk Penalty
                dyn_rep = dyn_map.get(cand.frame_id)
                dyn_penalty = 0.0
                if dyn_rep and dyn_rep.overall_scene_status == DynamicEvidenceCategory.DYNAMIC_EVIDENCE:
                    dyn_penalty = 0.20

                # Composite greedy heuristic gain
                greedy_gain = round(max(0.0, (0.35 * gain_time + 0.30 * gain_traj + 0.20 * gain_orient + 0.15 * gain_feat) - dyn_penalty), 4)

                # Determine Dominant Selection Reason
                is_forced_gap = dt >= cfg.max_temporal_gap_seconds
                is_traj_novel = traj_dist >= cfg.min_trajectory_baseline_meters
                is_feat_novel = gain_feat >= 0.40

                detailed_reasons: List[str] = []
                if is_forced_gap:
                    primary_reason = SelectionReason.TEMPORAL_GAP_COVERAGE
                    detailed_reasons.append(f"Forced selection: Temporal gap ({dt:.2f}s) >= max limit ({cfg.max_temporal_gap_seconds:.2f}s).")
                elif is_traj_novel:
                    primary_reason = SelectionReason.TRAJECTORY_DIVERSITY
                    detailed_reasons.append(f"Trajectory baseline ({traj_dist:.2f}m) >= threshold ({cfg.min_trajectory_baseline_meters:.2f}m).")
                elif gain_orient >= 0.5:
                    primary_reason = SelectionReason.ORIENTATION_DIVERSITY
                    detailed_reasons.append("Significant platform orientation change detected.")
                elif is_feat_novel:
                    primary_reason = SelectionReason.FEATURE_NOVELTY
                    detailed_reasons.append(f"Feature novelty gain ({gain_feat:.2f}) exceeds threshold.")
                else:
                    primary_reason = SelectionReason.QUALITY_TIE_BREAK
                    detailed_reasons.append("Sufficient spacing and satisfactory quality.")

                marginal_breakdown = {
                    "temporal_gap_gain": round(gain_time, 4),
                    "trajectory_diversity_gain": round(gain_traj, 4),
                    "orientation_diversity_gain": round(gain_orient, 4),
                    "feature_novelty_gain": round(gain_feat, 4),
                    "dynamic_risk_penalty": round(dyn_penalty, 4),
                }

                # Evaluate selection trigger
                is_novel = is_forced_gap or is_traj_novel or is_feat_novel or (gain_orient >= 0.5)

                if is_novel:
                    selected.append(
                        SelectedKeyframe(
                            frame_id=cand.frame_id,
                            frame_index=cand.frame_index,
                            timestamp_seconds=cand.timestamp_seconds,
                            primary_reason=primary_reason,
                            detailed_reasons=detailed_reasons,
                            marginal_gain_breakdown=marginal_breakdown,
                            greedy_heuristic_gain=greedy_gain,
                            expected_trajectory_displacement_meters=expected_disp,
                        )
                    )
                    last_selected_time = cand.timestamp_seconds
                    last_selected_enu = cand_enu
                    last_selected_quat = cand_quat
                else:
                    deprioritized.append(
                        DeprioritizedCandidate(
                            frame_id=cand.frame_id,
                            frame_index=cand.frame_index,
                            timestamp_seconds=cand.timestamp_seconds,
                            rejection_reason=RejectionReason.HIGH_REDUNDANCY,
                            explanation="Marginal geometric and temporal information gain below threshold.",
                        )
                    )

            # Step 3: Final Boundary Anchor (Safe Evaluation)
            if cfg.force_first_and_last_frames and len(surviving_candidates) > 1:
                last_safe = surviving_candidates[-1]
                is_true_final = (last_safe.frame_id == frames[-1].frame_id)

                if selected[-1].frame_id != last_safe.frame_id:
                    final_reason = SelectionReason.FINAL_ANCHOR if is_true_final else SelectionReason.FINAL_BOUNDARY_FALLBACK
                    final_detail = "Sequence temporal boundary final anchor." if is_true_final else f"Last frame was unsafe; anchored nearest safe candidate (frame {last_safe.frame_id})."

                    speed_final = speed_map.get(last_safe.frame_id)
                    disp_final = round(float(speed_final * (last_safe.timestamp_seconds - last_selected_time)), 3) if speed_final is not None else None

                    if len(selected) < max_budget:
                        selected.append(
                            SelectedKeyframe(
                                frame_id=last_safe.frame_id,
                                frame_index=last_safe.frame_index,
                                timestamp_seconds=last_safe.timestamp_seconds,
                                primary_reason=final_reason,
                                detailed_reasons=[final_detail],
                                marginal_gain_breakdown={"temporal": 1.0, "feature": 1.0, "trajectory": 1.0},
                                greedy_heuristic_gain=1.0,
                                expected_trajectory_displacement_meters=disp_final,
                            )
                        )
                    elif len(selected) > 1:
                        selected[-1] = SelectedKeyframe(
                            frame_id=last_safe.frame_id,
                            frame_index=last_safe.frame_index,
                            timestamp_seconds=last_safe.timestamp_seconds,
                            primary_reason=final_reason,
                            detailed_reasons=[f"{final_detail} (Replaced previous non-anchor candidate due to budget limit)."],
                            marginal_gain_breakdown={"temporal": 1.0, "feature": 1.0, "trajectory": 1.0},
                            greedy_heuristic_gain=1.0,
                            expected_trajectory_displacement_meters=disp_final,
                        )
        else:
            diagnostics.append("BOUNDARY_UNCOVERED: No safe candidates found across sequence boundaries.")

        selected_ids = [k.frame_id for k in selected]
        selected_indices = [k.frame_index for k in selected]
        selected_timestamps = [k.timestamp_seconds for k in selected]
        reduction_ratio = round(float(total_frames - len(selected)) / max(total_frames, 1), 4)

        provenance = {
            "total_input_frames": total_frames,
            "selected_keyframes_count": len(selected),
            "deprioritized_count": len(deprioritized),
            "selection_algorithm": "GreedyMarginalGain_v1.1",
            "min_spacing_seconds": cfg.min_temporal_spacing_seconds,
            "max_gap_seconds": cfg.max_temporal_gap_seconds,
            "heuristic_defaults_active": True,
        }

        return KeyframeSelectionResult(
            selected_keyframe_ids=selected_ids,
            selected_frame_indices=selected_indices,
            selected_timestamps_seconds=selected_timestamps,
            selected_keyframes=selected,
            deprioritized_candidates=deprioritized,
            total_input_frames=total_frames,
            selected_count=len(selected),
            reduction_ratio=reduction_ratio,
            fallback_used=fallback_used,
            diagnostics=diagnostics,
            provenance=provenance,
            config_version=cfg.config_version,
        )
