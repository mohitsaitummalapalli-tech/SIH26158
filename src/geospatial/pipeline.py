"""End-to-end Geospatial and Metric Reconstruction Pipeline.

Coordinates:
1. Origin selection (FIRST_VALID_POSITION, MEDIAN_POSITION, EXPLICIT_ORIGIN).
2. Telemetry synchronization & observation modeling with lever-arm kinematics.
3. Geometric scale observability and degeneracy gates.
4. Robust Sim(3) estimation (minimal RANSAC + Huber IRLS).
5. Analytical parameter uncertainty quantification.
6. Metric scale state machine promotion.
7. Independent Ground Control Point (GCP) validation.
8. Immutable provenance hash generation.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.geospatial.coordinates import (
    GeospatialAnchorOrigin,
    AltitudeReferenceType,
    wgs84_to_ecef,
    ecef_to_wgs84,
    wgs84_to_enu,
)
from src.geospatial.sim3 import Sim3, Sim3TransformContract, UncertaintyType
from src.geospatial.lever_arm import LeverArm, LeverArmStatus
from src.geospatial.telemetry_observation import (
    TelemetryObservation,
    ObservationClassification,
    GnssAccuracyInterpretation,
)
from src.geospatial.synchronization import (
    RawTelemetryRecord,
    TelemetrySynchronizer,
)
from src.geospatial.observability import (
    check_scale_observability,
    ScaleObservabilityReport,
    FullSim3ObservabilityStatus,
)
from src.geospatial.robust_estimation import (
    RobustSim3Estimator,
    RobustSim3Result,
)
from src.geospatial.metric_state import (
    MetricScaleStatus,
    MetricStateMachine,
)
from src.geospatial.validation import (
    GroundControlPoint,
    MetricValidator,
    ValidationReport,
)
from src.geospatial.uncertainty import (
    UncertaintyPropagator,
    Sim3UncertaintyReport,
)


@dataclass
class GeospatialMetricReconstructionResult:
    """Certified final outcome of Phase 3E.5 Geospatial & Metric Reconstruction."""

    metric_scale_status: MetricScaleStatus
    full_sim3_observability: FullSim3ObservabilityStatus
    is_metric_scale: bool
    depth_unit: str  # "METRES" or "RECONSTRUCTION_UNITS"
    anchor_origin: GeospatialAnchorOrigin
    sim3_transform: Optional[Sim3TransformContract]

    # Lever-Arm Provenance
    lever_arm_status: LeverArmStatus
    lever_arm_vector_m: Tuple[float, float, float]

    # GNSS Uncertainty Provenance
    gnss_accuracy_interpretation: GnssAccuracyInterpretation = GnssAccuracyInterpretation.ONE_SIGMA_STANDARD_DEVIATION
    gnss_uncertainty_source: str = "reported"

    # Residual & Inlier Diagnostics
    inlier_count: int = 0
    total_telemetry_count: int = 0
    inlier_ratio: float = 0.0
    horizontal_rmse_m: float = 0.0
    vertical_rmse_m: float = 0.0
    total_3d_rmse_m: float = 0.0
    max_residual_m: float = 0.0

    # Attitude Consistency
    mean_attitude_residual_deg: Optional[float] = None

    # Rejection Accounting
    rejected_telemetry_summary: Dict[str, int] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""

    # Observability & Axial Resolution
    axial_rotation_resolved: bool = True
    rotational_null_direction: Optional[Tuple[float, float, float]] = None

    # Optional independent validation report
    validation_report: Optional[ValidationReport] = None


class GeospatialMetricReconstructor:
    """Production coordinator executing the complete Phase 3E.5 pipeline."""

    def __init__(
        self,
        tau_disp_dimless: float = 1e-6,
        tau_collinear: float = 1e-4,
        tau_min_baseline_m: float = 10.0,
        tau_tri_degen: float = 1e-4,
        tau_rel_edge: float = 1e-4,
        tau_inlier_mahalanobis: float = 3.0,
        k_huber: float = 1.345,
        tau_rmse_uncertain_m: float = 5.0,
        tau_rel_scale_uncertainty: float = 0.15,
        max_gap_s: float = 1.0,
        clock_offset_s: float = 0.0,
        gnss_accuracy_interpretation: GnssAccuracyInterpretation = GnssAccuracyInterpretation.ONE_SIGMA_STANDARD_DEVIATION,
    ) -> None:
        self.tau_disp_dimless = tau_disp_dimless
        self.tau_collinear = tau_collinear
        self.tau_min_baseline_m = tau_min_baseline_m
        self.tau_tri_degen = tau_tri_degen
        self.tau_rel_edge = tau_rel_edge
        self.tau_inlier_mahalanobis = tau_inlier_mahalanobis
        self.k_huber = k_huber
        self.tau_rmse_uncertain_m = tau_rmse_uncertain_m
        self.tau_rel_scale_uncertainty = tau_rel_scale_uncertainty
        self.max_gap_s = max_gap_s
        self.clock_offset_s = clock_offset_s
        self.gnss_accuracy_interpretation = gnss_accuracy_interpretation

    def reconstruct(
        self,
        camera_centers_rec: Dict[str, np.ndarray],
        camera_timestamps_s: Dict[str, float],
        telemetry_records: List[RawTelemetryRecord],
        camera_rotations_rec: Optional[Dict[str, np.ndarray]] = None,
        lever_arm: Optional[LeverArm] = None,
        anchor_origin: Optional[GeospatialAnchorOrigin] = None,
        origin_policy: str = "FIRST_VALID_POSITION",
        checkpoints: Optional[List[GroundControlPoint]] = None,
    ) -> GeospatialMetricReconstructionResult:
        """Execute complete geospatial metric reconstruction workflow.
        
        Args:
            camera_centers_rec: Map of frame_id -> 3D camera optical center in reconstruction gauge.
            camera_timestamps_s: Map of frame_id -> shutter trigger time in seconds.
            telemetry_records: List of RawTelemetryRecord instances from flight log.
            camera_rotations_rec: Optional map of frame_id -> (3, 3) camera orientation in reconstruction frame.
            lever_arm: Optional airframe lever-arm definition (defaults to uncalibrated).
            anchor_origin: Optional explicit topocentric anchor (if None, derived via policy).
            origin_policy: Origin derivation policy ("FIRST_VALID_POSITION" or "MEDIAN_POSITION").
            checkpoints: Optional independent surveyed Ground Control Points for validation.
            
        Returns:
            GeospatialMetricReconstructionResult with certified metric state and diagnostics.
        """
        state_machine = MetricStateMachine(MetricScaleStatus.NOT_METRIC)
        lever_arm_inst = lever_arm or LeverArm.uncalibrated()
        total_frames = len(camera_centers_rec)

        # Determine GNSS uncertainty provenance tag
        if self.gnss_accuracy_interpretation == GnssAccuracyInterpretation.CEP_50:
            gnss_uncertainty_source = "converted_cep50"
        elif self.gnss_accuracy_interpretation == GnssAccuracyInterpretation.TWO_SIGMA_95:
            gnss_uncertainty_source = "converted_2sigma95"
        elif self.gnss_accuracy_interpretation == GnssAccuracyInterpretation.UNKNOWN_VENDOR_ACCURACY:
            gnss_uncertainty_source = "fallback"
        else:
            gnss_uncertainty_source = "reported"

        # Deterministic sorting of frame keys to guarantee bit-for-bit permutation invariance
        sorted_frame_ids = sorted(camera_centers_rec.keys())

        # 1. Establish Topocentric ENU Anchor Datum
        if anchor_origin is None:
            if not telemetry_records:
                state_machine.transition(
                    MetricScaleStatus.METRIC_ALIGNMENT_FAILED,
                    reason="No telemetry records provided to establish geospatial anchor",
                )
                dummy_anchor = GeospatialAnchorOrigin(
                    lat_deg=0.0,
                    lon_deg=0.0,
                    ellipsoidal_height_m=0.0,
                    altitude_reference=AltitudeReferenceType.UNKNOWN,
                    origin_policy=origin_policy,
                )
                return self._build_failure_result(
                    state_machine=state_machine,
                    anchor=dummy_anchor,
                    lever_arm=lever_arm_inst,
                    total_count=total_frames,
                    rejected_counts={"NO_TELEMETRY": total_frames},
                    failure_reason="NO_TELEMETRY",
                )

            if origin_policy == "MEDIAN_POSITION":
                lats = [r.latitude_deg for r in telemetry_records]
                lons = [r.longitude_deg for r in telemetry_records]
                alts = [r.altitude_m for r in telemetry_records]
                anchor = GeospatialAnchorOrigin(
                    lat_deg=float(np.median(lats)),
                    lon_deg=float(np.median(lons)),
                    ellipsoidal_height_m=float(np.median(alts)),
                    altitude_reference=AltitudeReferenceType.ELLIPSOIDAL_WGS84,
                    origin_policy="MEDIAN_POSITION",
                )
            else:
                first_rec = telemetry_records[0]
                anchor = GeospatialAnchorOrigin(
                    lat_deg=first_rec.latitude_deg,
                    lon_deg=first_rec.longitude_deg,
                    ellipsoidal_height_m=first_rec.altitude_m,
                    altitude_reference=AltitudeReferenceType.ELLIPSOIDAL_WGS84,
                    origin_policy="FIRST_VALID_POSITION",
                )
        else:
            anchor = anchor_origin

        # 2. Telemetry Synchronization
        synchronizer = TelemetrySynchronizer(
            telemetry_records=telemetry_records,
            anchor=anchor,
            clock_offset_s=self.clock_offset_s,
            max_gap_s=self.max_gap_s,
            lever_arm=lever_arm_inst,
            gnss_accuracy_interpretation=self.gnss_accuracy_interpretation,
        )

        observations: List[TelemetryObservation] = []
        valid_observations: List[TelemetryObservation] = []
        rejected_counts: Dict[str, int] = {}

        for fid in sorted_frame_ids:
            t_frame = camera_timestamps_s[fid]
            c_pt = camera_centers_rec[fid]
            r_rec = camera_rotations_rec.get(fid) if camera_rotations_rec else None
            obs = synchronizer.synchronize_frame(fid, t_frame, c_pt, r_cam_rec=r_rec)
            observations.append(obs)

            if obs.classification == ObservationClassification.VALID:
                valid_observations.append(obs)
            else:
                cat = obs.classification.value
                rejected_counts[cat] = rejected_counts.get(cat, 0) + 1

        # Check sufficient synchronized telemetry
        if len(valid_observations) < 4:
            state_machine.transition(
                MetricScaleStatus.METRIC_ALIGNMENT_FAILED,
                reason=f"Insufficient valid synchronized telemetry observations: {len(valid_observations)} < 4",
            )
            return self._build_failure_result(
                state_machine=state_machine,
                anchor=anchor,
                lever_arm=lever_arm_inst,
                total_count=total_frames,
                rejected_counts=rejected_counts,
                failure_reason="INSUFFICIENT_TELEMETRY",
            )

        # 3. Geometric Scale Observability Check
        c_recs = np.array([obs.c_rec for obs in valid_observations], dtype=np.float64)
        z_enus = np.array([obs.z_gnss_enu for obs in valid_observations], dtype=np.float64)

        obs_report = check_scale_observability(
            camera_centers_rec=c_recs,
            gnss_positions_enu=z_enus,
            tau_disp_dimless=self.tau_disp_dimless,
            tau_collinear=self.tau_collinear,
            tau_min_baseline_m=self.tau_min_baseline_m,
            min_inlier_count=4,
        )

        if not obs_report.scale_observable:
            state_machine.transition(
                MetricScaleStatus.METRIC_ALIGNMENT_FAILED,
                reason="; ".join(obs_report.scale_failure_reasons),
            )
            return self._build_failure_result(
                state_machine=state_machine,
                anchor=anchor,
                lever_arm=lever_arm_inst,
                total_count=total_frames,
                rejected_counts=rejected_counts,
                failure_reason="; ".join(obs_report.scale_failure_reasons),
                full_sim3_observability=obs_report.full_sim3_observability,
                diagnostics={"observability_report": obs_report.__dict__},
            )

        # 4. Robust Sim(3) Estimation (RANSAC + Huber IRLS + Collinear Fallback)
        estimator = RobustSim3Estimator(
            tau_tri_degen=self.tau_tri_degen,
            tau_rel_edge=self.tau_rel_edge,
            tau_inlier_mahalanobis=self.tau_inlier_mahalanobis,
            k_huber=self.k_huber,
            tau_min_baseline_m=self.tau_min_baseline_m,
        )
        est_result = estimator.estimate(valid_observations, lever_arm_inst)

        if not est_result.success or est_result.sim3 is None:
            state_machine.transition(
                MetricScaleStatus.METRIC_ALIGNMENT_FAILED,
                reason=est_result.failure_reason or "Robust estimation failed",
            )
            return self._build_failure_result(
                state_machine=state_machine,
                anchor=anchor,
                lever_arm=lever_arm_inst,
                total_count=total_frames,
                rejected_counts=rejected_counts,
                failure_reason=est_result.failure_reason,
                full_sim3_observability=obs_report.full_sim3_observability,
                diagnostics={"estimation_diagnostics": est_result.diagnostics.__dict__},
            )

        sim3_est = est_result.sim3

        # 5. Collinear Axial Rotation Resolution Check via Attitude
        axial_rotation_resolved = True
        rotational_null_direction = sim3_est.rotational_null_direction

        if obs_report.is_collinear:
            inlier_obs = [valid_observations[idx] for idx in est_result.inlier_indices]
            att_rotations = []
            for obs in inlier_obs:
                if obs.r_cam_rec is not None and obs.r_body_to_enu is not None:
                    r_cam_enu = obs.r_body_to_enu @ lever_arm_inst.mounting_rotation_camera_to_body
                    r_att_i = r_cam_enu @ obs.r_cam_rec.T
                    att_rotations.append(r_att_i)

            if len(att_rotations) >= 1:
                # SVD chordal L2 average
                m_sum = sum(att_rotations)
                u_m, _, vt_m = np.linalg.svd(m_sum)
                r_att_mean = u_m @ vt_m
                if np.linalg.det(r_att_mean) < 0:
                    r_att_mean = u_m @ np.diag([1.0, 1.0, -1.0]) @ vt_m

                # Attitude dispersion across inliers
                res_deg_list = []
                for r_att_i in att_rotations:
                    tr = float(np.trace(r_att_i @ r_att_mean.T))
                    ang = math.acos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
                    res_deg_list.append(math.degrees(ang))
                mean_att_disp_deg = float(np.mean(res_deg_list))

                # Along-track alignment consistency with position line
                c_diff = c_recs[-1] - c_recs[0]
                z_diff = z_enus[-1] - z_enus[0]
                norm_c = float(np.linalg.norm(c_diff))
                norm_z = float(np.linalg.norm(z_diff))
                if norm_c > 0 and norm_z > 0:
                    u_rec_dir = c_diff / norm_c
                    u_geo_dir = z_diff / norm_z
                    pred_geo = r_att_mean @ u_rec_dir
                    dot_val = float(np.dot(pred_geo, u_geo_dir))
                    along_err_deg = math.degrees(math.acos(np.clip(dot_val, -1.0, 1.0)))
                else:
                    along_err_deg = 0.0

                if mean_att_disp_deg <= 10.0 and along_err_deg <= 10.0:
                    # Valid attitude resolved the axial rotation!
                    sim3_est.rotation = r_att_mean
                    c_mean = np.mean(c_recs[est_result.inlier_indices], axis=0)
                    target_cams_inlier = np.array([
                        lever_arm_inst.unapply_lever_arm(obs.z_gnss_enu, obs.r_body_to_enu)
                        for obs in inlier_obs
                    ], dtype=np.float64)
                    z_mean = np.mean(target_cams_inlier, axis=0)
                    sim3_est.translation = z_mean - sim3_est.scale * (r_att_mean @ c_mean)
                    axial_rotation_resolved = True
                    rotational_null_direction = None
                    obs_report.full_sim3_observability = FullSim3ObservabilityStatus.FULL_SIM3_OBSERVABLE
                    sim3_est.axial_rotation_resolved = True
                    sim3_est.rotational_null_direction = None
                else:
                    # Attitude is conflicting/wrong
                    axial_rotation_resolved = False
                    sim3_est.axial_rotation_resolved = False
                    obs_report.full_sim3_observability = FullSim3ObservabilityStatus.FULL_SIM3_NOT_OBSERVABLE_COLLINEAR
            else:
                # No attitude provided: preserve unresolved rotational DOF
                axial_rotation_resolved = False
                sim3_est.axial_rotation_resolved = False
                obs_report.full_sim3_observability = FullSim3ObservabilityStatus.FULL_SIM3_NOT_OBSERVABLE_COLLINEAR
        else:
            axial_rotation_resolved = True
            rotational_null_direction = None
            sim3_est.axial_rotation_resolved = True
            sim3_est.rotational_null_direction = None

        # 6. Uncertainty Quantification via Huber-Weighted Fisher Matrix
        unc_report = UncertaintyPropagator.estimate_parameter_covariance(
            sim3=sim3_est,
            observations=valid_observations,
            inlier_indices=est_result.inlier_indices,
            lever_arm=lever_arm_inst,
            huber_weights=est_result.diagnostics.final_huber_weights,
        )

        if obs_report.is_collinear and not axial_rotation_resolved:
            # Under collinear flight without attitude, rotation around trajectory axis is underconstrained
            unc_report.uncertainty_type = UncertaintyType.HEURISTIC_UNCERTAINTY
            if unc_report.fallback_reason is None:
                unc_report.fallback_reason = "Collinear trajectory has unconstrained 1D rotational null space"

        sim3_est.scale_uncertainty_1sigma = unc_report.scale_uncertainty_1sigma
        sim3_est.uncertainty_type = unc_report.uncertainty_type
        sim3_est.fisher_condition_number = unc_report.fisher_condition_number

        # 7. Evaluate State Machine
        state_machine.evaluate_estimation(
            estimation_success=True,
            is_observable=obs_report.scale_observable,
            inlier_count=est_result.diagnostics.final_inlier_count,
            rmse_3d_m=est_result.diagnostics.total_3d_rmse_m,
            relative_scale_uncertainty=unc_report.relative_scale_uncertainty,
            tau_rmse_uncertain_m=self.tau_rmse_uncertain_m,
            tau_rel_uncertainty=self.tau_rel_scale_uncertainty,
        )

        # 8. Independent Ground Control Point Validation
        val_report: Optional[ValidationReport] = None
        if checkpoints:
            validator = MetricValidator()
            val_report = validator.validate(sim3_est, checkpoints)
            state_machine.evaluate_validation(
                is_validated=val_report.is_validated,
                validation_rmse_m=val_report.total_3d_rmse_m,
                tolerance_m=val_report.tolerance_m,
            )

        # Update rejected counts from RANSAC
        for r_idx in est_result.rejected_indices:
            cat = ObservationClassification.OUTLIER_POSITION.value
            rejected_counts[cat] = rejected_counts.get(cat, 0) + 1

        # 9. Provenance Hash
        prov_hash = UncertaintyPropagator.compute_provenance_hash(
            anchor_lat=anchor.lat_deg,
            anchor_lon=anchor.lon_deg,
            anchor_alt=anchor.ellipsoidal_height_m,
            sim3_scale=sim3_est.scale,
            sim3_translation=(float(sim3_est.translation[0]), float(sim3_est.translation[1]), float(sim3_est.translation[2])),
            inlier_count=est_result.diagnostics.final_inlier_count,
            rmse_m=est_result.diagnostics.total_3d_rmse_m,
        )

        is_metric = state_machine.is_metric
        depth_unit = "METRES" if is_metric else "RECONSTRUCTION_UNITS"

        sim3_contract = sim3_est.to_contract(
            inlier_count=est_result.diagnostics.final_inlier_count,
            residual_rmse_m=est_result.diagnostics.total_3d_rmse_m,
        )

        return GeospatialMetricReconstructionResult(
            metric_scale_status=state_machine.current_state,
            full_sim3_observability=obs_report.full_sim3_observability,
            is_metric_scale=is_metric,
            depth_unit=depth_unit,
            anchor_origin=anchor,
            sim3_transform=sim3_contract,
            axial_rotation_resolved=axial_rotation_resolved,
            rotational_null_direction=rotational_null_direction,
            lever_arm_status=lever_arm_inst.status,
            lever_arm_vector_m=(
                float(lever_arm_inst.vector_body[0]),
                float(lever_arm_inst.vector_body[1]),
                float(lever_arm_inst.vector_body[2]),
            ),
            gnss_accuracy_interpretation=self.gnss_accuracy_interpretation,
            gnss_uncertainty_source=gnss_uncertainty_source,
            inlier_count=est_result.diagnostics.final_inlier_count,
            total_telemetry_count=total_frames,
            inlier_ratio=est_result.diagnostics.inlier_ratio,
            horizontal_rmse_m=est_result.diagnostics.horizontal_rmse_m,
            vertical_rmse_m=est_result.diagnostics.vertical_rmse_m,
            total_3d_rmse_m=est_result.diagnostics.total_3d_rmse_m,
            max_residual_m=est_result.diagnostics.max_residual_m,
            mean_attitude_residual_deg=est_result.diagnostics.mean_attitude_residual_deg,
            rejected_telemetry_summary=rejected_counts,
            diagnostics={
                "observability": obs_report.__dict__,
                "estimation": est_result.diagnostics.__dict__,
                "uncertainty": {
                    "uncertainty_type": unc_report.uncertainty_type.value,
                    "scale_uncertainty_1sigma": unc_report.scale_uncertainty_1sigma,
                    "relative_scale_uncertainty": unc_report.relative_scale_uncertainty,
                    "rotation_uncertainty_rad": unc_report.rotation_uncertainty_rad,
                    "translation_uncertainty_m": unc_report.translation_uncertainty_m,
                    "sigma_log_scale": unc_report.sigma_log_scale,
                    "rotational_null_direction": unc_report.rotational_null_direction,
                    "unconstrained_parameter_directions": unc_report.unconstrained_parameter_directions,
                    "axial_rotation_resolved": unc_report.axial_rotation_resolved,
                    "fisher_condition_number": unc_report.fisher_condition_number,
                    "regularization_used": unc_report.regularization_used,
                    "regularization_value": unc_report.regularization_value,
                    "parameter_scales": unc_report.parameter_scales,
                    "fallback_reason": unc_report.fallback_reason,
                },
                "state_machine_history": [
                    {"from": t.previous_state.value, "to": t.current_state.value, "reason": t.reason}
                    for t in state_machine.history
                ],
            },
            provenance_hash=prov_hash,
            validation_report=val_report,
        )

    def _build_failure_result(
        self,
        state_machine: MetricStateMachine,
        anchor: GeospatialAnchorOrigin,
        lever_arm: LeverArm,
        total_count: int,
        rejected_counts: Dict[str, int],
        failure_reason: Optional[str] = None,
        full_sim3_observability: FullSim3ObservabilityStatus = FullSim3ObservabilityStatus.FULL_SIM3_NOT_OBSERVABLE_STATIONARY,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> GeospatialMetricReconstructionResult:
        """Construct deterministic failure result preserving gauge uncalibrated state."""
        prov_hash = UncertaintyPropagator.compute_provenance_hash(
            anchor_lat=anchor.lat_deg,
            anchor_lon=anchor.lon_deg,
            anchor_alt=anchor.ellipsoidal_height_m,
            sim3_scale=1.0,
            sim3_translation=(0.0, 0.0, 0.0),
            inlier_count=0,
            rmse_m=0.0,
        )
        diag = diagnostics or {}
        diag["failure_reason"] = failure_reason or "Alignment failed"
        diag["state_machine_history"] = [
            {"from": t.previous_state.value, "to": t.current_state.value, "reason": t.reason}
            for t in state_machine.history
        ]

        if self.gnss_accuracy_interpretation == GnssAccuracyInterpretation.CEP_50:
            gnss_uncertainty_source = "converted_cep50"
        elif self.gnss_accuracy_interpretation == GnssAccuracyInterpretation.TWO_SIGMA_95:
            gnss_uncertainty_source = "converted_2sigma95"
        elif self.gnss_accuracy_interpretation == GnssAccuracyInterpretation.UNKNOWN_VENDOR_ACCURACY:
            gnss_uncertainty_source = "fallback"
        else:
            gnss_uncertainty_source = "reported"

        return GeospatialMetricReconstructionResult(
            metric_scale_status=MetricScaleStatus.METRIC_ALIGNMENT_FAILED,
            full_sim3_observability=full_sim3_observability,
            is_metric_scale=False,
            depth_unit="RECONSTRUCTION_UNITS",
            anchor_origin=anchor,
            sim3_transform=None,
            axial_rotation_resolved=False,
            rotational_null_direction=None,
            lever_arm_status=lever_arm.status,
            lever_arm_vector_m=(
                float(lever_arm.vector_body[0]),
                float(lever_arm.vector_body[1]),
                float(lever_arm.vector_body[2]),
            ),
            gnss_accuracy_interpretation=self.gnss_accuracy_interpretation,
            gnss_uncertainty_source=gnss_uncertainty_source,
            inlier_count=0,
            total_telemetry_count=total_count,
            inlier_ratio=0.0,
            horizontal_rmse_m=0.0,
            vertical_rmse_m=0.0,
            total_3d_rmse_m=0.0,
            max_residual_m=0.0,
            rejected_telemetry_summary=rejected_counts,
            diagnostics=diag,
            provenance_hash=prov_hash,
        )
