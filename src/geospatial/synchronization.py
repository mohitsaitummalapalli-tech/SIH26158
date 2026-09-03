"""Temporal synchronization engine: aligns video frame triggers with GNSS telemetry.

Uses the Phase 1 canonical timeline standards:
- Clock offset handling: t_corr = t_frame + delta_t_clock
- Bracketing telemetry interval search
- Gap rejection: delta_t_interval > tau_gap (default 1.0s)
- Continuous trajectory interpolation (Linear / Hermite for position, SLERP for orientation)
- Dynamic speed-proportional uncertainty inflation
"""

from dataclasses import dataclass
import math
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from src.geospatial.coordinates import (
    GeospatialAnchorOrigin,
    wgs84_to_ecef,
    ecef_to_enu,
    wgs84_to_enu,
)
from src.geospatial.lever_arm import LeverArm
from src.geospatial.telemetry_observation import (
    TelemetryObservation,
    ObservationClassification,
    GnssAccuracyInterpretation,
    construct_gnss_covariance,
)


@dataclass
class RawTelemetryRecord:
    """Standardized raw telemetry epoch."""
    timestamp_seconds: float
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    horizontal_accuracy_m: Optional[float] = None
    vertical_accuracy_m: Optional[float] = None
    is_rtk_fixed: bool = False


def _euler_to_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Convert Z-Y-X Euler angles (yaw, pitch, roll in degrees) to active (3, 3) rotation matrix."""
    psi = math.radians(yaw_deg)
    theta = math.radians(pitch_deg)
    phi = math.radians(roll_deg)

    r_z = np.array([
        [math.cos(psi), -math.sin(psi), 0.0],
        [math.sin(psi), math.cos(psi), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    r_y = np.array([
        [math.cos(theta), 0.0, math.sin(theta)],
        [0.0, 1.0, 0.0],
        [-math.sin(theta), 0.0, math.cos(theta)],
    ], dtype=np.float64)

    r_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(phi), -math.sin(phi)],
        [0.0, math.sin(phi), math.cos(phi)],
    ], dtype=np.float64)

    return r_z @ r_y @ r_x


def _rotation_matrix_to_quaternion(r: np.ndarray) -> np.ndarray:
    """Convert (3, 3) rotation matrix to unit quaternion [w, x, y, z]."""
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif (r[0, 0] > r[1, 1]) and (r[0, 0] > r[2, 2]):
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(q)
    return q / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical Linear Interpolation (SLERP) between unit quaternions q0 and q1."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        # Linear interpolation if nearly identical
        res = q0 + t * (q1 - q0)
        return res / np.linalg.norm(res)

    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = math.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = math.sin(theta_t)

    s0 = math.cos(theta_t) - dot * sin_theta_t / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = (s0 * q0) + (s1 * q1)
    return res / np.linalg.norm(res)


def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [w, x, y, z] to (3, 3) active rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


class TelemetrySynchronizer:
    """Interpolates continuous drone trajectory to align with reconstructed optical frame epochs."""

    def __init__(
        self,
        telemetry_records: List[RawTelemetryRecord],
        anchor: GeospatialAnchorOrigin,
        clock_offset_s: float = 0.0,
        max_gap_s: float = 1.0,
        shutter_timing_uncertainty_s: float = 0.01,
        lever_arm: Optional[LeverArm] = None,
        gnss_accuracy_interpretation: GnssAccuracyInterpretation = GnssAccuracyInterpretation.ONE_SIGMA_STANDARD_DEVIATION,
    ) -> None:
        self.anchor = anchor
        self.clock_offset_s = clock_offset_s
        self.max_gap_s = max_gap_s
        self.shutter_timing_uncertainty_s = shutter_timing_uncertainty_s
        self.lever_arm = lever_arm or LeverArm.uncalibrated()
        self.gnss_accuracy_interpretation = gnss_accuracy_interpretation

        # Sort and deduplicate records by timestamp
        records_sorted = sorted(telemetry_records, key=lambda r: r.timestamp_seconds)
        self.records: List[RawTelemetryRecord] = []
        for rec in records_sorted:
            if not self.records or (rec.timestamp_seconds - self.records[-1].timestamp_seconds) > 1e-9:
                self.records.append(rec)

        # Precompute ENU positions
        self.enu_positions: List[np.ndarray] = []
        for r in self.records:
            enu = wgs84_to_enu(r.latitude_deg, r.longitude_deg, r.altitude_m, self.anchor)
            self.enu_positions.append(np.array(enu, dtype=np.float64))

    def synchronize_frame(
        self,
        frame_id: str,
        frame_timestamp_s: float,
        c_rec: np.ndarray,
        r_cam_rec: Optional[np.ndarray] = None,
    ) -> TelemetryObservation:
        """Interpolate telemetry stream to shutter trigger epoch.
        
        Args:
            frame_id: Unique string frame identifier.
            frame_timestamp_s: Shutter trigger time in seconds.
            c_rec: Optical center in reconstruction gauge (3,).
            r_cam_rec: Optional camera rotation in reconstruction gauge (3, 3).
            
        Returns:
            TelemetryObservation with classification status.
        """
        c_rec_arr = np.asarray(c_rec, dtype=np.float64).reshape((3,))
        t_corr = frame_timestamp_s + self.clock_offset_s

        if not self.records or len(self.records) < 2:
            return TelemetryObservation(
                frame_id=frame_id,
                timestamp_seconds=t_corr,
                c_rec=c_rec_arr,
                z_gnss_enu=np.zeros(3, dtype=np.float64),
                r_cam_rec=r_cam_rec,
                classification=ObservationClassification.INVALID_GNSS,
                rejection_reason="Insufficient telemetry stream records (< 2)",
            )

        t_min = self.records[0].timestamp_seconds
        t_max = self.records[-1].timestamp_seconds

        # Out-of-range check
        if t_corr < t_min or t_corr > t_max:
            # Rejection: out of bounds
            return TelemetryObservation(
                frame_id=frame_id,
                timestamp_seconds=t_corr,
                c_rec=c_rec_arr,
                z_gnss_enu=np.zeros(3, dtype=np.float64),
                r_cam_rec=r_cam_rec,
                classification=ObservationClassification.TEMPORAL_MISMATCH,
                rejection_reason=f"Timestamp {t_corr:.3f}s out of telemetry range [{t_min:.3f}s, {t_max:.3f}s]",
            )

        # Exact match check
        for idx, rec in enumerate(self.records):
            if abs(rec.timestamp_seconds - t_corr) < 1e-6:
                z_enu = self.enu_positions[idx]
                r_body = _euler_to_rotation_matrix(rec.yaw_deg, rec.pitch_deg, rec.roll_deg)
                cov, _ = construct_gnss_covariance(
                    horizontal_accuracy_m=rec.horizontal_accuracy_m,
                    vertical_accuracy_m=rec.vertical_accuracy_m,
                    shutter_timing_uncertainty_s=self.shutter_timing_uncertainty_s,
                    lever_arm=self.lever_arm,
                    r_body_to_enu=r_body,
                    is_rtk_fixed=rec.is_rtk_fixed,
                )
                return TelemetryObservation(
                    frame_id=frame_id,
                    timestamp_seconds=t_corr,
                    c_rec=c_rec_arr,
                    z_gnss_enu=z_enu,
                    r_body_to_enu=r_body,
                    covariance_enu=cov,
                    horizontal_accuracy_m=rec.horizontal_accuracy_m,
                    vertical_accuracy_m=rec.vertical_accuracy_m,
                    is_rtk_fixed=rec.is_rtk_fixed,
                    r_cam_rec=r_cam_rec,
                    classification=ObservationClassification.VALID,
                )

        # Binary search for bracketing interval [idx_k, idx_k+1]
        timestamps = [r.timestamp_seconds for r in self.records]
        k = int(np.searchsorted(timestamps, t_corr)) - 1
        k = max(0, min(k, len(self.records) - 2))

        rec_0 = self.records[k]
        rec_1 = self.records[k + 1]
        dt_interval = rec_1.timestamp_seconds - rec_0.timestamp_seconds

        # Gap check
        if dt_interval > self.max_gap_s:
            return TelemetryObservation(
                frame_id=frame_id,
                timestamp_seconds=t_corr,
                c_rec=c_rec_arr,
                z_gnss_enu=np.zeros(3, dtype=np.float64),
                classification=ObservationClassification.TEMPORAL_MISMATCH,
                rejection_reason=f"Telemetry gap {dt_interval:.3f}s exceeds threshold {self.max_gap_s:.3f}s",
            )

        # Interpolation fraction alpha in [0.0, 1.0]
        alpha = (t_corr - rec_0.timestamp_seconds) / dt_interval if dt_interval > 0 else 0.0

        # Position interpolation in ENU
        pos_0 = self.enu_positions[k]
        pos_1 = self.enu_positions[k + 1]
        z_interp = pos_0 + alpha * (pos_1 - pos_0)

        # Velocity estimation
        v_est = (pos_1 - pos_0) / dt_interval if dt_interval > 0 else np.zeros(3, dtype=np.float64)

        # Attitude interpolation via SLERP
        r0 = _euler_to_rotation_matrix(rec_0.yaw_deg, rec_0.pitch_deg, rec_0.roll_deg)
        r1 = _euler_to_rotation_matrix(rec_1.yaw_deg, rec_1.pitch_deg, rec_1.roll_deg)
        q0 = _rotation_matrix_to_quaternion(r0)
        q1 = _rotation_matrix_to_quaternion(r1)
        q_interp = _quaternion_slerp(q0, q1, alpha)
        r_body_interp = _quaternion_to_rotation_matrix(q_interp)

        # Accuracy interpolation (conservative max)
        h_acc = max(rec_0.horizontal_accuracy_m or 0.0, rec_1.horizontal_accuracy_m or 0.0) or None
        v_acc = max(rec_0.vertical_accuracy_m or 0.0, rec_1.vertical_accuracy_m or 0.0) or None
        rtk = rec_0.is_rtk_fixed and rec_1.is_rtk_fixed

        cov, _ = construct_gnss_covariance(
            horizontal_accuracy_m=h_acc,
            vertical_accuracy_m=v_acc,
            velocity_enu=v_est,
            shutter_timing_uncertainty_s=self.shutter_timing_uncertainty_s,
            lever_arm=self.lever_arm,
            r_body_to_enu=r_body_interp,
            is_rtk_fixed=rtk,
            interpretation=self.gnss_accuracy_interpretation,
        )

        return TelemetryObservation(
            frame_id=frame_id,
            timestamp_seconds=t_corr,
            c_rec=c_rec_arr,
            z_gnss_enu=z_interp,
            r_body_to_enu=r_body_interp,
            covariance_enu=cov,
            horizontal_accuracy_m=h_acc,
            vertical_accuracy_m=v_acc,
            velocity_enu=v_est,
            is_rtk_fixed=rtk,
            r_cam_rec=r_cam_rec,
            classification=ObservationClassification.VALID,
        )
