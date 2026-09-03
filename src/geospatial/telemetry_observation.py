"""GNSS telemetry observation model, residual computation, and quality classification.

Forward Observation Equation:
    z_gnss,i = s * R * C_rec,i + t + R_body,i * L_body + epsilon_i
    epsilon_i ~ N(0, Sigma_i)

Residual:
    r_i = z_gnss,i - (s * R * C_rec,i + t + R_body,i * L_body)

Mahalanobis Distance:
    d_i = sqrt(r_i^T * Sigma_i^-1 * r_i)
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from src.geospatial.sim3 import Sim3
from src.geospatial.lever_arm import LeverArm, LeverArmStatus


class ObservationClassification(str, Enum):
    """Scientific classification of telemetry observations for georeferencing."""
    VALID = "VALID"                                         # High-quality observation accepted in inlier consensus
    INVALID_GNSS = "INVALID_GNSS"                           # Missing lat/lon, NaN coordinates, or fix not valid
    LOW_QUALITY_GNSS = "LOW_QUALITY_GNSS"                   # DOP too high or reported accuracy worse than gate
    TEMPORAL_MISMATCH = "TEMPORAL_MISMATCH"                 # Shutter time falls in excessive telemetry gap (> 1.0s)
    OUTLIER_POSITION = "OUTLIER_POSITION"                   # Isolated spatial outlier rejected by RANSAC consensus
    HIGH_RESIDUAL = "HIGH_RESIDUAL"                         # Residual exceeds inlier boundary after IRLS convergence
    INSUFFICIENT_GEOMETRIC_SUPPORT = "INSUFFICIENT_GEOMETRIC_SUPPORT" # Rejected due to collinear or hover degeneracy


class GnssAccuracyInterpretation(str, Enum):
    """Explicit interpretation semantics for reported GNSS accuracy metadata."""
    ONE_SIGMA_STANDARD_DEVIATION = "ONE_SIGMA_STANDARD_DEVIATION"
    CEP_50 = "CEP_50"
    TWO_SIGMA_95 = "TWO_SIGMA_95"
    RMS_ERROR = "RMS_ERROR"
    UNKNOWN_VENDOR_ACCURACY = "UNKNOWN_VENDOR_ACCURACY"


@dataclass
class TelemetryObservation:
    """Rigorous observation pairing reconstructed camera optical center with GNSS telemetry."""

    frame_id: str
    timestamp_seconds: float
    c_rec: np.ndarray                                 # Optical center in reconstruction gauge (3,)
    z_gnss_enu: np.ndarray                            # Observed antenna position in local ENU (3,)
    r_body_to_enu: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64)) # Body-to-ENU active rotation (3, 3)
    covariance_enu: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64)) # Total observation covariance (3, 3)
    horizontal_accuracy_m: Optional[float] = None
    vertical_accuracy_m: Optional[float] = None
    velocity_enu: Optional[np.ndarray] = None         # Estimated vehicle velocity in ENU [m/s]
    r_cam_rec: Optional[np.ndarray] = None            # Camera orientation in reconstruction frame (3, 3)
    is_rtk_fixed: bool = False
    classification: ObservationClassification = ObservationClassification.VALID
    rejection_reason: Optional[str] = None

    def __post_init__(self) -> None:
        self.c_rec = np.asarray(self.c_rec, dtype=np.float64).reshape((3,))
        self.z_gnss_enu = np.asarray(self.z_gnss_enu, dtype=np.float64).reshape((3,))
        self.r_body_to_enu = np.asarray(self.r_body_to_enu, dtype=np.float64).reshape((3, 3))
        self.covariance_enu = np.asarray(self.covariance_enu, dtype=np.float64).reshape((3, 3))
        if self.r_cam_rec is not None:
            self.r_cam_rec = np.asarray(self.r_cam_rec, dtype=np.float64).reshape((3, 3))

    def compute_modeled_antenna_enu(self, sim3: Sim3, lever_arm: LeverArm) -> np.ndarray:
        """Compute modeled GNSS antenna position in local ENU:
        
        z_hat = s * R * C_rec + t + R_body * L_body
        """
        c_cam_enu = sim3.transform_camera_center(self.c_rec)
        return lever_arm.predict_antenna_enu(c_cam_enu, self.r_body_to_enu)

    def compute_residual(self, sim3: Sim3, lever_arm: LeverArm) -> np.ndarray:
        """Compute 3D residual vector in local ENU:
        
        r_i = z_gnss,i - (s * R * C_rec,i + t + R_body,i * L_body)
        """
        z_hat = self.compute_modeled_antenna_enu(sim3, lever_arm)
        return self.z_gnss_enu - z_hat

    def compute_mahalanobis(self, sim3: Sim3, lever_arm: LeverArm) -> float:
        """Compute dimensionless Mahalanobis distance:
        
        d_i = sqrt(r_i^T * Sigma_i^-1 * r_i)
        """
        r = self.compute_residual(sim3, lever_arm)
        # Numerical guard: add small epsilon to diagonal if near singular
        cov = self.covariance_enu.copy()
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov + np.eye(3) * 1e-6)

        d_sq = float(r.T @ inv_cov @ r)
        return math.sqrt(max(0.0, d_sq))


def construct_gnss_covariance(
    horizontal_accuracy_m: Optional[float] = None,
    vertical_accuracy_m: Optional[float] = None,
    velocity_enu: Optional[np.ndarray] = None,
    shutter_timing_uncertainty_s: float = 0.01,
    lever_arm: Optional[LeverArm] = None,
    r_body_to_enu: Optional[np.ndarray] = None,
    is_rtk_fixed: bool = False,
    fallback_horizontal_m: float = 3.0,
    fallback_vertical_m: float = 5.0,
    interpretation: GnssAccuracyInterpretation = GnssAccuracyInterpretation.ONE_SIGMA_STANDARD_DEVIATION,
) -> Tuple[np.ndarray, bool]:
    """Construct total 3x3 observation covariance matrix in local ENU.
    
    Combines:
    1. Direct reported GNSS horizontal and vertical accuracies according to interpretation semantics.
    2. Timing uncertainty inflation: (sigma_time * ||v||)^2 * I_3.
    3. Lever-arm uncertainty in ENU.
    4. Conservative fallback floors for missing metadata (CONFIGURATION HEURISTICS).
    
    Returns:
        (covariance_3x3, metadata_missing_flag)
    """
    metadata_missing = False

    # 1. Base GNSS accuracy
    if is_rtk_fixed:
        sigma_h = horizontal_accuracy_m if (horizontal_accuracy_m is not None and horizontal_accuracy_m > 0) else 0.03
        sigma_v = vertical_accuracy_m if (vertical_accuracy_m is not None and vertical_accuracy_m > 0) else 0.05
    elif interpretation == GnssAccuracyInterpretation.UNKNOWN_VENDOR_ACCURACY:
        sigma_h = fallback_horizontal_m
        sigma_v = fallback_vertical_m
        metadata_missing = True
    else:
        # Convert horizontal accuracy according to specified semantics
        if horizontal_accuracy_m is not None and horizontal_accuracy_m > 0 and not math.isnan(horizontal_accuracy_m):
            if interpretation == GnssAccuracyInterpretation.CEP_50:
                # Under circular normal distribution: CEP50 = sqrt(2 ln 2) * sigma approx 1.177410 * sigma
                sigma_h = horizontal_accuracy_m / 1.1774100225154747
            elif interpretation == GnssAccuracyInterpretation.TWO_SIGMA_95:
                # 2D 95% circular error radius = sqrt(-2 ln 0.05) * sigma approx 2.447747 * sigma
                sigma_h = horizontal_accuracy_m / 2.447746893674682
            else:
                sigma_h = horizontal_accuracy_m
        else:
            sigma_h = fallback_horizontal_m
            metadata_missing = True

        # Convert vertical accuracy according to specified semantics
        if vertical_accuracy_m is not None and vertical_accuracy_m > 0 and not math.isnan(vertical_accuracy_m):
            if interpretation == GnssAccuracyInterpretation.CEP_50:
                sigma_v = vertical_accuracy_m / 1.1774100225154747
            elif interpretation == GnssAccuracyInterpretation.TWO_SIGMA_95:
                # 1D 95% confidence interval is 1.959964 * sigma
                sigma_v = vertical_accuracy_m / 1.959963984540054
            else:
                sigma_v = vertical_accuracy_m
        else:
            sigma_v = fallback_vertical_m
            metadata_missing = True

    cov_gnss = np.diag([sigma_h ** 2, sigma_h ** 2, sigma_v ** 2])

    # 2. Timing / velocity uncertainty inflation
    cov_timing = np.zeros((3, 3), dtype=np.float64)
    if velocity_enu is not None:
        v = np.asarray(velocity_enu, dtype=np.float64).reshape((3,))
        speed = float(np.linalg.norm(v))
        if speed > 0 and shutter_timing_uncertainty_s > 0:
            var_timing = (shutter_timing_uncertainty_s * speed) ** 2
            cov_timing = np.eye(3, dtype=np.float64) * var_timing

    # 3. Lever-arm uncertainty
    cov_lever = np.zeros((3, 3), dtype=np.float64)
    if lever_arm is not None:
        r_body = r_body_to_enu if r_body_to_enu is not None else np.eye(3, dtype=np.float64)
        cov_lever = lever_arm.effective_covariance_enu(r_body)

    total_cov = cov_gnss + cov_timing + cov_lever
    return total_cov, metadata_missing
