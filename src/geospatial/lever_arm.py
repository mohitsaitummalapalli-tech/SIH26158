"""Physical lever-arm kinematics between drone camera optical center and GNSS antenna.

Vector convention:
    L_body = P_antenna_body - P_camera_body
Displacement from camera optical center to GNSS antenna phase center, expressed in the
drone platform body coordinate frame (FLU: +X Forward, +Y Left, +Z Up).

Kinematic forward relationship in local ENU:
    C_antenna_geo = C_camera_geo + R_body * L_body

Forward GNSS observation model:
    z_gnss = s * R * C_rec + t + R_body * L_body + epsilon
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Optional, Tuple, Union
import numpy as np


class LeverArmStatus(str, Enum):
    """Provenance and calibration status of the airframe lever arm."""
    LEVER_ARM_CALIBRATED = "LEVER_ARM_CALIBRATED"       # Measured via factory/survey calibration
    LEVER_ARM_UNCALIBRATED = "LEVER_ARM_UNCALIBRATED"   # Unmeasured; nominal fallback uncertainty applied
    LEVER_ARM_ZERO = "LEVER_ARM_ZERO"                   # Coincident by design/assumption (e.g. handheld phone)


@dataclass
class LeverArm:
    """Physical airframe lever-arm container with active rotation into local ENU."""

    vector_body: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    status: LeverArmStatus = LeverArmStatus.LEVER_ARM_UNCALIBRATED
    covariance_body: Optional[np.ndarray] = None
    heuristic_uncertainty_m: float = 0.20  # CONFIGURATION HEURISTIC for uncalibrated multi-rotors
    mounting_rotation_camera_to_body: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.vector_body = np.asarray(self.vector_body, dtype=np.float64).reshape((3,))
        if np.any(np.isnan(self.vector_body)) or np.any(np.isinf(self.vector_body)):
            raise ValueError(f"Lever-arm vector contains NaN or Inf: {self.vector_body}")

        if self.covariance_body is not None:
            self.covariance_body = np.asarray(self.covariance_body, dtype=np.float64)
            if self.covariance_body.shape != (3, 3):
                raise ValueError(f"Lever-arm covariance must be (3, 3), got {self.covariance_body.shape}")

        if self.mounting_rotation_camera_to_body is None:
            self.mounting_rotation_camera_to_body = np.eye(3, dtype=np.float64)
        else:
            self.mounting_rotation_camera_to_body = np.asarray(self.mounting_rotation_camera_to_body, dtype=np.float64)
            if self.mounting_rotation_camera_to_body.shape != (3, 3):
                raise ValueError(f"Mounting rotation must be (3, 3), got {self.mounting_rotation_camera_to_body.shape}")

    @classmethod
    def calibrated(
        cls,
        dx_forward_m: float,
        dy_left_m: float,
        dz_up_m: float,
        covariance_3x3: Optional[np.ndarray] = None,
        mounting_rotation_camera_to_body: Optional[np.ndarray] = None,
    ) -> "LeverArm":
        """Construct a calibrated lever arm with known physical offsets in FLU airframe frame."""
        vec = np.array([dx_forward_m, dy_left_m, dz_up_m], dtype=np.float64)
        return cls(
            vector_body=vec,
            status=LeverArmStatus.LEVER_ARM_CALIBRATED,
            covariance_body=covariance_3x3,
            mounting_rotation_camera_to_body=mounting_rotation_camera_to_body,
        )

    @classmethod
    def uncalibrated(cls, heuristic_uncertainty_m: float = 0.20) -> "LeverArm":
        """Construct an uncalibrated lever arm.
        
        Defaults to zero displacement with explicit heuristic covariance inflation.
        """
        return cls(
            vector_body=np.zeros(3, dtype=np.float64),
            status=LeverArmStatus.LEVER_ARM_UNCALIBRATED,
            heuristic_uncertainty_m=heuristic_uncertainty_m,
        )

    @classmethod
    def zero(cls) -> "LeverArm":
        """Construct a zero lever arm (camera and antenna phase center coincident)."""
        return cls(
            vector_body=np.zeros(3, dtype=np.float64),
            status=LeverArmStatus.LEVER_ARM_ZERO,
            heuristic_uncertainty_m=0.0,
        )

    def transform_to_enu(self, r_body_to_enu: np.ndarray) -> np.ndarray:
        """Transform lever-arm offset vector from body frame to local ENU.
        
        Args:
            r_body_to_enu: (3, 3) active rotation matrix rotating FLU vectors to ENU.
            
        Returns:
            (3,) displacement vector in meters in local ENU frame.
        """
        r_body = np.asarray(r_body_to_enu, dtype=np.float64)
        if r_body.shape != (3, 3):
            raise ValueError(f"Body rotation matrix must be (3, 3), got {r_body.shape}")
        return r_body @ self.vector_body

    def predict_antenna_enu(self, c_cam_enu: np.ndarray, r_body_to_enu: np.ndarray) -> np.ndarray:
        """Predict physical GNSS antenna position given camera optical center in ENU.
        
        C_antenna_geo = C_cam_geo + R_body * L_body
        """
        cam = np.asarray(c_cam_enu, dtype=np.float64).reshape((3,))
        return cam + self.transform_to_enu(r_body_to_enu)

    def predict_camera_enu(self, c_antenna_enu: np.ndarray, r_body_to_enu: np.ndarray) -> np.ndarray:
        """Predict camera optical center given observed GNSS antenna position in ENU.
        
        C_cam_geo = C_antenna_geo - R_body * L_body
        """
        antenna = np.asarray(c_antenna_enu, dtype=np.float64).reshape((3,))
        return antenna - self.transform_to_enu(r_body_to_enu)

    def unapply_lever_arm(self, c_antenna_enu: np.ndarray, r_body_to_enu: np.ndarray) -> np.ndarray:
        """Subtract lever arm from antenna position to yield camera optical center in ENU."""
        return self.predict_camera_enu(c_antenna_enu, r_body_to_enu)

    def effective_covariance_enu(self, r_body_to_enu: np.ndarray) -> np.ndarray:
        """Compute effective lever-arm uncertainty covariance in local ENU frame.
        
        If calibrated: R_body * Sigma_lever * R_body^T
        If uncalibrated: (sigma_heuristic)^2 * I_3
        """
        r_body = np.asarray(r_body_to_enu, dtype=np.float64)
        if self.status == LeverArmStatus.LEVER_ARM_CALIBRATED and self.covariance_body is not None:
            return r_body @ self.covariance_body @ r_body.T
        elif self.status == LeverArmStatus.LEVER_ARM_UNCALIBRATED:
            var = self.heuristic_uncertainty_m ** 2
            return np.eye(3, dtype=np.float64) * var
        else:
            return np.zeros((3, 3), dtype=np.float64)
