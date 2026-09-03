"""Independent metric validation against ground control points and surveyed references.

Strict Scientific Separation:
- Telemetry GNSS: Observation source used for ESTIMATION.
- GCPs / Survey Checkpoints: Independent reference used strictly for VALIDATION.

Terminology Invariant:
- Alignment residuals against telemetry are labeled 'alignment residual' or 'telemetry consistency'.
- 'Reconstruction accuracy' is strictly reserved for independent certified ground-truth comparisons.
"""

from dataclasses import dataclass, field
import math
from typing import List, Dict, Tuple, Optional
import numpy as np

from src.geospatial.sim3 import Sim3


@dataclass
class GroundControlPoint:
    """Surveyed physical reference target."""
    point_id: str
    point_rec: np.ndarray             # 3D position in reconstruction gauge
    gcp_enu: np.ndarray               # Certified 3D survey position in local ENU meters
    survey_accuracy_m: float = 0.05   # Certified 1-sigma survey precision (meters)

    def __post_init__(self) -> None:
        self.point_rec = np.asarray(self.point_rec, dtype=np.float64).reshape((3,))
        self.gcp_enu = np.asarray(self.gcp_enu, dtype=np.float64).reshape((3,))


@dataclass
class CheckpointResidual:
    """Individual checkpoint validation discrepancy."""
    point_id: str
    residual_enu_m: np.ndarray
    horizontal_error_m: float
    vertical_error_m: float
    total_3d_error_m: float
    passed_tolerance: bool


@dataclass
class ValidationReport:
    """Complete independent validation verdict and error diagnostics."""
    is_validated: bool
    num_checkpoints: int
    horizontal_rmse_m: float
    vertical_rmse_m: float
    total_3d_rmse_m: float
    max_error_m: float
    tolerance_m: float
    residuals: List[CheckpointResidual] = field(default_factory=list)
    verdict_summary: str = ""


class MetricValidator:
    """Validates estimated Sim(3) model against independent surveyed references."""

    def __init__(self, tolerance_multiplier: float = 3.0) -> None:
        self.tolerance_multiplier = tolerance_multiplier

    def validate(
        self,
        sim3: Sim3,
        checkpoints: List[GroundControlPoint],
    ) -> ValidationReport:
        """Validate estimated Sim(3) against a list of hold-out GroundControlPoints.
        
        Args:
            sim3: Converged Sim(3) similarity transformation.
            checkpoints: Independent surveyed checkpoints.
            
        Returns:
            ValidationReport with RMSE and pass/fail verdict.
        """
        if not checkpoints:
            return ValidationReport(
                is_validated=False,
                num_checkpoints=0,
                horizontal_rmse_m=0.0,
                vertical_rmse_m=0.0,
                total_3d_rmse_m=0.0,
                max_error_m=0.0,
                tolerance_m=0.0,
                residuals=[],
                verdict_summary="NO_CHECKPOINTS_SUPPLIED: Independent metric validation requires reference points.",
            )

        residuals_list: List[CheckpointResidual] = []
        errors_3d: List[float] = []
        errors_h: List[float] = []
        errors_v: List[float] = []

        # Mean survey precision
        mean_survey_sigma = float(np.mean([cp.survey_accuracy_m for cp in checkpoints]))
        tolerance_m = self.tolerance_multiplier * mean_survey_sigma

        for cp in checkpoints:
            # Model prediction: X_geo = s * R * X_rec + t
            predicted_enu = sim3.transform_point(cp.point_rec)
            r_vec = predicted_enu - cp.gcp_enu

            err_h = float(np.linalg.norm(r_vec[:2]))
            err_v = float(abs(r_vec[2]))
            err_3d = float(np.linalg.norm(r_vec))

            passed = err_3d <= (self.tolerance_multiplier * cp.survey_accuracy_m)

            errors_h.append(err_h)
            errors_v.append(err_v)
            errors_3d.append(err_3d)

            residuals_list.append(CheckpointResidual(
                point_id=cp.point_id,
                residual_enu_m=r_vec,
                horizontal_error_m=err_h,
                vertical_error_m=err_v,
                total_3d_error_m=err_3d,
                passed_tolerance=passed,
            ))

        h_rmse = float(np.sqrt(np.mean(np.array(errors_h) ** 2)))
        v_rmse = float(np.sqrt(np.mean(np.array(errors_v) ** 2)))
        total_rmse = float(np.sqrt(np.mean(np.array(errors_3d) ** 2)))
        max_err = float(np.max(errors_3d))

        is_validated = total_rmse <= tolerance_m

        if is_validated:
            summary = f"VALIDATED: Checkpoint RMSE={total_rmse:.3f}m <= tolerance={tolerance_m:.3f}m across {len(checkpoints)} points."
        else:
            summary = f"VALIDATION_FAILED: Checkpoint RMSE={total_rmse:.3f}m > tolerance={tolerance_m:.3f}m."

        return ValidationReport(
            is_validated=is_validated,
            num_checkpoints=len(checkpoints),
            horizontal_rmse_m=h_rmse,
            vertical_rmse_m=v_rmse,
            total_3d_rmse_m=total_rmse,
            max_error_m=max_err,
            tolerance_m=tolerance_m,
            residuals=residuals_list,
            verdict_summary=summary,
        )
