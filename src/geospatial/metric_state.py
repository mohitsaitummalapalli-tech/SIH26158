"""Metric scale state machine governing the certification of 3D reconstruction units.

States:
- NOT_METRIC: Default state. Arbitrary dimensionless reconstruction gauge.
- METRIC_SCALE_ESTIMATED: Statistical Sim(3) fit converged with sufficient observability.
- METRIC_SCALE_VALIDATED: Verified against independent surveyed checkpoints/GCPs.
- METRIC_SCALE_UNCERTAIN: Estimation converged but with high residuals (>5.0m) or weak geometry.
- METRIC_ALIGNMENT_FAILED: Optimization divergence, collinearity, stationary hover, or <4 inliers.

Invariant: No reconstruction is promoted to METRIC_SCALE_VALIDATED without independent ground truth.
"""

from enum import Enum
from typing import Optional, List
from dataclasses import dataclass


class MetricScaleStatus(str, Enum):
    """Certified state of metric scale calibration."""
    NOT_METRIC = "NOT_METRIC"                           # Uncalibrated dimensionless SfM gauge
    METRIC_SCALE_ESTIMATED = "METRIC_SCALE_ESTIMATED"   # Statistically estimated from GNSS trajectory
    METRIC_SCALE_VALIDATED = "METRIC_SCALE_VALIDATED"   # Independently validated against ground truth
    METRIC_SCALE_UNCERTAIN = "METRIC_SCALE_UNCERTAIN"   # Estimated but exceeds residual or uncertainty gates
    METRIC_ALIGNMENT_FAILED = "METRIC_ALIGNMENT_FAILED" # Estimation failed or degenerate flight geometry


@dataclass
class MetricStateTransition:
    """Audit record of a metric scale state transition."""
    previous_state: MetricScaleStatus
    current_state: MetricScaleStatus
    reason: str


class MetricStateMachine:
    """Deterministic state machine governing metric scale promotion."""

    def __init__(self, initial_state: MetricScaleStatus = MetricScaleStatus.NOT_METRIC) -> None:
        self._state: MetricScaleStatus = initial_state
        self._history: List[MetricStateTransition] = []

    @property
    def current_state(self) -> MetricScaleStatus:
        return self._state

    @property
    def is_metric(self) -> bool:
        """Only true if scale has been successfully estimated or validated."""
        return self._state in (
            MetricScaleStatus.METRIC_SCALE_ESTIMATED,
            MetricScaleStatus.METRIC_SCALE_VALIDATED,
        )

    @property
    def history(self) -> List[MetricStateTransition]:
        return list(self._history)

    def transition(self, target_state: MetricScaleStatus, reason: str) -> None:
        """Record and execute state transition."""
        prev = self._state
        self._state = target_state
        self._history.append(MetricStateTransition(previous_state=prev, current_state=target_state, reason=reason))

    def evaluate_estimation(
        self,
        estimation_success: bool,
        is_observable: bool,
        inlier_count: int,
        rmse_3d_m: float,
        relative_scale_uncertainty: float,
        tau_rmse_uncertain_m: float = 5.0,
        tau_rel_uncertainty: float = 0.15,
        observability_failure_reason: Optional[str] = None,
    ) -> MetricScaleStatus:
        """Evaluate estimation outcome and determine new state.
        
        Args:
            estimation_success: Boolean indicator from robust estimator.
            is_observable: Boolean indicator from scale observability check.
            inlier_count: Number of consensus inlier GNSS points.
            rmse_3d_m: Total 3D RMSE on inliers in local ENU meters.
            relative_scale_uncertainty: 1-sigma relative uncertainty (sigma_s / s).
            tau_rmse_uncertain_m: Metric threshold between estimated and uncertain (default 5.0m).
            tau_rel_uncertainty: Dimensionless threshold on scale uncertainty (default 15%).
            observability_failure_reason: Reason string if observability failed.
            
        Returns:
            The resulting MetricScaleStatus.
        """
        if not estimation_success:
            self.transition(
                MetricScaleStatus.METRIC_ALIGNMENT_FAILED,
                reason="Robust Sim(3) estimation failed to converge or find inliers",
            )
            return self._state

        if not is_observable:
            self.transition(
                MetricScaleStatus.METRIC_ALIGNMENT_FAILED,
                reason=f"Geometric observability failed: {observability_failure_reason or 'Degenerate geometry'}",
            )
            return self._state

        if inlier_count < 4:
            self.transition(
                MetricScaleStatus.METRIC_ALIGNMENT_FAILED,
                reason=f"Insufficient inliers for metric certification: {inlier_count} < 4",
            )
            return self._state

        # Check uncertainty and residual gates
        if rmse_3d_m > tau_rmse_uncertain_m or relative_scale_uncertainty > tau_rel_uncertainty:
            self.transition(
                MetricScaleStatus.METRIC_SCALE_UNCERTAIN,
                reason=f"High residual/uncertainty: RMSE={rmse_3d_m:.2f}m (gate {tau_rmse_uncertain_m:.2f}m), "
                       f"rel_unc={relative_scale_uncertainty:.1%} (gate {tau_rel_uncertainty:.1%})",
            )
            return self._state

        self.transition(
            MetricScaleStatus.METRIC_SCALE_ESTIMATED,
            reason=f"Sim(3) converged with {inlier_count} inliers, RMSE={rmse_3d_m:.2f}m, "
                   f"rel_unc={relative_scale_uncertainty:.1%}",
        )
        return self._state

    def evaluate_validation(self, is_validated: bool, validation_rmse_m: float, tolerance_m: float) -> MetricScaleStatus:
        """Evaluate independent checkpoint validation outcome."""
        if self._state not in (
            MetricScaleStatus.METRIC_SCALE_ESTIMATED,
            MetricScaleStatus.METRIC_SCALE_UNCERTAIN,
        ):
            # Cannot validate if estimation failed or is not metric
            return self._state

        if is_validated:
            self.transition(
                MetricScaleStatus.METRIC_SCALE_VALIDATED,
                reason=f"Validated against independent surveyed checkpoints: RMSE={validation_rmse_m:.3f}m <= {tolerance_m:.3f}m",
            )
        else:
            self.transition(
                MetricScaleStatus.METRIC_SCALE_UNCERTAIN,
                reason=f"Validation checkpoints contradict estimated alignment: RMSE={validation_rmse_m:.3f}m > {tolerance_m:.3f}m",
            )
        return self._state
