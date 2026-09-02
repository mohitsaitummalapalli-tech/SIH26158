"""Validation module: Metric definitions and strict accuracy claim verification."""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from datetime import datetime, timezone


class InvalidAccuracyClaimError(ValueError):
    """Raised when an accuracy or benchmark metric lacks mandatory scientific provenance or violates mathematical bounds."""
    pass


# Non-negative metrics that cannot physically be less than 0.0
NON_NEGATIVE_METRIC_PREFIXES: Set[str] = {
    "rmse", "mae", "ate", "rpe", "chamfer", "distance", "error", 
    "time", "vram", "memory", "residual", "variance", "std"
}

PERCENTAGE_METRICS: Set[str] = {
    "accuracy_at_tau_pct", "completeness_at_tau_pct", "success_rate_pct",
    "scale_drift_pct", "inlier_ratio_pct"
}


@dataclass(frozen=True)
class ProvenanceRecord:
    """Scientific provenance metadata required for every recorded benchmark metric."""
    dataset_name: str
    ground_truth_reference: str  # e.g., "Terrestrial LiDAR scan 2026-08 (RIEGL VZ-400i)"
    ground_truth_format: str     # e.g., "las", "ply", "rtk_checkpoints"
    calculation_method: str      # Name or path of the reproducible evaluation script/algorithm
    dataset_checksum_sha256: Optional[str] = None
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class AccuracyMetric:
    """A verified, mathematically defined accuracy metric with full provenance.
    
    Enforces SIH26158 Rule 3: No accuracy claim is valid without:
    1. Named Dataset
    2. Defined Ground Truth
    3. Defined Metric Name & Valid Unit
    4. Reproducible Calculation Method
    5. Valid Mathematical Bounds (No NaN, Inf, or negative distances/errors)
    """
    metric_name: str
    value: float
    unit: str
    provenance: ProvenanceRecord
    threshold_tau_meters: Optional[float] = None
    confidence_interval_95: Optional[List[float]] = None
    sample_size: Optional[int] = None
    extra_context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that all required provenance and mathematical constraints are satisfied."""
        # 1. Metric name check
        if not self.metric_name or not self.metric_name.strip():
            raise InvalidAccuracyClaimError("Metric claim rejected: 'metric_name' must not be empty.")
        
        norm_name = self.metric_name.strip().lower()

        # 2. Unit check
        if not self.unit or not self.unit.strip():
            raise InvalidAccuracyClaimError("Metric claim rejected: 'unit' must not be empty.")

        # 3. Provenance presence check
        if self.provenance is None:
            raise InvalidAccuracyClaimError("Metric claim rejected: 'provenance' record is missing.")
            
        # 4. Dataset name check
        if not self.provenance.dataset_name or not self.provenance.dataset_name.strip():
            raise InvalidAccuracyClaimError(
                "Metric claim rejected: 'dataset_name' is missing or empty. "
                "Accuracy claims cannot be made without a named dataset."
            )
            
        # 5. Ground truth reference check
        if not self.provenance.ground_truth_reference or not self.provenance.ground_truth_reference.strip():
            raise InvalidAccuracyClaimError(
                "Metric claim rejected: 'ground_truth_reference' is missing or empty. "
                "Accuracy claims cannot be made without a defined ground truth source."
            )
            
        # 6. Calculation method check
        if not self.provenance.calculation_method or not self.provenance.calculation_method.strip():
            raise InvalidAccuracyClaimError(
                "Metric claim rejected: 'calculation_method' is missing or empty. "
                "Accuracy claims must provide a reproducible calculation procedure."
            )

        # 7. Numeric validity: Reject NaN and Infinite values
        if math.isnan(self.value):
            raise InvalidAccuracyClaimError(f"Metric claim rejected: '{self.metric_name}' value cannot be NaN.")
        
        if math.isinf(self.value):
            raise InvalidAccuracyClaimError(f"Metric claim rejected: '{self.metric_name}' value cannot be Infinite.")

        # 8. Negative error check where mathematically impossible
        is_non_negative = any(prefix in norm_name for prefix in NON_NEGATIVE_METRIC_PREFIXES)
        if is_non_negative and self.value < 0.0:
            raise InvalidAccuracyClaimError(
                f"Metric claim rejected: '{self.metric_name}' value ({self.value}) cannot be negative."
            )

        # 9. Percentage bounds check
        if norm_name in PERCENTAGE_METRICS:
            if self.value < 0.0 or self.value > 100.0:
                raise InvalidAccuracyClaimError(
                    f"Metric claim rejected: Percentage metric '{self.metric_name}' must be in range [0, 100], got {self.value}."
                )

        # 10. Threshold validation
        if self.threshold_tau_meters is not None:
            if math.isnan(self.threshold_tau_meters) or self.threshold_tau_meters <= 0.0:
                raise InvalidAccuracyClaimError(
                    f"Metric claim rejected: Threshold tau ({self.threshold_tau_meters}) must be a positive non-zero number."
                )


class StrictMetricValidator:
    """Validator guardrail ensuring no unverified metrics enter the system."""
    
    @staticmethod
    def validate_metric_claim(
        metric_name: str,
        value: float,
        unit: str,
        dataset_name: str,
        ground_truth_ref: str,
        calc_method: str,
        threshold: Optional[float] = None,
        checksum: Optional[str] = None
    ) -> AccuracyMetric:
        """Construct and validate an accuracy metric record."""
        provenance = ProvenanceRecord(
            dataset_name=dataset_name,
            ground_truth_reference=ground_truth_ref,
            ground_truth_format="las",
            calculation_method=calc_method,
            dataset_checksum_sha256=checksum
        )
        return AccuracyMetric(
            metric_name=metric_name,
            value=value,
            unit=unit,
            provenance=provenance,
            threshold_tau_meters=threshold
        )
