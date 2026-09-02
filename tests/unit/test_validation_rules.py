"""Test suite verifying strict enforcement of SIH26158 Engineering Rules (Accuracy Claims)."""

import pytest
import math
from src.validation import (
    AccuracyMetric,
    ProvenanceRecord,
    InvalidAccuracyClaimError,
    StrictMetricValidator
)


@pytest.fixture
def valid_provenance():
    return ProvenanceRecord(
        dataset_name="urban_corridor_seq01",
        ground_truth_reference="Terrestrial LiDAR RIEGL VZ-400i (2026-08)",
        ground_truth_format="las",
        calculation_method="benchmarks.metrics.compute_chamfer_distance",
        dataset_checksum_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


# Test Case 1: Valid metric creation
def test_valid_accuracy_metric_creation(valid_provenance):
    metric = AccuracyMetric(
        metric_name="chamfer_distance",
        value=0.142,
        unit="meters",
        provenance=valid_provenance,
        threshold_tau_meters=0.10
    )
    assert metric.value == 0.142
    assert metric.unit == "meters"
    assert metric.provenance.dataset_name == "urban_corridor_seq01"
    assert metric.provenance.dataset_checksum_sha256 is not None


# Test Case 2: Reject empty metric name
def test_reject_empty_metric_name(valid_provenance):
    with pytest.raises(InvalidAccuracyClaimError, match="metric_name"):
        AccuracyMetric(
            metric_name="",
            value=0.15,
            unit="meters",
            provenance=valid_provenance
        )


# Test Case 3: Reject empty dataset name (Rule 1 & 3)
def test_reject_empty_dataset_name():
    provenance = ProvenanceRecord(
        dataset_name="",  # Missing
        ground_truth_reference="LiDAR Scan",
        ground_truth_format="las",
        calculation_method="benchmarks.metrics.compute_chamfer"
    )
    with pytest.raises(InvalidAccuracyClaimError, match="dataset_name"):
        AccuracyMetric(
            metric_name="ate_rmse",
            value=0.25,
            unit="meters",
            provenance=provenance
        )


# Test Case 4: Reject empty ground truth reference (Rule 1 & 3)
def test_reject_empty_ground_truth_reference():
    provenance = ProvenanceRecord(
        dataset_name="urban_corridor_seq01",
        ground_truth_reference="   ",  # Blank
        ground_truth_format="las",
        calculation_method="benchmarks.metrics.compute_chamfer"
    )
    with pytest.raises(InvalidAccuracyClaimError, match="ground_truth_reference"):
        AccuracyMetric(
            metric_name="ate_rmse",
            value=0.25,
            unit="meters",
            provenance=provenance
        )


# Test Case 5: Reject empty calculation method
def test_reject_empty_calculation_method():
    provenance = ProvenanceRecord(
        dataset_name="urban_corridor_seq01",
        ground_truth_reference="LiDAR Scan",
        ground_truth_format="las",
        calculation_method=""  # Blank
    )
    with pytest.raises(InvalidAccuracyClaimError, match="calculation_method"):
        AccuracyMetric(
            metric_name="ate_rmse",
            value=0.25,
            unit="meters",
            provenance=provenance
        )


# Test Case 6: Reject empty unit
def test_reject_empty_unit(valid_provenance):
    with pytest.raises(InvalidAccuracyClaimError, match="unit"):
        AccuracyMetric(
            metric_name="ate_rmse",
            value=0.25,
            unit="",  # Missing unit
            provenance=valid_provenance
        )


# Test Case 7: Reject missing provenance
def test_reject_none_provenance():
    with pytest.raises(InvalidAccuracyClaimError, match="provenance"):
        AccuracyMetric(
            metric_name="ate_rmse",
            value=0.25,
            unit="meters",
            provenance=None  # type: ignore
        )


# Test Case 8: Reject NaN metric value
def test_reject_nan_metric_value(valid_provenance):
    with pytest.raises(InvalidAccuracyClaimError, match="cannot be NaN"):
        AccuracyMetric(
            metric_name="chamfer_distance",
            value=float("nan"),
            unit="meters",
            provenance=valid_provenance
        )


# Test Case 9: Reject Infinite metric value
def test_reject_infinite_metric_value(valid_provenance):
    with pytest.raises(InvalidAccuracyClaimError, match="cannot be Infinite"):
        AccuracyMetric(
            metric_name="ate_rmse",
            value=float("inf"),
            unit="meters",
            provenance=valid_provenance
        )


# Test Case 10: Reject negative error/distance values
def test_reject_negative_error_metric(valid_provenance):
    with pytest.raises(InvalidAccuracyClaimError, match="cannot be negative"):
        AccuracyMetric(
            metric_name="ate_rmse",
            value=-0.05,
            unit="meters",
            provenance=valid_provenance
        )

    with pytest.raises(InvalidAccuracyClaimError, match="cannot be negative"):
        AccuracyMetric(
            metric_name="chamfer_distance",
            value=-1.2,
            unit="meters",
            provenance=valid_provenance
        )


# Test Case 11: Reject out-of-bounds percentage
def test_reject_out_of_bounds_percentage(valid_provenance):
    with pytest.raises(InvalidAccuracyClaimError, match=r"\[0, 100\]"):
        AccuracyMetric(
            metric_name="accuracy_at_tau_pct",
            value=105.4,  # Over 100%
            unit="percent",
            provenance=valid_provenance
        )

    with pytest.raises(InvalidAccuracyClaimError, match=r"\[0, 100\]"):
        AccuracyMetric(
            metric_name="completeness_at_tau_pct",
            value=-5.0,  # Below 0%
            unit="percent",
            provenance=valid_provenance
        )


# Test Case 12: Reject invalid threshold tau
def test_reject_invalid_threshold_tau(valid_provenance):
    with pytest.raises(InvalidAccuracyClaimError, match="positive non-zero"):
        AccuracyMetric(
            metric_name="accuracy_at_tau_pct",
            value=78.5,
            unit="percent",
            provenance=valid_provenance,
            threshold_tau_meters=-0.10  # Negative threshold
        )


# Test Case 13: StrictMetricValidator helper test
def test_strict_metric_validator_helper():
    metric = StrictMetricValidator.validate_metric_claim(
        metric_name="ate_rmse",
        value=0.31,
        unit="meters",
        dataset_name="seq02_rural_corridor",
        ground_truth_ref="RTK-GPS GNSS Station Checkpoints",
        calc_method="src.validation.metrics.compute_ate_rmse",
        threshold=0.5,
        checksum="abcd1234ef5678"
    )
    assert metric.value == 0.31
    assert metric.threshold_tau_meters == 0.5
    assert metric.provenance.dataset_checksum_sha256 == "abcd1234ef5678"
