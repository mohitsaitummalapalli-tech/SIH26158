"""Phase 3E.6 Benchmark Data Models & Type Specifications.

Encodes the non-collapse seven-axis evaluation model, 6-tier evidence hierarchy,
6-state result model, disjoint reference partition rules, and immutable records.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple


class ContractViolationError(ValueError):
    """Raised when an architectural, anti-leakage, or scientific contract is breached."""
    pass


class EvidenceLevel(IntEnum):
    """Six-tier ground truth evidence hierarchy."""
    LEVEL_0_NO_GROUND_TRUTH = 0
    LEVEL_1_TELEMETRY_ONLY = 1
    LEVEL_2_SYNTHETIC_KNOWN_GEOMETRY = 2
    LEVEL_3_INDEPENDENT_MEASURED_DISTANCES = 3
    LEVEL_4_SURVEYED_CHECKPOINTS = 4
    LEVEL_5_INDEPENDENT_REFERENCE_SCAN = 5


class BenchmarkStatus(str, Enum):
    """Evaluation result state model."""
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUABLE = "NOT_EVALUABLE"
    CONTRACT_VIOLATION = "CONTRACT_VIOLATION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INCONCLUSIVE = "INCONCLUSIVE"


class VisibilityState(str, Enum):
    """Five-state visibility evidence taxonomy."""
    OBSERVED = "OBSERVED"
    PHYSICALLY_OCCLUDED = "PHYSICALLY_OCCLUDED"
    UNOBSERVED = "UNOBSERVED"
    RECONSTRUCTION_MISSING = "RECONSTRUCTION_MISSING"
    UNDETERMINED = "UNDETERMINED"


class ReproducibilityLevel(str, Enum):
    """Four-tier environmental reproducibility standards."""
    R0_METADATA = "R0_METADATA"
    R1_NUMERICAL = "R1_NUMERICAL"
    R2_DETERMINISTIC = "R2_DETERMINISTIC"
    R3_BITWISE = "R3_BITWISE"


class LatencyTier(str, Enum):
    """Operational throughput classification."""
    OFFLINE_BATCH = "OFFLINE_BATCH"
    NEAR_REAL_TIME = "NEAR_REAL_TIME"
    REAL_TIME = "REAL_TIME"
    NOT_CLASSIFIED = "NOT_CLASSIFIED"


class ValidationScope(str, Enum):
    """Scope of benchmark validation execution."""
    METRIC_ENGINE_CONTRACT = "METRIC_ENGINE_CONTRACT"
    END_TO_END_RECONSTRUCTION = "END_TO_END_RECONSTRUCTION"


class TaxonomyClass(str, Enum):
    """Dataset taxonomy classes."""
    CLASS_A_SYNTHETIC_CONTROLLED = "CLASS_A_SYNTHETIC_CONTROLLED"
    CLASS_B_SEMI_SYNTHETIC_SENSOR = "CLASS_B_SEMI_SYNTHETIC_SENSOR"
    CLASS_C_REAL_TELEMETRY = "CLASS_C_REAL_TELEMETRY"
    CLASS_D_REAL_SURVEYED = "CLASS_D_REAL_SURVEYED"
    CLASS_E_REAL_UNREFERENCED = "CLASS_E_REAL_UNREFERENCED"


class QualityAxis(str, Enum):
    """The seven orthogonal quality axes of the Non-Collapse Axiom."""
    AXIS_A_VISUAL = "A_VISUAL"
    AXIS_B_GEOMETRIC = "B_GEOMETRIC"
    AXIS_C_METRIC_SCALE = "C_METRIC_SCALE"
    AXIS_D_GEOSPATIAL = "D_GEOSPATIAL"
    AXIS_E_TEXTURE = "E_TEXTURE"
    AXIS_F_UNCERTAINTY = "F_UNCERTAINTY"
    AXIS_G_COMPLETENESS = "G_COMPLETENESS"


class UncertaintyEvaluationMode(str, Enum):
    """Modes for uncertainty assessment."""
    HEURISTIC_CONFIDENCE_RANKING = "HEURISTIC_CONFIDENCE_RANKING"
    PROBABILISTIC_UNCERTAINTY = "PROBABILISTIC_UNCERTAINTY"


class UncertaintyStatus(str, Enum):
    """Status states for uncertainty calibration."""
    NOT_EVALUABLE = "NOT_EVALUABLE"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
    EVALUATED = "EVALUATED"
    CALIBRATION_SUPPORTED = "CALIBRATION_SUPPORTED"
    CALIBRATION_NOT_SUPPORTED = "CALIBRATION_NOT_SUPPORTED"


@dataclass(frozen=True)
class ReferencePartition:
    """Rigorous partition of surveyed / reference entities.
    
    Must guarantee pairwise disjoint IDs:
    - ESTIMATION_REFERENCE_SET: Used for parameter estimation (e.g. GCPs in Sim(3) fit).
    - CALIBRATION_REFERENCE_SET: Used for sensor/intrinsics calibration or lever-arm refinement.
    - VALIDATION_REFERENCE_SET: Strictly withheld from all estimation and optimization;
      used solely for hold-out verification post-convergence.
    """
    estimation_set_ids: Set[str] = field(default_factory=set)
    calibration_set_ids: Set[str] = field(default_factory=set)
    validation_set_ids: Set[str] = field(default_factory=set)

    def validate_disjointness(self) -> None:
        """Enforces pairwise disjointness. Raises ContractViolationError on any overlap."""
        est_val = self.estimation_set_ids.intersection(self.validation_set_ids)
        if est_val:
            raise ContractViolationError(
                f"Data leakage detected! ESTIMATION_REFERENCE_SET intersects VALIDATION_REFERENCE_SET: {est_val}"
            )

        cal_val = self.calibration_set_ids.intersection(self.validation_set_ids)
        if cal_val:
            raise ContractViolationError(
                f"Data leakage detected! CALIBRATION_REFERENCE_SET intersects VALIDATION_REFERENCE_SET: {cal_val}"
            )

        est_cal = self.estimation_set_ids.intersection(self.calibration_set_ids)
        if est_cal:
            raise ContractViolationError(
                f"Partition overlap detected! ESTIMATION_REFERENCE_SET intersects CALIBRATION_REFERENCE_SET: {est_cal}"
            )


@dataclass(frozen=True)
class AcquisitionConditions:
    """Environmental flight and scene parameters."""
    lighting: str = "overcast"
    wind_speed_mps: float = 0.0
    weather: str = "clear"
    scene_type: str = "quarry"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraCalibrationMeta:
    """Nominal or pre-calibrated camera intrinsics."""
    model: str = "PINHOLE"
    focal_length_px: float = 1000.0
    principal_point_px: Tuple[float, float] = (500.0, 500.0)
    distortion_coefficients: List[float] = field(default_factory=list)


@dataclass(frozen=True)
class TelemetryMeta:
    """Telemetry capture and accuracy specifications."""
    has_telemetry: bool = False
    sampling_rate_hz: float = 1.0
    gnss_format: str = "WGS84_ELLIPSOIDAL"
    horizontal_accuracy_m: float = 1.5
    vertical_accuracy_m: float = 2.5


@dataclass(frozen=True)
class GroundTruthMeta:
    """Independent ground truth description and reference partition."""
    has_ground_truth: bool = False
    ground_truth_type: str = "NONE"
    coordinate_reference_system: Optional[str] = None
    survey_accuracy_m: Optional[float] = None
    total_targets_surveyed: int = 0
    partition: Optional[ReferencePartition] = None


@dataclass(frozen=True)
class DatasetManifest:
    """Immutable, auditable manifest for a validation dataset."""
    dataset_id: str
    taxonomy_class: TaxonomyClass
    acquisition_conditions: AcquisitionConditions
    frame_count: int
    image_resolution: Tuple[int, int]
    camera_calibration: CameraCalibrationMeta
    telemetry_metadata: TelemetryMeta
    ground_truth_metadata: GroundTruthMeta
    sha256_checksum: str


@dataclass(frozen=True)
class StatisticalSummary:
    """Five-figure statistical error summary. Never reports only a mean."""
    mae: float
    rmse: float
    median: float
    p95: float
    maximum: float
    sample_count: int
    unit: str

    def __post_init__(self) -> None:
        for val, name in [
            (self.mae, "mae"),
            (self.rmse, "rmse"),
            (self.median, "median"),
            (self.p95, "p95"),
            (self.maximum, "maximum"),
        ]:
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"StatisticalSummary error: '{name}' is non-finite ({val}).")
            if val < -1e-9:
                raise ValueError(f"StatisticalSummary error: '{name}' cannot be negative ({val}).")


@dataclass(frozen=True)
class ClaimAuthorization:
    """Explicit claim authorization record output by the Claim Policy engine."""
    claims_allowed: List[str]
    claims_blocked: List[str]
    policy_status: BenchmarkStatus
    violation_reasons: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class StageExecutionRecord:
    """Immutable audit record for a single pipeline execution stage."""
    stage_name: str
    dataset_id: str
    software_commit: str
    config_hash: str
    input_hashes: Dict[str, str]
    output_hashes: Dict[str, str]
    started_at_utc: str
    finished_at_utc: str
    elapsed_seconds: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimingProfile:
    """End-to-end timing and throughput measurement profile."""
    total_wall_clock_sec: float
    stage_wall_clock_sec: Dict[str, float]
    input_frames: int
    decoded_duration_sec: float
    pipeline_fps: float
    real_time_factor: float
    latency_tier: LatencyTier
    hardware_environment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete, non-collapsed result container conforming to Benchmark Result Schema v2."""
    benchmark_id: str
    dataset_id: str
    evidence_level: EvidenceLevel
    software_commit: str
    result_state: BenchmarkStatus
    evaluable_axes: List[QualityAxis]
    not_evaluable_axes: List[QualityAxis]
    claim_authorization: ClaimAuthorization
    metrics: Dict[str, Any]
    timing_profile: TimingProfile
    reproducibility: Dict[str, Any]
    validation_scope: ValidationScope = ValidationScope.END_TO_END_RECONSTRUCTION
    validation_context: Dict[str, Any] = field(default_factory=dict)
    stage_records: List[StageExecutionRecord] = field(default_factory=list)
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "benchmark_id": self.benchmark_id,
            "dataset_id": self.dataset_id,
            "evidence_level": int(self.evidence_level),
            "software_commit": self.software_commit,
            "result_state": self.result_state.value,
            "validation_scope": self.validation_scope.value,
            "validation_context": self.validation_context,
            "evaluable_axes": [axis.value for axis in self.evaluable_axes],
            "not_evaluable_axes": [axis.value for axis in self.not_evaluable_axes],
            "claim_authorization": {
                "claims_allowed": self.claim_authorization.claims_allowed,
                "claims_blocked": self.claim_authorization.claims_blocked,
                "policy_status": self.claim_authorization.policy_status.value,
                "violation_reasons": self.claim_authorization.violation_reasons,
            },
            "metrics": self.metrics,
            "timing_profile": {
                "total_wall_clock_sec": self.timing_profile.total_wall_clock_sec,
                "stage_wall_clock_sec": self.timing_profile.stage_wall_clock_sec,
                "input_frames": self.timing_profile.input_frames,
                "decoded_duration_sec": self.timing_profile.decoded_duration_sec,
                "pipeline_fps": self.timing_profile.pipeline_fps,
                "real_time_factor": self.timing_profile.real_time_factor,
                "latency_tier": self.timing_profile.latency_tier.value,
                "hardware_environment": self.timing_profile.hardware_environment,
            },
            "reproducibility": self.reproducibility,
            "created_at_utc": self.created_at_utc,
        }
