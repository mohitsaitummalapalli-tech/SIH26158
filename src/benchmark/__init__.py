"""Phase 3E.6 End-to-End Validation, Benchmarking & Evidence Architecture Package."""

from src.benchmark.models import (
    ContractViolationError,
    EvidenceLevel,
    BenchmarkStatus,
    VisibilityState,
    ReproducibilityLevel,
    LatencyTier,
    TaxonomyClass,
    QualityAxis,
    ValidationScope,
    UncertaintyEvaluationMode,
    UncertaintyStatus,
    ReferencePartition,
    AcquisitionConditions,
    CameraCalibrationMeta,
    TelemetryMeta,
    GroundTruthMeta,
    DatasetManifest,
    StatisticalSummary,
    ClaimAuthorization,
    StageExecutionRecord,
    TimingProfile,
    BenchmarkResult,
)
from src.benchmark.reproducibility import (
    verify_reproducibility_level,
)
from src.benchmark.claim_policy import (
    ClaimPolicyEngine,
    ALL_CANONICAL_CLAIMS,
    CLAIM_POLICY_MATRIX,
)
from src.benchmark.metrics_geometry import (
    compute_statistical_summary,
    compute_point_to_point_distances,
    compute_point_to_plane_distances,
    compute_bidirectional_chamfer,
    compute_hausdorff_distances,
    compute_f_score_at_tau,
    compute_normal_angular_deviation,
)
from src.benchmark.metrics_metric_scale import (
    ValidationSegment,
    compute_relative_scale_error,
    evaluate_metric_scale,
)
from src.benchmark.metrics_geospatial import (
    CheckpointReference,
    CheckpointEvaluationResult,
    evaluate_geospatial_checkpoints,
)
from src.benchmark.metrics_trajectory import (
    RawTrajectoryResult,
    Sim3AlignedTrajectoryResult,
    RpeDriftResult,
    evaluate_raw_trajectory_ate,
    evaluate_sim3_aligned_trajectory_ate,
    evaluate_rpe_drift,
    solve_umeyama_sim3,
    DISCLAIMER_SIM3_ALIGNMENT,
)
from src.benchmark.metrics_texture import (
    TextureDiagnosticMetadata,
    TextureDiagnosticResult,
    compute_masked_psnr,
    compute_masked_ssim,
    compute_seam_gradient_discontinuity,
    evaluate_texture_diagnostics,
    DISCLAIMER_TEXTURE_DIAGNOSTIC,
)
from src.benchmark.metrics_completeness import (
    CompletenessEvaluationResult,
    classify_visibility_evidence,
    evaluate_roi_completeness,
)
from src.benchmark.metrics_uncertainty import (
    HeuristicRankingResult,
    ProbabilisticCoverageResult,
    compute_spearman_rank_correlation,
    compute_bootstrap_confidence_interval,
    evaluate_heuristic_confidence_ranking,
    evaluate_probabilistic_coverage,
    transform_spatial_covariance,
)
from src.benchmark.robustness_perturbations import (
    EnvelopeClassification,
    PerturbationRecord,
    apply_gaussian_image_blur,
    apply_linear_motion_blur,
    apply_gnss_gaussian_noise,
    inject_gnss_outliers,
    simulate_telemetry_dropout,
    simulate_shutter_clock_bias,
    apply_focal_length_perturbation,
    subsample_frame_dropping,
    generate_collinear_trajectory,
)
from src.benchmark.timing_profiler import (
    BenchmarkTimingProfiler,
    get_system_hardware_environment,
)
from src.benchmark.engine import BenchmarkEngine

__all__ = [
    # Models
    "ContractViolationError",
    "EvidenceLevel",
    "BenchmarkStatus",
    "VisibilityState",
    "ReproducibilityLevel",
    "LatencyTier",
    "TaxonomyClass",
    "QualityAxis",
    "UncertaintyEvaluationMode",
    "UncertaintyStatus",
    "ReferencePartition",
    "AcquisitionConditions",
    "CameraCalibrationMeta",
    "TelemetryMeta",
    "GroundTruthMeta",
    "DatasetManifest",
    "StatisticalSummary",
    "ClaimAuthorization",
    "StageExecutionRecord",
    "TimingProfile",
    "BenchmarkResult",
    # Claim Policy
    "ClaimPolicyEngine",
    "ALL_CANONICAL_CLAIMS",
    "CLAIM_POLICY_MATRIX",
    # Geometry Metrics
    "compute_statistical_summary",
    "compute_point_to_point_distances",
    "compute_point_to_plane_distances",
    "compute_bidirectional_chamfer",
    "compute_hausdorff_distances",
    "compute_f_score_at_tau",
    "compute_normal_angular_deviation",
    # Metric Scale
    "ValidationSegment",
    "compute_relative_scale_error",
    "evaluate_metric_scale",
    # Geospatial
    "CheckpointReference",
    "CheckpointEvaluationResult",
    "evaluate_geospatial_checkpoints",
    # Trajectory
    "RawTrajectoryResult",
    "Sim3AlignedTrajectoryResult",
    "RpeDriftResult",
    "evaluate_raw_trajectory_ate",
    "evaluate_sim3_aligned_trajectory_ate",
    "evaluate_rpe_drift",
    "solve_umeyama_sim3",
    "DISCLAIMER_SIM3_ALIGNMENT",
    # Texture
    "TextureDiagnosticMetadata",
    "TextureDiagnosticResult",
    "compute_masked_psnr",
    "compute_masked_ssim",
    "compute_seam_gradient_discontinuity",
    "evaluate_texture_diagnostics",
    "DISCLAIMER_TEXTURE_DIAGNOSTIC",
    # Completeness
    "CompletenessEvaluationResult",
    "classify_visibility_evidence",
    "evaluate_roi_completeness",
    # Uncertainty
    "HeuristicRankingResult",
    "ProbabilisticCoverageResult",
    "compute_spearman_rank_correlation",
    "compute_bootstrap_confidence_interval",
    "evaluate_heuristic_confidence_ranking",
    "evaluate_probabilistic_coverage",
    # Robustness
    "EnvelopeClassification",
    "PerturbationRecord",
    "apply_gaussian_image_blur",
    "apply_linear_motion_blur",
    "apply_gnss_gaussian_noise",
    "inject_gnss_outliers",
    "simulate_telemetry_dropout",
    "simulate_shutter_clock_bias",
    "apply_focal_length_perturbation",
    "subsample_frame_dropping",
    "generate_collinear_trajectory",
    # Timing
    "BenchmarkTimingProfiler",
    "get_system_hardware_environment",
    # Engine
    "BenchmarkEngine",
]
