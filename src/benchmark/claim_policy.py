"""Phase 3E.6 Machine-Readable Claim-Policy & Anti-Leakage Engine.

Enforces strict mathematical gating of scientific claims against ground-truth evidence levels,
anti-leakage partition invariants, and zero-tolerance self-evaluation checks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
from src.benchmark.models import (
    ContractViolationError,
    EvidenceLevel,
    BenchmarkStatus,
    TaxonomyClass,
    ReferencePartition,
    ClaimAuthorization,
    DatasetManifest,
)

# Canonical scientific claims registered in the architecture
ALL_CANONICAL_CLAIMS: Set[str] = {
    "reprojection_consistency",
    "relative_motion_profile",
    "visual_appearance_psnr",
    "gnss_fitting_residual",
    "synthetic_recovery_acc",
    "segment_scale_accuracy",
    "horizontal_checkpoint_rmse",
    "vertical_checkpoint_rmse",
    "surveyed_3d_accuracy",
    "surface_mesh_chamfer",
    "probabilistic_gaussian_coverage",
    "radiometric_color_accuracy",
    "universal_drone_accuracy",
}

# Base claim permission matrix by EvidenceLevel
CLAIM_POLICY_MATRIX: Dict[EvidenceLevel, Set[str]] = {
    EvidenceLevel.LEVEL_0_NO_GROUND_TRUTH: {
        "reprojection_consistency",
        "relative_motion_profile",
        "visual_appearance_psnr",
    },
    EvidenceLevel.LEVEL_1_TELEMETRY_ONLY: {
        "reprojection_consistency",
        "relative_motion_profile",
        "visual_appearance_psnr",
        "gnss_fitting_residual",
    },
    EvidenceLevel.LEVEL_2_SYNTHETIC_KNOWN_GEOMETRY: {
        "reprojection_consistency",
        "relative_motion_profile",
        "visual_appearance_psnr",
        "gnss_fitting_residual",
        "synthetic_recovery_acc",
        "segment_scale_accuracy",
        "surface_mesh_chamfer",
    },
    EvidenceLevel.LEVEL_3_INDEPENDENT_MEASURED_DISTANCES: {
        "reprojection_consistency",
        "relative_motion_profile",
        "visual_appearance_psnr",
        "gnss_fitting_residual",
        "segment_scale_accuracy",
    },
    EvidenceLevel.LEVEL_4_SURVEYED_CHECKPOINTS: {
        "reprojection_consistency",
        "relative_motion_profile",
        "visual_appearance_psnr",
        "gnss_fitting_residual",
        "segment_scale_accuracy",
        "horizontal_checkpoint_rmse",
        "vertical_checkpoint_rmse",
        "surveyed_3d_accuracy",
    },
    EvidenceLevel.LEVEL_5_INDEPENDENT_REFERENCE_SCAN: {
        "reprojection_consistency",
        "relative_motion_profile",
        "visual_appearance_psnr",
        "gnss_fitting_residual",
        "segment_scale_accuracy",
        "horizontal_checkpoint_rmse",
        "vertical_checkpoint_rmse",
        "surveyed_3d_accuracy",
        "surface_mesh_chamfer",
    },
}


class ClaimPolicyEngine:
    """Rigorous policy auditor ensuring zero ungrounded accuracy claims enter the benchmark."""

    @staticmethod
    def audit_reference_partition(partition: Optional[ReferencePartition]) -> None:
        """Enforces pairwise disjointness of estimation, calibration, and validation reference sets."""
        if partition is not None:
            partition.validate_disjointness()

    @staticmethod
    def verify_no_self_evaluation(
        reconstruction_hash: str,
        reference_hash: str,
        reconstruction_cloud: Any = None,
        reference_cloud: Any = None,
    ) -> None:
        """Rejects using the evaluated reconstruction as ground truth (MUT-01)."""
        if reconstruction_hash and reference_hash and reconstruction_hash.strip() == reference_hash.strip():
            raise ContractViolationError(
                f"Self-evaluation cheat detected! Reconstruction hash matches reference hash: {reconstruction_hash}"
            )
        if reconstruction_cloud is not None and reference_cloud is not None:
            if reconstruction_cloud is reference_cloud:
                raise ContractViolationError(
                    "Self-evaluation cheat detected! Reconstruction object is identical in memory to reference object."
                )

    @staticmethod
    def verify_no_validation_alignment(
        validation_targets_pre_alignment: Any,
        validation_targets_post_alignment: Any,
    ) -> None:
        """Verifies that validation checkpoints were not subjected to ICP or tweak alignment (MUT-02)."""
        import numpy as np
        pre = np.asarray(validation_targets_pre_alignment, dtype=np.float64)
        post = np.asarray(validation_targets_post_alignment, dtype=np.float64)
        if not np.allclose(pre, post, atol=1e-9):
            raise ContractViolationError(
                "Integrity violation (MUT-02): Validation checkpoints were modified or aligned during evaluation!"
            )

    @staticmethod
    def audit_claim_authorization(
        evidence_level: EvidenceLevel,
        requested_claims: Optional[List[str]] = None,
        independent_reference_available: bool = False,
        holdout_enforced: bool = False,
        metric_scale_validated: bool = False,
        geospatial_reference_available: bool = False,
        surface_reference_available: bool = False,
        radiometric_calibration_available: bool = False,
        probabilistic_model_declared: bool = False,
        partition: Optional[ReferencePartition] = None,
    ) -> ClaimAuthorization:
        """Evaluates allowed and blocked claims under the evidence matrix."""
        # Check partition disjointness
        if partition is not None:
            partition.validate_disjointness()

        base_allowed = CLAIM_POLICY_MATRIX.get(evidence_level, set()).copy()
        
        # universal_drone_accuracy is NEVER allowed under any circumstance
        base_allowed.discard("universal_drone_accuracy")

        # Refine allowed set based on actual physical prerequisites
        if not independent_reference_available and evidence_level >= EvidenceLevel.LEVEL_3_INDEPENDENT_MEASURED_DISTANCES:
            base_allowed.discard("segment_scale_accuracy")
            base_allowed.discard("horizontal_checkpoint_rmse")
            base_allowed.discard("vertical_checkpoint_rmse")
            base_allowed.discard("surveyed_3d_accuracy")
            base_allowed.discard("surface_mesh_chamfer")

        if not holdout_enforced:
            base_allowed.discard("horizontal_checkpoint_rmse")
            base_allowed.discard("vertical_checkpoint_rmse")
            base_allowed.discard("surveyed_3d_accuracy")

        if not surface_reference_available:
            base_allowed.discard("surface_mesh_chamfer")

        if not geospatial_reference_available:
            base_allowed.discard("horizontal_checkpoint_rmse")
            base_allowed.discard("vertical_checkpoint_rmse")
            base_allowed.discard("surveyed_3d_accuracy")

        # Radiometric claim requires explicit radiometric calibration
        if radiometric_calibration_available:
            base_allowed.add("radiometric_color_accuracy")
        else:
            base_allowed.discard("radiometric_color_accuracy")

        # Probabilistic uncertainty claims require explicit declared model
        if probabilistic_model_declared and evidence_level >= EvidenceLevel.LEVEL_2_SYNTHETIC_KNOWN_GEOMETRY:
            base_allowed.add("probabilistic_gaussian_coverage")
        else:
            base_allowed.discard("probabilistic_gaussian_coverage")

        allowed_list = sorted(list(base_allowed))
        blocked_list = sorted(list(ALL_CANONICAL_CLAIMS - base_allowed))

        violations: List[str] = []
        policy_status = BenchmarkStatus.PASS

        if requested_claims:
            for req in requested_claims:
                if req not in base_allowed:
                    violations.append(
                        f"Prohibited claim '{req}' requested under EvidenceLevel {evidence_level.name}."
                    )
            if violations:
                policy_status = BenchmarkStatus.CONTRACT_VIOLATION

        return ClaimAuthorization(
            claims_allowed=allowed_list,
            claims_blocked=blocked_list,
            policy_status=policy_status,
            violation_reasons=violations,
        )

    @classmethod
    def enforce_claim_emission(
        cls,
        evidence_level: EvidenceLevel,
        claim_key: str,
        **kwargs: Any,
    ) -> None:
        """Enforces that a specific claim emission is strictly authorized. Raises ContractViolationError if blocked."""
        auth = cls.audit_claim_authorization(evidence_level, requested_claims=[claim_key], **kwargs)
        if auth.policy_status == BenchmarkStatus.CONTRACT_VIOLATION:
            raise ContractViolationError(
                f"Claim Policy Violation: Claim '{claim_key}' is strictly blocked for {evidence_level.name}. "
                f"Reasons: {auth.violation_reasons}"
            )
