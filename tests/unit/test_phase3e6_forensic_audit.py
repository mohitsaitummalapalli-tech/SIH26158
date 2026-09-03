"""Phase 3E.6 Forensic Audit & Adversarial Mutation Test Suite.

Contains explicit adversarial mutation tests targeting potential cheats,
mathematical errors, and reporting integrity breaches:
- MUT-01: Reconstruction used as GT (self-evaluation cheat)
- MUT-02: ICP hides error on hold-out checkpoints
- MUT-03: Validation reference contamination (GCP/CKP leakage)
- MUT-04: GNSS residual claimed as independent metric accuracy
- MUT-05: Selective failure dropping to inflate benchmark pass rates
- MUT-06: Unidirectional Chamfer reporting cheat
- MUT-07: ENU axis swap (East-North confusion)
- MUT-08: Inverted scale ratio denominator (|d_est - d_ref| / d_est)
- MUT-09: Trajectory alignment removes error without documenting Sim(3) scale & disclaimer
- MUT-10: Heuristic score claimed as calibrated Gaussian probability
- MUT-11: Hidden synthetic truth leaked to reconstruction algorithm
- MUT-12: Chronological PTS order violation
- MUT-13: Unjustified reproducibility failure when within R1 tolerance
- MUT-14: Unsupported occlusion classification without optical rays
- MUT-15: Completeness claim without bounded reference ROI
- MUT-16: Radiometric claim without certified calibration
- MUT-17: Universal PSNR/SSIM threshold assertion
- MUT-18: Covariance / unit scaling inversion
"""

import math
import numpy as np
import pytest

from src.benchmark.models import (
    ContractViolationError,
    EvidenceLevel,
    BenchmarkStatus,
    VisibilityState,
    ReproducibilityLevel,
    ReferencePartition,
)
from src.benchmark.claim_policy import ClaimPolicyEngine
from src.benchmark.metrics_geometry import (
    compute_point_to_point_distances,
    compute_bidirectional_chamfer,
)
from src.benchmark.metrics_metric_scale import (
    compute_relative_scale_error,
)
from src.benchmark.metrics_geospatial import (
    CheckpointReference,
    evaluate_geospatial_checkpoints,
)
from src.benchmark.metrics_trajectory import (
    evaluate_sim3_aligned_trajectory_ate,
    DISCLAIMER_SIM3_ALIGNMENT,
)
from src.benchmark.metrics_texture import (
    TextureDiagnosticMetadata,
    evaluate_texture_diagnostics,
)
from src.benchmark.metrics_completeness import (
    classify_visibility_evidence,
    evaluate_roi_completeness,
)
from src.benchmark.metrics_uncertainty import (
    evaluate_probabilistic_coverage,
    transform_spatial_covariance,
)
from src.benchmark.engine import BenchmarkEngine
from src.benchmark.reproducibility import verify_reproducibility_level


def test_mut_01_reconstruction_used_as_gt():
    """MUT-01: Self-evaluation cheat must be detected and rejected."""
    cloud = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(ContractViolationError, match="Self-evaluation cheat detected"):
        ClaimPolicyEngine.verify_no_self_evaluation(
            reconstruction_hash="same_hash_123",
            reference_hash="same_hash_123",
            reconstruction_cloud=cloud,
            reference_cloud=cloud,
        )


def test_mut_02_icp_hides_error():
    """MUT-02: Modifying hold-out checkpoints via alignment before evaluation must fail."""
    pre_align = np.array([10.0, 20.0, 5.0])
    post_align = np.array([10.05, 20.02, 5.01])  # Shifted via ICP
    with pytest.raises(ContractViolationError, match="MUT-02"):
        ClaimPolicyEngine.verify_no_validation_alignment(pre_align, post_align)


def test_mut_03_validation_contamination():
    """MUT-03: Leaking validation checkpoints into estimation set must fail."""
    partition = ReferencePartition(
        estimation_set_ids={"CKP_LEAKED", "GCP1"},
        validation_set_ids={"CKP_LEAKED", "CKP2"},
    )
    with pytest.raises(ContractViolationError, match="Data leakage detected"):
        partition.validate_disjointness()


def test_mut_04_gnss_residual_claimed_as_accuracy():
    """MUT-04: Claiming GNSS training residual as independent surveyed accuracy must be blocked."""
    auth = ClaimPolicyEngine.audit_claim_authorization(EvidenceLevel.LEVEL_1_TELEMETRY_ONLY)
    assert "gnss_fitting_residual" in auth.claims_allowed
    assert "horizontal_checkpoint_rmse" in auth.claims_blocked
    with pytest.raises(ContractViolationError, match="strictly blocked"):
        ClaimPolicyEngine.enforce_claim_emission(
            EvidenceLevel.LEVEL_1_TELEMETRY_ONLY,
            "horizontal_checkpoint_rmse",
        )


def test_mut_05_selective_failure_dropping():
    """MUT-05: Manifest execution counter must detect dropped failed runs via production audit."""
    executed_runs = ["RUN_01_PASS", "RUN_02_FAIL", "RUN_03_PASS"]
    reported_runs = ["RUN_01_PASS", "RUN_03_PASS"]  # Silently dropped RUN_02_FAIL
    
    # 1. Dropped run triggers ContractViolationError
    with pytest.raises(ContractViolationError, match="Selective reporting cheat detected"):
        BenchmarkEngine.audit_execution_completeness(executed_runs, reported_runs)

    # 2. Duplicate executed or reported runs triggers ContractViolationError
    with pytest.raises(ContractViolationError, match="Duplicate run IDs"):
        BenchmarkEngine.audit_execution_completeness(["R1", "R1"], ["R1"])

    # 3. Exact matching execution passes
    BenchmarkEngine.audit_execution_completeness(
        ["RUN_01_PASS", "RUN_02_FAIL"],
        ["RUN_01_PASS", "RUN_02_FAIL"],
    )


def test_mut_06_unidirectional_chamfer():
    """MUT-06: Bidirectional Chamfer must evaluate both forward and backward distances."""
    pts_a = np.array([[0.0, 0.0, 0.0]])
    pts_b = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])  # b has a distant point
    res = compute_bidirectional_chamfer(pts_a, pts_b, est_hash="h1", gt_hash="h2")
    # Forward distance (a -> b) is 0.0! Backward distance (b -> a) is 5.0!
    assert res["forward_summary"].mae == 0.0
    assert res["backward_summary"].mae == 5.0
    # True bidirectional Chamfer must be non-zero (2.5m)
    assert res["chamfer_distance"] > 0.0


def test_mut_07_enu_axis_swap():
    """MUT-07: Swapping East and North coordinate axes in residual vector must be caught."""
    true_enu = (100.0, 200.0, 50.0)
    swapped_est = (200.0, 100.0, 50.0)  # East and North swapped
    ckp = CheckpointReference("CKP1", east_m=true_enu[0], north_m=true_enu[1], up_m=true_enu[2])
    res = evaluate_geospatial_checkpoints([ckp], {"CKP1": swapped_est})
    assert res.residuals_per_target["CKP1"]["delta_east_m"] == 100.0
    assert res.residuals_per_target["CKP1"]["delta_north_m"] == -100.0
    assert res.rmse_horizontal > 140.0  # Massive error due to axis swap


def test_mut_08_inverted_scale_ratio():
    """MUT-08: Inverting scale ratio formula (|d_est - d_ref| / d_est) must be rejected."""
    # When d_est = 2.0 and d_ref = 1.0:
    # Correct relative error = |2 - 1| / 1 = 1.0 (100%)
    # Inverted formula = |2 - 1| / 2 = 0.5 (50% - cheat!)
    d_est, d_ref = 2.0, 1.0
    err = compute_relative_scale_error(d_est, d_ref)
    assert math.isclose(err, 1.0, abs_tol=1e-5)
    assert not math.isclose(err, 0.5, abs_tol=1e-5)


def test_mut_09_undocumented_sim3_error_removal():
    """MUT-09: Suppressing Sim(3) alignment disclaimer must trigger ContractViolationError."""
    c_est = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    c_ref = c_est * 5.0
    with pytest.raises(ContractViolationError, match="MUT-09"):
        evaluate_sim3_aligned_trajectory_ate(c_est, c_ref, suppress_disclaimer=True)


def test_mut_10_heuristic_score_claimed_as_probability():
    """MUT-10: Claiming heuristic scores as Gaussian probability without declared model must fail."""
    sigmas = np.ones(50)
    errors = np.ones(50)
    with pytest.raises(ContractViolationError, match="MUT-10"):
        evaluate_probabilistic_coverage(sigmas, errors, declared_probabilistic_model=None)


def test_mut_11_hidden_synthetic_truth_leakage():
    """MUT-11: Accessing hidden evaluation ground truth in reconstruction pipeline triggers violation via production audit."""
    public_params = {"bounds": [-10, 10], "focal": 1000.0, "artifact_id": "ART_PUBLIC_001"}
    hidden_truth = {"true_depth_maps": np.zeros((10, 10)), "artifact_id": "ART_HIDDEN_TRUTH_001"}
    
    # 1. Privileged key presence triggers ContractViolationError
    with pytest.raises(ContractViolationError, match="Hidden synthetic truth leakage detected"):
        BenchmarkEngine.verify_input_isolation(
            reconstruction_inputs={**public_params, **hidden_truth},
            hidden_evaluation_artifacts=hidden_truth,
        )

    # 2. Checksum overlap triggers ContractViolationError
    with pytest.raises(ContractViolationError, match="Checksum overlap"):
        BenchmarkEngine.verify_input_isolation(
            reconstruction_inputs={"input_mesh": {"checksum_sha256": "leaked_truth_hash_123"}},
            hidden_evaluation_artifacts={"evaluation_gt": {"checksum_sha256": "leaked_truth_hash_123"}},
        )

    # 3. Clean isolated inputs pass verification
    BenchmarkEngine.verify_input_isolation(
        reconstruction_inputs={"camera_frames": {"checksum_sha256": "video_frame_hash_001"}},
        hidden_evaluation_artifacts={"evaluation_gt": {"checksum_sha256": "ground_truth_cad_hash_999"}},
    )


def test_mut_12_chronological_order_violation():
    """MUT-12: Non-chronological shuffled video frames must be rejected by production temporal order validator."""
    shuffled_pts = [0.0, 0.033, 0.066, 0.022]  # Non-monotonic PTS

    # 1. Shuffled sequence triggers ContractViolationError
    with pytest.raises(ContractViolationError, match="Chronological PTS ordering violation"):
        BenchmarkEngine.verify_temporal_order(shuffled_pts)

    # 2. Duplicate sequence without permission triggers ContractViolationError
    with pytest.raises(ContractViolationError, match="non-increasing"):
        BenchmarkEngine.verify_temporal_order([0.0, 0.033, 0.033, 0.066], allow_duplicates=False)

    # 3. Monotonic canonical PTS sequence passes
    BenchmarkEngine.verify_temporal_order([0.0, 0.033, 0.066, 0.099])


def test_mut_13_unjustified_reproducibility_failure():
    """MUT-13: Production reproducibility validator must accept runs within R1 tolerance and reject out-of-tolerance runs."""
    arr1 = np.array([1.00000001, 2.00000002])
    arr2 = np.array([1.00000002, 2.00000001])
    # Binary hashes differ:
    assert arr1.tobytes() != arr2.tobytes()

    # 1. Within R1 tolerance (1e-5): Production validator passes
    assert verify_reproducibility_level(arr1, arr2, ReproducibilityLevel.R1_NUMERICAL, tolerance=1e-5) is True

    # 2. Out of tolerance: Production validator catches violation
    out_of_tol_arr = np.array([1.01, 2.02])
    with pytest.raises(ContractViolationError, match="Level R1 numerical violation"):
        verify_reproducibility_level(arr1, out_of_tol_arr, ReproducibilityLevel.R1_NUMERICAL, tolerance=1e-5)


def test_mut_14_unsupported_occlusion_classification():
    """MUT-14: Classifying missing geometry as PHYSICALLY_OCCLUDED without ray evidence must fail."""
    state = classify_visibility_evidence(
        has_optical_ray_intersection=False,
        ray_intersection_angle_deg=0.0,
        in_camera_frustum=True,
        ray_hits_foreground=False,
        has_ray_tracing_evidence=False,
        is_reconstructed=False,
    )
    # Must be UNDETERMINED, never PHYSICALLY_OCCLUDED!
    assert state == VisibilityState.UNDETERMINED
    assert state != VisibilityState.PHYSICALLY_OCCLUDED


def test_mut_15_completeness_without_roi():
    """MUT-15: Completeness claim without explicit reference ROI must trigger ContractViolationError."""
    pts_a = np.array([[0.0, 0.0, 0.0]])
    pts_b = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ContractViolationError, match="MUT-15"):
        evaluate_roi_completeness(pts_a, pts_b, tau_meters=0.1, roi_bounds=None)


def test_mut_16_radiometric_claim_without_calibration():
    """MUT-16: Radiometric color accuracy claim without calibration must trigger ContractViolationError."""
    img = np.ones((10, 10, 3), dtype=np.float64)
    meta = TextureDiagnosticMetadata(image_resolution=(10, 10), radiometric_calibration_certified=False)
    with pytest.raises(ContractViolationError, match="MUT-16"):
        evaluate_texture_diagnostics(img, img, metadata=meta, claim_colorimetric_accuracy=True)


def test_mut_17_universal_psnr_ssim_threshold():
    """MUT-17: Asserting universal PSNR/SSIM acceptance gate must fail."""
    img = np.ones((10, 10), dtype=np.float64)
    meta = TextureDiagnosticMetadata(image_resolution=(10, 10))
    with pytest.raises(ContractViolationError, match="MUT-17"):
        evaluate_texture_diagnostics(img, img, metadata=meta, assert_universal_gate=True)


def test_mut_18_covariance_unit_scaling_error():
    """MUT-18: Production covariance spatial unit scaling verifies quadratic s^2 law against independent oracle."""
    cov_m = np.array([[4.0, 0.5], [0.5, 9.0]], dtype=np.float64)  # Covariance in m^2
    s = 0.001  # Convert metres to kilometres (s = 1e-3)

    # 1. Independent mathematical oracle computation: Sigma_expected = (s^2) * Sigma
    # Explicitly computed without calling the production scaling logic
    expected_cov_km = np.array([
        [4.0 * (0.001 ** 2), 0.5 * (0.001 ** 2)],
        [0.5 * (0.001 ** 2), 9.0 * (0.001 ** 2)],
    ], dtype=np.float64)

    # 2. Execute production function
    actual_cov_km = transform_spatial_covariance(cov_m, scale_factor_s=s, verify_round_trip=True)

    # 3. Assert production output numerically matches independent mathematical oracle
    assert np.allclose(actual_cov_km, expected_cov_km, atol=1e-15, rtol=1e-12)

    # 4. Explicit separate inverse round-trip verification: (1/s)^2 * Sigma' == Sigma
    recovered_cov_m = transform_spatial_covariance(actual_cov_km, scale_factor_s=1.0 / s, verify_round_trip=False)
    assert np.allclose(recovered_cov_m, cov_m, atol=1e-12, rtol=1e-9)

    # 5. Adversarial Mutation: An incorrect implementation using linear scaling (s * Sigma)
    # MUST be detected and rejected by the independent oracle comparison
    mutated_linear_cov_km = s * cov_m  # Linear cheat (e.g. s * Sigma instead of s^2 * Sigma)
    # Linear scaling produces 4e-3 instead of 4e-6 (1000x error!)
    diff_vs_oracle = np.max(np.abs(mutated_linear_cov_km - expected_cov_km))
    assert diff_vs_oracle > 1e-4, "Adversarial linear mutation was not detected by oracle!"
    assert not np.allclose(actual_cov_km, mutated_linear_cov_km, atol=1e-8)

    # 6. Production validation rejects non-positive or degenerate scale factors
    with pytest.raises(ContractViolationError, match="MUT-18"):
        transform_spatial_covariance(cov_m, scale_factor_s=-0.001)

    with pytest.raises(ContractViolationError, match="MUT-18"):
        transform_spatial_covariance(cov_m, scale_factor_s=0.0)
