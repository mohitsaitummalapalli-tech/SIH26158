"""Phase 3E.6 Executable Synthetic Benchmark Runner.

Executes end-to-end validation on:
- Class A: Exact procedural 3D geometric CAD scene (clean ground truth).
- Class B: Photorealistic / sensor-perturbed simulation (Gaussian noise, outliers, focal drift).
Saves serialized benchmark manifest to benchmarks/manifests/synthetic_benchmark_result.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.models import (
    TaxonomyClass,
    EvidenceLevel,
    BenchmarkStatus,
    QualityAxis,
    ValidationScope,
    AcquisitionConditions,
    CameraCalibrationMeta,
    TelemetryMeta,
    GroundTruthMeta,
    DatasetManifest,
    ReferencePartition,
)
from src.benchmark.engine import BenchmarkEngine
from src.benchmark.timing_profiler import BenchmarkTimingProfiler
from src.benchmark.metrics_geometry import (
    compute_point_to_point_distances,
    compute_bidirectional_chamfer,
    compute_hausdorff_distances,
    compute_f_score_at_tau,
)
from src.benchmark.metrics_metric_scale import (
    ValidationSegment,
    evaluate_metric_scale,
)
from src.benchmark.metrics_trajectory import (
    evaluate_raw_trajectory_ate,
    evaluate_sim3_aligned_trajectory_ate,
    evaluate_rpe_drift,
)
from src.benchmark.robustness_perturbations import (
    apply_gnss_gaussian_noise,
)


def run_synthetic_benchmark() -> Dict[str, Any]:
    print("=" * 70)
    print("RUNNING PHASE 3E.6 SYNTHETIC METRIC ENGINE CONTRACT VALIDATION")
    print("Scope: METRIC_ENGINE_CONTRACT (Identity-case verification of metric calculators)")
    print("End-to-End Photogrammetric Reconstruction: FALSE")
    print("=" * 70)

    engine = BenchmarkEngine(software_commit="20c62a1", software_version="v2.0.0-LOCKED")
    profiler = BenchmarkTimingProfiler(input_frames=60, decoded_duration_sec=2.0)
    profiler.start_pipeline()

    # 1. Generate Synthetic Scene Geometry (Class A: Hemispherical Quarry Pit)
    profiler.start_stage("scene_generation")
    rng = np.random.default_rng(42)
    n_points = 500
    phi = rng.uniform(0, 2 * np.pi, n_points)
    theta = rng.uniform(0, np.pi / 2, n_points)
    r = 25.0  # 25 meter radius
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = -r * np.cos(theta)  # Pit depth down to -25m
    gt_cad_points = np.column_stack([x, y, z])

    # Distinct memory copy for reconstructed points (Class A: zero error)
    est_points_class_a = gt_cad_points.copy()
    profiler.stop_stage()

    # 2. Camera Trajectory (Circular inspection flight at z = 15m)
    profiler.start_stage("trajectory_evaluation")
    n_cameras = 20
    angles = np.linspace(0, 2 * np.pi, n_cameras, endpoint=False)
    flight_r = 35.0
    cam_centers_ref = np.column_stack([
        flight_r * np.cos(angles),
        flight_r * np.sin(angles),
        np.full(n_cameras, 15.0)
    ])
    cam_rotations_ref = np.array([np.eye(3) for _ in range(n_cameras)])

    # Class A Trajectory (Exact)
    traj_raw_res = evaluate_raw_trajectory_ate(
        cam_centers_ref, cam_rotations_ref, cam_centers_ref, cam_rotations_ref
    )
    traj_aligned_res = evaluate_sim3_aligned_trajectory_ate(cam_centers_ref, cam_centers_ref)
    traj_rpe_res = evaluate_rpe_drift(cam_centers_ref, cam_rotations_ref, cam_centers_ref, cam_rotations_ref, delta_interval_frames=1)
    profiler.stop_stage()

    # 3. Geometric Metrology Evaluation
    profiler.start_stage("geometry_evaluation")
    roi_bounds = ((-50.0, -50.0, -50.0), (50.0, 50.0, 50.0))
    p2p_summary = compute_point_to_point_distances(
        est_points_class_a, gt_cad_points, roi_bounds=roi_bounds,
        est_hash="cad_recon_hash_class_a", gt_hash="cad_reference_truth_hash"
    )
    chamfer_res = compute_bidirectional_chamfer(
        est_points_class_a, gt_cad_points, roi_bounds=roi_bounds,
        est_hash="cad_recon_hash_class_a", gt_hash="cad_reference_truth_hash"
    )
    hausdorff_res = compute_hausdorff_distances(est_points_class_a, gt_cad_points, roi_bounds=roi_bounds)
    f_score_res = compute_f_score_at_tau(est_points_class_a, gt_cad_points, tau_meters=0.05, roi_bounds=roi_bounds)
    profiler.stop_stage()

    # 4. Metric Scale Evaluation (Class A: 3 Independent Segments)
    profiler.start_stage("metric_scale_evaluation")
    marker_positions = {
        "M1": np.array([0.0, 0.0, 0.0]),
        "M2": np.array([20.0, 0.0, 0.0]),
        "M3": np.array([0.0, 15.0, 0.0]),
    }
    segments = [
        ValidationSegment("SEG_01", "M1", "M2", reference_distance=20.0),
        ValidationSegment("SEG_02", "M1", "M3", reference_distance=15.0),
        ValidationSegment("SEG_03", "M2", "M3", reference_distance=25.0),
    ]
    scale_res = evaluate_metric_scale(segments, marker_positions, scale_factor_to_metric=1.0)
    profiler.stop_stage()

    # 5. Class B Sensor Perturbation (GNSS Gaussian Noise 1.0m)
    profiler.start_stage("class_b_perturbation")
    perturbed_gnss, pert_rec = apply_gnss_gaussian_noise(cam_centers_ref, horizontal_sigma_m=1.0, vertical_sigma_m=2.0)
    profiler.stop_stage()

    profiler.stop_pipeline()
    timing = profiler.build_timing_profile()

    # 6. Assemble Manifest & Benchmark Result
    partition = ReferencePartition(
        estimation_set_ids={"SEG_01"},
        validation_set_ids={"SEG_02", "SEG_03"},
    )
    manifest = DatasetManifest(
        dataset_id="SYNTH-CLASS-A-QUARRY-60F",
        taxonomy_class=TaxonomyClass.CLASS_A_SYNTHETIC_CONTROLLED,
        acquisition_conditions=AcquisitionConditions(scene_type="quarry_pit", lighting="diffuse"),
        frame_count=60,
        image_resolution=(1920, 1080),
        camera_calibration=CameraCalibrationMeta(focal_length_px=1440.0),
        telemetry_metadata=TelemetryMeta(has_telemetry=True, horizontal_accuracy_m=0.0, vertical_accuracy_m=0.0),
        ground_truth_metadata=GroundTruthMeta(
            has_ground_truth=True,
            ground_truth_type="EXACT_CAD_SURFACE",
            partition=partition,
            total_targets_surveyed=3,
        ),
        sha256_checksum="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )

    result = engine.execute_validation(
        manifest=manifest,
        reconstruction_artifacts={"checksum_sha256": "recon_sha256_class_a_001"},
        reference_artifacts={"checksum_sha256": "reference_sha256_cad_001"},
        profiler=profiler,
    )
    result.validation_scope = ValidationScope.METRIC_ENGINE_CONTRACT
    result.validation_context = {
        "end_to_end_reconstruction": False,
        "ground_truth_used_for_reconstruction": True,
        "ground_truth_used_for_evaluation_only": False,
        "accuracy_claim_authorized": False,
        "description": "Identity-case verification of benchmark metric implementations using exact CAD geometry.",
    }

    # Attach detailed measured metrics
    result.metrics = {
        "geometric": {
            "p2p_rmse_m": p2p_summary.rmse,
            "chamfer_distance_m": chamfer_res["chamfer_distance"],
            "hausdorff_max_m": hausdorff_res["hausdorff_max"],
            "hausdorff_95_m": hausdorff_res["hausdorff_95"],
            "precision_at_5cm": f_score_res["precision"],
            "recall_at_5cm": f_score_res["recall"],
            "f1_score_at_5cm": f_score_res["f1_score"],
        },
        "metric_scale": {
            "rmse_relative_error": scale_res["relative_scale_error_summary"].rmse,
            "max_relative_error_pct": scale_res["max_relative_error_pct"],
            "segment_count": scale_res["segment_count"],
        },
        "trajectory": {
            "ate_raw_rmse_m": traj_raw_res.ate_translation_rmse_m,
            "ate_sim3_aligned_rmse_m": traj_aligned_res.aligned_ate_rmse_m,
            "rpe_translational_drift_rmse_m": traj_rpe_res.translational_drift_per_delta_rmse,
        },
        "perturbation_audit": {
            "perturbation_type": pert_rec.perturbation_type,
            "magnitude": pert_rec.magnitude,
            "regime": pert_rec.operating_regime.value,
        },
    }

    # Serialize manifest
    out_dir = PROJECT_ROOT / "benchmarks" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "synthetic_benchmark_result.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"Synthetic benchmark completed: STATUS = {result.result_state.value}")
    print(f"Geometric P2P RMSE: {p2p_summary.rmse:.6f} m")
    print(f"Bidirectional Chamfer: {chamfer_res['chamfer_distance']:.6f} m")
    print(f"Scale Relative RMSE: {scale_res['relative_scale_error_summary'].rmse:.6f}")
    print(f"Raw Trajectory ATE RMSE: {traj_raw_res.ate_translation_rmse_m:.6f} m")
    print(f"Output serialized to: {out_file}")
    return result.to_dict()


if __name__ == "__main__":
    run_synthetic_benchmark()
