"""Phase 3E.6 Geospatial Validation & Checkpoint Residual Analysis.

Evaluates hold-out ground checkpoints (CKPs) against surveyed ENU/WGS84 references.
Strictly prohibits validation checkpoint participation in estimation or alignment fitting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.benchmark.models import (
    StatisticalSummary,
    ContractViolationError,
    ReferencePartition,
)
from src.benchmark.claim_policy import ClaimPolicyEngine
from src.benchmark.metrics_geometry import compute_statistical_summary


@dataclass(frozen=True)
class CheckpointReference:
    """A surveyed ground checkpoint (CKP) strictly withheld from estimation."""
    target_id: str
    east_m: float
    north_m: float
    up_m: float
    crs: str = "EPSG:32643"  # UTM or local tangent ENU


@dataclass(frozen=True)
class CheckpointEvaluationResult:
    """Detailed residual summary across hold-out validation checkpoints."""
    target_count: int
    rmse_east: float
    rmse_north: float
    rmse_up: float
    rmse_horizontal: float
    rmse_3d: float
    mae_3d: float
    median_3d: float
    p95_3d: float
    maximum_3d: float
    residuals_per_target: Dict[str, Dict[str, float]]
    statistical_summary_horizontal: StatisticalSummary
    statistical_summary_3d: StatisticalSummary


def evaluate_geospatial_checkpoints(
    checkpoints: List[CheckpointReference],
    estimated_enu_coordinates: Dict[str, Tuple[float, float, float]],
    partition: Optional[ReferencePartition] = None,
    pre_alignment_positions: Optional[Dict[str, np.ndarray]] = None,
    post_alignment_positions: Optional[Dict[str, np.ndarray]] = None,
) -> CheckpointEvaluationResult:
    """Evaluates independent hold-out checkpoint residuals in topocentric ENU coordinates.
    
    Enforces anti-leakage invariants:
    1. Checkpoints must belong exclusively to VALIDATION_REFERENCE_SET.
    2. Checkpoints must NOT have been modified by ICP or post-fit alignment.
    """
    if len(checkpoints) == 0:
        raise ValueError("Cannot evaluate geospatial accuracy: Zero validation checkpoints provided.")

    # 1. Anti-leakage partition audit
    ckp_ids = {ckp.target_id for ckp in checkpoints}
    if partition is not None:
        partition.validate_disjointness()
        leaked_in_est = ckp_ids.intersection(partition.estimation_set_ids)
        if leaked_in_est:
            raise ContractViolationError(
                f"Validation reference contamination (MUT-03): Checkpoints found in estimation set: {leaked_in_est}"
            )
        leaked_in_cal = ckp_ids.intersection(partition.calibration_set_ids)
        if leaked_in_cal:
            raise ContractViolationError(
                f"Validation reference contamination: Checkpoints found in calibration set: {leaked_in_cal}"
            )

    # 2. Anti-alignment audit (MUT-02)
    if pre_alignment_positions is not None and post_alignment_positions is not None:
        for ckp in checkpoints:
            tid = ckp.target_id
            if tid in pre_alignment_positions and tid in post_alignment_positions:
                ClaimPolicyEngine.verify_no_validation_alignment(
                    pre_alignment_positions[tid],
                    post_alignment_positions[tid],
                )

    # 3. Residual calculation
    delta_e_list = []
    delta_n_list = []
    delta_u_list = []
    horiz_errors = []
    errors_3d = []
    target_residuals = {}

    for ckp in checkpoints:
        tid = ckp.target_id
        if tid not in estimated_enu_coordinates:
            raise KeyError(f"Missing estimated coordinate for validation checkpoint: {tid}")

        est_e, est_n, est_u = estimated_enu_coordinates[tid]

        de = float(est_e - ckp.east_m)
        dn = float(est_n - ckp.north_m)
        du = float(est_u - ckp.up_m)

        err_horiz = float(math.sqrt(de ** 2 + dn ** 2))
        err_3d = float(math.sqrt(de ** 2 + dn ** 2 + du ** 2))

        delta_e_list.append(de)
        delta_n_list.append(dn)
        delta_u_list.append(du)
        horiz_errors.append(err_horiz)
        errors_3d.append(err_3d)

        target_residuals[tid] = {
            "delta_east_m": de,
            "delta_north_m": dn,
            "delta_up_m": du,
            "error_horizontal_m": err_horiz,
            "error_3d_m": err_3d,
        }

    k = len(checkpoints)
    arr_de = np.array(delta_e_list, dtype=np.float64)
    arr_dn = np.array(delta_n_list, dtype=np.float64)
    arr_du = np.array(delta_u_list, dtype=np.float64)
    arr_horiz = np.array(horiz_errors, dtype=np.float64)
    arr_3d = np.array(errors_3d, dtype=np.float64)

    rmse_e = float(np.sqrt(np.mean(arr_de ** 2)))
    rmse_n = float(np.sqrt(np.mean(arr_dn ** 2)))
    rmse_u = float(np.sqrt(np.mean(arr_du ** 2)))
    rmse_horiz = float(np.sqrt(rmse_e ** 2 + rmse_n ** 2))
    rmse_3d = float(np.sqrt(rmse_e ** 2 + rmse_n ** 2 + rmse_u ** 2))

    stat_horiz = compute_statistical_summary(arr_horiz, unit="meters")
    stat_3d = compute_statistical_summary(arr_3d, unit="meters")

    return CheckpointEvaluationResult(
        target_count=k,
        rmse_east=rmse_e,
        rmse_north=rmse_n,
        rmse_up=rmse_u,
        rmse_horizontal=rmse_horiz,
        rmse_3d=rmse_3d,
        mae_3d=stat_3d.mae,
        median_3d=stat_3d.median,
        p95_3d=stat_3d.p95,
        maximum_3d=stat_3d.maximum,
        residuals_per_target=target_residuals,
        statistical_summary_horizontal=stat_horiz,
        statistical_summary_3d=stat_3d,
    )
