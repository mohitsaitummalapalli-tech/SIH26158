"""Robust 7-DoF Sim(3) estimation using deterministic 3-point RANSAC and Huber-weighted IRLS.

Stage 1: Deterministic 3-Point RANSAC
- Rejects degenerate triplets via dimensionless isoperimetric quotient Q = 4*pi*A / P^2 < 1e-4.
- Evaluates geometry in both reconstruction gauge and target ENU.
- Deterministic hypothesis tie-breaking:
    1. Higher inlier count
    2. Lower sum of squared Mahalanobis residuals on inliers
    3. Lexicographical triplet indices (i, j, k)

Stage 2: Iteratively Reweighted Least Squares (IRLS) with Huber Loss
- Tuning parameter k_Huber = 1.345 (95% Gaussian efficiency).
- Mahalanobis residual: d_i = sqrt(r_i^T * Sigma_i^-1 * r_i).
- Huber weights: w_i = 1.0 if d_i <= k_Huber else k_Huber / d_i.
- Strictly preserves Sim(3); affine and nonlinear warps are forbidden.
"""

from dataclasses import dataclass, field
import itertools
import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from src.geospatial.sim3 import Sim3, solve_sim3_umeyama
from src.geospatial.lever_arm import LeverArm
from src.geospatial.telemetry_observation import TelemetryObservation, ObservationClassification


@dataclass
class EstimationDiagnostics:
    """Detailed diagnostics of robust Sim(3) estimation."""
    iterations: int
    converged: bool
    ransac_hypotheses_evaluated: int
    degenerate_triplets_rejected: int
    winning_triplet: Optional[Tuple[int, int, int]]
    initial_inlier_count: int
    final_inlier_count: int
    inlier_ratio: float
    horizontal_rmse_m: float
    vertical_rmse_m: float
    total_3d_rmse_m: float
    max_residual_m: float
    mean_attitude_residual_deg: Optional[float] = None
    final_huber_weights: Optional[np.ndarray] = None


@dataclass
class RobustSim3Result:
    """Outcome of robust geospatial similarity estimation."""
    success: bool
    sim3: Optional[Sim3]
    inlier_indices: List[int]
    rejected_indices: List[int]
    residuals_m: np.ndarray
    mahalanobis_distances: np.ndarray
    diagnostics: EstimationDiagnostics
    final_huber_weights: Optional[np.ndarray] = None
    failure_reason: Optional[str] = None


def compute_min_edge(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute minimum pairwise Euclidean edge length of a 3-point triangle."""
    d01 = float(np.linalg.norm(p1 - p0))
    d12 = float(np.linalg.norm(p2 - p1))
    d20 = float(np.linalg.norm(p0 - p2))
    return min(d01, d12, d20)


def compute_isoperimetric_quotient(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute dimensionless isoperimetric quotient Q = 4 * pi * Area / Perimeter^2.
    
    Q = 1.0 for equilateral triangle, Q -> 0 for degenerate/collinear triplets.
    """
    v01 = p1 - p0
    v02 = p2 - p0
    cross = np.cross(v01, v02)
    area = 0.5 * float(np.linalg.norm(cross))

    d01 = float(np.linalg.norm(v01))
    d12 = float(np.linalg.norm(p2 - p1))
    d20 = float(np.linalg.norm(v02))
    perimeter = d01 + d12 + d20

    if perimeter <= 0.0 or math.isnan(perimeter):
        return 0.0

    return (4.0 * math.pi * area) / (perimeter * perimeter)


class RobustSim3Estimator:
    """Two-stage robust estimator combining minimal RANSAC and M-estimator IRLS."""

    def __init__(
        self,
        tau_tri_degen: float = 1e-4,
        tau_rel_edge: float = 1e-4,
        tau_inlier_mahalanobis: float = 3.0,
        k_huber: float = 1.345,
        tau_conv_sim3: float = 1e-6,
        max_irls_iterations: int = 50,
        max_ransac_samples: int = 500,
        tau_min_baseline_m: float = 10.0,
    ) -> None:
        self.tau_tri_degen = tau_tri_degen
        self.tau_rel_edge = tau_rel_edge
        self.tau_inlier_mahalanobis = tau_inlier_mahalanobis
        self.k_huber = k_huber
        self.tau_conv_sim3 = tau_conv_sim3
        self.max_irls_iterations = max_irls_iterations
        self.max_ransac_samples = max_ransac_samples
        self.tau_min_baseline_m = tau_min_baseline_m

    def estimate(
        self,
        observations: List[TelemetryObservation],
        lever_arm: LeverArm,
    ) -> RobustSim3Result:
        """Execute robust Sim(3) estimation over given telemetry observations.
        
        Args:
            observations: List of TelemetryObservation pairs.
            lever_arm: Physical airframe lever-arm definition.
            
        Returns:
            RobustSim3Result with estimated transform, inliers, and audit diagnostics.
        """
        n = len(observations)
        if n < 3:
            return RobustSim3Result(
                success=False,
                sim3=None,
                inlier_indices=[],
                rejected_indices=list(range(n)),
                residuals_m=np.zeros((n, 3)),
                mahalanobis_distances=np.zeros(n),
                diagnostics=EstimationDiagnostics(
                    iterations=0,
                    converged=False,
                    ransac_hypotheses_evaluated=0,
                    degenerate_triplets_rejected=0,
                    winning_triplet=None,
                    initial_inlier_count=0,
                    final_inlier_count=0,
                    inlier_ratio=0.0,
                    horizontal_rmse_m=0.0,
                    vertical_rmse_m=0.0,
                    total_3d_rmse_m=0.0,
                    max_residual_m=0.0,
                ),
                failure_reason="INSUFFICIENT_OBSERVATIONS: Fewer than 3 telemetry points provided",
            )

        # Extract points in source and destination frames
        c_recs = np.array([obs.c_rec for obs in observations], dtype=np.float64)
        target_cams = np.array([
            lever_arm.unapply_lever_arm(obs.z_gnss_enu, obs.r_body_to_enu)
            for obs in observations
        ], dtype=np.float64)

        # Precompute pairwise distance matrices in both frames
        diff_rec = c_recs[:, np.newaxis, :] - c_recs[np.newaxis, :, :]
        dists_rec = np.linalg.norm(diff_rec, axis=-1)

        diff_target = target_cams[:, np.newaxis, :] - target_cams[np.newaxis, :, :]
        dists_target = np.linalg.norm(diff_target, axis=-1)

        # Global diameter
        d_max_rec = float(np.max(dists_rec))
        d_max_target = float(np.max(dists_target))

        # --- Stage 1: Deterministic Minimal 3-Point RANSAC ---
        triplet_generator = itertools.combinations(range(n), 3)

        best_inliers: List[int] = []
        best_sum_sq_mahal = float("inf")
        best_triplet: Optional[Tuple[int, int, int]] = None
        best_sim3: Optional[Sim3] = None

        evaluated_hypotheses = 0
        rejected_degenerate_triplets = 0

        for sample_count, triplet in enumerate(triplet_generator, start=1):
            if sample_count > self.max_ransac_samples:
                break

            i, j, k = triplet

            # 1. Dimensionless edge ratio guard
            min_e_rec = compute_min_edge(c_recs[i], c_recs[j], c_recs[k])
            rho_rec = min_e_rec / d_max_rec if d_max_rec > 0 else 0.0
            if rho_rec < self.tau_rel_edge:
                rejected_degenerate_triplets += 1
                continue

            min_e_target = compute_min_edge(target_cams[i], target_cams[j], target_cams[k])
            rho_target = min_e_target / d_max_target if d_max_target > 0 else 0.0
            if rho_target < self.tau_rel_edge:
                rejected_degenerate_triplets += 1
                continue

            # 2. Dimensionless isoperimetric quotient guard (collinearity rejection)
            q_rec = compute_isoperimetric_quotient(c_recs[i], c_recs[j], c_recs[k])
            if q_rec < self.tau_tri_degen:
                rejected_degenerate_triplets += 1
                continue

            q_target = compute_isoperimetric_quotient(target_cams[i], target_cams[j], target_cams[k])
            if q_target < self.tau_tri_degen:
                rejected_degenerate_triplets += 1
                continue

            evaluated_hypotheses += 1

            # Solve minimal 3-point similarity
            try:
                candidate_sim3 = solve_sim3_umeyama(
                    src_points=c_recs[[i, j, k]],
                    dst_points=target_cams[[i, j, k]],
                )
            except Exception:
                continue

            # Evaluate consensus across all observations
            inliers: List[int] = []
            sum_sq = 0.0
            for idx, obs in enumerate(observations):
                d = obs.compute_mahalanobis(candidate_sim3, lever_arm)
                if d <= self.tau_inlier_mahalanobis:
                    inliers.append(idx)
                    sum_sq += d * d

            # Deterministic tie-breaking
            is_better = False
            if len(inliers) > len(best_inliers):
                is_better = True
            elif len(inliers) == len(best_inliers) and len(inliers) > 0:
                if sum_sq < best_sum_sq_mahal - 1e-9:
                    is_better = True

            if is_better:
                best_inliers = inliers
                best_sum_sq_mahal = sum_sq
                best_triplet = triplet
                best_sim3 = candidate_sim3

        if best_sim3 is None or len(best_inliers) < 3:
            # Check if failure was caused by collinear trajectory geometry
            diff_c = c_recs[:, None] - c_recs[None, :]
            c_span = float(np.max(np.linalg.norm(diff_c, axis=-1)))
            diff_z = target_cams[:, None] - target_cams[None, :]
            z_span = float(np.max(np.linalg.norm(diff_z, axis=-1)))

            c_cov = np.cov(c_recs, rowvar=False)
            evals_c = np.linalg.eigvalsh(c_cov)
            is_collinear_geom = (evals_c[2] > 1e-12) and (evals_c[1] / evals_c[2] < 1e-3)

            if is_collinear_geom and c_span > 1e-6 and z_span >= self.tau_min_baseline_m:
                # Dedicated collinear trajectory estimation path:
                # Along-track scale, direction, and translation are observable.
                # Axial rotation about trajectory line remains unobservable from positions alone.
                try:
                    collinear_sim3 = solve_sim3_umeyama(c_recs, target_cams)
                    u_geo = target_cams[-1] - target_cams[0]
                    norm_u = float(np.linalg.norm(u_geo))
                    null_dir = tuple((u_geo / norm_u).tolist()) if norm_u > 0 else (1.0, 0.0, 0.0)
                    collinear_sim3.axial_rotation_resolved = False
                    collinear_sim3.rotational_null_direction = null_dir

                    inliers = []
                    sum_sq = 0.0
                    for idx, obs in enumerate(observations):
                        d = obs.compute_mahalanobis(collinear_sim3, lever_arm)
                        if d <= self.tau_inlier_mahalanobis:
                            inliers.append(idx)
                            sum_sq += d * d

                    if len(inliers) >= 4:
                        best_sim3 = collinear_sim3
                        best_inliers = inliers
                        best_sum_sq_mahal = sum_sq
                except Exception:
                    pass

        if best_sim3 is None or len(best_inliers) < 3:
            return RobustSim3Result(
                success=False,
                sim3=None,
                inlier_indices=[],
                rejected_indices=list(range(n)),
                residuals_m=np.zeros((n, 3)),
                mahalanobis_distances=np.zeros(n),
                diagnostics=EstimationDiagnostics(
                    iterations=0,
                    converged=False,
                    ransac_hypotheses_evaluated=evaluated_hypotheses,
                    degenerate_triplets_rejected=rejected_degenerate_triplets,
                    winning_triplet=None,
                    initial_inlier_count=0,
                    final_inlier_count=0,
                    inlier_ratio=0.0,
                    horizontal_rmse_m=0.0,
                    vertical_rmse_m=0.0,
                    total_3d_rmse_m=0.0,
                    max_residual_m=0.0,
                ),
                failure_reason="NO_NON_DEGENERATE_SAMPLE_FOUND: RANSAC failed to find consensus hypothesis",
            )

        initial_inlier_count = len(best_inliers)

        # --- Stage 2: Iteratively Reweighted Least Squares (IRLS) with Huber Loss ---
        current_sim3 = best_sim3
        converged = False
        iteration = 0

        for iteration in range(1, self.max_irls_iterations + 1):
            # Compute Mahalanobis residuals and Huber weights on inliers
            inlier_sub = best_inliers
            d_vals = np.array([observations[idx].compute_mahalanobis(current_sim3, lever_arm) for idx in inlier_sub])

            # Huber weights
            weights = np.zeros(len(inlier_sub), dtype=np.float64)
            for m_idx, d in enumerate(d_vals):
                if d <= self.k_huber:
                    weights[m_idx] = 1.0
                else:
                    weights[m_idx] = self.k_huber / max(d, 1e-9)

            # Solve weighted Umeyama
            try:
                updated_sim3 = solve_sim3_umeyama(
                    src_points=c_recs[inlier_sub],
                    dst_points=target_cams[inlier_sub],
                    weights=weights,
                )
                updated_sim3.axial_rotation_resolved = current_sim3.axial_rotation_resolved
                updated_sim3.rotational_null_direction = current_sim3.rotational_null_direction
            except Exception:
                break

            # Parameter convergence check
            # delta_theta = [ln(s_new/s_old), ||R_new * R_old^T - I||_F, ||t_new - t_old||]
            delta_s = abs(math.log(updated_sim3.scale / current_sim3.scale))
            r_diff = np.linalg.norm(updated_sim3.rotation @ current_sim3.rotation.T - np.eye(3))
            t_diff = float(np.linalg.norm(updated_sim3.translation - current_sim3.translation))
            delta_norm = delta_s + r_diff + t_diff

            current_sim3 = updated_sim3
            if delta_norm < self.tau_conv_sim3:
                converged = True
                break

        # Final consensus inlier and residual evaluation
        final_inliers: List[int] = []
        final_rejected: List[int] = []
        residuals_m = np.zeros((n, 3), dtype=np.float64)
        mahal_distances = np.zeros(n, dtype=np.float64)

        for idx, obs in enumerate(observations):
            r = obs.compute_residual(current_sim3, lever_arm)
            d = obs.compute_mahalanobis(current_sim3, lever_arm)
            residuals_m[idx] = r
            mahal_distances[idx] = d

            if d <= self.tau_inlier_mahalanobis:
                final_inliers.append(idx)
                obs.classification = ObservationClassification.VALID
                obs.rejection_reason = None
            else:
                final_rejected.append(idx)
                obs.classification = ObservationClassification.OUTLIER_POSITION
                obs.rejection_reason = f"Residual Mahalanobis distance {d:.2f} exceeds threshold {self.tau_inlier_mahalanobis:.2f}"

        # Final Huber weights on inliers
        if final_inliers:
            final_huber_weights = np.zeros(len(final_inliers), dtype=np.float64)
            for m_idx, inlier_idx in enumerate(final_inliers):
                d = mahal_distances[inlier_idx]
                if d <= self.k_huber:
                    final_huber_weights[m_idx] = 1.0
                else:
                    final_huber_weights[m_idx] = self.k_huber / max(d, 1e-9)
        else:
            final_huber_weights = None

        # Residual metrics on final inliers
        if final_inliers:
            inlier_res = residuals_m[final_inliers]
            horiz_norms = np.linalg.norm(inlier_res[:, :2], axis=1)
            vert_norms = np.abs(inlier_res[:, 2])
            total_norms = np.linalg.norm(inlier_res, axis=1)

            horizontal_rmse = float(np.sqrt(np.mean(horiz_norms ** 2)))
            vertical_rmse = float(np.sqrt(np.mean(vert_norms ** 2)))
            total_3d_rmse = float(np.sqrt(np.mean(total_norms ** 2)))
            max_residual = float(np.max(total_norms))
        else:
            horizontal_rmse = 0.0
            vertical_rmse = 0.0
            total_3d_rmse = 0.0
            max_residual = 0.0

        # Attitude consistency cross-check (if drone body orientations are available)
        att_residuals: List[float] = []
        for idx in final_inliers:
            obs = observations[idx]
            # r_body_to_enu aligns camera optical axis if mounting is aligned
            r_geo = obs.r_body_to_enu
            # R_est * I
            diff_rot = r_geo @ current_sim3.rotation.T
            tr = float(np.trace(diff_rot))
            cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
            angle_deg = math.degrees(math.acos(cos_theta))
            att_residuals.append(angle_deg)

        mean_att_deg = float(np.mean(att_residuals)) if att_residuals else None

        diagnostics = EstimationDiagnostics(
            iterations=iteration,
            converged=converged,
            ransac_hypotheses_evaluated=evaluated_hypotheses,
            degenerate_triplets_rejected=rejected_degenerate_triplets,
            winning_triplet=best_triplet,
            initial_inlier_count=initial_inlier_count,
            final_inlier_count=len(final_inliers),
            inlier_ratio=len(final_inliers) / n if n > 0 else 0.0,
            horizontal_rmse_m=horizontal_rmse,
            vertical_rmse_m=vertical_rmse,
            total_3d_rmse_m=total_3d_rmse,
            max_residual_m=max_residual,
            mean_attitude_residual_deg=mean_att_deg,
            final_huber_weights=final_huber_weights,
        )

        return RobustSim3Result(
            success=True,
            sim3=current_sim3,
            inlier_indices=final_inliers,
            rejected_indices=final_rejected,
            residuals_m=residuals_m,
            mahalanobis_distances=mahal_distances,
            diagnostics=diagnostics,
            final_huber_weights=final_huber_weights,
        )
