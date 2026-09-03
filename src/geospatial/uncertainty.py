"""Analytical parameter covariance estimation, uncertainty propagation, and provenance hashing.

Parameterization:
    theta = [ln(s), omega_x, omega_y, omega_z, t_x, t_y, t_z]^T in R^7

Fisher Information Matrix (Hessian):
    H = sum_{i in inliers} w_i * J_i^T * Sigma_i^-1 * J_i
    Sigma_theta = H^-1

Uncertainty Quantification:
    sigma_s = s * sqrt(Sigma_theta[0, 0])
    sigma_rot = sqrt(Tr(Sigma_theta[1:4, 1:4]))  [radians]
    sigma_trans = sqrt(Tr(Sigma_theta[4:7, 4:7])) [meters]
"""

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from src.geospatial.sim3 import Sim3, UncertaintyType
from src.geospatial.telemetry_observation import TelemetryObservation
from src.geospatial.lever_arm import LeverArm


@dataclass
class Sim3UncertaintyReport:
    """Rigorous transformation parameter uncertainty report."""
    uncertainty_type: UncertaintyType
    scale_uncertainty_1sigma: float
    relative_scale_uncertainty: float
    rotation_uncertainty_rad: float
    translation_uncertainty_m: float
    sigma_log_scale: float = 0.0
    rotational_null_direction: Optional[Tuple[float, float, float]] = None
    unconstrained_parameter_directions: List[str] = field(default_factory=list)
    axial_rotation_resolved: bool = True
    covariance_7x7: Optional[np.ndarray] = None
    fisher_condition_number: Optional[float] = None
    regularization_used: bool = False
    regularization_value: float = 0.0
    parameter_scales: Optional[Tuple[float, float, float]] = None
    fallback_reason: Optional[str] = None


def _skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Construct 3x3 skew-symmetric matrix from 3D vector."""
    x, y, z = v[0], v[1], v[2]
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=np.float64)


class UncertaintyPropagator:
    """Computes analytical parameter covariance and point uncertainty."""

    @staticmethod
    def estimate_parameter_covariance(
        sim3: Sim3,
        observations: List[TelemetryObservation],
        inlier_indices: List[int],
        lever_arm: LeverArm,
        huber_weights: Optional[np.ndarray] = None,
    ) -> Sim3UncertaintyReport:
        """Estimate 7x7 parameter covariance matrix via dimensionless normalized Huber-weighted Fisher matrix."""
        if len(inlier_indices) < 4:
            # Insufficient points for full rank 7x7 Hessian
            rel_unc = 0.50
            return Sim3UncertaintyReport(
                uncertainty_type=UncertaintyType.HEURISTIC_UNCERTAINTY,
                scale_uncertainty_1sigma=sim3.scale * rel_unc,
                relative_scale_uncertainty=rel_unc,
                rotation_uncertainty_rad=math.radians(5.0),
                translation_uncertainty_m=3.0,
                covariance_7x7=None,
                fisher_condition_number=None,
                regularization_used=True,
                regularization_value=0.0,
                parameter_scales=None,
                fallback_reason="Insufficient inliers (< 4) for Fisher information matrix estimation",
            )

        H = np.zeros((7, 7), dtype=np.float64)

        for k, idx in enumerate(inlier_indices):
            obs = observations[idx]
            # Modeled camera point: p_cam = s * R * C_rec
            p_cam = sim3.scale * (sim3.rotation @ obs.c_rec)

            # Jacobian J = d(r_i) / d(theta) where r_i = z - (p_cam + t + R_body * L)
            # d(r)/d(ln s) = - p_cam       [units L]
            # d(r)/d(omega) = [p_cam]_x    [units L]
            # d(r)/d(t) = - I_3            [units 1]
            J = np.zeros((3, 7), dtype=np.float64)
            J[:, 0] = -p_cam
            J[:, 1:4] = _skew_symmetric(p_cam)
            J[:, 4:7] = -np.eye(3, dtype=np.float64)

            # Measurement weight / inverse covariance [units L^-2]
            cov = obs.covariance_enu
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov + np.eye(3) * 1e-4)

            # Weight from converged robust Huber IRLS
            w_i = float(huber_weights[k]) if (huber_weights is not None and k < len(huber_weights)) else 1.0

            # Accumulate Huber-weighted Fisher Information (Hessian)
            H += w_i * (J.T @ inv_cov @ J)

        # Compute characteristic physical baseline span among inliers
        z_inliers = np.array([observations[idx].z_gnss_enu for idx in inlier_indices], dtype=np.float64)
        if len(z_inliers) > 1:
            diffs = z_inliers[:, np.newaxis, :] - z_inliers[np.newaxis, :, :]
            b_gnss = float(np.max(np.linalg.norm(diffs, axis=-1)))
        else:
            b_gnss = 0.0

        s_ln_s = 1.0  # dimensionless scale prior
        s_rot = 1.0   # dimensionless rotation prior (radians)
        s_pos = b_gnss if b_gnss > 0.0 else 1.0  # physical coordinate position scale [L]

        # Normalization Scale Matrices:
        # Physical translation t has dimension [L].
        # In parameter-normalization space, dimensionless parameter scaling is:
        #   theta_tilde = S_param @ theta, where S_param = diag(1, 1, 1, 1, 1/s_pos, 1/s_pos, 1/s_pos)
        #   with s_pos = D_geo (physical baseline extent).
        # Defining S = S_param^(-1) = diag(1, 1, 1, 1, s_pos, s_pos, s_pos) [dimensions 1, 1, 1, 1, L, L, L]:
        scale_vec = np.array([s_ln_s, s_rot, s_rot, s_rot, s_pos, s_pos, s_pos], dtype=np.float64)
        S = np.diag(scale_vec)

        # Dimensionless normalized Hessian:
        #   H_tilde = S @ H @ S = S_param^(-1) @ H @ S_param^(-1)
        # Dimensions: [L] * [L^(-2)] * [L] = [1] (strictly dimensionless)
        H_tilde = S @ H @ S

        # Dimensionless spectral conditioning evaluation
        eigenvals = np.linalg.eigvalsh(H_tilde)
        min_eig = float(np.min(eigenvals))
        max_eig = float(np.max(eigenvals))
        kappa = float(max_eig / max(min_eig, 1e-15))

        regularization_used = False
        regularization_value = 0.0
        fallback_reason: Optional[str] = None

        if min_eig >= 1e-8 and kappa <= 1e8:
            try:
                cov_tilde = np.linalg.inv(H_tilde)
                unc_type = UncertaintyType.ESTIMATED_COVARIANCE
            except np.linalg.LinAlgError:
                lambda_reg = 1e-6  # DIMENSIONLESS NUMERICAL HEURISTIC
                cov_tilde = np.linalg.pinv(H_tilde + np.eye(7) * lambda_reg)
                unc_type = UncertaintyType.HEURISTIC_UNCERTAINTY
                regularization_used = True
                regularization_value = lambda_reg
                fallback_reason = "Inversion failure on normalized Hessian; applied dimensionless heuristic damping"
        else:
            # Singular, ill-conditioned (e.g. collinear flight), or rank-deficient Hessian
            lambda_reg = 1e-6  # DIMENSIONLESS NUMERICAL HEURISTIC
            cov_tilde = np.linalg.pinv(H_tilde + np.eye(7) * lambda_reg)
            unc_type = UncertaintyType.HEURISTIC_UNCERTAINTY
            regularization_used = True
            regularization_value = lambda_reg
            fallback_reason = f"Ill-conditioned normalized Fisher matrix (kappa={kappa:.2e} > 1e8 or min_eig={min_eig:.2e} < 1e-8); applied dimensionless regularization lambda_reg=1e-6"

        # Transform covariance back to physical parameterization:
        # Since H_tilde = S @ H @ S, for nonsingular H we have:
        #   H = S^(-1) @ H_tilde @ S^(-1)
        # Taking the matrix inverse:
        #   Sigma_theta = H^(-1) = (S^(-1) @ H_tilde @ S^(-1))^(-1)
        #               = (S^(-1))^(-1) @ H_tilde^(-1) @ (S^(-1))^(-1)
        #               = S @ cov_tilde @ S
        # Equivalently, in terms of parameter scale S_param = S^(-1) (where S_pos = 1/D_geo):
        #   Sigma_theta = S_param^(-1) @ cov_tilde @ S_param^(-1)
        # Both formulations identically multiply the translation block by s_pos^2 = D_geo^2,
        # restoring physical units [L^2] and preserving translation scale invariance (sigma_km = sigma_m / 1000).
        cov_theta = S @ cov_tilde @ S

        var_ln_s = max(0.0, float(cov_theta[0, 0]))
        sigma_ln_s = math.sqrt(var_ln_s)
        # delta s = s * delta(ln s)
        sigma_s = sim3.scale * sigma_ln_s
        rel_scale_unc = sigma_ln_s

        var_rot = max(0.0, float(np.trace(cov_theta[1:4, 1:4])))
        sigma_rot = math.sqrt(var_rot)

        var_trans = max(0.0, float(np.trace(cov_theta[4:7, 4:7])))
        sigma_trans = math.sqrt(var_trans)

        # Inspect rotational submatrix of H_tilde to identify any unconstrained axial null modes
        H_rot = H_tilde[1:4, 1:4]
        rot_evals, rot_evecs = np.linalg.eigh(H_rot)
        rotational_null_direction: Optional[Tuple[float, float, float]] = None
        unconstrained_directions: List[str] = []

        if rot_evals[0] < 1e-4:
            # Trajectory is collinear: rotation around the line is unconstrained by positions
            null_vec = rot_evecs[:, 0]
            norm_null = float(np.linalg.norm(null_vec))
            if norm_null > 0:
                null_vec = null_vec / norm_null
                rotational_null_direction = (float(null_vec[0]), float(null_vec[1]), float(null_vec[2]))
            unconstrained_directions.append("axial_rotation_about_trajectory")

        axial_resolved = sim3.axial_rotation_resolved
        if not axial_resolved and "axial_rotation_about_trajectory" not in unconstrained_directions:
            unconstrained_directions.append("axial_rotation_about_trajectory")
            if rotational_null_direction is None:
                rotational_null_direction = sim3.rotational_null_direction

        return Sim3UncertaintyReport(
            uncertainty_type=unc_type,
            scale_uncertainty_1sigma=sigma_s,
            relative_scale_uncertainty=rel_scale_unc,
            rotation_uncertainty_rad=sigma_rot,
            translation_uncertainty_m=sigma_trans,
            sigma_log_scale=sigma_ln_s,
            rotational_null_direction=rotational_null_direction,
            unconstrained_parameter_directions=unconstrained_directions,
            axial_rotation_resolved=axial_resolved,
            covariance_7x7=cov_theta,
            fisher_condition_number=kappa,
            regularization_used=regularization_used,
            regularization_value=regularization_value,
            parameter_scales=(s_ln_s, s_rot, s_pos),
            fallback_reason=fallback_reason,
        )

    @staticmethod
    def compute_provenance_hash(
        anchor_lat: float,
        anchor_lon: float,
        anchor_alt: float,
        sim3_scale: float,
        sim3_translation: Tuple[float, float, float],
        inlier_count: int,
        rmse_m: float,
    ) -> str:
        """Generate SHA-256 provenance signature for auditability."""
        payload = {
            "anchor": [round(anchor_lat, 8), round(anchor_lon, 8), round(anchor_alt, 3)],
            "scale": round(sim3_scale, 8),
            "translation": [round(x, 4) for x in sim3_translation],
            "inlier_count": inlier_count,
            "rmse_m": round(rmse_m, 4),
        }
        raw_json = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw_json).hexdigest()
