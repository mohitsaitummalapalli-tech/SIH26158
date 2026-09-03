"""Rigorous 7-DoF Similarity Transformation Sim(3) for Geospatial Metric Alignment.

Governs the mapping between dimensionless reconstruction coordinates X_rec and local metric ENU coordinates X_geo:
    X_geo = s * R * X_rec + t

Properties:
- s in R_{>0}: Global isotropic scale factor [meters / reconstruction_unit]
- R in SO(3): 3x3 Orthonormal rotation matrix (R^T R = I, det(R) = +1)
- t in R^3: Translation vector in local ENU frame [meters]

Strictly preserves Euclidean angles, shape proportions, and collinearity.
Affine shear and nonlinear warps are forbidden.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Optional, List, Tuple, Union, Dict, Any
import numpy as np


class UncertaintyType(str, Enum):
    """Classification of transformation uncertainty provenance."""
    ESTIMATED_COVARIANCE = "ESTIMATED_COVARIANCE"
    HEURISTIC_UNCERTAINTY = "HEURISTIC_UNCERTAINTY"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class Sim3TransformContract:
    """Contract representation of a 7-DoF Sim(3) transformation."""
    scale: float
    rotation_matrix: List[List[float]]
    translation_enu: Tuple[float, float, float]
    scale_uncertainty_1sigma: float = 0.0
    uncertainty_type: UncertaintyType = UncertaintyType.UNAVAILABLE
    source_crs: str = "reconstruction_gauge"
    target_crs: str = "local_topocentric_enu"
    residual_rmse_m: Optional[float] = None
    inlier_count: int = 0
    fisher_condition_number: Optional[float] = None
    axial_rotation_resolved: bool = True
    rotational_null_direction: Optional[Tuple[float, float, float]] = None


class Sim3:
    """Mathematical 7-DoF Similarity Transformation representation and operations."""

    def __init__(
        self,
        scale: float,
        rotation: Union[np.ndarray, List[List[float]]],
        translation: Union[np.ndarray, List[float], Tuple[float, float, float]],
        scale_uncertainty_1sigma: float = 0.0,
        uncertainty_type: UncertaintyType = UncertaintyType.UNAVAILABLE,
        covariance_7x7: Optional[np.ndarray] = None,
        fisher_condition_number: Optional[float] = None,
        axial_rotation_resolved: bool = True,
        rotational_null_direction: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self.scale = float(scale)
        self.rotation = np.array(rotation, dtype=np.float64)
        self.translation = np.array(translation, dtype=np.float64).reshape((3,))
        self.scale_uncertainty_1sigma = float(scale_uncertainty_1sigma)
        self.uncertainty_type = uncertainty_type
        self.covariance_7x7 = covariance_7x7
        self.fisher_condition_number = fisher_condition_number
        self.axial_rotation_resolved = axial_rotation_resolved
        self.rotational_null_direction = rotational_null_direction

        self.validate()

    def validate(self) -> None:
        """Validate mathematical invariants of Sim(3)."""
        if math.isnan(self.scale) or math.isinf(self.scale) or self.scale <= 0.0:
            raise ValueError(f"Sim(3) scale must be finite and strictly positive, got {self.scale}")

        if self.rotation.shape != (3, 3):
            raise ValueError(f"Rotation matrix must be (3, 3), got {self.rotation.shape}")
        if np.any(np.isnan(self.rotation)) or np.any(np.isinf(self.rotation)):
            raise ValueError("Rotation matrix contains NaN or Inf values")

        # Check orthonormality: R^T R approx I
        rt_r = self.rotation.T @ self.rotation
        identity_diff = np.max(np.abs(rt_r - np.eye(3)))
        if identity_diff > 1e-4:
            raise ValueError(f"Rotation matrix is not orthonormal; ||R^T R - I||_max = {identity_diff:.2e}")

        # Check special orthogonal group: det(R) approx +1
        det_r = float(np.linalg.det(self.rotation))
        if abs(det_r - 1.0) > 1e-4:
            raise ValueError(f"Rotation determinant must be +1 (SO(3)), got {det_r:.6f}")

        if self.translation.shape != (3,):
            raise ValueError(f"Translation vector must be (3,), got {self.translation.shape}")
        if np.any(np.isnan(self.translation)) or np.any(np.isinf(self.translation)):
            raise ValueError("Translation vector contains NaN or Inf values")

    @classmethod
    def identity(cls) -> "Sim3":
        """Create identity Sim(3) transformation (s=1.0, R=I, t=0)."""
        return cls(
            scale=1.0,
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
            scale_uncertainty_1sigma=0.0,
            uncertainty_type=UncertaintyType.UNAVAILABLE,
        )

    def transform_point(self, point_rec: np.ndarray) -> np.ndarray:
        """Apply forward transformation to a point or array of points.
        
        X_geo = s * R * X_rec + t
        
        Args:
            point_rec: Array of shape (3,) or (N, 3).
            
        Returns:
            Transformed array of matching shape.
        """
        pt = np.asarray(point_rec, dtype=np.float64)
        if pt.ndim == 1:
            if pt.shape != (3,):
                raise ValueError(f"Point must be shape (3,), got {pt.shape}")
            return self.scale * (self.rotation @ pt) + self.translation
        elif pt.ndim == 2:
            if pt.shape[1] != 3:
                raise ValueError(f"Points array must be shape (N, 3), got {pt.shape}")
            # (N, 3) @ R^T * s + t
            return (pt @ self.rotation.T) * self.scale + self.translation
        else:
            raise ValueError(f"Expected 1D or 2D array, got ndim={pt.ndim}")

    def transform_camera_center(self, c_rec: np.ndarray) -> np.ndarray:
        """Transform optical camera center: C_geo = s * R * C_rec + t."""
        return self.transform_point(c_rec)

    def inverse(self) -> "Sim3":
        """Compute exact mathematical inverse of Sim(3).
        
        X_rec = (1/s) * R^T * (X_geo - t) = (1/s) * R^T * X_geo - (1/s) * R^T * t
        """
        inv_scale = 1.0 / self.scale
        inv_rotation = self.rotation.T
        inv_translation = -inv_scale * (inv_rotation @ self.translation)

        # Scale uncertainty transforms via error propagation: sigma(1/s) = sigma(s) / s^2
        inv_scale_unc = self.scale_uncertainty_1sigma / (self.scale * self.scale) if self.scale > 0 else 0.0

        return Sim3(
            scale=inv_scale,
            rotation=inv_rotation,
            translation=inv_translation,
            scale_uncertainty_1sigma=inv_scale_unc,
            uncertainty_type=self.uncertainty_type,
        )

    def transform_inverse(self, point_geo: np.ndarray) -> np.ndarray:
        """Apply inverse transformation: X_rec = (1/s) * R^T * (X_geo - t)."""
        return self.inverse().transform_point(point_geo)

    def compose(self, other: "Sim3") -> "Sim3":
        """Compose two Sim(3) transformations: T_result = self o other.
        
        T_result(X) = T_self(T_other(X))
                    = s1 * R1 * (s2 * R2 * X + t2) + t1
                    = (s1 * s2) * (R1 * R2) * X + (s1 * R1 * t2 + t1)
        """
        comp_scale = self.scale * other.scale
        comp_rotation = self.rotation @ other.rotation
        comp_translation = self.scale * (self.rotation @ other.translation) + self.translation

        return Sim3(
            scale=comp_scale,
            rotation=comp_rotation,
            translation=comp_translation,
            uncertainty_type=UncertaintyType.UNAVAILABLE,
        )

    def to_contract(self, inlier_count: int = 0, residual_rmse_m: Optional[float] = None) -> Sim3TransformContract:
        """Serialize into clean data contract."""
        return Sim3TransformContract(
            scale=self.scale,
            rotation_matrix=self.rotation.tolist(),
            translation_enu=(float(self.translation[0]), float(self.translation[1]), float(self.translation[2])),
            scale_uncertainty_1sigma=self.scale_uncertainty_1sigma,
            uncertainty_type=self.uncertainty_type,
            residual_rmse_m=residual_rmse_m,
            inlier_count=inlier_count,
            fisher_condition_number=self.fisher_condition_number,
            axial_rotation_resolved=self.axial_rotation_resolved,
            rotational_null_direction=self.rotational_null_direction,
        )


def solve_sim3_umeyama(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Sim3:
    """Closed-form similarity estimation between corresponding 3D point sets via Umeyama/Horn.
    
    Finds (s, R, t) minimizing:
        sum_i w_i || dst_i - (s * R * src_i + t) ||^2
        
    Args:
        src_points: Array of shape (N, 3) in source coordinates (e.g., reconstruction gauge).
        dst_points: Array of shape (N, 3) in destination coordinates (e.g., local ENU).
        weights: Optional positive weights array of shape (N,).
        
    Returns:
        Converged Sim3 instance.
        
    Raises:
        ValueError: If fewer than 3 points, collinear/degenerate geometry, or zero variance.
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)

    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Point shapes must match and be (N, 3), got src={src.shape}, dst={dst.shape}")

    n = src.shape[0]
    if n < 3:
        raise ValueError(f"At least 3 point correspondences required for Sim(3), got {n}")

    if weights is None:
        w = np.ones(n, dtype=np.float64) / n
    else:
        w_raw = np.asarray(weights, dtype=np.float64).reshape((n,))
        sum_w = np.sum(w_raw)
        if sum_w <= 0.0 or np.any(np.isnan(w_raw)) or np.any(w_raw < 0):
            raise ValueError(f"Invalid weights: sum must be positive and non-negative, got sum={sum_w}")
        w = w_raw / sum_w

    # Weighted centroids
    src_mean = np.sum(src * w[:, np.newaxis], axis=0)
    dst_mean = np.sum(dst * w[:, np.newaxis], axis=0)

    # Demeaned points
    src_demeaned = src - src_mean
    dst_demeaned = dst - dst_mean

    # Weighted variance of source points
    src_var = np.sum(w[:, np.newaxis] * (src_demeaned ** 2))
    if src_var < 1e-15:
        raise ValueError(f"Source points have zero/degenerate variance: {src_var}")

    # Weighted cross-covariance matrix: H = dst^T * W * src
    # (3, N) @ (N, 3) -> (3, 3)
    H = (dst_demeaned * w[:, np.newaxis]).T @ src_demeaned

    # Singular Value Decomposition of cross-covariance
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Reflection detection / correction
    d = np.linalg.det(U) * np.linalg.det(V)
    D = np.diag([1.0, 1.0, 1.0 if d >= 0 else -1.0])

    R = U @ D @ Vt
    # Enforce strictly orthonormal rotation
    u_rot, _, vt_rot = np.linalg.svd(R)
    R = u_rot @ vt_rot
    if np.linalg.det(R) < 0:
        R = u_rot @ np.diag([1.0, 1.0, -1.0]) @ vt_rot

    # Scale factor
    scale = float(np.sum(D @ S) / src_var)
    if scale <= 0.0:
        raise ValueError(f"Degenerate or negative scale computed in Umeyama: {scale}")

    # Translation
    translation = dst_mean - scale * (R @ src_mean)

    return Sim3(scale=scale, rotation=R, translation=translation)
