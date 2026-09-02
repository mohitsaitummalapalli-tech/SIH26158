"""Phase 3D.1: Global Bundle Adjustment Optimizer Implementation.

SCIENTIFIC OBJECTIVE:
Refine relative camera poses, 3D landmark coordinates, and optional intrinsics jointly
by minimizing image-space reprojection errors across all valid 2D track observations.

CRITICAL PRINCIPLES:
1. Monocular Relative Reconstruction: Bundle Adjustment operates strictly in relative
   reconstruction space (SCALE_AMBIGUOUS). It does not establish absolute metric scale
   or georeferencing without certified external metric ground truth.
2. Sim(3) Gauge Preservation & Exact Parameter Dimension:
   - Camera 0 is fixed at [I | 0] (0 DoF).
   - Camera 1 translation has unit norm (||t_10|| = 1.0), parameterized as a 2-DoF direction
     on S^2 in local tangent space, while its rotation has 3 DoF.
   - Cameras 2...M-1 have 6 DoF each (3 rotation, 3 translation).
   - Landmarks have 3 DoF each.
   Total optimization dimension: dim(Theta) = 6M - 7 + 3N for M >= 2.
3. Residuals in Pixels: Image-space reprojection residual r_ij is measured in pixels.
   Lower reprojection error demonstrates internal geometric consistency (LEVEL_1),
   not physical 3D ground-truth accuracy.
4. Minimal Rotation Parameterization: Rotations are parameterized minimally via 3-DoF
   Lie algebra so(3) / axis-angle vectors, avoiding 9-DoF unconstrained matrix drift.
5. Robust Huber Objective: Loss rho_delta(e) is evaluated on residual norm e = ||r||_2,
   smoothly bridging quadratic and linear growth at delta (HEURISTIC_DEFAULT).
6. Sparse Block Structure: Observation residuals depend only on the observing camera
   and the observed landmark. Solved via scipy least_squares with TRF and sparse Jacobian pattern.
7. Rollback Safety: If optimization fails validation, original reconstruction is preserved.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Optional, List, Dict, Any, Tuple, Set

import numpy as np
from scipy.optimize import least_squares
import scipy.sparse as sp

from src.geometry.contracts import (
    EvaluationLevel,
    PipelineStageStatus,
    MeasurementType,
    GaugeFixingPolicy,
    GeometryFailureReason,
    CameraIntrinsics,
    ExtrinsicPose,
    TriangulatedTrack,
    SparseReconstructionResult,
)
from src.geometry.sfm import SfMCamera, SfMTrack


class BAFailureReason(str, Enum):
    """Explicit failure taxonomy for global Bundle Adjustment."""
    INVALID_INPUT_RECONSTRUCTION = "INVALID_INPUT_RECONSTRUCTION"
    INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"
    INVALID_CAMERA_STATE = "INVALID_CAMERA_STATE"
    INVALID_LANDMARK_STATE = "INVALID_LANDMARK_STATE"
    GAUGE_CONSTRAINT_INVALID = "GAUGE_CONSTRAINT_INVALID"
    PROJECTION_FAILURE = "PROJECTION_FAILURE"
    OPTIMIZATION_FAILED = "OPTIMIZATION_FAILED"
    OPTIMIZATION_DIVERGED = "OPTIMIZATION_DIVERGED"
    MAX_ITERATIONS_REACHED = "MAX_ITERATIONS_REACHED"
    NUMERICAL_SINGULARITY = "NUMERICAL_SINGULARITY"
    POST_OPTIMIZATION_VALIDATION_FAILED = "POST_OPTIMIZATION_VALIDATION_FAILED"


@dataclass(frozen=True)
class BundleAdjustmentConfig:
    """Configurable heuristic engineering defaults (HEURISTIC_DEFAULT) for Bundle Adjustment.
    
    All numerical parameters in this configuration are engineering defaults and must not
    be construed as universal mathematical constants. Satisfaction of a convergence threshold
    is a local numerical termination condition, NOT a mathematical proof of the global optimum.
    """
    loss_function: str = "HUBER"                        # HEURISTIC_DEFAULT: Robust loss type ("HUBER" or "SQUARED")
    huber_delta_px: float = 2.0                         # HEURISTIC_DEFAULT: Huber threshold in pixels
    optimize_intrinsics: bool = False                   # HEURISTIC_DEFAULT: Intrinsics held fixed by default
    max_iterations: int = 50                            # HEURISTIC_DEFAULT: Maximum non-linear solver iterations
    cost_tolerance: float = 1e-6                        # HEURISTIC_DEFAULT: Relative cost reduction stopping tolerance
    parameter_tolerance: float = 1e-6                   # HEURISTIC_DEFAULT: Step norm stopping tolerance
    gradient_tolerance: float = 1e-8                    # HEURISTIC_DEFAULT: Gradient infinity-norm stopping tolerance
    min_registered_cameras: int = 2                     # HEURISTIC_DEFAULT: Minimum registered cameras required
    min_landmarks: int = 10                             # HEURISTIC_DEFAULT: Minimum 3D landmarks required
    min_observations_per_landmark: int = 2              # HEURISTIC_DEFAULT: Minimum observation count for optimization
    max_reprojection_rmse_px: float = 4.0               # HEURISTIC_DEFAULT: Post-optimization RMSE acceptance ceiling
    rmse_divergence_tolerance_px: float = 2.0           # HEURISTIC_DEFAULT: Maximum allowable RMSE increase sanity check
    gauge_policy: GaugeFixingPolicy = GaugeFixingPolicy.FIX_FIRST_CAMERA_AND_UNIT_BASELINE
    config_version: str = "BAConfig_v1.1"


@dataclass
class BAReprojectionMetrics:
    """Image-space reprojection residual statistics before and after optimization."""
    mean_error_px: float
    rmse_px: float
    median_error_px: float
    percentile_90_px: float
    max_error_px: float
    total_observations: int
    measurement_type: MeasurementType = MeasurementType.ESTIMATED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_error_px": self.mean_error_px,
            "rmse_px": self.rmse_px,
            "median_error_px": self.median_error_px,
            "percentile_90_px": self.percentile_90_px,
            "max_error_px": self.max_error_px,
            "total_observations": self.total_observations,
            "measurement_type": self.measurement_type.value,
        }


@dataclass
class BundleAdjustmentResult:
    """Typed result contract for global Bundle Adjustment."""
    status: PipelineStageStatus
    refined_reconstruction: Optional[SparseReconstructionResult]
    metrics_before: BAReprojectionMetrics
    metrics_after: Optional[BAReprojectionMetrics]
    cost_before: float
    cost_after: float
    total_iterations: int
    convergence_reason: str
    gauge_preserved: bool
    is_metric_scale: bool = False
    has_monocular_scale_ambiguity: bool = True
    failure_reason: Optional[BAFailureReason] = None
    diagnostics: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "has_refined_reconstruction": self.refined_reconstruction is not None,
            "metrics_before": self.metrics_before.to_dict(),
            "metrics_after": self.metrics_after.to_dict() if self.metrics_after else None,
            "cost_before": self.cost_before,
            "cost_after": self.cost_after,
            "total_iterations": self.total_iterations,
            "convergence_reason": self.convergence_reason,
            "gauge_preserved": self.gauge_preserved,
            "is_metric_scale": self.is_metric_scale,
            "has_monocular_scale_ambiguity": self.has_monocular_scale_ambiguity,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "diagnostics": self.diagnostics,
            "provenance": self.provenance,
        }


class BARotationMath:
    """Minimal 3-DoF Lie algebra so(3) rotation parameterization and conversions."""

    @staticmethod
    def rodrigues_to_rotation(omega: np.ndarray) -> np.ndarray:
        """Convert axis-angle vector omega in so(3) to SO(3) rotation matrix R via Rodrigues formula.
        
        R = I + (sin(theta)/theta) * [omega]_x + ((1 - cos(theta))/theta^2) * [omega]_x^2
        Uses Taylor series near theta -> 0 to maintain numerical stability without singularities.
        """
        omega = np.asarray(omega, dtype=np.float64).ravel()
        theta_sq = float(np.dot(omega, omega))
        theta = math.sqrt(theta_sq)

        wx = np.array([
            [0.0, -omega[2], omega[1]],
            [omega[2], 0.0, -omega[0]],
            [-omega[1], omega[0], 0.0],
        ], dtype=np.float64)

        if theta < 1e-7:
            return np.eye(3) + wx + 0.5 * (wx @ wx)

        a = math.sin(theta) / theta
        b = (1.0 - math.cos(theta)) / theta_sq
        return np.eye(3) + a * wx + b * (wx @ wx)

    @staticmethod
    def rotation_to_rodrigues(R: np.ndarray) -> np.ndarray:
        """Convert SO(3) rotation matrix to minimal 3-DoF axis-angle vector omega in so(3)."""
        R = np.asarray(R, dtype=np.float64)
        tr = float(np.trace(R))
        cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
        theta = math.acos(cos_theta)

        if theta < 1e-7:
            return np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ], dtype=np.float64) * 0.5

        scale = theta / (2.0 * math.sin(theta))
        return np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ], dtype=np.float64) * scale


class BATranslationDirectionMath:
    """Manages 2-DoF tangent-space parameterization of unit translation directions on S^2.
    
    Camera 1 baseline magnitude is held constant at 1.0 reconstruction units to fix the
    monocular scale gauge freedom. Its translation is therefore constrained to S^2, leaving
    exactly 2 DoF for translation direction.
    """

    @staticmethod
    def construct_tangent_basis(d0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Construct an orthonormal basis (b1, b2) for the tangent plane T_d0(S^2).
        
        Guarantees:
        - b1 . d0 = 0, b2 . d0 = 0, b1 . b2 = 0
        - ||b1|| = 1, ||b2|| = 1
        """
        d = np.asarray(d0, dtype=np.float64).ravel()
        norm_d = float(np.linalg.norm(d))
        if norm_d < 1e-8:
            d = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            d = d / norm_d

        # Pick reference vector not collinear with d
        if abs(float(d[0])) < 0.8:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        b1 = np.cross(d, ref)
        norm_b1 = float(np.linalg.norm(b1))
        if norm_b1 < 1e-8:
            b1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            b1 = b1 / norm_b1

        b2 = np.cross(d, b1)
        b2 = b2 / max(1e-8, float(np.linalg.norm(b2)))
        return b1, b2

    @staticmethod
    def tangent_to_direction(d0: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Map 2-DoF tangent vector alpha in R^2 to unit direction vector on S^2.
        
        Uses geodesic exponential map on S^2:
        theta = ||alpha||_2
        direction = cos(theta) * d0 + (sin(theta)/theta) * (alpha_1 * b1 + alpha_2 * b2)
        Guarantees ||direction|| == 1.0 unconditionally for any alpha in R^2.
        """
        d0 = np.asarray(d0, dtype=np.float64).ravel()
        d0 = d0 / max(1e-8, float(np.linalg.norm(d0)))
        b1, b2 = BATranslationDirectionMath.construct_tangent_basis(d0)

        alpha = np.asarray(alpha, dtype=np.float64).ravel()
        theta = float(np.linalg.norm(alpha))

        if theta < 1e-7:
            v = d0 + alpha[0] * b1 + alpha[1] * b2
            return v / float(np.linalg.norm(v))

        delta_dir = (alpha[0] * b1 + alpha[1] * b2) / theta
        res = math.cos(theta) * d0 + math.sin(theta) * delta_dir
        return res / float(np.linalg.norm(res))

    @staticmethod
    def direction_to_tangent(d0: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Map unit direction on S^2 to 2-DoF tangent perturbation alpha in R^2 via log map."""
        d0 = np.asarray(d0, dtype=np.float64).ravel()
        d0 = d0 / max(1e-8, float(np.linalg.norm(d0)))
        direction = np.asarray(direction, dtype=np.float64).ravel()
        direction = direction / max(1e-8, float(np.linalg.norm(direction)))

        b1, b2 = BATranslationDirectionMath.construct_tangent_basis(d0)
        dot_val = np.clip(float(np.dot(d0, direction)), -1.0, 1.0)
        theta = math.acos(dot_val)

        if theta < 1e-7:
            return np.array([float(np.dot(direction, b1)), float(np.dot(direction, b2))], dtype=np.float64)

        proj1 = float(np.dot(direction, b1))
        proj2 = float(np.dot(direction, b2))
        proj_norm = math.sqrt(proj1**2 + proj2**2)
        if proj_norm < 1e-8:
            return np.zeros(2, dtype=np.float64)
        scale = theta / proj_norm
        return np.array([proj1 * scale, proj2 * scale], dtype=np.float64)


@dataclass(frozen=True)
class BAParameterLayout:
    """Explicit, deterministic memory layout of parameter vector Theta."""
    camera_order: List[str]
    track_order: List[int]
    camera_offsets: Dict[str, int]
    landmark_offsets: Dict[int, int]
    total_dimension: int
    camera_count: int
    landmark_count: int


class BAParameterManager:
    """Manages parameter packing, unpacking, and gauge constraints for Bundle Adjustment.
    
    GAUGE CONSTRAINTS:
    - Camera 0 is fixed at [I | 0] (0 parameters in Theta).
    - Camera 1 translation magnitude is fixed to 1.0 reconstruction units.
      Camera 1 contributes 3 rotation params + 2 translation direction params on S^2 = 5 params.
    - Cameras 2...M-1 contribute 6 params each (3 rotation, 3 translation).
    - Landmarks contribute 3 params each (X, Y, Z).
    
    DIMENSION INVARIANT:
    For M cameras and N landmarks with M >= 2:
    total_params = 5 + 6*(M - 2) + 3*N = 6*M - 7 + 3*N
    """

    def __init__(
        self,
        camera_order: List[str],
        track_order: List[int],
        config: BundleAdjustmentConfig,
        ref_cam1_direction: Optional[np.ndarray] = None,
    ):
        self.camera_order = list(camera_order)
        self.track_order = list(track_order)
        self.config = config

        self.ref_camera_id = self.camera_order[0] if len(self.camera_order) > 0 else ""
        self.cam1_id = self.camera_order[1] if len(self.camera_order) > 1 else ""
        self.other_camera_ids = self.camera_order[2:] if len(self.camera_order) > 2 else []

        if ref_cam1_direction is not None:
            norm_d = float(np.linalg.norm(ref_cam1_direction))
            self.ref_cam1_direction = ref_cam1_direction / (norm_d if norm_d > 1e-8 else 1.0)
        else:
            self.ref_cam1_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    @property
    def layout(self) -> BAParameterLayout:
        """Deterministic memory layout of parameter vector Theta."""
        cam_offsets: Dict[str, int] = {}
        offset = 0
        if self.cam1_id:
            cam_offsets[self.cam1_id] = offset
            offset += 5
        for cid in self.other_camera_ids:
            cam_offsets[cid] = offset
            offset += 6

        lm_offsets: Dict[int, int] = {}
        for tid in self.track_order:
            lm_offsets[tid] = offset
            offset += 3

        return BAParameterLayout(
            camera_order=list(self.camera_order),
            track_order=list(self.track_order),
            camera_offsets=cam_offsets,
            landmark_offsets=lm_offsets,
            total_dimension=self.total_params,
            camera_count=len(self.camera_order),
            landmark_count=len(self.track_order),
        )

    @property
    def num_camera_params(self) -> int:
        """Camera parameter count under Sim(3) gauge fixing (6M - 7 for M >= 2)."""
        M = len(self.camera_order)
        if M < 2:
            return 0
        return 6 * M - 7

    @property
    def num_landmark_params(self) -> int:
        """3 parameters per 3D landmark (X, Y, Z)."""
        return len(self.track_order) * 3

    @property
    def total_params(self) -> int:
        """Total dimension of optimization state vector Theta = 6M - 7 + 3N (M >= 2)."""
        return self.num_camera_params + self.num_landmark_params

    def pack_parameters(
        self,
        cameras: Dict[str, SfMCamera],
        landmarks: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Pack optimizable camera poses and landmarks into parameter vector Theta."""
        params = np.zeros(self.total_params, dtype=np.float64)
        offset = 0

        # Camera 1: 3 rotation + 2 translation direction on S^2
        if self.cam1_id and self.cam1_id in cameras:
            cam1 = cameras[self.cam1_id]
            omega1 = BARotationMath.rotation_to_rodrigues(cam1.R_cw)
            params[offset:offset + 3] = omega1
            offset += 3

            t1 = cam1.t_cw
            norm_t1 = float(np.linalg.norm(t1))
            if norm_t1 > 1e-8:
                self.ref_cam1_direction = t1 / norm_t1
            alpha1 = BATranslationDirectionMath.direction_to_tangent(self.ref_cam1_direction, t1)
            params[offset:offset + 2] = alpha1
            offset += 2

        # Cameras 2...M-1: 3 rotation + 3 translation
        for cam_id in self.other_camera_ids:
            cam = cameras[cam_id]
            omega = BARotationMath.rotation_to_rodrigues(cam.R_cw)
            params[offset:offset + 3] = omega
            params[offset + 3:offset + 6] = cam.t_cw
            offset += 6

        # Landmarks: 3 parameters each
        for t_id in self.track_order:
            params[offset:offset + 3] = landmarks[t_id]
            offset += 3

        return params

    def unpack_parameters(
        self,
        params: np.ndarray,
        base_cameras: Dict[str, SfMCamera],
    ) -> Tuple[Dict[str, SfMCamera], Dict[int, np.ndarray]]:
        """Unpack parameter vector Theta into cameras and landmarks, strictly enforcing gauge preservation."""
        new_cameras: Dict[str, SfMCamera] = {}
        new_landmarks: Dict[int, np.ndarray] = {}

        # 1. Camera 0 remains gauge-fixed at [I | 0]
        if self.ref_camera_id and self.ref_camera_id in base_cameras:
            ref_cam = base_cameras[self.ref_camera_id]
            new_cameras[self.ref_camera_id] = SfMCamera(
                frame_id=self.ref_camera_id,
                R_cw=np.eye(3, dtype=np.float64),
                t_cw=np.zeros(3, dtype=np.float64),
                intrinsics=ref_cam.intrinsics,
                is_registered=True,
                registration_order=0,
            )

        offset = 0

        # 2. Camera 1: 3 rotation + 2 translation-direction on S^2 (norm strictly 1.0)
        if self.cam1_id and self.cam1_id in base_cameras:
            base_cam1 = base_cameras[self.cam1_id]
            omega1 = params[offset:offset + 3]
            R_cw1 = BARotationMath.rodrigues_to_rotation(omega1)
            offset += 3

            alpha1 = params[offset:offset + 2]
            unit_dir = BATranslationDirectionMath.tangent_to_direction(self.ref_cam1_direction, alpha1)
            offset += 2

            # Translation magnitude is strictly 1.0 reconstruction units
            t_cw1 = unit_dir * 1.0

            new_cameras[self.cam1_id] = SfMCamera(
                frame_id=self.cam1_id,
                R_cw=R_cw1,
                t_cw=t_cw1,
                intrinsics=base_cam1.intrinsics,
                is_registered=True,
                registration_order=base_cam1.registration_order,
            )

        # 3. Cameras 2...M-1: 3 rotation + 3 translation
        for cam_id in self.other_camera_ids:
            base_cam = base_cameras[cam_id]
            omega = params[offset:offset + 3]
            t_cw = params[offset + 3:offset + 6].copy()
            R_cw = BARotationMath.rodrigues_to_rotation(omega)
            offset += 6

            new_cameras[cam_id] = SfMCamera(
                frame_id=cam_id,
                R_cw=R_cw,
                t_cw=t_cw,
                intrinsics=base_cam.intrinsics,
                is_registered=True,
                registration_order=base_cam.registration_order,
            )

        # 4. Landmarks: 3 per track
        for t_id in self.track_order:
            new_landmarks[t_id] = params[offset:offset + 3].copy()
            offset += 3

        return new_cameras, new_landmarks


class BAResidualCalculator:
    """Computes 2D image-space reprojection residuals and robust Huber loss."""

    @staticmethod
    def project_point(
        K: CameraIntrinsics,
        R_cw: np.ndarray,
        t_cw: np.ndarray,
        X_w: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Project 3D world point into camera raster.
        
        Returns:
            pixel_coord: np.ndarray [u, v] or None if non-finite/behind camera
            depth_z: optical depth Z_c
        """
        pt_c = R_cw @ X_w + t_cw
        z_c = float(pt_c[2])

        # Strict positive depth check (cheirality)
        if z_c <= 1e-6 or not np.all(np.isfinite(pt_c)):
            return None, z_c

        u = K.fx * (pt_c[0] / z_c) + K.cx
        v = K.fy * (pt_c[1] / z_c) + K.cy

        if not (math.isfinite(u) and math.isfinite(v)):
            return None, z_c

        return np.array([u, v], dtype=np.float64), z_c

    @staticmethod
    def compute_residual(
        observed_px: Tuple[float, float],
        projected_px: np.ndarray,
    ) -> np.ndarray:
        """Compute 2D reprojection residual vector: r = x_observed - x_projected (pixels)."""
        return np.array([observed_px[0] - projected_px[0], observed_px[1] - projected_px[1]], dtype=np.float64)

    @staticmethod
    def huber_loss(residual: np.ndarray, delta_px: float = 2.0) -> Tuple[float, float]:
        """Evaluate Huber robust loss function.
        
        Let e = ||residual||_2 in pixels.
        rho_delta(e) = 0.5 * e^2                 if e <= delta_px
                     = delta_px * (e - 0.5 * delta_px) if e > delta_px
        
        Continuous and C^1 smooth everywhere, including at e = delta_px:
        - Value at delta_px: 0.5 * delta_px^2
        - First derivative: rho'(e) = e for e <= delta_px; rho'(e) = delta_px for e > delta_px.
        
        Returns:
            loss_value: scalar robust loss rho_delta(e)
            weight: effective M-estimator weight w(e) = rho'(e) / e in range (0, 1]
        """
        e_sq = float(np.dot(residual, residual))
        e = math.sqrt(e_sq)

        if e <= delta_px:
            return 0.5 * e_sq, 1.0
        else:
            loss_val = delta_px * (e - 0.5 * delta_px)
            weight = delta_px / max(1e-8, e)
            return loss_val, weight

    @classmethod
    def evaluate_reconstruction_metrics(
        cls,
        cameras: Dict[str, SfMCamera],
        landmarks: Dict[int, np.ndarray],
        tracks: Dict[int, SfMTrack],
        delta_px: float = 2.0,
    ) -> Tuple[BAReprojectionMetrics, float, List[str]]:
        """Compute reprojection error statistics and total robust cost across all observations."""
        errors_px: List[float] = []
        diagnostics: List[str] = []
        total_cost = 0.0

        for t_id, track in tracks.items():
            if t_id not in landmarks:
                continue
            X_w = landmarks[t_id]

            for cam_id, obs_px in track.observations.items():
                if cam_id not in cameras:
                    continue
                cam = cameras[cam_id]

                proj_px, z_c = cls.project_point(cam.intrinsics, cam.R_cw, cam.t_cw, X_w)
                if proj_px is None or z_c <= 1e-6:
                    diagnostics.append(f"Cheirality or projection failure for track {t_id} in camera {cam_id}.")
                    continue

                r = cls.compute_residual(obs_px, proj_px)
                err = float(np.linalg.norm(r))
                errors_px.append(err)

                loss_val, _ = cls.huber_loss(r, delta_px=delta_px)
                total_cost += loss_val

        if len(errors_px) == 0:
            return BAReprojectionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0), 0.0, diagnostics

        mean_err = float(np.mean(errors_px))
        rmse_err = math.sqrt(float(np.mean(np.array(errors_px) ** 2)))
        med_err = float(np.median(errors_px))
        p90_err = float(np.percentile(errors_px, 90))
        max_err = float(np.max(errors_px))

        metrics = BAReprojectionMetrics(
            mean_error_px=mean_err,
            rmse_px=rmse_err,
            median_error_px=med_err,
            percentile_90_px=p90_err,
            max_error_px=max_err,
            total_observations=len(errors_px),
        )
        return metrics, total_cost, diagnostics


class BAResidualEvaluator:
    """Evaluates observation residuals, per-observation errors, and robust objective."""

    @classmethod
    def evaluate(
        cls,
        params: np.ndarray,
        param_manager: BAParameterManager,
        base_cameras: Dict[str, SfMCamera],
        tracks: Dict[int, SfMTrack],
        delta_px: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, float, int, int, int]:
        """Evaluate residuals vector, error norms, and depth validity.
        
        Returns:
            residuals: 1D array of shape (2 * num_observations,) containing [u_obs - u_proj, v_obs - v_proj]
            error_norms: 1D array of shape (num_observations,)
            robust_cost: float scalar sum of rho_delta(e)
            valid_count: int count of valid projections
            positive_depth_count: int count of projections with Z_c > 0
            invalid_depth_count: int count of projections with Z_c <= 0 or non-finite
        """
        cameras, landmarks = param_manager.unpack_parameters(params, base_cameras)
        residuals_list: List[float] = []
        error_norms_list: List[float] = []
        robust_cost = 0.0
        valid_count = 0
        positive_depth_count = 0
        invalid_depth_count = 0

        for t_id in param_manager.track_order:
            if t_id not in landmarks or t_id not in tracks:
                continue
            X_w = landmarks[t_id]
            track = tracks[t_id]

            for cam_id in param_manager.camera_order:
                if cam_id not in track.observations or cam_id not in cameras:
                    continue
                obs_px = track.observations[cam_id]
                cam = cameras[cam_id]

                proj_px, z_c = BAResidualCalculator.project_point(cam.intrinsics, cam.R_cw, cam.t_cw, X_w)
                if proj_px is None or z_c <= 1e-6:
                    invalid_depth_count += 1
                    # Avoid NaN/Inf in optimizer without fabricating huge fake residuals
                    r_u = float(obs_px[0] - cam.intrinsics.cx)
                    r_v = float(obs_px[1] - cam.intrinsics.cy)
                    residuals_list.extend([r_u, r_v])
                    e = math.sqrt(r_u**2 + r_v**2)
                    error_norms_list.append(e)
                    loss_val, _ = BAResidualCalculator.huber_loss(np.array([r_u, r_v]), delta_px)
                    robust_cost += loss_val
                else:
                    valid_count += 1
                    positive_depth_count += 1
                    r = BAResidualCalculator.compute_residual(obs_px, proj_px)
                    residuals_list.extend([float(r[0]), float(r[1])])
                    e = float(np.linalg.norm(r))
                    error_norms_list.append(e)
                    loss_val, _ = BAResidualCalculator.huber_loss(r, delta_px)
                    robust_cost += loss_val

        return (
            np.array(residuals_list, dtype=np.float64),
            np.array(error_norms_list, dtype=np.float64),
            float(robust_cost),
            valid_count,
            positive_depth_count,
            invalid_depth_count,
        )


class BASparsityStructure:
    """Defines and audits the sparse block structure of the Bundle Adjustment problem.
    
    Sparsity depends on camera/track connectivity; typical multi-view reconstructions exhibit
    substantial structural zeros, but percentages are problem-dependent, not universal constants.
    """

    def __init__(
        self,
        camera_order: List[str],
        track_order: List[int],
        observation_pairs: List[Tuple[str, int]],
    ):
        self.camera_order = list(camera_order)
        self.track_order = list(track_order)
        self.observation_pairs = list(observation_pairs)

        self.cam_to_idx = {cid: idx for idx, cid in enumerate(self.camera_order[1:])}  # Optimizable cams
        self.track_to_idx = {tid: idx for idx, tid in enumerate(self.track_order)}

    @property
    def num_residuals(self) -> int:
        """2 residual dimensions (u, v) per observation."""
        return len(self.observation_pairs) * 2

    @property
    def total_jacobian_entries(self) -> int:
        """Dense matrix entries = num_residuals * num_parameters."""
        M = len(self.camera_order)
        n_cam_params = 6 * M - 7 if M >= 2 else 0
        n_params = n_cam_params + len(self.track_to_idx) * 3
        return self.num_residuals * n_params

    @property
    def sparse_nonzeros(self) -> int:
        """Non-zero Jacobian entries: each 2D residual has nonzeros only for its camera and landmark."""
        nnz = 0
        first_opt_cam = self.camera_order[1] if len(self.camera_order) > 1 else ""
        for cam_id, track_id in self.observation_pairs:
            if cam_id == first_opt_cam:
                # Camera 1 has 5 parameters: 2 residuals * 5 = 10 nonzeros
                nnz += 10
            elif cam_id in self.cam_to_idx:
                # Cameras 2...M-1 have 6 parameters: 2 residuals * 6 = 12 nonzeros
                nnz += 12
            # Landmark has 3 parameters: 2 residuals * 3 = 6 nonzeros
            if track_id in self.track_to_idx:
                nnz += 6
        return nnz

    @property
    def sparsity_ratio(self) -> float:
        """Fraction of Jacobian entries that are structurally zero."""
        if self.total_jacobian_entries == 0:
            return 1.0
        return 1.0 - (self.sparse_nonzeros / self.total_jacobian_entries)


class BAPostOptimizationValidator:
    """Validates optimized variables against geometric safety gates, robust objective, and gauge constraints."""

    @staticmethod
    def validate(
        cameras: Dict[str, SfMCamera],
        landmarks: Dict[int, np.ndarray],
        tracks: Dict[int, SfMTrack],
        ref_camera_id: str,
        metrics_before: BAReprojectionMetrics,
        metrics_after: BAReprojectionMetrics,
        cost_before: float,
        cost_after: float,
        config: BundleAdjustmentConfig,
    ) -> Tuple[bool, Optional[BAFailureReason], List[str]]:
        """Run post-optimization validation suite.
        
        CRITERIA:
        1. Camera state validity: all poses finite and rotations valid SO(3).
        2. Landmark state validity: all landmarks strictly finite.
        3. Gauge preservation:
           - Camera 0 remains at identity origin [I | 0].
           - Camera 1 baseline magnitude remains 1.0 within numerical tolerance.
        4. Positive depth: all observations have positive optical depth (Z_c > 0).
        5. Observation count: valid observation count is preserved.
        6. Primary optimization acceptance: robust cost must not increase (cost_after <= cost_before + epsilon).
        7. Reprojection RMSE sanity check: heuristic divergence ceiling (HEURISTIC_DEFAULT).
        """
        diagnostics: List[str] = []

        # 1. Finite camera poses
        for cid, cam in cameras.items():
            if not (np.all(np.isfinite(cam.R_cw)) and np.all(np.isfinite(cam.t_cw))):
                diagnostics.append(f"Non-finite camera pose detected for {cid}.")
                return False, BAFailureReason.INVALID_CAMERA_STATE, diagnostics

        # 2. Finite landmarks
        for tid, pt in landmarks.items():
            if not np.all(np.isfinite(pt)):
                diagnostics.append(f"Non-finite 3D landmark coordinates for track {tid}.")
                return False, BAFailureReason.INVALID_LANDMARK_STATE, diagnostics

        # 3. Gauge preservation
        # A. Reference camera at origin
        if ref_camera_id in cameras:
            ref_cam = cameras[ref_camera_id]
            if not (np.allclose(ref_cam.R_cw, np.eye(3), atol=1e-5) and np.allclose(ref_cam.t_cw, np.zeros(3), atol=1e-5)):
                diagnostics.append("Gauge constraint violated: Reference camera pose drifted from origin [I | 0].")
                return False, BAFailureReason.GAUGE_CONSTRAINT_INVALID, diagnostics

        # B. Camera 1 unit baseline norm
        cam_ids = sorted(list(cameras.keys()))
        if len(cam_ids) >= 2:
            cam1_id = cam_ids[1]
            norm_t1 = float(np.linalg.norm(cameras[cam1_id].t_cw))
            if abs(norm_t1 - 1.0) > 1e-4:
                diagnostics.append(f"Gauge constraint violated: Camera 1 baseline norm is {norm_t1:.5f} (expected 1.0).")
                return False, BAFailureReason.GAUGE_CONSTRAINT_INVALID, diagnostics

        # 4. Cheirality check: Landmarks must have positive optical depth in observing cameras
        non_positive_depth_count = 0
        for tid, track in tracks.items():
            if tid not in landmarks:
                continue
            pt = landmarks[tid]
            for cid in track.observations:
                if cid in cameras:
                    _, z_c = BAResidualCalculator.project_point(cameras[cid].intrinsics, cameras[cid].R_cw, cameras[cid].t_cw, pt)
                    if z_c <= 1e-6:
                        non_positive_depth_count += 1

        if non_positive_depth_count > 0:
            diagnostics.append(f"Cheirality violation: {non_positive_depth_count} observations have non-positive depth.")
            return False, BAFailureReason.POST_OPTIMIZATION_VALIDATION_FAILED, diagnostics

        # 5. Valid observation count preserved
        if metrics_after.total_observations < metrics_before.total_observations:
            diagnostics.append(
                f"Observation count dropped: {metrics_after.total_observations} vs {metrics_before.total_observations} before."
            )
            return False, BAFailureReason.POST_OPTIMIZATION_VALIDATION_FAILED, diagnostics

        # 6. Primary optimization acceptance: robust cost must not increase
        if cost_after > cost_before + 1e-4:
            diagnostics.append(
                f"Optimization diverged: Robust cost increased from {cost_before:.4f} to {cost_after:.4f}."
            )
            return False, BAFailureReason.OPTIMIZATION_DIVERGED, diagnostics

        # 7. Heuristic RMSE divergence ceiling (HEURISTIC_DEFAULT sanity check)
        if metrics_after.rmse_px > metrics_before.rmse_px + config.rmse_divergence_tolerance_px:
            diagnostics.append(
                f"HEURISTIC_DEFAULT sanity failure: Post-BA RMSE ({metrics_after.rmse_px:.2f}px) exceeds "
                f"Pre-BA RMSE ({metrics_before.rmse_px:.2f}px) by > {config.rmse_divergence_tolerance_px}px."
            )
            return False, BAFailureReason.OPTIMIZATION_DIVERGED, diagnostics

        return True, None, diagnostics


class IBundleAdjustmentOptimizer(ABC):
    """Abstract interface defining the execution protocol for BA solvers."""

    @abstractmethod
    def initialize(
        self,
        reconstruction: SparseReconstructionResult,
        config: BundleAdjustmentConfig,
    ) -> bool:
        """Initialize optimizer state from Phase 3C SparseReconstructionResult."""
        pass

    @abstractmethod
    def evaluate(self, params: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute residual vector and scalar robust cost for given parameter state Theta."""
        pass

    @abstractmethod
    def iterate(self) -> Tuple[bool, float, str]:
        """Perform one non-linear optimization step."""
        pass

    @abstractmethod
    def converged(self) -> Tuple[bool, str]:
        """Check whether convergence criteria are met."""
        pass

    @abstractmethod
    def finalize(self) -> BundleAdjustmentResult:
        """Assemble and return the typed BundleAdjustmentResult."""
        pass


class BundleAdjustmentEngine(IBundleAdjustmentOptimizer):
    """Production non-linear least-squares Bundle Adjustment optimizer using scipy least_squares (TRF).
    
    Uses block-sparse Jacobian pattern and minimal Lie algebra so(3) + S^2 tangent-space parameterization.
    """

    def __init__(self, config: Optional[BundleAdjustmentConfig] = None):
        self.config = config or BundleAdjustmentConfig()
        self.param_manager: Optional[BAParameterManager] = None
        self.base_cameras: Dict[str, SfMCamera] = {}
        self.tracks: Dict[int, SfMTrack] = {}
        self.reconstruction: Optional[SparseReconstructionResult] = None
        self.metrics_before: Optional[BAReprojectionMetrics] = None
        self.cost_before: float = 0.0
        self.solver_info: Dict[str, Any] = {}

    def initialize(
        self,
        reconstruction: SparseReconstructionResult,
        config: Optional[BundleAdjustmentConfig] = None,
        tracks: Optional[Dict[int, SfMTrack]] = None,
        intrinsics_map: Optional[Dict[str, CameraIntrinsics]] = None,
    ) -> bool:
        """Initialize optimizer state and parameter layout from sparse reconstruction."""
        if config is not None:
            self.config = config
        self.reconstruction = reconstruction

        cam_ids = sorted(list(reconstruction.camera_poses.keys()))
        if len(cam_ids) < self.config.min_registered_cameras:
            return False

        # Build base cameras
        self.base_cameras = {}
        for idx, cid in enumerate(cam_ids):
            pose = reconstruction.camera_poses[cid]
            R_cw = np.array(pose.rotation_matrix, dtype=np.float64)
            c_w = np.array(pose.translation_vector, dtype=np.float64)
            t_cw = -R_cw @ c_w

            if intrinsics_map and cid in intrinsics_map:
                K = intrinsics_map[cid]
            elif cid in reconstruction.intrinsics:
                K = reconstruction.intrinsics[cid]
            else:
                K = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000)
            self.base_cameras[cid] = SfMCamera(
                frame_id=cid,
                R_cw=R_cw,
                t_cw=t_cw,
                intrinsics=K,
                is_registered=True,
                registration_order=idx,
            )

        # Build tracks
        self.tracks = {}
        if tracks is not None:
            self.tracks = dict(tracks)
        else:
            for tid, t_track in reconstruction.points3d.items():
                self.tracks[tid] = SfMTrack(
                    track_id=tid,
                    world_point=np.array(t_track.world_point, dtype=np.float64),
                    observations=dict(t_track.observations),
                    keypoint_indices={cid: tid for cid in t_track.observations},
                )

        if len(self.tracks) < self.config.min_landmarks:
            return False

        track_ids = sorted(list(self.tracks.keys()))
        ref_cam1_dir = self.base_cameras[cam_ids[1]].t_cw if len(cam_ids) > 1 else None
        self.param_manager = BAParameterManager(
            camera_order=cam_ids,
            track_order=track_ids,
            config=self.config,
            ref_cam1_direction=ref_cam1_dir,
        )
        return True

    def build_sparse_jacobian_pattern(self) -> Any:
        """Construct observation-based block-sparse Jacobian pattern for scipy least_squares."""
        if self.param_manager is None:
            raise ValueError("Parameter manager not initialized.")

        layout = self.param_manager.layout
        total_dim = layout.total_dimension

        # Count total residual dimensions
        num_obs = 0
        for tid in layout.track_order:
            trk = self.tracks[tid]
            for cid in layout.camera_order:
                if cid in trk.observations:
                    num_obs += 1

        num_residuals = num_obs * 2
        sparsity = sp.lil_matrix((num_residuals, total_dim), dtype=int)

        r_idx = 0
        for tid in layout.track_order:
            trk = self.tracks[tid]
            lm_offset = layout.landmark_offsets[tid]

            for cid in layout.camera_order:
                if cid not in trk.observations:
                    continue

                # Camera parameters
                if cid in layout.camera_offsets:
                    c_offset = layout.camera_offsets[cid]
                    c_len = 5 if cid == layout.camera_order[1] else 6
                    sparsity[r_idx:r_idx + 2, c_offset:c_offset + c_len] = 1

                # Landmark parameters (3)
                sparsity[r_idx:r_idx + 2, lm_offset:lm_offset + 3] = 1
                r_idx += 2

        return sparsity.tocsr()

    def evaluate(self, params: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute residual vector and scalar cost for given parameter state Theta."""
        if self.param_manager is None:
            raise ValueError("Parameter manager not initialized.")
        residuals, _, cost, _, _, _ = BAResidualEvaluator.evaluate(
            params, self.param_manager, self.base_cameras, self.tracks, self.config.huber_delta_px
        )
        return residuals, cost

    def iterate(self) -> Tuple[bool, float, str]:
        """Single iteration step wrapper (managed internally by TRF solver)."""
        return True, 0.0, "TRF_SOLVER"

    def converged(self) -> Tuple[bool, str]:
        """Check convergence status."""
        return True, "CONVERGED"

    def finalize(self) -> BundleAdjustmentResult:
        """Return result after optimization."""
        if self.reconstruction is None or self.metrics_before is None:
            raise ValueError("Optimizer not run.")
        return BundleAdjustmentResult(
            status=PipelineStageStatus.SUCCESS,
            refined_reconstruction=self.reconstruction,
            metrics_before=self.metrics_before,
            metrics_after=self.metrics_before,
            cost_before=self.cost_before,
            cost_after=self.cost_before,
            total_iterations=0,
            convergence_reason="NOOP",
            gauge_preserved=True,
        )

    def optimize(
        self,
        reconstruction: SparseReconstructionResult,
        tracks: Optional[Dict[int, SfMTrack]] = None,
        intrinsics_map: Optional[Dict[str, CameraIntrinsics]] = None,
        config: Optional[BundleAdjustmentConfig] = None,
    ) -> BundleAdjustmentResult:
        """Execute full nonlinear Bundle Adjustment optimization with safety gates and rollback."""
        cfg = config or self.config
        t_start = time.perf_counter()

        # 1. Precondition validation: camera count
        cam_ids = sorted(list(reconstruction.camera_poses.keys()))
        if len(cam_ids) < cfg.min_registered_cameras:
            m_empty = BAReprojectionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
            return BundleAdjustmentResult(
                status=PipelineStageStatus.FAILED,
                refined_reconstruction=None,
                metrics_before=m_empty,
                metrics_after=None,
                cost_before=0.0,
                cost_after=0.0,
                total_iterations=0,
                convergence_reason="INSUFFICIENT_CAMERAS",
                gauge_preserved=False,
                failure_reason=BAFailureReason.INVALID_INPUT_RECONSTRUCTION,
                diagnostics=[f"Need at least {cfg.min_registered_cameras} cameras, got {len(cam_ids)}."],
            )

        # 2. Precondition validation: landmark count
        if len(reconstruction.points3d) < cfg.min_landmarks:
            m_empty = BAReprojectionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
            return BundleAdjustmentResult(
                status=PipelineStageStatus.FAILED,
                refined_reconstruction=None,
                metrics_before=m_empty,
                metrics_after=None,
                cost_before=0.0,
                cost_after=0.0,
                total_iterations=0,
                convergence_reason="INSUFFICIENT_LANDMARKS",
                gauge_preserved=False,
                failure_reason=BAFailureReason.INSUFFICIENT_OBSERVATIONS,
                diagnostics=[f"Need at least {cfg.min_landmarks} landmarks, got {len(reconstruction.points3d)}."],
            )

        # 3. Precondition validation: Camera 0 gauge
        ref_pose = reconstruction.camera_poses[cam_ids[0]]
        R0 = np.array(ref_pose.rotation_matrix)
        c0 = np.array(ref_pose.translation_vector)
        if not (np.allclose(R0, np.eye(3), atol=1e-3) and np.allclose(c0, np.zeros(3), atol=1e-3)):
            m_empty = BAReprojectionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
            return BundleAdjustmentResult(
                status=PipelineStageStatus.FAILED,
                refined_reconstruction=None,
                metrics_before=m_empty,
                metrics_after=None,
                cost_before=0.0,
                cost_after=0.0,
                total_iterations=0,
                convergence_reason="GAUGE_VIOLATED_INITIAL",
                gauge_preserved=False,
                failure_reason=BAFailureReason.GAUGE_CONSTRAINT_INVALID,
                diagnostics=["Camera 0 is not aligned with origin [I | 0]."],
            )

        # 4. Precondition validation: Camera 1 unit baseline magnitude
        cam1_pose = reconstruction.camera_poses[cam_ids[1]]
        R1 = np.array(cam1_pose.rotation_matrix)
        c1 = np.array(cam1_pose.translation_vector)
        t1 = -R1 @ c1
        norm_t1 = float(np.linalg.norm(t1))
        if abs(norm_t1 - 1.0) > 1e-2:
            m_empty = BAReprojectionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
            return BundleAdjustmentResult(
                status=PipelineStageStatus.FAILED,
                refined_reconstruction=None,
                metrics_before=m_empty,
                metrics_after=None,
                cost_before=0.0,
                cost_after=0.0,
                total_iterations=0,
                convergence_reason="GAUGE_SCALE_VIOLATED_INITIAL",
                gauge_preserved=False,
                failure_reason=BAFailureReason.GAUGE_CONSTRAINT_INVALID,
                diagnostics=[f"Camera 1 baseline norm {norm_t1:.4f} != 1.0."],
            )

        # 5. Precondition validation: parameter finiteness
        for cid, p in reconstruction.camera_poses.items():
            if not (np.all(np.isfinite(p.rotation_matrix)) and np.all(np.isfinite(p.translation_vector))):
                m_empty = BAReprojectionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
                return BundleAdjustmentResult(
                    status=PipelineStageStatus.FAILED,
                    refined_reconstruction=None,
                    metrics_before=m_empty,
                    metrics_after=None,
                    cost_before=0.0,
                    cost_after=0.0,
                    total_iterations=0,
                    convergence_reason="NON_FINITE_CAMERA",
                    gauge_preserved=False,
                    failure_reason=BAFailureReason.INVALID_CAMERA_STATE,
                    diagnostics=[f"Camera {cid} contains non-finite values."],
                )

        for tid, pt in reconstruction.points3d.items():
            if not np.all(np.isfinite(pt.world_point)):
                m_empty = BAReprojectionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
                return BundleAdjustmentResult(
                    status=PipelineStageStatus.FAILED,
                    refined_reconstruction=None,
                    metrics_before=m_empty,
                    metrics_after=None,
                    cost_before=0.0,
                    cost_after=0.0,
                    total_iterations=0,
                    convergence_reason="NON_FINITE_LANDMARK",
                    gauge_preserved=False,
                    failure_reason=BAFailureReason.INVALID_LANDMARK_STATE,
                    diagnostics=[f"Landmark {tid} contains non-finite coordinates."],
                )

        # Initialize engine
        init_ok = self.initialize(reconstruction, cfg, tracks, intrinsics_map)
        if not init_ok or self.param_manager is None:
            m_empty = BAReprojectionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
            return BundleAdjustmentResult(
                status=PipelineStageStatus.FAILED,
                refined_reconstruction=None,
                metrics_before=m_empty,
                metrics_after=None,
                cost_before=0.0,
                cost_after=0.0,
                total_iterations=0,
                convergence_reason="INITIALIZATION_FAILED",
                gauge_preserved=False,
                failure_reason=BAFailureReason.INVALID_INPUT_RECONSTRUCTION,
                diagnostics=["Failed to initialize parameter manager."],
            )

        # 6. Evaluate initial state
        p0 = self.param_manager.pack_parameters(
            self.base_cameras, {t: trk.world_point for t, trk in self.tracks.items()}
        )
        res0, norms0, cost_before, val_cnt0, pos_z0, inv_z0 = BAResidualEvaluator.evaluate(
            p0, self.param_manager, self.base_cameras, self.tracks, cfg.huber_delta_px
        )

        if inv_z0 > 0:
            m_before_fail = BAReprojectionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, val_cnt0)
            return BundleAdjustmentResult(
                status=PipelineStageStatus.FAILED,
                refined_reconstruction=None,
                metrics_before=m_before_fail,
                metrics_after=None,
                cost_before=cost_before,
                cost_after=cost_before,
                total_iterations=0,
                convergence_reason="INITIAL_INVALID_DEPTH",
                gauge_preserved=False,
                failure_reason=BAFailureReason.PROJECTION_FAILURE,
                diagnostics=[f"Initial reconstruction has {inv_z0} non-positive depth observations."],
            )

        mean_e0 = float(np.mean(norms0)) if len(norms0) > 0 else 0.0
        rmse_e0 = math.sqrt(float(np.mean(norms0**2))) if len(norms0) > 0 else 0.0
        med_e0 = float(np.median(norms0)) if len(norms0) > 0 else 0.0
        p90_e0 = float(np.percentile(norms0, 90)) if len(norms0) > 0 else 0.0
        max_e0 = float(np.max(norms0)) if len(norms0) > 0 else 0.0

        metrics_before = BAReprojectionMetrics(
            mean_error_px=mean_e0,
            rmse_px=rmse_e0,
            median_error_px=med_e0,
            percentile_90_px=p90_e0,
            max_error_px=max_e0,
            total_observations=len(norms0),
        )
        self.metrics_before = metrics_before
        self.cost_before = cost_before

        # 7. Build sparse Jacobian pattern
        jac_sparsity = self.build_sparse_jacobian_pattern()

        # 8. Define residual function for least_squares
        def residual_fun(params: np.ndarray) -> np.ndarray:
            assert self.param_manager is not None
            r, _, _, _, _, _ = BAResidualEvaluator.evaluate(
                params, self.param_manager, self.base_cameras, self.tracks, cfg.huber_delta_px
            )
            return r

        # 9. Execute non-linear solver (TRF)
        try:
            opt_res = least_squares(
                residual_fun,
                p0,
                jac_sparsity=jac_sparsity,
                method="trf",
                loss="huber",
                f_scale=cfg.huber_delta_px,
                ftol=cfg.cost_tolerance,
                xtol=cfg.parameter_tolerance,
                gtol=cfg.gradient_tolerance,
                max_nfev=cfg.max_iterations * 10,
                verbose=0,
            )
        except Exception as e:
            return BundleAdjustmentResult(
                status=PipelineStageStatus.FAILED,
                refined_reconstruction=None,
                metrics_before=metrics_before,
                metrics_after=None,
                cost_before=cost_before,
                cost_after=cost_before,
                total_iterations=0,
                convergence_reason="SOLVER_EXCEPTION",
                gauge_preserved=False,
                failure_reason=BAFailureReason.OPTIMIZATION_FAILED,
                diagnostics=[f"Solver encountered exception: {str(e)}"],
            )

        # 10. Unpack candidate parameters
        p_opt = opt_res.x
        cand_cams, cand_lms = self.param_manager.unpack_parameters(p_opt, self.base_cameras)

        # Evaluate candidate metrics
        res_opt, norms_opt, cost_after, val_cnt_opt, pos_z_opt, inv_z_opt = BAResidualEvaluator.evaluate(
            p_opt, self.param_manager, self.base_cameras, self.tracks, cfg.huber_delta_px
        )

        mean_e_opt = float(np.mean(norms_opt)) if len(norms_opt) > 0 else 0.0
        rmse_e_opt = math.sqrt(float(np.mean(norms_opt**2))) if len(norms_opt) > 0 else 0.0
        med_e_opt = float(np.median(norms_opt)) if len(norms_opt) > 0 else 0.0
        p90_e_opt = float(np.percentile(norms_opt, 90)) if len(norms_opt) > 0 else 0.0
        max_e_opt = float(np.max(norms_opt)) if len(norms_opt) > 0 else 0.0

        metrics_after = BAReprojectionMetrics(
            mean_error_px=mean_e_opt,
            rmse_px=rmse_e_opt,
            median_error_px=med_e_opt,
            percentile_90_px=p90_e_opt,
            max_error_px=max_e_opt,
            total_observations=len(norms_opt),
        )

        # 11. Run Post-Optimization Validation Suite
        ref_cid = cam_ids[0]
        valid, val_reason, diagnostics = BAPostOptimizationValidator.validate(
            cameras=cand_cams,
            landmarks=cand_lms,
            tracks=self.tracks,
            ref_camera_id=ref_cid,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            cost_before=cost_before,
            cost_after=cost_after,
            config=cfg,
        )

        runtime_s = float(time.perf_counter() - t_start)
        provenance = {
            "solver_method": "scipy_least_squares_trf",
            "parameter_dimension": len(p0),
            "observation_count": len(norms0),
            "nfev": int(opt_res.nfev),
            "njev": int(getattr(opt_res, "njev", 0)),
            "optimality": float(opt_res.optimality) if hasattr(opt_res, "optimality") else 0.0,
            "runtime_seconds": runtime_s,
            "cost_before": cost_before,
            "cost_after": cost_after,
        }

        # 12. Rollback safety: if validation failed or invalid depth detected, rollback!
        if not valid or inv_z_opt > 0:
            return BundleAdjustmentResult(
                status=PipelineStageStatus.FAILED,
                refined_reconstruction=None,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                cost_before=cost_before,
                cost_after=cost_after,
                total_iterations=int(opt_res.nfev),
                convergence_reason=opt_res.message,
                gauge_preserved=False,
                failure_reason=val_reason or BAFailureReason.POST_OPTIMIZATION_VALIDATION_FAILED,
                diagnostics=diagnostics,
                provenance=provenance,
            )

        # 13. Construct refined SparseReconstructionResult
        refined_camera_poses: Dict[str, ExtrinsicPose] = {}
        for cid, cam in cand_cams.items():
            refined_camera_poses[cid] = ExtrinsicPose(
                rotation_matrix=cam.R_cw.tolist(),
                translation_vector=cam.camera_center.tolist(),
                coordinate_convention="opencv_optical",
                scale_factor=1.0,
                is_metric=False,
            )

        refined_points3d: Dict[int, TriangulatedTrack] = {}
        for tid, pt_coords in cand_lms.items():
            orig_track = reconstruction.points3d[tid]
            refined_points3d[tid] = TriangulatedTrack(
                track_id=tid,
                world_point=pt_coords.copy(),
                observations=orig_track.observations,
                reprojection_errors=orig_track.reprojection_errors,
                cheirality_valid=True,
                triangulation_angle_deg=orig_track.triangulation_angle_deg,
                measurement_type=MeasurementType.ESTIMATED,
            )

        refined_reconstruction = SparseReconstructionResult(
            camera_poses=refined_camera_poses,
            intrinsics={cid: cam.intrinsics for cid, cam in cand_cams.items()},
            points3d=refined_points3d,
            mean_reprojection_rmse_px=metrics_after.rmse_px,
            percentile_90_reprojection_error_px=metrics_after.percentile_90_px,
            total_registered_cameras=len(cand_cams),
            total_triangulated_points=len(cand_lms),
            mean_track_length=float(np.mean([len(t.observations) for t in refined_points3d.values()])) if refined_points3d else 0.0,
            gauge_policy=cfg.gauge_policy,
            is_metric_scale=False,
            has_monocular_scale_ambiguity=True,
            registered_frame_ids=cam_ids,
            unregistered_frame_ids=reconstruction.unregistered_frame_ids,
            failed_frame_ids=reconstruction.failed_frame_ids,
            camera_centers={cid: cam.camera_center.tolist() for cid, cam in cand_cams.items()},
            status=PipelineStageStatus.SUCCESS,
            provenance={
                "ba_iterations": int(opt_res.nfev),
                "cost_reduction": float(cost_before - cost_after),
                "runtime_seconds": runtime_s,
            },
        )

        return BundleAdjustmentResult(
            status=PipelineStageStatus.SUCCESS,
            refined_reconstruction=refined_reconstruction,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            cost_before=cost_before,
            cost_after=cost_after,
            total_iterations=int(opt_res.nfev),
            convergence_reason=opt_res.message,
            gauge_preserved=True,
            is_metric_scale=False,
            has_monocular_scale_ambiguity=True,
            diagnostics=diagnostics,
            provenance=provenance,
        )
