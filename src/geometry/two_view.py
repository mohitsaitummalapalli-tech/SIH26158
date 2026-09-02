"""Phase 3B: Two-View Geometry & Robust Geometric Verification.

Implements robust Fundamental and Essential matrix estimation, RANSAC outlier
rejection, relative pose hypothesis decomposition, cheirality validation, and
two-view geometric diagnostics.

IMPORTANT SCIENTIFIC PRINCIPLES:
- Fundamental Matrix F operates on uncalibrated pixel raster coordinates: x_2_px^T * F * x_1_px = 0.
- Essential Matrix E operates on calibrated normalized coordinates: x_2_norm^T * E * x_1_norm = 0.
- Relative translation direction is estimated with arbitrary unit norm (||t|| = 1.0).
- Monocular translation magnitude is strictly UNOBSERVABLE without external metric constraints.
- Relative reconstruction scale is SCALE_AMBIGUOUS.
- Successful two-view geometry does NOT imply successful full SfM or metric 3D reconstruction.
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np

from src.geometry.contracts import (
    MeasurementType,
    GeometryFailureReason,
    DistortionModel,
    DistortionStatus,
    CameraIntrinsics,
    FeatureCorrespondences,
    TwoViewGeometryResult,
)


@dataclass(frozen=True)
class TwoViewConfig:
    """Configurable heuristic engineering defaults (HEURISTIC_DEFAULT) for two-view geometry."""
    ransac_confidence: float = 0.999                  # HEURISTIC_DEFAULT
    ransac_threshold_px: float = 2.0                  # HEURISTIC_DEFAULT (F-matrix Sampson inlier threshold in pixels)
    ransac_threshold_norm: float = 0.002              # HEURISTIC_DEFAULT (E-matrix normalized coordinate inlier threshold)
    max_iterations: int = 2000                        # HEURISTIC_DEFAULT
    min_inlier_ratio: float = 0.20                    # HEURISTIC_DEFAULT
    min_inliers: int = 15                             # HEURISTIC_DEFAULT
    weak_baseline_parallax_deg: float = 1.0           # HEURISTIC_DEFAULT (Parallax angle below which baseline is weak)
    pure_rotation_parallax_deg: float = 0.5           # HEURISTIC_DEFAULT (Parallax angle below which pure rotation is flagged)
    min_cheirality_ratio: float = 0.65                # HEURISTIC_DEFAULT (Fraction of inliers requiring positive depth)
    homography_inlier_ratio_threshold: float = 0.85   # HEURISTIC_DEFAULT (Planar degeneracy indicator)
    config_version: str = "TwoViewConfig_v1.0"


class TwoViewGeometryEstimator:
    """Estimates robust two-view geometry (Fundamental and Essential pathways)."""

    def __init__(self, config: Optional[TwoViewConfig] = None):
        self.config = config or TwoViewConfig()

    def estimate_fundamental(
        self,
        correspondences: FeatureCorrespondences,
    ) -> TwoViewGeometryResult:
        """Estimate Fundamental Matrix F from uncalibrated pixel correspondences."""
        frame_a_id = correspondences.frame_a_id
        frame_b_id = correspondences.frame_b_id
        pts_a = np.asarray(correspondences.points_a, dtype=np.float64)
        pts_b = np.asarray(correspondences.points_b, dtype=np.float64)
        n_pts = len(pts_a)

        # 1. Minimum correspondences check (8-point algorithm requires >= 8 points)
        if n_pts < 8:
            return TwoViewGeometryResult(
                frame_a_id=frame_a_id,
                frame_b_id=frame_b_id,
                has_calibrated_intrinsics=False,
                input_correspondence_count=n_pts,
                model_used="FUNDAMENTAL_MATRIX",
                f_status="FAILED",
                failure_reason=GeometryFailureReason.INSUFFICIENT_MATCHES,
                diagnostics=[f"Correspondence count {n_pts} < minimum 8 required for Fundamental matrix."],
                provenance={"estimator": "cv2.findFundamentalMat", "config_version": self.config.config_version},
            )

        # 2. RANSAC Fundamental Matrix estimation
        F, inlier_mask = cv2.findFundamentalMat(
            pts_a,
            pts_b,
            cv2.FM_RANSAC,
            ransacReprojThreshold=self.config.ransac_threshold_px,
            confidence=self.config.ransac_confidence,
            maxIters=self.config.max_iterations,
        )

        if F is None or inlier_mask is None or F.shape != (3, 3):
            return TwoViewGeometryResult(
                frame_a_id=frame_a_id,
                frame_b_id=frame_b_id,
                has_calibrated_intrinsics=False,
                input_correspondence_count=n_pts,
                model_used="FUNDAMENTAL_MATRIX",
                f_status="FAILED",
                failure_reason=GeometryFailureReason.GEOMETRIC_VERIFICATION_FAILED,
                diagnostics=["RANSAC Fundamental matrix estimation failed to converge."],
                provenance={"estimator": "cv2.findFundamentalMat", "config_version": self.config.config_version},
            )

        mask_bool = (inlier_mask.ravel() == 1)
        inlier_count = int(np.sum(mask_bool))
        inlier_ratio = float(inlier_count / max(1, n_pts))

        # 3. Compute Sampson error residuals for inliers
        sampson_errors = self._compute_sampson_errors_f(pts_a[mask_bool], pts_b[mask_bool], F)
        mean_err = float(np.mean(sampson_errors)) if len(sampson_errors) > 0 else 0.0
        median_err = float(np.median(sampson_errors)) if len(sampson_errors) > 0 else 0.0

        # 4. Check failure thresholds
        failure_reason = None
        is_degenerate = False
        diagnostics = []

        if inlier_count < self.config.min_inliers:
            failure_reason = GeometryFailureReason.INSUFFICIENT_MATCHES
            diagnostics.append(f"Inlier count {inlier_count} < minimum threshold {self.config.min_inliers}.")
        elif inlier_ratio < self.config.min_inlier_ratio:
            failure_reason = GeometryFailureReason.GEOMETRIC_VERIFICATION_FAILED
            diagnostics.append(f"Inlier ratio {inlier_ratio:.3f} < minimum threshold {self.config.min_inlier_ratio:.3f}.")

        # 5. Check planar degeneracy via Homography fitting
        if inlier_count >= 8:
            H, h_mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, self.config.ransac_threshold_px)
            if H is not None and h_mask is not None:
                h_inlier_ratio = float(np.sum(h_mask.ravel() == 1) / max(1, n_pts))
                if h_inlier_ratio >= self.config.homography_inlier_ratio_threshold:
                    is_degenerate = True
                    diagnostics.append(
                        f"Planar configuration detected: Homography inlier ratio {h_inlier_ratio:.3f} exceeds threshold."
                    )

        return TwoViewGeometryResult(
            frame_a_id=frame_a_id,
            frame_b_id=frame_b_id,
            fundamental_matrix=F,
            essential_matrix=None,
            has_calibrated_intrinsics=False,
            input_correspondence_count=n_pts,
            inlier_mask=mask_bool,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            mean_reprojection_error_px=mean_err,
            mean_epipolar_residual=mean_err,
            median_epipolar_residual=median_err,
            is_degenerate=is_degenerate,
            model_used="FUNDAMENTAL_MATRIX",
            f_status="SUCCESS" if failure_reason is None else "FAILED",
            failure_reason=failure_reason,
            diagnostics=diagnostics,
            provenance={
                "estimator": "cv2.findFundamentalMat",
                "ransac_threshold_px": self.config.ransac_threshold_px,
                "ransac_confidence": self.config.ransac_confidence,
                "config_version": self.config.config_version,
            },
        )

    def estimate_essential(
        self,
        correspondences: FeatureCorrespondences,
        intrinsics: CameraIntrinsics,
    ) -> TwoViewGeometryResult:
        """Estimate Essential Matrix E, recover relative pose hypotheses, and validate cheirality."""
        frame_a_id = correspondences.frame_a_id
        frame_b_id = correspondences.frame_b_id
        pts_a = np.asarray(correspondences.points_a, dtype=np.float64)
        pts_b = np.asarray(correspondences.points_b, dtype=np.float64)
        n_pts = len(pts_a)

        # 1. Calibration and distortion validation
        if not intrinsics.is_calibrated:
            return TwoViewGeometryResult(
                frame_a_id=frame_a_id,
                frame_b_id=frame_b_id,
                has_calibrated_intrinsics=False,
                input_correspondence_count=n_pts,
                model_used="ESSENTIAL_MATRIX",
                e_status="FAILED",
                failure_reason=GeometryFailureReason.CALIBRATION_UNAVAILABLE,
                diagnostics=[
                    f"Camera intrinsics not calibrated or unsupported distortion model: {intrinsics.distortion_status}."
                ],
                provenance={"config_version": self.config.config_version},
            )

        if n_pts < 8:
            return TwoViewGeometryResult(
                frame_a_id=frame_a_id,
                frame_b_id=frame_b_id,
                has_calibrated_intrinsics=True,
                input_correspondence_count=n_pts,
                model_used="ESSENTIAL_MATRIX",
                e_status="FAILED",
                failure_reason=GeometryFailureReason.INSUFFICIENT_MATCHES,
                diagnostics=[f"Correspondence count {n_pts} < minimum 8 required for Essential matrix."],
                provenance={"config_version": self.config.config_version},
            )

        # 2. Coordinate normalization with distortion handling
        pts_a_norm, ok_a, err_a = self._normalize_points(pts_a, intrinsics)
        pts_b_norm, ok_b, err_b = self._normalize_points(pts_b, intrinsics)

        if not ok_a or not ok_b:
            return TwoViewGeometryResult(
                frame_a_id=frame_a_id,
                frame_b_id=frame_b_id,
                has_calibrated_intrinsics=False,
                input_correspondence_count=n_pts,
                model_used="ESSENTIAL_MATRIX",
                e_status="FAILED",
                failure_reason=GeometryFailureReason.CALIBRATION_UNAVAILABLE,
                diagnostics=[f"Normalization error: {err_a or err_b}"],
                provenance={"config_version": self.config.config_version},
            )

        # 3. RANSAC Essential Matrix estimation on normalized coordinates
        # Using cv2.findEssentialMat with normalized coordinates (focal=1.0, pp=(0,0))
        E, inlier_mask = cv2.findEssentialMat(
            pts_a_norm,
            pts_b_norm,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=self.config.ransac_confidence,
            threshold=self.config.ransac_threshold_norm,
            maxIters=self.config.max_iterations,
        )

        if E is None or inlier_mask is None or E.shape != (3, 3):
            return TwoViewGeometryResult(
                frame_a_id=frame_a_id,
                frame_b_id=frame_b_id,
                has_calibrated_intrinsics=True,
                input_correspondence_count=n_pts,
                model_used="ESSENTIAL_MATRIX",
                e_status="FAILED",
                failure_reason=GeometryFailureReason.GEOMETRIC_VERIFICATION_FAILED,
                diagnostics=["RANSAC Essential matrix estimation failed to converge."],
                provenance={"estimator": "cv2.findEssentialMat", "config_version": self.config.config_version},
            )

        mask_bool = (inlier_mask.ravel() == 1)
        inlier_count = int(np.sum(mask_bool))
        inlier_ratio = float(inlier_count / max(1, n_pts))

        if inlier_count < self.config.min_inliers:
            return TwoViewGeometryResult(
                frame_a_id=frame_a_id,
                frame_b_id=frame_b_id,
                essential_matrix=E,
                has_calibrated_intrinsics=True,
                input_correspondence_count=n_pts,
                inlier_mask=mask_bool,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                model_used="ESSENTIAL_MATRIX",
                e_status="FAILED",
                failure_reason=GeometryFailureReason.INSUFFICIENT_MATCHES,
                diagnostics=[f"Inlier count {inlier_count} < minimum threshold {self.config.min_inliers}."],
                provenance={"estimator": "cv2.findEssentialMat", "config_version": self.config.config_version},
            )

        # 4. Decompose E into 4 pose hypotheses and test cheirality
        pts_a_inliers = pts_a_norm[mask_bool]
        pts_b_inliers = pts_b_norm[mask_bool]

        hypotheses = self._decompose_essential_matrix(E)
        R_first, t_first = hypotheses[0]
        best_hypo = (R_first, t_first, 0.0)
        max_cheirality_count = -1
        cheirality_counts = []
        parallaxes = []

        for R_cand, t_cand in hypotheses:
            passed_count, med_parallax = self._evaluate_cheirality(pts_a_inliers, pts_b_inliers, R_cand, t_cand)
            cheirality_counts.append(passed_count)
            parallaxes.append(med_parallax)
            if passed_count > max_cheirality_count:
                max_cheirality_count = passed_count
                best_hypo = (R_cand, t_cand, med_parallax)

        # Sort cheirality counts descending to check uniqueness
        sorted_counts = sorted(cheirality_counts, reverse=True)
        top1 = sorted_counts[0]
        top2 = sorted_counts[1] if len(sorted_counts) > 1 else 0

        cheirality_ratio = float(top1 / max(1, inlier_count))
        R_rel, t_rel, median_parallax = best_hypo

        # 5. Residual statistics on inliers (Sampson distance in normalized coordinates)
        sampson_errs = self._compute_sampson_errors_e(pts_a_inliers, pts_b_inliers, E)
        mean_err = float(np.mean(sampson_errs)) if len(sampson_errs) > 0 else 0.0
        median_err = float(np.median(sampson_errs)) if len(sampson_errs) > 0 else 0.0
        # Convert mean normalized error to approx pixel error: err_px ≈ err_norm * f
        f_mean = 0.5 * (intrinsics.fx + intrinsics.fy)
        mean_px_err = float(mean_err * f_mean)

        # 6. Condition and Failure Checks
        failure_reason = None
        is_degenerate = False
        diagnostics = []

        # Check cheirality sufficiency and ambiguity
        if cheirality_ratio < self.config.min_cheirality_ratio:
            failure_reason = GeometryFailureReason.CHEIRALITY_VIOLATION
            diagnostics.append(
                f"Cheirality ratio {cheirality_ratio:.3f} < minimum threshold {self.config.min_cheirality_ratio:.3f}."
            )
        elif top1 > 0 and top2 >= 0.90 * top1 and top1 < 0.85 * inlier_count:
            # Ambiguous pose: two hypotheses have nearly identical pass rates
            failure_reason = GeometryFailureReason.DEGENERATE_GEOMETRY
            is_degenerate = True
            diagnostics.append(f"Ambiguous pose recovery: top hypotheses counts ({top1}, {top2}) are indistinguishable.")

        # Check pure rotation and weak baseline parallax
        if median_parallax < self.config.pure_rotation_parallax_deg:
            failure_reason = GeometryFailureReason.PURE_ROTATION_RISK
            is_degenerate = True
            diagnostics.append(
                f"Pure rotation risk: Median parallax {median_parallax:.2f}° < threshold {self.config.pure_rotation_parallax_deg:.2f}°."
            )
        elif median_parallax < self.config.weak_baseline_parallax_deg:
            failure_reason = GeometryFailureReason.WEAK_BASELINE
            diagnostics.append(
                f"Weak baseline: Median parallax {median_parallax:.2f}° < threshold {self.config.weak_baseline_parallax_deg:.2f}°."
            )

        # Check inlier ratio
        if inlier_ratio < self.config.min_inlier_ratio and failure_reason is None:
            failure_reason = GeometryFailureReason.GEOMETRIC_VERIFICATION_FAILED
            diagnostics.append(f"Inlier ratio {inlier_ratio:.3f} < threshold {self.config.min_inlier_ratio:.3f}.")

        return TwoViewGeometryResult(
            frame_a_id=frame_a_id,
            frame_b_id=frame_b_id,
            essential_matrix=E,
            relative_rotation=R_rel,
            relative_translation=t_rel,
            has_calibrated_intrinsics=True,
            input_correspondence_count=n_pts,
            inlier_mask=mask_bool,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            mean_reprojection_error_px=mean_px_err,
            mean_epipolar_residual=mean_err,
            median_epipolar_residual=median_err,
            cheirality_passed_count=top1,
            cheirality_ratio=cheirality_ratio,
            median_parallax_deg=median_parallax,
            is_degenerate=is_degenerate,
            model_used="ESSENTIAL_MATRIX",
            e_status="SUCCESS" if failure_reason is None else "FAILED",
            scale_status="SCALE_AMBIGUOUS",
            translation_magnitude_status="UNOBSERVABLE",
            relative_rotation_measurement=MeasurementType.ESTIMATED,
            relative_translation_measurement=MeasurementType.ESTIMATED,
            failure_reason=failure_reason,
            diagnostics=diagnostics,
            provenance={
                "estimator": "cv2.findEssentialMat",
                "ransac_threshold_norm": self.config.ransac_threshold_norm,
                "ransac_confidence": self.config.ransac_confidence,
                "config_version": self.config.config_version,
            },
        )

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------

    def _normalize_points(
        self,
        points_px: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> Tuple[np.ndarray, bool, str]:
        """Convert pixel coordinates to normalized camera coordinates with distortion handling."""
        pts = np.asarray(points_px, dtype=np.float64)
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float64), True, ""

        if intrinsics.distortion_status == DistortionStatus.CALIBRATION_UNAVAILABLE:
            return np.empty((0, 2), dtype=np.float64), False, "CALIBRATION_UNAVAILABLE"

        if (
            intrinsics.distortion_status == DistortionStatus.UNSUPPORTED_MODEL
            or intrinsics.distortion_model == DistortionModel.UNSUPPORTED_UNKNOWN
        ):
            return np.empty((0, 2), dtype=np.float64), False, "UNSUPPORTED_DISTORTION_MODEL"

        # 1. NONE_RECTIFIED: Simple pinhole unprojection
        if intrinsics.distortion_model == DistortionModel.NONE_RECTIFIED:
            u_norm = (pts[:, 0] - intrinsics.cx) / intrinsics.fx
            v_norm = (pts[:, 1] - intrinsics.cy) / intrinsics.fy
            return np.column_stack((u_norm, v_norm)), True, ""

        # 2. BROWN_CONRADY_RADIAL_TANGENTIAL
        K = np.array(intrinsics.matrix_3x3, dtype=np.float64)
        dist = np.array([intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2, intrinsics.k3], dtype=np.float64)

        if intrinsics.distortion_model == DistortionModel.BROWN_CONRADY_RADIAL_TANGENTIAL:
            pts_cv = pts.reshape((-1, 1, 2)).astype(np.float32)
            undist = cv2.undistortPoints(pts_cv, K, dist)
            return undist.reshape((-1, 2)).astype(np.float64), True, ""

        # 3. FISHEYE_EQUIDISTANT
        if intrinsics.distortion_model == DistortionModel.FISHEYE_EQUIDISTANT:
            pts_cv = pts.reshape((-1, 1, 2)).astype(np.float32)
            D_fish = np.array([intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2], dtype=np.float64)
            undist = cv2.fisheye.undistortPoints(pts_cv, K, D_fish)
            return undist.reshape((-1, 2)).astype(np.float64), True, ""

        return np.empty((0, 2), dtype=np.float64), False, f"Unknown distortion model {intrinsics.distortion_model}"

    def _decompose_essential_matrix(self, E: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Decompose essential matrix into 4 valid (R, t) hypothesis tuples."""
        U, S, Vt = np.linalg.svd(E)

        # Enforce positive determinant on U and V
        if np.linalg.det(U) < 0:
            U[:, 2] *= -1
        if np.linalg.det(Vt) < 0:
            Vt[2, :] *= -1

        W = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ], dtype=np.float64)

        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt
        t = U[:, 2]
        t = t / np.linalg.norm(t)

        hypotheses = [
            (R1, t),
            (R1, -t),
            (R2, t),
            (R2, -t),
        ]
        return hypotheses

    def _evaluate_cheirality(
        self,
        pts_a_norm: np.ndarray,
        pts_b_norm: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
    ) -> Tuple[int, float]:
        """Triangulate points and count positive cheirality (Z_c1 > 0 and Z_c2 > 0)."""
        P1 = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float64)

        P2 = np.column_stack((R, t.reshape(3, 1)))
        C2 = -R.T @ t  # Camera 2 center in Camera 1 frame

        passed_count = 0
        parallaxes = []

        for i in range(len(pts_a_norm)):
            x1 = pts_a_norm[i, 0]
            y1 = pts_a_norm[i, 1]
            x2 = pts_b_norm[i, 0]
            y2 = pts_b_norm[i, 1]

            # Direct Linear Transform (DLT) triangulation
            A = np.array([
                x1 * P1[2, :] - P1[0, :],
                y1 * P1[2, :] - P1[1, :],
                x2 * P2[2, :] - P2[0, :],
                y2 * P2[2, :] - P2[1, :],
            ], dtype=np.float64)

            _, _, Vt_A = np.linalg.svd(A)
            X_hom = Vt_A[3, :]

            if abs(X_hom[3]) < 1e-8:
                continue

            X1 = X_hom[:3] / X_hom[3]
            z1 = X1[2]

            # Transform into camera 2 coordinates
            X2 = R @ X1 + t
            z2 = X2[2]

            if z1 > 1e-4 and z2 > 1e-4:
                passed_count += 1
                # Compute parallax angle
                norm1 = np.linalg.norm(X1)
                norm2 = np.linalg.norm(X1 - C2)
                if norm1 > 1e-6 and norm2 > 1e-6:
                    ray1 = X1 / norm1
                    ray2 = (X1 - C2) / norm2
                    cos_theta = np.clip(np.dot(ray1, ray2), -1.0, 1.0)
                    parallax_deg = float(np.degrees(np.arccos(cos_theta)))
                    parallaxes.append(parallax_deg)

        median_parallax = float(np.median(parallaxes)) if len(parallaxes) > 0 else 0.0
        return passed_count, median_parallax

    def _compute_sampson_errors_f(
        self,
        pts_a: np.ndarray,
        pts_b: np.ndarray,
        F: np.ndarray,
    ) -> np.ndarray:
        """Compute Sampson epipolar distance in pixels for points under Fundamental matrix F."""
        if len(pts_a) == 0:
            return np.empty((0,), dtype=np.float64)

        errors = []
        for i in range(len(pts_a)):
            xa = np.array([pts_a[i, 0], pts_a[i, 1], 1.0], dtype=np.float64)
            xb = np.array([pts_b[i, 0], pts_b[i, 1], 1.0], dtype=np.float64)

            num = float((xb @ F @ xa) ** 2)
            Fx1 = F @ xa
            Ft_x2 = F.T @ xb

            denom = float(Fx1[0] ** 2 + Fx1[1] ** 2 + Ft_x2[0] ** 2 + Ft_x2[1] ** 2)
            if denom > 1e-12:
                errors.append(math.sqrt(num / denom))
            else:
                errors.append(0.0)

        return np.array(errors, dtype=np.float64)

    def _compute_sampson_errors_e(
        self,
        pts_a_norm: np.ndarray,
        pts_b_norm: np.ndarray,
        E: np.ndarray,
    ) -> np.ndarray:
        """Compute Sampson epipolar error in normalized ray coordinates under Essential matrix E."""
        if len(pts_a_norm) == 0:
            return np.empty((0,), dtype=np.float64)

        errors = []
        for i in range(len(pts_a_norm)):
            xa = np.array([pts_a_norm[i, 0], pts_a_norm[i, 1], 1.0], dtype=np.float64)
            xb = np.array([pts_b_norm[i, 0], pts_b_norm[i, 1], 1.0], dtype=np.float64)

            num = float((xb @ E @ xa) ** 2)
            Ex1 = E @ xa
            Et_x2 = E.T @ xb

            denom = float(Ex1[0] ** 2 + Ex1[1] ** 2 + Et_x2[0] ** 2 + Et_x2[1] ** 2)
            if denom > 1e-12:
                errors.append(math.sqrt(num / denom))
            else:
                errors.append(0.0)

        return np.array(errors, dtype=np.float64)
