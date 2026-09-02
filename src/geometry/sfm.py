"""Phase 3C: Incremental Structure-from-Motion (SfM).

Implements relative monocular incremental camera registration, 3D landmark track
maintenance, PnP pose estimation, multi-view triangulation, and relative sparse
reconstruction with monocular scale ambiguity.

IMPORTANT SCIENTIFIC PRINCIPLES:
- Gauge Fixing: Camera 0 is fixed at origin [I | 0], Camera 1 translation is unit baseline (||t|| = 1.0).
- Scale Ambiguity: Monocular reconstruction is strictly SCALE_AMBIGUOUS without external metric ground truth.
- Translation vs Center: Camera pose transforms world points to camera coordinates: X_c = R_cw * X_w + t_cw.
  The camera optical center in world coordinates is C_w = -R_cw^T * t_cw.
- PnP Translation != Metric Scale: Translation solved via PnP is expressed in relative reconstruction units.
- Reprojection Error: Measured in pixels, serves as image-space geometric consistency diagnostic only.
- Global Bundle Adjustment belongs to Phase 3D.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Set

import cv2
import numpy as np

from src.geometry.contracts import (
    EvaluationLevel,
    PipelineStageStatus,
    MeasurementType,
    GaugeFixingPolicy,
    GeometryFailureReason,
    CameraIntrinsics,
    ExtrinsicPose,
    FeatureCorrespondences,
    TwoViewGeometryResult,
    TriangulatedTrack,
    SparseReconstructionResult,
)


@dataclass(frozen=True)
class SfMConfig:
    """Configurable heuristic engineering defaults (HEURISTIC_DEFAULT) for incremental SfM.
    
    NOTE ON PNP THRESHOLDS:
    - min_candidate_correspondences (default 6) is a configurable HEURISTIC_DEFAULT minimum
      candidate correspondence threshold to consider a frame for PnP registration in this pipeline.
      It is distinct from solver-specific minimal sampling configurations (such as the classic P3P
      algebraic formulation which samples 3 points and uses a 4th point for solution disambiguation,
      or uncalibrated 11-DoF DLT which samples 6 points), practical RANSAC requirements, and project
      acceptance thresholds (min_pnp_inliers = 15).
    - These numbers (3, 4, 6, 15) reflect specific solver formulations and engineering heuristics,
      NOT universal mathematical laws for all PnP methods.
    - All numerical thresholds in this configuration are engineering heuristics.
    """
    min_candidate_correspondences: int = 6             # HEURISTIC_DEFAULT: Minimum 2D-3D correspondences to consider a frame
    min_pnp_inliers: int = 15                          # HEURISTIC_DEFAULT: Minimum 2D-3D inliers to accept camera pose
    min_pnp_inlier_ratio: float = 0.25                 # HEURISTIC_DEFAULT: Minimum inlier ratio for PnP RANSAC
    pnp_reprojection_threshold_px: float = 4.0         # HEURISTIC_DEFAULT: PnP RANSAC inlier threshold in pixels
    pnp_confidence: float = 0.999                     # HEURISTIC_DEFAULT: PnP RANSAC confidence
    max_pnp_iterations: int = 2000                     # HEURISTIC_DEFAULT: Max RANSAC iterations for PnP
    planar_svd_ratio_threshold: float = 1e-4          # HEURISTIC_DEFAULT: SVD singular value ratio (sigma_3/sigma_1) for PLANARITY_RISK
    min_triangulation_angle_deg: float = 1.0          # HEURISTIC_DEFAULT: Minimum ray parallax for valid triangulation
    max_reprojection_error_px: float = 4.0            # HEURISTIC_DEFAULT: Maximum acceptable reprojection residual in pixels
    min_track_observations: int = 2                   # HEURISTIC_DEFAULT: Minimum views to retain a landmark track
    min_registered_cameras_for_sparse: int = 2        # HEURISTIC_DEFAULT: Minimum cameras to declare sparse reconstruction
    min_points_for_sparse: int = 15                   # HEURISTIC_DEFAULT: Minimum 3D points for sparse reconstruction
    config_version: str = "SfMConfig_v1.0"


@dataclass(frozen=True)
class CandidateEvaluation:
    """Evaluation of an unregistered candidate frame for next-camera selection."""
    frame_id: str
    available_2d3d_correspondences: int
    estimated_registration_sufficiency: bool
    selection_reason: str


@dataclass
class SfMCamera:
    """Rigid 6-DoF camera in incremental reconstruction space."""
    frame_id: str
    R_cw: np.ndarray                                   # Shape (3, 3) SO(3) world-to-camera rotation
    t_cw: np.ndarray                                   # Shape (3,) world-to-camera translation vector
    intrinsics: CameraIntrinsics
    is_registered: bool = True
    registration_order: int = 0
    pnp_inlier_count: int = 0
    pnp_inlier_ratio: float = 0.0
    reprojection_rmse_px: float = 0.0

    @property
    def camera_center(self) -> np.ndarray:
        """Camera optical center in world/model coordinates: C_w = -R_cw^T * t_cw."""
        return -self.R_cw.T @ self.t_cw

    @property
    def R_wc(self) -> np.ndarray:
        """Camera-to-world rotation matrix: R_wc = R_cw^T."""
        return self.R_cw.T

    def project(self, point_world: np.ndarray) -> Tuple[np.ndarray, float]:
        """Project a 3D world point into the camera raster.
        
        Returns:
            pixel_coord: np.ndarray of shape (2,) [u, v]
            depth_z: float optical depth along camera principal axis Z_c
        """
        pt_c = self.R_cw @ point_world + self.t_cw
        z_c = float(pt_c[2])
        if z_c <= 1e-6:
            return np.array([-1.0, -1.0], dtype=np.float64), z_c

        u = self.intrinsics.fx * (pt_c[0] / z_c) + self.intrinsics.cx
        v = self.intrinsics.fy * (pt_c[1] / z_c) + self.intrinsics.cy
        return np.array([u, v], dtype=np.float64), z_c

    def to_extrinsic_pose(self) -> ExtrinsicPose:
        """Map to standard ExtrinsicPose contract."""
        return ExtrinsicPose(
            rotation_matrix=self.R_cw.tolist(),
            translation_vector=self.camera_center.tolist(),
            coordinate_convention="opencv_optical",
            scale_factor=1.0,
            is_metric=False,
        )


@dataclass
class SfMTrack:
    """A multi-view 3D landmark track maintaining physical observations."""
    track_id: int
    world_point: np.ndarray                           # Shape (3,) float64 in relative SfM coordinate frame
    observations: Dict[str, Tuple[float, float]]      # frame_id -> (u, v) pixel observation
    keypoint_indices: Dict[str, int]                  # frame_id -> feature index
    reprojection_errors: Dict[str, float] = field(default_factory=dict) # frame_id -> error px
    triangulation_angle_deg: float = 0.0
    cheirality_valid: bool = True
    is_valid: bool = True

    def to_triangulated_track(self) -> TriangulatedTrack:
        """Convert to typed contract TriangulatedTrack."""
        return TriangulatedTrack(
            track_id=self.track_id,
            world_point=self.world_point.copy(),
            observations=dict(self.observations),
            reprojection_errors=dict(self.reprojection_errors),
            cheirality_valid=self.cheirality_valid,
            triangulation_angle_deg=self.triangulation_angle_deg,
            measurement_type=MeasurementType.ESTIMATED,
        )


class MatchGraph:
    """Graph of verified pairwise feature correspondences across keyframes."""

    def __init__(self):
        # (frame_a, frame_b) -> FeatureCorrespondences
        self.edges: Dict[Tuple[str, str], FeatureCorrespondences] = {}
        # frame_id -> set of connected frame_ids
        self.adjacency: Dict[str, Set[str]] = {}

    def add_edge(
        self,
        frame_a_id: str,
        frame_b_id: str,
        correspondences: FeatureCorrespondences,
        inlier_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Add verified pairwise correspondences edge to match graph."""
        pts_a = np.asarray(correspondences.points_a, dtype=np.float64)
        pts_b = np.asarray(correspondences.points_b, dtype=np.float64)
        dists = np.asarray(correspondences.descriptor_distances, dtype=np.float64)

        if inlier_mask is not None and len(inlier_mask) == len(pts_a):
            mask_bool = (np.asarray(inlier_mask).ravel() == 1)
            pts_a = pts_a[mask_bool]
            pts_b = pts_b[mask_bool]
            dists = dists[mask_bool] if len(dists) == len(mask_bool) else np.zeros(len(pts_a))

        filtered_corr = FeatureCorrespondences(
            frame_a_id=frame_a_id,
            frame_b_id=frame_b_id,
            points_a=pts_a,
            points_b=pts_b,
            descriptor_distances=dists,
            match_count=len(pts_a),
            descriptor_type=correspondences.descriptor_type,
            provenance=dict(correspondences.provenance),
        )

        self.edges[(frame_a_id, frame_b_id)] = filtered_corr
        self.adjacency.setdefault(frame_a_id, set()).add(frame_b_id)
        self.adjacency.setdefault(frame_b_id, set()).add(frame_a_id)

    def get_edge(self, frame_a_id: str, frame_b_id: str) -> Optional[Tuple[FeatureCorrespondences, bool]]:
        """Retrieve correspondences between two frames. Returns (corr, is_reversed)."""
        if (frame_a_id, frame_b_id) in self.edges:
            return self.edges[(frame_a_id, frame_b_id)], False
        if (frame_b_id, frame_a_id) in self.edges:
            return self.edges[(frame_b_id, frame_a_id)], True
        return None

    def get_neighbors(self, frame_id: str) -> List[str]:
        """Return list of connected frame IDs sorted for determinism."""
        return sorted(list(self.adjacency.get(frame_id, set())))


class IncrementalSfMEngine:
    """Deterministic classical Incremental Structure-from-Motion engine."""

    def __init__(self, config: Optional[SfMConfig] = None):
        self.config = config or SfMConfig()
        self.cameras: Dict[str, SfMCamera] = {}
        self.tracks: Dict[int, SfMTrack] = {}
        self._next_track_id: int = 0
        self.last_candidate_evaluation: Optional[CandidateEvaluation] = None

    def initialize_two_view(
        self,
        two_view_result: TwoViewGeometryResult,
        correspondences: FeatureCorrespondences,
        intrinsics_map: Dict[str, CameraIntrinsics],
    ) -> bool:
        """Initialize the relative reconstruction coordinate frame from an accepted Phase 3B seed."""
        frame_a = two_view_result.frame_a_id
        frame_b = two_view_result.frame_b_id

        # 1. Validate Phase 3B input prerequisites
        if not two_view_result.has_calibrated_intrinsics:
            return False
        if two_view_result.e_status != "SUCCESS":
            return False
        if two_view_result.relative_rotation is None or two_view_result.relative_translation is None:
            return False
        if frame_a not in intrinsics_map or frame_b not in intrinsics_map:
            return False

        K_a = intrinsics_map[frame_a]
        K_b = intrinsics_map[frame_b]
        if not K_a.is_calibrated or not K_b.is_calibrated:
            return False

        # 2. Fix the initial gauge: Camera 0 at origin, Camera 1 at relative pose
        # World frame = Camera A optical frame
        R_a = np.eye(3, dtype=np.float64)
        t_a = np.zeros(3, dtype=np.float64)

        R_b = np.asarray(two_view_result.relative_rotation, dtype=np.float64)
        t_b = np.asarray(two_view_result.relative_translation, dtype=np.float64)
        # Enforce unit baseline for relative gauge
        norm_t = np.linalg.norm(t_b)
        if norm_t > 1e-8:
            t_b = t_b / norm_t

        cam_a = SfMCamera(
            frame_id=frame_a, R_cw=R_a, t_cw=t_a, intrinsics=K_a,
            is_registered=True, registration_order=0,
        )
        cam_b = SfMCamera(
            frame_id=frame_b, R_cw=R_b, t_cw=t_b, intrinsics=K_b,
            is_registered=True, registration_order=1,
            pnp_inlier_count=two_view_result.inlier_count,
            pnp_inlier_ratio=two_view_result.inlier_ratio,
        )

        self.cameras[frame_a] = cam_a
        self.cameras[frame_b] = cam_b

        # 3. Seed initial 3D landmark tracks using inliers
        pts_a = np.asarray(correspondences.points_a, dtype=np.float64)
        pts_b = np.asarray(correspondences.points_b, dtype=np.float64)
        mask = two_view_result.inlier_mask
        if mask is not None and len(mask) == len(pts_a):
            pts_a = pts_a[mask]
            pts_b = pts_b[mask]

        P_a = np.array(K_a.matrix_3x3, dtype=np.float64) @ np.column_stack((R_a, t_a))
        P_b = np.array(K_b.matrix_3x3, dtype=np.float64) @ np.column_stack((R_b, t_b))

        seeded_points = 0
        for i in range(len(pts_a)):
            pt_a = pts_a[i]
            pt_b = pts_b[i]

            pt_3d, ok, err_a, err_b, parallax = self._triangulate_point(pt_a, pt_b, P_a, P_b, cam_a, cam_b)
            if ok and pt_3d is not None:
                track = SfMTrack(
                    track_id=self._next_track_id,
                    world_point=pt_3d,
                    observations={frame_a: (float(pt_a[0]), float(pt_a[1])), frame_b: (float(pt_b[0]), float(pt_b[1]))},
                    keypoint_indices={frame_a: i, frame_b: i},
                    reprojection_errors={frame_a: err_a, frame_b: err_b},
                    triangulation_angle_deg=parallax,
                    cheirality_valid=True,
                    is_valid=True,
                )
                self.tracks[self._next_track_id] = track
                self._next_track_id += 1
                seeded_points += 1

        return seeded_points >= self.config.min_points_for_sparse

    def find_2d_3d_correspondences(
        self,
        candidate_frame_id: str,
        match_graph: MatchGraph,
    ) -> Tuple[np.ndarray, np.ndarray, List[int], List[int], List[str]]:
        """Construct unambiguous 2D-3D correspondences between candidate frame and existing tracks."""
        pts_3d: List[np.ndarray] = []
        pts_2d: List[Tuple[float, float]] = []
        track_ids: List[int] = []
        cand_indices: List[int] = []
        diagnostics: List[str] = []

        seen_cand_pts: Set[Tuple[int, int]] = set()
        seen_tracks: Set[int] = set()

        # Iterate over all registered cameras connected to this candidate frame
        for reg_frame_id in sorted(self.cameras.keys()):
            edge_info = match_graph.get_edge(reg_frame_id, candidate_frame_id)
            if edge_info is None:
                continue

            corr, is_reversed = edge_info
            pts_reg = corr.points_b if is_reversed else corr.points_a
            pts_cand = corr.points_a if is_reversed else corr.points_b

            # Map from (u, v) in reg_frame to existing track_id
            for track_id, track in self.tracks.items():
                if not track.is_valid or reg_frame_id not in track.observations:
                    continue

                u_reg, v_reg = track.observations[reg_frame_id]

                # Find matching feature index in pts_reg
                dists_to_track = np.hypot(pts_reg[:, 0] - u_reg, pts_reg[:, 1] - v_reg)
                closest_idx = int(np.argmin(dists_to_track))
                if dists_to_track[closest_idx] < 1.5:  # Feature matches within 1.5 pixels
                    cand_pt = (float(pts_cand[closest_idx, 0]), float(pts_cand[closest_idx, 1]))
                    cand_key = (round(cand_pt[0] * 10), round(cand_pt[1] * 10))

                    # Uniqueness checks (strictly avoid TRACK_CONFLICT)
                    if track_id in seen_tracks:
                        diagnostics.append(f"TRACK_CONFLICT: Track {track_id} already matched in candidate frame {candidate_frame_id}.")
                        continue
                    if cand_key in seen_cand_pts:
                        diagnostics.append(f"TRACK_CONFLICT: Candidate feature {cand_pt} matches multiple 3D tracks.")
                        continue

                    seen_tracks.add(track_id)
                    seen_cand_pts.add(cand_key)

                    pts_3d.append(track.world_point)
                    pts_2d.append(cand_pt)
                    track_ids.append(track_id)
                    cand_indices.append(closest_idx)

        if len(pts_3d) == 0:
            return np.empty((0, 3)), np.empty((0, 2)), [], [], diagnostics

        return np.array(pts_3d, dtype=np.float64), np.array(pts_2d, dtype=np.float64), track_ids, cand_indices, diagnostics

    def register_camera_pnp(
        self,
        candidate_frame_id: str,
        pts_3d: np.ndarray,
        pts_2d: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> Tuple[bool, Optional[SfMCamera], List[int], Optional[GeometryFailureReason], List[str]]:
        """Register candidate camera via robust PnP RANSAC.
        
        NOTE ON PNP THRESHOLDS:
        - min_candidate_correspondences (default 6) is a configurable HEURISTIC_DEFAULT minimum
          candidate correspondence threshold for this implementation.
        - Solver-specific minimal configurations (e.g. P3P needs 3 or 4 points), practical RANSAC
          requirements, and project acceptance thresholds (min_pnp_inliers = 15) are strictly distinguished.
        """
        diagnostics: List[str] = []
        n_pts = len(pts_3d)

        # 1. Candidate correspondence threshold check (HEURISTIC_DEFAULT)
        if n_pts < self.config.min_candidate_correspondences:
            diagnostics.append(
                f"Candidate correspondences {n_pts} < heuristic minimum candidate threshold "
                f"{self.config.min_candidate_correspondences} (INSUFFICIENT_2D_3D_CORRESPONDENCES)."
            )
            return False, None, [], GeometryFailureReason.INSUFFICIENT_2D_3D_CORRESPONDENCES, diagnostics

        if n_pts < self.config.min_pnp_inliers:
            diagnostics.append(
                f"Correspondence count {n_pts} < heuristic acceptance threshold {self.config.min_pnp_inliers}."
            )

        # 2. Planar configuration risk diagnostic (HEURISTIC_DEFAULT)
        # Planar 3D configurations do not mathematically invalidate PnP in general,
        # but pose a numerical conditioning and depth ambiguity risk (PLANARITY_RISK) for certain solvers.
        pts_centered = pts_3d - np.mean(pts_3d, axis=0)
        _, singular_vals, _ = np.linalg.svd(pts_centered)
        if len(singular_vals) >= 3 and (singular_vals[2] / max(1e-6, singular_vals[0])) < self.config.planar_svd_ratio_threshold:
            diagnostics.append(
                f"PLANARITY_RISK: Near-planar 3D point configuration detected (sigma_3/sigma_1 < {self.config.planar_svd_ratio_threshold}). "
                "May affect numerical conditioning of PnP pose estimation."
            )

        # 3. Solve PnP RANSAC
        K_mat = np.array(intrinsics.matrix_3x3, dtype=np.float64)
        dist_coeffs = np.array(
            [intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2, intrinsics.k3],
            dtype=np.float64,
        )

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d.reshape(-1, 1, 3).astype(np.float64),
            pts_2d.reshape(-1, 1, 2).astype(np.float64),
            K_mat,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            confidence=self.config.pnp_confidence,
            reprojectionError=self.config.pnp_reprojection_threshold_px,
            iterationsCount=self.config.max_pnp_iterations,
        )

        if not success or inliers is None or len(inliers) == 0:
            diagnostics.append("PnP RANSAC failed to converge.")
            return False, None, [], GeometryFailureReason.CAMERA_REGISTRATION_FAILED, diagnostics

        inlier_arr = np.asarray(inliers, dtype=np.int64).ravel()
        inlier_indices: List[int] = [int(x) for x in inlier_arr.tolist()]
        inlier_count = len(inlier_indices)
        inlier_ratio = float(inlier_count / max(1, n_pts))

        if inlier_count < self.config.min_pnp_inliers:
            diagnostics.append(f"PnP inlier count {inlier_count} < threshold {self.config.min_pnp_inliers}.")
            return False, None, inlier_indices, GeometryFailureReason.CAMERA_REGISTRATION_FAILED, diagnostics

        if inlier_ratio < self.config.min_pnp_inlier_ratio:
            diagnostics.append(f"PnP inlier ratio {inlier_ratio:.3f} < threshold {self.config.min_pnp_inlier_ratio:.3f}.")
            return False, None, inlier_indices, GeometryFailureReason.CAMERA_REGISTRATION_FAILED, diagnostics

        # 4. Extract rotation and translation
        R_cw, _ = cv2.Rodrigues(rvec)
        t_cw = tvec.ravel()

        # Compute reprojection RMSE on inliers
        inlier_pts3d = pts_3d[inlier_arr]
        inlier_pts2d = pts_2d[inlier_arr]

        errors = []
        for i in range(len(inlier_pts3d)):
            pt_c = R_cw @ inlier_pts3d[i] + t_cw
            if pt_c[2] > 1e-4:
                u = intrinsics.fx * (pt_c[0] / pt_c[2]) + intrinsics.cx
                v = intrinsics.fy * (pt_c[1] / pt_c[2]) + intrinsics.cy
                errors.append((u - inlier_pts2d[i, 0]) ** 2 + (v - inlier_pts2d[i, 1]) ** 2)

        rmse = math.sqrt(float(np.mean(errors))) if len(errors) > 0 else 999.0
        if rmse > self.config.max_reprojection_error_px:
            diagnostics.append(f"PnP inlier reprojection RMSE {rmse:.2f}px exceeds threshold {self.config.max_reprojection_error_px:.2f}px.")
            return False, None, inlier_indices, GeometryFailureReason.CAMERA_REGISTRATION_FAILED, diagnostics

        reg_order = len(self.cameras)
        camera = SfMCamera(
            frame_id=candidate_frame_id,
            R_cw=R_cw,
            t_cw=t_cw,
            intrinsics=intrinsics,
            is_registered=True,
            registration_order=reg_order,
            pnp_inlier_count=inlier_count,
            pnp_inlier_ratio=inlier_ratio,
            reprojection_rmse_px=rmse,
        )

        return True, camera, inlier_indices, None, diagnostics

    def triangulate_new_tracks(
        self,
        new_camera: SfMCamera,
        match_graph: MatchGraph,
    ) -> int:
        """Triangulate unmapped feature matches between new camera and all registered cameras."""
        new_frame_id = new_camera.frame_id
        P_new = np.array(new_camera.intrinsics.matrix_3x3, dtype=np.float64) @ np.column_stack((new_camera.R_cw, new_camera.t_cw))
        new_points_count = 0

        # Existing observed features in new_camera
        existing_obs_in_new = set()
        for track in self.tracks.values():
            if new_frame_id in track.observations:
                existing_obs_in_new.add(track.observations[new_frame_id])

        for reg_frame_id, reg_cam in sorted(self.cameras.items()):
            if reg_frame_id == new_frame_id:
                continue

            edge_info = match_graph.get_edge(reg_frame_id, new_frame_id)
            if edge_info is None:
                continue

            corr, is_reversed = edge_info
            pts_reg = corr.points_b if is_reversed else corr.points_a
            pts_new = corr.points_a if is_reversed else corr.points_b

            P_reg = np.array(reg_cam.intrinsics.matrix_3x3, dtype=np.float64) @ np.column_stack((reg_cam.R_cw, reg_cam.t_cw))

            for i in range(len(pts_reg)):
                pt_reg = pts_reg[i]
                pt_new = pts_new[i]
                pt_new_tuple = (float(pt_new[0]), float(pt_new[1]))

                if pt_new_tuple in existing_obs_in_new:
                    continue

                # Triangulate
                pt_3d, ok, err_reg, err_new, parallax = self._triangulate_point(
                    pt_reg, pt_new, P_reg, P_new, reg_cam, new_camera
                )

                if ok and pt_3d is not None:
                    track = SfMTrack(
                        track_id=self._next_track_id,
                        world_point=pt_3d,
                        observations={
                            reg_frame_id: (float(pt_reg[0]), float(pt_reg[1])),
                            new_frame_id: pt_new_tuple,
                        },
                        keypoint_indices={reg_frame_id: i, new_frame_id: i},
                        reprojection_errors={reg_frame_id: err_reg, new_frame_id: err_new},
                        triangulation_angle_deg=parallax,
                        cheirality_valid=True,
                        is_valid=True,
                    )
                    self.tracks[self._next_track_id] = track
                    self._next_track_id += 1
                    existing_obs_in_new.add(pt_new_tuple)
                    new_points_count += 1

        return new_points_count

    def update_existing_tracks(
        self,
        new_camera: SfMCamera,
        track_ids: List[int],
        pts_2d: np.ndarray,
        inlier_indices: List[int],
    ) -> None:
        """Update existing 3D tracks with new 2D observations from PnP inliers.
        
        TRACK INVARIANTS:
        - Invariant A: One keyframe cannot contribute multiple observations to the same track.
        - Invariant B: One keypoint in one keyframe cannot belong to multiple tracks.
        """
        new_frame_id = new_camera.frame_id
        seen_new_pts: Set[Tuple[int, int]] = set()

        for idx in inlier_indices:
            t_id = track_ids[idx]
            if t_id in self.tracks:
                track = self.tracks[t_id]
                u, v = float(pts_2d[idx, 0]), float(pts_2d[idx, 1])
                pt_key = (round(u * 10), round(v * 10))

                # Invariant A: Do not allow duplicate observation of same track in new_frame_id
                if new_frame_id in track.observations:
                    continue

                # Invariant B: Do not allow one keypoint to belong to multiple tracks
                if pt_key in seen_new_pts:
                    continue

                seen_new_pts.add(pt_key)
                track.observations[new_frame_id] = (u, v)

                # Compute reprojection error for this observation
                proj_pt, z_c = new_camera.project(track.world_point)
                if z_c > 1e-4:
                    err = float(np.hypot(proj_pt[0] - u, proj_pt[1] - v))
                    track.reprojection_errors[new_frame_id] = err

    def evaluate_candidates(
        self,
        unregistered_frame_ids: List[str],
        match_graph: MatchGraph,
    ) -> List[CandidateEvaluation]:
        """Evaluate candidate frames distinguishing eligibility from registration sufficiency."""
        evaluations: List[CandidateEvaluation] = []

        for frame_id in sorted(unregistered_frame_ids):
            pts_3d, _, _, _, _ = self.find_2d_3d_correspondences(frame_id, match_graph)
            n_corrs = len(pts_3d)
            is_eligible = (n_corrs >= self.config.min_candidate_correspondences)
            is_sufficient = (n_corrs >= self.config.min_pnp_inliers)

            if is_sufficient:
                reason = (
                    f"Candidate {frame_id} has {n_corrs} 2D-3D correspondences (>= {self.config.min_pnp_inliers}), "
                    "sufficient for robust PnP registration."
                )
            elif is_eligible:
                reason = (
                    f"Candidate {frame_id} has {n_corrs} 2D-3D correspondences (>= {self.config.min_candidate_correspondences} "
                    f"eligibility threshold, but < {self.config.min_pnp_inliers} sufficiency threshold). "
                    "Selected for consideration; registration may be rejected by PnP."
                )
            else:
                reason = (
                    f"Candidate {frame_id} has {n_corrs} 2D-3D correspondences (< {self.config.min_candidate_correspondences} "
                    "eligibility threshold). Ineligible for registration."
                )

            evaluations.append(
                CandidateEvaluation(
                    frame_id=frame_id,
                    available_2d3d_correspondences=n_corrs,
                    estimated_registration_sufficiency=is_sufficient,
                    selection_reason=reason,
                )
            )

        return evaluations

    def select_next_candidate_frame(
        self,
        unregistered_frame_ids: List[str],
        match_graph: MatchGraph,
    ) -> Optional[str]:
        """Deterministically select next candidate camera maximizing verified 2D-3D correspondences."""
        evaluations = self.evaluate_candidates(unregistered_frame_ids, match_graph)

        # Filter eligible candidates (>= min_candidate_correspondences)
        eligible = [e for e in evaluations if e.available_2d3d_correspondences >= self.config.min_candidate_correspondences]
        if not eligible:
            self.last_candidate_evaluation = None
            return None

        # Sort: max correspondences descending, frame_id ascending for deterministic tie-breaking
        eligible.sort(key=lambda e: (-e.available_2d3d_correspondences, e.frame_id))
        best_cand = eligible[0]
        self.last_candidate_evaluation = best_cand
        return best_cand.frame_id

    def reconstruct(
        self,
        keyframe_ids: List[str],
        intrinsics_map: Dict[str, CameraIntrinsics],
        match_graph: MatchGraph,
        initial_two_view: TwoViewGeometryResult,
        initial_correspondences: FeatureCorrespondences,
    ) -> SparseReconstructionResult:
        """Execute full incremental Structure-from-Motion pipeline."""
        diagnostics: List[str] = []
        registered_frame_ids: List[str] = []
        failed_frame_ids: List[str] = []

        # 1. Initialize two-view seed
        ok_init = self.initialize_two_view(initial_two_view, initial_correspondences, intrinsics_map)
        if not ok_init:
            return SparseReconstructionResult(
                camera_poses={},
                intrinsics={},
                points3d={},
                mean_reprojection_rmse_px=0.0,
                percentile_90_reprojection_error_px=0.0,
                total_registered_cameras=0,
                total_triangulated_points=0,
                mean_track_length=0.0,
                is_metric_scale=False,
                evaluation_level=EvaluationLevel.LEVEL_1_IMAGE_SPACE_CONSISTENCY,
                has_monocular_scale_ambiguity=True,
                gauge_policy=GaugeFixingPolicy.FIX_FIRST_CAMERA_AND_UNIT_BASELINE,
                status=PipelineStageStatus.FAILED,
                failure_reason=GeometryFailureReason.SPARSE_RECONSTRUCTION_INSUFFICIENT,
                diagnostics=["Two-view seed initialization failed."],
                provenance={"config_version": self.config.config_version},
            )

        registered_frame_ids.extend([initial_two_view.frame_a_id, initial_two_view.frame_b_id])
        unregistered = [fid for fid in keyframe_ids if fid not in registered_frame_ids]

        # 2. Incremental Registration Loop
        while len(unregistered) > 0:
            next_cand = self.select_next_candidate_frame(unregistered, match_graph)
            if next_cand is None:
                diagnostics.append(f"Reconstruction stalled with {len(unregistered)} frames remaining.")
                break

            unregistered.remove(next_cand)
            cand_intrinsics = intrinsics_map.get(next_cand)
            if cand_intrinsics is None or not cand_intrinsics.is_calibrated:
                failed_frame_ids.append(next_cand)
                diagnostics.append(f"Frame {next_cand} lacks valid calibration.")
                continue

            pts_3d, pts_2d, track_ids, _, corr_diag = self.find_2d_3d_correspondences(next_cand, match_graph)
            diagnostics.extend(corr_diag)

            reg_ok, new_cam, inlier_indices, fail_reason, pnp_diag = self.register_camera_pnp(
                next_cand, pts_3d, pts_2d, cand_intrinsics
            )
            diagnostics.extend(pnp_diag)

            if reg_ok and new_cam is not None:
                self.cameras[next_cand] = new_cam
                registered_frame_ids.append(next_cand)

                # Update existing tracks with observations
                self.update_existing_tracks(new_cam, track_ids, pts_2d, inlier_indices)

                # Triangulate newly visible points
                self.triangulate_new_tracks(new_cam, match_graph)
            else:
                failed_frame_ids.append(next_cand)
                diagnostics.append(f"Registration failed for frame {next_cand}: {fail_reason}.")

        # 3. Assemble final sparse reconstruction contract
        camera_poses = {fid: cam.to_extrinsic_pose() for fid, cam in self.cameras.items()}
        camera_centers = {fid: cam.camera_center.tolist() for fid, cam in self.cameras.items()}
        points3d_map = {t_id: track.to_triangulated_track() for t_id, track in self.tracks.items() if track.is_valid}

        # Reprojection error summary across all observations
        all_errors = []
        track_lengths = []
        for track in points3d_map.values():
            all_errors.extend(track.reprojection_errors.values())
            track_lengths.append(len(track.observations))

        mean_rmse = float(np.mean(all_errors)) if len(all_errors) > 0 else 0.0
        p90_err = float(np.percentile(all_errors, 90)) if len(all_errors) > 0 else 0.0
        mean_len = float(np.mean(track_lengths)) if len(track_lengths) > 0 else 0.0

        n_cams = len(self.cameras)
        n_pts = len(points3d_map)

        stage_status = PipelineStageStatus.SUCCESS
        failure_reason = None
        if n_cams < self.config.min_registered_cameras_for_sparse:
            stage_status = PipelineStageStatus.FAILED
            failure_reason = GeometryFailureReason.CAMERA_REGISTRATION_FAILED
        elif n_pts < self.config.min_points_for_sparse:
            stage_status = PipelineStageStatus.FAILED
            failure_reason = GeometryFailureReason.SPARSE_RECONSTRUCTION_INSUFFICIENT

        return SparseReconstructionResult(
            camera_poses=camera_poses,
            intrinsics=intrinsics_map,
            points3d=points3d_map,
            mean_reprojection_rmse_px=mean_rmse,
            percentile_90_reprojection_error_px=p90_err,
            total_registered_cameras=n_cams,
            total_triangulated_points=n_pts,
            mean_track_length=mean_len,
            is_metric_scale=False,
            evaluation_level=EvaluationLevel.LEVEL_1_IMAGE_SPACE_CONSISTENCY,
            has_monocular_scale_ambiguity=True,
            gauge_policy=GaugeFixingPolicy.FIX_FIRST_CAMERA_AND_UNIT_BASELINE,
            registered_frame_ids=registered_frame_ids,
            unregistered_frame_ids=unregistered,
            failed_frame_ids=failed_frame_ids,
            camera_centers=camera_centers,
            status=stage_status,
            failure_reason=failure_reason,
            diagnostics=diagnostics,
            provenance={
                "engine": "IncrementalSfMEngine",
                "config_version": self.config.config_version,
                "pnp_method": "SOLVEPNP_ITERATIVE",
                "triangulation_method": "DLT_SVD",
            },
        )

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------

    def _triangulate_point(
        self,
        pt_a: np.ndarray,
        pt_b: np.ndarray,
        P_a: np.ndarray,
        P_b: np.ndarray,
        cam_a: SfMCamera,
        cam_b: SfMCamera,
    ) -> Tuple[Optional[np.ndarray], bool, float, float, float]:
        """Triangulate a 3D point via DLT and validate cheirality, parallax, and residuals."""
        u1, v1 = pt_a[0], pt_a[1]
        u2, v2 = pt_b[0], pt_b[1]

        # Linear Direct Linear Transform (DLT) matrix
        A = np.array([
            u1 * P_a[2, :] - P_a[0, :],
            v1 * P_a[2, :] - P_a[1, :],
            u2 * P_b[2, :] - P_b[0, :],
            v2 * P_b[2, :] - P_b[1, :],
        ], dtype=np.float64)

        if not np.all(np.isfinite(A)):
            return None, False, 0.0, 0.0, 0.0

        try:
            _, _, Vt = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            return None, False, 0.0, 0.0, 0.0

        X_hom = Vt[3, :]

        # Check finite and non-degenerate scale
        if abs(X_hom[3]) < 1e-8:
            return None, False, 0.0, 0.0, 0.0

        X_w = X_hom[:3] / X_hom[3]
        if not np.all(np.isfinite(X_w)):
            return None, False, 0.0, 0.0, 0.0

        # Check positive optical depth (cheirality) in both cameras
        proj_a, z_a = cam_a.project(X_w)
        proj_b, z_b = cam_b.project(X_w)
        if z_a <= 1e-4 or z_b <= 1e-4:
            return None, False, 0.0, 0.0, 0.0

        # Check ray parallax angle
        ray_a = X_w - cam_a.camera_center
        ray_b = X_w - cam_b.camera_center
        norm_a = np.linalg.norm(ray_a)
        norm_b = np.linalg.norm(ray_b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return None, False, 0.0, 0.0, 0.0

        cos_angle = np.clip(np.dot(ray_a / norm_a, ray_b / norm_b), -1.0, 1.0)
        parallax_deg = float(np.degrees(np.arccos(cos_angle)))
        if parallax_deg < self.config.min_triangulation_angle_deg:
            return None, False, 0.0, 0.0, parallax_deg

        # Check reprojection residuals
        err_a = float(np.hypot(proj_a[0] - u1, proj_a[1] - v1))
        err_b = float(np.hypot(proj_b[0] - u2, proj_b[1] - v2))
        if err_a > self.config.max_reprojection_error_px or err_b > self.config.max_reprojection_error_px:
            return None, False, err_a, err_b, parallax_deg

        return X_w, True, err_a, err_b, parallax_deg
