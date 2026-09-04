"""Phase 3F: Authentic Synthetic Multi-View Scene Rendering Fixture.

TEST/DATASET GENERATION INFRASTRUCTURE ONLY.
This module is strictly external to the reconstruction pipeline.

Generates realistic 2D perspective renderings of a known 3D scene consisting
of textured non-coplanar geometric planar facets observed from multiple
calibrated camera viewpoints.

Reconstruction pipelines receiving output from this fixture receive ONLY:
- Rendered RGB numpy images
- CameraIntrinsics (fx, fy, cx, cy, width, height)

Hidden ground-truth 3D coordinates and true camera poses remain strictly
contained within the evaluation dictionary and NEVER enter the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from src.geometry.contracts import CameraIntrinsics, DistortionModel, DistortionStatus


def _create_high_contrast_texture(size: int = 300, seed: int = 42) -> np.ndarray:
    """Creates a feature-rich, high-contrast texture with unique non-repetitive geometric and alphanumeric markers."""
    np.random.seed(seed)
    tex = np.zeros((size, size, 3), dtype=np.uint8)

    # 1. Subtle smooth gradient background
    for r in range(size):
        for c in range(size):
            tex[r, c] = [int(r / size * 180 + 30), int(c / size * 180 + 30), 120]

    # 2. Add randomized, distinct geometric and text markers
    for i in range(150):
        x = int(np.random.randint(15, size - 15))
        y = int(np.random.randint(15, size - 15))
        color = (int(np.random.randint(50, 255)), int(np.random.randint(50, 255)), int(np.random.randint(50, 255)))
        s = i % 4
        if s == 0:
            cv2.circle(tex, (x, y), int(np.random.randint(4, 10)), color, -1)
        elif s == 1:
            cv2.rectangle(tex, (x - 6, y - 6), (x + 6, y + 6), color, -1)
        elif s == 2:
            cv2.putText(tex, f"{seed}_{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        else:
            cv2.drawMarker(tex, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=8, thickness=2)

    return tex


@dataclass
class Synthetic3DFacet:
    """A 3D planar quadrilateral facet with associated 2D texture."""
    corners_3d: np.ndarray  # Shape (4, 3) in world coordinates
    texture: np.ndarray     # Shape (H, W, 3) uint8


def create_ground_truth_scene() -> List[Synthetic3DFacet]:
    """Constructs a deterministic 3D scene of non-coplanar textured planar facets across 5 distinct depth layers."""
    facets = []

    # Layer 1: Foreground near left, depth Z ~ 4.2m - 4.5m
    f1_corners = np.array([
        [-1.6, -1.0, 4.2],
        [-0.5, -1.0, 4.5],
        [-0.5,  0.2, 4.5],
        [-1.6,  0.2, 4.2],
    ], dtype=np.float64)
    facets.append(Synthetic3DFacet(corners_3d=f1_corners, texture=_create_high_contrast_texture(300, seed=101)))

    # Layer 2: Near right, depth Z ~ 5.5m - 5.8m
    f2_corners = np.array([
        [ 0.4, -1.0, 5.5],
        [ 1.7, -1.0, 5.8],
        [ 1.7,  0.2, 5.8],
        [ 0.4,  0.2, 5.5],
    ], dtype=np.float64)
    facets.append(Synthetic3DFacet(corners_3d=f2_corners, texture=_create_high_contrast_texture(300, seed=102)))

    # Layer 3: Mid ground, slanted ramp, depth Z ~ 6.5m - 7.2m
    f3_corners = np.array([
        [-0.8,  0.6, 6.5],
        [ 0.8,  0.6, 6.5],
        [ 0.8,  1.8, 7.2],
        [-0.8,  1.8, 7.2],
    ], dtype=np.float64)
    facets.append(Synthetic3DFacet(corners_3d=f3_corners, texture=_create_high_contrast_texture(300, seed=103)))

    # Layer 4: Background center, depth Z ~ 8.5m
    f4_corners = np.array([
        [-1.5, -1.8, 8.5],
        [ 1.5, -1.8, 8.5],
        [ 1.5, -0.6, 8.5],
        [-1.5, -0.6, 8.5],
    ], dtype=np.float64)
    facets.append(Synthetic3DFacet(corners_3d=f4_corners, texture=_create_high_contrast_texture(300, seed=104)))

    # Layer 5: Far background rear wall, depth Z ~ 10.5m
    f5_corners = np.array([
        [-1.0, -1.5, 10.5],
        [ 1.0, -1.5, 10.5],
        [ 1.0,  0.5, 10.5],
        [-1.0,  0.5, 10.5],
    ], dtype=np.float64)
    facets.append(Synthetic3DFacet(corners_3d=f5_corners, texture=_create_high_contrast_texture(300, seed=105)))

    return facets


def render_camera_view(
    facets: List[Synthetic3DFacet],
    R_cw: np.ndarray,
    t_cw: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Renders perspective projection of textured 3D facets onto a 2D image canvas."""
    img = np.full((intrinsics.height, intrinsics.width, 3), 40, dtype=np.uint8)

    # Sort facets by median distance (painter's algorithm: back-to-front)
    def facet_depth(facet: Synthetic3DFacet) -> float:
        pts_c = (R_cw @ facet.corners_3d.T + t_cw.reshape(3, 1)).T
        return float(np.median(pts_c[:, 2]))

    sorted_facets = sorted(facets, key=facet_depth, reverse=True)

    for facet in sorted_facets:
        pts_3d = facet.corners_3d  # (4, 3)
        pts_c = (R_cw @ pts_3d.T + t_cw.reshape(3, 1)).T

        # Check all vertices have positive depth in front of camera
        if np.any(pts_c[:, 2] <= 0.1):
            continue

        # Project to image plane
        u = intrinsics.fx * (pts_c[:, 0] / pts_c[:, 2]) + intrinsics.cx
        v = intrinsics.fy * (pts_c[:, 1] / pts_c[:, 2]) + intrinsics.cy
        dst_pts = np.column_stack((u, v)).astype(np.float32)

        # Texture source corners (top-left, top-right, bottom-right, bottom-left)
        th, tw = facet.texture.shape[:2]
        src_pts = np.array([
            [0, 0],
            [tw - 1, 0],
            [tw - 1, th - 1],
            [0, th - 1],
        ], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            facet.texture,
            H,
            (intrinsics.width, intrinsics.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        mask = (warped > 0)
        img[mask] = warped[mask]

    return img


def generate_synthetic_multiview_dataset(
    n_views: int = 3,
    baseline_step: float = 0.35,
) -> Tuple[List[np.ndarray], CameraIntrinsics, Dict[str, Any]]:
    """Generates an authentic multi-view synthetic dataset.

    Returns:
    - rendered_images: List of RGB images (Input to reconstruction pipeline)
    - intrinsics: CameraIntrinsics (Input to reconstruction pipeline)
    - hidden_evaluation_truth: Ground truth 3D facets and poses (KEPT OUTSIDE RECONSTRUCTION)
    """
    intrinsics = CameraIntrinsics(
        fx=800.0,
        fy=800.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
        distortion_model=DistortionModel.NONE_RECTIFIED,
        distortion_status=DistortionStatus.RECTIFIED_ZERO_DISTORTION,
    )

    facets = create_ground_truth_scene()
    rendered_images: List[np.ndarray] = []
    hidden_poses: List[Dict[str, np.ndarray]] = []

    for i in range(n_views):
        # Camera translated horizontally along X axis
        tx = -i * baseline_step
        # Small yaw rotation to keep scene centered
        yaw = np.radians(-i * 2.5)
        R_cw = np.array([
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0,         1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ], dtype=np.float64)
        t_cw = np.array([tx * np.cos(yaw), 0.0, tx * np.sin(yaw)], dtype=np.float64)

        img = render_camera_view(facets, R_cw, t_cw, intrinsics)
        rendered_images.append(img)
        hidden_poses.append({"R_cw": R_cw, "t_cw": t_cw})

    hidden_evaluation_truth = {
        "true_3d_facets": facets,
        "true_camera_poses": hidden_poses,
        "n_views": n_views,
    }

    return rendered_images, intrinsics, hidden_evaluation_truth
