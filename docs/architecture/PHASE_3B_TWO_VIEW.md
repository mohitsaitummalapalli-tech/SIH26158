# Phase 3B: Two-View Geometry & Robust Geometric Verification Architecture

## 1. Executive Summary & Objective

Phase 3B implements the deterministic **Two-View Epipolar Geometry and Robust Geometric Verification** subsystem. It takes candidate 2D descriptor correspondences (`FeatureCorrespondences`) generated in Phase 3A, performs coordinate normalization with explicit lens distortion compensation, robustly estimates epipolar geometry models ($\mathbf{F}$ and $\mathbf{E}$), rejects outlier matches via RANSAC, decomposes the Essential matrix into relative camera pose hypotheses, and validates physical cheirality and triangulation parallax.

> **CRITICAL SCIENTIFIC PRINCIPLES & GUARDRAILS:**
> 1. **TWO-VIEW SUCCESS $\neq$ FULL SFM OR METRIC RECONSTRUCTION**:
>    - Successful two-view geometry establishes a relative two-view visual ray constraint.
>    - It does **not** imply successful incremental multi-view SfM, bundle adjustment convergence, or metric 3D point cloud accuracy.
> 2. **UNOBSERVABLE TRANSLATION MAGNITUDE & SCALE AMBIGUITY**:
>    - Under monocular pinhole camera geometry, the absolute baseline translation magnitude is strictly **`UNOBSERVABLE`**.
>    - The recovered translation vector is a unit direction only ($\|\mathbf{t}_{21}\| = 1.0$).
>    - The relative reconstruction scale is **`SCALE_AMBIGUOUS`** and requires independent physical baseline constraints or Ground Control Points (GCPs).
> 3. **FUNDAMENTAL VS. ESSENTIAL MATRIX CALIBRATION**:
>    - **Path A (Fundamental Matrix $\mathbf{F}$)**: Operates in uncalibrated pixel raster space ($\mathbf{x}_{2, \text{px}}^T \mathbf{F} \mathbf{x}_{1, \text{px}} = 0$). No metric camera pose can be derived from $\mathbf{F}$ alone.
>    - **Path B (Essential Matrix $\mathbf{E}$)**: Operates in calibrated normalized ray space ($\mathbf{x}_{2, \text{norm}}^T \mathbf{E} \mathbf{x}_{1, \text{norm}} = 0$). Requires valid, verified camera calibration (`CameraIntrinsics.is_calibrated == True`).
> 4. **TELEMETRY PROXY $\neq$ OPTICAL BASELINE**:
>    - Onboard drone GNSS/navigation telemetry positions are labeled `TRAJECTORY_PROXY`. They must not be conflated with the optical camera center baseline without explicit lever-arm calibration.

---

## 2. Pipeline Architecture

```
                 Phase 3A FeatureCorrespondences (pts_a, pts_b)
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
    [Path A: Uncalibrated]                                [Path B: Calibrated]
    Fundamental Matrix (F)                                Essential Matrix (E)
            │                                                     │
            ▼                                                     ▼
     RANSAC Estimation                                     Distortion Model
 (x_2_px^T * F * x_1_px = 0)                       (Rectified / Brown-Conrady / Fisheye)
            │                                                     │
            ▼                                                     ▼
  Sampson Pixel Residuals                             Normalized Coordinate Conversion
            │                                           (x_norm = K^-1 * x_undist)
            ▼                                                     │
    TwoViewGeometryResult                                         ▼
   (F-matrix inliers only)                                RANSAC Estimation
                                                      (x_2_norm^T * E * x_1_norm = 0)
                                                                  │
                                                                  ▼
                                                      Four-Way SVD Pose Decomposition
                                                      (R1, +t), (R1, -t), (R2, +t), (R2, -t)
                                                                  │
                                                                  ▼
                                                      Triangulation & Cheirality Test
                                                         (Z_c1 > 0  and  Z_c2 > 0)
                                                                  │
                                                                  ▼
                                                      Parallax & Degeneracy Diagnostics
                                                      (Weak Baseline / Pure Rotation)
                                                                  │
                                                                  ▼
                                                        TwoViewGeometryResult
                                                       (E-matrix, R_rel, t_rel direction)
```

---

## 3. Mathematical Formulations & Data Contracts

### 3.1 Uncalibrated Epipolar Geometry (Fundamental Matrix)
For correspondences $\mathbf{x}_{1, \text{px}} = [u_1, v_1, 1]^T$ and $\mathbf{x}_{2, \text{px}} = [u_2, v_2, 1]^T$ in pixel raster space:
$$\mathbf{x}_{2, \text{px}}^T \mathbf{F} \, \mathbf{x}_{1, \text{px}} = 0$$
where $\mathbf{F} \in \mathbb{R}^{3 \times 3}$ is a rank-2 matrix with $\det(\mathbf{F}) = 0$.
The first-order geometric residual is evaluated using the **Sampson Distance**:
$$d_{\text{Sampson}}^2(\mathbf{x}_1, \mathbf{x}_2; \mathbf{F}) = \frac{(\mathbf{x}_{2, \text{px}}^T \mathbf{F} \mathbf{x}_{1, \text{px}})^2}{(\mathbf{F} \mathbf{x}_{1, \text{px}})_1^2 + (\mathbf{F} \mathbf{x}_{1, \text{px}})_2^2 + (\mathbf{F}^T \mathbf{x}_{2, \text{px}})_1^2 + (\mathbf{F}^T \mathbf{x}_{2, \text{px}})_2^2} \quad [\text{px}^2]$$

### 3.2 Calibrated Epipolar Geometry (Essential Matrix)
When camera calibration $\mathbf{K}$ is available, pixel coordinates are mapped to normalized rays:
$$\mathbf{x}_{\text{norm}} = \mathbf{K}^{-1} \mathbf{x}_{\text{undist}}$$
The coplanarity constraint in normalized camera space is:
$$\mathbf{x}_{2, \text{norm}}^T \mathbf{E} \, \mathbf{x}_{1, \text{norm}} = 0 \quad \text{with } \mathbf{E} = [\mathbf{t}_{21}]_\times \mathbf{R}_{21} = \mathbf{K}_2^T \mathbf{F} \mathbf{K}_1$$
where singular values satisfy $\sigma_1 = \sigma_2 > 0, \; \sigma_3 = 0$.

### 3.3 SVD Relative Pose Recovery & Four-Way Decomposition
Singular Value Decomposition of $\mathbf{E} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$ with $\mathbf{W} = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$:
1. $\mathbf{R}_1 = \mathbf{U} \mathbf{W} \mathbf{V}^T, \quad \mathbf{t} = +\mathbf{U}[:, 2]$
2. $\mathbf{R}_1 = \mathbf{U} \mathbf{W} \mathbf{V}^T, \quad \mathbf{t} = -\mathbf{U}[:, 2]$
3. $\mathbf{R}_2 = \mathbf{U} \mathbf{W}^T \mathbf{V}^T, \quad \mathbf{t} = +\mathbf{U}[:, 2]$
4. $\mathbf{R}_2 = \mathbf{U} \mathbf{W}^T \mathbf{V}^T, \quad \mathbf{t} = -\mathbf{U}[:, 2]$
Determinants are enforced as $+1$ ($\det(\mathbf{R}) = +1, \det(\mathbf{U}) > 0, \det(\mathbf{V}) > 0$).
Translation is normalized to unit norm: $\|\mathbf{t}\| = 1.0$.

### 3.4 Cheirality & Linear DLT Triangulation
For each hypothesis $(\mathbf{R}, \mathbf{t})$, projection matrices are $\mathbf{P}_1 = [\mathbf{I} \mid \mathbf{0}]$ and $\mathbf{P}_2 = [\mathbf{R} \mid \mathbf{t}]$.
The 3D point $\tilde{\mathbf{X}} = [X, Y, Z, W]^T$ is triangulated via Direct Linear Transform (DLT):
$$\mathbf{A} \tilde{\mathbf{X}} = \mathbf{0}, \quad \mathbf{A} = \begin{bmatrix} x_1 \mathbf{P}_{1, 3}^T - \mathbf{P}_{1, 1}^T \\ y_1 \mathbf{P}_{1, 3}^T - \mathbf{P}_{1, 2}^T \\ x_2 \mathbf{P}_{2, 3}^T - \mathbf{P}_{2, 1}^T \\ y_2 \mathbf{P}_{2, 3}^T - \mathbf{P}_{2, 2}^T \end{bmatrix}$$
The optical depths in both camera frames must be strictly positive:
$$Z_{c1} = \frac{Z}{W} > 0 \quad \text{and} \quad Z_{c2} = (\mathbf{R} \mathbf{X}_1 + \mathbf{t})_z > 0$$
The unique physically realizable camera pose is the hypothesis that maximizes the cheirality pass count.

### 3.5 Parallax Angle & Degeneracy Diagnostics
The ray parallax angle $\theta_i$ subtended at 3D point $\mathbf{X}_1$ from camera centers $\mathbf{C}_1 = \mathbf{0}$ and $\mathbf{C}_2 = -\mathbf{R}^T \mathbf{t}$ is:
$$\cos \theta_i = \frac{\mathbf{X}_1}{\|\mathbf{X}_1\|} \cdot \frac{\mathbf{X}_1 - \mathbf{C}_2}{\|\mathbf{X}_1 - \mathbf{C}_2\|}$$
- **Pure Rotation Risk** (`PURE_ROTATION_RISK`): Median parallax $\theta_{\text{med}} < 0.5^{\circ}$ (`HEURISTIC_DEFAULT`).
- **Weak Baseline** (`WEAK_BASELINE`): Median parallax $\theta_{\text{med}} < 1.0^{\circ}$ (`HEURISTIC_DEFAULT`).
- **Planar Scene Risk** (`DEGENERATE_GEOMETRY`): Homography inlier ratio $\ge 0.85$ (`HEURISTIC_DEFAULT`).

---

## 4. Coordinate Frames & Handedness Standards

```
                      +U (Up / Zenith)
                       │
                       │
                       │
                       └────────────► +E (East)
                      /
                     /
                    ▼ +N (North)
               Local Tangent ENU Frame (World)
                       │
                       ▼ Extrinsics: X_c = R_cw * X_w + t_cw
                       │
                       ┌────────────► +X_c (Right)
                      /│
                     / │
                    /  ▼ +Y_c (Down)
                   ▼ +Z_c (Forward / Optical Axis)
               Camera Optical Frame (OpenCV Standard)
```

- **Relative Transformation**: $\mathbf{X}_{c2} = \mathbf{R}_{21} \mathbf{X}_{c1} + \mathbf{t}_{21}$
- **Relative Camera Center**: $\mathbf{C}_2 = -\mathbf{R}_{21}^T \mathbf{t}_{21}$ (Location of Camera 2 optical center expressed in Camera 1 frame).

---

## 5. Threshold Classifications in `TwoViewConfig`

All numerical thresholds are configured with explicit `HEURISTIC_DEFAULT` semantics:
- `ransac_threshold_px = 2.0` $\to$ `HEURISTIC_DEFAULT` (Sampson distance in pixels)
- `ransac_threshold_norm = 0.002` $\to$ `HEURISTIC_DEFAULT` (Normalized coordinate residual)
- `ransac_confidence = 0.999` $\to$ `HEURISTIC_DEFAULT`
- `max_iterations = 2000` $\to$ `HEURISTIC_DEFAULT`
- `min_inlier_ratio = 0.20` $\to$ `HEURISTIC_DEFAULT`
- `min_inliers = 15` $\to$ `HEURISTIC_DEFAULT`
- `weak_baseline_parallax_deg = 1.0` $\to$ `HEURISTIC_DEFAULT`
- `pure_rotation_parallax_deg = 0.5` $\to$ `HEURISTIC_DEFAULT`
- `min_cheirality_ratio = 0.65` $\to$ `HEURISTIC_DEFAULT`
- `homography_inlier_ratio_threshold = 0.85` $\to$ `HEURISTIC_DEFAULT`

---

## 6. Phase 3B Output Contract for Phase 3C

`TwoViewGeometryResult` provides calibrated initial two-view geometric constraints for **Phase 3C Incremental SfM**:
- `frame_a_id: str`, `frame_b_id: str`
- `essential_matrix: np.ndarray` (Shape $3 \times 3$)
- `relative_rotation: np.ndarray` (Shape $3 \times 3$, `SO(3)`)
- `relative_translation: np.ndarray` (Shape $3$, unit vector)
- `inlier_mask: np.ndarray` (Shape $N$, boolean)
- `inlier_count: int`
- `cheirality_passed_count: int`
- `median_parallax_deg: float`
- `scale_status: "SCALE_AMBIGUOUS"`
- `translation_magnitude_status: "UNOBSERVABLE"`
