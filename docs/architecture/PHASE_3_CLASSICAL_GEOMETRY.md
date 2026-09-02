# Phase 3: Classical Geometry Baseline Architecture, Mathematical Contracts & Scientific Evaluation Protocol

## 1. Executive Summary & Objective

Phase 3 establishes the deterministic **Classical Multi-View Geometry Baseline** for the reconstruction pipeline. Its objective is to reconstruct accurate 3D scene geometry from the quality-assessed and coverage-selected keyframe stream (`KeyframeSelectionResult`) using classical structure-from-motion (SfM) and multi-view stereo (MVS) principles.

This document formalizes the mathematical specifications, coordinate frames, typed contracts, error taxonomy, failure states, gauge fixing policies, lens distortion models, and the rigorous scientific evaluation hierarchy comparing the classical baseline against foundation AI geometry models (e.g. DUSt3R, VGGT).

> **CRITICAL SCIENTIFIC PRINCIPLES & GUARDRAILS:**
> 1. **NO RECONSTRUCTION CLAIMS BEFORE EMPIRICAL EVALUATION**:
>    - Metric accuracy, visual completeness, and real-drone compatibility cannot be claimed until empirical validation is conducted on benchmark datasets.
> 2. **THREE DISTINCT LEVELS OF GEOMETRY EVALUATION**:
>    - **Level 1 (Image-Space Geometric Consistency)**: Reprojection residuals and 2D track consistency.
>    - **Level 2 (Relative / Scale-Aligned 3D Geometry)**: Relative structure and shape fidelity up to an unknown global similarity transform ($\text{Sim}(3)$).
>    - **Level 3 (Absolute Metric / Geospatial Accuracy)**: Physical scale in meters and georeferenced coordinates verified against certified ground truth or surveyed Ground Control Points (GCPs).
>    - *Scale-alignment (e.g. Procrustes / Umeyama) is an evaluation tool, NOT proof of intrinsic metric correctness.*
> 3. **GROUND TRUTH PROVENANCE (ATE RMSE)**:
>    - Onboard GNSS/navigation telemetry is a sensor measurement classified as `TRAJECTORY_PROXY`.
>    - Absolute Trajectory Error (ATE RMSE) is strictly `GROUND_TRUTH_DEPENDENT` and requires an independent certified reference trajectory. Telemetry alone must NOT be treated as ground truth.
> 4. **MONOCULAR GAUGE FREEDOM & SCALE AMBIGUITY**:
>    - Monocular visual reconstructions possess an inherent 7-DoF similarity gauge ambiguity ($\text{Sim}(3)$: 3 rotation, 3 translation, 1 scale).
>    - Gauge fixing during Bundle Adjustment fixes arbitrary numerical coordinates in relative reconstruction space; it does NOT establish physical metric scale.
> 5. **COMPLETENESS SEMANTICS**:
>    - **Reference Point Completeness** (`REFERENCE_POINT_COMPLETENESS`): Fraction of sampled reference points with a reconstructed neighbor within distance tolerance $\tau$. Assumes uniform surface point density.
>    - **Surface Area Completeness** (`SURFACE_AREA_COMPLETENESS`): Area-weighted surface coverage on a continuous reference surface mesh within tolerance $\tau$.
>    - *Raw point count is a descriptive density statistic, not a quality or completeness metric.*
> 6. **FUNDAMENTAL VS. ESSENTIAL MATRIX CALIBRATION**:
>    - **Fundamental Matrix $\mathbf{F}$**: Operates on uncalibrated pixel raster coordinates ($\mathbf{x}_{2, \text{px}}^T \mathbf{F} \mathbf{x}_{1, \text{px}} = 0$).
>    - **Essential Matrix $\mathbf{E}$**: Operates on calibrated normalized coordinates ($\mathbf{x}_{2, \text{norm}}^T \mathbf{E} \mathbf{x}_{1, \text{norm}} = 0$). Requires camera intrinsics $\mathbf{K}$.
> 7. **SPARSE SFM SUCCESS $\neq$ DENSE MVS SUCCESS**:
>    - A successful sparse SfM camera registration does not imply successful dense depth estimation or complete surface coverage.

---

## 2. Target Pipeline Architecture

```
                    KeyframeSelectionResult (Selected Keyframes)
                                       │
                                       ▼
                     [Stage 5A] 2D Feature Extraction
                               (ORB / SIFT / SuperPoint)
                                       │
                                       ▼
                     [Stage 5B] Feature Matching & Filtering
                          (Mutual Cross-Check / Ratio Test)
                                       │
                                       ▼
                     [Stage 5C] Two-View Geometric Verification
                         (Fundamental F / Essential E / RANSAC)
                                       │
                                       ▼
                     [Stage 5D] Two-View Seed Initialization
                          (R_rel, t_rel, Initial Triangulation)
                                       │
                                       ▼
                     [Stage 5E] Incremental Camera Registration
                             (P3P / EPnP + RANSAC Pose Solving)
                                       │
                                       ▼
                     [Stage 5F] Multi-View Track Triangulation
                             (DLT + Parallax Angle Filtering)
                                       │
                                       ▼
                     [Stage 5G] Global Bundle Adjustment (BA)
                       (Gauge-Fixed Huber-Loss Joint Optimization)
                                       │
                                       ▼
                         SparseReconstructionResult
                                       │
                                       ▼
                     [Stage 5H] Dense Multi-View Stereo (MVS)
                         (PatchMatch Depth Estimation & Fusion)
                                       │
                                       ▼
                              DenseMVSOutput
```

---

## 3. Mathematical Formulations & Contracts

### 3.1 Pinhole Camera Model & Lens Distortion
Given a 3D world landmark point $\mathbf{X}_w = [E, N, U]^T \in \mathbb{R}^3$ in local ENU space:
1. **Extrinsic Transformation to Camera Optical Frame**:
   $$\mathbf{X}_c = \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = \mathbf{R}_{cw} \mathbf{X}_w + \mathbf{t}_{cw}$$
   where $\mathbf{R}_{cw} \in \text{SO}(3)$ and $\mathbf{t}_{cw} \in \mathbb{R}^3$.
   The optical center in world coordinates is:
   $$\mathbf{C}_w = -\mathbf{R}_{cw}^T \mathbf{t}_{cw}$$
2. **Normalized Undistorted Camera Coordinates**:
   $$\mathbf{x}_n = \begin{bmatrix} x_n \\ y_n \\ 1 \end{bmatrix} = \begin{bmatrix} X_c / Z_c \\ Y_c / Z_c \\ 1 \end{bmatrix} \quad (\text{requires } Z_c > 0)$$
3. **Lens Distortion Modeling**:
   - `NONE_RECTIFIED`: Raster images already undistorted ($x_d = x_n, y_d = y_n$).
   - `BROWN_CONRADY_RADIAL_TANGENTIAL`:
     $$r^2 = x_n^2 + y_n^2$$
     $$x_d = x_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2 p_1 x_n y_n + p_2 (r^2 + 2 x_n^2)$$
     $$y_d = y_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1 (r^2 + 2 y_n^2) + 2 p_2 x_n y_n$$
4. **Intrinsic Projection to Pixel Raster**:
   $$u = f_x x_d + c_x, \quad v = f_y y_d + c_y$$

### 3.2 Fundamental vs. Essential Matrix Epipolar Geometry
1. **Fundamental Matrix $\mathbf{F} \in \mathbb{R}^{3 \times 3}$ (Uncalibrated Pixel Space)**:
   $$\mathbf{x}_{2, \text{px}}^T \mathbf{F} \, \mathbf{x}_{1, \text{px}} = 0 \quad (\text{Rank 2 matrix, 7 DoF})$$
2. **Essential Matrix $\mathbf{E} \in \mathbb{R}^{3 \times 3}$ (Calibrated Normalized Space)**:
   $$\mathbf{x}_{2, \text{norm}}^T \mathbf{E} \, \mathbf{x}_{1, \text{norm}} = 0 \quad \text{with } \mathbf{E} = [\mathbf{t}_{21}]_\times \mathbf{R}_{21} = \mathbf{K}_2^T \mathbf{F} \mathbf{K}_1$$
3. **Cheirality Constraint (Mathematical Condition)**:
   Triangulated 3D points must possess strictly positive optical depth in both observing camera frames:
   $$Z_{c, 1} > 0 \quad \text{and} \quad Z_{c, 2} > 0$$

### 3.3 Multi-View Triangulation (Direct Linear Transform)
For an observed landmark $\mathbf{X}_w$ across views $i \in \{1, \dots, M\}$:
$$\mathbf{x}_i \times (\mathbf{P}_i \tilde{\mathbf{X}}_w) = \mathbf{0}$$
- **Validation Gates**:
  - Parallax angle $\theta_{\text{parallax}} \ge 1.0^{\circ}$ (`HEURISTIC_DEFAULT`).
  - Cheirality $Z_{c, i} > 0 \;\forall i$ (Mathematical Condition).
  - Mean reprojection error $\le 2.0\text{ px}$ (`HEURISTIC_DEFAULT`).

### 3.4 Global Bundle Adjustment & Gauge Fixing Policy
Joint optimization of registered camera poses $\{\mathbf{R}_i, \mathbf{t}_i\}$, intrinsics $\{\mathbf{K}_i\}$, and 3D landmarks $\{\mathbf{X}_j\}$:
$$\min_{\{\mathbf{R}_i, \mathbf{t}_i, \mathbf{K}_i, \mathbf{X}_j\}} \sum_{i=1}^M \sum_{j \in \mathcal{V}_i} \rho_{\text{Huber}}\left( \left\| \mathbf{x}_{ij} - \pi(\mathbf{K}_i, \mathbf{R}_i, \mathbf{t}_i, \mathbf{X}_j) \right\|^2_{\mathbf{\Sigma}_{ij}^{-1}} \right)$$

**Gauge Fixing Policy (`GaugeFixingPolicy`)**:
- `FIX_FIRST_CAMERA_AND_UNIT_BASELINE`: Camera 0 fixed to $[\mathbf{I} \mid \mathbf{0}]$ and $\|\mathbf{t}_{10}\| = 1.0$. Fixes 7-DoF ambiguity in relative reconstruction space.
- *Note: Gauge fixing resolves numerical singularity in the Levenberg-Marquardt Hessian; it does NOT establish physical metric scale.*

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

| Coordinate Space | Axes Definition | Handedness | Unit | Reference |
| :--- | :--- | :--- | :--- | :--- |
| **Pixel Raster Space** | $u$ (Right, col), $v$ (Down, row) | 2D Plane | Pixels ($\text{px}$) | Sensor array origin (top-left) |
| **Normalized Camera Space** | $x_n = (u-c_x)/f_x$, $y_n = (v-c_y)/f_y$ | 2D Ray | Dimensionless | Pinhole optical axis |
| **Camera Optical Frame** | $+X_c$ Right, $+Y_c$ Down, $+Z_c$ Forward | Right-handed | Metric / Scale | Optical perspective center |
| **Local ENU World Frame** | $+E$ East, $+N$ North, $+U$ Up | Right-handed | Meters ($\text{m}$) | Tangent plane origin on WGS84 |
| **GNSS Platform Trajectory** | Flight path positions $\mathbf{p}_{\text{GNSS}}$ | Right-handed | Meters ($\text{m}$) | Navigation antenna center (`TRAJECTORY_PROXY`) |

---

## 5. Threshold Semantics & Classifications

| Parameter / Failure Gate | Value | Classification | Description |
| :--- | :--- | :--- | :--- |
| `min_feature_count` | `100` | `HEURISTIC_DEFAULT` | Engineering threshold for minimum keypoints per frame. |
| `min_candidate_matches` | `30` | `HEURISTIC_DEFAULT` | Engineering threshold for candidate pair matching. |
| `min_inlier_ratio` | `0.20` | `HEURISTIC_DEFAULT` | Minimum RANSAC inlier ratio to accept two-view geometry. |
| `weak_baseline_parallax_deg` | `1.0 deg` | `HEURISTIC_DEFAULT` | Minimum ray parallax to avoid ill-conditioned depth triangulation. |
| `min_sparse_points` | `50` | `HEURISTIC_DEFAULT` | Minimum landmark count to declare successful sparse reconstruction. |
| `min_registered_cameras` | `3` | `HEURISTIC_DEFAULT` | Minimum cameras required to form a multi-view model. |
| `max_reprojection_rmse_px` | `2.0 px` | `HEURISTIC_DEFAULT` | Maximum acceptable root-mean-square reprojection residual. |
| `fundamental_epipolar` | $\mathbf{x}_{2, \text{px}}^T \mathbf{F} \mathbf{x}_{1, \text{px}} = 0$ | **Mathematical Condition** | Exact epipolar constraint in uncalibrated pixel space. |
| `essential_epipolar` | $\mathbf{x}_{2, \text{norm}}^T \mathbf{E} \mathbf{x}_{1, \text{norm}} = 0$ | **Mathematical Condition** | Exact coplanarity constraint in normalized calibrated space. |
| `positive_cheirality` | $Z_c > 0$ | **Mathematical Condition** | Physical requirement that landmarks lie in front of cameras. |
| `surface_distance_tolerance` | $\tau = 0.05\text{ m}$ | `GROUND_TRUTH_DEPENDENT` | Spatial distance threshold for reference completeness evaluation. |

---

## 6. Scientific Evaluation Protocol: Classical vs. Foundation AI Geometry

The comparison between classical geometry and foundation AI geometry models (e.g. DUSt3R, VGGT) is structured into **5 distinct evaluation dimensions**:

### Dimension A: Image-Space Consistency (Level 1)
- **Reprojection RMSE**: Root-mean-square pixel error $\text{RMSE}_{\text{reproj}} = \sqrt{\frac{1}{N} \sum \|\mathbf{x} - \hat{\mathbf{x}}\|^2} \quad [\text{px}]$.
- **Track Consistency**: Mean track length across multi-view observations.
- **Inlier Ratio**: Ratio of geometrically verified matches passing epipolar filtering.

### Dimension B: Relative / Scale-Aligned 3D Geometric Accuracy (Level 2)
- **Surface Accuracy / Chamfer Distance**: Mean bidirectional distance to ground-truth LiDAR surface after $\text{Sim}(3)$ alignment.
- **Reference Point Completeness** (`REFERENCE_POINT_COMPLETENESS`):
  $$\text{Completeness}(\mathcal{S}_{\text{ref}}, \tau) = \frac{\left|\{\mathbf{p} \in \mathcal{S}_{\text{ref}} \mid \min_{\mathbf{q} \in \mathcal{S}_{\text{rec}}} \|\mathbf{p} - \mathbf{q}\| \le \tau\}\right|}{|\mathcal{S}_{\text{ref}}|} \in [0.0, 1.0]$$
- **Point Density**: Descriptive count of reconstructed points per unit area ($\text{pts/m}^2$). *Point count alone is not a quality metric.*

### Dimension C: Absolute Metric / Geospatial Accuracy (Level 3 - GROUND_TRUTH_DEPENDENT)
- **Scale Error Percentage**: $\frac{|s_{\text{estimated}} - s_{\text{true}}|}{s_{\text{true}}} \times 100\%$.
- **Absolute Trajectory Error (ATE)**: RMSE of camera centers against certified ground-truth survey coordinates in meters.
- **GCP Residual Error**: Metric distance error at surveyed Ground Control Points.

### Dimension D: Robustness & Failure Modes
- **Degradation Breakdown**: Failure rate across severe motion blur, low texture (asphalt, water), near-pure rotation ($B < 0.2\text{ m}$), and dynamic objects.
- **Cheirality Violations**: Number of landmarks or depth pixels falling behind the camera plane.

### Dimension E: Computational Footprint
- **Runtime Latency**: Total pipeline elapsed time (seconds) and throughput (FPS).
- **Memory Footprint**: Peak CPU RAM (MB) and peak GPU VRAM (GB).

---

## 7. Failure Taxonomy (Non-Silent State Transitions)

- **`INSUFFICIENT_FEATURES`**: Detected feature count $< 100$ (`HEURISTIC_DEFAULT`).
- **`INSUFFICIENT_MATCHES`**: Candidate pair matches $< 30$ (`HEURISTIC_DEFAULT`).
- **`GEOMETRIC_VERIFICATION_FAILED`**: RANSAC inlier ratio $< 20\%$ (`HEURISTIC_DEFAULT`).
- **`DEGENERATE_GEOMETRY`**: Planar or collinear feature distribution causing homography ambiguity.
- **`PURE_ROTATION_RISK`**: Baseline translation $B \approx 0$ relative to scene depth.
- **`WEAK_BASELINE`**: Baseline parallax angle $< 1.0^{\circ}$ (`HEURISTIC_DEFAULT`).
- **`CALIBRATION_UNAVAILABLE`**: Camera intrinsics missing when Essential Matrix $\mathbf{E}$ or 3D metric triangulation is requested.
- **`CHEIRALITY_VIOLATION`**: Triangulated points fall behind camera sensor planes ($Z_c \le 0$).
- **`CAMERA_REGISTRATION_FAILED`**: PnP RANSAC fails to register a new camera.
- **`TRIANGULATION_FAILED`**: Landmark rays fail to converge within reprojection threshold.
- **`SPARSE_RECONSTRUCTION_INSUFFICIENT`**: $< 50$ 3D points or $< 3$ cameras registered (`HEURISTIC_DEFAULT`).
- **`BUNDLE_ADJUSTMENT_DIVERGED`**: Non-linear optimization gradient exceeds numerical bounds.
- **`MVS_DEPTH_ESTIMATION_FAILED`**: Multi-view PatchMatch stereo fails photometric consistency.

---

## 8. Phase 3 Implementation Roadmap

The execution of Phase 3 will follow this exact phased order:
- **Phase 3A**: Classical Feature Extraction & Robust Descriptor Matching (`src/geometry/features.py`).
- **Phase 3B**: Epipolar Geometry, Essential Matrix Estimation & Two-View Initialization (`src/geometry/two_view.py`).
- **Phase 3C**: Incremental Structure-from-Motion, PnP Camera Registration & Triangulation (`src/geometry/sfm.py`).
- **Phase 3D**: Global Bundle Adjustment & Sparse Point Cloud Generation (`src/geometry/bundle_adjustment.py`).
- **Phase 3E**: Dense Multi-View Stereo (MVS) Depth Estimation & Point Cloud Fusion (`src/geometry/mvs.py`).
