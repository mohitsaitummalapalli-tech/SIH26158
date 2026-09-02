# Phase 3C: Incremental Structure-from-Motion (SfM) Architecture

## 1. Executive Summary & Objective

Phase 3C implements the deterministic **Incremental Structure-from-Motion (SfM)** pipeline. It consumes verified two-view epipolar seed geometry from Phase 3B, fixes the 7-DoF similarity gauge freedom, maintains multi-view 3D landmark tracks across keyframes, robustly registers additional cameras via Perspective-n-Point (PnP + RANSAC), triangulates new 3D points with cheirality and parallax verification, and outputs a relative sparse reconstruction (`SparseReconstructionResult`).

> **CRITICAL SCIENTIFIC PRINCIPLES & GUARDRAILS:**
> 1. **MONOCULAR RELATIVE SFM & SCALE AMBIGUITY**:
>    - The reconstruction produced by Phase 3C is strictly **`SCALE_AMBIGUOUS`** and expressed in arbitrary relative reconstruction units (`has_monocular_scale_ambiguity = True`, `is_metric_scale = False`).
>    - Monocular camera baseline translation magnitude is physically unobservable without certified external metric scale references (e.g. calibrated stereo baseline or surveyed Ground Control Points).
> 2. **NO GNSS POSE SUBSTITUTION**:
>    - Onboard drone telemetry is treated strictly as `TRAJECTORY_PROXY`.
>    - GNSS coordinates are **never** copied or substituted into the visual SfM camera poses.
> 3. **PnP TRANSLATION $\neq$ METRIC SCALE**:
>    - The translation vector $\mathbf{t}_{cw}$ solved via PnP is scaled strictly to the gauge fixed by the initial two-view pair ($\|\mathbf{t}_{10}\| = 1.0$).
> 4. **REPROJECTION ERROR $\neq$ 3D METRIC ACCURACY**:
>    - Reprojection error measured in pixels is an image-space consistency diagnostic (`LEVEL_1_IMAGE_SPACE_CONSISTENCY`), not proof of physical 3D accuracy.
> 5. **GLOBAL BUNDLE ADJUSTMENT SCOPE**:
>    - Incremental SfM performs local geometric verification and track extension. Full joint nonlinear optimization (Bundle Adjustment) is strictly isolated to Phase 3D.
> 6. **SYNTHETIC VALIDATION DISCLAIMER**:
>    - Synthetic validation measures implementation behavior under controlled assumptions; it does not establish real-UAV performance.

---

## 2. Pipeline Architecture & Inter-Phase Integration

```
       Phase 3A: Candidate Descriptor Matching (ClassicalDescriptorMatcher)
                                          │
                                          ▼
                      [FeatureMatchResult.to_correspondences()]
                                          │
                                          ▼
          Phase 3B: Robust Two-View Epipolar Geometry (TwoViewGeometryEstimator)
                                          │
                                          ▼
                     [TwoViewGeometryResult (Essential Matrix Seed)]
                                          │
                                          ▼
                       [Initial Gauge Fixing (Camera 0 & 1)]
                          Camera 0: R_0 = I, t_0 = 0
                          Camera 1: R_1 = R_rel, t_1 = t_rel (||t_1|| = 1.0)
                                          │
                                          ▼
                        [Initial Triangulated Landmark Tracks]
                          DLT Triangulation (Cheirality Z > 0, Parallax >= 1.0 deg)
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │  Incremental Registration Loop  │
                         └────────────────┬────────────────┘
                                          │
                                          ▼
                        [Candidate Frame Eligibility & Evaluation]
                          available_2d3d_correspondences (>= 6)
                          estimated_registration_sufficiency (>= 15)
                          Deterministic alphabetical tie-breaking
                                          │
                                          ▼
                        [Unambiguous 2D-3D Correspondence Matching]
                          Strict uniqueness enforcement (Invariants A & B)
                          Explicit TRACK_CONFLICT logging
                                          │
                                          ▼
                         [Perspective-n-Point (PnP + RANSAC)]
                          Solve R_cw, t_cw via cv2.solvePnPRansac
                          Validate inliers (>= 15, ratio >= 0.25) & RMSE
                          PLANARITY_RISK diagnostic
                                          │
                                          ▼
                        [Multi-View Track Observations Update]
                          Append observations to existing 3D tracks without mutation
                                          │
                                          ▼
                          [Multi-View Point Triangulation]
                          Triangulate newly visible 2D matches with registered cams
                                          │
                                          ▼
                         [SparseReconstructionResult Output]
```

---

## 3. Mathematical Formulations & Coordinate Standards

### 3.1 Camera Projection & Extrinsic Representation
Standard OpenCV optical camera convention:
- **Camera Optical Frame**: $+X$ right, $+Y$ down, $+Z$ forward (along principal optical axis).
- **World-to-Camera Transformation**:
  $$\mathbf{X}_c = \mathbf{R}_{cw} \mathbf{X}_w + \mathbf{t}_{cw}$$
  where $\mathbf{R}_{cw} \in SO(3)$ and $\mathbf{t}_{cw} \in \mathbb{R}^3$.
- **Camera Optical Center in World Coordinates**:
  $$\mathbf{C}_w = -\mathbf{R}_{cw}^T \mathbf{t}_{cw}$$
- **Pinhole Projection**:
  $$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim \mathbf{K} (\mathbf{R}_{cw} \mathbf{X}_w + \mathbf{t}_{cw})$$

### 3.2 Gauge Fixing Policy
To anchor the 7 degrees of freedom (3 rotation, 3 translation, 1 scale) in monocular SfM:
$$\text{Camera 0}: \quad \mathbf{R}_0 = \mathbf{I}_{3 \times 3}, \quad \mathbf{t}_0 = \mathbf{0}_{3 \times 1}$$
$$\text{Camera 1}: \quad \mathbf{R}_1 = \mathbf{R}_{\text{rel}}, \quad \mathbf{t}_1 = \frac{\mathbf{t}_{\text{rel}}}{\|\mathbf{t}_{\text{rel}}\|}$$
This implements `GaugeFixingPolicy.FIX_FIRST_CAMERA_AND_UNIT_BASELINE`.

### 3.3 Direct Linear Transform (DLT) Triangulation
For calibrated projection matrices $\mathbf{P}_1 = \mathbf{K}_1 [\mathbf{R}_1 \mid \mathbf{t}_1]$ and $\mathbf{P}_2 = \mathbf{K}_2 [\mathbf{R}_2 \mid \mathbf{t}_2]$ and pixel observations $\mathbf{p}_1 = (u_1, v_1)$, $\mathbf{p}_2 = (u_2, v_2)$:
$$\mathbf{A} \tilde{\mathbf{X}} = \mathbf{0}, \quad \mathbf{A} = \begin{bmatrix} u_1 \mathbf{P}_{1, 3}^T - \mathbf{P}_{1, 1}^T \\ v_1 \mathbf{P}_{1, 3}^T - \mathbf{P}_{1, 2}^T \\ u_2 \mathbf{P}_{2, 3}^T - \mathbf{P}_{2, 1}^T \\ v_2 \mathbf{P}_{2, 3}^T - \mathbf{P}_{2, 2}^T \end{bmatrix}$$
Solve via SVD $\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$, $\tilde{\mathbf{X}} = \mathbf{V}[3, :]$, dehomogenize $\mathbf{X}_w = \tilde{\mathbf{X}}[:3] / \tilde{\mathbf{X}}[3]$.
- **Cheirality Requirement**: $Z_{c1} = (\mathbf{R}_1 \mathbf{X}_w + \mathbf{t}_1)_z > 0$ and $Z_{c2} = (\mathbf{R}_2 \mathbf{X}_w + \mathbf{t}_2)_z > 0$.
- **Ray Parallax Requirement**:
  $$\mathbf{r}_1 = \frac{\mathbf{X}_w - \mathbf{C}_1}{\|\mathbf{X}_w - \mathbf{C}_1\|}, \quad \mathbf{r}_2 = \frac{\mathbf{X}_w - \mathbf{C}_2}{\|\mathbf{X}_w - \mathbf{C}_2\|}, \quad \theta = \arccos(\mathbf{r}_1 \cdot \mathbf{r}_2) \ge 1.0^{\circ}$$

---

## 4. PnP Threshold Semantics & Candidate Distinction

### 4.1 Distinction Between Minimal Configurations, Practical RANSAC, and Acceptance
- **Configurable Candidate Threshold (`min_candidate_correspondences = 6`)**:
  `6 is a configurable HEURISTIC_DEFAULT minimum candidate correspondence threshold for this implementation.`
  It is NOT a universal mathematical law.
- **Solver-Specific Minimal Configuration**:
  Minimal sample size for P3P is 3 points (producing up to 4 solutions) plus a 4th point for disambiguation; uncalibrated DLT requires 6 points.
- **Project Acceptance Threshold (`min_pnp_inliers = 15`)**:
  To ensure geometric stability and avoid degenerate pose solutions, the project requires at least 15 verified inliers after RANSAC.

### 4.2 Candidate Eligibility vs. Registration Sufficiency
The engine distinguishes candidate selection from registration success:
- **Eligibility**: Candidate frame has $\ge 6$ 2D–3D correspondences (`available_2d3d_correspondences >= min_candidate_correspondences`).
- **Sufficiency**: Candidate frame has $\ge 15$ correspondences (`estimated_registration_sufficiency = True`).
- If a candidate has $6 \le N < 15$ correspondences, it may be selected, but its diagnostic explicitly indicates that registration sufficiency is False and PnP rejection may follow.

---

## 5. Planarity Risk Semantics

Planar 3D configurations do not mathematically invalidate PnP in general (calibrated P3P/EPnP handles planar scenes well). However, planar point distributions create numerical conditioning and depth ambiguity risks (`PLANARITY_RISK`).

The engine computes an SVD on centered 3D points:
$$\text{If } \frac{\sigma_3}{\max(10^{-6}, \sigma_1)} < \text{planar\_svd\_ratio\_threshold} \ (10^{-4})$$
A `PLANARITY_RISK` diagnostic is recorded without unconditionally aborting the solver, clearly distinguishing numerical risk from algorithmic failure.

---

## 6. Track Invariants & Provenance

The engine enforces strict track uniqueness in both directions:
1. **Invariant A (No Duplicate Views per Track)**:
   One keyframe cannot contribute multiple observations to the same landmark track.
2. **Invariant B (No Keypoint Multi-Assignment)**:
   One 2D keypoint in a keyframe cannot belong to multiple 3D landmark tracks.
3. **Explicit Conflict Logging**:
   Conflicting associations produce an explicit `TRACK_CONFLICT` diagnostic rather than silent overwriting or merging.

---

## 7. Failure Taxonomy

Non-silent explicit failure codes:
- `INSUFFICIENT_MATCHES`: Insufficient feature correspondences to initialize.
- `INSUFFICIENT_2D_3D_CORRESPONDENCES`: Candidate camera has $< 6$ candidate correspondences.
- `CAMERA_REGISTRATION_FAILED`: PnP RANSAC failed to converge or inlier count/ratio/RMSE violated thresholds.
- `TRIANGULATION_FAILED`: Point triangulation ill-conditioned or non-finite.
- `CHEIRALITY_VIOLATION`: Triangulated point has negative or zero optical depth in observing cameras.
- `TRACK_CONFLICT`: Ambiguous feature-to-track correspondence assignments detected.
- `RECONSTRUCTION_STALLED`: No unregistered candidate frames have sufficient 2D–3D overlap.
- `SPARSE_RECONSTRUCTION_INSUFFICIENT`: Total registered cameras $< 2$ or triangulated points $< 15$.
