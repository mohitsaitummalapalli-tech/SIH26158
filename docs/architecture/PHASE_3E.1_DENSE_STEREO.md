# Phase 3E.1 Architecture: Classical Dense Stereo Baseline

## 1. Overview & Algorithmic Scope

Phase 3E.1 implements the **first real dense stereo depth estimation baseline** for the SIH 26158 UAV 3D reconstruction system.

### Algorithmic Policy:
- **Selected Method**: Classical Semi-Global Block Matching (OpenCV `StereoSGBM`) with calibrated epipolar rectification and bidirectional Left-Right Consistency (LRC) validation.
- **Explicitly Excluded at this stage**: Learned/deep stereo networks (e.g. DUSt3R, MASt3R, VGGT, MASt3R-SLAM), Gaussian Splatting, NeRF, and CUDA kernels. These are designated for future comparative experimental milestones once this classical geometric baseline is validated.
- **Contract Adherence**: Satisfies the Phase 3E.0 `IMVSDepthEstimator` interface and outputs typed `DepthMap` and `DepthConfidenceMap` data structures.

---

## 2. Mathematical Formulation & Camera Geometry

### 2.1 Coordinate Conventions
Camera extrinsic convention adheres strictly to Phase 3 contracts:
$$\mathbf{X}_c = \mathbf{R}_{cw} \mathbf{X}_w + \mathbf{t}_{cw}$$
$$\mathbf{C}_w = -\mathbf{R}_{cw}^T \mathbf{t}_{cw} \iff \mathbf{t}_{cw} = -\mathbf{R}_{cw} \mathbf{C}_w$$

Pinhole perspective projection:
$$u = f_x \frac{X_c}{Z_c} + c_x, \quad v = f_y \frac{Y_c}{Z_c} + c_y$$

### 2.2 Critical Distinction: Disparity is NOT Depth
Stereo matching evaluates the 1D horizontal shift (disparity $d$) between rectified epipolar scanlines:
$$d = u_{rect, ref} - u_{rect, src}$$

Cartesian optical depth $Z_c$ along the camera optical axis is calculated strictly via:
$$Z_{rect} = \frac{f'_{rect} \cdot B_{rect}}{d}$$
where:
- $f'_{rect} = P_1[0, 0]$ is the rectified horizontal focal length.
- $B_{rect} = \frac{|P_2[0, 3]|}{P_1[0, 0]}$ is the baseline in `RECONSTRUCTION_UNITS` between rectified optical centers.
- $d > 0$ is the valid subpixel disparity in pixels ($d = \text{disparity\_raw} / 16.0$ from SGBM fixed-point representation).

`DepthMap` stores **strictly optical depth $Z_c$**, never:
- Euclidean range $\|\mathbf{X}_c\|_2$
- Raw disparity $d$
- Inverse depth $1/Z$

---

## 3. Dedicated Stereo Rectification Layer

Rectification is handled by the dedicated `StereoRectifier` class.

Given reference camera $(\mathbf{K}_1, \mathbf{D}_1, \mathbf{R}_{ref}, \mathbf{C}_{ref})$ and source camera $(\mathbf{K}_2, \mathbf{D}_2, \mathbf{R}_{src}, \mathbf{C}_{src})$:
1. **Relative Rigid Transform**:
   $$\mathbf{R}_{rel} = \mathbf{R}_{src} \mathbf{R}_{ref}^T$$
   $$\mathbf{t}_{rel} = \mathbf{R}_{src} (\mathbf{C}_{ref} - \mathbf{C}_{src})$$
2. **OpenCV Epipolar Rectification**:
   `cv2.stereoRectify` computes $3 \times 3$ rotation matrices $\mathbf{R}_1, \mathbf{R}_2$, $3 \times 4$ projection matrices $\mathbf{P}_1, \mathbf{P}_2$, and $4 \times 4$ disparity-to-depth matrix $\mathbf{Q}$.
3. **Calibrated Distortion & Remapping**:
   `cv2.initUndistortRectifyMap` and `cv2.remap` transform the reference and source image planes so that epipolar lines become horizontal and collinear across identical rows ($v_1 = v_2$).
4. **Rectified Intrinsics Preservation**:
   The rectified focal length and principal point are extracted from $\mathbf{P}_1$ and stored in `StereoRectificationResult.rectified_intrinsics_ref`.

---

## 4. Disparity Validation & Left-Right Consistency

### 4.1 Strict Disparity Validation (Zero-Tolerance)
Disparities are rejected if:
- Non-finite (NaN, $+\infty$, $-\infty$)
- Non-positive ($d \le 0.0$)
- Outside the configured search range $[d_{min}, d_{min} + \Delta d]$
- Out-of-bounds in source image ($u_{src} = u - d < 0$ or $\ge W$)

**Project Policy Enforced**:
- `np.nan_to_num()` is **strictly prohibited**.
- Invalid values are **never clipped into validity**.
- Invalid pixels are **never assigned fake depth 0.0 or -1.0 as valid**.

### 4.2 Bidirectional Left-Right Consistency (LRC)
1. Left disparity $d_L(u, v)$ is computed (reference $\to$ source).
2. Right disparity $d_R(u_{src}, v)$ is computed (source $\to$ reference).
3. The corresponding source column is $c_{src} = \text{round}(u - d_L)$.
4. The right matcher convention yields negative disparity; its magnitude $|d_R(c_{src}, v)|$ is compared:
   $$|d_L(u, v) - |d_R(c_{src}, v)|| \le \tau_{lr} \quad (\text{HEURISTIC\_DEFAULT: } 1.5\text{ px})$$
5. Disagreements exceeding $\tau_{lr}$ are classified as `PointVisibilityState.INCONSISTENT` and excluded from the valid depth mask.

---

## 5. Visibility Taxonomy & Occlusion Handling

Every pixel is explicitly categorized into `PointVisibilityState`:
- `VALID`: Disparity is positive, within range, satisfies LRC, and yields $Z \in [Z_{min}, Z_{max}]$.
- `INVALID_DEPTH`: Disparity is non-positive, non-finite, or out of search range.
- `INCONSISTENT`: Disparity fails bidirectional left-right consistency ($> \tau_{lr}$).
- `OCCLUDED`: Projected ray falls outside the source image bounds ($u - d < 0$ or $\ge W$).

---

## 6. Confidence Semantics

Confidence is explicitly labeled `# HEURISTIC_SCORE` and strictly bounded in $[0, 1]$.
It is **NOT** a Bayesian posterior probability, statistical variance, or accuracy percentage.

Confidence combines:
1. **Local Gradient Texture**: Computed via Sobel gradient magnitude on the reference image.
2. **Left-Right Agreement**: Scaled exponentially $\exp(-|d_L - |d_R|| / \tau_{lr})$.
3. **Dynamic Motion Risk Attenuation**: Attenuated by Phase 2 dynamic risk:
   $$\text{confidence} = \text{confidence}_{raw} \times (1.0 - 0.5 \cdot \text{dynamic\_risk})$$

---

## 7. Scale Gauge & Provenance Preservation

1. **Reconstruction Units**: All depth values are strictly in `RECONSTRUCTION_UNITS`.
2. **No Metric Scale Claim**: `DepthMap.provenance["is_metric"] = False`. GNSS metadata is never silently injected into stereo depth estimation.
3. **Traceability**: Output structures preserve `reference_frame_id`, `source_frame_id`, rectified camera intrinsics, baseline distance, algorithm name, and matching parameters.
