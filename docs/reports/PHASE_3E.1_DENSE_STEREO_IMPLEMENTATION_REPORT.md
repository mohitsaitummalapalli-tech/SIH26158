# Phase 3E.1 Dense Stereo Baseline Implementation Report

## 1. Algorithm
- **Baseline Engine**: OpenCV Semi-Global Block Matching (`cv2.StereoSGBM` with `MODE_SGBM_3WAY`).
- **Pipeline Structure**:
  1. Calibrated stereo rectification via `StereoRectifier` using `cv2.stereoRectify`.
  2. Subpixel undistortion and remapping via `cv2.initUndistortRectifyMap` and `cv2.remap`.
  3. Bidirectional matching: Reference $\to$ Source (`disp_left_raw`) and Source $\to$ Reference (`disp_right_raw`).
  4. Disparity validation, Left-Right Consistency (LRC) enforcement, and disparity-to-optical-depth conversion.
  5. Composite heuristic confidence evaluation and Phase 2 dynamic motion risk attenuation.
- **Contract Fulfillment**: Implements `IMVSDepthEstimator` from Phase 3E.0 (`src.geometry.mvs`).

---

## 2. Mathematical Formulation
- **Stereo Disparity**:
  Epipolar lines are collinear along image raster rows ($v_{ref} = v_{src}$). The horizontal shift between corresponding points is disparity $d = u_{ref} - u_{src}$.
- **Depth Formulation**:
  Optical depth along the camera principal axis $Z_c$ is determined via:
  $$Z_{rect} = \frac{f'_{rect} \cdot B_{rect}}{d}$$
  where $f'_{rect}$ is the rectified horizontal focal length from $\mathbf{P}_1[0, 0]$, $B_{rect} = \frac{|\mathbf{P}_2[0, 3]|}{f'_{rect}}$ is the rectified stereo baseline distance in `RECONSTRUCTION_UNITS`, and $d > 0$ is the subpixel disparity in pixels.

---

## 3. Camera Convention
Preserves the project's exact camera conventions:
$$\mathbf{X}_c = \mathbf{R}_{cw} \mathbf{X}_w + \mathbf{t}_{cw}$$
$$\mathbf{C}_w = -\mathbf{R}_{cw}^T \mathbf{t}_{cw} \iff \mathbf{t}_{cw} = -\mathbf{R}_{cw} \mathbf{C}_w$$
$$u = f_x \frac{X_c}{Z_c} + c_x, \quad v = f_y \frac{Y_c}{Z_c} + c_y$$
`DepthMap` stores strictly optical depth $Z_c$ along the camera principal axis. It does **not** store Euclidean range $\|\mathbf{X}_c\|_2$, raw disparity, or inverse depth.

---

## 4. Rectification
- Dedicated class `StereoRectifier` computes relative rigid pose:
  $$\mathbf{R}_{rel} = \mathbf{R}_{src} \mathbf{R}_{ref}^T, \quad \mathbf{t}_{rel} = \mathbf{R}_{src} (\mathbf{C}_{ref} - \mathbf{C}_{src})$$
- Computes rectification transforms $\mathbf{R}_1, \mathbf{R}_2$ and projection matrices $\mathbf{P}_1, \mathbf{P}_2$ with `flags=cv2.CALIB_ZERO_DISPARITY` and `alpha=0.0`.
  > **Note on Rectification Crop**: `alpha=0.0` requests the valid-region-maximizing rectification crop; actual pixel validity remains governed by rectification validity/ROI handling and downstream validity checks.
- Typed container `StereoRectificationResult` holds $\mathbf{R}_1, \mathbf{R}_2, \mathbf{P}_1, \mathbf{P}_2, \mathbf{Q}$, rectified camera intrinsics, baseline distance, and valid ROIs.

---

## 5. Disparity Representation
- Disparity is computed in subpixel resolution ($d = \text{disparity\_raw} / 16.0$ from SGBM fixed-point representation).
- Only positive disparities $d > 0.0$ corresponding to forward-facing scene geometry are valid.

---

## 6. Disparity-to-Depth Conversion
- Explicit conversion $Z = (f'_{rect} \cdot B_{rect}) / d$ executed strictly when $d > 0$, $d \in [d_{min}, d_{max}]$, and $Z \in [Z_{min}, Z_{max}]$.
- Baseline $B_{rect}$ is derived directly from the relative camera centers in `RECONSTRUCTION_UNITS`.

---

## 7. Invalid-Depth Handling
- Zero tolerance: Disparities $\le 0$, non-finite (NaN, $\pm\infty$), or out-of-range are rejected.
- **Prohibitions Honored**:
  - `np.nan_to_num()`: 0 occurrences.
  - Value clipping into validity: 0 occurrences.
  - Fake default depth assignment (e.g. 0 or -1): 0 occurrences.
- Invalid pixels have `valid_mask == False` and are tagged with the appropriate `PointVisibilityState`.

---

## 8. Left-Right Consistency (LRC)
- Bidirectional verification compares reference disparity $d_L(u, v)$ against the source disparity at the corresponding pixel $c_{src} = \text{round}(u - d_L)$:
  $$|d_L(u, v) - |d_R(c_{src}, v)|| \le \tau_{lr} \quad (\text{HEURISTIC\_DEFAULT: } 1.5\text{ px})$$
- Inconsistent pixels are classified as `PointVisibilityState.INCONSISTENT` and discarded from valid depth estimation.

---

## 9. Confidence Semantics
- Strictly labeled `# HEURISTIC_SCORE` and bounded in $[0, 1]$.
- Blends normalized local Sobel gradient texture with exponential left-right consistency agreement:
  $$\text{confidence}_{raw} = 0.4 \cdot \text{texture} + 0.6 \cdot \exp(-|d_L - |d_R|| / \tau_{lr})$$
- Never claimed to be a Bayesian probability, posterior distribution, or physical measurement uncertainty.

---

## 10. Dynamic-Scene Limitations
- Propagates Phase 2 dynamic risk scores from `MVSInput.dynamic_risk_scores`.
- Confidence is attenuated: $\text{confidence} = \text{confidence}_{raw} \times (1.0 - 0.5 \cdot \text{dynamic\_risk})$.
- **Explicit Limitation**: "Dynamic-scene handling is frame/view-risk aware but not pixel-level semantic motion segmentation." Classical StereoSGBM cannot distinguish moving vehicles from static terrain at the pixel level.

---

## 11. Provenance
- `DenseStereoResult`, `DepthMap`, and `DepthConfidenceMap` store complete metadata:
  - `reference_frame_id`, `source_frame_id`
  - `rectified_fx`, `baseline_units`
  - `is_metric: False`, `unit: RECONSTRUCTION_UNITS`
  - `pair_swapped: bool` (records whether reference and source roles were canonicalized)
  - Algorithm name (`ClassicalStereoSGBM_DenseStereoConfig_v1.0`), matching parameters, and timestamp.

---

## 12. Synthetic Test Methodology & Accuracy Scoping
- All claims of stereo correspondence, disparity validation, and depth recovery are **strictly scoped to synthetic calibrated test scenes** with:
  - Known fronto-parallel textured plane at $Z_c \in \{2.0, 5.0, 25.0, 50.0\}$ reconstruction units.
  - Translated + rotated camera configurations ($\mathbf{R} \neq \mathbf{I}$, yaw/pitch angles).
  - Asymmetric intrinsics ($f_x \neq f_y$, $950 \neq 920$) and offset principal point ($c_x = 340 \neq 320$).
  - Mismatched image pairs testing LRC rejection.
  - Uniform/blank and noise images testing non-finite and zero disparity rejection.
- **Determinism Scope**: Repeated execution in the same software/hardware environment yields bit-exact outputs for the tested inputs.
- **No Real-Flight Generalization**: Real UAV optical performance, sensor noise, vibration, and environmental lighting remain uncharacterized.

---

### 13. Test Results
All 24 dedicated Phase 3E.1 unit tests passed:
- `test_rectification_geometry_translated_and_rotated_camera`: PASSED
- `test_degenerate_coincident_baseline_rejected`: PASSED
- `test_known_disparity_and_depth_recovery_on_synthetic_plane`: PASSED
- `test_invalid_nan_inf_zero_disparity_rejection`: PASSED
- `test_left_right_disparity_consistency_rejection`: PASSED
- `test_confidence_semantics_and_range_bounds`: PASSED
- `test_dynamic_risk_propagation_attenuates_confidence`: PASSED
- `test_deterministic_execution`: PASSED
- `test_reconstruction_unit_preservation_and_no_metric_claim`: PASSED
- `test_mvs_input_interface_compliance`: PASSED
- `test_out_of_bounds_correspondence_tagged_as_occluded`: PASSED
- `test_nan_and_inf_disparities_explicitly_rejected`: PASSED
- `test_zero_disparity_rejection_no_infinite_depth`: PASSED
- `test_provenance_preservation_details`: PASSED
- `test_translated_and_rotated_stereo_depth_estimation`: PASSED
- `test_arbitrary_camera_ordering_handled_consistently`: PASSED
- `test_camera_center_recovery_and_baseline_not_equal_to_translation_diff`: PASSED
- `test_adversarial_rotation_combinations_and_reversals`: PASSED
- `test_regression_pose_translation_is_extrinsic_not_optical_center`: PASSED
- `test_multi_depth_and_multi_baseline_synthetic_planes`: PASSED
- `test_3d_point_backprojection_roundtrip_off_axis`: PASSED
- `test_distortion_coefficients_handling_in_rectification`: PASSED
- `test_different_ref_and_src_intrinsics`: PASSED
- `test_provenance_records_pair_swap_flag`: PASSED

---

## 14. Forensic Audit #1: Camera Pair Ordering and Disparity Sign
- **Issue Investigated**: Whether `ClassicalStereoSGBMEstimator` correctly handles arbitrary reference/source camera ordering without producing inverted disparity or empty depth maps.
- **Mathematical Root Cause**: `cv2.stereoRectify` expects `T` vector defining $X_2 = R X_1 + T$. When $t_{rel}[0] \ge 0$ (source camera is to the left of reference camera), SGBM searches in the positive disparity direction which corresponds to matching points to the right, yielding predominantly negative raw disparities that were rejected by positive disparity checks.
- **Resolution**: Implemented automatic pair canonicalization in `ClassicalStereoSGBMEstimator.compute_dense_stereo` which evaluates $t_{rel}[0]$ and swaps pair roles internally when necessary, guaranteeing positive SGBM search direction and accurate depth recovery for both ordering permutations.

---

## 15. Forensic Audit #2: Camera Center vs Extrinsic Translation
- **Issue Investigated**: Whether pair-order canonicalization and `StereoRectifier` use actual camera optical centers $\mathbf{C}_w$ or incorrectly interpret extrinsic translation $\mathbf{t}_{cw}$ as optical centers.
- **Mathematical Derivation**:
  - Optical convention: $\mathbf{X}_c = \mathbf{R}_{cw}\mathbf{X}_w + \mathbf{t}_{cw}$
  - Camera optical center: $\mathbf{C}_w = -\mathbf{R}_{cw}^T \mathbf{t}_{cw} \iff \mathbf{t}_{cw} = -\mathbf{R}_{cw}\mathbf{C}_w$
  - Relative transform: $\mathbf{X}_{c, src} = (\mathbf{R}_{src}\mathbf{R}_{ref}^T)\mathbf{X}_{c, ref} + (\mathbf{t}_{src} - \mathbf{R}_{src}\mathbf{R}_{ref}^T \mathbf{t}_{ref})$
  - Relative translation: $\mathbf{t}_{rel} = \mathbf{t}_{src} - \mathbf{R}_{rel}\mathbf{t}_{ref} = \mathbf{R}_{src}(\mathbf{C}_{ref} - \mathbf{C}_{src})$
  - Physical baseline: $B = \|\mathbf{C}_{src} - \mathbf{C}_{ref}\|_2 = \|\mathbf{t}_{rel}\|_2 \neq \|\mathbf{t}_{src} - \mathbf{t}_{ref}\|_2$ (when $\mathbf{R}_{ref} \neq \mathbf{R}_{src}$).
- **Resolution**: Updated `StereoRectifier` and `ClassicalStereoSGBMEstimator` to strictly recover $\mathbf{C}_w = -\mathbf{R}_{cw}^T\mathbf{t}_{cw}$, compute $\mathbf{t}_{rel} = \mathbf{t}_{src} - \mathbf{R}_{rel}\mathbf{t}_{ref}$, and compute physical baseline from true optical centers.
- **Adversarial Verification**: Added adversarial tests across identity rotations, ref-only rotated, src-only rotated, both rotated, reversed ordering, asymmetric intrinsics, translated cameras, and varied baselines.

---

## 16. Full Regression Status
- **Pre-Phase 3E.1 Baseline**: 407 passed.
- **Current Total**: **431 passed in 19.21s, 0 failures, 0 regressions**.

---

## 17. Pyright Result
- **Command**: `npx pyright src/geometry/ tests/unit/test_phase3*.py`
- **Output**: **`0 errors, 0 warnings, 0 informations`**.

---

## 18. Known Limitations
1. **Low-Texture Regions**: StereoSGBM relies on local intensity variation and produces holes/invalid disparities over featureless flat surfaces (e.g. water, uniform asphalt, sky).
2. **Repetitive Patterns**: Ambiguous epipolar matching costs on repeating agricultural rows or tiled roofs.
3. **Motion Parallax & Dynamic Objects**: Vehicles in motion violate epipolar geometry, producing invalid disparities or false depths.
4. **Resolution Scaling**: Rectification and matching on large UAV images ($4K$) require downsampling or tiling to avoid memory bottlenecks.

---

## 19. What Remains Unimplemented
- Learned deep stereo estimators (DUSt3R, MASt3R, VGGT).
- Multi-baseline view aggregation beyond primary source pairs.
- CUDA-accelerated PatchMatch stereo kernels.
- Pixel-level semantic dynamic object segmentation masks.

---

## 20. Whether Real-Data Validation Exists
**Real-data validation not yet performed.** All results in this phase are verified against synthetic calibrated geometric benchmarks.

---

## 21. Scientific Claims
- **Metric Scale**: NOT CLAIMED. All geometry is strictly in `RECONSTRUCTION_UNITS`.
- **Real-Time Execution**: NOT CLAIMED.
- **Survey-Grade Accuracy**: NOT CLAIMED.
- **Robustness**: Validated only under synthetic test conditions with defined thresholds.

---

## 22. Final Phase Status

```
================================================================================
               PHASE 3E.1 — IMPLEMENTED, AUDITED & READY FOR LOCK
================================================================================
  Test Suite:    431/431 PASSED (24 dedicated Phase 3E.1 tests, 0 regressions)
  Pyright Types: 0 errors, 0 warnings, 0 informations
  Audit #1:      Arbitrary camera pair ordering canonicalization verified
  Audit #2:      Camera center C_w = -R^T t_cw vs extrinsic translation t_cw verified
  Audit Final:   Sections A through T falsification criteria verified
  Scale Guard:   RECONSTRUCTION_UNITS strictly enforced; 0 metric claims
  Dependencies:  Existing cv2/numpy/scipy/pydantic; 0 new packages installed
  Status:        READY FOR LOCK
================================================================================
```
