# Phase 3E.2 Dense 3D Point Generation & Geometric Validation Implementation Report

## 1. Objective
Phase 3E.2 implements the canonical 3D unprojection, coordinate transformation, and geometric validation layer that maps validated 2D dense stereo depth maps (Phase 3E.1) into structured 3D point observations (`DensePointObservation`) and consolidated point clouds (`DensePointCloud`) in relative reconstruction coordinates (`RECONSTRUCTION_UNITS`).

---

## 2. Existing Conventions Discovered
Through forensic inspection of `src/geometry/dense_stereo.py`, `src/geometry/mvs.py`, `src/geometry/sfm.py`, and `src/geometry/bundle_adjustment.py`:
- **Camera Optical Model**:
  $$\mathbf{X}_c = \mathbf{R}_{cw} \mathbf{X}_w + \mathbf{t}_{cw}$$
  $$\mathbf{C}_w = -\mathbf{R}_{cw}^T \mathbf{t}_{cw} \iff \mathbf{t}_{cw} = -\mathbf{R}_{cw} \mathbf{C}_w$$
- **Extrinsics vs Optical Center**:
  `ExtrinsicPose.translation_vector` represents extrinsic translation $\mathbf{t}_{cw}$, **NOT** the optical center $\mathbf{C}_w$.
- **Stereo Rectification Rotation**:
  `cv2.stereoRectify` computes rotation $\mathbf{R}_1$ bringing original camera coordinates into the rectified frame:
  $$\mathbf{X}_{rect} = \mathbf{R}_1 \mathbf{X}_{c, orig} \iff \mathbf{X}_{c, orig} = \mathbf{R}_1^T \mathbf{X}_{rect}$$
- **Depth Map Representation**:
  Depth is optical depth along the camera principal axis $Z_{rect}$ in `RECONSTRUCTION_UNITS`, with `is_metric: False`.

---

## 3. Mathematical Derivations

### 3.1 Rectified Pixel to Rectified 3D Point
Given rectified pixel $(u_{rect}, v_{rect})$, rectified optical depth $Z_{rect}$, and rectified projection matrix $\mathbf{P}_1$:
$$f'_{rect} = \mathbf{P}_1[0, 0], \quad c'_{x, rect} = \mathbf{P}_1[0, 2], \quad c'_{y, rect} = \mathbf{P}_1[1, 2]$$
$$\mathbf{X}_{rect} = \begin{bmatrix} Z_{rect} \frac{u_{rect} - c'_{x, rect}}{f'_{rect}} \\ Z_{rect} \frac{v_{rect} - c'_{y, rect}}{f'_{rect}} \\ Z_{rect} \end{bmatrix}$$

### 3.2 Rectified Frame to Original Camera Frame
$$\mathbf{X}_{c, orig} = \mathbf{R}_1^T \mathbf{X}_{rect}$$

### 3.3 Original Camera Frame to World Frame
$$\mathbf{X}_w = \mathbf{R}_{ref}^T (\mathbf{X}_{c, orig} - \mathbf{t}_{ref}) = \mathbf{R}_{ref}^T (\mathbf{R}_1^T \mathbf{X}_{rect} - \mathbf{t}_{ref})$$
Equivalently:
$$\mathbf{X}_w = \mathbf{C}_{w, ref} + \mathbf{R}_{ref}^T \mathbf{R}_1^T \mathbf{X}_{rect}$$

---

## 4. Point Generation Algorithm
1. Extract $(\mathbf{P}_1, \mathbf{R}_1)$ from `stereo_result.rectification`.
2. Iterate through each pixel $(row, col)$ in the depth raster:
   - Filter invalid mask pixels (`valid_mask == False`).
   - Validate disparity $d > 0.0$.
   - Backproject $(col, row, Z_{rect}) \to \mathbf{X}_{rect} \to \mathbf{X}_{c, orig} \to \mathbf{X}_w$.
3. Execute `DensePointGeometricValidator.validate()`:
   - Check coordinate finiteness ($\mathbf{X}_w, \mathbf{X}_{c, orig}, \mathbf{X}_{rect}$).
   - Enforce cheirality ($Z_{c, orig} > 0$ and $Z_{rect} > 0$).
   - Check depth bounds $Z_{rect} \in [Z_{min}, Z_{max}]$.
   - Enforce minimum stereo confidence $C \ge C_{min}$.
   - Perform forward projection $\mathbf{X}_w \to (u', v')$ and check reprojection error $\Delta_{reproj} \le \tau_{reproj}$.
4. Construct `ValidatedDensePoint` with full metadata and provenance.
5. Aggregate validated points into `DensePointCloud` container.

---

## 5. Geometric Validation Rules & Rejection Taxonomy
Explicit rejection reasons tracked in `DensePointGenerationResult.rejection_breakdown`:
- `INVALID_DEPTH_VALUE`
- `NON_POSITIVE_DEPTH`
- `OUT_OF_DEPTH_BOUNDS`
- `NON_FINITE_COORDINATES`
- `CHEIRALITY_VIOLATION`
- `REPROJECTION_ERROR_EXCEEDED`
- `LOW_CONFIDENCE`
- `OCCLUDED_OR_INCONSISTENT`
- `INVALID_DISPARITY`
- `MASKED_OUT`

---

## 6. Confidence Semantics
- Stereo confidence is preserved directly from Phase 3E.1 (`DepthConfidenceMap.overall_confidence`).
- Explicitly labeled `HEURISTIC_SCORE` and bounded in $[0, 1]$.
- No claims of Bayesian posterior probabilities or physical measurement uncertainties.

---

## 7. Provenance Design
Every point records:
- `reference_frame_id`, `source_frame_id`
- `pixel_coord_rect`: $(u, v)$
- `depth`: $Z_{rect}$ in `RECONSTRUCTION_UNITS`
- `disparity`: $d$ in pixels
- `stereo_confidence`: float in $[0, 1]$
- `reprojection_error_px`: float in pixels
- `is_metric`: `False`
- `pair_swapped`: bool

---

## 8. Falsification Matrix

| Mathematical Invariant | Test Method | Failure Condition Caught If Buggy |
|---|---|---|
| Inversion of Rectification $\mathbf{R}_1^T$ | `test_rectification_rotation_consistency_and_falsification` | Fails if code omits $\mathbf{R}_1^T$ (error $> 2.58$ units on 15 deg tilt) |
| World Translation $\mathbf{X}_w = \mathbf{R}^T(\mathbf{X}_c - \mathbf{t})$ | `test_regression_proving_tcw_is_extrinsic_translation_not_optical_center` | Fails if code treats $\mathbf{t}$ as $\mathbf{C}_w$ or adds $\mathbf{t}$ directly (error $> 5.0$ units) |
| Off-Axis Pixel Round-Trip | `test_off_axis_points_sweep` | Fails if principal point offset or anisotropic focal lengths are mishandled |
| Cheirality & Depth Bounds | `test_geometric_validator_rejection_modes` | Fails if $Z \le 0$ or out-of-bounds depths pass validation |
| Reprojection Error Rejection | `test_geometric_validator_rejection_modes` | Fails if points with inconsistent image projections are accepted |
| Deterministic Execution | `test_deterministic_repeated_execution` | Fails if point generation outputs vary across repeated runs |
| Scale Guard | `test_full_dense_point_generator_pipeline` | Fails if `is_metric_scale` is True or metric units are asserted |

---

## 9. Verification Results
- **Phase 3E.2 Tests**: **9 / 9 PASSED** (`tests/unit/test_phase3e2_dense_point_generation.py`)
- **Total Test Suite**: **440 / 440 PASSED in 50.16s** (0 failures, 0 regressions)
- **Pyright Static Type Checking**: **`0 errors, 0 warnings, 0 informations`**

---

## 10. Known Limitations
1. **Single-Pair Point Cloud**: Generates points from individual reference-source stereo observations; multi-view volumetric fusion across dozens of video frames is deferred to Phase 3E.3.
2. **Surface Topology**: Points represent unorganized 3D point observations without surface normal estimation or polygon meshing.
3. **Occlusion Boundaries**: Points at sharp depth discontinuities may exhibit edge bleed if stereo matching lacked subpixel segmentation.

---

## 11. Real-Data Boundary & Explicit Non-Claims
- **Proven**: Exact mathematical backprojection, rectification rotation inversion, and geometric validation on calibrated synthetic benchmarks.
- **Not Proven**: Real UAV rolling shutter dynamics, atmospheric haze, flight vibration, and real-world metric surface precision.
- **Explicit Non-Claims**: No claims of "metric accuracy", "production readiness", or "100% precision" are made.

---

## 12. Final Phase Status

```
================================================================================
               PHASE 3E.2 — IMPLEMENTED, AUDITED & READY FOR REVIEW
================================================================================
  Test Suite:    440/440 PASSED (9 dedicated Phase 3E.2 tests, 0 regressions)
  Pyright Types: 0 errors, 0 warnings, 0 informations
  Scale Guard:   RECONSTRUCTION_UNITS strictly enforced; 0 metric claims
  Dependencies:  Existing cv2/numpy/scipy/pydantic; 0 new packages installed
  Status:        READY FOR AUDIT (NOT LOCKED)
================================================================================
```
