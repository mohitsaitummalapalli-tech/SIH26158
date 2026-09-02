# Phase 3E.0 Deep Scientific & Mathematical Audit Report

## 1. Executive Verdict

```
================================================================================
                    PHASE 3E.0 — AUDIT PASS
================================================================================
  Camera Convention: PASS (Exact SO(3) + Optical Pinhole)
  Depth Semantics:   PASS (Strictly Z_c, Optical Depth along Principal Axis)
  Scale & Gauge:     PASS (Reconstruction Units Only, Zero Metric Claims)
  Invalid Depth:     PASS (Zero-Tolerant, No NaN/Inf/Zero Replacement)
  Consistency Check: PASS (Rigorous 2-Way Reprojection + Relative Depth)
  Test Suite:        PASS (407/407 Tests Passing, 0 Regressions)
  Pyright Types:     PASS (0 Errors, 0 Warnings, 0 Informations)
================================================================================
```

A forensic scientific audit was conducted across all 15 audit dimensions specified for **Phase 3E.0 (Dense Multi-View Stereo Architecture & Contracts)**. 

The implementation in [`src/geometry/mvs.py`](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/src/geometry/mvs.py), its exports in [`src/geometry/__init__.py`](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/src/geometry/__init__.py), its test suite in [`tests/unit/test_phase3e_mvs_contracts.py`](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/tests/unit/test_phase3e_mvs_contracts.py), and architectural documentation in [`docs/architecture/PHASE_3E_DENSE_MVS.md`](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/docs/architecture/PHASE_3E_DENSE_MVS.md) strictly adhere to all mathematical invariants and project guardrails.

---

## 2. File Inventory & Categorization

| File | Item | Type | Category |
| :--- | :--- | :--- | :--- |
| `src/geometry/mvs.py` | `MVSFailureReason` | Enum (13 members) | **A. Pure Contract** |
| `src/geometry/mvs.py` | `PointVisibilityState` | Enum (7 members) | **A. Pure Contract** |
| `src/geometry/mvs.py` | `PointValidationStatus` | Enum (4 members) | **A. Pure Contract** |
| `src/geometry/mvs.py` | `DepthUnit` | Enum (2 members) | **A. Pure Contract** |
| `src/geometry/mvs.py` | `MVSConfig` | Frozen Dataclass | **A. Pure Contract** (Heuristic defaults) |
| `src/geometry/mvs.py` | `MVSInput` | Dataclass | **A. Pure Contract** |
| `src/geometry/mvs.py` | `MVSViewPair` | Dataclass | **A. Pure Contract** |
| `src/geometry/mvs.py` | `MVSViewGraph` | Dataclass | **A. Pure Contract** |
| `src/geometry/mvs.py` | `DepthMap` | Dataclass | **A. Pure Contract** + query helper |
| `src/geometry/mvs.py` | `DepthConfidenceMap` | Dataclass | **A. Pure Contract** |
| `src/geometry/mvs.py` | `DensePointObservation` | Dataclass | **A. Pure Contract** |
| `src/geometry/mvs.py` | `DensePointCloud` | Dataclass | **A. Pure Contract** |
| `src/geometry/mvs.py` | `MVSViewPairSelector` | Abstract Base Class | **A. Pure Contract** |
| `src/geometry/mvs.py` | `DepthConsistencyChecker` | Abstract Base Class | **A. Pure Contract** |
| `src/geometry/mvs.py` | `DensePointFusion` | Abstract Base Class | **A. Pure Contract** |
| `src/geometry/mvs.py` | `IMVSDepthEstimator` | Abstract Base Class | **A. Pure Contract** |
| `src/geometry/mvs.py` | `MVSGeometryMath.backproject_pixel` | Static Method | **B. Mathematical Utility** |
| `src/geometry/mvs.py` | `MVSGeometryMath.project_world_point` | Static Method | **B. Mathematical Utility** |
| `src/geometry/mvs.py` | `depth_to_world_points` | Function | **B. Mathematical Utility** |
| `src/geometry/mvs.py` | `HeuristicViewPairSelector` | Concrete Class | **C. Reference Implementation** |
| `src/geometry/mvs.py` | `GeometricDepthConsistencyChecker` | Concrete Class | **C. Reference Implementation** |
| `src/geometry/mvs.py` | `VoxelGridDensePointFusion` | Concrete Class | **C. Reference Implementation** |
| `src/geometry/mvs.py` | `MVSValidator` | Concrete Class | **C. Reference Implementation** |
| *Entire Subsystem* | *Production Dense MVS Solver* | N/A | **D. Actual MVS Algorithms: NONE (0)** |

**Finding**: No actual dense MVS matching algorithm (PatchMatch, semi-global matching, plane sweep, or learned stereo network) was implemented. Implementation is cleanly confined to architecture, contracts, and reference utilities.

---

## 3. Camera Convention Audit

### Mathematical Specification:
$$\mathbf{X}_c = \mathbf{R}_{cw} \mathbf{X}_w + \mathbf{t}_{cw}$$
$$\mathbf{C}_w = -\mathbf{R}_{cw}^T \mathbf{t}_{cw} \iff \mathbf{t}_{cw} = -\mathbf{R}_{cw} \mathbf{C}_w$$
$$u = f_x \frac{X_c}{Z_c} + c_x, \quad v = f_y \frac{Y_c}{Z_c} + c_y$$
$$\mathbf{X}_c = Z_c \mathbf{K}^{-1} [u, v, 1]^T, \quad \mathbf{X}_w = \mathbf{R}_{cw}^T (\mathbf{X}_c - \mathbf{t}_{cw})$$

### Code Inspection:
1. **Backprojection** ([`src/geometry/mvs.py:309-317`](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/src/geometry/mvs.py#L309-L317)):
   ```python
   x_c = depth_z * (u - K.cx) / K.fx
   y_c = depth_z * (v - K.cy) / K.fy
   z_c = depth_z
   X_c = np.array([x_c, y_c, z_c], dtype=np.float64)
   X_w = R_cw.T @ (X_c - t_cw)
   ```
   Matches $Z_c \mathbf{K}^{-1} [u, v, 1]^T$ and $\mathbf{R}_{cw}^T (\mathbf{X}_c - \mathbf{t}_{cw})$ identically.
2. **Projection** ([`src/geometry/mvs.py:330-340`](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/src/geometry/mvs.py#L330-L340)):
   ```python
   X_c = R_cw @ X_w + t_cw
   z_c = float(X_c[2])
   ...
   u = K.fx * (float(X_c[0]) / z_c) + K.cx
   v = K.fy * (float(X_c[1]) / z_c) + K.cy
   ```
   Matches $\mathbf{X}_c = \mathbf{R}_{cw} \mathbf{X}_w + \mathbf{t}_{cw}$ and pinhole projection identically.
3. **Camera Center Conversion** ([`src/geometry/mvs.py:355-357`](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/src/geometry/mvs.py#L355-L357)):
   ```python
   R_cw = np.array(extrinsic_pose.rotation_matrix, dtype=np.float64)
   c_w = np.array(extrinsic_pose.translation_vector, dtype=np.float64)
   t_cw = -R_cw @ c_w
   ```
   $\mathbf{t}_{cw} = -\mathbf{R}_{cw} \mathbf{C}_w \iff \mathbf{C}_w = -\mathbf{R}_{cw}^T \mathbf{t}_{cw}$.

**Verdict**: **PASS**.

---

## 4. Depth Semantics Audit

Trace of depth across the entire lifecycle:
1. `DepthMap.depth_array`: Stored as optical depth $Z_c$ along the principal optical axis.
2. `DepthMap.get_depth_at(u, v)`: Retrieves continuous depth $Z_c$ with bounds and validity checks.
3. `MVSGeometryMath.backproject_pixel`: Ray scaling uses $x_c = Z_c \frac{u-c_x}{f_x}$, $y_c = Z_c \frac{v-c_y}{f_y}$, $z_c = Z_c$. If depth were radial distance $d = \|\mathbf{X}_c\|_2$, the ray would be normalized by $\sqrt{x_n^2 + y_n^2 + 1}$. The absence of this normalization confirms the quantity is strictly optical depth $Z_c$.
4. `GeometricDepthConsistencyChecker`: Compares $Z_{src, proj} = X_{c, src}[2]$ with $Z_{src, obs}$ from the source depth map. Both terms are optical depths $Z_c$.
5. Nowhere is disparity $d$ conflated with Cartesian depth.

**Verdict**: **PASS**.

---

## 5. Invalid-Depth Audit

Code inspection for non-finite and non-positive depth handling:
- **`MVSGeometryMath.backproject_pixel`** ([line 307](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/src/geometry/mvs.py#L307)):
  ```python
  if not (math.isfinite(depth_z) and depth_z > 1e-6):
      return None, False
  ```
  Immediately rejects NaN, Inf, $\le 0$, and negative depths.
- **`MVSGeometryMath.project_world_point`** ([line 333](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/src/geometry/mvs.py#L333)):
  ```python
  if z_c <= 1e-6 or not np.all(np.isfinite(X_c)):
      return None, z_c, False
  ```
  Points on or behind the camera sensor plane ($Z_c \le 0$) fail cheirality and are rejected.
- **`depth_to_world_points`** ([lines 364-374](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/src/geometry/mvs.py#L364-L374)):
  Strictly ignores pixels with `valid_mask == False`, out-of-range depths ($z \notin [Z_{min}, Z_{max}]$), or failed backprojections.
- **Search for fallback defaults**:
  - `np.nan_to_num`: **0 occurrences**.
  - Depth clipping (`np.clip` on depth): **0 occurrences**.
  - Defaulting to `0.0` or `-1.0`: **0 occurrences**.

**Verdict**: **PASS**. Invalid depths are strictly dropped, never fabricated into 3D points.

---

## 6. Cross-View Consistency Audit

Inspection of `GeometricDepthConsistencyChecker` ([lines 570-652](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/src/geometry/mvs.py#L570-L652)):

For reference pixel $(u_{ref}, v_{ref}, z_{ref})$:
1. **Backprojects** to world point $\mathbf{X}_w = \mathbf{R}_{ref}^T (\mathbf{X}_c - \mathbf{t}_{ref})$.
2. **Transforms** to source camera $\mathbf{X}_{c, src} = \mathbf{R}_{src} \mathbf{X}_w + \mathbf{t}_{src}$.
3. **Cheirality check**: $Z_{src, proj} = X_{c, src}[2] > 10^{-6}$. If false $\to$ `PointVisibilityState.OCCLUDED`.
4. **Raster bounds check**: $0 \le u_{src} < W_{src}$ and $0 \le v_{src} < H_{src}$. If outside $\to$ `PointVisibilityState.OCCLUDED`.
5. **Source depth query**: $Z_{src, obs}$ retrieved via `src_depth.get_depth_at(u_{src}, v_{src})`. If invalid $\to$ `PointVisibilityState.INCONSISTENT`.
6. **Relative depth consistency**:
   $$\frac{|Z_{src, proj} - Z_{src, obs}|}{Z_{src, proj}} \le \tau_{depth} \quad (\text{HEURISTIC\_DEFAULT: } 0.05)$$
   If violated $\to$ `PointVisibilityState.INCONSISTENT`.
7. **Two-Way Reprojection back-check**:
   Backprojects source observation $\mathbf{X}_{w, src}$ and projects into reference camera $\mathbf{p}_{ref, back}$.
   Verifies:
   $$\|\mathbf{p}_{ref} - \mathbf{p}_{ref, back}\|_2 \le \tau_{reproj} \quad (\text{HEURISTIC\_DEFAULT: } 1.5\text{ px})$$
   If violated $\to$ `PointVisibilityState.INCONSISTENT`.
8. Only when both pass: `consistency_mask = True`, `visibility_state = VALID`.

**Hand-Check Verification**:
- Two cameras with baseline $B = 1.0$, focal length $f = 1000$, principal point $(500, 500)$.
- Point at $(0, 0, 10)$ in world.
- Ref pixel: $(500, 500)$, depth $10.0$.
- Source pixel: $(400, 500)$, depth $10.0$.
- Both checks evaluate to $0.0 \le 0.05$ and $0.0\text{ px} \le 1.5\text{ px}$, correctly marking `VALID`.
- If source depth is corrupted to $13.0$, relative error is $30\% > 5\%$, correctly marking `INCONSISTENT`.

**Verdict**: **PASS**.

---

## 7. View-Pair Selection Audit

Inspection of `HeuristicViewPairSelector` ([lines 472-567](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/src/geometry/mvs.py#L472-L567)):
- **GNSS vs. Optical Baseline**: Baseline is computed strictly from camera optical centers $\mathbf{C}_w$ in `RECONSTRUCTION_UNITS` ($\|\mathbf{C}_{src} - \mathbf{C}_{ref}\|_2$). It does not touch GNSS metadata and is explicitly named `baseline_proxy`.
- **Relative Rotation**: Geodesic SO(3) distance $\theta = \arccos((\text{Tr}(\mathbf{R}_{rel}) - 1) / 2)$.
- **Rejection Criteria**:
  - Coincident centers ($\|\mathbf{C}\| < 10^{-4}$): Rejected with `"Coincident camera centers (degenerate baseline)"`.
  - Viewing angle $> 40.0^\circ$: Rejected with `"Viewing angle too steep"`.
  - Overlap proxy $< 0.25$: Rejected with `"Insufficient visual overlap"`.
- **Deterministic Ordering**: `frame_ids` sorted; candidates sorted by `viewpoint_suitability_score` in descending order.
- **Dynamic Scene Interaction**:
  ```python
  dyn_risk = max(mvs_input.dynamic_risk_scores.get(ref_id, 0.0), mvs_input.dynamic_risk_scores.get(src_id, 0.0))
  suitability *= (1.0 - 0.5 * dyn_risk)
  ```
  High dynamic risk directly down-weights suitability score and re-ranks or filters pairs.

**Verdict**: **PASS**.

---

## 8. Confidence Semantics Audit

Inspection of `DepthConfidenceMap`:
- Photometric and geometric confidence arrays are explicitly labeled `# HEURISTIC_SCORE` in $[0, 1]$.
- No claim of probabilistic Bayesian posterior or ground truth certainty is made.
- Multi-criteria metadata is preserved:
  - `photometric_confidence`
  - `geometric_consistency_confidence`
  - `support_view_count`
  - `visibility_state`
  - `overall_confidence`

**Verdict**: **PASS**.

---

## 9. Dense Fusion Audit

Inspection of `VoxelGridDensePointFusion` ([lines 655-745](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/src/geometry/mvs.py#L655-L745)):
- **Spatial Deduplication**: 3D voxel grid with resolution $\Delta_{voxel} = 0.02$ reconstruction units. Voxel keys sorted deterministically.
- **Multi-View Support**: Points must be observed by $\ge \text{min\_consistent\_views}$ (`HEURISTIC_DEFAULT: 2`) unique camera frames; single-view observations are discarded.
- **Centroid Computation**: Confidence-weighted spatial average $\mathbf{X}_{fused} = \frac{\sum w_i \mathbf{X}_i}{\sum w_i}$.
- **Discontinuity Preservation**: Small voxel resolution ($\Delta = 0.02$) prevents cross-surface blurring. Documented limitation notes that fine geometry closer than $\Delta_{voxel}$ merges.
- **Provenance Retention**: `source_frame_ids: List[List[str]]` tracks the exact contributing camera IDs per point.

**Verdict**: **PASS**.

---

## 10. Dynamic Scene Audit

Tracing dynamic risk:
- Phase 2 dynamic risk scores are accepted in `MVSInput.dynamic_risk_scores`.
- Applied in `HeuristicViewPairSelector` to penalize pair suitability.
- Stored on `MVSViewPair.dynamic_risk`.
- **Status in Phase 3E.0**: Dynamic risk affects pair selection and propagates into pair metadata. Per-pixel dynamic segmentation masking will execute inside the Phase 3E.1 depth estimator when raw image tensors are loaded.

**Verdict**: **PASS**.

---

## 11. Gauge & Scale Audit

Search across all Phase 3E code for scale terms:
- `depth_unit`: `DepthUnit.RECONSTRUCTION_UNITS`.
- `is_metric_scale`: `False`.
- `MVSValidator.validate_point_cloud`:
  ```python
  if point_cloud.is_metric_scale:
      diags.append("Violation of Phase 3E constraint: DensePointCloud claims metric scale without external calibration.")
      return False, MVSFailureReason.DENSE_FUSION_FAILED, diags
  ```
  Actively flags and fails any illegal claim of metric scale!
- Monocular gauge is inherited directly from Phase 3D.1 without scale distortion.

**Verdict**: **PASS**.

---

## 12. Provenance Audit

Trace of dense point provenance:
- **Observation level**: Tracks `reference_frame_id`, `pixel_coord`, `depth`, and `source_view_support_count`.
- **Fused point cloud level**: Tracks `source_frame_ids` (list of observing camera IDs), `support_counts`, `confidences`, `visibility_states`, and algorithm metadata.
- **Memory optimization**: Individual $(u, v)$ coordinates are aggregated into camera ID lists upon fusion to prevent unbounded memory growth on large UAV datasets ($10^6$ points).

**Verdict**: **PASS**.

---

## 13. Test Quality Audit

| Contract Invariant | Test Method | Classification | Justification |
| :--- | :--- | :--- | :--- |
| Camera Convention | `test_camera_convention_projection_backprojection_roundtrip` | **STRONG TEST** | Exact numeric roundtrip recovery at $10^{-8}$ with translated pose. |
| Depth Backprojection | `test_known_camera_known_depth_exact_backprojection` | **STRONG TEST** | Compares center and off-axis pixels against analytic ground truth. |
| Invalid Depth | `test_invalid_and_non_positive_depth_rejection` | **STRONG TEST** | Verifies rejection of 6 invalid states ($0, -5, -10^{-8}, \text{NaN}, \pm\infty$). |
| Depth Query | `test_depth_map_query_invariants` | **STRONG TEST** | Tests valid, masked, and out-of-bounds queries. |
| View-Pair Multi-Source | `test_view_pair_selector_multi_source_and_ordering` | **STRONG TEST** | Verifies ranking and multi-source view constraint. |
| Degenerate Baseline / Angle | `test_view_pair_selector_rejection_rules` | **STRONG TEST** | Verifies zero baseline and steep angle rejections. |
| Dynamic Scene Risk | `test_dynamic_risk_propagation_into_view_pairs` | **STRONG TEST** | Verifies dynamic risk strictly penalizes score and persists. |
| Consistency Acceptance | `test_cross_view_geometric_depth_consistency_acceptance` | **STRONG TEST** | Exact 3D point across two views accepted as `VALID`. |
| Consistency Disagreement | `test_cross_view_geometric_depth_inconsistency_rejection` | **STRONG TEST** | Corrupted source depth ($+30\%$) rejected as `INCONSISTENT`. |
| Occlusion & Bounds | `test_cross_view_occlusion_and_bounds_rejection` | **STRONG TEST** | Out-of-bounds projection tagged `OCCLUDED`. |
| Depth Filtering | `test_depth_to_world_points_backprojection_filtering` | **STRONG TEST** | Masked and too-close pixels excluded from backprojection. |
| Fusion Deduplication | `test_dense_point_fusion_spatial_deduplication_and_support_count` | **STRONG TEST** | 3 multi-view points in voxel fuse to centroid with support count 3. |
| Support Filtering | `test_dense_point_fusion_filters_insufficient_multi_view_support` | **STRONG TEST** | Single-view point rejected when `min_consistent_views=2`. |
| Gauge Preservation | `test_reconstruction_unit_preservation_no_meters_claim` | **STRONG TEST** | Verifies `RECONSTRUCTION_UNITS` and `is_metric_scale=False`. |
| Scale Violation Guard | `test_validator_rejects_premature_metric_scale_claim` | **STRONG TEST** | Validator fails if metric scale is claimed. |
| Input Validation | `test_validator_input_and_depth_map_checks` | **STRONG TEST** | Checks insufficient frames and array shape mismatches. |
| Extensible Estimator | `test_abstract_depth_estimator_interface_compliance` | **STRONG TEST** | Subclasses abstract estimator and validates plugin pipeline. |
| Taxonomy Completeness | `test_failure_and_visibility_taxonomies_completeness` | **STRONG TEST** | Validates all enum members are distinct and non-empty. |

---

## 14. Recommended Adversarial Tests for Phase 3E.1

While all Phase 3E.0 contract invariants pass, the following 12 adversarial test cases are documented for Phase 3E.1 test suite expansion:
1. **Rotated Camera Roundtrip**: Test backprojection roundtrip with full 3-axis rotation ($R \neq I$).
2. **Convergent Stereo Pair**: Consistency check where source camera is pitched/yawed toward reference optical axis.
3. **Asymmetric Intrinsics**: Test camera model with $f_x \neq f_y$ and non-centered $(c_x, c_y)$.
4. **Negative Cheirality**: Point located behind source camera ($Z_{src, proj} < 0$).
5. **Depth Discontinuity**: Two distinct surfaces within the same visual neighborhood.
6. **Extreme Baseline Parallax**: Baseline larger than scene depth ($B > Z$).
7. **Empty Depth Map**: All pixels masked in depth map.
8. **Malformed Pose Matrix**: Non-SO(3) matrix with $\det(R) \neq 1$.
9. **Fusion Key Ordering**: Multi-threaded or shuffled input order invariant test.
10. **Maximum Dynamic Risk**: Dynamic risk = 1.0 test case.
11. **Grid Resolution Boundary**: Points placed exactly on voxel grid boundary.
12. **Extreme Gauge Magnitude**: Poses scaled by $10^4$ to ensure no numerical overflow.

---

## 15. Scientific Claim Audit

All claims in documentation and reports were audited against empirical evidence:
- No claim of "sub-millimeter precision", "survey-grade accuracy", or "real-time execution".
- The gauge limitation is explicitly stated: *"Phase 3E dense geometry remains in the SfM reconstruction gauge. Absolute metric scale is not established here."*
- All default thresholds are explicitly labeled `HEURISTIC_DEFAULT`.
- All confidence scores are explicitly labeled `HEURISTIC_SCORE`.

**Verdict**: **PASS**.

---

## 16. Dependency & Architecture Audit

- `pyproject.toml` and `requirements.txt` inspected:
  - Pinned dependencies: `numpy>=1.24.0`, `scipy>=1.10.0`, `pydantic>=2.0.0`.
  - No new external libraries or models were installed.
- Repository remains clean and lightweight.

**Verdict**: **PASS**.

---

## 17. Final Status

```
================================================================================
                         PHASE 3E.0 — AUDIT PASS
================================================================================
   Full Regression: 407/407 PASSED | Pyright: 0 ERRORS | Baseline Regressions: 0
================================================================================
```
