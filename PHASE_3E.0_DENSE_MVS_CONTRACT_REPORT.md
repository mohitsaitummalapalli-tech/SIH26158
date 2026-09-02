# Phase 3E.0 Architecture & Contract Report: Dense Multi-View Stereo (MVS)

## 1. Executive Summary

Phase 3E.0 establishes the formal mathematical architecture, typed data contracts, abstract interfaces, validation suites, and synthetic unit tests for the **Dense Multi-View Stereo (MVS)** subsystem.

In strict compliance with the Phase 3E.0 directive:
- **No production MVS depth solver has been implemented yet.**
- **No external heavy models (DUSt3R, VGGT, etc.) or weights have been downloaded.**
- **All dense geometry remains strictly in the relative SfM coordinate gauge (`RECONSTRUCTION_UNITS`).**
- **No claim of absolute metric accuracy or physical meters is made.**
- **Full test suite passes: 407 passed, 0 failures.**
- **Pyright verification: 0 errors, 0 warnings, 0 informations.**

---

## 2. Files Created and Modified

| File | Action | Purpose |
| :--- | :--- | :--- |
| [`src/geometry/mvs.py`](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/src/geometry/mvs.py) | **CREATED** | Core Phase 3E.0 contracts, enums, math utilities, abstract interfaces, reference implementations (`HeuristicViewPairSelector`, `GeometricDepthConsistencyChecker`, `VoxelGridDensePointFusion`, `MVSValidator`). |
| [`src/geometry/__init__.py`](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/src/geometry/__init__.py) | **MODIFIED** | Exported all Phase 3E.0 typed symbols and interfaces into geometry namespace. |
| [`tests/unit/test_phase3e_mvs_contracts.py`](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/tests/unit/test_phase3e_mvs_contracts.py) | **CREATED** | 18 synthetic mathematical contract verification tests covering all 11 required contract behaviors (A–K). |
| [`docs/architecture/PHASE_3E_DENSE_MVS.md`](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/docs/architecture/PHASE_3E_DENSE_MVS.md) | **CREATED** | Detailed architecture specification covering 16 required sections, conventions, and limitations. |
| [`PHASE_3E.0_DENSE_MVS_CONTRACT_REPORT.md`](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/PHASE_3E.0_DENSE_MVS_CONTRACT_REPORT.md) | **CREATED** | Formal milestone contract report. |
| [`docs/reports/PHASE_3E.0_DENSE_MVS_CONTRACT_REPORT.md`](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/docs/reports/PHASE_3E.0_DENSE_MVS_CONTRACT_REPORT.md) | **CREATED** | Persistent copy under documentation records. |

---

## 3. Architecture & Decoupled Pipeline Stages

Phase 3E defines ten decoupled stages with explicit typed boundaries:

1. **Optimized SfM Input**: Consumes `SparseReconstructionResult` from Phase 3D.1 containing registered camera poses and calibrated intrinsics.
2. **MVS Candidate View Selection**: `MVSViewPairSelector` evaluates baseline parallax, rotation angles, visual overlap, and Phase 2 dynamic risk.
3. **Reference / Source View Pairs**: Represented deterministically by `MVSViewGraph` and `MVSViewPair`.
4. **Dense Correspondence / Depth Estimation Interface**: `IMVSDepthEstimator` provides an extensible contract for future algorithms (classical, plane-sweep, learned).
5. **Depth Map & Confidence Representation**: `DepthMap` (optical depth $Z_c$) and `DepthConfidenceMap` (photometric, geometric, and support criteria).
6. **Cross-View Geometric Consistency**: `DepthConsistencyChecker` performs two-way reprojection and relative depth verification.
7. **Depth Filtering & Occlusion Tagging**: Categorizes observations into `PointVisibilityState` (`VISIBLE`, `OCCLUDED`, `INCONSISTENT`, `VALID`).
8. **3D Backprojection**: `depth_to_world_points` backprojects verified depth pixels into `DensePointObservation` records.
9. **Dense Point Fusion**: `DensePointFusion` performs spatial voxel deduplication and multi-view support thresholding.
10. **Dense Point Cloud Output**: `DensePointCloud` storing coordinates in `RECONSTRUCTION_UNITS`, support counts, and provenance.

---

## 4. Mathematical Conventions & Contracts

### 4.1. Camera Coordinate Convention
Preserves OpenCV optical pinhole convention across all contracts:
$$\mathbf{X}_c = \mathbf{R}_{cw} \mathbf{X}_w + \mathbf{t}_{cw}$$
$$\mathbf{C}_w = -\mathbf{R}_{cw}^T \mathbf{t}_{cw}$$
$$u = f_x \frac{X_c}{Z_c} + c_x, \quad v = f_y \frac{Y_c}{Z_c} + c_y$$

### 4.2. Optical Depth vs. Radial Distance
Depth is strictly defined as optical distance along the principal camera axis:
$$\text{Depth} \equiv Z_c$$
Disparity is explicitly distinguished from world depth and cannot be treated as Cartesian depth without triangulation or baseline calibration.

### 4.3. Reconstruction Units & Gauge Invariance
Dense geometry remains in the monocular gauge established in Phase 3C/3D.1. Depth values and 3D point coordinates are strictly typed with `DepthUnit.RECONSTRUCTION_UNITS`. Any attempt to claim `is_metric_scale=True` without external georeferenced calibration is explicitly rejected by `MVSValidator`.

---

## 5. Verification Results

### 5.1. Phase 3E.0 Synthetic Tests
18 dedicated synthetic unit tests in `tests/unit/test_phase3e_mvs_contracts.py` passed:
- `test_camera_convention_projection_backprojection_roundtrip`: **PASSED**
- `test_known_camera_known_depth_exact_backprojection`: **PASSED**
- `test_invalid_and_non_positive_depth_rejection`: **PASSED**
- `test_depth_map_query_invariants`: **PASSED**
- `test_view_pair_selector_multi_source_and_ordering`: **PASSED**
- `test_view_pair_selector_rejection_rules`: **PASSED**
- `test_dynamic_risk_propagation_into_view_pairs`: **PASSED**
- `test_cross_view_geometric_depth_consistency_acceptance`: **PASSED**
- `test_cross_view_geometric_depth_inconsistency_rejection`: **PASSED**
- `test_cross_view_occlusion_and_bounds_rejection`: **PASSED**
- `test_depth_to_world_points_backprojection_filtering`: **PASSED**
- `test_dense_point_fusion_spatial_deduplication_and_support_count`: **PASSED**
- `test_dense_point_fusion_filters_insufficient_multi_view_support`: **PASSED**
- `test_reconstruction_unit_preservation_no_meters_claim`: **PASSED**
- `test_validator_rejects_premature_metric_scale_claim`: **PASSED**
- `test_validator_input_and_depth_map_checks`: **PASSED**
- `test_abstract_depth_estimator_interface_compliance`: **PASSED**
- `test_failure_and_visibility_taxonomies_completeness`: **PASSED**

### 5.2. Full Project Regression
```text
============================= 407 passed in 9.23s =============================
```
- Total test count increased from **389 to 407**.
- 0 failures, 0 regressions across all phases (Phase 0, 1, 2, 3A, 3B, 3C, 3D, 3E.0).

### 5.3. Static Type Analysis (Pyright)
```bash
npx -y pyright src/geometry/ tests/unit/test_phase3*.py
```
**Result**: `0 errors, 0 warnings, 0 informations`.

---

## 6. Implementation Deliberately NOT Started

In accordance with Phase 3E.0 boundaries, the following were intentionally omitted and deferred to Phase 3E.1 or later:
1. Dense PatchMatch depth propagation / cost volume solvers.
2. GPU/CUDA acceleration kernels.
3. Learned depth estimation models (DUSt3R, VGGT, MASt3R).
4. Dense mesh surface generation (Poisson reconstruction, Marching Cubes).
5. Texture atlas generation and UV unwrapping.
6. Metric georeferencing and absolute scale restoration (Phase 4).
7. LAS/LAZ point cloud or GIS vector exports.

---

## 7. Unresolved Risks & Architectural Guards

1. **Memory Pressure on High-Res UAV Streams**:
   - *Guard*: `MVSInput` stores path references (`image_paths`), preventing simultaneous RAM loading.
2. **Dynamic Moving Objects**:
   - *Guard*: Frame dynamic risk scores from Phase 2 down-weight stereo view-pair suitability and penalize inconsistent transient depths.
3. **Textureless Surfaces (Water, Uniform Roads)**:
   - *Guard*: `DepthConfidenceMap` and `min_consistent_views` prevent spurious low-confidence depth hypotheses from entering the dense point cloud.

---

## 8. Git Working Tree Status

As instructed by Section 25:
- **No git commit has been created.**
- **No git push has been executed.**
- Phase 3E.0 is presented for user audit and review before committing.

---

## 9. Final Phase Status

```
================================================================================
                    PHASE 3E.0 — READY FOR AUDIT
================================================================================
   Tests: 407/407 PASSED | Pyright: 0 ERRORS | Baseline Regressions: 0
================================================================================
```
