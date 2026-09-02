# Code Health & IDE Diagnostic Audit Report

## 1. Executive Summary

A comprehensive code health and static analyzer audit was conducted following the lock of **Phase 3B (Two-View Geometry & Robust Geometric Verification)**.

Prior to cleanup, the IDE reported **12 diagnostic problems**, notably a red badge with **4 errors** in `tests/unit/test_phase3b_two_view.py` and related errors in `src/geometry/two_view.py`, `src/geometry/features.py`, and `tests/unit/test_dynamic_scene_analysis.py`.

All genuine code defects, missing typing imports, unsafe Optional unpackings, and array constructor ambiguities have been resolved without altering scientific behavior or weakening validation.

---

## 2. Diagnostics Identified & Root Cause Analysis

### 2.1 `tests/unit/test_phase3b_two_view.py` (4 Red Diagnostics)
- **Line 30**: `R_rel: Optional[np.ndarray] = None` — `error: "Optional" is not defined (reportUndefinedVariable)`
- **Line 31**: `t_rel: Optional[np.ndarray] = None` — `error: "Optional" is not defined (reportUndefinedVariable)`
- **Line 34**: `intrinsics: Optional[CameraIntrinsics] = None` — `error: "Optional" is not defined (reportUndefinedVariable)`
- **Line 39**: `-> Tuple[...]` — `error: "Tuple" is not defined (reportUndefinedVariable)`
- **Lines 308, 309, 468**: `np.linalg.norm(t_est)`, `np.dot(t_est, ...)`, `assert_array_almost_equal` — `error: Argument of type "ndarray | None" cannot be assigned to parameter of type "ArrayLike"`
- **Root Cause**: `Optional` and `Tuple` were missing from the module's imports from `typing`. In Python 3.14, runtime function execution does not evaluate annotations unless introspected, allowing pytest to pass while static analyzers / IDE language servers correctly flagged undefined symbols. In addition, `t_est` and `relative_rotation` are typed as `Optional[np.ndarray]` on `TwoViewGeometryResult`, requiring explicit `assert ... is not None` before array math.
- **Fix Applied**: Added `from typing import Optional, Tuple` and explicit non-None assertions (`assert t_est is not None`, `assert R_est is not None`, `assert res1.relative_rotation is not None`) to narrow the type safely.

### 2.2 `src/geometry/two_view.py`
- **Line 297**: `R_rel, t_rel, median_parallax = best_hypo` — `error: "None" is not iterable (reportGeneralTypeIssues)`
- **Lines 17–38**: Unused imports (`asdict`, `Enum`, `Union`, `GeometryMathContracts`, `SpatialMatchDiagnostics`, `SpatialDistributionCalculator`).
- **Root Cause**: `best_hypo` was initialized to `None` prior to the hypothesis loop. Even though the loop over the 4 SVD decomposition hypotheses is non-empty, static analysis inferred `best_hypo` could remain `None`, causing an unsafe tuple unpacking error.
- **Fix Applied**: Initialized `best_hypo` to `(hypotheses[0][0], hypotheses[0][1], 0.0)` to guarantee a strictly non-None 3-tuple, and pruned all unused imports.

### 2.3 `src/geometry/features.py`
- **Line 304**: `self._orb = cv2.ORB_create(...)` — `error: "ORB_create" is not a known attribute of module "cv2" (reportAttributeAccessIssue)`
- **Root Cause**: `cv2.ORB_create` is a legacy Python binding alias not declared in modern typed stubs. The official OpenCV C++/Python factory method is `cv2.ORB.create(...)`.
- **Fix Applied**: Replaced `cv2.ORB_create` with standard `cv2.ORB.create(...)`.

### 2.4 `tests/unit/test_dynamic_scene_analysis.py`
- **Line 63, 178, 245, 360**: `M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])` — `error: Argument of type "list[list[int]]" cannot be assigned to parameter "value" of type "_ConvertibleToFloat | None" in function "__new__"`
- **Line 64, 179, 246, 361**: `img = cv2.warpAffine(img, M, ...)` — `error: Argument of type "float32" cannot be assigned to parameter "M" of type "MatLike"`
- **Line 97**: `def detect_candidate_regions(self, frame):` — `error: Method "detect_candidate_regions" overrides class "DynamicRegionProvider" in an incompatible manner`
- **Line 336**: `np.array_equal(f0.data, copy0)` — `error: Argument of type "ndarray | None" cannot be assigned to parameter "a1" of type "ArrayLike"`
- **Root Cause**: Calling `np.float32([[...]])` attempts to call the scalar type constructor on a nested list rather than creating a 2D array via `np.array(..., dtype=np.float32)`. Overriding `detect_candidate_regions` lacked type annotations matching the abstract base class `DynamicRegionProvider`.
- **Fix Applied**: Converted all affine matrix creations to `np.array([[...]], dtype=np.float32)`, typed the mock method signature with `DecodedFrame` and return types, and asserted `f0.data is not None`.

---

## 3. Verification & Diagnostic Status After Fixes

### 3.1 Pyright Language Server Verification
Running `npx -y pyright` on all updated files:
```bash
npx -y pyright tests/unit/test_phase3b_two_view.py src/geometry/two_view.py src/geometry/features.py tests/unit/test_dynamic_scene_analysis.py
```
**Result**: `0 errors, 0 warnings, 0 informations`

### 3.2 Compilation Verification
```bash
python -m py_compile src/geometry/contracts.py src/geometry/features.py src/geometry/two_view.py
```
**Result**: `Exit code 0 (Clean compilation)`

### 3.3 Phase 3B Test Suite
```bash
python -m pytest tests/unit/test_phase3b_two_view.py -v
```
**Result**: `26 passed in 1.94s (0 failures)`

### 3.4 Full Workspace Regression Suite
```bash
python -m pytest -v
```
**Result**: `307 passed in 4.98s (0 failures)`

---

## 4. Audit Decision & Scientific Invariance Confirmation

- **Phase 3B Scientific Invariance**: Verified. No mathematical formulas, geometric thresholds, coordinate conventions, or failure taxonomies were altered.
- **Locked Baseline**: All 307 tests remain completely green.
- **Status**: **PASS — CODE HEALTH AUDIT COMPLETE**.
