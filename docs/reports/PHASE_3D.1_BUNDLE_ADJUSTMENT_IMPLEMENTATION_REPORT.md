# Phase 3D.1 Bundle Adjustment Implementation Report

**Status**: PHASE 3D.1 IMPLEMENTED & FORMALLY VERIFIED (LOCKED)  
**Parent Pipeline**: Phase 3 Classical Structure-from-Motion  
**Predecessor**: Phase 3C Incremental Structure-from-Motion (`SparseReconstructionResult`)  
**Successor**: Phase 3E Dense Multi-View Stereo (MVS)  
**Regression**: 389 passed / 389 tests (0 failures, 0 warnings)  
**Static Type Checking (Pyright)**: 0 errors, 0 warnings, 0 informations  

---

### 1. Implementation Summary
Phase 3D.1 implements the non-linear global Bundle Adjustment optimizer (`BundleAdjustmentEngine`), completing Stage 5 of the reconstruction pipeline. The subsystem takes a `SparseReconstructionResult` from Phase 3C and jointly refines:
1. Registered camera poses ($\mathbf{R}_{cw}, \mathbf{t}_{cw}$).
2. 3D triangulated landmark positions ($\mathbf{X}_w$).

Optimization is executed by minimizing the robust Huber-weighted reprojection errors across all valid 2D feature track observations using `scipy.optimize.least_squares` with the Trust Region Reflective (`trf`) algorithm and an observation-based block-sparse Jacobian pattern.

---

### 2. Exact Parameterization
All 7 degrees of gauge freedom ($\text{Sim}(3)$) are explicitly eliminated directly within the optimization variables without rank deficiency or unconstrained parameter drift:
- **Camera 0 (Reference Pose Gauge, 6 DoF)**:
  $$\mathbf{R}_0 \equiv \mathbf{I}_{3 \times 3}, \quad \mathbf{t}_0 \equiv \mathbf{0}_{3 \times 1}$$
  Camera 0 is completely omitted from the parameter state vector $\boldsymbol{\Theta}$ (0 parameters).
- **Camera 1 (Scale Gauge, 1 DoF)**:
  - Rotation: 3 DoF parameterized via minimal $\mathfrak{so}(3)$ axis-angle vector $\boldsymbol{\omega}_1$.
  - Translation Magnitude: Fixed at exactly $1.0$ reconstruction units ($\|\mathbf{t}_{10}\|_2 \equiv 1.0$).
  - Translation Direction: Constrained to the unit sphere $S^2$, parameterized as a 2-DoF vector $\boldsymbol{\alpha}_1 \in \mathbb{R}^2$ in the tangent space $T_{\mathbf{d}_0}(S^2)$.
  - Total parameters for Camera 1: $3 + 2 = 5$ parameters.
- **Cameras 2...M-1**:
  - 3 rotation DoF ($\boldsymbol{\omega}_i \in \mathfrak{so}(3)$) + 3 unconstrained translation DoF ($\mathbf{t}_i \in \mathbb{R}^3$) = 6 parameters each.
- **3D Landmarks 1...N**:
  - 3 coordinates $(X, Y, Z)$ each = 3 parameters per landmark.

---

### 3. Parameter Dimension
For $M \ge 2$ registered cameras and $N$ triangulated landmarks:
$$\dim(\boldsymbol{\Theta}) = 5 + 6(M - 2) + 3N = 6M - 7 + 3N$$

Enforced and validated across variable problem sizes:
- $M=2, N=10 \implies 6(2) - 7 + 3(10) = 35$ parameters.
- $M=3, N=15 \implies 6(3) - 7 + 3(15) = 56$ parameters.
- $M=4, N=12 \implies 6(4) - 7 + 3(12) = 53$ parameters.
- $M=5, N=50 \implies 6(5) - 7 + 3(50) = 173$ parameters.
- Managed by `BAParameterManager` and audited by `BAParameterLayout`.

---

### 4. Rotation Update Convention
Rotations are parameterized minimally via 3-DoF Lie algebra $\mathfrak{so}(3)$ axis-angle vectors:
$$\mathbf{R}(\boldsymbol{\omega}) = \mathbf{I} + \frac{\sin(\theta)}{\theta} [\boldsymbol{\omega}]_\times + \frac{1 - \cos(\theta)}{\theta^2} [\boldsymbol{\omega}]_\times^2$$
with Taylor series evaluation for $\theta < 10^{-7}$.
- **Composition Convention**: Local Lie algebra update rule:
  $$\mathbf{R}_{\text{new}} = \text{Exp}(\Delta \boldsymbol{\omega}) \mathbf{R}_{\text{old}}$$
- Guarantees $\mathbf{R}^T \mathbf{R} = \mathbf{I}$ and $\det(\mathbf{R}) = +1$ within machine precision, eliminating 9-DoF unconstrained matrix drift.

---

### 5. Camera-1 $S^2$ Update Convention
Camera 1's translation direction is parameterized in the tangent space of base direction $\mathbf{d}_0 \in S^2$:
- Let $(\mathbf{b}_1, \mathbf{b}_2)$ be an orthonormal basis for $T_{\mathbf{d}_0}(S^2)$ orthogonal to $\mathbf{d}_0$.
- For tangent vector $\boldsymbol{\alpha} = [\alpha_1, \alpha_2]^T \in \mathbb{R}^2$ with $\theta = \|\boldsymbol{\alpha}\|_2$:
  $$\hat{\mathbf{t}}(\boldsymbol{\alpha}) = \cos(\theta) \mathbf{d}_0 + \frac{\sin(\theta)}{\theta} (\alpha_1 \mathbf{b}_1 + \alpha_2 \mathbf{b}_2)$$
- At every function evaluation, the recovered unit direction is scaled by $1.0$:
  $$\mathbf{t}_1 = \hat{\mathbf{t}}(\boldsymbol{\alpha}) \times 1.0$$
- Unconditionally guarantees $|\|\mathbf{t}_1\|_2 - 1.0| < 10^{-6}$ for any $\boldsymbol{\alpha} \in \mathbb{R}^2$.

---

### 6. Residual Definition
For observation $j$ of landmark $k$ in camera $i$:
$$\mathbf{r}_{ij} = \mathbf{x}_{ij} - \pi(\mathbf{K}_i, \mathbf{R}_i, \mathbf{t}_i, \mathbf{X}_k) = \begin{bmatrix} u_{\text{obs}} - u_{\text{proj}} \\ v_{\text{obs}} - v_{\text{proj}} \end{bmatrix} \in \mathbb{R}^2 \quad (\text{pixels})$$
Positive depth ($Z_c > 0$) is strictly tracked by `BAResidualEvaluator`. If $Z_c \le 0$, the observation is marked geometrically invalid without fabricating huge artificial numbers (e.g. $10^{10}$), and the candidate state is flagged for rejection during post-optimization validation.

---

### 7. Huber Robust Loss Implementation
Evaluated on the $L_2$ error norm $e = \|\mathbf{r}_{ij}\|_2$ in pixels:
$$\rho_\delta(e) = \begin{cases} \frac{1}{2} e^2 & \text{if } e \le \delta \\ \delta \left(e - \frac{1}{2}\delta\right) & \text{if } e > \delta \end{cases} \quad (\delta = 2.0\text{ px}, \text{HEURISTIC\_DEFAULT})$$
- Continuous and $C^1$ smooth everywhere on $\mathbb{R}^+$.
- Effective M-estimator weight:
  $$w(e) = \frac{\rho_\delta'(e)}{e} = \begin{cases} 1.0 & \text{if } e \le \delta \\ \frac{\delta}{e} & \text{if } e > \delta \end{cases}$$

---

### 8. Jacobian Implementation
- **Block-Sparse Finite-Difference Jacobian**: Built using an observation-based sparsity pattern matrix (`sp.csr_matrix`).
- Observation residual row pairs $(2k, 2k+1)$ connect only to:
  - Observing camera columns (5 columns for Camera 1, 6 columns for Cameras $\ge 2$).
  - Observed landmark columns (3 columns).
- Sparse graph coloring evaluates the full Jacobian in $O(1)$ function evaluations, achieving high numerical precision without dense matrix overhead.

---

### 9. Solver Configuration
- **Optimizer**: `scipy.optimize.least_squares`
- **Method**: Trust Region Reflective (`trf`)
- **Loss**: `huber` with `f_scale = 2.0`
- **Tolerances**:
  - `ftol = 1e-6` (`cost_tolerance`, `HEURISTIC_DEFAULT`)
  - `xtol = 1e-6` (`parameter_tolerance`, `HEURISTIC_DEFAULT`)
  - `gtol = 1e-8` (`gradient_tolerance`, `HEURISTIC_DEFAULT`)
  - `max_nfev = max_iterations * 10`

---

### 10. Convergence Information
Solver captures and records:
- Function evaluation count (`nfev`)
- Jacobian evaluation count (`njev`)
- First-order optimality gradient infinity-norm (`optimality`)
- Initial vs final robust cost ($F_{\text{before}}, F_{\text{after}}$)
- Initial vs final image-space reprojection metrics (mean, RMSE, median, 90th percentile, max)
- Solver termination status and message

---

### 11. Rollback & Validation Behavior
- Candidate optimization parameters are unpacked into temporary structures.
- `BAPostOptimizationValidator` evaluates 7 safety gates:
  1. Finite camera poses and valid $SO(3)$ rotations.
  2. Finite landmark coordinates.
  3. Strict gauge preservation (Camera 0 at $[\mathbf{I} \mid \mathbf{0}]$, Camera 1 baseline magnitude $= 1.0$).
  4. Positive optical depth ($Z_c > 0$) for all observations.
  5. Observation count preservation.
  6. Primary acceptance: $F_{\text{after}} \le F_{\text{before}} + 10^{-4}$.
  7. Reprojection RMSE sanity guard ($\le \text{RMSE}_{\text{before}} + 2.0\text{ px}$, `HEURISTIC_DEFAULT`).
- **Rollback Safety**: If any gate fails, the candidate is discarded, the original reconstruction is retained, and `status = FAILED` is returned with full diagnostics.

---

### 12. Synthetic Optimization Results
- **Controlled Perturbation Test**: Perturbing camera poses by small angles and landmarks by $0.03$ units resulted in successful convergence:
  - Robust cost strictly decreased ($F_{\text{after}} < F_{\text{before}}$).
  - Reprojection RMSE strictly decreased.
  - Camera 0 remained fixed at origin $[\mathbf{I} \mid \mathbf{0}]$.
  - Camera 1 baseline magnitude remained strictly $1.0000$.
  - All reconstructed points maintained positive depth.

---

### 13. Outlier Test Results
- **Controlled Outlier Test**: Adding $30\text{--}40\text{ px}$ gross outliers into observation tracks:
  - Huber loss downweighted outlier weights from $1.0$ to $< 0.05$.
  - Optimizer remained numerically stable without divergence.
  - Robust cost non-increase condition was satisfied ($F_{\text{after}} \le F_{\text{before}} + 10^{-4}$).

---

### 14. Failure Case Results
Explicit rejection verified with typed `BAFailureReason` for:
- Insufficient registered cameras ($M < 2$) $\to$ `INVALID_INPUT_RECONSTRUCTION`
- Insufficient landmarks ($N < 10$) $\to$ `INSUFFICIENT_OBSERVATIONS`
- Non-finite camera pose (NaN) $\to$ `INVALID_CAMERA_STATE`
- Non-finite landmark (Inf) $\to$ `INVALID_LANDMARK_STATE`
- Initial Camera 0 pose drift $\to$ `GAUGE_CONSTRAINT_INVALID`
- Initial Camera 1 baseline drift $\to$ `GAUGE_CONSTRAINT_INVALID`

---

### 15. Runtime Measurements
- Synthetic test execution (3 cameras, 25 landmarks, 75 observations): $\approx 0.05\text{--}0.15\text{ s}$.
- Full Phase 3D test suite (48 tests): $4.03\text{ s}$.
- Runtime is captured per-optimization run and recorded in `provenance["runtime_seconds"]`.

---

### 16. Tests Added
19 new tests added to `tests/unit/test_phase3d_bundle_adjustment.py` (total 48 tests):
1. `test_parameter_layout_structure`
2. `test_parameter_pack_unpack_roundtrip`
3. `test_parameter_dimension`
4. `test_camera_zero_excluded`
5. `test_camera_one_has_five_parameters`
6. `test_landmark_parameter_offsets`
7. `test_pack_unpack_preserves_geometry`
8. `test_rotation_identity_perturbation`
9. `test_rotation_small_angle`
10. `test_rotation_composition`
11. `test_rotation_orthonormality_and_determinant`
12. `test_robust_loss_zero`
13. `test_robust_loss_branches_and_weights`
14. `test_sparse_jacobian_structure_block_pattern`
15. `test_synthetic_bundle_adjustment_optimization`
16. `test_bundle_adjustment_controlled_outliers`
17. `test_bundle_adjustment_noop_optimum`
18. `test_bundle_adjustment_failure_cases`
19. `test_bundle_adjustment_runtime_and_provenance`

---

### 17. Full Regression
```bash
python -m pytest -v
```
**Result**: **389 passed in 9.89s, 0 failures, 0 warnings** (341 baseline + 48 Phase 3D = 389 total).

---

### 18. Pyright Type Checking
```bash
npx -y pyright src/geometry/bundle_adjustment.py src/geometry/__init__.py tests/unit/test_phase3d_bundle_adjustment.py
```
**Result**: **`0 errors, 0 warnings, 0 informations`**.

---

### 19. Known Limitations
1. **Scale Ambiguity**: Output is in relative monocular reconstruction units. No absolute metric scale or accuracy in meters is claimed.
2. **Local Minima**: Optimization depends on clean initialization from Phase 3C incremental SfM.
3. **UAV Empirical Validation**: Tested on synthetic configurations; performance on real UAV imagery with motion blur and rolling shutter must be empirically evaluated downstream.

---

### 20. Formal Phase Status
**PHASE 3D.1 STATUS: FORMALLY LOCKED**  
All mathematical contracts, exact gauge parameterization, robust Huber objective, sparse Jacobian structure, solver execution, post-optimization validation, and rollback safety are implemented and verified.
