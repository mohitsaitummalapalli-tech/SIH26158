# Phase 3D Architecture Specification: Global Bundle Adjustment (Implementation & Contracts)

**Status**: PHASE 3D.1 IMPLEMENTED (Formally Verified)  
**Parent Pipeline**: Phase 3 Classical Structure-from-Motion  
**Predecessor**: Phase 3C Incremental Structure-from-Motion (`SparseReconstructionResult`)  
**Successor**: Phase 3E Dense Multi-View Stereo (MVS)  
**Evaluation Level**: `LEVEL_1_IMAGE_SPACE` (Internal geometric reprojection consistency)

---

## 1. Scientific Purpose & Core Principles

Global Bundle Adjustment (BA) is the joint non-linear optimization of camera poses and 3D landmark positions using all geometrically verified 2D feature observations.

$$\text{SparseReconstructionResult} \xrightarrow{\text{Bundle Adjustment}} \text{Refined Sparse Reconstruction}$$

### Non-Negotiable Scientific Principles
1. **Monocular Relative Reconstruction**:
   Bundle Adjustment operates strictly in relative reconstruction coordinates. It is inherently **`SCALE_AMBIGUOUS`** and does **NOT** establish absolute physical scale, absolute metric accuracy, or georeferencing without certified external metric ground truth.
2. **Reprojection Residuals in Pixels**:
   The objective minimizes image-space reprojection residuals $\mathbf{r}_{ij} \in \mathbb{R}^2$ measured in **pixels**. Lower reprojection RMSE signifies internal ray-intersection consistency, **not** physical ground-truth accuracy.
3. **No GNSS / Metric Fusion**:
   GNSS telemetry positions are strictly `TRAJECTORY_PROXY` data and must **never** be injected into the visual bundle adjustment cost function or substituted into camera extrinsics.
4. **Gauge Invariance & Parameter Consistency**:
   Monocular reconstruction has 7 degrees of gauge freedom ($\text{Sim}(3)$: 3 rotation, 3 translation, 1 scale). The optimization state vector explicitly eliminates all 7 gauge degrees of freedom rather than using an unconstrained parameterization.

---

## 2. Mathematical Optimization Problem

### Objective Function
Given $M$ registered cameras and $N$ triangulated 3D landmarks, the objective function minimizes the robust sum of Huber-weighted reprojection errors:

$$\min_{\boldsymbol{\Theta}} F(\boldsymbol{\Theta}) = \sum_{i=1}^M \sum_{j \in \mathcal{V}_i} \rho_\delta \left( \left\| \mathbf{x}_{ij} - \pi(\mathbf{K}_i, \mathbf{R}_i, \mathbf{t}_i, \mathbf{X}_j) \right\|_2 \right)$$

where:
- $\mathcal{V}_i$ is the set of landmark tracks observed in camera $i$.
- $\mathbf{x}_{ij} \in \mathbb{R}^2$ is the detected feature coordinate $[u, v]^T$ in pixels (`OBSERVED`).
- $\pi(\cdot)$ is the calibrated pinhole projection operator (`ESTIMATED`).
- $\rho_\delta(\cdot)$ is the Huber robust loss function (`HEURISTIC_DEFAULT`).
- $\boldsymbol{\Theta}$ is the gauge-constrained parameter state vector.

---

## 3. Exact Gauge-Constrained Optimization Dimension

To eliminate the 7-DoF similarity ambiguity ($\text{Sim}(3)$) directly within the optimization variables without rank deficiency or redundant drift:
1. **Camera 0 (Reference Pose Gauge, 6 DoF)**:
   $$\mathbf{R}_0 \equiv \mathbf{I}_{3 \times 3}, \quad \mathbf{t}_0 \equiv \mathbf{0}_{3 \times 1}$$
   Camera 0 parameters are held constant and **omitted** from $\boldsymbol{\Theta}$ (0 DoF).
2. **Camera 1 (Scale Gauge, 1 DoF)**:
   - Rotation: 3 DoF via Lie algebra $\mathfrak{so}(3)$ axis-angle vector $\boldsymbol{\omega}_1$.
   - Translation Magnitude: Fixed at $1.0$ reconstruction units ($\|\mathbf{t}_{10}\|_2 \equiv 1.0$).
   - Translation Direction: Constrained to the unit sphere $S^2$, parameterized as a 2-DoF vector $\boldsymbol{\alpha}_1 \in \mathbb{R}^2$ in the local tangent plane $T_{\mathbf{d}_0}(S^2)$.
   - Camera 1 contributes exactly $3 + 2 = 5$ parameters.
3. **Cameras 2...M-1**:
   - 3 rotation DoF + 3 translation DoF = 6 parameters each ($6(M - 2)$ parameters).
4. **3D Landmarks 1...N**:
   - 3 coordinates $(X, Y, Z)$ each ($3N$ parameters).

### Exact State Dimension Formula
For $M \ge 2$ cameras and $N$ landmarks:
$$\dim(\boldsymbol{\Theta}) = 5 + 6(M - 2) + 3N = 6M - 7 + 3N$$

| Cameras ($M$) | Landmarks ($N$) | Exact Parameter Dimension ($\dim(\boldsymbol{\Theta})$) |
|---|---|---|
| 2 | 10 | $6(2) - 7 + 3(10) = 5 + 30 = 35$ |
| 3 | 15 | $6(3) - 7 + 3(15) = 11 + 45 = 56$ |
| 5 | 50 | $6(5) - 7 + 3(50) = 23 + 150 = 173$ |
| 10 | 100 | $6(10) - 7 + 3(100) = 53 + 300 = 353$ |

---

## 4. Camera 1 Unit Baseline Direction Parameterization on $S^2$

Let $\mathbf{d}_0 \in S^2$ be the base reference unit direction.
An orthonormal basis $(\mathbf{b}_1, \mathbf{b}_2)$ for the tangent space $T_{\mathbf{d}_0}(S^2)$ is constructed such that:
$$\mathbf{b}_1 \cdot \mathbf{d}_0 = 0, \quad \mathbf{b}_2 \cdot \mathbf{d}_0 = 0, \quad \mathbf{b}_1 \cdot \mathbf{b}_2 = 0, \quad \|\mathbf{b}_1\|_2 = 1, \quad \|\mathbf{b}_2\|_2 = 1$$

### Tangent to Direction (Exponential / Retraction Map)
For tangent perturbation $\boldsymbol{\alpha} = [\alpha_1, \alpha_2]^T \in \mathbb{R}^2$:
$$\theta = \|\boldsymbol{\alpha}\|_2$$
$$\hat{\mathbf{t}}(\boldsymbol{\alpha}) = \cos(\theta) \mathbf{d}_0 + \frac{\sin(\theta)}{\theta} (\alpha_1 \mathbf{b}_1 + \alpha_2 \mathbf{b}_2)$$
For $\theta < 10^{-7}$, the Taylor series expansion is evaluated:
$$\hat{\mathbf{t}}(\boldsymbol{\alpha}) \approx \frac{\mathbf{d}_0 + \alpha_1 \mathbf{b}_1 + \alpha_2 \mathbf{b}_2}{\|\mathbf{d}_0 + \alpha_1 \mathbf{b}_1 + \alpha_2 \mathbf{b}_2\|_2}$$
This guarantees $\|\hat{\mathbf{t}}(\boldsymbol{\alpha})\|_2 \equiv 1.0$ unconditionally for any $\boldsymbol{\alpha} \in \mathbb{R}^2$.

---

## 5. Robust Huber Loss Notation & $C^1$ Continuity

Let $e = \|\mathbf{r}_{ij}\|_2$ be the 2D reprojection error norm in pixels.
The Huber robust loss function $\rho_\delta(e)$ is defined as:

$$\rho_\delta(e) = \begin{cases} \frac{1}{2} e^2 & \text{if } e \le \delta \\ \delta \left(e - \frac{1}{2}\delta\right) & \text{if } e > \delta \end{cases}$$

- Threshold parameter: $\delta = 2.0\text{ px}$ (`HEURISTIC_DEFAULT`).
- Value at transition $e = \delta$: $\rho_\delta(\delta) = \frac{1}{2}\delta^2$.
- First derivative:
  $$\rho_\delta'(e) = \begin{cases} e & \text{if } e \le \delta \\ \delta & \text{if } e > \delta \end{cases}$$
- The loss function is continuously differentiable ($C^1$ smooth) everywhere on $\mathbb{R}^+$, including across the transition boundary $e = \delta$.
- Effective M-estimator weight:
  $$w(e) = \frac{\rho_\delta'(e)}{e} = \begin{cases} 1.0 & \text{if } e \le \delta \\ \frac{\delta}{e} & \text{if } e > \delta \end{cases}$$

---

## 6. Solver Architecture (`BundleAdjustmentEngine`)

Phase 3D.1 implements the non-linear least-squares solver using `scipy.optimize.least_squares` with the Trust Region Reflective (`trf`) algorithm.

### Solver Components
1. **Parameter Manager & Layout (`BAParameterManager`, `BAParameterLayout`)**:
   - Reversible parameter vector packing and unpacking.
   - Exact dimension invariant $6M - 7 + 3N$ for $M \ge 2$.
   - Explicit parameter offsets for cameras and 3D landmarks.
2. **Residual Engine (`BAResidualEvaluator`)**:
   - Evaluates 2D residuals: $\mathbf{r}_{ij} = [u_{\text{obs}} - u_{\text{proj}}, v_{\text{obs}} - v_{\text{proj}}]^T$.
   - Computes robust Huber objective and per-observation error norms.
   - Explicitly records optical depth validity ($Z_c > 0$) without fabricating large artificial residuals.
3. **Jacobian Strategy**:
   - Evaluates finite-difference Jacobians using an explicit observation-based **block-sparse Jacobian mask** (`sp.csr_matrix`).
   - Observations connect one camera and one landmark; residuals have nonzeros only in corresponding parameter columns (5 for Camera 1, 6 for Cameras $\ge 2$, 3 for Landmark).
   - Scipy's graph-coloring finite-difference algorithm dramatically reduces function evaluations to $O(1)$ relative to problem size.

---

## 7. Post-Optimization Acceptance Semantics & Rollback

Post-optimization acceptance is evaluated by [`BAPostOptimizationValidator`](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/src/geometry/bundle_adjustment.py#L765) based on **robust cost reduction and geometric validity**, rather than raw RMSE:

1. **Camera State Validity**: All camera poses must be finite and $\mathbf{R}_i \in SO(3)$.
2. **Landmark State Validity**: All landmarks $\mathbf{X}_j$ must be finite.
3. **Gauge Preservation**:
   - Reference camera $\mathbf{R}_0 = \mathbf{I}, \mathbf{t}_0 = \mathbf{0}$.
   - Camera 1 baseline magnitude is exactly $1.0$ ($|\|\mathbf{t}_1\| - 1.0| \le 10^{-4}$).
4. **Positive Optical Depth (Cheirality)**: $Z_c > 0$ for all observing cameras.
5. **Observation Count Preservation**: No valid observations silently dropped.
6. **Primary Optimization Acceptance**:
   $$F_{\text{after}} \le F_{\text{before}} + \epsilon_{\text{cost}} \quad (\epsilon_{\text{cost}} = 10^{-4})$$
   If the robust cost increased, the optimization is rejected as `OPTIMIZATION_DIVERGED`.
7. **Heuristic RMSE Sanity Ceiling**:
   $$\text{RMSE}_{\text{after}} \le \text{RMSE}_{\text{before}} + \tau_{\text{rmse}} \quad (\tau_{\text{rmse}} = 2.0\text{ px}, \text{HEURISTIC\_DEFAULT})$$
   This check guards against severe divergence. It is strictly an engineering heuristic and does not prove theoretical convergence.
8. **Rollback Safety**:
   If validation fails, the engine rejects the candidate state, preserves the original reconstruction, and returns `status = FAILED` with typed failure reason and diagnostic trace.

---

## 8. Convergence Semantics

Configurable stopping criteria in `BundleAdjustmentConfig`:
- Relative cost reduction: $\Delta F / F < 10^{-6}$ (`cost_tolerance`, `HEURISTIC_DEFAULT`).
- Parameter step norm: $\|\Delta \boldsymbol{\Theta}\|_2 < 10^{-6}$ (`parameter_tolerance`, `HEURISTIC_DEFAULT`).
- Gradient infinity-norm: $\|\mathbf{g}\|_\infty < 10^{-8}$ (`gradient_tolerance`, `HEURISTIC_DEFAULT`).
- Iteration limit: `max_iterations = 50` (`HEURISTIC_DEFAULT`).

> [!NOTE]
> Satisfaction of numerical convergence thresholds indicates a local stationary point or step stall. It is **NOT** a mathematical proof of finding the global optimum.

---

## 9. Failure Taxonomy (`BAFailureReason`)
- `INVALID_INPUT_RECONSTRUCTION`
- `INSUFFICIENT_OBSERVATIONS`
- `INVALID_CAMERA_STATE`
- `INVALID_LANDMARK_STATE`
- `GAUGE_CONSTRAINT_INVALID`
- `PROJECTION_FAILURE`
- `OPTIMIZATION_FAILED`
- `OPTIMIZATION_DIVERGED`
- `MAX_ITERATIONS_REACHED`
- `NUMERICAL_SINGULARITY`
- `POST_OPTIMIZATION_VALIDATION_FAILED`

---

## 10. Scientific Disclaimers & Known Limitations
1. **Scale Ambiguity**: The output is strictly monocular relative reconstruction. No metric accuracy or distance in meters is claimed.
2. **Local Extrema**: Non-linear least squares does not guarantee finding the global optimum; convergence depends on the quality of Phase 3C incremental initialization.
3. **Real-UAV Performance**: Evaluated on synthetic geometric configurations. Robustness on real UAV imagery with high motion blur or rolling shutter requires downstream empirical evaluation.
