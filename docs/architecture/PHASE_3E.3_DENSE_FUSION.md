# Phase 3E.3 Multi-View Dense Point Fusion Architecture Specification

## 1. Overview & Purpose
Phase 3E.3 defines the multi-view geometric fusion layer that aggregates independent, calibrated 3D dense point observations (`DensePointObservation` from Phase 3E.2) into a consolidated, multi-view validated point cloud (`DensePointCloud`).

```
Phase 3E.1 (Dense Stereo)
      ↓
[DepthMap, DepthConfidenceMap]
      ↓
Phase 3E.2 (Dense Point Generation)
      ↓
List[DensePointObservation]
      ↓
Phase 3E.3 (Multi-View Dense Fusion)
      ↓
Consolidated DensePointCloud (RECONSTRUCTION_UNITS)
```

---

## 2. Mathematical Model

### 2.1 Coordinate Space & Gauge Constraint
All input observations and output fused points are defined strictly in the relative coordinate gauge established by SfM (Phase 3D.1):
- `depth_unit = DepthUnit.RECONSTRUCTION_UNITS`
- `is_metric_scale = False`
Absolute metric scale is not claimed without certified external ground truth.

### 2.2 Compatibility Predicate & Anti-Chaining
Two observations $\mathbf{X}_i$ and $\mathbf{X}_j$ are considered geometrically compatible for inclusion into cluster $\mathcal{C}$ if and only if:
1. Distance to cluster centroid:
   $$\|\mathbf{X}_i - \bar{\mathbf{X}}_{\mathcal{C}}\|_2 \le \tau_{spatial}$$
2. Bounding diameter constraint (Anti-Chaining):
   $$\max_{\mathbf{X}_k \in \mathcal{C}} \|\mathbf{X}_i - \mathbf{X}_k\|_2 \le \tau_{diameter}$$
This prevents unconstrained transitive chaining ($A \sim B \sim C \dots$) from creating elongated false surfaces.

### 2.3 Fused Point Estimation
For an accepted cluster $\mathcal{C} = \{(\mathbf{X}_i, c_i)\}_{i=1}^M$:
$$\mathbf{X}_{fused} = \frac{\sum_{i=1}^M w_i \mathbf{X}_i}{\sum_{i=1}^M w_i}$$
where $w_i = c_i$ (heuristic confidence weights).

Fused confidence is aggregated as:
$$c_{fused} = \frac{\sum_{i=1}^M w_i c_i}{\sum_{i=1}^M w_i}, \quad c_{fused} \in [0, 1] \text{ (HEURISTIC\_SCORE)}$$

---

## 3. Spatial Indexing & Computational Complexity
- **Role of Spatial Hashing**: Spatial hashing is strictly an acceleration/indexing structure, not a geometric validity criterion.
- **Complexity Qualification**:
  - **Expected Time Complexity**: $O(N)$ under bounded local point density and bounded cluster size because each observation examines a constant 27-voxel neighborhood.
  - **Worst-Case Time Complexity**: $O(N^2)$ when an arbitrarily large number of observations occupy the same spatial neighborhood, because candidate/member distance checks can then grow quadratically.

---

## 4. Distinct-View Support Semantics
- A cluster is validated only if:
  $$\text{cardinality}\left(\bigcup_{i=1}^M \{\text{ref\_frame\_id}_i\}\right) \ge N_{min\_views}$$
- Multiple observations originating from the *same* camera frame merge into a single spatial observation but do **not** inflate the distinct-view count.
- Observations with insufficient distinct views are either rejected (`REJECT_SINGLE_VIEW`) or tagged `PointValidationStatus.OBSERVED` (`RETAIN_AS_OBSERVED`).

---

## 5. Determinism & Input-Order Invariance
- **Determinism Mechanism**: Input-order invariance is achieved by canonical sorting of observations prior to the deterministic greedy clustering pass.
- Sort key: $\text{key}(o) = (\text{frame\_id}, u, v, X, Y, Z, c)$.
- Greedy sequential clustering is inherently order-dependent; canonical pre-sorting establishes a unique processing sequence to guarantee bit-exact identical outputs under arbitrary input observation order permutations.

---

## 6. Rejection Taxonomy
Explicit reasons tracked in `FusionRejectionReason`:
- `NON_FINITE_COORDINATES`
- `NON_FINITE_CONFIDENCE`
- `OUT_OF_BOUNDS_CONFIDENCE`
- `LOW_CONFIDENCE`
- `INSUFFICIENT_DISTINCT_VIEWS`
- `ISOLATED_OBSERVATION`
- `SPATIAL_CLUSTER_DIAMETER_EXCEEDED`
- `DYNAMIC_RISK_EXCEEDED`
- `INVALID_VISIBILITY_STATE`
- `INVALID_VALIDATION_STATUS`
- `CLUSTER_MERGE_REJECTED`

---

## 7. Real-Data Boundary & Limitations
- **Proven**: Spatial clustering, deterministic input-order invariance via canonicalization, distinct-view consensus, scale equivariance in `RECONSTRUCTION_UNITS`, and anti-chaining guards on synthetic and calibrated benchmarks.
- **Not Proven**: Real-world atmospheric scattering, rolling shutter deformations, reflective surface reconstruction, and absolute physical metric accuracy.
