# Phase 3E.4 Surface Reconstruction Architecture & Mathematical Contract Specification

## 1. Executive Summary & Objective
Phase 3E.4 establishes the scientific contract and algorithmic design for converting the consolidated, multi-view fused dense point cloud from Phase 3E.3 (`DensePointCloud`) into a structured, validated surface representation (`SurfaceMesh`) in relative reconstruction coordinates (`RECONSTRUCTION_UNITS`).

### Core Scientific Principles
- **Geometric Extrapolation Constraint**: The reconstruction algorithm must enforce an explicit geometric scale constraint (such as an $\alpha$-parameter or maximum edge length) to prevent unconstrained bridging across wide unobserved gaps.
- **Support $\neq$ Geometric Certainty**: Limiting the geometric scale of admissible facets bounds interpolation distance, but does **not** prove that the interpolated surface corresponds to a physically observed continuous surface.
- **Open Boundaries vs. Watertightness**: Unobserved spaces, occlusions, and visibility boundaries must remain open boundaries. No artificial closing cap, base plane, or watertight closure is imposed without observational evidence.
- **Gauge Semantics**: Output geometry is strictly in relative reconstruction coordinates (`depth_unit = DepthUnit.RECONSTRUCTION_UNITS`, `is_metric_scale = False`). No metric scale is claimed.

---

## 2. Pipeline Integration & Data Flow

```
Phase 3D.1 (Bundle Adjustment)
      ↓  Optimized camera poses & sparse gauge
Phase 3E.1 (Dense Stereo Matching)
      ↓  Disparity & Optical Depth Maps
Phase 3E.2 (Dense Point Unprojection)
      ↓  DensePointObservation (World XYZ, Pixel ID, Depth, Heuristic Confidence)
Phase 3E.3 (Multi-View Dense Fusion)
      ↓  Consolidated DensePointCloud (Distinct-view filtered, anti-chained)
Phase 3E.4 (Surface Reconstruction Contract & Engine)
      ↓
SurfaceMesh (Vertices, Faces, Normals, Observation Support Scores, Boundary Masks)
```

---

## 3. Input Contract Analysis (from Phase 3E.3)
The primary input to Phase 3E.4 is `DensePointCloud` produced by Phase 3E.3:
- `points`: $(N, 3)$ `np.float64` array of 3D coordinates in `RECONSTRUCTION_UNITS`.
- `confidences`: $(N,)$ `np.float32` array in $[0, 1]$, representing aggregated `HEURISTIC_SCORE`.
- `support_counts`: $(N,)$ `np.int32` array representing distinct-view support count ($\ge N_{min\_views}$).
- `source_frame_ids`: `List[List[str]]` recording contributing reference/source frame IDs per point.
- `visibility_states`: `List[PointVisibilityState]`, strictly `PointVisibilityState.VALID`.
- `validation_statuses`: `List[PointValidationStatus]`, `VALIDATED` or `OBSERVED`.
- `depth_unit`: `DepthUnit.RECONSTRUCTION_UNITS` (scale-ambiguous).
- `is_metric_scale`: `False`.
- `provenance`: Traceability dictionary from Phase 3E.3 fusion.

---

## 4. Evaluation of Surface Reconstruction Candidates

| Candidate Method | Normal Requirement | Extrapolation / Interpolation Behavior | Hole / Boundary Handling | Thin Structures | Dependency Footprint | Determinism | Suitability for Aerial UAV Video |
|---|---|---|---|---|---|---|---|
| **Screened Poisson** | Strict (requires oriented normals) | High volumetric interpolation; tends to close all open boundaries into a watertight manifold | Obscures unobserved regions unless aggressively trimmed by octree depth/density | Smooths out sharp geometric features | Heavy C++ binary wrapper (e.g. Open3D/PoissonRecon) | Sensitive to octree solver ordering | Moderate (risk of generating continuous solids under roofs) |
| **Ball Pivoting (BPA)** | Strict (requires oriented normals) | Conservative geometric interpolation; only places facets where a sphere of radius $\rho$ touches 3 points | Preserves holes when point spacing exceeds $2\rho$ | Preserves thin structures when $\rho < d_{separation}$ | Moderate (requires kd-tree spatial indexing) | Deterministic if seed front is canonically ordered | High (conservative relative to volumetric Poisson) |
| **3D Alpha Complex via Delaunay** | None for triangulation (normals estimated post-hoc) | Explicit geometric scale constraint governed by circumradius $\alpha_{radius}$ and edge cap $\alpha_{edge}$ | Preserves open boundaries and unobserved gaps exceeding scale thresholds | Preserves thin structures when $\alpha < d_{separation}$ | Pure `scipy.spatial.Delaunay` (zero new binary dependencies) | Deterministic repeated execution under declared runtime | **Very High** (exact topological filtering of 3D simplices) |
| **2.5D Visibility-Aware Delaunay** | None | Planar/depth-guided 2.5D triangulation | Effective for nadir terrain sweeps; fails on complex vertical overhangs | Fails on multi-layered vertical geometry | Pure `scipy.spatial` | Deterministic repeated execution | High for purely 2.5D terrain |

### 4.1 Selected Primary Mathematical Method: 3D Alpha-Complex Simplices
The primary surface extraction pipeline uses 3D Delaunay tetrahedralization filtered by dual scale constraints ($\alpha_{radius}$ and $\alpha_{edge}$):

1. **Simplicial Complex Construction**:
   Compute the 3D Delaunay tetrahedralization $\mathcal{T} = \{\sigma_k\}_{k=1}^K$ of the non-coplanar input points $\mathbf{X} \in \mathbb{R}^{N \times 3}$ ($N \ge 4$) using `scipy.spatial.Delaunay`.
   *Capability & Scope*: `scipy.spatial.Delaunay` partitions the convex hull of 3D points in general position into non-overlapping tetrahedra. If the point cloud is degenerate (e.g., all points strictly coplanar in 3D), 3D tetrahedralization raises `QhullError`; such degeneracies are intercepted by planar dimensionality checks before Delaunay execution.

2. **Tetrahedral Circumradius Filtering ($\alpha_{radius}$)**:
   For each tetrahedron $\sigma = (\mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3)$, compute its circumsphere radius $R_\sigma$:
   $$R_\sigma = \frac{\|\mathbf{a} \times \mathbf{b}\| \|\mathbf{c} \times \mathbf{d}\|}{12 V_\sigma} \quad \text{(or via Cayley-Menger determinant / circumcenter solver)}$$
   Retain only tetrahedra satisfying $R_\sigma \le \alpha_{radius}$.

3. **Boundary Facet Extraction**:
   Extract all triangular 2-simplices (faces) that belong to exactly one admissible tetrahedron (i.e. the boundary facets of the retained 3D alpha complex).

4. **Facet Edge Length Filtering ($\alpha_{edge}$)**:
   Discard any extracted triangle face where the maximum edge length exceeds $\alpha_{edge}$:
   $$\max_{e \in \text{edges}(f)} \|e\|_2 \le \alpha_{edge}$$
   *Separation Note*: $\alpha_{radius}$ governs tetrahedral volumetric inclusion, whereas $\alpha_{edge}$ bounds facet boundary lengths.

---

## 5. Normal Estimation Contract

Surface normals $\mathbf{n}_i \in \mathbb{R}^3$ are computed for each vertex $\mathbf{X}_i$ using local neighborhood covariance Principal Component Analysis (PCA):

### 5.1 Local Neighborhood Covariance
For vertex $\mathbf{X}_i$, query its $k$-nearest neighbors $\mathcal{N}(i)$ (with $k \ge 3$):
$$\mathbf{C}_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} (\mathbf{X}_j - \bar{\mathbf{X}})(\mathbf{X}_j - \bar{\mathbf{X}})^T$$
where $\bar{\mathbf{X}} = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{X}_j$.

### 5.2 Eigenvalue Conditioning & Degeneracy Metrics
Compute eigenvalues and unit eigenvectors:
$$\mathbf{C}_i \mathbf{e}_m = \lambda_m \mathbf{e}_m, \quad 0 \le \lambda_0 \le \lambda_1 \le \lambda_2$$

- **Normal Vector Candidate**: The unoriented normal candidate is the eigenvector $\mathbf{e}_0$ corresponding to the minimal eigenvalue $\lambda_0$.
- **Planarity Metric ($\mathcal{P}$)**:
  $$\mathcal{P} = \frac{\lambda_1 - \lambda_0}{\lambda_2} \in [0, 1]$$
- **Linearity / Collinearity Degeneracy ($\mathcal{L}$)**:
  $$\mathcal{L} = \frac{\lambda_2 - \lambda_1}{\lambda_2} \in [0, 1]$$
- **Sphericity / Volumetric Scatter ($\mathcal{S}$)**:
  $$\mathcal{S} = \frac{\lambda_0}{\lambda_2} \in [0, 1]$$

**Validity Rule**:
- A neighborhood is considered **reliably planar** if $\mathcal{P} \ge \tau_{planarity}$ (e.g. 0.2) and $\mathcal{S} \le \tau_{sphericity}$ (e.g. 0.3).
- If $\mathcal{L} \approx 1$ (collinear points, where $\lambda_1 \approx \lambda_0 \approx 0$) or $\mathcal{S} \approx 1$ (isotropic noise, $\lambda_0 \approx \lambda_1 \approx \lambda_2$), the normal is tagged `UNCERTAIN` or `DEGENERATE`.

### 5.3 Viewpoint-Assisted Normal Orientation with Vector Cancellation Guard
When camera optical centers $\{\mathbf{C}_{w, m}\}_{m \in \mathcal{M}(i)}$ for contributing source frames are provided via a camera pose dictionary `camera_centers: Dict[str, np.ndarray]`:
1. Aggregate the unit viewing directions from the vertex to all contributing camera centers:
   $$\mathbf{v}_{view} = \frac{1}{|\mathcal{M}(i)|} \sum_{m \in \mathcal{M}(i)} \frac{\mathbf{C}_{w, m} - \mathbf{X}_i}{\|\mathbf{C}_{w, m} - \mathbf{X}_i\|_2 + \epsilon}$$
2. **Vector Cancellation Guard**:
   If $\|\mathbf{v}_{view}\|_2 < \tau_{cancellation}$ (e.g. $10^{-3}$, which occurs when contributing camera viewpoints are nearly opposing or symmetrical around the point), the aggregate viewing direction is ill-defined. In this case, the estimator safely falls back to the deterministic unoriented normal sign convention (canonical component sign).
3. **Orientation Assignment**:
   If $\|\mathbf{v}_{view}\|_2 \ge \tau_{cancellation}$:
   $$\mathbf{n}_i = \begin{cases} \mathbf{e}_0 & \text{if } \mathbf{e}_0 \cdot \mathbf{v}_{view} \ge 0 \\ -\mathbf{e}_0 & \text{if } \mathbf{e}_0 \cdot \mathbf{v}_{view} < 0 \end{cases}$$
4. **Fallback / Unoriented Mode**: If `camera_centers` is omitted or empty, the estimator produces deterministic unoriented normals using a canonical sign convention (e.g., first non-zero component positive).

---

## 6. Surface Representation Contract (`SurfaceMesh`)

```python
@dataclass
class SurfaceMesh:
    """Consolidated surface mesh representation in relative reconstruction coordinates."""
    vertices: np.ndarray                                # (V, 3) float64 in RECONSTRUCTION_UNITS
    faces: np.ndarray                                   # (F, 3) int32 triangle vertex indices
    vertex_normals: Optional[np.ndarray]                # (V, 3) float64 unit normal vectors or None
    face_normals: Optional[np.ndarray]                  # (F, 3) float64 unit face normal vectors or None
    vertex_confidences: np.ndarray                      # (V,) float32 in [0, 1], HEURISTIC_SCORE
    vertex_support_counts: np.ndarray                   # (V,) int32 distinct frame counts
    face_support_scores: np.ndarray                     # (F,) float32 aggregated heuristic support score [0, 1]
    face_areas: np.ndarray                              # (F,) float64 face area in RECONSTRUCTION_UNITS^2
    is_boundary_vertex: np.ndarray                      # (V,) bool, True if vertex is incident to an open mesh boundary
    is_boundary_face: np.ndarray                        # (F,) bool, True if face has at least one open boundary edge
    total_vertices: int
    total_faces: int
    depth_unit: DepthUnit = DepthUnit.RECONSTRUCTION_UNITS
    is_metric_scale: bool = False
    provenance: Dict[str, Any] = field(default_factory=dict)
```

---

## 7. Support & Coverage Model

### 7.1 Distinct View Support vs. Heuristic Support Score
- **Target Views ($N_{target\_views}$)**: A configurable positive integer (default: 3 or 4) defining the nominal multi-view observation redundancy expected for high-confidence geometry.
- **Face Support Score ($S_f$)**:
  $$S_f = \frac{1}{3} \sum_{k=1}^3 \min\left(1.0, \frac{\text{support\_count}(v_k)}{N_{target\_views}}\right) \in [0, 1]$$
  *Qualification*: $S_f$ is strictly a **`HEURISTIC_SUPPORT_SCORE`**. It does **not** represent a Bayesian posterior probability, physical measurement certainty, or geometric accuracy metric.

### 7.2 Exact Boundary Semantics & Separation from Support
In the extracted 2-manifold simplicial surface mesh $\mathcal{S} = \{f_j\}_{j=1}^F$ with undirected edges $\mathcal{E}$:
1. **Topological Boundary Edge**: For every edge $e = \{u, v\} \in \mathcal{E}$, let $\mathcal{F}(e) = \{f \in \mathcal{S} \mid e \subset f\}$ be the set of incident faces. An edge $e$ is an **open boundary edge** iff $|\mathcal{F}(e)| = 1$.
2. **Boundary Vertex Marking**: A vertex $v$ is marked `is_boundary_vertex[v] = True` iff $v$ is an endpoint of at least one topological boundary edge $e$ with $|\mathcal{F}(e)| = 1$; otherwise `is_boundary_vertex[v] = False`.
3. **Boundary Face Marking**: A face $f = (v_0, v_1, v_2)$ is marked `is_boundary_face[f] = True` iff at least one of its three constituent edges has $|\mathcal{F}(e)| = 1$.
4. **Independence Principle**: A boundary vertex (e.g. on a well-illuminated, multi-view observed roof ridge) can have maximal support ($S_f = 1.0$) while being a topological boundary. Conversely, a closed interior face can have low support ($S_f < 0.5$) if formed by marginally supported points. Topology and support are tracked in completely independent fields.

---

## 8. Geometry Safety & Scale-Aware Tolerances

To preserve scale equivariance across different relative reconstruction unit scalings, geometric validity checks use dimensionless relative ratios:

1. **Dimensionless Triangle Aspect Ratio**:
   $$\mathcal{A}_f = \frac{\max(e_1, e_2, e_3)}{h_{min}} \le \tau_{aspect} \quad (\text{e.g. } \tau_{aspect} = 20.0)$$
   where $h_{min} = \frac{2 \text{Area}}{\max(e_1, e_2, e_3)}$ is the minimum altitude.
2. **Dimensionless Relative Area Degeneracy**:
   $$\frac{\text{Area}_f}{\max(e_1, e_2, e_3)^2} \ge \tau_{rel\_area} \quad (\text{e.g. } \tau_{rel\_area} = 10^{-6})$$
   Faces failing this relative criterion are flagged as `DEGENERATE_TRIANGLE` and rejected.
3. **Finite Coordinates**: Vertices with NaN or Inf coordinates are rejected prior to triangulation.

---

## 9. Hole and Boundary Policy
1. **Open Boundary Preservation**: The boundary of observed geometry remains open; the mesh is **not** forced to be watertight.
2. **Unobserved Spaces**: Occlusions, shadows, and areas outside the camera frustum remain empty holes.
3. **No Topological Hallucination**: No synthetic base plane or artificial closing cap is added.

---

## 10. Provenance Design
- Each vertex maps $1:1$ to a fused point in `DensePointCloud`.
- Each face inherits the union of contributing source frame IDs from its 3 constituent vertices.
- Complete traceability back to Phase 3E.1 stereo pairs is preserved in `SurfaceMesh.provenance`.

---

## 11. Scale & Determinism Rules
- **Scale**: `depth_unit = DepthUnit.RECONSTRUCTION_UNITS`, `is_metric_scale = False`.
- **Numerical Scale Equivariance**: Under geometric scaling $\mathbf{X} \to s \mathbf{X}$, $\alpha_{radius} \to s \alpha_{radius}$, $\alpha_{edge} \to s \alpha_{edge}$ ($s > 0$), the extracted face topology (index connectivity) is invariant, while vertex coordinates and areas scale as $s \mathbf{X}$ and $s^2 \text{Area}$. In standard 64-bit IEEE-754 floating-point arithmetic, this holds within standard numerical roundoff tolerances ($\sim 10^{-14}$) across well-conditioned dynamic ranges ($10^{-6} \le s \le 10^6$).
- **Determinism**: Deterministic repeated execution under the same declared software/runtime environment is achieved by canonically sorting input points $(\mathbf{X}, c, \text{frame\_id})$ prior to Delaunay tetrahedralization.

---

## 12. Dependency Analysis
- **Core Dependencies**: `numpy>=1.24.0`, `scipy>=1.10.0` (`scipy.spatial.cKDTree`, `scipy.spatial.Delaunay`), `pydantic>=2.0.0`.
- **Zero New Dependencies Required**: No OpenCV C++ GUI, Open3D, Trimesh, or PyVista binary extensions are introduced.
- **Python Compatibility**: Strictly compatible with Python 3.10, 3.11, and 3.12.

---

## 13. Rejection Taxonomy (`SurfaceFailureReason`)
- `NON_FINITE_VERTICES`: Non-finite (NaN/Inf) vertex coordinates.
- `INSUFFICIENT_NON_COPLANAR_POINTS`: Fewer than 4 non-coplanar points for 3D Delaunay.
- `DEGENERATE_TRIANGLE`: Faces failing relative area or aspect ratio thresholds.
- `EDGE_LENGTH_EXCEEDED`: Triangle edge $> \alpha_{edge}$.
- `TETRAHEDRON_CIRCUMRADIUS_EXCEEDED`: Delaunay tetrahedron $> \alpha_{radius}$.
- `DEGENERATE_NORMAL_COVARIANCE`: PCA neighborhood collinear or spherically degenerate.
- `EMPTY_INPUT_CLOUD`: Input `DensePointCloud` contains 0 valid points.

---

## 14. Detailed 20-Scenario Test Plan

1. **Minimal Valid 3D Simplex**: 4 non-coplanar points forming a non-degenerate regular tetrahedron yield 4 triangular faces under $\alpha_{radius} \ge R_{tetra}$.
2. **Planar Point Cloud Handling**: Coplanar 3D points detected; dimensionality guard prevents unhandled `QhullError`.
3. **Regular 3D Grid Surface**: Grid on a curved paraboloid surface correctly triangulated into a continuous open sheet.
4. **Sparse Points Rejection**: Points with pairwise spacing $> \alpha_{edge}$ yield 0 faces.
5. **Duplicate Point Deduplication**: Coincident vertices safely deduplicated prior to triangulation.
6. **NaN/Inf Coordinate Rejection**: Non-finite coordinates intercepted with `NON_FINITE_VERTICES`.
7. **Degenerate Needle Triangle Filtering**: Collinear triple rejected by dimensionless relative area check.
8. **Disconnected Spatial Clusters**: Two separate point clusters separated by $d > \alpha_{radius}$ yield disconnected mesh components without bridging faces.
9. **Hole Preservation in Point Grid**: Grid with an intentional square hole preserves the open hole when hole dimension $> \alpha_{edge}$.
10. **Boundary Edge & Vertex Flagging**: Vertices on open hole and exterior perimeters flagged in `is_boundary_vertex`.
11. **Unsupported Gap Rejection**: Two parallel vertical walls separated by $d > \alpha_{radius}$ do not form bridge faces across the gap.
12. **Thin-Structure Separation Test**: Two parallel sheets separated by physical gap $d_{gap}$ triangulate as two separate surfaces without inter-layer bridging when $\alpha_{radius} < d_{gap}$ and $\alpha_{edge} < d_{gap}$.
13. **PCA Normal Estimation on Planar Patch**: Planar surface produces normals aligned with the plane's mathematical normal vector.
14. **PCA Degeneracy Detection**: Collinear line of points triggers `DEGENERATE_NORMAL_COVARIANCE` or falls back to unoriented fallback.
15. **Viewpoint-Assisted Normal Orientation with Cancellation Guard**: Opposing camera centers trigger cancellation guard and fallback to canonical sign without error.
16. **Support Score vs Topology Separation**: High-support vertices on boundary edges maintain $S_f = 1.0$ while `is_boundary_vertex = True`.
17. **Provenance Traceability**: `SurfaceMesh.provenance` retains source frame IDs from input `DensePointCloud`.
18. **Dimensionless Scale Equivariance**: Scaling coordinates and scale parameters by factor $s = 3.0$ preserves face topology and scales areas by $s^2 = 9.0$ within floating-point roundoff.
19. **Deterministic Repeated Execution**: Repeated execution on permuted input point orders under the same runtime environment produces bit-exact identical vertex and face arrays.
20. **Mutation Testing**: Deliberately disabled $\alpha$ checks or area filters are caught by adversarial tests.

---

## 15. Explicit Acceptance Criteria
1. **$\alpha$-Parameter Semantics**: $\alpha_{radius}$ strictly filters 3D Delaunay tetrahedra; $\alpha_{edge}$ strictly bounds triangle facet edge lengths.
2. **Boundary Extraction Semantics**: Boundary facets are extracted strictly as 2-simplices incident to exactly one admissible tetrahedron; vertices and faces are marked based on incident edge multiplicity $|\mathcal{F}(e)| = 1$.
3. **Normal Degeneracy Detection & Cancellation Guard**: PCA conditioning metrics ($\mathcal{P}, \mathcal{L}, \mathcal{S}$) reliably identify collinear/degenerate neighborhoods, and opposing viewpoints trigger the cancellation guard.
4. **Support vs. Topology Distinction**: `is_boundary_vertex` is mathematically independent of `vertex_support_counts` and `face_support_scores`.
5. **Scale-Aware Tolerances**: Area and aspect ratio checks use dimensionless relative ratios, satisfying numerical scale equivariance across well-conditioned dynamic ranges.
6. **Deterministic Execution**: Canonical pre-sorting ensures reproducible mesh generation under the same software runtime environment.

---

## 16. Explicit List of Non-Claims
Phase 3E.4 **WILL NOT CLAIM**:
1. Absolute metric accuracy in meters.
2. Watertight manifold geometry over unobserved regions.
3. Elimination of all geometric ambiguity without external ground truth.
4. Correctness on specular, transparent, or untextured surfaces.
5. Bayesian posterior probability or physical measurement precision.
