# Phase 3E.4 Step 4 — Multi-View Surface Texture Reconstruction
## Architecture & Mathematical Contract Specification (Final Forensic Revision)

**Status**: CONTRACT & DESIGN SPECIFICATION ONLY
**Baseline Lock**: `46fcc74` (Phase 3E.4 Step 2 locked at `cfea803`, Step 3 locked at `43b6b9d`)
**Implementation State**: NO CODE IMPLEMENTATION / NO GIT COMMIT

---

## 1. Executive Summary & Anti-Hallucination Core Principles

Phase 3E.4 Step 4 establishes the formal mathematical contract and pipeline architecture for **Multi-View Surface Texture Reconstruction**. The objective is to reconstruct trustworthy, photometrically consistent surface texture from calibrated multi-view drone frames across the reconstructed `SurfaceMesh` (Step 2), strictly governed by visibility-qualified observations evaluated at exact 3D surface points.

### 1.1 The Ten Core Principles

1. **Strict Exact-Point Evidence Invariant (Anti-Hallucination)**:
   A discrete UV texel $(x, y)$ or 3D surface point $\mathbf{P}_w$ may receive an RGB texture value **if and only if its exact continuous 3D coordinate has valid, visibility-qualified observation evidence**. Being inside a triangle with an observed centroid or adjacent to an observed vertex does **NOT** grant texture evidence to an unobserved texel. If a surface point lacks direct observation evidence, the system **MUST NOT** fabricate, smoothly interpolate across unobserved boundaries, or hallucinate RGB colors. Unobserved texels must strictly evaluate to `OperationalTextureState.UNOBSERVED` with sentinel null-values (`RGB = (0, 0, 0)`, `alpha = 0.0`, `confidence = 0.0`).
2. **Zero Ingestion of Rejected Candidates**:
   Any camera frame rejected during visibility qualification (due to negative optical depth, out-of-bounds, mesh occlusion, or low composite score) is **permanently disqualified** from contributing to surface texture.
3. **Full Candidate History & Provenance Preservation**:
   Every reconstructed surface texture unit preserves the complete lineage of contributing source frame IDs, fractional pixel coordinates, individual observation scores, photometric weights, residual deviations, and outlier classification records.
4. **Candidate-Order Invariance & Runtime Determinism**:
   The fused texture result must be identical regardless of candidate dictionary insertion order, triangle ordering in the input mesh, camera indexing order, or floating-point evaluation order. All chart layouts, packing algorithms, and tie-breaking rules must be deterministic.
5. **Photometric Robustness Without Radiometric Claims**:
   The system must explicitly accommodate real-world drone aerial capture variations—including automatic exposure adjustments, vignetting, minor illumination gradients, and compression artifacts—strictly via robust statistical consistency weighting. The system **MUST NOT claim true radiometric calibration or physical surface reflectance recovery**; true radiometric calibration requires a validated physical sensor-illumination model.
6. **Statistically Grounded Photometric Outlier Rejection**:
   Contradictory RGB observations (caused by dynamic elements, transient reflections, moving shadows, or residual registration inaccuracies) must not be naively averaged. A robust M-estimator with Tukey biweight influence must downweight and eliminate photometric outliers.
7. **Decoupled Heuristic Texture Confidence**:
   Every reconstructed texture element carries a normalized texture confidence score $C_{\text{tex}} \in [0, 1]$, explicitly defined as a deterministic heuristic composite of observation count, observation quality, view angular diversity, and photometric consensus. It is **explicitly defined as a heuristic score, NOT a statistical probability**.
8. **Explicit Texture State Taxonomy**:
   Reconstructed regions are partitioned into mutually exclusive, auditable operational states (`OBSERVED_TEXTURE`, `WEAK_TEXTURE`, `PHOTOMETRIC_CONFLICT`, `UNOBSERVED`, `INVALID_INPUT`). These states must never be collapsed into an ambiguous single RGB fallback.
9. **Zero Metric Scale Leakage**:
   Reconstruction units ($[L]$) remain abstract and uncalibrated (`depth_unit = RECONSTRUCTION_UNITS`, `is_metric_scale = False`). Texture parameterization, UV texel density, and photometric color processing must remain invariant under coordinate scaling $\mathbf{X}_w \to s \mathbf{X}_w$.
10. **Headless & Fully Automated Execution**:
    The texture pipeline must operate entirely in headless Python/NumPy environments with zero dependence on Blender, external GUI tools, or interactive artist workflows.

---

## 2. Surface Representation Comparative Analysis & Scientific Selection

Texture reconstruction on 3D polygonal surfaces can be implemented via three primary representations. Below is an explicit comparative evaluation against the SIH26158 project requirements.

| Dimension | Option A: Per-Vertex Color Fusion | Option B: Per-Face Color Fusion | Option C: UV Texture Atlas Parameterization |
|---|---|---|---|
| **Data Structure** | RGB array of shape $(V, 3)$, confidence $(V,)$ | RGB array of shape $(F, 3)$, confidence $(F,)$ | 2D Image Atlas $(H_{\text{tex}}, W_{\text{tex}}, 3)$ + parallel Confidence Map |
| **Evidence Granularity** | Exact vertex coordinates $\mathbf{V}_i$ | Exact centroid coordinates $\mathbf{C}_j$ | Continuous surface point $\mathbf{P}_w(x, y)$ |
| **Spatial Detail Resolution** | Constrained by mesh vertex density. High-frequency aerial patterns (road lines, roof shingles) are completely blurred over sparse planar faces. | Piecewise constant flat shading. Produces angular, faceted visual artifacts; cannot resolve intra-triangle detail. | **Decoupled from mesh vertex density**. Can represent sub-facet detail (centimeter-level road lines, pavement markers, roof textures). |
| **Sampling Complexity** | Minimal ($O(V \cdot K)$) | Minimal ($O(F \cdot K)$) | Moderate ($O(N_{\text{texels}} \cdot K)$); requires rasterization/barycentric mapping. |
| **Seam & Discontinuity Handling** | Continuous across shared vertices; no UV seams. | Discontinuous across all face boundaries. | Requires explicit UV chart gutter padding and boundary color management. |
| **Standard Export Compliance** | glTF / PLY vertex colors (`COLOR_0`) | Custom face property or duplicated vertices | **Standard industry glTF 2.0 PBR material / OBJ+MTL texture maps**. |
| **Anti-Hallucination Auditability** | Direct per-vertex audit | Direct per-face audit | Exact texel-to-surface-to-camera audit trail via parallel metadata maps. |

### 2.1 Scientific Architecture Selection: Hybrid UV Texture Atlas with Vertex Fallback

**Selected Representation**: **Option C (UV Texture Atlas Parameterization)** is selected as the primary representation for Phase 3E.4 Step 4, supplemented by a zero-overhead **Option A (Per-Vertex Color)** projection.

**Scientific Justification**:
1. **Preservation of Observed Real-World Details**:
   In Phase 3E.4 Step 2, planar regions (such as flat roofs, parking lots, and road segments) are optimally represented by large, simplified Delaunay triangles to prevent geometric noise. In Option A or B, a large $10\text{ m} \times 10\text{ m}$ roof facet would receive only 3 vertex colors or a single flat color, destroying all painted lines, HVAC textures, and tile patterns. A UV Texture Atlas maintains high-frequency visual evidence independent of geometric decimation.
2. **Standard Interoperability**:
   Downstream visualization, GIS orthomosaics, and 3D web viewers require standard UV-mapped texture images (`albedo_map.png` + `confidence_map.png`).
3. **Anti-Hallucination Compatibility**:
   Every texel $(u, v)$ inside a valid UV chart maps uniquely via barycentric coordinates to an exact 3D surface point $\mathbf{P}_w(x, y) \in \mathcal{T}_j$. Step 3 visibility and quality gates evaluated at $\mathbf{P}_w(x, y)$ govern which cameras may color that texel. Texels outside valid charts or lacking direct observation evidence receive explicit sentinel null-values.

---

## 3. Automated Deterministic UV Parameterization & Chart Packing

To maintain headless, deterministic, and repeatable execution without external GUI tools, Step 4 employs an automated, conformal planar chart parameterization and deterministic bin packing.

### 3.1 Planar Conformal Chart Generation
1. **Connected Component Clustering**:
   Mesh faces $\mathcal{F}$ are clustered into contiguous surface charts $\mathcal{C}_k \subset \mathcal{F}$ based on surface normal continuity:
   $$\mathbf{n}_a \cdot \mathbf{n}_b \ge \cos(\theta_{\text{chart\_max}}), \quad \text{where } \theta_{\text{chart\_max}} = 45^\circ \text{ (configurable)}$$
   Charts are bounded to a maximum diameter in reconstruction units to prevent severe projective distortion.
2. **Local Planar Projection**:
   For each chart $\mathcal{C}_k$, compute the area-weighted average normal $\bar{\mathbf{n}}_k$. Construct an orthonormal local coordinate basis $(\mathbf{e}_{k, 1}, \mathbf{e}_{k, 2}, \bar{\mathbf{n}}_k)$ where $\mathbf{e}_{k, 1}$ is deterministically aligned with the chart's primary principal axis via PCA on vertex coordinates.
3. **2D Local Parameterization**:
   Each 3D vertex $\mathbf{V}_i \in \mathcal{C}_k$ is mapped to 2D chart coordinates:
   $$u_i' = \mathbf{V}_i \cdot \mathbf{e}_{k, 1}, \quad v_i' = \mathbf{V}_i \cdot \mathbf{e}_{k, 2}$$

### 3.2 Deterministic Chart Packing into Texture Atlas
1. **Resolution & Texel Density Policy**:
   Let $A_{\text{mesh}}$ be the total 3D surface area of the mesh in reconstruction units. For a target texture atlas dimension $H_{\text{tex}} \times W_{\text{tex}}$ (e.g., $2048 \times 2048$), compute the uniform texel scale factor:
   $$\rho_{\text{texel}} = \sqrt{\frac{W_{\text{tex}} \cdot H_{\text{tex}} \cdot \eta_{\text{pack}}}{A_{\text{mesh}}}} \quad [\text{texels / reconstruction-unit}]$$
   where $\eta_{\text{pack}} \in (0, 1)$ is the target packing efficiency (default $0.70$).
2. **Chart AABB Bounding & Gutter Padding**:
   Each chart polygon in 2D is scaled by $\rho_{\text{texel}}$ and enclosed in an axis-aligned bounding box of size $(w_k, h_k)$. A gutter padding of $p_{\text{gutter}} \ge 4$ texels is added to all chart borders to completely prevent bilinear texture bleeding across chart boundaries.
3. **Deterministic Shelf-Bin Packing**:
   Charts are sorted deterministically by primary key: height descending, secondary key: width descending, tertiary key: minimum face index ascending. Packed into horizontal shelves from top to bottom, left to right.
4. **Normalized UV Coordinates**:
   Final coordinates are normalized to $[0.0, 1.0]^2$:
   $$u = \frac{x_{\text{atlas}} + 0.5}{W_{\text{tex}}}, \quad v = \frac{y_{\text{atlas}} + 0.5}{H_{\text{tex}}}$$

---

## 4. Texel-to-Surface Mapping & Anti-Hallucination Evidence Contract

### 4.1 Exact UV Texel → Triangle → Barycentric Coordinates → 3D Surface Point
For each discrete integer texel coordinate $(x, y) \in [0, W_{\text{tex}}-1] \times [0, H_{\text{tex}}-1]$:
1. Compute continuous normalized UV atlas coordinates:
   $$u = \frac{x + 0.5}{W_{\text{tex}}}, \quad v = \frac{y + 0.5}{H_{\text{tex}}}$$
2. Locate the chart $\mathcal{C}_k$ and specific triangle $\mathcal{T}_j = (\mathbf{V}_0, \mathbf{V}_1, \mathbf{V}_2)$ whose UV triangle vertices $(\mathbf{u}_0, \mathbf{u}_1, \mathbf{u}_2)$ contain $(u, v)$.
3. Compute exact 2D barycentric coordinates $(\lambda_0, \lambda_1, \lambda_2)$ by solving:
   $$\begin{pmatrix} u_1 - u_0 & u_2 - u_0 \\ v_1 - v_0 & v_2 - v_0 \end{pmatrix} \begin{pmatrix} \lambda_1 \\ \lambda_2 \end{pmatrix} = \begin{pmatrix} u - u_0 \\ v - v_0 \end{pmatrix}, \quad \lambda_0 = 1.0 - \lambda_1 - \lambda_2$$
   Inside-triangle condition: $\lambda_i \ge -\tau_{\text{bary}}$ for all $i \in \{0, 1, 2\}$, where $\tau_{\text{bary}} = 10^{-6}$.
4. If $(u, v)$ falls outside all valid triangle charts (e.g. atlas background, gutter padding), it is assigned:
   $$\text{state} = \text{UNOBSERVED}, \quad \text{RGB} = (0, 0, 0), \quad \alpha = 0.0, \quad C_{\text{tex}} = 0.0$$
5. For points inside triangle $\mathcal{T}_j$, reconstruct the continuous 3D surface point $\mathbf{P}_w(x, y)$:
   $$\mathbf{P}_w(x, y) = \lambda_0 \mathbf{V}_0 + \lambda_1 \mathbf{V}_1 + \lambda_2 \mathbf{V}_2$$
6. Compute the interpolated surface normal:
   $$\mathbf{n}_w(x, y) = \frac{\lambda_0 \mathbf{n}_{v0} + \lambda_1 \mathbf{n}_{v1} + \lambda_2 \mathbf{n}_{v2}}{\|\lambda_0 \mathbf{n}_{v0} + \lambda_1 \mathbf{n}_{v1} + \lambda_2 \mathbf{n}_{v2}\|_2}$$

> [!CRITICAL]
> **Anti-Hallucination Contract Rule**:
> **UV parameterization is strictly a geometric coordinate mapping; it NEVER creates, implies, or substitutes for photographic observation evidence.**
> A texel $(x, y)$ inside triangle $\mathcal{T}_j$ does NOT inherit observation evidence merely because triangle $\mathcal{T}_j$ had an observed centroid or vertex in Step 3. Observation evidence must be directly evaluated for the specific point $\mathbf{P}_w(x, y)$.

### 4.2 Formal Evidence Retrieval Interface for Arbitrary Surface Points
To ensure an arbitrary UV texel's 3D surface point $\mathbf{P}_w(x, y)$ obtains genuine visibility-qualified evidence without heuristic shortcutting, Step 4 exposes a formal point-level evaluation interface consuming the locked Step 3 BVH and camera models:

```python
def evaluate_surface_point_observations(
    point_w: np.ndarray,          # (3,) exact continuous surface point P_w(x, y)
    normal_w: np.ndarray,         # (3,) interpolated unit normal n_w(x, y)
    containing_face_idx: int,     # Face index j containing point_w
    candidate_cameras: Dict[str, TextureSourceCamera],
    bvh: DeterministicAABBBVH,
    config: TextureAssociationConfig,
) -> List[TextureObservation]:
    """Evaluates strict, visibility-qualified observations for an exact 3D surface point.

    Guarantees that centroid or vertex visibility is NEVER falsely attributed
    to an arbitrary surface point that may be locally occluded.
    """
```

For the point $\mathbf{P}_w(x, y)$, this pipeline executes the strict Step 3 geometric gates:
1. **Gate 1 (Finite Parameters)**: Verify $\mathbf{P}_w, \mathbf{n}_w, \mathbf{C}_w, \mathbf{R}_{cw}, \mathbf{t}_{cw}$ are finite.
2. **Gate 2 (Coincident Distance Guard)**: Verify $d_{\text{cam}} = \|\mathbf{C}_w - \mathbf{P}_w(x, y)\|_2 > 0.0$.
3. **Gate 3 (Optical Depth Cheirality)**: $\mathbf{X}_c = \mathbf{R}_{cw} \mathbf{P}_w(x, y) + \mathbf{t}_{cw}$, verify $X_{c, z} > 0.0$.
4. **Gate 4 (Sensor Margin Bounds)**: Verify $m_{\text{border}} \le u_{\text{cam}} \le W_{\text{cam}} - 1 - m_{\text{border}}$ and $m_{\text{border}} \le v_{\text{cam}} \le H_{\text{cam}} - 1 - m_{\text{border}}$.
5. **Gate 5 (Finite Line Segment Raycast Occlusion)**:
   $$\mathbf{O} = \mathbf{P}_w(x, y) + \epsilon \mathbf{v}_{\text{view}}, \quad \mathbf{E} = \mathbf{C}_w - \epsilon \mathbf{v}_{\text{view}}, \quad \mathbf{D} = \mathbf{E} - \mathbf{O}, \quad t \in [0.0, 1.0]$$
   where $\epsilon = 10^{-6} \cdot d_{\text{cam}}$ (forensic contract value).
   Occlusion query against BVH excludes strictly the containing face $\{ \mathcal{T}_j \}$.
6. **Gate 6 (Quality & Composite Score)**: Calculate $S_{\text{composite}}$ and threshold against `min_composite_score`.

**No Evidence Rule**:
Let $\mathcal{K}(x, y) = \{(\mathbf{c}_m, w_m, \text{frame\_id}_m)\}_{m=1}^M$ be the sampled colors and prior weights from cameras passing all 6 gates for point $\mathbf{P}_w(x, y)$.
If $\mathcal{K}(x, y) = \emptyset$, the texel has **ZERO valid evidence**:
$$\text{state} = \text{UNOBSERVED}, \quad \text{RGB} = (0, 0, 0), \quad \alpha = 0.0, \quad C_{\text{tex}} = 0.0$$
**NO texture value is generated.**

---

## 5. Multi-View Photometric Residuals & Robust M-Estimator Fusion

When multiple cameras observe the same surface point $\mathbf{P}_w(x, y)$, sampled RGB values differ due to automatic exposure, perspective foreshortening, sensor noise, or transient photometric outliers (moving objects, reflections, dynamic shadows).

### 5.1 Observation Prior Weight
Each candidate observation $m \in \mathcal{K}(x, y)$ inherits a prior weight $w_m$ from Step 3:
$$w_m = S_{\text{geom}, m} \times S_{\text{frame}, m} \times (1.0 - \text{dynamic\_risk}_m)$$
where:
$$S_{\text{geom}, m} = |\mathbf{n}_w(x, y) \cdot \mathbf{v}_{\text{view}, m}| \times \frac{d_{\min}}{d_m} \times s_{\text{margin}, m}$$

### 5.2 Exact Mathematical Formulation of Tukey Biweight Fusion

For a texel with candidate observations $(\mathbf{c}_m, w_m)_{m=1}^M$ where $\mathbf{c}_m = (r_m, g_m, b_m)^T \in [0, 255]^3$:

1. **Initial Robust Color Anchor ($\mathbf{c}_{\text{anchor}}$)**:
   Compute perceived luminance for each candidate:
   $$L_m = 0.2126 r_m + 0.7152 g_m + 0.0722 b_m$$
   Sort candidates by luminance: $L_{(1)} \le L_{(2)} \le \dots \le L_{(M)}$.
   Find weighted median candidate index $k^*$ such that:
   $$\sum_{k=1}^{k^*} w_{(k)} \ge 0.5 \sum_{m=1}^M w_m$$
   Initialize color anchor $\mathbf{c}_{\text{anchor}}^{(0)} = \mathbf{c}_{(k^*)}$.

2. **Photometric Residuals ($r_m$) Against Anchor**:
   At iteration $t$, the photometric residual $r_m^{(t)}$ for observation $m$ is defined directly against the robust color anchor $\mathbf{c}_{\text{anchor}}^{(t)}$:
   $$r_m^{(t)} = \frac{\|\mathbf{c}_m - \mathbf{c}_{\text{anchor}}^{(t)}\|_2}{\sqrt{3} \cdot 255.0} \in [0.0, 1.0]$$

3. **Robust Scale Estimate ($\hat{\sigma}$)**:
   The robust scale $\hat{\sigma}^{(t)}$ is computed consistently from the centered residuals $\{r_m^{(t)}\}_{m=1}^M$:
   $$\text{med\_r}^{(t)} = \text{median}_w \left( \{r_m^{(t)}\}_{m=1}^M \right)$$
   $$\hat{\sigma}^{(t)} = 1.4826 \times \text{median}_w \left( \left\{ |r_m^{(t)} - \text{med\_r}^{(t)}| \right\}_{m=1}^M \right) + \epsilon_{\text{scale}}$$
   where $\epsilon_{\text{scale}} = 10^{-4}$ prevents division by zero when observations are identical.

4. **Normalized Residual ($u_m$)**:
   The normalized residual $u_m^{(t)}$ scales the residual by the robust scale:
   $$u_m^{(t)} = \frac{r_m^{(t)}}{k_{\text{tukey}} \cdot \hat{\sigma}^{(t)}}$$
   where $k_{\text{tukey}} = 4.685$ is the standard tuning constant providing 95% Gaussian asymptotic efficiency.

5. **Tukey Biweight Influence Weight**:
   $$\psi(u_m^{(t)}) = \begin{cases} \left( 1.0 - (u_m^{(t)})^2 \right)^2 & \text{if } |u_m^{(t)}| \le 1.0 \\ 0.0 & \text{if } |u_m^{(t)}| > 1.0 \end{cases}$$
   Observations with $|u_m^{(t)}| \le 1.0$ are classified as **inliers**; observations with $|u_m^{(t)}| > 1.0$ are classified as **photometric outliers** (weight strictly 0.0).

6. **Iterative Reweighted Anchor Update**:
   Combine the geometric prior weight with the robust photometric weight:
   $$\tilde{w}_m^{(t)} = w_m \cdot \psi(u_m^{(t)})$$
   $$\mathbf{c}_{\text{anchor}}^{(t+1)} = \frac{\sum_{m=1}^M \tilde{w}_m^{(t)} \mathbf{c}_m}{\sum_{m=1}^M \tilde{w}_m^{(t)}}$$
   Iterate until convergence $\|\mathbf{c}_{\text{anchor}}^{(t+1)} - \mathbf{c}_{\text{anchor}}^{(t)}\|_2 < \tau_{\text{conv}} = 0.5 \text{ RGB units}$ (or maximum 5 iterations). Let $\mathbf{c}^*$ and $\psi(u_m^*)$ denote converged values.

---

## 6. Photometric Conflict Definition & Confidence Formulation

### 6.1 Mathematical Definition of `PHOTOMETRIC_CONFLICT`
An observation $m$ is an inlier if $\psi(u_m^*) > 0.0$. Let:
$$M_{\text{inliers}} = \sum_{m=1}^M \mathbb{I}(\psi(u_m^*) > 0.0)$$
Define the inlier consensus ratio:
$$\kappa_{\text{consensus}} = \frac{\sum_{m \in \text{inliers}} w_m \cdot \psi(u_m^*)}{\sum_{m=1}^M w_m} \in [0.0, 1.0]$$
Define the weighted mean inlier residual:
$$\bar{r}_{\text{inlier}} = \frac{\sum_{m \in \text{inliers}} w_m r_m^*}{\sum_{m \in \text{inliers}} w_m}$$

**Formal Conflict Condition**:
When $M \ge 2$ observations exist, the texel is classified as **`PHOTOMETRIC_CONFLICT`** if any of the following hold:
1. $M_{\text{inliers}} < 1$ (zero mutual consensus).
2. $\kappa_{\text{consensus}} < \tau_{\text{consensus\_min}}$ (heuristic threshold, default $0.35$).
3. $\bar{r}_{\text{inlier}} > \tau_{\text{conflict}}$ (heuristic threshold, default $0.20$).

**Action on Conflict**:
No RGB average is fabricated. The texel is assigned:
$$\text{state} = \text{PHOTOMETRIC\_CONFLICT}, \quad \text{RGB} = (0, 0, 0), \quad \alpha = 0.0, \quad C_{\text{tex}} = 0.0$$
and all contradictory candidate frames are recorded in provenance.

### 6.2 Exact Mathematical Confidence Equations ($C_{\text{tex}}$)
Texture confidence $C_{\text{tex}} \in [0.0, 1.0]$ measures the trustworthiness of the fused color. It is explicitly defined as a **heuristic composite score, NOT a statistical probability**:
$$C_{\text{tex}} = C_{\text{count}} \times C_{\text{qual}} \times C_{\text{geom}} \times C_{\text{cons}}$$

1. **Observation Count Factor ($C_{\text{count}}$)**:
   $$C_{\text{count}} = \min\left(1.0, \frac{M_{\text{inliers}}}{N_{\text{target}}}\right), \quad \text{where default } N_{\text{target}} = 4$$
   For one inlier observation ($M_{\text{inliers}} = 1$):
   $$C_{\text{count}} = \frac{1}{4} = 0.25$$
2. **Quality Factor ($C_{\text{qual}}$)**:
   $$C_{\text{qual}} = \frac{\sum_{m \in \text{inliers}} w_m S_{\text{frame}, m}}{\sum_{m \in \text{inliers}} w_m} \in [0.0, 1.0]$$
3. **Geometric Factor ($C_{\text{geom}}$)**:
   $$C_{\text{geom}} = \max_{m \in \text{inliers}} |\mathbf{n}_w(\mathbf{P}_w) \cdot \mathbf{v}_{\text{view}, m}| \in [0.0, 1.0]$$
4. **Photometric Consensus Factor ($C_{\text{cons}}$)**:
   - **Multiple Observations ($M \ge 2, M_{\text{inliers}} \ge 2$)**:
     $$C_{\text{cons}} = \max\left(0.0, 1.0 - \frac{\bar{r}_{\text{inlier}}}{\tau_{\text{conflict}}}\right) \in [0.0, 1.0]$$
   - **Single Observation ($M = 1, M_{\text{inliers}} = 1$)**:
     Because inter-view consensus cannot be evaluated with a single view, $C_{\text{cons}}$ is assigned the **documented neutral value**:
     $$C_{\text{cons}} = 0.50 \quad (\text{neutral consensus heuristic})$$

### 6.3 Mathematical Derivation for Single-Observation Confidence
For a single valid observation ($M = 1, M_{\text{inliers}} = 1$):
$$C_{\text{tex}} = C_{\text{count}} \times C_{\text{qual}} \times C_{\text{geom}} \times C_{\text{cons}}$$
$$C_{\text{tex}} = 0.25 \times S_{\text{frame}} \times |\mathbf{n}_w \cdot \mathbf{v}_{\text{view}}| \times 0.50 = 0.125 \times S_{\text{frame}} \times |\mathbf{n}_w \cdot \mathbf{v}_{\text{view}}|$$
Since $S_{\text{frame}} \le 1.0$ and $|\mathbf{n}_w \cdot \mathbf{v}_{\text{view}}| \le 1.0$:
$$C_{\text{tex}} \le 0.125 \times 1.0 \times 1.0 = 0.125$$
Because the minimum confidence threshold for `OBSERVED_TEXTURE` is $\tau_{\text{conf\_min}} = 0.20$, **any single observation strictly evaluates to $C_{\text{tex}} \le 0.125 < 0.20$**, mathematically guaranteeing classification as `WEAK_TEXTURE`. An unverified single observation can never claim high confidence.

---

## 7. Anti-Hallucination Rule for Vertex-Color Fallback

In addition to the UV atlas, Step 4 generates per-vertex fallback colors on `SurfaceMesh.vertices`.

> [!CRITICAL]
> **Vertex-Color Fallback Anti-Hallucination Invariant**:
> A mesh vertex $\mathbf{V}_i$ may receive a vertex color **if and only if vertex $\mathbf{V}_i$ itself has accepted Step 3 observation evidence** (`TextureSampleType.VERTEX`).
> **The vertex-color fallback MUST NOT interpolate, propagate, or smear color from an observed facet centroid or adjacent face onto an unobserved vertex.**
> If vertex $\mathbf{V}_i$ has no accepted Step 3 observations:
> $$\text{vertex\_color}[\mathbf{V}_i] = (0, 0, 0), \quad \text{vertex\_confidence}[\mathbf{V}_i] = 0.0, \quad \text{vertex\_state}[\mathbf{V}_i] = \text{UNOBSERVED}$$

---

## 8. UV Seam Consistency & Discontinuity Management

1. **Exact 3D Edge Alignment**:
   When two adjacent triangles $\mathcal{T}_a$ and $\mathcal{T}_b$ are placed into different UV charts, their shared 3D edge corresponds to separate 2D boundary texels. For any texel within distance $\delta_{\text{seam}} = 0.5 / \rho_{\text{texel}}$ of a shared 3D edge, its 3D sample point $\mathbf{P}_w$ is projected onto the exact shared 3D segment.
2. **Evidence-Driven Observation Governance Across Seams**:
   Seam texels sample cameras strictly based on genuine 3D line-of-sight visibility to the shared edge point $\mathbf{P}_w$. Cameras unoccluded to the edge evaluate identical 3D ray geometries, preventing artificial color discontinuities across chart seams while remaining 100% evidence-driven.
3. **Gutter Dilation (Anti-Bleeding)**:
   Chart interiors are dilated by 2 texels into the padding gutter using exact edge-normal extrapolation. This prevents mipmapping and bilinear texture filtering from sampling black unobserved atlas background pixels at chart edges.

---

## 9. Coverage Metric Definitions

Coverage metrics provide an auditable summary of texture reconstruction completeness. They are defined over the active parameterized surface area:

$$\text{observed\_texel\_ratio} = \frac{N_{\text{OBSERVED\_TEXTURE}}}{N_{\text{surface\_texels}}}$$
$$\text{weakly\_observed\_texel\_ratio} = \frac{N_{\text{WEAK\_TEXTURE}}}{N_{\text{surface\_texels}}}$$
$$\text{unobserved\_texel\_ratio} = \frac{N_{\text{UNOBSERVED}} + N_{\text{PHOTOMETRIC\_CONFLICT}}}{N_{\text{surface\_texels}}}$$

$$\text{observed\_texel\_ratio} + \text{weakly\_observed\_texel\_ratio} + \text{unobserved\_texel\_ratio} \equiv 1.0$$

> [!IMPORTANT]
> **Audit Warning on Terminology**:
> These ratios report **texel observation coverage across the reconstructed mesh**. They do **NOT** represent physical ground-truth accuracy, photographic resolution, or real-world surface completeness.

---

## 10. Formal Data Structures & Interfaces

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from src.geometry.mvs import DepthUnit
from src.geometry.surface_reconstruction import SurfaceMesh
from src.geometry.texture_association import (
    DeterministicAABBBVH,
    SurfaceTextureAssociationMap,
    TextureAssociationConfig,
    TextureObservation,
    TextureSampleType,
    TextureSourceCamera,
)


class OperationalTextureState(str, Enum):
    """Mutually exclusive operational state for each textured element."""
    OBSERVED_TEXTURE = "OBSERVED_TEXTURE"
    WEAK_TEXTURE = "WEAK_TEXTURE"
    PHOTOMETRIC_CONFLICT = "PHOTOMETRIC_CONFLICT"
    UNOBSERVED = "UNOBSERVED"
    INVALID_INPUT = "INVALID_INPUT"


class TextureReconstructionConfig(BaseModel):
    """Configuration governing UV parameterization, fusion, and confidence thresholds."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    atlas_width: int = Field(default=2048, ge=256, le=8192)
    atlas_height: int = Field(default=2048, ge=256, le=8192)
    gutter_padding_px: int = Field(default=4, ge=2, le=32)
    chart_max_normal_angle_deg: float = Field(default=45.0, gt=0.0, le=90.0)
    target_packing_efficiency: float = Field(default=0.70, gt=0.1, le=0.95)

    tukey_tuning_constant: float = Field(default=4.685, gt=0.0)
    max_m_estimator_iterations: int = Field(default=5, ge=1, le=20)
    convergence_threshold_rgb: float = Field(default=0.5, gt=0.0)

    min_confidence_observed: float = Field(default=0.20, ge=0.0, le=1.0)
    photometric_conflict_threshold: float = Field(default=0.20, gt=0.0, le=1.0)
    min_consensus_fraction: float = Field(default=0.35, gt=0.0, le=1.0)
    target_observation_count: int = Field(default=4, ge=1)


@dataclass(frozen=True)
class CandidateColorSample:
    """Individual sampled color evidence from an accepted Step 3 observation."""
    frame_id: str
    camera_pixel: Tuple[float, float]
    raw_rgb: Tuple[float, float, float]
    prior_weight: float
    tukey_weight: float
    is_inlier: bool
    residual: float


@dataclass(frozen=True)
class FusedTextureElement:
    """Fused visual and diagnostic result for a surface point."""
    state: OperationalTextureState
    rgb: Tuple[int, int, int]
    confidence: float
    inlier_count: int
    total_candidate_count: int
    contributing_frames: List[str]


@dataclass(frozen=True)
class ReconstructedTextureAtlas:
    """Complete multi-view reconstructed texture atlas with diagnostics and provenance."""
    # 2D Texture maps
    albedo_atlas: np.ndarray          # (H, W, 3) uint8 RGB
    confidence_atlas: np.ndarray      # (H, W) float32 in [0, 1]
    state_atlas: np.ndarray           # (H, W) uint8 enum representation

    # Mesh parameterization
    uv_coordinates: np.ndarray        # (V, 2) or (F, 3, 2) float32 in [0, 1]
    vertex_colors: np.ndarray         # (V, 3) uint8 fallback vertex colors
    vertex_confidences: np.ndarray    # (V,) float32
    vertex_states: List[OperationalTextureState]

    # Coverage statistics
    total_surface_texels: int
    observed_texel_ratio: float
    weakly_observed_texel_ratio: float
    unobserved_texel_ratio: float
    photometric_conflict_texel_ratio: float

    # Scale and metadata
    depth_unit: DepthUnit = DepthUnit.RECONSTRUCTION_UNITS
    is_metric_scale: bool = False
    config: TextureReconstructionConfig = field(default_factory=TextureAssociationConfig)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
```

---

## 11. 25-Scenario Adversarial Verification & Test Plan

| Scenario ID | Test Name | Focus & Invariant Verified |
|---|---|---|
| **SCEN-01** | `test_anti_hallucination_zero_step3_yields_unobserved` | Surface point with zero Step 3 observations yields strictly `UNOBSERVED`, `confidence=0.0`, `alpha=0.0`. |
| **SCEN-02** | `test_observed_centroid_with_unobserved_texel` | Facet centroid has Step 3 observation, but corner texel's line-of-sight is occluded: corner texel strictly evaluates to `UNOBSERVED` with `alpha=0.0`. |
| **SCEN-03** | `test_observed_vertex_vs_unsupported_neighboring_texel` | Vertex has Step 3 observation, but adjacent interior texel has no line of sight: texel strictly evaluates to `UNOBSERVED`. |
| **SCEN-04** | `test_vertex_fallback_cannot_propagate_unsupported_evidence` | Face centroid is observed, but incident vertex has no direct vertex observation: vertex color strictly remains `UNOBSERVED`, `confidence=0.0`. |
| **SCEN-05** | `test_rejected_step3_candidates_never_enter_fusion` | Candidates marked `REJECTED` in Step 3 are strictly excluded from texel sampling. |
| **SCEN-06** | `test_single_observation_confidence_exactness` | Single valid observation assigns documented neutral consensus $C_{\text{cons}} = 0.50$, resulting in $C_{\text{tex}} \le 0.125$ (`WEAK_TEXTURE`). |
| **SCEN-07** | `test_multiple_agreeing_observations_fusion` | Multiple views with identical RGB produce exact color, maximum consensus $C_{\text{cons}} = 1.0$, high confidence. |
| **SCEN-08** | `test_tukey_residual_scale_sensitivity` | Outlier candidate with residual $r > 4.685 \hat{\sigma}$ receives $\psi(u) = 0.0$ and zero weight in fused color. |
| **SCEN-09** | `test_photometric_conflict_threshold_rejection` | Observations split into polarized clusters with $\bar{r}_{\text{inlier}} > \tau_{\text{conflict}}$: marked `PHOTOMETRIC_CONFLICT`, `alpha=0.0`. |
| **SCEN-10** | `test_deterministic_candidate_order_invariance` | Permuting candidate list order produces bit-for-bit identical fused RGB and confidence. |
| **SCEN-11** | `test_deterministic_chart_packing_order` | Permuting mesh triangle input order produces identical UV chart layout and packing. |
| **SCEN-12** | `test_seam_equivalent_surface_points_with_different_candidates` | Shared 3D edge evaluated in two charts: candidate sets and Tukey fusion remain evidence-driven without color divergence. |
| **SCEN-13** | `test_gutter_padding_anti_bleeding` | Padding texels between charts do not bleed unobserved background into chart interiors. |
| **SCEN-14** | `test_texel_barycentric_interpolation_bounds` | Reconstructed 3D surface point $\mathbf{P}_w(x, y)$ strictly satisfies $\lambda_i \in [0, 1]$ and $\sum \lambda_i = 1$. |
| **SCEN-15** | `test_grazing_angle_downweighting` | Cameras with $|\mathbf{n} \cdot \mathbf{v}_{\text{view}}| \approx 0$ receive lower prior weights than normal-aligned views. |
| **SCEN-16** | `test_coverage_metric_partition_unity` | Proves $\text{observed\_ratio} + \text{weak\_ratio} + \text{unobserved\_ratio} \equiv 1.0$ within floating tolerance. |
| **SCEN-17** | `test_no_metric_scale_leakage` | Verifies `depth_unit == RECONSTRUCTION_UNITS` and `is_metric_scale == False` are strictly maintained. |
| **SCEN-18** | `test_extreme_scale_sweep_invariance` | Mesh coordinates scaled by $10^{-6}$ to $10^6$: UV coordinates, states, and normalized scores are invariant. |
| **SCEN-19** | `test_nan_inf_pixel_handling` | Corrupted image pixels (NaN, Inf) are detected and rejected without poisoning fusion. |
| **SCEN-20** | `test_invalid_rgb_range_rejection` | Pixel values outside $[0, 255]$ are rejected as invalid input. |
| **SCEN-21** | `test_provenance_audit_trail_completeness` | Every reconstructed element records contributing frame IDs, pixel coordinates, and weights. |
| **SCEN-22** | `test_zero_distance_camera_handling` | Coincident camera-sample pairs do not cause divide-by-zero during texel projection. |
| **SCEN-23** | `test_headless_execution_no_gui_dependency` | Full UV parameterization and atlas generation runs with zero display/OpenGL/Blender dependencies. |
| **SCEN-24** | `test_partially_observed_mesh_boundaries` | Half-observed, half-occluded mesh correctly transitions across boundary with zero color bleed. |
| **SCEN-25** | `test_repeated_execution_hash_identity` | Executing the pipeline 10 consecutive times on identical input yields identical SHA-256 atlas hashes. |

---

## 12. Design Approval & Next Steps Gate

This document serves as the **Design & Mathematical Contract** for Phase 3E.4 Step 4.
**DO NOT WRITE PRODUCTION CODE. DO NOT COMMIT.**
Awaiting explicit user review and approval before proceeding to Step 4 implementation.
