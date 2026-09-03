# Phase 3E.4 Step 3 — Visibility-Aware Surface-to-Image Texture Association
## Architecture & Mathematical Contract Specification (Final Forensic Revision)

**Status**: CONTRACT & DESIGN SPECIFICATION
**Baseline Lock**: `cfea803` (Phase 3E.4 Step 2 Locked)
**Implementation State**: IMPLEMENTED & AUDITED

---

## 1. Executive Summary & Core Principles

Phase 3E.4 Step 3 defines the rigorous mathematical contract and software architecture for **Visibility-Aware Surface-to-Image Texture Association**.

The objective is to establish an unambiguous, occlusion-verified, and quality-scored mapping between discrete surface samples of the reconstructed 3D mesh (`SurfaceMesh` from Step 2) and calibrated camera frames (Phase 3D.1 / 3E) so that downstream texture mapping utilizes only valid, geometrically unobstructed image evidence.

### Fundamental Scientific Invariants
1. **Projection Cheirality & Inviolability**: Every texture observation must satisfy rigid coordinate projection into positive camera space ($X_{c, z} > 0$) and strictly inside active sensor bounds.
2. **Mesh-Relative Visibility Semantics**:
   - `VISIBLE` strictly means: **"No occluding triangle exists in the supplied reconstructed mesh along the tested finite segment."**
   - It does **not** prove physical real-world visibility. The reconstructed mesh may be incomplete and omit physical occluders. Step 3 never claims that mesh-relative visibility equals ground-truth scene visibility.
3. **Proximity $\neq$ Visibility**: Physical proximity between a camera and a surface point does not imply visibility. Intervening mesh triangles must be proven absent via finite segment raycasting.
4. **Complete Audit Trail**: Every evaluated sample-camera pair generates exactly one `CandidateDecisionRecord`. All decisions (accepted, rejected at any gate, or accepted-but-not-retained due to top-$K$ limits) remain 100% auditable.
5. **Observed vs. Unobserved Semantics**:
   - A surface sample is `OBSERVED` if and only if **at least one candidate survives all geometric, visibility, and quality gates and is retained as a `TextureObservation`**.
   - If geometrically visible candidates exist but all fail quality thresholds (`LOW_QUALITY_SCORE`), the sample is `UNOBSERVED`.
   - `sample_coverage_ratio = (number of OBSERVED samples) / (total samples)`. This is strictly sample count coverage, **not** surface-area coverage.
6. **Heuristic Quality Scoring**: Observation scores are strictly `HEURISTIC_SCORE \in [0, 1]`, representing observational favorability, **never probabilities**.
7. **No Ray-Origin Normal-Sign Dependence**: The ray origin is offset strictly along the line-of-sight vector toward the camera ($\mathbf{v}_{\text{view}}$). Arbitrary unoriented PCA normal signs cannot flip the ray direction or falsely reject visible geometry.
8. **Normal Invariance**: Flipping $\mathbf{n} \to -\mathbf{n}$ leaves ray geometry, visibility, $s_{\text{angle}}$, and the composite score completely unchanged (following from $s_{\text{angle}} = |\mathbf{n} \cdot \mathbf{v}_{\text{view}}|$).
9. **No Texture Hallucination**: Unseen, unobserved, or occluded geometry is strictly tagged `UNOBSERVED`. Step 3 executes zero generative inpainting, hole completion, or AI texture synthesis.
10. **Gauge & Unit Preservation**: Depth and distance quantities remain in `DepthUnit.RECONSTRUCTION_UNITS` with `is_metric_scale = False`.

---

## 2. Mathematical Formulations

### 2.1 Coordinate Transform & Camera Optical Center
Given world surface sample point $\mathbf{X}_w \in \mathbb{R}^3$, world-to-camera rotation matrix $\mathbf{R}_{cw} \in SO(3)$, and translation vector $\mathbf{t}_{cw} \in \mathbb{R}^3$:
$$\mathbf{X}_c = \mathbf{R}_{cw} \mathbf{X}_w + \mathbf{t}_{cw}$$

The camera optical center in world coordinates is:
$$\mathbf{C}_w = -\mathbf{R}_{cw}^T \mathbf{t}_{cw}$$

### 2.2 Camera Model & Undistorted Domain Contract
Step 3 operates strictly under the **calibrated undistorted pinhole image domain**:
> **Camera Model Contract**: All input camera matrices $\mathbf{K}$ and pixel coordinates $(u, v)$ correspond to the calibrated undistorted image domain (as rectified in Phase 1 / 2). Step 3 does not apply raw non-linear lens distortion models.

For camera coordinates $\mathbf{X}_c = [X_{c, x}, X_{c, y}, X_{c, z}]^T$, if $X_{c, z} > 0$:
$$u = f_x \frac{X_{c, x}}{X_{c, z}} + c_x, \quad v = f_y \frac{X_{c, y}}{X_{c, z}} + c_y$$

Active sensor bounds gate:
$$u \in [m_{\text{border}}, W - 1 - m_{\text{border}}], \quad v \in [m_{\text{border}}, H - 1 - m_{\text{border}}]$$
where $m_{\text{border}}$ is `image_border_margin_px` (default $4.0\text{ px}$).

### 2.3 Scale-Equivariant Finite Ray Segment Formulation
Let $d_{\text{cam}} = \|\mathbf{C}_w - \mathbf{X}_w\|_2$.

**Zero Distance Guard**:
If $d_{\text{cam}} \le 0.0$ (camera optical center coincides with surface sample), the query is immediately rejected with status `DEGENERATE_CAMERA`.

If $d_{\text{cam}} > 0.0$, the unit line-of-sight vector from sample $\mathbf{X}_w$ to camera $\mathbf{C}_w$ is:
$$\mathbf{v}_{\text{view}} = \frac{\mathbf{C}_w - \mathbf{X}_w}{d_{\text{cam}}}$$

The finite line segment endpoints are defined strictly along $\mathbf{v}_{\text{view}}$ (independent of surface normal orientation):
$$\epsilon = \epsilon_{\text{ratio}} \cdot d_{\text{cam}}$$
$$\mathbf{O} = \mathbf{X}_w + \epsilon \mathbf{v}_{\text{view}}$$
$$\mathbf{E} = \mathbf{C}_w - \epsilon \mathbf{v}_{\text{view}}$$
$$\mathbf{D} = \mathbf{E} - \mathbf{O} = (1.0 - 2\epsilon_{\text{ratio}}) (\mathbf{C}_w - \mathbf{X}_w)$$
$$\mathbf{R}(t) = \mathbf{O} + t \mathbf{D}, \quad t \in [0.0, 1.0]$$
where $\epsilon_{\text{ratio}}$ is dimensionless (default $10^{-6}$).

### 2.4 Scale-Safe Möller-Trumbore Ray-Triangle Intersection
For a candidate mesh triangle facet $\mathcal{T} = (\mathbf{V}_0, \mathbf{V}_1, \mathbf{V}_2)$ with edge vectors:
$$\mathbf{E}_1 = \mathbf{V}_1 - \mathbf{V}_0, \quad \mathbf{E}_2 = \mathbf{V}_2 - \mathbf{V}_0$$
$$\mathbf{P} = \mathbf{D} \times \mathbf{E}_2, \quad \det = \mathbf{E}_1 \cdot \mathbf{P}$$

#### Numerical Tolerance Semantics
The tolerances:
$$\tau_{\det} = 10^{-7}, \quad \tau_{\text{bary}} = 10^{-6}, \quad \tau_t = 10^{-6}$$
are **dimensionless numerical heuristics controlling floating-point classification and roundoff boundaries**.
They are **NOT**:
- physical tolerances
- spatial accuracy guarantees
- metric accuracy estimates
- reconstruction accuracy claims

The dimensionless mathematical formulations are:

#### Dimensionless Determinant Parallelism Gate
$$|\det| \le \tau_{\det} \cdot \|\mathbf{E}_1\|_2 \cdot \|\mathbf{D}\|_2 \cdot \|\mathbf{E}_2\|_2$$
If true, the ray segment is parallel to the facet (no intersection). Otherwise, solve barycentric coordinates:
$$\mathbf{T} = \mathbf{O} - \mathbf{V}_0, \quad u_{\text{bary}} = \frac{\mathbf{T} \cdot \mathbf{P}}{\det}$$
$$\mathbf{Q} = \mathbf{T} \times \mathbf{E}_1, \quad v_{\text{bary}} = \frac{\mathbf{D} \cdot \mathbf{Q}}{\det}$$
$$t_{\text{hit}} = \frac{\mathbf{E}_2 \cdot \mathbf{Q}}{\det}$$

#### Dimensionless Barycentric & Segment Hit Gate
$$\text{Hit} \iff \begin{cases}
u_{\text{bary}} \ge -\tau_{\text{bary}} \\
v_{\text{bary}} \ge -\tau_{\text{bary}} \\
u_{\text{bary}} + v_{\text{bary}} \le 1.0 + \tau_{\text{bary}} \\
-\tau_t \le t_{\text{hit}} \le 1.0 + \tau_t
\end{cases}$$

An occlusion occurs if and only if a candidate non-self triangle satisfies the hit condition.

---

## 3. Explicit Self-Intersection Exclusion Rule

Topological exclusion is enforced at the data structure level:
1. **Facet-Centroid Samples** ($\mathbf{P}_j$ derived from triangle index $j$):
   $$\mathcal{T}_{\text{query}} = \mathcal{T}_{\text{mesh}} \setminus \{j\}$$
   Triangle facet $j$ is never tested for occlusion.
2. **Vertex Samples** ($\mathbf{P}_i$ derived from vertex index $i$):
   Let $\mathcal{F}(v_i)$ be all triangles incident to vertex $i$:
   $$\mathcal{T}_{\text{query}} = \mathcal{T}_{\text{mesh}} \setminus \mathcal{F}(v_i)$$
   All incident triangles are excluded from the occlusion test.

---

## 4. Decoupling Normal Orientation from Visibility

1. **No Hard Backface Culling**: Geometric visibility depends purely on cheirality ($X_{c, z} > 0$), sensor frustum bounds, and line segment occlusion. Unoriented local PCA normal signs will **never** cause a visible surface to be rejected as `BACKFACING`.
2. **Continuous Acute Angle Weighting**: Surface normal $\mathbf{n}_w$ is used exclusively as an observation quality factor:
   $$s_{\text{angle}} = |\mathbf{n}_w \cdot \mathbf{v}_{\text{view}}| \in [0, 1]$$
3. **Normal Flip Invariance**:
   Because $s_{\text{angle}} = |\mathbf{n}_w \cdot \mathbf{v}_{\text{view}}|$ and ray geometry is defined along $\mathbf{v}_{\text{view}}$, flipping $\mathbf{n} \to -\mathbf{n}$ leaves:
   - ray geometry $\mathbf{R}(t)$
   - ray visibility result (`VISIBLE` / `OCCLUDED`)
   - angular factor $s_{\text{angle}}$
   - composite score $S_{\text{composite}}$
   **completely unchanged**.

---

## 5. Deterministic AABB BVH Contract

To avoid brute-force $O(M \cdot F)$ scaling while introducing zero external C++ dependencies, Step 3 mandates an in-memory **Axis-Aligned Bounding Box (AABB) Bounding Volume Hierarchy (BVH)**:

### 5.1 Construction Contract
- **Node Bounding Box**: $[\mathbf{x}_{\min}, \mathbf{x}_{\max}]$ enclosing all child triangles.
- **Deterministic Object-Median Split**:
  1. For the current triangle set, compute centroids $\mathbf{c}_k = \frac{1}{3}(\mathbf{V}_0 + \mathbf{V}_1 + \mathbf{V}_2)$.
  2. Select split axis $a^* = \arg\max_{a \in \{0, 1, 2\}} (\max_k c_{k, a} - \min_k c_{k, a})$.
  3. Sort deterministically by primary key $c_{k, a^*}$ and secondary tie-breaker triangle index $k$.
  4. Split at median index.
  5. Terminate when leaf size $\le N_{\text{leaf\_max}}$ (default 4).

### 5.2 Traversal & Complexity Claims
- Slab-method ray-box intersection tests determine traversal into child nodes.
- Möller-Trumbore is evaluated only on non-excluded triangles in intersecting leaf nodes.
- **Complexity Specification**:
  - **BVH Construction**: Expected $O(F \log F)$ for deterministic median splitting.
  - **Ray Traversal**: Expected sublinear behavior relative to exhaustive linear scanning.
  - **Worst-Case**: $O(F)$ per ray (e.g. pathological geometry with completely overlapping bounding boxes).
  - *No unqualified $O(M \log F)$ claim is made.*

---

## 6. Heuristic Scoring & Two-Pass Distance Normalization

### 6.1 Two-Pass Normalization Workflow
1. **Pass 1 (Geometric Visibility & Discovery)**:
   For sample $\mathbf{X}_w$, evaluate all candidate cameras. Collect visible candidates $\mathcal{C}_{\text{vis}}(\mathbf{X}_w)$.
   Compute:
   $$d_{\min}(\mathbf{X}_w) = \min_{k \in \mathcal{C}_{\text{vis}}(\mathbf{X}_w)} \|\mathbf{C}_{w, k} - \mathbf{X}_w\|_2$$
2. **Pass 2 (Scoring & Ranking)**:
   For each visible camera $k \in \mathcal{C}_{\text{vis}}(\mathbf{X}_w)$:
   $$s_{\text{dist}, k} = \frac{d_{\min}(\mathbf{X}_w)}{\|\mathbf{C}_{w, k} - \mathbf{X}_w\|_2} \in (0, 1]$$
   *(Note: $s_{\text{dist}} = 1.0$ for the closest geometrically visible candidate camera; this reflects relative spatial proximity, not image resolution).*

### 6.2 Composite Heuristic Score Formulation
$$S_{\text{composite}} = S_{\text{geom}} \times S_{\text{frame}} \times S_{\text{dynamic}} \in [0, 1]$$

1. **Geometric Score**:
   $$S_{\text{geom}} = s_{\text{dist}} \times s_{\text{angle}} \times s_{\text{margin}}$$
   $$s_{\text{margin}} = \min\left(1.0, \frac{\max(0.0, d_{\text{edge}} - m_{\text{border}})}{\text{margin\_score\_reference\_px}}\right)$$
   where `margin_score_reference_px` (default $32.0\text{ px}$) is a configurable heuristic pixel parameter.
2. **Frame Quality Score ($S_{\text{frame}}$)**:
   Consumes Phase 1 / 2B analytics:
   $$S_{\text{frame}} = q_{\text{sharpness}} \times (1.0 - p_{\text{blur}}) \times q_{\text{exposure}}$$
   - **Missing Metrics Contract**: If any metric is missing, the documented neutral fallback $0.5$ is assigned, and `metrics_missing=True` is recorded in decision record provenance.
   - **Invalid Metrics Contract**: If any input metric is `NaN`, `Inf`, or outside $[0, 1]$, the candidate is **explicitly rejected** with status `INVALID_QUALITY_METRICS` (never silently clipped or substituted).
3. **Dynamic Scene Risk Score ($S_{\text{dynamic}}$)**:
   Consumes Phase 2A / 2C analytics:
   $$S_{\text{dynamic}} = 1.0 - \text{clip}(\text{dynamic\_risk\_score}, 0.0, 1.0)$$

---

## 7. Data Contracts & Type Definitions

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from src.geometry.mvs import DepthUnit

class TextureSampleType(str, Enum):
    """Geometric primitive type being textured."""
    VERTEX = "VERTEX"
    FACET_CENTROID = "FACET_CENTROID"

class SampleObservationState(str, Enum):
    """Sample-level observation status."""
    OBSERVED = "OBSERVED"
    UNOBSERVED = "UNOBSERVED"

class TextureQueryStatus(str, Enum):
    """Per-camera query evaluation status."""
    VISIBLE = "VISIBLE"
    NEGATIVE_DEPTH = "NEGATIVE_DEPTH"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    OCCLUDED = "OCCLUDED"
    LOW_QUALITY_SCORE = "LOW_QUALITY_SCORE"
    DEGENERATE_CAMERA = "DEGENERATE_CAMERA"
    NON_FINITE_PARAMETERS = "NON_FINITE_PARAMETERS"
    INVALID_QUALITY_METRICS = "INVALID_QUALITY_METRICS"

class DecisionStatus(str, Enum):
    """Auditable decision classification."""
    ACCEPTED_RETAINED = "ACCEPTED_RETAINED"
    ACCEPTED_NOT_RETAINED = "ACCEPTED_NOT_RETAINED"
    REJECTED = "REJECTED"

class TextureAssociationConfig(BaseModel):
    """Configuration governing projection, occlusion, and scoring tolerances."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    image_border_margin_px: float = Field(
        default=4.0, ge=0.0,
        description="Sensor margin in pixels excluded to prevent lens edge artifacts."
    )
    margin_score_reference_px: float = Field(
        default=32.0, gt=0.0,
        description="Configurable pixel-domain transition distance for margin scoring."
    )
    ray_offset_epsilon_ratio: float = Field(
        default=1e-6, gt=0.0, lt=1e-1,
        description="Dimensionless ratio of ray length used to offset ray origin and target."
    )
    tau_det: float = Field(
        default=1e-7, gt=0.0,
        description="Dimensionless determinant tolerance ratio for ray-triangle parallelism."
    )
    tau_bary: float = Field(
        default=1e-6, ge=0.0,
        description="Dimensionless barycentric coordinate numerical tolerance."
    )
    tau_t: float = Field(
        default=1e-6, ge=0.0,
        description="Dimensionless ray segment parameter numerical tolerance."
    )
    min_composite_score: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description="Minimum composite heuristic score required to accept observation."
    )
    max_observations_per_sample: int = Field(
        default=8, ge=1,
        description="Maximum number of candidate observations preserved per sample (top-K)."
    )

@dataclass(frozen=True)
class CandidateDecisionRecord:
    """Exact audit record for EVERY evaluated sample-camera pair."""
    sample_type: TextureSampleType
    sample_index: int
    frame_id: str
    decision: DecisionStatus
    query_status: TextureQueryStatus
    projected_pixels: Optional[Tuple[float, float]]
    depth: Optional[float]
    distance_to_cam: Optional[float]
    composite_score: Optional[float]
    rejection_reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TextureObservation:
    """Individual accepted, unoccluded surface texture observation."""
    sample_type: TextureSampleType
    sample_index: int
    frame_id: str
    pixel_coords: Tuple[float, float]
    depth: float
    incidence_angle_deg: float
    distance_to_cam: float
    geometric_score: float
    frame_quality_score: float
    dynamic_risk_score: float
    composite_score: float
    provenance: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SurfaceTextureAssociationMap:
    """Consolidated association output mapping samples to ranked observations."""
    sample_type: TextureSampleType
    total_samples: int
    sample_states: List[SampleObservationState]
    observations_by_sample: Dict[int, List[TextureObservation]]
    best_observation_by_sample: Dict[int, Optional[TextureObservation]]
    decision_records: List[CandidateDecisionRecord]
    sample_coverage_ratio: float
    depth_unit: DepthUnit = DepthUnit.RECONSTRUCTION_UNITS
    is_metric_scale: bool = False
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def rejection_records(self) -> List[CandidateDecisionRecord]:
        """Derived convenience view of all rejected candidates."""
        return [r for r in self.decision_records if r.decision == DecisionStatus.REJECTED]
```

---

## 8. Association & Visibility Algorithm

```text
Algorithm: Visibility-Aware Texture Association (Final Revision)
Inputs:
  - SurfaceMesh: V, F, normals, topology
  - Cameras: Dict[frame_id -> {R_cw, t_cw, K (undistorted), W, H, quality, dynamic_risk}]
  - Config: TextureAssociationConfig
Outputs:
  - SurfaceTextureAssociationMap

1. Canonically sort surface samples (by index) and cameras (by lexicographical frame_id).
2. Construct deterministic AABB BVH over mesh faces F (median centroid split).
3. Pre-compute incident face exclusion sets for each sample:
     - FACET_CENTROID j: excluded_faces = {j}
     - VERTEX i: excluded_faces = incident_faces(i)

4. Initialize decision_records = []
5. For each sample s in {0, ..., S - 1}:
     Pass 1 (Geometric Visibility & Discovery):
       visible_candidates = []
       For each frame_id, cam in Cameras:
         a. If cam parameters non-finite:
              decision_records.append(Record(REJECTED, NON_FINITE_PARAMETERS)), continue.
         b. Compute X_c = R_cw * P_s + t_cw.
         c. If X_c[2] <= 0:
              decision_records.append(Record(REJECTED, NEGATIVE_DEPTH)), continue.
         d. Compute (u, v) in undistorted pinhole domain.
         e. If (u, v) outside [margin, Dim - 1 - margin]:
              decision_records.append(Record(REJECTED, OUT_OF_BOUNDS, (u, v))), continue.
         f. d_cam = ||C_w - P_s||. If d_cam <= 0:
              decision_records.append(Record(REJECTED, DEGENERATE_CAMERA)), continue.
         g. Form finite segment R(t) = O + tD for t in [0, 1] using:
              O = P_s + eps * v_view, E = C_w - eps * v_view, D = E - O.
         h. Traverse BVH against non-excluded triangles using scale-safe Möller-Trumbore.
         i. If hit detected in t in [-tau_t, 1 + tau_t]:
              decision_records.append(Record(REJECTED, OCCLUDED, hit_triangle)), continue.
         j. Add to visible_candidates: (frame_id, cam, (u, v), X_c[2], d_cam).

     If visible_candidates is empty:
       sample_states[s] = UNOBSERVED
       continue

     d_min = min(c.d_cam for c in visible_candidates)

     Pass 2 (Quality Scoring, Acceptance & Top-K Ranking):
       scored_candidates = []
       For each cand in visible_candidates:
         Validate Phase-2 quality metrics:
           - If NaN, Inf, or < 0 or > 1:
               decision_records.append(Record(REJECTED, INVALID_QUALITY_METRICS))
               continue
           - If missing:
               use fallback 0.5 with metrics_missing=True in provenance
         s_dist = d_min / cand.d_cam
         s_angle = |n_s · v_view|
         s_margin = min(1.0, max(0.0, d_edge - margin) / margin_score_reference_px)
         S_geom = s_dist * s_angle * s_margin
         S_comp = S_geom * S_frame * S_dynamic
         If S_comp < min_composite_score:
           decision_records.append(Record(REJECTED, LOW_QUALITY_SCORE, score=S_comp))
         Else:
           scored_candidates.append(cand with S_comp)

     If scored_candidates is empty:
       sample_states[s] = UNOBSERVED
       continue

     sample_states[s] = OBSERVED
     Sort scored_candidates deterministically by (S_comp DESC, frame_id ASC).

     For rank, cand in enumerate(scored_candidates):
       If rank < max_observations_per_sample:
         decision_records.append(Record(ACCEPTED_RETAINED, VISIBLE, score=cand.S_comp))
         observations_by_sample[s].append(TextureObservation(cand))
       Else:
         decision_records.append(Record(ACCEPTED_NOT_RETAINED, VISIBLE, score=cand.S_comp))

     best_observation_by_sample[s] = observations_by_sample[s][0]

6. Compute sample_coverage_ratio = count(OBSERVED) / S.
7. Return SurfaceTextureAssociationMap.
```

---

## 9. Comprehensive 22-Scenario Verification Test Plan

1. **Complete Audit Trail Cardinality**:
   Verify `len(decision_records) == total_samples * total_cameras`.
2. **Accepted-Retained vs. Accepted-Not-Retained Audit**:
   Configure `max_observations_per_sample = 2` with 5 valid cameras. Verify 2 records are `ACCEPTED_RETAINED` and 3 are `ACCEPTED_NOT_RETAINED`.
3. **Observed vs. Unobserved State Invariant**:
   Verify sample is `OBSERVED` if and only if `len(observations_by_sample[s]) > 0`.
4. **All-Visible-But-Low-Score $\to$ UNOBSERVED**:
   Place camera in visible line of sight with $S_{\text{frame}} \times S_{\text{dynamic}} < \text{min\_composite\_score}$. Verify query is `LOW_QUALITY_SCORE` and sample state is `UNOBSERVED`.
5. **Zero Camera-Sample Distance Guard**:
   Place $\mathbf{C}_w = \mathbf{P}_w$. Verify query returns `DEGENERATE_CAMERA` without division-by-zero crash.
6. **Undistorted Pinhole Camera Model Contract**:
   Verify projection matches analytical $K [R | t]$ in undistorted coordinates.
7. **Flipped Normal Invariance Test**:
   Invert surface normal $\mathbf{n} \to -\mathbf{n}$. Verify ray segment geometry, visibility result, $s_{\text{angle}}$, and composite score are **all identical** (following from $s_{\text{angle}} = |\mathbf{n} \cdot \mathbf{v}_{\text{view}}|$).
8. **Facet-Centroid Self-Exclusion Test**:
   Raycast from triangle centroid; verify containing triangle $j$ is never flagged as an occluder.
9. **Vertex Incident-Face Exclusion Test**:
   Raycast from sharp concave crease vertex; verify incident triangles sharing the vertex do not self-occlude.
10. **Dimensionless Determinant Parallelism Gate**:
    Verify parallelism threshold behaves identically across scales $10^{-6}$ to $10^{6}$.
11. **Dimensionless Barycentric & Segment Hit Gate**:
    Ray grazing triangle edge within $\tau_{\text{bary}}$ is cleanly evaluated without numerical overflow.
12. **Scale Sweep ($10^{-6}$ to $10^{6}$)**:
    Scale entire scene and cameras by $10^{-6}, 10^{-4}, 1.0, 10^4, 10^6$. Verify invariant $(u, v)$, ray decisions, and normalized scores.
13. **Mesh-Relative Visibility Semantics**:
    Demonstrate that a missing physical occluder in an open mesh evaluates as `VISIBLE`, confirming visibility is mesh-relative.
14. **Deterministic AABB BVH vs. Brute-Force Equivalence**:
    Verify BVH output matches exhaustive linear scan across 1,000 triangles and 200 rays.
15. **Open Mesh & Hole Traversal**:
    Ray passing through an open boundary/hole in a mesh is recognized as `VISIBLE`.
16. **Candidate-First $d_{\min}$ Normalization**:
    Test two visible cameras at distances 2.0 and 4.0; verify $s_{\text{dist}} = 1.0$ and $0.5$ respectively.
17. **Missing Phase-2 Metrics Test**:
    Verify documented neutral fallback $0.5$ is applied, `metrics_missing=True` is recorded, and provenance contains fallback audit entry.
18. **Invalid Phase-2 Metrics Rejection Test**:
    Pass `NaN`, `Inf`, and out-of-range values ($<0$ or $>1$) in quality metrics. Verify candidate is explicitly rejected with `INVALID_QUALITY_METRICS` (never silently clipped or substituted).
19. **Runtime Environment Determinism**:
    50 random permutations of input camera dictionaries produce identical association maps.
20. **Canonical Camera Ordering**:
    Cameras sorted lexicographically by `frame_id` guarantee deterministic tie-breaking.
21. **Complete Rejection Provenance**:
    Every rejected query records exact reason (`NEGATIVE_DEPTH`, `OUT_OF_BOUNDS`, `OCCLUDED`, `LOW_QUALITY_SCORE`, `INVALID_QUALITY_METRICS`).
22. **Zero Metric Scale Leakage**:
    Verify `depth_unit == DepthUnit.RECONSTRUCTION_UNITS` and `is_metric_scale == False`.

---

## 10. Explicit Non-Goals & Scope Boundaries

1. **No Metric Scale Promotion**: Output remains strictly in `RECONSTRUCTION_UNITS` with `is_metric_scale = False`.
2. **No Physical Scene Visibility Claims**: Visibility is strictly mesh-relative.
3. **No AI / Generative Inpainting**: Unobserved geometry remains untextured.
4. **No UV Atlas Generation or Chart Packing**: Step 3 establishes multi-view observation association, not UV atlasing.
5. **No Photometric Multi-Band Blending**: Seam blending is deferred to downstream rendering stages.
6. **No Raw Lens Distortion Modeling**: Operates strictly in the calibrated undistorted pinhole domain.

---

**DOCUMENTATION COMPLETE. NO CODE IMPLEMENTED. NO GIT COMMIT PERFORMED.**
