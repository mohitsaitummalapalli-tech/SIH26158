# Phase 3E Architecture Specification: Dense Multi-View Stereo (MVS)

## 1. Scientific Purpose

Phase 3E converts the gauge-constrained, bundle-adjusted sparse reconstruction from Phase 3D.1 into dense surface and 3D point geometry using Multi-View Stereo (MVS).

While Phase 3C (Incremental SfM) and Phase 3D.1 (Bundle Adjustment) establish registered camera poses and triangulate sparse feature tracks at salient keypoints, sparse landmarks leave the vast majority of scene surface unobserved. Multi-View Stereo densely samples the visual surface by establishing dense stereo correspondences across overlapping camera frustums, computing per-pixel optical depth maps, filtering depth via cross-view geometric consistency, and fusing consistent 3D observations into a consolidated dense point cloud.

> [!IMPORTANT]
> **Reconstruction Unit Limitation**:
> Phase 3E dense geometry remains strictly in the SfM reconstruction gauge. Absolute metric scale is not established here. All depths and 3D coordinates are expressed in `RECONSTRUCTION_UNITS`. No metric or meter accuracy is claimed until subsequent georeferencing and Sim(3) alignment against calibrated ground truth.

---

## 2. Pipeline Architecture

The Phase 3E dense reconstruction pipeline executes through ten strictly decoupled stages:

```
Optimized SfM (Phase 3D.1)
    ↓
MVS Candidate View Selection (MVSViewPairSelector)
    ↓
Reference / Source View Pairs (MVSViewGraph)
    ↓
Dense Correspondence / Depth Estimation Contract (IMVSDepthEstimator)
    ↓
Depth Map & Confidence Evaluation (DepthMap, DepthConfidenceMap)
    ↓
Cross-View Geometric Consistency (DepthConsistencyChecker)
    ↓
Depth Filtering & Occlusion Tagging (PointVisibilityState)
    ↓
3D Backprojection (depth_to_world_points)
    ↓
Dense Point Fusion (DensePointFusion)
    ↓
Dense Point Cloud (DensePointCloud)
```

---

## 3. Inputs and Data Contracts

Phase 3E consumes the typed contract `MVSInput`:

```python
@dataclass
class MVSInput:
    selected_frame_ids: List[str]
    image_dimensions: Dict[str, Tuple[int, int]]       # frame_id -> (height, width)
    camera_intrinsics: Dict[str, CameraIntrinsics]      # Calibrated pinhole intrinsics
    camera_poses: Dict[str, ExtrinsicPose]             # Optimized poses from Phase 3D.1
    image_paths: Optional[Dict[str, str]] = None       # File paths for streaming (no RAM duplication)
    sparse_landmarks: Optional[Dict[int, TriangulatedTrack]] = None # SfM priors
    dynamic_risk_scores: Dict[str, float] = field(default_factory=dict) # Phase 2 risk
    coordinate_convention: str = "opencv_optical"
    depth_unit: DepthUnit = DepthUnit.RECONSTRUCTION_UNITS
    provenance: Dict[str, Any] = field(default_factory=dict)
```

### Memory & Streaming Safety
To prevent out-of-memory errors on high-resolution UAV video sequences, `MVSInput` stores image path references rather than loading full-resolution uncompressed pixel arrays into memory simultaneously.

---

## 4. Camera Coordinate Convention

Phase 3E strictly preserves the project-wide Phase 3 OpenCV optical camera convention:

$$\mathbf{X}_c = \mathbf{R}_{cw} \mathbf{X}_w + \mathbf{t}_{cw}$$

$$\mathbf{C}_w = -\mathbf{R}_{cw}^T \mathbf{t}_{cw}$$

$$u = f_x \frac{X_c}{Z_c} + c_x, \quad v = f_y \frac{Y_c}{Z_c} + c_y$$

where:
- $+X$ points right in the image plane,
- $+Y$ points downward in the image plane,
- $+Z$ points forward along the optical optical axis (optical depth).

---

## 5. View-Pair Selection & MVS View Graph

Dense depth estimation requires stereo camera pairs with sufficient baseline parallax, adequate visual overlap, and bounded angular divergence.

### Selection Criteria
1. **Geometric Baseline Proxy**: Physical distance between camera centers $\|\mathbf{C}_{src} - \mathbf{C}_{ref}\|$ in reconstruction units. Coincident camera centers ($\|\mathbf{C}\| < 10^{-4}$) are rejected.
2. **Triangulation Angle**: Geodesic rotation angle $\theta \in [0, \pi]$ between camera optical axes. Must fall within $[\theta_{min}, \theta_{max}] = [2.0^\circ, 40.0^\circ]$ (`HEURISTIC_DEFAULT`). Angles exceeding $40^\circ$ suffer severe affine and foreshortening distortion.
3. **Visual Frustum Overlap**: Estimated shared visual coverage proxy based on relative rotation and sequence index distance. Minimum required overlap: $\ge 0.25$ (`HEURISTIC_DEFAULT`).
4. **Dynamic Scene Risk**: Penalty applied from Phase 2 temporal motion analysis:
   $$\text{suitability} = (0.6 \cdot \text{overlap} + 0.4 \cdot \text{baseline}) \times (1.0 - 0.5 \cdot \text{dynamic\_risk})$$

### Output Contract: `MVSViewGraph`
Stores selected edges (`MVSViewPair`) ranked deterministically by suitability score, along with detailed audit logs for every rejected candidate pair.

---

## 6. Depth Map Semantics

The `DepthMap` contract defines 2D optical depth in the reference camera coordinate frame:

- **Coordinate Definition**: Optical depth refers strictly to $Z_c$ (distance along the principal axis), NOT radial distance $\|\mathbf{X}_c\|_2$, nor raw horizontal disparity $d$.
- **Dimensions**: Strictly matches the reference camera sensor dimensions $(H, W)$.
- **Valid Mask**: Explicit boolean raster marking pixels with finite, positive depth ($Z_c \ge Z_{min}$).
- **Continuous Query**: `get_depth_at(u, v)` allows bilinear or nearest depth lookup with boundary verification.

---

## 7. Depth Confidence Semantics

Confidence is explicitly decoupled from depth values. A computed depth value does not imply correctness.

The `DepthConfidenceMap` contract maintains:
1. **Photometric Confidence**: Normalized matching correlation (e.g. NCC, census, or cost-volume peak ratio) in $[0, 1]$ (`HEURISTIC_SCORE`).
2. **Geometric Consistency Confidence**: Proportion of source views confirming depth within tolerance (`HEURISTIC_SCORE`).
3. **Support-View Count**: Integer count of consistent source views.
4. **Visibility State**: Explicit enum classifying every pixel (`PointVisibilityState`).
5. **Overall Confidence**: Calibrated or heuristic blend thresholded by `confidence_threshold` (`HEURISTIC_DEFAULT: 0.5`).

---

## 8. Cross-View Geometric Consistency

The `DepthConsistencyChecker` verifies depth hypotheses across stereo views using two-way projection checks:

Given reference pixel $(u_{ref}, v_{ref})$ with depth $Z_{ref}$:
1. **Backprojection**: $\mathbf{X}_c = Z_{ref} \mathbf{K}_{ref}^{-1} [u_{ref}, v_{ref}, 1]^T \implies \mathbf{X}_w = \mathbf{R}_{ref}^T (\mathbf{X}_c - \mathbf{t}_{ref})$.
2. **Source Projection**: $\mathbf{X}_{c, src} = \mathbf{R}_{src} \mathbf{X}_w + \mathbf{t}_{src}$.
3. **Cheirality & Bounds Check**: Verify $Z_{src, proj} > 0$ and $[u_{src}, v_{src}]^T \in [0, W) \times [0, H)$.
4. **Source Depth Lookup**: Read observed depth $Z_{src, obs}$ at $(u_{src}, v_{src})$.
5. **Relative Depth Disparity Check**:
   $$\frac{|Z_{src, proj} - Z_{src, obs}|}{Z_{src, proj}} \le \tau_{depth} \quad (\text{HEURISTIC\_DEFAULT: } 0.05)$$
6. **Reprojection Back-Check**: Backproject $\mathbf{X}_{w, src}$ from source depth and project into reference camera. Verify:
   $$\|\mathbf{p}_{ref} - \mathbf{p}_{ref, back}\|_2 \le \tau_{reproj} \quad (\text{HEURISTIC\_DEFAULT: } 1.5\text{ px})$$

---

## 9. Occlusion & Visibility State Taxonomy

Dense geometry explicitly represents visibility and failure modes via `PointVisibilityState`:
- `VISIBLE`: Confirmed visible and consistent across reference and source views.
- `OCCLUDED`: Occluded by foreground surface in source view, or outside source camera frustum.
- `INCONSISTENT`: Reprojection or relative depth disagreement exceeds tolerance.
- `INVALID_DEPTH`: Depth is non-positive, NaN, Infinite, or outside configured bounds.
- `INSUFFICIENT_SUPPORT`: Observed by fewer than `min_consistent_views` source cameras.
- `LOW_CONFIDENCE`: Below minimum confidence threshold.
- `VALID`: Successfully passed all validation, occlusion, and consistency gates.

Missing or occluded depth must never be silently converted to zero or backprojected as real world geometry.

---

## 10. 3D Backprojection Contract

`depth_to_world_points` deterministically converts verified depth pixels into `DensePointObservation` records:

$$\mathbf{X}_c = \begin{bmatrix} Z_c \frac{u - c_x}{f_x} \\ Z_c \frac{v - c_y}{f_y} \\ Z_c \end{bmatrix}$$

$$\mathbf{X}_w = \mathbf{R}_{cw}^T (\mathbf{X}_c - \mathbf{t}_{cw})$$

Points with depth outside $[Z_{min}, Z_{max}]$ or failing validity masks are excluded from backprojection.

---

## 11. Dense Point Fusion Contract

The `DensePointFusion` interface consolidates multiple multi-view `DensePointObservation` records into a single `DensePointCloud`:

1. **Spatial Deduplication**: Observations are indexed into a deterministic 3D voxel grid with resolution $\Delta_{voxel}$ (`HEURISTIC_DEFAULT: 0.02` reconstruction units).
2. **Confidence-Weighted Centroid**:
   $$\mathbf{X}_{fused} = \frac{\sum_i w_i \mathbf{X}_{w, i}}{\sum_i w_i}, \quad w_i = \text{confidence}_i$$
3. **Multi-View Support Enforcement**: Voxel clusters observed by fewer than `min_consistent_views` (`HEURISTIC_DEFAULT: 2`) unique frame IDs are discarded.
4. **Deterministic Sorting**: Voxel keys are sorted deterministically to ensure bit-exact, order-independent output.

---

## 12. Failure Taxonomy

Explicit failure reasons defined in `MVSFailureReason`:
- `INSUFFICIENT_VALID_VIEWS`: Fewer registered frames than required for multi-view stereo.
- `INVALID_CAMERA_CALIBRATION`: Missing or uncalibrated intrinsics.
- `MISSING_IMAGE`: Specified image frame reference cannot be accessed.
- `INCOMPATIBLE_IMAGE_DIMENSIONS`: Raster dimensions do not match camera calibration.
- `INVALID_POSE`: Non-finite or non-SO(3) camera pose.
- `INVALID_INTRINSICS`: Non-positive focal length or optical center out of bounds.
- `NO_USABLE_VIEW_PAIRS`: All candidate pairs fail baseline or overlap constraints.
- `INSUFFICIENT_OVERLAP`: Overlap between selected views is below threshold.
- `INSUFFICIENT_GEOMETRIC_BASELINE`: Baseline parallax insufficient for stable triangulation.
- `DEPTH_ESTIMATION_FAILED`: Depth solver failed to produce valid depth map.
- `EXCESSIVE_OCCLUSION`: Majority of pixels occluded across candidate views.
- `INSUFFICIENT_CONSISTENCY`: Cross-view consistency verification failed.
- `DENSE_FUSION_FAILED`: Point cloud fusion failed or produced invalid coordinate outputs.

---

## 13. Dynamic Scene Interaction

UAV video sequences frequently contain moving objects (vehicles, pedestrians, swaying vegetation). Phase 3E integrates dynamic risk metadata from Phase 2:
- View-pair selection penalizes pairs where either view exhibits high dynamic motion risk.
- Confidence maps incorporate dynamic risk attenuation.
- High-risk pixels are prioritized for rejection if cross-view consistency indicates transient surface movement.
- Dynamic objects are prevented from being fused into static world geometry.

---

## 14. Sparse SfM Priors

Sparse 3D landmarks from Phase 3D.1 provide robust geometric priors:
- **Depth Range Initialization**: 5th and 95th percentiles of sparse landmark depths in camera frame define adaptive search bounds $[Z_{min}, Z_{max}]$.
- **Consistency Anchors**: Dense depth hypotheses in the neighborhood of sparse landmarks can be verified against triangulated track depths.
- Dense depth is NOT forced to interpolate noisy sparse points directly.

---

## 15. Plugin Architecture: `IMVSDepthEstimator`

To prevent architectural lock-in, depth estimation algorithms satisfy an abstract plugin interface:

```python
class IMVSDepthEstimator(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def estimate_depth_map(
        self,
        ref_frame_id: str,
        source_frame_ids: List[str],
        mvs_input: MVSInput,
        config: MVSConfig,
    ) -> Tuple[DepthMap, DepthConfidenceMap]:
        pass
```

Future implementations can plug in classical PatchMatch MVS, plane-sweep stereo, or foundation depth models without altering downstream consistency, backprojection, or fusion contracts.

---

## 16. Summary of Architectural Guarantees

| Contract Element | Specification | Rationale |
| :--- | :--- | :--- |
| **Coordinate Gauge** | `RECONSTRUCTION_UNITS` | Monocular scale ambiguity; no metric scale claims. |
| **Camera Model** | OpenCV optical pinhole | Direct compatibility with Phase 3 contracts. |
| **Depth Definition** | Optical depth $Z_c$ | Strict geometric definition; disparity is not world depth. |
| **Occlusion State** | Typed enum (`PointVisibilityState`) | No silent zero-filling; unobserved geometry is explicit. |
| **Consistency** | 2-way reprojection + relative depth | Discards transient and non-static false matches. |
| **Multi-View Support** | Minimum $\ge 2$ unique views | Filters isolated monocular artifacts. |
| **Execution** | Fully deterministic | Sorted keys, fixed tie-breaking, reproducible results. |
