# Phase 3A: Classical Feature Extraction & Robust Descriptor Matching Architecture

## 1. Executive Summary & Objective

Phase 3A implements the deterministic **Classical 2D Feature Extraction and Robust Descriptor Matching** subsystem. It converts quality-assessed, coverage-selected canonical RGB keyframes (`KeyframeSelectionResult` / `DecodedFrame`) into detected 2D interest points, binary ORB descriptors, candidate descriptor correspondences, and spatial match distribution diagnostics.

> **CRITICAL SCIENTIFIC PRINCIPLES & GUARDRAILS:**
> 1. **DESCRIPTOR SIMILARITY $\neq$ GEOMETRIC CORRECTNESS**:
>    - This module outputs candidate descriptor matches only.
>    - It makes **no claims** regarding epipolar coplanarity, relative camera poses, or 3D coordinate accuracy.
>    - Geometric verification via Essential/Fundamental matrix estimation and RANSAC inlier filtering belongs strictly to **Phase 3B**.
> 2. **COORDINATE OBSERVATION LEVEL**:
>    - All keypoint locations $(u, v)$ are directly observed in pixel raster coordinates (`PIXEL_OBSERVED`).
>    - Origin $(0, 0)$ is at top-left, $+u$ extends rightward across columns, $+v$ extends downward across rows.
> 3. **THRESHOLD SEMANTICS**:
>    - All parameters (e.g. `max_features = 2000`, `lowe_ratio = 0.75`, `min_accepted_matches = 30`, `max_descriptor_distance = 64.0`) are explicitly classified as `HEURISTIC_DEFAULT`.
> 4. **DETERMINISM**:
>    - The classical baseline utilizes OpenCV's `ORB` (Oriented FAST and Rotated BRIEF) algorithm, providing reproducible binary feature representations without external deep-learning dependencies.

---

## 2. Pipeline Architecture

```
                    DecodedFrame (Canonical RGB uint8)
                                       │
                                       ▼
                   [1] Input Validation & Grayscale Conversion
                            (ITU-R BT.601 Weights)
                                       │
                                       ▼
                   [2] ORB Multi-Scale Feature Detection
                       (FAST Corners + Pyramid Octaves)
                                       │
                                       ▼
                   [3] Oriented BRIEF Descriptor Computation
                         (256-bit Binary Descriptors)
                                       │
                                       ▼
                            FeatureExtractionResult
                                       │
                                       ▼
                   [4] Brute-Force Hamming Descriptor Matching
                             (k=2 Nearest Neighbors)
                                       │
                                       ▼
                   [5] Match Filtering & Spatial Diagnostics
                     ├── Lowe Ratio Test (d1 <= 0.75 * d2)
                     ├── Mutual Nearest Neighbor Cross-Check
                     └── Spatial Distribution & Grid Occupancy
                                       │
                                       ▼
                             FeatureMatchResult
                                       │
                                       ▼ (Phase 3B Input)
                            FeatureCorrespondences
```

---

## 3. Mathematical Formulations & Data Contracts

### 3.1 Pixel Raster Coordinates
Observed interest point coordinates:
$$\mathbf{p} = \begin{bmatrix} u \\ v \end{bmatrix} \in \mathbb{R}^2, \quad 0 \le u < \text{width}, \; 0 \le v < \text{height}$$

### 3.2 Binary Descriptor Representation & Hamming Distance
For 256-bit binary ORB descriptors $\mathbf{d}_a, \mathbf{d}_b \in \{0, 1\}^{256}$ represented as 32-byte arrays (`uint8`):
$$d_H(\mathbf{d}_a, \mathbf{d}_b) = \sum_{k=1}^{256} (\mathbf{d}_a[k] \oplus \mathbf{d}_b[k]) \in [0, 256]$$
The matching strategy enforces:
$$d_H(\mathbf{d}_a, \mathbf{d}_b) \le d_{\text{max}} \quad (\text{default } d_{\text{max}} = 64.0, \text{ HEURISTIC\_DEFAULT})$$

### 3.3 Lowe Ratio Test (Descriptor Ambiguity Filter)
For the nearest neighbor distance $d_1 = d_H(\mathbf{d}_a, \mathbf{d}_{b, 1})$ and second-nearest neighbor distance $d_2 = d_H(\mathbf{d}_a, \mathbf{d}_{b, 2})$:
$$\text{Accept if } d_1 \le r_{\text{Lowe}} \cdot d_2 \quad (\text{default } r_{\text{Lowe}} = 0.75, \text{ HEURISTIC\_DEFAULT})$$
*This heuristic rejects ambiguous descriptor matches occurring in repetitive patterns.*

### 3.4 Mutual Nearest Neighbor Consistency (Cross-Check)
A candidate match $(i, j)$ between Frame A and Frame B is accepted only if:
$$\text{BestMatch}(i \in A) = j \in B \quad \text{and} \quad \text{BestMatch}(j \in B) = i \in A$$

### 3.5 Spatial Distribution Diagnostics
To detect degenerate spatial clustering (e.g. all matches isolated in one corner):
1. **Grid Occupancy Ratio**:
   $$\text{Occupancy} = \frac{\text{Number of occupied cells in an } 8 \times 8 \text{ grid}}{64} \in [0.0, 1.0]$$
2. **Convex Hull Area Fraction**:
   $$\text{HullFraction} = \frac{\text{Area}(\text{ConvexHull}(\{\mathbf{p}_i\}))}{\text{width} \times \text{height}} \in [0.0, 1.0]$$
3. **Spatial Shannon Entropy**:
   $$H_{\text{spatial}} = -\frac{1}{\log_2(K)} \sum_{k=1}^K p_k \log_2(p_k + \epsilon) \in [0.0, 1.0]$$
   where $p_k$ is the empirical match point probability in grid bin $k$.

---

## 4. Failure Taxonomy

Non-silent explicit failure codes defined in `FeatureFailureReason`:
- **`NO_FEATURES_DETECTED`**: 0 keypoints found in the input frame (e.g. blank/textureless frame).
- **`INSUFFICIENT_FEATURES`**: Detected keypoints $< 100$ (`HEURISTIC_DEFAULT`). Status marked as `DEGRADED`.
- **`DESCRIPTOR_EXTRACTION_FAILED`**: Descriptor computation error on detected keypoints.
- **`NO_CANDIDATE_MATCHES`**: 0 matches survived Lowe ratio and mutual consistency filters.
- **`INSUFFICIENT_DESCRIPTOR_MATCHES`**: Accepted matches $< 30$ (`HEURISTIC_DEFAULT`). Status marked as `DEGRADED`.
- **`INVALID_IMAGE`**: Input image is not a `uint8` array of shape `(H, W, 3)`.
- **`UNSUPPORTED_FEATURE_CONFIGURATION`**: Unknown detector or matcher configuration.

---

## 5. Phase 3A Output Contract for Phase 3B

`FeatureMatchResult.to_correspondences()` produces a typed `FeatureCorrespondences` object ready for Phase 3B two-view epipolar geometry:
- `frame_a_id: str`
- `frame_b_id: str`
- `points_a: np.ndarray` (Shape $M \times 2$, `float64` pixel coordinates)
- `points_b: np.ndarray` (Shape $M \times 2$, `float64` pixel coordinates)
- `descriptor_distances: np.ndarray` (Shape $M$, `float64` Hamming distances)
- `match_count: int` ($M$)
- `descriptor_type: str` (e.g. `"ORB_256BIT_RATIO_AND_MUTUAL"`)
- `provenance: Dict[str, Any]`

---

## 6. Limitations & Explicit Guardrails

1. **Scale Pyramid Parameters vs. Robustness Claims**:
   - `scale_factor` is an internal ORB image-pyramid configuration parameter.
   - `n_levels` controls the discrete number of pyramid levels.
   - These parameters do NOT establish a universal maximum scale-change or viewpoint-change tolerance.
   - Actual matching robustness must be empirically evaluated on representative image transformations and UAV datasets.
2. **UAV Performance Not Yet Established**:
   - Descriptor matching performance on real UAV imagery has not yet been established by the current unit/integration tests.
   - The synthetic unit tests confirm descriptor-level pipeline mechanics only.
3. **No Unwarranted Generalization Claims**:
   - No claims of real-flight UAV accuracy, real-flight robustness, universal scale invariance, or geometric correctness are made.
   - All candidate descriptor matches remain strictly unverified until Phase 3B epipolar geometric estimation and RANSAC filtering.
