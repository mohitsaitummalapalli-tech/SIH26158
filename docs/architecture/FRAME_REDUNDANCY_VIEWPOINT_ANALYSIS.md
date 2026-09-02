# Frame Redundancy & Viewpoint Diversity Diagnostics Architecture

## 1. Executive Summary & Objective

The Frame Redundancy and Viewpoint Diversity Diagnostics subsystem quantifies the relative novelty and geometric contribution of individual video frames across the sequence timeline.

---

## 2. Scientific Interpretation Boundaries

> **CRITICAL SCIENTIFIC PRINCIPLES:**
> 1. **VISUAL SIMILARITY $\neq$ GEOMETRIC REDUNDANCY**:
>    - High appearance similarity can occur between frames from different viewpoints observing planar surfaces, while drastic illumination changes can obscure identical 3D geometry.
> 2. **FEATURE-MATCH COUNT $\neq$ RECONSTRUCTION ACCURACY**:
>    - A low match count may stem from smooth low-texture surfaces, specular highlights, or lighting transitions, **NOT** necessarily unsuitability for 3D reconstruction.
> 3. **GNSS TRAJECTORY BASELINE $\neq$ OPTICAL CAMERA BASELINE**:
>    - Without lever-arm offset calibration, ENU coordinates represent the aircraft navigation GNSS receiver position, **NOT** the camera optical center.
> 4. **AIRCRAFT ATTITUDE $\neq$ CAMERA VIEW ANGLE**:
>    - Without gimbal encoder angle calibration, aircraft orientation is a platform navigation proxy, **NOT** the true optical principal axis direction.
> 5. **DIAGNOSTIC EVIDENCE ONLY**:
>    - This subsystem produces diagnostic evidence reports (`FrameRedundancyReport`). It does **NOT** drop frames, rank usefulness, or make keyframe exclusion decisions at this stage.

---

## 3. Pipeline Integration

```
                         DynamicSceneReport
                                 │
                                 ▼
                 FrameRedundancyViewpointAnalyzer
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
     Visual ZNCC         ORB Feature Match      Trajectory / Quat
    (Similarity)         (Overlap & Hull)       (Baseline & Angle)
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 ▼
                       FrameRedundancyReport
                   (Structured JSON Diagnostic)
                                 │
                                 ▼
                    Phase 2G Keyframe Selection
```

---

## 4. Mathematical Formulations & Metrics

### 1. Temporal Separation
$$\Delta t = |t_b - t_a| \quad [\text{seconds}]$$
Always evaluated using canonical presentation timestamps ($t_{\text{PTS}}$), never naive frame indices.

### 2. Appearance Similarity (Normalized Zero-Mean Cross-Correlation)
$$\text{ZNCC}(I_a, I_b) = \frac{1}{HW \sigma_a \sigma_b} \sum_{x, y} (I_a(x, y) - \mu_a)(I_b(x, y) - \mu_b) \in [-1.0, 1.0]$$
$$\mathcal{S}_{\text{visual}} = \frac{1}{2}(\text{ZNCC} + 1.0) \in [0.0, 1.0]$$

### 3. Feature Overlap Ratio
Using classical ORB keypoint extraction and mutual cross-check Hamming matching:
$$\mathcal{R}_{\text{match}} = \frac{N_{\text{matches}}}{\min(N_{\text{kp}, a}, N_{\text{kp}, b})} \in [0.0, 1.0]$$

### 4. Spatial Match Distribution
- **Convex Hull Coverage Ratio**:
  $$\mathcal{C}_{\text{hull}} = \frac{\text{Area}(\text{ConvexHull}(\{\mathbf{x}_{\text{match}}\}))}{H \cdot W} \in [0.0, 1.0]$$
- **Grid Occupancy Ratio**:
  Fraction of $4 \times 4$ spatial grid cells containing at least one verified feature correspondence:
  $$\mathcal{O}_{\text{grid}} = \frac{|\{(r, c) \mid \exists \mathbf{x} \in \text{cell}_{r, c}\}|}{16} \in [0.0, 1.0]$$

### 5. Trajectory-Based Baseline ($\Delta \mathbf{x}_{\text{ENU}}$)
Given synchronized ENU positions $\mathbf{x}_a, \mathbf{x}_b$:
$$B_{\text{trajectory}} = \|\mathbf{x}_a - \mathbf{x}_b\|_{\text{ENU}} = \sqrt{(E_a - E_b)^2 + (N_a - N_b)^2 + (U_a - U_b)^2} \quad [\text{m}]$$

### 6. Trajectory Orientation Change ($\Delta \theta_{\text{att}}$)
Geodesic rotation angle between aircraft attitude unit quaternions $\mathbf{q}_a, \mathbf{q}_b$:
$$\Delta \theta_{\text{att}} = 2 \arccos(|\mathbf{q}_a \cdot \mathbf{q}_b|) \quad [\text{degrees}]$$

---

## 5. Computational Complexity & Windowing

Comparing every frame against all prior frames incurs $O(N^2)$ complexity. To ensure bounded memory and linear execution time on long video sequences, the analyzer implements a temporal sliding window strategy:
$$\text{Candidate Pairs for Frame } i = \{j \mid \max(0, i - W) \le j < i\}$$
- **Default Window Size**: $W = 5$ frames.
- **Computational Complexity**: $O(W \cdot N)$, operating in linear time with sequence duration.

---

## 6. Configurable Heuristic Thresholds (`HEURISTIC_DEFAULT`)

| Parameter | Default Value | Status | Description |
| :--- | :--- | :--- | :--- |
| `temporal_window_frames` | `5` | `HEURISTIC_DEFAULT` | Number of prior candidate frames evaluated per frame. |
| `orb_max_features` | `500` | `HEURISTIC_DEFAULT` | Maximum ORB feature keypoints extracted per frame. |
| `visual_similarity_threshold` | `0.90` | `HEURISTIC_DEFAULT` | Appearance correlation threshold for visual redundancy. |
| `match_ratio_redundancy_threshold` | `0.60` | `HEURISTIC_DEFAULT` | Feature match ratio threshold for novelty evaluation. |
| `trajectory_baseline_threshold_meters` | `1.0 m` | `HEURISTIC_DEFAULT` | Minimum spatial baseline to indicate physical displacement. |
| `orientation_change_threshold_degrees` | `2.0 deg` | `HEURISTIC_DEFAULT` | Minimum attitude change to indicate viewpoint rotation. |
| `temporal_redundancy_time_threshold_seconds` | `0.5 s` | `HEURISTIC_DEFAULT` | Inter-frame temporal proximity threshold. |

---

## 7. Provenance & Immutability

- **Immutability**: Source frame buffers are never modified.
- **Provenance**: Records target frame ID, reference frame IDs, timestamps, evaluated pair count, window size limit, and detector backend (`OpenCV_ORB_v1.0`).
