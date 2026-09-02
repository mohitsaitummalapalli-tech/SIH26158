# Temporal Motion & Blur Diagnostic Subsystem Architecture

## 1. Executive Summary & Objective

The Temporal Motion and Blur Diagnostic Subsystem evaluates deterministic, sequence-aware motion kinematics and heuristic blur indicators across consecutive `DecodedFrame` sequences from the canonical timeline.

---

## 2. Scientific Interpretation Boundaries

> **CRITICAL SCIENTIFIC PRINCIPLE:**
> The mathematical concepts in this subsystem must be strictly separated and never confounded:
>
> 1. **INTER-FRAME APPARENT MOTION ($\mathbf{v} \text{ in px/s}$)**:
>    - The 2D optical displacement between discrete video presentation frames.
>    - **It is NOT 6-DoF camera pose estimation or visual odometry.**
>
> 2. **SINGLE-FRAME OPTICAL SHARPNESS ($S_{\text{Laplacian}}$)**:
>    - High-frequency spatial edge energy evaluated on a single frame.
>    - **It is an image statistic, not a guarantee of geometric reconstruction fidelity.**
>
> 3. **INTRA-EXPOSURE MOTION BLUR**:
>    - Optical degradation accumulated on sensor photosites during the physical shutter opening time ($t_{\text{shutter}}$).
>    - **`motion_blur_indicator` is a heuristic diagnostic proxy combining inter-frame velocity with relative sharpness drop; it is NOT a physical measurement of intra-exposure blur or a calibrated probability.**
>
> 4. **SPATIAL MOTION HETEROGENEITY**:
>    - Variance in displacement across spatial tiles.
>    - **It is a pattern diagnostic indicator, NOT definitive dynamic object segmentation.**
>
> 5. **DIAGNOSTICS ONLY**:
>    - This subsystem produces diagnostic reports. It does **NOT** drop or reject frames at this stage.

---

## 3. Pipeline Integration

```
                         DecodedFrame (t-1, t, t+1)
                                     │
                                     ▼
                           TemporalMotionAnalyzer
                                     │
                 ┌───────────────────┼───────────────────┐
                 ▼                   ▼                   ▼
         Dense Optical Flow    Velocity Scaling     Spatial Grid
        (Farnebäck Alg.)         (Δt Normalized)       (3x3 Tiles)
                 │                   │                   │
                 └───────────────────┼───────────────────┘
                                     ▼
                        TemporalMotionBlurReport
                      (Structured JSON Diagnostic)
```

---

## 4. Mathematical Formulations & Method

### 1. Dense Displacement Estimation (Farnebäck Optical Flow)
Given adjacent grayscale images $I_1(x, y)$ and $I_2(x, y)$, quadratic polynomial expansion approximates local neighborhoods:
$$f_1(\mathbf{x}) \sim \mathbf{x}^T \mathbf{A}_1 \mathbf{x} + \mathbf{b}_1^T \mathbf{x} + c_1$$
$$f_2(\mathbf{x}) = f_1(\mathbf{x} - \mathbf{d}) \sim \mathbf{x}^T \mathbf{A}_2 \mathbf{x} + \mathbf{b}_2^T \mathbf{x} + c_2$$
Solving for the displacement vector field $\mathbf{d}(x, y) = (u(x, y), v(x, y))^T$ minimizes the photometric residual across multi-scale spatial pyramids.

### 2. Timestamp-Based Velocity Normalization
For adjacent frames with presentation timestamps $t_1$ and $t_2$ ($\Delta t = t_2 - t_1 > 0$):
$$\mathbf{v}(x, y) = \frac{\mathbf{d}(x, y)}{\Delta t} = \left( \frac{u(x, y)}{\Delta t}, \frac{v(x, y)}{\Delta t} \right) \quad [\text{px/s}]$$
All temporal calculations use true canonical timestamps ($t_{\text{PTS}}$), never naive frame index differences.

### 3. Directional Coherence Score
Measures the spatial alignment of displacement unit vectors across the image:
$$\mathbf{u}_{\text{unit}}(x, y) = \frac{\mathbf{d}(x, y)}{\|\mathbf{d}(x, y)\|} \quad (\text{for } \|\mathbf{d}\| > 0.1)$$
$$\mathcal{S}_{\text{coherence}} = \left\| \frac{1}{N} \sum_{x, y} \mathbf{u}_{\text{unit}}(x, y) \right\| \in [0.0, 1.0]$$

### 4. Spatial Grid Tiling ($M \times N$)
Subdivides the frame into regular spatial tiles (default: $3 \times 3$). Evaluates localized mean displacement, velocity, and variance to identify regional anomalies:
- **`POTENTIAL_CAMERA_MOTION`**: Global coherent motion ($\mathcal{S}_{\text{coherence}} \ge \tau_{\text{coherence}}$) without isolated outlier tiles.
- **`POTENTIAL_LOCAL_MOTION`**: Isolated tile displacement significantly exceeds the global median ($\text{Tile}_{\text{disp}} > \tau_{\text{local}} \times \max(\text{Median}_{\text{global}}, \tau_{\text{disp}})$).
- **`MIXED_MOTION`**: Combination of dominant camera motion and regional variations.
- **`LOW_APPARENT_MOTION`**: Low flow displacement in textured scenes ($\|\mathbf{d}\| < \tau_{\text{disp}}$).
- **`INSUFFICIENT_EVIDENCE`**: Flat / low-texture scene, missing adjacent frames, or large temporal gap ($> 2.0\text{ s}$).

### 5. Relative Sharpness & Motion Blur Diagnostic Indicator
To prevent reliance on uncalibrated universal absolute thresholds, sharpness is evaluated relative to neighboring frames:
$$\Delta S_{\text{rel}} = \max\left(0.0, 1.0 - \frac{S_{\text{target}}}{\max(\bar{S}_{\text{neighbor}}, 10^{-4})}\right)$$
The `motion_blur_indicator` index is computed as:
$$\text{Indicator}_{\text{motion\_blur}} = \min\left(1.0, \frac{\|\mathbf{v}\|}{100.0}\right) \times \Delta S_{\text{rel}} \in [0.0, 1.0]$$
- High inter-frame velocity + relative sharpness drop $\to$ Elevated motion blur indicator.
- High inter-frame velocity + pristine edge preservation $\to$ Low motion blur indicator (fast shutter speed).
- Low inter-frame velocity + degraded sharpness $\to$ Static defocus or low-texture terrain.

---

## 5. Configurable Heuristic Thresholds (`HEURISTIC_DEFAULT`)

| Parameter | Default Value | Status | Description |
| :--- | :--- | :--- | :--- |
| `coherence_threshold` | `0.70` | `HEURISTIC_DEFAULT` | Directional alignment threshold for coherent camera motion. |
| `local_motion_ratio_threshold` | `2.0` | `HEURISTIC_DEFAULT` | Outlier tile ratio to flag localized motion heterogeneity. |
| `displacement_lower_bound` | `0.5 px` | `HEURISTIC_DEFAULT` | Minimum displacement to distinguish motion from sensor noise. |
| `low_texture_variance_threshold`| `10.0` | `HEURISTIC_DEFAULT` | Luma variance threshold for low-texture flat surfaces. |
| `max_temporal_gap_seconds` | `2.0 s` | `HEURISTIC_DEFAULT` | Maximum allowable time delta between adjacent frames. |

> **CALIBRATION NOTICE:** All default thresholds are provisional heuristic baselines. They must be empirically calibrated against real drone flight datasets with ground-truth trajectory and shutter timing data.

---

## 6. Exposure Metadata Integration (Future Roadmap)

- **Shutter Duration ($t_{\text{exposure}}$)**: When available from camera EXIF / XMP metadata, exposure duration will be coupled with inter-frame velocity to derive physical intra-exposure smear bounds ($\text{Smear}_{\text{px}} \approx \|\mathbf{v}\| \times t_{\text{exposure}}$).
- **Current State**: Fields are defined in `TemporalMotionBlurReport.future_exposure_metadata` without fabricating synthetic values.

---

## 7. Edge Case Handling

1. **First / Last Frame**: Analyzed using forward-only or backward-only adjacent frames.
2. **Unequal Sampling ($\Delta t_1 \neq \Delta t_2$)**: Correctly normalized by each independent interval.
3. **Large Temporal Gaps**: Gaps exceeding `max_temporal_gap_seconds` ($2.0\text{ s}$) are excluded from velocity computation.
4. **Corrupted Neighbor**: Handled gracefully without crashing the analysis pipeline.
5. **Low Texture / Sky**: Explicitly tagged with `low_texture_indicator = True` and categorized as `INSUFFICIENT_EVIDENCE`.

---

## 8. Immutability & Provenance

- **Immutability**: Input frame pixel buffers are never modified.
- **Provenance**: Every report records `target_frame_id`, `neighbor_frame_ids`, `time_deltas_seconds`, `analysis_method`, and all heuristic thresholds applied.
