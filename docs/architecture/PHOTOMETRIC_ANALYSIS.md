# Photometric & Illumination Stability Diagnostics Architecture

## 1. Executive Summary & Objective

The Photometric and Illumination Stability Diagnostic Subsystem provides deterministic, explainable measurements of spatial illumination distributions, dynamic range, clipping fractions, and frame-to-frame photometric transitions across the canonical video timeline.

---

## 2. Scientific Interpretation Boundaries

> **CRITICAL SCIENTIFIC PRINCIPLES:**
> 1. **IMAGE LUMINANCE $\neq$ PHYSICAL SCENE ILLUMINATION**:
>    - Decoded pixel luminance reflects quantized sensor digital numbers (DN) after analog gain, ADC, and ISP tone mapping.
>    - **It does NOT measure physical scene radiance ($\text{W}/(\text{sr}\cdot\text{m}^2)$) or ground illuminance ($\text{lux}$).**
>
> 2. **HISTOGRAM DISTANCE $\neq$ DEFINITIVE EXPOSURE CHANGE**:
>    - Bhattacharyya distance measures luma distribution divergence between discrete frames.
>    - **It can be caused by camera auto-exposure, scene content change, cloud transit, or orientation changes.**
>
> 3. **LOCALIZED BRIGHTNESS VARIATION $\neq$ DEFINITIVE SHADOW / GLARE SEGMENTATION**:
>    - Spatial tile variances indicate regional photometric heterogeneity.
>    - **They do NOT constitute semantic shadow or specular highlight segmentation without physical surface albedo estimation.**
>
> 4. **COLOR STATISTICS $\neq$ COLORIMETRIC CALIBRATION**:
>    - Per-channel RGB statistics describe image appearance.
>    - **They do NOT represent white-balance calibration or spectral reflectance.**
>
> 5. **DIAGNOSTICS ONLY**:
>    - This subsystem produces diagnostic reports only. It does **NOT** modify frame buffers or filter frames.

---

## 3. Pipeline Integration

```
                         DecodedFrame (t-1, t)
                                   │
                                   ▼
                         PhotometricAnalyzer
                                   │
                 ┌─────────────────┼─────────────────┐
                 ▼                 ▼                 ▼
          Extended Luma     Spatial 3x3 Grid    Luma Histogram
          (p1..p99, Range)   (Patterns/Tiles)    (Bhattacharyya)
                 │                 │                 │
                 └─────────────────┼─────────────────┘
                                   ▼
                      PhotometricStabilityReport
                     (Structured JSON Diagnostic)
```

---

## 4. Mathematical Formulations & Method

### 1. Canonical Luminance Conversion (ITU-R BT.601)
$$Y(x, y) = 0.299 \cdot R(x, y) + 0.587 \cdot G(x, y) + 0.114 \cdot B(x, y) \in [0.0, 255.0]$$

### 2. Normalized Luminance Histogram
For $K$ bins across range $[0.0, 255.0]$:
$$H_k = \frac{1}{N} \sum_{x, y} \mathbf{1}_{Y(x, y) \in \text{bin}_k}, \quad \sum_{k=1}^K H_k = 1.0$$

### 3. Classical Bhattacharyya Histogram Distance
Given normalized histograms $P$ and $Q$ of target and neighbor frames:
$$\text{BC}(P, Q) = \sum_{k=1}^K \sqrt{P_k \cdot Q_k} \in [0.0, 1.0]$$
$$D_B(P, Q) = -\ln(\text{BC}(P, Q)) \ge 0.0$$
- $D_B(P, Q) = 0.0$: Identical luminance distributions.
- $D_B(P, Q) \ge 0.15$: Significant distribution shift.

### 4. Spatial Illumination Grid ($M \times N$)
Subdivides the frame into regular spatial tiles (default: $3 \times 3$). Computes mean, median, percentile spread ($p_{90} - p_{10}$), and clipping fractions per tile.
- **`UNIFORM`**: Tile mean standard deviation $\sigma_{\text{tiles}} < \tau_{\text{uniform}}$ ($8.0$).
- **`GRADIENT`**: Monotonic cross-tile illumination gradient ($\Delta Y_{\text{axis}} > 12.0$).
- **`LOCALIZED_BRIGHTNESS`**: Center/edge tile mean significantly exceeds median ($\ge 1.5 \times \text{median}$).
- **`LOCALIZED_DARKNESS`**: Tile mean significantly below median ($\le 0.5 \times \text{median}$).
- **`MIXED`**: Complex multi-modal spatial variation.
- **`INSUFFICIENT_EVIDENCE`**: Non-finite or corrupted image data.

### 5. Frame-to-Frame Temporal Photometric Change
When a valid temporal neighbor frame is provided ($\Delta t \le \tau_{\Delta t}$):
- $\Delta \mu_Y = \mu_{\text{target}} - \mu_{\text{neighbor}}$
- $\Delta \tilde{Y} = \tilde{Y}_{\text{target}} - \tilde{Y}_{\text{neighbor}}$
- Categorization:
  - **`STABLE`**: $D_B < 0.15$ and $|\Delta \mu_Y| < 5.0$.
  - **`POTENTIAL_EXPOSURE_TRANSITION`**: Large global shift ($|\Delta \mu_Y| \ge 15.0$ or $D_B \ge 0.15$).
  - **`POTENTIAL_LOCAL_ILLUMINATION_CHANGE`**: High histogram distance driven by localized spatial spot.
  - **`MIXED_CHANGE`**: Intermediate photometric variation.

---

## 5. Configurable Heuristic Thresholds (`HEURISTIC_DEFAULT`)

| Parameter | Default Value | Status | Description |
| :--- | :--- | :--- | :--- |
| `histogram_bins` | `64` | `HEURISTIC_DEFAULT` | Number of bins for 1D luma histogram. |
| `tile_grid_rows` | `3` | `HEURISTIC_DEFAULT` | Spatial grid rows. |
| `tile_grid_cols` | `3` | `HEURISTIC_DEFAULT` | Spatial grid columns. |
| `shadow_threshold` | `5.0` | `HEURISTIC_DEFAULT` | Luminance value below which pixels are marked clipped shadows. |
| `highlight_threshold` | `250.0` | `HEURISTIC_DEFAULT` | Luminance value above which pixels are marked clipped highlights. |
| `max_temporal_gap_seconds` | `2.0 s` | `HEURISTIC_DEFAULT` | Maximum allowable inter-frame time delta for comparison. |
| `spatial_uniformity_std_thresh` | `8.0` | `HEURISTIC_DEFAULT` | Standard deviation threshold across tile means for uniform pattern. |
| `bhattacharyya_change_threshold`| `0.15` | `HEURISTIC_DEFAULT` | Histogram distance threshold for exposure transitions. |

---

## 6. Immutability & Provenance

- **Immutability**: DecodedFrame data arrays are read-only and never modified (no gamma, no stretching, no white-balance).
- **Provenance**: Records target frame ID, neighbor frame ID, timestamps, time deltas, analysis dimensions, and histogram bin configurations.
