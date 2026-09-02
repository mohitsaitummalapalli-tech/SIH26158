# Frame Quality Assessment Architecture & Statistical Contracts

## 1. Executive Summary & Objective

The Frame Quality Assessment subsystem evaluates deterministic, explainable mathematical image statistics on in-memory `DecodedFrame` objects without mutating the underlying image data.

> **CRITICAL SCIENTIFIC INTEGRITY RULE:**
> The statistical metrics computed in this subsystem (Laplacian variance, luma distributions, clipping fractions, Tenengrad energy) are **DIAGNOSTIC IMAGE PROXIES**. They do **NOT** represent ground-truth motion blur measurements, illumination correctness, or guaranteed 3D reconstruction accuracy. They are **NOT** used to drop frames in this phase. Frame filtering and keyframe selection are formally handled in Stage 4 / Phase 2C.

---

## 2. Subsystem Architecture

```
                       DecodedFrame (Canonical RGB uint8)
                                     │
                                     ▼
                           FrameQualityAnalyzer
                                     │
                 ┌───────────────────┼───────────────────┐
                 ▼                   ▼                   ▼
        Luminance & Clipping      Contrast           Sharpness
          (ITU-R BT.601)      (Percentiles/Std)   (Laplacian/Sobel)
                 │                   │                   │
                 └───────────────────┼───────────────────┘
                                     ▼
                            Spatial Grid Tiling
                               (3x3 Sub-Grid)
                                     │
                                     ▼
                             FrameQualityReport
                      (Structured JSON Diagnostic)
```

---

## 3. Mathematical Definitions & Statistical Formulations

### 1. Luminance Conversion (ITU-R BT.601)
Every canonical RGB frame ($\text{dtype}=\text{uint8}$, values in $[0, 255]$) is converted to a floating-point scalar luminance matrix $Y$:
$$Y(x, y) = 0.299 \cdot R(x, y) + 0.587 \cdot G(x, y) + 0.114 \cdot B(x, y)$$
- **Range**: $[0.0, 255.0]$
- **Statistics**: Mean ($\mu_Y$), Median ($\tilde{Y}$), Standard Deviation ($\sigma_Y$), and Percentiles ($p_5, p_{25}, p_{75}, p_{95}$).

### 2. Clipping & Saturation Fractions
Measures extreme values where dynamic range is compressed or lost:
- **Shadow Clipping Fraction ($f_{\text{shadow}}$)**:
  $$f_{\text{shadow}} = \frac{1}{N} \sum_{x, y} \mathbf{1}_{\{Y(x, y) \le \tau_{\text{shadow}}\}}$$
  Default threshold: $\tau_{\text{shadow}} = 5.0$.
- **Highlight Clipping Fraction ($f_{\text{highlight}}$)**:
  $$f_{\text{highlight}} = \frac{1}{N} \sum_{x, y} \mathbf{1}_{\{Y(x, y) \ge \tau_{\text{highlight}}\}}$$
  Default threshold: $\tau_{\text{highlight}} = 250.0$.
- **Channel Clipping**: Evaluated separately per channel ($R, G, B$) to identify single-channel saturation (e.g. blue sky, red glare).

### 3. Contrast Indicators
- **Luminance Standard Deviation**: $\sigma_Y = \sqrt{\frac{1}{N}\sum (Y(x, y) - \mu_Y)^2}$
- **Percentile Spread (90–10)**: $\Delta p_{90, 10} = p_{90} - p_{10}$
- **Michelson Contrast Proxy**: $\mathcal{C}_M = \frac{p_{95} - p_{5}}{p_{95} + p_{5} + \epsilon}$

### 4. Sharpness & Edge Energy Statistics
- **Variance of Laplacian ($\text{Var}(\nabla^2 Y)$)**:
  $$\nabla^2 Y = \frac{\partial^2 Y}{\partial x^2} + \frac{\partial^2 Y}{\partial y^2}$$
  $$\text{Var}(\nabla^2 Y) = \frac{1}{N} \sum_{x, y} \left( \nabla^2 Y(x, y) - \overline{\nabla^2 Y} \right)^2$$
- **Tenengrad Sobel Gradient Energy**:
  $$\text{Energy}_{\text{Tenengrad}} = \frac{1}{N} \sum_{x, y} \left( G_x(x, y)^2 + G_y(x, y)^2 \right)$$
  where $G_x = S_x * Y$ and $G_y = S_y * Y$ using $3 \times 3$ Sobel operators.
- **Modified Laplacian**:
  $$\text{ML}(x, y) = |2 Y(x, y) - Y(x - 1, y) - Y(x + 1, y)| + |2 Y(x, y) - Y(x, y - 1) - Y(x, y + 1)|$$

### 5. Spatial Grid Tiling
- Subdivides the image into a configurable $M \times N$ regular grid (default: $3 \times 3$ tiles).
- Evaluates $\mu_Y, \sigma_Y,$ and $\text{Var}(\nabla^2 Y)$ per tile to identify localized optical aberrations (e.g. vignetting, corner blur, lens flare).

### 6. High-Frequency Residual Indicator (Compression Proxy)
- Measures residual high-frequency energy $(Y - G_\sigma(Y))$ where $G_\sigma$ is a $5 \times 5$ Gaussian kernel ($\sigma = 1.0$).
- Conservative metric proxy for compression/blockiness artifacts without asserting codec metadata claims.

---

## 4. Quality Status Categories

| Status | Diagnostic Condition |
| :--- | :--- |
| **`VALID`** | Sharpness and clipping statistics fall within nominal operational parameters. |
| **`DEGRADED`** | Moderate blur ($\text{Var}(\nabla^2 Y) < \tau_{\text{degraded}}$) or moderate clipping ($\ge 20\%$ pixels clipped). |
| **`SEVERELY_DEGRADED`** | Severe blur ($\text{Var}(\nabla^2 Y) < \tau_{\text{severe}}$) or severe clipping ($\ge 40\%$ pixels clipped). |
| **`ANALYSIS_ERROR`** | Frame decode failure, image dimensions below minimum ($< 32 \times 32$), or non-finite values. |

---

## 5. Configuration & Provenance

### `QualityAssessmentConfig`:
- Fully typed, immutable configuration defining clipping thresholds, tile grid resolution, blur boundaries, and minimum dimension constraints.

### Provenance Tracking:
Every `FrameQualityReport` retains:
- `source_frame_id`, `source_frame_index`, `source_timestamp_seconds`
- `source_video`, `decoder_backend`
- `config_version` (`"QualityAssessment_v1.0"`)
- `analysis_dimensions` ($W \times H$)

---

## 6. Original Frame Preservation (Immutability)

- The analysis is **purely analytical** and strictly read-only.
- The input `DecodedFrame.data` buffer is never modified, sharpened, blurred, clipped, or tone-mapped.
