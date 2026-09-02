# Temporal Synchronization Engine Architecture & Scientific Semantics

## 1. Executive Summary & Objective

The Temporal Synchronization subsystem establishes a mathematically rigorous, datum-aware temporal and spatial association between video presentation timestamps (`CanonicalTimeline`) and continuous/discrete aerial spatial observations (`CanonicalTelemetryStream`).

> **CORE SCIENTIFIC PRINCIPLE:**
> - **Interpolation interval is NOT equivalent to propagated temporal uncertainty.**
> - Latitude and Longitude are NOT generic flat Cartesian coordinates. Spatial interpolation operates in 3D Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates to maintain metric consistency and handle antimeridian wrap-around.
> - Vertical datums (Ellipsoidal vs. MSL vs. Takeoff-Relative) must NOT be combined without explicit geoid undulation models.

---

## 2. Separation of Temporal Quantities & Uncertainties

To prevent misleading uncertainty claims, the observation data contract explicitly separates diagnostic temporal metrics from statistical uncertainties:

| Property | Symbol | Data Type | Physical Meaning |
| :--- | :--- | :--- | :--- |
| **`video_timestamp_seconds`** | $t_{\text{video}}$ | `float` (s) | True presentation timestamp of the video frame from stream origin |
| **`telemetry_timestamp_seconds`** | $t_{\text{telemetry}}$ | `Optional[float]` (s) | Target timestamp in the telemetry timebase ($t_{\text{telemetry}} = f(t_{\text{video}})$) |
| **`interpolation_fraction`** | $\alpha$ | `Optional[float]` | Normalized temporal position parameter $\alpha = \frac{t_{\text{target}} - t_0}{t_1 - t_0} \in [0.0, 1.0]$ |
| **`bracketing_interval_seconds`** | $\Delta t_{\text{gap}}$ | `Optional[float]` (s) | Diagnostic temporal separation between bounding telemetry records ($t_1 - t_0$) |
| **`clock_offset_seconds`** | $\Delta t_{\text{offset}}$ | `float` (s) | Scalar time bias applied between video and telemetry clocks |
| **`clock_offset_status`** | — | `ClockOffsetStatus` | Provenance of clock offset: `IDENTITY`, `KNOWN_APPLIED`, `ESTIMATED`, `VALIDATED` |
| **`timebase_uncertainty_seconds`** | $\sigma_t$ | `Optional[float]` (s) | Measured or declared clock jitter/uncertainty (independent of sample spacing) |

> **IMPORTANT:**
> Diagnostic half-interval ($\frac{\Delta t_{\text{gap}}}{2}$) is not a measurement uncertainty and must not be used as a substitute for an empirical sensor noise covariance.

---

## 3. Metric Spatial Interpolation (WGS84 $\to$ ECEF $\to$ WGS84)

Geodetic coordinates $(\phi, \lambda, h)$ on the curved Earth ellipsoid cannot be interpolated via linear 2D algebra without introducing distortion, chord errors, and antimeridian singularities.

### Conversion to Earth-Centered Earth-Fixed (ECEF):
Using standard WGS84 ellipsoid parameters ($a = 6378137.0\text{ m}, e^2 \approx 0.00669437999014$):
$$N(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2\phi}}$$
$$X = (N(\phi) + h) \cos\phi \cos\lambda$$
$$Y = (N(\phi) + h) \cos\phi \sin\lambda$$
$$Z = (N(\phi)(1 - e^2) + h) \sin\phi$$

### Metric Interpolation in $\mathbb{R}^3$:
$$\mathbf{X}(\alpha) = \mathbf{X}_0 + \alpha (\mathbf{X}_1 - \mathbf{X}_0), \quad \alpha = \frac{t_{\text{target}} - t_0}{t_1 - t_0}$$

### Inverse Geodetic Mapping (Bowring Inversion):
$$\mathbf{X}(\alpha) \xrightarrow{\text{Bowring}} (\phi_{\text{interp}}, \lambda_{\text{interp}}, h_{\text{interp}})$$

### Antimeridian Wrap-Around Protection:
Trajectories traversing longitude $\pm 180.0^\circ$ (e.g. from $+179.9^\circ \to -179.9^\circ$) are naturally continuous in ECEF $\mathbb{R}^3$, preventing catastrophic interpolation through the $0.0^\circ$ Prime Meridian.

---

## 4. Vertical Datum & Altitude Compatibility Policy

Interpolating altitudes with disparate vertical reference frames produces corrupted geometric scale:
- `ELLIPSOIDAL` $\leftrightarrow$ `ELLIPSOIDAL`: **Compatible** (Interpolated in ECEF).
- `MSL` $\leftrightarrow$ `MSL`: **Compatible** (Interpolated in ECEF).
- `RELATIVE_TO_TAKEOFF` $\leftrightarrow$ `RELATIVE_TO_TAKEOFF`: **Compatible**.
- `ELLIPSOIDAL` $\leftrightarrow$ `MSL`: **INCOMPATIBLE**. Interpolation is refused; frame status is marked `SyncStatus.INCOMPATIBLE_REFERENCE` with `position = None`.
- `ELLIPSOIDAL` $\leftrightarrow$ `UNKNOWN`: **INCOMPATIBLE**.

---

## 5. Clock Offset Semantics

A configured constant offset is not empirically validated unless ground truth evidence exists. The status reflects its true provenance:
- **`SyncStatus.OFFSET_APPLIED`**: A known/configured offset $\Delta t_{\text{offset}}$ was applied via `ConstantOffsetClockModel`.
- **`SyncStatus.OFFSET_ESTIMATED`**: Estimated via visual-inertial correlation (Phase 1C/1D).
- **`SyncStatus.OFFSET_VALIDATED`**: Empirically verified against an external hardware synchronization pulse.

---

## 6. Known Scientific Limitations

1. **Lever-Arm Displacement**: Spatial coordinates represent the GNSS antenna phase center. Translation to camera projection center via $\mathbf{p}_{\text{cam}} = \mathbf{p}_{\text{GNSS}} + \mathbf{R} \mathbf{t}_{\text{lever}}$ is handled in downstream pose modules.
2. **Propagated Uncertainty Budget**: Complete covariance propagation (combining GNSS DOP, lever-arm variance, interpolation variance, and clock jitter) is formally conducted in Stage 13 & 14 (Uncertainty Quantification) rather than approximated in the ingestion layer.
