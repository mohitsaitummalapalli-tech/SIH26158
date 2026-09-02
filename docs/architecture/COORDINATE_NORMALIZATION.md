# Coordinate Normalization Architecture: WGS84 → ECEF → Local Topocentric ENU

## 1. Executive Summary & Objective

The Coordinate Normalization subsystem provides a mathematically rigorous transformation pipeline converting global geodetic navigation coordinates into metric 3D Euclidean coordinates suitable for downstream computer vision and photogrammetric bundle adjustment.

> **CRITICAL SCIENTIFIC DISTINCTION:**
> - **Coordinate Normalization $\neq$ Georeferencing**: This subsystem performs coordinate normalization (datum validation and local metric Euclidean framing). Georeferencing occurs downstream when reconstructed 3D models are aligned to global coordinates via 7-DoF Sim(3) transformations.
> - **Coordinate Conversion Accuracy $\neq$ Sensor Accuracy $\neq$ Reconstruction Accuracy**: The numerical round-trip precision of these algorithms reflects the mathematical stability of the closed-form implementation. It does **not** represent claims about physical GNSS receiver precision, drone navigation accuracy, or final 3D reconstruction quality.

---

## 2. Authoritative WGS84 Constants & Reference Ellipsoid

All calculations utilize the National Geospatial-Intelligence Agency (NGA) / Department of Defense WGS 84 Reference System (NIMA TR8350.2, 3rd Edition):

| Parameter | Symbol | Value | Source / Standard |
| :--- | :--- | :--- | :--- |
| **Semi-Major Axis** | $a$ | $6378137.0\text{ m}$ | Defined constant |
| **Reciprocal Flattening** | $1/f$ | $298.257223563$ | Defined constant |
| **Flattening** | $f$ | $1 / 298.257223563 \approx 0.003352810664747$ | Derived: $f = 1 - (b/a)$ |
| **Semi-Minor Axis** | $b$ | $6356752.314245179\text{ m}$ | Derived: $b = a(1 - f)$ |
| **First Eccentricity Squared** | $e^2$ | $0.006694379990141316$ | Derived: $e^2 = 2f - f^2 = \frac{a^2 - b^2}{a^2}$ |
| **Second Eccentricity Squared** | $e'^2$ | $0.006739496742276434$ | Derived: $e'^2 = \frac{e^2}{1 - e^2} = \frac{a^2 - b^2}{b^2}$ |

---

## 3. Coordinate Frame Definitions & Conventions

```
          Global WGS84 ECEF Frame                    Local Topocentric ENU Frame
               (EPSG:4978)                                  (Metric Tangent)
                   +Z (Pole)                                     +U (Up)
                   ▲                                             ▲
                   │                                             │
                   │                                             │
                   │                                             │
                   └──────────► +Y (90°E)                        └──────────► +E (East)
                  /                                             /
                 /                                             /
                ▼                                             ▼
               +X (Prime Meridian)                           +N (North)
```

### 1. Global ECEF Frame (`EPSG:4978` Cartesian $\mathbb{R}^3$)
- **Origin**: Earth center of mass $(0, 0, 0)$.
- **$+X$ Axis**: Equatorial axis passing through the intersection of the Equator and the WGS84 Reference Meridian ($0^\circ$ latitude, $0^\circ$ longitude).
- **$+Y$ Axis**: Equatorial axis completing the right-handed orthogonal system, passing through $0^\circ$ latitude and $90^\circ\text{ E}$ longitude.
- **$+Z$ Axis**: Aligned with the WGS84 Terrestrial Reference Pole (conventional terrestrial north axis).

### 2. Local Topocentric ENU Frame (Metric Euclidean $\mathbb{R}^3$)
- **Origin**: Anchor geodetic position $(\phi_0, \lambda_0, h_0)$ with corresponding ECEF anchor $\mathbf{X}_0 = (X_0, Y_0, Z_0)^T$.
- **$+E$ (East)**: Tangent to the parallel of latitude in the direction of increasing longitude.
- **$+N$ (North)**: Tangent to the meridian in the direction of increasing latitude (True North).
- **$+U$ (Up)**: Normal to the WGS84 reference ellipsoid, pointing outward away from Earth center.
- **Handedness**: Right-handed orthonormal basis ($\mathbf{e} \times \mathbf{n} = \mathbf{u}, \det(\mathbf{R}) = +1$).

---

## 4. Mathematical Transformation Formulations

### 1. WGS84 Geodetic $(\phi, \lambda, h) \to$ ECEF $(X, Y, Z)$
Given latitude $\phi \in [-\frac{\pi}{2}, \frac{\pi}{2}]$, longitude $\lambda \in [-\pi, \pi]$ (converted from degrees to radians), and ellipsoidal height $h \in \mathbb{R}$:

1. **Prime Vertical Radius of Curvature**:
   $$N(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2\phi}}$$

2. **Cartesian Coordinates in $\mathbb{R}^3$**:
   $$X = (N(\phi) + h) \cos\phi \cos\lambda$$
   $$Y = (N(\phi) + h) \cos\phi \sin\lambda$$
   $$Z = (N(\phi)(1 - e^2) + h) \sin\phi$$

### 2. ECEF $(X, Y, Z) \to$ WGS84 Geodetic $(\phi, \lambda, h)$ (Bowring Inversion)
1. Equatorial distance: $p = \sqrt{X^2 + Y^2}$
2. Parametric latitude: $\theta = \text{atan2}(Z \cdot a, p \cdot b)$
3. Geodetic latitude:
   $$\phi = \text{atan2}\left(Z + e'^2 \cdot b \sin^3\theta, \; p - e^2 \cdot a \cos^3\theta\right)$$
4. Longitude: $\lambda = \text{atan2}(Y, X)$
5. Ellipsoidal Height:
   $$h = \begin{cases} \frac{p}{\cos\phi} - N(\phi) & \text{if } |\cos\phi| > 10^{-4} \\ \frac{Z}{\sin\phi} - N(\phi)(1 - e^2) & \text{if } |\cos\phi| \le 10^{-4} \end{cases}$$

### 3. ECEF $\to$ Local Topocentric ENU $(e, n, u)$
$$\Delta \mathbf{X} = \begin{bmatrix} X - X_0 \\ Y - Y_0 \\ Z - Z_0 \end{bmatrix}$$

$$\begin{bmatrix} e \\ n \\ u \end{bmatrix} = \mathbf{R}_{\text{ECEF}\to\text{ENU}} \Delta \mathbf{X} = \begin{bmatrix} -\sin\lambda_0 & \cos\lambda_0 & 0 \\ -\sin\phi_0 \cos\lambda_0 & -\sin\phi_0 \sin\lambda_0 & \cos\phi_0 \\ \cos\phi_0 \cos\lambda_0 & \cos\phi_0 \sin\lambda_0 & \sin\phi_0 \end{bmatrix} \begin{bmatrix} X - X_0 \\ Y - Y_0 \\ Z - Z_0 \end{bmatrix}$$

---

## 5. Origin Selection Strategies (`OriginPolicy`)

| Policy | Description | Use Case |
| :--- | :--- | :--- |
| **`FIRST_VALID_POSITION`** | Uses the first valid ellipsoidal observation in the stream | Standard linear drone flight ingestion |
| **`MEDIAN_POSITION`** | Converts all valid positions to ECEF, calculates metric 3D median, and converts back to geodetic anchor | Complex / loitering / non-linear multi-rotor surveys |
| **`EXPLICIT_ORIGIN`** | User-configured anchor origin | Fixed ground-control base stations / multi-flight harmonization |

---

## 6. Numerical Precision & Integrity Statements

- **Mathematical Round-Trip Precision**:
  - Angular round-trip tolerance: $|\Delta\phi|, |\Delta\lambda| < 1 \times 10^{-7\circ}$ ($< 1.1\text{ cm}$).
  - Height round-trip tolerance: $|\Delta h| < 1.0\text{ mm}$.
  - Basis vector orthogonality: $|\mathbf{e}\cdot\mathbf{n}|, |\mathbf{n}\cdot\mathbf{u}|, |\mathbf{e}\cdot\mathbf{u}| < 1 \times 10^{-7}$.
- **Important Qualification**:
  These tolerances document the mathematical conditioning and numerical convergence of the algorithm implementation. They are **not** empirical claims regarding GNSS receiver accuracy, flight path accuracy, or photogrammetric 3D model accuracy.
