# Geospatial Accuracy, Coordinate Transformations & Metric Rigor

## 1. The 5-Stage Spatial Transformation Pipeline

A core scientific principle of SIH26158 is that **solving a 7-DoF $\text{Sim}(3)$ alignment does NOT automatically solve georeferencing**. 

Reconstruction from uncalibrated or relative AI pointmaps must progress through five distinct, verifiable transformations:

```
Stage 1: Relative Reconstruction
         Dense pointmap / point cloud in unscaled, arbitrary coordinate frame X_m
                           │
                           ▼
Stage 2: Metric Scale Recovery
         Estimation of metric scale factor s via GNSS baselines, barometric delta, or VIO
                           │
                           ▼
Stage 3: Similarity Alignment (Sim(3) to Local Euclidean Frame)
         Optimal 7-DoF alignment (s, R_align, t_align) into Local Topocentric ENU
         (East-North-Up tangent plane at local geodetic origin)
                           │
                           ▼
Stage 4: Geographic Coordinate Transformation
         Local ENU ──► Geocentric ECEF (EPSG:4978) ──► Projected CRS (WGS84 UTM EPSG:326XX)
                           │
                           ▼
Stage 5: Independent Ground Truth Validation
         Validation against independent survey-grade checkpoints (RTK GCPs) & LiDAR
```

---

## 2. Coordinate Systems & Geodetic Definitions

### 2.1 Coordinate Reference Systems (CRS)
1. **Model / Reconstruction Frame ($\mathbf{X}_m \in \mathbb{R}^3$):** Dimensionless or arbitrarily scaled relative frame produced by multi-view AI geometry or monocular SfM.
2. **Local Topocentric Tangent Plane (East-North-Up / ENU):** Cartesian frame centered at a local reference geodetic point $(\phi_0, \lambda_0, h_0)$. $X$ points East, $Y$ points North, $Z$ points Up along local ellipsoidal normal.
3. **Earth-Centered Earth-Fixed (ECEF: EPSG:4978):** Cartesian system $(X, Y, Z)$ in meters with origin at Earth's center of mass, $Z$ along rotation axis, $X$ through Prime Meridian.
4. **WGS84 Geodetic (EPSG:4326):** Latitude $(\phi)$, Longitude $(\lambda)$, Ellipsoidal Height ($h$).
5. **Universal Transverse Mercator (UTM: EPSG:326XX / EPSG:327XX):** Conformal cylindrical map projection dividing the Earth into 60 six-degree longitudinal zones. Preserves local angles and provides metric Cartesian coordinates $(E, N, h)$ for planar engineering measurements.

### 2.2 Geodetic to ECEF Conversion (WGS84 Ellipsoid)
Given semi-major axis $a = 6378137.0\text{ m}$ and first eccentricity squared $e^2 = 0.00669437999014$:
$$N(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2 \phi}}$$
$$\begin{bmatrix} X_{ecef} \\ Y_{ecef} \\ Z_{ecef} \end{bmatrix} = \begin{bmatrix} (N(\phi) + h)\cos \phi \cos \lambda \\ (N(\phi) + h)\cos \phi \sin \lambda \\ (N(\phi)(1 - e^2) + h)\sin \phi \end{bmatrix}$$

---

## 3. Metric Scale Recovery & Umeyama $\text{Sim}(3)$ Alignment

To align estimated camera centers $\mathbf{C}_m = \{\mathbf{c}_{m,1}, \dots, \mathbf{c}_{m,N}\} \subset \mathbb{R}^3$ to synchronized spatial anchors in local ENU coordinates $\mathbf{C}_{enu} = \{\mathbf{c}_{enu,1}, \dots, \mathbf{c}_{enu,N}\} \subset \mathbb{R}^3$, we solve the closed-form Umeyama formulation:

$$\min_{s, \mathbf{R}, \mathbf{t}} \frac{1}{N} \sum_{i=1}^N \left\| \mathbf{c}_{enu, i} - (s \mathbf{R} \mathbf{c}_{m, i} + \mathbf{t}) \right\|^2$$

### 3.1 Closed-Form Solution Steps
1. Compute centroids:
   $$\boldsymbol{\mu}_m = \frac{1}{N}\sum_{i=1}^N \mathbf{c}_{m,i}, \quad \boldsymbol{\mu}_{enu} = \frac{1}{N}\sum_{i=1}^N \mathbf{c}_{enu,i}$$
2. Compute centered coordinates and variances:
   $$\sigma_m^2 = \frac{1}{N}\sum_{i=1}^N \|\mathbf{c}_{m,i} - \boldsymbol{\mu}_m\|^2$$
3. Cross-covariance matrix $\mathbf{\Sigma}_{m, enu} \in \mathbb{R}^{3 \times 3}$:
   $$\mathbf{\Sigma}_{m, enu} = \frac{1}{N}\sum_{i=1}^N (\mathbf{c}_{enu,i} - \boldsymbol{\mu}_{enu})(\mathbf{c}_{m,i} - \boldsymbol{\mu}_m)^T$$
4. Compute SVD: $\mathbf{\Sigma}_{m, enu} = \mathbf{U} \mathbf{D} \mathbf{V}^T$, where $\mathbf{D} = \text{diag}(d_1, d_2, d_3)$ with $d_1 \ge d_2 \ge d_3 \ge 0$.
5. Construct sign-correction matrix $\mathbf{S}' = \text{diag}(1, 1, \det(\mathbf{U}\mathbf{V}^T))$ to enforce proper rotation ($\det(\mathbf{R}) = +1$, avoiding reflections).
6. Optimal parameters:
   $$\mathbf{R}_{align} = \mathbf{U} \mathbf{S}' \mathbf{V}^T \in SO(3)$$
   $$s = \frac{1}{\sigma_m^2} \text{tr}(\mathbf{D} \mathbf{S}') \in \mathbb{R}^+$$
   $$\mathbf{t}_{align} = \boldsymbol{\mu}_{enu} - s \mathbf{R}_{align} \boldsymbol{\mu}_m \in \mathbb{R}^3$$

---

## 4. Distinction of Error Domains

Engineering rigor requires maintaining strict separation between three distinct error domains:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ERROR DOMAINS                                    │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
      ┌────────────────────────────┼────────────────────────────┐
      ▼                            ▼                            ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
│ Camera Trajectory│     │ 3D Geometric     │     │ Absolute Geospatial      │
│ Error (ATE, RPE) │     │ Error (Chamfer)  │     │ Error (GCP RMSE)         │
├──────────────────┤     ├──────────────────┤     ├──────────────────────────┤
│ Evaluates flight │     │ Evaluates dense  │     │ Evaluates real-world     │
│ path estimation; │     │ surface shape &  │     │ positioning accuracy on  │
│ does NOT measure │     │ point cloud      │     │ independent ground       │
│ reconstructed    │     │ fidelity vs      │     │ survey checkpoints       │
│ scene geometry!  │     │ LiDAR scan.      │     │ in global CRS.           │
└──────────────────┘     └──────────────────┘     └──────────────────────────┘
```

### 4.1 Camera Trajectory Error: ATE and RPE
- **Absolute Trajectory Error (ATE RMSE):**
  $$\text{ATE}_{\text{RMSE}} = \sqrt{\frac{1}{N}\sum_{i=1}^N \left\| \mathbf{c}_{enu, i} - (s \mathbf{R}_{align} \mathbf{c}_{m, i} + \mathbf{t}_{align}) \right\|^2}$$
- **Relative Pose Error (RPE) over time interval $\Delta t$:**
  Given relative motion $\mathbf{Q}_{i} = \mathbf{P}_{i}^{-1} \mathbf{P}_{i+\Delta t}$ and ground truth $\mathbf{Q}_{i}^{gt} = (\mathbf{P}_{i}^{gt})^{-1} \mathbf{P}_{i+\Delta t}^{gt}$:
  $$\mathbf{E}_{i} = (\mathbf{Q}_{i}^{gt})^{-1} \mathbf{Q}_{i}$$
  $$\text{RPE}_{\text{trans}} = \sqrt{\frac{1}{M}\sum_{i=1}^M \|\text{trans}(\mathbf{E}_i)\|^2}, \quad \text{RPE}_{\text{rot}} = \sqrt{\frac{1}{M}\sum_{i=1}^M \angle(\text{rot}(\mathbf{E}_i))^2}$$

### 4.2 3D Geometric Error: Chamfer Distance & Completeness
Given reconstructed dense point cloud $\mathcal{P} = \{\mathbf{p}_1, \dots, \mathbf{p}_{|\mathcal{P}|}\}$ and ground-truth LiDAR point cloud $\mathcal{G} = \{\mathbf{g}_1, \dots, \mathbf{g}_{|\mathcal{G}|}\}$ in a shared local metric frame:

1. **Chamfer Distance ($m$):**
   $$d_{\text{Chamfer}}(\mathcal{P}, \mathcal{G}) = \frac{1}{2|\mathcal{P}|}\sum_{\mathbf{p} \in \mathcal{P}} \min_{\mathbf{g} \in \mathcal{G}} \|\mathbf{p} - \mathbf{g}\| + \frac{1}{2|\mathcal{G}|}\sum_{\mathbf{g} \in \mathcal{G}} \min_{\mathbf{p} \in \mathcal{P}} \|\mathbf{g} - \mathbf{p}\|$$
2. **Accuracy (Precision at threshold $\tau$):**
   $$\text{Accuracy}(\tau) = \frac{1}{|\mathcal{P}|} \sum_{\mathbf{p} \in \mathcal{P}} \mathbb{I}\left( \min_{\mathbf{g} \in \mathcal{G}} \|\mathbf{p} - \mathbf{g}\| < \tau \right) \times 100\%$$
3. **Completeness (Recall at threshold $\tau$):**
   $$\text{Completeness}(\tau) = \frac{1}{|\mathcal{G}|} \sum_{\mathbf{g} \in \mathcal{G}} \mathbb{I}\left( \min_{\mathbf{p} \in \mathcal{P}} \|\mathbf{g} - \mathbf{p}\| < \tau \right) \times 100\%$$
4. **F-Score at threshold $\tau$:**
   $$F_1(\tau) = 2 \cdot \frac{\text{Accuracy}(\tau) \cdot \text{Completeness}(\tau)}{\text{Accuracy}(\tau) + \text{Completeness}(\tau)}$$

### 4.3 Absolute Geospatial Error: Ground Checkpoint RMSE
For $K$ independent ground survey checkpoints $\{\mathbf{p}_k^{GCP}\}$ measured via survey-grade RTK GNSS:
$$\text{RMSE}_{X} = \sqrt{\frac{1}{K}\sum_{k=1}^K (X_k - X_k^{GCP})^2}, \quad \text{RMSE}_{Y} = \sqrt{\frac{1}{K}\sum_{k=1}^K (Y_k - Y_k^{GCP})^2}, \quad \text{RMSE}_{Z} = \sqrt{\frac{1}{K}\sum_{k=1}^K (Z_k - Z_k^{GCP})^2}$$
$$\text{RMSE}_{XYZ} = \sqrt{\text{RMSE}_X^2 + \text{RMSE}_Y^2 + \text{RMSE}_Z^2}$$
$$\text{MAE}_{XYZ} = \frac{1}{K}\sum_{k=1}^K \sqrt{(X_k - X_k^{GCP})^2 + (Y_k - Y_k^{GCP})^2 + (Z_k - Z_k^{GCP})^2}$$
