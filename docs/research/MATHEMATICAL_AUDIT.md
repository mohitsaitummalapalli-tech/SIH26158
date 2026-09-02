# Mathematical Audit & Theoretical Validation

This document records the mathematical verification of all photogrammetric, geometric, and geospatial formulations implemented and documented in SIH26158.

---

## 1. Verified Formulations & Equations

| Topic / Formulation | File Location | Mathematical Definition | Audit Finding & Verification Status |
| :--- | :--- | :--- | :--- |
| **Pinhole Camera Model** | `docs/research/CLASSICAL_METHODS.md` | $\lambda \mathbf{x} = \mathbf{K} [\mathbf{R} \mid \mathbf{t}] \mathbf{X}_w$ | **CORRECT**: Explicitly defines intrinsic matrix $\mathbf{K}$, normalized ray projection, and homogeneous coordinates. |
| **Essential vs. Fundamental Matrix** | `docs/research/CLASSICAL_METHODS.md` | $\hat{\mathbf{x}}_j^T \mathbf{E}_{ij} \hat{\mathbf{x}}_i = 0$ vs $\mathbf{x}_j^T \mathbf{F}_{ij} \mathbf{x}_i = 0$ | **CORRECTED**: Explicitly distinguished normalized camera coordinates $\hat{\mathbf{x}} = \mathbf{K}^{-1}\mathbf{x}$ for Essential matrix from raw pixel coordinates for Fundamental matrix. |
| **Epipolar Triangulation** | `docs/research/CLASSICAL_METHODS.md` | $\mathbf{A} \mathbf{X} = \mathbf{0}$ via DLT SVD | **CORRECT**: Overdetermined algebraic error minimization via right-singular vector. |
| **Bundle Adjustment & Gauge Freedom** | `docs/research/CLASSICAL_METHODS.md` | $\min \sum \rho(\|\mathbf{x} - \pi(\cdot)\|^2)$ with 7-DoF gauge ambiguity | **CORRECT**: Rigorously defines non-linear reprojection optimization and the 7-DoF null space ($\text{Sim}(3)$) in monocular SfM. |
| **5-Stage Spatial Transformation** | `docs/research/GEOSPATIAL_ACCURACY.md` | Relative $\to$ Scale $\to \text{Sim}(3) \to$ Geodetic/UTM $\to$ Independent Validation | **CORRECT**: Corrects common misconception that $\text{Sim}(3)$ automatically solves georeferencing without datum transformations. |
| **Umeyama $\text{Sim}(3)$ Formulation** | `docs/research/GEOSPATIAL_ACCURACY.md` | Closed-form SVD of cross-covariance with chirality check $\det(\mathbf{R}) = +1$ | **CORRECT**: Explicitly enforces sign-correction matrix $\mathbf{S}'$ to guarantee proper rotation and prevent mirrored reflections. |
| **WGS84 Geodetic to ECEF** | `docs/research/GEOSPATIAL_ACCURACY.md` | $N(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2 \phi}}$ | **CORRECT**: Standard EPSG:4978 ellipsoidal formulation with WGS84 constants ($a=6378137.0\text{m}$, $e^2=0.00669437999014$). |
| **ATE (Absolute Trajectory Error)** | `docs/research/GEOSPATIAL_ACCURACY.md` | $\text{ATE}_{\text{RMSE}} = \sqrt{\frac{1}{N}\sum \|\mathbf{c}^{gt} - \mathbf{S}\mathbf{c}\|^2}$ | **CORRECT**: Measures camera center trajectory root-mean-square error after $\text{Sim}(3)$ alignment. |
| **RPE (Relative Pose Error)** | `docs/research/GEOSPATIAL_ACCURACY.md` | $\mathbf{E}_i = (\mathbf{Q}_i^{gt})^{-1} \mathbf{Q}_i$ | **CORRECT**: Measures drift rate per unit time/distance for translation and rotation. |
| **Chamfer Distance & F-Score** | `docs/research/GEOSPATIAL_ACCURACY.md` | Symmetric cloud-to-cloud distance and $F_1(\tau)$ precision/recall at threshold $\tau$ | **CORRECT**: Standard metric definition for dense point cloud evaluation against LiDAR. |
| **Checkpoint Survey RMSE / MAE** | `docs/research/GEOSPATIAL_ACCURACY.md` | $\text{RMSE}_{XYZ} = \sqrt{\text{RMSE}_X^2 + \text{RMSE}_Y^2 + \text{RMSE}_Z^2}$ | **CORRECT**: Standard geospatial surveying standard for independent GCP evaluation. |

---

## 2. Distinction of Error Domains

The mathematical framework enforces strict separation between:
1. **Camera Trajectory Error (ATE / RPE):** Evaluates flight path estimation accuracy, but does NOT guarantee 3D scene surface accuracy.
2. **3D Geometric Error (Chamfer Distance / Cloud-to-Mesh RMSE):** Evaluates relative object/surface shape fidelity in local metric frame.
3. **Absolute Geospatial Error (Checkpoint RMSE XYZ):** Evaluates real-world positioning accuracy in global CRS.
