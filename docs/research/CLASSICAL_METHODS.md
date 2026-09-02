# Classical Photogrammetry & SfM Methods: Mathematical Rigor

## 1. Classical Pipeline Anatomy

Standard photogrammetric 3D reconstruction follows a multi-stage sequential pipeline:

```
Video Frames ──► Feature Detection ──► Feature Matching ──► Epipolar Verification (RANSAC)
                     (SIFT/ORB)         (Flann/Brute-Force)           │
                                                                      ▼
3D Point Cloud ◄── Multi-View Stereo ◄── Bundle Adjustment ◄── Incremental / Global SfM
(MVS: PMVS/PatchMatch)  (Dense Matching)      (Levenberg-Marquardt)   (Camera Poses)
```

---

## 2. Core Mathematical Formulations

### 2.1 Pinhole Camera Model & Projection
A 3D point in world coordinate frame $\mathbf{X}_w = [X_w, Y_w, Z_w, 1]^T$ projects to homogeneous pixel coordinates $\mathbf{x} = [u, v, 1]^T$ via:
$$\lambda \mathbf{x} = \mathbf{K} [\mathbf{R} \mid \mathbf{t}] \mathbf{X}_w = \mathbf{P} \mathbf{X}_w$$
where $\lambda$ is depth in the camera frame, $\mathbf{P} \in \mathbb{R}^{3 \times 4}$ is the camera projection matrix, and $\mathbf{K}$ is the upper-triangular intrinsic calibration matrix:
$$\mathbf{K} = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
where $f_x, f_y$ are focal lengths in pixel units, $(c_x, c_y)$ is the principal point, and $s$ is the skew coefficient (typically $0$).

The extrinsic parameters $[\mathbf{R} \mid \mathbf{t}] \in SE(3)$ transform points from the world frame to the camera frame:
$$\mathbf{X}_c = \mathbf{R} \mathbf{X}_w + \mathbf{t}$$

### 2.2 Epipolar Geometry: Essential vs. Fundamental Matrix
For a calibrated camera pair with intrinsics $\mathbf{K}_i, \mathbf{K}_j$, matching homogeneous pixel points $\mathbf{x}_i, \mathbf{x}_j$ are converted to **normalized camera ray coordinates**:
$$\hat{\mathbf{x}}_i = \mathbf{K}_i^{-1} \mathbf{x}_i, \quad \hat{\mathbf{x}}_j = \mathbf{K}_j^{-1} \mathbf{x}_j$$

The **Essential Matrix** $\mathbf{E}_{ij} \in \mathbb{R}^{3 \times 3}$ relates normalized rays:
$$\hat{\mathbf{x}}_j^T \mathbf{E}_{ij} \hat{\mathbf{x}}_i = 0, \quad \text{where } \mathbf{E}_{ij} = [\mathbf{t}_{ij}]_\times \mathbf{R}_{ij}$$
where $[\mathbf{t}_{ij}]_\times$ is the skew-symmetric matrix of relative translation $\mathbf{t}_{ij} = [t_x, t_y, t_z]^T$:
$$[\mathbf{t}_{ij}]_\times = \begin{bmatrix} 0 & -t_z & t_y \\ t_z & 0 & -t_x \\ -t_y & t_x & 0 \end{bmatrix}$$

The **Fundamental Matrix** $\mathbf{F}_{ij} \in \mathbb{R}^{3 \times 3}$ relates uncalibrated pixel coordinates directly:
$$\mathbf{x}_j^T \mathbf{F}_{ij} \mathbf{x}_i = 0, \quad \text{where } \mathbf{F}_{ij} = \mathbf{K}_j^{-T} \mathbf{E}_{ij} \mathbf{K}_i^{-1}$$

From $\mathbf{E}_{ij}$, SVD decomposition $\mathbf{E}_{ij} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$ yields four candidate $(\mathbf{R}_{ij}, \mathbf{t}_{ij})$ solutions. The unique physically valid configuration is selected via chirality verification (cheirality check: all triangulated 3D points must lie in front of both cameras, $Z > 0$).

Notice that the baseline length $\|\mathbf{t}_{ij}\|$ is strictly undetermined: epipolar geometry recovers translation direction only, creating inherent monocular scale ambiguity.

### 2.3 Direct Linear Transformation (DLT) Triangulation
Given matching rays in two views with projection matrices $\mathbf{P}_1, \mathbf{P}_2$, a 3D point $\mathbf{X}$ satisfies $\mathbf{x}_1 \times (\mathbf{P}_1 \mathbf{X}) = \mathbf{0}$ and $\mathbf{x}_2 \times (\mathbf{P}_2 \mathbf{X}) = \mathbf{0}$. This forms an over-determined linear system $\mathbf{A} \mathbf{X} = \mathbf{0}$:
$$\mathbf{A} = \begin{bmatrix} u_1 \mathbf{p}_1^{3T} - \mathbf{p}_1^{1T} \\ v_1 \mathbf{p}_1^{3T} - \mathbf{p}_1^{2T} \\ u_2 \mathbf{p}_2^{3T} - \mathbf{p}_2^{1T} \\ v_2 \mathbf{p}_2^{3T} - \mathbf{p}_2^{2T} \end{bmatrix}$$
where $\mathbf{p}_i^{kT}$ denotes the $k$-th row of projection matrix $\mathbf{P}_i$. The optimal solution $\mathbf{X}$ is the right singular vector of $\mathbf{A}$ corresponding to its smallest singular value.

### 2.4 Global Bundle Adjustment & Gauge Freedom
Bundle adjustment refines camera extrinsics $\mathbf{P}_i = \{\mathbf{R}_i, \mathbf{t}_i\}$, intrinsics $\mathbf{K}_i$, and 3D structure $\mathbf{X}_k$ by minimizing non-linear reprojection error:
$$\min_{\{\mathbf{P}_i\}, \{\mathbf{X}_k\}} \sum_{i=1}^M \sum_{k=1}^N \rho \left( \left\| \mathbf{x}_{ik} - \pi(\mathbf{K}_i, \mathbf{P}_i, \mathbf{X}_k) \right\|^2_{\mathbf{\Sigma}_{ik}^{-1}} \right)$$
where $\pi(\cdot)$ is the projection function, $\rho(\cdot)$ is a robust Huber/Cauchy loss, and $\mathbf{\Sigma}_{ik}$ is the feature measurement covariance.

**Gauge Freedom:** Monocular reconstruction possesses a 7-dimensional null space (Gauge Ambiguity) corresponding to the group $\text{Sim}(3)$:
- 3 degrees of freedom in global rotation.
- 3 degrees of freedom in global translation.
- 1 degree of freedom in global scale.
Without external constraints (GCPs, RTK GNSS, or IMU priors), fixing the gauge arbitrarily leaves scale and global orientation ungrounded.

---

## 3. Failure Mechanics in Single-Pass Linear Trajectories

1. **Gauge Drift Accumulation:** In a 1D sequence without cross-pass loop closures, errors in consecutive relative rotations $\Delta \mathbf{R}_{i, i+1}$ accumulate monotonically. Small uncalibrated radial distortion $k_1$ or focal length drift induces non-linear deformation:
   $$\Delta Z(x) \approx \alpha x^2 + \beta x$$
   manifesting as the classic photogrammetric "bowl" or "dome" effect.
2. **Degenerate Triangulation Baseline:** When the UAV moves purely forward along its optical axis, disparity rays become nearly parallel ($B \to 0$), causing the depth variance $\sigma_Z \approx \frac{Z^2}{f B} \sigma_p$ to approach infinity.
