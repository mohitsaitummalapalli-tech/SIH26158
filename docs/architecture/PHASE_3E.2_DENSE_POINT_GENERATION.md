# Phase 3E.2: Dense 3D Point Generation & Geometric Validation Architecture

## 1. Executive Summary
Phase 3E.2 establishes the canonical 3D unprojection, coordinate transformation, and geometric validation layer that maps validated 2D dense stereo depth maps (Phase 3E.1) into structured 3D point observations and point clouds in relative reconstruction coordinates (`RECONSTRUCTION_UNITS`).

```
Dense Stereo Depth Map (Phase 3E.1)
  + Rectified Intrinsics P1
  + Rectification Rotation R1
  + Reference Camera Pose [R_ref | t_ref]
        │
        ▼
[Step 1: Rectified Backprojection]
  X_rect = [ Z_rect * (u - cx_rect) / fx_rect ]
           [ Z_rect * (v - cy_rect) / fy_rect ]
           [ Z_rect                            ]
        │
        ▼
[Step 2: Invert Rectification Rotation]
  X_c_orig = R1^T * X_rect
        │
        ▼
[Step 3: World Transformation]
  X_w = R_ref^T * (X_c_orig - t_ref) = C_w + R_ref^T * X_c_orig
        │
        ▼
[Step 4: Multi-Criteria Geometric Validation]
  - Finite Coordinates Check
  - Cheirality Check (Z > 0 in original & rectified frames)
  - Depth Bounds Check [min_depth, max_depth]
  - Confidence Threshold Check
  - Forward Reprojection Consistency Check (error <= max_px)
        │
        ▼
Validated Dense 3D Point Observations & Dense Point Cloud
```

---

## 2. Mathematical Foundations

### 2.1 Camera Coordinate System
Under the project's standard optical camera model:
$$\mathbf{X}_{c, orig} = \mathbf{R}_{ref} \mathbf{X}_w + \mathbf{t}_{ref}$$
$$\mathbf{C}_{w, ref} = -\mathbf{R}_{ref}^T \mathbf{t}_{ref} \iff \mathbf{t}_{ref} = -\mathbf{R}_{ref} \mathbf{C}_{w, ref}$$
where $\mathbf{X}_w$ is the 3D world coordinate, $\mathbf{R}_{ref} \in \text{SO}(3)$ is the world-to-camera rotation, and $\mathbf{t}_{ref} \in \mathbb{R}^3$ is the extrinsic translation vector.

### 2.2 Rectified Coordinate Inversion
Stereo rectification (`cv2.stereoRectify`) applies a 3D rotation transform $\mathbf{R}_1$ to bring the unrectified reference camera frame into epipolar alignment:
$$\mathbf{X}_{rect} = \mathbf{R}_1 \mathbf{X}_{c, orig}$$
Therefore, to map points from rectified camera coordinates $\mathbf{X}_{rect}$ back to the original camera optical frame:
$$\mathbf{X}_{c, orig} = \mathbf{R}_1^T \mathbf{X}_{rect}$$

### 2.3 World Coordinate Transformation
Combining the rectification inversion with the camera extrinsic transformation:
$$\mathbf{X}_w = \mathbf{R}_{ref}^T (\mathbf{X}_{c, orig} - \mathbf{t}_{ref}) = \mathbf{R}_{ref}^T (\mathbf{R}_1^T \mathbf{X}_{rect} - \mathbf{t}_{ref})$$
Equivalently expressed using camera optical center $\mathbf{C}_{w, ref}$:
$$\mathbf{X}_w = \mathbf{C}_{w, ref} + \mathbf{R}_{ref}^T \mathbf{R}_1^T \mathbf{X}_{rect}$$

---

## 3. Geometric Validation Pipeline

Every generated 3D point must satisfy six explicit physical and mathematical criteria:

1. **Finite Coordinates**:
   $$\forall k \in \{0, 1, 2\}, \quad \mathbf{X}_w[k], \mathbf{X}_{c, orig}[k], \mathbf{X}_{rect}[k] \in (-\infty, \infty)$$
2. **Cheirality Invariant**:
   Points must lie strictly in front of the optical planes of both the original reference camera and the rectified camera:
   $$Z_{c, orig} = \mathbf{X}_{c, orig}[2] > 10^{-6}, \quad Z_{rect} = \mathbf{X}_{rect}[2] > 10^{-6}$$
3. **Depth Range Bounds**:
   $$Z_{rect} \in [Z_{min}, Z_{max}] \quad (\text{HEURISTIC\_DEFAULT: } [0.5, 100.0]\text{ reconstruction units})$$
4. **Disparity Validity**:
   $$d > 0.0$$
5. **Confidence Filter**:
   $$C \ge C_{min} \quad (\text{HEURISTIC\_DEFAULT: } 0.20)$$
6. **Reprojection Consistency**:
   Forward projecting $\mathbf{X}_w$ to the rectified image plane $(u'_{rect}, v'_{rect})$ must match the originating pixel $(u_{rect}, v_{rect})$:
   $$\Delta_{reproj} = \sqrt{(u'_{rect} - u_{rect})^2 + (v'_{rect} - v_{rect})^2} \le \tau_{reproj} \quad (\text{HEURISTIC\_DEFAULT: } 2.0\text{ px})$$

---

## 4. Rejection Taxonomy
When a pixel fails geometric validation, the rejection reason is explicitly logged into `DensePointGenerationResult.rejection_breakdown`:
- `INVALID_DEPTH_VALUE`
- `NON_POSITIVE_DEPTH`
- `OUT_OF_DEPTH_BOUNDS`
- `NON_FINITE_COORDINATES`
- `CHEIRALITY_VIOLATION`
- `REPROJECTION_ERROR_EXCEEDED`
- `LOW_CONFIDENCE`
- `OCCLUDED_OR_INCONSISTENT`
- `INVALID_DISPARITY`
- `MASKED_OUT`

---

## 5. Scope & Real-Data Boundaries

### Proven:
- Bit-exact coordinate round-trip $(\mathbf{X}_w \to \mathbf{X}_c \to \mathbf{X}_{rect} \to (u, v, z) \to \mathbf{X}_{rect} \to \mathbf{X}_c \to \mathbf{X}_w)$ within $10^{-10}$ tolerance.
- Exact inversion of non-identity rectification rotation $\mathbf{R}_1^T$.
- Rigorous enforcement of $\mathbf{t}_{cw}$ vs $\mathbf{C}_w$.

### Explicitly Excluded (Deferred to Subsequent Phases):
- Multi-view voxel fusion / TSDF integration (Phase 3E.3).
- Surface mesh generation / Poisson reconstruction.
- Absolute metric scale estimation without certified ground truth.
- Real-world UAV aerodynamic flight vibration and rolling shutter modeling.
