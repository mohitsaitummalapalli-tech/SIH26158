# 3D Reconstruction Landscape: State of the Art & Paradigms

## 1. Overview of 3D Reconstruction Paradigms

Three major paradigms dominate modern 3D vision and photogrammetry:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       3D RECONSTRUCTION PARADIGMS                            │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
      ┌────────────────────────────┼────────────────────────────┐
      ▼                            ▼                            ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
│    Classical     │     │ Neural Radiance  │     │ Foundation AI Geometry   │
│ Photogrammetry   │     │ Fields & 3DGS    │     │ Models (DUSt3R, VGGT)    │
├──────────────────┤     ├──────────────────┤     ├──────────────────────────┤
│ • SfM + MVS      │     │ • NeRF, Instant- │     │ • Direct Pointmap        │
│ • Epipolar geom  │     │   NGP, 3DGS      │     │   Regression             │
│ • Triangulation  │     │ • Volumetric /   │     │ • Uncalibrated, zero-    │
│ • Bundle Adjust. │     │   Gaussian splat │     │   shot multi-view stereo │
│ • High metric    │     │ • Photorealistic │     │ • Robust to low overlap  │
│   rigor on grids │     │   rendering      │     │ • Unconstrained scale    │
└──────────────────┘     └──────────────────┘     └──────────────────────────┘
```

---

## 2. Comparative Breakdown

| Paradigm | Strengths | Critical Weaknesses in Single-Pass Drone Video |
| :--- | :--- | :--- |
| **Classical Photogrammetry (COLMAP, OpenSfM, Pix4D)** | • Strong geometric foundation via bundle adjustment.<br>• Proven metric scale when calibrated with GCPs/RTK.<br>• Exact epipolar constraints and outlier rejection (RANSAC). | • Fails when overlap is low or baseline is acute.<br>• Prone to catastrophic track breakage on homogeneous surfaces (water, asphalt, uniform roofs).<br>• Monocular scale drift and severe bending in linear trajectories.<br>• High compute latency ($O(N^2)$ to $O(N^3)$). |
| **Novel View Synthesis (NeRF, 3D Gaussian Splatting)** | • Unmatched photorealistic view synthesis.<br>• Continuous representation.<br>• 3DGS achieves real-time rendering ($>100\text{ fps}$). | • Requires pre-computed accurate camera poses (typically from COLMAP).<br>• Poor extrapolation outside the flight trajectory.<br>• Extracted surfaces often contain floaters, hollow shells, and lack metric survey-grade surfaces.<br>• Expensive per-scene optimization time. |
| **Foundation AI Geometry (DUSt3R, MASt3R, VGGT)** | • Direct regression of 3D pointmaps in camera/world frame without explicit camera calibration.<br>• Robust across wide baselines and low overlap.<br>• Handles textureless regions via learned priors.<br>• Feed-forward inference is rapid. | • Uncalibrated relative coordinate outputs require global alignment and metric scaling.<br>• High GPU VRAM requirements during global alignment optimization.<br>• Can smooth out sharp geometric edges or hallucinate unobserved surfaces if unconstrained. |

---

## 3. The Single-Pass Challenge Landscape

In conventional mapping, drones fly cross-hatch trajectories with $>75\%$ lateral overlap. A single-pass trajectory lacks lateral cross-ties, causing:
1. **Critical Gauge Ambiguity:** Incremental bundle adjustment lacks cross-track geometric constraints, resulting in vertical parabolic curling ("bowl effect").
2. **Ill-conditioned Triangulation:** Small angular baseline between consecutive frames magnifies depth variance:
   $$\sigma_Z \approx \frac{Z^2}{f \cdot B} \sigma_p$$
   where $B$ is baseline, $Z$ is depth, $f$ is focal length, and $\sigma_p$ is pixel disparity error. As $B \to 0$, depth uncertainty explodes.
3. **Occlusion Asymmetry:** Oblique single-pass views only observe one side of buildings/structures; hidden facades must be recognized as unobserved rather than falsely reconstructed.

---

## 4. Strategic Direction for SIH26158

Neither pure classical photogrammetry nor raw unconstrained AI models solve the single-pass problem in isolation. The optimal path requires an **AI-Geometry + Geospatial Fusion Strategy**:
1. Leverage **Foundation AI Geometry** (e.g. DUSt3R / VGGT) to obtain robust pairwise pointmaps and relative poses even in low-overlap/monocular motion conditions where classical feature matching fails.
2. Formulate a **Global Pose & Pointmap Graph Optimization** constrained by physical drone telemetry (GNSS, barometric altitude, IMU gravity vector).
3. Fuse multi-view pointmaps into a global metric coordinate frame (UTM) with rigorous **Uncertainty Quantification** to prevent hallucinated geometry.
