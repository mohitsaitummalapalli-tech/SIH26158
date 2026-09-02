# AI Geometry Models: Architecture, Benchmarks & Specifications

## 1. Overview of Foundation 3D Vision Models

Recent advances in Vision Transformers (ViT) have enabled direct, uncalibrated, feed-forward 3D point cloud and pose estimation from unposed image pairs or video streams. Two prominent architectures are evaluated for SIH26158:
1. **DUSt3R** (3D Reconstruction Made Easy) & **MASt3R** (Grounding Matching in 3D).
2. **VGGT** (Visual Geometry Grounded Transformer).

---

## 2. DUSt3R & MASt3R Architecture Analysis

### 2.1 Core Principle: Direct Pointmap Regression
Instead of the classical multi-stage pipeline (features $\to$ epipolar geometry $\to$ triangulation $\to$ depth maps), DUSt3R frames multi-view stereo as a direct regression of **pixel-aligned 3D pointmaps**:

$$\mathcal{F}_{\theta}: (I_1, I_2) \longmapsto \left( \mathbf{X}^{1,1}, \mathbf{X}^{2,1}, \mathbf{C}^{1,1}, \mathbf{C}^{2,1} \right)$$

where:
- $I_1, I_2 \in \mathbb{R}^{H \times W \times 3}$ are two input images.
- $\mathbf{X}^{1,1} \in \mathbb{R}^{H \times W \times 3}$ is the 3D pointmap of image 1 expressed in camera 1's coordinate frame.
- $\mathbf{X}^{2,1} \in \mathbb{R}^{H \times W \times 3}$ is the 3D pointmap of image 2 **expressed in camera 1's coordinate frame**.
- $\mathbf{C}^{1,1}, \mathbf{C}^{2,1} \in \mathbb{R}^{H \times W}$ are per-pixel confidence maps.

### 2.2 Global Multi-View Alignment
For a sequence of $N$ images, DUSt3R constructs a connectivity graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ and solves a global optimization over rigid transformations $\mathbf{P}_v = (\sigma_v, \mathbf{R}_v, \mathbf{t}_v) \in \text{Sim}(3)$:

$$\min_{\{\mathbf{P}_v\}, \{\mathbf{X}_v\}} \sum_{e=(u,v)\in\mathcal{E}} \sum_{i=1}^{HW} C_i^{v,u} \left\| \mathbf{X}_i^u - \mathbf{P}_v \mathbf{X}_i^{v,u} \right\|^2$$

This yields jointly optimized camera poses and unified 3D point clouds without requiring prior camera calibration.

### 2.3 Strengths & Limitations for Drone Video
- **Strengths:** Robust to extreme view changes, handles textureless surfaces (roads, roofs) smoothly, produces dense depth with confidence maps.
- **Limitations:** Output is in an arbitrary coordinate frame (requires metric scale recovery via GNSS/IMU); optimization across long sequences ($N > 100$) can be memory intensive without windowed sub-graph optimization.

---

## 3. VGGT (Visual Geometry Grounded Transformer) Analysis

### 3.1 Architecture & Mechanism
VGGT models full sequence geometry using an autoregressive or transformer encoder-decoder backbone conditioned on visual features and learned geometric embeddings:
- Directly models spatial-temporal correspondences across sequential video frames.
- Predicts dense depth, camera motion trajectories, and surface normal priors simultaneously.
- Employs self-attention across wide temporal horizons to enforce trajectory consistency.

### 3.2 Strengths & Limitations for Drone Video
- **Strengths:** Exploits continuous temporal structure of drone flight video; less prone to pairwise drift over smooth linear trajectories.
- **Limitations:** Higher compute footprint per frame; requires strict VRAM management during long sequence execution.

---

## 4. Hardware, Compute & Licensing Matrix

| Model / Architecture | Open Source License | Parameter Size | Minimum VRAM (Pairwise) | VRAM (Global Opt, 100 Frames) | Native Output Scale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DUSt3R (ViT-Large)** | CC BY-NC-SA 4.0 / Research | $\sim 560\text{M}$ | 8 GB | 24 GB (or 16 GB with fp16 + gradient checkpointing) | Arbitrary relative scale |
| **MASt3R (ViT-Large)** | CC BY-NC-SA 4.0 / Research | $\sim 560\text{M}$ | 8 GB | 24 GB | Arbitrary relative scale |
| **VGGT** | Academic / Research | $\sim 300\text{M}-700\text{M}$ | 12 GB | 32 GB | Arbitrary / normalized relative scale |
| **COLMAP Baseline** | BSD-3-Clause (Commercial OK) | N/A (Algorithmic) | 4 GB | Scalable on CPU / GPU | Scale undetermined without GCPs/EXIF |

---

## 5. Architectural Adaptation Strategy for Phase 1

1. **Sliding-Window Subgraph Optimization:** Decompose $N$-frame drone flight tracks into overlapping local windows of $W = 15\text{--}30$ keyframes.
2. **Telemetry-Anchored Global Sim(3) Alignment:** Use GNSS camera trajectory $[\mathbf{c}_1, \dots, \mathbf{c}_N]$ and barometric altitude to initialize and constrain the 7-DoF similarity transformations $\mathbf{S} \in \text{Sim}(3)$.
3. **Confidence-Weighted Point Density Pruning:** Utilize predicted confidence maps $\mathbf{C}$ to reject uncertain background pixels, horizon artifacts, and dynamic obstacles before surface meshing.
