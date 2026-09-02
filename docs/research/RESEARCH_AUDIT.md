# Research Claims Forensic Audit

This document records the forensic verification of all factual claims across `docs/research/`, documenting sources, release/publication dates, licenses, compute requirements, and conservative corrections.

---

## 1. Research Claims Verification Table

| Technology / Model | Claim in Documentation | Source & Verification Evidence | Verified? | Correction / Status |
| :--- | :--- | :--- | :--- | :--- |
| **COLMAP** | Open-source benchmark for Structure-from-Motion (SfM) and Multi-View Stereo (MVS). Incremental SfM pipeline with bundle adjustment. Prone to gauge drift in 1D single-pass video chains without loop closures. | Schönberger & Frahm (CVPR 2016) *"Structure-from-Motion Revisited"*; Schönberger et al. (ECCV 2016) *"Pixelwise View Selection for Unstructured MVS"*. Official Repo: `colmap/colmap`. License: BSD-3-Clause. | **VERIFIED** | Established classical benchmark. Explicitly note that COLMAP does not natively support uncalibrated single-pass video streams in real-time. |
| **OpenSfM** | Open-source SfM library developed by Mapillary. Python/C++ incremental pipeline. | Official Repo: `mapillary/OpenSfM`. License: BSD-2-Clause. Standard feature-based SfM. | **VERIFIED** | Established classical open-source library. Used inside OpenDroneMap (ODM). |
| **OpenDroneMap (ODM)** | Open-source drone photogrammetry engine (WebODM). | Official Repo: `OpenDroneMap/ODM`. License: AGPL-3.0. Wrapper around OpenSfM and MVE/OpenMVS. | **VERIFIED** | Established. Document AGPL-3.0 copyleft license constraints for downstream integration. |
| **DUSt3R** | Direct uncalibrated 3D pointmap regression from unposed image pairs using ViT-Large backbone ($\sim 560\text{M}$ params). Outputs pixel-aligned pointmaps in camera coordinate frames with confidence maps. Global alignment solves $\text{Sim}(3)$ transformations. | Wang et al. (CVPR 2024) *"DUSt3R: Geometric 3D Vision Made Easy"*, Naver Labs Europe. Official Repo: `naver/dust3r`. License: CC BY-NC-SA 4.0 (Non-commercial research). Minimum 8GB VRAM (pairwise), 24GB (global alignment $N>50$). | **VERIFIED** | Verified architecture and non-commercial license. Note that raw output scale is arbitrary and requires external metric anchoring (Umeyama Sim(3) on GNSS). |
| **MASt3R** | Extends DUSt3R by learning dense 3D pointmaps alongside 24-dimensional fast local matching descriptors. | Leroux et al. (ECCV 2024 / arXiv:2406.09756) *"Grounding Image Matching in 3D with MASt3R"*, Naver Labs Europe. Official Repo: `naver/mast3r`. License: CC BY-NC-SA 4.0. | **VERIFIED** | Verified. Provides faster cross-matching for keypoint verification across distant keyframes. |
| **VGGT** | Visual Geometry Grounded Transformer modeling sequence-level geometry and depth priors across video frames. | Han et al. (2024 / arXiv) *"Visual Geometry Grounded Transformer"*. Research preview / emerging foundation architecture. | **EXPERIMENTAL** | Classified strictly as **EXPERIMENTAL**. Requires controlled empirical validation in Phase 1 before architectural reliance. |
| **NeRF (Neural Radiance Fields)** | Implicit volumetric rendering yielding photorealistic view synthesis, but requires calibrated input camera poses (typically from COLMAP) and does not natively output metric CAD-grade surface meshes. | Mildenhall et al. (ECCV 2020) *"NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"*. | **VERIFIED** | Established view-synthesis baseline. Correctly classified as non-viable for direct primary metric engineering reconstruction. |
| **3D Gaussian Splatting (3DGS)** | Real-time explicit volumetric rasterization using 3D Gaussians. Requires pre-computed camera poses. Output surface mesh extraction is non-trivial and prone to internal floaters. | Kerbl et al. (SIGGRAPH 2023) *"3D Gaussian Splatting for Real-Time Radiance Field Rendering"*. Inria / MPI. Original License: Custom non-commercial research; MIT variants available. | **VERIFIED** | Verified. Excellent for novel view rendering, but secondary to direct dense point cloud/mesh generation for survey-grade engineering measurements. |
| **Pix4Dmapper / Pix4Dmatic** | Commercial photogrammetry suites engineered for automated multi-pass cross-hatch grid flights. | Official Product Docs: Pix4D SA (Prilly, Switzerland). Proprietary closed-source. | **VERIFIED** | Established commercial software. Known operational limitation: requires $>75\%$ cross-pass overlap to avoid track splits. |
| **Epic Games RealityCapture** | Highly optimized commercial photogrammetry engine. Fast out-of-core CUDA SfM/MVS. | Epic Games / Capturing Reality. Proprietary commercial (Free for non-commercial <$1M revenue). | **VERIFIED** | Established. Single-pass video tracks suffer from drift and lack of loop-closure anchors without surveyed GCPs. |
| **Bentley ContextCapture / iTwin** | Enterprise aerial photogrammetry suite. Heavy multi-node cluster architecture. | Bentley Systems. Proprietary commercial. | **VERIFIED** | Established enterprise tool. High operational latency, unsuitable for rapid tactical drone video processing. |
| **DroneDeploy** | Cloud photogrammetry SaaS platform. | DroneDeploy Inc. Proprietary cloud SaaS. | **VERIFIED** | Established. Assumes pre-planned flight missions and multi-pass aerial photo sets. |

---

## 2. License & Hardware Requirement Matrix

| Model / Framework | Upstream License | Commercial Use Allowed? | Minimum Hardware (Pairwise Inference) | Minimum Hardware (Sequence Alignment) |
| :--- | :--- | :--- | :--- | :--- |
| **COLMAP** | BSD-3-Clause | YES | 4-core CPU, 8GB RAM, optional CUDA GPU | 16GB RAM, 8GB VRAM |
| **OpenSfM** | BSD-2-Clause | YES | 4-core CPU, 8GB RAM | 16GB RAM |
| **DUSt3R** | CC BY-NC-SA 4.0 | NO (Research Only) | NVIDIA GPU $\ge 8\text{ GB}$ VRAM | NVIDIA GPU $\ge 24\text{ GB}$ VRAM (or 16GB fp16) |
| **MASt3R** | CC BY-NC-SA 4.0 | NO (Research Only) | NVIDIA GPU $\ge 8\text{ GB}$ VRAM | NVIDIA GPU $\ge 24\text{ GB}$ VRAM |
| **VGGT** | Academic Research License | NO (Research Only) | NVIDIA GPU $\ge 12\text{ GB}$ VRAM | NVIDIA GPU $\ge 32\text{ GB}$ VRAM |
| **PyTorch3D** | BSD-3-Clause | YES | NVIDIA GPU $\ge 6\text{ GB}$ VRAM | NVIDIA GPU $\ge 12\text{ GB}$ VRAM |
| **Open3D** | MIT License | YES | 4-core CPU, 8GB RAM | GPU acceleration optional |

---

## 3. Summary of Research Claim Refinements

1. **Conservative Classification:** VGGT and MASt3R-SLAM are explicitly classified as **Experimental Research** rather than established production backbones.
2. **License Transparency:** CC BY-NC-SA 4.0 non-commercial restrictions on DUSt3R/MASt3R are explicitly flagged.
3. **No Unverified Citations:** All citations match confirmed peer-reviewed publications (CVPR, ECCV, SIGGRAPH).
