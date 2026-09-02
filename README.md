# SIH26158: Single-Pass Drone Video to Accurate 3D Model Generation System

[![Phase](https://img.shields.io/badge/Phase-Phase%200%20(Research%20Foundation)-blue)](#current-project-status)
[![Standards](https://img.shields.io/badge/Standards-Strict%20Reproducibility-brightgreen)](#engineering-rules)

## Project Overview

**SIH26158** is a research-grade engineering initiative designed to reconstruct georeferenced, metrically accurate, textured 3D representations (point clouds, meshes, terrain, structures) from a **single continuous drone video flight path**.

Traditional aerial photogrammetry (e.g., standard SfM / MVS) demands multi-pass cross-hatch flight plans with high forward and lateral overlap (75–85%), calibrated ground control points (GCPs), and lengthy offline processing. In contrast, single-pass video acquisition faces severe technical constraints:
- Minimal parallax and narrow viewing angles.
- High motion blur, dynamic compression artifacts, and rolling shutter distortion.
- Scale ambiguity and rapid trajectory drift in linear trajectories.
- Requirement for metric scale accuracy and geospatial alignment without extensive ground surveying.

This repository provides the foundational architecture, mathematical specifications, experiment protocols, and scaffolding to rigorously evaluate classical photogrammetry baselines, foundational AI geometry models (e.g., DUSt3R, VGGT), and hybrid fusion pipelines.

---

## Current Project Status

> **PHASE 0 — RESEARCH FOUNDATION**
> - Mathematical and architectural definitions established.
> - Core data contracts, schemas, and metric validation guardrails constructed.
> - Reconstruction engines are intentionally uninstantiated pending empirical validation in Phase 1.
> - No fake or synthetic reconstruction outputs are presented.

---

## Engineering Rules

To ensure academic and industrial rigor, this project strictly enforces the following eight engineering rules across all code, tests, documentation, and experimental evaluations:

1. **No fabricated metrics.**
   Every reported number must originate from an executed, recorded, and verifiable benchmark run.
2. **No fake reconstruction results.**
   Never generate or display synthetic/placeholder 3D models claiming to be drone reconstructions.
3. **No unverified accuracy claims.**
   NO accuracy claim is valid unless it explicitly records:
   - **Named Dataset**
   - **Defined Ground Truth Reference** (e.g., LiDAR point cloud, RTK-surveyed check points)
   - **Defined Metric** (e.g., RMSE, MAE, Chamfer distance, F-score at threshold $\tau$)
   - **Reproducible Calculation Method**
4. **Research experiments must remain separate from production source code.**
   Exploratory scripts, benchmarks, and baseline notebooks live strictly under `experiments/` and `benchmarks/`, consuming versioned contracts from `src/`.
5. **Every major phase requires tests and a checkpoint.**
   No progression to subsequent development phases without verifiable integration tests and architecture approval.
6. **Unknown/unobserved geometry must never be silently presented as observed truth.**
   Inpainting, hallucinated neural geometry, or extrapolated surface areas must carry explicit confidence/uncertainty labels.
7. **Every external model must have documented license and hardware requirements.**
   Third-party models (DUSt3R, VGGT, MASt3R, COLMAP, OpenSfM) must have documented compute budgets, VRAM consumption, and licensing terms.
8. **Every benchmark must be reproducible.**
   All benchmark runs require deterministic random seeds, pinned dependencies, recorded hardware environment specifications, and configuration manifests.

---

## Repository Structure

```
SIH26158/
├── README.md                           # Project manifesto and engineering rules
├── docs/
│   ├── problem/
│   │   └── SIH26158_REQUIREMENTS.md    # Problem statement, inputs, outputs, challenges
│   ├── research/
│   │   ├── 3D_RECONSTRUCTION_LANDSCAPE.md # State-of-the-art overview
│   │   ├── CLASSICAL_METHODS.md        # SfM/MVS analysis & drift mechanics
│   │   ├── AI_GEOMETRY_MODELS.md       # DUSt3R, VGGT, MASt3R analysis
│   │   ├── GEOSPATIAL_ACCURACY.md      # Georeferencing, CRS, Sim(3), metric validation
│   │   └── COMPETITOR_ANALYSIS.md      # Pix4D, RealityCapture, OpenSfM benchmark comparison
│   ├── architecture/
│   │   └── SYSTEM_ARCHITECTURE.md      # Provisional pipeline architecture
│   └── experiments/
│       └── EXPERIMENT_PROTOCOL.md      # 4-way evaluation protocol (Classical, DUSt3R, VGGT, Fusion)
│
├── configs/
│   ├── development/                    # Development runtime parameters
│   └── testing/                        # Test suite configs
│
├── data/
│   ├── raw/                            # Drone video and raw telemetry (.SRT, .CSV, EXIF)
│   ├── processed/                      # Extracted keyframes, synced poses
│   ├── synthetic/                      # Synthetic validation sets (e.g. AirSim / Unreal)
│   └── ground_truth/                   # Survey-grade LiDAR point clouds / GCP coordinates
│
├── experiments/
│   ├── baseline/                       # Classical COLMAP / OpenSfM baseline pipeline
│   ├── dust3r/                         # DUSt3R pointmap alignment pipeline
│   ├── vggt/                           # Visual Geometry Grounded Transformer pipeline
│   └── fusion/                         # Hybrid AI + Classical fusion experiments
│
├── src/
│   ├── ingestion/                      # Video stream parsing and telemetry extraction
│   ├── preprocessing/                  # Frame extraction and undistortion
│   ├── quality/                        # Sharpness, exposure, and motion blur filtering
│   ├── geometry/                       # Camera pose estimation and relative orientation
│   ├── reconstruction/                 # AI & classical dense pointmap / mesh generation
│   ├── fusion/                         # Multi-view point cloud and depth fusion
│   ├── geospatial/                     # WGS84/UTM transforms, Sim(3) alignment, georeferencing
│   ├── validation/                     # Rigorous metric validation against ground truth
│   ├── uncertainty/                    # Covariance and spatial confidence estimation
│   └── api/                            # Pipeline orchestration API
│
├── tests/
│   ├── unit/                           # Module-level unit tests
│   ├── integration/                    # Interface and pipeline integration tests
│   └── regression/                     # Accuracy and performance regression tests
│
├── benchmarks/                         # Benchmark runner scripts and metrics aggregators
├── scripts/                            # Environment verification & utility scripts
└── deployment/                         # Containerization and hardware profiles
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (Recommended: $\ge 16\text{ GB}$ VRAM for AI geometry evaluation)
- PyTorch 2.2+ (for model inference phases)

### Installation
```bash
git clone https://github.com/organization/SIH26158.git
cd SIH26158
pip install -e .[dev]
```

### Running Verification Tests
```bash
pytest -v
```

---

## License & Compliance

All internal pipeline architecture and validation harnesses are licensed under Apache 2.0 unless otherwise stated in submodules. All external foundation models must adhere to their respective upstream licensing conditions as documented in [docs/research/AI_GEOMETRY_MODELS.md](docs/research/AI_GEOMETRY_MODELS.md).

