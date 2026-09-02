# Experiment Protocol Forensic Audit

This document records the verification of the 4-way reconstruction benchmark protocol to ensure fair, unbiased, and reproducible experimental comparisons.

---

## 1. Protocol Fairness Checklist

| Fairness Dimension | Audit Verification | Status |
| :--- | :--- | :--- |
| **Input Data Equivalence** | All pipelines receive identical video frames, resolutions, and timestamps. | **SATISFIED** |
| **Coordinate Alignment Uniformity** | Umeyama $\text{Sim}(3)$ alignment is applied with identical GNSS priors across all pipelines before computing metric errors. | **SATISFIED** |
| **Ground Truth Explicitness** | Protocol explicitly designates which metrics require ground truth (LiDAR, RTK GPS, Checkpoint GCPs) vs internal metrics. | **SATISFIED** |
| **Visual vs Metric Separation** | Prohibits visual quality metrics (PSNR, SSIM) from substituting for spatial metric accuracy. | **SATISFIED** |
| **Failure Penalty Handling** | Failed or fragmented pipeline runs cannot be omitted; they are recorded as partial failures with completeness penalties. | **SATISFIED** |
| **Hardware & Seed Reproducibility** | Manifest mandates recording of GPU model, VRAM, CUDA version, library pins, and deterministic random seeds. | **SATISFIED** |
