# Environment & Python Version Compatibility Audit

## 1. Executive Summary

This document audits the software runtime environment, Python version constraints, and dependency compatibility across the SIH26158 pipeline.

---

## 2. Declared vs Runtime Python Version Analysis

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Declared Constraint (`pyproject.toml`)** | `>=3.10, <3.13` | CPython 3.10, 3.11, 3.12 supported. |
| **Current Execution Host** | Python 3.14.7 | Active development interpreter. |
| **Compatibility Status** | **Out of Declared Target Range** | Preserving `>=3.10, <3.13` in `pyproject.toml`. |

### Technical Rationale for Declared Boundary (`<3.13`):
1. **PyTorch & CUDA Ecosystem**:
   - PyTorch (`torch>=2.2.0`), `torchvision`, and NVIDIA CUDA 12.x wheels officially target Python 3.10, 3.11, and 3.12.
2. **Deep Learning / 3D Reconstruction Extensions**:
   - Downstream models planned for Stages 5–8 (DUSt3R, RoMa, CroCo, PyTorch3D, VGGT, SpConv) require custom C++/CUDA kernel compilations (`pybind11`, `torch.utils.cpp_extension`) that fail to build on Python 3.14 due to C-API internal refactorings.
3. **Geospatial & Demuxing Libraries**:
   - `rasterio`, `gdal`, and `av` (PyAV) have stabilized binary wheels for Python 3.10–3.12.

---

## 3. Media Decoding & OpenCV Backend Compatibility

- **OpenCV**: Fully operational across Python 3.10–3.14 (`cv2==5.0.0` or `4.x`).
- **Media Foundation / MSMF / FFMPEG**:
  - `cv2.VideoCapture` leverages Windows Media Foundation (`cv2.CAP_MSMF`) or FFMPEG dynamically.
  - Seeking semantics (`CAP_PROP_POS_FRAMES`) differ across container demuxers on complex GOP/VFR streams.
- **Reference Architecture**:
  - Sequential streaming (`iter_frames()`) is the authoritative scientific reference path.
  - Random-access seeking is an optimization path.

---

## 4. Policy Decision

- **Rule**: Do **NOT** modify `pyproject.toml` to expand `requires-python` to 3.14. The constraint `>=3.10, <3.13` remains strictly enforced for production Docker and CI/CD targets.
