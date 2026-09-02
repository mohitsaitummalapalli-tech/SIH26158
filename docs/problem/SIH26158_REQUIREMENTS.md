# SIH26158 Problem Requirements Specification & Traceability Matrix

## 1. Executive Summary & Problem Definition

**Official Problem Statement:**
Reconstruct a georeferenced, metrically accurate, textured 3D representation (dense point cloud, 3D surface mesh, digital elevation/surface model) from a **single continuous drone video flight path**.

In standard surveying, photogrammetry relies on multi-pass cross-hatch flight plans (75–85% forward and lateral overlap) with pre-surveyed Ground Control Points (GCPs). In tactical reconnaissance, disaster response, and rapid linear infrastructure mapping, only a **single-pass moving-UAV video stream** is available.

---

## 2. Requirement Traceability & Classification Matrix

| Requirement Area | Specific Requirement Item | Classification | Official Problem Statement vs. Engineering Interpretation | Documented / Implemented In | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input Source** | Single-pass moving-UAV video (MP4/MOV/HEVC) | **MUST** | **Official Requirement:** Continuous single-pass video without cross-hatch passes | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/ingestion/` | Complete |
| **Input Telemetry** | GPS / GNSS coordinates ($\ge 1\text{ Hz}$) | **MUST** | **Official Requirement:** Geolocation coordinates associated with flight path | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/ingestion/` | Complete |
| **Input Metadata** | Flight metadata (timestamps, gimbal angles, heading) | **MUST** | **Official Requirement:** Flight state and camera orientation data | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/ingestion/` | Complete |
| **Auxiliary Input** | High-rate IMU ($\ge 50\text{ Hz}$) | **SHOULD** | **Engineering Interpretation:** Exploit for gravity alignment & VIO if available | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/ingestion/` | Complete |
| **Auxiliary Input** | Barometric altitude | **SHOULD** | **Engineering Interpretation:** Exploit for vertical drift damping | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/ingestion/` | Complete |
| **Auxiliary Input** | Pre-calibrated camera intrinsics & distortion | **SHOULD** | **Engineering Interpretation:** Exploit when known; support zero-shot if missing | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/geometry/` | Complete |
| **Auxiliary Input** | RTK / PPK differential corrections | **NICE-TO-HAVE** | **Engineering Interpretation:** Supported for centimeter-grade scale locking | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/geospatial/` | Complete |
| **Auxiliary Input** | Extensive Ground Control Points (GCPs) | **NON-GOAL** | **Official Requirement:** System must work *without* extensive physical GCP arrays | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/geospatial/` | Complete |
| **Output Geometry** | Terrain & topography (DSM/DTM) | **MUST** | **Official Requirement:** Bare-earth and elevation profiles | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/reconstruction/` | Complete |
| **Output Geometry** | Man-made structures & buildings | **MUST** | **Official Requirement:** 3D structural geometry | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/reconstruction/` | Complete |
| **Output Geometry** | Building facades | **MUST** | **Official Requirement:** Vertical facade geometry | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/reconstruction/` | Complete |
| **Output Geometry** | Rooftops & roof superstructures | **MUST** | **Official Requirement:** Roof geometry | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/reconstruction/` | Complete |
| **Output Geometry** | Roads & transportation infrastructure | **MUST** | **Official Requirement:** Road surface and linear corridors | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/reconstruction/` | Complete |
| **Output Geometry** | Vegetation & vertical obstacles | **MUST** | **Official Requirement:** Canopy, isolated trees, hazard poles | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/reconstruction/` | Complete |
| **Output Asset** | Dense point clouds (`.las`, `.laz`, `.ply`) | **MUST** | **Official Requirement:** Discrete 3D spatial points with color/normals | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/reconstruction/` | Complete |
| **Output Asset** | Textured 3D surface meshes (`.obj`, `.glb`) | **MUST** | **Official Requirement:** Continuous textured surfaces | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/reconstruction/` | Complete |
| **Property** | Georeferencing (WGS84 / UTM CRS) | **MUST** | **Official Requirement:** Aligned to real-world coordinate systems | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/geospatial/` | Complete |
| **Property** | Metric scale accuracy | **MUST** | **Official Requirement:** True Euclidean dimensions in meters | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/validation/` | Complete |
| **Property** | Suitability for 3D visualization | **MUST** | **Official Requirement:** High visual fidelity, textured rendering | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/api/` | Complete |
| **Property** | Suitability for metric measurement | **MUST** | **Official Requirement:** Distances, heights, surface areas, volumes | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/validation/` | Complete |
| **Property** | Suitability for GIS analysis | **MUST** | **Official Requirement:** Exportable to QGIS, ArcGIS, CAD | `docs/problem/SIH26158_REQUIREMENTS.md`, `src/api/` | Complete |
| **Operational Target** | Near-real-time / rapid streaming turnaround | **SHOULD** | **Official Requirement:** Low-latency turnaround vs traditional multi-hour SfM | `docs/architecture/SYSTEM_ARCHITECTURE.md`, `src/api/` | Complete |
| **Multi-pass Grids** | Requiring 80% cross-hatch overlap flights | **NON-GOAL** | **Official Requirement:** The system is explicitly targeted at *single-pass* video | `docs/problem/SIH26158_REQUIREMENTS.md`, `README.md` | Complete |

---

## 3. Operational & Technical Challenges Matrix

| Challenge ID | Challenge Description | Physical Cause | Engineering Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **CH-01** | Limited viewing angles / 1D baseline | Single linear pass provides minimal angular diversity | AI foundation models (DUSt3R/VGGT) provide learned depth priors; keyframe selection maximizes baseline |
| **CH-02** | Motion blur & rolling shutter distortion | UAV linear speed and CMOS rolling shutter readout | Laplacian variance frame filtering, optical flow sharpness metrics, rolling shutter compensation models |
| **CH-03** | Video compression artifacts | Inter-frame H.264/H.265 DCT quantization noise | Compression-aware keyframe extraction, de-blocking preprocessing |
| **CH-04** | Dynamic objects & transient motion | Moving vehicles, pedestrians, swaying trees | Epipolar outlier pruning, temporal motion segmentation, uncertainty weighting |
| **CH-05** | Illumination & auto-exposure variance | Auto-exposure/white-balance adjustments during flight | Photometric normalization, multi-band texture blending |
| **CH-06** | GNSS / sensor noise | Consumer GNSS jitter ($\pm 2\text{--}5\text{m}$) | Sliding-window spline trajectory smoothing, robust graph optimization |
| **CH-07** | Occlusions & asymmetric observation | Single pass observes only one facade side | Explicit uncertainty fields marking unobserved geometry as missing rather than hallucinating truth |
| **CH-08** | Metric accuracy without extensive GCPs | No ground survey markers on tactical paths | Telemetry-constrained $\text{Sim}(3)$ Umeyama alignment combined with barometric/VIO scaling |

---

## 4. Strict Scientific Guardrail Mandate

> **RULE 3 ENFORCEMENT:**
> NO accuracy claim is valid without:
> 1. **Named Dataset**
> 2. **Defined Ground Truth Reference**
> 3. **Defined Metric Name & Unit**
> 4. **Reproducible Calculation Method**
