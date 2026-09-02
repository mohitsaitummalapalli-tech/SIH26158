# Provisional System Architecture Specification (Phase 0 Baseline)

> **NOTICE: PROVISIONAL RESEARCH ARCHITECTURE**
> This architecture represents the research blueprint. All modular boundaries, contracts, and algorithmic components are **provisional and subject to empirical validation** in Phase 1 benchmarking.

---

## 1. End-to-End Pipeline Architecture Flow

```
[ Continuous Drone Video ] + [ GPS/IMU Telemetry ]
                 │
                 ▼
     ┌────────────────────────────────────────────────────────┐
     │ 1. Ingestion & Temporal Synchronization Engine         │
     │    (Demuxing, UTC clock sync, GPS spline interp)      │
     └───────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
     ┌────────────────────────────────────────────────────────┐
     │ 2. Preprocessing & Rolling Shutter Compensation        │
     │    (Frame extraction, lens undistortion, motion model) │
     └───────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
     ┌────────────────────────────────────────────────────────┐
     │ 3. Frame Quality & Dynamic Object Masking              │
     │    (Laplacian sharpness, exposure, motion seg mask)   │
     └───────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
     ┌────────────────────────────────────────────────────────┐
     │ 4. Baseline-Adaptive Keyframe Selection                │
     │    (Parallax baseline scoring, overlap preservation)   │
     └───────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
     ┌────────────────────────────────────────────────────────┐
     │ 5. Relative Camera Pose & Intrinsics Estimation        │
     │    (Visual-Inertial initialization, zero-shot priors)  │
     └───────────────────────────┬────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
     ┌───────────────────────────────┐ ┌───────────────────────────────┐
     │ 6. Classical Geometry Engine  │ │ 7. AI Geometry Engine         │
     │    (SIFT/ORB keypoints,       │ │    (DUSt3R/VGGT direct dense  │
     │     Epipolar RANSAC filter)   │ │     pointmaps + confidences)  │
     └───────────────┬───────────────┘ └───────────────┬───────────────┘
                     │                                 │
                     └───────────────┬─────────────────┘
                                     │
                                     ▼
     ┌────────────────────────────────────────────────────────┐
     │ 8. Multi-View Geometry & Scale Fusion Graph            │
     │    (Sliding-window pose graph, GNSS Sim(3) constraint) │
     └───────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
     ┌────────────────────────────────────────────────────────┐
     │ 9. Dense Point Cloud Generation & Outlier Pruning      │
     │    (Confidence thresholding, normal estimation)        │
     └───────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
     ┌────────────────────────────────────────────────────────┐
     │ 10. Continuous Surface Reconstruction (Meshing)        │
     │     (Screened Poisson, TSDF voxel fusion, Alpha-wraps) │
     └───────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
     ┌────────────────────────────────────────────────────────┐
     │ 11. UV Texture Projection & Photometric De-ghosting    │
     │     (Multi-band exposure blending, view selection)     │
     └───────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
     ┌────────────────────────────────────────────────────────┐
     │ 12. Georeferencing & Spatial Coordinate Transformation │
     │     (Local ENU ──► ECEF ──► WGS84 UTM Projection)      │
     └───────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
     ┌────────────────────────────────────────────────────────┐
     │ 13. Spatial Uncertainty & Extrapolation Quantification │
     │     (Per-vertex confidence, unobserved facade tagging) │
     └───────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
     ┌────────────────────────────────────────────────────────┐
     │ 14. Strict Accuracy & Provenance Validation Harness    │
     │     (Ground-truth LiDAR/checkpoint evaluation)         │
     └───────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
     ┌────────────────────────────────────────────────────────┐
     │ 15. 3D Viewer & Asset Export Gateway                   │
     │     (.LAS, .PLY, .OBJ, .GLB, GeoTIFF DSM rasters)      │
     └────────────────────────────────────────────────────────┘
```

---

## 2. Stage-by-Stage Contract Specifications

### Stage 1: Ingestion, Synchronization & Coordinate Normalization
- **Input:** Video container (`.mp4`, `.mov`), telemetry logs (`.srt`, `.csv`, KLV streams).
- **Output:** `CanonicalFlightDataset` (immutable, provenance-preserving aggregate combining `CanonicalTimeline`, `CanonicalTelemetryStream`, `SynchronizedTrajectory`, `NormalizedTelemetryStream`, and per-frame `CanonicalFrameObservation` records).
- **Dependencies:** `ISOBMFFParser`, `DJISRTAdapter`, `GenericCSVAdapter`, `TemporalSynchronizationEngine`, `CoordinateNormalizer`, `CanonicalDatasetBuilder`.
- **Failure Modes:** Telemetry gap exceeding `max_interpolation_gap_seconds` (`UNSYNCHRONIZED`), out-of-range timestamps (`OUT_OF_RANGE`), incompatible vertical datums (`INCOMPATIBLE_REFERENCE`).
- **Metrics:** Frame synchronization success ratio, interpolation bracketing interval, ENU coordinate round-trip precision ($< 1\text{ mm}$).

### Stage 2: Frame Decoding, Preprocessing & Rolling Shutter Compensation
- **Input:** `CanonicalFlightDataset` (referenced frames & telemetry).
- **Processing:** `FrameDecoder` (`DatasetFrameDecoder`, `OpenCVFrameDecoder`) streaming random-access and sequential decoding to `DecodedFrame` (canonical RGB `uint8` arrays), followed by lens undistortion and readout time compensation.
- **Output:** Decoded, lens-undistorted, rolling-shutter-corrected RGB image arrays.
- **Dependencies:** `FrameDecoder`, `OpenCVFrameDecoder`, Lens calibration model (OpenCV pinhole/Brown-Conrady).
- **Failure Modes:** File access failure, corrupted frames, decode timeout, inaccurate distortion parameters.
- **Metrics:** Decode throughput (FPS), reprojection residual of straight lines.

### Stage 3: Frame Quality, Motion, Photometric, Dynamic Scene & Redundancy Diagnostics
- **Input:** `DecodedFrame` (canonical RGB `uint8` arrays) + temporal sequence context ($t-1, t, t+1$) + ENU / attitude telemetry.
- **Processing:** 
  1. `FrameQualityAnalyzer` evaluating single-frame image statistics (ITU-R BT.601 luminance, shadow/highlight clipping, contrast spreads, Laplacian/Tenengrad sharpness, spatial grid tiles, and high-frequency residual proxies).
  2. `TemporalMotionAnalyzer` evaluating sequence-aware motion kinematics (Farnebäck dense optical flow, $\Delta t$-normalized velocity, spatial grid tiling, directional coherence, and motion-blur indicators).
  3. `PhotometricAnalyzer` evaluating spatial illumination uniformity, dynamic range, clipping, and frame-to-frame Bhattacharyya histogram transitions.
  4. `DynamicSceneAnalyzer` evaluating detector-agnostic candidate regions (`DynamicRegionProvider`), relative motion discrepancy, and temporal persistence.
  5. `FrameRedundancyViewpointAnalyzer` evaluating pair-wise visual appearance correlation (ZNCC), ORB feature overlap, spatial match distributions, and trajectory-based baseline/orientation changes.
- **Output:** 
  - `FrameQualityReport` (`QualityStatus`: `VALID`, `DEGRADED`, `SEVERELY_DEGRADED`, `ANALYSIS_ERROR`).
  - `TemporalMotionBlurReport` (`MotionCategory`: `POTENTIAL_CAMERA_MOTION`, `POTENTIAL_LOCAL_MOTION`, `MIXED_MOTION`, `LOW_APPARENT_MOTION`, `INSUFFICIENT_EVIDENCE`).
  - `PhotometricStabilityReport` (`SpatialIlluminationPattern`: `UNIFORM`, `GRADIENT`, `LOCALIZED_BRIGHTNESS`, `LOCALIZED_DARKNESS`, `MIXED`; `PhotometricChangeCategory`: `STABLE`, `POTENTIAL_EXPOSURE_TRANSITION`, `POTENTIAL_LOCAL_ILLUMINATION_CHANGE`).
  - `DynamicSceneReport` (`DynamicEvidenceCategory`: `STATIC_EVIDENCE`, `POSSIBLY_DYNAMIC`, `DYNAMIC_EVIDENCE`, `INSUFFICIENT_EVIDENCE`).
  - `FrameRedundancyReport` (`FramePairRelation` with visual similarity, feature match ratio, convex hull coverage, trajectory baseline meters, and orientation change).
- **Dependencies:** `FrameQualityAnalyzer`, `TemporalMotionAnalyzer`, `PhotometricAnalyzer`, `DynamicSceneAnalyzer`, `FrameRedundancyViewpointAnalyzer`, OpenCV.
- **Failure Modes:** Missing adjacent frames, temporal gaps $> 2.0\text{ s}$, low-texture flat scenes, non-finite pixels.
- **Metrics:** Laplacian variance ($\text{Var}(\nabla^2 Y)$), apparent velocity ($\text{px/s}$), directional coherence, Bhattacharyya distance ($D_B$), relative motion discrepancy ($\Delta \mathbf{v}_{\mathcal{R}}$), visual similarity (ZNCC), match ratio, trajectory baseline ($B_{\text{trajectory}}$).

### Stage 4: Coverage-Aware Keyframe Selection
- **Input:** `DecodedFrame` stream + Stage 3 diagnostic reports (`FrameQualityReport`, `TemporalMotionBlurReport`, `PhotometricStabilityReport`, `DynamicSceneReport`, `FrameRedundancyReport`) + Synchronized ENU trajectory / attitude metadata.
- **Processing:** `CoverageAwareKeyframeSelector` executing two-stage filtering:
  1. Stage A: Hard Safety Gate (decode status verification, `SEVERELY_DEGRADED` quality exclusion, extreme dynamic contamination filter).
  2. Stage B: Greedy Marginal-Gain Selection (temporal boundary anchoring, minimum spacing $\Delta t \ge 0.25\text{ s}$, maximum gap constraints, trajectory/orientation diversity proxies, feature novelty gains, dynamic risk penalties).
- **Output:** `KeyframeSelectionResult` containing selected keyframe IDs/indices, timestamps, machine-readable `primary_reason` codes (`TEMPORAL_GAP_COVERAGE`, `TRAJECTORY_DIVERSITY`, `ORIENTATION_DIVERSITY`, `FEATURE_NOVELTY`, `INITIAL_ANCHOR`, `FINAL_ANCHOR`, `FALLBACK`), and deprioritized candidate diagnostics.
- **Dependencies:** `CoverageAwareKeyframeSelector`, `KeyframeSelectionConfig`.
- **Failure Modes:** Insufficient candidates (handled via deterministic temporal fallback), excessive dynamic contamination.
- **Metrics:** Selection count, sequence reduction ratio, mean keyframe baseline, temporal gap distribution.

### Stage 5: Classical Geometry & Incremental Structure-from-Motion (Phase 3)
- **Sub-stages:**
  - **Phase 3A (`features.py`):** ORB feature detection, descriptor extraction, mutual & ratio test matching (`FeatureMatchResult`).
  - **Phase 3B (`two_view.py`):** Robust two-view geometry, essential matrix estimation ($\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$), RANSAC verification, cheirality validation (`TwoViewGeometryResult`).
  - **Phase 3C (`sfm.py`):** Incremental SfM, candidate evaluation, PnP camera registration, multi-view DLT triangulation, gauge fixing (`SparseReconstructionResult`).
  - **Phase 3D (`bundle_adjustment.py`):** Global Bundle Adjustment (`SparseReconstructionResult` $\to$ Huber-loss nonlinear optimization $\to$ Refined `SparseReconstructionResult`).
- **Input:** `KeyframeSelectionResult` selected keyframe stream + camera calibration priors.
- **Output:** `TwoViewGeometryResult`, `SparseReconstructionResult`, and `BundleAdjustmentResult` containing refined camera poses $\{\mathbf{R}_i, \mathbf{t}_i\}$, 3D point landmark tracks $\mathbf{X}_j$, and reprojection RMSE statistics in reconstruction units.
- **Dependencies:** `CameraIntrinsics`, `ExtrinsicPose`, `TwoViewGeometryResult`, `SparseReconstructionResult`, `BundleAdjustmentResult`, `GeometryMathContracts`.
- **Failure Modes:** Explicit failure taxonomy: `INSUFFICIENT_MATCHES`, `GEOMETRIC_VERIFICATION_FAILED`, `PURE_ROTATION_RISK`, `WEAK_BASELINE`, `CHEIRALITY_VIOLATION`, `BUNDLE_ADJUSTMENT_DIVERGED`, `POST_OPTIMIZATION_VALIDATION_FAILED`.
- **Metrics:** Inlier ratio, mean reprojection RMSE ($< 2.0\text{ px}$), 90th percentile reprojection error, gauge preservation in reconstruction units. Monocular reconstruction scale remains uncalibrated (`SCALE_AMBIGUOUS`) without metric fusion.

### Stage 6 & 7: Dense Multi-View Stereo & Foundation AI Geometry Engines
- **Input:** `SparseReconstructionResult` + selected keyframe image pairs (`DenseMVSInput`).
- **Output:** Dense Multi-View Stereo point clouds (`DenseMVSOutput`) + Dense pixel-aligned 3D pointmaps $\mathbf{X}^{j,i}$ with confidence maps $\mathbf{C}^{j,i}$ (AI).
- **Dependencies:** Classical PatchMatch Stereo / MVS + Foundation AI backbones.
- **Failure Modes:** `MVS_DEPTH_ESTIMATION_FAILED`, textureless stereo matching degradation.
- **Metrics:** Dense point count ($\ge 50,000$), depth confidence consistency, pointmap alignment residual.

### Stage 8: Multi-View Geometry & Scale Fusion Graph
- **Input:** Pairwise pointmaps, relative poses, and synchronized GNSS track.
- **Output:** Globally registered pointmap coordinates in unified metric frame.
- **Dependencies:** Sliding-window graph optimizer ($\text{Sim}(3)$ constraints).
- **Failure Modes:** Scale drift accumulation along linear trajectory; gauge bending.
- **Metrics:** Graph edge residual error, trajectory scale consistency.

### Stage 9: Dense Point Cloud Generation & Outlier Pruning
- **Input:** Registered multi-view pointmaps.
- **Output:** Clean, merged dense point cloud with RGB color, normals, and uncertainty scalars.
- **Dependencies:** Statistical Outlier Removal (SOR), normal estimation via local plane fitting.
- **Failure Modes:** Ghost point clusters, vegetation noise.
- **Metrics:** Point density ($pts/m^2$), normal consistency angle.

### Stage 10: Continuous Surface Reconstruction (Meshing)
- **Input:** Merged oriented point cloud.
- **Output:** 3D triangular surface mesh (`.obj`, `.ply`).
- **Dependencies:** Screened Poisson Surface Reconstruction / TSDF grid.
- **Failure Modes:** Bubble artifacts in unobserved regions; non-manifold edges.
- **Metrics:** Vertex count, face count, manifold validity check.

### Stage 11: UV Texture Projection & Photometric De-ghosting
- **Input:** 3D surface mesh + calibrated keyframes.
- **Output:** UV texture atlas with seamless photometric blending.
- **Dependencies:** UV unwrapper, multi-band exposure compensator.
- **Failure Modes:** Texture ghosting on vertical facades; exposure boundary seams.
- **Metrics:** Reprojection photometric MSE.

### Stage 12: Georeferencing & Spatial Coordinate Transformation
- **Input:** Local metric mesh/point cloud + GNSS track.
- **Output:** Georeferenced assets in target Projected CRS (WGS84 UTM EPSG:326XX).
- **Dependencies:** Umeyama $\text{Sim}(3)$ solver, Proj/GDAL coordinate transformation.
- **Failure Modes:** Incorrect UTM zone selection; geoid height omission.
- **Metrics:** Transformation residual RMSE against GNSS anchors.

### Stage 13: Spatial Uncertainty & Extrapolation Quantification
- **Input:** Reconstructed geometry + multi-view observation counts.
- **Output:** Per-vertex spatial uncertainty scalar field; explicit unobserved facade tags.
- **Dependencies:** Ray-cast visibility aggregator, neural confidence field.
- **Failure Modes:** Inpainted geometry falsely marked as confident.
- **Metrics:** Unobserved vertex ratio, spatial variance trace.

### Stage 14: Strict Accuracy & Provenance Validation Harness
- **Input:** Reconstructed point cloud/mesh + ground truth LiDAR / RTK checkpoints.
- **Output:** Verified `AccuracyMetric` records with full cryptographic SHA-256 provenance.
- **Dependencies:** Strict Metric Validator (Rule 3 enforcement).
- **Failure Modes:** Rejection of unprovenanced or mathematically impossible claims.
- **Metrics:** ATE RMSE, Chamfer Distance, Checkpoint RMSE XYZ, Completeness at $\tau$.

### Stage 15: 3D Viewer & Asset Export Gateway
- **Input:** Validated georeferenced point cloud, textured mesh, DSM raster.
- **Output:** Standard geospatial files (`.las`, `.laz`, `.obj`, `.glb`, GeoTIFF) and WebGL stream.
- **Dependencies:** Asset serializers, WebGL viewer interface.
- **Failure Modes:** Export format schema invalidity.
- **Metrics:** Serialization time, file size efficiency.

---

## 3. Coordinate Frame Conventions

To prevent spatial ambiguity across modules, the following standards are strictly enforced:

| Subsystem | Coordinate Frame | Convention | Axes Definition |
| :--- | :--- | :--- | :--- |
| **Camera Sensor** | Image Pixel Frame | 2D $(u, v)$ | $u \to \text{Right (cols)}$, $v \to \text{Down (rows)}$, Origin at top-left |
| **Camera Optical** | OpenCV Camera Frame | Right-Handed 3D | $X \to \text{Right}$, $Y \to \text{Down}$, $Z \to \text{Forward (optical axis)}$ |
| **Drone Body** | UAV Body Frame | Right-Handed 3D | Forward-Right-Down (FRD) or North-East-Down (NED) |
| **Local Spatial** | Topocentric ENU | Right-Handed 3D | $X \to \text{East}$, $Y \to \text{North}$, $Z \to \text{Up (ellipsoidal normal)}$ |
| **Geocentric** | ECEF (EPSG:4978) | Right-Handed 3D | Origin at Earth Center of Mass, $Z$ along polar axis |
| **Geospatial Map**| WGS84 UTM | Projected Metric | $E \to \text{Easting (m)}$, $N \to \text{Northing (m)}$, $h \to \text{Height (m)}$ |
