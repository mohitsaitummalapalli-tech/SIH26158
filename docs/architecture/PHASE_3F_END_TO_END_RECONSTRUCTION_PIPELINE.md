# Phase 3F: End-to-End Reconstruction Pipeline Integration Contract

**Document Status**: `DESIGNED` (Pre-Implementation Architectural Contract)
**Target Baseline**: `019deb2` (Phase 3E.6 Locked Baseline)
**Contract Version**: `v1.0-frozen`

---

## 1. Executive Summary & Scientific Principles

The objective of Phase 3F is to synthesize all verified, mathematically robust subsystem components developed in Phases 1A through 3E.6 into a single-pass, executable drone-video reconstruction pipeline (`src/pipeline/`).

### Scientific Invariants & Anti-Fabrication Principles
1. **No Faking / Shortcuts**: The pipeline must execute true algorithmic stages (video decoding, keyframe selection, ORB feature extraction and matching, two-view epipolar geometry, incremental SfM, bundle adjustment, dense stereo SGBM, unprojection, voxel fusion, alpha-complex surface meshing, visibility-aware texture association, texture atlas reconstruction, and geospatial Sim(3) registration).
2. **Strict Reconstruction vs. Evaluation Boundary**:
   - **Reconstruction Inputs**: Only raw sensor frames (video/images), declared camera intrinsics, and optional declared flight telemetry are permitted into reconstruction stages.
   - **Hidden Evaluation Truth**: Ground-truth 3D CAD points, true camera poses, true depth maps, and surveyed checkpoints must remain strictly isolated outside the reconstruction pipeline and may ONLY be consumed by post-reconstruction validation harnesses.
   - **Anti-Cheat Rule**: Reconstructing via `gt_points.copy()`, `gt_mesh.copy()`, or injecting true camera poses into SfM constitutes an immediate scientific contract violation.
3. **Synthetic Image Test Authenticity & Isolation**:
   - Any end-to-end synthetic validation test MUST follow:
     $$\text{Known 3D Scene} \xrightarrow{\text{Render}} \text{Synthetic Images} \xrightarrow{\text{Pipeline}} \text{Reconstructed Model} \xrightarrow{\text{Compare}} \text{Hidden Truth}$$
   - The synthetic renderer is **test/dataset generation infrastructure**, NOT part of the reconstruction algorithm being benchmarked.
   - The renderer may know: hidden 3D scene, hidden camera poses, hidden visibility, hidden normals, hidden textures.
   - The reconstruction pipeline may **NOT** know any of those hidden values.
   - The rendered RGB images and explicitly permitted camera intrinsics / metadata are the **ONLY** data crossing the reconstruction boundary.
   - **Strictly Prohibited Shortcuts**:
     - `gt_points.copy()`
     - `gt_mesh.copy()`
     - `gt_pose.copy()`
     - `true_depth -> reconstruction`
     - `true_landmarks -> reconstruction`
     - `known_camera_pose -> SfM`
     - or any equivalent shortcut.
   - If a genuine rendering path cannot be executed with existing project capabilities/dependencies, the test status MUST be explicitly reported as `END_TO_END_SYNTHETIC_TEST = NOT_EVALUABLE`.
4. **Non-Negotiable Failure Reporting**:
   - If the actual end-to-end synthetic run cannot reconstruct successfully from the rendered images, record:
     `END_TO_END_SYNTHETIC_EXECUTION = FAILED` or `END_TO_END_SYNTHETIC_EXECUTION = NOT_EVALUABLE` with full diagnostics.
   - **Never replace the failed result with an identity/reference result.**
   - A successful test requires the actual pipeline stages to execute on the rendered images and achieve convergence.
5. **Preservation of Coordinate Frames & Units**:
   - Internal SfM and dense geometry operate strictly in `RECONSTRUCTION_UNITS` under the arbitrary monocular gauge ($\|t_{10}\| = 1.0$).
   - The pipeline MUST NOT convert reconstruction units to metres unless the geospatial stage executes with certified metric evidence (valid GNSS/telemetry baseline).
   - If telemetry is absent or unobservable, the reconstruction remains fully valid in `RECONSTRUCTION_UNITS` with `metric_scale_status = SCALE_AMBIGUOUS` and `geospatial_status = NOT_EVALUABLE`.
6. **No Design Claims as Execution Results**:
   - Status classifications must be rigorously separated: `DESIGNED`, `IMPLEMENTED`, `EXECUTED`, `MEASURED`, `NOT_EVALUABLE`.
   - An architectural design or unexecuted test script MUST NEVER be described as `PASS`.

---

## 2. Pipeline Stage Classification & Execution Policy

Every stage in the pipeline DAG is classified into one of three execution categories:
- **MANDATORY**: Critical geometric dependency. If this stage fails or receives insufficient input, the pipeline MUST immediately halt and report `PipelineStatus.FAILED` or `PipelineStatus.INSUFFICIENT_INPUT`.
- **CONDITIONAL**: Executed only if configured or if required input evidence is present. If skipped or failed, subsequent dependent stages are bypassed or adjusted, but prior valid geometric reconstruction is preserved.
- **OPTIONAL**: Diagnostic or enhancement stage (e.g. texturing). Failure or absence of data leaves the geometric model intact.

### Stage Classification Table

| Stage ID | Stage Name | Category | Prerequisite Artifacts | Output Artifact | Failure Policy |
|---|---|---|---|---|---|
| `STG-01` | Ingestion | `MANDATORY` | Raw video file / image stream | `VideoArtifact`, `CanonicalTimelineArtifact` | Halt with `INSUFFICIENT_INPUT` / `FAILED` |
| `STG-02` | Decoding | `MANDATORY` | `CanonicalTimelineArtifact` | `DecodedFramesArtifact` | Halt if zero frames successfully decoded |
| `STG-03` | Frame Intelligence | `MANDATORY` | `DecodedFramesArtifact` | `FrameQualityArtifact` | Halt if all frames fail quality thresholds |
| `STG-04` | Keyframe Selection | `MANDATORY` | `FrameQualityArtifact`, `DecodedFramesArtifact` | `KeyframeSetArtifact` | Halt with `INSUFFICIENT_INPUT` if $< 2$ keyframes |
| `STG-05` | Correspondence | `MANDATORY` | `KeyframeSetArtifact` | `CorrespondenceArtifact` | Halt if candidate matches insufficient for two-view |
| `STG-06` | Two-View Geometry | `MANDATORY` | `CorrespondenceArtifact` | `TwoViewGeometryArtifact` | Attempt alternate keyframe pairs; halt if all degenerate |
| `STG-07` | Incremental SfM | `MANDATORY` | `TwoViewGeometryArtifact`, `KeyframeSetArtifact` | `SfMArtifact` | Halt with `FAILED` if track registration fails |
| `STG-08` | Bundle Adjustment | `MANDATORY` | `SfMArtifact` | `BundleAdjustmentArtifact` | Rollback to pre-BA SfM on non-convergence; halt if corrupt |
| `STG-09` | Dense Stereo | `CONDITIONAL` | `BundleAdjustmentArtifact`, `KeyframeSetArtifact` | `DenseStereoArtifact` | If disabled/fails, bypass dense path; output sparse mesh |
| `STG-10` | Dense Point Gen | `CONDITIONAL` | `DenseStereoArtifact`, `BundleAdjustmentArtifact` | `DensePointArtifact` | Bypassed if dense stereo skipped |
| `STG-11` | Dense Fusion | `CONDITIONAL` | `DensePointArtifact` | `DenseFusionArtifact` | Bypassed if dense points unavailable |
| `STG-12` | Surface Meshing | `CONDITIONAL` | `DenseFusionArtifact` (or `BundleAdjustmentArtifact`) | `SurfaceArtifact` | Bypassed if mesh generation disabled |
| `STG-13` | Texture Assoc | `OPTIONAL` | `SurfaceArtifact`, `KeyframeSetArtifact` | `TextureAssociationArtifact` | Surface remains untextured; pipeline succeeds |
| `STG-14` | Texture Recon | `OPTIONAL` | `TextureAssociationArtifact`, `SurfaceArtifact` | `TexturedSurfaceArtifact` | Output untextured mesh on failure |
| `STG-15` | Geospatial Sim(3) | `CONDITIONAL` | `BundleAdjustmentArtifact`, Telemetry Stream | `GeospatialArtifact` | If telemetry absent/unobservable: `NOT_EVALUABLE`, keep gauge |
| `STG-16` | Final Validation | `MANDATORY` | All prior produced artifacts | `FinalReconstructionArtifact` | Validates internal consistency, units, and provenance |

---

## 3. Artifact Model & Executable Immutability Contract

### 3.1 Base Artifact Contract
Every artifact produced or consumed by any stage inherits from `PipelineArtifact` and contains mandatory cryptographic metadata:

```python
@dataclass(frozen=True)
class PipelineArtifact:
    artifact_id: str                      # Unique UUIDv4 string
    artifact_type: str                    # Type discriminator (e.g. "SfMArtifact")
    producer_stage: str                   # Stage ID that created this artifact
    input_artifact_ids: List[str]         # Cryptographic parents in DAG
    units: str                            # "RECONSTRUCTION_UNITS", "METRES", "PIXELS"
    coordinate_frame: str                 # "LOCAL_GAUGE", "TOPOCENTRIC_ENU", "ECEF"
    content_hash: str                     # SHA-256 digest of payload contents
    payload: Any                          # Immutable domain data structure
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 3.2 Executable Immutability Verification
Before any stage consumes an input artifact:
1. The consumer computes `actual_hash = sha256(serialize_canonical(artifact.payload))`.
2. The consumer verifies `actual_hash == artifact.content_hash`.
3. **If hashes mismatch**: A `ContractViolationError("Artifact tampering detected: content_hash mismatch")` is raised immediately, halting execution.
4. **Forensic Guarantee**: Any in-memory mutation of points, camera poses, or texture maps after stage completion is immediately detected and rejected.

---

## 4. Pipeline Execution Lifecycle & DAG Flow

```
[Raw Drone Video] / [Raw Image Stream]
                 │
                 ▼
       ┌──────────────────┐
       │ STG-01 Ingestion │ (MANDATORY: Builds Canonical Timeline)
       └─────────┬────────┘
                 ▼
       ┌──────────────────┐
       │ STG-02 Decoding  │ (MANDATORY: OpenCV RGB Decoded Frames)
       └─────────┬────────┘
                 ▼
       ┌────────────────────────┐
       │ STG-03 Quality Assess  │ (MANDATORY: Sharpness, Exposure, Motion Blur)
       └─────────┬──────────────┘
                 ▼
       ┌────────────────────────┐
       │ STG-04 Keyframe Select │ (MANDATORY: Greedy Marginal Gain Selection)
       └─────────┬──────────────┘
                 ▼
       ┌────────────────────────┐
       │ STG-05 Correspondence  │ (MANDATORY: ORB Extraction + Ratio Matching)
       └─────────┬──────────────┘
                 ▼
       ┌────────────────────────┐
       │ STG-06 Two-View Geomet │ (MANDATORY: Essential Matrix + Cheirality)
       └─────────┬──────────────┘
                 ▼
       ┌────────────────────────┐
       │ STG-07 Incremental SfM │ (MANDATORY: Seed Pose + PnP + Triangulation)
       └─────────┬──────────────┘
                 ▼
       ┌────────────────────────┐
       │ STG-08 Bundle Adjust   │ (MANDATORY: Gauge-Preserving Huber Refinement)
       └─────────┬──────────────┘
                 │
        ┌────────┴────────────────────────┐
        ▼                                 ▼
┌──────────────────────┐        ┌─────────────────────────┐
│ STG-09 Dense Stereo  │        │ STG-15 Geospatial Sim(3)│ (CONDITIONAL:
│ (CONDITIONAL: SGBM)  │        │ Valid telemetry present?│  Telemetry vs Gauge)
└───────┬──────────────┘        └─────────┬───────────────┘
        ▼                                 │
┌──────────────────────┐                  │
│ STG-10 Dense Points  │                  │
│ (Optical-Z Unproj)   │                  │
└───────┬──────────────┘                  │
        ▼                                 │
┌──────────────────────┐                  │
│ STG-11 Dense Fusion  │                  │
│ (Voxel-Grid Fusion)  │                  │
└───────┬──────────────┘                  │
        ▼                                 │
┌──────────────────────┐                  │
│ STG-12 Surface Mesh  │                  │
│ (Alpha Complex Mesh) │                  │
└───────┬──────────────┘                  │
        ▼                                 │
┌──────────────────────┐                  │
│ STG-13 Texture Assoc │                  │
│ (Ray-Casting BVH)    │                  │
└───────┬──────────────┘                  │
        ▼                                 │
┌──────────────────────┐                  │
│ STG-14 Texture Recon │                  │
│ (Atlas Reconstruction│                  │
└───────┬──────────────┘                  │
        │                                 │
        └────────────────┬────────────────┘
                         ▼
             ┌────────────────────────┐
             │ STG-16 Final Validation│ (MANDATORY: Provenance & Packaging)
             └───────────┬────────────┘
                         ▼
            [FinalReconstructionArtifact]
```

### 4.1 Step 13 Texture Association: Normal-Sign Independent Visibility Contract

In strict conformance with the locked Phase 3E.4 Step 3 architectural contract:
1. **Normal-Sign Independent Visibility**: Line-of-sight visibility between surface samples $P_w$ and camera optical centers $C_w$ is evaluated purely geometrically without hard back-face culling. Observations are never rejected solely because $\mathbf{n} \cdot \mathbf{v} \le 0$.
2. **Geometric Raycasting Primitives**: Calibrated pinhole projection ($K, R_{cw}, t_{cw}$), positive optical depth ($X_{c,z} > 0$), sensor margin boundaries, and finite line segment raycasting $(P_w + \epsilon \mathbf{v}, C_w - \epsilon \mathbf{v})$ via Möller–Trumbore BVH traversal.
3. **Incidence Scoring**: Surface normal orientation and incidence angles serve strictly as heuristic quality metrics during candidate scoring and multi-view atlas blending, never as hard acceptance/rejection visibility gates.
4. **Adversarial Forensic Protection (MUT-3F-26)**: Any attempt to reintroduce hard back-face culling or undocumented rejection gates in `TextureAssociationArtifact` triggers `ContractViolationError`.
5. **Regression Verification (TEST-3F-60)**: `VISIBLE_TRIANGLE_REVERSED_NORMAL` verifies that geometrically visible facets with inverted normals ($\mathbf{n} \cdot \mathbf{v} \le 0$) remain `OBSERVED` with complete mathematical provenance.

---

## 5. Reconstruction vs. Evaluation Isolation Boundary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RECONSTRUCTION ISOLATION DOMAIN                          │
│                                                                             │
│  Permitted Inputs:                                                          │
│  - Raw video streams / frame raster images                                   │
│  - Camera intrinsic matrix K, distortion coefficients D                      │
│  - Presentation timestamps (PTS)                                            │
│  - Flight telemetry observations (timestamp, lat, lon, alt, cov)            │
│                                                                             │
│  FORBIDDEN INSIDE THIS DOMAIN:                                              │
│  - True camera poses (R_cw, t_cw)                                           │
│  - True scene 3D points / mesh geometry                                     │
│  - True ground-truth depth maps                                             │
│  - Ground Control Point (GCP) 3D benchmark surveyed coordinates             │
│  - True visibility / occlusion masks                                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Reconstruction Artifacts
                                       │ (Points, Poses, Mesh, Texture, Sim(3))
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     INDEPENDENT EVALUATION DOMAIN                           │
│                                                                             │
│  - Benchmark Engine (Phase 3E.6)                                            │
│  - Hidden Ground Truth Repository (CAD, true poses, true depth)             │
│  - Metric Point-to-Point RMSE, Chamfer, Hausdorff, Scale Residuals          │
│  - Evidence Level Assessment (Level 0, Level 1, Level 2, Level 3)           │
└─────────────────────────────────────────────────────────────────────────────┘
```

Any penetration of Hidden Ground Truth into the Reconstruction Isolation Domain triggers an immediate `DataLeakageError` (verified forensically in `MUT-3F-10`).

---

## 6. Failure Taxonomy & Handling

The pipeline implements an explicit, typed failure taxonomy:

```python
class PipelineStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    INSUFFICIENT_INPUT = "INSUFFICIENT_INPUT"
    CONTRACT_VIOLATION = "CONTRACT_VIOLATION"
    DEGRADED_COMPLETION = "DEGRADED_COMPLETION"
```

1. **`INSUFFICIENT_INPUT`**:
   - Zero video frames decoded (`STG-02`).
   - All frames discarded by frame quality filter (`STG-03`).
   - Fewer than 2 usable keyframes selected (`STG-04`).
   - Insufficient 2D-2D feature correspondences ($< 15$) across all keyframe pairs (`STG-05`).
2. **`FAILED`**:
   - Degenerate epipolar geometry across all candidate keyframe pairs (`STG-06`).
   - SfM fails to register minimum required cameras or triangulate landmark tracks (`STG-07`).
   - Numerical corruption (NaN/Inf) encountered in core optimization algorithms.
3. **`CONTRACT_VIOLATION`**:
   - Artifact content hash mismatch (immutability breach).
   - Hidden ground-truth key leakage into reconstruction inputs.
   - Non-monotonic frame timestamps or frame reordering.
   - Uncertified conversion from `RECONSTRUCTION_UNITS` to metres.
4. **`DEGRADED_COMPLETION`**:
   - Core geometric SfM/BA succeeds, but optional dense stereo or texturing fails. The pipeline preserves and exports the sparse or untextured geometric model with explicit warning diagnostics.

---

## 7. Determinism & Provenance Specification

Every pipeline execution records a `PipelineRunMetadata` block:
- `run_id`: Unique execution identifier.
- `git_commit`: Current software commit (`019deb2`).
- `config_hash`: SHA-256 digest of serialized `PipelineConfig`.
- `random_seed`: Configured random seed (default: `42`).
- `execution_timestamp_utc`: ISO 8601 start timestamp.
- `stage_records`: List of `StageExecutionRecord` entries tracking individual start/end times, durations, and output artifact hashes.

---

## 8. Integration & Forensic Test Plan

### 8.1 Systematic Integration Scenarios (`tests/integration/test_phase3f_pipeline.py`)
1. `TEST-3F-01`: Video ingestion $\to$ canonical timeline construction.
2. `TEST-3F-02`: Timeline $\to$ frame decoding into canonical RGB.
3. `TEST-3F-03`: Frames $\to$ quality assessment & intelligence metrics.
4. `TEST-3F-04`: Keyframe selection $\to$ ORB feature extraction & descriptor matching.
5. `TEST-3F-05`: Correspondences $\to$ robust two-view geometry & cheirality check.
6. `TEST-3F-06`: Two-view geometry $\to$ incremental SfM camera registration & triangulation.
7. `TEST-3F-07`: Incremental SfM $\to$ global bundle adjustment refinement.
8. `TEST-3F-08`: Bundle adjustment $\to$ dense stereo disparity estimation.
9. `TEST-3F-09`: Dense stereo $\to$ dense 3D point cloud generation.
10. `TEST-3F-10`: Dense point cloud $\to$ multi-view voxel-grid fusion.
11. `TEST-3F-11`: Fused points $\to$ alpha-complex surface reconstruction.
12. `TEST-3F-12`: Surface mesh $\to$ visibility-aware texture association.
13. `TEST-3F-13`: Texture association $\to$ texture atlas reconstruction.
14. `TEST-3F-14`: Telemetry present $\to$ valid geospatial Sim(3) metric transformation.
15. `TEST-3F-15`: Telemetry absent $\to$ geospatial stage emits `NOT_EVALUABLE` while geometric reconstruction succeeds in `RECONSTRUCTION_UNITS`.
16. `TEST-3F-16`: Full pipeline execution on authentic synthetic image sequence (rendered images as sole input).
17. `TEST-3F-17`: Artifact immutability verification across all inter-stage handoffs.
18. `TEST-3F-18`: Deterministic rerun verification (bitwise or identical structure across runs).

### 8.2 Adversarial Forensic Mutations (`tests/integration/test_phase3f_forensic.py`)
1. `MUT-3F-01`: Injecting true camera pose into SfM triggers `DataLeakageError`.
2. `MUT-3F-02`: Injecting true depth maps into dense reconstruction triggers `DataLeakageError`.
3. `MUT-3F-03`: Injecting true CAD mesh into surface meshing stage triggers `DataLeakageError`.
4. `MUT-3F-04`: Skipping a failed mandatory stage while marking pipeline `SUCCESS` triggers `ContractViolationError`.
5. `MUT-3F-05`: Substituting SfM output with hidden ground-truth landmarks triggers `ContractViolationError`.
6. `MUT-3F-06`: Converting reconstruction units to metres without valid metric evidence triggers `ContractViolationError`.
7. `MUT-3F-07`: Claiming GNSS fitting residual as independent accuracy triggers `ContractViolationError`.
8. `MUT-3F-08`: Mutating artifact payload after creation triggers `ContractViolationError` (immutability check).
9. `MUT-3F-09`: Presenting out-of-order temporal frames triggers `ContractViolationError`.
10. `MUT-3F-10`: Feeding validation reference objects into reconstruction stages triggers `DataLeakageError`.
