# Canonical Flight Dataset Architecture & Pipeline Specification

## 1. Executive Summary & Objective

The `CanonicalFlightDataset` is the primary integrated, immutable, provenance-preserving data structure that serves as the single input artifact for all downstream processing stages (Frame Quality, Keyframing, Pose Estimation, Bundle Adjustment, and Reconstruction).

It unifies:
1. **Video Container & Presentation Timeline** (`CanonicalTimeline`, `VideoMetadata`)
2. **Standardized Spatial Telemetry** (`CanonicalTelemetryStream`)
3. **Continuous Temporal Synchronization** (`SynchronizedTrajectory`)
4. **Metric Cartesian Normalization** (`NormalizedTelemetryStream`, `GeodeticOrigin`)

> **SCIENTIFIC INTEGRITY STATEMENT:**
> The `CanonicalFlightDataset` represents validated, synchronized, and normalized telemetry observations and video frame metadata. It does **NOT** imply:
> - Accurate optical camera pose
> - High-precision ground truth GNSS accuracy
> - Completed metric 3D reconstruction
> - Successful 7-DoF georeferencing alignment
>
> Downstream computer vision and bundle adjustment stages must establish optical geometry independently rather than treating uncorrected telemetry as ground truth.

---

## 2. Integrated Dataset Pipeline Architecture

```
  Video File (MP4/MOV)              Telemetry File (SRT/CSV/KLV)
           │                                      │
           ▼                                      ▼
     [VideoSource]                        [TelemetryAdapter]
           │                                      │
           ▼                                      ▼
   CanonicalTimeline                   CanonicalTelemetryStream
           │                                      │
           └──────────────────┬───────────────────┘
                              │
                              ▼
               [TemporalSynchronizationEngine]
                              │
                              ▼
                   [CoordinateNormalizer]
                              │
                              ▼
                   ╔══════════════════════╗
                   ║ CanonicalFlightDataset ║
                   ╚══════════════════════╝
```

---

## 3. Dataset Contract & Frame-Level Association

Each frame observation in `CanonicalFlightDataset.frame_observations` provides an immutable, one-to-one record:

| Field | Type | Description |
| :--- | :--- | :--- |
| `frame_id` | `str` | Unique video frame identifier (e.g. `"frame_0000"`) |
| `frame_index` | `int` | 0-indexed presentation sequence number |
| `video_timestamp_seconds` | `float` | Exact presentation timestamp from video timebase origin |
| `is_keyframe` | `bool` | Flag indicating intra-coded / designated keyframe |
| `sync_status` | `SyncStatus` | `EXACT`, `INTERPOLATED`, `OFFSET_APPLIED`, `EXTRAPOLATED`, `UNSYNCHRONIZED`, `OUT_OF_RANGE` |
| `is_synchronized` | `bool` | True if valid spatial coordinates were successfully aligned |
| `original_position` | `Optional[TelemetryPosition]` | Raw geodetic coordinates $(\phi, \lambda, h)$ and vertical reference |
| `ecef_position` | `Optional[ECEFCoordinates]` | Global metric Cartesian $(X, Y, Z)$ coordinates (EPSG:4978) |
| `enu_position` | `Optional[ENUCoordinates]` | Local metric tangent $(e, n, u)$ coordinates relative to `origin` |
| `orientation` | `Optional[TelemetryOrientation]` | Gimbal angles and body attitude (SLERP-interpolated) |
| `velocity` | `Optional[TelemetryVelocity]` | Ground speed, climb rate, and velocity components $(v_N, v_E, v_D)$ |
| `quality` | `Optional[TelemetryQuality]` | GNSS fix type, satellite count, DOP metrics |
| `source_record_indices` | `List[int]` | Indices of raw telemetry samples that contributed to observation |

---

## 4. Dataset Validity States

| Status | Definition | Typical Cause |
| :--- | :--- | :--- |
| **`VALID`** | Video timeline and telemetry aligned; 100% of frames have synchronized metric coordinates. | Complete, continuous flight log with valid ellipsoidal GNSS. |
| **`PARTIALLY_VALID`** | Video and telemetry aligned, but some frames are out of range or telemetry gaps exist, or optional sensors (IMU/velocity) are missing. | Telemetry started after video recording, or consumer drone log without IMU. |
| **`INVALID`** | Structural failure: missing timeline, zero telemetry samples, zero synchronized frames, or fatal schema errors. | Unreadable file, corrupt header, or disjoint timebases. |

---

## 5. Missing Optional Data Policy

The dataset architecture strictly adheres to non-fabrication principles:
- If IMU attitude $(\psi, \theta, \phi)$ is absent: `orientation = None`. The system records a `MISSING_OPTIONAL_IMU` issue with `INFO` severity. No fake zero-degree angles are injected.
- If ground speed is absent: `velocity = None`.
- If GNSS fix quality is absent: `quality = None`.

---

## 6. Metadata Serialization & Manifest Format

The dataset manifest is serialized to JSON (`dataset.to_json()`) containing:
- Provenance manifest (source paths, SHA-256 hashes, software version)
- Video timing summary & metadata
- Local ENU origin definition
- Diagnostic validation issues
- Frame-level telemetry and coordinate records

> **NOTE:** Raw video pixel buffers are never embedded in the metadata manifest. The manifest references frames by logical sequence index and unique identifier, allowing fast, reproducible serialization and auditability.
