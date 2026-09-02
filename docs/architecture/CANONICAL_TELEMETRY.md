# Canonical Telemetry Architecture & Data Contracts

## 1. Objective & Purpose

The Canonical Telemetry subsystem defines vendor-agnostic, mathematically rigorous data contracts for spatial, kinematic, and attitude telemetry observations extracted from drone flight logs (DJI SRT, KLV, ArduPilot/PX4 CSV, EXIF).

> **ARCHITECTURAL PRINCIPLE:**
> The canonical data contracts decouple the internal reconstruction and georeferencing pipeline from proprietary vendor log quirks, time formats, and reference frames. No vendor-specific assumptions are baked into core contracts.

---

## 2. Canonical Telemetry Schema Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             TelemetryRecord                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  timestamp: float (seconds, ≥ 0.0)                                          │
│  timestamp_semantics: TimestampSemantics (VIDEO_RELATIVE, UTC, GPS_TIME, …) │
│  timestamp_utc: Optional[str] (ISO 8601 UTC)                                │
│  extra_metadata: Dict[str, Any]                                             │
└────────┬───────────────────┬──────────────────┬──────────────┬──────────────┘
         │                   │                  │              │
         ▼                   ▼                  ▼              ▼
┌──────────────────┐ ┌───────────────┐ ┌────────────────┐ ┌──────────────────┐
│TelemetryPosition │ │TelemetryOrient│ │TelemetryVeloc  │ │TelemetryQuality  │
├──────────────────┤ ├───────────────┤ ├────────────────┤ ├──────────────────┤
│ latitude_deg     │ │ heading_deg   │ │ speed_mps      │ │ fix_type         │
│ longitude_deg    │ │ pitch_deg     │ │ north_vel_mps  │ │ satellites_vis   │
│ altitude_meters  │ │ roll_deg      │ │ east_vel_mps   │ │ hdop / vdop      │
│ altitude_ref     │ │ gimbal_pitch  │ │ down_vel_mps   │ │ horiz_acc_meters │
│ position_ref     │ │ gimbal_roll   │ │ climb_rate_mps │ │ vert_acc_meters  │
│                  │ │ gimbal_yaw    │ │                │                    │
└──────────────────┘ └───────────────┘ └────────────────┘ └──────────────────┘
         │
         ▼
┌──────────────────┐
│TelemetryProven   │
├──────────────────┤
│ source_type      │ (e.g. "dji_srt", "ardupilot_csv", "embedded_klv")
│ source_identifier│ (file path, stream URI, device serial)
│ record_index     │ (original 0-based sample position)
│ extraction_method│ ("CanonicalTelemetry_v1.0")
│ source_checksum  │ (SHA256 of raw source telemetry log)
└──────────────────┘
```

---

## 3. Standard Units & Conventions

To prevent dimensional mixing errors, all telemetry adapter outputs must conform to these standard SI and angular units:

| Quantity | Canonical Unit | Valid Mathematical Range | Internal Representation |
| :--- | :--- | :--- | :--- |
| **Latitude** | Decimal Degrees | $[-90.0^\circ, +90.0^\circ]$ | `float` (IEEE 754 64-bit) |
| **Longitude** | Decimal Degrees | $[-180.0^\circ, +180.0^\circ]$ | `float` (IEEE 754 64-bit) |
| **Altitude / Elevation** | Meters ($\text{m}$) | $\mathbb{R}$ (finite float) | `float` (IEEE 754 64-bit) |
| **Linear Velocity / Speed**| Meters/sec ($\text{m/s}$) | $[0, +\infty)$ | `float` (IEEE 754 64-bit) |
| **Drone Heading / Yaw** | Degrees ($^\circ$) | $[0.0^\circ, 360.0^\circ)$ ($0^\circ = \text{True North}$) | `Optional[float]` |
| **Drone Body Pitch** | Degrees ($^\circ$) | $[-90.0^\circ, +90.0^\circ]$ ($+ = \text{Nose Up}$) | `Optional[float]` |
| **Drone Body Roll** | Degrees ($^\circ$) | $[-180.0^\circ, +180.0^\circ]$ ($+ = \text{Right Wing Down}$) | `Optional[float]` |
| **Gimbal Pitch** | Degrees ($^\circ$) | $[-90.0^\circ, +30.0^\circ]$ ($0^\circ = \text{Horizon}, -90^\circ = \text{Nadir}$) | `Optional[float]` |
| **Gimbal Roll** | Degrees ($^\circ$) | $[-180.0^\circ, +180.0^\circ]$ | `Optional[float]` |
| **Gimbal Yaw** | Degrees ($^\circ$) | $[0.0^\circ, 360.0^\circ)$ | `Optional[float]` |

> **RULE:** Angular values are NEVER stored as radians in canonical contracts. Radians are converted explicitly only at the point of trigonometric calculation in georeferencing math routines.

---

## 4. Coordinate Reference Systems (CRS)

The pipeline explicitly differentiates between distinct spatial coordinate reference systems:
1. **`WGS84_GEODETIC` (EPSG:4326):** Latitude and Longitude on the WGS84 reference ellipsoid.
2. **`ECEF` (EPSG:4978):** Earth-Centered, Earth-Fixed Cartesian coordinates $(X, Y, Z)$ in meters.
3. **`LOCAL_ENU`:** Topocentric tangent Euclidean coordinates (East, North, Up) in meters anchored at a reference geodetic origin.
4. **`CAMERA_FRAME`:** Optical coordinate frame ($X$-right, $Y$-down, $Z$-forward).

> **CRITICAL DISTINCTION:**
> GPS coordinates represent the GNSS antenna phase center location, NOT the camera optical center. Camera center pose requires lever-arm offset translation and gimbal attitude rotation.

---

## 5. Altitude Reference Datum

Altitude without an explicit vertical datum is a primary source of metric scaling and georeferencing errors. `AltitudeReference` explicitly records one of:

- **`ELLIPSOIDAL`:** Geometric height ($h$) above the WGS84 reference ellipsoid (standard in RTK/PPK GNSS receivers).
- **`MSL`:** Orthometric height ($H$) above the geoid / Mean Sea Level (e.g., EGM96/EGM2008).
- **`AGL`:** Height above local ground terrain (from ultrasonic, lidar, or radar altimeters).
- **`RELATIVE_TO_TAKEOFF`:** Barometric altitude relative to the home/takeoff point ($h=0$).
- **`UNKNOWN`:** Preserved when raw telemetry does not declare its vertical reference. *The system never guesses or assumes ellipsoidal height.*

---

## 6. Timestamp Semantics & Synchronization Design

Every `TelemetryRecord` explicitly declares its `TimestampSemantics`:
- `VIDEO_RELATIVE`: Time in seconds relative to video start ($t=0.0\text{s}$).
- `UTC_TIMESTAMP`: Absolute ISO 8601 UTC time.
- `GPS_TIME`: Time since GPS epoch (January 6, 1980).
- `MONOTONIC_SYSTEM_TIME`: Monotonic hardware clock time.
- `SENSOR_LOG_TIME`: Flight controller internal log counter.

### Duplicate Timestamps Handling
Telemetry logs recorded at high frequency (e.g. 50Hz IMU, 10Hz GPS) may produce multiple samples with identical timestamps. The `CanonicalTelemetryStream` preserves duplicate timestamps as valid distinct records.

### Downstream Synchronization Interface (Phase 1B.2 preview)
In Phase 1B.2, synchronization between `CanonicalTimeline` and `CanonicalTelemetryStream` will apply:
1. Timebase alignment (matching UTC or video start time offset).
2. Monotonic sorting via `stream.get_chronological_records()`.
3. Bounded temporal interpolation (linear or $C^2$ cubic spline) within maximum allowable time delta ($\Delta t \le \tau_{\max}$).
