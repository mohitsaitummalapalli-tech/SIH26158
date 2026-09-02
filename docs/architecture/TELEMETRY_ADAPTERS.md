# Telemetry Adapter Layer Architecture & Vendor Ingestion

## 1. Executive Summary & Objective

The Telemetry Adapter Layer provides modular, decoupled ingestion adapters that translate disparate raw telemetry logs (DJI SRT, ArduPilot/PX4 CSV, and future KLV metadata) into the unified `CanonicalTelemetryStream` without imposing vendor-specific assumptions or hard-coded limits onto downstream reconstruction modules.

---

## 2. Adapter Architecture

```
                               ┌────────────────────────────────┐
                               │     Raw Telemetry Source       │
                               │  (.SRT, .CSV, KLV stream, ...) │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │   TelemetryAdapter Interface   │
                               ├────────────────────────────────┤
                               │ + parse_records()              │
                               │ + parse_stream()               │
                               │ + compute_sha256()             │
                               └───────────────┬────────────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               │                               │                               │
               ▼                               ▼                               ▼
    ┌────────────────────┐          ┌────────────────────┐          ┌────────────────────┐
    │   DJISRTAdapter    │          │  GenericCSVAdapter │          │ KLVAdapterInterface│
    ├────────────────────┤          ├────────────────────┤          ├────────────────────┤
    │ Regex tag parsing, │          │ Explicit column &  │          │ SMPTE 336M / MISB  │
    │ multi-series tags, │          │ unit configuration │          │ ST 0601 interface  │
    │ timecode mapping   │          │ conversion         │          │ (Future phase)     │
    └──────────┬─────────┘          └──────────┬─────────┘          └──────────┬─────────┘
               │                               │                               │
               └───────────────────────────────┼───────────────────────────────┘
                                               │
                                               ▼
                              ┌─────────────────────────────────┐
                              │  ParsedTelemetryRecord (Status) │
                              │  - VALID                        │
                              │  - PARTIALLY_VALID              │
                              │  - INVALID                      │
                              └────────────────┬────────────────┘
                                               │
                                               ▼
                              ┌─────────────────────────────────┐
                              │    CanonicalTelemetryStream     │
                              │  (Immutable, Chronological,     │
                              │   Cryptographic Provenance)     │
                              └─────────────────────────────────┘
```

---

## 3. DJI SRT Telemetry Semantics & Tag Variations

DJI drone firmware generates `.SRT` subtitle companion files containing flight parameters updated at video framerate (~30Hz) or sensor update rate. Due to firmware evolution across DJI aircraft generations, tag formats and field availability vary substantially:

### Known Format Variations:
1. **Bracketed Tag Format (Mavic 2/3, Air 2S, Mini 3/4):**
   ```
   1
   00:00:00,000 --> 00:00:00,033
   [iso: 100] [shutter: 1/1000] [fnum: 2.8] [focal_len: 240] [latitude: 18.5204] [longitude: 73.8567] [rel_alt: 120.500 abs_alt: 560.200]
   [gb_pitch: -45.0 gb_roll: 0.0 gb_yaw: 145.2] [drone_pitch: -5.0 drone_roll: 1.2 drone_yaw: 145.2]
   ```
2. **GPS Tuple & Colon Tag Format (Phantom 4, Inspire 2):**
   ```
   1
   00:00:00,000 --> 00:00:00,033
   HOME(73.8567,18.5204) 2023.08.15 14:30:12
   GPS(73.8567,18.5204,18) D 15.20m, H 120.50m, H.S 5.20m/s, V.S 0.10m/s
   ```
   *(Note: In DJI `GPS(lon, lat, sats)` tuples, longitude is listed first).*

### Semantics & Policies:
- **Timestamp**: Derived directly from the subtitle timecode start (`HH:MM:SS,mmm`) as video-relative seconds ($t \ge 0.0\text{s}$). If an absolute ISO date/time string is detected in the text, it is recorded in `timestamp_utc`.
- **Altitude**:
  - `abs_alt` / `altitude`: Mapped to `AltitudeReference.MSL`.
  - `rel_alt` / `height`: Mapped to `AltitudeReference.RELATIVE_TO_TAKEOFF`.
- **Optional Quantities**: Missing fields remain `None`. Zero values are never fabricated for missing sensors.
- **Unknown Tags**: Non-standard tags (e.g. `focal_len`, `dzoom`, `color_md`, `ct`) are captured in `extra_metadata`.

---

## 4. Generic CSV Adapter & Mapping Configuration

The `GenericCSVAdapter` ingests arbitrary structured tabular logs (e.g., ArduPilot DataFlash CSV, PX4 ULog CSV, custom ground station logs) using an explicit `CSVColumnMapping`:

```python
mapping = CSVColumnMapping(
    timestamp_col="time_boot_ms",
    latitude_col="lat",
    longitude_col="lon",
    altitude_col="alt_amsl",
    heading_col="yaw_deg",
    pitch_col="pitch_deg",
    roll_col="roll_deg",
    speed_col="ground_speed_mps",
    timestamp_unit="milliseconds",
    angle_unit="degrees",
    altitude_unit="meters",
    altitude_reference=AltitudeReference.MSL,
    position_reference=PositionReference.WGS84_GEODETIC
)
```

### Validation & Error Policy:
- Missing required mapped columns in the CSV header raises an immediate configuration error.
- Ambiguous or unstated unit mappings are rejected rather than guessed.
- Unit conversions (e.g. radians to degrees, feet to meters, milliseconds to seconds) are computed deterministically.

---

## 5. Record Status & Malformed Record Policy

Raw telemetry records are never silently dropped or discarded. Each block or row is parsed into a `ParsedTelemetryRecord` classified as:

1. **`VALID`**: Complete record containing valid spatial coordinates, finite timestamp, orientation, and kinematics.
2. **`PARTIALLY_VALID`**: Recoverable record containing valid spatial coordinates and timestamp, with non-critical optional fields (e.g. roll, pitch, speed) absent.
3. **`INVALID`**: Unrecoverable record (e.g., unparseable text, coordinate out-of-bounds, $\text{NaN}/\pm\infty$) accompanied by a deterministic `rejection_reason`.

---

## 6. KLV (Key-Length-Value) Metadata Architecture

For defense, surveying, and industrial aerial video streams carrying embedded SMPTE 336M / MISB ST 0601 metadata tracks in MPEG Transport Streams (TS) or MP4 containers:
- `KLVPacket` defines the binary contract (`key: bytes`, `length: int`, `value: bytes`, `timestamp_utc: Optional[str]`, `schema_identifier: str`).
- `KLVAdapterInterface` specifies the demuxer interface for future implementation.
- *No speculative decoder is implemented until concrete vendor samples and schema standards are provided.*

---

## 7. Cryptographic Provenance

Every canonical record preserves lineage back to its source:
- `source_type`: e.g. `"dji_srt"`, `"generic_csv"`.
- `source_identifier`: Absolute file path or stream URI.
- `record_index`: 0-indexed position within raw source log.
- `extraction_method`: Adapter version identifier (e.g. `"DJISRTAdapter_v1.0"`).
- `source_checksum`: SHA-256 hash of the raw telemetry source file.
