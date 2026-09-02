# Video Ingestion Subsystem & Canonical Timeline Specification

## 1. Overview & Purpose

The **Video Ingestion Subsystem (Phase 1A)** is responsible for ingesting single-pass drone video containers (`.mp4`, `.mov`), extracting structural container metadata, and constructing a **Canonical Video Timeline**.

This subsystem establishes the immutable visual backbone for the reconstruction system before telemetry synchronization (Phase 1B) or geometric reconstruction (Phase 2).

---

## 2. Supported Container Inputs

| Format / Container | File Extensions | Supported Codecs | Notes |
| :--- | :--- | :--- | :--- |
| **ISO Base Media File Format (MP4)** | `.mp4`, `.MP4` | AVC/H.264 (`avc1`), HEVC/H.265 (`hvc1`, `hev1`), MPEG-4 Visual (`mp4v`) | Primary format for DJI, Autel, and standard drone platforms. |
| **Apple QuickTime Movie (MOV)** | `.mov`, `.MOV` | Apple ProRes (`apcn`, `apch`, `apco`, `ap4h`), AVC/H.264, HEVC | Common for high-bitrate aerial cinematic payloads. |

Unsupported formats (e.g. `.avi`, `.mkv`, `.flv`) are deterministically rejected with `UnsupportedVideoFormatError`.

---

## 3. Metadata Extraction Engine

Video container parsing is performed via `ISOBMFFParser` directly from binary container box tables:
- **`ftyp`**: Major brand, minor version, compatible brands.
- **`mvhd`**: Movie duration and movie timescale.
- **`tkhd`**: Video track display width and height (16.16 fixed-point).
- **`mdhd`**: Media stream timescale and media duration in ticks.
- **`hdlr`**: Handler reference (`vide` for video stream, `soun` for audio, `meta` for metadata).
- **`stsd`**: Visual sample description entry (FourCC codec, compressor name, raster dimensions).
- **`stts`**: Time-to-Sample table (decoding time deltas $\Delta t_i$).
- **`ctts`**: Composition Time-to-Sample table (presentation time offsets).
- **`stss`**: Sync Sample table (IDR/keyframe indices).
- **`stsz`**: Sample Size table (frame/sample count).

---

## 4. Timestamp Semantics & Exact PTS Mapping

> **CRITICAL RULE:**
> The system **NEVER** assumes $\text{timestamp} = \frac{\text{frame\_index}}{\text{nominal\_fps}}$.

Real drone video feeds often have minor clock jitter, variable frame rates (VFR), or B-frame reordering. The system extracts exact Presentation TimeStamps (PTS):

$$\text{DTS}_k = \sum_{i=1}^{k-1} \Delta t_i$$
$$\text{PTS}_k = \text{DTS}_k + \text{CTTS\_Offset}_k$$
$$\text{timestamp\_seconds}_k = \frac{\text{PTS}_k}{\text{timescale}}$$

- **Internal Time Representation:** `timestamp_seconds` is stored as an IEEE 754 64-bit float representing elapsed presentation time from stream origin ($t = 0.0\text{ s}$).
- **Monotonicity:** Canonical timelines enforce $\text{timestamp\_seconds}_k \ge \text{timestamp\_seconds}_{k-1}$.

---

## 5. Canonical Frame Identity & Timeline

### 5.1 `CanonicalFrame` Data Contract
```python
@dataclass(frozen=True)
class CanonicalFrame:
    frame_id: str             # Deterministic ID: f"{video_id}_{frame_index:06d}"
    frame_index: int          # 0-indexed sequential frame index
    timestamp_seconds: float  # Exact PTS in seconds
    pts: int                  # Raw Presentation TimeStamp ticks
    timescale: int            # Media stream ticks per second
    source_video: str         # Absolute file path of source video
    width: int                # Raster width in pixels
    height: int               # Raster height in pixels
    is_keyframe: bool         # True if sync/IDR frame
    extra_metadata: Dict      # Codec fourcc, compressor name
```

### 5.2 `CanonicalTimeline` Data Contract
- Stores the discrete sequence of `CanonicalFrame` objects.
- Provides $O(1)$ index lookup (`timeline.get_frame(idx)`) and $O(\log N)$ binary search timestamp lookup (`timeline.get_frame_at_timestamp(t)`).

---

## 6. Cryptographic Provenance

Every ingested video records complete provenance metadata in `VideoProvenance`:
- `source_file_path`: Absolute path of source asset.
- `file_size_bytes`: Byte length on disk.
- `sha256_checksum`: Cryptographic SHA-256 digest of source video file.
- `ingestion_timestamp_utc`: ISO 8601 UTC timestamp of ingestion operation.
- `metadata_extractor`: Implementation version (`ISOBMFFParser_v1.0`).
- `timestamp_source`: Source of timestamps (`container_pts`).

---

## 7. Deterministic Error Taxonomy

| Error Class | Condition |
| :--- | :--- |
| `VideoNotFoundError` | File path does not exist or is not a regular file. |
| `UnsupportedVideoFormatError` | File extension is not `.mp4` or `.mov`. |
| `CorruptVideoError` | File is 0 bytes, truncated, or lacks valid `ftyp`/`moov` box headers. |
| `InvalidVideoMetadataError` | Video track is missing, dimensions $\le 0$, timescale $\le 0$, duration $\le 0$, or timestamp non-monotonicity detected. |

---

## 8. Decoupled Architecture & Future Telemetry Synchronization Interface (Phase 1B)

The ingestion architecture completely decouples video stream parsing from telemetry stream parsing:

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│       VideoSource (Phase 1A)    │       │     TelemetrySource (Phase 1B)  │
│  (MP4 / MOV ISOBMFF Ingestion)  │       │   (DJI .SRT / KLV / CSV Parser) │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │                                         │
                 ▼                                         ▼
        CanonicalTimeline                         TelemetryStream
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │  Timeline Synchronizer        │
                      │  (Cubic Spline Interpolator   │
                      │   Spatial Frame Alignment)    │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                             SynchronizedFrameStream
```

In Phase 1B, the `TelemetrySource` will independently extract timestamped GNSS/IMU records, and the `TimelineSynchronizer` will attach spatial priors to each `CanonicalFrame` without mutating the core `CanonicalTimeline`.
