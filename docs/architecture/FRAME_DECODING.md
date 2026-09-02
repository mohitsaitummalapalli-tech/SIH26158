# Frame Decoding & Canonical Image Representation Architecture

## 1. Executive Summary & Objective

The Frame Decoding subsystem provides a modular, pluggable, and memory-safe interface for decoding discrete video presentation frames into deterministic in-memory RGB array representations (`DecodedFrame`).

> **CRITICAL SCIENTIFIC DISTINCTION:**
> Decoded frames produced in this phase are verified for **VALID PIXEL REPRESENTATION** only. They are **NOT** "quality frames" or "keyframes" yet. Frame quality assessment, blur scoring, illumination analysis, and dynamic object masking are formally handled in Stage 3 / Phase 2B.

---

## 2. Decoder Architecture & Pluggable Backends

```
                        CanonicalFlightDataset
                                  │
                                  ▼
                        DatasetFrameDecoder
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
        OpenCVFrameDecoder               SyntheticFrameDecoder
     (Production Media Ingestion)         (Deterministic Fixtures)
                 │                                 │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                             DecodedFrame
                       (Canonical RGB uint8)
```

### Decoder Classes:
- **`FrameDecoder`**: Abstract base class enforcing single-frame random access (`decode_frame`) and sequential streaming iteration (`iter_frames`).
- **`OpenCVFrameDecoder`**: Production backend utilizing OpenCV `VideoCapture` with explicit color space conversion (BGR $\to$ RGB).
- **`SyntheticFrameDecoder`**: Procedural, deterministic test card generator for unit and integration testing without external file dependencies.
- **`DatasetFrameDecoder`**: High-level wrapper binding a `CanonicalFlightDataset` directly to a `FrameDecoder` backend.

---

## 3. Random-Access vs Sequential Decoding Semantics Audit

### 1. OpenCV Demuxer Mechanics (`CAP_PROP_POS_FRAMES`):
- `cv2.VideoCapture` delegates seeking to platform-specific multimedia backends (e.g. `MSMF` on Windows, `FFMPEG` or `GStreamer` on Linux).
- **Constant Frame Rate (CFR) & Closed GOP**: Seeking by frame index is generally accurate.
- **Variable Frame Rate (VFR) & B-Frames (Open GOP)**:
  - Compressed video containers store frames in *decode order* (DTS) which differs from *presentation display order* (PTS) when B-frames are present.
  - Calling `CAP_PROP_POS_FRAMES` can seek to the nearest preceding IDR keyframe and may not correctly advance composition timestamps (CTTS) across all backend builds.
- **Scientific Reference Architecture**:
  - **Sequential Streaming (`iter_frames()`)** starting from frame 0 is the **authoritative scientific ground truth** for guaranteed presentation-order decoding.
  - **Random Access (`decode_frame()`)** is an optimized seeking path for fast previewing and selective sampling, but cannot guarantee bit-exact PTS alignment on arbitrary proprietary VFR drone streams without sequential index verification.

---

## 4. Timestamp Integrity & Canonical PTS Preservation

- **Timestamp Source Policy**:
  - The timestamp of every `DecodedFrame` originates strictly from the canonical Phase 1A timeline (`CanonicalTimeline.frames[i].timestamp_seconds`).
  - Naive derivations ($t = \text{index} / \text{FPS}$) are **forbidden** when a canonical timeline is present, preventing time-drift errors on VFR streams.
  - `DecodedFrame.timestamp_source` explicitly records `"canonical_timeline_pts"` (or `"inferred_nominal_fps"` only if an unparsed raw video is decoded directly without metadata).

---

## 5. Canonical Pixel & Image Contracts

### 1. Representation Standards:
- **Channel Order**: Canonical **RGB** (Red = Channel 0, Green = Channel 1, Blue = Channel 2). OpenCV's default BGR format is converted to RGB immediately upon frame demuxing.
- **Data Type**: `uint8` (`numpy.uint8` with values in $[0, 255]$).
- **Array Shape**: $(H, W, 3)$ for standard color frames; $(H, W)$ for monochrome/grayscale.
- **Alpha Channel**: Dropped / ignored during 3-channel RGB standard ingestion.
- **Color Space**: sRGB computational representation (no radiometric calibration claimed at this stage).

### 2. Contract (`DecodedFrame`):
- `frame_id`: Matches canonical frame identifier (e.g. `"flight_000042"`).
- `frame_index`: 0-based sequential presentation index.
- `timestamp_seconds`: True presentation timestamp ($t_{\text{video}}$) from `CanonicalTimeline`.
- `timestamp_source`: Explicit lineage tag (`"canonical_timeline_pts"`).
- `width`, `height`, `channels`: Verified image dimensions.
- `data`: In-memory `numpy.ndarray` (or `None` on failure).
- `decode_status`: `DecodeStatus` enum (`SUCCESS`, `CORRUPTED`, `DECODER_ERROR`, `INDEX_OUT_OF_BOUNDS`, `FILE_NOT_FOUND`).
- `is_resized`: Explicit boolean flag indicating whether the frame was resized from native resolution.
- `original_width`, `original_height`: Preserved native dimensions when resized.

---

## 6. Memory Control & Streaming Generator

For aerial drone surveys (e.g. 4K 60fps videos spanning 30+ minutes and containing $> 100,000$ frames):
- **Rule**: Entire video sequences are **NEVER** pre-loaded into RAM or VRAM simultaneously.
- **Sequential Streaming**: `iter_frames(start_index, stop_index, step)` operates as a Python generator yielding one `DecodedFrame` at a time. The caller can process the frame buffer and allow garbage collection before decoding the next frame.
- **Random Access**: `decode_frame(frame_index)` seeks directly to the target frame index via hardware demuxer index positioning (`CAP_PROP_POS_FRAMES`).

---

## 7. Resizing Policy

- **Default**: Original resolution decoding (`target_width=None, target_height=None`).
- **Explicit Resizing**: If a pipeline stage requests downsampling (e.g. for keyframe screening), `is_resized` is set to `True`, and `original_width` / `original_height` are recorded in the `DecodedFrame` metadata. Frames are never silently resized.
- **Geometric Integrity**: Resized pixel arrays carry no claim of metric camera calibration until rescaled camera intrinsics matrices ($K$) are formally derived.

---

## 8. Error Behavior & Integrity

- **Missing Files**: Returns `DecodedFrame` with `DecodeStatus.FILE_NOT_FOUND` and `data=None`.
- **Corrupt Streams**: Returns `DecodedFrame` with `DecodeStatus.CORRUPTED` and `data=None` without crashing sequential iteration or fabricating synthetic substitute frames.
- **Out of Bounds**: Returns `DecodedFrame` with `DecodeStatus.INDEX_OUT_OF_BOUNDS`.
- **Integrity Validation**: `frame.validate()` checks array instance type, dtype matching, shape matching, and finite contents.

---

## 9. Environment Compatibility

- Declared Python Target: `>=3.10, <3.13` (CPython 3.10, 3.11, 3.12).
- Downstream AI / Reconstruction packages (PyTorch, CUDA 12, DUSt3R, RoMa, CroCo) require Python 3.10–3.12.
- For full details, see [`docs/research/ENVIRONMENT_COMPATIBILITY.md`](file:///c:/Users/mohit/OneDrive/Desktop/SIH%2026158/docs/research/ENVIRONMENT_COMPATIBILITY.md).

---

## 10. Final Audit Verdict

```
================================================================================
                           PHASE 2A.1 AUDIT STATUS
================================================================================
            CONDITIONAL PASS — RANDOM ACCESS BACKEND LIMITED
================================================================================
```
*Rationale: The canonical image contract, PTS preservation, sequential streaming, and RGB pixel representation are mathematically verified. Random-access seeking via OpenCV CAP_PROP_POS_FRAMES is acknowledged as an optimization path with known backend-dependent limitations on VFR/B-frame streams; sequential iteration is the authoritative ground truth.*
