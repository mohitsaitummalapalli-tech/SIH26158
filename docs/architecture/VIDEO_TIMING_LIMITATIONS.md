# Video Timing Concepts, Semantics & Architectural Limitations

## 1. Distinction of Timing Concepts

To avoid subtle timing and synchronization errors when pairing drone video with GPS/IMU telemetry, the system enforces strict theoretical distinctions between six distinct timing concepts:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TIMING TAXONOMY                                   │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
      ┌────────────────────────────┼────────────────────────────┐
      ▼                            ▼                            ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
│ Decoding Time    │     │ Presentation Time│     │ Actual Presentation     │
│ (DTS)            │     │ (PTS)            │     │ Timing (Seconds)         │
├──────────────────┤     ├──────────────────┤     ├──────────────────────────┤
│ Stream-level     │     │ True display     │     │ Floating-point elapsed   │
│ decompression    │     │ order timestamp: │     │ presentation time:       │
│ order:           │     │ PTS = DTS + CTTS │     │ t = PTS / timescale +    │
│ DTS = Σ delta    │     │ in stream ticks. │     │ edit_list_offset.        │
└──────────────────┘     └──────────────────┘     └──────────────────────────┘
      ▲                            ▲                            ▲
      │                            │                            │
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
│ Structural       │     │ Frame Index      │     │ Nominal FPS              │
│ Container Time   │     │ (0-based)        │     │ (Estimated Rate)         │
├──────────────────┤     ├──────────────────┤     ├──────────────────────────┤
│ Movie-level total│     │ Discrete sample  │     │ Total frames / duration; │
│ duration ticks in│     │ position index   │     │ NOT a guarantee of       │
│ mvhd / mdhd.     │     │ k ∈ {0, ..., N-1}│     │ constant inter-frame dt! │
└──────────────────┘     └──────────────────┘     └──────────────────────────┘
```

---

## 2. Timing Definitions & Mathematical Formulas

1. **Decoding TimeStamp (DTS):**
   The time at which a video sample must be decoded by the video codec. In streams without B-frames (e.g. baseline H.264), $\text{PTS} = \text{DTS}$. When B-frames are present, frames are decoded out of display order.
2. **Composition Time Offset (CTTS):**
   The presentation offset between decoding time and display time:
   $$\text{PTS}_k = \text{DTS}_k + \text{CTTS\_Offset}_k$$
3. **Stream Timescale ($\nu$):**
   Integer frequency representing time units per second (e.g. $\nu = 1000$ for millisecond ticks, $\nu = 60000$ for $59.94\text{ fps}$ / $60\text{ fps}$ broadcast tracks).
4. **Presentation Timestamp in Seconds ($t_k$):**
   The continuous presentation timestamp from video origin:
   $$t_k = \frac{\text{PTS}_k}{\nu} + t_{\text{edit\_offset}}$$
5. **Frame Index ($k$):**
   Zero-based integer index ($0 \le k < N$).
6. **Nominal Framerate ($\text{FPS}_{\text{nominal}}$):**
   $$\text{FPS}_{\text{nominal}} = \frac{N}{t_{\text{duration}}}$$
   > **CRITICAL WARNING:**
   > In Variable Frame Rate (VFR) recordings or dropped-frame streams, $\Delta t_k = t_{k+1} - t_k \ne \frac{1}{\text{FPS}_{\text{nominal}}}$. The pipeline must never calculate frame times as $k / \text{FPS}_{\text{nominal}}$.

---

## 3. Edit List (`elst`) Semantics

The Track Edit Box (`edts` $\to$ `elst`) specifies how media samples are mapped onto the overall movie timeline:
- **Initial Empty Edit ($\text{media\_time} = -1$):** Indicates an initial dwell or delay before the video stream starts. The parser adds this segment duration to all sample presentation timestamps ($t_{\text{edit\_offset}} = \frac{\text{segment\_duration}}{\text{movie\_timescale}}$).
- **Non-Zero Media Start ($\text{media\_time} > 0$):** Indicates that initial decoded samples prior to $\text{media\_time}$ are trimmed.

---

## 4. Fragmented Media Limitations (`fMP4`)

- **Structure:** Fragmented MP4 files use movie fragments (`moof` + `traf` + `trun`) to deliver streaming video chunks without a finalized master sample table (`stbl`) in `moov`.
- **Handling:** Fragmented MP4s are detected via `moof`/`mvex` boxes and rejected with a descriptive `UnsupportedFragmentedMP4Error`.
- **Reason:** Aerial drone flight recordings (DJI, Autel, Skydio, GoPro) are stored as standard contiguous MP4/MOV files with complete sample tables.

---

## 5. Confirmed Architectural Limitations

1. **Reference Demuxer Availability:** `ffmpeg` / `ffprobe` binaries are currently not installed on the system PATH. All parser tests currently run on deterministic synthetic binaries (`TEST_SYNTHETIC`). Full external binary cross-validation is flagged as **CONDITIONAL PASS — REFERENCE VALIDATION LIMITED**.
2. **Pixel Raster Extraction:** The `ISOBMFFParser` extracts box headers, track metadata, and presentation timestamps. Compressed bitstream raster decoding (YUV $\to$ RGB tensor) requires codec decoders (FFmpeg / PyAV / OpenCV) during downstream frame preprocessing.
3. **Pixel Format / Chroma Subsampling:** Detailed chroma subsampling metadata (`yuv420p` vs `yuv422p10le`) requires Annex-B NAL bitstream inspection and is marked `REFERENCE-ONLY` at the container level.

---

## 6. Real-File Compatibility Policy

> **MANDATORY POLICY:**
> Compatibility with specific drone manufacturers (e.g. DJI Mavic 3 Enterprise, Autel EVO II, PX4 companion computers) cannot be claimed based solely on synthetic binary unit tests.
> 
> Official manufacturer compatibility certification requires empirical verification against verified real-world flight video assets with corresponding ground-truth telemetry tracks.
