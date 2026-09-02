"""Test utility for generating minimal compliant synthetic MP4/MOV test binaries.

DISCLAIMER:
THIS IS SYNTHETIC TEST DATA GENERATED SOLELY FOR UNIT TESTING INGESTION BOX PARSING.
IT IS NOT A REAL DRONE FLIGHT DATASET.
"""

import struct
from typing import List, Tuple, Optional


def build_box(box_type: bytes, payload: bytes) -> bytes:
    """Pack an ISOBMFF box with uint32 size and 4-byte fourcc."""
    size = len(payload) + 8
    return struct.pack(">I4s", size, box_type) + payload


def build_box64(box_type: bytes, payload: bytes) -> bytes:
    """Pack an ISOBMFF box with 64-bit size (size=1) and 4-byte fourcc."""
    size = len(payload) + 16
    return struct.pack(">I4sQ", 1, box_type, size) + payload


def create_synthetic_mp4(
    filepath: str,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    num_frames: int = 30,
    timescale: int = 1000,
    keyframe_indices: Optional[List[int]] = None,
    codec_fourcc: bytes = b"avc1",
    ctts_offsets: Optional[List[int]] = None,
    vfr_deltas: Optional[List[Tuple[int, int]]] = None,
    edit_list_delay_ms: Optional[int] = None,
    is_mov: bool = False,
    use_64bit_box: bool = False,
    include_unknown_box: bool = False,
    include_secondary_video_track: bool = False
) -> str:
    """Create a minimal structurally valid MP4/MOV file for deterministic unit testing."""
    if keyframe_indices is None:
        keyframe_indices = [0]

    # 1. ftyp box
    brand = b"qt  " if is_mov else b"isom"
    ftyp_payload = brand + struct.pack(">I", 512) + (b"qt  " if is_mov else b"isomiso2mp41")
    ftyp_box = build_box(b"ftyp", ftyp_payload)

    # 2. Timing (stts)
    if vfr_deltas:
        stts_payload = struct.pack(">II", 0, len(vfr_deltas))
        total_duration_ticks = 0
        total_frames = 0
        for count, delta in vfr_deltas:
            stts_payload += struct.pack(">II", count, delta)
            total_duration_ticks += count * delta
            total_frames += count
        num_frames = total_frames
        duration_ticks = total_duration_ticks
    else:
        frame_delta = timescale // fps
        duration_ticks = num_frames * frame_delta
        stts_payload = struct.pack(">II", 0, 1) + struct.pack(">II", num_frames, frame_delta)
    stts_box = build_box(b"stts", stts_payload)

    # 3. Composition offsets (ctts) for B-frames
    ctts_box = b""
    if ctts_offsets is not None and len(ctts_offsets) == num_frames:
        ctts_payload = struct.pack(">II", 0, num_frames)
        for off in ctts_offsets:
            ctts_payload += struct.pack(">II", 1, off)
        ctts_box = build_box(b"ctts", ctts_payload)

    # 4. Sync samples / Keyframes (stss)
    stss_payload = struct.pack(">II", 0, len(keyframe_indices))
    for kf in keyframe_indices:
        stss_payload += struct.pack(">I", kf + 1)
    stss_box = build_box(b"stss", stss_payload)

    # 5. Sample sizes (stsz)
    stsz_payload = struct.pack(">III", 0, 0, num_frames)
    for _ in range(num_frames):
        stsz_payload += struct.pack(">I", 128)
    stsz_box = build_box(b"stsz", stsz_payload)

    # 6. Chunk offset (stco)
    stco_payload = struct.pack(">II", 0, 1) + struct.pack(">I", 1024)
    stco_box = build_box(b"stco", stco_payload)

    # 7. Sample description (stsd)
    compressor_name = b"Test Synthetic Codec"
    vse_payload = (
        b"\x00" * 6 +
        struct.pack(">H", 1) +
        b"\x00" * 16 +
        struct.pack(">HH", width, height) +
        struct.pack(">II", 0x00480000, 0x00480000) +
        struct.pack(">I", 0) +
        struct.pack(">H", 1) +
        struct.pack(">B", len(compressor_name)) +
        compressor_name.ljust(31, b"\x00") +
        struct.pack(">h", 24) +
        struct.pack(">h", -1)
    )
    vse_box = build_box(codec_fourcc, vse_payload)
    stsd_payload = struct.pack(">II", 0, 1) + vse_box
    stsd_box = build_box(b"stsd", stsd_payload)

    # 8. stbl box
    stbl_payload = stsd_box + stts_box + ctts_box + stss_box + stsz_box + stco_box
    stbl_box = build_box(b"stbl", stbl_payload)

    # 9. minf box
    dref_box = build_box(b"dref", struct.pack(">II", 0, 1) + build_box(b"url ", struct.pack(">I", 1)))
    dinf_box = build_box(b"dinf", dref_box)
    vmhd_box = build_box(b"vmhd", struct.pack(">I", 1) + struct.pack(">HHHH", 0, 0, 0, 0))
    minf_payload = vmhd_box + dinf_box + stbl_box
    minf_box = build_box(b"minf", minf_payload)

    # 10. mdia box
    hdlr_payload = struct.pack(">II4s", 0, 0, b"vide") + b"\x00" * 12 + b"VideoHandler\x00"
    hdlr_box = build_box(b"hdlr", hdlr_payload)
    mdhd_payload = struct.pack(">B", 0) + b"\x00" * 3 + struct.pack(">IIII", 0, 0, timescale, duration_ticks) + struct.pack(">H", 0) + struct.pack(">H", 0)
    mdhd_box = build_box(b"mdhd", mdhd_payload)
    mdia_payload = mdhd_box + hdlr_box + minf_box
    mdia_box = build_box(b"mdia", mdia_payload)

    # 11. Optional Edit List (edts / elst)
    edts_box = b""
    if edit_list_delay_ms is not None:
        # Initial empty edit (delay)
        elst_payload = struct.pack(">II", 0, 2)
        # Entry 1: empty edit (dwell)
        elst_payload += struct.pack(">Ii", edit_list_delay_ms, -1) + struct.pack(">hh", 1, 0)
        # Entry 2: normal playback
        elst_payload += struct.pack(">Ii", duration_ticks, 0) + struct.pack(">hh", 1, 0)
        elst_box = build_box(b"elst", elst_payload)
        edts_box = build_box(b"edts", elst_box)

    # 12. trak box
    w_fixed = int(width * 65536)
    h_fixed = int(height * 65536)
    tkhd_payload = struct.pack(">B", 0) + b"\x00" * 3 + struct.pack(">IIII", 0, 0, 1, 0) + struct.pack(">I", duration_ticks) + b"\x00" * 48 + struct.pack(">II", w_fixed, h_fixed)
    tkhd_box = build_box(b"tkhd", tkhd_payload)
    trak_payload = tkhd_box + edts_box + mdia_box
    trak_box = build_box(b"trak", trak_payload)

    # 13. Optional secondary video track (lower res)
    trak2_box = b""
    if include_secondary_video_track:
        w2_fixed = int(640 * 65536)
        h2_fixed = int(360 * 65536)
        tkhd2_payload = struct.pack(">B", 0) + b"\x00" * 3 + struct.pack(">IIII", 0, 0, 2, 0) + struct.pack(">I", duration_ticks) + b"\x00" * 48 + struct.pack(">II", w2_fixed, h2_fixed)
        tkhd2_box = build_box(b"tkhd", tkhd2_payload)
        trak2_box = build_box(b"trak", tkhd2_box + mdia_box)

    # 14. mvhd box & moov box
    mvhd_payload = struct.pack(">B", 0) + b"\x00" * 3 + struct.pack(">IIII", 0, 0, timescale, duration_ticks) + struct.pack(">I", 0x00010000) + struct.pack(">h", 0x0100) + b"\x00" * 70 + struct.pack(">I", 2)
    mvhd_box = build_box(b"mvhd", mvhd_payload)
    moov_payload = mvhd_box + trak_box + trak2_box
    moov_box = build_box(b"moov", moov_payload)

    # 15. Optional unknown extension box
    unknown_box = b""
    if include_unknown_box:
        unknown_box = build_box(b"free", b"Unknown custom metadata payload to be ignored safely")

    # 16. mdat box (use 64-bit if requested)
    mdat_payload = b"\x00" * (num_frames * 128)
    if use_64bit_box:
        mdat_box = build_box64(b"mdat", mdat_payload)
    else:
        mdat_box = build_box(b"mdat", mdat_payload)

    with open(filepath, "wb") as f:
        f.write(ftyp_box)
        f.write(unknown_box)
        f.write(moov_box)
        f.write(mdat_box)

    return filepath


def create_synthetic_fragmented_mp4(filepath: str) -> str:
    """Create a synthetic fragmented MP4 containing moov+mvex and moof boxes."""
    ftyp_box = build_box(b"ftyp", b"isom" + struct.pack(">I", 512) + b"isomiso2mp41")
    
    # moov with mvex
    mvhd_box = build_box(b"mvhd", struct.pack(">B", 0) + b"\x00" * 3 + struct.pack(">IIII", 0, 0, 1000, 0) + struct.pack(">I", 0x00010000) + struct.pack(">h", 0x0100) + b"\x00" * 70 + struct.pack(">I", 2))
    mvex_box = build_box(b"mvex", build_box(b"trex", struct.pack(">IIIIII", 0, 1, 1, 0, 0, 0)))
    moov_box = build_box(b"moov", mvhd_box + mvex_box)
    
    # moof (fragment header)
    mfhd_box = build_box(b"mfhd", struct.pack(">II", 0, 1))
    traf_box = build_box(b"traf", build_box(b"tfhd", struct.pack(">II", 0, 1)))
    moof_box = build_box(b"moof", mfhd_box + traf_box)
    mdat_box = build_box(b"mdat", b"\x00" * 256)

    with open(filepath, "wb") as f:
        f.write(ftyp_box)
        f.write(moov_box)
        f.write(moof_box)
        f.write(mdat_box)

    return filepath
