"""Pure-Python ISO Base Media File Format (ISOBMFF / MP4 / QuickTime MOV) parser."""

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, BinaryIO
from src.ingestion.exceptions import (
    CorruptVideoError,
    InvalidVideoMetadataError,
    UnsupportedFragmentedMP4Error,
)


@dataclass
class EditListEntry:
    """Entry in track edit list table (elst)."""
    segment_duration: int  # in movie timescale units
    media_time: int        # in media timescale units (-1 for empty edit / delay)
    media_rate: float      # 1.0 = normal playback rate


@dataclass
class VideoTrackInfo:
    """Raw parsed properties of a video track in an ISOBMFF/MOV container."""
    track_id: int
    codec_fourcc: str
    width: int
    height: int
    timescale: int
    duration_ticks: int
    sample_count: int
    dts_list: List[int] = field(default_factory=list)
    pts_list: List[int] = field(default_factory=list)
    keyframe_indices: List[int] = field(default_factory=list)  # 0-indexed sample indices
    compressor_name: Optional[str] = None
    is_variable_frame_rate: bool = False
    has_b_frames: bool = False
    edit_list_entries: List[EditListEntry] = field(default_factory=list)


@dataclass
class ParsedContainerInfo:
    """Parsed container-level metadata."""
    major_brand: str
    minor_version: int
    compatible_brands: List[str]
    movie_timescale: int
    movie_duration_ticks: int
    creation_time_epoch: Optional[int]
    modification_time_epoch: Optional[int]
    video_tracks: List[VideoTrackInfo] = field(default_factory=list)
    has_audio: bool = False
    has_metadata_track: bool = False
    is_fragmented: bool = False


class ISOBMFFParser:
    """Parser for ISO Base Media File Format (MP4) and QuickTime (.MOV) files."""

    def __init__(self, file_obj: BinaryIO, file_size: int) -> None:
        self.f = file_obj
        self.file_size = file_size

    @classmethod
    def parse_file(cls, filepath: str) -> ParsedContainerInfo:
        with open(filepath, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                raise CorruptVideoError(f"Video file '{filepath}' is empty (0 bytes).")
            f.seek(0)
            parser = cls(f, size)
            return parser.parse()

    def parse(self) -> ParsedContainerInfo:
        major_brand = ""
        minor_version = 0
        compatible_brands: List[str] = []
        movie_timescale = 1000
        movie_duration_ticks = 0
        creation_epoch: Optional[int] = None
        modification_epoch: Optional[int] = None
        video_tracks: List[VideoTrackInfo] = []
        has_audio = False
        has_metadata_track = False
        is_fragmented = False

        self.f.seek(0)
        offset = 0

        while offset < self.file_size:
            header = self._read_box_header(offset)
            if header is None:
                break
            box_type, box_size, header_size, payload_offset = header

            if box_type == b"ftyp":
                major_brand, minor_version, compatible_brands = self._parse_ftyp(payload_offset, box_size - header_size)
            elif box_type == b"moov":
                # Parse moov children
                moov_offset = payload_offset
                moov_end = offset + box_size

                while moov_offset < moov_end:
                    child_header = self._read_box_header(moov_offset)
                    if child_header is None:
                        break
                    c_type, c_size, c_hsize, c_payload = child_header

                    if c_type == b"mvhd":
                        mvhd_res = self._parse_mvhd(c_payload, c_size - c_hsize)
                        movie_timescale, movie_duration_ticks, creation_epoch, modification_epoch = mvhd_res
                    elif c_type == b"trak":
                        trak_res = self._parse_trak(c_payload, c_size - c_hsize)
                        if trak_res is not None:
                            handler_type, track_info = trak_res
                            if handler_type == b"vide" and track_info is not None:
                                video_tracks.append(track_info)
                            elif handler_type == b"soun":
                                has_audio = True
                            elif handler_type == b"meta":
                                has_metadata_track = True
                    elif c_type == b"mvex":
                        # Movie extends box -> signals fragmented MP4
                        is_fragmented = True

                    moov_offset += c_size
            elif box_type == b"moof":
                # Movie fragment box -> fragmented MP4
                is_fragmented = True

            offset += box_size

        if is_fragmented and not video_tracks:
            raise UnsupportedFragmentedMP4Error(
                "Fragmented MP4 detected (found 'moof' or 'mvex' box). "
                "Non-fragmented moov-based video containers are required for canonical timeline creation."
            )

        if not video_tracks:
            if movie_duration_ticks == 0 and not has_audio and not has_metadata_track:
                raise CorruptVideoError("Corrupt MP4/MOV container: missing or truncated 'moov' box.")
            raise InvalidVideoMetadataError("No valid video stream found in container.")

        return ParsedContainerInfo(
            major_brand=major_brand,
            minor_version=minor_version,
            compatible_brands=compatible_brands,
            movie_timescale=movie_timescale,
            movie_duration_ticks=movie_duration_ticks,
            creation_time_epoch=creation_epoch,
            modification_time_epoch=modification_epoch,
            video_tracks=video_tracks,
            has_audio=has_audio,
            has_metadata_track=has_metadata_track,
            is_fragmented=is_fragmented,
        )

    def _read_box_header(self, offset: int) -> Optional[Tuple[bytes, int, int, int]]:
        if offset + 8 > self.file_size:
            return None
        self.f.seek(offset)
        data = self.f.read(8)
        if len(data) < 8:
            return None
        size, box_type = struct.unpack(">I4s", data)
        header_size = 8

        if size == 1:
            # 64-bit largesize box
            if offset + 16 > self.file_size:
                raise CorruptVideoError(f"Corrupt 64-bit box header at offset {offset}.")
            data64 = self.f.read(8)
            size = struct.unpack(">Q", data64)[0]
            header_size = 16
        elif size == 0:
            # Box extends to end of file
            size = self.file_size - offset

        if size < header_size or offset + size > self.file_size:
            raise CorruptVideoError(f"Box {box_type.decode('ascii', errors='replace')} has invalid size {size} exceeding file size {self.file_size}.")

        return box_type, size, header_size, offset + header_size

    def _parse_ftyp(self, offset: int, length: int) -> Tuple[str, int, List[str]]:
        if length < 8:
            return "", 0, []
        self.f.seek(offset)
        data = self.f.read(length)
        major_brand = data[:4].decode("latin-1", errors="replace").strip()
        minor_ver = struct.unpack(">I", data[4:8])[0]
        compat = []
        for i in range(8, length, 4):
            if i + 4 <= length:
                brand = data[i:i+4].decode("latin-1", errors="replace").strip()
                if brand:
                    compat.append(brand)
        return major_brand, minor_ver, compat

    def _parse_mvhd(self, offset: int, length: int) -> Tuple[int, int, Optional[int], Optional[int]]:
        if length < 24:
            raise CorruptVideoError("Corrupt 'mvhd' box: insufficient length.")
        self.f.seek(offset)
        version = struct.unpack(">B", self.f.read(1))[0]
        self.f.read(3)  # flags

        if version == 0:
            data = self.f.read(16)
            creation_time, mod_time, timescale, duration = struct.unpack(">IIII", data)
        elif version == 1:
            data = self.f.read(28)
            creation_time, mod_time, timescale, duration = struct.unpack(">QQIQ", data)
        else:
            raise CorruptVideoError(f"Unsupported 'mvhd' version: {version}")

        # QuickTime / MP4 epoch begins Jan 1, 1904. Convert to Unix Epoch (Jan 1, 1970).
        unix_creation = creation_time - 2082844800 if creation_time > 2082844800 else None
        unix_mod = mod_time - 2082844800 if mod_time > 2082844800 else None

        return timescale, duration, unix_creation, unix_mod

    def _parse_trak(self, offset: int, length: int) -> Optional[Tuple[bytes, Optional[VideoTrackInfo]]]:
        trak_end = offset + length
        curr = offset
        track_id = 0
        width = 0
        height = 0
        handler_type = b""
        mdia_offset = None
        mdia_len = 0
        edit_list_entries: List[EditListEntry] = []

        while curr < trak_end:
            h = self._read_box_header(curr)
            if h is None:
                break
            b_type, b_size, b_hsize, b_payload = h

            if b_type == b"tkhd":
                tkhd_data = self._parse_tkhd(b_payload, b_size - b_hsize)
                if tkhd_data:
                    track_id, w_fixed, h_fixed = tkhd_data
                    width = int(w_fixed)
                    height = int(h_fixed)
            elif b_type == b"edts":
                edit_list_entries = self._parse_edts(b_payload, b_size - b_hsize)
            elif b_type == b"mdia":
                mdia_offset = b_payload
                mdia_len = b_size - b_hsize

            curr += b_size

        if mdia_offset is None:
            return None

        # Parse mdia
        mdia_curr = mdia_offset
        mdia_end = mdia_offset + mdia_len
        timescale = 1000
        duration = 0
        stbl_offset = None
        stbl_len = 0

        while mdia_curr < mdia_end:
            h = self._read_box_header(mdia_curr)
            if h is None:
                break
            b_type, b_size, b_hsize, b_payload = h

            if b_type == b"mdhd":
                timescale, duration = self._parse_mdhd(b_payload, b_size - b_hsize)
            elif b_type == b"hdlr":
                handler_type = self._parse_hdlr(b_payload, b_size - b_hsize)
            elif b_type == b"minf":
                minf_curr = b_payload
                minf_end = b_payload + (b_size - b_hsize)
                while minf_curr < minf_end:
                    mh = self._read_box_header(minf_curr)
                    if mh is None:
                        break
                    mb_type, mb_size, mb_hsize, mb_payload = mh
                    if mb_type == b"stbl":
                        stbl_offset = mb_payload
                        stbl_len = mb_size - mb_hsize
                    minf_curr += mb_size

            mdia_curr += b_size

        if handler_type != b"vide":
            return handler_type, None

        # Parse sample table (stbl) for video track
        codec_fourcc = "unknown"
        sample_count = 0
        dts_list: List[int] = []
        pts_list: List[int] = []
        keyframe_indices: List[int] = []
        compressor_name = None
        is_vfr = False
        has_b_frames = False

        if stbl_offset is not None:
            stbl_curr = stbl_offset
            stbl_end = stbl_offset + stbl_len
            stts_entries: List[Tuple[int, int]] = []
            ctts_entries: List[Tuple[int, int]] = []

            while stbl_curr < stbl_end:
                h = self._read_box_header(stbl_curr)
                if h is None:
                    break
                b_type, b_size, b_hsize, b_payload = h

                if b_type == b"stsd":
                    stsd_res = self._parse_stsd_video(b_payload, b_size - b_hsize)
                    if stsd_res:
                        codec_fourcc, stsd_w, stsd_h, compressor_name = stsd_res
                        if width == 0 and stsd_w > 0:
                            width = stsd_w
                        if height == 0 and stsd_h > 0:
                            height = stsd_h
                elif b_type == b"stts":
                    stts_entries = self._parse_stts(b_payload, b_size - b_hsize)
                    if len(stts_entries) > 1:
                        # Multiple delta entries signal Variable Frame Rate
                        unique_deltas = {d for _, d in stts_entries}
                        if len(unique_deltas) > 1:
                            is_vfr = True
                elif b_type == b"ctts":
                    ctts_entries = self._parse_ctts(b_payload, b_size - b_hsize)
                    if ctts_entries:
                        has_b_frames = True
                elif b_type == b"stss":
                    keyframe_indices = self._parse_stss(b_payload, b_size - b_hsize)
                elif b_type == b"stsz":
                    sample_count = self._parse_stsz_count(b_payload, b_size - b_hsize)

                stbl_curr += b_size

            dts_list, sample_count = self._build_dts(stts_entries, sample_count)
            pts_list = self._build_pts(dts_list, ctts_entries)

            if not keyframe_indices and sample_count > 0:
                keyframe_indices = list(range(sample_count))

        return handler_type, VideoTrackInfo(
            track_id=track_id,
            codec_fourcc=codec_fourcc,
            width=width,
            height=height,
            timescale=timescale,
            duration_ticks=duration,
            sample_count=sample_count,
            dts_list=dts_list,
            pts_list=pts_list,
            keyframe_indices=keyframe_indices,
            compressor_name=compressor_name,
            is_variable_frame_rate=is_vfr,
            has_b_frames=has_b_frames,
            edit_list_entries=edit_list_entries,
        )

    def _parse_tkhd(self, offset: int, length: int) -> Optional[Tuple[int, float, float]]:
        if length < 84:
            return None
        self.f.seek(offset)
        version = struct.unpack(">B", self.f.read(1))[0]
        self.f.read(3)  # flags

        if version == 0:
            self.f.read(8)
            track_id = struct.unpack(">I", self.f.read(4))[0]
            self.f.read(60)
            w_raw, h_raw = struct.unpack(">II", self.f.read(8))
        else:
            self.f.read(16)
            track_id = struct.unpack(">I", self.f.read(4))[0]
            self.f.read(60)
            w_raw, h_raw = struct.unpack(">II", self.f.read(8))

        width = w_raw / 65536.0  # 16.16 fixed point
        height = h_raw / 65536.0
        return track_id, width, height

    def _parse_edts(self, offset: int, length: int) -> List[EditListEntry]:
        """Parse edts container for elst box."""
        edts_end = offset + length
        curr = offset
        entries: List[EditListEntry] = []

        while curr < edts_end:
            h = self._read_box_header(curr)
            if h is None:
                break
            b_type, b_size, b_hsize, b_payload = h
            if b_type == b"elst":
                entries = self._parse_elst(b_payload, b_size - b_hsize)
            curr += b_size
        return entries

    def _parse_elst(self, offset: int, length: int) -> List[EditListEntry]:
        if length < 8:
            return []
        self.f.seek(offset)
        version = struct.unpack(">B", self.f.read(1))[0]
        self.f.seek(offset + 4)
        entry_count = struct.unpack(">I", self.f.read(4))[0]
        entries: List[EditListEntry] = []

        for _ in range(entry_count):
            if version == 0:
                seg_dur, media_time = struct.unpack(">Ii", self.f.read(8))
            else:
                seg_dur, media_time = struct.unpack(">Qq", self.f.read(16))
            rate_int, rate_frac = struct.unpack(">hh", self.f.read(4))
            media_rate = rate_int + (rate_frac / 65536.0)
            entries.append(EditListEntry(
                segment_duration=seg_dur,
                media_time=media_time,
                media_rate=media_rate
            ))
        return entries

    def _parse_mdhd(self, offset: int, length: int) -> Tuple[int, int]:
        if length < 20:
            return 1000, 0
        self.f.seek(offset)
        version = struct.unpack(">B", self.f.read(1))[0]
        self.f.read(3)

        if version == 0:
            self.f.read(8)
            timescale, duration = struct.unpack(">II", self.f.read(8))
        else:
            self.f.read(16)
            timescale, duration = struct.unpack(">IQ", self.f.read(12))

        return timescale, duration

    def _parse_hdlr(self, offset: int, length: int) -> bytes:
        if length < 16:
            return b""
        self.f.seek(offset + 8)
        handler_type = self.f.read(4)
        return handler_type

    def _parse_stsd_video(self, offset: int, length: int) -> Optional[Tuple[str, int, int, Optional[str]]]:
        if length < 16:
            return None
        self.f.seek(offset + 4)
        entry_count = struct.unpack(">I", self.f.read(4))[0]
        if entry_count == 0:
            return None

        entry_size, format_fourcc = struct.unpack(">I4s", self.f.read(8))
        codec_name = format_fourcc.decode("latin-1", errors="replace").strip()

        self.f.read(24)
        w, h = struct.unpack(">HH", self.f.read(4))

        self.f.read(14)
        compressor_len = struct.unpack(">B", self.f.read(1))[0]
        compressor_name = None
        if 0 < compressor_len <= 31:
            raw_cname = self.f.read(compressor_len)
            compressor_name = raw_cname.decode("utf-8", errors="replace").strip()

        return codec_name, w, h, compressor_name

    def _parse_stts(self, offset: int, length: int) -> List[Tuple[int, int]]:
        if length < 8:
            return []
        self.f.seek(offset + 4)
        entry_count = struct.unpack(">I", self.f.read(4))[0]
        entries = []
        for _ in range(entry_count):
            count, delta = struct.unpack(">II", self.f.read(8))
            entries.append((count, delta))
        return entries

    def _parse_ctts(self, offset: int, length: int) -> List[Tuple[int, int]]:
        if length < 8:
            return []
        self.f.seek(offset)
        version = struct.unpack(">B", self.f.read(1))[0]
        self.f.seek(offset + 4)
        entry_count = struct.unpack(">I", self.f.read(4))[0]
        entries = []
        for _ in range(entry_count):
            if version == 0:
                count, offset_val = struct.unpack(">II", self.f.read(8))
            else:
                count, offset_val = struct.unpack(">Ii", self.f.read(8))
            entries.append((count, offset_val))
        return entries

    def _parse_stss(self, offset: int, length: int) -> List[int]:
        if length < 8:
            return []
        self.f.seek(offset + 4)
        entry_count = struct.unpack(">I", self.f.read(4))[0]
        keyframes = []
        for _ in range(entry_count):
            sample_num = struct.unpack(">I", self.f.read(4))[0]
            keyframes.append(sample_num - 1)
        return keyframes

    def _parse_stsz_count(self, offset: int, length: int) -> int:
        if length < 12:
            return 0
        self.f.seek(offset + 4)
        sample_size, sample_count = struct.unpack(">II", self.f.read(8))
        return sample_count

    def _build_dts(self, stts_entries: List[Tuple[int, int]], expected_count: int) -> Tuple[List[int], int]:
        dts_list: List[int] = []
        current_dts = 0
        for count, delta in stts_entries:
            for _ in range(count):
                dts_list.append(current_dts)
                current_dts += delta

        total = len(dts_list) if dts_list else expected_count
        return dts_list, total

    def _build_pts(self, dts_list: List[int], ctts_entries: List[Tuple[int, int]]) -> List[int]:
        if not ctts_entries:
            return list(dts_list)

        offsets: List[int] = []
        for count, offset in ctts_entries:
            for _ in range(count):
                offsets.append(offset)

        if len(offsets) != len(dts_list):
            return list(dts_list)

        return [dts + off for dts, off in zip(dts_list, offsets)]
