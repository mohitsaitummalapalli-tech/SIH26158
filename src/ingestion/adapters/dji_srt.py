"""DJI SRT subtitle telemetry log parser and adapter."""

import re
import os
import math
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone

from src.ingestion.adapters.base import (
    TelemetryAdapter,
    ParsedTelemetryRecord,
    RecordStatus,
)
from src.ingestion.canonical_telemetry import (
    TelemetryRecord,
    TelemetryPosition,
    TelemetryOrientation,
    TelemetryVelocity,
    TelemetryQuality,
    TelemetryProvenance,
    AltitudeReference,
    PositionReference,
    TimestampSemantics,
    FixType,
)
from src.ingestion.exceptions import InvalidTelemetryDataError


class DJISRTAdapter(TelemetryAdapter):
    """Production parser for DJI video subtitle telemetry (.SRT) files."""

    # Regex for standard SRT timestamp line: 00:00:01,500 --> 00:00:01,533 or with '.' decimal
    SRT_TIMECODE_REGEX = re.compile(
        r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{3})\s*-->\s*(?P<end_h>\d{2}):(?P<end_m>\d{2}):(?P<end_s>\d{2})[,.](?P<end_ms>\d{3})"
    )

    # Regex for GPS(lon,lat,sat) or HOME(lon,lat)
    GPS_TUPLE_REGEX = re.compile(
        r"GPS\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)(?:\s*,\s*(\d+))?\s*\)", re.IGNORECASE
    )

    # Regex for speed tokens: H.S 5.20m/s, V.S 0.10m/s, H 120.50m
    SPEED_TOKEN_REGEX = re.compile(
        r"(?:H\.S|HS|H_S|Speed)\s*[:=]?\s*([-\d.,]+)(?:\s*m/s)?", re.IGNORECASE
    )
    VSPEED_TOKEN_REGEX = re.compile(
        r"(?:V\.S|VS|V_S|Vspeed)\s*[:=]?\s*([-\d.,]+)(?:\s*m/s)?", re.IGNORECASE
    )

    # Regex for ISO-like timestamp in text payload
    ISO_DATE_REGEX = re.compile(
        r"(\d{4}[-./]\d{2}[-./]\d{2}\s+\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)"
    )

    def _parse_timecode_to_seconds(self, h: str, m: str, s: str, ms: str) -> float:
        """Convert HH:MM:SS,mmm to float seconds."""
        return int(h) * 3600.0 + int(m) * 60.0 + int(s) + int(ms) / 1000.0

    def _clean_number(self, val_str: str) -> Optional[float]:
        """Parse float handling optional comma decimal and metric suffixes (e.g., '120.5m', '5.2m/s')."""
        if not val_str:
            return None
        cleaned = val_str.strip().rstrip("msMS/").strip().replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _parse_single_block(
        self, raw_index: int, block_lines: List[str]
    ) -> ParsedTelemetryRecord:
        raw_text = "\n".join(block_lines)
        if not block_lines:
            return ParsedTelemetryRecord(
                status=RecordStatus.INVALID,
                record=None,
                raw_record_index=raw_index,
                raw_text=raw_text,
                rejection_reason="Empty subtitle block.",
            )

        # 1. Parse Subtitle Timecode
        timestamp_sec: Optional[float] = None
        timecode_line_idx = -1
        for i, line in enumerate(block_lines):
            match = self.SRT_TIMECODE_REGEX.match(line.strip())
            if match:
                timestamp_sec = self._parse_timecode_to_seconds(
                    match.group("h"), match.group("m"), match.group("s"), match.group("ms")
                )
                timecode_line_idx = i
                break

        if timestamp_sec is None:
            return ParsedTelemetryRecord(
                status=RecordStatus.INVALID,
                record=None,
                raw_record_index=raw_index,
                raw_text=raw_text,
                rejection_reason="Missing or unparseable SRT timecode line.",
            )

        # 2. Extract Payload Lines
        payload_lines = block_lines[timecode_line_idx + 1 :]
        payload_text = " ".join(payload_lines)

        lat: Optional[float] = None
        lon: Optional[float] = None
        alt: Optional[float] = None
        rel_alt: Optional[float] = None
        abs_alt: Optional[float] = None
        alt_ref = AltitudeReference.UNKNOWN

        heading: Optional[float] = None
        pitch: Optional[float] = None
        roll: Optional[float] = None
        gb_pitch: Optional[float] = None
        gb_roll: Optional[float] = None
        gb_yaw: Optional[float] = None

        speed: Optional[float] = None
        climb_rate: Optional[float] = None
        satellites: Optional[int] = None
        timestamp_utc: Optional[str] = None
        extra_metadata: Dict[str, Any] = {}

        # Check for ISO date string in payload
        date_match = self.ISO_DATE_REGEX.search(payload_text)
        if date_match:
            timestamp_utc = date_match.group(1).replace(".", "-").replace("/", "-")

        # Parse all bracketed blocks: [ ... ]
        bracket_blocks = re.findall(r"\[([^\]]+)\]", payload_text)
        for block in bracket_blocks:
            # Find all key-value pairs within bracket
            pairs = re.findall(r"([a-zA-Z0-9_.]+)\s*[:=]\s*([-\d.,a-zA-Z_/]+)", block)
            if pairs:
                for k_raw, v_raw in pairs:
                    k = k_raw.lower().strip()
                    num_val = self._clean_number(v_raw)
                    if k in {"latitude", "lat"}:
                        lat = num_val
                    elif k in {"longitude", "lon", "long"}:
                        lon = num_val
                    elif k in {"altitude", "alt", "abs_alt"}:
                        abs_alt = num_val
                    elif k in {"rel_alt", "relative_alt", "height"}:
                        rel_alt = num_val
                    elif k in {"drone_heading", "heading", "yaw", "drone_yaw"}:
                        heading = num_val
                    elif k in {"drone_pitch", "pitch"}:
                        pitch = num_val
                    elif k in {"drone_roll", "roll"}:
                        roll = num_val
                    elif k in {"gimbal_pitch", "gb_pitch"}:
                        gb_pitch = num_val
                    elif k in {"gimbal_roll", "gb_roll"}:
                        gb_roll = num_val
                    elif k in {"gimbal_yaw", "gb_yaw"}:
                        gb_yaw = num_val
                    elif k in {"hspeed", "speed", "h_s", "h.s"}:
                        speed = num_val
                    elif k in {"vspeed", "v_s", "v.s", "climb_rate"}:
                        climb_rate = num_val
                    elif k in {"sats", "satellites", "gps_count"}:
                        satellites = int(num_val) if num_val is not None else None
                    else:
                        extra_metadata[k] = num_val if num_val is not None else v_raw.strip()
            else:
                # Bracket with single value or tag
                parts = block.strip().split()
                if len(parts) == 2:
                    k = parts[0].lower().strip()
                    num_val = self._clean_number(parts[1])
                    extra_metadata[k] = num_val if num_val is not None else parts[1]

        # Check for unbracketed GPS(lon, lat, sats) format
        gps_match = self.GPS_TUPLE_REGEX.search(payload_text)
        if gps_match:
            lon_val = self._clean_number(gps_match.group(1))
            lat_val = self._clean_number(gps_match.group(2))
            if lon is None:
                lon = lon_val
            if lat is None:
                lat = lat_val
            if gps_match.group(3) and satellites is None:
                satellites = int(gps_match.group(3))

        # Check for speed in unbracketed text (e.g. H.S 5.20m/s, V.S 0.10m/s)
        if speed is None:
            hs_match = self.SPEED_TOKEN_REGEX.search(payload_text)
            if hs_match:
                speed = self._clean_number(hs_match.group(1))

        if climb_rate is None:
            vs_match = self.VSPEED_TOKEN_REGEX.search(payload_text)
            if vs_match:
                climb_rate = self._clean_number(vs_match.group(1))

        # Resolve Altitude
        if abs_alt is not None:
            alt = abs_alt
            alt_ref = AltitudeReference.MSL
        elif rel_alt is not None:
            alt = rel_alt
            alt_ref = AltitudeReference.RELATIVE_TO_TAKEOFF
        else:
            alt = None
            alt_ref = AltitudeReference.UNKNOWN

        # 3. Validation & TelemetryRecord Construction
        if lat is None or lon is None or alt is None:
            return ParsedTelemetryRecord(
                status=RecordStatus.INVALID,
                record=None,
                raw_record_index=raw_index,
                raw_text=raw_text,
                rejection_reason=f"Missing essential spatial coordinates (lat={lat}, lon={lon}, alt={alt}).",
            )

        try:
            position = TelemetryPosition(
                latitude_deg=lat,
                longitude_deg=lon,
                altitude_meters=alt,
                altitude_reference=alt_ref,
                position_reference=PositionReference.WGS84_GEODETIC,
            )
        except InvalidTelemetryDataError as e:
            return ParsedTelemetryRecord(
                status=RecordStatus.INVALID,
                record=None,
                raw_record_index=raw_index,
                raw_text=raw_text,
                rejection_reason=f"Position validation failed: {str(e)}",
            )

        orientation = None
        if any(x is not None for x in [heading, pitch, roll, gb_pitch, gb_roll, gb_yaw]):
            try:
                orientation = TelemetryOrientation(
                    heading_deg=heading,
                    pitch_deg=pitch,
                    roll_deg=roll,
                    gimbal_pitch_deg=gb_pitch,
                    gimbal_roll_deg=gb_roll,
                    gimbal_yaw_deg=gb_yaw,
                )
            except InvalidTelemetryDataError as e:
                return ParsedTelemetryRecord(
                    status=RecordStatus.INVALID,
                    record=None,
                    raw_record_index=raw_index,
                    raw_text=raw_text,
                    rejection_reason=f"Orientation validation failed: {str(e)}",
                )

        velocity = None
        if speed is not None or climb_rate is not None:
            try:
                velocity = TelemetryVelocity(
                    speed_mps=speed,
                    climb_rate_mps=climb_rate,
                )
            except InvalidTelemetryDataError as e:
                return ParsedTelemetryRecord(
                    status=RecordStatus.INVALID,
                    record=None,
                    raw_record_index=raw_index,
                    raw_text=raw_text,
                    rejection_reason=f"Velocity validation failed: {str(e)}",
                )

        quality = None
        if satellites is not None:
            quality = TelemetryQuality(
                fix_type=FixType.FIX_3D if satellites >= 4 else FixType.NO_FIX,
                satellites_visible=satellites,
            )

        provenance = TelemetryProvenance(
            source_type="dji_srt",
            source_identifier=self.source_path,
            record_index=raw_index,
            extraction_method="DJISRTAdapter_v1.0",
            source_checksum=self.compute_sha256(),
        )

        try:
            record = TelemetryRecord(
                timestamp=timestamp_sec,
                position=position,
                timestamp_semantics=TimestampSemantics.VIDEO_RELATIVE,
                timestamp_utc=timestamp_utc,
                orientation=orientation,
                velocity=velocity,
                quality=quality,
                provenance=provenance,
                extra_metadata=extra_metadata,
            )
        except InvalidTelemetryDataError as e:
            return ParsedTelemetryRecord(
                status=RecordStatus.INVALID,
                record=None,
                raw_record_index=raw_index,
                raw_text=raw_text,
                rejection_reason=f"Record validation failed: {str(e)}",
            )

        # Status determination: VALID if position, orientation, and velocity all present; else PARTIALLY_VALID
        status = RecordStatus.VALID if (orientation is not None and velocity is not None) else RecordStatus.PARTIALLY_VALID

        return ParsedTelemetryRecord(
            status=status,
            record=record,
            raw_record_index=raw_index,
            raw_text=raw_text,
            rejection_reason=None,
        )

    def parse_records(self) -> List[ParsedTelemetryRecord]:
        """Parse all SRT subtitle blocks in the file into classified records."""
        with open(self.source_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Split into blocks separated by blank lines
        raw_blocks = re.split(r"\n\s*\n", content.strip())
        results: List[ParsedTelemetryRecord] = []

        for idx, block_str in enumerate(raw_blocks):
            lines = [l.strip() for l in block_str.strip().splitlines() if l.strip()]
            if not lines:
                continue
            parsed = self._parse_single_block(raw_index=idx, block_lines=lines)
            results.append(parsed)

        return results
