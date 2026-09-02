"""Configurable generic CSV telemetry log adapter."""

import csv
import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

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
from src.ingestion.exceptions import (
    InvalidTelemetryDataError,
    InvalidVideoMetadataError,
)


@dataclass(frozen=True)
class CSVColumnMapping:
    """Explicit configuration mapping CSV columns to canonical telemetry contracts."""
    timestamp_col: str
    latitude_col: str
    longitude_col: str
    altitude_col: str
    heading_col: Optional[str] = None
    pitch_col: Optional[str] = None
    roll_col: Optional[str] = None
    gimbal_pitch_col: Optional[str] = None
    gimbal_roll_col: Optional[str] = None
    gimbal_yaw_col: Optional[str] = None
    speed_col: Optional[str] = None
    climb_rate_col: Optional[str] = None
    satellites_col: Optional[str] = None
    fix_type_col: Optional[str] = None
    timestamp_utc_col: Optional[str] = None

    # Unit declarations
    timestamp_unit: str = "seconds"       # "seconds", "milliseconds", "microseconds"
    angle_unit: str = "degrees"           # "degrees", "radians"
    altitude_unit: str = "meters"         # "meters", "feet"
    speed_unit: str = "mps"               # "mps", "kmph", "knots"

    # Reference semantics
    timestamp_semantics: TimestampSemantics = TimestampSemantics.VIDEO_RELATIVE
    altitude_reference: AltitudeReference = AltitudeReference.UNKNOWN
    position_reference: PositionReference = PositionReference.WGS84_GEODETIC

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate mapping definition completeness and unit declarations."""
        if not self.timestamp_col:
            raise ValueError("CSVColumnMapping requires a non-empty 'timestamp_col'.")
        if not self.latitude_col:
            raise ValueError("CSVColumnMapping requires a non-empty 'latitude_col'.")
        if not self.longitude_col:
            raise ValueError("CSVColumnMapping requires a non-empty 'longitude_col'.")
        if not self.altitude_col:
            raise ValueError("CSVColumnMapping requires a non-empty 'altitude_col'.")

        valid_time_units = {"seconds", "milliseconds", "microseconds"}
        if self.timestamp_unit not in valid_time_units:
            raise ValueError(f"Ambiguous or unsupported timestamp_unit '{self.timestamp_unit}'. Supported: {valid_time_units}")

        valid_angle_units = {"degrees", "radians"}
        if self.angle_unit not in valid_angle_units:
            raise ValueError(f"Ambiguous or unsupported angle_unit '{self.angle_unit}'. Supported: {valid_angle_units}")

        valid_alt_units = {"meters", "feet"}
        if self.altitude_unit not in valid_alt_units:
            raise ValueError(f"Ambiguous or unsupported altitude_unit '{self.altitude_unit}'. Supported: {valid_alt_units}")


class GenericCSVAdapter(TelemetryAdapter):
    """Production adapter for structured CSV/TSV flight logs with explicit mapping."""

    def __init__(self, source_path: str, mapping: CSVColumnMapping) -> None:
        super().__init__(source_path)
        mapping.validate()
        self.mapping = mapping

    def _convert_timestamp(self, raw_val: float) -> float:
        unit = self.mapping.timestamp_unit.lower()
        if unit == "seconds":
            return raw_val
        elif unit == "milliseconds":
            return raw_val / 1000.0
        elif unit == "microseconds":
            return raw_val / 1000000.0
        return raw_val

    def _convert_angle(self, raw_val: Optional[float]) -> Optional[float]:
        if raw_val is None:
            return None
        if self.mapping.angle_unit.lower() == "radians":
            return math.degrees(raw_val)
        return raw_val

    def _convert_altitude(self, raw_val: float) -> float:
        if self.mapping.altitude_unit.lower() == "feet":
            return raw_val * 0.3048
        return raw_val

    def _convert_speed(self, raw_val: Optional[float]) -> Optional[float]:
        if raw_val is None:
            return None
        unit = self.mapping.speed_unit.lower()
        if unit == "kmph":
            return raw_val / 3.6
        elif unit == "knots":
            return raw_val * 0.514444
        return raw_val

    def _parse_row(self, row_idx: int, row: Dict[str, str], raw_text: str) -> ParsedTelemetryRecord:
        m = self.mapping
        try:
            # 1. Mandatory Fields
            if not row.get(m.timestamp_col) or not row.get(m.latitude_col) or not row.get(m.longitude_col) or not row.get(m.altitude_col):
                return ParsedTelemetryRecord(
                    status=RecordStatus.INVALID,
                    record=None,
                    raw_record_index=row_idx,
                    raw_text=raw_text,
                    rejection_reason="Missing mandatory coordinate/timestamp values in CSV row.",
                )

            raw_ts = float(row[m.timestamp_col].strip().replace(",", "."))
            ts = self._convert_timestamp(raw_ts)

            raw_lat = float(row[m.latitude_col].strip().replace(",", "."))
            raw_lon = float(row[m.longitude_col].strip().replace(",", "."))
            raw_alt = float(row[m.altitude_col].strip().replace(",", "."))
            alt = self._convert_altitude(raw_alt)

            position = TelemetryPosition(
                latitude_deg=raw_lat,
                longitude_deg=raw_lon,
                altitude_meters=alt,
                altitude_reference=m.altitude_reference,
                position_reference=m.position_reference,
            )

            # 2. Orientation Fields
            heading = self._convert_angle(float(row[m.heading_col].strip()) if m.heading_col and row.get(m.heading_col) else None)
            pitch = self._convert_angle(float(row[m.pitch_col].strip()) if m.pitch_col and row.get(m.pitch_col) else None)
            roll = self._convert_angle(float(row[m.roll_col].strip()) if m.roll_col and row.get(m.roll_col) else None)
            gb_pitch = self._convert_angle(float(row[m.gimbal_pitch_col].strip()) if m.gimbal_pitch_col and row.get(m.gimbal_pitch_col) else None)
            gb_roll = self._convert_angle(float(row[m.gimbal_roll_col].strip()) if m.gimbal_roll_col and row.get(m.gimbal_roll_col) else None)
            gb_yaw = self._convert_angle(float(row[m.gimbal_yaw_col].strip()) if m.gimbal_yaw_col and row.get(m.gimbal_yaw_col) else None)

            orientation = None
            if any(x is not None for x in [heading, pitch, roll, gb_pitch, gb_roll, gb_yaw]):
                orientation = TelemetryOrientation(
                    heading_deg=heading,
                    pitch_deg=pitch,
                    roll_deg=roll,
                    gimbal_pitch_deg=gb_pitch,
                    gimbal_roll_deg=gb_roll,
                    gimbal_yaw_deg=gb_yaw,
                )

            # 3. Velocity Fields
            speed = self._convert_speed(float(row[m.speed_col].strip()) if m.speed_col and row.get(m.speed_col) else None)
            climb = self._convert_speed(float(row[m.climb_rate_col].strip()) if m.climb_rate_col and row.get(m.climb_rate_col) else None)
            velocity = None
            if speed is not None or climb is not None:
                velocity = TelemetryVelocity(speed_mps=speed, climb_rate_mps=climb)

            # 4. Quality Fields
            sats = int(row[m.satellites_col].strip()) if m.satellites_col and row.get(m.satellites_col) else None
            quality = None
            if sats is not None:
                quality = TelemetryQuality(
                    fix_type=FixType.FIX_3D if sats >= 4 else FixType.NO_FIX,
                    satellites_visible=sats,
                )

            provenance = TelemetryProvenance(
                source_type="generic_csv",
                source_identifier=self.source_path,
                record_index=row_idx,
                extraction_method="GenericCSVAdapter_v1.0",
                source_checksum=self.compute_sha256(),
            )

            ts_utc = row.get(m.timestamp_utc_col).strip() if m.timestamp_utc_col and row.get(m.timestamp_utc_col) else None

            record = TelemetryRecord(
                timestamp=ts,
                position=position,
                timestamp_semantics=m.timestamp_semantics,
                timestamp_utc=ts_utc,
                orientation=orientation,
                velocity=velocity,
                quality=quality,
                provenance=provenance,
            )

            status = RecordStatus.VALID if (orientation is not None and velocity is not None) else RecordStatus.PARTIALLY_VALID
            return ParsedTelemetryRecord(
                status=status,
                record=record,
                raw_record_index=row_idx,
                raw_text=raw_text,
                rejection_reason=None,
            )

        except (ValueError, InvalidTelemetryDataError) as e:
            return ParsedTelemetryRecord(
                status=RecordStatus.INVALID,
                record=None,
                raw_record_index=row_idx,
                raw_text=raw_text,
                rejection_reason=f"CSV row parsing/validation failed: {str(e)}",
            )

    def parse_records(self) -> List[ParsedTelemetryRecord]:
        """Parse all rows from the CSV file."""
        results: List[ParsedTelemetryRecord] = []
        with open(self.source_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return results

            # Verify required mapped columns exist in header
            m = self.mapping
            required_cols = [m.timestamp_col, m.latitude_col, m.longitude_col, m.altitude_col]
            missing_cols = [col for col in required_cols if col not in reader.fieldnames]
            if missing_cols:
                raise ValueError(f"CSV header is missing required mapped columns: {missing_cols}")

            for idx, row in enumerate(reader):
                raw_text = ",".join([str(v) for v in row.values()])
                parsed = self._parse_row(row_idx=idx, row=row, raw_text=raw_text)
                results.append(parsed)

        return results
