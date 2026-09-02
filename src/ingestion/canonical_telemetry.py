"""Canonical telemetry contracts, coordinate references, and validation rules."""

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Iterator
from src.ingestion.exceptions import InvalidTelemetryDataError


class AltitudeReference(str, Enum):
    """Reference datum for elevation and altitude measurements."""
    ELLIPSOIDAL = "ELLIPSOIDAL"           # Height above WGS84 reference ellipsoid (h)
    MSL = "MSL"                           # Orthometric height above Mean Sea Level / Geoid (H)
    AGL = "AGL"                           # Height Above Ground Level (radar/sonar/terrain-relative)
    RELATIVE_TO_TAKEOFF = "RELATIVE_TO_TAKEOFF" # Barometric height relative to home/takeoff point
    UNKNOWN = "UNKNOWN"                   # Default when unstated by source data


class PositionReference(str, Enum):
    """Spatial coordinate reference system for horizontal coordinates."""
    WGS84_GEODETIC = "WGS84_GEODETIC"     # Latitude, Longitude in decimal degrees (EPSG:4326)
    ECEF = "ECEF"                         # Earth-Centered Earth-Fixed Cartesian in meters (EPSG:4978)
    LOCAL_ENU = "LOCAL_ENU"               # Topocentric East-North-Up tangent plane in meters
    UNKNOWN = "UNKNOWN"


class TimestampSemantics(str, Enum):
    """Semantic origin and clock reference for telemetry timestamps."""
    GPS_TIME = "GPS_TIME"                 # Time from GPS constellation epoch
    UTC_TIMESTAMP = "UTC_TIMESTAMP"       # ISO 8601 UTC clock timestamp / epoch seconds
    VIDEO_RELATIVE = "VIDEO_RELATIVE"     # Elapsed presentation time from video stream start (t=0.0s)
    MONOTONIC_SYSTEM_TIME = "MONOTONIC_SYSTEM_TIME" # Hardware monotonic clock ticks / seconds
    SENSOR_LOG_TIME = "SENSOR_LOG_TIME"   # Flight controller internal log counter
    UNKNOWN = "UNKNOWN"


class FixType(str, Enum):
    """GNSS positioning solution quality status."""
    NO_FIX = "NO_FIX"
    FIX_2D = "2D_FIX"
    FIX_3D = "3D_FIX"
    DGPS = "DGPS"
    RTK_FLOAT = "RTK_FLOAT"
    RTK_FIXED = "RTK_FIXED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class TelemetryPosition:
    """3D spatial coordinate representation.
    
    Units & Bounds:
    - latitude_deg: Decimal degrees [-90.0, 90.0]
    - longitude_deg: Decimal degrees [-180.0, 180.0]
    - altitude_meters: Linear meters (finite float)
    - altitude_reference: Declared altitude datum (preserves UNKNOWN if unstated)
    - position_reference: Declared horizontal coordinate reference system
    """
    latitude_deg: float
    longitude_deg: float
    altitude_meters: float
    altitude_reference: AltitudeReference = AltitudeReference.UNKNOWN
    position_reference: PositionReference = PositionReference.WGS84_GEODETIC

    def __post_init__(self) -> None:
        if math.isnan(self.latitude_deg) or math.isinf(self.latitude_deg):
            raise InvalidTelemetryDataError(f"Telemetry latitude cannot be NaN or Infinite (got {self.latitude_deg}).")
        if self.latitude_deg < -90.0 or self.latitude_deg > 90.0:
            raise InvalidTelemetryDataError(f"Telemetry latitude out of valid range [-90.0, 90.0]: {self.latitude_deg}°")

        if math.isnan(self.longitude_deg) or math.isinf(self.longitude_deg):
            raise InvalidTelemetryDataError(f"Telemetry longitude cannot be NaN or Infinite (got {self.longitude_deg}).")
        if self.longitude_deg < -180.0 or self.longitude_deg > 180.0:
            raise InvalidTelemetryDataError(f"Telemetry longitude out of valid range [-180.0, 180.0]: {self.longitude_deg}°")

        if math.isnan(self.altitude_meters) or math.isinf(self.altitude_meters):
            raise InvalidTelemetryDataError(f"Telemetry altitude cannot be NaN or Infinite (got {self.altitude_meters}).")


@dataclass(frozen=True)
class TelemetryOrientation:
    """3D angular orientation representation for drone body and camera gimbal.
    
    Units & Conventions:
    - All angular quantities are strictly represented in DEGREES.
    - Preserves angular values without imposing vendor-specific physical stops in the canonical contract.
    - heading_deg: Drone flight heading / yaw in degrees.
    - pitch_deg: Drone body pitch in degrees.
    - roll_deg: Drone body roll in degrees.
    - gimbal_pitch_deg: Camera gimbal pitch in degrees.
    - gimbal_roll_deg: Camera gimbal roll in degrees.
    - gimbal_yaw_deg: Camera gimbal yaw / heading in degrees.
    """
    heading_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    gimbal_pitch_deg: Optional[float] = None
    gimbal_roll_deg: Optional[float] = None
    gimbal_yaw_deg: Optional[float] = None

    def __post_init__(self) -> None:
        for name, val in [
            ("heading_deg", self.heading_deg),
            ("pitch_deg", self.pitch_deg),
            ("roll_deg", self.roll_deg),
            ("gimbal_pitch_deg", self.gimbal_pitch_deg),
            ("gimbal_roll_deg", self.gimbal_roll_deg),
            ("gimbal_yaw_deg", self.gimbal_yaw_deg),
        ]:
            if val is not None and (math.isnan(val) or math.isinf(val)):
                raise InvalidTelemetryDataError(f"TelemetryOrientation field '{name}' cannot be NaN or Infinite (got {val}).")


@dataclass(frozen=True)
class TelemetryVelocity:
    """Kinematic velocity vector and ground speed in linear meters per second.
    
    Units:
    - speed_mps: Scalar 2D or 3D ground speed in meters per second (>= 0.0).
    - north_velocity_mps: Linear velocity component along True North (+North) in m/s.
    - east_velocity_mps: Linear velocity component along East (+East) in m/s.
    - down_velocity_mps: Linear velocity component along Down (+Down) in m/s.
    - climb_rate_mps: Vertical ascent rate (+Up) in m/s.
    """
    speed_mps: Optional[float] = None
    north_velocity_mps: Optional[float] = None
    east_velocity_mps: Optional[float] = None
    down_velocity_mps: Optional[float] = None
    climb_rate_mps: Optional[float] = None

    def __post_init__(self) -> None:
        if self.speed_mps is not None:
            if math.isnan(self.speed_mps) or math.isinf(self.speed_mps) or self.speed_mps < 0.0:
                raise InvalidTelemetryDataError(f"TelemetryVelocity 'speed_mps' must be non-negative finite float, got {self.speed_mps}.")


@dataclass(frozen=True)
class TelemetryQuality:
    """Quality and statistical uncertainty metadata associated with telemetry observation."""
    fix_type: FixType = FixType.UNKNOWN
    satellites_visible: Optional[int] = None
    hdop: Optional[float] = None
    vdop: Optional[float] = None
    pdop: Optional[float] = None
    horizontal_accuracy_meters: Optional[float] = None  # 1-sigma horizontal uncertainty in meters
    vertical_accuracy_meters: Optional[float] = None    # 1-sigma vertical uncertainty in meters


@dataclass(frozen=True)
class TelemetryProvenance:
    """Audit and lineage tracking metadata for an individual telemetry observation."""
    source_type: str = "unknown"             # e.g., "dji_srt", "ardupilot_csv", "embedded_klv"
    source_identifier: str = ""             # File path, stream ID, or device serial
    record_index: Optional[int] = None      # 0-indexed position in raw source data
    extraction_method: str = "CanonicalTelemetry_v1.0"
    source_checksum: Optional[str] = None   # SHA-256 checksum of raw source telemetry log


@dataclass(frozen=True)
class TelemetryRecord:
    """Canonical telemetry observation record representing the state of an aerial sensor platform.
    
    Timestamp & Reference Semantics:
    - timestamp: Observation time in seconds (finite float >= 0.0).
    - timestamp_semantics: Semantic origin (e.g. VIDEO_RELATIVE, UTC_TIMESTAMP, GPS_TIME).
    - position: Mandatory 3D spatial coordinate (lat, lon, alt, references).
    - orientation: Optional angular attitude (heading, pitch, roll, gimbal angles).
    - velocity: Optional kinematic motion in meters per second.
    - quality: Optional GNSS solution quality and uncertainty metrics.
    - provenance: Optional lineage tracking back to original source record.
    """
    timestamp: float
    position: TelemetryPosition
    timestamp_semantics: TimestampSemantics = TimestampSemantics.VIDEO_RELATIVE
    timestamp_utc: Optional[str] = None
    orientation: Optional[TelemetryOrientation] = None
    velocity: Optional[TelemetryVelocity] = None
    quality: Optional[TelemetryQuality] = None
    provenance: Optional[TelemetryProvenance] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        timestamp: Optional[float] = None,
        position: Optional[TelemetryPosition] = None,
        timestamp_semantics: TimestampSemantics = TimestampSemantics.VIDEO_RELATIVE,
        timestamp_utc: Optional[str] = None,
        orientation: Optional[TelemetryOrientation] = None,
        velocity: Optional[TelemetryVelocity] = None,
        quality: Optional[TelemetryQuality] = None,
        provenance: Optional[TelemetryProvenance] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        # Legacy Phase 0 backward-compatible parameters:
        timestamp_seconds: Optional[float] = None,
        latitude_deg: Optional[float] = None,
        longitude_deg: Optional[float] = None,
        altitude_meters: Optional[float] = None,
        gimbal_pitch_deg: Optional[float] = None,
        gimbal_roll_deg: Optional[float] = None,
        gimbal_yaw_deg: Optional[float] = None,
        drone_heading_deg: Optional[float] = None,
        speed_mps: Optional[float] = None,
        is_rtk_fixed: bool = False,
        accuracy_horizontal_meters: Optional[float] = None,
        accuracy_vertical_meters: Optional[float] = None,
    ) -> None:
        # Resolve timestamp
        resolved_ts = timestamp if timestamp is not None else timestamp_seconds
        if resolved_ts is None:
            raise InvalidTelemetryDataError("TelemetryRecord requires a valid, non-null timestamp.")
        if math.isnan(resolved_ts) or math.isinf(resolved_ts) or resolved_ts < 0.0:
            raise InvalidTelemetryDataError(f"TelemetryRecord timestamp must be non-negative finite float, got {resolved_ts}.")

        # Resolve position
        if position is not None:
            resolved_pos = position
        elif latitude_deg is not None and longitude_deg is not None and altitude_meters is not None:
            resolved_pos = TelemetryPosition(
                latitude_deg=latitude_deg,
                longitude_deg=longitude_deg,
                altitude_meters=altitude_meters
            )
        else:
            raise InvalidTelemetryDataError("TelemetryRecord requires a valid TelemetryPosition or (lat, lon, alt) coordinates.")

        # Resolve orientation
        resolved_ori = orientation
        if resolved_ori is None and any(x is not None for x in [gimbal_pitch_deg, gimbal_roll_deg, gimbal_yaw_deg, drone_heading_deg]):
            resolved_ori = TelemetryOrientation(
                heading_deg=drone_heading_deg,
                gimbal_pitch_deg=gimbal_pitch_deg,
                gimbal_roll_deg=gimbal_roll_deg,
                gimbal_yaw_deg=gimbal_yaw_deg,
            )

        # Resolve velocity
        resolved_vel = velocity
        if resolved_vel is None and speed_mps is not None:
            resolved_vel = TelemetryVelocity(speed_mps=speed_mps)

        # Resolve quality
        resolved_qual = quality
        if resolved_qual is None and (is_rtk_fixed or accuracy_horizontal_meters is not None or accuracy_vertical_meters is not None):
            resolved_qual = TelemetryQuality(
                fix_type=FixType.RTK_FIXED if is_rtk_fixed else FixType.UNKNOWN,
                horizontal_accuracy_meters=accuracy_horizontal_meters,
                vertical_accuracy_meters=accuracy_vertical_meters,
            )

        object.__setattr__(self, "timestamp", float(resolved_ts))
        object.__setattr__(self, "position", resolved_pos)
        object.__setattr__(self, "timestamp_semantics", timestamp_semantics)
        object.__setattr__(self, "timestamp_utc", timestamp_utc)
        object.__setattr__(self, "orientation", resolved_ori)
        object.__setattr__(self, "velocity", resolved_vel)
        object.__setattr__(self, "quality", resolved_qual)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "extra_metadata", extra_metadata or {})

    # Convenience properties for backwards compatibility with Phase 0/1A
    @property
    def timestamp_seconds(self) -> float:
        return self.timestamp

    @property
    def latitude_deg(self) -> float:
        return self.position.latitude_deg

    @property
    def longitude_deg(self) -> float:
        return self.position.longitude_deg

    @property
    def altitude_meters(self) -> float:
        return self.position.altitude_meters

    @property
    def gimbal_pitch_deg(self) -> Optional[float]:
        return self.orientation.gimbal_pitch_deg if self.orientation else None

    @property
    def gimbal_roll_deg(self) -> Optional[float]:
        return self.orientation.gimbal_roll_deg if self.orientation else None

    @property
    def gimbal_yaw_deg(self) -> Optional[float]:
        return self.orientation.gimbal_yaw_deg if self.orientation else None

    @property
    def drone_heading_deg(self) -> Optional[float]:
        return self.orientation.heading_deg if self.orientation else None

    @property
    def speed_mps(self) -> Optional[float]:
        return self.velocity.speed_mps if self.velocity else None

    @property
    def is_rtk_fixed(self) -> bool:
        if self.quality:
            return self.quality.fix_type == FixType.RTK_FIXED
        return False

    @property
    def accuracy_horizontal_meters(self) -> Optional[float]:
        return self.quality.horizontal_accuracy_meters if self.quality else None

    @property
    def accuracy_vertical_meters(self) -> Optional[float]:
        return self.quality.vertical_accuracy_meters if self.quality else None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize TelemetryRecord to dictionary representation."""
        return {
            "timestamp": self.timestamp,
            "timestamp_semantics": self.timestamp_semantics.value,
            "timestamp_utc": self.timestamp_utc,
            "position": asdict(self.position),
            "orientation": asdict(self.orientation) if self.orientation else None,
            "velocity": asdict(self.velocity) if self.velocity else None,
            "quality": asdict(self.quality) if self.quality else None,
            "provenance": asdict(self.provenance) if self.provenance else None,
            "extra_metadata": self.extra_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetryRecord":
        """Reconstruct TelemetryRecord from serialized dictionary representation."""
        pos_data = data["position"]
        position = TelemetryPosition(
            latitude_deg=pos_data["latitude_deg"],
            longitude_deg=pos_data["longitude_deg"],
            altitude_meters=pos_data["altitude_meters"],
            altitude_reference=AltitudeReference(pos_data.get("altitude_reference", AltitudeReference.UNKNOWN.value)),
            position_reference=PositionReference(pos_data.get("position_reference", PositionReference.WGS84_GEODETIC.value)),
        )

        orientation = None
        if data.get("orientation"):
            orientation = TelemetryOrientation(**data["orientation"])

        velocity = None
        if data.get("velocity"):
            velocity = TelemetryVelocity(**data["velocity"])

        quality = None
        if data.get("quality"):
            q_data = dict(data["quality"])
            q_data["fix_type"] = FixType(q_data.get("fix_type", FixType.UNKNOWN.value))
            quality = TelemetryQuality(**q_data)

        provenance = None
        if data.get("provenance"):
            provenance = TelemetryProvenance(**data["provenance"])

        return cls(
            timestamp=data["timestamp"],
            position=position,
            timestamp_semantics=TimestampSemantics(data.get("timestamp_semantics", TimestampSemantics.VIDEO_RELATIVE.value)),
            timestamp_utc=data.get("timestamp_utc"),
            orientation=orientation,
            velocity=velocity,
            quality=quality,
            provenance=provenance,
            extra_metadata=data.get("extra_metadata", {}),
        )


@dataclass
class CanonicalTelemetryStream:
    """Discrete, chronologically structured telemetry stream preserving original source records."""
    stream_id: str
    records: List[TelemetryRecord]
    provenance: Optional[TelemetryProvenance] = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> TelemetryRecord:
        return self.records[idx]

    def __iter__(self) -> Iterator[TelemetryRecord]:
        return iter(self.records)

    def has_duplicate_timestamps(self) -> bool:
        """Check if multiple records share identical timestamps."""
        seen = set()
        for r in self.records:
            if r.timestamp in seen:
                return True
            seen.add(r.timestamp)
        return False

    def get_chronological_records(self) -> List[TelemetryRecord]:
        """Return a sorted view of telemetry records without altering underlying source order."""
        return sorted(self.records, key=lambda r: r.timestamp)
