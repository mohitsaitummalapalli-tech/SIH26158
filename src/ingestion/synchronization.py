"""Temporal synchronization engine: associates video timelines with telemetry streams in metric Cartesian space."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Iterator

from src.ingestion.canonical_timeline import CanonicalFrame, CanonicalTimeline
from src.ingestion.canonical_telemetry import (
    TelemetryRecord,
    TelemetryPosition,
    TelemetryOrientation,
    TelemetryVelocity,
    TelemetryQuality,
    TelemetryProvenance,
    CanonicalTelemetryStream,
    AltitudeReference,
    PositionReference,
    TimestampSemantics,
    FixType,
)
from src.ingestion.exceptions import IngestionError


# --- WGS84 Geodetic Ellipsoid Constants ---
WGS84_A = 6378137.0                      # Semi-major axis in meters
WGS84_F = 1.0 / 298.257223563            # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)      # Semi-minor axis in meters (6356752.314245)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)     # First eccentricity squared (~0.00669437999014)
WGS84_EP2 = (WGS84_A**2 - WGS84_B**2) / (WGS84_B**2) # Second eccentricity squared


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> Tuple[float, float, float]:
    """Convert WGS84 geodetic coordinates (lat, lon in degrees, alt in meters) to ECEF Cartesian (X, Y, Z in meters)."""
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)

    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_lam = math.sin(lam)
    cos_lam = math.cos(lam)

    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)

    x = (n + alt_m) * cos_phi * cos_lam
    y = (n + alt_m) * cos_phi * sin_lam
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_phi

    return (x, y, z)


def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert ECEF Cartesian coordinates (X, Y, Z in meters) to WGS84 geodetic (lat, lon in degrees, alt in meters).
    
    Uses Bowring's closed-form method with sub-millimeter precision.
    """
    p = math.sqrt(x * x + y * y)
    if p < 1e-6:
        # Polar singularity
        lat_deg = 90.0 if z > 0 else -90.0
        lon_deg = 0.0
        alt_m = abs(z) - WGS84_B
        return (lat_deg, lon_deg, alt_m)

    theta = math.atan2(z * WGS84_A, p * WGS84_B)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    phi = math.atan2(
        z + WGS84_EP2 * WGS84_B * (sin_theta**3),
        p - WGS84_E2 * WGS84_A * (cos_theta**3)
    )
    lam = math.atan2(y, x)

    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)

    lat_deg = math.degrees(phi)
    lon_deg = math.degrees(lam)
    alt_m = (p / cos_phi) - n if abs(cos_phi) > 1e-4 else (z / sin_phi) - n * (1.0 - WGS84_E2)

    return (lat_deg, lon_deg, alt_m)


class SyncStatus(str, Enum):
    """Synchronization classification status for a video frame observation."""
    EXACT = "EXACT"                         # Exact timestamp match with a telemetry observation
    INTERPOLATED = "INTERPOLATED"           # Interpolated in metric Cartesian space between bounding observations
    NEAREST = "NEAREST"                     # Matched to nearest telemetry observation (within tolerance)
    OFFSET_APPLIED = "OFFSET_APPLIED"       # Synchronized using a configured constant time offset
    OFFSET_ESTIMATED = "OFFSET_ESTIMATED"   # Synchronized using an empirically estimated time offset
    OFFSET_VALIDATED = "OFFSET_VALIDATED"   # Synchronized using an empirically validated time offset
    EXTRAPOLATED = "EXTRAPOLATED"           # Extrapolated outside telemetry boundary (explicitly enabled)
    UNSYNCHRONIZED = "UNSYNCHRONIZED"       # Telemetry gap exceeded max_interpolation_gap
    OUT_OF_RANGE = "OUT_OF_RANGE"           # Frame timestamp falls before telemetry start or after telemetry end
    INCOMPATIBLE_REFERENCE = "INCOMPATIBLE_REFERENCE" # Incompatible vertical or horizontal coordinate datums

    # Backward compatibility alias
    OFFSET_CORRECTED = "OFFSET_APPLIED"


class ClockOffsetStatus(str, Enum):
    """Declared status and provenance of clock offset applied during synchronization."""
    IDENTITY = "IDENTITY"                   # No offset applied (identity zero bias)
    KNOWN_APPLIED = "KNOWN_APPLIED"         # Constant offset supplied by configuration (unvalidated)
    ESTIMATED = "ESTIMATED"                 # Offset estimated via correlation / visual feature alignment
    VALIDATED = "VALIDATED"                 # Offset validated against ground-truth timing reference
    UNKNOWN = "UNKNOWN"                     # Unspecified timing relationship


# --- Rotation Math for Safe Orientation SLERP ---

def _euler_to_quaternion(yaw_deg: float, pitch_deg: float, roll_deg: float) -> Tuple[float, float, float, float]:
    """Convert Z-Y-X Euler angles (yaw, pitch, roll in degrees) to unit quaternion (w, x, y, z)."""
    psi = math.radians(yaw_deg)
    theta = math.radians(pitch_deg)
    phi = math.radians(roll_deg)

    cy = math.cos(psi * 0.5)
    sy = math.sin(psi * 0.5)
    cp = math.cos(theta * 0.5)
    sp = math.sin(theta * 0.5)
    cr = math.cos(phi * 0.5)
    sr = math.sin(phi * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm > 0:
        return (qw / norm, qx / norm, qy / norm, qz / norm)
    return (1.0, 0.0, 0.0, 0.0)


def _quaternion_to_euler(w: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert unit quaternion (w, x, y, z) back to Z-Y-X Euler angles (yaw, pitch, roll in degrees)."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    yaw_deg = math.degrees(yaw) % 360.0
    pitch_deg = math.degrees(pitch)
    roll_deg = math.degrees(roll)

    return (yaw_deg, pitch_deg, roll_deg)


def _slerp_quaternions(
    q0: Tuple[float, float, float, float],
    q1: Tuple[float, float, float, float],
    alpha: float
) -> Tuple[float, float, float, float]:
    """Spherical Linear Interpolation (SLERP) on SO(3) unit quaternions along shortest arc."""
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1

    dot = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1

    if dot < 0.0:
        w1, x1, y1, z1 = -w1, -x1, -y1, -z1
        dot = -dot

    if dot > 0.9995:
        w = w0 + alpha * (w1 - w0)
        x = x0 + alpha * (x1 - x0)
        y = y0 + alpha * (y1 - y0)
        z = z0 + alpha * (z1 - z0)
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        return (w / norm, x / norm, y / norm, z / norm)

    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    w = s0 * w0 + s1 * w1
    x = s0 * x0 + s1 * x1
    y = s0 * y0 + s1 * y1
    z = s0 * z0 + s1 * z1

    norm = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / norm, x / norm, y / norm, z / norm)


# --- Clock Offset Models ---

class ClockOffsetModel(ABC):
    """Abstract mapping between video presentation time and telemetry time domain: t_telemetry = f(t_video)."""

    @abstractmethod
    def video_to_telemetry_time(self, video_timestamp: float) -> float:
        """Map video timestamp to telemetry time domain."""
        pass

    @abstractmethod
    def get_offset_seconds(self) -> float:
        """Return scalar time offset added to video timestamps."""
        pass

    @abstractmethod
    def get_offset_status(self) -> ClockOffsetStatus:
        """Return declared provenance/status of the offset."""
        pass


class IdentityClockModel(ClockOffsetModel):
    """Direct identity mapping assuming synchronized zero-epoch clocks: t_telemetry = t_video."""

    def video_to_telemetry_time(self, video_timestamp: float) -> float:
        return video_timestamp

    def get_offset_seconds(self) -> float:
        return 0.0

    def get_offset_status(self) -> ClockOffsetStatus:
        return ClockOffsetStatus.IDENTITY


class ConstantOffsetClockModel(ClockOffsetModel):
    """Constant time bias model: t_telemetry = t_video + delta_t."""

    def __init__(self, offset_seconds: float, status: ClockOffsetStatus = ClockOffsetStatus.KNOWN_APPLIED) -> None:
        if math.isnan(offset_seconds) or math.isinf(offset_seconds):
            raise ValueError(f"Constant clock offset cannot be NaN or Infinite (got {offset_seconds}).")
        self.offset_seconds = float(offset_seconds)
        self.status = status

    def video_to_telemetry_time(self, video_timestamp: float) -> float:
        return video_timestamp + self.offset_seconds

    def get_offset_seconds(self) -> float:
        return self.offset_seconds

    def get_offset_status(self) -> ClockOffsetStatus:
        return self.status


# --- Synchronization Configuration & Result Contracts ---

@dataclass(frozen=True)
class SynchronizationConfig:
    """Configuration parameters controlling temporal matching, interpolation, and tolerances."""
    max_interpolation_gap_seconds: float = 2.0  # Max gap beyond which interpolation is refused (UNSYNCHRONIZED)
    exact_match_tolerance_seconds: float = 1e-4  # Epsilon for identifying identical timestamps (0.1ms)
    allow_extrapolation: bool = False           # Whether to linearly extrapolate outside telemetry coverage
    interpolate_orientation: bool = True        # Enable safe quaternion SLERP for attitude angles
    use_ecef_interpolation: bool = True         # Use metric ECEF Cartesian interpolation for geodetic positions
    clock_model: ClockOffsetModel = field(default_factory=IdentityClockModel)


@dataclass(frozen=True)
class SynchronizedFrameObservation:
    """Rigorous temporal association between a video frame and spatial telemetry.
    
    Scientific Principle:
    - bracketing_interval_seconds: Diagnostic spacing between bounding telemetry samples (t_1 - t_0).
    - interpolation_fraction: Parameter alpha in [0.0, 1.0] indicating temporal placement.
    - timebase_uncertainty_seconds: Empirical timebase jitter if measured (distinct from gap size).
    """
    video_frame_id: str
    video_timestamp_seconds: float
    telemetry_timestamp_seconds: Optional[float]
    status: SyncStatus
    position: Optional[TelemetryPosition]
    orientation: Optional[TelemetryOrientation] = None
    velocity: Optional[TelemetryVelocity] = None
    quality: Optional[TelemetryQuality] = None
    interpolation_method: str = "none"
    interpolation_fraction: Optional[float] = None
    bracketing_interval_seconds: Optional[float] = None
    clock_offset_seconds: float = 0.0
    clock_offset_status: ClockOffsetStatus = ClockOffsetStatus.IDENTITY
    timebase_uncertainty_seconds: Optional[float] = None
    source_record_indices: List[int] = field(default_factory=list)
    source_provenance: Optional[TelemetryProvenance] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_synchronized(self) -> bool:
        """Return True if observation contains valid, synchronized spatial coordinates."""
        return self.status in {
            SyncStatus.EXACT,
            SyncStatus.INTERPOLATED,
            SyncStatus.NEAREST,
            SyncStatus.OFFSET_APPLIED,
            SyncStatus.OFFSET_ESTIMATED,
            SyncStatus.OFFSET_VALIDATED,
        }

    # Backward compatibility properties
    @property
    def time_offset_applied(self) -> float:
        return self.clock_offset_seconds

    @property
    def temporal_uncertainty_seconds(self) -> Optional[float]:
        """Deprecated alias: returns bracketing_interval_seconds / 2.0 for diagnostic backwards-compatibility."""
        if self.bracketing_interval_seconds is not None:
            return self.bracketing_interval_seconds * 0.5
        return self.timebase_uncertainty_seconds


@dataclass
class SynchronizedTrajectory:
    """Sequence of synchronized frame-telemetry observations representing flight path."""
    trajectory_id: str
    observations: List[SynchronizedFrameObservation]
    config: SynchronizationConfig
    provenance_summary: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> SynchronizedFrameObservation:
        return self.observations[idx]

    def __iter__(self) -> Iterator[SynchronizedFrameObservation]:
        return iter(self.observations)

    @property
    def synchronized_count(self) -> int:
        """Count of frames successfully synchronized with telemetry."""
        return sum(1 for obs in self.observations if obs.is_synchronized)

    @property
    def unsynchronized_count(self) -> int:
        """Count of frames with out-of-range or unsynchronized status."""
        return len(self.observations) - self.synchronized_count


# --- Synchronization Engine Implementation ---

class TemporalSynchronizationEngine:
    """Core synchronization engine aligning CanonicalTimeline with CanonicalTelemetryStream."""

    @classmethod
    def are_altitude_references_compatible(cls, ref0: AltitudeReference, ref1: AltitudeReference) -> bool:
        """Verify vertical datum compatibility. Incompatible datums must not be interpolated."""
        if ref0 == ref1:
            return True
        # Specific conversions would be handled via geoid models in georeferencing, not direct interpolation
        return False

    @classmethod
    def _interpolate_position_ecef(
        cls, pos0: TelemetryPosition, pos1: TelemetryPosition, alpha: float
    ) -> TelemetryPosition:
        """Metric Cartesian ECEF interpolation, correctly handling ellipsoidal curvature and antimeridian wraparound."""
        # Convert WGS84 Geodetic to ECEF (X, Y, Z in meters)
        x0, y0, z0 = geodetic_to_ecef(pos0.latitude_deg, pos0.longitude_deg, pos0.altitude_meters)
        x1, y1, z1 = geodetic_to_ecef(pos1.latitude_deg, pos1.longitude_deg, pos1.altitude_meters)

        # Linear interpolation in metric Euclidean ECEF space
        x = x0 + alpha * (x1 - x0)
        y = y0 + alpha * (y1 - y0)
        z = z0 + alpha * (z1 - z0)

        # Convert interpolated ECEF back to WGS84 Geodetic (lat, lon, alt)
        lat_deg, lon_deg, alt_m = ecef_to_geodetic(x, y, z)

        return TelemetryPosition(
            latitude_deg=lat_deg,
            longitude_deg=lon_deg,
            altitude_meters=alt_m,
            altitude_reference=pos0.altitude_reference,
            position_reference=PositionReference.WGS84_GEODETIC,
        )

    @classmethod
    def _interpolate_position_geodetic(
        cls, pos0: TelemetryPosition, pos1: TelemetryPosition, alpha: float
    ) -> TelemetryPosition:
        """Shortest-arc geodetic interpolation with antimeridian wraparound protection."""
        lat = pos0.latitude_deg + alpha * (pos1.latitude_deg - pos0.latitude_deg)

        # Shortest-path longitude wrap handling (+179.9 to -179.9 does NOT traverse 0 deg)
        dlon = (pos1.longitude_deg - pos0.longitude_deg + 180.0) % 360.0 - 180.0
        lon = (pos0.longitude_deg + alpha * dlon + 180.0) % 360.0 - 180.0

        alt = pos0.altitude_meters + alpha * (pos1.altitude_meters - pos0.altitude_meters)

        return TelemetryPosition(
            latitude_deg=lat,
            longitude_deg=lon,
            altitude_meters=alt,
            altitude_reference=pos0.altitude_reference,
            position_reference=pos0.position_reference,
        )

    @classmethod
    def _interpolate_orientation(
        cls, ori0: TelemetryOrientation, ori1: TelemetryOrientation, alpha: float
    ) -> TelemetryOrientation:
        """Rotation-aware Spherical Linear Interpolation (SLERP) for attitude angles."""
        heading: Optional[float] = None
        pitch: Optional[float] = None
        roll: Optional[float] = None

        if all(x is not None for x in [ori0.heading_deg, ori0.pitch_deg, ori0.roll_deg, ori1.heading_deg, ori1.pitch_deg, ori1.roll_deg]):
            q0 = _euler_to_quaternion(ori0.heading_deg, ori0.pitch_deg, ori0.roll_deg)
            q1 = _euler_to_quaternion(ori1.heading_deg, ori1.pitch_deg, ori1.roll_deg)
            q_interp = _slerp_quaternions(q0, q1, alpha)
            heading, pitch, roll = _quaternion_to_euler(*q_interp)

        gb_pitch: Optional[float] = None
        gb_roll: Optional[float] = None
        gb_yaw: Optional[float] = None

        if ori0.gimbal_pitch_deg is not None and ori1.gimbal_pitch_deg is not None:
            gb_pitch = ori0.gimbal_pitch_deg + alpha * (ori1.gimbal_pitch_deg - ori0.gimbal_pitch_deg)
        if ori0.gimbal_roll_deg is not None and ori1.gimbal_roll_deg is not None:
            gb_roll = ori0.gimbal_roll_deg + alpha * (ori1.gimbal_roll_deg - ori0.gimbal_roll_deg)
        if ori0.gimbal_yaw_deg is not None and ori1.gimbal_yaw_deg is not None:
            dyaw = (ori1.gimbal_yaw_deg - ori0.gimbal_yaw_deg + 180.0) % 360.0 - 180.0
            gb_yaw = (ori0.gimbal_yaw_deg + alpha * dyaw) % 360.0

        return TelemetryOrientation(
            heading_deg=heading,
            pitch_deg=pitch,
            roll_deg=roll,
            gimbal_pitch_deg=gb_pitch,
            gimbal_roll_deg=gb_roll,
            gimbal_yaw_deg=gb_yaw,
        )

    @classmethod
    def _interpolate_velocity(
        cls, vel0: TelemetryVelocity, vel1: TelemetryVelocity, alpha: float
    ) -> TelemetryVelocity:
        """Linear interpolation of speed and velocity components."""
        speed = None
        if vel0.speed_mps is not None and vel1.speed_mps is not None:
            speed = vel0.speed_mps + alpha * (vel1.speed_mps - vel0.speed_mps)

        climb = None
        if vel0.climb_rate_mps is not None and vel1.climb_rate_mps is not None:
            climb = vel0.climb_rate_mps + alpha * (vel1.climb_rate_mps - vel0.climb_rate_mps)

        north = None
        if vel0.north_velocity_mps is not None and vel1.north_velocity_mps is not None:
            north = vel0.north_velocity_mps + alpha * (vel1.north_velocity_mps - vel0.north_velocity_mps)

        east = None
        if vel0.east_velocity_mps is not None and vel1.east_velocity_mps is not None:
            east = vel0.east_velocity_mps + alpha * (vel1.east_velocity_mps - vel0.east_velocity_mps)

        down = None
        if vel0.down_velocity_mps is not None and vel1.down_velocity_mps is not None:
            down = vel0.down_velocity_mps + alpha * (vel1.down_velocity_mps - vel0.down_velocity_mps)

        return TelemetryVelocity(
            speed_mps=speed,
            north_velocity_mps=north,
            east_velocity_mps=east,
            down_velocity_mps=down,
            climb_rate_mps=climb,
        )

    @classmethod
    def synchronize(
        cls,
        timeline: CanonicalTimeline,
        telemetry_stream: CanonicalTelemetryStream,
        config: Optional[SynchronizationConfig] = None,
    ) -> SynchronizedTrajectory:
        """Synchronize video frames with telemetry stream with provenance and metric spatial safety."""
        cfg = config or SynchronizationConfig()
        offset_sec = cfg.clock_model.get_offset_seconds()
        offset_status = cfg.clock_model.get_offset_status()

        sorted_records = telemetry_stream.get_chronological_records()
        observations: List[SynchronizedFrameObservation] = []

        if not sorted_records:
            for frame in timeline:
                obs = SynchronizedFrameObservation(
                    video_frame_id=frame.frame_id,
                    video_timestamp_seconds=frame.timestamp_seconds,
                    telemetry_timestamp_seconds=None,
                    status=SyncStatus.UNSYNCHRONIZED,
                    position=None,
                    interpolation_method="none",
                    clock_offset_seconds=offset_sec,
                    clock_offset_status=offset_status,
                )
                observations.append(obs)
            return SynchronizedTrajectory(
                trajectory_id=timeline.video_id,
                observations=observations,
                config=cfg,
            )

        t_min = sorted_records[0].timestamp
        t_max = sorted_records[-1].timestamp

        for frame in timeline:
            v_ts = frame.timestamp_seconds
            t_target = cfg.clock_model.video_to_telemetry_time(v_ts)

            # 1. Out-of-Range Check
            if t_target < t_min:
                if cfg.allow_extrapolation:
                    rec0 = sorted_records[0]
                    obs = SynchronizedFrameObservation(
                        video_frame_id=frame.frame_id,
                        video_timestamp_seconds=v_ts,
                        telemetry_timestamp_seconds=rec0.timestamp,
                        status=SyncStatus.EXTRAPOLATED,
                        position=rec0.position,
                        orientation=rec0.orientation,
                        velocity=rec0.velocity,
                        quality=rec0.quality,
                        interpolation_method="extrapolated_hold_start",
                        clock_offset_seconds=offset_sec,
                        clock_offset_status=offset_status,
                        source_record_indices=[rec0.provenance.record_index] if rec0.provenance and rec0.provenance.record_index is not None else [0],
                        source_provenance=rec0.provenance,
                    )
                else:
                    obs = SynchronizedFrameObservation(
                        video_frame_id=frame.frame_id,
                        video_timestamp_seconds=v_ts,
                        telemetry_timestamp_seconds=None,
                        status=SyncStatus.OUT_OF_RANGE,
                        position=None,
                        interpolation_method="none",
                        clock_offset_seconds=offset_sec,
                        clock_offset_status=offset_status,
                    )
                observations.append(obs)
                continue

            if t_target > t_max:
                if cfg.allow_extrapolation:
                    rec_last = sorted_records[-1]
                    obs = SynchronizedFrameObservation(
                        video_frame_id=frame.frame_id,
                        video_timestamp_seconds=v_ts,
                        telemetry_timestamp_seconds=rec_last.timestamp,
                        status=SyncStatus.EXTRAPOLATED,
                        position=rec_last.position,
                        orientation=rec_last.orientation,
                        velocity=rec_last.velocity,
                        quality=rec_last.quality,
                        interpolation_method="extrapolated_hold_end",
                        clock_offset_seconds=offset_sec,
                        clock_offset_status=offset_status,
                        source_record_indices=[rec_last.provenance.record_index] if rec_last.provenance and rec_last.provenance.record_index is not None else [len(sorted_records) - 1],
                        source_provenance=rec_last.provenance,
                    )
                else:
                    obs = SynchronizedFrameObservation(
                        video_frame_id=frame.frame_id,
                        video_timestamp_seconds=v_ts,
                        telemetry_timestamp_seconds=None,
                        status=SyncStatus.OUT_OF_RANGE,
                        position=None,
                        interpolation_method="none",
                        clock_offset_seconds=offset_sec,
                        clock_offset_status=offset_status,
                    )
                observations.append(obs)
                continue

            # 2. Binary search bounding interval [rec_left, rec_right]
            low = 0
            high = len(sorted_records) - 1
            left_idx = 0
            while low <= high:
                mid = (low + high) // 2
                if sorted_records[mid].timestamp <= t_target:
                    left_idx = mid
                    low = mid + 1
                else:
                    high = mid - 1

            rec_left = sorted_records[left_idx]

            # Exact match check
            if abs(rec_left.timestamp - t_target) <= cfg.exact_match_tolerance_seconds:
                status = SyncStatus.OFFSET_APPLIED if offset_sec != 0.0 else SyncStatus.EXACT
                obs = SynchronizedFrameObservation(
                    video_frame_id=frame.frame_id,
                    video_timestamp_seconds=v_ts,
                    telemetry_timestamp_seconds=rec_left.timestamp,
                    status=status,
                    position=rec_left.position,
                    orientation=rec_left.orientation,
                    velocity=rec_left.velocity,
                    quality=rec_left.quality,
                    interpolation_method="exact_match",
                    interpolation_fraction=0.0,
                    bracketing_interval_seconds=0.0,
                    clock_offset_seconds=offset_sec,
                    clock_offset_status=offset_status,
                    source_record_indices=[rec_left.provenance.record_index] if rec_left.provenance and rec_left.provenance.record_index is not None else [left_idx],
                    source_provenance=rec_left.provenance,
                    timebase_uncertainty_seconds=abs(rec_left.timestamp - t_target),
                )
                observations.append(obs)
                continue

            right_idx = min(left_idx + 1, len(sorted_records) - 1)
            rec_right = sorted_records[right_idx]

            if abs(rec_right.timestamp - t_target) <= cfg.exact_match_tolerance_seconds:
                status = SyncStatus.OFFSET_APPLIED if offset_sec != 0.0 else SyncStatus.EXACT
                obs = SynchronizedFrameObservation(
                    video_frame_id=frame.frame_id,
                    video_timestamp_seconds=v_ts,
                    telemetry_timestamp_seconds=rec_right.timestamp,
                    status=status,
                    position=rec_right.position,
                    orientation=rec_right.orientation,
                    velocity=rec_right.velocity,
                    quality=rec_right.quality,
                    interpolation_method="exact_match",
                    interpolation_fraction=1.0,
                    bracketing_interval_seconds=0.0,
                    clock_offset_seconds=offset_sec,
                    clock_offset_status=offset_status,
                    source_record_indices=[rec_right.provenance.record_index] if rec_right.provenance and rec_right.provenance.record_index is not None else [right_idx],
                    source_provenance=rec_right.provenance,
                    timebase_uncertainty_seconds=abs(rec_right.timestamp - t_target),
                )
                observations.append(obs)
                continue

            # 3. Telemetry Gap Check
            dt_gap = rec_right.timestamp - rec_left.timestamp
            if dt_gap > cfg.max_interpolation_gap_seconds:
                obs = SynchronizedFrameObservation(
                    video_frame_id=frame.frame_id,
                    video_timestamp_seconds=v_ts,
                    telemetry_timestamp_seconds=None,
                    status=SyncStatus.UNSYNCHRONIZED,
                    position=None,
                    interpolation_method="gap_exceeded",
                    bracketing_interval_seconds=dt_gap,
                    clock_offset_seconds=offset_sec,
                    clock_offset_status=offset_status,
                    extra_metadata={"gap_duration_seconds": dt_gap},
                )
                observations.append(obs)
                continue

            # 4. Altitude Reference Compatibility Check
            if not cls.are_altitude_references_compatible(
                rec_left.position.altitude_reference, rec_right.position.altitude_reference
            ):
                obs = SynchronizedFrameObservation(
                    video_frame_id=frame.frame_id,
                    video_timestamp_seconds=v_ts,
                    telemetry_timestamp_seconds=None,
                    status=SyncStatus.INCOMPATIBLE_REFERENCE,
                    position=None,
                    interpolation_method="incompatible_vertical_reference",
                    bracketing_interval_seconds=dt_gap,
                    clock_offset_seconds=offset_sec,
                    clock_offset_status=offset_status,
                    extra_metadata={
                        "ref_left": rec_left.position.altitude_reference.value,
                        "ref_right": rec_right.position.altitude_reference.value,
                    },
                )
                observations.append(obs)
                continue

            # 5. Interpolate between rec_left and rec_right
            alpha = (t_target - rec_left.timestamp) / dt_gap if dt_gap > 0 else 0.0
            alpha = max(0.0, min(1.0, alpha))

            # Interpolate Position in metric Cartesian ECEF space (or geodetic shortest-arc)
            if cfg.use_ecef_interpolation and rec_left.position.position_reference == PositionReference.WGS84_GEODETIC:
                interp_pos = cls._interpolate_position_ecef(rec_left.position, rec_right.position, alpha)
            else:
                interp_pos = cls._interpolate_position_geodetic(rec_left.position, rec_right.position, alpha)

            # Interpolate Orientation (Optional, SLERP)
            interp_ori: Optional[TelemetryOrientation] = None
            if cfg.interpolate_orientation and rec_left.orientation and rec_right.orientation:
                interp_ori = cls._interpolate_orientation(rec_left.orientation, rec_right.orientation, alpha)
            elif rec_left.orientation or rec_right.orientation:
                interp_ori = rec_left.orientation if alpha < 0.5 else rec_right.orientation

            # Interpolate Velocity (Optional, linear)
            interp_vel: Optional[TelemetryVelocity] = None
            if rec_left.velocity and rec_right.velocity:
                interp_vel = cls._interpolate_velocity(rec_left.velocity, rec_right.velocity, alpha)
            elif rec_left.velocity or rec_right.velocity:
                interp_vel = rec_left.velocity if alpha < 0.5 else rec_right.velocity

            # Quality: Nearest Neighbor Hold
            interp_qual = rec_left.quality if alpha < 0.5 else rec_right.quality

            src_indices = []
            if rec_left.provenance and rec_left.provenance.record_index is not None:
                src_indices.append(rec_left.provenance.record_index)
            else:
                src_indices.append(left_idx)
            if rec_right.provenance and rec_right.provenance.record_index is not None:
                src_indices.append(rec_right.provenance.record_index)
            else:
                src_indices.append(right_idx)

            status = SyncStatus.OFFSET_APPLIED if offset_sec != 0.0 else SyncStatus.INTERPOLATED

            obs = SynchronizedFrameObservation(
                video_frame_id=frame.frame_id,
                video_timestamp_seconds=v_ts,
                telemetry_timestamp_seconds=t_target,
                status=status,
                position=interp_pos,
                orientation=interp_ori,
                velocity=interp_vel,
                quality=interp_qual,
                interpolation_method="ecef_pos_slerp_ori" if cfg.use_ecef_interpolation else "geodetic_pos_slerp_ori",
                interpolation_fraction=alpha,
                bracketing_interval_seconds=dt_gap,
                clock_offset_seconds=offset_sec,
                clock_offset_status=offset_status,
                source_record_indices=src_indices,
                source_provenance=rec_left.provenance,
                timebase_uncertainty_seconds=None, # Explicitly distinct from gap size
            )
            observations.append(obs)

        return SynchronizedTrajectory(
            trajectory_id=timeline.video_id,
            observations=observations,
            config=cfg,
            provenance_summary={
                "video_source": timeline.source_path,
                "telemetry_source": telemetry_stream.provenance.source_identifier if telemetry_stream.provenance else "",
                "total_frames": len(timeline),
                "total_telemetry_samples": len(telemetry_stream),
                "offset_seconds": offset_sec,
                "offset_status": offset_status.value,
            },
        )
