"""Architectural interface specification for future KLV (Key-Length-Value) metadata adapters.

NOTICE:
No speculative KLV decoder is implemented in this phase.
This module defines the typed interface and data contracts required for future SMPTE 336M / MISB ST 0601 decoders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from src.ingestion.adapters.base import TelemetryAdapter, ParsedTelemetryRecord


@dataclass(frozen=True)
class KLVPacket:
    """Raw KLV data packet following SMPTE 336M specifications.
    
    Attributes:
    - key: 16-byte SMPTE universal label or local tag.
    - length: Byte length of the value payload (BER encoded).
    - value: Raw payload byte string.
    - timestamp_utc: Optional microsecond UTC timestamp (MISB Tag 2).
    - schema_identifier: Standard schema identifier (e.g. "MISB_ST_0601", "STANAG_4609").
    """
    key: bytes
    length: int
    value: bytes
    timestamp_utc: Optional[str] = None
    schema_identifier: str = "MISB_ST_0601"


class KLVAdapterInterface(TelemetryAdapter, ABC):
    """Abstract interface for standards-compliant KLV metadata demuxers and decoders."""

    @abstractmethod
    def parse_klv_packets(self) -> List[KLVPacket]:
        """Extract raw KLV packets from transport stream or container metadata track."""
        pass

    @abstractmethod
    def decode_packet_to_telemetry(self, packet: KLVPacket) -> Optional[ParsedTelemetryRecord]:
        """Decode a single KLV packet into a canonical ParsedTelemetryRecord using registered schema."""
        pass
