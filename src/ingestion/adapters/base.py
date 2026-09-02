"""Base adapter definitions and parsed record status classifications."""

import os
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

from src.ingestion.canonical_telemetry import (
    TelemetryRecord,
    CanonicalTelemetryStream,
    TelemetryProvenance,
)


class RecordStatus(str, Enum):
    """Classification status for raw telemetry records during ingestion."""
    VALID = "VALID"                       # Complete record meeting all structural & coordinate constraints
    PARTIALLY_VALID = "PARTIALLY_VALID"   # Recoverable record with some non-critical fields missing/unparseable
    INVALID = "INVALID"                   # Unrecoverable record (corrupt coordinates, unparseable payload)


@dataclass(frozen=True)
class ParsedTelemetryRecord:
    """Wrapped result of parsing an individual raw telemetry log record."""
    status: RecordStatus
    record: Optional[TelemetryRecord]
    raw_record_index: int
    raw_text: str
    rejection_reason: Optional[str] = None


class TelemetryAdapter(ABC):
    """Abstract base class for all telemetry ingestion adapters."""

    def __init__(self, source_path: str) -> None:
        self.source_path = os.path.abspath(source_path)
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Telemetry source file not found at: '{self.source_path}'")
        self._sha256: Optional[str] = None

    def compute_sha256(self) -> str:
        """Compute cryptographic hash of source telemetry file for provenance."""
        if self._sha256 is None:
            hasher = hashlib.sha256()
            with open(self.source_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            self._sha256 = hasher.hexdigest()
        return self._sha256

    @abstractmethod
    def parse_records(self) -> List[ParsedTelemetryRecord]:
        """Parse all records from the source with individual status classifications."""
        pass

    def parse_stream(self) -> CanonicalTelemetryStream:
        """Construct a CanonicalTelemetryStream containing all valid and partially valid records."""
        parsed_records = self.parse_records()
        valid_records: List[TelemetryRecord] = [
            pr.record for pr in parsed_records if pr.record is not None
        ]
        provenance = TelemetryProvenance(
            source_type=self.__class__.__name__,
            source_identifier=self.source_path,
            source_checksum=self.compute_sha256()
        )
        return CanonicalTelemetryStream(
            stream_id=os.path.basename(self.source_path),
            records=valid_records,
            provenance=provenance
        )
