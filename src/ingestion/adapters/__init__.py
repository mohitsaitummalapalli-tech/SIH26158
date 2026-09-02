"""Telemetry adapter implementations and interfaces."""

from src.ingestion.adapters.base import (
    TelemetryAdapter,
    ParsedTelemetryRecord,
    RecordStatus,
)
from src.ingestion.adapters.dji_srt import DJISRTAdapter
from src.ingestion.adapters.generic_csv import GenericCSVAdapter, CSVColumnMapping
from src.ingestion.adapters.klv_interface import KLVAdapterInterface, KLVPacket

__all__ = [
    "TelemetryAdapter",
    "ParsedTelemetryRecord",
    "RecordStatus",
    "DJISRTAdapter",
    "GenericCSVAdapter",
    "CSVColumnMapping",
    "KLVAdapterInterface",
    "KLVPacket",
]
