"""Exceptions for video ingestion and metadata validation."""


class IngestionError(Exception):
    """Base exception for all ingestion-related errors."""
    pass


class VideoNotFoundError(IngestionError, FileNotFoundError):
    """Raised when the specified video file path does not exist on disk."""
    pass


class UnsupportedVideoFormatError(IngestionError, ValueError):
    """Raised when a video container format or file extension is not supported."""
    pass


class UnsupportedFragmentedMP4Error(UnsupportedVideoFormatError):
    """Raised when a fragmented MP4 (moof/mvex) is encountered without supported demuxing."""
    pass


class CorruptVideoError(IngestionError, ValueError):
    """Raised when a video file is truncated, header is unreadable, or container structure is corrupted."""
    pass


class InvalidVideoMetadataError(IngestionError, ValueError):
    """Raised when extracted video metadata violates physical or logical constraints (e.g. zero duration, negative dimensions)."""
    pass


class InvalidTelemetryDataError(IngestionError, ValueError):
    """Raised when telemetry coordinates, timestamps, or quality records violate physical bounds or schema constraints."""
    pass
