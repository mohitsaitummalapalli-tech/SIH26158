"""Phase 3F: End-to-End Pipeline Error Taxonomy.

Strictly typed exception hierarchy for non-silent pipeline failures, contract
violations, insufficient inputs, and anti-leakage isolation breaches.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


from src.benchmark.models import ContractViolationError as BenchmarkContractViolationError


class PipelineError(Exception):
    """Base exception for all Phase 3F pipeline errors."""

    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.diagnostics = diagnostics or {}
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        stage_prefix = f"[{self.stage}] " if self.stage else ""
        return f"{stage_prefix}{self.message}"


class ContractViolationError(PipelineError, BenchmarkContractViolationError):
    """Raised when an immutable contract, artifact hash, or lifecycle invariant is breached."""
    pass


class StageExecutionError(PipelineError):
    """Raised when an algorithmic stage encounters numerical or runtime failure."""
    pass


class InsufficientInputError(PipelineError):
    """Raised when required input evidence (e.g. keyframes, tracks, inliers) is insufficient."""
    pass


class DataLeakageError(PipelineError):
    """Raised when hidden evaluation truth or privileged ground truth crosses into reconstruction."""
    pass
