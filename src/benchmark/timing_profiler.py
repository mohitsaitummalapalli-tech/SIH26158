"""Phase 3E.6 Computational Performance & Timing Profiler.

Measures wall-clock latency breakdown, end-to-end PIPELINE_FPS, Real-Time Factor (RTF),
and operational latency tiers under strictly defined environmental standards.
Rejects claiming real-time capability based on isolated sub-stages.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import importlib.util

psutil: Any = None
_HAS_PSUTIL = False
try:
    if importlib.util.find_spec("psutil") is not None:
        psutil = importlib.import_module("psutil")
        _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

from src.benchmark.models import (
    LatencyTier,
    TimingProfile,
)


def get_system_hardware_environment() -> Dict[str, Any]:
    """Inspects and returns hardware and runtime environment metadata."""
    total_ram_gb = 0.0
    cpu_count_logical = 1
    cpu_count_physical = 1

    if _HAS_PSUTIL and psutil is not None:
        try:
            mem = psutil.virtual_memory()
            total_ram_gb = round(mem.total / (1024 ** 3), 2)
            cpu_count_logical = psutil.cpu_count(logical=True) or 1
            cpu_count_physical = psutil.cpu_count(logical=False) or 1
        except Exception:
            pass
    else:
        cpu_count_logical = os.cpu_count() or 1
        cpu_count_physical = cpu_count_logical

    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "total_ram_gb": total_ram_gb,
        "cpu_count_logical": cpu_count_logical,
        "cpu_count_physical": cpu_count_physical,
        "gpu_available": False,
    }


class BenchmarkTimingProfiler:
    """End-to-end and per-stage timing profiler."""

    def __init__(self, input_frames: int = 0, decoded_duration_sec: float = 0.0) -> None:
        self.input_frames = input_frames
        self.decoded_duration_sec = decoded_duration_sec
        self.start_wall_clock: float = 0.0
        self.end_wall_clock: float = 0.0
        self.stage_timings: Dict[str, float] = {}
        self._active_stage: Optional[str] = None
        self._active_stage_start: float = 0.0

    def start_pipeline(self) -> None:
        """Starts end-to-end pipeline clock."""
        self.start_wall_clock = time.perf_counter()

    def stop_pipeline(self) -> None:
        """Stops end-to-end pipeline clock."""
        self.end_wall_clock = time.perf_counter()

    def start_stage(self, stage_name: str) -> None:
        """Begins timing a named pipeline sub-stage."""
        if self._active_stage is not None:
            self.stop_stage()
        self._active_stage = stage_name
        self._active_stage_start = time.perf_counter()

    def stop_stage(self) -> float:
        """Stops timing current sub-stage and records elapsed wall-clock seconds."""
        if self._active_stage is None:
            return 0.0
        elapsed = time.perf_counter() - self._active_stage_start
        self.stage_timings[self._active_stage] = elapsed
        self._active_stage = None
        return elapsed

    def build_timing_profile(self) -> TimingProfile:
        """Calculates throughput metrics and assigns strict latency tier."""
        if self.end_wall_clock <= self.start_wall_clock:
            self.end_wall_clock = time.perf_counter()

        total_wall = float(self.end_wall_clock - self.start_wall_clock)
        if total_wall <= 1e-9:
            total_wall = 1e-6

        pipeline_fps = float(self.input_frames / total_wall) if self.input_frames > 0 else 0.0
        
        if self.decoded_duration_sec > 1e-6:
            rtf = float(total_wall / self.decoded_duration_sec)
        else:
            rtf = 0.0

        # Latency tier classification strictly based on end-to-end pipeline FPS
        if self.input_frames == 0:
            tier = LatencyTier.NOT_CLASSIFIED
        elif pipeline_fps < 1.0:
            tier = LatencyTier.OFFLINE_BATCH
        elif pipeline_fps < 15.0:
            tier = LatencyTier.NEAR_REAL_TIME
        elif pipeline_fps >= 30.0:
            tier = LatencyTier.REAL_TIME
        else:
            tier = LatencyTier.NEAR_REAL_TIME

        return TimingProfile(
            total_wall_clock_sec=total_wall,
            stage_wall_clock_sec=self.stage_timings.copy(),
            input_frames=self.input_frames,
            decoded_duration_sec=self.decoded_duration_sec,
            pipeline_fps=pipeline_fps,
            real_time_factor=rtf,
            latency_tier=tier,
            hardware_environment=get_system_hardware_environment(),
        )
