"""API & Pipeline orchestration interfaces for SIH26158."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class PipelineStage(str, Enum):
    IDLE = "IDLE"
    INGESTION = "INGESTION"
    FRAME_EXTRACTION = "FRAME_EXTRACTION"
    QUALITY_FILTERING = "QUALITY_FILTERING"
    KEYFRAME_SELECTION = "KEYFRAME_SELECTION"
    POSE_ESTIMATION = "POSE_ESTIMATION"
    GEOMETRY_RECONSTRUCTION = "GEOMETRY_RECONSTRUCTION"
    FUSION = "FUSION"
    SURFACE_MESHING = "SURFACE_MESHING"
    TEXTURE_PROJECTION = "TEXTURE_PROJECTION"
    GEOREFERENCING = "GEOREFERENCING"
    VALIDATION = "VALIDATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class PipelineRunStatus:
    """Real-time execution status of a reconstruction pipeline job."""
    job_id: str
    current_stage: PipelineStage
    progress_percentage: float = 0.0
    processed_keyframes_count: int = 0
    estimated_remaining_seconds: Optional[float] = None
    error_message: Optional[str] = None
    output_artifacts: Dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineRunConfig:
    """Top-level pipeline configuration submitted to the API."""
    video_path: str
    telemetry_path: Optional[str] = None
    output_directory: str = "data/processed/"
    target_engine: str = "fusion"
    extract_fps: float = 2.0
    enable_georeferencing: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["ply", "obj", "geotiff"])
