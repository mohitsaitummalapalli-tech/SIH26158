"""Integration test suite verifying cross-module interfaces."""

import pytest
from src.api import PipelineRunConfig, PipelineRunStatus, PipelineStage
from src.ingestion import IngestionConfig
from src.quality import QualityFilterConfig
from src.reconstruction import ReconstructionConfig
from src.fusion import FusionGraphConfig


def test_pipeline_configuration_compatibility():
    """Verify top-level pipeline configuration initializes sub-module configs cleanly."""
    run_config = PipelineRunConfig(
        video_path="data/raw/sample_flight.mp4",
        telemetry_path="data/raw/sample_flight.srt",
        output_directory="data/processed/run_001/",
        target_engine="fusion",
        extract_fps=2.0
    )
    assert run_config.target_engine == "fusion"
    assert run_config.extract_fps == 2.0
    
    ingestion_cfg = IngestionConfig(target_fps=run_config.extract_fps)
    assert ingestion_cfg.target_fps == 2.0

    recon_cfg = ReconstructionConfig(target_engine=run_config.target_engine)
    assert recon_cfg.target_engine == "fusion"


def test_pipeline_status_transitions():
    """Verify pipeline state flow contracts."""
    status = PipelineRunStatus(
        job_id="job_20260902_001",
        current_stage=PipelineStage.IDLE,
        progress_percentage=0.0
    )
    assert status.current_stage == PipelineStage.IDLE
    
    # Progress through stages
    status.current_stage = PipelineStage.INGESTION
    status.progress_percentage = 10.0
    assert status.current_stage == PipelineStage.INGESTION
    assert status.progress_percentage == 10.0

    status.current_stage = PipelineStage.COMPLETED
    status.progress_percentage = 100.0
    assert status.current_stage == PipelineStage.COMPLETED
