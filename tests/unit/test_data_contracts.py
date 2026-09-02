"""Test suite verifying integrity and immutability of core data contracts."""

import pytest
from src.ingestion import TelemetryRecord, VideoMetadata
from src.quality import FrameQualityScore
from src.geometry import CameraIntrinsics, ExtrinsicPose
from src.geospatial import Sim3Transform, CRSInfo
from src.uncertainty import UncertaintyField


def test_telemetry_record_immutability():
    """Verify TelemetryRecord is frozen/immutable."""
    record = TelemetryRecord(
        timestamp_seconds=12.5,
        latitude_deg=18.5204,
        longitude_deg=73.8567,
        altitude_meters=540.2,
        is_rtk_fixed=True
    )
    assert record.latitude_deg == 18.5204
    with pytest.raises(Exception):
        record.latitude_deg = 19.0  # type: ignore


def test_camera_intrinsics_matrix():
    """Verify intrinsic matrix generation."""
    intrinsics = CameraIntrinsics(
        fx=1000.0,
        fy=1000.0,
        cx=960.0,
        cy=540.0,
        width=1920,
        height=1080
    )
    mat = intrinsics.matrix_3x3
    assert len(mat) == 3
    assert mat[0][0] == 1000.0
    assert mat[0][2] == 960.0
    assert mat[1][1] == 1000.0
    assert mat[2][2] == 1.0


def test_frame_quality_score():
    """Verify frame quality evaluation contracts."""
    quality = FrameQualityScore(
        frame_index=42,
        laplacian_variance=125.4,
        exposure_balance_score=0.92,
        motion_blur_metric=0.15,
        composite_quality_score=85.0,
        passed_filter=True
    )
    assert quality.frame_index == 42
    assert quality.passed_filter is True


def test_sim3_transform_contract():
    """Verify 7-DoF Sim(3) transform contract."""
    sim3 = Sim3Transform(
        scale=2.35,
        source_crs="local_model",
        target_crs="EPSG:32643",
        num_anchors_used=24
    )
    assert sim3.scale == 2.35
    assert len(sim3.rotation_matrix) == 3
    assert len(sim3.translation_vector) == 3


def test_uncertainty_field_contract():
    """Verify uncertainty field flags and unobserved geometry marking."""
    uncertainty = UncertaintyField(
        num_observations=6,
        mean_reprojection_error_pixels=0.85,
        triangulation_angle_deg=14.2,
        spatial_covariance_trace=0.004,
        neural_confidence_score=0.94,
        is_observed_geometry=True
    )
    assert uncertainty.is_observed_geometry is True
    assert uncertainty.neural_confidence_score == 0.94
