"""Test suite verifying clean importability of all Phase 0 packages."""

import pytest

def test_root_package_import():
    import src
    assert src.__version__ == "0.1.0"

def test_all_submodules_importable():
    import src.ingestion
    import src.preprocessing
    import src.quality
    import src.geometry
    import src.reconstruction
    import src.fusion
    import src.geospatial
    import src.validation
    import src.uncertainty
    import src.api
    
    assert src.ingestion.TelemetryRecord is not None
    assert src.ingestion.CanonicalFlightDataset is not None
    assert src.preprocessing.FrameData is not None
    assert src.preprocessing.FrameDecoder is not None
    assert src.preprocessing.DecodedFrame is not None
    assert src.quality.FrameQualityScore is not None
    assert src.quality.FrameQualityAnalyzer is not None
    assert src.quality.FrameQualityReport is not None
    assert src.quality.TemporalMotionAnalyzer is not None
    assert src.quality.TemporalMotionBlurReport is not None
    assert src.quality.PhotometricAnalyzer is not None
    assert src.quality.PhotometricStabilityReport is not None
    assert src.quality.DynamicSceneAnalyzer is not None
    assert src.quality.DynamicSceneReport is not None
    assert src.quality.FrameRedundancyViewpointAnalyzer is not None
    assert src.quality.FrameRedundancyReport is not None
    assert src.quality.CoverageAwareKeyframeSelector is not None
    assert src.quality.KeyframeSelectionResult is not None
    assert src.geometry.CameraIntrinsics is not None
    assert src.geometry.ExtrinsicPose is not None
    assert src.geometry.TwoViewGeometryResult is not None
    assert src.geometry.SparseReconstructionResult is not None
    assert src.geometry.ClassicalFeatureExtractor is not None
    assert src.geometry.ClassicalDescriptorMatcher is not None
    assert src.geometry.TwoViewGeometryEstimator is not None
    assert src.geometry.IncrementalSfMEngine is not None
    assert src.geometry.BundleAdjustmentResult is not None
    assert src.reconstruction.PointCloudMetadata is not None
    assert src.fusion.FusionGraphConfig is not None
    assert src.geospatial.Sim3Transform is not None
    assert src.geospatial.CoordinateNormalizer is not None
    assert src.validation.AccuracyMetric is not None
    assert src.uncertainty.UncertaintyField is not None
    assert src.api.PipelineStage is not None
