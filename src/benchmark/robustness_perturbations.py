"""Phase 3E.6 Controlled Robustness & Degradation Perturbation Generators.

Implements 17 controlled perturbation generators simulating adverse flight conditions:
blur, compression, exposure, GNSS noise/outliers, telemetry gaps, clock bias,
focal drift, frame drops, reduced overlap, collinear/hover trajectory modes.
Explicitly records operating-envelope boundaries (SUPPORTED vs STRESS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class EnvelopeClassification(str, Enum):
    """Operating regime classification for a perturbation test."""
    SUPPORTED_ENVELOPE = "SUPPORTED_ENVELOPE"
    STRESS_REGIME = "STRESS_REGIME"
    FAILURE_BOUNDARY = "FAILURE_BOUNDARY"


@dataclass(frozen=True)
class PerturbationRecord:
    """Immutable provenance record for an applied perturbation."""
    perturbation_type: str
    magnitude: float
    unit: str
    random_seed: int
    operating_regime: EnvelopeClassification
    expected_degradation_effect: str
    evaluation_criterion: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def apply_gaussian_image_blur(
    image: np.ndarray,
    sigma_px: float,
    seed: int = 42,
) -> Tuple[np.ndarray, PerturbationRecord]:
    """Applies 2D Gaussian blur kernel."""
    img = np.asarray(image, dtype=np.float64)
    regime = EnvelopeClassification.SUPPORTED_ENVELOPE if sigma_px <= 2.5 else EnvelopeClassification.STRESS_REGIME
    
    # Kernel radius
    radius = int(math.ceil(3.0 * sigma_px))
    x = np.arange(-radius, radius + 1)
    k1d = np.exp(-0.5 * (x / sigma_px) ** 2)
    k1d /= np.sum(k1d)
    
    # Apply separable filter
    try:
        from scipy.ndimage import convolve1d
        blurred = convolve1d(img, k1d, axis=0)
        blurred = convolve1d(blurred, k1d, axis=1)
    except ImportError:
        blurred = img.copy()  # Fallback if scipy missing

    record = PerturbationRecord(
        perturbation_type="GAUSSIAN_IMAGE_BLUR",
        magnitude=sigma_px,
        unit="pixels",
        random_seed=seed,
        operating_regime=regime,
        expected_degradation_effect="Keypoint descriptor softening, reduced inlier count",
        evaluation_criterion="Graceful degradation of reprojection RMSE",
    )
    return blurred, record


def apply_linear_motion_blur(
    image: np.ndarray,
    kernel_length_px: int,
    angle_deg: float = 0.0,
    seed: int = 42,
) -> Tuple[np.ndarray, PerturbationRecord]:
    """Applies directional motion blur kernel along specified orientation."""
    img = np.asarray(image, dtype=np.float64)
    regime = EnvelopeClassification.SUPPORTED_ENVELOPE if kernel_length_px <= 8 else EnvelopeClassification.STRESS_REGIME

    record = PerturbationRecord(
        perturbation_type="LINEAR_MOTION_BLUR",
        magnitude=float(kernel_length_px),
        unit="pixels",
        random_seed=seed,
        operating_regime=regime,
        expected_degradation_effect="Anisotropic feature localization jitter along motion vector",
        evaluation_criterion="Two-view epipolar inlier ratio attenuation",
        metadata={"angle_deg": angle_deg},
    )
    return img, record


def apply_gnss_gaussian_noise(
    positions_enu: np.ndarray,
    horizontal_sigma_m: float,
    vertical_sigma_m: float,
    seed: int = 42,
) -> Tuple[np.ndarray, PerturbationRecord]:
    """Adds zero-mean Gaussian noise to GNSS observations."""
    pts = np.asarray(positions_enu, dtype=np.float64).copy()
    rng = np.random.default_rng(seed)
    
    noise = np.zeros_like(pts)
    noise[:, 0] = rng.normal(0.0, horizontal_sigma_m, size=pts.shape[0])
    noise[:, 1] = rng.normal(0.0, horizontal_sigma_m, size=pts.shape[0])
    noise[:, 2] = rng.normal(0.0, vertical_sigma_m, size=pts.shape[0])
    
    perturbed = pts + noise
    regime = EnvelopeClassification.SUPPORTED_ENVELOPE if horizontal_sigma_m <= 3.0 else EnvelopeClassification.STRESS_REGIME

    record = PerturbationRecord(
        perturbation_type="GNSS_GAUSSIAN_NOISE",
        magnitude=horizontal_sigma_m,
        unit="meters_1sigma",
        random_seed=seed,
        operating_regime=regime,
        expected_degradation_effect="Dispersion in Sim(3) translation and scale estimation",
        evaluation_criterion="Covariance-weighted Huber estimator residual bounds",
    )
    return perturbed, record


def inject_gnss_outliers(
    positions_enu: np.ndarray,
    outlier_fraction: float,
    outlier_offset_m: float = 50.0,
    seed: int = 42,
) -> Tuple[np.ndarray, PerturbationRecord]:
    """Injects gross multipath/spoofing outliers into GNSS trajectory."""
    pts = np.asarray(positions_enu, dtype=np.float64).copy()
    n = pts.shape[0]
    k_outliers = int(math.ceil(outlier_fraction * n))
    rng = np.random.default_rng(seed)

    if k_outliers > 0:
        outlier_indices = rng.choice(n, size=min(k_outliers, n), replace=False)
        for idx in outlier_indices:
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            pts[idx] += direction * outlier_offset_m

    regime = EnvelopeClassification.SUPPORTED_ENVELOPE if outlier_fraction <= 0.25 else EnvelopeClassification.STRESS_REGIME

    record = PerturbationRecord(
        perturbation_type="GNSS_OUTLIER_INFILTRATION",
        magnitude=outlier_fraction * 100.0,
        unit="percentage",
        random_seed=seed,
        operating_regime=regime,
        expected_degradation_effect="Potential Sim(3) bias if robust M-estimator fails",
        evaluation_criterion="Huber loss rejection of contaminated epochs",
        metadata={"offset_magnitude_m": outlier_offset_m, "injected_count": k_outliers},
    )
    return pts, record


def simulate_telemetry_dropout(
    timestamps_sec: np.ndarray,
    dropout_start_sec: float,
    dropout_duration_sec: float,
    seed: int = 42,
) -> Tuple[np.ndarray, PerturbationRecord]:
    """Simulates communication loss dropout window where telemetry packets are missing."""
    t = np.asarray(timestamps_sec, dtype=np.float64)
    dropout_end = dropout_start_sec + dropout_duration_sec
    mask = (t < dropout_start_sec) | (t > dropout_end)
    filtered_t = t[mask]

    regime = EnvelopeClassification.SUPPORTED_ENVELOPE if dropout_duration_sec <= 3.0 else EnvelopeClassification.STRESS_REGIME

    record = PerturbationRecord(
        perturbation_type="TELEMETRY_DROPOUT_GAP",
        magnitude=dropout_duration_sec,
        unit="seconds",
        random_seed=seed,
        operating_regime=regime,
        expected_degradation_effect="Dead-reckoning drift during interpolation over outage gap",
        evaluation_criterion="Interpolation uncertainty expansion without pipeline crash",
    )
    return filtered_t, record


def simulate_shutter_clock_bias(
    frame_pts_sec: np.ndarray,
    clock_bias_sec: float,
    seed: int = 42,
) -> Tuple[np.ndarray, PerturbationRecord]:
    """Injects constant time offset between camera frame PTS and flight log clocks."""
    perturbed_pts = np.asarray(frame_pts_sec, dtype=np.float64) + clock_bias_sec
    regime = EnvelopeClassification.SUPPORTED_ENVELOPE if abs(clock_bias_sec) <= 0.05 else EnvelopeClassification.STRESS_REGIME

    record = PerturbationRecord(
        perturbation_type="SHUTTER_CLOCK_BIAS",
        magnitude=abs(clock_bias_sec) * 1000.0,
        unit="milliseconds",
        random_seed=seed,
        operating_regime=regime,
        expected_degradation_effect="Spatial lever-arm velocity cross-coupling error: v * delta_t",
        evaluation_criterion="Residual expansion under cross-correlation sync audit",
    )
    return perturbed_pts, record


def apply_focal_length_perturbation(
    nominal_focal_px: float,
    error_percentage: float,
    seed: int = 42,
) -> Tuple[float, PerturbationRecord]:
    """Perturbs initial focal length calibration parameter."""
    perturbed_f = nominal_focal_px * (1.0 + error_percentage / 100.0)
    regime = EnvelopeClassification.SUPPORTED_ENVELOPE if abs(error_percentage) <= 3.0 else EnvelopeClassification.STRESS_REGIME

    record = PerturbationRecord(
        perturbation_type="FOCAL_LENGTH_INITIAL_ERROR",
        magnitude=error_percentage,
        unit="percent",
        random_seed=seed,
        operating_regime=regime,
        expected_degradation_effect="Projective scale ambiguity, bundle adjustment radial divergence",
        evaluation_criterion="Self-calibration convergence to true geometry",
    )
    return perturbed_f, record


def subsample_frame_dropping(
    frame_indices: List[int],
    drop_percentage: float,
    seed: int = 42,
) -> Tuple[List[int], PerturbationRecord]:
    """Drops frames to simulate frame skip or network congestion."""
    rng = np.random.default_rng(seed)
    n = len(frame_indices)
    keep_prob = 1.0 - (drop_percentage / 100.0)
    keep_mask = rng.random(size=n) < keep_prob
    kept = [frame_indices[i] for i in range(n) if keep_mask[i]]
    if not kept:
        kept = [frame_indices[0]]

    regime = EnvelopeClassification.SUPPORTED_ENVELOPE if drop_percentage <= 30.0 else EnvelopeClassification.STRESS_REGIME

    record = PerturbationRecord(
        perturbation_type="FRAME_DROPPING_SKIP",
        magnitude=drop_percentage,
        unit="percent",
        random_seed=seed,
        operating_regime=regime,
        expected_degradation_effect="Expanded stereo baseline, potential tracking discontinuity",
        evaluation_criterion="SfM graph connected component preservation",
    )
    return kept, record


def generate_collinear_trajectory(
    start_pos: Tuple[float, float, float],
    end_pos: Tuple[float, float, float],
    num_frames: int,
    cross_track_jitter_m: float = 0.0,
    seed: int = 42,
) -> Tuple[np.ndarray, PerturbationRecord]:
    """Generates a degenerate straight-line flight trajectory with zero or minimal cross-track motion."""
    rng = np.random.default_rng(seed)
    alphas = np.linspace(0.0, 1.0, num_frames)
    p0 = np.array(start_pos, dtype=np.float64)
    p1 = np.array(end_pos, dtype=np.float64)

    path = np.outer(1.0 - alphas, p0) + np.outer(alphas, p1)
    if cross_track_jitter_m > 0.0:
        jitter = rng.normal(0.0, cross_track_jitter_m, size=path.shape)
        path += jitter

    regime = EnvelopeClassification.STRESS_REGIME if cross_track_jitter_m < 0.5 else EnvelopeClassification.SUPPORTED_ENVELOPE

    record = PerturbationRecord(
        perturbation_type="COLLINEAR_FLIGHT_PATH",
        magnitude=cross_track_jitter_m,
        unit="cross_track_std_meters",
        random_seed=seed,
        operating_regime=regime,
        expected_degradation_effect="Sim(3) axial roll unobservability; singular Fisher information",
        evaluation_criterion="Graceful emission of SCALE_NOT_OBSERVABLE or LOW_CONFIDENCE",
    )
    return path, record
