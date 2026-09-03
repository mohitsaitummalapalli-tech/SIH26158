"""Phase 3E.6 Texture & Photometric Reprojection Diagnostics.

Implements masked PSNR, masked SSIM, and seam edge gradient discontinuity.
Strictly designates all photometric scores as DIAGNOSTIC ONLY (no universal pass/fail gates).
Rejects colorimetric ground-truth claims without explicit radiometric calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.benchmark.models import (
    StatisticalSummary,
    ContractViolationError,
)

DISCLAIMER_TEXTURE_DIAGNOSTIC = (
    "Reprojection PSNR/SSIM reflects photometric agreement with source keyframe video; "
    "it DOES NOT represent radiometric ground-truth accuracy or geometric surface correctness."
)


@dataclass(frozen=True)
class TextureDiagnosticMetadata:
    """Mandatory rendering and image context for texture diagnostics."""
    image_resolution: Tuple[int, int]
    color_space: str = "sRGB"
    mask_definition: str = "FOREGROUND_NON_SKY"
    renderer: str = "SOFTWARE_RASTERIZER"
    interpolation: str = "BILINEAR"
    compression_state: str = "H264_LOSSY"
    exposure_assumptions: str = "TIME_VARYING_AUTO_EXPOSURE"
    radiometric_calibration_certified: bool = False


@dataclass(frozen=True)
class TextureDiagnosticResult:
    """Results container for photometric diagnostics."""
    masked_psnr_db: float
    masked_ssim: float
    seam_gradient_discontinuity_index: float
    valid_pixel_count: int
    total_pixel_count: int
    foreground_ratio: float
    metadata: TextureDiagnosticMetadata
    is_diagnostic_only: bool = True
    disclaimer: str = DISCLAIMER_TEXTURE_DIAGNOSTIC


def compute_masked_psnr(
    rendered_image: np.ndarray,
    reference_image: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB over valid masked foreground pixels."""
    ren = np.asarray(rendered_image, dtype=np.float64)
    ref = np.asarray(reference_image, dtype=np.float64)

    if ren.shape != ref.shape:
        raise ValueError(f"Shape mismatch between rendered {ren.shape} and reference {ref.shape}")

    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape[:2] != ren.shape[:2]:
            raise ValueError("Mask dimensions do not match image dimensions.")
        ren_valid = ren[mask]
        ref_valid = ref[mask]
    else:
        ren_valid = ren.ravel()
        ref_valid = ref.ravel()

    if ren_valid.size == 0:
        return 0.0

    mse = float(np.mean((ren_valid - ref_valid) ** 2))
    if mse < 1e-12:
        return 100.0  # Numerical identity

    max_val = 255.0 if np.max(ref) > 1.0 else 1.0
    return float(10.0 * np.log10((max_val ** 2) / mse))


def compute_masked_ssim(
    rendered_image: np.ndarray,
    reference_image: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """Computes Mean Structural Similarity Index (SSIM) over masked foreground."""
    ren = np.asarray(rendered_image, dtype=np.float64)
    ref = np.asarray(reference_image, dtype=np.float64)

    if ren.shape != ref.shape:
        raise ValueError("Image shape mismatch in SSIM computation.")

    # Convert RGB to grayscale luminance if 3-channel
    if ren.ndim == 3 and ren.shape[2] == 3:
        ren_gray = 0.2989 * ren[:, :, 0] + 0.5870 * ren[:, :, 1] + 0.1140 * ren[:, :, 2]
        ref_gray = 0.2989 * ref[:, :, 0] + 0.5870 * ref[:, :, 1] + 0.1140 * ref[:, :, 2]
    else:
        ren_gray = ren
        ref_gray = ref

    max_val = 255.0 if np.max(ref_gray) > 1.0 else 1.0
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        x = ren_gray[mask]
        y = ref_gray[mask]
    else:
        x = ren_gray.ravel()
        y = ref_gray.ravel()

    if x.size == 0:
        return 0.0

    mu_x = float(np.mean(x))
    mu_y = float(np.mean(y))
    sigma_x2 = float(np.var(x))
    sigma_y2 = float(np.var(y))
    sigma_xy = float(np.mean((x - mu_x) * (y - mu_y)))

    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x2 + sigma_y2 + c2)

    ssim_val = num / den
    return float(np.clip(ssim_val, 0.0, 1.0))


def compute_seam_gradient_discontinuity(
    textured_mesh_rendering: np.ndarray,
    seam_boundary_mask: np.ndarray,
) -> float:
    """Computes photometric gradient discontinuity index across texture chart boundary seams."""
    img = np.asarray(textured_mesh_rendering, dtype=np.float64)
    mask = np.asarray(seam_boundary_mask, dtype=bool)

    if img.ndim == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img

    # Sobel / difference gradients
    grad_y, grad_x = np.gradient(gray)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    if np.any(mask):
        seam_grad = float(np.mean(grad_mag[mask]))
    else:
        seam_grad = 0.0

    return seam_grad


def evaluate_texture_diagnostics(
    rendered_image: np.ndarray,
    reference_image: np.ndarray,
    metadata: TextureDiagnosticMetadata,
    valid_mask: Optional[np.ndarray] = None,
    seam_mask: Optional[np.ndarray] = None,
    assert_universal_gate: bool = False,
    claim_colorimetric_accuracy: bool = False,
) -> TextureDiagnosticResult:
    """Evaluates texture and photometric agreement.
    
    Guarantees:
    1. Rejects universal hardcoded gates like PSNR > 35 without dataset pre-registration (MUT-17).
    2. Rejects colorimetric ground truth claims without certified radiometric calibration (MUT-16).
    """
    if assert_universal_gate:
        raise ContractViolationError(
            "Contract Violation (MUT-17): Universal PSNR/SSIM acceptance gates are strictly rejected. "
            "Photometric thresholds must be pre-registered and dataset-specific."
        )

    if claim_colorimetric_accuracy and not metadata.radiometric_calibration_certified:
        raise ContractViolationError(
            "Contract Violation (MUT-16): Radiometric/colorimetric accuracy claimed without certified calibration."
        )

    psnr_val = compute_masked_psnr(rendered_image, reference_image, valid_mask)
    ssim_val = compute_masked_ssim(rendered_image, reference_image, valid_mask)
    
    if seam_mask is not None:
        seam_disc = compute_seam_gradient_discontinuity(rendered_image, seam_mask)
    else:
        seam_disc = 0.0

    total_px = rendered_image.shape[0] * rendered_image.shape[1]
    if valid_mask is not None:
        valid_px = int(np.sum(valid_mask))
    else:
        valid_px = total_px
    fg_ratio = float(valid_px / total_px) if total_px > 0 else 0.0

    return TextureDiagnosticResult(
        masked_psnr_db=psnr_val,
        masked_ssim=ssim_val,
        seam_gradient_discontinuity_index=seam_disc,
        valid_pixel_count=valid_px,
        total_pixel_count=total_px,
        foreground_ratio=fg_ratio,
        metadata=metadata,
    )
