# Architecture Forensic Audit

This document records the architectural inspection of the 15-stage pipeline, verifying stage-by-stage data contracts, coordinate system conventions, temporal synchronization, and failure handling.

---

## 1. Architectural Audit Checklist

| Architectural Requirement | Verification Finding | Status |
| :--- | :--- | :--- |
| **Temporal Synchronization** | Stage 1 explicitly defines UTC clock sync and cubic spline interpolation from 1Hz GNSS to 30fps frames with $<5\text{ ms}$ jitter target. | **AUDITED & SATISFIED** |
| **Coordinate Frame Conventions** | System explicitly defines OpenCV optical frame ($X$-right, $Y$-down, $Z$-forward), Drone FRD/NED, Local Topocentric ENU, ECEF, and WGS84 UTM. | **AUDITED & SATISFIED** |
| **Rolling Shutter Handling** | Stage 2 explicitly includes rolling shutter motion compensation and Brown-Conrady lens undistortion. | **AUDITED & SATISFIED** |
| **Dynamic Object Masking** | Stage 3 explicitly adds transient motion segmentation to prevent dynamic objects (cars, pedestrians) from contaminating static geometry. | **AUDITED & SATISFIED** |
| **Outlier & Floater Pruning** | Stage 9 explicitly introduces Statistical Outlier Removal (SOR) and normal consistency filtering. | **AUDITED & SATISFIED** |
| **Scale & Metric Recovery** | Stage 8 & 12 enforce 7-DoF $\text{Sim}(3)$ Umeyama alignment constrained by GNSS telemetry. | **AUDITED & SATISFIED** |
| **Uncertainty Propagation** | Stage 13 generates per-vertex spatial covariance and explicitly tags unobserved facades as missing. | **AUDITED & SATISFIED** |
| **Scientific Provenance Guardrail** | Stage 14 enforces strict metric validation requiring dataset name, ground truth, and SHA-256 checksums. | **AUDITED & SATISFIED** |

---

## 2. Interface Contract Consistency

Every stage now has explicitly documented:
- `INPUT`
- `OUTPUT`
- `FAILURE MODES`
- `METRICS`
- `DEPENDENCIES`

No missing or orphaned stages were detected.
