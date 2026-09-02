# Coverage-Aware Keyframe Selection Architecture & Scientific Policies

## 1. Executive Summary & Objective

The Coverage-Aware Keyframe Selection Subsystem produces a compact, non-redundant subset of candidate frame references from the canonical video stream while maximizing temporal, spatial, and trajectory coverage for downstream 3D reconstruction.

---

## 2. Scientific Interpretation Boundaries & Principles

> **CRITICAL SCIENTIFIC PRINCIPLES:**
> 1. **FRAME QUALITY $\neq$ FRAME USEFULNESS**:
>    - A high-quality static frame may be geometrically redundant, while a moderately sharp frame with unique viewpoint parallax may be critically informative.
> 2. **VISUAL NOVELTY $\neq$ GEOMETRIC NOVELTY**:
>    - High visual appearance similarity across planar surfaces does not imply identical 3D camera geometry.
> 3. **GNSS TRAJECTORY DIVERSITY $\neq$ CAMERA VIEWPOINT DIVERSITY**:
>    - Without lever-arm and camera calibration, ENU coordinates and attitude quaternions are platform navigation proxies (`TRAJECTORY_PROXY` / `ORIENTATION_PROXY`).
> 4. **DYNAMIC AREA FRACTION $\neq$ GEOMETRIC CONTAMINATION**:
>    - A large dynamic candidate region with near-zero feature matches in the dynamic area does not corrupt static background correspondences.
> 5. **GREEDY HEURISTIC $\neq$ GLOBALLY OPTIMAL SELECTION**:
>    - The sequential greedy marginal-gain selector is an explainable engineering heuristic (`GREEDY_HEURISTIC_GAIN`), not a proven global optimum.
> 6. **SELECTION DOES NOT PHYSICALLY DELETE FRAMES**:
>    - Keyframe selection outputs references (`KeyframeSelectionResult`) without altering or deleting source video data.

---

## 3. Pipeline Integration

```
       FrameQualityReport + TemporalMotionBlurReport + PhotometricStabilityReport
                                    +
                 DynamicSceneReport + FrameRedundancyReport
                                    │
                                    ▼
                     CoverageAwareKeyframeSelector
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        STAGE A: Hard Safety Gate           STAGE B: Greedy Marginal Gain
        - Corrupt frame rejection           - Boundary anchor safety & fallback
        - SEVERELY_DEGRADED filter          - Min spacing (Δt ≥ 0.25s)
        - Dynamic area & match check        - Max gap coverage (Δt ≤ 2.5s)
                  │                         - Trajectory baseline proxy (B ≥ 0.5m)
                  │                         - Orientation change proxy (Δθ ≥ 2.0°)
                  │                         - Feature novelty (1 - match_ratio)
                  │                         - Speed-aware displacement proxy
                  └─────────────────┬─────────────────┘
                                    ▼
                         KeyframeSelectionResult
                      (Selected References & Provenance)
                                    │
                                    ▼
                     Stage 5: Relative Pose Estimation
```

---

## 4. Two-Stage Selection Architecture

### Stage A: Hard Safety Gate
Evaluates every frame against strict safety criteria:
1. **Decode Status**: Rejects frames where `decode_status != DecodeStatus.SUCCESS` or data buffer is missing (`UNSAFE_DECODE_FAILURE`).
2. **Severe Image Degradation**: Deprioritizes frames marked `QualityStatus.SEVERELY_DEGRADED` (`UNSAFE_SEVERELY_DEGRADED`).
3. **Dynamic Contamination**: Rejects frames where dynamic area fraction exceeds $60\%$ unless feature-match diagnostics demonstrate that static background correspondences remain dominant ($< 15\%$ dynamic match contamination), in which case the candidate is retained with a diagnostic warning.

### Stage B: Greedy Marginal-Gain Coverage Selection
Iterates through surviving candidate frames:
1. **Boundary Anchor Safety Policy**:
   - Evaluates boundary frames through the normal safety gate.
   - If initial frame is safe: `INITIAL_ANCHOR`.
   - If initial frame is unsafe: searches forward for nearest safe frame $\to$ `INITIAL_BOUNDARY_FALLBACK`.
   - If final frame is safe: `FINAL_ANCHOR`.
   - If final frame is unsafe: searches backward for nearest safe frame $\to$ `FINAL_BOUNDARY_FALLBACK`.
   - If no safe candidate exists: `BOUNDARY_UNCOVERED`.
2. **Minimum Temporal Spacing**: Enforces $\Delta t \ge \tau_{\min\_spacing}$ ($0.25\text{ s}$).
3. **Marginal Information Gain Evaluation**:
   - **Temporal Gap Gain**: $\Delta \mathcal{G}_{\text{time}} = \min\left(1.0, \frac{\Delta t}{\tau_{\max\_gap}}\right)$
   - **Trajectory Diversity Gain (Proxy)**: $\Delta \mathcal{G}_{\text{traj}} = \min\left(1.0, \frac{B_{\text{trajectory}}}{2 \cdot \tau_{\text{baseline}}}\right)$
   - **Orientation Diversity Gain (Proxy)**: $\Delta \mathcal{G}_{\text{orient}} = \min\left(1.0, \frac{\Delta \theta_{\text{att}}}{2 \cdot \tau_{\text{orient}}}\right)$
   - **Feature Novelty Gain**: $\Delta \mathcal{G}_{\text{feat}} = \max\left(0.0, 1.0 - \min(\mathcal{R}_{\text{match}})\right)$
   - **Speed-Aware Trajectory Displacement**: $\Delta d_{\text{exp}} = v_{\text{ground}} \times \Delta t \quad [\text{m}]$ (`TRAJECTORY_PROXY`)
   - **Dynamic Risk Penalty**: Deducts $0.20$ if candidate exhibits active dynamic scene evidence.

---

## 5. Machine-Readable Explainability

Every selected keyframe records an explicit `primary_reason`, `greedy_heuristic_gain`, and full breakdown:
- **`INITIAL_ANCHOR`**: Sequence temporal origin anchor.
- **`INITIAL_BOUNDARY_FALLBACK`**: Nearest safe candidate replacing an unsafe initial frame.
- **`FINAL_ANCHOR`**: Sequence temporal boundary anchor.
- **`FINAL_BOUNDARY_FALLBACK`**: Nearest safe candidate replacing an unsafe final frame.
- **`BOUNDARY_UNCOVERED`**: Sequence boundary where no safe frame could be anchored.
- **`TEMPORAL_GAP_COVERAGE`**: Forced selection due to elapsed time approaching $\tau_{\max\_gap}$.
- **`TRAJECTORY_DIVERSITY`**: Significant spatial translation ($B_{\text{trajectory}} \ge \tau_{\text{baseline}}$).
- **`ORIENTATION_DIVERSITY`**: Significant platform attitude change ($\Delta \theta_{\text{att}} \ge \tau_{\text{orient}}$).
- **`FEATURE_NOVELTY`**: High feature novelty ($1 - \text{match\_ratio} \ge 0.40$).
- **`QUALITY_TIE_BREAK`**: Spaced candidate with satisfactory quality metrics.
- **`FALLBACK`**: Deterministic fallback anchor when all candidates fail gates.

---

## 6. Budget & Coverage Incompatibility Detection

If a user configures `max_keyframe_count` that is mathematically insufficient to cover the sequence duration $T = t_{\max} - t_{\min}$ given $\tau_{\max\_gap}$:
$$\text{min\_required} = \left\lceil \frac{T}{\tau_{\max\_gap}} \right\rceil + 1$$
If $\text{max\_keyframe\_count} < \text{min\_required}$, the selector records an explicit diagnostic:
`BUDGET_INCOMPATIBLE_WITH_COVERAGE_CONSTRAINT`
and enforces the budget constraint deterministically without fabricating full coverage.

---

## 7. Configurable Heuristic Thresholds (`HEURISTIC_DEFAULT`)

| Parameter | Default Value | Status | Description |
| :--- | :--- | :--- | :--- |
| `min_temporal_spacing_seconds` | `0.25 s` | `HEURISTIC_DEFAULT` | Minimum allowed elapsed time between consecutive keyframes. |
| `max_temporal_gap_seconds` | `2.5 s` | `HEURISTIC_DEFAULT` | Maximum allowable time interval before forced keyframe selection. |
| `min_trajectory_baseline_meters`| `0.5 m` | `HEURISTIC_DEFAULT` | Minimum trajectory distance to indicate spatial novelty. |
| `min_orientation_change_degrees`| `2.0 deg` | `HEURISTIC_DEFAULT` | Minimum attitude rotation to indicate viewpoint diversity. |
| `max_dynamic_contamination_fraction` | `0.60` | `HEURISTIC_DEFAULT` | Maximum acceptable dynamic area fraction. |
| `force_first_and_last_frames` | `True` | `HEURISTIC_DEFAULT` | Forces sequence boundary keyframe anchoring when safe. |

---

## 8. Computational Complexity & Provenance

- **Candidate Generation & Selection Complexity**: Linear in input sequence length, bounded at $O(W \cdot N)$ where $W$ is the temporal redundancy window ($W=5$).
- **Provenance**: Records input frame count, selected keyframe count, deprioritized candidate count, heuristic default status, and selection algorithm version (`GreedyMarginalGain_v1.1`).
