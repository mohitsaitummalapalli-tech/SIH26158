# Dynamic Scene Analysis Contract & Temporal Evidence Architecture

## 1. Executive Summary & Objective

The Dynamic Scene Analysis subsystem establishes a detector-agnostic architectural contract and deterministic temporal evidence layer for identifying candidate dynamic regions across video sequences.

---

## 2. Scientific Interpretation Boundaries

> **CRITICAL SCIENTIFIC PRINCIPLES:**
> 1. **IMAGE MOTION $\neq$ OBJECT MOTION $\neq$ SEMANTIC OBJECT IDENTITY**:
>    - These three concepts are fundamentally distinct and must never be conflated.
> 2. **SEMANTIC CLASS DOES NOT IMPLY DYNAMIC BEHAVIOR**:
>    - A semantic label (e.g. `car`, `truck`, `boat`, `person`) indicates object type, **NOT** whether the object is moving.
>    - Stationary vehicles (parked cars) and static structures must be evaluated based on empirical temporal motion discrepancy.
> 3. **DYNAMIC REGIONS DO NOT REQUIRE SEMANTIC CLASSIFICATION**:
>    - Unlabeled or novel moving entities (machinery, wildlife, industrial equipment) receive dynamic evidence directly from motion vector field discrepancies.
> 4. **LOCAL MOTION DISCREPANCY IS EVIDENCE, NOT PROOF**:
>    - 3D parallax, occluding geometric edges, and depth discontinuities under camera translation can create local optical flow disparities.
>    - Local flow discrepancy provides **diagnostic hypothesis evidence**, not absolute ground-truth object motion.
> 5. **DIAGNOSTIC EVIDENCE ONLY**:
>    - This subsystem produces diagnostic evidence reports (`DynamicSceneReport`). It does **NOT** drop frames or erase/mask pixels permanently.

---

## 3. Pipeline Integration

```
                         DecodedFrame (t-1, t, t+1)
                                     │
                                     ▼
                        DynamicRegionProvider (ABC)
                   (e.g., Synthetic / External Adapter)
                                     │
                                     ▼
                            Candidate Regions
                                     │
                                     ▼
                           DynamicSceneAnalyzer
                                     │
                 ┌───────────────────┼───────────────────┐
                 ▼                   ▼                   ▼
         Dominant Flow       Local Discrepancy    Persistence Track
        (Global Vector)      (v_local - v_glob)     (Frame History)
                 │                   │                   │
                 └───────────────────┼───────────────────┘
                                     ▼
                             DynamicSceneReport
                        (Structured JSON Diagnostic)
```

---

## 4. Detector-Agnostic Provider Interface

To maintain strict modularity without coupling the core pipeline to heavy AI frameworks (e.g. YOLO, SAM, Mask R-CNN), the subsystem defines an abstract base interface:

```python
class DynamicRegionProvider(ABC):
    @abstractmethod
    def detect_candidate_regions(
        self, frame: DecodedFrame
    ) -> List[Tuple[Tuple[int, int, int, int], Optional[str], Optional[float], RegionMaskReference]]:
        """Extract candidate bounding boxes and optional semantic annotations."""
        pass
```

Future adapters (e.g. `YOLOv8RegionProvider`, `SAMSegmenterProvider`, `GroundingDINORegionProvider`) implement this interface without requiring core geometry pipeline modifications.

---

## 5. Mathematical Formulations & Evidence Model

### 1. Relative Motion Discrepancy
Given dominant global camera velocity $\mathbf{v}_{\text{global}}$ and mean local velocity $\mathbf{v}_{\text{local}}$ within candidate region $\mathcal{R}$:
$$\Delta \mathbf{v}_{\mathcal{R}} = \|\mathbf{v}_{\text{local}} - \mathbf{v}_{\text{global}}\| \quad [\text{px/s}]$$

### 2. Temporal Persistence Count ($K_{\text{persist}}$)
Tracks the spatial presence of candidate regions across consecutive frames:
$$K_{\text{persist}} = \sum_{t' = t - K + 1}^t \mathbf{1}_{\mathcal{R} \in \text{Frame}_{t'}}$$

### 3. Dynamic Evidence Index ($\mathcal{E}_{\text{dynamic}} \in [0.0, 1.0]$)
A heuristic diagnostic score combining relative motion discrepancy with temporal persistence:
$$\mathcal{E}_{\text{dynamic}} = 0.70 \cdot \min\left(1.0, \frac{\Delta \mathbf{v}_{\mathcal{R}}}{2 \cdot \tau_{\text{discrepancy}}}\right) + 0.30 \cdot \left(\min\left(1.0, \frac{\Delta \mathbf{v}_{\mathcal{R}}}{2 \cdot \tau_{\text{discrepancy}}}\right) \cdot \min\left(1.0, \frac{K_{\text{persist}}}{\tau_{\text{persist}}}\right)\right)$$

### 4. Dynamic Evidence Categories
- **`STATIC_EVIDENCE`**: Motion discrepancy $\Delta \mathbf{v}_{\mathcal{R}} < 1.0\text{ px/s}$ (region moves coherently with global camera motion).
- **`POSSIBLY_DYNAMIC`**: Transient local discrepancy or intermediate velocity difference.
- **`DYNAMIC_EVIDENCE`**: Significant motion discrepancy ($\Delta \mathbf{v}_{\mathcal{R}} \ge \tau_{\text{discrepancy}}$) persistent across consecutive frames ($K_{\text{persist}} \ge \tau_{\text{persist}}$).
- **`INSUFFICIENT_EVIDENCE`**: Isolated frame without adjacent temporal context or large temporal gap ($> 2.0\text{ s}$).

---

## 6. Configurable Heuristic Thresholds (`HEURISTIC_DEFAULT`)

| Parameter | Default Value | Status | Description |
| :--- | :--- | :--- | :--- |
| `motion_discrepancy_threshold_px_s` | `4.0 px/s` | `HEURISTIC_DEFAULT` | Velocity discrepancy threshold from dominant global flow. |
| `min_persistence_frames` | `2` | `HEURISTIC_DEFAULT` | Consecutive frame count to establish temporal persistence. |
| `dynamic_score_threshold` | `0.50` | `HEURISTIC_DEFAULT` | Diagnostic evidence index bound for `DYNAMIC_EVIDENCE`. |
| `max_temporal_gap_seconds` | `2.0 s` | `HEURISTIC_DEFAULT` | Maximum allowable inter-frame time delta. |

---

## 7. Provenance & Immutability

- **Immutability**: Source frame buffers are never altered.
- **Provenance**: Records target frame ID, timestamps, region provider class name, candidate region count, bounding boxes, and discrepancy thresholds applied.
