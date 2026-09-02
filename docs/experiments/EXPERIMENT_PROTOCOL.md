# Empirical Experiment Protocol: 4-Way Reconstruction Benchmark

## 1. Objective

To establish a strictly reproducible, scientifically rigorous benchmark evaluating four distinct 3D reconstruction pipelines on identical single-pass drone video trajectories:

- **Pipeline A (Classical Baseline):** Incremental SfM + Multi-View Stereo (COLMAP / OpenSfM).
- **Pipeline B (DUSt3R Baseline):** Direct pairwise ViT pointmap regression + global alignment.
- **Pipeline C (VGGT Baseline):** Visual Geometry Grounded Transformer sequence modeling.
- **Pipeline D (Hybrid Fusion):** AI geometry pointmaps regularized by classical epipolar constraints and GNSS/IMU $\text{Sim}(3)$ graph optimization.

---

## 2. Fair Comparison & Control Rules

To prevent biased or unfair benchmarks, all pipelines must be evaluated under identical experimental conditions:

1. **Identical Input Streams:** Exactly the same video frames, downsampling rates, and telemetry timestamps.
2. **Identical Evaluation Bounding Volume:** Metric error is evaluated exclusively within a fixed 3D bounding box defined by the ground truth survey area.
3. **Identical Coordinate Alignment Method:** All unscaled relative models must undergo identical Umeyama $\text{Sim}(3)$ alignment using the same synchronized GNSS camera trajectory priors before computing metric errors.
4. **Handling Missing or Fragmented Outputs:** If a pipeline fails to register a frame, crashes with OOM, or produces disconnected fragments, the run is recorded as a **Partial Failure** with penalty metrics (completeness = 0 for unregistered regions, rather than dropping the failed run from statistics).
5. **Separation of Visual vs. Metric Metrics:** Visual photorealism metrics (PSNR, SSIM, LPIPS) **must never substitute for metric geometric accuracy**.

---

## 3. Metric Taxonomy & Ground Truth Requirements

| Metric Name | Unit | Requires Ground Truth? | Description & Requirement |
| :--- | :--- | :--- | :--- |
| `reconstruction_success` | Boolean | NO | Whether pipeline completed without crashing or fragmenting. |
| `wall_clock_time_seconds` | Seconds ($s$) | NO | Total runtime from video frame input to final export. |
| `throughput_fps` | Frames/sec | NO | Number of processed video frames per second of compute. |
| `peak_vram_mb` | Megabytes ($MB$) | NO | Maximum CUDA GPU memory allocated during the job. |
| `reprojection_error_pixels` | Pixels ($px$) | NO | Self-consistency optical residual on keypoint tracks. |
| `ate_rmse_meters` | Meters ($m$) | **YES (RTK GPS)** | Absolute Trajectory Error of camera centers vs RTK ground truth. |
| `rpe_trans_mps` | Meters/sec | **YES (RTK GPS)** | Relative translation drift rate per second of flight. |
| `rpe_rot_deg_per_sec` | Deg/sec | **YES (RTK GPS)** | Relative rotation drift rate per second of flight. |
| `chamfer_distance_meters` | Meters ($m$) | **YES (LiDAR)** | Mean bidirectional point-to-point distance to reference LiDAR. |
| `accuracy_at_tau_pct` | Percent ($\%$) | **YES (LiDAR)** | Fraction of reconstructed points within $\tau = 0.10\text{m}, 0.50\text{m}$ of LiDAR. |
| `completeness_at_tau_pct` | Percent ($\%$) | **YES (LiDAR)** | Fraction of ground truth LiDAR points covered by reconstruction within $\tau$. |
| `f1_score_at_tau` | Scalar $[0, 1]$ | **YES (LiDAR)** | Harmonic mean of accuracy and completeness at threshold $\tau$. |
| `checkpoint_rmse_xyz_m` | Meters ($m$) | **YES (Survey GCPs)** | 3D RMSE measured at independent ground control checkpoints. |
| `scale_drift_pct` | Percent ($\%$) | **YES (RTK/LiDAR)** | Gauge scale expansion/contraction across trajectory length. |

---

## 4. Standardized Test Sequences & Datasets

All evaluations must be conducted on verified benchmark datasets containing synchronized video, RTK GNSS logs, and millimeter/centimeter-accurate LiDAR ground truth:

1. **Benchmark Sequence 01: Linear Urban Corridor**
   - **Environment:** Multi-story buildings, paved roads, vehicles, powerlines.
   - **Trajectory:** 400-meter straight single-pass flight, altitude $45\text{m}$, speed $8\text{ m/s}$, camera pitch $-45^\circ$.
   - **Ground Truth:** Terrestrial + Aerial LiDAR scan ($\pm 15\text{ mm}$ precision) + 12 RTK check points.
2. **Benchmark Sequence 02: Topographic Rural / Terrain**
   - **Environment:** Undulating hills, sparse vegetation, agricultural fields, dirt roads.
   - **Trajectory:** 600-meter linear single-pass flight, altitude $70\text{m}$, camera pitch $-60^\circ$.
   - **Ground Truth:** Airborne LiDAR DEM/DSM ($\pm 50\text{ mm}$ precision).
3. **Benchmark Sequence 03: Industrial Infrastructure / Complex Facades**
   - **Environment:** Warehouses, vertical tanks, metal structures, reflective roofs.
   - **Trajectory:** 300-meter linear flight with varying gimbal yaw.
   - **Ground Truth:** High-density LiDAR mesh.

---

## 5. Execution & Recording Protocol

For each test run:
1. **Manifest Creation:** Record machine hardware (CPU, GPU model, CUDA version, RAM, VRAM), library version pins, random seed, and input parameters.
2. **Execution:** Launch the benchmark script via isolated CLI command:
   ```bash
   python -m benchmarks.run_benchmark --pipeline [baseline|dust3r|vggt|fusion] --dataset-config configs/testing/seq01_urban.json --output-dir experiments/results/seq01_run/
   ```
3. **Validation & Log Archival:** The validation harness computes the metrics and writes `metrics.json` accompanied by a full cryptographic SHA256 checksum of input video and ground truth data.
4. **No Manual Alteration:** Output metric files must never be edited manually.
