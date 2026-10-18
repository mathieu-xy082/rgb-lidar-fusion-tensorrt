# Changelog

All notable changes to `rgb-lidar-fusion-tensorrt` will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `CameraDepthModel.forward_features()` exposes full-resolution `[B, 32, H, W]`
  final-decoder features for the future BEV lift, preserving the depth output,
  existing checkpoint schema, and feature-to-backbone gradients.
- Full resized KITTI `P2`, camera-from-LiDAR transform, and depth scale carried
  separately through dataset, model batches, holdout, and fixed evaluation.
- Differentiable metric camera-z conversion and camera-to-LiDAR backprojection,
  preserving full `P2` translation, v1 projection resize, and existing inputs.
  Geometry tests cover frame axes, non-zero translation, resizing, per-sample
  calibration, round trips, and depth gradients; validity and BEV binning remain
  separate follow-ups.

### Changed

- Standardized calibration fields, geometry arguments, and batch keys on
  `t_vehicle_to_camera` (`T_vehicle_to_camera` in equations), with an explicit
  projection/backprojection derivation and a tested numerical example in code.
  Legacy JSON manifests using `t_camera_from_vehicle` remain readable; Python
  callers should use the new name. Transform values and model weights are unchanged.

## [1.1.0] - 2026-10-18 - Merge branch 'ec/camera-depth-backbone' into main 
### Added

- Dense `CameraDepthModel` with deterministic sparse LiDAR holdout,
  leakage-safe resplatting, masked Smooth L1 training, and checkpoint support.
- KITTI camera-depth runner with deterministic train/validation splits, fixed
  per-frame validation masks, and held-out MAE/RMSE in meters.
- Random-point and spatial-block LiDAR holdout, with achieved holdout fractions
  and leakage-safe splatted-depth baseline errors and coverage.
- Matched 512-frame experiment configurations and cross-evaluation of both
  checkpoints under both holdout modes.
- Three notebooks for LiDAR projection, hybrid-input exploration, training,
  metric analysis, splatting ablation, and prediction inspection.
- Notebook GPU preflight with driver recovery checks, competing-process and
  free-memory guards, synchronized CUDA probes, and allocator diagnostics.
- Standalone bounded FP32 GPU diagnostic with read-only checkpoint loading,
  synchronized phase markers, incremental JSONL logs, and memory measurements.
- Optional dependency groups for notebook execution and visualization.
- Documentation index, current milestones, explicit BEV design links, and
  separated active/archive workstreams.
- Recorded camera-depth experiment results, sparse-supervision limitations,
  and feature/geometry/validity prerequisites for the future BEV lift.

### Changed

- Pinned PyTorch to `2.13.0+cu129` through the official CUDA 12.9 package index.
- Set camera-depth training to `320x1024`; YAML defaults use batch size 2,
  while the local 4 GiB GPU notebook and matched experiments use batch size 1.
- Reworked the README to distinguish the implemented pipeline, legacy baseline,
  camera-depth representation, and target BEV architecture.

### Fixed

- Restore Adam state against parameters already placed on the training device,
  avoiding mixed CPU/CUDA optimizer tensors when resuming a checkpoint.
- Persist camera-depth metrics and run metadata after each completed epoch,
  preserving completed history if a later epoch fails.
- Release unused CUDA allocator cache before the notebook's free-memory check,
  without lowering the safety threshold or deleting referenced tensors.
- Stop issuing explicit cache-cleanup calls after a failed CUDA smoke probe.

### Validated

- Camera-depth smoke training completed on 64 KITTI Object frames on CPU and
  the local NVIDIA GPU.
- Separate random-point and spatial-block GPU runs completed on 512 KITTI
  Object frames (410 train / 102 validation), five epochs and 2050 steps each.
  All four cross-evaluation combinations completed. See the
  [camera-depth workstream](docs/workstreams/camera-depth-backbone.md) for the
  scores, achieved holdout fractions, and interpretation limits.
- Sparse-versus-enriched input ablation on 64 frames improved held-out errors
  with splatted depth/confidence inputs; this is a single-seed result.
- After the local NVIDIA driver/GSP update from `575.64.03` to `615.71.09`,
  a 100-step GPU diagnostic, the remaining random-point epoch, the full block
  run, and cross-evaluation completed successfully. This does not establish
  the cause of the earlier Xid failures.
- Clean pre-merge validation on 2026-10-18: `232 passed, 1 skipped` (CUDA-only
  regression), lockfile check, deterministic projection/demo artifact checks,
  and a two-epoch CPU baseline training smoke. Remote CI was not rerun as part
  of these local checks.

### Known limitations

- Depth metrics supervise only hidden measured LiDAR pixels, not every dense
  prediction. Broad scene generalization, learned RGB semantics, and prediction uncertainty
  are not established by this single-seed pilot.
- Random and block holdout produce different target sets and achieved removal
  fractions; comparisons must use the same evaluation masks. Splat baseline
  errors include zero predictions outside splat support and require coverage
  to be interpreted correctly.
- Qualitative review remains before camera-depth v1 workstream closure.
  Mixed holdout and larger-dataset/sensitivity studies are optional follow-ups,
  not prerequisites for the first geometric BEV prototype.
- The feature-output interface, BEV lift, LiDAR BEV branch, and 3D detection
  heads are not implemented. Existing ONNX export covers the legacy baseline,
  not `CameraDepthModel`; real TensorRT engine validation remains deferred.
- The local 4 GiB GPU has limited headroom. AMP remains disabled, and the
  successful post-update runs do not guarantee future GPU stability.

## 1.0.0 - 2026-08-02

First stable portfolio/demo release of the RGB-LiDAR Fusion TensorRT prototype. This release establishes the reviewed technical foundation for an AV/robotics ML deployment pipeline: geometry, dataset contracts, enriched LiDAR representation, PyTorch model/training loop, ONNX parity scaffolding, TensorRT runtime planning, reproducible demo artifacts, and GitHub Actions CI.

### Added

- GitHub Actions CI equivalent to the former GitLab validation surface, with lockfile checking, full validation, and deterministic CPU training smoke coverage.
- PDM-based Python project structure with staged validation commands and lightweight default dependencies.
- KITTI calibration parsing and camera/LiDAR projection utilities.
- Synthetic and local KITTI smoke paths for projected LiDAR overlays and sparse LiDAR map generation.
- Lightweight KITTI dataset layer with manifest validation, source-path metadata, and clear dataset index errors.
- Deterministic local LiDAR surface splatting with `depth_expanded` and `confidence` channels.
- Baseline PyTorch RGB+LiDAR fusion model with the nominal enriched LiDAR contract:
  - `rgb: [B, 3, H, W]`
  - `lidar_maps: [B, 8, H, W]`
  - generic concatenated model-batch input `inputs: [B, 11, H, W]`.
- Dataset-to-model and model-input adapter contracts so downstream PyTorch, ONNX, demo, and training code avoid manual tensor slicing.
- ONNX export and ONNX Runtime parity validation for deterministic synthetic baseline inputs.
- Reproducible demo artifact pipeline producing overlay, sparse maps, splatted maps, enriched baseline inputs, and SHA-256 integrity metadata under ignored `results/` paths.
- TensorRT runtime environment detection, guarded build/infer/benchmark script stubs, and documented FP16 benchmark schema for future NVIDIA runtime validation.
- CPU-safe / GPU-ready baseline training loop with deterministic seed handling, explicit device diagnostics, checkpoint save/resume, metrics JSON/CSV, run metadata, and synthetic smoke configuration.
- Artifact policy for keeping datasets, checkpoints, ONNX exports, TensorRT engines, logs, and generated demo outputs out of Git.

### Validated

- GitHub Actions CI passed on the release branch and is integrated into `main`.
- Local validation on `main` passed with `91 passed` plus deterministic smoke/demo artifact generation.
- CPU training smoke completed two epochs, wrote checkpoint/metrics/run metadata outside the repository, and produced finite decreasing losses on the synthetic target.

### Deferred

- Real KITTI training target and trained checkpoint generation.
- GPU-backed training run with recorded device/runtime metadata.
- ONNX export from a trained checkpoint rather than synthetic/random baseline weights.
- TensorRT FP16 engine build and latency benchmark on a verified NVIDIA runtime.
- Detection-style head, post-processing, KITTI metrics, NMS, and real perception-quality evaluation.
