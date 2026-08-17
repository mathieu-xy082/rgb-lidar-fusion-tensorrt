# Changelog

All notable changes to `rgb-lidar-fusion-tensorrt` will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Dense `CameraDepthModel` with deterministic sparse LiDAR holdout,
  leakage-safe resplatting, masked Smooth L1 training, and checkpoint support.
- Local KITTI camera-depth runner using a meaningful multi-frame sample.
- Documentation index, current milestones, explicit BEV design links, and
  separated active/archive workstreams.

### Changed

- Pinned PyTorch to `2.13.0+cu129` for compatibility with the local CUDA 12.9
  driver.
- Set the default camera-depth training configuration to `320x1024`, batch
  size 2.
- Reworked the README to distinguish the implemented pipeline, legacy baseline,
  camera-depth representation, and target BEV architecture.

### Validated

- Camera-depth smoke training completed on 64 KITTI Object frames on CPU and
  the local NVIDIA GPU.
- Full validation passes with 109 tests plus deterministic smoke/demo artifact
  generation.

### Known limitations

- The camera-depth run proves trainability, not depth generalization; validation
  metrics and visualizations are still required.
- The local 4 GiB GPU has limited headroom at `320x1024`. Mixed precision
  remains disabled after an NVIDIA Xid 62 failure pending a clean CUDA retest.

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
