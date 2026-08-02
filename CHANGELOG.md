# Changelog

All notable changes to `rgb-lidar-fusion-tensorrt` will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until the first stable release is cut, the package version remains `0.1.0`; after `ec/ci-github` is integrated into `main`, the next dedicated release commit should promote the project to `1.0.0` and move the relevant entries under `## 1.0.0 - YYYY-MM-DD`.

## Unreleased

### Added

- GitHub Actions CI equivalent to the former GitLab validation surface, with lockfile checking, full validation, and deterministic CPU training smoke coverage.

### Release preparation

- Prepare the repository for the first stable release by documenting the validated roadmap milestones below.
- Next release step after integration into `main`: bump `[project].version` from `0.1.0` to `1.0.0`, move these release-preparation notes into a dated `1.0.0` section if still relevant, tag `v1.0.0`, and publish a stable GitHub release.

## 0.1.0 - 2026-08-02

Initial validated prototype baseline. This version is still pre-stable from a packaging perspective, but it contains the reviewed technical foundation that should become the first stable portfolio/demo release once GitHub CI is integrated.

### Added

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

- Local validation on the integrated training baseline passed with `91 passed` plus deterministic smoke/demo artifact generation.
- CPU training smoke completed two epochs, wrote checkpoint/metrics/run metadata outside the repository, and produced finite decreasing losses on the synthetic target.
- The current repository state is suitable as the technical base for a stable `v1.0.0` release once GitHub Actions CI is merged and the release commit is created.

### Deferred

- Real KITTI training target and trained checkpoint generation.
- GPU-backed training run with recorded device/runtime metadata.
- ONNX export from a trained checkpoint rather than synthetic/random baseline weights.
- TensorRT FP16 engine build and latency benchmark on a verified NVIDIA runtime.
- Detection-style head, post-processing, KITTI metrics, NMS, and real perception-quality evaluation.
