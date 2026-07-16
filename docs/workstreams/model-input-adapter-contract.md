# Workstream — Model input adapter contract

Branch: `feature/model-input-adapter-contract`

Parent branch: `main`

## Goal

Close the contract gap between the NumPy dataset/model batch adapter and the PyTorch `BaselineFusionModel` forward API.

The integrated dataset/model batch branch exposes a generic concatenated representation:

```text
inputs: [B, 9, H, W]   # RGB + six sparse LiDAR maps
inputs: [B, 11, H, W]  # RGB + six sparse LiDAR maps + depth_expanded/confidence
```

The integrated baseline model consumes explicit tensors:

```text
rgb:        [B, 3, H, W]
lidar_maps: [B, 6, H, W]  # sparse mode
lidar_maps: [B, 8, H, W]  # enriched mode
```

This workstream should add a small, tested adapter that converts the batch dictionary into model-call inputs without each downstream script re-implementing channel slicing.

## Scope

- Add an adapter function, for example `model_batch_to_baseline_inputs(batch, lidar_mode="enriched")`.
- Validate `input_channels` before slicing, not only numeric channel counts.
- For `lidar_mode="sparse"`, return:
  - `rgb = inputs[:, 0:3]`
  - `lidar_maps = inputs[:, 3:9]`
- For `lidar_mode="enriched"`, require or build the eight LiDAR channels:
  - six sparse maps
  - `depth_expanded`
  - `confidence`
- Preserve NumPy-first behavior; only convert to torch tensors in an explicit optional helper if useful.
- Add tests that call `BaselineFusionModel(**model_inputs)` on synthetic batches for both sparse and enriched modes when the `ml` group is available.
- Document whether downstream ONNX/demo scripts should use the generic concatenated batch or the explicit model input adapter.

## Acceptance criteria

- `PDM_IGNORE_ACTIVE_VENV=1 pdm run validate` passes.
- New tests cover sparse and enriched slicing, wrong/missing channel metadata, and shape mismatch errors.
- A synthetic dataset batch can be transformed into explicit `rgb` / `lidar_maps` tensors and consumed by `BaselineFusionModel`.
- No generated datasets, ONNX files, TensorRT engines, or benchmark artifacts are committed.
- The branch reports `BRANCHE PRÊTE POUR REVIEW: feature/model-input-adapter-contract` only after implementation, tests, and docs are complete.
