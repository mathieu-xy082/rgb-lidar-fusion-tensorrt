# Workstream — Model input adapter contract

Branch: `feature/model-input-adapter-contract`

Parent branch: `main`

## Goal

Close the contract gap between the NumPy dataset/model batch adapter and the PyTorch `BaselineFusionModel` forward API.

The dataset/model batch adapter exposes a generic concatenated representation:

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

This workstream adds a small, tested adapter that converts the batch dictionary into model-call inputs without each downstream script re-implementing channel slicing. The nominal project contract is now enriched by default: sparse-only batches are treated as an ablation/debug path, while the default `[B, 11, H, W]` batch can represent sparse LiDAR as the degenerate limit of local Gaussian splatting.

In that identity limit:

```text
depth_expanded = normalized_camera_depth
confidence = point_mask
```

This corresponds to a zero-radius / infinitely peaked local kernel: no information is propagated to neighboring pixels, but downstream ONNX/demo/TensorRT code can still consume one stable enriched model contract.

## Scope

- Add an adapter function, for example `model_batch_to_baseline_inputs(batch, lidar_mode="enriched")`.
- Validate `input_channels` before slicing, not only numeric channel counts.
- For `lidar_mode="sparse"`, return:
  - `rgb = inputs[:, 0:3]`
  - `lidar_maps = inputs[:, 3:9]`
- For `lidar_mode="enriched"`, return the eight LiDAR channels:
  - six sparse maps
  - `depth_expanded`
  - `confidence`
- Make enriched batch production the default; preserve `include_splatted_depth=False` only for sparse-only ablations.
- Preserve NumPy-first behavior; only convert to torch tensors in an explicit optional helper if useful.
- Add tests that call `BaselineFusionModel(**model_inputs)` on synthetic batches for both sparse and enriched modes when the `ml` group is available.
- Document whether downstream ONNX/demo scripts should use the generic concatenated batch or the explicit model input adapter.

## Acceptance criteria

- `PDM_IGNORE_ACTIVE_VENV=1 pdm run validate` passes.
- New tests cover sparse and enriched slicing, wrong/missing channel metadata, and shape mismatch errors.
- A synthetic dataset batch can be transformed into explicit `rgb` / `lidar_maps` tensors and consumed by `BaselineFusionModel`.
- No generated datasets, ONNX files, TensorRT engines, or benchmark artifacts are committed.
- The branch reports `BRANCHE PRÊTE POUR REVIEW: feature/model-input-adapter-contract` only after implementation, tests, and docs are complete.
