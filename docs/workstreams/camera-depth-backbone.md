# Workstream — Camera-depth backbone

Branch: `ec/camera-depth-backbone`

Parent branch: `main`

## Goal

Replace the current scalar training smoke target with a dense camera-depth
backbone that predicts:

```text
depth_pred: [B, 1, H, W]
```

The goal is not to implement BEV yet. The goal is to train and validate a useful
image-space depth representation before using it for the future depth-aware lift
to BEV.

## Scope

- Add a small CNN encoder/decoder consuming RGB plus sparse/enriched LiDAR maps.
- Add a masked depth loss.
- Add a deterministic sparse LiDAR holdout path.
- Verify forward, loss, backward, finite gradients, and checkpoint compatibility.
- Keep the implementation CPU-safe for tests and GPU-ready for later training.
- Use a non-trivial local KITTI sample for the meaningful smoke training path.
- Do not implement the BEV lift, YOLO-like auxiliary head, or CenterPoint-like
  BEV heads in this branch.

## Progress

- [x] Create `ec/camera-depth-backbone` directly from `main`.
- [x] Add the dense `CameraDepthModel` encoder/decoder.
- [x] Add deterministic sparse LiDAR holdout and leakage-safe resplatting.
- [x] Add masked depth loss and a finite CPU training-step test.
- [x] Connect the camera-depth path to downloaded KITTI object frames.
- [x] Add the local multi-frame KITTI smoke runner and its configuration.
- [x] Add checkpoint/resume and depth-specific training metrics.
- [x] Run and record the first meaningful KITTI smoke experiment.

## Sparse LiDAR channels

The six raw sparse LiDAR channels remain:

```text
0: normalized_camera_depth
1: normalized_vehicle_x
2: normalized_vehicle_y
3: normalized_vehicle_z
4: intensity
5: point_mask
```

The enriched representation appends:

```text
6: depth_expanded
7: confidence
```

## Holdout then splatting

For sparse depth holdout training, the important rule is:

```text
Any channel given to the model must be computed only from kept points.
Any point used as target must be absent from every input channel.
```

The correct order is:

```text
1. Project all LiDAR points into the image.
2. Build the full sparse maps.
3. Split valid LiDAR pixels into kept pixels and holdout pixels.
4. Build the model input from kept pixels only.
5. Recompute depth_expanded/confidence from the kept sparse depth and kept mask.
6. Predict depth_pred.
7. Compute the loss only on holdout pixels.
```

In other words, `depth_expanded` and `confidence` must be recomputed **after
holdout**, from `sparse_depth_kept` and `sparse_mask_kept`.

Correct:

```text
depth_expanded_kept, confidence_kept
= splat_sparse_depth(sparse_depth_kept, sparse_mask_kept)
```

Incorrect:

```text
depth_expanded_full, confidence_full
= splat_sparse_depth(sparse_depth_full, sparse_mask_full)
```

The incorrect path leaks target information into the model input because holdout
points can influence the splatted channels.

## Training target

Recommended first useful target:

```text
Input:
  RGB
  sparse LiDAR kept
  optional splatted channels recomputed from kept points only

Target:
  full sparse depth

Loss mask:
  holdout_lidar_mask
```

Loss:

```text
loss = SmoothL1(
    depth_pred[holdout_lidar_mask],
    sparse_depth_full[holdout_lidar_mask]
)
```

This tests whether RGB plus visible LiDAR context can recover measured LiDAR
depth at pixels deliberately hidden from the input.

## Data policy for smoke training

The CI test can remain tiny and synthetic, because its role is only to validate
shape checks, deterministic holdout, finite loss, and backward propagation.

The training smoke that is meant to say something useful about the camera-depth
backbone should use a more substantial local KITTI sample:

```text
data/kitti/training/
  image_2/
  velodyne/
  calib/
```

Recommended local bootstrap:

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm run python docs/onboarding/scripts/download_kitti_samples.py \
  --source official \
  --count 64
```

The exact count can be adjusted for the machine, but the default meaningful
target should be dozens of frames rather than the 3-frame lightweight mirror.
The downloaded data must stay ignored by Git.

Run the camera-depth smoke after the download:

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm run train-camera-depth -- \
  --config configs/training/kitti_camera_depth.yaml
```

The default configuration uses 64 frames resized to `320x1024`, batch size 2,
a 20% sparse LiDAR holdout, and local splatting with a 2-pixel radius.
Checkpoints and metrics are written below
`results/training/kitti_camera_depth/` and remain ignored by Git.

The branch should therefore support two validation levels:

```text
CI smoke:
  synthetic or fixture-sized batch, no external downloads

local KITTI smoke:
  downloaded KITTI object frames, sparse holdout, masked depth loss
```

## First KITTI smoke result

Run completed locally on 2026-08-09 with the committed default configuration:

```text
dataset: KITTI Object training frames 000000..000063
device: CPU (local CUDA driver incompatible with the installed PyTorch build)
image shape: 128x416
batch size: 2
epochs: 1
optimizer steps: 32
mean training loss: 0.028757
mean gradient norm: 0.248513
checkpoint: results/training/kitti_camera_depth/checkpoints/latest.pt
```

This result proves that the downloaded KITTI path, projection, holdout,
resplatting, dense forward pass, masked loss, backward pass, and checkpoint path
work together on a non-trivial sample. It is not a validation metric and does
not yet demonstrate depth generalization.

The first real run also exposed saturation in the original `Softplus` output
head: after several batches, the prediction approached zero and all gradients
vanished. The camera-depth head now emits an unconstrained linear regression
value during training. Positivity can be applied at inference or revisited with
a calibrated depth parameterization after the basic learning behavior has been
measured.

### GPU validation

The same 64-frame smoke completed on the local GPU after pinning the PyTorch
CUDA build to the installed driver capability:

```text
driver: NVIDIA 575.64.03 (CUDA 12.9 capability)
PyTorch: 2.13.0+cu129
GPU: NVIDIA RTX 500 Ada Generation Laptop GPU
compute capability: 8.9
CUDA available: true
epochs: 1
optimizer steps: 32
mean training loss: 0.028765
mean gradient norm: 0.243644
checkpoint: results/training/kitti_camera_depth_gpu/checkpoints/latest.pt
```

The project binds `torch` to the official PyTorch CUDA 12.9 package index in
`pyproject.toml`. This prevents dependency resolution from silently selecting a
CUDA 13 build that requires a newer NVIDIA driver.

### Resolution decision

The active training resolution is `320x1024` with batch size 2. A complete
64-frame FP32 epoch ran on the RTX with 32 optimizer steps, mean loss `0.033595`,
and mean gradient norm `0.331645`.

The 4 GiB GPU has little remaining headroom at this setting: cuDNN reported one
failed optional 240 MiB workspace allocation, selected a fallback algorithm,
and completed the run. A subsequent mixed-precision experiment caused NVIDIA
Xid 62 and left the GPU in `Reset Required` state, so mixed precision is not
enabled in the committed configuration. The machine must be restarted before
further CUDA validation.

## Acceptance criteria

- `CameraDepthModel` returns `depth_pred: [B, 1, H, W]`.
- Masked depth loss validates shape compatibility and rejects empty masks.
- Holdout generation is deterministic under a seed.
- Splatted input channels, when enabled, are recomputed from kept points only.
- Tests cover data-leakage-sensitive behavior: holdout pixels must be zeroed from
  all six sparse LiDAR input channels. Splatted channels must be recomputed from
  kept points only; a neighbouring kept point may legitimately splat onto the
  holdout location.
- A smoke training step produces finite loss and finite non-zero gradients.
- A local KITTI smoke path can train on a non-trivial downloaded sample outside
  Git, initially targeting roughly 64 frames when resources allow.
- Generated checkpoints, metrics, and artifacts remain outside Git.
