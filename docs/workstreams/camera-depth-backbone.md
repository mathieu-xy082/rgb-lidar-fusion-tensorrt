# Workstream — Camera-depth backbone

Branch: `ec/camera-depth-backbone`

Parent branch: `main`

Status: v1 implementation, validation metrics, analysis notebook, and clean
multi-epoch GPU runs are complete; structured LiDAR holdout, dataset scale-up,
sensitivity, and dense-depth validation remain.

Related documents:

- [camera-depth design](../bev-cam-depth-backbone.md);
- [target BEV architecture](../bev-model-roadmap.md);
- [project milestones](../milestones.md).

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

The groups below mirror the `Scope` items so that implementation progress and
remaining work can be reviewed without inferring coverage from commit history.

### 1. CNN encoder/decoder for RGB plus sparse/enriched LiDAR

Scope item: add a small CNN encoder/decoder consuming RGB plus sparse/enriched
LiDAR maps.

- [x] Add `CameraDepthModel` with a two-level encoder, skip-connected decoder,
  and dense `depth_pred: [B, 1, H, W]` output.
- [x] Support the six-channel sparse and eight-channel enriched LiDAR contracts
  through `lidar_mode`.
- [x] Validate RGB/LiDAR batch, channel, and spatial compatibility before the
  forward pass.
- [x] Test finite dense output while preserving an odd input resolution.
- [ ] Add a direct forward test for `lidar_mode="sparse"`; the code path exists,
  but current camera-depth forward coverage exercises enriched mode only.

### 2. Masked depth loss

Scope item: add a masked depth loss.

- [x] Add masked Smooth L1 evaluated only on selected LiDAR target pixels.
- [x] Validate prediction, target, and mask shapes.
- [x] Reject empty masks and non-finite selected targets.
- [x] Test that unselected pixels do not influence the loss and that an empty
  mask fails explicitly.

### 3. Deterministic sparse LiDAR holdout

Scope item: add a deterministic sparse LiDAR holdout path.

- [x] Select holdout pixels deterministically from a configured seed.
- [x] Remove each held-out point from all six sparse LiDAR input channels.
- [x] Recompute `depth_expanded` and `confidence` from kept points only.
- [x] Test deterministic masks, target preservation, and leakage-safe
  resplatting.

### 4. Trainability and checkpoint compatibility

Scope item: verify forward, loss, backward, finite gradients, and checkpoint
compatibility.

- [x] Test dense forward and masked loss.
- [x] Run a CPU optimizer step with finite loss and finite non-zero gradients.
- [x] Save model/optimizer state, epoch, step, and configuration.
- [x] Test checkpoint resume and metrics-history preservation on the KITTI
  camera-depth runner.

### 5. CPU-safe and GPU-ready execution

Scope item: keep the implementation CPU-safe for tests and GPU-ready for later
training.

- [x] Keep CI tests independent of CUDA and external datasets.
- [x] Add explicit `auto`/`cpu`/`cuda` selection and device diagnostics.
- [x] Pin the PyTorch CUDA 12.9 build to the local NVIDIA driver capability.
- [x] Complete and record real GPU smoke runs, including the active
  `320x1024`, batch-size-2 configuration.

### 6. Meaningful local KITTI smoke

Scope item: use a non-trivial local KITTI sample for the meaningful smoke
training path.

- [x] Add `KittiObjectDepthDataset` for RGB, Velodyne, and calibration frame
  triplets.
- [x] Add the configurable `train-camera-depth` runner.
- [x] Require the configured sample count instead of silently accepting an
  incomplete download.
- [x] Complete and record a 64-frame KITTI Object smoke on CPU and GPU.

### 7. Scope boundary

Scope item: do not implement the BEV lift, YOLO-like auxiliary head, or
CenterPoint-like BEV heads in this branch.

- [x] Keep this branch limited to the image-space camera-depth representation.
- [x] Track the BEV lift and detection heads as subsequent milestones.

### Follow-up: representation validation

These tasks are not missing pieces of the v1 holdout smoke. They are required
before claiming that the learned image-space representation generalizes well
enough to feed the BEV lift.

- [x] Separate training and validation samples deterministically.
- [x] Add held-out depth metrics on the validation split, evaluated without
  gradients or optimizer updates.
- [x] Keep validation holdout masks fixed per dataset index and independent of
  batch size.
- [x] Compare model MAE/RMSE against the leakage-safe splatted-depth baseline.
- [x] Visualize depth predictions, errors, holdout targets, and confidence in
  `notebooks/03_camera_depth_training_analysis.ipynb`.
- [x] Add a notebook GPU preflight that blocks training on driver recovery,
  insufficient free memory, competing CUDA processes, or a failed CUDA probe.
- [x] Add a reproducible sparse-versus-enriched input ablation path.
- [x] Run the sparse-input ablation and record whether splatting improves
  held-out validation MAE/RMSE.
- [x] Add deterministic LiDAR block dropout to batch preparation while keeping
  RGB visible.
- [x] Recompute the fixed splatting channels after block dropout and verify
  that every target measurement is absent from all model inputs.
- [ ] Report random-point and spatial-block validation metrics separately.
- [ ] Scale the representation experiment beyond the 64-frame smoke sample.
- [ ] Measure sensitivity to holdout fraction and splatting radius.
- [x] Repeat a clean GPU run after the recorded Xid 62 failure.
- [ ] Document what sparse holdout supervision can and cannot establish about
  dense depth quality.

### Next experiment: LiDAR block dropout

Development increments:

- [x] Extract `build_depth_holdout_mask()` from batch preparation; preserve the
  seeded random-point selection and test it independently of PyTorch.
- [x] Add spatial block selection and its region mask with deterministic tests.
- [x] Connect block masks to input removal and leakage-safe resplatting.
- [ ] Expose holdout settings in the runner and a dedicated pilot configuration.
- [ ] Add separate random-point/block validation and notebook visualizations.
- [ ] Run the smoke and matched 512-frame experiments.

The extracted random-point helper accepts a valid-depth mask `[B, 1, H, W]`
and returns a boolean target mask of the same shape. It retains the previous
singleton behavior: a sample with only one valid measurement uses that point
as its target and has no kept LiDAR context.

`build_spatial_depth_holdout_masks()` returns `(holdout_mask, region_mask)`:
the first selects only real measurements to supervise; the second covers entire
rectangles for visualization. This helper is tested independently. It samples
blocks around
not-yet-held-out measurements, preserves complete rectangles at image borders,
and rejects candidates that would remove every valid measurement. It requires
at least two measurements per sample and raises if no safe block is found
within its bounded attempt budget; it does not fall back to random points.

The holdout fraction is a target rather than an exact constraint: a complete
block may exceed it, and the maximum block count may stop selection earlier.
Future run metrics must record the achieved fraction. For example, on KITTI
frame `000000` at `320x1024`, seed 13 and the default 16..48 by 32..128 pixel
blocks (maximum 8) select 3,128 of 20,195 measurements (15.49%) for a requested
20%. This is a mask-generation check, not a training result.

Batch preparation now accepts `holdout_mode="spatial_blocks"` and the three
block settings. The default remains `holdout_mode="random_points"`. Both paths
remove targets from all six sparse channels before recomputing enriched inputs,
even when the incoming batch already contains full-point splats. The returned
`holdout_region_mask: [B, 1, H, W]` covers rectangles in spatial mode and copies
`loss_mask` in random-point mode. RGB and `depth_target` remain unchanged.

Tests verify that changing removed measurements cannot change model inputs,
that kept points may still splat into held-out regions, and that a CPU optimizer
step with spatial holdout produces finite non-zero gradients. Runner settings,
validation protocol selection, and notebook controls remain to be connected.

The enriched input remains the primary representation. The current fixed
Gaussian splatting is a strong baseline and should not be removed while testing
generalization. Instead of hiding only isolated LiDAR pixels, training should
also hide spatially coherent image regions:

```text
1. Build the full six-channel sparse LiDAR maps.
2. Sample one or more deterministic image-space blocks.
3. Mark sparse LiDAR measurements inside the blocks as holdout targets.
4. Remove those measurements from all six sparse input channels.
5. Recompute depth_expanded and confidence from kept points only.
6. Keep RGB visible inside the blocks.
7. Evaluate the prediction only at the removed raw LiDAR measurements.
```

This is a LiDAR-modality dropout, not RGB cutout. It asks the network to use RGB
semantics and surrounding geometry to complete partially observed surfaces such
as vegetation, pedestrians, vehicles, and walls. Masking RGB and LiDAR together
is a separate robustness experiment and is not the initial priority.

The first implementation should support both holdout modes under configuration:

```text
random_points: current independent sparse-point holdout
spatial_blocks: variable-size rectangular or irregular LiDAR holes
```

Tests must cover deterministic masks, non-empty kept and target sets, removal
from all six sparse channels, and leakage-safe resplatting. Validation should
retain fixed masks per sample and publish metrics by holdout mode and block-size
bucket; a single aggregate score would hide whether performance collapses as
the missing region grows.

No model architecture change is required to start this experiment. Block
dropout changes input and target preparation while preserving the enriched
eight-channel LiDAR contract and `depth_pred: [B, 1, H, W]`. The current model
has only two downsampling levels, so its useful spatial context may become a
limit for larger holes. Freeze the architecture for the first comparison; add
a deeper pyramid level, dilation, or another context module only if validation
error clearly increases with block size.

### Dataset scale-up

The existing 64 KITTI Object frames are a smoke and debugging dataset. They are
enough to validate projection, holdout, resplatting, optimization, and the first
sparse-versus-enriched comparison, but they are not enough to establish
generalization across scenes and object appearances.

Use the official range downloader to extend the existing local dataset; files
already present are skipped:

```bash
# First meaningful block-dropout pilot: 512 frame triplets.
PDM_IGNORE_ACTIVE_VENV=1 pdm run python docs/onboarding/scripts/download_kitti_samples.py \
  --source official \
  --count 512
```

Then set `sample_count: 512` in an experiment-specific training configuration.
Keep the 64-frame configuration as the fast smoke path instead of silently
changing its purpose. The official KITTI Object training set contains 7,481
image/Velodyne/calibration triplets, and the current downloader accepts
`--count 7481`; downloading the complete archives directly is more efficient
than thousands of range requests when the full set is desired.

Recommended progression:

```text
64 frames:    functional smoke and fast regression checks
512 frames:   first block-dropout pilot and hyperparameter debugging
7,481 frames: final KITTI Object representation experiments
```

Each scale must use a fixed train/validation split and fixed validation masks.
Once block sizes or other hyperparameters have been selected, reserve a final
untouched test subset rather than repeatedly interpreting validation results as
test performance. Additional datasets such as KITTI Raw, nuScenes, or Waymo can
improve environmental diversity later, but they require dataset-specific
calibration adapters and should not be mixed into the first block-dropout
implementation.

### Splatting ablation result

The matched five-epoch GPU runs completed on 2026-10-11 with seed 13, the same
51/13 train/validation split, and the same fixed validation holdout masks. Both
runs reached their best validation MAE at epoch 4:

```text
input mode   best MAE   best RMSE
sparse        3.513 m     5.634 m
enriched      1.302 m     3.031 m
```

Adding `depth_expanded` and `confidence` reduced MAE by 2.211 m (63%) and RMSE
by 2.603 m (46%) in this run. The enriched model remained better at every
epoch. This is strong evidence that splatting helps the current architecture on
held-out LiDAR reconstruction, but it is still a single-seed result and does
not validate depth at pixels without LiDAR ground truth.

The epoch metrics written to `metrics.json` and `metrics.csv` now contain:

```text
loss / grad_norm             training diagnostics
val_loss                     normalized masked Smooth L1
val_mae_m / val_rmse_m       model errors in meters
val_splat_mae_m / rmse_m     depth_expanded baseline errors in meters
val_splat_coverage           holdout pixels reached by a kept-point splat
val_pixel_count              number of held-out validation measurements
```

`validation_holdout_seed + dataset_index` defines each validation mask. The
same pixels are therefore evaluated after every epoch, even if validation batch
size changes. These are sparse-point generalization metrics, not dense ground
truth metrics.

The analysis notebook can launch or load a run, plot training/validation curves,
compare the model against the splatted-depth baseline, and visualize a
representative validation frame. Reusable loading and plotting logic lives in
`src/rgb_lidar_fusion/camera_depth_analysis.py` rather than only in the notebook.

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

Fast local smoke bootstrap:

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm run python docs/onboarding/scripts/download_kitti_samples.py \
  --source official \
  --count 64
```

The 64-frame configuration is intentionally retained as a fast local smoke. A
representation experiment intended to measure generalization should use the
512-frame pilot or the full 7,481-frame KITTI Object training set described
above. The downloaded data must stay ignored by Git.

Run the camera-depth smoke after the download:

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm run train-camera-depth -- \
  --config configs/training/kitti_camera_depth.yaml
```

The default smoke configuration uses 64 frames resized to `320x1024`, batch
size 2, a 20% sparse LiDAR holdout, and local splatting with a 2-pixel radius.
Checkpoints and metrics are written below
`results/training/kitti_camera_depth/` and remain ignored by Git.

The branch should therefore support two validation levels:

```text
CI smoke:
  synthetic or fixture-sized batch, no external downloads

local KITTI smoke:
  downloaded KITTI object frames, sparse holdout, masked depth loss

representation experiment:
  fixed split over at least the 512-frame pilot, random and block holdout metrics
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
