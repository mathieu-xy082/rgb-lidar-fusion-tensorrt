# Workstream — Camera-depth backbone

Branch: `ec/camera-depth-backbone`

Parent branch: `main`

Updated: 2026-10-15.

Status: v1 implementation and sparse-supervised evaluation are complete,
including matched five-epoch random/block runs on 512 frames and cross-evaluation.
Qualitative prediction review remains before workstream closure. Dense-depth
quality is not established; further training studies are not prerequisites for
starting an experimental BEV lift.

Related documents:

- [camera-depth design](../bev-cam-depth-backbone.md);
- [target BEV architecture](../bev-model-roadmap.md);
- [project milestones](../milestones.md).

## Current handoff

- Completed: dense forward/loss/backward, leakage-safe random and block holdout,
  checkpoint resume, validation metrics, notebooks, and the 512-frame pilot.
- Active execution setting: enriched inputs, `320x1024`, batch size 1, FP32,
  PyTorch `2.13.0+cu129`, NVIDIA driver/GSP `615.71.09`.
- Next: inspect both checkpoints on the same validation frames and masks, then
  close v1 as functional and evaluated under sparse supervision.
- Subsequent development: expose backbone features and implement the geometric
  contract described in [BEV connection prerequisites](../bev-cam-depth-backbone.md#prerequis-du-raccord-bev).
- Deferred: mixed holdout, block-size/radius/fraction sensitivity, more data,
  additional seeds, and dense or semi-dense supervision.

The GPU incident sections below are historical records, not the current setup.

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
- [x] Add a direct forward test for `lidar_mode="sparse"`.

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
- [x] Pin the PyTorch CUDA 12.9 build; retain it after the driver update.
- [x] Complete and record real GPU smoke runs at `320x1024`, including the
  historical batch-size-2 smoke and the batch-size-1 representation pilot.
- [x] Complete the fifth random-point epoch and the full five-epoch block run
  after updating the local driver/GSP to `615.71.09`.

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

This section tracks the sparse-supervised representation pilot. Completion
allows an experimental BEV handoff, not a claim of reliable dense depth or
production-ready perception.

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
- [x] Report random-point and spatial-block validation metrics separately.
- [x] Scale the representation experiment to the matched 512-frame pilot.
- [x] Train both holdout modes for five epochs and cross-evaluate both models
  on fixed random-point and spatial-block masks.
- [x] Repeat a clean GPU run after the recorded Xid 62 failure.
- [x] Document what sparse holdout supervision can and cannot establish about
  dense depth quality.

### Remaining v1 closure

- [ ] Inspect a small set of identical validation frames for both checkpoints,
  under both holdout modes; review depth discontinuities, nearby objects,
  distant surfaces, and obvious invalid predictions.
- [ ] Record the qualitative findings and close v1 if no major inconsistency
  blocks the geometric prototype. No additional training score is required.

### Deferred representation studies

These are optional improvements, not blockers for the first BEV prototype:

- [ ] Measure sensitivity to holdout fraction and splatting radius.
- [ ] Evaluate fixed small/medium/large blocks before making claims about
  receptive field or spatial-context limits.
- [ ] Test mixed random/block training and, for a controlled geometry study,
  bring achieved holdout fractions closer together.
- [ ] Expand scene diversity and seeds, and add dense/semi-dense validation
  before claiming broadly reliable depth completion.
- [ ] Add progress reporting to long notebook cross-evaluations.

### LiDAR block dropout implementation

Development increments:

- [x] Extract `build_depth_holdout_mask()` from batch preparation; preserve the
  seeded random-point selection and test it independently of PyTorch.
- [x] Add spatial block selection and its region mask with deterministic tests.
- [x] Connect block masks to input removal and leakage-safe resplatting.
- [x] Expose holdout settings in the runner and a dedicated pilot configuration.
- [x] Add separate random-point/block validation and notebook visualizations.
- [x] Run the smoke and matched 512-frame experiments.
- [x] Complete the four train/evaluation holdout combinations.

The extracted random-point helper accepts a valid-depth mask `[B, 1, H, W]`
and returns a boolean target mask of the same shape. It retains the previous
singleton behavior: a sample with only one valid measurement uses that point
as its target and has no kept LiDAR context.

`build_spatial_depth_holdout_masks()` returns `(holdout_mask, region_mask)`:
the first selects only real measurements to supervise; the second covers entire
rectangles for visualization. This helper is tested independently. It samples
blocks around not-yet-held-out measurements, preserves complete rectangles at
image borders, and rejects candidates that would remove every valid measurement. It requires
at least two measurements per sample and raises if no safe block is found
within its bounded attempt budget; it does not fall back to random points.

The holdout fraction is a target rather than an exact constraint: a complete
block may exceed it, and the maximum block count may stop selection earlier.
Run metrics now record the achieved fraction. For example, on KITTI
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
step with spatial holdout produces finite non-zero gradients.

The runner now accepts `holdout_mode` with the existing `random_points` default.
Its flat scalar YAML format expresses block ranges as:

```yaml
holdout_mode: spatial_blocks
block_height_min_px: 16
block_height_max_px: 48
block_width_min_px: 32
block_width_max_px: 128
block_max_count: 8
```

Training and validation use the same configured mode and block ranges. Validation
seeds remain `validation_holdout_seed + dataset_index`, so per-frame masks are
independent of validation batch size and sample order. The checkpoint stores the
configuration, and `run_metadata.json` records the effective holdout settings.
Epoch JSON/CSV metrics and CLI output include `train_holdout_fraction` and
`val_holdout_fraction`, computed as total removed measurements divided by total
valid measurements over the respective split, rather than image-area fractions.

The dedicated pilot config is
[`kitti_camera_depth_block_dropout.yaml`](../../configs/training/kitti_camera_depth_block_dropout.yaml):
512 frames, enriched inputs, five epochs, `320x1024`, a YAML default batch size
of 2, and a dedicated output directory. The completed local runs override batch
size to 1 in the notebook; reproduce that override for matched comparisons.
The CLI entry point below uses the YAML defaults, not the notebook override:

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm run train-camera-depth -- \
  --config configs/training/kitti_camera_depth_block_dropout.yaml
```

The matched random-point pilot config is
[`kitti_camera_depth_random_holdout_512.yaml`](../../configs/training/kitti_camera_depth_random_holdout_512.yaml).
It uses the same dataset, split, seeds, normalization, input mode, and training
hyperparameters as the spatial pilot, with its own output directory.

The analysis notebook now offers a CPU holdout preview before training: regions,
kept sparse depth, removed measurements, and recomputed splat depth/confidence.
It requires KITTI data but no checkpoint. Sample prediction plots reconstruct
the holdout from the saved checkpoint configuration, including spatial regions.

`compare_camera_depth_holdout_checkpoints()` evaluates both checkpoints on both
holdout modes over the same validation split. It rejects incompatible dataset
protocols, training hyperparameters, epochs, or optimizer-step counts. Each
evaluation mode uses the same fixed per-frame masks for both models; spatial
validation uses the block ranges from the spatial checkpoint. Evaluation batch
size may differ from training batch size. The notebook flag
`RUN_HOLDOUT_COMPARISON` only evaluates existing checkpoints and saves a JSON
report with metric rows, checkpoint paths, evaluation settings, and achieved
holdout fractions. MAE/RMSE matrices show training mode by evaluation mode;
these are checkpoint scores, not historical best-epoch scores.

CPU fixture tests cover both holdout modes, sparse/enriched inputs, validation,
metrics persistence, checkpoint resume, cross-evaluation consistency, and
execution of the preview/comparison/sample notebook cells. The notebook has
also been executed on local KITTI data on CPU with all training flags disabled.
The matched 512-frame experiments and cross-evaluation are complete; block-size
sensitivity remains a deferred study. Sparse LiDAR supervision does not establish depth quality
at pixels without LiDAR measurements. In large holes, splat coverage may be low:
the current splat baseline errors include zero-valued predictions outside its
support and must be interpreted together with coverage.

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

The original 64 KITTI Object frames are a smoke and debugging dataset. They are
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

This first scale-up is complete: the local pilot uses 512 matching
image/Velodyne/calibration triplets, split into 410 train and 102 validation
frames. Further expansion is not required to begin the geometric BEV prototype.

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

### Matched 512-frame results

The completed pilot uses 410 train / 102 validation frames, seed/split seed 13,
validation holdout seed 1013, enriched inputs, `320x1024`, batch size 1, and five
epochs (2050 optimizer steps) per model. Both models started from the same
seeded initialization; the block model did not resume from the random model.
Validation masks are identical between models within each evaluation mode.

The local report is
`results/analysis/camera_depth_holdout_comparison.json`; the corresponding runs
are under `results/training/kitti_camera_depth_random_holdout_512/` and
`results/training/kitti_camera_depth_block_dropout_512/`. These artifacts remain
ignored by Git; the rounded results below preserve the conclusions in docs.

| Training holdout | Evaluation holdout | MAE (m) | RMSE (m) |
|---|---|---:|---:|
| Random points | Random points | 0.602 | 1.674 |
| Random points | Spatial blocks | 4.883 | 8.327 |
| Spatial blocks | Random points | 1.744 | 3.249 |
| Spatial blocks | Spatial blocks | 2.117 | 4.225 |

| Evaluation holdout | Achieved fraction | Target pixels | Splat coverage | Splat MAE / RMSE (m) |
|---|---:|---:|---:|---:|
| Random points | 20.0% | 386015 | 80.5% | 3.349 / 8.529 |
| Spatial blocks | 10.7% | 206250 | 4.2% | 16.110 / 19.776 |

On the same spatial targets, block training reduced MAE by approximately 57%
and RMSE by 49% relative to random training. Random training remained more
accurate on random-point targets. This supports a robustness/accuracy trade-off
under the tested missing-data patterns, not a universally better checkpoint.

### Interpretation limits

- Scores compare models within the same evaluation mode, not across different
  target sets. Only hidden, measured LiDAR pixels have reference depths.
- Training and evaluation holdout fractions differ between modes (about 20%
  random versus 11% spatial). Equal epoch/step budgets do not equalize the
  number or distribution of supervised measurements.
- Splat baseline errors include zero predictions outside splat support. The
  large spatial baseline error must be read with its 4.2% coverage; it is not
  the error on only the covered pixels.
- A dense output is not a dense validation result. Sky, occluded areas, object
  boundaries, and pixels without LiDAR references remain insufficiently tested.
- Depth supervision and these scores do not establish learned object semantics,
  calibrated prediction uncertainty, or dependence on the RGB modality.
- This is a single-seed pilot on a fixed KITTI subset. It does not establish
  broad scene generalization, a held-out final test score, or BEV accuracy.
- Keep both checkpoints for the BEV prototype. Do not require a mixed-holdout
  run, SOTA scores, or a larger network before testing the geometric connection.

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

The resolution selected for the original smoke was `320x1024`, batch size 2. A complete
64-frame FP32 epoch ran on the RTX with 32 optimizer steps, mean loss `0.033595`,
and mean gradient norm `0.331645`.

The 4 GiB GPU has little remaining headroom at this setting: cuDNN reported one
failed optional 240 MiB workspace allocation, selected a fallback algorithm,
and completed the run. A subsequent mixed-precision experiment caused NVIDIA
Xid 62 and left the GPU in `Reset Required` state, so mixed precision is not
enabled in the committed configuration. At that point, the machine required a
restart before further CUDA validation. Later recovery and completed runs are
recorded below.

### Interrupted 512-frame GPU run

On 2026-10-13, the matched random-point pilot at `320x1024`, batch size 1,
failed after its first completed epoch. Kernel logs reported Xid 62 followed
by Xid 154, and NVIDIA SMI reported `GPU Recovery Action: Reset`. This run used
FP32 without mixed precision; the exact trigger has not been established.

The last checkpoint was inspected on CPU: epoch 1, step 410, finite model
weights, and optimizer state present. A separate copy was retained under
`results/training/kitti_camera_depth_random_holdout_512/checkpoints/` before
recovery. Training can resume from that completed epoch once the GPU is healthy;
progress within the interrupted epoch was not saved.

The failure exposed delayed sidecar persistence: metrics and run metadata had
only been written at the end of the entire run, so epoch-1 metrics were not
retained despite the valid checkpoint. The camera-depth runner now writes JSON,
CSV, and metadata after each completed epoch alongside its checkpoint. CPU
tests simulate a later-epoch failure and verify that completed metric rows
survive and remain intact after resume.

The first GPU resume attempt exposed a separate device-placement bug: Adam's
moments were restored against CPU parameters before the training step moved
the model to CUDA. Both runners now move the model before optimizer creation.
Checkpoint tensors are loaded through CPU, then PyTorch restores optimizer
moments onto the parameter device while preserving the step-counter policy.
Regression coverage checks initialization order and includes a CUDA-only
checkpoint-resume optimizer step.

The CUDA regression test passed on the local GPU. A separate tiny-input probe
also loaded the actual epoch-1/step-410 camera-depth checkpoint and completed
one GPU optimizer step with finite loss and gradients. The checkpoint SHA-256
was unchanged before and after the probe; the long resumed epoch remains to
be validated.

The subsequent resumed run failed again on 2026-10-13 at 18:19:58 local time,
this time reporting `CUDNN_STATUS_EXECUTION_FAILED` during backward. Kernel logs
again recorded Xid 62, channel cleanup events (Xid 45), and Xid 154 requesting
a GPU reset. NVIDIA SMI confirmed `GPU Recovery Action: Reset`; the displayed
98% utilization must not be interpreted as successful training progress.
The checkpoint still matches the retained epoch-1/step-410 copy byte for byte;
no additional completed epoch was saved.

This recurrence is distinct from the corrected Adam device-placement bug.
Neither an out-of-memory condition nor a specific cuDNN defect has been
established. Recover the GPU before running further CUDA operations. Because
desktop applications also use this GPU, prefer a controlled machine reboot
over attempting an in-session GPU reset. Before another full training run,
isolate repeated forward/backward steps at the actual input resolution in a
standalone process, record the PyTorch/CUDA/cuDNN/driver versions and GPU memory
usage, and compare backend settings one at a time if needed. A short passing
probe does not establish long-run stability.

### Standalone GPU diagnostic

Run this from the repository root, with no notebook training or other CUDA
compute process running. It requires the project's ML dependencies. The script
sets `CUDA_LAUNCH_BLOCKING=1` before importing PyTorch and refuses to proceed
when the existing GPU preflight reports insufficient memory, competing compute
processes, or a required GPU recovery action.

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm run python scripts/probe_camera_depth_gpu.py \
  --steps 5 --height 320 --width 1024 --batch-size 1 \
  --checkpoint results/training/kitti_camera_depth_random_holdout_512/checkpoints/latest.pt
```

The diagnostic uses `CameraDepthModel`, the masked depth loss, and Adam in FP32.
It repeatedly processes a fixed synthetic batch, not KITTI frames; this isolates
the model/backend from dataset preparation but does not measure prediction
quality or reproduce every condition of the interrupted training run. Loading
the optional checkpoint exercises optimizer resume, but updated weights are
discarded when the process exits. No training checkpoint or metrics are saved.

A fresh JSONL log under `results/diagnostics/` records library versions, NVIDIA
SMI snapshots, the checkpoint epoch/step, and synchronized phase markers for
input transfer, forward, backward, gradient checking, and the optimizer update.
Each completed step reports loss, gradient norm, duration, and CUDA allocated,
reserved, peak-allocated, and globally free memory. Log rows are flushed as they
are written. After a failure, diagnostics use NVIDIA SMI rather than issuing
more CUDA work; there is no automatic retry. Existing log files are not
overwritten, including when `--output` explicitly selects a path.

Only after a healthy-device baseline, repeat the same command with
`--disable-cudnn` to isolate that backend if needed. This is a diagnostic
comparison, not a permanent training change or a guaranteed workaround.
`--device cpu --height 9 --width 13 --steps 1` checks the script itself without
validating CUDA. A passing five-step GPU test is necessary evidence for further
investigation, not proof that a full epoch will be stable.

After the reboot on 2026-10-13, the five-step cuDNN-enabled diagnostic passed
at `320x1024`, batch size 1, with the actual epoch-1/step-410 checkpoint.
It recorded PyTorch `2.13.0+cu129`, CUDA build `12.9`, cuDNN version `92000`,
and NVIDIA driver `575.64.03`. Peak PyTorch allocation was approximately
848 MiB, reserved memory 1586 MiB, and device-wide free memory during completed
steps approximately 1294 MiB. All losses and gradients were finite. The
checkpoint SHA-256 was unchanged afterward, NVIDIA reported recovery action
`None`, and no Xid entries were present in the current boot's kernel journal.
The log is `results/diagnostics/camera_depth_gpu_20261013T163247Z_16672.jsonl`.
This short synthetic success does not explain the earlier long-run failures;
the next diagnostic should increase the step count without changing backend,
input dimensions, or batch size, before returning to KITTI training.

### Notebook memory between completed runs

An idle notebook can retain unused CUDA allocator cache after a successful run.
The original preflight compared NVIDIA free memory against the safety threshold
before releasing that cache, potentially blocking a sequential run even though
PyTorch could reuse the reserved blocks. The notebook now opts into
`run_gpu_preflight(release_cached_memory=True)`: after driver/process checks,
it collects unreachable objects, releases unused cache, and refreshes the NVIDIA
memory measurement. The report separates live PyTorch allocations, reserved
memory, and the cache released. The 2500 MiB threshold remains unchanged.

This is not a GPU reset or a fix for Xid failures. Referenced tensors and models
remain allocated; unexpectedly high active memory can require deleting their
owning references or restarting the kernel. Cleanup is skipped for an unhealthy
driver or an uninitialized CUDA context and must not run concurrently with
training in the same process. Regression tests cover remeasurement after cache
release, retained active allocations, reset/uninitialized states, and stopping
after a cleanup failure without attempting another CUDA probe.

The notebook user confirmed that cache cleanup restored preflight readiness.
The resumed random-point run subsequently completed epochs 3 and 4 and saved
the epoch-4/step-1640 checkpoint. Validation MAE/RMSE were `0.637557 m` /
`1.737027 m` on 386015 held-out pixels. Epochs 2 through 4 have persisted metric
rows; epoch-1 metrics remain unavailable from the original interrupted run.
At that point, the pilot still needed its fifth epoch and the block-holdout run;
both have since completed, as summarized in the current results above.
These successful runs do not establish the cause of the earlier Xid failures.

### Recurrent failure while attempting epoch 5

On 2026-10-13 at 22:56:44 local time, the epoch-5 resume attempt failed during
backward with `CUDA error: unspecified launch failure`. Kernel logs again
recorded Xid 62, Xid 45 cleanup events for the Python process, and Xid 154
requesting a GPU reset. NVIDIA SMI confirmed recovery action `Reset`.
Cache cleanup and the earlier successful short probes did not resolve this
recurring failure; the exact cause remains unknown.

At the time of this failure, the latest checkpoint was reloaded on CPU and
contained finite weights at epoch 4, step 1640. A separate identical copy was
retained as `checkpoints/epoch_004_step_01640_before_driver_diagnostics.pt`;
the checkpoint and its copy then had
SHA-256 `4a127b740df41f6f64ff26c038ae075a1117b7e25c1866d3299342965fd2c466`.
No completed fifth epoch had been recorded at that point.

The loaded open kernel driver and GSP firmware are `575.64.03`, with kernel
`6.14.0-33-generic`, PyTorch `2.13.0+cu129`, CUDA build `12.9`, and cuDNN `92000`.
Local `ubuntu-drivers devices` recommends `nvidia-driver-615-open`; the current
APT package index offers `615.71.09-0ubuntu0.24.04.2`. This is a driver-update
candidate, not an established fix. Before another long GPU run, retain an
NVIDIA bug report from the failed state and review an Ubuntu-managed driver
update with the user. No driver, firmware, power setting, or dependency was
changed during this diagnosis. After recovery, compare controlled backend
probes and real KITTI training; do not interpret a passing short synthetic run
as proof of long-run stability or assume that the GPU hardware is defective.

### Driver update and completed runs

The driver and GSP firmware were subsequently updated through Ubuntu packages
to `615.71.09`, without changing PyTorch or the training architecture. After
reboot, a 100-step synthetic checkpoint-resume probe completed in approximately
34 seconds; NVIDIA reported recovery action `None` and no Xid entries were
observed in that boot's journal at the check. The random-point run then reached
epoch 5 / step 2050, and the separate block-holdout run completed all five
epochs. The four cross-evaluation combinations also completed successfully.

These runs provide evidence of successful execution after the update, not proof
that the driver was the sole cause or that future failures are impossible. AMP
remains disabled and the active notebook batch size remains 1. The original
random run's epoch-1 metrics were lost before per-epoch sidecar persistence was
implemented; its model/optimizer checkpoint survived and epochs 2 through 5
have saved metrics. The block run has metrics for all five epochs.

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
