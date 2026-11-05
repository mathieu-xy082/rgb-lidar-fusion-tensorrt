# Workstream - BEV fusion backbone

Branch: `ec/bev-fusion-backbone`

Parent branch: `main` after release 1.3.0.

Updated: 2026-11-05.

Status: increments 1 to 4 are complete: contract, feature fusion, shared
three-scale backbone, assembled center diagnostic and checkpoint helpers are
implemented and CPU-tested. The multimodal training runner/notebook and the
four-frame/512-frame experiments are **not implemented/run yet**. Existing
standalone encoders, checkpoints and runners remain unchanged.

References:

- [target architecture](../bev-model-roadmap.md);
- [camera lift and geometry](camera-depth-bev-lift.md);
- [LiDAR encoder design](../bev-lidar-encoder.md);
- [closed LiDAR center pilot](lidar-center-diagnostic.md#6-pilot-results-and-prototype-closure);
- [project milestones](../milestones.md).

## Scope

1. Formalize the aligned BEV inputs, support/evidence and output contract.
2. Implement feature fusion independently of both encoders and detection.
3. Add a lightweight shared multi-scale BEV backbone.
4. Assemble the multimodal model with explicit preprocessing/checkpoint policies.
5. Inspect representations and validate four-frame memorization in a notebook.
6. Run a disjoint 512-frame pilot and compare against a matched LiDAR-only baseline.

The optional Car-center/XY-offset head supplies an existing diagnostic loss;
it is not the final 3D detector. Size, height, yaw, velocity, additional classes,
auxiliary YOLO supervision, temporal fusion, ONNX/TensorRT export and a
real-time deployment claim stay outside this increment.

## Fusion contract

This section specifies the first implementation. `BEVFeatureFusion` implements
the cell-local part; `SharedBEVBackbone` adds spatial context and
`BEVFusionDiagnostic` assembles both encoders. Existing `CameraBEVFeatures`, `LidarBEVFeatures`,
`BEVGridConfig` and encoder APIs remain the integration points.

### Geometry and sample identity

Both branches must represent the **same frame**, in the same batch order,
Velodyne frame and metric grid. KITTI axes are x forward, y left, z up;
this is not an implicit ROS base_link, world or gravity-aligned frame.

The initial grid is:

```text
x_range_m = [0, 50)
y_range_m = [-25, 25)
z_range_m = [-3, 3)
resolution_m = 0.20
(Hbev, Wbev) = (250, 250)
```

Rows increase along x, columns along y. Cell `(row, col)` has center:

```text
x = x_min + (row + 0.5) * resolution
y = y_min + (col + 0.5) * resolution
```

Preserve half-open bounds, height filtering relative to the LiDAR origin and
the existing floating-point border guards. The camera lift must be recomputed
on this grid, not resized from a previous 0.5 m BEV image.

Existing camera/LiDAR BEV feature containers do not store their grid. The fusion
boundary therefore receives explicit camera and LiDAR `BEVGridConfig`
values and rejects differences in XYZ bounds or resolution, even for equal
tensor shapes. The assembly layer must associate them with the actual
preprocessing and sample IDs; tensor shapes alone cannot prove calibration,
frame identity or synchronization. No silent transpose, crop or interpolation.

### Encoder inputs and lift policy

The image-space contract stays `rgb[B,3,H,W]` plus
`lidar_maps[B,8,H,W]`, concatenated into 11 channels by the camera backbone.
The eight LiDAR channels remain ordered as:

| Index | Content |
| --- | --- |
| 0 | Sparse camera-z depth divided by `max_depth_m` |
| 1, 2, 3 | Sparse vehicle x, y, z divided by the same scale |
| 4 | Sparse LiDAR intensity |
| 5 | Sparse point-presence mask |
| 6 | Splatted camera-z depth, with the same normalization |
| 7 | Heuristic splat confidence |

Splatting propagates **depth only**, with a local confidence score; XYZ,
intensity and the original point mask remain sparse. It does not guarantee
depth at every pixel. The scale is explicit and must match the camera
checkpoint, currently 80 m in the pilots; it is not the BEV ROI extent.

For the first assembled model, choose `source="splatted"` in
`prepare_camera_depth_support` and `weight_by_splat_confidence=True` in
`lift_camera_features_to_bev`. Start with `min_splat_confidence=0.0` (the
existing policy still requires finite confidence strictly greater than zero),
and `0 < camera_z <= max_depth_m`, followed by the common XYZ grid filter.
Record these settings in the future run/checkpoint configuration. Full resized
P2 and `T_vehicle_to_camera` remain explicit geometry inputs, not CNN channels.

The LiDAR branch independently receives the same frame's raw XYZI cloud,
without camera-FOV filtering, then its existing pillars/PFN/scatter/context.
Its default cap stays 64 retained points per pillar and its feature width 32.
The two feature paths share LiDAR information; they are not statistically
independent sensors. Camera features currently learned depth, not explicit
object semantics. A fusion benefit must be measured, not assumed.

Dense predicted depth is a later alternative, not an automatic fallback.
It requires an explicit support policy and does not inherit splat confidence
as learned uncertainty. No held-out targets or annotations enter the lift.

### BEV inputs and evidence

For a non-empty batch, use `B >= 1` and a shared positive spatial shape:

| Input | Shape / type | Meaning |
| --- | --- | --- |
| `camera_bev.features` | `[B,Ccam,Hbev,Wbev]`, float | Aggregated camera decoder features; default Ccam=32 |
| `camera_bev.pixel_count` | `[B,1,Hbev,Wbev]`, int64 | Number of accepted image pixels per cell |
| `camera_bev.weight_sum` | `[B,1,Hbev,Wbev]`, float | Sum of accepted splat weights in the selected policy |
| `lidar_bev.features` | `[B,Clidar,Hbev,Wbev]`, float | Features **after** `LidarBEVContext`; default Clidar=32 |
| `lidar_bev.point_count` | `[B,1,Hbev,Wbev]`, int64 | Raw accepted points **before** the per-pillar cap |

The fusion boundary validates matching batch/spatial sizes, explicit grid
equality, configured channel widths, finite features, non-negative finite
evidence and consistent camera count/mass support. All tensors must already
share the module's device; features and floating evidence share its
float32/float64 dtype. No implicit transfer or dtype conversion, except casting
the derived boolean masks for concatenation. Start training with batch 1 FP32;
float64 is for numerical tests, not the proposed GPU training configuration.

Derive two distinct boolean maps:

```text
camera_support = camera_bev.pixel_count > 0
lidar_observed_mask = lidar_bev.point_count > 0
```

Camera support describes accepted projected/splatted image contributions, not
independent LiDAR measurements or a camera-FOV/free-space mask. `weight_sum` is
an aggregation mass, not a calibrated BEV probability; many splatted pixels
may originate from one raw return. LiDAR observation means direct raw support,
not traversability, free space or object presence.

An unsupported camera cell has zero lifted features. LiDAR context can remain
non-zero in an unobserved cell: **do not multiply contextualized LiDAR features
by `lidar_observed_mask`**. Both masks remain evidence even when downstream
convolutions propagate features into other cells. Zero evidence never creates
a negative training label by itself.

Empty-support samples with a valid batch/grid are allowed; an entirely empty
batch is rejected at this new boundary without changing the existing LiDAR
encoder's empty-batch contract. Zero-valued branch bundles permit controlled
absence tests, not a claim of trained sensor-failure robustness. No NaN/Inf is
silently promoted to a valid feature, and neither input bundle is mutated.

### Concatenation and output

Keep a fixed concatenation order along the channel dimension:

```text
Fcat = concat(Fcamera, Flidar_context,
              float(camera_support), float(lidar_observed_mask))

default: [B, 32 + 32 + 1 + 1, Hbev, Wbev] = [B, 66, 250, 250]
Ffusion = Conv1x1(Fcat)  # default 66 -> 32, stride 1, bias=False
Fshared = spatial_backbone(Ffusion)  # default [B, 32, 250, 250]
```

The first projection is linear, with no normalization, activation, attention,
learned gate or confidence multiplier. It mixes channels **within each cell**;
the shared backbone adds spatial context. Its output is a learned feature
vector, not a set of class scores, heights or detections. Channel widths may
be configured explicitly; defaults give the 66-to-32 contract above.

The implemented `FusedBEVFeatures` output bundles features with the unchanged
grid, camera pixel counts/weight sums, LiDAR pre-cap counts and their separate
derived support masks. Preserve evidence snapshots independently of the learned
activation field; do not reduce both supports to one ambiguous "valid" flag
or erase fused/context features outside measured cells. The camera's
image-space `pixel_mask` can remain available for inspection but is not an
extra BEV channel. These classes were added in increment 2; increment 1 was
documentary only.

Gradients must reach both input feature maps and the trainable fusion/shared
parameters. When encoders are unfrozen, they may reach camera decoder/encoder
and PFN/context parameters. Existing floor/scatter placement does **not**
provide coordinate gradients to train depth or calibration; with splatted
depth and a feature-only objective, `depth_head` is not on the loss path.
Counts and boolean evidence are not learned uncertainty or gradient targets.

## Progress

### 1. Scope: formalize the fusion contract

- [x] Fix axes, half-open ROI, 20 cm grid, batch/sample alignment and grid checks.
- [x] Preserve RGB/enriched inputs and select splatted, confidence-weighted lift.
- [x] Distinguish camera support, LiDAR observations and contextual activations.
- [x] Specify concatenation, 1x1 projection, output evidence and gradient limits.
- [x] Define incremental validation, compatibility and non-goals below.
- [x] Link this workstream from the README, documentation index and roadmap.

Exit criterion: a reviewable contract consistent with existing interfaces,
without model changes, checkpoint migration or a training run. Implementation
and behavioral tests are recorded with the corresponding increments below.

### 2. Scope: implement the fusion block

- [x] Add explicit grid-checked fusion and an output bundle preserving evidence.
- [x] Implement the ordered concatenation and configurable bias-free 1x1 projection.
- [x] Test channel order with controlled weights, no mutation, device/dtype and
  malformed-grid rejection (including equal shapes with different metric bounds).
- [x] Test camera-only, LiDAR-only, zero-support cells/samples, batch separation,
  and preservation of LiDAR context outside direct observations.
- [x] Test finite non-zero gradients to both feature branches and projection.

Exit criterion: a separately testable fusion module; no encoder or detector needed.

Implemented in `rgb_lidar_fusion.bev_fusion`:

```python
from rgb_lidar_fusion.bev_fusion import BEVFeatureFusion

fusion = BEVFeatureFusion()  # camera=32, LiDAR=32, output=32
fused = fusion(
    camera_bev,
    lidar_context_bev,
    camera_grid=camera_grid,
    lidar_grid=lidar_grid,
)
# fused.features: [B, 32, Hbev, Wbev]
# fused.camera_support / fused.lidar_observed_mask: bool [B, 1, Hbev, Wbev]
```

The only learned layer is `projection`, a bias-free 1x1 convolution with 2,112
parameters at default widths. Count/mass maps are cloned into the output, while
feature gradients are retained. Inconsistent camera count/mass support and
non-zero camera features outside its support are rejected, not silently repaired.
Tiny positive aggregation masses remain accepted; their magnitude is not an
additional multiplier at fusion. LiDAR context is kept even where **both**
support maps are false. The image-space camera `pixel_mask` is not consumed.

CPU validation: 81 fusion tests cover the contract, signed linear outputs,
odd/non-square/full-size grid shapes, count/mass snapshots, branch-absence
controls, double-precision gradcheck and malformed inputs. A synthetic weighted
lift plus the existing PFN/scatter/context connects to the new fusion with finite
non-zero gradients to image features, point features and encoder/fusion weights.
This is not the assembled multimodal model, an optimizer run or GPU validation.
The full regression suite passed with 714 tests and one CUDA-only skip; 81
links/anchors were checked across the seven Markdown files updated in this
increment. Existing notebook execution state and checkpoint schemas are unchanged.

### 3. Scope: implement shared spatial context

- [x] Specify a small three-scale residual backbone at widths 32/64/96,
  as a first functional reference, not a demonstrated latency/accuracy optimum.
- [x] Define down/up sampling, cell-center alignment, normalization and padding
  before implementation; restoring tensor shape alone does not prove alignment.
- [x] Preserve stride-1 output on odd/non-square grids; inspect controlled
  impulse/gradient support and account for ROI borders without wraparound.
- [x] Preserve evidence and test forward/backward, resource use and a wider
  theoretical context than the existing 19-cell LiDAR block.

Exit criterion: aligned shared features with tested spatial dependencies, not
a claim that larger receptive fields automatically improve object detection.

Implemented as `rgb_lidar_fusion.bev_backbone.SharedBEVBackbone`:

```python
from rgb_lidar_fusion.bev_backbone import SharedBEVBackbone

backbone = SharedBEVBackbone(channels=(32, 64, 96))
shared = backbone(fused)  # FusedBEVFeatures in/out, not a bare tensor
# shared.features: [B, 32, Hbev, Wbev]
```

#### Architecture and alignment

| Stage | Operation | Default shape (B=1) |
| --- | --- | --- |
| Fine | Two-convolution residual block | `[1,32,250,250]` |
| Middle | Stride-2 convolution, ReLU, residual block | `[1,64,125,125]` |
| Deep | Stride-2 convolution, ReLU, residual block | `[1,96,63,63]` |
| Middle decoder | Transposed convolution + middle skip, ReLU | `[1,64,125,125]` |
| Fine decoder | Transposed convolution + fine skip, ReLU, residual block | `[1,32,250,250]` |

Each residual block is `ReLU(x + Conv3x3(ReLU(Conv3x3(x))))`. All 12
convolutions use 3x3 kernels, padding 1 and no bias. There is no normalization:
zero features stay zero, and normalization statistics do not add non-local
dependencies or batch coupling. Widths are configurable as three positive
integers; input/output width is the first value and must match fusion's output.

The stride-2 encoders retain the index-zero center. For fine-grid cell-center
coordinate `x0 = x_min + resolution/2`, internal indices are anchored at
`x0 + 2*i*resolution`, then `x0 + 4*i*resolution` (likewise y). These internal
lattices are not coarse `BEVGridConfig` instances with the same minimum bound:
that would introduce a half-cell origin change. Paired 3x3/stride-2/padding-1
transposed convolutions use the actual skip tensor's `output_size`, selecting
output padding 0 or 1 on the high-index side. No fractional interpolation,
cropping or metric-grid modification occurs; center-tap tests verify indices,
not just matching tensor dimensions.

The controlled gradient support relative to the **fused input grid** depends
on each output index modulo four:

| Index phase | Input offsets per axis | Cells | At 20 cm |
| --- | --- | --- | --- |
| 0 | `[-21,21]` | 43 | 8.6 m |
| 1 | `[-22,20]` | 43 | 8.6 m |
| 2 | `[-19,19]` | 39 | 7.8 m |
| 3 | `[-20,22]` | 43 | 8.6 m |

Rows/columns can have different phases; borders clip these dependencies.
This is wider than the LiDAR context's 19 cells but is not translation
invariance, the effective learned field, or a bound on upstream camera/raw-point
dependencies. Stride-phase sensitivity and learned transposed-convolution
artifacts remain quality risks to inspect after training.

Camera pixel counts/mass, LiDAR pre-cap counts and the metric grid stay at the
original resolution. Evidence tensors are cloned and are not pooled, inferred
from activation magnitude or reapplied as masks. Shared features can spread
into cells with neither camera contributions nor direct LiDAR observations.

#### CPU validation and resources

The 77 backbone/probe tests include default intermediate sizes, singleton and
odd/non-square grids, center-tap alignment, positive-weight receptive fields
in train/eval modes, ROI clipping, zero-input/skip paths, evidence snapshots,
invalid input rejection, double-precision gradcheck and fusion-to-backbone
gradients to both feature branches. No full encoder assembly is claimed.

Reproduce the bounded CPU-only resource control with:

```bash
pdm run python scripts/probe_bev_backbone.py
```

The local FP32 `[1,32,250,250]` control uses random weights, one CPU thread,
one warmup and median of three iterations. The backbone has **423,936
parameters**, occupying **1.617 MiB** in FP32; this is not peak activation or
training memory. A local run measured **142.97 ms forward** and **400.17 ms
forward/backward**, including validation/evidence copying and a scalar loss
for backward. Timing varies with host/load. The ignored JSON report is
`results/analysis/bev_fusion/backbone_cpu.json`.

No dataset, checkpoint, optimizer, encoders, lift, fusion or detection head
is involved in that probe; no GPU, end-to-end latency or peak-memory result
is claimed. New training is deferred until assembly and the guarded notebook.
The full regression suite passed with **791 tests and one CUDA-only skip**,
without exclusions. The seven updated Markdown files passed 81 link/anchor
checks. Existing encoder APIs, checkpoints and notebook execution state are unchanged.

### 4. Scope: assemble the multimodal model

- [x] Compose existing camera features/lift and LiDAR encoding with the new modules.
- [x] Preserve old standalone state dictionaries; explicitly load camera and
  PFN/context weights without requiring the old auxiliary head.
- [x] Audit checkpoint provenance, preprocessing, sample IDs and train/validation
  membership for **both** encoders before reusing pretrained weights, recording
  the legacy camera provenance limitation below rather than claiming proof.
- [x] Add the existing center head/loss/decoder as a diagnostic consumer, keeping
  KITTI labels and supervision masks out of feature construction.
- [x] Define a new multimodal checkpoint kind, configuration, resume/optimizer
  policy and a synthetic end-to-end backward test without changing old runners.

Exit criterion: RGB + enriched maps + calibration + raw cloud -> shared BEV ->
diagnostic centers, with explicit evidence and reproducible weight loading.

#### Model and preprocessing API

`bev_model.BEVFusionConfig` holds architecture/preprocessing settings. Its
reviewable default JSON is `configs/bev_fusion.json`: 320x1024, the common
20 cm grid, 80 m depth normalization, splat radius 2/sigma 1, 64 points per
pillar, seed 13, LiDAR width 32 and backbone widths 32/64/96. This is **not**
an executable training configuration: no epochs, paths, device or automatic run.

`bev_batch.prepare_bev_fusion_batch(items, raw_clouds, config=...)` reuses the
dataset/model adapters and raw pillar builder, producing CPU tensors plus
ordered sample IDs. Items must be resized/calibrated sparse dataset items;
raw XYZI arrays are looked up by the same ID, not an unchecked sequence zip.
Splatting is rebuilt from the configuration, with no holdout by default.
Targets/annotations are excluded from model kwargs. The adapter does not prove
that caller-provided arrays actually originate from the declared scan or
calibration; that remains the dataset producer's responsibility. Non-camera
LiDAR coverage is retained. Device transfer remains explicit.

```python
from rgb_lidar_fusion.bev_batch import prepare_bev_fusion_batch
from rgb_lidar_fusion.bev_model import BEVFusionConfig, BEVFusionDiagnostic

config = BEVFusionConfig()
batch = prepare_bev_fusion_batch(items, raw_clouds_by_id, config=config)
model = BEVFusionDiagnostic(config)
# Fresh random weights until the explicitly audited encoder warm start.
prediction = model(**batch.inputs)  # existing CenterPrediction
representations = model.encode(**batch.inputs)  # separate inspection pass
```

Forward uses `camera.forward_features`, splatted/weighted lift,
PFN/scatter/context, fusion, shared backbone and the existing `LidarCenterHead`.
`encode()` returns `BEVFusionRepresentations` with `camera_bev`, `lidar_bev`,
`fused_bev` and `shared_bev`, without evaluating the diagnostic head. Existing
`lidar_center_loss` and `decode_lidar_centers` consume the forward output; labels
and supervision masks remain outside the model. No camera depth prediction is
computed. Its old depth-head keys remain loadable, but its parameters are
excluded from optimization because the center loss does not traverse that head.

`freeze_encoders=True` is the default: camera/PFN/context parameters are frozen,
and `train()` keeps these modules in eval mode, including PFN BatchNorm buffers.
Fusion, shared backbone and center head remain trainable. An explicitly
unfrozen configuration permits backward through both encoders. With
`camera_enabled=False`, the camera computation is skipped and camera BEV
features/counts/mass are zero; the raw LiDAR branch and common architecture
remain intact for the later matched control. This is not sensor-failure recovery.

#### Weight provenance and checkpoint policy

`bev_checkpoint.initialize_bev_fusion_encoders(...)` is an FP32 CPU-only warm
start. Before changing any destination module, it validates preprocessing,
strict state keys/shapes/dtypes/finite values, the target split, all current
dataset hashes and absence of source training IDs in target validation. It
loads **only camera, PFN and context**; the old LiDAR center head need not be
present and neither old head nor Adam state is reused. Overfit-source LiDAR
checkpoints are rejected. Move the model to the desired device only afterwards.

For the legacy camera checkpoint, original sorted `camera_source_ids` must be
declared explicitly. Its config/seed/fraction reconstruct the split, but the
checkpoint did not save IDs or file hashes. A correct declared mapping is a
caller prerequisite; historical camera membership/content cannot be independently
proven. Do not fabricate a mapping from a seed alone. The audit report retains
this limitation along with source checkpoint SHA-256, paths, epochs and IDs.
LiDAR's saved fingerprints are checked against the current target manifest.
These checks do not establish scene independence or prevent model-selection bias.

Use `make_bev_fusion_optimizer(model)` **after device placement**. It creates
single-group Adam with `foreach=False` and only the ordered trainable parameters;
frozen encoders and the unused depth head are excluded. The first schedule is
therefore fusion/backbone/diagnostic-head training, not silent full fine-tuning.

`save_bev_fusion_checkpoint` writes an atomic, separate `bev_fusion_center_v1`
payload containing config, full model weights, split/hashes, counters, audit
report, metrics and trainable parameter names. Supplying Adam creates a
`purpose="resume"` checkpoint; omitting it creates `purpose="inference"`.
The future runner will manage latest/best selection; these helpers do not
select an epoch or implement a training loop.

`load_bev_fusion_checkpoint` returns a CPU/eval model and payload using
`weights_only=True`, without requiring the original encoder checkpoint files
or dataset for inference. `restore_bev_fusion_training` requires the new resume
format, identical model configuration, split and verified current files, plus
the matching optimizer created after model placement. It restores the saved
Adam settings/state rather than importing an old encoder optimizer. Changing
the freeze/ablation policy requires an explicit new training stage, not resume.
No CUDA selection, retry or allocation is performed automatically by these APIs.

#### Validation and local audit

The 59 assembly/checkpoint tests cover JSON settings, target-independent batch
preparation, empty support, raw LiDAR outside camera FOV, frozen encoder/BN
state, feature-only inspection, explicit LiDAR-only control, finite end-to-end
backward to both unfrozen encoders and all new blocks, and absence of depth-head
execution. Checkpoint tests cover strict warm starts without the old head,
pre-mutation refusals, leakage/hash/config failures, CPU inference round-trip
and an exact two-step CPU Adam save/resume control. These are synthetic unit
optimizations, not KITTI training or validation-quality results.

A read-only CPU audit loaded the existing local camera and LiDAR encoders:

| Source | Epoch / step | Train / validation IDs | Checkpoint SHA-256 |
| --- | --- | --- | --- |
| Camera block-holdout `latest.pt` | 5 / 2050 | 410 / 102, declared sorted-ID mapping | `3561742ce6de01bf71e6ae290b8e5f707011cb16434e06f9bf3cee59d7d13ece` |
| LiDAR center `best.pt` | 9 / 3690 | 410 / 102, saved IDs and file hashes | `eac8f0d39edce3b1b44706936cc8aaf2d88c2dc5d040245c93994455fb5daf5f` |

The current 512-frame manifest passed file-hash checks and both source train
sets were disjoint from target validation under the declared camera mapping.
This audit performed no real-frame forward, GPU execution, optimizer update,
checkpoint write or new detection-quality measurement. Legacy camera provenance
remains limited as stated above. The four-frame/notebook experiment is next.
Full regression: **850 tests passed, one CUDA-only skip**, no exclusions;
81 links/anchors checked across the seven updated Markdown files. Existing
notebooks, data, source checkpoint files and standalone APIs were not modified.

### 5. Scope: inspect and memorize four frames

- [ ] Add a notebook with training opt-in, CPU inspection and child-process
  GPU training/cleanup, reusing the existing notebook safety pattern.
- [ ] Inspect camera/LiDAR support, separate feature maps, fused features and targets.
- [ ] Freeze both encoders initially (`eval()` plus disabled parameter gradients,
  including fixed PFN BatchNorm statistics); train fusion/backbone/diagnostic head.
- [ ] Validate four-frame memorization, finite metrics and peak GPU memory at
  batch 1; do not lower existing camera/GPU guards to make a run start.
- [ ] If needed, change the training schedule explicitly and record it rather
  than silently unfreezing encoders; no automatic retry after a CUDA failure.

Exit criterion: an inspectable small optimization run and a measured resource
budget, not validation quality. Discard these fitted weights before the pilot.

### 6. Scope: disjoint pilot and comparison

- [ ] Reuse the saved 410/102 frame IDs; validate dataset/checkpoint hashes and
  ensure no supervised or depth pretraining used the pilot validation frames.
- [ ] Use complete available inputs for the center diagnostic by default; depth
  holdout is not required by this loss. Keep any corruption experiment explicit.
- [ ] Compare matched LiDAR-only versus fused runs with the same shared-backbone
  and head design, initial shared/head weights, split, schedule and evaluation.
  Zero the camera feature/support block for the baseline, not the raw LiDAR.
- [ ] Report the baseline separately from the older LiDAR-only pilot, whose
  architecture has no shared backbone; it does not isolate fusion's contribution.
- [ ] Reuse Car annotations, ignored-domain policy, score/top-k and XY matching;
  report losses, TP/FP/FN, precision/recall and distance over matched centers only.
- [ ] Save latest/best, inspect validation successes and failures, record GPU
  memory/timing scope and optionally evaluate a gradual encoder-unfreezing stage.

Exit criterion: a reproducible train/validation workflow and an honest
comparison, including a neutral/negative result if fusion does not help. The
random frame split is not a scene-independent test set; selected validation
scores are not unbiased test results or full 3D AP. If actual LiDAR returns are
removed in a later corruption test, remove them **before both** pillar building
and image projection/splatting to avoid leaking the same returns through the
other branch. A deliberate branch-only ablation is a different experiment.

## Validation record

2026-11-05, increment 1: existing feature containers, geometry/support policy,
encoder composition and diagnostic APIs were read to establish this contract.
No fusion implementation, training, GPU execution or new model-quality result
is claimed. Documentary validation: 95 links/anchors checked across the nine
changed Markdown files; regression suite: 633 passed, one CUDA-only test skipped,
no exclusions. Implementation evidence will be added increment by increment.
