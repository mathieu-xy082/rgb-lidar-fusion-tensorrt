# Workstream - LiDAR BEV encoder

Branch: `ec/lidar-bev-encoder`

Parent branch: `main` (camera-depth BEV lift / release 1.2.0).

Updated: 2026-11-03.

Status: scopes 1 to 4 implemented and initially validated: raw XYZI grouping, masked
PFN, stride-1 scatter, and residual dilated pre-fusion context, preserving
measurement evidence. A small multi-resolution reference was compared on CPU.
Joint camera/LiDAR inspection at 20 cm and a bounded LiDAR-only GPU diagnostic
are recorded below. Developer review of the artifacts remains separate from
this technical validation; no deployment/stability claim is made.
No fusion, detector, or LiDAR training integration is implemented.

References:

- [LiDAR encoder design and decisions](../bev-lidar-encoder.md);
- [target architecture](../bev-model-roadmap.md);
- [existing camera lift contract](camera-depth-bev-lift.md#grid-and-aggregation-contract);
- [project milestones](../milestones.md).

## Scope

1. Define raw XYZI input, pillar construction, and a shared metric grid.
2. Add a PFN and scatter producing LiDAR BEV features and measurement evidence.
3. Add pre-fusion spatial context targeting approximately 3.8 x 3.8 m.
4. Inspect camera/LiDAR alignment and measure the encoder's resource use.

Keep fusion, the shared BEV backbone, detection heads/losses, full detector
training, and ONNX/TensorRT export outside this first encoder scope. Preserve
camera-depth training, input/checkpoint contracts, and previous inspection
results at 0.5 m; the new shared-grid inspection will use 0.2 m explicitly.

## Progress

### Preparation - design decisions

- [x] Select 0.20 x 0.20 m pillars without vertical subdivision.
- [x] Reuse Velodyne axes and `BEVGridConfig` row-x/column-y conventions.
- [x] Target contextualized LiDAR features before camera fusion.
- [x] Record a 19-cell receptive-field candidate and its limitations.
- [x] Distinguish agreed choices, proposed parameters, and unimplemented work.

These preparation checks record design discussion only. Validation for scopes
1 to 3 is recorded below; a theoretical field is not object-recognition evidence.

### 1. Scope: raw XYZI input, pillars, and shared grid

- [x] Define the batched raw-point input contract, including empty samples.
- [x] Reuse explicit metric bounds with 0.20 m XY resolution; start with the
  proposed x=[0,50), y=[-25,25), z=[-3,3) ROI without a camera-FOV filter.
- [x] Build pillar indices, retained-point masks, counts, and nine point features.
- [x] Specify reproducible overflow selection; start by testing the proposed
  maximum of 64 retained points per pillar and report discarded measurements.
- [x] Test finite-input policy, half-open bounds, cell centers, non-square
  grids, batch separation, overflow, and absence of input mutation.

Exit criterion: independently inspectable, deterministic pillar construction
matching the existing camera-grid geometry, without a neural network yet.

Implemented in `rgb_lidar_fusion.lidar_pillars.build_lidar_pillars`. Inputs are
a sequence of variable-length float32/float64 NumPy `[Ni, 4]` arrays. Output
`PillarBatch` contains float32 `features[P, M, 9]`, int64 `indices[P, 3]`
ordered as batch/row-x/column-y, boolean `point_mask[P, M]`, original
`point_indices[P, M]` with -1 padding, and pre-cap
`point_count[B, 1, Hbev, Wbev]`. Here P counts occupied pillars across the batch,
and M is the configurable cap, default 64. `retained_count`, `dropped_count`,
and `observed_mask` are derived properties. Empty samples and empty batches
preserve explicit shapes; no total-pillar cap is applied.

XYZ is metric and intensity is copied, without clipping or normalization.
Non-finite XYZI and values not representable as float32 are rejected before
ROI filtering. Finite points outside the ROI are ignored, without camera-FOV
filtering. Float32 floor/index guards match the camera lift, including exclusion
of near-upper-bound indices rounded outside the grid; no edge clamping is used.

Overflow uses sampling without replacement from a local generator keyed by
seed/batch/row/column, default seed 13, without consuming global RNG state.
Repeating input order and seed reproduces the selected source indices; changing
point order or batch position may change selection. Other pillars' overflow
does not affect a cell's selection. Centroids use retained points only; all
padded features remain zero, with a separate false mask. This CPU NumPy stage
does not provide autograd or connect raw points to the existing training runner.

Validation on 2026-11-03: the complete CPU suite passed with 415 tests and one
CUDA-only skip. Tests include direct equality with camera-lift cell counts
on a non-square 20 cm grid at float32 boundaries, loader compatibility,
source-index traceability, and no input mutation/global RNG consumption.

A read-only smoke check on real KITTI frame `000000`, with x=[0,50),
y=[-25,25), z=[-3,3), resolution 0.20 m, produced 6,246 occupied pillars from
115,384 raw points. Accepted points: 62,984; retained: 55,465; overflow: 7,519.
Shapes were `[6246, 64, 9]` and `[1, 1, 250, 250]`; features were finite and the
source cloud unchanged. This checks preprocessing, not learned features,
camera-BEV inspection at 20 cm, recognition quality, or GPU performance.

### 2. Scope: PFN, scatter, and measurement evidence

- [x] Confirm the PFN width and normalization implementation; use the standard
  Linear/BatchNorm/ReLU/channel-wise-max pattern as the proposed reference.
- [x] Exclude padding from pooling and prevent padding from biasing normalization.
- [x] Scatter learned vectors while retaining pre-cap `point_count` and
  `observed_mask` separately from contextualized activations.
- [x] Test point-order invariance without overflow, padding, single-point
  pillars, empty samples/batches, expected scatter cells, and finite backward.

Exit criterion: finite `[B, C2, Hbev, Wbev]` features plus explicit measurement
evidence, with gradients to PFN parameters and retained point features. Discrete
pillar assignment does not promise gradients through cell selection.

Implemented in `rgb_lidar_fusion.lidar_bev` as two explicit components:

- `PillarFeatureNet(features, point_mask)` returns `[P, C]`, with C=32 by
  default and configurable positive width. It applies bias-free Linear,
  BatchNorm1d (`eps=1e-3`, `momentum=0.01`), and ReLU to **valid points only**,
  then a masked per-pillar/channel max. Empty pillars return zero vectors.
- `scatter_pillar_features(pillar_features, indices=..., point_count=...)`
  returns `LidarBEVFeatures`: `[B, C, Hbev, Wbev]` features, a cloned pre-cap
  count snapshot, and a derived observed mask. It preserves row-x/column-y and
  stride 1. All positive-count cells must have exactly one pillar; duplicate,
  missing, out-of-grid, and empty-cell indices are rejected.

Inputs are explicit Torch tensors already placed on the module's dtype/device;
there is no hidden NumPy conversion, device move, or dataset/runner integration.
The PFN supports float32/float64. Non-finite selected features are rejected;
masked padding may contain arbitrary values without entering Linear or BN.

Training batches with at least two valid points update BN statistics across
those points, even when each pillar contains only one point. With zero or one
valid point in the entire call, training uses existing running statistics
without updating them or the BN batch counter; affine/input gradients remain
connected. Initial statistics are mean=0/variance=1, not calibrated statistics.
This explicit degenerate-input policy is not an alternative normalization study.

Validation on 2026-11-03: 57 new PFN/scatter tests passed; the complete suite
passed with 472 tests and one CUDA-only skip. Checks include a direct reference
for Linear/BN/ReLU/max, unchanged BN statistics under extra/altered padding,
point/pillar permutations, singleton and empty cases, malformed contracts,
double-precision gradchecks, non-square placement, and finite parameter/input
backward with zero padding gradients. Existing camera tests remain unchanged.

A CPU forward/backward on real frame `000000` produced `[6246, 32]` pillar
vectors and `[1, 32, 250, 250]` BEV features. All 6,246 observed cells and 62,984
pre-cap measurements were preserved; the raw cloud was unchanged. The PFN was
randomly initialized, with no optimizer step or checkpoint. This is not object
recognition evidence, a 20 cm camera/LiDAR visual inspection, or a GPU benchmark.
Spatial context still belongs to scope 3; this scatter does not spread features
into neighboring cells.

### 3. Scope: pre-fusion spatial context

- [x] Specify width, normalization, activation, and any residual connections.
- [x] Evaluate the four-convolution candidate with dilations `(1, 2, 2, 4)`,
  stride 1, matching padding, and a 19 x 19 cell theoretical receptive field.
- [x] Verify preserved grid alignment and receptive-field support on a
  controlled convolution test, independently of random trained activations.
- [x] Compare with a small multi-resolution alternative before declaring the
  context architecture validated; document any output-stride alignment.
- [x] Test finite forward/backward and distinguish propagated features from
  direct observations in empty cells.

Exit criterion: a documented context choice with checked spatial dependencies,
shape/alignment, and resource trade-offs, without claiming learned detections.

Implemented as `lidar_bev.LidarBEVContext`, default width 32, with two residual
blocks. Each block applies Conv/ReLU/Conv, adds its input, then applies ReLU.
The dilation pairs are `(1, 2)` and `(2, 4)`; all convolutions use 3x3 kernels,
stride 1, padding equal to dilation, and no bias. No spatial normalization is
added, preserving strict local dependencies and a zero response to empty input.
The existing PFN BatchNorm is unchanged. This is a first functional reference,
not a detector-quality or state-of-the-art architecture selection.

`context(scattered_bev)` accepts/returns `LidarBEVFeatures`, preserving shape,
axes, cell centers, and cloned pre-cap measurement counts. Features may spread
into cells whose observed mask stays false; no measurement mask is reapplied
to erase context. Empty samples and empty batches have finite zero backward.
No new device move, trainable placement, or change to PFN/scatter keys is added.

Controlled positive-weight tests verify an exact, contiguous 19x19 input-cell
support for an interior output, in train and eval modes. The support is relative
to the scattered feature grid; 20 cm cells give 3.8 m. Impulse tests check its
center and ROI clipping without wraparound; residual identity tests and odd,
even, non-square shapes check that shape preservation does not hide a shift.

Validation on 2026-11-03: 45 new context/comparison tests passed; the complete
suite passed with 517 tests and one CUDA-only skip. Tests include numerical
gradients, unchanged evidence, finite parameter/input backward through the
complete PFN/scatter/context path, and zero padding gradients. Camera tests,
depth checkpoints, and notebooks were not modified.

On real KITTI `000000`, CPU forward/backward with random PFN/context weights
returned `[1,32,250,250]`, preserving 6,246 observed cells and 62,984 pre-cap
points. Features became non-zero in 21,069 unobserved cells; these are contextual
activations, not new measurements or detected objects. The raw cloud was
unchanged. No optimizer step, checkpoint, learned metric, or GPU run was produced.

#### CPU comparison and first context choice

Reproduce with:

```bash
pdm run python scripts/compare_lidar_bev_context.py --iterations 7 --warmup 2
```

The comparison-only alternative uses a fine 3x3 Conv/ReLU, a stride-2 3x3
Conv/ReLU, three coarse 3x3 Conv/ReLU layers, then a stride-2 3x3 transposed
convolution plus the fine branch and ReLU. Width remains 32, with no bias or
normalization. The downsampled index-zero center is preserved; explicit
transposed-convolution `output_size` restores even/odd/non-square dimensions,
without a fractional resize. Its field is 17 cells at even output indices and
19 at odd indices, independently per axis, as checked by controlled gradients.
It is an experimental reference in the script, not a second production API.

Local CPU control, FP32 `[1,32,250,250]`, one Torch thread, two warmups,
median of seven iterations, random weights, context only:

| Block | Parameters | Forward (ms) | Forward + backward (ms) |
| --- | ---: | ---: | ---: |
| Dilated residual | 36,864 | 73.8 | 289.2 |
| Multi-resolution reference | 55,296 | 38.0 | 106.7 |

The JSON report is ignored under
`results/analysis/lidar_bev/context_comparison_cpu.json`. Both variants use the
same input validation/evidence-copying path. Timing excludes raw grouping,
PFN, detection, optimizer updates, and GPU work; reported parameter storage is
not peak activation/workspace memory. CPU measurements vary with the host/load.

The dilated block is retained as the first functional reference for its uniform
19-cell context, direct stride-1 grid alignment, and fewer parameters. The
multi-resolution alternative is faster in this CPU control; the dilated choice
is **not** a demonstrated latency/accuracy optimum. Detection supervision and
GPU measurements are needed before selecting a deployment architecture.

### 4. Scope: inspection, compatibility, and resources

- [x] Recompute camera BEV directly at 0.20 m from image features and depth support.
- [x] Inspect synthetic and real KITTI alignment on identical grid bounds,
  including raw LiDAR outside camera coverage.
- [x] Keep 0.20 m artifacts separate from the historical 0.50 m inspections.
- [x] Run focused tests and existing camera-depth/lift regression checks.
- [x] Measure batch-1 GPU forward/backward latency and peak memory on the 4 GB
  device; report settings and results rather than promising real-time operation.

Exit criterion: reproducible inspection and validation report with no camera
regression, explicit grid/evidence contracts, and measured resource limits.

#### Joint inspection

```bash
pdm run python scripts/inspect_camera_bev.py \
  --checkpoint results/training/kitti_camera_depth_block_dropout_512/checkpoints/latest.pt \
  --source splatted --weight-by-splat-confidence --with-lidar --resolution 0.2
```

The existing camera-only mode keeps its 0.5 m default, four-panel figure and
artifact schema. The opt-in joint mode writes under
`results/analysis/camera_lidar_bev/res_0.2m/` by default, with a resolution/joint
suffix even for explicit output directories. It recomputes camera BEV directly
from full-resolution features, input depth/support, and calibration; it does
not resize the historical 100x100 BEV. Existing 0.5 m artifacts are untouched.

The nine panels show RGB, native ROI LiDAR, camera-FOV LiDAR, camera pixel
counts, pre-cap LiDAR counts, support overlap, and separate camera/PFN/context
feature RMS. Both feature grids share axes, bounds, origin and 20 cm cells.
LiDAR weights are explicitly random (seed 13); RMS and overlap are not object
scores or accuracy metrics. CPU initialization is isolated from global RNG
state and does not seed CUDA. No holdout or learned-depth placement is used.
The NPZ adds both LiDAR feature maps, evidence, source pillar indices, retained/
dropped counts, and a categorical support map to the unchanged camera fields.

On real `000000`, the epoch-5/step-2050 camera checkpoint gave 191,636 accepted
splatted pixels in 2,579 cells. Native LiDAR retained all 62,984 ROI measurements
in 6,246 cells before the per-pillar cap. Shared support: 2,408 cells; camera
only: 171; LiDAR only: 3,838. Camera-only cells can arise from splatting and
rounded projection/backprojection; they are not new raw measurements or proof
of a calibration defect. Different support does not imply misalignment.

The generated figure was inspected for axis orientation, scene placement and
the expected larger native LiDAR coverage. No gross shift/inversion was seen.
The context panel spreads activations without changing raw evidence. This is
one geometric prototype inspection, not semantic validation or developer signoff.
The trusted camera checkpoint SHA-256 remains
`3561742ce6de01bf71e6ae290b8e5f707011cb16434e06f9bf3cee59d7d13ece`.

#### Bounded GPU diagnostic and memory guard

```bash
timeout 90s pdm run python scripts/probe_lidar_bev.py \
  --velodyne data/kitti/training/velodyne/000000.bin \
  --device cuda --warmup 1 --iterations 1
```

Run in a fresh process. Incremental JSONL logs under `results/diagnostics/`
record preflight, grouping, transfer, synchronized forward/backward, validation,
and memory peaks. Existing log paths are refused. CPU mode tests the tool; it
does not validate GPU health. The default is one warmup and three measured
iterations per mode; the first hardware control above deliberately used one.
No optimizer, checkpoint write, or automatic retry is performed. On failure,
the tool does not synchronize again, empty the cache, or query CUDA counters.

**The LiDAR-only threshold is 2,048 MiB (2 GiB)**, configurable through
`--minimum-free-memory-mib`. This is a diagnostic starting margin, not an
established requirement for every scene. The shared/camera preflight default
remains **2,500 MiB**; recovery-action and concurrent-compute-process checks are
unchanged. A lower memory threshold does not repair or prevent driver failures.

Hardware control on 2026-11-03: RTX 500 Ada Generation Laptop, driver
615.71.09, Torch 2.13.0+cu129, FP32 batch 1, 6,246 pillars/55,465 retained points.
Preflight reported ready, 2,270 MiB free, no recovery action and no competing
CUDA compute process. Both modes completed after one warmup and one measured
iteration each:

| LiDAR encoder mode | Measured wall time | Peak PyTorch allocation |
| --- | ---: | ---: |
| Eval forward | 11.0 ms | 122.9 MiB |
| Train forward + backward | 35.3 ms | 180.5 MiB |

CPU grouping took 95.7 ms separately; initial model setup/transfer took 150.8 ms.
The GPU wall times exclude these, camera processing, I/O, optimizer state,
fusion, heads and export. Peaks are PyTorch allocations, not all driver/desktop
VRAM. This is a single indicative sample with synchronized diagnostics, not a
p50/p95 benchmark, an end-to-end FPS result, or a long-run driver stability test.
In particular, raw CPU grouping remains a latency bottleneck to investigate.
Log: `results/diagnostics/lidar_bev_20261103T152254Z_182519.jsonl` (ignored).

Local regression validation: 528 tests passed, one CUDA-only test skipped in
the sandboxed suite. The real GPU diagnostic was run separately with hardware
access. Tests cover joint sparse/splatted views, raw points outside camera FOV,
checkpoint/input preservation, RNG isolation, incremental logs, no overwrite,
preflight refusal before setup, and failure logging without retry. No notebook
execution state was modified. Broader scenes, developer visual review, trained
detection quality and deployment benchmarking remain follow-ups.

## Deferred validation

An encoder with finite gradients is not a trained detector. Autonomous LiDAR
object recognition requires a later detection head, 3D supervision, and a
LiDAR-only evaluation against the fused detector. The shared BEV backbone must
still supply wider context; 3.8 m does not cover every car or orientation.
