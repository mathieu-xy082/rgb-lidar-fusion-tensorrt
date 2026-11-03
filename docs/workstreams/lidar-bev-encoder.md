# Workstream - LiDAR BEV encoder

Branch: `ec/lidar-bev-encoder`

Parent branch: `main` (camera-depth BEV lift / release 1.2.0).

Updated: 2026-11-03.

Status: scope 1 implemented and CPU-tested: raw XYZI grouping, nine point
features, measurement counts, and reproducible overflow selection. No PFN,
context block, fusion, or detector is implemented. The pre-fusion context
objective is agreed; neural architecture parameters remain proposed choices.

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

These preparation checks record design discussion only. Scope 1 validation is
recorded below; a 3.8 m theoretical receptive field is not object-recognition evidence.

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

- [ ] Confirm the PFN width and normalization implementation; use the standard
  Linear/BatchNorm/ReLU/channel-wise-max pattern as the proposed reference.
- [ ] Exclude padding from pooling and prevent padding from biasing normalization.
- [ ] Scatter learned vectors while retaining pre-cap `point_count` and
  `observed_mask` separately from contextualized activations.
- [ ] Test point-order invariance without overflow, padding, single-point
  pillars, empty samples/batches, expected scatter cells, and finite backward.

Exit criterion: finite `[B, C2, Hbev, Wbev]` features plus explicit measurement
evidence, with gradients to PFN parameters and retained point features. Discrete
pillar assignment does not promise gradients through cell selection.

### 3. Scope: pre-fusion spatial context

- [ ] Specify width, normalization, activation, and any residual connections.
- [ ] Evaluate the four-convolution candidate with dilations `(1, 2, 2, 4)`,
  stride 1, matching padding, and a 19 x 19 cell theoretical receptive field.
- [ ] Verify preserved grid alignment and receptive-field support on a
  controlled convolution test, independently of random trained activations.
- [ ] Compare with a small multi-resolution alternative before declaring the
  context architecture validated; document any output-stride alignment.
- [ ] Test finite forward/backward and distinguish propagated features from
  direct observations in empty cells.

Exit criterion: a documented context choice with checked spatial dependencies,
shape/alignment, and resource trade-offs, without claiming learned detections.

### 4. Scope: inspection, compatibility, and resources

- [ ] Recompute camera BEV directly at 0.20 m from image features and depth support.
- [ ] Inspect synthetic and real KITTI alignment on identical grid bounds,
  including raw LiDAR outside camera coverage.
- [ ] Keep 0.20 m artifacts separate from the historical 0.50 m inspections.
- [ ] Run focused tests and existing camera-depth/lift regression checks.
- [ ] Measure batch-1 GPU forward/backward latency and peak memory on the 4 GB
  device; report settings and results rather than promising real-time operation.

Exit criterion: reproducible inspection and validation report with no camera
regression, explicit grid/evidence contracts, and measured resource limits.

## Deferred validation

An encoder with finite gradients is not a trained detector. Autonomous LiDAR
object recognition requires a later detection head, 3D supervision, and a
LiDAR-only evaluation against the fused detector. The shared BEV backbone must
still supply wider context; 3.8 m does not cover every car or orientation.
