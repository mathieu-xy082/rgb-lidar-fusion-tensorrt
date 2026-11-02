# Workstream - LiDAR BEV encoder

Branch: `ec/lidar-bev-encoder`

Parent branch: `main` (camera-depth BEV lift / release 1.2.0).

Updated: 2026-11-02.

Status: design recorded; implementation has not started. The 20 cm pillar
resolution and pre-fusion context objective are agreed. PFN parameters and the
dilated context block remain proposed choices, not measured results.

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

These checks record design discussion only; no encoder test or experiment has
been run. A 3.8 m theoretical receptive field is not object-recognition evidence.

### 1. Scope: raw XYZI input, pillars, and shared grid

- [ ] Define the batched raw-point input contract, including empty samples.
- [ ] Reuse explicit metric bounds with 0.20 m XY resolution; start with the
  proposed x=[0,50), y=[-25,25), z=[-3,3) ROI without a camera-FOV filter.
- [ ] Build pillar indices, retained-point masks, counts, and nine point features.
- [ ] Specify reproducible overflow selection; start by testing the proposed
  maximum of 64 retained points per pillar and report discarded measurements.
- [ ] Test finite-input policy, half-open bounds, cell centers, non-square
  grids, batch separation, overflow, and absence of input mutation.

Exit criterion: independently inspectable, deterministic pillar construction
matching the existing camera-grid geometry, without a neural network yet.

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
