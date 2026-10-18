# Workstream - Camera-depth BEV lift

Branch: `ec/camera-depth-bev-lift`

Parent branch: `main` (camera-depth v1 / release 1.1.0).

Updated: 2026-10-18.

Status: feature access is implemented; geometry, validity policy, and BEV
projection are not implemented yet. No new training is required for this step.

References:

- [BEV connection prerequisites](../bev-cam-depth-backbone.md#prerequis-du-raccord-bev);
- [target architecture](../bev-model-roadmap.md);
- [project milestones](../milestones.md);
- [camera-depth v1 experiments](camera-depth-backbone.md).

## Scope

Implement the camera-depth-to-BEV connection in small, independently testable
increments. Preserve existing depth training, inference, and checkpoints. Keep
the LiDAR BEV encoder, fusion backbone, and detection heads out of this scope.

## Progress

### 1. Expose features without breaking the backbone

- [x] Add `CameraDepthModel.forward_features(rgb, lidar_maps)`.
- [x] Select the existing final decoder output, before `depth_head`.
- [x] Document full-resolution shape and image-grid alignment.
- [x] Preserve the public depth output, state-dict schema, and parameters.
- [x] Test sparse/enriched modes, even/odd shapes, and exact v1 depth parity.
- [x] Test a feature-only backward through encoder/decoder to both inputs.

The contract is `image_features: [B, 32, H, W]`. Spatial index `[v, u]` is the
same image-grid index as RGB and predicted depth, including odd resolutions.
There is no additional feature-grid stride or offset; camera-coordinate
conventions and backprojection belong to increment 2, not this API.

`forward()` still returns `[B, 1, H, W]` and is implemented as
`depth_head(forward_features(rgb, lidar_maps))`. Call those two components
explicitly when both depth and features are needed from one backbone pass.
No features are cached on the module, no tensor is detached, and no extra
convolution, weight, activation, or resize is introduced. A feature-only loss
does not traverse `depth_head`; its parameters correctly receive no gradient.

Tests independently reproduce the v1 forward calculation, freeze the names and
shapes of its 13 convolutions, load checkpoints with `strict=True`, and verify
finite, non-zero gradients through all encoder/decoder stages.

CPU validation on 2026-10-18: 24 camera-depth model tests passed; the full suite
excluding notebook-state checks passed with 239 tests and one CUDA-only skip.
All three notebook syntax checks passed. Notebook execution state was left
untouched rather than cleaned as part of this increment.

A read-only CPU probe also compared both real epoch-5/step-2050 checkpoints
against the committed v1 implementation on synthetic inputs at `17x23`.
Feature shape was `[1, 32, 17, 23]`, both depth paths matched v1 bit for bit,
and checkpoint SHA-256 hashes were unchanged. This checks compatibility, not
new KITTI prediction quality or GPU execution; no training was launched.

### 2. Specify metric depth and calibration

- [ ] Expose resized calibration matrices to the lift, preserving KITTI `P2`.
- [ ] Define camera-z depth in meters and the camera/LiDAR/BEV transforms.
- [ ] Test synthetic backprojection/reprojection and resized calibration.

### 3. Define valid support and confidence

- [ ] Separate valid lift pixels, splat confidence, and learned depth.
- [ ] Reject invalid depths explicitly; do not treat every dense pixel as reliable.

### 4. Implement and verify the BEV grid

- [ ] Define metric bounds, axes, resolution, and aggregation.
- [ ] Test expected cells, collisions, empty cells, and out-of-grid points.
- [ ] Verify a synthetic feature-to-BEV backward, then inspect a KITTI frame.

## Limits

Feature extraction does not perform a lift, infer semantic classes, or establish
depth reliability. The full-resolution representation is a simple alignment
choice for the first prototype, not a latency or memory optimization. A later
lower-resolution API will need its own explicit alignment contract.
