# Workstream - Camera-depth BEV lift

Branch: `ec/camera-depth-bev-lift`

Parent branch: `main` (camera-depth v1 / release 1.1.0).

Updated: 2026-10-18.

Status: feature access and camera/LiDAR geometry are implemented; validity
policy and BEV projection remain. No new training is required for these steps.

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

- [x] Expose resized calibration matrices to the lift, preserving KITTI `P2`.
- [x] Define camera-z depth in meters and the camera-to-LiDAR transform.
- [x] Test synthetic backprojection/reprojection and resized calibration.
- [x] Preserve geometry through stacking, training holdout, and fixed evaluation.
- [x] Check a differentiable depth-to-XYZ path, without changing v1 projection.

#### Geometry contract

`KittiObjectDepthDataset` now supplies these additional float32 arrays. Their
batched forms survive `dataset_items_to_model_batch`, camera-depth holdout, and
`prepare_camera_depth_evaluation_batch` without entering the CNN:

| Key | Batched shape | Meaning |
| --- | --- | --- |
| `camera_projection` | `[B, 3, 4]` | Full resized KITTI P2, including translation |
| `t_camera_from_vehicle` | `[B, 4, 4]` | R0_rect @ Tr_velo_to_cam, meters |
| `max_depth_m` | `[B]` | Per-sample normalization scale |

Legacy manifest/synthetic items without these fields still work. They do not
silently receive an identity calibration. Partial geometry or mixed batches
with/without geometry are rejected.

The geometry functions live in `rgb_lidar_fusion.camera_geometry`:

- `normalized_depth_to_meters(depth, max_depth_m=...)` multiplies normalized
  `[B, 1, H, W]` depth by an explicit scalar or per-sample `[B]` scale. Use the
  checkpoint configuration and the same scale in the dataset; there is no
  hidden 80 m default, clamping, or conversion from Euclidean range.
- `backproject_depth_to_vehicle(depth_m, camera_projection=...,
  t_camera_from_vehicle=...)` returns `[B, 3, H, W]` metric XYZ on the same
  image grid. Matrices and depth must share float32/float64 dtype and device. It retains
  autograd and does not rasterize or round coordinates.

For P2 = [A | t], rectified projection has A's last row [0, 0, 1]. With a known
camera z and pixel (u, v), the homogeneous projection scale is **z + t_z**, not
necessarily z. The implementation solves:

```text
A * X_camera = (z + t_z) * [u, v, 1] - t
X_vehicle_h = inverse(T_camera_from_vehicle) * [X_camera, 1]
```

It uses linear solves, not a second application of P2 translation or an
assumption that K alone is sufficient. Non-rectified projection matrices are
rejected; this is not an arbitrary-camera/distortion model.

Camera coordinates are in the rectified reference camera frame, **before**
P2, with x right, y down, z forward. In KITTI, vehicle coordinates mean
Velodyne: x forward, y left, z up. This is not a calibrated ROS base_link,
world frame, or gravity-aligned frame. The first BEV prototype will use this
LiDAR frame; grid origin, row/column direction, bounds, and height selection
remain increment 4. No additional camera-to-BEV transform is invented here.

#### Pixel alignment and compatibility

The full-resolution feature/depth grid uses integer pixel centers (u, v), with
top-left (0, 0). The historical dataset projection resize is preserved:
u' = sx*u, v' = sy*v. Complete P2 rows, including translation, are scaled;
extrinsics and camera-z in meters do not change.

This convention is an approximation to RGB pixel-center alignment: PIL's
bilinear center mapping is u' = sx*(u + 0.5) - 0.5 (likewise for v), not simple
scaling. Correcting that subpixel offset would change v1 inputs and requires
an explicit preprocessing migration; it is not silently done in this step.
Sparse maps also round continuous projected coordinates with `rint` and clip
at borders, so backprojecting their grid depth need not recover each original
raw LiDAR point exactly. Tests of invertibility use continuous geometry before
this rasterization; they do not claim to undo its quantization.

These functions perform geometry only. Negative, zero, out-of-range, or
non-finite depth is not made reliable by conversion; support filtering and
confidence remain increment 3. No holdout target is used as an input to a
learned lift. The checkpoint schema, model inputs, and training loss stay
unchanged.

Validation on 2026-10-18: geometry tests cover scalar/per-sample scales, known
XYZ points, batched non-identity transforms, non-zero P2 translations (including
t_z), even/odd/tiny grids, resize compatibility, invalid inputs, and numerical
depth-gradient checks. Dataset/batch tests verify unchanged sparse projection,
separate geometry arrays, no CNN channel change, and deterministic evaluation
order. The CPU suite excluding saved notebook-state checks passed with 274 tests
and one CUDA-only skip; the three notebook syntax checks passed separately.

A read-only CPU probe on real KITTI frame 000000 at 320x1024 backprojected
20,082 interior measured grid pixels, then reprojected through the existing
NumPy production path. Maximum pixel error was 0.00012207 px; maximum camera-z
error was 0.00000954 m. This verifies algebraic consistency after rasterization,
not recovery of original subpixel raw points, dense depth quality, or GPU
behavior. No training, checkpoint, or notebook output was modified.

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
