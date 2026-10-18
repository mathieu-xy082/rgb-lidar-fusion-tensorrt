# Workstream - Camera-depth BEV lift

Branch: `ec/camera-depth-bev-lift`

Parent branch: `main` (camera-depth v1 / release 1.1.0).

Updated: 2026-10-18.

Status: feature access, camera/LiDAR geometry, and depth-source/support policies
are implemented. BEV projection and its use of the validity mask remain.
No new training is required for these steps.

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
| `t_vehicle_to_camera` | `[B, 4, 4]` | R0_rect @ Tr_velo_to_cam, meters |
| `max_depth_m` | `[B]` | Per-sample normalization scale |

Legacy manifest/synthetic items without these fields still work. They do not
silently receive an identity calibration. Partial geometry or mixed batches
with/without geometry are rejected.

The canonical notation is `t_vehicle_to_camera` in Python and
`T_vehicle_to_camera` in equations: `X_camera_h = T_vehicle_to_camera *
X_vehicle_h`. Its inverse maps camera coordinates back to LiDAR coordinates;
the rename does not invert the stored matrix. Existing JSON calibration
manifests may still use the old `t_camera_from_vehicle` key (not both names
at once). Python calibration fields, keyword arguments, and batch keys now use
the new name; model parameters and checkpoint weights are unaffected.

The geometry functions live in `rgb_lidar_fusion.camera_geometry`:

- `normalized_depth_to_meters(depth, max_depth_m=...)` multiplies normalized
  `[B, 1, H, W]` depth by an explicit scalar or per-sample `[B]` scale. Use the
  checkpoint configuration and the same scale in the dataset; there is no
  hidden 80 m default, clamping, or conversion from Euclidean range.
- `backproject_depth_to_vehicle(depth_m, camera_projection=...,
  t_vehicle_to_camera=...)` returns `[B, 3, H, W]` metric XYZ on the same
  image grid. Matrices and depth must share float32/float64 dtype and device. It retains
  autograd and does not rasterize or round coordinates.

For P2 = [A | t], rectified projection has A's last row [0, 0, 1]. With a known
camera z and pixel (u, v), the homogeneous projection scale is **z + t_z**, not
necessarily z. The implementation solves:

```text
A * X_camera = (z + t_z) * [u, v, 1] - t
X_vehicle_h = inverse(T_vehicle_to_camera) * [X_camera, 1]
```

It uses linear solves, not a second application of P2 translation or an
assumption that K alone is sufficient. Non-rectified projection matrices are
rejected; this is not an arbitrary-camera/distortion model.
The implementation comments detail homogeneous division, the z + t_z scale,
a numerical example with non-zero t_z, and the inverse extrinsic solve. A
regression test reproduces the numerical example.

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
confidence are handled separately by increment 3. No holdout target is used as
an input to a learned lift. The checkpoint schema, model inputs, and training loss stay
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

- [x] Separate valid lift pixels, splat confidence, and learned depth.
- [x] Require finite camera-z, explicit support, and an explicit metric range.
- [x] Select sparse, splatted, or predicted depth without accessing targets.
- [x] Require an explicit support mask for predicted depth; no dense default.
- [x] Exclude invalid values without promoting them to valid via a clamp.
- [x] Test both holdout modes for target leakage and finite geometry backward.

The new `camera_depth_support` module contains `build_depth_valid_mask` and
`prepare_camera_depth_support`. It returns a `CameraDepthSupport` object:

| Field | Meaning |
| --- | --- |
| `source` | Explicit choice: `sparse`, `splatted`, or `predicted` |
| `depth_m` | `[B, 1, H, W]` camera-z in meters; zero placeholders at invalid pixels |
| `valid_mask` | Boolean `[B, 1, H, W]`, governing future geometry inclusion |
| `splat_confidence` | Original heuristic map for the splatted source only; otherwise `None` |

The mask is `support & isfinite(z) & (min < z) & (z <= max)`, with
`0 <= min < max`. `max_valid_depth_m` is required separately from the
checkpoint normalization `max_depth_m`: an 80 m normalization scale does not
force an 80 m lift range. The default lower bound is zero, strictly excluded.

Policies are deliberately distinct:

- `sparse`: channel 0 depth, supported only by finite positive channel 5
  measurement presence. For training/evaluation, use retained input maps after
  holdout, never the full target maps.
- `splatted`: channels 6/7 depth and score from post-holdout resplatting. Support
  requires finite `0 < confidence <= 1` and `confidence >= min_splat_confidence`.
  The default threshold is zero: any positive valid score is eligible, not
  guaranteed geometrically accurate. A stricter threshold is configurable.
- `predicted`: normalized `depth_prediction` plus a required caller-supplied
  boolean `prediction_support_mask`. There is no implicit all-pixel domain or
  inherited splat uncertainty. Retained/splatted support can define a cautious
  first domain; selecting a wider domain is an explicit experimental policy,
  not evidence that the learned depth is reliable there.

Example using the actual prepared model input (not `depth_target`):

```python
selected = prepare_camera_depth_support(
    torch.from_numpy(depth_batch["lidar_maps"]),
    source="splatted",
    max_depth_m=checkpoint_config["max_depth_m"],
    max_valid_depth_m=50.0,
)
```

Invalid values are zeroed only in the returned metric tensor to prevent
NaN/Inf contamination of a subsequent dense solve. No input is mutated or
detached. **The corresponding mask stays false**: placeholders can still
produce finite XYZ after backprojection, especially with translated cameras,
and must never contribute to BEV cells. The BEV implementation in increment 4
must apply this mask before indexing/aggregation. An empty mask is valid.

Splat confidence is not a probability of correct depth, an object score, or a
learned uncertainty. The original map is returned unchanged, including scores
rejected by the mask, so only valid entries may be used for future weighting.
No uncertainty head, trainable parameter, training-loss change, or checkpoint
change is introduced. These utilities are tested but are not called by the
current training/inference path; their BEV integration remains increment 4.

Tests cover boundaries, invalid depths/scores, empty support, channel contracts,
per-sample scales, explicit prediction domains, no input mutation, finite
backward through geometry with an invalid prediction, and unchanged selection
after modifying only hidden targets under random-point and block holdout.

CPU validation on 2026-10-18: 35 support-policy tests passed; the suite excluding
saved notebook-state checks passed with 313 tests and one CUDA-only skip.
Notebook syntax was checked separately, without changing their saved state.
No GPU training or new checkpoint was produced. This step remains separate
from the preceding notation/math-comment commit `b04546c`.

### 4. Implement and verify the BEV grid

- [ ] Define metric bounds, axes, resolution, and aggregation.
- [ ] Test expected cells, collisions, empty cells, and out-of-grid points.
- [ ] Verify a synthetic feature-to-BEV backward, then inspect a KITTI frame.

## Limits

Feature extraction does not perform a lift, infer semantic classes, or establish
depth reliability. The full-resolution representation is a simple alignment
choice for the first prototype, not a latency or memory optimization. A later
lower-resolution API will need its own explicit alignment contract.
