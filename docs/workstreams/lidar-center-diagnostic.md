# Workstream - Optional LiDAR center diagnostic

Updated: 2026-11-03.

Base: LiDAR encoder prototype through `2817ef3`. Current development branch:
`ec/lidar-bev-encoder`.

Status: first CPU notebook, optional Car-center objective, and local label
preparation implemented. No supervised training run or trained LiDAR checkpoint
exists. This head is optional, not a new requirement of the fused architecture.

References:

- [LiDAR encoder design](../bev-lidar-encoder.md);
- [encoder prototype and GPU diagnostics](lidar-bev-encoder.md);
- [notebook usage](../../notebooks/README.md#lidar-bev-notebook);
- [target architecture](../bev-model-roadmap.md).

## Scope and Progress

### 1. First notebook

- [x] Add `04_lidar_bev_analysis.ipynb`, CPU-only by default, no network or optimizer.
- [x] Keep reusable loading, inference and plotting in `lidar_bev_analysis`.
- [x] Show native RGB/annotations, full-ROI cloud, counts, PFN/context RMS,
  supervision domain, center targets, and untrained scores.
- [x] Initialize one fixed-seed model across frames, without seeding CUDA.
- [x] Execute the code cells on local `000006` and `000366`; keep outputs/counters
  absent in the versioned notebook. Optional GPU probing stays a subprocess.

The sample IDs have four and two Car centers in the configured ROI. GT centers
are circled in green so a one-cell peak remains visible at the full BEV scale.
RMS, untrained scores and the displayed objective are not detection-quality
results. Missing labels permit geometry inspection but never imply background.
Inference leaves model parameters and BN buffers unchanged; targets do not enter
the encoder inputs, as verified by changing labels while preserving predictions.

### 2. Optional supervised objective

- [x] Add `LidarCenterDiagnostic`: existing PFN/scatter/context plus two 1x1 heads.
- [x] Predict one Car-center heatmap and two sigmoid XY offsets per cell.
- [x] Define masked Gaussian focal loss and center-only L1 offset loss.
- [x] Decode local maxima/top-k into metric center points, not oriented 3D boxes.
- [x] Test geometry, ignored-region gradients, coordinate decoding and backward
  through the complete LiDAR branch. No existing encoder/camera API is changed.

`lidar_center_model` contains the composition/head/loss/decoder;
`lidar_center_targets` is NumPy-first target construction. Channels are row-x
and column-y fractions from each cell's lower bound:

```text
x = x_min + (row + offset_x) * resolution
y = y_min + (col + offset_y) * resolution
```

No extra half-cell shift is added. Targets use a fixed radius-2 Gaussian with
unit peaks, not the size-adaptive targets of a full CenterPoint reproduction.
Heatmap focal exponents are 2 and 4; the sum is normalized by max(Ncenters,1).
Offset L1 averages the two coordinates at centers. Entirely unknown batches
yield differentiable zero; the future runner must track/skip them, not report
them as successful learning. Center collisions raise instead of overwriting.

KITTI locations are bottom centers in rectified reference-camera coordinates.
Subtract h/2 along camera y before solving `T_vehicle_to_camera` to get the
geometric center. Corners use the original yaw/dimensions and the same inverse
transform. P2 translation is used only for image projection, never twice for
the label frame conversion. Native image dimensions/calibration/2D labels stay
aligned; this notebook does not resize them.

#### Conservative supervision domain

KITTI annotations do not cover the entire native LiDAR ROI. A negative cell is
eligible only if its whole vertical column over `grid.z_range_m` projects inside
the image at positive camera and homogeneous projection depth. This is stricter
than testing one ground plane and intentionally leaves extra near-field cells
unknown. It is not a general visibility/occlusion estimator.

DontCare, Van/other non-Car classes, and out-of-ROI Car rectangles exclude their
image frustums at all depths. Intersection uses the projected height segment's
bounding rectangle, conservatively over-ignoring possible cells. Explicit Car
peaks override exclusions only at their annotated center cells. Gaussian tails
do not make ignored space supervised. Raw inputs remain full-ROI, not filtered
using labels. No negative targets are inferred from missing LiDAR returns.

### 3. Labels and audit

- [x] Fetch the official label ZIP only; leave RGB/cloud/calibration files intact.
- [x] Install exactly the 512 IDs with local images, without overwriting conflicts.
- [x] Validate selected ZIP CRCs, record archive SHA-256 and source URL.
- [x] Parse all labels and check local counterparts, geometry and targets.
- [x] Inspect annotated views; automated checks do not prove every human label.

```bash
pdm run python scripts/prepare_kitti_labels.py --download
# Recheck current files without any network:
pdm run python scripts/prepare_kitti_labels.py
```

Official archive: `data_object_label_2.zip`, 5,601,213 bytes. SHA-256:
`4efc76220d867e1c31bb980bbf8cbc02599f02a9cb4350effa98dbb04aaed880`.
This is recorded provenance, not an independently published official checksum.
The cache and installed `data/kitti/training/label_2` remain ignored by Git.

Local audit: 512 frames, 1,947 Car labels, 1,715 Car centers within the configured
50x50 m XY / [-3,3) m Z ROI; all 512 frames have a nonempty supervision domain.
No format/counterpart/target errors or bbox warnings were found. The smaller
center count reflects ROI filtering, not missing label files. A valid partially
visible Truck in `000301` has negative center depth; the parser preserves it
instead of rejecting a box whose visible part can be in front of the camera.
DontCare sentinel 3D values are never treated as physical geometry.

Report: ignored `results/analysis/lidar_bev/annotation_audit.json`. Class counts:
Car 1,947; Van 179; Truck 71; Pedestrian 290; Cyclist 125; Person_sitting 19;
Tram 40; Misc 76; DontCare 1,010.

## Next Increment - Not Implemented

- [ ] Choose/document the train/validation IDs and label policy. The consecutive
  512-frame subset is a pilot, not a full benchmark or guaranteed independent
  scene split; do not silently claim sequence-disjoint generalization.
- [ ] Add a runner with fixed configuration, metrics, checkpoint/resume and
  optimizer-state device handling. Validate overfit on a few samples first.
- [ ] Evaluate held-out center localization/precision/recall with an explicit
  matching policy and ignored-domain handling; this is not 3D box AP.
- [ ] Extend the notebook with trained checkpoint loading and train/val curves.
- [ ] Profile head + optimizer memory before authorizing larger GPU runs.

## Validation

Local suite: 560 passed, one CUDA-only skip. Tests cover label sentinels and
malformed rows, center/corner transforms independent of P2 translation,
visibility/ignore masks, collisions, loss values and masked gradients,
metric decoding, target-independent encoder inputs, immutable inference state,
ZIP selection/conflict safety, audit failure, and notebook safe defaults/state.
All notebook code cells were executed on CPU with the two default real frames;
figures were rendered outside Git. No GPU probe, network action from the
notebook, optimizer step or trained checkpoint was produced.

## References

- [KITTI 3D benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):
  label download and image-visible annotation limits.
- [CenterPoint](https://github.com/tianweiy/CenterPoint): center-based detection
  reference, not a claim of reproducing its full architecture or evaluation.
- [OpenPCDet box utilities](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/utils/box_utils.py):
  camera bottom-center convention and reference box geometry.
