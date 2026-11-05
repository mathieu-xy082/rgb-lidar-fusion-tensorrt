# Workstream - Optional LiDAR center diagnostic

Updated: 2026-11-05.

Base: LiDAR encoder prototype through `2817ef3`. Current development branch:
`ec/lidar-bev-encoder`.

Status: prototype increment closed on 2026-11-05. Four-frame memorization,
the 410/102 CUDA pilot and qualitative validation review are complete. Retain
epoch 9 (`best.pt`) by minimum validation loss, not epoch 20 (`latest.pt`).
The training/validation chain is validated, not production detection quality.
This head remains optional, not a requirement of the fused architecture.

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
yield differentiable zero; the train/validation runner skips and reports them
instead of treating them as successful learning. The overfit runner requires
positive frames. Center collisions raise instead of overwriting.

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

### 4. Four-frame memorization

- [x] Train PFN/context/head on explicit IDs, with deterministic point selection,
  fixed configuration, metrics, atomic checkpoints and checked Adam/BN resume.
- [x] Add notebook `05` with opt-in subprocess training and CPU checkpoint views.
- [x] Validate CUDA run on `000006`, `000036`, `000123`, `000366`: 200 epochs,
  800 updates, 13/13 centers matched and no FP at score threshold 0.3/top-k 50,
  within the masked domain and a 1 m matching radius. Mean matched XY distance
  is 0.01026 m, loss 203.38168 to 0.67673. This is memorization, not physical
  centimeter accuracy or generalization. Artifacts remain ignored locally.

### 5. Disjoint train/validation pilot

- [x] Select the first 512 sorted local KITTI image IDs, require all counterpart
  files, and use the existing seeded splitter: 410 train / 102 validation, seed 13.
- [x] Persist exact IDs and file hashes in `split.json` and checkpoints; reject
  configuration, dataset or split changes on resume.
- [x] Initialize fresh weights. `000123` and `000366` are in this validation
  split, so the four-frame weights must not seed this experiment.
- [x] Stream frame preparation instead of caching all padded pillars. Include
  annotated no-Car frames as negatives; skip/report unknown supervision only.
- [x] Evaluate both fixed splits using `eval()` and `no_grad()`, without changing
  parameters/BN. Only train IDs reach the optimizer.
- [x] Save latest state each epoch, plus best validation-loss weights. Embed best
  state/history in latest so interrupted sidecars can be repaired on resume.
- [x] Add notebook `06`: safe defaults, CUDA subprocess with the 2048 MiB guard,
  train/validation curves and inspection restricted to saved validation IDs.
- [x] Exercise a real eight-frame CPU smoke run (6 train / 2 validation), including
  Car-negative frames. Check all 512 file counterparts/hashes and split disjointness.
- [x] Complete the CUDA pilot for 20 epochs/8,200 updates and inspect curves,
  reported peak allocation and representative validation frames. Results below.

Configuration: `configs/training/kitti_lidar_center_512.json`. Explicit command:

```bash
pdm run python scripts/train_lidar_center_split.py --config configs/training/kitti_lidar_center_512.json
```

Outputs: ignored `results/training/kitti_lidar_center_512/`. `latest.pt` is for
resume; `best.pt` is inference-only, selected by minimum `val_loss`, including
the epoch-0 baseline. Epoch count is a total target, not additional epochs.
The new checkpoint kind is distinct from the overfit kind; both remain loadable
for CPU inspection. A nonempty output directory requires explicit resume.

Losses are macro-averages of per-supervised-frame losses. Report train update
loss separately from eval-mode `train_eval_loss` and `val_loss`. Precision/recall
use score-ordered greedy one-to-one XY matching after local-max/top-k decoding
within each frame's supervision domain. Match radius, score threshold and top-k
are fixed in the config, not tuned per frame. Recall is null when no Car target
exists; matched error is null when there is no TP. This is not full-box AP.

### 6. Pilot results and prototype closure

Local run: `results/training/kitti_lidar_center_512/`, CUDA FP32, batch 1,
32 channels, 20 cm grid, Adam at 0.001, seed/split seed 13. All 410 training
and 102 validation frames were evaluated; none were skipped. The split includes
1,370 train and 345 validation Car centers, with 74 and 10 no-Car frames
respectively. The model started fresh; the four-frame weights were not reused.

Twenty epochs/8,200 updates completed in about 44 min 47 s. Reported peak
PyTorch allocated memory was 212.40 MiB, not total device memory. This completed
run demonstrates feasibility on this GPU, not universal driver stability,
end-to-end real-time performance, or export readiness.

| Metric | Best validation-loss epoch 9 | Final epoch 20 |
| --- | ---: | ---: |
| Step | 3,690 | 8,200 |
| Train loss, eval mode | 0.97393 | 0.41241 |
| Validation loss | **2.01414** | 3.49954 |
| Validation precision | 71.66% | 73.38% |
| Validation recall | 63.77% | 59.13% |
| Validation TP / FP / FN | 220 / 87 / 125 | 204 / 74 / 141 |
| Mean XY distance, matched centers only | 0.18851 m | 0.20163 m |

Matching uses score >= 0.3, top-k 50 per frame and the inclusive 1 m radius,
restricted to the supervision domain. Metrics depend on this operating point;
the best loss does not imply the maximum precision or recall. Continued train
improvement while validation loss rises is evidence of overfitting. Do not
extend training solely because the training loss is still decreasing.

Keep `checkpoints/best.pt` at epoch 9 for prototype inspection/future feature
reuse. Keep `latest.pt` at epoch 20 only for an explicit future resume. Both
checkpoint states were checked finite on CPU. Best-checkpoint SHA-256:

```text
eac8f0d39edce3b1b44706936cc8aaf2d88c2dc5d040245c93994455fb5daf5f
```

Qualitative review confirmed by the developer using the epoch-9 checkpoint:

| Validation position | KITTI ID | Observation in the evaluated domain |
| --- | --- | --- |
| 0 | `000009` | One center found, no FP/FN; XY distance 0.13303 m |
| 28 | `000148` | One annotated target missed; no decoded prediction |
| 79 | `000388` | One FP without a matching annotated Car center |

These examples deliberately include success and failure; they are not an
exhaustive failure analysis. RGB was a visual reference, never a LiDAR-branch
input. An FP is defined relative to annotations/domain, not proof of the
physical absence of any vehicle in the entire scene.

Exit decision: the independently trainable LiDAR branch and its disjoint
validation workflow are sufficient for this prototype increment. No extra
optimization of the auxiliary head is required before returning to camera/LiDAR
BEV integration. Fusion, the shared BEV backbone, full 3D heads and deployment
remain separate work. Preserve measurement counts/masks when reusing learned
features; a high activation does not become a new measurement.

## Deferred Quality Work

These are future evaluation improvements, not blockers for prototype closure:

- [ ] Check sequence/scene grouping or acquire appropriate mapping before claiming
  scene-disjoint generalization. A seeded random frame split is not sufficient.
- [ ] Keep an independent test set for final estimates after model selection;
  best-checkpoint validation metrics are not unbiased test results.
- [ ] Broaden the FP/FN analysis, improve the precision/recall operating point
  and profile the fused model when its architecture is defined.

## Validation

Initial validation (2026-11-03): 560 passed, one CUDA-only skip. Tests cover label sentinels and
malformed rows, center/corner transforms independent of P2 translation,
visibility/ignore masks, collisions, loss values and masked gradients,
metric decoding, target-independent encoder inputs, immutable inference state,
ZIP selection/conflict safety, audit failure, and notebook safe defaults/state.
All notebook code cells were executed on CPU with the two default real frames;
figures were rendered outside Git. No GPU probe, network action from the
notebook, optimizer step or trained checkpoint was produced at that initial stage.

Train/validation tests additionally cover reproducible disjoint splits,
validation exclusion from optimizer steps, no-grad/BN immutability, negative
frames, unknown-region skipping, bounded live-frame memory, best/latest selection,
exact CPU resume, changed-file/config refusal, sidecar repair, and failed CUDA
preflight without allocation or retry. Notebook code is validated with default
training disabled and a separate real CPU smoke checkpoint; no full 512-frame
training result is claimed by these development checks.

Pre-pilot development regression check: 631 passed, one CUDA-only skip, two existing
notebook-state checks deselected after confirming that committed notebook `04`
already contains outputs/execution counts and `RUN_ANNOTATION_AUDIT=True`.
They fail on that pre-existing state, not the new runner. Tests ran in a copy
using Git versions of the locally executed `03`/`05`; those working notebooks
were not edited or cleared. The real eight-frame CPU run and notebook `06`
default/smoke-checkpoint cells completed, with curves and a validation frame
rendered and inspected outside Git. The existing 200-epoch CUDA overfit
checkpoint still loads on CPU with the shared loader.

Closure cleanup removes execution outputs/counters from versioned notebooks and
restores opt-in flags, including the previously committed notebook `04` state.
The complete executed `03`/`04`/`05`/`06` sessions were backed up under ignored
`results/notebook_sessions/lidar-pilot-2026-11-05/`; datasets, metrics and trained
checkpoints stay outside Git. Reviewed block/CUDA presets are retained with
training disabled. Final closure regression: **633 passed, one CUDA-only skip**,
with no deselections. All 138 local links/anchors in modified Markdown files
were checked, SVG structure is covered by tests, and JavaScript syntax for
the four explorers passes `node --check`. Only status captions changed in the
explorers, not their numerical algorithms.

## References

- [KITTI 3D benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):
  label download and image-visible annotation limits.
- [CenterPoint](https://github.com/tianweiy/CenterPoint): center-based detection
  reference, not a claim of reproducing its full architecture or evaluation.
- [OpenPCDet box utilities](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/utils/box_utils.py):
  camera bottom-center convention and reference box geometry.
