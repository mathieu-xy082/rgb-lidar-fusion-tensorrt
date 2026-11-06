# Workstream - Camera-depth data provenance

Updated: 2026-11-06.

Branch: `fix/camera-depth-provenance`, based on `ec/bev-fusion-backbone`
at `7719e37`. Return this fix to the fusion branch before its training notebook;
it includes the source-audit integration and is not a main-only patch.

Status: follow-up implemented before the fused
training notebook. This fixes traceability/resume contracts, not the model or
the split algorithm. The previous camera-depth prototype remains closed.

References:

- [camera-depth history](camera-depth-backbone.md);
- [BEV fusion source-weight policy](bev-fusion-backbone.md#weight-provenance-and-checkpoint-policy);
- [milestones](../milestones.md).

## Resume note - 2026-11-05

Developer decision: **do not reuse weights for fusion if doing so requires a
legacy/unverified-provenance exception**. The API escape hatch documented below
is not the chosen project workflow; do not enable `allow_legacy_camera=True`.

When work resumes:

1. Reintegrate this fix into `ec/bev-fusion-backbone` after checking branch state.
2. Prepare a fresh camera-depth training run from random initialization, using
   the corrected runner, a new output directory and the agreed common KITTI
   train/validation membership. Record IDs, hashes and protocol from the start;
   do not resume old camera weights or manufacture a retrospective manifest.
3. Recheck the existing LiDAR checkpoint against that same dataset/split and
   preprocessing. Reuse it only if the ordinary audit passes without exception;
   otherwise rerun the required LiDAR training too.
4. Use only accepted, traced encoder checkpoints for the upcoming fusion
   runner/notebook and four-frame diagnostic. Keep old pilots as historical
   artifacts, not as an exception-based fusion initialization.

No training is started by this note. Run parameters, GPU preflight and execution
will be reviewed together after resuming; no further action is required now.

## Restart preparation - 2026-11-06

- [x] Fix merged into `ec/bev-fusion-backbone` at `0cb0393`.
- [x] Compared the camera configuration's actual ordered 512 IDs and seeded
  410/102 partition with the saved LiDAR split: source/train/validation IDs match.
- [x] Checked all four input-file hashes per frame against the LiDAR manifest;
  current local files match. The LiDAR `best.pt` embeds the same split (epoch 9).
  This is a preliminary data check, not the full encoder warm-start audit.
- [x] Prepared notebook `03` for a fresh enriched spatial-block run: 320x1024,
  batch 1, five epochs, seed/split seed 13. New output directory:
  `results/training/kitti_camera_depth_block_dropout_512_provenance` (absent at
  preparation time). Training and resume remain disabled by default.
- [x] At developer request, set notebook `03`'s free-memory preflight threshold
  to 2048 MiB (2 GiB); the shared utility default is unchanged. This gate is not
  a guarantee of sufficient memory throughout training.
- [x] GPU preflight passed and the developer launched the fresh camera run.
  Epoch 2 was recorded at step 820: validation MAE 2.488615 m, RMSE 4.848676 m,
  identical to the historical spatial-block pilot at that epoch. Completion of
  the five-epoch run is still pending; resume with the same directory and
  `RESUME_EXISTING=True`, keeping `EPOCHS=5` as the total target.
- [ ] Inspect its recorded manifest, validation metrics and predictions.
- [ ] Run the complete ordinary encoder audit with the new camera checkpoint
  and LiDAR `best.pt` before fusion initialization; retrain LiDAR if required.

No training or GPU preflight was launched by the preparation itself; the
developer subsequently started the run as recorded above. The checks
describe local data at preparation time, not independent scenes or guaranteed
future file contents. Historical checkpoints remain untouched.

## Problem and interpretation

A seeded random frame split is a normal experimental baseline. The gap was
that old camera checkpoints saved config/counters, while run metadata saved
only split indices. Without the ordered IDs and input hashes, changing a
directory's contents could silently change what those indices meant. Resume
also loaded model/Adam without checking preprocessing or split identity.

This is a reproducibility risk, not evidence that past validation frames were
used for training or that their reported scores are false. A frame-disjoint
split still does not establish sequence/scene independence; validation used
for model selection is not an unbiased test set.

## Implemented contract

- [x] Snapshot actual `dataset.samples` ordering, source/train/validation IDs,
  saved index partitions and SHA-256 of each image, Velodyne and calibration file.
- [x] Record resolved preprocessing/holdout, normalization, seeds, batch size,
  learning rate and loss settings, including effective defaults.
- [x] Embed the manifest in atomic `camera_depth_v2` checkpoints and save an
  identical `split.json`. Model architecture/state-dict keys are unchanged:
  "v2" identifies the checkpoint envelope, not a new neural network.
- [x] Validate manifest, protocol and current data before CUDA seeding, device
  selection or model allocation on resume; use saved indices, never reroll.
- [x] Reject incompatible resumes, fewer requested epochs, nonempty fresh output
  directories and resumes aimed at another run's checkpoint path.
- [x] Verify each triplet before and after loading it during training/validation
  to detect changes after the initial snapshot. No label files are required
  by the depth task; LiDAR-center/fusion manifests additionally hash labels.
- [x] Expose provenance and manifest-backed validation indices to notebook `03`
  and checkpoint comparison. Verify current data for new checkpoints; warn
  explicitly for legacy reconstructions instead of presenting them as verified.
- [x] Prefer recorded camera membership/hashes in BEV warm starts. Default to
  refusing legacy camera provenance, with an explicit exception documented below.
- [x] Include `fix/**` pushes in the existing CI branch filters.

Implementation: `camera_depth_manifest`, the camera runner in `training`,
`camera_depth_analysis`, and the source audit in `bev_checkpoint`.

The same IDs, ordering and bytes can be relocated to a different dataset root;
the original root is provenance metadata, not the content identity. Adding or
changing frames inside the selected population, modifying a file, or changing
the effective protocol prevents resume. Extending epochs or changing execution
device/output-independent operational settings does not alter the manifest
protocol; numerical equivalence across devices/library versions is not promised.

## Existing checkpoints

The completed five-epoch random/block pilots have **no historical manifest**.
They remain readable for inference and qualitative inspection, with
`provenance_status="legacy_unverified"`. Existing files are not rewritten.
Generating hashes today cannot prove what bytes were used during those runs.

Verified camera training resume rejects these old envelopes; start a fresh
run in a distinct output directory for a genuinely recorded lineage. No
automatic re-training or checkpoint conversion is performed.

For the existing BEV prototype only, a deliberate warm-start exception is:

```python
report = initialize_bev_fusion_encoders(
    model,
    camera_checkpoint=camera_path,
    lidar_checkpoint=lidar_path,
    split=target_split,
    camera_source_ids=original_sorted_ids,
    allow_legacy_camera=True,
)
```

That ID list must be a defensible original population, not an invented mapping
from the seed alone. The returned report remains `legacy_unverified`; current
target-data checks do not certify the camera's historical bytes. Retain that
report in future fused checkpoints. New `camera_depth_v2` sources use recorded
IDs/hashes directly and require neither this flag nor a caller-supplied mapping.

## Validation

CPU fixture tests cover saved membership/protocol/content, all three input-file
mutation cases, protocol mismatches before device initialization, same-size
membership changes, in-read changes, legacy refusal/readability, saved-index
reuse without a splitter call and exact uninterrupted/resumed CPU results.
BEV tests cover default legacy refusal and recorded-manifest warm starts/hash
rejection. Notebook tests execute the new validation-index helper path.

Regression: **874 tests passed, one CUDA-only skip**, without exclusions;
eight expected warnings identify deliberate legacy validation reconstructions.
83 links/anchors were checked across the eight Markdown files changed here.

No new real-KITTI training, GPU work, modification of historical checkpoint
files or claim of retrospective provenance verification is part of this fix.
