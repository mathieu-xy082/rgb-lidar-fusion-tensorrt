# Notebooks

Notebooks disponibles, dans l'ordre recommandé :

1. `01_project_lidar_on_image.ipynb` — verify KITTI calibration by projecting
   raw Velodyne points over the corresponding RGB image.
2. `02_visualize_hybrid_input.ipynb` — inspect the 6 sparse LiDAR maps, the 2
   splatted maps, and the complete 11-channel RGB + LiDAR model input.
3. `03_camera_depth_training_analysis.ipynb` — launch or load a camera-depth run,
   plot training/validation curves, compare against the splatted-depth baseline,
   preview random/spatial holdout regions before training, cross-evaluate matched
   checkpoints, and inspect representative depth/error maps.
4. `04_lidar_bev_analysis.ipynb` — CPU inspection of native LiDAR, PFN/context
   features, KITTI Car annotations, conservative supervision masks, and the
   optional untrained Car-center head. Change `SAMPLE_IDS` to inspect frames.
5. `05_lidar_center_overfit.ipynb` — optional four-frame memorization run,
   checkpoint inspection and center metrics on the same training frames.
6. `06_lidar_center_training_analysis.ipynb` — separate 410/102 train/validation
   pilot, CUDA subprocess, saved split, best/latest checkpoints and held-out views.
7. `07_bev_fusion_overfit.ipynb` — audited camera/LiDAR warm start, CPU inspection
   of supports/features and optional four-train-frame fusion memorization in a
   guarded CUDA subprocess. Encoders stay frozen; this is not validation quality.
8. `08_bev_fusion_training_analysis.ipynb` — matched 410/102 LiDAR-only/fusion
   pilot, opt-in one-variant child process, curves, guarded comparison and CPU
   inspection of saved validation IDs, with PNG/JSON exports.

Keep notebooks lightweight; move reusable code into `src/rgb_lidar_fusion/`.

## BEV Fusion Notebook

Notebook `07` first audits the fresh traced camera weights, LiDAR `best.pt`,
input hashes and common 410/102 membership. The four diagnostic IDs are
`000006`, `000036`, `000052`, `000060`, all in training, with 15 Car centers.
Complete inputs are used; there is no depth holdout. CPU plots show separate
camera/LiDAR evidence, four feature RMS maps, targets and decoded centers.
Camera, PFN and LiDAR context stay frozen, including PFN BatchNorm statistics.
Inspection cells also save 150 dpi PNGs beneath
`results/analysis/bev_fusion/<output-directory-name>/`, with printed paths:
`initial_<sample_id>.png` before learning and
`epoch_<four-digit-epoch>_<sample_id>.png` for the saved checkpoint. These
ignored artifacts are separate from the training output directory and remain
available without retaining notebook outputs.

Start with setup, parameters and CPU inspection only. Training remains disabled
with `RUN_OVERFIT=False`. Once inspected, explicitly choose `RUN_OVERFIT=True`
and rerun parameters/training. The notebook proposes 200 epochs, batch 1, CUDA
in a child process with the existing 2048 MiB free-memory gate. The JSON preset
defaults to CPU/100 epochs. Each batch is logged; evaluations/checkpoints are
saved at 0, 1, every 5 epochs and the final epoch. Interrupt stops and waits for
the child, releasing its GPU resources when it exits; there is no retry/fallback.

Follow the JSONL in another terminal:

```bash
tail -f results/training/kitti_bev_fusion_overfit_4_cuda.jsonl
```

Resume only the fused run with `RESUME_EXISTING=True`, keeping `EPOCHS` as the
total target. Checkpoint metadata states memorization scope and records source
audits, actual four-frame IDs, matching settings and resource measurements.
RMS is not object confidence; a low loss without TP/recall is not memorization
success. Discard these fitted weights before the later disjoint comparison.

The completed local 200-epoch CUDA control finds all 15 centers with no FP/FN;
the source encoder parameters/buffers remain unchanged. This closes the
memorization check, not generalization or camera benefit. The next pilot's
protocol and fresh matched-start audit are in `bev_pilot` and
`configs/training/kitti_bev_fusion_pilot.json`. The streaming runner is
`scripts/train_bev_pilot.py`, with an explicit `--variant lidar_only` or
`--variant fusion`; notebook `08` now pilots these runs. The shared
budget is 10 epochs per variant, with complete inputs and frozen encoders.
Latest/best checkpoints are separate under each variant's directory; only
latest can resume Adam. Do not extend `07` to learn the 512 frames or reuse
its fitted weights. Do not launch both GPU variants concurrently.

## Matched BEV Pilot Notebook

Notebook `08` reads `configs/training/kitti_bev_fusion_pilot.json` and defaults
to `VARIANT='lidar_only'`, `EPOCHS=10`, `DEVICE='cuda'`, with `RUN_TRAINING=False`,
`RESUME_EXISTING=False` and `RUN_COMPARISON=False`. GPU work occurs only in the
training child process. Audits, curves and validation inspection stay on CPU.

1. Execute setup, parameters and audit. Expect 410 train / 102 validation,
   overlap zero and identical fresh-block hashes. No training is started.
2. Set `RUN_TRAINING=True`, rerun parameters, then the training cell. The child
   receives the exact resolved configuration, saved outside the variant's output
   directory. Follow phases/processed counts in the cell or the sibling JSONL.
3. After completion, disable training before inspection. To resume an interrupted
   variant, keep that variant/output root, enable `RESUME_EXISTING=True` and keep
   `EPOCHS` as the total target. Only its `latest.pt` resumes Adam.
4. Train `VARIANT='fusion'` separately with the same protocol/budget and
   `RESUME_EXISTING=False` for its first launch. Never use `07`'s fitted weights.
5. Set `RUN_TRAINING=False` and `RUN_COMPARISON=True` after both runs finish.
   Curves can show partial runs, but the final comparison refuses incomplete or
   incompatible budgets, source weights, initialization, split or evaluation.
6. Change `VALIDATION_SAMPLE_POSITION` and `INSPECT_SELECTION` (`best`/`latest`)
   to inspect saved validation IDs only. Best is selected by validation loss,
   possibly at epoch 0; it is not necessarily the final epoch.

Artifacts are ignored beneath
`results/analysis/bev_fusion/kitti_bev_fusion_pilot_512/`: `training_curves.png`,
`comparison.json`, effective launch configs, and validation PNGs named with
variant, selection, actual weight epoch and sample ID. Training logs are:

```bash
tail -f results/training/kitti_bev_fusion_pilot_512_lidar_only.jsonl
tail -f results/training/kitti_bev_fusion_pilot_512_fusion.jsonl
```

The notebook reads authoritative latest checkpoints, not potentially stale
metrics sidecars. Comparison reports both equal-budget final scores and each
run's independently selected best scores. It does not reuse the older LiDAR
pilot without a shared backbone as the control. Center error concerns matched
objects only; always also examine TP/FP/FN and precision/recall. Validation used
for selection is not an unbiased or scene-independent test set, nor full 3D AP.

The first paired CUDA pilot is complete: ten epochs / 4100 updates per run.
Selected checkpoints are LiDAR-only epoch 9 and fusion epoch 6; final epoch-10
scores are reported separately. Fusion's selected model has fewer false
positives but lower recall, not a demonstrated global improvement. The
[workstream result record](../docs/workstreams/bev-fusion-backbone.md#6d-results-and-prototype-closure)
preserves metrics, artifact hashes, resources and scope. Existing notebook
outputs/launch flags are session state, not documentation to commit or a reason
to rerun training. Use the saved checkpoints for any later inspection.

Install and launch from the repository root:

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G ml -G visualization -G notebook
PDM_IGNORE_ACTIVE_VENV=1 pdm run notebooks
```

The first two notebooks and notebook `04` never start training. The camera-depth notebook does
not start training on `Run All` by default. Set
`RUN_TRAINING = True` in its parameter cell when the configuration and output
directory have been reviewed. Before training, its GPU preflight reports the
driver recovery state, free memory, competing CUDA processes, and the result of
a synchronized CUDA smoke test. A failed preflight blocks GPU training.

In notebook `03`, the first code cell sets `CUDA_LAUNCH_BLOCKING=1` before importing PyTorch for
the ongoing GPU investigation. Keep it before imports in a fresh kernel;
synchronized execution helps locate failures but does not fix driver faults.
Versioned notebooks contain no saved outputs or execution counters.

Between completed runs, the notebook preflight collects unreachable objects and
releases unused PyTorch CUDA cache before measuring free memory. The report
distinguishes memory allocated to live tensors from memory reserved by the
allocator. This does not delete referenced models/tensors, lower the free-memory
threshold, or recover a failed GPU. If active allocations remain unexpectedly
high (for example after a traceback), restart the kernel and rerun setup before
training. Never run the cleanup concurrently with training.

In notebook `03`, the reviewed parameters select the 512-frame spatial-block pilot on CUDA with a
batch size of 1, five epochs and a new output directory:
`results/training/kitti_camera_depth_block_dropout_512_provenance`.
Its preflight requires at least 2048 MiB (2 GiB) of free GPU memory; passing
this threshold does not guarantee that training will fit in memory.
Keep `RESUME_EXISTING = False` for its first launch: the model starts from random
weights and records IDs, hashes and protocol in `split.json` and `camera_depth_v2`
checkpoints. Historical checkpoints without a manifest must not initialize this
run or the upcoming fusion experiment. The opt-in holdout comparison still points
to the two historical runs, not this new run.
To continue this new run after an interruption, set `RESUME_EXISTING = True`;
`EPOCHS` is the total target epoch count, including epochs already completed.
Metrics and run metadata are persisted after every completed camera-depth epoch.

Set `RUN_SPLATTING_ABLATION = True` to train the sparse-input counterpart of
the primary enriched run. The notebook then compares their best held-out
MAE/RMSE using the same dataset split, holdout seeds, and hyperparameters.

The holdout preview cell uses CPU and KITTI data only; it needs no checkpoint.
It shows the selected regions, kept and removed measurements, and recomputed
depth/confidence maps for both `random_points` and `spatial_blocks`.

For the 512-frame block-dropout experiment, the matched training configurations
are `configs/training/kitti_camera_depth_random_holdout_512.yaml` and
`configs/training/kitti_camera_depth_block_dropout.yaml`. After both runs exist,
set `RUN_HOLDOUT_COMPARISON = True` and review `HOLDOUT_CHECKPOINTS`. This flag
only runs validation. It compares each checkpoint on both fixed holdout modes
and writes the four metric rows to
`results/analysis/camera_depth_holdout_comparison.json`. Dataset, split,
normalization, input mode, training settings, epoch, and step must match.
Evaluation batch size can differ from training batch size.

Sample prediction plots use the recorded validation indices and verify current
data for manifest-backed checkpoints. Historical checkpoints reconstruct the
split from configuration with an explicit unverified-provenance warning.
The holdout mode comes from the saved configuration; spatial regions appear in red on RGB.

## LiDAR BEV Notebook

Notebook `04` needs no camera-depth checkpoint. It initializes one CPU model
with fixed random weights for all selected frames. Green rectangles/circles
represent Car ground truth; decoded predictions are red crosses and are not
trained detections. Feature RMS is not a semantic score. Loss components are
the defined objective evaluated without gradients, not a training curve.

The annotation helper downloads only labels for images already present:

```bash
pdm run python scripts/prepare_kitti_labels.py --download
```

Omit `--download` for a local audit only. The report is saved under ignored
`results/analysis/lidar_bev/annotation_audit.json`. It checks structure,
calibration, counterpart files and target construction, not perfect human
annotations. The notebook reads the cached report by default; set
`RUN_ANNOTATION_AUDIT=True` to recheck current files after dataset changes.
Missing labels mean unknown supervision, not a negative training frame.

`RUN_GPU_PROBE=False` keeps inspection CPU-only. Opting in runs the existing
encoder-only diagnostic in a separate bounded process with a 2048 MiB guard;
it does not profile the new head or Adam training. It cannot repair a bad driver.
No dataset download is triggered by Run All. Notebooks `05` and `06` add explicit
training and trained-checkpoint inspection without changing notebook `04`. See the
[optional center diagnostic workstream](../docs/workstreams/lidar-center-diagnostic.md).

## LiDAR Training Notebooks

Notebook `05` remains a bounded one-to-four-frame diagnostic. Its CUDA run on
four frames reached 200 epochs/800 updates with all 13 centers matched. These
are memorized training frames, not validation results. Its disabled-by-default
CUDA/200 preset points to that local run; the CLI JSON retains CPU/100 defaults.

For the 512-frame pilot, open `06`, execute setup/parameters and the split check.
Expect `train=410 validation=102 overlap=0`. The first 512 sorted local KITTI IDs
are split with seed 13; this is frame-disjoint, not guaranteed scene-disjoint.
The model starts fresh, not from the four-frame checkpoint, because some of
those IDs belong to validation. No Car-negative frames are filtered out.

Notebook `06` now preserves the completed CUDA/20-epoch preset with
`RUN_TRAINING=False`. Use the curve and validation-inspection cells to review it.
The JSON config retains a five-epoch starting point for a new run. Starting
again requires an explicit `RUN_TRAINING=True` and a fresh output directory,
or a checked resume with a larger total epoch count; do not overwrite the pilot.
The child process checks GPU readiness with a 2048 MiB guard and prepares data
one frame at a time, rather than caching 512 padded point tensors. No automatic
CUDA fallback/retry is attempted. Both flags default to false; Run All does
local split/hash checks but starts no optimizer or GPU preflight.

Equivalent explicit terminal command, from the repository root:

```bash
pdm run python scripts/train_lidar_center_split.py --config configs/training/kitti_lidar_center_512.json
```

Results live under ignored `results/training/kitti_lidar_center_512/`:

- `split.json`: exact train/validation IDs, policy and input hashes;
- `metrics.json`: update loss, separate eval-mode train/validation losses and
  center precision/recall/XY matching errors, including counts and skipped IDs;
- `checkpoints/latest.pt`: model, Adam, BN, history, split and embedded best weights;
- `checkpoints/best.pt`: CPU-loadable inference checkpoint at minimum `val_loss`;
- `run_metadata.json`: configuration, best epoch and evaluation limitations.

Resume with `RESUME_EXISTING=True` and a larger total `EPOCHS`. Configuration,
split and files are checked; differing runs cannot silently share a directory.
Use `VALIDATION_SAMPLE_POSITION` to inspect a saved validation ID from `best.pt`.
Validation uses `eval()`/`no_grad()` and never updates weights or BN statistics.
The first epoch-0 evaluation is also eligible for best-checkpoint selection.
Do not interpret matched-center error alone as detector quality: inspect TP/FP/FN.
Undefined recall/error is `None`, not a perfect score. The metric is not 3D AP,
and selecting on validation does not create an independent test benchmark.

The completed pilot ran for 20 epochs/8,200 updates. `best.pt` selects epoch 9
by validation loss: 2.01414 versus 3.49954 at epoch 20, while eval-mode train
loss fell from 0.97393 to 0.41241. This overfitting signal does not justify
simply extending training. Validation precision/recall at epoch 9 are
71.66%/63.77% (220 TP, 87 FP, 125 FN) at the configured operating point.

Reviewed examples using `VALIDATION_SAMPLE_POSITION`: 0 (`000009`, found),
28 (`000148`, missed), 79 (`000388`, FP). These cases close the prototype
workflow review, not detector-quality validation. See the
[pilot closure report](../docs/workstreams/lidar-center-diagnostic.md#6-pilot-results-and-prototype-closure).
Executed notebook sessions were backed up locally under ignored
`results/notebook_sessions/lidar-pilot-2026-11-05/` before clearing versioned
outputs/counters. Source presets are retained with training/audit actions off.
