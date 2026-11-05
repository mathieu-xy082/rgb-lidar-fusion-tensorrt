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

Keep notebooks lightweight; move reusable code into `src/rgb_lidar_fusion/`.

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
batch size of 1. To continue an interrupted run, set `RESUME_EXISTING = True`;
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

Sample prediction plots use the saved checkpoint configuration to reconstruct
the validation split and holdout mode. Spatial regions appear in red on RGB.

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
