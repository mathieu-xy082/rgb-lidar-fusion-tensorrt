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

The default parameters select the 512-frame random-point pilot on CUDA with a
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
No dataset download is triggered by Run All. The supervised train/validation
runner, checkpoint loading and loss curves are the next increment, not hidden
behind an unimplemented training flag. See the
[optional center diagnostic workstream](../docs/workstreams/lidar-center-diagnostic.md).
