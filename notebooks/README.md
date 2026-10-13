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

Keep notebooks lightweight; move reusable code into `src/rgb_lidar_fusion/`.

Install and launch from the repository root:

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G ml -G visualization -G notebook
PDM_IGNORE_ACTIVE_VENV=1 pdm run notebooks
```

The first two notebooks never start training. The camera-depth notebook does
not start training on `Run All` by default. Set
`RUN_TRAINING = True` in its parameter cell when the configuration and output
directory have been reviewed. Before training, its GPU preflight reports the
driver recovery state, free memory, competing CUDA processes, and the result of
a synchronized CUDA smoke test. A failed preflight blocks GPU training.

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
