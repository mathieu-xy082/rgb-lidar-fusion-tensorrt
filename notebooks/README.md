# Notebooks

Notebook disponible :

1. `03_camera_depth_training_analysis.ipynb` — launch or load a camera-depth run,
   plot training/validation curves, compare against the splatted-depth baseline,
   and inspect representative depth/error maps.

Planned exploratory notebooks:

1. `01_project_lidar_on_image.ipynb` — visualize KITTI LiDAR projected on RGB images.
2. `02_visualize_hybrid_input.ipynb` — inspect RGB + sparse LiDAR feature maps.

Keep notebooks lightweight; move reusable code into `src/rgb_lidar_fusion/`.

Install and launch from the repository root:

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G ml -G visualization -G notebook
PDM_IGNORE_ACTIVE_VENV=1 pdm run notebooks
```

The camera-depth notebook does not start training on `Run All` by default. Set
`RUN_TRAINING = True` in its parameter cell when the configuration and output
directory have been reviewed. Before training, its GPU preflight reports the
driver recovery state, free memory, competing CUDA processes, and the result of
a synchronized CUDA smoke test. A failed preflight blocks GPU training.

Set `RUN_SPLATTING_ABLATION = True` to train the sparse-input counterpart of
the primary enriched run. The notebook then compares their best held-out
MAE/RMSE using the same dataset split, holdout seeds, and hyperparameters.
