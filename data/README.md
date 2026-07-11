# Data

Do not commit datasets to this repository.

Recommended starting point: KITTI object detection dataset.

Expected local layout, once downloaded manually:

```text
data/kitti/
  training/
    image_2/
    velodyne/
    calib/
    label_2/
  testing/
    image_2/
    velodyne/
    calib/
```

The first milestone only needs a few calibration files, RGB images, and Velodyne point clouds to validate projection.

## Ajouter un petit échantillon local KITTI

1. Télécharger manuellement un extrait KITTI object detection hors Git.
2. Copier seulement quelques fichiers localement, par exemple :

```text
data/kitti/training/
  calib/000000.txt
  velodyne/000000.bin
  image_2/000000.png
```

3. Ne pas committer ces fichiers : ils restent des données externes lourdes.
4. Vérifier la lecture calibration/Velodyne avec un petit snippet reproductible :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm run python - <<'PY'
from rgb_lidar_fusion.calibration import parse_kitti_calibration_file
from rgb_lidar_fusion.project_lidar import load_velodyne_bin, project_lidar_to_image

calib = parse_kitti_calibration_file('data/kitti/training/calib/000000.txt')
points = load_velodyne_bin('data/kitti/training/velodyne/000000.bin')
projected = project_lidar_to_image(points, calib, image_shape=(375, 1242))
print(f'loaded_points={points.shape[0]}')
print(f'projected_points={projected.pixels.shape[0]}')
PY
```

Pour une première visualisation sans données KITTI, `PDM_IGNORE_ACTIVE_VENV=1 pdm run smoke` génère la projection synthétique et les cartes LiDAR sparse en mémoire.

Pour écrire un overlay synthétique visualisable sans committer d'image générée :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm run python scripts/smoke_project_lidar.py \
  --overlay-output /tmp/rgb_lidar_synthetic_overlay.ppm
```

Garder les sorties sous `/tmp`, `results/` local ignoré, ou un autre dossier de
travail hors Git.
