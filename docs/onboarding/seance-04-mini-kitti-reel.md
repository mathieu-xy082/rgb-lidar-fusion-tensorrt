# Séance 4 — Mini-échantillon KITTI réel

## Objectif

Remplacer la fixture synthétique à 3 points par une vraie scène KITTI :

```text
image caméra réelle
+ scan Velodyne réel
+ calibration KITTI
→ overlay LiDAR sur image
→ sparse maps réelles
→ visualisation du splatting
```

## Étape A — Télécharger un sample léger

Les gros zips officiels KITTI sont volumineux. Pour l'onboarding, on utilise un
mini-sample versionné dans `kuixu/kitti_object_vis`.

```bash
mkdir -p data/kitti/training/image_2 data/kitti/training/velodyne data/kitti/training/calib
base='https://raw.githubusercontent.com/kuixu/kitti_object_vis/master/data/object/training'
wget -O data/kitti/training/image_2/000000.png "$base/image_2/000000.png"
wget -O data/kitti/training/velodyne/000000.bin "$base/velodyne/000000.bin"
wget -O data/kitti/training/calib/000000.txt "$base/calib/000000.txt"
```

Ces fichiers restent hors Git grâce à `.gitignore`.

## Étape B — Convertir l'image caméra PNG en PPM

Le script de projection du repo lit un canvas PPM simple. Pour éviter ImageMagick
ou Pillow, utiliser le helper standard-library :

```bash
mkdir -p results/onboarding/kitti_000000
pdm run python docs/onboarding/scripts/png_to_ppm.py \
  data/kitti/training/image_2/000000.png \
  results/onboarding/kitti_000000/image_000000.ppm
```

## Étape C — Projeter le vrai LiDAR sur la vraie image

```bash
pdm run python scripts/smoke_project_lidar.py \
  --calib-file data/kitti/training/calib/000000.txt \
  --velodyne-file data/kitti/training/velodyne/000000.bin \
  --image-file results/onboarding/kitti_000000/image_000000.ppm \
  --overlay-output results/onboarding/kitti_000000/overlay_000000.ppm \
  --sparse-output results/onboarding/kitti_000000/sparse_maps_000000.npz
```

Sortie observée sur ce sample :

```text
image_shape=370x1224
mode=kitti
loaded_points=115384
projected_points=20285
lidar_maps_shape=(6, 370, 1224)
occupied_pixels=20235
splatted_pixels=245529
projected_splatted_pixels=245529
splat_max_confidence=1.000
```

## Étape D — Visualiser sparse vs splatted

```bash
pdm run python docs/onboarding/scripts/render_splatting_ppm.py \
  --sparse-npz results/onboarding/kitti_000000/sparse_maps_000000.npz \
  --output-dir results/onboarding/kitti_000000/visualizations
```

Images générées :

```text
results/onboarding/kitti_000000/overlay_000000.ppm
results/onboarding/kitti_000000/visualizations/sparse_depth.ppm
results/onboarding/kitti_000000/visualizations/splatted_depth.ppm
results/onboarding/kitti_000000/visualizations/splatted_confidence.ppm
```

À observer :

```text
sparse_pixels=20235
splatted_pixels=245529
```

Le splatting devient visuellement intéressant ici parce que le LiDAR réel fournit
des dizaines de milliers de points projetés, pas seulement 3 points synthétiques.
