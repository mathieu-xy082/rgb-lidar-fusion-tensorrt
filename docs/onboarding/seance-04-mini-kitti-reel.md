# Séance 4 — Mini-échantillon KITTI réel

## Objectif

Remplacer la fixture synthétique à 3 points par une vraie scène KITTI :

```text
image caméra réelle
+ scan Velodyne réel
+ calibration KITTI
→ overlay LiDAR sur image
→ sparse maps réelles
→ artefact ML splatté structuré
→ visualisation du splatting
→ galerie multi-frames navigable
```

## Étape A — Télécharger plusieurs samples KITTI

Deux sources sont disponibles.

### Option rapide — miroir léger, 3 frames seulement

```bash
pdm run python docs/onboarding/scripts/download_kitti_samples.py --count 3
```

Limite du miroir léger :

```text
000000,000001,000002
```

Pour les lister :

```bash
pdm run python docs/onboarding/scripts/download_kitti_samples.py --list-available
```

### Option complète — archives KITTI officielles, plus de frames

Pour récupérer 10 frames sans télécharger les gros zips complets :

```bash
pdm run python docs/onboarding/scripts/download_kitti_samples.py --source official --count 10
```

Ou explicitement :

```bash
pdm run python docs/onboarding/scripts/download_kitti_samples.py \
  --source official \
  --ids 000000,000001,000002,000003,000004,000005,000006,000007,000008,000009
```

Le script lit les répertoires ZIP distants par HTTP Range, puis extrait seulement
les fichiers demandés depuis les archives officielles KITTI S3.

Il affiche les commandes/étapes équivalentes :

```text
COMMAND: HTTP Range central-directory ...data_object_image_2.zip...
COMMAND: extract training/image_2/000003.png from ... -> data/kitti/...
```

Ces fichiers restent hors Git grâce à `.gitignore`.

## Étape B — Générer tous les rendus et la galerie

```bash
pdm run python docs/onboarding/scripts/render_kitti_sequence.py --count 3 --alpha 0.45
```

Ce script boucle image par image et affiche les commandes de base qu'il lance :

```text
COMMAND: ... png_to_ppm.py ...
COMMAND: ... smoke_project_lidar.py ...
COMMAND: ... render_splatting_ppm.py ...
```

Pour chaque frame, il produit :

```text
results/onboarding/kitti_<id>/image_<id>.ppm
results/onboarding/kitti_<id>/overlay_<id>.ppm
results/onboarding/kitti_<id>/sparse_maps_<id>.npz
results/onboarding/kitti_<id>/splatted_maps_<id>.npz
results/onboarding/kitti_<id>/visualizations/sparse_depth_overlay.ppm
results/onboarding/kitti_<id>/visualizations/splatted_depth_overlay.ppm
```

`overlay_<id>.ppm` et les fichiers `visualizations/*.ppm` sont des vues humaines/debug.
L'entrée ML structurée est `splatted_maps_<id>.npz` : elle garde les 6 canaux sparse
originaux, puis ajoute `normalized_camera_depth_splatted` et `splat_confidence`.
Le contrat versionné minimal est inspectable sans dépendances ML :

```bash
pdm run python - <<'PY'
import numpy as np
z = np.load('results/onboarding/kitti_000000/splatted_maps_000000.npz')
print(str(z['schema_version'].item()))
print(z['splatted_lidar_maps'].shape)
print([str(x) for x in z['splatted_channel_names']])
PY
```

Il génère aussi :

```text
results/onboarding/kitti_gallery.html
```

## Étape C — Ouvrir la galerie navigable

Depuis le dossier `results/onboarding` :

```bash
cd results/onboarding
python -m http.server 8000
```

Puis ouvrir :

```text
http://127.0.0.1:8000/kitti_gallery.html
```

Le script `render_kitti_sequence.py` affiche aussi cette commande à la fin.

## Navigation

Dans la galerie :

```text
← / →       frame précédente / suivante
↑ / ↓       frame précédente / suivante
n / p       next / previous
1 / 2 / 3 / 4 focus sur un panneau
```

Les 4 panneaux affichés sont :

```text
1. Camera
2. LiDAR overlay
3. Sparse depth overlay
4. Splatted depth overlay
```

## Variante tout-en-un avec serveur

Si tu veux que le script démarre le serveur directement :

```bash
pdm run python docs/onboarding/scripts/render_kitti_sequence.py --count 3 --alpha 0.45 --serve
```

Puis ouvrir :

```text
http://127.0.0.1:8000/kitti_gallery.html
```

## Sorties observées sur les premiers samples

Sample `000000` :

```text
image_shape=370x1224
loaded_points=115384
projected_points=20285
occupied_pixels=20235
splatted_pixels=245529
```

Sample `000001` :

```text
image_shape=375x1242
loaded_points=120268
projected_points=18630
occupied_pixels=18622
splatted_pixels=225910
```

Le splatting devient visuellement intéressant ici parce que le LiDAR réel fournit
des dizaines de milliers de points projetés, pas seulement 3 points synthétiques.
