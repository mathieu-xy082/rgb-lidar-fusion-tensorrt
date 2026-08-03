# Séance 6 — Inspecter les `.npz` et les tenseurs LiDAR

## Objectif

Passer de la visualisation humaine à la compréhension des artefacts numériques.

On veut comprendre :

```text
overlay_<id>.ppm       = image de debug
sparse_maps_<id>.npz   = tenseurs LiDAR sparse
splatted_maps_<id>.npz = tenseurs LiDAR enrichis après splatting, selon branche dédiée
```

Cette séance pourra être ajustée selon l'état de la branche
`ec/splatted-structured-maps`.

## Prérequis

Avoir généré au moins une frame KITTI réelle :

```bash
pdm run python docs/onboarding/scripts/download_kitti_samples.py --source official --count 1
pdm run python docs/onboarding/scripts/render_kitti_sequence.py --count 1 --alpha 0.45
```

## Étape A — Inspecter le sparse `.npz`

```bash
python - <<'PY'
import numpy as np

path = "results/onboarding/kitti_000000/sparse_maps_000000.npz"
z = np.load(path)

print("files:", z.files)

maps = z["lidar_maps"]
print("shape:", maps.shape)
print("dtype:", maps.dtype)

channel_names = [
    "normalized_camera_depth",
    "normalized_vehicle_x",
    "normalized_vehicle_y",
    "normalized_vehicle_z",
    "intensity",
    "point_mask",
]

for i, name in enumerate(channel_names):
    c = maps[i]
    print(i, name, "min=", float(c.min()), "max=", float(c.max()), "nonzero=", int((c != 0).sum()))
PY
```

## Interprétation attendue

```text
shape = (6, H, W)
```

Cela veut dire :

```text
6 canaux LiDAR alignés sur l'image caméra
même hauteur H
même largeur W
```

Le canal `point_mask` indique les pixels où un point LiDAR réel existe.

## Étape B — Comparer pixels image et pixels LiDAR

```bash
python - <<'PY'
import numpy as np

maps = np.load("results/onboarding/kitti_000000/sparse_maps_000000.npz")["lidar_maps"]
_, h, w = maps.shape
mask = maps[5] > 0

print("image_pixels:", h * w)
print("sparse_lidar_pixels:", int(mask.sum()))
print("coverage_percent:", 100.0 * mask.sum() / (h * w))
PY
```

## À retenir

Le LiDAR projeté est utile mais très sparse : beaucoup de pixels caméra n'ont
aucune mesure LiDAR directe.

C'est une des raisons du splatting.

## Étape C — Inspecter le futur `.npz` splatté si disponible

Quand `ec/splatted-structured-maps` sera prête, on inspectera :

```bash
python - <<'PY'
import numpy as np

path = "results/onboarding/kitti_000000/splatted_maps_000000.npz"
z = np.load(path)
print(z.files)
print("splatted_lidar_maps:", z["splatted_lidar_maps"].shape)
print("channels:", z["splatted_channel_names"])
PY
```

Forme attendue :

```text
(8, H, W)
```

## Mini-exercice

Pourquoi `overlay_<id>.ppm` n'est-il pas une bonne entrée ML principale ?

Réponse attendue : parce que c'est une image colorisée où les informations
caméra/LiDAR sont mélangées visuellement, alors qu'un `.npz` conserve des canaux
numériques séparés.
