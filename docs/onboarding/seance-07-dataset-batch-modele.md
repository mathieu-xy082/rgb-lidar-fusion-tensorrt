# Séance 7 — Dataset → batch → modèle

## Objectif

Comprendre comment les artefacts deviennent des entrées consommables par le
modèle.

Chaîne étudiée :

```text
fichier / artefact
→ sample dataset
→ batch NumPy
→ tenseurs PyTorch
→ modèle baseline
```

## Fichiers principaux

```text
src/rgb_lidar_fusion/dataset.py
src/rgb_lidar_fusion/model_batch.py
src/rgb_lidar_fusion/baseline_model.py
tests/test_kitti_dataset.py
tests/test_model_batch.py
tests/test_baseline_model.py
```

## Étape A — Lire un sample dataset synthétique

On commence avec la fixture synthétique pour rester déterministe.

```bash
pdm run python - <<'PY'
from rgb_lidar_fusion.dataset import KittiSparseLidarDataset

dataset = KittiSparseLidarDataset("tests/fixtures/synthetic_kitti")
sample = dataset[0]

print("len:", len(dataset))
print("keys:", sample.keys())
print("image:", sample["image"].shape, sample["image"].dtype)
print("lidar_maps:", sample["lidar_maps"].shape, sample["lidar_maps"].dtype)
print("target:", sample["target"])
print("meta:", sample["meta"])
PY
```

## Interprétation

Le dataset retourne un dictionnaire :

```python
{
    "image": [3, H, W],
    "lidar_maps": [6, H, W],
    "target": ...,
    "meta": ...,
}
```

Le dataset ne lance pas le modèle. Il prépare un sample cohérent.

## Étape B — Transformer un sample en batch

```bash
pdm run python - <<'PY'
from rgb_lidar_fusion.dataset import KittiSparseLidarDataset
from rgb_lidar_fusion.model_batch import dataset_items_to_model_batch

dataset = KittiSparseLidarDataset("tests/fixtures/synthetic_kitti")
batch = dataset_items_to_model_batch([dataset[0]])

print("batch keys:", batch.keys())
print("inputs:", batch["inputs"].shape)
print("input_channels:")
for i, name in enumerate(batch["input_channels"]):
    print(i, name)
PY
```

## Interprétation

Le batch concatène conceptuellement :

```text
RGB channels
+ sparse LiDAR channels
+ éventuellement depth_expanded/confidence
```

## Étape C — Adapter le batch pour le modèle baseline

```bash
pdm run python - <<'PY'
from rgb_lidar_fusion.dataset import KittiSparseLidarDataset
from rgb_lidar_fusion.model_batch import dataset_items_to_model_batch, model_batch_to_baseline_inputs

dataset = KittiSparseLidarDataset("tests/fixtures/synthetic_kitti")
batch = dataset_items_to_model_batch([dataset[0]])
inputs = model_batch_to_baseline_inputs(batch, lidar_mode="enriched")

print("rgb:", inputs["rgb"].shape)
print("lidar_maps:", inputs["lidar_maps"].shape)
PY
```

Forme attendue :

```text
rgb: [B, 3, H, W]
lidar_maps: [B, 8, H, W]
```

## À retenir

```text
dataset.py     = produit un sample
model_batch.py = prépare/adapte le batch
baseline_model.py = consomme rgb + lidar_maps
```

## Mini-exercice

Si `rgb` a 3 canaux et `lidar_maps` a 8 canaux, combien de canaux voit la
première convolution après concaténation interne ?

Réponse attendue :

```text
11 canaux
```
