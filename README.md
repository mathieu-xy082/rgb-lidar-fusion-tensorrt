# RGB-LiDAR Fusion TensorRT Prototype

Prototype démonstrateur de fusion caméra/LiDAR pour perception AV/robotics.

Objectif : construire un pipeline complet et démontrable :

```text
KITTI RGB image
+ LiDAR projected sparse geometry maps
→ PyTorch detector
→ ONNX export
→ TensorRT FP16 engine
→ real-time inference benchmark
→ demo video / README professionnel
```

Ce repository vise d'abord un prototype propre, mesuré et présentable — pas un modèle SOTA.

## Motivation

Le projet montre l'ownership d'un pipeline ML deployment réaliste :

- géométrie caméra/LiDAR ;
- dataset PyTorch ;
- modèle de perception multimodal ;
- export ONNX ;
- optimisation TensorRT FP16 ;
- benchmark de latence ;
- packaging demo AV/robotics.

## Milestones

### Milestone 1 — Visualisation LiDAR projeté

Produire des images RGB avec points LiDAR colorés par profondeur.

Livrables :

```text
src/rgb_lidar_fusion/calibration.py
src/rgb_lidar_fusion/project_lidar.py
results/projected_lidar_examples/*.png
```

### Milestone 2 — Dataset PyTorch

Créer un dataset qui retourne :

```python
{
    "image": Tensor[3, H, W],
    "lidar_maps": Tensor[5, H, W],
    "target": boxes/classes/depth,
}
```

Puis construire :

```python
input = torch.cat([image, lidar_maps], dim=0)
```

### Milestone 3 — Baseline modèle

D'abord simple : adapter une première convolution à `8` ou `9` canaux.

Ensuite, si utile : deux branches RGB / LiDAR maps avec fusion intermédiaire.

### Milestone 4 — Export ONNX

- exporter le modèle ;
- vérifier les shapes ;
- comparer PyTorch vs ONNX Runtime ;
- documenter les limitations.

### Milestone 5 — TensorRT FP16

```text
PyTorch checkpoint
→ ONNX export
→ TensorRT engine FP16
→ inference TensorRT
→ benchmark latency p50/p95/FPS
```

## Structure

```text
configs/                    Configuration dataset/modèle
src/rgb_lidar_fusion/        Package Python principal
tests/                       Tests unitaires
notebooks/                   Exploration visuelle
results/                     Résultats générés non versionnés
scripts/                     Commandes utilitaires
```

## Installation dev

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Dataset

Dataset recommandé pour démarrer : **KITTI object detection**.

Les données ne doivent pas être commitées. Voir `data/README.md`.

## Status

Initial skeleton créé. Première cible : projection géométrique LiDAR → image et génération de cartes sparse depth/geometry.
