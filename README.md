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

Ce projet utilise **PDM**, comme XoloLingua, pour garder un environnement reproductible et des commandes standardisées.

```bash
pdm install -G dev
pdm run validate
```

Commandes utiles :

```bash
pdm run test      # lance pytest
pdm run smoke     # lance le smoke test de projection synthétique
pdm run validate  # test + smoke
```

Les dépendances lourdes PyTorch / ONNX / TensorRT ne sont pas installées par défaut. Elles seront ajoutées par groupes PDM au moment des milestones correspondants. Voir `docs/dependency-roadmap.md`.

## Dataset

Dataset recommandé pour démarrer : **KITTI object detection**.

Les données ne doivent pas être commitées. Voir `data/README.md`.

## Status

Initial skeleton créé. La branche `feature/kitti-calibration-projection` couvre maintenant une première lecture testée des fichiers calibration KITTI (`P2`, `R0_rect`, `Tr_velo_to_cam`) et des nuages Velodyne `.bin`, puis projette les points vers l'image et génère des cartes sparse synthétiques. Prochaine cible : générer un overlay image visualisable à partir d'un petit échantillon KITTI local non committé.
