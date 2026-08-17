# RGB-LiDAR Fusion TensorRT

Prototype de perception multimodale caméra/LiDAR pour expérimenter une chaîne
complète allant de la projection géométrique à une future détection 3D en BEV.

Le dépôt ne cherche pas encore une performance SOTA. Il cherche à construire des
contrats de données explicites, des expériences reproductibles et un chemin de
déploiement PyTorch -> ONNX -> TensorRT mesurable.

## État actuel

Le socle implémenté couvre :

- la lecture des calibrations KITTI et la projection du nuage LiDAR dans l'image ;
- six cartes LiDAR sparse alignées sur les pixels caméra ;
- un splatting local qui ajoute une profondeur étendue et sa confiance ;
- un baseline historique RGB/LiDAR utilisé pour valider training, ONNX et les
  outils TensorRT ;
- un backbone camera-depth dense entraîné par holdout de mesures LiDAR ;
- des smoke tests CPU, un runner KITTI local et le checkpointing ;
- un générateur déterministe d'artefacts de démonstration.

La prochaine direction structurante est le lift des features camera-depth vers
une grille BEV. Le point cloud brut conservera ensuite sa propre branche BEV
avant la fusion et les têtes de détection 3D.

L'état détaillé et les critères de sortie sont suivis dans
[docs/milestones.md](docs/milestones.md).

## Architecture

### Pipeline implémenté

```text
KITTI RGB + point cloud + calibration
                |
                v
Projection LiDAR dans l'image
                |
                v
sparse_lidar_maps: [B, 6, H, W]
                |
                v
Splatting local
                |
                v
enriched_lidar_maps: [B, 8, H, W]
          +---------------------+
          |                     |
          v                     v
Legacy fusion baseline      CameraDepthModel
prediction: [B, D]          depth_pred: [B, 1, H, W]
training / ONNX             masked LiDAR holdout loss
```

Le baseline historique prouve la chaîne d'intégration et de déploiement, mais sa
sortie `[B, D]` n'est pas une détection ni une profondeur exploitable. Le
`CameraDepthModel` constitue la première représentation apprise directement
utile à la future branche caméra BEV.

### Architecture cible

```text
RGB + splatted depth/confidence + calibration
                |
                v
Camera-depth backbone -> depth-aware lift -> camera BEV

Raw point cloud -> LiDAR encoder -> LiDAR BEV

camera BEV + LiDAR BEV
                |
                v
BEV fusion backbone
                |
                v
CenterPoint-like 3D detection heads
```

La head YOLO-like reste une supervision auxiliaire ou un prior possible ; elle
ne doit pas bloquer la détection 3D par une cascade de frustums rigide.

La conception complète est décrite dans
[docs/bev-model-roadmap.md](docs/bev-model-roadmap.md). Le sous-système
camera-depth qui prépare le lift est détaillé dans
[docs/bev-cam-depth-backbone.md](docs/bev-cam-depth-backbone.md).

## Contrat LiDAR image-space

Les six canaux sparse conservent les mesures projetées :

```text
0  normalized_camera_depth
1  normalized_vehicle_x
2  normalized_vehicle_y
3  normalized_vehicle_z
4  intensity
5  point_mask
```

La représentation enriched ajoute :

```text
6  normalized_camera_depth_splatted
7  splat_confidence
```

Le batch nominal contient donc `RGB[3] + LiDAR enriched[8]`, soit
`inputs: [B, 11, H, W]`. Le mode sparse-only à 9 canaux reste disponible pour
les ablations. Les consommateurs passent par les adaptateurs de
`rgb_lidar_fusion.model_batch` au lieu de découper ces canaux manuellement.

## Installation

Le projet utilise PDM et Python 3.11 ou supérieur.

```bash
pdm install -G dev
pdm run validate
```

Le groupe `ml` installe PyTorch CUDA 12.9 et Pillow :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev -G ml
```

PyTorch est actuellement fixé à `2.13.0+cu129`. Une machine sans GPU NVIDIA
peut exécuter les tests et les smoke trainings sur CPU.

## Commandes principales

```bash
# Tests, projection synthétique et artefacts de démonstration
pdm run validate

# Baseline historique, smoke training
pdm run train-baseline -- --config configs/training/synthetic_smoke.yaml

# Backbone camera-depth sur un échantillon KITTI local
pdm run train-camera-depth -- --config configs/training/kitti_camera_depth.yaml

# Export et parité du baseline historique
pdm install -G dev -G ml -G onnx
pdm run export_onnx
pdm run validate_onnx

# Diagnostic du runtime TensorRT
pdm run python scripts/tensorrt_benchmark.py --detect
```

Le runner camera-depth attend par défaut 64 frames KITTI Object sous
`data/kitti/training/`. Le téléchargement et l'inspection des données sont
expliqués dans le
[parcours onboarding](docs/onboarding/README.md).

Les datasets, checkpoints, métriques, fichiers ONNX, engines TensorRT et autres
artefacts générés restent hors Git, sous `data/`, `results/` ou un répertoire
temporaire.

## Documentation

Commencer par [docs/README.md](docs/README.md), qui indique le rôle, le statut et
le niveau d'autorité de chaque document.

- [Milestones](docs/milestones.md) : état vivant du projet et prochaines étapes.
- [Architecture BEV](docs/bev-model-roadmap.md) : cible de détection 3D multimodale.
- [Backbone camera-depth](docs/bev-cam-depth-backbone.md) : loss intermédiaire et
  préparation du lift BEV.
- [Onboarding](docs/onboarding/README.md) : prise en main pratique du dépôt.
- [Dépendances](docs/dependency-roadmap.md) : groupes PDM et environnements.
- [TensorRT](docs/tensorrt-benchmark-plan.md) : runtime et protocole de benchmark.
- [Articles](docs/articles/README.md) : bibliographie et positionnement.

## Structure

```text
configs/                  configurations dataset et entraînement
data/                     données locales ignorées par Git
docs/                     architecture, guides, workstreams et archives
scripts/                  points d'entrée training/export/runtime
src/rgb_lidar_fusion/     package Python principal
tests/                    tests unitaires et smoke tests
results/                  artefacts générés ignorés par Git
```

Le projet suit Semantic Versioning. Les changements notables sont consignés
dans [CHANGELOG.md](CHANGELOG.md).
