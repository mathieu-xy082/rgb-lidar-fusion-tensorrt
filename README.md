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
- un premier lift géométrique des features caméra vers BEV, avec masques,
  agrégation par cellule, tests de backward et inspection KITTI sur CPU ;
- une préparation des piliers LiDAR bruts et un PFN/scatter vers BEV, avec
  masques, compteurs et contexte dilaté résiduel testés, une inspection conjointe
  caméra/LiDAR et un premier diagnostic GPU borné ;
- une tête LiDAR facultative de localisation des centres de voitures, avec
  cibles KITTI, loss masquée, mémorisation sur quatre frames puis pilote CUDA
  de 20 epochs avec split disjoint 410 train / 102 validation ; bilan et
  limites dans le [workstream LiDAR centre](docs/workstreams/lidar-center-diagnostic.md#6-pilot-results-and-prototype-closure) ;
- des smoke tests CPU, un runner KITTI avec split train/validation, métriques de
  profondeur et checkpointing ;
- un générateur déterministe d'artefacts de démonstration.

Le lift BEV est un prototype PyTorch fonctionnel, relu et inspecté, clôturé pour
ce premier périmètre ; ce n'est pas une détection 3D ni un module validé temps réel.
Le point cloud brut dispose aussi d'un premier PFN/scatter et contexte BEV.
Le bloc de fusion et le backbone commun multi-échelle sont raccordés aux deux
encodeurs dans `BEVFusionDiagnostic`, avec une head centre/offset de diagnostic,
des contrôles de provenance et un format de checkpoint distinct. Les tests CPU
valident le chemin complet. Le notebook `07` a validé la mémorisation CUDA de
quatre frames du train : 15 centres retrouvés sans FP/FN, encodeurs gelés.
Le pilote 410/102 dispose de sa configuration, d'un audit d'initialisation
appariée et d'un runner par variante avec reprise et checkpoints latest/best.
Son notebook, les runs réels et la comparaison LiDAR seul/fusion restent à faire.
Les heads 3D complètes restent à développer. Le
[contrat de fusion BEV](docs/workstreams/bev-fusion-backbone.md#fusion-contract)
et son workstream distinguent ces étapes.

Les nouveaux runs caméra enregistrent IDs, empreintes des données et protocole
dans leurs checkpoints. La [politique de provenance](docs/workstreams/camera-depth-provenance.md)
précise les contrôles de reprise et les limites des anciens poids.

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

Le diagnostic multimodal dispose maintenant d'un adaptateur CPU pour les
entrées RGB/enriched et les piliers, et expose quatre représentations BEV. Sa
configuration de modèle est `configs/bev_fusion.json`. Le preset du diagnostic
est `configs/training/kitti_bev_fusion_overfit.json` ; le
[notebook 07](notebooks/07_bev_fusion_overfit.ipynb) permet l'inspection CPU puis
un lancement explicite en sous-processus. Le pilote disjoint dispose du runner
`scripts/train_bev_pilot.py` ; son notebook et ses expériences restent à faire.

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
La [conception de l'encodeur LiDAR](docs/bev-lidar-encoder.md) précise le
socle intégré dans main depuis la release 1.3.0 : piliers de 20 cm,
PFN/scatter et premier contexte spatial
avant fusion sont implémentés, inspectés sur une grille commune de 20 cm et
testés dans un premier diagnostic GPU. Cela ne valide ni la détection ni le
temps réel de la chaîne complète. Commandes et limites dans le
[workstream LiDAR](docs/workstreams/lidar-bev-encoder.md#4-scope-inspection-compatibility-and-resources).

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

Pour vérifier la projection LiDAR, inspecter l'entrée hybride et analyser
interactivement les entraînements camera-depth :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G ml -G visualization -G notebook
PDM_IGNORE_ACTIVE_VENV=1 pdm run notebooks
```

Suivre les notebooks `01` à `06` décrits dans leur
[README](notebooks/README.md).

## Commandes principales

```bash
# Tests, projection synthétique et artefacts de démonstration
pdm run validate

# Baseline historique, smoke training
pdm run train-baseline -- --config configs/training/synthetic_smoke.yaml

# Backbone camera-depth sur un échantillon KITTI local
pdm run train-camera-depth -- --config configs/training/kitti_camera_depth.yaml

# Notebook d'entraînement et d'analyse camera-depth
pdm run notebooks

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
