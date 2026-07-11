# Automation plan

Ce fichier documente les tâches Hermes planifiées pour le projet RGB-LiDAR Fusion TensorRT.

Repository local :

```text
/root/ai-projects/rgb-lidar-fusion-tensorrt
```

Repository GitLab :

```text
git@gitlab.com:zeratulakek/rgb-lidar-fusion-tensorrt.git
https://gitlab.com/zeratulakek/rgb-lidar-fusion-tensorrt
```

## Règle générale

- `main` reste stable.
- Chaque tâche secondaire travaille sur sa branche dédiée.
- Les branches secondaires doivent être reviewées avec Mathieu avant intégration.
- Une branche prête doit alerter explicitement : `BRANCHE PRÊTE POUR REVIEW: <branch>`.
- Les tâches ne doivent pas fusionner dans `main` sans instruction explicite.

## État CI

La première CI GitLab minimale a été créée sur `ci/gitlab-pipeline`, validée sur GitLab.com par Mathieu, puis intégrée dans `main` via :

```text
3dfd8e6 merge: integrate gitlab validation pipeline
```

La suite CI est suivie sur une nouvelle branche dédiée :

```text
ci/expand-pipeline-stages
```

Objectif : faire évoluer la CI vers plusieurs stages/jobs utiles sans sur-engineering.

La pipeline cible reste légère et explicite :

- `quality` vérifie que le lockfile PDM reste cohérent avec `pyproject.toml`.
- `test` installe les dépendances de développement puis lance la suite `pytest`.
- `smoke` réinstalle depuis le cache PDM et exécute le smoke test synthétique de projection LiDAR.

Le script local de référence reste `PDM_IGNORE_ACTIVE_VENV=1 pdm run validate`, qui agrège `test` et `smoke` sans ajouter de dépendances lourdes PyTorch / ONNX / TensorRT.

## Tâche maîtresse

| Job ID | Nom | Cadence | Répétitions | Rôle |
|---|---|---:|---:|---|
| `5cb7256a80fb` | RGB-LiDAR master roadmap coordinator | 12h | 60 | Surveille les branches, agrège les outputs des tâches secondaires, recommande reviews/intégrations |

La tâche maîtresse reçoit comme contexte les derniers outputs des tâches secondaires via `context_from`.

## Tâches secondaires

| Job ID | Nom | Branche | Cadence | Répétitions | Objectif | État |
|---|---|---|---:|---:|---|---|
| `7ba3e771823b` | RGB-LiDAR CI workstream | `ci/gitlab-pipeline` | 8h | 10 | CI GitLab PDM minimale | intégrée dans `main` |
| `47faa6f33ca9` | RGB-LiDAR CI staged pipeline workstream | `ci/expand-pipeline-stages` | 12h | 8 | CI multi-stages/jobs | active |
| `4e740e9f0926` | RGB-LiDAR KITTI projection workstream | `feature/kitti-calibration-projection` | 12h | 14 | Calibration/projection KITTI + visualisation | active |
| `e708c5234841` | RGB-LiDAR surface splatting workstream | `feature/lidar-surface-splatting` | 12h | 12 | Niveau 1 : splatting local simple + cartes `depth_expanded`/`confidence` | active |
| `c14fc66ace0b` | RGB-LiDAR dataset workstream | `feature/kitti-dataset-lidar-maps` | 12h | 14 | Dataset RGB + cartes LiDAR sparse/expanded | active |
| `5ba77d52d513` | RGB-LiDAR baseline model workstream | `feature/baseline-fusion-model` | 24h | 10 | Baseline PyTorch simple | active |
| `3b5996eb00b6` | RGB-LiDAR ONNX export workstream | `feature/onnx-export-validation` | 24h | 10 | Export ONNX + validation | active |
| `cbba862ccdfb` | RGB-LiDAR TensorRT benchmark planning workstream | `feature/tensorrt-benchmark-plan` | 48h | 6 | Plan/scripts TensorRT FP16 benchmark | active |

## Ordre recommandé de review

1. `ci/expand-pipeline-stages`
2. `feature/kitti-calibration-projection`
3. `feature/lidar-surface-splatting`
4. `feature/kitti-dataset-lidar-maps`
5. `feature/baseline-fusion-model`
6. `feature/onnx-export-validation`
7. `feature/tensorrt-benchmark-plan`

## Commande de validation locale

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate
```
