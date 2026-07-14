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
- Les jobs intégrés dans `main` sont pausés, pas supprimés, pour garder l'historique.
- Les branches actives doivent être rebasées sur leur base de référence après les mises à jour de `main`.

## État intégré dans `main`

| Bloc | Branche absorbée | Job | État job |
|---|---|---:|---|
| CI PDM minimale | `ci/gitlab-pipeline` | `7ba3e771823b` | pausé |
| CI staged pipeline | `ci/expand-pipeline-stages` | `47faa6f33ca9` | pausé |
| Calibration/projection KITTI | `feature/kitti-calibration-projection` | `4e740e9f0926` | pausé |
| Dataset sparse LiDAR | `feature/kitti-dataset-lidar-maps` | `c14fc66ace0b` | pausé |
| LiDAR local surface splatting | `feature/lidar-surface-splatting` | `e708c5234841` | pausé |

Validation de base sur `main` :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate
```

## Tâche maîtresse

| Job ID | Nom | Cadence | Répétitions restantes | Rôle |
|---|---|---:|---:|---|
| `5cb7256a80fb` | RGB-LiDAR master roadmap coordinator | 12h | borné à 40 runs total | Coordinateur read-only: agrège les outputs, signale les branches prêtes, recommande l'ordre de review/intégration |

La tâche maîtresse reçoit comme contexte les derniers outputs des tâches secondaires actives via `context_from`.

## Tâches secondaires actives

| Priorité | Job ID | Nom | Branche | Base de référence | Cadence | Répétitions | Objectif |
|---:|---|---|---|---|---:|---:|---|
| P0 | `5ba77d52d513` | RGB-LiDAR baseline model workstream | `feature/baseline-fusion-model` | `origin/main` | 24h | 10 total | Baseline PyTorch simple et contrat modèle minimal |
| P0 | `39955b4d68ca` | RGB-LiDAR P0 splatted baseline contract | `feature/baseline-use-splatted-lidar-channels` | `origin/feature/baseline-fusion-model` | 12h | 8 total | Décider/tester l'usage des cartes `depth_expanded` + `confidence` dans la baseline |
| P1 | `4c07230c6d83` | RGB-LiDAR P1 dataset-model contract | `feature/dataset-model-contract` | `origin/main` | 12h | 8 total | Adapter dataset + splatting vers batch modèle testé |
| P2 | `3b5996eb00b6` | RGB-LiDAR ONNX export workstream | `feature/onnx-export-validation` | `origin/main` | 24h | 10 total | Export/parité ONNX après stabilisation baseline |
| P3 | `cbba862ccdfb` | RGB-LiDAR TensorRT benchmark planning workstream | `feature/tensorrt-benchmark-plan` | `origin/main` | 48h | 6 total | Plan/scripts de benchmark TensorRT FP16 |
| P3 | `291f2424cf69` | RGB-LiDAR P3 TensorRT runtime environment check | `feature/tensorrt-runtime-environment-check` | `origin/feature/tensorrt-benchmark-plan` | 24h | 5 total | Diagnostic CPU-safe TensorRT/CUDA |
| P4 | `d293dd90feea` | RGB-LiDAR P4 demo artifact pipeline | `feature/demo-artifact-pipeline` | `origin/main` | 24h | 5 total | Artefacts de démonstration reproductibles et non versionnés |

## Ordre recommandé de review

1. `feature/baseline-fusion-model`
2. `feature/baseline-use-splatted-lidar-channels`
3. `feature/dataset-model-contract`
4. `feature/onnx-export-validation`
5. `feature/tensorrt-benchmark-plan`
6. `feature/tensorrt-runtime-environment-check`
7. `feature/demo-artifact-pipeline`

## Commandes de validation locale

Branches non-ML :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate
```

Branches qui utilisent PyTorch / groupe ML :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev -G ml
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate
```

## Politique d'intégration

Pour intégrer une branche :

1. Rebaser la branche sur sa base de référence actuelle.
2. Lancer la validation locale appropriée.
3. Vérifier les commits propres avec `git rev-list --left-right --count origin/main...origin/<branch>` ou la base pertinente.
4. Obtenir l'accord explicite de Mathieu.
5. Fast-forward `main` si la branche est directement basée sur `main`, ou intégrer d'abord sa branche parent si c'est une sous-branche.
6. Valider `main`, pousser, pauser le job de la branche absorbée.
7. Rebaser les branches actives restantes sur leur nouvelle base.
