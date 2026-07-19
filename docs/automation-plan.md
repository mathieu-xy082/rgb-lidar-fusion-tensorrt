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
- Les branches actives doivent être rebasées sur leur base de référence après les mises à jour de `main` qui changent code, contrats, dépendances ou CI.
- Les artefacts générés, checkpoints, datasets, ONNX et engines TensorRT restent hors Git.

## État intégré dans `main`

| Bloc | Branche absorbée | Job | État job |
|---|---|---:|---|
| CI PDM minimale | `ci/gitlab-pipeline` | `7ba3e771823b` | pausé |
| CI staged pipeline | `ci/expand-pipeline-stages` | `47faa6f33ca9` | pausé |
| Calibration/projection KITTI | `feature/kitti-calibration-projection` | `4e740e9f0926` | pausé |
| Dataset sparse LiDAR | `feature/kitti-dataset-lidar-maps` | `c14fc66ace0b` | pausé |
| LiDAR local surface splatting | `feature/lidar-surface-splatting` | `e708c5234841` | pausé |
| Baseline PyTorch | `feature/baseline-fusion-model` | `5ba77d52d513` | pausé |
| Baseline splatted/enriched contract | `feature/baseline-use-splatted-lidar-channels` | `39955b4d68ca` | pausé |
| Dataset-model contract | `feature/dataset-model-contract` | `4c07230c6d83` | pausé |
| Model input adapter contract | `feature/model-input-adapter-contract` | `883d17269097` | pausé |
| ONNX export validation | `feature/onnx-export-validation` | `3b5996eb00b6` | pausé |
| Demo artifact pipeline | `feature/demo-artifact-pipeline` | `d293dd90feea` | pausé |
| TensorRT benchmark plan | `feature/tensorrt-benchmark-plan` | `cbba862ccdfb` | pausé |
| TensorRT runtime environment check | `feature/tensorrt-runtime-environment-check` | `291f2424cf69` | pausé |

Validation de base sur `main` :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate
```

## Tâche maîtresse

| Job ID | Nom | Cadence | Répétitions | Rôle |
|---|---|---:|---:|---|
| `5cb7256a80fb` | RGB-LiDAR master roadmap coordinator | 12h | borné à 40 runs total | Coordinateur read-only: agrège les outputs, signale les branches prêtes, recommande l'ordre de review/intégration |

La tâche maîtresse reçoit comme contexte les derniers outputs des tâches secondaires actives via `context_from` et/ou le scanner déterministe `/root/.hermes/scripts/rgb_lidar_readiness_scan.py`.

## Tâches secondaires actives restantes

Aucune branche pré-training active ne reste après intégration ONNX + demo. La prochaine tâche secondaire doit être créée pour l'entraînement baseline.

## Prochaine tâche à créer maintenant

| Priorité | Nom proposé | Branche proposée | Objectif |
|---:|---|---|---|
| P5 | RGB-LiDAR baseline training loop | `feature/training-loop-baseline` | Boucle d'entraînement PyTorch CPU-safe/GPU-ready, checkpointing, metrics, smoke tests |

Prompt recommandé pour ce futur job :

```text
Projet: /root/ai-projects/rgb-lidar-fusion-tensorrt.
Branche dédiée: feature/training-loop-baseline.
Base: origin/main.
Objectif: implémenter une boucle d'entraînement PyTorch minimale mais sérieuse pour le modèle baseline RGB + LiDAR enriched. Rester CPU-safe en CI, GPU-ready pour runs dédiés. Ajouter scripts/config/tests/docs, ne jamais committer datasets/checkpoints/artefacts générés. Valider avec PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev -G ml puis pdm run validate et tests training ciblés. Alerter uniquement avec `BRANCHE PRÊTE POUR REVIEW: feature/training-loop-baseline` quand la branche est validée et poussée.
```

## Ordre recommandé de review à partir d'ici

1. créer `feature/training-loop-baseline`
2. préparer/réaliser première expérience GPU dédiée
3. exporter checkpoint entraîné vers ONNX
4. benchmark TensorRT sur runtime NVIDIA vérifié

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

Branches ONNX :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev -G ml -G onnx
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate_onnx
```

Training GPU réel :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev -G ml
PDM_IGNORE_ACTIVE_VENV=1 pdm run python scripts/train_baseline.py \
  --config configs/training/kitti_tiny.yaml \
  --device cuda
```

Le script training doit afficher un diagnostic clair si CUDA n'est pas disponible.

## Politique d'intégration

Pour intégrer une branche :

1. Rebaser la branche sur sa base de référence actuelle.
2. Lancer la validation locale appropriée.
3. Vérifier les commits propres avec `git rev-list --left-right --count origin/main...origin/<branch>` ou la base pertinente.
4. Obtenir l'accord explicite de Mathieu.
5. Intégrer dans `main` uniquement après accord.
6. Valider `main`, pousser, pauser le job de la branche absorbée.
7. Vérifier containment avec `git merge-base --is-ancestor`.
8. Supprimer la branche secondaire seulement après confirmation.
9. Rebaser les branches actives restantes sur leur nouvelle base si le changement impacte code/contrats/dépendances/CI.
