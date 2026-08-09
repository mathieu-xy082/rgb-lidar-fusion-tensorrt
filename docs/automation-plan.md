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
| `e146168ef424` | RGB-LiDAR master roadmap coordinator | 12h | 10 runs total | Coordinateur read-only: agrège les outputs, signale les branches prêtes, recommande l'ordre de review/intégration |

La tâche maîtresse reçoit comme contexte les derniers outputs des tâches secondaires actives via `context_from` et/ou le scanner déterministe `/root/.hermes/scripts/rgb_lidar_readiness_scan.py`.

## Tâches secondaires actives restantes

| Priorité | Job ID | Nom | Branche | Base de référence | Cadence | Répétitions | Objectif |
|---:|---|---|---|---|---:|---:|---|
| P0 | `a10d6c8fe7a1` | RGB-LiDAR structured splatted maps | `ec/splatted-structured-maps` | `origin/ec/onboarding` | 12h | 8 total | Générer et consommer un `.npz` structuré contenant les cartes LiDAR après splatting |

## Tâche active actuelle

| Priorité | Nom | Branche | Objectif |
|---:|---|---|---|
| P0 | RGB-LiDAR structured splatted maps | `ec/splatted-structured-maps` | Transformer les sorties de séance 4 en artefacts ML structurés `splatted_maps_<id>.npz` |

Prompt recommandé pour ce job :

```text
Projet: /root/ai-projects/rgb-lidar-fusion-tensorrt.
Branche dédiée: ec/splatted-structured-maps.
Base: origin/ec/onboarding.
Objectif: implémenter le plan docs/splatted-structured-maps-plan.md. Ajouter un contrat `.npz` versionné pour `splatted_maps_<id>.npz`, générer les cartes splattées structurées depuis le pipeline KITTI onboarding, permettre à la visualisation de consommer un `.npz` précomputé, et ajouter un mode dataset/model input opt-in pour la représentation splatted. Ne pas entraîner de modèle dans cette branche. Ne jamais committer `data/`, `results/`, datasets, checkpoints, ONNX ou engines TensorRT. Valider avec `pdm run validate` et un smoke réel sur `render_kitti_sequence.py --ids 000000 --alpha 0.45`. Alerter uniquement avec `BRANCHE PRÊTE POUR REVIEW: ec/splatted-structured-maps` quand la branche est validée et poussée.
```

## Ordre recommandé de review à partir d'ici

1. finaliser `ec/splatted-structured-maps` ;
2. review/intégration de ce contrat d'artefact dans la base onboarding/main selon décision de Mathieu ;
3. reprendre ensuite l'entraînement baseline avec `splatted_maps_<id>.npz` comme entrée structurée ;
4. exporter checkpoint entraîné vers ONNX ;
5. benchmark TensorRT sur runtime NVIDIA vérifié.

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
