# Automation plan

Ce fichier documente les tâches Hermes planifiées pour le projet RGB-LiDAR Fusion TensorRT.

Repository local :

```text
/root/ai-projects/rgb-lidar-fusion-tensorrt
```

Repository GitHub de référence :

```text
git@github.com:mathieu-xy082/rgb-lidar-fusion-tensorrt.git
https://github.com/mathieu-xy082/rgb-lidar-fusion-tensorrt
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
| Training loop baseline | `feature/training-loop-baseline` | ancien job terminé | intégré |
| Onboarding pratique KITTI | `ec/onboarding` | travail manuel | intégré, branche supprimée |
| Structured splatted LiDAR maps | `ec/splatted-structured-maps` | `a10d6c8fe7a1` | intégré, job terminé, branche supprimée |

Validation de base sur `main` :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate
```

## Tâche maîtresse

Le coordinateur borné `e146168ef424` et son workstream splatted
`a10d6c8fe7a1` ont terminé leur cycle et ne sont plus présents dans le
scheduler. Aucun coordinateur RGB-LiDAR récurrent n'est actif actuellement.

Le scanner déterministe `/root/.hermes/scripts/rgb_lidar_readiness_scan.py`
reste disponible pour initialiser un futur coordinateur. Il ne liste plus de
workstream actif et marque onboarding/splatted comme intégrés.

## Tâches secondaires actives restantes

Aucune. Un nouveau job doit être créé uniquement après sélection d'un prochain
objectif borné et création de sa branche depuis `origin/main`.

## Tâche active actuelle

Aucune tâche autonome. Les séances onboarding peuvent continuer manuellement
depuis `main`.

## Ordre recommandé de review à partir d'ici

1. poursuivre les séances onboarding depuis `main` ;
2. sélectionner un prochain objectif borné à partir des réflexions BEV/camera-depth ;
3. créer sa branche et son job dédié seulement après validation du périmètre ;
4. reprendre ensuite l'entraînement avec `splatted_maps_<id>.npz` comme entrée structurée ;
5. exporter un checkpoint entraîné vers ONNX puis benchmarker TensorRT sur un runtime NVIDIA vérifié.

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
