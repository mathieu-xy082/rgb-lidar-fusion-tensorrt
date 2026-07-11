# RGB-LiDAR Fusion TensorRT — Roadmap maîtresse

Ce document pilote le prototype démonstrateur `rgb-lidar-fusion-tensorrt`.

Objectif professionnel : produire un projet AV/robotics crédible et démontrable couvrant :

```text
KITTI RGB image
+ LiDAR projected sparse geometry maps
→ PyTorch detector
→ ONNX export
→ TensorRT FP16 engine
→ benchmark latency
→ demo / README professionnel
```

Le but n'est pas de battre l'état de l'art, mais de démontrer un ownership complet et propre d'un pipeline ML deployment réaliste.

## Principes de travail

1. `main` reste stable et reviewé.
2. Chaque tâche secondaire travaille sur une branche dédiée.
3. Une branche secondaire n'est pas intégrée sans review humaine avec Mathieu.
4. Une branche prête doit déclencher une alerte explicite : `BRANCHE PRÊTE POUR REVIEW`.
5. Les données KITTI, checkpoints, ONNX, engines TensorRT, vidéos et artefacts générés ne sont pas committés.
6. PDM est le workflow Python officiel.
7. Les dépendances lourdes sont ajoutées par groupe au moment utile, pas dans l'installation minimale.
8. Chaque itération doit finir soit par un commit poussé sur sa branche, soit par un message de blocage précis.

## Validation de base

Commande minimale attendue sur toutes les branches :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate
```

## Branches et tâches secondaires

| Ordre | Branche | Objectif | Dépendances | Critère de review |
|---:|---|---|---|---|
| 1 | `ci/gitlab-pipeline` | Créer une CI GitLab PDM minimale | Aucune | Pipeline GitLab exécute `pdm run validate` sur MR/main |
| 2 | `feature/kitti-calibration-projection` | Lire calibration KITTI + projeter LiDAR réel sur image | Aucune | Tests parser/projection + script générant overlay sur fixture ou instructions data |
| 3 | `feature/lidar-surface-splatting` | Niveau 1 : expansion locale simple des points LiDAR projetés par splatting spatial avec carte de confiance | Projection disponible ou fixtures synthétiques | Tests unitaires, visualisation synthétique, API produisant `depth_expanded` + `confidence` sans remplacer le sparse brut |
| 4 | `feature/kitti-dataset-lidar-maps` | Dataset PyTorch/KITTI produisant RGB + cartes LiDAR sparse et, si reviewé, cartes splattées | Projection/calibration reviewée ou disponible; splatting optionnel | Dataset testé sur fixture synthétique, API claire pour futures features |
| 5 | `feature/baseline-fusion-model` | Baseline modèle PyTorch simple `RGB + lidar_maps` | Dataset prêt ou branch accepté comme base | Forward pass testé, shapes documentées, groupe `ml` PDM justifié |
| 6 | `feature/onnx-export-validation` | Export ONNX + validation de parité | Baseline modèle prêt | Script export + test shape/parité, groupe `onnx` PDM |
| 7 | `feature/tensorrt-benchmark-plan` | Préparer chemin TensorRT FP16 et benchmark | ONNX prêt | Plan runtime CUDA/TensorRT, scripts ou stubs testables, limites documentées |

## Hypothèse architecturale — splatting local simple

Mathieu a proposé de ne pas considérer un point LiDAR projeté comme une information strictement ponctuelle. Un impact sur une carrosserie, par exemple, renseigne probablement une petite surface locale continue visible dans l'image.

Décision initiale : commencer par le **niveau 1 — splatting local simple**, avant toute estimation de plan local ou surface concave.

Principe :

```text
projected sparse LiDAR point
→ small local spatial kernel around pixel (u, v)
→ expanded depth map + confidence map
```

Règles de conception :

- conserver les cartes sparse brutes ;
- ajouter des cartes expandées séparées ;
- produire au minimum `depth_expanded` et `confidence` ;
- confiance décroissante avec la distance au point source ;
- rayon limité et paramétrable ;
- si plusieurs points influencent un pixel, résoudre de manière déterministe, idéalement en favorisant profondeur proche/confiance forte ;
- ne pas encore propager via modèle de plan ou surface concave ;
- documenter clairement que cette représentation est une hypothèse expérimentale, à comparer au sparse brut.

Évolutions possibles après review : propagation image-guidée / edge-aware, puis plan local, puis surface locale plus expressive.

Statut branche `feature/lidar-surface-splatting` : API niveau 1 ajoutée dans `rgb_lidar_fusion.lidar_splatting` avec tests synthétiques. Les cartes sparse restent exposées séparément dans le résultat `SplattedDepth`; la stratégie de conflit documentée est `nearest_depth_then_confidence`.

## Tâche maîtresse

La tâche maîtresse ne doit pas coder à la place des branches secondaires. Elle doit :

- surveiller `main` et les branches secondaires ;
- lire les derniers outputs des tâches secondaires ;
- repérer les branches prêtes pour review ;
- repérer les branches bloquées ;
- vérifier que les dépendances entre branches restent cohérentes ;
- proposer l'ordre de review/intégration ;
- alerter Mathieu avec un résumé court et actionnable.

Elle ne doit pas fusionner dans `main` sans instruction explicite.

## Règles d'intégration

Pour intégrer une branche secondaire :

1. Mathieu demande explicitement la review.
2. Rebaser ou mettre à jour la branche si nécessaire.
3. Relancer `PDM_IGNORE_ACTIVE_VENV=1 pdm run validate`.
4. Inspecter le diff.
5. Faire la review ensemble.
6. Intégrer dans `main` uniquement après accord.
7. Pousser `main`.
8. Supprimer la branche secondaire seulement après confirmation.

## CI GitLab cible

Première CI attendue :

```yaml
image: python:3.11

stages:
  - validate

validate:
  stage: validate
  script:
    - pip install pdm
    - pdm install -G dev
    - pdm run validate
```

Évolutions futures possibles :

- cache PDM/pip ;
- jobs séparés lint/test/smoke ;
- artefacts pour overlays de projection ;
- jobs optionnels ML/ONNX ;
- règles MR vs main ;
- badges README.

## Definition of done du premier démonstrateur

Un premier livrable présentable doit contenir :

- une visualisation convaincante de LiDAR projeté dans l'image ;
- un pipeline de features sparse cohérent ;
- un dataset ou pseudo-dataset testable ;
- une baseline modèle simple ;
- une trajectoire claire vers ONNX/TensorRT ;
- des tests reproductibles via PDM ;
- une CI verte ;
- un README expliquant le storytelling AV/robotics.
