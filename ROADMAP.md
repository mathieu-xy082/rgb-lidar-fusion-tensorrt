# RGB-LiDAR Fusion TensorRT — Roadmap maîtresse

Ce document pilote le prototype démonstrateur `rgb-lidar-fusion-tensorrt`.

Objectif professionnel : produire un projet AV/robotics crédible et démontrable couvrant :

```text
KITTI RGB image
+ LiDAR projected sparse geometry maps
+ LiDAR local surface splatting
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

## État intégré dans `main`

`main` contient maintenant le socle reviewé suivant :

| Bloc | Branche absorbée | Statut |
|---|---|---|
| CI staged PDM | `ci/expand-pipeline-stages` | intégré |
| Calibration/projection KITTI | `feature/kitti-calibration-projection` | intégré |
| Dataset sparse LiDAR léger | `feature/kitti-dataset-lidar-maps` | intégré |
| LiDAR local surface splatting niveau 1 | `feature/lidar-surface-splatting` | intégré |

La branche `feature/lidar-surface-splatting` a été intégrée après ajout de tests d'invariants. Le modèle de splatting actuellement accepté est :

- cartes sparse brutes conservées ;
- sorties expandées séparées `depth_expanded` et `confidence` ;
- kernel local carré paramétrable ;
- confiance gaussienne décroissante avec la distance pixel ;
- résolution déterministe des overlaps : profondeur proche, puis confiance forte, puis ordre source row-major ;
- aucune dépendance PyTorch/ONNX/TensorRT pour ce niveau 1.

## Validation de base

Commande minimale attendue sur `main` et sur toutes les branches non-ML :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate
```

Branches qui introduisent le modèle PyTorch :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev -G ml
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate
PDM_IGNORE_ACTIVE_VENV=1 pdm run pytest tests/test_baseline_model.py -q
```

## Branches actives rebasées sur le nouveau `main`

| Priorité review | Branche | SHA actuel | Objectif | Statut attendu avant intégration |
|---:|---|---:|---|---|
| 1 | `feature/baseline-fusion-model` | `7cb0945` | Baseline PyTorch consommant RGB + cartes LiDAR | Review fonctionnelle du contrat modèle, shapes, dépendances `ml`; idéalement adapter l'entrée pour exploiter sparse + splatting |
| 2 | `feature/onnx-export-validation` | `c8d4c43` | Préparer export/parité ONNX | À intégrer après baseline reviewé/mergé, ou garder comme cadrage documentaire si on veut figer le scope plus tôt |
| 3 | `feature/tensorrt-benchmark-plan` | `4f33afa` | Scaffolding runtime/benchmark TensorRT | À intégrer après clarification ONNX/runtime cible ; actuellement utile comme plan testable, pas encore benchmark réel GPU |

## Prochaines tâches / branches prioritaires

### P0 — Review et consolidation du baseline PyTorch

Branche existante : `feature/baseline-fusion-model`.

Objectif : transformer la baseline actuelle en première interface modèle vraiment alignée avec le socle intégré.

Critères :

- forward pass testé sur tenseurs synthétiques ;
- contrat d'entrée documenté : `rgb`, `lidar_maps`, et décision explicite sur l'utilisation de `depth_expanded` / `confidence` ;
- shapes batch/channel/H/W verrouillées par tests ;
- groupe PDM `ml` justifié et minimal ;
- pas de dépendance torchvision/pretrained tant que non nécessaire ;
- validation locale avec `pdm install -G dev -G ml`.

Si une sous-branche est nécessaire avant intégration :

```text
feature/baseline-use-splatted-lidar-channels
```

But de cette sous-branche : décider si la première baseline consomme seulement les 6 cartes sparse, ou un tenseur enrichi incluant `depth_expanded` + `confidence`.

### P1 — Contrat dataset → modèle

Nouvelle branche recommandée :

```text
feature/dataset-model-contract
```

Objectif : écrire le contrat minimal entre `KittiSparseLidarDataset`, le splatting et le modèle.

Livrables attendus :

- helper ou adapter transformant un item dataset NumPy en batch modèle ;
- option explicite pour générer/attacher les cartes splattées ;
- tests sans données KITTI réelles, basés sur fixtures synthétiques ;
- documentation courte : quelles cartes sont entraînables maintenant, quelles cartes restent expérimentales.

### P2 — Export ONNX réel après baseline merge

Branche existante : `feature/onnx-export-validation`.

Objectif suivant après baseline intégré : remplacer le simple cadrage par un export exécutable.

Livrables attendus :

- groupe PDM `onnx` avec `onnx` + `onnxruntime` ;
- `scripts/export_onnx.py` écrivant dans `results/onnx/` ou autre dossier ignoré ;
- test de shape ONNX ;
- test de parité PyTorch vs ONNX Runtime sur tenseurs déterministes ;
- aucune inclusion de fichier `.onnx` dans Git.

### P3 — TensorRT runtime target decision

Branche existante : `feature/tensorrt-benchmark-plan`.

Objectif : passer du scaffolding à une décision runtime vérifiable.

Questions à trancher :

- cible locale CUDA/TensorRT ou container NVIDIA ;
- version CUDA/TensorRT ;
- disponibilité des bindings Python ;
- chemin reproductible ONNX → engine FP16 ;
- métriques minimales : latence p50/p95, FPS, warmup, batch size.

Nouvelle sous-branche possible :

```text
feature/tensorrt-runtime-environment-check
```

But : ajouter une commande de diagnostic qui détecte proprement l'absence de TensorRT/CUDA et produit un message actionnable au lieu d'échouer brutalement.

### P4 — Artefacts de démonstration non versionnés

Nouvelle branche recommandée :

```text
feature/demo-artifact-pipeline
```

Objectif : préparer la démonstration professionnelle sans committer d'artefacts lourds.

Livrables attendus :

- script générant overlay/sparse/splatted maps dans `results/` ;
- README expliquant comment générer localement une image ou courte séquence de démo ;
- `.gitignore` vérifié pour images/vidéos/checkpoints/ONNX/engines ;
- éventuellement capture synthétique légère en texte/PPM si utile.

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
- si plusieurs points influencent un pixel, résoudre de manière déterministe, en favorisant profondeur proche/confiance forte ;
- ne pas encore propager via modèle de plan ou surface concave ;
- documenter clairement que cette représentation est une hypothèse expérimentale, à comparer au sparse brut.

Évolutions possibles après baseline : propagation image-guidée / edge-aware, puis plan local, puis surface locale plus expressive.

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
8. Pauser le job autonome de la branche absorbée.
9. Rebaser les branches restantes sur le nouveau `origin/main`.
10. Supprimer la branche secondaire seulement après confirmation.

## CI GitLab cible

La CI actuelle doit rester alignée avec la validation de base. Évolutions futures possibles :

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
- le splatting local comme innovation expérimentale testée ;
- un dataset ou pseudo-dataset testable ;
- une baseline modèle simple ;
- une trajectoire claire vers ONNX/TensorRT ;
- des tests reproductibles via PDM ;
- une CI verte ;
- un README expliquant le storytelling AV/robotics.
