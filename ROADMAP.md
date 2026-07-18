# RGB-LiDAR Fusion TensorRT — Roadmap maîtresse

Ce document pilote le prototype démonstrateur `rgb-lidar-fusion-tensorrt`.

Objectif professionnel : produire un projet AV/robotics crédible et démontrable couvrant toute la chaîne :

```text
KITTI/synthetic RGB image
+ projected sparse LiDAR geometry maps
+ enriched local LiDAR splatting channels
→ PyTorch baseline model
→ PyTorch training loop + checkpoint
→ ONNX export/parity of the trained model
→ TensorRT FP16 runtime/benchmark path
→ reproducible demo artifacts / portfolio README
```

Le but n'est pas de battre l'état de l'art. Le but est de démontrer un ownership complet et propre d'un pipeline ML deployment réaliste : géométrie, dataset, modèle, entraînement, export, runtime, benchmark et packaging.

## Principes de travail

1. `main` reste stable et reviewé.
2. Chaque tâche secondaire travaille sur une branche dédiée.
3. Une branche secondaire n'est pas intégrée sans review humaine avec Mathieu.
4. Une branche prête doit déclencher une alerte explicite : `BRANCHE PRÊTE POUR REVIEW: <branch>`.
5. Les données KITTI, checkpoints, ONNX, engines TensorRT, vidéos et artefacts générés ne sont pas committés.
6. PDM est le workflow Python officiel.
7. Les dépendances lourdes sont ajoutées par groupe au moment utile, pas dans l'installation minimale.
8. Les scripts doivent rester CPU-safe pour CI quand c'est possible, et explicites quand une ressource GPU/TensorRT manque.
9. Chaque itération doit finir soit par un commit poussé sur sa branche, soit par un message de blocage précis.

## État intégré dans `main`

`main` contient maintenant le socle reviewé suivant :

| Bloc | Branche absorbée | Statut |
|---|---|---|
| CI staged PDM | `ci/expand-pipeline-stages` | intégré |
| Calibration/projection KITTI | `feature/kitti-calibration-projection` | intégré |
| Dataset sparse LiDAR léger | `feature/kitti-dataset-lidar-maps` | intégré |
| LiDAR local surface splatting niveau 1 | `feature/lidar-surface-splatting` | intégré |
| Baseline PyTorch RGB+LiDAR | `feature/baseline-fusion-model` | intégré |
| Contrat baseline enriched/splatted | `feature/baseline-use-splatted-lidar-channels` | intégré |
| Contrat dataset → modèle | `feature/dataset-model-contract` | intégré |
| Contrat adaptateur d'entrée modèle | `feature/model-input-adapter-contract` | intégré |
| Export/parité ONNX baseline enriched | `feature/onnx-export-validation` | intégré |
| Demo artifact pipeline | `feature/demo-artifact-pipeline` | intégré |
| Plan benchmark TensorRT | `feature/tensorrt-benchmark-plan` | intégré |
| Diagnostic runtime TensorRT/CUDA | `feature/tensorrt-runtime-environment-check` | intégré |

## Contrat nominal actuel

Le contrat downstream nominal est **LiDAR enriched**.

```text
rgb:        [B, 3, H, W]
lidar_maps: [B, 8, H, W]
```

Canaux LiDAR :

```text
0: normalized sparse depth
1: x_vehicle
2: y_vehicle
3: z_vehicle
4: intensity
5: sparse point mask
6: depth_expanded
7: confidence
```

La représentation générique batch reste :

```text
inputs: [B, 11, H, W]
```

soit RGB 3 + LiDAR enriched 8. Les consommateurs PyTorch/ONNX/demo/training ne doivent pas découper ce tenseur à la main. Ils doivent passer par l'adaptateur :

```python
batch = dataset_items_to_model_batch(items)
arrays = model_batch_to_baseline_inputs(batch, lidar_mode="enriched")
tensors = model_batch_to_torch_tensors(arrays)
```

Le mode sparse-only 6 canaux reste disponible pour ablation/debug explicite, pas comme chemin par défaut.

## Validation de base

Commande minimale attendue sur `main` et sur les branches légères :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate
```

Branches PyTorch/training :

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

TensorRT reste dépendant de la cible CUDA/TensorRT réelle et doit échouer avec diagnostic clair quand indisponible.

## Phase pré-training

La phase pré-training est terminée côté socle : ONNX et demo sont intégrées dans `main`. La prochaine branche active doit partir de `origin/main` et viser directement la boucle d'entraînement baseline.

## Nouveau jalon central — entraînement modèle

### P5 — Training loop baseline CPU-safe / GPU-ready

Nouvelle branche recommandée maintenant :

```text
feature/training-loop-baseline
```

Objectif : ajouter une boucle d'entraînement PyTorch minimale mais sérieuse, qui fonctionne en smoke test CPU et qui peut être lancée sur GPU dédié pour produire un checkpoint réel.

Livrables attendus :

```text
configs/training/synthetic_smoke.yaml
configs/training/kitti_tiny.yaml
docs/gpu-training-plan.md
scripts/train_baseline.py
src/rgb_lidar_fusion/training.py
tests/test_training_loop.py
```

Critères :

- seed déterministe ;
- `cuda` si disponible, sinon `cpu` ;
- logs explicites : device, epochs, batch size, loss, durée ;
- checkpoint save/resume ;
- metrics JSON/CSV dans un dossier ignoré ;
- test CI sur dataset synthétique minuscule ;
- perte finie ;
- gradients finis et non nuls ;
- au moins un paramètre modifié après une step ;
- aucun dataset/checkpoint committé.

### Cible d'apprentissage initiale

Ne pas viser directement la détection KITTI complète comme première étape. Le premier objectif doit prouver la trainabilité et la reproductibilité sans multiplier les fronts techniques.

Ordre recommandé :

1. target synthétique pour smoke test CI ;
2. petit subset local KITTI ou pseudo-target dense/heatmap ;
3. seulement ensuite, tête plus proche d'une vraie détection.

## P6 — Première exécution GPU dédiée

Quand `feature/training-loop-baseline` est intégrée ou prête à tester :

```text
cloud/local NVIDIA GPU
→ clone repo
→ install PDM groups
→ prepare tiny dataset path
→ train
→ collect checkpoint + metrics
```

Ressources acceptables :

| Usage | GPU suggéré |
|---|---|
| smoke CUDA | petit NVIDIA quelconque |
| baseline expérimental | RTX 3060/4060/4070, T4, L4 |
| itération confortable | RTX 4080/4090, A10 |
| benchmark TensorRT sérieux | GPU NVIDIA compatible CUDA/TensorRT |

Critère de succès : produire un checkpoint et un fichier metrics traçables, avec commande, device, durée et tendance de loss documentés.

Voir : `docs/gpu-training-plan.md`.

## P7 — Export/deployment depuis checkpoint entraîné

Après obtention d'un checkpoint :

1. exporter ONNX depuis checkpoint, pas seulement depuis poids random ;
2. valider PyTorch vs ONNX Runtime ;
3. générer artefacts demo avec le checkpoint ;
4. benchmark TensorRT sur cible compatible.

Branches possibles :

```text
feature/onnx-trained-checkpoint-export
feature/trained-demo-artifacts
feature/tensorrt-trained-benchmark
```

## Hypothèse architecturale — splatting local simple

Mathieu a proposé de ne pas considérer un point LiDAR projeté comme une information strictement ponctuelle. Un impact sur une carrosserie, par exemple, renseigne probablement une petite surface locale continue visible dans l'image.

Décision actuelle : le contrat nominal downstream est enriched. Les cartes sparse brutes sont conservées, et `depth_expanded` / `confidence` complètent l'entrée modèle. Quand aucune expansion non dégénérée n'est demandée, l'identité dégénérée est cohérente :

```text
depth_expanded = normalized sparse depth
confidence = sparse point mask
```

Évolutions possibles après training baseline : propagation image-guidée / edge-aware, plan local, puis surface locale plus expressive.

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

## Ordre recommandé à partir d'ici

```text
1. Review/intégrer feature/onnx-export-validation
2. Review/intégrer feature/demo-artifact-pipeline
3. Créer feature/training-loop-baseline
4. Implémenter training CPU-safe / GPU-ready
5. Choisir et préparer accès GPU dédié
6. Lancer première expérience GPU documentée
7. Exporter le checkpoint entraîné vers ONNX
8. Benchmark TensorRT sur runtime NVIDIA vérifié
```

## CI GitLab cible

La CI doit rester fiable et légère :

- validation de base `dev` sur chaque push ;
- tests PyTorch CPU via groupe `ml` quand les dépendances sont disponibles ;
- ONNX optionnel ou séparé si le temps CI devient trop long ;
- pas de job TensorRT obligatoire sans runner GPU ;
- artefacts générés ignorés ou explicitement uploadés comme artefacts CI, jamais committés.

## Definition of done du démonstrateur présentable

Un livrable portfolio convaincant doit contenir :

- visualisation LiDAR projeté dans l'image ;
- représentation sparse + enriched/splatted cohérente ;
- dataset/pseudo-dataset testable ;
- baseline modèle simple ;
- training loop avec checkpoint réel ;
- métriques honnêtes, même modestes ;
- export ONNX validé ;
- plan ou benchmark TensorRT sur cible NVIDIA ;
- demo artifacts reproductibles ;
- README expliquant clairement que le projet démontre l'ownership pipeline AV/robotics plutôt qu'une performance SOTA.
