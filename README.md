# RGB-LiDAR Fusion TensorRT Prototype

Prototype démonstrateur de fusion caméra/LiDAR pour perception AV/robotics.

Objectif : construire un pipeline complet et démontrable :

```text
KITTI RGB image
+ LiDAR projected sparse geometry maps
→ PyTorch baseline + training loop
→ checkpoint entraîné, même modeste
→ ONNX export
→ TensorRT FP16 engine
→ real-time inference benchmark
→ demo video / README professionnel
```

Ce repository vise d'abord un prototype propre, mesuré et présentable — pas un modèle SOTA.

## Motivation

Le projet montre l'ownership d'un pipeline ML deployment réaliste :

- géométrie caméra/LiDAR ;
- dataset PyTorch ;
- modèle de perception multimodal ;
- entraînement PyTorch reproductible et checkpointing ;
- export ONNX ;
- optimisation TensorRT FP16 ;
- benchmark de latence ;
- packaging demo AV/robotics.

## Milestones

### Milestone 1 — Visualisation LiDAR projeté

Produire des images RGB avec points LiDAR colorés par profondeur.

Livrables :

```text
src/rgb_lidar_fusion/calibration.py
src/rgb_lidar_fusion/project_lidar.py
results/projected_lidar_examples/*.png
```

### Milestone 2 — Dataset PyTorch

Créer un dataset qui retourne :

```python
{
    "image": Tensor[3, H, W],
    "lidar_maps": Tensor[6, H, W],  # depth, xyz geometry, intensity, mask
    "target": boxes/classes/depth,
}
```

Puis construire :

```python
input = torch.cat([image, lidar_maps], dim=0)
```

### Milestone 3 — Baseline modèle

D'abord simple : adapter une première convolution à `8` ou `9` canaux.

Ensuite, si utile : deux branches RGB / LiDAR maps avec fusion intermédiaire.

### Milestone 4 — Training baseline

- ajouter une boucle d'entraînement PyTorch CPU-safe / GPU-ready ;
- sauvegarder checkpoints et métriques hors Git ;
- valider un smoke test synthétique en CI ;
- préparer un run sur GPU dédié pour produire un checkpoint réel.

Voir `docs/gpu-training-plan.md`.

### Milestone 5 — Export ONNX

- exporter le modèle ;
- vérifier les shapes ;
- comparer PyTorch vs ONNX Runtime ;
- documenter les limitations.

### Milestone 6 — TensorRT FP16

```text
PyTorch checkpoint
→ ONNX export
→ TensorRT engine FP16
→ inference TensorRT
→ benchmark latency p50/p95/FPS
```

Le chemin TensorRT est préparé sans supposer que la machine de développement a CUDA/TensorRT. Voir `docs/tensorrt-benchmark-plan.md` pour la stratégie local vs container, les scripts stubs sûrs et le schéma de benchmark.

La roadmap maîtresse complète, incluant l'ordre ONNX/demo → training → GPU → checkpoint → export/benchmark, est documentée dans `ROADMAP.md`.

## Structure

```text
configs/                    Configuration dataset/modèle
docs/articles/              Bibliographie et positionnement littérature
docs/onboarding/            Parcours pratique guidé
src/rgb_lidar_fusion/        Package Python principal
tests/                       Tests unitaires
notebooks/                   Exploration visuelle
results/                     Résultats générés non versionnés
```


## Versioning

Le projet suit Semantic Versioning. Les changements notables et les éléments
validés pour la future release stable sont suivis dans `CHANGELOG.md`.

## Onboarding pratique

Un parcours de tutorat progressif est disponible dans `docs/onboarding/` pour
s'approprier le repo par la pratique : validation locale, artefacts demo,
projection LiDAR, splatting, mini-échantillon KITTI réel et contrats de batch
modèle.

## Installation dev

Ce projet utilise **PDM**, comme XoloLingua, pour garder un environnement reproductible et des commandes standardisées.

```bash
pdm install -G dev
pdm run validate
```

Commandes utiles :

```bash
pdm run test      # lance pytest
pdm run smoke     # lance le smoke test de projection synthétique
pdm run demo-artifacts  # régénère un bundle de démo sous results/demo_artifacts/
pdm run validate  # test + smoke
```

## Artefacts de démonstration reproductibles

La branche `feature/demo-artifact-pipeline` fournit un générateur déterministe
qui produit un mini-bundle synthétique sans données KITTI externes ni dépendances
lourdes :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm run demo-artifacts
```

La commande écrit sous `results/demo_artifacts/` :

```text
projected_lidar_examples/synthetic_overlay.ppm
sparse_maps/synthetic_sparse_maps.npz
splatted_maps/synthetic_splatted_maps.npz
model_inputs/synthetic_baseline_inputs.npz
manifest.json
```

La commande réinitialise le dossier de sortie ciblé avant d'écrire le bundle, puis
renseigne la taille image, le nombre de points projetés, les pixels sparse
occupés, les pixels couverts par le splatting, les chemins de sortie relatifs au
dossier de génération, les shapes des entrées baseline RGB+LiDAR enrichi, ainsi
que les tailles et SHA-256 de chaque artefact pour vérifier la reproductibilité
entre machines. Pour choisir un autre dossier de sortie :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm run python -m rgb_lidar_fusion.demo_artifacts \
  --output-dir /tmp/rgb_lidar_demo_artifacts
```

Les overlays/images, vidéos, exports ONNX, engines TensorRT, checkpoints, logs
et benchmarks générés sont explicitement exclus du versionnement.

Pour générer une visualisation synthétique sans dépendance image externe :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm run python scripts/smoke_project_lidar.py \
  --overlay-output /tmp/rgb_lidar_synthetic_overlay.ppm \
  --sparse-output /tmp/rgb_lidar_synthetic_sparse_maps.npz
```

Pour créer en plus un mini-échantillon synthétique au format KITTI (`training/calib/000000.txt`
et `training/velodyne/000000.bin`) utilisable comme fixture locale hors Git :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm run python scripts/smoke_project_lidar.py \
  --write-synthetic-sample /tmp/rgb_lidar_synthetic_kitti
```

Le fichier produit est un PPM ASCII (`P3`) ouvrable avec la plupart des viewers
image ou convertible localement. Le `.npz` optionnel contient `lidar_maps` au
format `[6, H, W]` (`depth`, xyz véhicule, intensité, masque). Les overlays,
cartes sparse générées et données KITTI restent hors Git.

Pour projeter un petit échantillon KITTI local déjà téléchargé sans committer les
données, fournir la calibration, le Velodyne `.bin` et la taille image
`HAUTEURxLARGEUR` :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm run python scripts/smoke_project_lidar.py \
  --calib-file data/kitti/training/calib/000000.txt \
  --velodyne-file data/kitti/training/velodyne/000000.bin \
  --image-size 375x1242 \
  --overlay-output /tmp/rgb_lidar_kitti_000000_overlay.ppm \
  --sparse-output /tmp/rgb_lidar_kitti_000000_sparse_maps.npz
```

La commande écrit l'overlay sur un fond noir de même résolution que l'image KITTI
et affiche le nombre de points chargés/projetés. Garder le PPM généré sous
`/tmp`, `results/` local ignoré, ou un autre dossier hors Git.

Pour dessiner sur un canvas RGB local sans ajouter Pillow/OpenCV au socle léger,
convertir ponctuellement l'image KITTI en PPM ASCII (`P3`) hors Git puis fournir
ce canvas :

```bash
magick data/kitti/training/image_2/000000.png /tmp/kitti_000000.ppm
PDM_IGNORE_ACTIVE_VENV=1 pdm run python scripts/smoke_project_lidar.py \
  --calib-file data/kitti/training/calib/000000.txt \
  --velodyne-file data/kitti/training/velodyne/000000.bin \
  --image-file /tmp/kitti_000000.ppm \
  --overlay-output /tmp/rgb_lidar_kitti_000000_overlay.ppm
```

`--image-file` accepte uniquement un PPM ASCII `P3`; sa taille est inférée
automatiquement si `--image-size` n'est pas fourni. Si `--image-size` est aussi
fourni, il doit correspondre au canvas PPM. Les PNG/JPEG restent une dépendance
future optionnelle.

Pour le baseline PyTorch de Milestone 3, installer aussi le groupe `ml` :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev -G ml
PDM_IGNORE_ACTIVE_VENV=1 pdm run pytest tests/test_baseline_model.py -q
```

Le modèle minimal `BaselineFusionModel` consomme deux tenseurs alignés spatialement.
Le contrat par défaut est désormais le LiDAR **enriched** à 8 canaux : les 6 cartes
sparse brutes restent intactes et `depth_expanded`/`confidence` sont ajoutées en
complément, pas en remplacement.

```python
rgb: Tensor[B, 3, H, W]
lidar_maps: Tensor[B, 8, H, W]
# channels 0..5: normalized depth, vehicle xyz, intensity, sparse point mask
# channel 6: depth_expanded from local LiDAR splatting
# channel 7: confidence from local LiDAR splatting
output: Tensor[B, output_dim]
```

Une option explicite `BaselineFusionModel(lidar_mode="sparse")` conserve le chemin
6 canaux pour comparer un baseline sparse-only, mais le chemin recommandé pour les
itérations ONNX/TensorRT suivantes est `lidar_mode="enriched"`.

Le batch dataset reste NumPy-first et expose aussi une représentation générique
concaténée `inputs: [B, C, H, W]`. Par défaut, il produit le contrat enriched
nominal `inputs: [B, 11, H, W]` : RGB + 6 cartes LiDAR sparse +
`depth_expanded`/`confidence`. Si aucune expansion locale non dégénérée n'est
demandée, ces deux derniers canaux utilisent la limite identité d'une gaussienne
infiniment piquée : `depth_expanded = depth` et `confidence = mask`. Le chemin
`inputs: [B, 9, H, W]` reste disponible uniquement via
`include_splatted_depth=False` pour les ablations sparse-only.

Les chemins downstream PyTorch/ONNX/demo ne doivent pas reslicer `inputs` à la
main : utiliser l'adaptateur explicite, qui valide `input_channels` avant de
retourner les tenseurs attendus par le modèle.

```python
from rgb_lidar_fusion.model_batch import (
    dataset_items_to_model_batch,
    model_batch_to_baseline_inputs,
    model_batch_to_torch_tensors,
)

batch = dataset_items_to_model_batch(items)
arrays = model_batch_to_baseline_inputs(batch, lidar_mode="enriched")
# arrays["rgb"]: [B, 3, H, W]
# arrays["lidar_maps"]: [B, 8, H, W]

tensors = model_batch_to_torch_tensors(arrays)  # optional; requires group `ml`
output = BaselineFusionModel(lidar_mode="enriched")(**tensors)
```

### Training baseline CPU-safe / GPU-ready

La boucle minimale d'entraînement reste volontairement légère : stdlib + PyTorch,
seed déterministe, sélection `cuda` si disponible sinon `cpu`, smoke synthétique
CI-safe, checkpoints et métriques hors Git sous `results/training/` par défaut.
Elle passe toujours par le contrat canonique dataset→modèle :
`dataset_items_to_model_batch(...)`, `model_batch_to_baseline_inputs(...,
lidar_mode="enriched")`, puis `model_batch_to_torch_tensors(...)`.

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev -G ml
PDM_IGNORE_ACTIVE_VENV=1 pdm run train-baseline -- --config configs/training/synthetic_smoke.yaml
```

Le runner imprime un diagnostic explicite de sélection device, par exemple
`device=cpu requested=auto cuda_available=False` en CI sans GPU. Lorsqu'un GPU
CUDA est sélectionné, le même diagnostic inclut aussi le nom matériel via
`cuda_device_name='...'`, afin de rendre le run dédié traçable dans les logs.

Sorties générées ignorées par Git :

```text
results/training/synthetic_smoke/checkpoints/latest.pt
results/training/synthetic_smoke/metrics.json
results/training/synthetic_smoke/metrics.csv
results/training/synthetic_smoke/run_metadata.json
```

Reprise :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm run train-baseline -- \
  --config configs/training/synthetic_smoke.yaml \
  --resume-from results/training/synthetic_smoke/checkpoints/latest.pt
```

`configs/training/kitti_tiny.yaml` documente le prochain incrément local KITTI,
mais le runner actuel reste synthétique tant qu'un schéma de pseudo-targets KITTI
n'est pas validé.

Les dépendances lourdes ONNX / TensorRT ne sont pas installées par défaut. Elles sont isolées par groupes PDM ou par setup runtime dédié au moment des milestones correspondants. Voir `docs/dependency-roadmap.md`, `docs/gpu-training-plan.md`, et `docs/onnx-export-validation.md`.

Pour l'export ONNX intégré, installer les groupes optionnels `ml` et `onnx`, puis lancer l'export et la validation de parité synthétique :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev -G ml -G onnx
PDM_IGNORE_ACTIVE_VENV=1 pdm run export_onnx
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate_onnx
```

Les scripts utilisent `BaselineFusionModel(output_dim=4, lidar_mode="enriched")`
avec des tenseurs déterministes `rgb: [2, 3, 32, 48]` et
`lidar_maps: [2, 8, 32, 48]`, puis vérifient la sortie ONNX
`prediction: [2, 4]` et la parité ONNX Runtime. Les fichiers générés
`results/onnx/baseline_fusion.onnx` et l'éventuel sidecar `.onnx.data` sont
ignorés par Git et ne doivent pas être commités.

## LiDAR local surface splatting — niveau 1

Le module `rgb_lidar_fusion.lidar_splatting` ajoute une représentation séparée des cartes sparse brutes :

```python
from rgb_lidar_fusion.lidar_splatting import SplattingConfig, splat_sparse_depth, splat_projected_depth

splat = splat_sparse_depth(
    sparse_depth=lidar_maps[0],
    sparse_mask=lidar_maps[5] > 0,
    config=SplattingConfig(radius_px=2, sigma_px=1.0),
)
depth_expanded = splat.depth_expanded
confidence = splat.confidence
```

Pour appeler le splatting directement depuis des points LiDAR déjà projetés :

```python
splat = splat_projected_depth(
    pixels=projected.pixels,              # [N, 2] en colonnes u, v
    depths=projected.camera_points[:, 2], # profondeur caméra positive
    image_shape=(height, width),
    config=SplattingConfig(radius_px=2, sigma_px=1.0),
)
```

Règle volontairement simple : chaque point LiDAR valide est copié dans un voisinage carré de rayon `radius_px`; la confiance suit une décroissance gaussienne `exp(-distance_px² / (2 * sigma_px²))`. En cas de chevauchement, la résolution est déterministe : la profondeur la plus proche gagne, puis la confiance la plus forte, puis l'ordre source row-major. La configuration identité `radius_px=0` représente le mode sparse comme enriched dégénéré : seul le pixel source reçoit `depth_expanded=depth` et `confidence=mask`.

Limites assumées pour cette étape : pas de propagation image-guidée/edge-aware, pas d'estimation de plan local, pas de surface concave, et aucune dépendance PyTorch/ONNX/TensorRT. Les sorties `depth_expanded` et `confidence` complètent les cartes sparse, elles ne les remplacent pas.

## Dataset

Dataset recommandé pour démarrer : **KITTI object detection**.

Les données ne doivent pas être commitées. Voir `data/README.md`.

## Status

Le socle intégré couvre maintenant la CI staged, la calibration/projection KITTI, le dataset sparse LiDAR léger, le splatting local déterministe, le baseline PyTorch enriched, le contrat dataset→modèle, l'adaptateur modèle, l'export/parité ONNX baseline, et les plans TensorRT. La prochaine cible review est la demo, puis une branche dédiée `feature/training-loop-baseline` pour préparer l'entraînement réel sur GPU.
