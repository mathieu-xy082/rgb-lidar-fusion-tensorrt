# RGB-LiDAR Fusion TensorRT Prototype

Prototype démonstrateur de fusion caméra/LiDAR pour perception AV/robotics.

Objectif : construire un pipeline complet et démontrable :

```text
KITTI RGB image
+ LiDAR projected sparse geometry maps
→ PyTorch detector
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
    "lidar_maps": Tensor[5, H, W],
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

### Milestone 4 — Export ONNX

- exporter le modèle ;
- vérifier les shapes ;
- comparer PyTorch vs ONNX Runtime ;
- documenter les limitations.

### Milestone 5 — TensorRT FP16

```text
PyTorch checkpoint
→ ONNX export
→ TensorRT engine FP16
→ inference TensorRT
→ benchmark latency p50/p95/FPS
```

## Structure

```text
configs/                    Configuration dataset/modèle
src/rgb_lidar_fusion/        Package Python principal
tests/                       Tests unitaires
notebooks/                   Exploration visuelle
results/                     Résultats générés non versionnés
scripts/                     Commandes utilitaires
```

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
pdm run validate  # test + smoke
```

Pour générer une visualisation synthétique sans dépendance image externe :

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm run python scripts/smoke_project_lidar.py \
  --overlay-output /tmp/rgb_lidar_synthetic_overlay.ppm \
  --sparse-output /tmp/rgb_lidar_synthetic_sparse_maps.npz
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

Les dépendances lourdes PyTorch / ONNX / TensorRT ne sont pas installées par défaut. Elles seront ajoutées par groupes PDM au moment des milestones correspondants. Voir `docs/dependency-roadmap.md`.

## Dataset

Dataset recommandé pour démarrer : **KITTI object detection**.

Les données ne doivent pas être commitées. Voir `data/README.md`.

## Status

Initial skeleton créé. La branche `feature/kitti-calibration-projection` couvre maintenant une première lecture testée des fichiers calibration KITTI (`P2`, `R0_rect`, `Tr_velo_to_cam`) et des nuages Velodyne `.bin`, puis projette les points vers l'image, génère des cartes sparse exportables en `.npz`, et produit un overlay PPM synthétique ou KITTI local visualisable sans dépendance lourde, soit sur fond noir, soit sur un canvas RGB PPM ASCII local. Prochaine cible : rendre la lecture PNG/JPEG optionnelle via une dépendance image légère ou un groupe PDM dédié.
