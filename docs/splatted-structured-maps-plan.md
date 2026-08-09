# Structured splatted LiDAR maps plan

Branche dédiée : `ec/splatted-structured-maps`

Base de référence : `origin/ec/onboarding`

## Objectif

Ajouter un artefact ML structuré `splatted_maps_<id>.npz` contenant les cartes LiDAR après splatting, séparées numériquement, pour que le code d'entraînement/inférence consomme des canaux explicites plutôt que des overlays visuels.

## Contexte

Le repo possède déjà :

- `src/rgb_lidar_fusion/lidar_splatting.py` avec `SplattingConfig`, `splat_sparse_depth`, `splat_projected_depth` ;
- `src/rgb_lidar_fusion/demo_artifacts.py` qui écrit déjà des artefacts synthétiques `splatted_maps/synthetic_splatted_maps.npz` ;
- `docs/onboarding/scripts/render_splatting_ppm.py` qui recalcule actuellement les splats pour visualiser ;
- `docs/onboarding/scripts/render_kitti_sequence.py` qui boucle sur les frames KITTI réelles ;
- `src/rgb_lidar_fusion/dataset.py` qui expose aujourd'hui un dataset sparse par défaut.

Distinction à conserver :

```text
overlay_<id>.ppm           = visualisation humaine/debug
sparse_maps_<id>.npz       = mesure LiDAR projetée sparse, fidèle à la source
splatted_maps_<id>.npz     = entrée ML structurée après splatting
```

## Contrat v1 proposé

Pour une frame KITTI `000003` :

```text
results/onboarding/kitti_000003/sparse_maps_000003.npz
results/onboarding/kitti_000003/splatted_maps_000003.npz
```

Clés minimales dans `splatted_maps_<id>.npz` :

```text
schema_version              "splatted-lidar-maps-v1"
source_sparse_maps          chemin source optionnel
splat_radius_px             int
splat_sigma_px              float
splat_conflict_strategy     string
sparse_lidar_maps           float32 [6, H, W]
splatted_lidar_maps         float32 [8, H, W]
splatted_channel_names      string array length 8
```

Canaux `splatted_lidar_maps` v1 :

```text
0 normalized_camera_depth_sparse
1 normalized_vehicle_x_sparse
2 normalized_vehicle_y_sparse
3 normalized_vehicle_z_sparse
4 intensity_sparse
5 point_mask_sparse
6 normalized_camera_depth_splatted
7 splat_confidence
```

Raison du choix v1 : conserver exactement les 6 canaux sparse existants, puis ajouter uniquement les deux canaux splattés déjà bien définis (`depth_expanded`, `confidence`). Les canaux `x/y/z/intensity` splattés sont reportés à une v2, car il faut définir précisément leur sémantique en cas de conflits/recouvrements.

## Tâches requises

### T1 — Helper de construction `[8,H,W]`

Ajouter une fonction centralisée, probablement dans `src/rgb_lidar_fusion/lidar_splatting.py` ou `src/rgb_lidar_fusion/splatted_maps.py` :

```python
def build_splatted_lidar_maps(sparse_lidar_maps, *, config=None): ...
```

Critères :

- entrée `float32 [6,H,W]` ;
- sortie `float32 [8,H,W]` ;
- les 6 premiers canaux sont identiques au sparse ;
- canal 6 = `depth_expanded` ;
- canal 7 = `confidence` ;
- tests sur shape, channel order, erreurs d'entrée.

### T2 — Save/load `.npz` versionné

Ajouter helpers :

```python
save_splatted_lidar_maps_npz(...)
load_splatted_lidar_maps_npz(...)
```

Critères :

- `schema_version` stable ;
- noms de canaux persistés ;
- configuration de splatting persistée ;
- roundtrip testé.

### T3 — Étendre `scripts/smoke_project_lidar.py`

Ajouter :

```bash
--splatted-output <path>
--splat-radius-px 2
--splat-sigma-px 1.0
```

Critères :

- `--sparse-output` reste compatible ;
- `--splatted-output` écrit le nouveau `.npz` ;
- logs explicites : `splatted_output=...`, `splatted_lidar_maps_shape=(8,H,W)`.

### T4 — Génération automatique dans la séquence KITTI

Mettre à jour `docs/onboarding/scripts/render_kitti_sequence.py` pour produire par frame :

```text
splatted_maps_<id>.npz
```

Critères :

- commande/étape équivalente affichée avec `COMMAND:` ;
- sortie ajoutée dans le répertoire `results/onboarding/kitti_<id>/` ;
- la galerie continue de fonctionner.

### T5 — Visualisation depuis `.npz` précomputé

Mettre à jour `docs/onboarding/scripts/render_splatting_ppm.py` :

```bash
--splatted-npz results/onboarding/kitti_000003/splatted_maps_000003.npz
```

Critères :

- si `--splatted-npz` est fourni, ne pas recalculer le splatting ;
- sinon, comportement actuel conservé ;
- tests ou smoke commands documentés.

### T6 — Dataset/model input opt-in

Ajouter un mode explicite :

```python
KittiSparseLidarDataset(..., lidar_representation="sparse")
KittiSparseLidarDataset(..., lidar_representation="splatted")
```

Critères :

- `sparse` reste le défaut ;
- `splatted` retourne `lidar_maps [8,H,W]` ;
- `meta["lidar_representation"]` et `meta["lidar_map_channels"]` reflètent le mode ;
- tests dataset/model batch mis à jour.

### T7 — Documentation onboarding

Mettre à jour `docs/onboarding/seance-04-mini-kitti-reel.md` :

- ajouter `splatted_maps_<id>.npz` aux sorties ;
- expliquer que c'est l'artefact ML, pas l'overlay ;
- ajouter une commande d'inspection `np.load(...)`.

### T8 — Validation finale

Commandes attendues :

```bash
pdm run validate
pdm run python docs/onboarding/scripts/render_kitti_sequence.py --ids 000000 --alpha 0.45
python - <<'PY'
import numpy as np
z = np.load('results/onboarding/kitti_000000/splatted_maps_000000.npz')
assert z['splatted_lidar_maps'].shape[0] == 8
print('ok')
PY
```

## Définition de fini

- `splatted_maps_<id>.npz` est généré pour des frames KITTI réelles ;
- le schéma est stable, versionné et auto-descriptif ;
- `splatted_lidar_maps.shape == (8,H,W)` ;
- dataset/model input peut choisir sparse vs splatted ;
- séance 4 documentée ;
- `pdm run validate` passe ;
- la branche `ec/splatted-structured-maps` est poussée ;
- alerte finale exacte : `BRANCHE PRÊTE POUR REVIEW: ec/splatted-structured-maps`.
