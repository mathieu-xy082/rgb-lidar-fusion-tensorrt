# Séance 3 — Cartes LiDAR sparse et enriched

## Objectif

Comprendre le passage :

```text
cartes LiDAR sparse 6 canaux
→ splatting local
→ cartes LiDAR enriched 8 canaux
```

## Fichiers à connaître

```text
src/rgb_lidar_fusion/lidar_splatting.py
src/rgb_lidar_fusion/project_lidar.py
src/rgb_lidar_fusion/model_batch.py
tests/test_lidar_splatting.py
tests/test_model_batch.py
```

## Étape A — Lire le module de splatting

```bash
sed -n '1,240p' src/rgb_lidar_fusion/lidar_splatting.py
```

À repérer :

- `SplattingConfig`
- `SplattingResult`
- `splat_sparse_depth(...)`
- `splat_projected_depth(...)`
- `depth_expanded`
- `confidence`

## Idée clé

Les cartes sparse contiennent des points très ponctuels :

```text
point_mask = 1 seulement sur les pixels LiDAR projetés
```

Le splatting produit deux cartes supplémentaires :

```text
depth_expanded : profondeur étalée localement
confidence      : confiance de cette expansion
```

## Étape B — Lire les tests

```bash
sed -n '1,240p' tests/test_lidar_splatting.py
```

Question : que se passe-t-il si deux splats se chevauchent ?

## Étape C — Relier à `project_lidar.py`

Dans `build_enriched_lidar_maps(...)`, les 6 canaux sparse sont copiés inchangés, puis les canaux 6 et 7 sont ajoutés.

Important : dans le code actuel, `build_enriched_lidar_maps(...)` est surtout une fonction helper/testée. Les chemins opérationnels principaux utilisent directement `splat_sparse_depth(...)` :

- `scripts/smoke_project_lidar.py` construit les cartes sparse, puis appelle directement `splat_sparse_depth(...)` pour afficher les métriques de splatting ;
- `src/rgb_lidar_fusion/demo_artifacts.py` appelle directement `splat_sparse_depth(...)` pour écrire les artefacts splattés ;
- `src/rgb_lidar_fusion/model_batch.py` appelle directement `splat_sparse_depth(...)` pour construire les canaux enriched dans le batch modèle.

Donc ce n'est pas un bug fonctionnel : le pipeline enriched existe, mais il est câblé via `model_batch.py` plutôt que via `build_enriched_lidar_maps(...)`.

Question : pourquoi ne remplace-t-on pas la sparse depth brute ?
