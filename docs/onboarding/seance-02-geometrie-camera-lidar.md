# Séance 2 — Géométrie caméra/LiDAR

## Objectif

Comprendre le chemin géométrique :

```text
point LiDAR 3D
→ transformation vers caméra
→ projection image
→ pixel u/v
→ filtrage des points visibles
```

## Fichiers à connaître

```text
src/rgb_lidar_fusion/calibration.py
src/rgb_lidar_fusion/project_lidar.py
scripts/smoke_project_lidar.py
tests/test_project_lidar.py
```

## Étape A — Lire les constantes et fonctions principales

Commandes :

```bash
cd .
sed -n '1,220p' src/rgb_lidar_fusion/project_lidar.py
```

À repérer :

- `LIDAR_MAP_CHANNELS`
- `ENRICHED_LIDAR_MAP_CHANNELS`
- `ProjectedLidar`
- `project_lidar_to_image(...)`
- `build_sparse_lidar_maps(...)`

## Étape B — Lancer le smoke de projection seul

```bash
pdm run smoke
```

À observer :

- taille image ;
- nombre de points projetés ;
- shape des cartes LiDAR sparse.

## Étape C — Lire le test de projection

```bash
sed -n '1,220p' tests/test_project_lidar.py
```

Question : quel type de point est filtré avant projection image ?

## Idée clé

Un point LiDAR doit être :

1. transformé dans le repère caméra ;
2. devant la caméra ;
3. projeté par la matrice caméra ;
4. dans les bornes image.

Sinon il ne devient pas un pixel utile.
