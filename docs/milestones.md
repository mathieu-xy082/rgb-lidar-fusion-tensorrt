# Milestones

Mise à jour : 2026-08-17.

Ce document est la source de vérité sur l'avancement du projet. Les détails de
branches et les comptes rendus d'expériences restent dans les workstreams.

## Terminé

| Jalon | Résultat |
|---|---|
| Géométrie KITTI | Lecture de calibration, projection du point cloud et visualisations |
| Cartes LiDAR sparse | Contrat `[6, H, W]` : profondeur, xyz véhicule, intensité, masque |
| Splatting local | Contrat enriched `[8, H, W]` avec profondeur étendue et confiance |
| Dataset et adaptateurs | Batches NumPy-first, contrat nominal RGB + LiDAR enriched |
| Baseline PyTorch historique | Forward, training synthétique, checkpoints et métriques |
| Export ONNX baseline | Export déterministe et validation de parité ONNX Runtime |
| Outillage TensorRT | Diagnostics, stubs build/infer et schéma p50/p95/FPS |
| Artefacts de démonstration | Bundle synthétique reproductible et ignoré par Git |
| Backbone camera-depth v1 | Sortie dense `[B, 1, H, W]`, holdout déterministe et loss masquée |
| Smoke training KITTI | 64 frames, CPU et GPU, checkpointing et métriques |

## En cours

### Stabiliser la représentation camera-depth

L'implémentation prouve que le pipeline complet fonctionne, mais pas encore que
la profondeur prédite généralise.

Critères avant de considérer cette représentation comme suffisamment validée :

- séparer clairement entraînement et validation ;
- ajouter des métriques de profondeur sur les points holdout ;
- visualiser les prédictions, erreurs et masques de confiance ;
- vérifier la sensibilité au taux de holdout et au rayon de splatting ;
- reproduire un run GPU après redémarrage de la machine ;
- documenter les limites de la supervision sparse.

La configuration active est `320x1024`, batch size 2. Elle tient sur le GPU
local de 4 GiB avec peu de marge. Un essai mixed precision a provoqué un
`NVIDIA Xid 62` ; AMP reste donc désactivé tant que la stabilité CUDA n'a pas
été revalidée.

Suivi détaillé :
[workstreams/camera-depth-backbone.md](workstreams/camera-depth-backbone.md).

## Prochain

### Lift camera-depth vers BEV

Construire un module testable qui transforme les features image-depth en grille
BEV à partir de la calibration, de la profondeur et de la confiance.

Premier critère de réussite :

```text
RGB + LiDAR enriched + calibration
-> camera-depth features
-> camera_bev: [B, C, Hbev, Wbev]
-> loss synthétique et backward finis
```

Le module doit rendre explicites :

- l'étendue métrique et la résolution de la grille BEV ;
- les conventions de repères caméra, véhicule et BEV ;
- le traitement des pixels sans profondeur fiable ;
- l'agrégation de plusieurs pixels dans une même cellule ;
- les shapes statiques nécessaires au futur export.

Architecture de référence :
[bev-model-roadmap.md](bev-model-roadmap.md).

## Plus tard

1. Ajouter un encodeur du point cloud de type PointPillars/VoxelNet vers
   `lidar_bev`.
2. Fusionner `camera_bev` et `lidar_bev` dans un backbone BEV commun.
3. Ajouter des heads CenterPoint-like : centre, offset, hauteur, dimensions,
   orientation, classe et éventuellement vitesse.
4. Évaluer une head YOLO-like auxiliaire sans en faire un prérequis bloquant de
   la détection 3D.
5. Définir les losses multi-tâches, le parsing des annotations KITTI 3D et les
   métriques de détection.
6. Exporter le modèle entraîné vers ONNX puis TensorRT et mesurer latence,
   mémoire et précision.

## Hors périmètre immédiat

- revendiquer une performance SOTA ;
- entraîner directement un détecteur 3D complet sans valider les représentations
  intermédiaires ;
- utiliser les détections 2D comme filtre dur empêchant la branche BEV de
  proposer un objet ;
- versionner datasets, checkpoints, ONNX ou engines TensorRT.
