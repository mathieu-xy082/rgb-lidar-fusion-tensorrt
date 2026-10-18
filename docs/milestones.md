# Milestones

Mise à jour : 2026-10-18.

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
| Pilotes camera-depth 512 frames | Modèles random/blocs entraînés séparément, 5 époques et 2050 steps chacun |
| Validation croisée camera-depth | Deux checkpoints évalués sur les mêmes masques random puis blocs, MAE/RMSE et couverture splattée |
| Accès aux features camera-depth | `forward_features()` expose `[B, 32, H, W]`, sans changer la sortie profondeur ni les checkpoints v1 |

## En cours

### Clôturer le backbone camera-depth v1

Le backbone est fonctionnel et évalué sous supervision sparse. Les pilotes
utilisent 410 frames d'entraînement et 102 de validation, avec des masques fixes.
Le modèle random est meilleur sur les points aléatoires ; le modèle entraîné
par blocs reconstruit mieux les trous spatiaux. Les résultats et limites sont
consignés dans le workstream, sans en déduire une profondeur dense fiable partout.

Reste avant clôture :

- [ ] Inspecter quelques frames identiques avec les deux checkpoints et les
  deux modes de holdout, notamment contours, objets proches et zones éloignées.
- [ ] Consigner les observations et clôturer v1 si aucune incohérence majeure
  n'empêche le prototype géométrique. Aucun nouveau seuil de MAE n'est requis.

La configuration exécutée est `320x1024`, batch size **1**, FP32. Les YAML des
pilotes gardent un défaut à 2 ; le notebook surcharge cette valeur. Après les
incidents Xid 62 avec le pilote `575.64.03`, le passage à `615.71.09` a été suivi
d'un test synthétique de 100 steps, de la fin du run random, du run blocs complet
et de la comparaison croisée. Cela ne prouve pas la cause des anciens crashs ni
une stabilité définitive. PyTorch reste `2.13.0+cu129` et AMP reste désactivé.

L'entraînement mixte, la sensibilité au taux de holdout/rayon/taille de blocs,
plusieurs seeds, davantage de données et la supervision dense restent des
améliorations non bloquantes pour le premier prototype BEV.

Suivi détaillé :
[workstreams/camera-depth-backbone.md](workstreams/camera-depth-backbone.md).

## Prochain

### Lift camera-depth vers BEV

Construire un module testable qui transforme les features image-depth en grille
BEV à partir de la calibration, de la profondeur et de la confiance.

Le raccord n'est pas encore implémenté. L'accès aux features du dernier décodeur
est disponible et testé, ainsi que la profondeur métrique, la calibration
redimensionnée et la rétroprojection caméra/LiDAR. Il reste à définir la
validité/confiance puis tester la grille et l'agrégation. La checklist technique
de référence est dans les
[prérequis du raccord BEV](bev-cam-depth-backbone.md#prerequis-du-raccord-bev).

Implémentation incrémentale :
[workstream camera-depth BEV lift](workstreams/camera-depth-bev-lift.md).

Premier critère de réussite :

```text
RGB + LiDAR enriched + calibration
-> camera-depth features
-> camera_bev: [B, C, Hbev, Wbev]
-> loss synthétique et backward finis
```

Commencer par une géométrie synthétique connue, puis mesurée/splattée avant
d'utiliser la profondeur apprise. Le premier backward doit atteindre les
features ; un scatter à indices discrets ne garantit pas un gradient vers la
profondeur via le placement des cellules.

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
