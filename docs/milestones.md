# Milestones

Mise à jour : 2026-11-03.

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
| Clôture du backbone v1 | Inspection held out de la frame `000366` et limites consignées ; prototype sparse-supervisé accepté |
| Premier raccord caméra BEV | Grille, masques, agrégation et gradients testés ; trois variantes inspectées sur `000000`, relecture terminée |
| Construction des piliers LiDAR | Entrée XYZI brute, grille 20 cm, neuf features, masques et sélection reproductible ; tests CPU et contrôle KITTI |

## Bilan camera-depth v1

### Prototype clôturé

Le backbone est fonctionnel et évalué sous supervision sparse. Les pilotes
utilisent 410 frames d'entraînement et 102 de validation, avec des masques fixes.
Le modèle random est meilleur sur les points aléatoires ; le modèle entraîné
par blocs reconstruit mieux les trous spatiaux. Les résultats et limites sont
consignés dans le workstream, sans en déduire une profondeur dense fiable partout.

Clôture acceptée le 2026-11-02 après inspection qualitative de la frame `000366`,
checkpoint epoch 5 / step 2050, holdout spatial. La structure prédite et une
grande partie des erreurs held out paraissent cohérentes ; une zone du bloc
gauche reste plus difficile. Les observations ne valident pas les pixels sans
vérité terrain LiDAR. Une inspection exhaustive des deux checkpoints sur des
frames/masques identiques reste un suivi facultatif, pas un test déclaré réalisé.

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

## Raccord caméra BEV clôturé

### Premier prototype géométrique

Construire un module testable qui transforme les features image-depth en grille
BEV à partir de la calibration, de la profondeur et de la confiance.

Un premier raccord fonctionnel est implémenté : features, profondeur/support,
calibration complète, rétroprojection et moyenne par cellule BEV. Les masques,
bords, collisions et gradients vers le backbone sont testés. Une frame KITTI a
été inspectée sur CPU avec le checkpoint existant et le LiDAR comme référence,
sans nouvel entraînement. Les variantes sparse, splattée non pondérée et
splattée pondérée sur `000000` ont été jugées visuellement cohérentes. La
relecture du 2026-11-02 n'a trouvé aucun défaut bloquant : 357 tests réussis,
un test CUDA ignoré. La branche `ec/camera-depth-bev-lift` a été intégrée dans
`main` par le merge `8290530`, puis publiée dans la release `1.2.0`. La checklist
technique de référence est dans les
[prérequis du raccord BEV](bev-cam-depth-backbone.md#prerequis-du-raccord-bev).

Implémentation incrémentale :
[workstream camera-depth BEV lift](workstreams/camera-depth-bev-lift.md).

Premier critère de réussite atteint sur les cas synthétiques :

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

Le module rend explicites :

- l'étendue métrique et la résolution de la grille BEV ;
- les conventions de repères caméra, véhicule et BEV ;
- le traitement des pixels sans profondeur fiable ;
- l'agrégation de plusieurs pixels dans une même cellule ;
- les shapes BEV fixées par la configuration ; l'export du lift reste à valider.

Inspection reproductible : `scripts/inspect_camera_bev.py`. Voir le
[contrat et la commande](workstreams/camera-depth-bev-lift.md#reproduce-the-inspection).

Architecture de référence :
[bev-model-roadmap.md](bev-model-roadmap.md).

## Prochaines étapes

### Chantier en cours : encodeur LiDAR BEV

Le [design LiDAR](bev-lidar-encoder.md) retient des piliers de 20 cm et un
contexte pré-fusion visé d'environ 3,8 x 3,8 m. Le lift caméra sera recalculé
sur la même grille ; pour la ROI initiale proposée, elle passe de 100 x 100
à 250 x 250 cellules. Le PFN de référence proposé et le candidat à convolutions
dilatées restent à implémenter et à valider, sans détection apprise à ce stade.

La construction déterministe des piliers est implémentée sans réseau neuronal :
les compteurs avant sélection, masques, indices source et features des points
retenus sont explicites. Validation du 2026-11-03 : 415 tests réussis et un test
CUDA ignoré, plus un contrôle CPU de la frame KITTI `000000`. Aucun entraînement
LiDAR ni nouveau checkpoint n'a été produit.

Le [workstream LiDAR](workstreams/lidar-bev-encoder.md#progress) regroupe les
tâches par périmètre : piliers/contrat, PFN/scatter, contexte, inspection et
mesures de ressources. Le prochain incrément sera le PFN puis le scatter de
features, avec prise en compte du padding dans la normalisation et l'agrégation.

### Ordre des incréments

1. Ajouter l'encodeur du point cloud de type PointPillars vers `lidar_bev`,
   avec son contexte spatial avant fusion et ses masques de mesures.
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
