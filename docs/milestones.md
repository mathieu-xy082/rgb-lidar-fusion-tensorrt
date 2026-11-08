# Milestones

Mise à jour : 2026-11-05.

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
| PFN et scatter LiDAR | Linear/BatchNorm/ReLU/max masqué, features BEV stride 1, comptes pré-cap conservés ; tests et backward CPU |
| Premier contexte LiDAR BEV | Deux blocs résiduels dilatés, support 19x19 vérifié, grille/mesures conservées ; comparaison multi-résolution CPU |
| Inspection conjointe et diagnostic LiDAR | Caméra/LiDAR sur grille 20 cm, couverture hors FOV préservée ; premier forward/backward GPU batch 1 et pic mémoire mesurés |
| Diagnostic centre LiDAR | Incrément prototype clôturé : mémorisation quatre frames, pilote 410/102 de 20 epochs CUDA, best epoch 9 et revue TP/FN/FP |
| Premier bloc de fusion BEV | Grilles explicites, concaténation features/supports, projection 1x1, compteurs conservés et backward CPU ; sans backbone spatial commun ni assemblage multimodal |
| Premier backbone BEV commun | Pyramide résiduelle 32/64/96, sortie stride 1 alignée, champ théorique 39/43 cellules, backward et mesure CPU ; sans assemblage aux encodeurs ni entraînement |
| Assemblage multimodal BEV | Adaptateur CPU, encodeurs/lift/fusion/backbone/head centre raccordés, backward synthétique, audit des sources et checkpoints distincts ; sans entraînement KITTI multimodal |

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

## Bilan LiDAR v1.3.0

La branche `ec/lidar-bev-encoder` est intégrée dans main et publiée dans la
release 1.3.0. Les contrôles ci-dessous retracent ses incréments successifs :
les mesures à poids aléatoires précèdent le pilote supervisé, sans le remplacer.

### Encodeur LiDAR BEV

Le [design LiDAR](bev-lidar-encoder.md) retient des piliers de 20 cm et un
contexte pré-fusion visé d'environ 3,8 x 3,8 m. Le lift caméra est recalculé
sur la même grille pour l'inspection conjointe : 250 x 250 cellules,
contre 100 x 100 dans les anciens artefacts à 0,50 m. Le PFN, le scatter et le
premier bloc de contexte dilaté sont implémentés et testés sur CPU.
Les premiers contrôles ci-dessous
n'utilisaient pas encore de supervision objet.

La construction déterministe des piliers est implémentée sans réseau neuronal :
les compteurs avant sélection, masques, indices source et features des points
retenus sont explicites. Validation du 2026-11-03 : 415 tests réussis et un test
CUDA ignoré, plus un contrôle CPU de la frame KITTI `000000`. Ce contrôle
initial n'avait produit ni entraînement LiDAR ni nouveau checkpoint.

Le PFN à 32 canaux et le scatter sont maintenant testés : 472 tests réussis,
un test CUDA ignoré. Padding exclu de BatchNorm/max, politique explicite pour
zéro/un point valide, cartes BEV et gradients finis, compteurs bruts conservés.
Le contrôle KITTI `000000` utilise des poids aléatoires, sans optimizer step ;
il ne constitue pas un résultat de détection ni une mesure de performance GPU.

Le contexte résiduel à dilatations `(1, 2, 2, 4)` est aussi implémenté : support
19x19 vérifié, sortie à stride 1 et mesures brutes conservées. La suite complète
atteint 517 tests réussis et un test CUDA ignoré. Une variante multi-résolution
est plus rapide dans le contrôle CPU local, mais le bloc dilaté reste la première
référence fonctionnelle pour son champ uniforme et son alignement direct ;
aucun optimum latence/précision ni déploiement GPU n'est démontré.

Le [workstream LiDAR](workstreams/lidar-bev-encoder.md#progress) regroupe les
tâches par périmètre : piliers/contrat, PFN/scatter, contexte, inspection et
mesures de ressources. L'inspection conjointe sur la grille de 20 cm et le
diagnostic GPU LiDAR sont maintenant disponibles : 528 tests réussis et un test
CUDA ignoré localement, plus un contrôle matériel distinct. Le garde-fou LiDAR
est de 2 048 MiB libres ; celui de la caméra reste à 2 500 MiB. Le premier
contrôle mesure 11,0 ms en forward, 35,3 ms en forward/backward et 180,5 MiB de
pic PyTorch, sans caméra ni optimiseur. Le grouping CPU prend 95,7 ms séparément :
ni temps réel global ni stabilité longue durée ne sont démontrés.

### Diagnostic supervisé facultatif

Le notebook `04_lidar_bev_analysis.ipynb` inspecte le LiDAR, ses features, les
boîtes KITTI et les cibles. Il utilise un modèle CPU à poids aléatoires, sans
réseau ni optimiseur automatique. Une head séparée prédit les centres Car et
leurs offsets XY ; sa loss focal/L1 et le backward complet sont testés.

Les 512 frames disposent maintenant de leurs labels officiels : 1 947 labels
Car, dont 1 715 centres dans la ROI. L'audit ne relève pas d'erreur de format,
de correspondance ou de construction des cibles ; cela ne garantit pas la
perfection des annotations humaines. Le hors-champ, DontCare et les classes
ignorées ne sont pas des négatifs implicites. Données et rapports restent hors Git.

La mémorisation sur `000006`, `000036`, `000123`, `000366` est validée :
200 epochs sur CUDA, 800 updates, 13 centres appariés, aucun faux positif au
seuil 0,3 dans le domaine annoté. Ce résultat porte sur les frames apprises.

Le notebook `06` a terminé 20 epochs CUDA, soit 8 200 updates, sur 410 frames
train / 102 validation disjointes, en partant de poids neufs. Le minimum de loss
validation est à l'epoch 9 : 2,01414, précision 71,66%, rappel 63,77%,
220 TP / 87 FP / 125 FN au seuil 0,3 et rayon d'appariement de 1 m. L'écart XY
moyen de 0,18851 m ne concerne que les centres appariés. À l'epoch 20, la loss
train diminue encore mais la loss validation remonte à 3,49954 : surapprentissage.

Les frames de validation `000009` (centre retrouvé), `000148` (centre manqué)
et `000388` (faux positif) ont été inspectées. Retenir `best.pt` à l'epoch 9,
pas `latest.pt` à l'epoch 20, pour l'inspection et la future réutilisation.
Le pic d'allocation PyTorch rapporté est 212,40 MiB ; ce n'est pas la mémoire
totale GPU ni une preuve de temps réel. L'indépendance des scènes reste à établir.

Décision : incrément prototype accepté pour la chaîne entraînement/validation,
sans revendication de détecteur final. Prochaine action : revenir au raccord
caméra/LiDAR BEV. Optimiser davantage cette head reste facultatif.
Suivi : [diagnostic centre LiDAR](workstreams/lidar-center-diagnostic.md).

## Prochaines étapes

### Chantier en cours : fusion et backbone BEV commun

Le [workstream de fusion](workstreams/bev-fusion-backbone.md) part de la release
1.3.0, sur `ec/bev-fusion-backbone`. Le premier incrément documentaire est
terminé : même grille de 20 cm et mêmes frames, features caméra/LiDAR à
32 canaux, deux masques de support distincts, projection 1x1 de 66 vers 32.
Le lift initial utilise la profondeur splattée pondérée ; compteurs et masse
restent des métadonnées séparées. Le bloc de fusion cellule par cellule est
maintenant implémenté : `BEVFeatureFusion` et sa sortie `FusedBEVFeatures`, avec
81 tests CPU. Le backbone commun `SharedBEVBackbone` ajoute trois échelles
32/64/96, des skips résiduels et une sortie alignée à la grille de départ.
Ses 77 tests contrôlent notamment les dimensions impaires, le champ théorique
39/43 cellules et les gradients ; un contrôle CPU du module seul est disponible.
Les deux encodeurs sont maintenant assemblés dans `BEVFusionDiagnostic`, avec
adaptateur CPU, encodeurs gelés par défaut, audit des poids et nouveau format
de checkpoint. Les tests vérifient le backward et une reprise Adam synthétique ;
ils ne constituent pas un entraînement multimodal KITTI.

Le runner quatre frames et le notebook `07` sont disponibles : encodeurs gelés,
inspection CPU, logs par batch, GPU en sous-processus et reprise contrôlée.
Les quatre IDs viennent uniquement du train commun. Un smoke KITTI CPU d'une
époque passe ; la mémorisation GPU et son budget mémoire restent à vérifier.
L'incrément suivant sera le pilote 410/102 avec comparaison LiDAR seul.
La head centre/offset existante sert de diagnostic, pas de head 3D finale.
Le [correctif de provenance caméra](workstreams/camera-depth-provenance.md)
est intégré. Le nouveau run caméra de cinq epochs a été vérifié et inspecté ;
l'audit complet passe avec LiDAR `best.pt` et le split 410/102, sans exception
legacy ni réentraînement LiDAR. Les anciens checkpoints restent historiques.

### Ordre des incréments

1. Fusionner `camera_bev` et `lidar_bev`, ajouter le backbone BEV commun et
   valider le raccord avec le diagnostic centre existant.
2. Ajouter des heads CenterPoint-like complètes : centre, offset, hauteur, dimensions,
   orientation, classe et éventuellement vitesse.
3. Évaluer une head YOLO-like auxiliaire sans en faire un prérequis bloquant de
   la détection 3D.
4. Étendre les cibles et le parsing KITTI existants, définir les losses
   multi-tâches et les métriques de détection 3D.
5. Exporter le modèle entraîné vers ONNX puis TensorRT et mesurer latence,
   mémoire et précision.

## Hors périmètre immédiat

- revendiquer une performance SOTA ;
- entraîner directement un détecteur 3D complet sans valider les représentations
  intermédiaires ;
- utiliser les détections 2D comme filtre dur empêchant la branche BEV de
  proposer un objet ;
- versionner datasets, checkpoints, ONNX ou engines TensorRT.
