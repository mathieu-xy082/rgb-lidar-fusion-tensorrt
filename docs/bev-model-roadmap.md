# BEV model roadmap

Date: 2026-08-05.

Mise à jour : 2026-11-05.

Statut : **design de référence**. Le backbone camera-depth préalable est
implémenté, évalué sur les pilotes random/blocs 512 frames et clôturé après
inspection qualitative. Le premier lift BEV géométrique est implémenté, relu
et inspecté ; il est intégré dans main depuis la release 1.2.0. La branche
LiDAR est intégrée dans main depuis la release 1.3.0 : piliers, PFN, scatter,
contexte spatial, inspection conjointe à 20 cm, diagnostic GPU et pilote centre
Car 410/102 avec revue TP/FN/FP. Le contrat et un premier bloc de fusion
cellule par cellule sont définis et testés sur CPU ; le backbone spatial
commun, l'assemblage multimodal et les heads 3D complètes restent à développer.

Documents liés :

- [milestones du projet](milestones.md) ;
- [design du backbone camera-depth](bev-cam-depth-backbone.md) ;
- [design de l'encodeur LiDAR BEV](bev-lidar-encoder.md) ;
- [contrat et incréments de fusion BEV](workstreams/bev-fusion-backbone.md) ;
- [workstream camera-depth](workstreams/camera-depth-backbone.md).

## Question

Le dépôt a d'abord fusionné une image RGB avec des cartes LiDAR sparse/enriched
dans le plan image, puis ajouté un backbone camera-depth dense. La prochaine
direction est plus ambitieuse :
combiner la sémantique image de type YOLO, la profondeur dense-ish apportée par
le splatting, et une philosophie de détection 3D en BEV proche de CenterPoint.

La question centrale est :

```text
Comment passer de RGB + splatted depth + point cloud
vers une détection 3D BEV qui conserve à la fois
la sémantique image et la géométrie LiDAR native ?
```

## Conclusion actuelle

La direction recommandée n'est pas de construire une cascade dure :

```text
YOLO detections -> frustums -> CenterPoint
```

Cette cascade serait fragile : une erreur ou un oubli de la détection 2D
empêcherait la branche 3D de récupérer l'objet.

La direction recommandée est plutôt :

```text
RGB + splatted depth/confidence
        |
        v
Image/depth branch -> camera-depth BEV features

Point cloud
        |
        v
LiDAR BEV branch -> LiDAR BEV features

camera-depth BEV features + LiDAR BEV features
        |
        v
Fused BEV backbone
        |
        v
CenterPoint-like 3D detection head
```

Une head 2D de type YOLO peut ensuite être ajoutée comme supervision auxiliaire
ou prior soft, mais la décision 3D principale devrait rester dans l'espace BEV.

## Rôles des branches

### Image + splatted depth

Cette branche apporte :

- la sémantique dense de l'image RGB ;
- des indices de profondeur par pixel via `depth_expanded` ;
- une estimation heuristique de support/fiabilité du splatting via `confidence` ;
- une contrainte géométrique image -> 3D grâce à la calibration caméra/LiDAR.

Son rôle est de produire des features BEV camera-depth, pas seulement des boîtes
2D. Le splatting permet de contraindre beaucoup mieux le lift image -> BEV qu'une
approche monoculaire pure, car une profondeur est disponible sur une grande partie
des pixels utiles.

Ces rôles décrivent la cible : la loss de profondeur actuelle n'établit pas une
sémantique objet apprise, et `confidence` n'est pas une incertitude calibrée de
la profondeur prédite.

### Point cloud

Le point cloud ne doit pas être réduit à un validateur de frustums YOLO.

Il reste la source géométrique métrique native :

- points 3D réellement mesurés ;
- structure verticale ;
- densité locale ;
- hauteur des objets ;
- surfaces utiles pour dimensions, orientation et confiance 3D.

Même si le splatting densifie la profondeur dans l'image, il reste une
interpolation locale. Le point cloud brut garde donc un rôle central pour la
précision 3D.

### BEV CenterPoint-like head

La head BEV devrait prédire la détection 3D finale :

```text
heatmap_center
offset_xy
z
size_lwh
yaw_sin_cos
class_logits
optional_velocity
```

Cette tête reprend l'idée principale de CenterPoint : représenter les objets par
leurs centres dans la vue BEV, puis régresser les attributs 3D autour de ces
centres.

## Architecture cible

Schéma conceptuel :

```text
RGB image
splatted depth
confidence
calibration
    |
    v
Image encoder / light neck
    |
    v
Depth-aware lift to BEV
    |
    v
camera_bev: [B, C1, Hbev, Wbev]

point cloud
    |
    v
PointPillars-like encoder (piliers XY de 20 cm)
    |
    v
Contexte LiDAR avant fusion (cible : environ 3,8 m)
    |
    v
lidar_bev: [B, C2, Hbev, Wbev]

concat(camera_bev, lidar_bev, camera_support, lidar_observed_mask)
    |
    v
BEV fusion backbone
    |
    v
CenterPoint-like 3D heads
```

Plus tard :

```text
Image encoder / neck
    |
    +-- auxiliary YOLO-like 2D head
    |
    +-- depth-aware lift to BEV
```

La head YOLO-like ne devrait pas bloquer la branche BEV. Elle sert plutôt à
renforcer la sémantique image et à fournir des pertes auxiliaires ou des priors
soft.

## Phase BEV 1 - Lift camera-depth

Construire d'abord le chemin caméra vers BEV :

```text
RGB + splatted depth/confidence
        |
        v
Depth-aware lift vers BEV
utilisant splatted depth + calibration + confidence
        |
        v
BEV camera-depth features
```

Cette phase doit stabiliser le repère, la résolution, la projection
image/depth -> BEV, les règles d'agrégation et le traitement de la confiance.
Elle doit être testable sans dépendre encore d'un encodeur de point cloud ou
d'une head de détection.

Avant le raccord au backbone, suivre les
[prérequis du raccord BEV](bev-cam-depth-backbone.md#prerequis-du-raccord-bev) :
accès aux features compatible avec les checkpoints, profondeur caméra en
mètres, calibration redimensionnée complète, masque de validité et politique
de confiance, puis grille/agrégation et tests de géométrie/backward.
Le `forward()` du modèle retourne toujours uniquement la profondeur.
`forward_features()` expose désormais les features du dernier décodeur en
`[B, 32, H, W]` sur la même grille image. La conversion en mètres, la calibration
complète dans les batches, la rétroprojection caméra/LiDAR et la sélection
explicite de profondeur/support/confiance sont implémentées. Le premier lift
fonctionnel les raccorde à une grille métrique avec moyenne des features,
masques et compteurs ; les gradients vers le backbone sont testés et une frame
KITTI a été inspectée sur CPU. L'entraînement depth reste inchangé et le lift
n'est pas encore validé en temps réel ou à l'export. Suivi :
[workstream BEV lift](workstreams/camera-depth-bev-lift.md).

Le premier prototype peut utiliser une profondeur synthétique puis
mesurée/splattée sur son support connu pour isoler les erreurs de géométrie.
La profondeur apprise pourra être branchée ensuite avec le masque de support
explicite prévu par cette API. Perfectionner l'entraînement camera-depth n'est
pas un prérequis.

## Phase BEV 2 - Branche LiDAR et fusion

```text
Point cloud
        |
        v
Piliers XY de 20 cm -> PFN -> scatter -> contexte LiDAR
        |
        v
BEV LiDAR features

BEV camera-depth features + BEV LiDAR features
        |
        v
Fusion BEV + petite head CenterPoint-like
```

Objectif de cette phase :

```text
Prouver que la représentation splattée + point cloud peut produire
une carte BEV cohérente et exploitable par une head 3D.
```

La branche LiDAR doit rester une source géométrique autonome. La fusion ne doit
pas dépendre d'une détection 2D préalable.

Le [design de l'encodeur LiDAR](bev-lidar-encoder.md) retient des piliers de
0,20 x 0,20 m et une grille commune avec le lift caméra. Pour la ROI initiale
proposée de 50 x 50 m, les sorties ont 250 x 250 cellules. Le lift caméra
est recalculé à cette résolution pour l'inspection conjointe, sans interpoler
les anciennes cartes à 0,50 m.

Avant fusion, le bloc LiDAR vise un champ réceptif théorique d'environ
3,8 x 3,8 m. Quatre convolutions de dilatations `(1, 2, 2, 4)`, en deux blocs
résiduels à 32 canaux, sont implémentées et vérifiées sur CPU. Une variante
multi-résolution est comparée, sans validation de précision ou performance GPU.
Le PFN implémenté utilise BatchNorm sur les seuls points valides plutôt que la
variante LayerNorm initialement évoquée, avec une sortie à 32 canaux et un
scatter à stride 1. Le backbone BEV commun apportera ensuite le contexte
plus large. Un champ réceptif théorique ne prouve pas une reconnaissance
d'objet : le diagnostic supervisé ci-dessous constitue une validation distincte.
Le [workstream LiDAR](workstreams/lidar-bev-encoder.md) sépare les tâches de
construction des piliers, PFN/scatter, contexte et validation du raccord.

Un [diagnostic centre LiDAR facultatif](workstreams/lidar-center-diagnostic.md)
valide l'apprentissage de cette branche seule : heatmap Car et offset XY,
annotations KITTI, mémorisation quatre frames et pilote CUDA 410/102 terminé.
L'epoch 9 est retenue après revue TP/FN/FP. Ce n'est ni une tête obligatoire
avant fusion ni un détecteur 3D complet. Les mêmes encodeurs pourront
recevoir les gradients de la future tête finale, sans tête LiDAR séparée.

Le [contrat de fusion](workstreams/bev-fusion-backbone.md#fusion-contract)
fixe maintenant les deux entrées à 32 canaux sur la grille commune, deux
masques de support distincts et une projection 1x1 de 66 vers 32 canaux.
Compteurs de pixels, masse de splatting et comptes LiDAR pré-cap restent
séparés des activations. Un petit backbone multi-échelle commun et la head
centre existante permettront de valider le raccord avant les heads 3D finales.
Le bloc `BEVFeatureFusion` implémente cette projection, avec conservation des
compteurs et gradients testés. Le backbone spatial commun et l'assemblage avec
les encodeurs restent à développer ; il n'y a pas encore de modèle fusionné entraîné.

## Phase BEV 3 - Supervision image auxiliaire

Ajouter une head image auxiliaire de type YOLO.

Objectif :

- apprendre explicitement la sémantique image ;
- fournir une supervision 2D plus facile à obtenir/inspecter ;
- vérifier la cohérence entre détections 2D et boîtes 3D projetées ;
- éviter que les features image ne soient utilisées uniquement comme texture
  faiblement contrainte.

Cette head peut rester légère au début. Elle n'a pas besoin d'être un YOLO
complet pretrained tant que le contrat multimodal n'est pas stable.

## Phase BEV 4 - Raffinement objet

Ajouter un raffinement frustum/object-level.

Pour chaque prédiction BEV candidate :

1. projeter la boîte 3D ou son centre dans l'image ;
2. récupérer les features image autour du frustum ;
3. récupérer les features point cloud/BEV autour de l'objet ;
4. raffiner score, dimensions, hauteur, yaw et éventuellement vitesse.

Cette étape se rapproche de l'esprit du second stage de CenterPoint et des
approches frustum, mais elle intervient après une proposition BEV principale,
pas comme filtre obligatoire avant la détection 3D.

## Pertes possibles

### Pertes principales BEV

- focal loss ou BCE sur `heatmap_center` ;
- L1/SmoothL1 sur `offset_xy` ;
- L1/SmoothL1 sur `z` ;
- L1/SmoothL1 sur `size_lwh` ;
- L1/SmoothL1 ou loss angulaire sur `yaw_sin_cos` ;
- cross entropy ou BCE sur `class_logits`.

### Pertes auxiliaires image

- détection 2D type YOLO ;
- cohérence entre boîte 3D projetée et boîte 2D ;
- cohérence entre profondeur prédite/observée et `depth_expanded` ;
- pénalisation des prédictions 3D qui contredisent fortement les pixels à forte
  `confidence`.

## Décisions encore ouvertes pour le détecteur complet

- extension du pilote KITTI object actuel vers d'autres datasets/scènes ;
- extension de Car vers piéton/cycliste et autres classes ;
- limites x/y/z adaptées au dataset et à la plateforme ; la première ROI
  proposée réutilise l'inspection KITTI, avec des piliers de 20 cm ;
- détails et budget du backbone BEV commun ; PFN/contexte actuels à 32 canaux ;
- alternatives à la profondeur splattée pondérée retenue pour le premier
  raccord, notamment support explicite pour une profondeur prédite ;
- extension du diagnostic centre/offset aux attributs et métriques 3D complets.

## Risques

- Le splatting peut créer une fausse densité aux bords d'objets ou derrière les
  surfaces visibles.
- Une cascade dure YOLO -> frustum -> BEV peut perdre le rappel 3D.
- Une architecture trop exotique trop tôt rendra le debugging difficile.
- Le passage image -> BEV peut dominer la complexité avant même la head 3D.
- KITTI complet ajoute rapidement parsing labels, matching, métriques, NMS et
  visualisation 3D.

## Positionnement

Le projet ne devrait pas être présenté comme un YOLO modifié ni comme une
implémentation CenterPoint directe.

Le positionnement plus juste est :

> Détection 3D BEV multimodale, avec lift image-depth contraint par splatted
> LiDAR et fusion avec géométrie point cloud native, déployable vers ONNX /
> TensorRT.

## Prochain incrément concret

Le backbone v1 et le premier raccord géométrique sont intégrés dans main
(release 1.2.0). Les
[prérequis de raccord](bev-cam-depth-backbone.md#prerequis-du-raccord-bev) sont
implémentés ; le critère synthétique RGB + enriched LiDAR + calibration vers
`camera_bev`, avec grille explicite et backward fonctionnel, est atteint.

La branche LiDAR de la [phase BEV 2](#phase-bev-2---branche-lidar-et-fusion)
et son [diagnostic centre](workstreams/lidar-center-diagnostic.md) sont intégrés
dans main depuis la release 1.3.0. Le pilote 410/102 et sa revue qualitative
clôturent ce prototype, sans valider une qualité de détection finale.

La branche active est `ec/bev-fusion-backbone`. Suivre son
[workstream](workstreams/bev-fusion-backbone.md#progress) : contrat et bloc de
fusion terminés, puis backbone commun, assemblage, mémorisation quatre frames
et comparaison LiDAR seul/fusion sur le split sauvegardé. Préserver les
checkpoints existants et vérifier leurs IDs d'entraînement avant réutilisation.
Aucun nouveau cycle d'optimisation de la profondeur ni head YOLO n'est requis.
