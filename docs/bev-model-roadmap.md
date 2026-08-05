# BEV model roadmap

Date: 2026-08-05.

## Question

Le baseline actuel fusionne une image RGB avec des cartes LiDAR sparse/enriched
dans le plan image. La prochaine direction possible est plus ambitieuse :
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
- une estimation de fiabilité locale via `confidence` ;
- une contrainte géométrique image -> 3D grâce à la calibration caméra/LiDAR.

Son rôle est de produire des features BEV camera-depth, pas seulement des boîtes
2D. Le splatting permet de contraindre beaucoup mieux le lift image -> BEV qu'une
approche monoculaire pure, car une profondeur est disponible sur une grande partie
des pixels utiles.

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
PointPillars/VoxelNet-like encoder
    |
    v
lidar_bev: [B, C2, Hbev, Wbev]

concat(camera_bev, lidar_bev)
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

## Étape 1 recommandée

Construire le socle BEV minimal :

```text
RGB + splatted depth/confidence
        |
        v
Depth-aware lift vers BEV
utilisant splatted depth + calibration + confidence
        |
        v
BEV camera-depth features

Point cloud
        |
        v
LiDAR encoder PointPillars/VoxelNet simplifié
        |
        v
BEV LiDAR features

BEV camera-depth features + BEV LiDAR features
        |
        v
Fusion BEV + petite head CenterPoint-like
```

Objectif de cette étape :

```text
Prouver que la représentation splattée + point cloud peut produire
une carte BEV cohérente et exploitable par une head 3D.
```

Cette étape ne nécessite pas encore une vraie head YOLO. Elle doit d'abord
stabiliser :

- le repère BEV ;
- la résolution BEV ;
- la projection image/depth -> BEV ;
- l'encodage point cloud -> BEV ;
- la concaténation/fusion des features BEV ;
- une head CenterPoint-like minimale.

## Étape 2 recommandée

Ajouter une head image auxiliaire de type YOLO.

Objectif :

- apprendre explicitement la sémantique image ;
- fournir une supervision 2D plus facile à obtenir/inspecter ;
- vérifier la cohérence entre détections 2D et boîtes 3D projetées ;
- éviter que les features image ne soient utilisées uniquement comme texture
  faiblement contrainte.

Cette head peut rester légère au début. Elle n'a pas besoin d'être un YOLO
complet pretrained tant que le contrat multimodal n'est pas stable.

## Étape 3 recommandée

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

## Décisions à prendre avant implémentation

- dataset cible initial : KITTI object, nuScenes mini, ou dataset synthétique ;
- classes initiales : voiture seule, puis piéton/cycliste ;
- repère BEV : limites x/y/z, résolution en mètres par cellule ;
- convention d'axes : véhicule/KITTI/caméra ;
- représentation LiDAR BEV : pillars simples, voxels, ou histogrammes manuels ;
- méthode de lift image-depth -> BEV : scatter dur, pooling, ou features
  pondérées par `confidence` ;
- première cible de training : heatmap BEV synthétique, pseudo-target KITTI, ou
  vraies annotations 3D.

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

Créer une branche dédiée, par exemple :

```text
feature/bev-fusion-prototype
```

Livrables minimaux :

```text
src/rgb_lidar_fusion/bev_projection.py
src/rgb_lidar_fusion/bev_model.py
tests/test_bev_projection.py
tests/test_bev_model.py
configs/training/bev_synthetic_smoke.yaml
```

Critère de réussite du premier incrément :

```text
Un batch synthétique RGB + enriched LiDAR + point cloud produit
camera_bev, lidar_bev, fused_bev et des sorties CenterPoint-like
avec shapes validées et backward PyTorch fonctionnel.
```
