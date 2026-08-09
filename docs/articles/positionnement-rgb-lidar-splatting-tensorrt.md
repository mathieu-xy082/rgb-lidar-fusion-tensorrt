# Positionnement initial — RGB/LiDAR splatting + TensorRT

Date : 2026-08-03.

## Question

Le travail entamé — projection caméra/LiDAR, cartes sparse enrichies, splatting,
pipeline orienté TensorRT — est-il une idée originale qui pourrait mériter une
publication ?

## Réponse courte

En l'état actuel, **l'idée générale n'est probablement pas originale au sens
publication ML/CV majeure**.

La littérature contient déjà beaucoup de variantes de :

- projection LiDAR ↔ caméra ;
- fusion image/LiDAR en vue image, vue BEV ou espace points ;
- depth completion à partir d'une image RGB et d'un LiDAR sparse ;
- ajout de canaux auxiliaires, densification ou propagation spatiale ;
- fusion optimisée pour la détection 3D.

En revanche, le projet peut devenir publiable si la contribution est resserrée sur
un angle spécifique, mesurable et insuffisamment couvert : **fusion RGB/LiDAR
simple, explicable et fortement optimisée pour l'inférence embarquée TensorRT**,
avec ablations propres et mesures temps réel.

## Où se situe notre approche

Le prototype actuel construit un chemin :

```text
image RGB KITTI
+ scan Velodyne
+ calibration
→ projection LiDAR dans l'image
→ sparse maps multi-canaux
→ splatting / densification locale
→ artefacts compatibles modèle CNN / export ONNX / TensorRT
```

Cela ressemble davantage à une famille de méthodes de **depth completion / early
fusion image-space** qu'à une approche moderne BEV/transformer.

### Proches voisins

| Voisin littérature | Lien avec notre travail | Différence possible |
|---|---|---|
| MV3D / AVOD | Fusion multi-vues avec image, BEV, front-view | Architectures lourdes orientées détection complète |
| Deep Continuous Fusion | Fusion continue entre features image et features 3D | Fusion feature-level plus sophistiquée que nos cartes image-space |
| PointPainting | Ajoute des infos image/sémantiques aux points LiDAR | Direction inverse : image → points plutôt que points → image |
| Sparse-to-Dense / DeepLiDAR | Image RGB + LiDAR sparse pour prédire depth dense | Notre objectif peut être des canaux utiles à perception temps réel, pas nécessairement depth dense supervisée |
| BEVFusion / TransFusion | Fusion moderne en BEV/transformer | Très performants mais plus complexes/lourds |
| SPLATNet | Splatting/lattice pour point clouds | Splatting plus général ; pas exactement pipeline caméra KITTI/TensorRT image-space |

## Ce qui n'est pas suffisant pour publier

Les éléments suivants seuls ne suffisent probablement pas :

- projeter un nuage LiDAR dans une image ;
- produire une carte depth sparse ;
- appliquer un splatting local ;
- visualiser des overlays ;
- exporter un petit réseau vers ONNX/TensorRT ;
- montrer que le pipeline fonctionne sur quelques frames KITTI.

Ces briques sont utiles pour apprendre, prototyper et industrialiser, mais elles
sont déjà très proches de techniques connues.

## Angles potentiellement publiables

### 1. Angle "embedded real-time fusion baseline"

Contribution possible : une baseline RGB/LiDAR explicable, faible coût,
exportable TensorRT, avec comparaison qualité/latence.

Il faudrait démontrer :

- latence GPU réaliste, éventuellement Jetson/Orin ou GPU embarqué comparable ;
- débit FPS avec batch=1 ;
- mémoire GPU ;
- coût des étapes prétraitement + réseau + post-traitement ;
- comparaison à des méthodes plus lourdes.

### 2. Angle "splatting ablation study"

Contribution possible : étude systématique des canaux sparse/splatted :

- depth brute ;
- confidence ;
- densité locale ;
- inverse depth ;
- distance au point mesuré ;
- rayon de splat adaptatif à la profondeur ;
- occlusion handling / z-buffer variants.

Il faudrait montrer qu'un choix simple améliore significativement une tâche aval.

### 3. Angle "robustesse opérationnelle"

Contribution possible : fusion robuste à des défauts réalistes :

- LiDAR low-channel / subsampling ;
- bruit calibration extrinsèque ;
- désynchronisation temporelle ;
- pluie/brouillard/occlusions ;
- trous de LiDAR par distance ou matériau.

Cet angle colle bien à une expertise AV/robotique appliquée.

### 4. Angle "distillation / teacher-student"

Contribution possible : utiliser une méthode lourde RGB/LiDAR comme teacher, puis
entraîner un modèle image-space/splatted léger comme student TensorRT.

Ce serait plus clairement ML :

```text
BEVFusion/TransFusion teacher
→ pseudo-labels/features
→ student CNN léger avec RGB + splatted LiDAR channels
→ TensorRT INT8
```

## Hypothèse de contribution réaliste pour le repo

Une contribution plausible serait :

> Une étude d'un pipeline RGB + LiDAR splatté, image-space, déployable TensorRT,
> évalué sur KITTI/nuScenes avec ablations de canaux et contraintes de latence,
> montrant un compromis qualité/latence favorable par rapport à des baselines plus
> lourdes pour un cas embarqué.

Ce serait probablement mieux positionné comme :

- workshop CVPR/ICCV/ECCV orienté autonomous driving / embedded perception ;
- conférence robotique appliquée ;
- article technique/engineering report solide ;
- éventuellement papier court si les résultats sont très propres.

## Verdict initial

| Question | Réponse |
|---|---|
| L'idée brute est-elle originale ? | Plutôt non. |
| Le repo a-t-il une valeur ML/AV sérieuse ? | Oui, s'il devient expérimentalement rigoureux. |
| Peut-on viser une publication ? | Possible, mais pas sur la simple projection/splatting. |
| Le meilleur angle ? | Baseline explicable, temps réel TensorRT, ablations + robustesse. |

## Prochaines étapes recommandées

1. Définir la tâche aval : depth completion, segmentation, détection 2D enrichie,
   détection 3D, ou perception proxy.
2. Choisir un benchmark reproductible : KITTI depth completion ou KITTI object.
3. Implémenter 2-3 baselines minimales : RGB seul, sparse depth, splatted depth.
4. Mesurer qualité + latence bout-en-bout.
5. Ajouter des ablations contrôlées : canaux, rayon de splat, confidence, low-LiDAR.
6. Ne parler de publication qu'après un tableau résultat clair.

## Risque principal

Le risque est de construire un pipeline techniquement propre mais trop proche de
méthodes établies pour être vendu comme nouveauté scientifique. Pour éviter cela,
il faut documenter très tôt :

- ce qu'on compare ;
- ce qu'on optimise ;
- quel compromis nouveau est démontré ;
- quelle contrainte terrain justifie l'approche.
