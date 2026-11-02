# BEV LiDAR encoder

Mise à jour : 2026-11-02.

Statut : **design, non implémenté**. Ce document distingue les choix retenus,
les paramètres proposés pour un premier prototype et les variantes à comparer.
Il ne décrit ni un détecteur entraîné ni une reproduction complète de PointPillars.

Documents liés :

- [architecture BEV cible](bev-model-roadmap.md) ;
- [contrats de la branche caméra](bev-cam-depth-backbone.md) ;
- [workstream LiDAR BEV](workstreams/lidar-bev-encoder.md) ;
- [milestones](milestones.md).

## Objectif et responsabilités

Construire une branche géométrique à partir du point cloud brut, indépendante
d'une détection 2D et compatible avec la grille du lift caméra existant.

```text
Point cloud XYZI
    |
    v
Piliers XY de 20 cm -> PFN -> scatter BEV
    |
    v
Bloc de contexte LiDAR avant fusion
    |
    v
lidar_bev: [B, C2, Hbev, Wbev]

camera_bev + lidar_bev
    |
    v
Fusion -> backbone BEV commun -> head CenterPoint-like
```

Le PFN (Pillar Feature Network) agrège les points d'un pilier. Le bloc de
contexte associe les piliers voisins avant la fusion. Le backbone commun
apporte ensuite un contexte plus large aux features multimodales.

Le bloc LiDAR ne doit donc pas rester uniquement local à l'échelle de quelques
piliers : nous visons un **champ réceptif théorique d'environ 3,8 x 3,8 m avant
fusion**. Ce contexte prépare l'apprentissage de formes d'objets, sans garantir
qu'une voiture entière, notamment selon son orientation, tienne dans cette zone.

## Choix retenus

- Représentation par piliers de **0,20 x 0,20 m**, sans subdivision verticale.
- Repère Velodyne/KITTI : x avant, y gauche, z haut ; pas de nouvelle
  transformation ou de compensation de gravité implicite.
- Réutilisation de `camera_bev.BEVGridConfig` : bornes XYZ semi-ouvertes,
  lignes suivant x, colonnes suivant y, même origine et mêmes centres de cellules.
- Grille commune caméra/LiDAR à 20 cm pour la première connexion à stride 1.
- Features LiDAR contextualisées avant fusion, puis backbone BEV commun.
- Premier incrément limité à une représentation testable : ni head de détection,
  ni nouvelle loss objet, ni promesse de précision ou de temps réel.

## Grille et raccord caméra

Pour conserver la zone utilisée lors de l'inspection caméra, la configuration
initiale proposée est x=[0,50), y=[-25,25), z=[-3,3), en mètres. Ces limites
restent configurables ; z est relatif à l'origine LiDAR, pas au sol.

Avec une résolution de 0,20 m, `Hbev = Wbev = 250` :

```text
row = floor((x - x_min) / 0.20)
col = floor((y - y_min) / 0.20)
```

Le point cloud est filtré par cette ROI, **pas par le champ de vue caméra**.
Les cellules hors couverture caméra peuvent donc conserver du support LiDAR.
Les masques des deux modalités restent distincts.

Les inspections caméra précédentes utilisaient 0,50 m, soit 100 x 100 cellules.
Le lift accepte déjà une résolution configurable : il faudra le recalculer à
0,20 m depuis les features image et leur support de profondeur, sans interpoler
la carte BEV 100 x 100. Changer ce binning ne nécessite pas, à lui seul, de
réentraîner le backbone camera-depth. L'inspection à 20 cm reste à effectuer ;
le défaut de la CLI reste actuellement à 0,50 m.

Le rapport des résolutions vaut 2,5 : un stride entier ne suffit pas à aligner
ces deux grilles. Une égalité de shape ne suffit pas non plus : axes, origine,
bornes, résolution et éventuel offset doivent coïncider. Certains frameworks
PointPillars utilisent les lignes pour y et les colonnes pour x, contrairement
à notre contrat ; une réutilisation demandera un adaptateur explicite.

## PFN de référence proposé

Pour chaque point retenu d'un pilier, les neuf features proposées reprennent
le PointPillars original :

```text
x, y, z, intensity,
x - mean_x, y - mean_y, z - mean_z,
x - pillar_center_x, y - pillar_center_y
```

Les coordonnées sont métriques dans le repère LiDAR ; les conventions
d'intensité doivent être explicites. Aucun canal de splatting image n'entre
dans ce PFN. Les moyennes sont calculées sur les points retenus, sans padding.

Premiers paramètres proposés, encore à valider :

- au plus 64 points par pilier, avec sélection reproductible en cas de
  dépassement et compteurs des points écartés ;
- PFN `Linear(9, 32) -> BatchNorm -> ReLU -> max par canal entre les points` ;
- scatter des vecteurs de piliers vers `[B, 32, 250, 250]` pour la ROI ci-dessus.

Le padding doit être exclu de l'agrégation et ne pas biaiser les statistiques
de normalisation ; un pilier vide ne doit pas créer de fausse observation.
La sélection, les masques et les cas vides seront testés avant le réseau complet.

**BatchNorm est la référence proposée**, contrairement à l'idée initiale de
LayerNorm. Un petit batch ne rend pas automatiquement BatchNorm inutilisable :
les statistiques peuvent agréger plusieurs points et piliers. LayerNorm reste
une variante possible, pas une amélioration démontrée. La largeur 32, contre
64 dans le PFN original, est un compromis proposé, pas une décision de précision.

## Contexte avant fusion : candidat dilaté

Quatre convolutions 3 x 3 de stride 1 sans dilation couvriraient seulement
9 x 9 cellules, soit 1,8 x 1,8 m. Ce contexte est inférieur à notre cible.

Un candidat léger conserve quatre convolutions, mais utilise les dilatations
`(1, 2, 2, 4)`, avec stride 1 et padding égal à la dilation :

| Couche | Dilation | Champ réceptif cumulé | Étendue en mètres |
|---|---:|---:|---:|
| Conv 1 | 1 | 3 x 3 | 0,6 x 0,6 |
| Conv 2 | 2 | 7 x 7 | 1,4 x 1,4 |
| Conv 3 | 2 | 11 x 11 | 2,2 x 2,2 |
| Conv 4 | 4 | 19 x 19 | 3,8 x 3,8 |

Pour ces convolutions à stride 1 : `R = 1 + sum((k_i - 1) * d_i) = 19`.
Le calcul porte sur les dépendances spatiales des convolutions, à l'intérieur
de la grille ; le padding ne fournit pas d'observations hors ROI.

À largeur constante, la dilation n'augmente pas le nombre de coefficients
des noyaux. Elle ne garantit cependant ni la même latence sur GPU, ni une
meilleure précision. Le champ réceptif effectif appris peut être plus restreint
que ce maximum théorique ; la sparsité LiDAR peut aussi limiter le contexte utile.

**Cette séquence est une proposition, pas un choix validé ni une architecture
PointPillars standard.** Elle devra être comparée à un petit backbone
multi-résolution. Si celui-ci change le stride, sa sortie et celle de la caméra
devront être réalignées explicitement. Largeur, normalisation du bloc et
connexions résiduelles restent à préciser avant son implémentation.

## Sorties et sens des masques

Contrat proposé :

| Sortie | Shape | Sens |
|---|---|---|
| `lidar_bev` | `[B, C2, Hbev, Wbev]` | Features après contexte LiDAR ; C2 proposé à 32 |
| `point_count` | `[B, 1, Hbev, Wbev]` | Mesures dans la ROI par cellule avant limitation des points |
| `observed_mask` | `[B, 1, Hbev, Wbev]` | Présence d'au moins une mesure LiDAR |

Les features peuvent devenir non nulles dans une cellule non observée grâce
aux convolutions voisines. Cela ne transforme pas cette cellule en mesure,
en espace libre ou en prédiction d'occupation. Les compteurs restent ceux des
mesures, pas ceux des activations. `C2` n'a pas besoin d'être égal au nombre de
canaux caméra `C1` pour une future concaténation.

## Validation et limites

Le [workstream](workstreams/lidar-bev-encoder.md#progress) détaille les incréments.
Les premiers critères sont géométriques et logiciels : cellules attendues,
bords, séparation des batches, padding, cas vides, valeurs finies, backward et
alignement caméra/LiDAR sur une grille non carrée pour détecter les transpositions.
Le champ réceptif convolutif sera également vérifié sur un cas contrôlé.

Un backward synthétique ne prouve aucune reconnaissance d'objet. Pour évaluer
l'autonomie de la branche, il faudra ultérieurement des annotations 3D, une
head et une évaluation LiDAR seul face à la variante fusionnée. Ce travail ne
bloque pas l'implémentation du premier encodeur.

Passer de 100 x 100 à 250 x 250 représente 6,25 fois plus de cellules pour les
tenseurs denses, pas nécessairement 6,25 fois la latence ou la mémoire globale.
Sur le GPU de 4 Go, commencer les mesures avec batch 1 et profiler forward,
backward et mémoire, sans extrapoler depuis la seule taille d'un tenseur.
L'export ONNX/TensorRT et le temps réel restent à valider séparément.

## Références

- [PointPillars, 2019](https://arxiv.org/abs/1812.05784) : PFN et backbone 2D
  multi-résolution ; notre cible n'en reproduit pas le détecteur complet.
- [Convolutions dilatées, ICLR 2016](https://arxiv.org/abs/1511.07122) :
  élargissement du contexte sans réduction de résolution ; la séquence proposée
  ici n'est pas une recette validée pour notre dataset.
- [PillarNeXt, CVPR 2023](https://arxiv.org/abs/2305.04925) : importance du
  backbone, du champ réceptif et de l'entraînement pour les approches par piliers ;
  ces résultats ne constituent pas une revendication SOTA pour notre projet.
