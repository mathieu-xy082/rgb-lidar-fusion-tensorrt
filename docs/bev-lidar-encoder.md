# BEV LiDAR encoder

Mise à jour : 2026-11-05.

Statut : **premier encodeur fonctionnel implémenté, inspecté sur CPU et testé
dans un diagnostic GPU borné**. La mémorisation sur quatre frames et un pilote
CUDA 410/102 de 20 epochs sont terminés ; la qualité n'est pas celle d'un
détecteur de production et l'indépendance des scènes n'est pas garantie.
Ce document distingue le contrat NumPy actuel, les choix retenus et les
paramètres/variantes à valider. Il ne décrit ni un détecteur entraîné ni une
reproduction complète de PointPillars.

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
- Socle encodeur limité à une représentation testable : aucune head ou loss
  objet obligatoire, ni promesse de précision ou de temps réel. Le diagnostic
  supervisé facultatif décrit ci-dessous reste séparé de ce contrat.

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
Le lift accepte une résolution configurable : l'inspection conjointe le recalcule
à 0,20 m depuis les features image et leur support de profondeur, sans interpoler
la carte BEV 100 x 100. Changer ce binning ne nécessite pas, à lui seul, de
réentraîner le backbone camera-depth. Le défaut caméra seul de la CLI reste
à 0,50 m. L'inspection conjointe est
maintenant disponible avec `--with-lidar --resolution 0.2`, dans un dossier
distinct des figures historiques.

Le rapport des résolutions vaut 2,5 : un stride entier ne suffit pas à aligner
ces deux grilles. Une égalité de shape ne suffit pas non plus : axes, origine,
bornes, résolution et éventuel offset doivent coïncider. Certains frameworks
PointPillars utilisent les lignes pour y et les colonnes pour x, contrairement
à notre contrat ; une réutilisation demandera un adaptateur explicite.

## Contrat de construction des piliers implémenté

`rgb_lidar_fusion.lidar_pillars.build_lidar_pillars` reçoit une liste ou un
tuple de nuages NumPy float32/float64 `[Ni, 4]`, avec Ni variable. Les nuages
vides `[0, 4]` et le batch vide sont acceptés. Aucun padding d'entrée ni filtre
de champ caméra n'est implicite. Les points sont convertis en float32 ; les
valeurs XYZI non finies ou non représentables dans ce format sont rejetées
avant le filtre de ROI. L'intensité est copiée sans normalisation ni clipping.

```python
from rgb_lidar_fusion.camera_bev import BEVGridConfig
from rgb_lidar_fusion.lidar_pillars import build_lidar_pillars
from rgb_lidar_fusion.project_lidar import load_velodyne_bin

grid = BEVGridConfig((0.0, 50.0), (-25.0, 25.0), (-3.0, 3.0), 0.20)
pillars = build_lidar_pillars(
    [load_velodyne_bin("data/kitti/training/velodyne/000000.bin")],
    grid=grid, max_points_per_pillar=64, seed=13,
)
```

Le résultat `PillarBatch` contient :

| Champ | Shape | Sens |
|---|---|---|
| `features` | `[P, M, 9]` | Features float32 des points retenus, padding nul |
| `indices` | `[P, 3]` | Indices int64 `(batch, row_x, col_y)` |
| `point_mask` | `[P, M]` | Points retenus, distincts du padding |
| `point_indices` | `[P, M]` | Indices dans le nuage d'origine ; -1 pour le padding |
| `point_count` | `[B, 1, Hbev, Wbev]` | Nombre de mesures par cellule avant limitation |
| `grid` | configuration | Grille métrique partagée fournie à l'appel |

P compte les piliers occupés de tout le batch ; M vaut 64 par défaut.
`retained_count[P]`, `dropped_count[P]` et `observed_mask[B,1,Hbev,Wbev]`
sont dérivés des indices, masques et compteurs. Les points hors ROI ne sont pas
comptés comme dépassements de capacité. Aucun plafond global de piliers n'est
ajouté. Les entrées ne sont pas modifiées.

Le tirage sans remise en cas de dépassement est local à chaque pilier et
reproductible à seed, batch et ordre d'entrée identiques. Il ne consomme pas
le RNG global et ne dépend pas du dépassement d'un autre pilier. La seed par
défaut est 13. Un changement de position dans le batch ou d'ordre des points
peut modifier la sélection. Les indices retenus sont remis dans l'ordre source.

L'indexation utilise le floor float32 et les gardes du lift caméra ; les rares
indices proches du bord supérieur arrondis hors grille sont exclus, pas clampés.
Ce prétraitement CPU n'est pas différentiable et n'est pas encore raccordé au
dataset ou au runner d'entraînement. L'import du module NumPy ne charge pas Torch ;
la configuration existante `BEVGridConfig` reste exposée par le module caméra.

## PFN et scatter implémentés

Pour chaque point retenu d'un pilier, les neuf features implémentées reprennent
le PointPillars original :

```text
x, y, z, intensity,
x - mean_x, y - mean_y, z - mean_z,
x - pillar_center_x, y - pillar_center_y
```

Les coordonnées sont métriques dans le repère LiDAR ; les conventions
d'intensité doivent être explicites. Aucun canal de splatting image n'entre
dans ce PFN. Les moyennes sont calculées sur les points retenus, sans padding.

Configuration actuelle :

- au plus 64 points par pilier par défaut, sélection et compteurs testés ;
- PFN `Linear(9, 32, bias=False) -> BatchNorm -> ReLU -> max par canal entre les points` ;
- scatter des vecteurs de piliers vers `[B, 32, 250, 250]` pour la ROI ci-dessus.

`lidar_bev.PillarFeatureNet` reçoit les features Torch `[P, M, 9]` et leur
masque booléen `[P, M]`. Seuls les points valides passent dans Linear,
BatchNorm1d (`eps=1e-3`, `momentum=0.01`) et ReLU ; le max est ensuite effectué
par pilier et canal. Le padding n'entre ni dans les statistiques ni dans
l'agrégation, même s'il contient NaN/Inf. Un pilier entièrement masqué rend
un vecteur nul. La largeur est configurable, avec 32 canaux par défaut.

**BatchNorm est la référence implémentée**, contrairement à l'idée initiale de
LayerNorm. Un petit batch ne rend pas automatiquement BatchNorm inutilisable :
les statistiques agrègent les points valides des différents piliers. En mode
training, avec zéro ou un seul point valide au total, le PFN utilise les
statistiques courantes sans les mettre à jour, ni incrémenter le compteur BN.
Les gradients vers les features et paramètres restent connectés. Les valeurs
initiales moyenne=0/variance=1 ne sont pas une calibration apprise.

LayerNorm reste une variante possible, pas une amélioration démontrée. La
largeur 32, contre 64 dans le PFN original, est un premier compromis logiciel,
pas un résultat validé en détection.

Le scatter reçoit un vecteur `[P, C]` par cellule occupée, les indices
`[P, 3]` et les compteurs pré-cap `[B, 1, Hbev, Wbev]`. Il conserve strictement
les axes et le stride 1, sans interpolation. Les indices doivent couvrir
exactement les cellules de compteur positif ; les doublons et coordonnées
invalides sont rejetés. Les cellules vides restent nulles. Aucun voisinage
spatial n'est encore traité.

À la suite de l'exemple de construction des piliers :

```python
import torch
from rgb_lidar_fusion.lidar_bev import PillarFeatureNet, scatter_pillar_features

pfn = PillarFeatureNet().eval()  # Poids aleatoires, pas un modele entraine.
point_features = torch.from_numpy(pillars.features)
with torch.no_grad():
    vectors = pfn(point_features, torch.from_numpy(pillars.point_mask))
    bev = scatter_pillar_features(
        vectors, indices=torch.from_numpy(pillars.indices),
        point_count=torch.from_numpy(pillars.point_count),
    )
lidar_bev = bev.features  # [B, 32, Hbev, Wbev], avant contexte spatial.
```

Les tenseurs doivent partager dtype et device avec le PFN, sans conversion ni
déplacement implicite. Le PFN/scatter accepte float32/float64 et fournit un
backward vers les features retenues et les paramètres ; le prétraitement NumPy
et les coordonnées discrètes ne deviennent pas différentiables. Aucun raccord
au runner d'entraînement n'est ajouté.

## Contexte avant fusion : premier bloc dilaté

Quatre convolutions 3 x 3 de stride 1 sans dilation couvriraient seulement
9 x 9 cellules, soit 1,8 x 1,8 m. Ce contexte est inférieur à notre cible.

Le premier bloc implémenté conserve quatre convolutions et utilise les dilatations
`(1, 2, 2, 4)`, avec stride 1 et padding égal à la dilation :

| Couche | Dilation | Champ réceptif cumulé | Étendue en mètres |
|---|---:|---:|---:|
| Conv 1 | 1 | 3 x 3 | 0,6 x 0,6 |
| Conv 2 | 2 | 7 x 7 | 1,4 x 1,4 |
| Conv 3 | 2 | 11 x 11 | 2,2 x 2,2 |
| Conv 4 | 4 | 19 x 19 | 3,8 x 3,8 |

`lidar_bev.LidarBEVContext` regroupe les couches en deux blocs résiduels :

```text
Bloc 1 : y = ReLU(x + Conv_d2(ReLU(Conv_d1(x))))
Bloc 2 : z = ReLU(y + Conv_d4(ReLU(Conv_d2(y))))
```

Ici, `Conv_dk` est une convolution 3x3 de dilation k ; x et z sont les cartes
BEV d'entrée et de sortie, sur la même grille.

La largeur reste constante, 32 par défaut ; les convolutions sont sans biais.
Aucune normalisation spatiale n'est ajoutée : les cellules vides n'entrent pas
dans de nouvelles statistiques globales et une entrée nulle reste nulle. La
BatchNorm du PFN ne change pas. Chaque convolution conserve les centres de
cellules grâce à stride 1 et padding égal à la dilation, y compris pour des
dimensions impaires ou non carrées.

Pour ces convolutions : `R = 1 + sum((k_i - 1) * d_i) = 19`. Le calcul porte
sur les dépendances du bloc par rapport à la grille issue du scatter, pas sur
une qualité apprise. Le padding ne fournit pas d'observations hors ROI. Des tests
contrôlés par gradient et impulsion vérifient le support complet 19x19, son
centrage et son absence de débordement vers le bord opposé.

À la suite de l'exemple PFN/scatter :

```python
from rgb_lidar_fusion.lidar_bev import LidarBEVContext

context = LidarBEVContext().eval()  # Poids aleatoires, pas un detecteur entraine.
with torch.no_grad():
    contextual_bev = context(bev)
lidar_bev = contextual_bev.features  # Meme grille ; compteurs/masque conserves.
```

Le module reçoit/retourne `LidarBEVFeatures`, sans modifier son entrée. Il copie
les compteurs de mesures et ne remasque pas les features propagées dans les
cellules sans observation. Les batches vides et cartes nulles restent valides,
avec backward fini. Aucun raccord au training, fusion ou head n'est ajouté.

À largeur constante, la dilation n'augmente pas le nombre de coefficients
des noyaux. Elle ne garantit cependant ni la même latence sur GPU, ni une
meilleure précision. Le champ réceptif effectif appris peut être plus restreint
que ce maximum théorique ; la sparsité LiDAR peut aussi limiter le contexte utile.

**Ce choix est validé fonctionnellement, pas en précision de détection ni en
performance GPU ; ce n'est pas le backbone PointPillars standard.** Le script
`scripts/compare_lidar_bev_context.py` le compare sur CPU à une petite variante
multi-résolution, avec un retour explicite à la grille d'origine par convolution
transposée. Cette variante est plus rapide sur le contrôle CPU local, mais son
champ varie entre 17 et 19 cellules selon la parité des indices de sortie.

Le bloc dilaté reste la première référence pour son contexte uniforme, son
alignement direct et ses 36 864 paramètres, contre 55 296 pour la variante.
Ce compromis n'est pas une preuve d'optimalité. Le
[workstream](workstreams/lidar-bev-encoder.md#cpu-comparison-and-first-context-choice)
donne la commande, les mesures CPU, la structure de comparaison et les limites.
Le choix pour le déploiement devra être réévalué avec supervision et mesures GPU.

## Sorties et sens des masques

`LidarBEVFeatures` contient `features`, une copie des `point_count` fournis et
la propriété `observed_mask`, avant comme après le bloc de contexte :

| Sortie | Shape | Sens |
|---|---|---|
| `lidar_bev` | `[B, C2, Hbev, Wbev]` | Features après contexte LiDAR ; PFN actuel à 32 canaux |
| `point_count` | `[B, 1, Hbev, Wbev]` | Mesures dans la ROI par cellule avant limitation des points |
| `observed_mask` | `[B, 1, Hbev, Wbev]` | Présence d'au moins une mesure LiDAR |

Les features peuvent devenir non nulles dans une cellule non observée grâce
aux convolutions voisines. Cela ne la transforme
pas en mesure, en espace libre ou en prédiction d'occupation. Les compteurs restent ceux des
mesures, pas ceux des activations. `C2` n'a pas besoin d'être égal au nombre de
canaux caméra `C1` pour une future concaténation.

## Validation et limites

Le [workstream](workstreams/lidar-bev-encoder.md#progress) détaille les incréments.
Les piliers, le PFN, le scatter et le contexte sont testés sur CPU, avec
forward/backward sur une frame KITTI réelle et poids aléatoires. À ce stade
initial, la suite passait à 528 tests réussis et un test CUDA ignoré. L'inspection conjointe
à 20 cm et un diagnostic GPU séparé sont consignés dans le
[workstream](workstreams/lidar-bev-encoder.md#4-scope-inspection-compatibility-and-resources).
Ce contrôle initial sans entraînement ne doit pas être confondu avec le
pilote supervisé et la revue développeur décrits dans le diagnostic Car ci-dessous.
Les premiers critères sont géométriques et logiciels : cellules attendues,
bords, séparation des batches, padding, cas vides, valeurs finies, backward et
alignement caméra/LiDAR sur une grille non carrée pour détecter les transpositions.
Le champ réceptif convolutif est vérifié sur des cas contrôlés.

Un backward synthétique ne prouve aucune reconnaissance d'objet. Pour évaluer
l'autonomie de la branche, il faudra ultérieurement des annotations 3D, une
head et une évaluation LiDAR seul face à la variante fusionnée. Ce travail ne
bloque pas l'implémentation du premier encodeur.

Passer de 100 x 100 à 250 x 250 représente 6,25 fois plus de cellules pour les
tenseurs denses, pas nécessairement 6,25 fois la latence ou la mémoire globale.
Le diagnostic `scripts/probe_lidar_bev.py` utilise un seuil local de **2 048 MiB
libres**, sans changer le seuil caméra/commun de 2 500 MiB ni les contrôles de
récupération/processus concurrents. Sur le GPU de 4 Go, un premier contrôle
batch 1 a mesuré 11,0 ms en forward et 35,3 ms en forward/backward, pic PyTorch
180,5 MiB. Ce n'est qu'une itération mesurée après warmup, sans caméra ni Adam.
Le regroupement CPU a pris séparément 95,7 ms : aucune cadence temps réel
globale ne doit être déduite du seul temps GPU. Plusieurs scènes, stabilité,
latence p50/p95 et mémoire du futur modèle fusionné restent à mesurer.
L'export ONNX/TensorRT reste à valider séparément.

## Diagnostic Car Facultatif

Le [workstream de diagnostic centre](workstreams/lidar-center-diagnostic.md)
ajoute un objectif Car-centre/offset XY et le notebook
`04_lidar_bev_analysis.ipynb`. Il sert à préparer l'entraînement et l'évaluation
de la branche LiDAR seule ; il n'est pas requis pour la cible fusionnée, dont
la loss finale pourra entraîner les deux branches.

Le PFN/contexte existant ne change pas. Les cibles proviennent de `label_2`,
avec conversion bottom-center caméra vers centre géométrique LiDAR et masques
conservateurs pour le hors-champ/DontCare. Les 512 labels locaux sont préparés
et audités ; les poids du notebook `04` restent aléatoires. Le notebook `05`
permet le surapprentissage de quatre frames, validé sur CUDA à 200 epochs.
Le notebook `06` et son runner utilisent un split fixe 410/102, des métriques
séparées et des checkpoints latest/best. Le pilote de 20 epochs est terminé :
`best.pt` retient l'epoch 9, avec précision 71,66% et rappel 63,77% au seuil 0,3
dans le domaine annoté. Les cas correct/manqué/faux positif ont été inspectés.
Le [bilan de clôture](workstreams/lidar-center-diagnostic.md#6-pilot-results-and-prototype-closure)
valide la chaîne train/validation du prototype, sans exiger plus d'optimisation
de cette head facultative avant de reprendre le raccord caméra/LiDAR BEV.

## Références

- [PointPillars, 2019](https://arxiv.org/abs/1812.05784) : PFN et backbone 2D
  multi-résolution ; notre cible n'en reproduit pas le détecteur complet.
- [Convolutions dilatées, ICLR 2016](https://arxiv.org/abs/1511.07122) :
  élargissement du contexte sans réduction de résolution ; la séquence retenue
  ici est testée fonctionnellement, pas validée en détection sur notre dataset.
- [PillarNeXt, CVPR 2023](https://arxiv.org/abs/2305.04925) : importance du
  backbone, du champ réceptif et de l'entraînement pour les approches par piliers ;
  ces résultats ne constituent pas une revendication SOTA pour notre projet.
