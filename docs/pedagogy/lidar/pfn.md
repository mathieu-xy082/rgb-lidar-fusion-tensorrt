# PFN : plusieurs points, un vecteur par pilier

**Lecture : environ 10 minutes.** Prérequis : un point LiDAR est une mesure de position et d’intensité. Une feature est un nombre qui décrit une donnée ; un canal est une composante de cette description.

[Vue globale](../README.md) · [Branche caméra](../camera/backbone-and-lift.md) · [Branche LiDAR](../architecture-lidar-bev-head.md)

[Explorateur PFN](../explorers/pfn.html)

## Où se trouve le PFN ?

![De la mesure LiDAR aux features BEV : piliers, PFN, scatter et contexte spatial](../assets/lidar-pfn.svg)

Un nuage LiDAR contient un nombre variable de points, répartis irrégulièrement dans l’espace. Une convolution 2D attend une grille régulière. La branche LiDAR construit le passage entre les deux : elle rassemble les points dans des colonnes verticales, les **piliers**, résume chaque pilier, puis range ces résumés dans une grille vue de dessus, la **BEV** (*Bird’s Eye View*).

Le PFN (*Pillar Feature Network*) fabrique le résumé, sans prédire lui-même des objets ni consulter les piliers voisins. Il a été entraîné avec le contexte et la head de centres dans le [pilote LiDAR](../../workstreams/lidar-center-diagnostic.md#6-pilot-results-and-prototype-closure). L’explorateur de cette fiche reste un calcul à poids fictifs, pas une lecture du checkpoint appris.

## Lire les dimensions

| Symbole | Signification | Explorateur | Défaut du projet |
|---|---|---:|---:|
| P | Nombre de piliers représentés, sur tout le batch | 3 | Variable |
| M | Places disponibles par pilier | 4 | 64 |
| V | Total de points retenus et valides | 8 dans le cas initial | Somme du masque |
| 9 | Caractéristiques initiales par point | 9 | 9 |
| C | Canaux appris par point, puis par pilier | 32 | 32, configurable |

**Linear transforme les neuf nombres d’un point en 32 nombres ; il ne transforme pas neuf points en 32 points.** Le max réduit ensuite la dimension des points, pas celle des canaux.

## Que contiennent les neuf nombres ?

| Canaux | Contenu | Utilité intuitive |
|---|---|---|
| 1 à 4 | x, y, z, intensité | Décrire la mesure d’origine |
| 5 à 7 | x − moyenne_x, y − moyenne_y, z − moyenne_z | Situer le point par rapport au groupe retenu |
| 8 et 9 | x − centre_x du pilier, y − centre_y du pilier | Situer le point à l’intérieur de sa cellule |

Les positions et écarts sont métriques ; l’intensité est copiée sans normalisation ni clipping. Les moyennes utilisent les points **retenus**, après la sélection éventuelle à M points, jamais le padding. Les piliers font 20 cm × 20 cm dans la configuration initiale et ne sont pas subdivisés suivant z ; les points sont toutefois filtrés par la plage de hauteur de la ROI.

## Les sept étapes

### 1. Entrée : des points et un masque

Les features ont la forme `[P, M, 9]`. Le masque `[P, M]` distingue les mesures des places artificielles, appelées padding. Une valeur nulle peut être une vraie mesure : on ne déduit donc pas la validité du contenu numérique.

### 2. Filtrage : ne conserver que les points valides

`features[point_mask]` donne `[V, 9]`. Le padding ne traverse aucune couche. Dans l’explorateur, le remplacer par NaN ne modifie aucune sortie. Le dépôt teste aussi ce comportement avec des valeurs non finies masquées.

### 3. Linear : apprendre 32 combinaisons

Une même matrice W, de taille `32 × 9`, transforme chaque point : **h = W x**, sans biais dans cette implémentation. Par exemple, un canal pourrait combiner hauteur, intensité et écarts géométriques ; les coefficients sont appris, leur sens n’est pas attribué à la main.

Les canaux ne correspondent pas nécessairement à des concepts nommables comme « roue » ou « voiture ». L’explorateur utilise des poids déterministes fictifs pour rendre les calculs reproductibles.

### 4. BatchNorm : normaliser chaque canal

En entraînement, la moyenne et la variance d’un canal sont calculées sur **tous les points valides de l’appel**, même s’ils proviennent de piliers différents. Le PFN n’est donc pas entièrement indépendant point par point pendant cette étape.

Pour un canal : `sortie = γ × (h − moyenne) / √(variance + 0,001) + β`. γ et β sont apprenables. La variance utilisée pour normaliser le lot est divisée par V. PyTorch maintient également des statistiques mémorisées ; le momentum du module vaut 0,01.

En évaluation, le module utilise ces statistiques mémorisées. Avec moins de deux points valides au total en entraînement, le dépôt les utilise aussi, sans mise à jour. Leurs valeurs initiales (moyenne 0, variance 1) ne constituent pas une calibration apprise.

L’explorateur fixe γ=1, β=0 et des statistiques mémorisées fictives. Il montre le forward, sans simuler leur mise à jour au fil de l’entraînement.

### 5. ReLU : remplacer les activations négatives par zéro

`ReLU(a) = max(0, a)`. La forme `[V, 32]` est conservée.

### 6. Regroupement : retrouver le pilier de chaque point

Les activations reviennent à leurs positions `[P, M, 32]`. Le code place `−∞` dans les emplacements masqués pour les exclure du max interne.

### 7. Max : une valeur par canal et par pilier

Pour chaque pilier p et canal c, `f[p,c] = max(h[i,c])` parmi les points valides du pilier, après ReLU.

| Point d’un même pilier | Canal A | Canal B | Canal C |
|---|---:|---:|---:|
| Point 1 | **0,8** | 0,1 | 0,4 |
| Point 2 | 0,2 | **0,9** | 0,3 |
| Point 3 | 0,5 | 0,2 | **0,7** |
| Résultat | **0,8** | **0,9** | **0,7** |

Cet exemple isolé illustre trois canaux fictifs. Aucun point ne gagne partout : le vecteur final combine les maxima de plusieurs points. Deux points et cinquante points donnent la même largeur de sortie, `[32]`. Le module renvoie explicitement zéro pour un pilier entièrement masqué ; le constructeur de piliers ne matérialise normalement que les cellules occupées.

## Pourquoi ces choix ?

| Choix retenu | Bénéfice recherché | Compromis ou limite |
|---|---|---|
| Piliers puis grille 2D | Rendre le nuage exploitable par des convolutions 2D | Pas de découpage vertical ; la hauteur est portée par les features |
| Linear partagé | Même transformation pour tous les points | Aucun contexte entre piliers à cette étape |
| Max par canal | Taille de sortie fixe, indépendante de l’ordre des points retenus | Perd l’identité des points et ne préserve pas explicitement leur nombre |
| 32 canaux | Premier compromis logiciel plus compact que les 64 du PFN original | Pas d’optimalité en détection démontrée |
| BatchNorm | Référence implémentée, statistiques sur les points valides | Dépend de la population du lot en entraînement ; LayerNorm reste une variante non démontrée |
| Masque explicite | Exclure les places artificielles des statistiques et du max | Doit rester cohérent avec les features |

L’ordre des points **retenus** n’affecte pas le max ; en revanche, au-delà de M mesures, changer l’ordre source peut modifier la sélection des points conservés. Les compteurs avant limitation sont préservés séparément.

## Après le PFN : placer, puis contextualiser

Le **scatter** reçoit `[P, 32]` et les indices `(batch, ligne_x, colonne_y)`. Il copie chaque vecteur dans sa cellule, sans interpolation, sans mélange de voisins. Pour la ROI x=[0,50), y=[−25,25) à 20 cm : sortie `[B, 32, 250, 250]`.

Le contexte spatial arrive ensuite : deux blocs résiduels, quatre convolutions 3 × 3 de dilatations `(1,2,2,4)`. Leur champ théorique est de 19 × 19 cellules, soit 3,8 × 3,8 m. Ce champ ne garantit aucune reconnaissance d’objet.

Une cellule sans mesure reste distinguée d’une cellule observée dont les activations seraient nulles. Après le contexte, une cellule non observée peut recevoir des activations de ses voisines : elle ne devient pas une mesure pour autant.

## Manipuler et vérifier sa compréhension

1. Ouvrir l’explorateur, puis suivre les sept étapes.
2. Au max, changer le canal et observer les gagnants, y compris les ex æquo.
3. Remplacer le padding par NaN : les sorties doivent rester identiques.
4. Comparer entraînement et évaluation : les statistiques utilisées changent.
5. Sélectionner un point valide, puis aucun : observer le traitement particulier de BatchNorm et les sorties nulles des piliers masqués.

Questions : pourquoi le résultat contient-il 32 nombres même avec deux points ? Pourquoi un zéro ne suffit-il pas à identifier du padding ? Le scatter consulte-t-il un voisin ?

## Retrouver l’implémentation

- [Features, sélection et masques : `lidar_pillars.py`](../../../src/rgb_lidar_fusion/lidar_pillars.py) — `PILLAR_FEATURE_CHANNELS`, `build_lidar_pillars`.
- [PFN, scatter et contexte : `lidar_bev.py`](../../../src/rgb_lidar_fusion/lidar_bev.py) — `PillarFeatureNet`, `scatter_pillar_features`, `LidarBEVContext`.
- [Tests PFN et scatter](../../../tests/test_lidar_bev.py) — référence Linear/BN/ReLU/max, padding, permutation, singleton, cas vides et gradients.
- [Contrat technique et décisions](../../bev-lidar-encoder.md).

Cette fiche explique l’implémentation ; elle ne remplace ni ses tests ni les [jalons du projet](../../milestones.md). L’explorateur est une simulation pédagogique en JavaScript, pas une exécution PyTorch et pas une reproduction complète du détecteur PointPillars.
