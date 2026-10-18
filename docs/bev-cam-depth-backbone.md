# BEV camera-depth backbone

Date: 2026-08-05.

Mise à jour : 2026-10-18.

Statut : **backbone v1 implémenté et évalué sous supervision sparse**. Les deux
pilotes random/blocs sur 512 frames et leur comparaison croisée sont terminés.
L'inspection qualitative reste à faire avant clôture du workstream. La qualité
dense, la sémantique apprise et le lift BEV ne sont pas encore validés.

Documents liés :

- [architecture BEV cible](bev-model-roadmap.md) ;
- [milestones du projet](milestones.md) ;
- [résultats du workstream](workstreams/camera-depth-backbone.md) ;
- [implémentation du raccord BEV](workstreams/camera-depth-bev-lift.md).

## Question

Avant d'implémenter le lift complet vers BEV, faut-il ajouter une tâche
intermédiaire pour tester et entraîner la branche image-depth ?

Le baseline à l'origine de cette réflexion produisait :

```text
prediction: [B, 1]
```

Cette sortie scalaire par image servait uniquement de smoke test de trainabilité. Elle
ne mesure pas une profondeur dense, une détection, une boîte 3D ou une carte BEV.

Dans la perspective BEV, une tâche intermédiaire plus utile serait :

```text
RGB + sparse LiDAR maps
        |
        v
camera-depth backbone
        |
        v
depth_pred: [B, 1, H, W]
        |
        v
masked depth loss
```

## Conclusion actuelle

Oui, il faut envisager une loss intermédiaire avant de prolonger le backbone vers :

```text
Depth-aware lift vers BEV
utilisant splatted depth + calibration + confidence
```

Le but est de vérifier que la branche image-depth apprend une information utile
dans le plan image avant de lui demander de produire des features BEV.

## Pourquoi remplacer `prediction [B, 1]`

La prédiction scalaire historique n'a pas de sens perception direct :

- elle ne prédit pas une profondeur par pixel ;
- elle ne prédit pas une boîte 2D ou 3D ;
- elle ne prédit pas une heatmap BEV ;
- elle ne teste pas la qualité du splatting ;
- elle ne teste pas la capacité du modèle à propager/interpréter la profondeur.

Son seul rôle est de valider :

```text
forward -> loss -> backward -> optimizer.step -> checkpoint -> metrics
```

Une sortie dense `depth_pred: [B, 1, H, W]` serait un meilleur premier objectif,
car elle prépare directement la future étape image-depth -> BEV.

## Loss intermédiaire recommandée

Forme générale :

```text
loss_depth = SmoothL1(
    depth_pred[valid_mask],
    target_depth[valid_mask]
)
```

ou :

```text
loss_depth = L1(
    depth_pred[valid_mask],
    target_depth[valid_mask]
)
```

La loss doit être masquée, car la supervision depth n'est pas fiable partout.

## Niveau 1 - Reproduire le splatting

Objectif : smoke test dense, simple mais faible.

```text
Input:
  RGB
  sparse depth
  sparse mask

Target:
  depth_expanded

Loss mask:
  confidence > 0
```

Important : ne pas donner `depth_expanded` en entrée si c'est aussi la target.
Sinon, la tâche devient presque triviale.

Cette tâche apprend surtout au modèle à approximer l'algorithme de splatting. Elle
est utile pour remplacer le smoke scalar actuel par un smoke dense, mais elle ne
prouve pas encore une vraie capacité de depth completion.

## Niveau 2 - Sparse depth holdout

Objectif : tester une vraie capacité d'interpolation locale à partir de RGB +
voisinage LiDAR.

Principe :

```text
1. partir d'une sparse depth complète issue de la projection LiDAR ;
2. masquer volontairement une partie des points LiDAR ;
3. donner au modèle la sparse depth partiellement masquée ;
4. demander au modèle de prédire la depth sur les points retirés.
```

Contrat :

```text
Input:
  RGB
  sparse depth kept
  sparse mask kept
  optional geometry channels kept

Target:
  sparse depth full

Loss mask:
  heldout_lidar_mask
```

Loss :

```text
loss = SmoothL1(
    depth_pred[heldout_lidar_mask],
    sparse_depth_full[heldout_lidar_mask]
)
```

Cette option est recommandée comme premier vrai incrément, car elle ne nécessite
pas de depth dense externe ni de labels 3D. Elle exploite directement les points
LiDAR projetés et teste si le modèle peut retrouver une profondeur mesurée mais
cachée.

## Niveau 3 - Vraie depth completion

Objectif : entraîner une prédiction dense ou semi-dense fiable.

```text
Input:
  RGB
  sparse LiDAR

Target:
  dense depth ou semi-dense depth fiable
```

C'est le meilleur objectif pour apprendre une profondeur image exploitable, mais
il demande un dataset ou une procédure de génération de targets plus sérieuse :

- dataset depth completion ;
- accumulation multi-frame ;
- pseudo-labels offline ;
- teacher model.

Cette étape peut venir plus tard, quand le pipeline dense et les masques de loss
sont stabilisés.

## Architecture temporaire

Une première architecture volontairement simple :

```text
rgb:          [B, 3, H, W]
lidar_sparse: [B, 6, H, W]
        |
        v
concat -> small CNN encoder/decoder
        |
        v
depth_pred:   [B, 1, H, W]
```

Option utile :

```text
depth_pred
feature_map
```

Le modèle peut retourner à la fois :

- `depth_pred`, pour la loss intermédiaire ;
- `feature_map`, pour préparer la réutilisation dans le futur lift BEV.

`CameraDepthModel.forward()` retourne toujours uniquement `depth_pred` pour
préserver le contrat v1. La méthode `forward_features()` expose maintenant la
sortie du dernier décodeur ; la tête existante permet d'obtenir la profondeur
depuis ces features sans recalculer le backbone. Les modes sparse et enriched
sont implémentés et testés ; le mode enriched constitue la référence des deux
pilotes 512 frames. Il fournit 8 canaux LiDAR en plus des 3 canaux RGB.

## Lien avec le futur lift BEV

Une fois la branche capable de prédire une depth map raisonnable ou de produire
des features depth-aware, elle peut alimenter :

```text
Depth-aware lift vers BEV
utilisant depth_pred / sparse depth / confidence + calibration
```

Le passage vers BEV pourra utiliser :

- la profondeur sparse brute ;
- `depth_expanded` et `confidence` ;
- une `depth_pred` apprise ;
- les features intermédiaires du backbone image-depth.

La calibration sert ensuite à convertir des pixels image en positions 3D, puis à
les accumuler dans une grille BEV.

## Prerequis du raccord BEV

Cette section est la référence pour le prochain incrément architectural.
Les éléments ci-dessous sont suivis dans le chantier du lift ; les points 1 et 2
sont implémentés, les points 3 et 4 restent à développer/tester. Aucun nouvel
entraînement préalable n'est requis pour commencer la géométrie.

### 1. Exposer les features sans casser le backbone

- [x] Exposer les features par `CameraDepthModel.forward_features(rgb, lidar_maps)`.
- [x] Conserver le retour de `forward()` et les noms/shapes des paramètres v1.
- [x] Fixer le contrat à `image_features: [B, 32, H, W]` : sortie de `decoder_0`,
  juste avant `depth_head`, sans couche ni paramètre supplémentaire.
- [x] Tester les modes sparse/enriched, les tailles paires/impaires, le
  chargement strict des checkpoints et l'équivalence avec le calcul v1.
- [x] Vérifier qu'une loss sur les features seules atteint les entrées,
  l'encodeur et le décodeur, avec des gradients finis et non nuls. La tête de
  profondeur n'est pas utilisée dans ce chemin et ne reçoit donc pas de gradient.

La grille exposée est à pleine résolution : `features[b, :, v, u]` est aligné
sur `rgb[b, :, v, u]` et `depth_pred[b, 0, v, u]`. Dans cette convention de grille,
le centre de l'échantillon en haut à gauche est `(u=0, v=0)`, avec un pas de 1.
Il n'y a ni conversion de stride ni décalage supplémentaire entre ces grilles,
même pour les tailles impaires. Cet alignement n'affirme pas qu'une feature est
une mesure 3D ponctuelle : son champ réceptif dépend du réseau. Le passage de
ces indices aux coordonnées caméra sera traité au point 2.

Pour calculer les deux sorties en une seule passe :

```python
image_features = model.forward_features(rgb, lidar_maps)  # [B, 32, H, W]
depth_pred = model.depth_head(image_features)              # [B, 1, H, W]
```

Ne pas appeler en plus `model(rgb, lidar_maps)` dans ce cas, ce qui recalculerait
le backbone. Les features ne sont ni détachées du graphe ni stockées sur le
modèle ; le code appelant décide de leur durée de vie. Pour une inspection sans
gradients, utiliser `model.eval()` avec `torch.no_grad()`, comme pour la profondeur.

### 2. Rendre la géométrie explicite

- [x] Convertir la profondeur normalisée en mètres avec le `max_depth_m` du
  checkpoint, actuellement 80 m. Elle représente le **z caméra rectifiée**,
  pas la distance euclidienne au capteur.
- [x] Fournir au module de lift les matrices de calibration effectivement adaptées
  à la résolution de l'image, pas seulement un chemin vers le fichier KITTI.
  Les tableaux `camera_projection: [B, 3, 4]`, `t_vehicle_to_camera: [B, 4, 4]`
  et `max_depth_m: [B]` sont transmis jusqu'aux batches de profondeur, sans
  devenir des canaux d'entrée du CNN ni dépendre du holdout.
- [x] Préserver la matrice KITTI `P2` complète et sa quatrième colonne de
  translation ; ne pas supposer que les seuls intrinsics `K` suffisent pour
  inverser exactement la projection existante.
- [x] Documenter et tester caméra rectifiée -> repère LiDAR/véhicule, avec
  les directions des transformations, les axes et les unités. Le repère dit
  véhicule dans le chemin KITTI actuel est celui du Velodyne, pas un `base_link`
  ROS déjà calibré. La grille BEV utilisera d'abord ce repère ; son indexation
  et ses bornes restent à fixer au point 4.

`camera_geometry.normalized_depth_to_meters()` applique l'échelle explicite du
checkpoint ; `backproject_depth_to_vehicle()` produit des XYZ métriques
`[B, 3, H, W]` sans couper le graphe autograd. Pour `P2 = [A | t]`, le calcul est
`A X_cam = (z + t_z) [u, v, 1] - t`, puis application de l'inverse de
`T_vehicle_to_camera`. Le z désigne le repère caméra rectifié de référence,
avant `P2`, et non un repère recentré sur la caméra 2.

La convention de resize v1 est conservée : `u' = sx*u`, `v' = sy*v`, lignes
complètes de `P2` redimensionnées. Elle diffère du mapping des centres de pixels
du rééchantillonnage RGB PIL par un décalage sous-pixel ; elle ne sera pas
corrigée silencieusement pour les checkpoints existants. L'arrondi des points
dans les cartes sparse empêche aussi une inversion exacte vers les points raw.
Voir le [contrat détaillé](workstreams/camera-depth-bev-lift.md#geometry-contract).
Une future grille de features sous-échantillonnée devra définir son propre
alignement profondeur/calibration et le traitement des discontinuités ; elle
n'est pas introduite ici. Validité et confiance restent le point suivant.

### 3. Séparer profondeur utilisable et confiance

- Définir un masque de lift : profondeur finie, strictement positive, dans la
  plage métrique retenue et avec une règle de support explicite. La head
  actuelle est une régression linéaire non contrainte ; une sortie finie seule
  ne constitue pas une profondeur exploitable.
- Exclure explicitement les valeurs invalides ; les remplacer par une valeur
  positive arbitraire ou un simple clamp ne suffit pas à les rendre valides.
- Distinguer `confidence` du splatting, masque de validité et incertitude de
  `depth_pred`. La confiance existante est heuristique et ne calibre pas les
  prédictions du réseau. Une head d'incertitude apprise n'est pas un prérequis.
- Pour le premier contrôle géométrique, utiliser une profondeur synthétique
  connue, puis mesurée/splattée avec son support explicite. Brancher ensuite la
  profondeur apprise comme source distincte, sans déclarer fiables tous les
  pixels d'une carte dense ni utiliser les targets holdout comme entrées.

### 4. Fixer la grille et vérifier le lift

- Définir les limites métriques, l'origine, les axes, la résolution, la sélection
  en hauteur et l'ordre d'indexation de `camera_bev: [B, C, Hbev, Wbev]`.
- Choisir une agrégation explicite lorsque plusieurs pixels arrivent dans la
  même cellule : somme/moyenne, et éventuelle pondération par la confiance.
  Tester aussi les cellules vides et l'exclusion des points hors grille.
- Commencer par des tests synthétiques de backprojection/reprojection,
  calibration redimensionnée et `P2` à translation non nulle ; vérifier ensuite
  le placement dans des cellules BEV attendues.
- Vérifier un backward fini et non nul vers les features/backbone. Avec un
  scatter à indices discrets, cela ne garantit pas un gradient vers la
  profondeur ou la calibration via les coordonnées : documenter cette limite.
- Inspecter ensuite une frame KITTI avec la géométrie LiDAR comme référence,
  sans ajouter encore l'encodeur LiDAR BEV ni les heads de détection.

Le critère de passage est un raccord géométriquement cohérent et testable, pas
un score de profondeur parfait. Les entraînements mêlant random/blocs, les études de
rayon/taille de blocs, les datasets plus grands, la supervision dense et l'export
ONNX/TensorRT du nouveau modèle peuvent venir après ce premier prototype.
L'export ONNX actuellement disponible concerne le baseline historique, pas
encore `CameraDepthModel` ou le lift.

## Séquence recommandée

```text
[fait]    1. Remplacer prediction [B, 1] par depth_pred [B, 1, H, W]
[fait]    2. Ajouter une masked depth loss
[fait]    3. Tester sparse holdout depth prediction
[fait]    4. Évaluer les deux pilotes 512 frames sur les deux holdouts
[actif]   5. Inspecter les prédictions et consigner les limites de v1
[fait]    6. Exposer les features sans modifier le contrat depth/checkpoints
[prochain]7. Définir la géométrie et réaliser le raccord camera-depth -> BEV
[prévu]   8. Ajouter une branche lidar_bev
[prévu]   9. Ajouter fusion BEV + CenterPoint-like head
```

## Risques

- Utiliser `depth_expanded` à la fois comme entrée et target rend la tâche
  artificiellement facile.
- Apprendre uniquement le splatting ne garantit pas une vraie depth completion.
- Une loss dense mal masquée peut pénaliser des pixels sans vérité terrain fiable.
- Une pseudo-depth trop bruitée peut dégrader le futur lift BEV.
- Si cette étape grossit trop, elle peut retarder le prototype BEV au lieu de le
  préparer.

## Prochain incrément concret

Les entraînements random/blocs et la comparaison croisée sont terminés. Il
reste à inspecter quelques prédictions sur les mêmes frames et masques, puis à
clôturer v1 comme fonctionnelle et évaluée sous supervision sparse, sans
revendiquer une profondeur dense fiable partout.

Le prochain chantier architectural est le module `camera-depth -> camera_bev`
décrit dans l'[architecture BEV](bev-model-roadmap.md). Il commence par les
[prérequis de raccord](#prerequis-du-raccord-bev), dont l'accès aux features est
maintenant disponible avec la géométrie caméra/LiDAR. Le prochain incrément porte
sur la validité et la confiance, en conservant l'architecture et les checkpoints
actuels. Aucun entraînement
mixte ou changement de backbone n'est requis avant les premiers tests géométriques.

Les critères à jour sont suivis dans les
[milestones](milestones.md), et les résultats d'exécution dans le
[workstream](workstreams/camera-depth-backbone.md).
