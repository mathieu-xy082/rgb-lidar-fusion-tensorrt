# BEV camera-depth backbone

Date: 2026-08-05.

Statut : **design partiellement implémenté**. La prédiction dense, le holdout,
la resplatisation sans fuite et le runner KITTI existent. La généralisation de
la profondeur et le lift vers BEV restent à valider ou implémenter.

Documents liés :

- [architecture BEV cible](bev-model-roadmap.md) ;
- [milestones du projet](milestones.md) ;
- [résultats du workstream](workstreams/camera-depth-backbone.md).

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

## Séquence recommandée

```text
[fait]    1. Remplacer prediction [B, 1] par depth_pred [B, 1, H, W]
[fait]    2. Ajouter une masked depth loss
[fait]    3. Tester sparse holdout depth prediction
[actif]   4. Évaluer la représentation camera-depth hors entraînement
[prochain]5. Réutiliser ce backbone/features pour camera-depth BEV lift
[prévu]   6. Ajouter une branche lidar_bev
[prévu]   7. Ajouter fusion BEV + CenterPoint-like head
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

Avant le lift, ajouter une séparation entraînement/validation, des métriques sur
les points holdout et des visualisations de profondeur/erreur. Le prochain
chantier architectural est ensuite le module `camera-depth -> camera_bev`
décrit dans l'[architecture BEV](bev-model-roadmap.md).

Les critères à jour sont suivis dans les
[milestones](milestones.md), et les résultats d'exécution dans le
[workstream](workstreams/camera-depth-backbone.md).
