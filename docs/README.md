# Documentation

Cet index est le point d'entrée de la documentation du dépôt. Il distingue les
contrats actuels, les choix d'architecture, les guides opérationnels et les
archives afin qu'un plan historique ne soit pas pris pour une instruction
encore active.

## Parcours recommandé

Pour comprendre le projet dans son état actuel :

1. lire le [README principal](../README.md) pour le périmètre et les commandes ;
2. consulter les [milestones](milestones.md) pour savoir ce qui est terminé,
   actif ou prévu ;
3. lire l'[architecture BEV cible](bev-model-roadmap.md) ;
4. approfondir le
   [backbone camera-depth](bev-cam-depth-backbone.md), premier sous-système de
   cette architecture ;
5. suivre l'[onboarding pratique](onboarding/README.md) pour exécuter le dépôt.

## Statuts

- **Actuel** : décrit un contrat ou un comportement implémenté.
- **Design** : décrit une cible ou une décision architecturale non entièrement
  implémentée.
- **Guide** : explique comment utiliser ou valider une partie du dépôt.
- **Workstream** : journal d'exécution d'un chantier actif.
- **Archive** : contexte historique, sans valeur d'instruction actuelle.

## Architecture et décisions

| Document | Statut | Rôle |
|---|---|---|
| [BEV model roadmap](bev-model-roadmap.md) | Design | Architecture cible camera BEV + LiDAR BEV + fusion + heads 3D |
| [BEV camera-depth backbone](bev-cam-depth-backbone.md) | Design partiellement implémenté | Justification de la prédiction dense, du holdout et du futur lift |
| [Camera-depth workstream](workstreams/camera-depth-backbone.md) | Workstream | Implémentation, expériences KITTI/GPU, risques et critères de sortie |
| [Milestones](milestones.md) | Actuel | État vivant du projet et ordre des prochains incréments |

## Guides techniques

| Document | Statut | Rôle |
|---|---|---|
| [Dependency roadmap](dependency-roadmap.md) | Guide | Groupes PDM et contraintes CPU/CUDA/ONNX/TensorRT |
| [ONNX export validation](onnx-export-validation.md) | Guide legacy | Export et parité du `BaselineFusionModel`, pas encore du modèle BEV |
| [TensorRT benchmark plan](tensorrt-benchmark-plan.md) | Guide | Contrat runtime, génération d'engine et métriques de benchmark |

## Prise en main

Le dossier [onboarding](onboarding/README.md) contient un parcours progressif :
validation locale, géométrie caméra/LiDAR, splatting, KITTI, tenseurs, modèles,
entraînement, ONNX et TensorRT.

## Recherche

Le dossier [articles](articles/README.md) regroupe la bibliographie et le
positionnement initial du projet. Ces documents donnent du contexte scientifique
mais ne définissent pas les contrats logiciels.

## Workstreams et archives

Les workstreams actifs vivent sous [workstreams/](workstreams/README.md). Ils
documentent les décisions d'implémentation et les résultats qui ne doivent pas
alourdir les documents d'architecture.

Les plans terminés et anciens workstreams vivent sous
[archive/](archive/README.md). Ils sont conservés pour la traçabilité, mais ne
doivent pas guider une nouvelle implémentation sans vérification dans le code et
les documents actuels.

## Règles d'entretien

- Le README reste court et décrit uniquement l'état observable du dépôt.
- Les milestones portent l'avancement produit, sans historique détaillé de
  branches.
- Un document de design renvoie vers son workstream d'implémentation.
- Un workstream terminé est archivé après synthèse de ses décisions durables.
- Toute mention de « prochain », « actuel » ou « non implémenté » doit être
  réévaluée lors de la clôture d'un workstream.
