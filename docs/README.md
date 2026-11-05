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
5. lire le [design de l'encodeur LiDAR BEV](bev-lidar-encoder.md), intégré dans
   main depuis la release 1.3.0, puis le
   [contrat du chantier de fusion BEV](workstreams/bev-fusion-backbone.md#fusion-contract) ;
6. suivre l'[onboarding pratique](onboarding/README.md) pour exécuter le dépôt.

## Statuts

- **Actuel** : décrit un contrat ou un comportement implémenté.
- **Design** : décrit une cible ou une décision architecturale non entièrement
  implémentée.
- **Guide** : explique comment utiliser ou valider une partie du dépôt.
- **Workstream** : journal d'exécution d'un chantier.
- **Archive** : contexte historique, sans valeur d'instruction actuelle.

## Architecture et décisions

| Document | Statut | Rôle |
|---|---|---|
| [BEV model roadmap](bev-model-roadmap.md) | Design | Architecture cible camera BEV + LiDAR BEV + fusion + heads 3D |
| [BEV camera-depth backbone](bev-cam-depth-backbone.md) | V1 et premier raccord BEV clôturés | Justification du holdout et [contrats techniques du lift](bev-cam-depth-backbone.md#prerequis-du-raccord-bev) |
| [Camera-depth workstream](workstreams/camera-depth-backbone.md) | Prototype clôturé | Pilotes 512 frames, validation croisée, inspection de 000366, limites et historique GPU |
| [Camera-depth BEV lift](workstreams/camera-depth-bev-lift.md) | Intégré dans main, release 1.2.0 | Raccord et backward testés ; trois variantes BEV inspectées sur 000000 |
| [BEV LiDAR encoder](bev-lidar-encoder.md) | Intégré dans main, release 1.3.0 | Contrats NumPy/Torch, contexte, inspection conjointe 20 cm et diagnostic GPU borné |
| [LiDAR BEV workstream](workstreams/lidar-bev-encoder.md) | Prototype intégré dans main, release 1.3.0 | Résultats CPU/GPU, seuil LiDAR 2 Gio et limites avant fusion/détection |
| [Diagnostic LiDAR centre](workstreams/lidar-center-diagnostic.md) | Prototype intégré dans main, release 1.3.0 | Pilote 410/102, 20 epochs CUDA, best epoch 9 et revue TP/FN/FP ; pas un détecteur de production |
| [BEV fusion backbone](workstreams/bev-fusion-backbone.md) | Fusion et backbone commun CPU implémentés ; assemblage à venir | Grille commune, features/supports, projection 66 vers 32, pyramide 32/64/96 et validation par incréments |
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

## Comprendre les architectures

Le [parcours pédagogique](pedagogy/README.md) part des deux branches parallèles,
camera-depth et LiDAR, puis situe leur future convergence en BEV. Chacune dispose
de fiches, de schémas et d’explorateurs HTML autonomes. Les opérateurs communs et
les objectifs diagnostiques sont distingués de la fusion et des heads 3D à venir.

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

