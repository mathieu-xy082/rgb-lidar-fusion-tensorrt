# Workstreams

Un workstream est un journal d'implémentation : périmètre, décisions, résultats
et risques d'un chantier. Il complète les documents d'architecture sans
devenir une source de vérité générale.

## En cours

- [Diagnostic LiDAR centre facultatif](lidar-center-diagnostic.md) : notebook
  `04`, head Car/offset XY, loss masquée, annotations des 512 frames téléchargées
  et auditées. Le runner supervisé et ses courbes restent à développer ; cette
  head n'est pas un prérequis de la future fusion BEV.
- [LiDAR BEV encoder](lidar-bev-encoder.md) : construction des piliers de 20 cm
  et PFN/scatter/contexte dilaté implémentés et testés sur CPU, avec masques,
  compteurs, backward et champ réceptif de 19 cellules vérifié. Une référence
  multi-résolution est comparée sur CPU. Inspection conjointe 20 cm et premier
  diagnostic GPU borné terminés ; revue visuelle développeur, plusieurs scènes
  et benchmarking de déploiement restent à faire. Les choix durables
  et les variantes proposées sont distingués dans le [design](../bev-lidar-encoder.md).

Ce chantier précède la fusion et les heads 3D. Voir les
[milestones](../milestones.md).

## Clôturés - intégrés dans main

- [Camera-depth BEV lift](camera-depth-bev-lift.md) : accès aux features
  implémenté sans modification du contrat de profondeur ni des checkpoints ;
  géométrie métrique, sélection du support, raccord à la grille BEV et backward
  testés. Trois variantes inspectées sur la frame 000000 et relecture terminée
  le 2026-11-02 ; prototype intégré dans main pour la release 1.2.0.
- [Camera-depth backbone](camera-depth-backbone.md) : v1 implémentée et évaluée,
  pilotes random/blocs sur 512 frames et comparaison croisée terminés ;
  inspection held out de la frame 000366 consignée et clôture du prototype
  acceptée le 2026-11-02, sans revendiquer une profondeur dense fiable partout.

Le premier raccord camera-depth vers BEV est fonctionnel. Ses
[prérequis techniques](../bev-cam-depth-backbone.md#prerequis-du-raccord-bev)
sont documentés ; accès aux features, géométrie caméra/LiDAR et politique de
validité/confiance sont raccordés à un premier lift BEV fonctionnel PyTorch.
Les études supplémentaires
d'entraînement ne sont pas des prérequis pour ce prototype géométrique.

Les journaux clôturés restent à leur emplacement actuel pour conserver les
liens de relecture. Leur futur archivage sous `docs/archive/workstreams/`
devra préserver ou mettre à jour ces liens ; aucun déplacement n'est effectué ici.
