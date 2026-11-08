# Workstreams

Un workstream est un journal d'implémentation : périmètre, décisions, résultats
et risques d'un chantier. Il complète les documents d'architecture sans
devenir une source de vérité générale.

## En cours

- [BEV fusion backbone](bev-fusion-backbone.md) : branche
  `ec/bev-fusion-backbone`, après la release 1.3.0. Contrat défini et fusion
  cellule par cellule implémentée/testée sur CPU : concaténation des features
  et supports, projection 1x1, compteurs préservés et backward. Backbone commun
  résiduel à trois échelles implémenté : alignement et champ 39/43 cellules
  contrôlés, mesure CPU reproductible. Diagnostic multimodal assemblé avec
  adaptateur CPU, head centre existante et checkpoints/audit distincts.
  Runner quatre frames et notebook `07` disponibles, avec inspection CPU,
  encodeurs gelés et lancement GPU explicite. Mémorisation CUDA validée :
  15 TP, aucun FP/FN ; pic alloué PyTorch 1512,6 MiB. Le pilote disjoint a
  maintenant son protocole, son audit d'initialisation appariée et son runner
  streaming 410/102 avec latest/best et reprise. Le notebook `08` et les
  contrôles de comparaison sont disponibles. Les deux runs CUDA de dix epochs
  sont terminés, comparés et inspectés : premier prototype multimodal de centres
  clôturé sur la branche. Le best fusion réduit les FP mais aussi le rappel ;
  aucun gain global n'est établi. Voir le [bilan](bev-fusion-backbone.md#6d-results-and-prototype-closure).
  La relecture d'intégration dans main reste à faire, pas un nouvel entraînement.

- [Camera-depth data provenance](camera-depth-provenance.md) : correctif intégré
  dans la branche de fusion, nouveau run cinq epochs tracé et inspecté.
  Audit ordinaire caméra/LiDAR passé sans exception ; aucun ancien checkpoint
  réécrit ou utilisé via l'exception legacy.

Ce chantier raccorde les deux branches avant les heads 3D finales. Voir les
[milestones](../milestones.md).

## Clôturés - intégrés dans main

- [LiDAR BEV encoder](lidar-bev-encoder.md) : piliers de 20 cm,
  PFN/scatter/contexte dilaté, inspection conjointe et diagnostic GPU borné ;
  intégré dans main pour la release 1.3.0. Les mesures initiales restent
  historiques ; le temps réel et le benchmarking de déploiement ne sont pas validés.
- [Diagnostic LiDAR centre facultatif](lidar-center-diagnostic.md) : inspection
  `04`, mémorisation quatre frames avec `05`, pilote disjoint 410/102 avec `06`
  terminé sur CUDA à 20 epochs. Incrément prototype clôturé après revue des
  courbes et de cas TP/FN/FP ; retenir l'epoch 9 selon la loss de validation.
  Cette head ne bloque pas la fusion BEV ; intégrée dans main pour la release 1.3.0.
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
