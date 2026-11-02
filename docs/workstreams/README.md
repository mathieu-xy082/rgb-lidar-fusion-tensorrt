# Workstreams

Un workstream est un journal d'implémentation : périmètre, décisions, résultats
et risques d'un chantier. Il complète les documents d'architecture sans
devenir une source de vérité générale.

## Prochain chantier à préparer

L'encodeur du point cloud brut vers LiDAR BEV, avec des conventions de repère
et de grille compatibles avec la branche caméra, précédera la fusion et les
heads 3D. Voir les [milestones](../milestones.md).

## Clôturés - intégration BEV en attente

- [Camera-depth BEV lift](camera-depth-bev-lift.md) : accès aux features
  implémenté sans modification du contrat de profondeur ni des checkpoints ;
  géométrie métrique, sélection du support, raccord à la grille BEV et backward
  testés. Trois variantes inspectées sur la frame 000000 et relecture terminée
  le 2026-11-02 ; prototype prêt pour intégration dans main.
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

Les journaux restent à leur emplacement actuel pendant la préparation de
l'intégration, pour conserver les liens de relecture. Leur archivage sous
`docs/archive/workstreams/` après intégration devra préserver ou mettre à
jour ces liens ; aucun déplacement n'est effectué ici.
