# Workstreams actifs

Un workstream est un journal d'implémentation : périmètre, décisions, résultats
et risques d'une branche active. Il complète les documents d'architecture sans
devenir une source de vérité générale.

## Actif

- [Camera-depth BEV lift](camera-depth-bev-lift.md) : accès aux features
  implémenté sans modification du contrat de profondeur ni des checkpoints ;
  géométrie métrique et rétroprojection caméra/LiDAR testées ; validité et
  grille BEV à développer ensuite.
- [Camera-depth backbone](camera-depth-backbone.md) : v1 implémentée et évaluée,
  pilotes random/blocs sur 512 frames et comparaison croisée terminés ;
  inspection qualitative restante avant clôture.

Le chantier architectural actif est le lift camera-depth vers BEV. Ses
[prérequis techniques](../bev-cam-depth-backbone.md#prerequis-du-raccord-bev)
sont documentés ; accès aux features et géométrie caméra/LiDAR sont implémentés.
Les études supplémentaires
d'entraînement ne sont pas des prérequis pour ce prototype géométrique.

À la clôture du chantier, les décisions durables doivent être reportées dans les
documents d'architecture ou les guides, puis le workstream déplacé sous
`docs/archive/workstreams/`.
