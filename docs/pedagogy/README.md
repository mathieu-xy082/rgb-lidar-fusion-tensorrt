# Explorateurs pédagogiques

Ce parcours en préparation expliquera les architectures du dépôt à travers des infographies et des outils interactifs en français.

## Parcours prévu

- Vue globale des branches caméra et LiDAR et de leur représentation BEV.
- Backbone camera-depth et lift géométrique caméra vers BEV.
- Construction des piliers LiDAR, features et masques.
- PFN : filtrage des points valides, Linear, BatchNorm, ReLU et max par pilier.
- Scatter des vecteurs de piliers vers la grille BEV.
- Contexte résiduel dilaté et champ réceptif.

Chaque support sera relié au code correspondant et montrera le rôle du module, ses dimensions, ses transformations et ses limites. Les exemples numériques seront explicitement synthétiques. Les modules implémentés seront distingués de l'architecture cible selon les [jalons](../milestones.md).

Le premier incrément prévu est une fiche PFN et un explorateur pas à pas. Les supports interactifs ne sont pas encore livrés.
