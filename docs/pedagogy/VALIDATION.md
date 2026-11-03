# Vérification du premier ensemble PFN

Référence de lecture : commit `3d9e786490ff655460ddc86f31cee64b5ae2ddea`, modules `lidar_pillars.py`, `lidar_bev.py` et tests `test_lidar_bev.py`.

## Vérifications effectuées

- Relecture du contrat : neuf features, Linear sans biais, BatchNorm eps=0,001, masque avant les couches, max par canal, sorties nulles pour les piliers entièrement masqués et fallback des statistiques pour moins de deux points valides.
- Calcul JavaScript confronté à un calcul NumPy indépendant : les 96 sorties du scénario initial concordent à une tolérance de 1e-12. Il s’agit d’une référence numérique synthétique, pas d’une comparaison à un checkpoint PyTorch.
- Vérification automatisée sous Node avec un DOM minimal : padding NaN sans effet, différence entre les modes BatchNorm, singleton identique en entraînement et évaluation pour les statistiques fixes choisies, scénario vide nul, rendu des sept étapes et sélection des 32 canaux.
- SVG rendu en PNG puis inspecté visuellement.
- Liens relatifs vérifiés contre les chemins de la branche.

## Limites de cette vérification

Le navigateur Chromium n’est pas installé dans l’environnement utilisé : le test Playwright n’a pas pu démarrer. Le DOM minimal ne vérifie ni la mise en page, ni les événements réels du navigateur, ni l’accessibilité complète. L’affichage HTML sur ordinateur et mobile reste à inspecter visuellement.

Aucun code de modèle n’est modifié et aucune suite PyTorch, mesure GPU ou expérience d’entraînement n’a été exécutée pour ce changement documentaire.

## Revue manuelle proposée

Ouvrir `explorers/pfn.html` localement :

1. Parcourir les sept étapes au clavier et à la souris ; vérifier les boutons aux extrémités.
2. Choisir plusieurs canaux et points ; lire le détail du calcul et les ex æquo du max.
3. Cocher le padding NaN ; vérifier que les sorties du max restent identiques.
4. Tester les deux modes BatchNorm et les trois scénarios, puis réinitialiser.
5. Réduire la fenêtre à 390 px : les contrôles doivent rester accessibles et les tableaux peuvent défiler horizontalement dans leur zone.
6. Vérifier que le vocabulaire et la densité conviennent à un lecteur débutant.
