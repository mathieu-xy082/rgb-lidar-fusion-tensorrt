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


## Page des trois rôles après rebase

La page `architecture-lidar-bev-head.md` et sa version autonome `explorers/architecture.html` décrivent l'encodeur local, le contexte BEV et la head de centres Car. Relecture de `lidar_bev.py`, `lidar_center_model.py`, des jalons, de la roadmap BEV et du workstream de diagnostic sur le commit `c6e0d6da157b42fcd8991c96b039e7b8fb31d96e`.

Les trois nouvelles figures SVG ont été rendues et inspectées. La logique de navigation entre les trois panneaux et de comparaison des deux parcours a été vérifiée sous Node avec un DOM minimal. Les identifiants des SVG embarqués sont uniques et la page ne charge aucune dépendance externe. Les liens relatifs sont vérifiés contre l'arbre de la branche. L'affichage HTML réel en navigateur reste à inspecter ; aucune nouvelle exécution PyTorch ou expérience d'entraînement n'est revendiquée.


## Formalisme et volumes des features

Ajout des formules des opérateurs et d'un laboratoire autonome 3×3/1×1, après lecture de `lidar_bev.py` et `lidar_center_model.py` au commit `a4a8017575bbdc314ac1cefed090df1930c433fc`. Le nombre de canaux est configuré, tandis que les paramètres sont apprenables. Les schémas de volumes sont explicitement non à l'échelle et ne représentent pas une hauteur physique.

Les deux nouveaux SVG ont été rendus et inspectés. Le JavaScript a été exécuté sous Node avec un DOM minimal : 36 configurations (3×3/1×1, dilatations 1/2/4, deux sorties, centre et bords) vérifiées contre une référence Python indépendante ; toutes les sommes concordent, avec biais et padding. Le mode 1×1 désactive la dilation. Les identifiants des figures embarquées et les liens sont contrôlés. L'affichage HTML et les événements réels en navigateur restent à inspecter. Aucun test PyTorch ou entraînement supplémentaire n'est déclaré.


## Écriture matricielle et rendu GitHub

Remplacement des macros `operatorname` signalées comme refusées par GitHub. Les opérateurs sont écrits avec des symboles définis dans le texte ; les produits Linear, BatchNorm, scatter conceptuel, convolution via patches et head 1×1 précisent leurs dimensions. ReLU et max restent décrits comme opérations non linéaires. Les petits exemples matriciels Linear, BatchNorm, scatter et patch ont été vérifiés avec NumPy. Délimiteurs mathématiques contrôlés et absence de `operatorname` vérifiée ; le rendu GitHub authentifié n'a pas été inspecté visuellement dans cet environnement.


## Ensemble camera-depth et lift BEV

Relecture des modules `camera_depth_model.py`, `camera_depth_support.py`, `camera_geometry.py`, `camera_bev.py` et du contrat technique au commit `26ad8ce2c354941fc828a052e66f184e5d48de1d`.

- Tableau des couches vérifié contre le code : 11/9 canaux d'entrée, stem 32, encodeurs 64/96, raccords 160/96, décodeurs 64/32 et head linéaire 32→1. Biais présents, sans BatchNorm ni pooling.
- Deux SVG rendus puis inspectés visuellement, dont les skips de concaténation et la séparation features/profondeur.
- Explorateur exécuté sous Node avec un DOM minimal : 28 rendus (sept étapes, deux résolutions, deux modes LiDAR), calcul de 54 rétroprojections, rejet par support, masques holdout fictifs et activation de la pondération uniquement pour la source splattée.
- Les 54 résultats géométriques comparés à des résolutions de systèmes NumPy indépendantes, avec P à quatrième colonne non nulle, changement de repère, reprojection et ROI. Moyenne pondérée et exemple central d'interpolation également vérifiés.
- Liens relatifs et identifiants HTML/SVG vérifiés ; formules en blocs math, sans operatorname ni underbrace.

Les activations du CNN et l'entraînement ne sont pas simulés. La grille holdout est illustrative, pas une reproduction du tirage du runner. L'exemple d'agrégation est distinct du pixel manipulé. La couverture visuelle/événementielle réelle de l'HTML en navigateur reste à vérifier ; aucune nouvelle exécution PyTorch, expérience d'entraînement ou validation GPU n'est revendiquée.
