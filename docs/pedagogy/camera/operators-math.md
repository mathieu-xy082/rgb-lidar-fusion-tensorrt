# Camera-depth : les opérations, puis la géométrie

[Vue globale](../README.md) · [Branche caméra](backbone-and-lift.md) · [Branche LiDAR](../architecture-lidar-bev-head.md)

[Explorateur camera-depth](../explorers/camera-depth.html) · [Convolutions : rappel commun](../operators-math.md)

**Convention : vecteurs colonnes à droite des matrices.** Les sommes servent à montrer les voisinages et les moyennes ; les produits matriciels servent aux projections et aux transformations de repères. Les formes `[B,C,H,W]` décrivent le stockage du code, pas l’orientation d’un vecteur mathématique.

## 1. Convolutions et réduction de résolution

Chaque `_ConvBlock` applique deux fois une convolution 3×3 avec biais, puis ReLU. Pour une couche de stride s et padding 1, en indexant le noyau par u,v de −1 à +1 :

```math
Y_{b,o,i,j}=b_o+\sum_c\sum_{u=-1}^{1}\sum_{v=-1}^{1}
K_{o,c,u,v}X_{b,c,si+u,sj+v}.
```

Les valeurs hors grille sont nulles. La majorité des couches utilise s=1 ; la première couche de chaque encodeur utilise s=2. Avec k=3 et p=1 :

```math
H_{\mathrm{out}}=\left\lceil H/s\right\rceil,
\qquad W_{\mathrm{out}}=\left\lceil W/s\right\rceil.
```

Ainsi, H=321 donne 161 puis 81 après les deux réductions. Il n’y a ni BatchNorm ni pooling dans ce modèle. Les poids du stem sont `[32,11,3,3]` puis `[32,32,3,3]` en mode enriched.

## 2. Interpolation bilinéaire : mélanger quatre voisins

Le décodeur emploie `interpolate(..., mode="bilinear", align_corners=False)` vers la taille exacte du skip. Pour la coordonnée horizontale u d’une sortie de largeur W_out, la position continue correspondante dans l’entrée vaut :

```math
\xi=(u+0.5)\frac{W_{\mathrm{in}}}{W_{\mathrm{out}}}-0.5.
```

Même principe verticalement pour η. En posant α=ξ−floor(ξ) et β=η−floor(η), la valeur interpolée est :

```math
f=(1-\alpha)(1-\beta)f_{00}
+\alpha(1-\beta)f_{10}
+(1-\alpha)\beta f_{01}
+\alpha\beta f_{11}.
```

Les indices 10 et 01 désignent ici respectivement le voisin horizontal et le voisin vertical. Aux bords, l’interpolation utilise la valeur du bord lorsque la coordonnée tombe au-delà du domaine. Chaque canal est interpolé séparément ; aucun nouveau poids n’est appris. Par exemple f00=0, f10=2, f01=4, f11=6 et α=β=0,5 donnent f=3.

Cette règle de rééchantillonnage des features ne remplace pas la convention de calibration v1 : lors du redimensionnement des entrées, les lignes complètes de P sont multipliées par sx et sy. Le décalage sous-pixel historique entre cette convention et le resize RGB est documenté dans le contrat technique, sans correction silencieuse des checkpoints.

## 3. Concaténation : empiler les canaux

Après interpolation, les deux cartes partagent B, H et W. La concaténation les réunit sur l’axe des canaux :

```math
D_1=\begin{bmatrix}U(E_2)\\E_1\end{bmatrix}_{\mathrm{canaux}},
\qquad 96+64=160.
```

Au second raccord : 64+32=96 canaux. Le décodeur applique ensuite ses convolutions pour revenir à 64 puis 32 canaux. Une concaténation ne fait ni moyenne ni addition des valeurs correspondantes ; les canaux sont conservés côte à côte jusqu’au mélange appris par les convolutions.

## 4. Head profondeur : une projection linéaire par pixel

Pour un pixel, sa feature f possède 32 composantes :

```math
\widehat d_n=wf+b,
\qquad w\in\mathbb R^{1\times32},\quad f\in\mathbb R^{32\times1}.
```

Il n’y a pas de sigmoid ou de ReLU dans cette head. La conversion en z caméra métrique utilise l’échelle D_max du checkpoint :

```math
z=D_{\mathrm{max}}\widehat d_n.
```

Cette multiplication ne rend pas valide une sortie négative ou non finie. La validité dépend ensuite d’un support explicite et d’une sélection métrique. La source du lift peut aussi être une profondeur sparse ou splattée, convertie avec la même échelle de normalisation adaptée à ses entrées.

## 5. Smooth L1 masquée : apprendre uniquement sur les mesures cachées

Pour l’ensemble H des pixels holdout et l’erreur normalisée e_p entre prédiction et cible :

```math
L=\frac1{|\mathcal H|}\sum_{p\in\mathcal H}\ell_\beta(e_p),
\qquad
\ell_\beta(e)=\begin{cases}
\frac{e^2}{2\beta},&|e|<\beta,\\
|e|-\frac{\beta}{2},&|e|\ge\beta.
\end{cases}
```

Cette expression vaut pour β>0 ; β=0 correspond à L1. La fonction du modèle utilise β=1 par défaut et rejette un masque ne sélectionnant aucune cible. La loss utilise les profondeurs normalisées, pas directement les mètres.

Les mesures cachées ne doivent pas alimenter le splatting ou les cartes d’entrée : les cartes sont reconstruites depuis les points conservés. Les pixels non supervisés ne sont pas des cibles nulles.

## 6. Rétroprojection : résoudre la calibration complète

La projection P est décomposée en A et t :

```math
P=[A\mid t],\qquad
A\in\mathbb R^{3\times3},\quad t\in\mathbb R^{3\times1}.
```

Pour le pixel q=(u,v,1) en colonne et la profondeur caméra z, la dernière ligne de A vaut `(0,0,1)` dans le contrat rectifié. La projection homogène vérifie :

```math
\lambda q=AX_{\mathrm{cam}}+t,
\qquad\lambda=z+t_z.
```

On résout donc :

```math
AX_{\mathrm{cam}}=(z+t_z)q-t.
```

Les deux membres sont des vecteurs 3×1. Le code utilise `torch.linalg.solve`, sans construire explicitement l’inverse de A. **La quatrième colonne t est conservée**, y compris t_z ; elle est distincte de la translation extrinsèque véhicule→caméra.

Exemple synthétique, également commenté dans le code :

```math
P=\begin{bmatrix}2&0&1&4\\0&2&1&-2\\0&0&1&0.5\end{bmatrix},
\quad q=\begin{bmatrix}2\\1\\1\end{bmatrix},\quad z=10.
```

On obtient λ=10,5, puis X_cam=(3,5 ; 1,25 ; 10) en mètres. Sa norme euclidienne est différente de z : « profondeur caméra » et « distance au capteur » ne sont pas interchangeables.

## 7. Changer de repère, puis discrétiser la BEV

La matrice T véhicule→caméra est de taille 4×4. Les coordonnées homogènes sont des colonnes 4×1 :

```math
T_{v\to c}\begin{bmatrix}X_v\\1\end{bmatrix}
=\begin{bmatrix}X_{\mathrm{cam}}\\1\end{bmatrix}.
```

On résout ce système pour revenir dans le repère véhicule/Velodyne, sans appliquer deux fois la translation de P. Dans ce repère, x pointe vers l’avant, y vers la gauche et z vers le haut.

Pour la résolution r de la grille :

```math
i=\left\lfloor\frac{x-x_{\min}}r\right\rfloor,
\qquad j=\left\lfloor\frac{y-y_{\min}}r\right\rfloor.
```

On filtre avant agrégation : support valide, profondeurs et features finies, bornes XYZ semi-ouvertes. Une position hors grille n’est jamais rabattue sur le bord. Les lignes suivent x et les colonnes suivent y, comme dans la branche LiDAR.

## 8. Moyenne des features par cellule

Pour les pixels acceptés dans la cellule g, leurs features f_p sont des vecteurs de 32 valeurs :

```math
F_g=\frac{\sum_{p\in\mathcal P_g}w_p f_p}{\sum_{p\in\mathcal P_g}w_p}.
```

Par défaut w_p=1. Pour la source splattée, une option utilise la confiance heuristique du splatting. Une cellule vide donne un vecteur nul ; un compteur séparé permet de l’identifier.

Exemple à deux canaux : features `(2 ; 4)` et `(8 ; 10)`, poids 1 et 0,5. La moyenne pondérée vaut `(4 ; 6)`, contre `(5 ; 7)` sans pondération. Les features sont moyennées ; le nombre de contributions et la somme des poids sont conservés séparément.

Le floor et les indices discrets ne donnent pas de gradient de placement vers la profondeur. Une loss sur F_g peut atteindre les features acceptées et le backbone, sans entraîner la head depth par ce seul chemin.

## Code de référence

[Modèle et loss](../../../src/rgb_lidar_fusion/camera_depth_model.py) · [Support](../../../src/rgb_lidar_fusion/camera_depth_support.py) · [Géométrie](../../../src/rgb_lidar_fusion/camera_geometry.py) · [Lift](../../../src/rgb_lidar_fusion/camera_bev.py).
