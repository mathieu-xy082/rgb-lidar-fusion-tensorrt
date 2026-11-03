# Des blocs aux opérations mathématiques

[Vue des trois étapes](architecture-lidar-bev-head.md) · [Laboratoire visuel des convolutions](explorers/operators.html) · [Parcours](README.md)

Chaque opérateur répond à une question différente : **combiner des canaux, normaliser, sélectionner, placer ou regarder les voisins**. Les formules ci-dessous correspondent aux opérateurs du code relu ; les petits exemples numériques sont fictifs.

![Cartes de features, noyaux 3×3, addition résiduelle et branches de la head](assets/tensor-operators.svg)

## Voir les volumes, puis ouvrir les opérations

![Volumes des features : piliers, PFN, scatter, contexte BEV et deux branches de head](assets/lidar-feature-volumes.svg)

Comme dans une illustration classique de CNN, chaque volume représente un tenseur. Ici, après le scatter, les cartes gardent 250×250 cellules : le contexte élargit les dépendances sans pooling ni réduction spatiale. La head reste convolutionnelle ; elle n’utilise pas de couches fully connected comme le classifieur VGG de l’image de référence.

## Notation : une forme n’est pas une opération

| Symbole | Sens |
|---|---|
| $B$ | Nombre de nuages du batch |
| $P,M,V$ | Piliers représentés, places par pilier, points valides au total |
| $C_{in},C_{out}$ | Canaux d’entrée et de sortie |
| $H,W$ | Lignes suivant x et colonnes suivant y dans la grille BEV |
| $X_{b,c,i,j}$ | Activation du batch b, canal c, ligne i, colonne j |
| $\rho(t)=\max(0,t)$ | ReLU, appliquée composante par composante |

Une carte `[B,32,H,W]` contient **32 valeurs par cellule**. Les dessins d’empilement représentent ces canaux, pas des étages de hauteur physique. Les 32 canaux du PFN sont des caractéristiques apprenables ; ce ne sont pas 32 classes d’objets.

**Convention des produits matriciels :** les vecteurs d’entrée sont des colonnes, placées à droite des poids : `sortie = poids × entrée`. Pour plusieurs points ou cellules, leurs vecteurs sont juxtaposés en colonnes. Les formes de tenseurs du code PyTorch restent celles indiquées entre crochets ; cette convention mathématique ne change ni le stockage ni l’implémentation. Les indices scalaires h[v,c] désignent toujours l’activation du point v et du canal c.

## 1. Linear : combiner les caractéristiques par produit matriciel

Ici, l’écriture matricielle rend visibles les dimensions. On représente chaque point par un vecteur colonne et on range les V points valides en colonnes dans X :

```math
h_v=W_{\mathrm{lin}}x_v,\qquad H_{\mathrm{lin}}=W_{\mathrm{lin}}X.
```

| Objet | Dimensions | Sens |
|---|---|---|
| x_v | 9 × 1 | Vecteur colonne d’un point valide |
| X | 9 × V | Une colonne par point valide |
| Poids W_lin | 32 × 9 | Une ligne de poids par canal de sortie |
| h_v | 32 × 1 | Vecteur colonne de sortie d’un point |
| Résultat H_lin | 32 × V | Une colonne de 32 activations par point |

Le PFN utilise `bias=False` : aucun biais n’est ajouté. Le même jeu de poids sert à tous les points. Le produit transforme les caractéristiques ; il conserve V points.

Pour ouvrir un seul résultat, le produit scalaire est plus concret. Avec trois entrées et un canal de sortie fictifs :

```math
x=\begin{bmatrix}2\\1\\-1\end{bmatrix},\qquad w=\begin{bmatrix}0.5&-1&2\end{bmatrix},
```

```math
h=wx=0.5\times2+(-1)\times1+2\times(-1)=-2.
```

Dans le vrai PFN, on effectue ce calcul avec neuf entrées et 32 lignes de poids. Les dimensions restent dans le tableau plutôt que sous chaque facteur de l’équation.

## 2. BatchNorm : normaliser un canal sur les points valides

On regarde **un canal à la fois**, comme une colonne contenant V valeurs. Pour V ≥ 2 en entraînement, on calcule sa moyenne et sa variance :

```math
\mu_c=\frac1V\sum_{v=1}^{V}h_{v,c},
\qquad
\sigma_c^2=\frac1V\sum_{v=1}^{V}(h_{v,c}-\mu_c)^2.
```

Puis on transforme chaque valeur du canal :

```math
\widehat h_{v,c}
=\gamma_c\frac{h_{v,c}-\mu_c}{\sqrt{\sigma_c^2+\varepsilon}}+\beta_c,
\qquad \varepsilon=10^{-3}.
```

| Symbole | Sens |
|---|---|
| h[v,c] | Activation du point v dans le canal c, avant BatchNorm |
| μ[c], σ²[c] | Moyenne et variance sur les V points valides |
| γ[c], β[c] | Facteur et décalage appris pour ce canal |
| ε | Petite constante ajoutée à la variance avant la racine |

**L’intuition :** soustraire la moyenne, diviser par l’écart-type stabilisé, puis appliquer un facteur et un décalage appris. On répète l’opération sur les 32 canaux ; la forme `[V,32]` ne change pas.

Exemple : pour un canal contenant **1 et 3**, la moyenne vaut 2 et la variance vaut 1. Avec γ = 1 et β = 0 :

```math
\frac{1-2}{\sqrt{1+0.001}}\approx-0.9995,
\qquad
\frac{3-2}{\sqrt{1+0.001}}\approx+0.9995.
```

Les points de **tous les piliers** participent aux statistiques, jamais le padding : il a été exclu avant Linear et BatchNorm. La normalisation dépend donc des autres points du lot en entraînement.

En évaluation, on utilise les statistiques mémorisées. Le dépôt fait de même en entraînement avec moins de deux points valides au total, sans mettre à jour ces statistiques. Pour V=0, aucune moyenne du lot vide n’est calculée.

<details>
<summary>Détail : les statistiques mémorisées</summary>

Le momentum de `BatchNorm1d` vaut 0,01. Le forward utilise la variance divisée par V, comme ci-dessus. Lors des mises à jour ordinaires, PyTorch utilise une correction non biaisée pour la variance mémorisée. Les paramètres appris γ et β sont distincts de ces statistiques.

</details>

## 3. ReLU et max : deux opérations non linéaires

ReLU s’applique à chaque valeur :

```math
a_{v,c}=\max(0,\widehat h_{v,c}).
```

Une valeur négative devient zéro ; la forme `[V,32]` est conservée. Ensuite, pour le pilier p, on garde le maximum de chaque canal parmi ses points valides :

```math
f_{p,c}=\max_{v\in\mathcal V_p}a_{v,c}.
```

L’ensemble des points valides du pilier est noté V_p dans cette formule. S’il est vide, le code renvoie explicitement un vecteur nul. Le padding est exclu du max par des valeurs internes −∞.

Exemple : deux points donnent respectivement `(0,8 ; 0,1)` et `(0,2 ; 0,9)` sur deux canaux. Le résultat du pilier est `(0,8 ; 0,9)` : un point différent fournit le maximum de chaque canal.

ReLU agit sur chaque valeur ; le max réduit la dimension des points et produit `[P,32]`. Aucun de ces deux opérateurs ne se résume à une multiplication par une matrice de poids fixe.

## 4. Scatter : placer les vecteurs dans la grille

Le pilier p possède un vecteur de 32 valeurs et un indice `(b_p, r_p, s_p)` : batch, ligne suivant x, colonne suivant y. Pour chaque canal c, le scatter effectue simplement :

```math
S_{b_p,c,r_p,s_p}=f_{p,c}.
```

**C’est une affectation à une position**, pas une transformation des valeurs. Les autres cellules sont initialisées à zéro. Les indices sont uniques : le code rejette les doublons et n’additionne pas des piliers concurrents.

La forme passe de `[P,32]` à `[B,32,H,W]`. Aucun voisinage n’est mélangé et aucune interpolation n’est effectuée. Les compteurs de mesures avant limitation restent distincts des activations.

## 5. Convolution 3×3 : combiner les voisins et les canaux

Pour calculer **une cellule d’un canal de sortie**, on multiplie les valeurs du voisinage par les coefficients du noyau, puis on additionne les produits sur tous les canaux d’entrée. Pour le contexte LiDAR, de stride 1 et sans biais :

```math
Y_{b,o,i,j}=\sum_{c=1}^{C_{\mathrm{in}}}\sum_{u=-1}^{1}\sum_{v=-1}^{1}
K_{o,c,u,v}\,X_{b,c,i+du,j+dv}.
```

| Indice ou paramètre | Sens |
|---|---|
| b | Élément du batch |
| o, c | Canal de sortie et canal d’entrée |
| i, j | Position de sortie dans la grille |
| u, v | Décalages du noyau 3×3, de −1 à +1 |
| d | Dilation : espacement entre les positions lues |

Les indices spatiaux du noyau sont centrés en zéro pour alléger la formule. Les valeurs hors grille viennent du padding nul. PyTorch appelle cette opération convolution, mais effectue une corrélation croisée : les poids ne sont pas retournés.

**Un canal de sortie utilise tous les canaux d’entrée.** Le noyau complet a la forme `[Cout,Cin,3,3]`. Pour 32 → 32, il possède 32 × 32 × 9 = **9 216 poids**. Les mêmes poids sont réutilisés aux différentes positions.

Exemple monocanal, sans biais :

```math
X_{\mathrm{patch}}=\begin{bmatrix}1&2&0\\0&1&3\\2&0&1\end{bmatrix},
\qquad
K=\begin{bmatrix}1&0&-1\\1&0&-1\\1&0&-1\end{bmatrix}.
```

On multiplie les éléments aux mêmes positions puis on somme :

```math
y=1+0+0+0+0-3+2+0-1=-1.
```

Il ne s’agit pas du produit matriciel des deux petits tableaux 3×3 : ce sont les produits **élément par élément**, suivis d’une somme. Le résultat −1 est avant activation ; une ReLU appliquée ensuite donnerait zéro.

### La même convolution vue comme un produit matriciel

Cette seconde vue aide à relier la convolution à Linear. Pour chaque position, on extrait le patch de tous les canaux et on l’aplatit en un vecteur colonne. L’ensemble de ces colonnes forme U_d(X), souvent appelé *unfold* ou *im2col*. Avec N = BHW pour le contexte actuel :

```math
Y_{\mathrm{flat}}=K_{\mathrm{flat}}U_d(X).
```

| Matrice | Dimensions générales | Pour 32 → 32 |
|---|---|---|
| Patches U_d(X) | (9 Cin) × N | 288 × N |
| Poids K_flat | Cout × (9 Cin) | 32 × 288 |
| Sortie Y_flat | Cout × N | 32 × N |

L’ordre d’aplatissement doit correspondre pour les patches et les poids. On reforme ensuite la sortie en `[B,Cout,H,W]`. Cette équivalence mathématique ne signifie pas que le backend alloue effectivement cette grande matrice.

### Dilation, stride et padding

- **Dilation d** : écart entre positions lues. À d=2, un noyau 3×3 lit neuf positions par canal dans une étendue 5×5.
- **Stride s** : déplacement entre deux positions de sortie ; s=1 dans le contexte actuel.
- **Padding p** : zéros ajoutés hors grille ; p=d conserve H et W pour un noyau 3×3 et s=1.

```math
H_{\mathrm{out}}=\left\lfloor\frac{H+2p-d(k-1)-1}{s}+1\right\rfloor.
```

Même formule pour W. La dilation modifie les positions lues, pas le nombre de poids. Pour les quatre couches actuelles de dilatations `(1,2,2,4)` et de stride 1, le champ théorique cumulé par axe vaut :

```math
R=1+2(1+2+2+4)=19.
```

Cette étendue décrit les dépendances possibles vis-à-vis de la grille issue du scatter. Aux bords, le padding n’ajoute aucune mesure réelle.

## 6. Bloc résiduel : ajouter deux chemins

Notons C_d la convolution 3×3 de dilation d et ρ la ReLU. Le premier bloc de `LidarBEVContext` effectue :

```math
F(X)=\mathcal C_2\!\left(\rho(\mathcal C_1(X))\right),
\qquad
Y=\rho\!\left(X+F(X)\right).
```

X et F(X) ont la même forme `[B,32,H,W]`. **Leur addition est élément par élément** : à une position donnée, 0,6 + (−0,2), puis ReLU, donne 0,4. Le second bloc utilise d=2 puis d=4. Chaque convolution possède ses propres poids, y compris les deux convolutions de dilation 2.

L’addition conserve 32 canaux ; une concaténation de deux cartes de 32 canaux en produirait 64. La future concaténation caméra/LiDAR est une autre opération, dans un autre module encore à développer.

## 7. Convolution 1×1 : projeter les canaux vers une prédiction

À chaque cellule, une convolution 1×1 combine les 32 canaux sans lire directement les cellules voisines. Ici, le produit matriciel est une lecture compacte : on range les N = BHW cellules en colonnes dans Z.

La head actuelle contient deux projections distinctes avec biais :

```math
Q=W_hZ+b_h,
\qquad
\Delta=\sigma\!\left(W_oZ+b_o\right).
```

| Objet | Dimensions | Sens |
|---|---|---|
| Z | 32 × N | Features d’une cellule par colonne |
| W_h | 1 × 32 | Poids pour le logit de centre Car |
| W_o | 2 × 32 | Poids pour les offsets XY |
| b_h, b_o | 1 × 1 et 2 × 1 | Biais colonnes diffusés sur les N colonnes |
| Q | 1 × N | Logits Car avant sigmoid |
| Δ | 2 × N | Offsets XY après sigmoid |

Dans ces formules, l’ajout du biais utilise la **diffusion** (*broadcasting*) : le même vecteur colonne est ajouté à chacune des N colonnes. La sigmoid s’applique à chaque valeur :

```math
\sigma(t)=\frac1{1+e^{-t}}.
```

Le forward retourne Q reformé en `[B,1,H,W]` et Δ en `[B,2,H,W]`. La sigmoid des logits Car donne les scores utilisés notamment au décodage. Contrairement aux convolutions du contexte, ces convolutions 1×1 ont un biais.

Les cellules sont traitées avec les mêmes poids ; elles portent déjà le contexte spatial acquis par les couches précédentes. Un score sigmoid n’est pas automatiquement une probabilité calibrée.

## 8. Décodage : du pic dans une carte aux mètres

Après sélection des maxima locaux et filtrage par score/top-k :

```math
x=x_{min}+(i+\delta^x)r,\qquad y=y_{min}+(j+\delta^y)r,
```

où r est la résolution métrique. Il n’y a pas de décalage supplémentaire de $1/2$ cellule : l’offset 0,5 désigne déjà le centre de la cellule. À r=0,20 m, i=10 et $\delta^x=0{,}25$, avec $x_{min}=0$, on obtient x=2,05 m.

Cela fournit des centres x/y et scores, pas des boîtes orientées. La tête 3D cible devra ajouter les autres attributs.

## Retrouver les opérateurs

- [Features des piliers](../../src/rgb_lidar_fusion/lidar_pillars.py).
- [Linear, BatchNorm, ReLU, max, scatter et blocs résiduels](../../src/rgb_lidar_fusion/lidar_bev.py).
- [Head 1×1, sigmoid et décodage](../../src/rgb_lidar_fusion/lidar_center_model.py).

Le laboratoire utilise des tenseurs et noyaux fictifs de petite taille, en JavaScript. Les formes 32→32 du modèle sont décrites ici ; il ne s’agit pas d’un checkpoint ni d’une exécution PyTorch.


