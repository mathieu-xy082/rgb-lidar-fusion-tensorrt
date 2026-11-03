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

## 1. Linear : un produit matriciel pour tous les points

On range les points valides en **lignes** dans la matrice $X$. Chaque ligne contient neuf caractéristiques. Avec les conventions de `nn.Linear` :

$$
\underbrace{X}_{V\times9}\;
\underbrace{W_{\mathrm{lin}}^{\mathsf T}}_{9\times32}
=\underbrace{H_{\mathrm{lin}}}_{V\times32}.
$$

| Objet | Dimensions | Sens |
|---|---|---|
| $X$ | $V\times9$ | Une ligne par point valide |
| $W_{\mathrm{lin}}$ | $32\times9$ | Une ligne de poids par canal de sortie |
| $W_{\mathrm{lin}}^{\mathsf T}$ | $9\times32$ | Transposée utilisée dans le produit |
| $H_{\mathrm{lin}}$ | $V\times32$ | Une ligne de 32 activations par point |

La dimension intérieure 9 se contracte ; V reste le nombre de points. Le PFN utilise **`bias=False`** : aucun biais n’est ajouté. La lettre W sans indice reste réservée à la largeur de la grille ; $W_{\mathrm{lin}}$ désigne les poids.

Exemple réduit à deux points, trois caractéristiques et deux canaux de sortie :

$$
\underbrace{\begin{bmatrix}2&1&-1\\0&2&1\end{bmatrix}}_{X:\;2\times3}
\underbrace{\begin{bmatrix}0.5&1\\-1&0\\2&-1\end{bmatrix}}_{W_{\mathrm{lin}}^{\mathsf T}:\;3\times2}
=
\underbrace{\begin{bmatrix}-2&3\\0&-1\end{bmatrix}}_{H_{\mathrm{lin}}:\;2\times2}.
$$

Chaque résultat est le produit d’une ligne d’entrée par une colonne de poids. Une même matrice de poids est partagée entre tous les points.

## 2. BatchNorm : statistiques par colonne, puis transformation matricielle

On part de $H_{\mathrm{lin}}\in\mathbb R^{V\times32}$. Pour $V\ge2$ en entraînement, notons $\mathbf1_V$ le vecteur colonne de V uns :

$$
\underbrace{\mu}_{1\times32}
=\frac1V\underbrace{\mathbf1_V^{\mathsf T}}_{1\times V}
\underbrace{H_{\mathrm{lin}}}_{V\times32},
\qquad
\underbrace{E}_{V\times32}=H_{\mathrm{lin}}-\mathbf1_V\mu.
$$

$$
\underbrace{v}_{1\times32}
=\frac1V\mathbf1_V^{\mathsf T}(E\odot E).
$$

$\odot$ signifie multiplication **élément par élément**, et non produit matriciel. Le vecteur v contient les variances des 32 canaux. La moyenne est soustraite à chaque ligne via $\mathbf1_V\mu$.

Pour chaque canal c, on définit $d_c=\gamma_c/\sqrt{v_c+\varepsilon}$, avec $\varepsilon=10^{-3}$. Puis :

$$
\underbrace{A_{\mathrm{BN}}}_{V\times32}
=\underbrace{E}_{V\times32}
\underbrace{D}_{32\times32}
+\underbrace{\mathbf1_V}_{V\times1}
\underbrace{\beta}_{1\times32},
\qquad D=\mathrm{diag}(d_1,\ldots,d_{32}).
$$

D est une matrice diagonale : elle applique un facteur différent à chaque colonne. Les paramètres $\gamma$ et $\beta$ sont appris. **Le code ne construit pas D** : il multiplie et additionne directement par canal avec diffusion (*broadcasting*). La matrice sert à lire l’opération.

Les points de tous les piliers participent aux statistiques ; le padding a été exclu avant Linear. Exemple : pour les valeurs 1 et 3 d’un canal, moyenne 2, variance 1 ; avec $\gamma=1,\beta=0$, sorties environ −0,9995 et +0,9995.

En évaluation, les statistiques mémorisées remplacent celles du lot. Le dépôt fait de même en entraînement si $V<2$, sans les mettre à jour ; les expressions divisant par V ne sont donc pas utilisées pour V=0. Le momentum vaut 0,01. La variance mémorisée utilise la correction non biaisée lors des mises à jour ordinaires de PyTorch ; le forward utilise la variance divisée par V.

## 3. ReLU et max : deux opérations différentes

$$
\underbrace{A}_{V\times32}=\rho\!\left(\underbrace{A_{\mathrm{BN}}}_{V\times32}\right),\qquad \rho(t)=\max(0,t).
$$

ReLU est non linéaire : elle s’applique élément par élément et ne se remplace pas par une matrice de poids fixe. Elle conserve la forme `[V,32]`. Ensuite, si $\mathcal V_p$ est l’ensemble des points valides du pilier p :

$$
f_{p,c}=\begin{cases}
\max_{v\in\mathcal V_p}A_{v,c},&\mathcal V_p\ne\varnothing,\\
0,&\mathcal V_p=\varnothing.
\end{cases}
$$

Le max est aussi une réduction non linéaire, pas un produit matriciel fixe. Il réduit les points **séparément pour chaque canal** : `[P,M,32] → [P,32]`. Avec $(0{,}8,0{,}2)$ pour le canal A et $(0{,}1,0{,}9)$ pour B, la sortie est $(0{,}8,0{,}9)$ : deux points différents fournissent les maxima. Les emplacements masqués valent $-\infty$ dans le tenseur interne de pooling, puis les piliers entièrement masqués sont explicitement mis à zéro.

## 4. Scatter : une matrice de placement, pour comprendre

On aplatit les positions spatiales de tout le batch : $N=BHW$. La sortie du PFN est $F\in\mathbb R^{P\times32}$. On définit une matrice de placement Q :

$$
Q\in\{0,1\}^{N\times P},\qquad
Q_{n,p}=1\;\text{si}\;n=b_pHW+r_pW+s_p.
$$

Chaque colonne contient un seul 1 : la position du pilier p. Les indices de piliers sont uniques, donc une ligne contient au plus un 1.

$$
\underbrace{S_{\mathrm{flat}}}_{N\times32}
=\underbrace{Q}_{N\times P}\;
\underbrace{F}_{P\times32}.
$$

Les lignes sans pilier restent nulles. On reforme ensuite `[B,H,W,32]`, puis on permute les axes vers `[B,32,H,W]`.

**Cette matrice est conceptuelle : le code utilise `index_copy`, sans allouer Q.** Il ne somme pas de piliers concurrents et n’interpole pas leurs positions ; les doublons sont rejetés. Les compteurs avant limitation restent distincts des activations. Placer n’est pas convoluer : aucun voisinage n’est mélangé.

## 5. Convolution 3×3 : extraire des patches, puis multiplier

Pour chaque position de sortie de chaque élément du batch, on extrait le patch 3×3 de **tous les canaux d’entrée**, à dilation d et avec padding nul. Ce patch est aplati en une ligne de $9C_{\mathrm{in}}$ valeurs. En empilant les lignes, on obtient $U_d(X)$, souvent appelé *unfold* ou *im2col*.

Pour le contexte actuel, stride 1 et padding d conservent H et W. On pose $N=BHW$ et on écrit :

$$
\underbrace{Y_{\mathrm{flat}}}_{N\times C_{\mathrm{out}}}
=
\underbrace{U_d(X)}_{N\times(9C_{\mathrm{in}})}
\underbrace{K_{\mathrm{flat}}^{\mathsf T}}_{(9C_{\mathrm{in}})\times C_{\mathrm{out}}}.
$$

| Objet | Forme générale | Pour 32 → 32 |
|---|---|---|
| Patches $U_d(X)$ | $N\times(9C_{\mathrm{in}})$ | $N\times288$ |
| Poids $K_{\mathrm{flat}}$ | $C_{\mathrm{out}}\times(9C_{\mathrm{in}})$ | $32\times288$ |
| Sortie aplatie | $N\times C_{\mathrm{out}}$ | $N\times32$ |

L’ordre d’aplatissement des canaux et des positions doit être identique pour les patches et les poids. On reforme la sortie en `[B,Cout,H,W]`. L’extraction de patches duplique des valeurs d’entrée ; c’est davantage qu’un simple changement de forme.

Cette écriture explique `Conv2d` ; **elle n’affirme pas que le backend matérialise cette grande matrice**. Les noyaux sont utilisés sans retournement : la convolution de PyTorch est une corrélation croisée. Le contexte n’ajoute aucun biais.

Exemple monocanal : un patch et un noyau 3×3 aplatis produisent une sortie scalaire :

$$
\underbrace{\begin{bmatrix}1&2&0&0&1&3&2&0&1\end{bmatrix}}_{1\times9}
\underbrace{\begin{bmatrix}1\\0\\-1\\1\\0\\-1\\1\\0\\-1\end{bmatrix}}_{9\times1}
=\underbrace{\begin{bmatrix}-1\end{bmatrix}}_{1\times1}.
$$

Avec plusieurs canaux, on prolonge les deux vecteurs avec les patches et les poids des autres canaux avant le produit. Chaque canal de sortie utilise tous les canaux d’entrée. Les poids du vrai contexte ont la forme `[32,32,3,3]`, soit 9 216 paramètres par convolution.

La valeur −1 est avant activation ; une ReLU appliquée ensuite donnerait zéro. Le laboratoire garde la vue produit par produit pour déplier concrètement le produit matriciel.

### Dilation, stride et padding

- **Dilation d** : espacement des positions lues dans chaque patch. À d=2, le noyau lit neuf positions par canal dans une étendue 5×5. La largeur $9C_{\mathrm{in}}$ de la matrice de patches ne change pas.
- **Stride s** : déplacement entre deux positions de sortie ; s=1 dans le contexte actuel.
- **Padding p** : zéros hors grille ; p=d conserve H et W pour k=3 et s=1.

$$
H_{\mathrm{out}}=\left\lfloor\frac{H+2p-d(k-1)-1}{s}+1\right\rfloor.
$$

Même formule pour W. Dans le cas général, la matrice de patches comporte $B H_{\mathrm{out}}W_{\mathrm{out}}$ lignes. La dilation ne change pas le nombre de poids. Avec les dilatations `(1,2,2,4)`, le champ théorique cumulé par axe vaut :

$$
R=1+2(1+2+2+4)=19.
$$

Cette étendue de dépendances est mesurée par rapport à la grille issue du scatter. Aux bords, le padding n’ajoute aucune mesure réelle.

## 6. Bloc résiduel : ajouter des tenseurs de mêmes dimensions

Notons $\mathcal C_d$ la convolution 3×3 de dilation d, **sans biais**. Chaque couche possède ses propres poids ; les deux couches de dilation 2 ne partagent pas leurs paramètres. Le premier bloc effectue :

$$
T=\rho\!\left(\mathcal C_1(X)\right),\qquad
F(X)=\mathcal C_2(T),\qquad
Y=\rho\!\left(X+F(X)\right).
$$

La définition matricielle de chaque $\mathcal C_d$ est celle de la section précédente. Tous les tenseurs X, T, F(X) et Y ont la forme `[B,32,H,W]`. Si l’on aplatit les positions en lignes :

$$
\underbrace{Y_{\mathrm{flat}}}_{N\times32}
=\rho\!\left(
\underbrace{X_{\mathrm{flat}}}_{N\times32}
+\underbrace{F(X)_{\mathrm{flat}}}_{N\times32}
\right).
$$

L’addition est élément par élément. Par exemple 0,6 + (−0,2), puis ReLU, donne 0,4. Le second bloc a la même structure avec d=2 puis d=4. Les activations non linéaires entre les convolutions empêchent de réduire tout le bloc à un unique produit par une matrice de poids fixe.

**Addition et concaténation sont différentes :** concaténer deux matrices $N\times32$ le long des canaux donnerait $N\times64$, alors que leur addition reste $N\times32$. La future concaténation caméra/LiDAR est un autre module, encore à développer.

## 7. Convolution 1×1 : un produit matriciel sur les canaux

On range chaque cellule de la carte en une ligne de $Z\in\mathbb R^{N\times32}$, avec $N=BHW$. Pour une convolution 1×1, le patch contient seulement les 32 canaux de cette cellule ; aucun voisin spatial direct n’est lu.

La head actuelle utilise **deux projections distinctes avec biais** :

$$
\underbrace{Q}_{N\times1}
=\underbrace{Z}_{N\times32}
\underbrace{W_h^{\mathsf T}}_{32\times1}
+\underbrace{\mathbf1_N}_{N\times1}
\underbrace{b_h}_{1\times1},
\qquad
P_h=\sigma(Q).
$$

$$
\underbrace{D_{\mathrm{logits}}}_{N\times2}
=\underbrace{Z}_{N\times32}
\underbrace{W_o^{\mathsf T}}_{32\times2}
+\underbrace{\mathbf1_N}_{N\times1}
\underbrace{b_o}_{1\times2},
\qquad
\Delta=\sigma(D_{\mathrm{logits}}).
$$

| Paramètre ou sortie | Dimensions | Rôle |
|---|---|---|
| $W_h$, $b_h$ | $1\times32$, $1\times1$ | Projection vers le logit Car |
| $W_o$, $b_o$ | $2\times32$, $1\times2$ | Projection vers les offsets XY |
| $Q$, $P_h$ | $N\times1$ | Logits puis scores après sigmoid |
| $\Delta$ | $N\times2$ | Fractions XY après sigmoid |

Les biais sont recopiés sur toutes les lignes par $\mathbf1_N b$. Le code les stocke comme vecteurs de longueur 1 et 2. La sigmoid est appliquée **élément par élément** :

$$
\sigma(t)=\frac1{1+e^{-t}}.
$$

Le forward retourne les logits Car, reformés en `[B,1,H,W]`, et les offsets après sigmoid en `[B,2,H,W]`. La sigmoid des logits Car sert notamment au décodage et à la loss ; les logits restent la sortie de la head pour ce canal.

Une convolution 1×1 mélange les canaux avec les mêmes poids à chaque cellule ; elle ne connecte pas toutes les positions spatiales entre elles comme une couche dense sur une image aplatie. Ses entrées portent néanmoins le contexte spatial acquis par les couches précédentes. Un score sigmoid n’est pas automatiquement une probabilité calibrée.

## 8. Décodage : du pic dans une carte aux mètres

Après sélection des maxima locaux et filtrage par score/top-k :

$$
x=x_{min}+(i+\delta^x)r,\qquad y=y_{min}+(j+\delta^y)r,
$$

où r est la résolution métrique. Il n’y a pas de décalage supplémentaire de $1/2$ cellule : l’offset 0,5 désigne déjà le centre de la cellule. À r=0,20 m, i=10 et $\delta^x=0{,}25$, avec $x_{min}=0$, on obtient x=2,05 m.

Cela fournit des centres x/y et scores, pas des boîtes orientées. La tête 3D cible devra ajouter les autres attributs.

## Retrouver les opérateurs

- [Features des piliers](../../src/rgb_lidar_fusion/lidar_pillars.py).
- [Linear, BatchNorm, ReLU, max, scatter et blocs résiduels](../../src/rgb_lidar_fusion/lidar_bev.py).
- [Head 1×1, sigmoid et décodage](../../src/rgb_lidar_fusion/lidar_center_model.py).

Le laboratoire utilise des tenseurs et noyaux fictifs de petite taille, en JavaScript. Les formes 32→32 du modèle sont décrites ici ; il ne s’agit pas d’un checkpoint ni d’une exécution PyTorch.

