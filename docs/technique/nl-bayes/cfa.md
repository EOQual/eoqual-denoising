# NL-Bayes CFA : débruiter et démosaïquer depuis la mosaïque

## 13. `nl_bayes_cfa` : débruiter et démosaïquer depuis la mosaïque

### 13.1 Pourquoi

Sur une image démosaïquée, ce qui reste après NL-Bayes est surtout du
bruit de basse fréquence en taches de couleur : le démosaïçage a corrélé
le bruit, alors que les deux filtres supposent σ²·I ([§8.2](parametres.md#82-image-démosaïquée), [§11.5](etudes.md#115-sensibilité-rgb-démosaïqué-zones-sombres)). Sur la
**mosaïque**, après stabilisation, le bruit est blanc et d'écart-type 1 :
le modèle σ²·I y est exact. `nl_bayes_cfa` estime, pour chaque patch, les
trois couleurs à partir des seules valeurs mesurées. C'est un problème
inverse linéaire à a priori gaussien, et NL-Bayes fournit cet a priori
par groupe. La méthode débruite et démosaïque à la fois : son entrée est
une mosaïque de Bayer (H, W), sa sortie une image RGB (H, W, 3).

**Sources** (seules sources, réimplémentation MIT) : G. Yu, G. Sapiro,
S. Mallat, « Solving Inverse Problems With Piecewise Linear Estimators:
From Gaussian Mixture Models to Structured Sparsity », *IEEE Trans. Image
Processing* 21(5), 2012, p. 2481-2499 (estimateurs linéaires par
morceaux, PLE : chaque patch est estimé avec la gaussienne qui lui
correspond) ; l'article NL-Bayes pour les groupes, la moyenne et la
covariance non locales (étape 2, [§4.4](methode.md#44-étape-2--la-covariance-vient-de-lestimée-de-base)).

### 13.2 Modèle et estimateur

Un patch couleur k × k est un vecteur x de dimension D = 3k² (les trois
couleurs de chaque pixel) ; la mosaïque en donne d = k² mesures :

```
y = U·x + n        U : sélection (d × D) des valeurs mesurées, fixée par le motif et la position du patch
                   n ~ N(0, σ²·I) dans le domaine stabilisé de la mosaïque (σ = 1)
x ~ N(μ, C)        a priori du groupe (patchs semblables)

x̂ = μ + C·Uᵀ·(U·C·Uᵀ + β·σ²·I)⁻¹·(y − U·μ)
```

- Toutes les composantes sont estimées, mesurées comme manquantes. Pour
  les composantes mesurées, on retrouve le filtre de l'étape 2 restreint
  à ces valeurs ; les manquantes sont prédites par leurs corrélations avec
  les mesures (C·Uᵀ).
- Seule U·C·Uᵀ + βσ²I (d × d) est inversée, et elle est définie positive
  même si C est singulière : pas de condition N > D.
- β joue le rôle de β₂ ([§4.5](methode.md#45-ce-que-fait-β)).

### 13.3 Algorithme

1. **Image pilote** : démosaïçage bilinéaire, puis NL-Bayes couleur
   (défaut), ou image fournie (par exemple Hamilton-Adams puis NL-Bayes).
   Le pilote sert à former les groupes et à estimer μ et C ; **aucune de
   ses valeurs ne passe dans le résultat**.
2. **Groupes** : pour chaque patch de référence (grille de pas `step`),
   les patchs à distance ≤ max(τ₀, d_N) dans le pilote (comme à l'étape 2,
   [§5.2](methode.md#52-étape-2-et-seuil-τ₀)) ; μ et C calculés sur les patchs pilotes.
3. **Phases** : U dépend de la position du patch dans le motif 2 × 2
   (quatre phases). Pour chaque phase présente dans le groupe, une seule
   résolution donne Gᵀ = (U·C·Uᵀ + βσ²I)⁻¹·U·C, puis x̂ = μ + Gᵀᵀ·(y − U·μ)
   pour tous les patchs de cette phase, y lu dans la mosaïque.
4. **Agrégation** : moyenne des estimées qui se recouvrent.
5. **Itération** : la sortie devient le pilote, et on reprend en 2 ;
   `n_iter` passes (8 par défaut), à la manière de l'EM de Yu et al.

Paramètres (`denoise_cfa` de nlbayes-ext, appelé par `backends/nl_bayes/cfa.py`) : `pattern` (`"RGGB"`, `"BGGR"`,
`"GRBG"`, `"GBRG"`), `k` = 3, `window_factor` = 7, `N` = 60, `beta` = 1,
`tau0` = 4 (sur [0, 255]), `n_iter` = 8, `pilot` (`"bilinear"`, `"none"`
ou une image), critère des zones homogènes (`use_area` = faux,
`area_z` = 2, `area_gamma`, `area_mean`, §13.5), `step`, `chunk`,
`n_threads`. Avec stabilisation :
`vst_denoise(mosaique_dn, "nl_bayes_cfa", gain=…, read_noise_dn=…,
pattern="RGGB")`. La méthode n'est pas encore branchée sur
`optimize_parameters` (la sortie RGB d'une entrée mosaïque n'y est pas
prévue).

### 13.4 Comparaison avec démosaïçage puis NL-Bayes

**Synthèse.** Hamilton-Adams puis `nl_bayes` couleur en Y₀U₀V₀ (« YUV »,
σ = C, β₂ = 1,2) contre `nl_bayes_cfa` sur la mosaïque (réglages par
défaut : N = 60, 8 passes ; « +zh » : critère des zones homogènes activé).
Erreurs contre la vraie image RGB, en niveaux d'affichage ; temps par
tuile de 128 × 128 et par fil. Protocole et détail ci-dessous.

| Jeu | Flux crête | PSNR (dB) YUV → CFA | Luminance, ombres YUV → CFA (+zh) | Couleur, ombres YUV → CFA | Luminance, clairs YUV → CFA | Couleur, clairs YUV → CFA | Temps YUV / CFA |
|---|---|---|---|---|---|---|---|
| Texturé | 300 e⁻ | 30,30 → **31,28** | 4,29 → 4,53 (4,35) | 2,88 → **2,33** | 6,67 → 6,12 | 4,75 → 4,03 | 0,3 / 2,0 s |
| | 1 500 e⁻ | 31,17 → **34,18** | 3,20 → 2,56 | 2,93 → **1,96** | 5,47 → 3,86 | 5,38 → 3,46 | 0,3 / 1,9 s |
| | 7 500 e⁻ | 31,25 → **35,10** | 3,00 → 1,80 | 3,34 → **1,80** | 5,09 → 3,21 | 5,66 → 3,36 | 0,3 / 1,9 s |
| Lisse | 300 e⁻ | 31,41 → **32,99** | 3,35 → 3,58 (3,38) | 2,07 → **1,83** | 5,42 → 4,85 | 4,34 → 2,90 | 0,3 / 2,7 s |
| | 1 500 e⁻ | 32,02 → **35,96** | 2,27 → 2,22 | 1,87 → **1,58** | 4,56 → 3,11 | 5,05 → 2,54 | 0,3 / 2,5 s |
| | 7 500 e⁻ | 32,28 → **37,49** | 1,95 → 1,49 | 2,06 → **1,47** | 4,19 → 2,32 | 5,15 → 2,44 | 0,3 / 2,2 s |
| Synthétique | 300 e⁻ | 31,69 → **36,52** | 7,35 → 4,31 (4,23) | 7,04 → **3,53** | 4,43 → 2,95 | 5,15 → 2,87 | 0,4 / 4,8 s |
| | 1 500 e⁻ | 31,32 → **37,36** | 6,61 → 2,89 | 7,13 → **3,02** | 4,57 → 2,56 | 5,67 → 2,76 | 0,4 / 4,8 s |
| | 7 500 e⁻ | 31,30 → **37,28** | 6,56 → 2,79 | 7,20 → **2,90** | 4,57 → 2,67 | 5,74 → 2,76 | 0,4 / 4,1 s |

`nl_bayes_cfa` gagne sur tous les indicateurs, sauf la luminance des
ombres à 300 e⁻ sur les orthophotos (+5 à +7 %), que le critère des zones
homogènes ramène au niveau de la chaîne YUV ; il coûte 7 à 12 fois plus de
temps. Filtrer les canaux ensemble à l'étape 2, plutôt que canal par
canal, améliore la chaîne YUV de 0 à 0,5 dB (et le pilote de `nl_bayes_cfa`, sans
effet notable sur son résultat à 8 passes) ; le gain de `nl_bayes_cfa`
baisse d'autant.

`examples/nl_bayes_cfa_study.py`, même chaîne capteur que le [§11.5](etudes.md#115-sensibilité-rgb-démosaïqué-zones-sombres) (flux
crête 300, 1 500 et 7 500 e⁻, lecture 3 e⁻, 2 tirages), toutes les chaînes
avec stabilisation de variance. Trois jeux de tuiles 128 × 128, tous avec
20 à 60 % de zones sombres (`--tiles`) :

- **texturé** : les 6 tuiles d'orthophoto les plus texturées (celles du
  [§11.5](etudes.md#115-sensibilité-rgb-démosaïqué-zones-sombres)) ;
- **lisse** : les 6 tuiles d'orthophoto les moins texturées ;
- **synthétique** : 4 scènes de grandes zones uniformes (cellules de
  Voronoï, moitié sombres, moitié claires), texture légère de ±2 niveaux
  d'affichage.

Le premier jeu, retenu pour sa texture, est défavorable aux zones
homogènes ; les deux autres corrigent ce biais. Référence de comparaison :
Hamilton-Adams puis NL-Bayes couleur en Y₀U₀V₀ (mode de l'article, [§8.1](parametres.md#81-trois-traitements-de-la-couleur)),
σ = C = √κ_Y + 0,2, β₂ = 1,2. Erreurs contre la **vraie** image RGB,
zones sombres en niveaux d'affichage ([§11.5](etudes.md#115-sensibilité-rgb-démosaïqué-zones-sombres)).

**Réglage de N et du nombre de passes** (`--parts grid`, pilote
bilinéaire, PSNR en dB, 300 / 1 500 / 7 500 e⁻) :

| N, passes | Texturé | Lisse | Synthétique | Temps / tuile |
|---|---|---|---|---|
| 30, 4 | 30,97 / 32,35 / 32,40 | 32,75 / 34,49 / 34,96 | 32,76 / 32,60 / 32,46 | 1,5 s |
| 60, 4 | 31,28 / 33,54 / 33,99 | 33,04 / 35,53 / 36,53 | 33,95 / 33,95 / 33,88 | 1,6 s |
| 90, 4 | 31,27 / 33,84 / 34,48 | 33,02 / 35,83 / 37,08 | 34,91 / 35,17 / 35,24 | 1,8 s |
| **60, 8** | 31,27 / 34,18 / 35,10 | 32,96 / 35,93 / 37,48 | 36,55 / 37,40 / 37,28 | 2,9 s |
| 90, 8 | 31,09 / 33,95 / 34,87 | 32,79 / 35,92 / 37,56 | 37,12 / 39,00 / 39,66 | 3,3 s |
| 120, 8 | 30,94 / 33,71 / 34,59 | 32,63 / 35,81 / 37,42 | 37,01 / 39,37 / 40,48 | 3,3 s |

- **N = 30 est trop petit** : un patch couleur 3 × 3 a D = 27 composantes,
  et 30 patchs estiment mal une covariance de cette dimension ([§3.4](methode.md#34-estimer-une-covariance-avec-peu-de-patchs)).
  N = 30 est la valeur de NL-Bayes en couleur, mais NL-Bayes filtre canal
  par canal, en dimension 9.
- **Valeurs par défaut : N = 60, 8 passes.** Sur les orthophotos, c'est
  le meilleur réglage de la grille ou il en est à 0,1 dB près ; sur la
  scène synthétique, il gagne 3,8 à 4,8 dB sur N = 30 et 4 passes.
  Plus de passes ou de patchs favorise encore les grandes zones uniformes
  (12 passes : +1 à +2,7 dB sur la synthétique), mais dégrade les
  orthophotos (−0,2 à −0,45 dB).

**Comparaison des chaînes** (`--parts chains`, réglages par défaut) :

| Jeu, flux | HA + NL-Bayes : PSNR / lum. sombre / coul. sombre | `nl_bayes_cfa` : PSNR / lum. sombre / coul. sombre | Gain apparié moyen (min.) |
|---|---|---|---|
| Texturé, 300 e⁻ | 30,30 / 4,29 / 2,88 | 31,28 / 4,53 / 2,33 | +0,99 (+0,10) |
| Texturé, 1 500 e⁻ | 31,17 / 3,20 / 2,93 | 34,18 / 2,56 / 1,96 | +3,01 (+2,10) |
| Texturé, 7 500 e⁻ | 31,25 / 3,00 / 3,34 | 35,10 / 1,80 / 1,80 | +3,85 (+2,74) |
| Lisse, 300 e⁻ | 31,41 / 3,35 / 2,07 | 32,99 / 3,58 / 1,83 | +1,57 (+0,81) |
| Lisse, 1 500 e⁻ | 32,02 / 2,27 / 1,87 | 35,96 / 2,22 / 1,58 | +3,94 (+2,82) |
| Lisse, 7 500 e⁻ | 32,28 / 1,95 / 2,06 | 37,49 / 1,49 / 1,47 | +5,20 (+3,64) |
| Synthétique, 300 e⁻ | 31,69 / 7,35 / 7,04 | 36,52 / 4,31 / 3,53 | +4,84 (+4,26) |
| Synthétique, 1 500 e⁻ | 31,32 / 6,61 / 7,13 | 37,36 / 2,89 / 3,02 | +6,04 (+4,76) |
| Synthétique, 7 500 e⁻ | 31,30 / 6,56 / 7,20 | 37,28 / 2,79 / 2,90 | +5,98 (+4,68) |

Constats :

1. **Gain net partout** : +1,0 à +6,0 dB en moyenne, positif sur chaque
   tuile (au plus juste : +0,10 dB, une tuile texturée à 300 e⁻). Il est le plus fort sur les scènes à grandes zones uniformes,
   où la chaîne actuelle garde le plus de bruit de couleur.
2. **Zones sombres** : l'erreur de couleur baisse de 11 à 60 %. L'erreur
   de luminance baisse de 2 à 57 %, sauf à 300 e⁻ sur les orthophotos,
   où elle augmente de 5 à 7 % (le critère des zones homogènes la ramène au
   niveau de la chaîne actuelle, §13.5).
3. **Démosaïçage** : sans bruit, `nl_bayes_cfa` dépasse Hamilton-Adams de
   4,3 dB (texturé), 5,7 dB (lisse) et 6,9 dB (synthétique). Une partie du
   gain vient du démosaïçage, pas seulement du débruitage.
4. **Pilote Hamilton-Adams** à 8 passes : +0,7 à +1,9 dB sur la
   synthétique, −0,3 à 0 dB sur les orthophotos. Le pilote bilinéaire
   interne reste le défaut ; à 8 passes, le pilote ne compte presque plus.
5. **β** : 0,7 gagne 0,3 à 0,4 dB sur les orthophotos à 300 e⁻ (0,2 dB
   au plus au-delà) et perd 0,07 à 0,16 dB sur la synthétique ; 1,4 perd
   jusqu'à 0,5 dB. β = 1 reste le défaut.
6. **Coût** : 2 à 5 s par tuile de 128 × 128 et par fil (8 passes), contre
   0,3 à 0,4 s pour Hamilton-Adams + NL-Bayes (7 à 12 fois plus).

### 13.5 Critère des zones homogènes

**Adaptation à la mosaïque.** Le critère de NL-Bayes ([§5.3](methode.md#53-critère-des-zones-homogènes)) compare la
variance des valeurs bruitées du groupe à (γσ)². Pour `nl_bayes_cfa`,
trois changements :

- variance calculée sur les **mesures de la mosaïque**, où le bruit est
  blanc de variance σ², et non sur le pilote, déjà débruité ;
- variance calculée **canal par canal**, autour de la moyenne de chaque
  canal, puis regroupée : une zone plate est de couleur constante, pas
  grise ;
- seuil **statistique** : s²/σ² ≤ 1 + z·√(2/(n − 3)), n le nombre de
  mesures du groupe. Sur du bruit seul, s²/σ² fluctue de ±√(2/(n − 3))
  autour de 1 ; avec 60 patchs 3 × 3 (n ≈ 540, ±6 %), le seuil fixe
  γ² = 1,10 de NL-Bayes n'est qu'à 1,6 fois cette fluctuation.

Un groupe plat est remplacé par la moyenne des mesures de chaque canal
(`area_mean="mosaic"`), ou par la moyenne du groupe pilote
(`area_mean="pilot"`). Options : `use_area`, `area_z` (2), `area_gamma`,
`area_mean`.

**Part des blocs sombres plats** (`--parts flatness`, s²/σ² ≤ 1,10 sur des
blocs 9 × 9 de la mosaïque stabilisée) : à 300 e⁻, 29 % (texturé), 36 %
(lisse), 46 % (synthétique) ; à 1 500 e⁻, 8, 11 et 25 % ; à 7 500 e⁻,
aucun. Au-delà de 300 e⁻, les ombres stabilisées sont texturées et le
critère, à juste titre, ne se déclenche presque pas.

**Effet à 300 e⁻** (réglages par défaut, z = 2, moyenne des mesures ; à
1 500 et 7 500 e⁻, écarts de PSNR ≤ 0,05 dB) :

| Jeu | PSNR sans → avec | Luminance sombre | Taches de luminance | Couleur sombre |
|---|---|---|---|---|
| Texturé | 31,27 → 31,29 | 4,54 → 4,37 (−4 %) | 1,99 → 1,72 (−14 %) | 2,34 → 2,39 (+2 %) |
| Lisse | 32,96 → 32,99 | 3,60 → 3,38 (−6 %) | 1,86 → 1,57 (−16 %) | 1,83 → 1,88 (+3 %) |
| Synthétique | 36,55 → 36,66 | 4,30 → 4,21 (−2 %) | 1,20 → 1,13 (−6 %) | 3,52 → 3,46 (−2 %) |

- **À faible flux, le critère ramène la luminance des ombres au niveau de
  Hamilton-Adams + NL-Bayes** (4,37 contre 4,34 ; 3,38 contre 3,42), en
  gardant tout le gain de couleur, pour 2 à 3 % de couleur en plus sur
  les orthophotos. Le PSNR ne baisse jamais. z de 1 à 3 ou γ = 1,05
  donnent le même PSNR à 0,03 dB près (z = 3 réduit un peu plus les taches
  de luminance) ; la moyenne du pilote ne change presque rien.
- **Il reste désactivé par défaut** : sur une zone parfaitement uniforme
  (test synthétique de 96 × 96), il coûte 10 dB (42,6 contre 52,5). La
  moyenne d'un groupe plat est calculée sur peu de mesures indépendantes
  par canal (une soixantaine de patchs 3 × 3 qui se chevauchent) et reste
  bruitée, alors que l'estimateur sans critère moyenne déjà sur toute la
  zone par ses passes successives.
- **Recommandation** : l'activer (`use_area=True`) à faible flux, quand la
  luminance des ombres compte ; il est sans effet à fort flux.

**Pourquoi le critère compte moins que dans NL-Bayes.** À l'étape 1 de
NL-Bayes, la covariance vient des patchs bruités : dans une zone plate,
elle garde des valeurs propres de bruit, que le gain non borné laisse
passer ([§4.3](methode.md#43-étape-1--la-covariance-du-signal-est-inconnue)), et le critère corrige ce défaut (+63 % d'erreur de
luminance des ombres sans lui à 300 e⁻, [§11.5](etudes.md#115-sensibilité-rgb-démosaïqué-zones-sombres)). `nl_bayes_cfa` estime
sa covariance sur un pilote déjà débruité, comme l'étape 2 de NL-Bayes :
dans une zone plate, C est presque nulle et l'estimateur rend déjà la
moyenne du groupe. Le critère y est donc un réglage fin, comme
`use_area2` pour NL-Bayes ([§11.5](etudes.md#115-sensibilité-rgb-démosaïqué-zones-sombres)).
