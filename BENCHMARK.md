# Benchmark — eoqual-denoising

Comparaison chiffrée des méthodes du catalogue, avec leurs réglages par
défaut puis avec des réglages optimisés. Pour le principe de chaque
méthode, voir `REFERENCE_TECHNIQUE.md` ; pour les licences,
`THIRD_PARTY_LICENSES.md`.

## En bref

- **Meilleure qualité** : `restormer`, `swinir`, `drunet` et `gsdrunet`
  sont à égalité à ±0,2 dB, et devant tout le reste. À σ = 0,1 :
  33,7 à 33,9 dB, contre 32,4 dB pour BM3D et 30,9 dB pour le meilleur
  du socle (`nl_means`).
- **Meilleur compromis qualité / vitesse** : `drunet` (1,5 s en CPU pour
  512×256), seul réseau conditionné par σ, donc robuste à tout niveau de
  bruit. C'est le choix par défaut recommandé quand PyTorch est
  disponible.
- **Sans PyTorch** : `nl_bayes` (32,44 dB, meilleur SSIM hors réseaux)
  et `nl_ridge` (32,19 dB, 32,50 une fois optimisé), deux
  réimplémentations MIT sans dépendance, au niveau de `bm3d` (32,38, non
  commercial). Pour une image couleur, `nl_bayes` en mode `yuv` (§8).
- **Optimiser les paramètres** rapporte jusqu'à **+1,3 dB** sur une image
  nouvelle pour les méthodes mal calibrées par défaut (`tv_bregman`,
  `epll`, `bilateral`), rien pour les réseaux et BM3D, déjà bien réglés.
  Une optimisation peut aussi **sur-apprendre** l'image de réglage
  (`nl_means` : +0,5 dB au réglage, −0,1 dB en évaluation).
- **Image brute de capteur** (bruit poisson-gaussien) : la stabilisation
  de variance (`vst_denoise`) apporte **+2 à +10 dB** (voir §4).

## 1. Protocole

| Élément | Choix |
|---|---|
| Image | `astronaut` 512×512, niveaux de gris, dynamique [0, 1] (NASA, domaine public, fournie par scikit-image) |
| Réglage / évaluation | **moitié gauche** (512×256) pour l'optimisation, **moitié droite** (512×256) pour toutes les mesures. Contenus différents, tirages de bruit différents (graines 1 et 2) |
| Bruit | blanc gaussien additif, σ ∈ {0,05 ; 0,1 ; 0,2}, **σ connu** et transmis aux méthodes |
| Scores | PSNR (dB) et SSIM, référence = image propre |
| Optimisation | σ = 0,1 ; optimisation bayésienne, 25 évaluations dont 8 aléatoires, graine 0, espaces de recherche par défaut (`REFERENCE_TECHNIQUE.md` §5.2) ; recherche exhaustive pour `swinir` (3 jeux de poids) |
| Machine | CPU arm64 (macOS), PyTorch sur CPU, un seul processus |
| Script | `examples/benchmark.py` (partie A et B), `examples/nl_bayes_benchmark.py` (§8) |

Séparer réglage et évaluation est indispensable : le gain mesuré sur
l'image de réglage elle-même est optimiste (colonne « Gain réglage » du
§3), seul le gain sur la moitié d'évaluation compte.

## 2. Réglages par défaut

### 2.1 PSNR (dB) — plus haut = meilleur

| Méthode | Famille | σ = 0,05 | σ = 0,1 | σ = 0,2 | Temps (σ = 0,1) |
|---|---|---|---|---|---|
| *(image bruitée)* | | 26,02 | 20,00 | 13,98 | — |
| `restormer` | réseau (aveugle) | **37,28** | **33,86** | **30,35** | 3,2 s |
| `drunet` | réseau (conditionné σ) | 37,22 | 33,82 | 30,29 | 1,5 s |
| `swinir` | réseau (poids par σ) | 35,65 | **33,86** | 29,62 | 14,2 s |
| `gsdrunet` | réseau (conditionné σ) | 37,04 | 33,68 | 30,10 | 1,5 s |
| `nl_bayes` | non local (MIT) | **36,08** ¹ | **32,44** ¹ | 28,67 | 1,5 s |
| `bm3d` | non local | 35,98 | 32,38 | 28,67 | 1,4 s |
| `nl_ridge` | non local (MIT) | 36,03 | 32,19 | 28,62 | 1,2 s |
| `nl_ridge_inria` | non local (code Inria) | 36,05 | 32,23 | 28,62 | 1,3 s |
| `nl_bayes_ipol` | non local (C++ IPOL) | 35,76 | 32,02 | 28,51 | 0,25 s |
| `nl_means` | non local | 34,31 | 30,87 | 26,69 | 0,32 s |
| `epll` | a priori de patchs | 34,80 | 30,61 | 25,89 | 5,4 s |
| `tgv` | variationnel | 33,92 | 30,16 | 26,34 | 3,4 s |
| `tv_chambolle` | variationnel | 33,61 | 29,75 | 25,87 | 0,03 s |
| `tv_bregman` | variationnel | 32,35 | 28,30 | 23,88 | 0,03 s |
| `wavelet_dict` | ondelettes | 32,18 | 27,89 | 23,56 | 0,21 s |
| `bilateral` | filtre local | 32,64 | 27,85 | 22,58 | < 0,01 s |
| `wavelet` | ondelettes | 31,49 | 27,63 | 24,60 | < 0,01 s |

¹ meilleur résultat hors réseaux.

### 2.2 SSIM — plus haut = meilleur (1 = identique)

| Méthode | σ = 0,05 | σ = 0,1 | σ = 0,2 |
|---|---|---|---|
| *(image bruitée)* | 0,504 | 0,310 | 0,172 |
| `restormer` | **0,965** | **0,941** | **0,902** |
| `drunet` | 0,964 | 0,938 | 0,898 |
| `swinir` | 0,947 | 0,938 | 0,882 |
| `gsdrunet` | 0,961 | 0,934 | 0,887 |
| `nl_bayes` | **0,955** ¹ | **0,922** ¹ | **0,863** ¹ |
| `bm3d` | 0,954 | 0,917 | 0,848 |
| `nl_ridge` | 0,943 | 0,880 | 0,810 |
| `nl_ridge_inria` | 0,943 | 0,883 | 0,811 |
| `nl_bayes_ipol` | 0,941 | 0,889 | 0,837 |
| `nl_means` | 0,939 | 0,881 | 0,758 |
| `tgv` | 0,925 | 0,844 | 0,702 |
| `epll` | 0,922 | 0,837 | 0,673 |
| `tv_chambolle` | 0,909 | 0,811 | 0,651 |
| `wavelet_dict` | 0,868 | 0,727 | 0,522 |
| `bilateral` | 0,824 | 0,627 | 0,404 |
| `tv_bregman` | 0,837 | 0,677 | 0,475 |
| `wavelet` | 0,793 | 0,707 | 0,614 |

### 2.3 Lecture

- **Trois paliers nets** : les réseaux (≈ 33,8 dB à σ = 0,1), puis les
  méthodes non locales à deux étapes NL-Bayes, BM3D et NL-Ridge
  (32,2 à 32,5 dB), puis
  le reste (≤ 30,9 dB). L'écart entre paliers (1,5 dB) dépasse largement
  les écarts au sein d'un palier.
- **Le SSIM confirme le PSNR**, à une nuance près : NL-Ridge perd plus en
  SSIM qu'en PSNR face à BM3D (0,883 contre 0,917 à σ = 0,1), signe d'une
  moins bonne préservation des structures fines.
- **SwinIR décroche à σ = 0,05** (−1,6 dB face aux autres réseaux) : il
  n'a de poids que pour 3 niveaux de bruit, et ce σ tombe entre deux
  (`REFERENCE_TECHNIQUE.md` §3.2). Il est aussi 10 fois plus lent.
- **Restormer en tête mais aveugle** : excellent dans son domaine
  d'entraînement (σ ≤ 50/255), il échoue au-delà, notamment à faible flux
  (`REFERENCE_TECHNIQUE.md` §3.3). DRUNet, à 0,05 dB près, n'a pas cette
  limite.
- **NL-Bayes natif** (moteur vectorisé) : en tête des méthodes
  classiques en SSIM à tous les niveaux de bruit, et en PSNR à σ ≤ 0,1 ;
  à égalité avec BM3D à σ = 0,2 (28,67 dB). Face à l'exécutable IPOL :
  +0,1 à +0,4 dB sur cette image (calcul en flottants, l'IPOL quantifiant
  sur 8 bits ; patchs de 5 pour les niveaux de gris), à égalité sur
  astronaut 256×256.
- **NL-Ridge natif** : reproduit le code des auteurs (32,19 contre
  32,23 dB, écart ≤ 0,04 dB à tous les σ) en temps comparable.
- **Méthodes classiques rapides** : `nl_means` (0,3 s) reste un bon
  compromis ; `tv_chambolle` est quasi instantanée pour 1 dB de moins.

## 3. Réglages optimisés (σ = 0,1)

PSNR sur la **moitié d'évaluation** avec les réglages par défaut, puis
avec les réglages optimisés sur la moitié de réglage.

| Méthode | Défaut | Optimisé | **Gain évaluation** | Gain réglage | Durée optim. | Meilleurs paramètres (multiples de σ) |
|---|---|---|---|---|---|---|
| `epll` | 30,61 | 31,89 | **+1,28** | +0,85 | 265 s | `sigma_scale` = 1,22 |
| `tv_bregman` | 28,30 | 29,57 | **+1,27** | +0,47 | 2 s | `weight` = 1,48/σ, anisotrope |
| `bilateral` | 27,85 | 28,83 | **+0,98** | +0,48 | 3 s | `sigma_color` = 2,66 σ, `sigma_space` = 1,72 px |
| `nl_ridge` ² | 32,19 | **32,50** | +0,31 | +0,02 | 170 s | `p1` = 9, `p2` = 9, `k1` = 15, `k2` = 118 |
| `nl_ridge_inria` ² | 32,23 | **32,53** | +0,30 | +0,06 | 129 s | `p1` = 10, `p2` = 7, `k1` = 17, `k2` = 86 |
| `nl_bayes_ipol` ² | 32,02 | 32,10 | +0,08 | +0,13 | 25 s | `sigma_scale` = 1,06, zones homogènes étape 1 |
| `nl_bayes` ² | 32,44 | 32,46 | +0,02 | +0,15 | 91 s | 10 dimensions ; `sigma_scale` = 0,92, `k2` = 7, `beta1` = 1,2, `tau0` = 2,1 |
| `tv_chambolle` | 29,75 | 29,82 | +0,07 | +0,01 | 2 s | `weight` = 0,78 σ |
| `wavelet` | 27,63 | 27,70 | +0,07 | +0,03 | 2 s | `db2`, BayesShrink, doux, `sigma_scale` = 0,98 |
| `wavelet_dict` | 27,89 | 27,96 | +0,07 | +0,01 | 25 s | `threshold` = 1,3 σ, 3 niveaux |
| `drunet` | 33,82 | 33,82 | 0,00 | 0,00 | 40 s | `sigma_scale` = 0,99 |
| `gsdrunet` | 33,68 | 33,67 | 0,00 | 0,00 | 39 s | `sigma_scale` = 0,99 |
| `bm3d` | 32,38 | 32,38 | 0,00 | 0,00 | 191 s | défauts (`sigma_scale` = 1, profil `np`) |
| `swinir` | 33,86 | 33,86 | 0,00 | — (exhaustif) | 43 s | poids « 15 » (= choix automatique) |
| `tgv` | 30,16 | 30,13 | −0,03 | 0,00 | 204 s | `weight` = 7,4 σ |
| `nl_means` | 30,87 | 30,74 | **−0,13** | +0,49 | 9 s | `h` = 0,81 σ, patchs 7, fenêtre ±3 |
| `restormer` | 33,86 | — | — | — | — | aucun paramètre réglable |

² 60 évaluations dont 20 aléatoires (espaces plus grands) ; les autres
lignes : 25 évaluations.

### 3.1 Lecture

- **NL-Ridge gagne +0,3 dB en généralisation** avec des groupes plus
  grands à l'étape 2 (k₂ ≈ 90-120 au lieu de 90 à ce σ) : une fois
  réglé, c'est la meilleure méthode classique (32,50 dB). NL-Bayes, déjà
  au mieux avec les paramètres de l'article, ne gagne rien sur une image
  nouvelle malgré ses 10 dimensions (+0,15 dB au réglage, +0,02 en
  évaluation : sur-apprentissage léger).

- **Les réseaux et BM3D sont déjà au mieux** avec σ connu : l'optimiseur
  retrouve `sigma_scale` ≈ 1, c'est-à-dire les défauts. L'optimisation ne
  sert alors qu'à compenser un σ mal estimé ou un bruit non conforme au
  modèle.
- **Trois méthodes gagnent plus de 0,9 dB sur une image nouvelle** :
  EPLL (σ sous-évalué de 22 % par défaut), TV Bregman (poids et TV
  anisotrope) et bilatéral (noyau spatial plus large). Ces gains se
  généralisent : ils sont même plus forts en évaluation qu'au réglage. Ce
  sont des candidats au **recalibrage des valeurs par défaut** (§8 de la
  référence technique), à confirmer sur d'autres images.
- **Sur-apprentissage de `nl_means`** : la fenêtre de recherche réduite à
  ±3 px gagne 0,5 dB sur la moitié de réglage mais en perd 0,1 sur
  l'autre moitié. C'est l'illustration du piège décrit au §5.5 de la
  référence technique : **toujours valider un réglage sur une autre
  image que celle qui a servi à l'optimiser.**
- **Coût** : de quelques secondes (socle) à 4 minutes (EPLL, TGV, BM3D)
  pour 25 évaluations sur 512×256 en CPU. Avec
  `n_jobs > 1`, les méthodes du socle se parallélisent sans perte.

## 4. Image brute de capteur : bruit poisson-gaussien

Résumé de `REFERENCE_TECHNIQUE.md` §4.5 (`examples/low_light_vst.py`,
`astronaut` 256×256, K = 2 e⁻/DN, bruit de lecture 0,5 e⁻). PSNR (dB) :

| Pic | Méthode | Directe (σ unique estimé) | GAT + inverse exacte | Gain |
|---|---|---|---|---|
| 2 e⁻ | `bm3d` | 16,18 | 20,31 | +4,1 |
| 2 e⁻ | `drunet` | 11,41 | **21,16** | +9,8 |
| 5 e⁻ | `drunet` | 15,27 | **24,36** | +9,1 |
| 20 e⁻ | `drunet` | 20,84 | **28,66** | +7,8 |

Quand le bruit photonique domine, appliquer un débruiteur gaussien
directement est une erreur : la stabilisation de variance gagne 4 à
10 dB. `restormer` ne profite pas de la GAT ; préférer `drunet`.

## 5. Recommandations

| Situation | Méthode conseillée | Pourquoi |
|---|---|---|
| Cas général, PyTorch disponible | `drunet` | Au niveau des meilleurs, rapide, conditionné par σ (robuste à tout niveau de bruit) |
| Qualité maximale, bruit modéré (σ ≤ 0,2) | `restormer` | Légèrement devant, mais aveugle : hors de son domaine il échoue |
| Image brute de capteur, faible flux | `vst_denoise(..., "drunet")` | +8 à +10 dB sur l'application directe |
| Sans PyTorch | `nl_bayes` | Meilleure méthode classique par défaut (PSNR et SSIM), MIT, aucune dépendance |
| Sans PyTorch, image couleur | `nl_bayes`, `color_mode="yuv"` | +0,8 à +1,2 dB sur le canal par canal, 2,5 fois plus rapide (§8.2) |
| Sans PyTorch, référence disponible | `nl_ridge` optimisé | 32,50 dB une fois réglé (§3) |
| Sans PyTorch, temps compté | `nl_means` | 7 fois plus rapide que `nl_bayes`, 1,6 dB de moins |
| Traitement massif, temps réel | `tv_chambolle` | ~10 fois plus rapide que `nl_means`, 1 dB de moins |
| Schéma plug-and-play (déconvolution) | `gsdrunet` | Convergence garantie, −0,15 dB seulement face à DRUNet |
| Nouveau capteur, référence disponible | `optimize_parameters` puis validation sur une autre scène | Gains jusqu'à +1,3 dB sur les méthodes mal calibrées |

## 6. Limites

- **Une seule image naturelle.** Le classement peut changer sur des
  images EO (statistiques, textures, dynamique) : les réseaux, entraînés
  sur des photos 8 bits, sont les plus exposés. Refaire ce benchmark sur
  des scènes du capteur visé avant toute décision.
- **Bruit blanc gaussien à σ connu**, sauf §4. Un σ estimé, un bruit
  coloré (`REFERENCE_TECHNIQUE.md` §9.1) ou non stationnaire changent la
  donne.
- **Temps en CPU** sur une seule machine, indicatifs ; sur GPU, les
  réseaux sont 10 à 100 fois plus rapides.
- **`nl_bayes` et `nl_ridge` natifs en numpy** : ~5 fois plus lents que
  le C++ OpenMP de l'IPOL (1,2 à 1,5 s contre 0,25 s pour 512×256) ; un
  portage PyTorch (GPU) est noté dans les évolutions.

## 7. Reproduire

Voir la commande en fin de §8 pour le benchmark NL-Bayes.



```bash
pip install "eoqual-denoising[deep,optim,bm3d,ptwt]"     # selon les licences acceptées
export EOQUAL_NLRIDGE_PATH=/chemin/NL-Ridge              # optionnel
python examples/benchmark.py --out benchmark.md          # ~25 min en CPU
python examples/benchmark.py --no-optim --sigmas 0.02 0.1 --out rapide.md
```

Le script ne mesure que les méthodes disponibles dans l'environnement, et
liste les autres avec la raison de leur absence.

## 8. NL-Bayes : moteurs et couleur

`examples/nl_bayes_benchmark.py`, moitié droite (512×256) d'astronaut,
graine 2, σ connu.

### 8.1 Niveaux de gris — moteur vectorisé et boucle (PSNR, dB)

| Moteur | σ = 0,05 | σ = 0,1 | σ = 0,2 | Temps (σ = 0,1) |
|---|---|---|---|---|
| boucle (article, astuce « paste ») | 36,12 | 32,39 | 28,67 | 2,34 s |
| vectorisé, pas 2 | 36,12 | **32,45** | **28,72** | 2,76 s |
| **vectorisé, pas 3 (défaut)** | 36,10 | 32,42 | 28,68 | **1,38 s** |
| vectorisé, pas 4 | 36,07 | 32,36 | 28,65 | 0,84 s |
| vectorisé, pas 3, un seul fil | 36,10 | 32,42 | 28,68 | 4,03 s |

Le moteur vectorisé remplace l'astuce « paste » (séquentielle) par une
grille de références. Au pas 3, il égale la boucle en qualité pour un
temps divisé par 1,7 ; au pas 4, par 2,8 pour −0,03 dB. En un seul fil,
il est plus lent que la boucle : son gain vient du traitement par lots
réparti sur plusieurs cœurs (résultat identique quel que soit le nombre
de fils).

### 8.2 Couleur RGB

| Traitement | PSNR σ = 0,05 | PSNR σ = 0,1 | PSNR σ = 0,2 | SSIM σ = 0,1 | Temps (σ = 0,1) |
|---|---|---|---|---|---|
| *(image bruitée)* | 26,02 | 20,00 | 13,98 | 0,313 | — |
| **`yuv` — Y₀U₀V₀ puis RGB (article, défaut)** | **36,24** | **33,07** | **29,85** | 0,910 | **1,65 s** |
| `rgb` — RGB, groupes communs | 36,02 | 32,77 | 29,31 | **0,912** | 1,92 s |
| `channel` — canal par canal | 35,76 | 32,23 | 28,61 | 0,906 | 4,13 s |
| IPOL C++, canal par canal | 35,44 | 31,83 | 28,17 | 0,872 | 0,76 s |

- **La méthode de l'article (`yuv`) l'emporte nettement** : +0,8 dB sur
  le canal par canal à σ = 0,1, +1,2 dB à σ = 0,2, pour un temps divisé
  par 2,5. Former les groupes sur la luminance Y₀ (moyenne des trois
  canaux, donc moins bruitée) donne des groupes plus fiables, que les
  canaux de chrominance partagent.
- **`rgb`** (groupes communs, sans changement d'espace) récupère une
  partie du gain (+0,5 dB) : l'essentiel vient du partage des groupes
  entre canaux, le reste de la transformée Y₀U₀V₀.
- **Le canal par canal** est le plus lent (trois débruitages complets)
  et le moins bon : chaque canal forme ses groupes sur une image plus
  bruitée et ignore la corrélation entre canaux.

Reproduire : `python examples/nl_bayes_benchmark.py --out nl_bayes.md`
(ajouter `EOQUAL_NLBAYES_EXE` pour la ligne IPOL).
