# Référence technique — eoqual-denoising

Détail de chaque méthode : principe, implémentation, limites connues,
évolutions envisagées. Pour le catalogue résumé et l'installation, voir
`README.md` ; pour l'audit des licences, `THIRD_PARTY_LICENSES.md` ; pour
les citations complètes, [bibliographie](bibliographie.md).

## 1. Vue d'ensemble

### 1.1 Modèle de bruit

Toutes les méthodes du catalogue supposent un **bruit blanc gaussien
additif et stationnaire** d'écart-type σ : y = x + n, n ~ N(0, σ²),
indépendant d'un pixel à l'autre et de même variance partout. Deux
écarts fréquents à ce modèle sont traités à part :

- **bruit poisson-gaussien** des images brutes de capteur (variance qui
  croît avec le signal) : stabilisation de variance, §5 ;
- **bruit d'une image démosaïquée** (corrélé et amplifié par
  l'interpolation des couleurs) : §6 ;
- **bruit coloré** en général (corrélé entre pixels voisins) : non
  traité, voir §10.1.

Le rôle de σ dans le calcul, son origine, son devenir dans ces deux cas
et le principe de son réglage sont rassemblés au §4 ; le réglage
lui-même, comme celui de tous les paramètres, est dans [optimisation](optimisation.md).

### 1.2 Les méthodes

| Méthode | Famille | Principe en une ligne | Section |
|---|---|---|---|
| `bilateral` | filtre local | moyenne locale pondérée par la distance spatiale et radiométrique | §2.1 |
| `nl_means` | non local | moyenne des pixels dont les voisinages se ressemblent | §2.2 |
| `tv_chambolle` | variationnel | variation totale (ROF), solveur dual de Chambolle | §2.3 |
| `tv_bregman` | variationnel | variation totale (ROF), solveur split Bregman | §2.4 |
| `wavelet` | transformée | seuillage des coefficients d'ondelettes | §2.5 |
| `nl_bayes` | non local, 2 étapes | estimateur bayésien gaussien sur groupes de patchs ; couleur | §2.6, [NL-Bayes](nl-bayes/index.md) |
| `nl_bayes_cfa` | non local, itératif | mosaïque de Bayer → RGB : estimateur gaussien (PLE) sur les groupes NL-Bayes, débruite et démosaïque | [NL-Bayes](nl-bayes/cfa.md) §13 |
| `nl_ridge` | non local, 2 étapes | combinaisons linéaires de patchs, poids en forme fermée (SURE puis ridge) | §2.7 |
| `drunet` | réseau | U-Net résiduel conditionné par σ | §3.1 |
| `swinir` | réseau | transformeur à fenêtres glissantes, poids par niveau de bruit | §3.2 |
| `restormer` | réseau | transformeur à attention entre canaux, aveugle | §3.3 |
| `gsdrunet` | réseau | DRUNet en pas de gradient d'un potentiel appris | §3.4 |
| `epll` | a priori de patchs | mélange gaussien appris sur des patchs naturels | §3.5 |
| `tgv` | variationnel | variation totale généralisée d'ordre 2 | §3.6 |
| `wavelet_dict` | transformée | seuillage ℓ1 dans un dictionnaire de plusieurs ondelettes | §3.7 |
| `bm3d` | non local, 2 étapes | filtrage collaboratif de groupes de blocs dans une transformée 3D | §3.8 |
| `nl_ridge_inria` | non local, 2 étapes | NL-Ridge, code des auteurs (référence) | §3.9 |
| `nl_bayes_ipol` | non local, 2 étapes | NL-Bayes, exécutable IPOL (référence) | §3.10 |

Le §2 regroupe les méthodes installées par défaut (le « socle »), le §3
celles qui demandent un extra pip ou un composant externe. Chaque
section décrit le principe, puis l'implémentation, les paramètres et les
limites. Le §4 traite du niveau de bruit σ, commun à toutes ; les §5
et §6 des deux cas où le bruit sort du modèle. L'optimisation des
paramètres fait l'objet d'un document séparé, [optimisation](optimisation.md).
L'utilisation commune à toutes les méthodes (appel,
unités, conventions) est décrite au §7, et les performances comparées
dans [benchmark](benchmark.md).

## 2. Méthodes du socle

Les valeurs par défaut marquées « calibrées » ont été choisies par
balayage sur `astronaut` (256×256, σ ∈ {0,02 ; 0,05 ; 0,1 ; 0,2} en
dynamique [0, 1]) pour maximiser le PSNR moyen. Ce sont des points de
départ raisonnables, pas des optima universels.

### 2.1 `bilateral` — filtre bilatéral

**Fichier** : `backends/bilateral/opencv_impl.py` — `cv2.bilateralFilter`.
**Référence** : Tomasi & Manduchi (1998).

Moyenne pondérée locale dont chaque poids est le produit d'un noyau
gaussien spatial (`sigma_space`, en pixels) et d'un noyau gaussien
radiométrique (`sigma_color`, unités de l'image). Les pixels de niveau
très différent du pixel central (de l'autre côté d'un contour) pèsent
peu, ce qui préserve les bords.

**Paramètres par défaut** : `sigma_space = 1.5` px ;
`sigma_color = 2.5 σ` (calibré : 2 σ est meilleur à faible bruit,
3 σ à fort bruit).

**Limites** : laisse un bruit résiduel « granuleux » dans les zones
homogènes et produit un effet de marches d'escalier sur les dégradés.
C'est la méthode la moins performante du socle à fort bruit (σ ≥ 0,1).

### 2.2 `nl_means` — Non-Local Means

**Fichier** : `backends/nl_means/skimage_impl.py` —
`skimage.restoration.denoise_nl_means`.
**Références** : Buades, Coll & Morel (2005) ; variante rapide Darbon et
al. (2008).

Chaque pixel est remplacé par une moyenne pondérée des pixels d'une
fenêtre de recherche (`patch_distance`), le poids décroissant avec la
distance entre le patch (`patch_size`) centré sur le pixel et celui
centré sur le candidat. Le paramètre `h` règle cette décroissance. Avec
`sigma` fourni, scikit-image retranche la contribution attendue du bruit
à la distance entre patchs.

**Paramètres par défaut** : patchs 7×7, fenêtre de recherche ±11 px,
`h = 0.6 σ` en mode rapide et `0.8 σ` en mode exact (recommandation de la
documentation scikit-image).

**Limites** : lissage excessif des textures fines peu répétitives (les
patchs similaires y sont rares). Coût en O(N · fenêtre²).

### 2.3 `tv_chambolle` — variation totale, solveur de Chambolle

**Fichier** : `backends/tv/skimage_impl.py` —
`skimage.restoration.denoise_tv_chambolle`.
**Références** : modèle ROF, Rudin, Osher & Fatemi (1992) ; solveur
dual, Chambolle (2004).

Minimise ½‖u − f‖² + `weight` · TV(u), où TV(u) est la variation totale
(norme L1 du gradient). La solution est constante par morceaux : le
bruit disparaît dans les zones homogènes et les contours francs sont
préservés.

**Paramètres par défaut** : `weight = 0.75 σ` (calibré) ; arrêt sur
variation relative `eps = 2e-4` ou 200 itérations.

**Limites** : effet « cartoon » (marches d'escalier sur les dégradés,
perte des textures de faible contraste) et biais de contraste sur les
petits objets.

### 2.4 `tv_bregman` — variation totale, split Bregman

**Fichier** : `backends/tv/skimage_impl.py` —
`skimage.restoration.denoise_tv_bregman`.
**Référence** : Goldstein & Osher (2009).

Même modèle ROF, résolu par la méthode split Bregman. **Attention au sens
du paramètre** : dans scikit-image, `weight` pondère ici l'*attache aux
données* (plus grand = moins lisse), à l'inverse de `tv_chambolle`.

**Paramètres par défaut** : `weight = 1/σ` (calibré) ; TV isotrope ; 100
itérations maximum.

**Limites** : celles du modèle ROF (voir §2.3). Avec ces réglages par
défaut, le résultat est légèrement inférieur à `tv_chambolle`
(0,5 à 1 dB) car le solveur converge vers une solution un peu moins
régularisée. Les deux sont gardés car leurs comportements en convergence
et en temps diffèrent selon la taille d'image.

### 2.5 `wavelet` — seuillage d'ondelettes

**Fichier** : `backends/wavelet/skimage_impl.py` —
`skimage.restoration.denoise_wavelet`.
**Références** : BayesShrink, Chang, Yu & Vetterli (2000) ; VisuShrink,
Donoho & Johnstone (1994).

Décomposition en ondelettes orthogonales, seuillage doux (ou dur) des
coefficients de détail, reconstruction. `BayesShrink` calcule un seuil
par sous-bande à partir de la variance estimée du signal. `VisuShrink`
applique le seuil universel σ√(2 ln N), plus lissant. L'image est
normalisée en [0, 1] avant l'appel (scikit-image tronque sa sortie à
cet intervalle), puis ramenée à son échelle d'origine.

**Paramètres par défaut** : ondelette `db2` (meilleure que `db1` et
`sym8` en moyenne sur le balayage), règle `rule="BayesShrink"`,
seuillage doux (`mode="soft"`).

**Limites** : transformée décimée, donc non invariante par translation,
d'où des artefacts de type pseudo-Gibbs près des contours. La méthode
est la plus rapide du socle mais la moins performante en PSNR.

### 2.6 `nl_bayes` — NL-Bayes, réimplémentation MIT

**Fichiers** : `backends/nl_bayes/native.py` (adaptateur) ; le calcul est
fait par la bibliothèque **nlbayes-ext** (MIT, même auteur,
https://github.com/oam31/nlbayes-ext, numpy et scipy seulement),
dépendance du socle. `nl_bayes_cfa` passe de même
par `backends/nl_bayes/cfa.py`.
**Références** : Buades, Lebrun & Morel (2013, SIAM) ; article IPOL de
Lebrun, Buades & Morel (2013), **seule source utilisée** (clean-room :
aucun code consulté, voir `THIRD_PARTY_LICENSES.md` §2.1).

**Principe** : pour chaque patch, les patchs semblables sont regroupés ;
le groupe est modélisé par une loi gaussienne (moyenne et covariance) et
chaque patch est estimé par un filtre de Wiener. Deux étapes : la
seconde forme ses groupes et sa covariance sur le résultat de la
première. Couleur traitée dans l'espace Y₀U₀V₀ de l'article.

**Résultats** : 32,42 dB à σ = 0,1 sur le benchmark principal, au niveau
de BM3D (32,38) et de NL-Ridge.

**Tout le détail** — mathématiques (covariance, filtre de Wiener), rôle de
chaque paramètre, place de σ, couleur et images démosaïquées, moteurs,
études de sensibilité et réglage : **[NL-Bayes](nl-bayes/index.md)**.

### 2.7 `nl_ridge` — NL-Ridge, réimplémentation MIT

**Fichier** : `backends/nl_ridge/native.py` — implémentation originale
numpy, sans dépendance.
**Références** : Herbreteau & Kervrann, ICIP 2022 (arXiv:2203.00570) et
SIAM J. Imaging Sci. 2025 (arXiv:2402.13816), **seules sources
utilisées** (clean-room : le code des auteurs, AGPL, n'a pas été
consulté pour l'algorithme).

Même schéma que NL-Bayes (groupes, deux étapes, agrégation), mais chaque
patch est estimé par une **combinaison linéaire des patchs de son
groupe**. Avec Y (n × k) les k patchs de n pixels d'un groupe :
X̂ = Y Θ, Θ (k × k) minimisant une estimation du risque quadratique.

- **Étape 1** (estimateur sans biais du risque, SURE) :
  Θ₁ = I − nσ²(YᵀY)⁻¹.
- **Étape 2** (« adaptation interne », régression ridge multivariée sur
  le groupe correspondant X̂₁ de l'image de l'étape 1) :
  Θ₂ = I − nσ²(X̂₁ᵀX̂₁ + nσ²I)⁻¹.
- **`constraints="affine"`** : poids de somme 1 (formules 6 et 15 de
  l'article 2025).
- **Reprojection pondérée** : chaque patch estimé j pèse 1/‖Θ·,j‖².
- **Accélération** : références sur une grille de pas 4, fenêtre 37×37.

**Paramètres** (tableau 1 des articles, σ sur [0, 255]) :

| σ | p₁ | p₂ | k₁ | k₂ |
|---|---|---|---|---|
| ≤ 15 | 7 | 7 | 18 | 55 |
| 15-35 | 9 | 9 | 18 | 90 |
| > 35 | 11 | 9 | 20 | 120 |

À noter : à l'étape 1, NL-Ridge a besoin de **n > k**, soit k₁ < p₁²
(YᵀY, de taille k₁ × k₁, est de rang au plus p₁²) — l'inverse de
NL-Bayes (N > k²) ; ses patchs sont donc grands et ses groupes petits.
Un réglage avec k₁ ≥ p₁² est refusé (`ValueError`) : la matrice serait
singulière et le résultat inutilisable (1,6 dB mesurés à p₁ = 5,
k₁ = 30). L'espace de recherche porte la contrainte correspondante
([optimisation](optimisation.md) §3).

**Validation** : reproduit le code des auteurs à 0,03 dB près sur
astronaut 256×256 (29,35 contre 29,34 dB à σ = 0,1), en temps comparable
(test `test_nl_ridge_native_matches_authors_code`). La variante affine
est 0,02 à 0,06 dB en dessous.

**Non porté** : bruits de Poisson et poisson-gaussien de l'article 2025
(utiliser `vst_denoise`), contraintes coniques et convexes (sans forme
fermée), couleur.

## 3. Méthodes en extra ou composant externe

**Réseaux de neurones (§3.1 à §3.4).** DRUNet,
SwinIR, Restormer et GS-DRUNet sont fournis par
[eoqual-neural](https://github.com/EOQual/eoqual-neural), qui regroupe les
réseaux d'EOQual par rôle (débruiteur appris, réseau déplié…). eoqual-denoising
n'en garde que l'adaptation à son contrat (`backends/neural.py` : unités du
capteur, σ estimé, `data_range`), derrière le même appel
`denoise(image, "drunet", sigma=…)` ; extra `neural`. Les descriptions
ci-dessous restent valables ; le code des réseaux est dans
`eoqual.neural.denoisers`.

### 3.1 `drunet` — réseau DRUNet (extra `neural`)

**Fichier** : `eoqual.neural.denoisers` (adaptation : `backends/neural.py`) — `deepinv.models.DRUNet`.
**Référence** : Zhang et al. (2022).

U-Net à quatre échelles et blocs résiduels. Le niveau de bruit lui est
fourni en canal d'entrée supplémentaire, ce qui permet d'utiliser un
**seul réseau pour toute la plage de σ**. Les poids niveaux de gris
pré-entraînés par l'équipe deepinv (`drunet_deepinv_gray_finetune_26k`,
~125 Mo) sont téléchargés au premier appel puis mis en cache par PyTorch
(`TORCH_HOME`). L'image est ramenée en [0, 1] via `data_range`. Le
modèle chargé est gardé en mémoire entre deux appels sur le même
périphérique.

**Limites** : entraîné sur des images naturelles 8 bits. Sur des images
EO aux statistiques différentes (fort contraste, saturations, très haute
dynamique), le gain n'est pas garanti et peut s'accompagner
d'hallucinations de texture. **Valider sur données capteur avant tout
usage opérationnel.** Temps CPU : ~2,5 s pour 256×256 sur la machine de développement ;
GPU fortement recommandé pour les grandes scènes.

### 3.2 `swinir` — transformeur SwinIR (extra `neural`)

**Fichier** : `eoqual.neural.denoisers` (adaptation : `backends/neural.py`) — `deepinv.models.SwinIR`.
**Référence** : Liang et al. (2021).

Transformeur à fenêtres glissantes (Swin) : l'auto-attention est calculée
dans des fenêtres locales de 8×8 pixels, décalées d'une couche à
l'autre pour propager l'information. **Non conditionné par σ** : les
auteurs publient un jeu de poids niveaux de gris par niveau de bruit
(σ = 15, 25, 50 en échelle [0, 255]). Le jeu le plus proche de σ est
choisi automatiquement (paramètre `noise_level` pour l'imposer).

**Limites** : entre deux niveaux entraînés, le résultat se dégrade
(σ = 0,05 ≈ 12,75/255, traité avec les poids « 15 » : 1,7 dB sous
DRUNet). C'est le modèle le plus lent du catalogue (~7 s en CPU pour
256×256), et chaque jeu de poids pèse ~45 Mo.

### 3.3 `restormer` — transformeur Restormer (extra `neural`)

**Fichier** : `eoqual.neural.denoisers` (adaptation : `backends/neural.py`) — `deepinv.models.Restormer`.
**Référence** : Zamir et al. (2022).

Transformeur dont l'attention porte sur les **canaux** plutôt que sur
les positions, ce qui rend son coût linéaire en nombre de pixels et
permet de traiter de grandes images. Poids niveaux de gris **aveugles** :
un seul réseau, entraîné sur σ ∈ [0, 50]/255 tiré au hasard, qui
n'utilise pas σ (le paramètre est accepté mais ignoré).

**Limites** : hors de son domaine d'entraînement (bruit normalisé
supérieur à 50/255), il échoue nettement, faute de pouvoir être informé
du niveau de bruit : à faible flux, même après stabilisation de variance
(§5), il perd 5 à 8,5 dB face à DRUNet. Ne l'utiliser que dans son domaine
(σ ≤ 0,2 en dynamique [0, 1]).

### 3.4 `gsdrunet` — GS-DRUNet (extra `neural`)

**Fichier** : `eoqual.neural.denoisers` (adaptation : `backends/neural.py`) — `deepinv.models.GSDRUNet`.
**Référence** : Hurault, Leclaire & Papadakis (2022).

Même architecture que DRUNet, mais le débruiteur est défini comme un
**pas de gradient d'un potentiel scalaire appris** : D(x) = x − ∇g(x).
Débruiter revient à descendre une fonctionnelle explicite. C'est ce qui
garantit la convergence quand il sert d'a priori dans un schéma
plug-and-play (déconvolution, super-résolution). Utilisé seul, il est
~0,2 dB sous DRUNet. Le calcul passe par l'autograd de PyTorch (gradient
du potentiel).

**Chargement des poids** : le fichier publié contient aussi des
compteurs de métriques d'entraînement, que deepinv 0.4.2 refuse en
chargement strict. La couche d'appel charge donc les poids elle-même,
ne garde que les clés du réseau et vérifie qu'aucune ne manque.

**Intérêt pour EOQual** : brique naturelle d'un futur schéma de
déconvolution plug-and-play convergent (lien avec la déconvolution et la
restauration de PSF).

### 3.5 `epll` — EPLL (extra `deep`)

**Fichier** : `backends/deepinv_zoo/models.py` — `deepinv.models.EPLLDenoiser`.
**Référence** : Zoran & Weiss (2011).

A priori **explicite** sur les patchs 6×6 : mélange de 200 gaussiennes
appris sur BSDS500. L'image restaurée maximise la vraisemblance moyenne
de tous ses patchs (qui se chevauchent), par un schéma
demi-quadratique à poids croissants. C'est un modèle génératif lisible
(chaque composante est une gaussienne de patch), entre les méthodes non
locales et les réseaux.

**Limites** : ~1 dB sous BM3D, et 3 à 4 s en CPU pour 256×256. Le
paramètre `batch_size` limite la mémoire sans changer le résultat.

### 3.6 `tgv` — variation totale généralisée (extra `deep`)

**Fichier** : `backends/deepinv_zoo/models.py` — `deepinv.models.TGVDenoiser`.
**Références** : Bredies, Kunisch & Pock (2010) ; solveur primal-dual de
Condat (2013).

TGV d'ordre 2 : minimise ½‖x − y‖² + λ₁‖r‖ + λ₂‖J(∇x − r)‖ avec un champ
auxiliaire r. Là où la TV force une image constante par morceaux, la TGV
autorise des morceaux **affines** : les contours restent francs, mais
les dégradés (éclairement, ombrages, rampes radiométriques) ne se
transforment plus en marches d'escalier. Sans apprentissage ni poids.
deepinv calcule λ₁ = 0,1·`weight` et λ₂ = 0,15·`weight`.

**Paramètres par défaut** : `weight = 7.5 σ` (calibré sur astronaut ;
l'équivalent du 0,75 σ de `tv_chambolle` à ce facteur 10 près) ; 1 000
itérations maximum.

**Limites** : sur astronaut, PSNR proche de `tv_chambolle` (±0,2 dB). Le
gain se voit sur les dégradés lisses plus que sur le PSNR global. ~3 s
en CPU pour 256×256 (Chambolle-Pock en PyTorch).

### 3.7 `wavelet_dict` — dictionnaire d'ondelettes redondant (extras `deep` + `ptwt`)

**Fichier** : `backends/deepinv_zoo/models.py` —
`deepinv.models.WaveletDictDenoiser`.
**Références** : Starck, Murtagh & Fadili (2010) ; schéma proximal
parallèle, Combettes & Pesquet (2008).

Régularisation ℓ1 dans un dictionnaire formé de plusieurs bases
d'ondelettes (par défaut `db8` et `db4`, 3 niveaux). À chaque itération,
l'image est seuillée dans chaque base, puis les résultats sont
moyennés. La redondance atténue les artefacts de bloc et de
pseudo-Gibbs d'une base unique.

**Paramètres par défaut** : seuil doux `1.25 σ` (calibré) ; 10 itérations
maximum.

**Pourquoi un extra séparé** : la transformée en ondelettes de deepinv
repose sur `ptwt` (PyTorch Wavelet Toolbox), sous **EUPL-1.2**, une
licence copyleft (voir `THIRD_PARTY_LICENSES.md` §2.2).

**Limites** : sur astronaut, au niveau de `wavelet` (scikit-image), donc
pas de gain en PSNR. Gardé pour sa formulation variationnelle, qui
s'intègre dans les schémas de restauration de deepinv.

### 3.8 `bm3d` — BM3D (extra `bm3d`, **non commercial**)

**Fichier** : `backends/bm3d/tau_wrapper.py` — paquet `bm3d` (TAU).
**Références** : Dabov et al. (2007) ; Mäkinen, Azzari & Foi (2020).

Regroupe les blocs similaires en piles 3D, applique une transformée 3D
séparable et un seuillage dur (étape 1), puis un filtrage de Wiener
collaboratif guidé par l'estimée de l'étape 1 (étape 2), et enfin agrège
les estimées pondérées de chaque bloc. `hard_threshold_only=True`
n'exécute que l'étape 1.

**Pourquoi en extra** : licence TAU limitée à l'usage **non commercial**
(voir `THIRD_PARTY_LICENSES.md` §2.2). La méthode semble en outre
couverte par un brevet (§3 du même document).

**Limites** : la version 4.x accepte aussi un bruit coloré (densité
spectrale `sigma_psd` 2D). Ce cas n'est pas encore exposé par la couche
d'appel (bruit blanc uniquement) — voir §10.1.

### 3.9 `nl_ridge_inria` — NL-Ridge, code des auteurs (composant externe, **AGPL-3.0**)

**Fichier** : `backends/nl_ridge/external.py`. Même méthode que
`nl_ridge` (§2.7), par le code des auteurs ; utile pour comparer.
**Références** : Herbreteau & Kervrann (2022, 2025).

Deux étapes non locales comme BM3D/NL-Bayes, mais l'estimateur de chaque
groupe de patchs similaires est une **régression ridge à forme fermée**
qui combine linéairement les patchs du groupe. Les coefficients sont
appris sans supervision sur l'image elle-même. Méthode entièrement
PyTorch, exécutable sur GPU.

**Mise en place** : cloner https://github.com/sherbret/NL-Ridge, puis
`export EOQUAL_NLRIDGE_PATH=/chemin/NL-Ridge` (ou paramètre
`nlridge_path=`) ; installer l'extra `deep` (torch, einops). L'image est
ramenée sur [0, 255] et les paramètres recommandés par les auteurs pour
le palier de σ correspondant sont appliqués automatiquement (σ ≤ 15 /
≤ 35 / au-delà). Tout paramètre de `NLRidge.forward` peut être passé en
mot-clé pour les remplacer.

**Limites** : voir l'avertissement AGPL dans `THIRD_PARTY_LICENSES.md`
§2.3. Les modèles de bruit Poisson et Poisson-gaussien du code amont ne
sont pas exposés : passer par `vst_denoise` (§5), commun à toutes les
méthodes.

### 3.10 `nl_bayes_ipol` — NL-Bayes, exécutable IPOL (externe, **GPL-3.0+**)

**Fichier** : `backends/nl_bayes/ipol_external.py`.
**Références** : Buades, Lebrun & Morel (2013) ; implémentation IPOL,
Lebrun, Buades & Morel (2013).

Même méthode que `nl_bayes` (§2.6), par l'implémentation C++ de
référence. Utile pour comparer ; `nl_bayes` est préférable en usage
courant (MIT, sans composant externe, sans quantification, meilleur
PSNR en niveaux de gris : 32,45 contre 32,02 dB).

Méthode et paramètres : [NL-Bayes](nl-bayes/index.md).

**Mise en place** : télécharger l'archive `nl-bayes_20130617.tar.gz` de
https://www.ipol.im/pub/art/2013/16/, installer libpng et fftw3, compiler
(`make`, ou `make OMP=1` pour le parallélisme), puis
`export EOQUAL_NLBAYES_EXE=/chemin/NL_Bayes` (ou paramètre
`nlbayes_exe=`). Sous macOS récent (clang), la compilation demande
d'ajouter `-include string.h` aux `CFLAGS` (déclaration manquante de
`strcmp` dans `io_png.c`), les chemins Homebrew de libpng, fftw et libomp,
et `-Xpreprocessor -fopenmp` / `-lomp` (le code inclut `omp.h` sans
condition). Ligne testée sur macOS arm64 :
`make CFLAGS="-O3 $INC -include string.h -Xpreprocessor -fopenmp"
CXXFLAGS="-O3 $INC -Xpreprocessor -fopenmp" LDFLAGS="$LIB -lpng -lfftw3f -lomp -lm"`,
avec `INC`/`LIB` pointant sur `/opt/homebrew/opt/{libpng,fftw,libomp}`.

**Échange** : l'image est ramenée sur [0, 255], **quantifiée sur 8 bits**
et écrite en PNG dans un dossier temporaire. Le programme est lancé avec
`add_noise = 0` (pas de re-bruitage) et l'estimée finale est relue (ou
celle de l'étape 1 avec `return_basic=True`). La quantification ajoute
une erreur d'écart-type ~0,29 niveau sur 255, négligeable sauf à très
faible σ.

**Limites** : voir l'avertissement GPL dans `THIRD_PARTY_LICENSES.md`
§2.3. Le protocole d'appel (ordre des 13 arguments, fichiers, échelle)
est testé avec un faux exécutable
(`test_nl_bayes_ipol_command_line_protocol`), et l'exécutable réel par
`test_nl_bayes_ipol` dès que `EOQUAL_NLBAYES_EXE` est défini. Les
paramètres de cette version 2013 de l'IPOL ne sont pas
réglables en ligne de commande, d'où un PSNR un peu inférieur à
`nl_bayes`, qui applique en plus les patchs de 5 recommandés pour les
niveaux de gris.

## 4. Le niveau de bruit σ

Ce chapitre rassemble ce qui concerne σ pour toutes les méthodes : ce
qu'il représente, comment les méthodes s'en servent, d'où vient sa
valeur, ce qu'il devient avec la stabilisation de variance et après un
démosaïçage, et comment le régler par optimisation. Ce qui est propre à
NL-Bayes (place exacte de σ dans ses formules, seuil τ₀, redondance avec
β et γ) est dans [NL-Bayes](nl-bayes/parametres.md) §7.

### 4.1 Ce que σ représente

Toutes les méthodes du catalogue supposent l'image bruitée de la forme :

```
y = x + n          n : bruit blanc gaussien, d'écart-type σ
```

- **blanc** : le bruit d'un pixel est indépendant de celui de ses voisins ;
- **stationnaire** : σ est le même partout dans l'image.

σ est donc **une seule valeur pour toute l'image**, et toute la difficulté
d'un débruiteur est de doser son effet par rapport à elle : trop fort, il
efface des détails ; trop faible, il laisse du bruit.

Deux cas réels sortent de ce modèle et sont traités à part :

| Cas | Pourquoi le modèle ne tient pas | Traitement |
|---|---|---|
| Image brute de capteur | la variance du bruit croît avec la luminance (bruit photonique) | stabilisation de variance (§4.4, §5) |
| Image démosaïquée | le bruit est corrélé entre pixels et entre canaux, et amplifié en luminance | facteur C (§4.5, §6) |

### 4.2 Comment les méthodes se servent de σ

| Famille | Usage de σ |
|---|---|
| filtres et méthodes variationnelles (`bilateral`, `nl_means`, `tv_*`, `tgv`) | un poids ou un seuil proportionnel à σ (espaces de recherche en multiples de σ, [optimisation](optimisation.md) §3) |
| ondelettes (`wavelet`, `wavelet_dict`) | seuil des coefficients proportionnel à σ |
| `bm3d` | seuil proportionnel à σ (étape 1), σ² comme variance du bruit dans le filtre de Wiener (étape 2) |
| `nl_bayes`, `nl_bayes_ipol` | σ² comme variance du bruit dans deux filtres de Wiener, γσ comme seuil des zones plates ([NL-Bayes](nl-bayes/parametres.md) §7) |
| `nl_bayes_cfa` | βσ² dans U·C·Uᵀ + βσ²I, bruit de la mosaïque (1 après stabilisation) ([NL-Bayes](nl-bayes/cfa.md) §13) |
| `nl_ridge`, `nl_ridge_inria` | nσ² dans les deux régressions de patchs, Θ₁ et Θ₂ (§2.7) |
| `drunet`, `gsdrunet` | carte de niveau de bruit en entrée du réseau |
| `swinir` | choix d'un jeu de poids entraîné (15, 25 ou 50) |
| `epll` | variance du bruit dans l'estimation par maximum de vraisemblance des patchs |
| `restormer` | aucun (réseau aveugle) |

Dans tous les cas, σ dose l'effet : trop fort, la méthode efface des
détails ; trop faible, elle laisse du bruit.

### 4.3 D'où vient la valeur de σ

1. **Valeur d'entrée** : `sigma=…`, dans les **unités de l'image** (DN,
   réflectance...).
2. **Sinon, estimation** par `estimate_noise_sigma` (estimateur MAD de
   Donoho & Johnstone, sous-bande d'ondelettes la plus fine) ; en couleur,
   moyenne des trois canaux. Deux pièges : il **surestime** σ sur une
   image très texturée, et il **trouve la moitié** du vrai σ sur une image
   démosaïquée (§6.6). Fournir σ dès qu'il est connu (caractérisation du
   capteur).
3. **Facteur `sigma_scale`** (optimisation seulement) : multiplie σ avant
   tout le reste.
4. **Échelle 0-255** (NL-Bayes, NL-Ridge) : σ est converti, avec
   l'image, sur l'échelle 0-255 de leurs articles via `data_range` (par
   défaut le min/max de l'image). C'est un calcul en flottants, pas une
   quantification. Pour une série d'images, fixer `data_range` à la
   dynamique du capteur (ex. `(0, 4095)`) ; détail pour NL-Bayes :
   [NL-Bayes](nl-bayes/parametres.md) §7.2.
5. **Couleur** : un seul σ, supposé identique sur les trois canaux.

### 4.4 Bruit qui dépend de la luminance : modèle (a, b) et stabilisation

Sur une image brute de capteur, la **variance** du bruit suit la
luminance L (niveau d'obscurité soustrait) :

```
σ²(L) = a + b·L        a : plancher (bruit de lecture²)        b : pente (bruit photonique, 1/gain)
```

**Convention, commune à tous les modules EOQual (eoqual-measure compris)
et aux fichiers de configuration des chaînes de débruitage : a est le
plancher, b la pente.** Un bruit stationnaire d'écart-type σ s'écrit
**a = σ², b = 0**. Les `a` et `b` de la PTC d'eoqual-measure s'emploient
tels quels. Attention aux articles : le modèle poissonnien-gaussien y est
souvent noté σ² = a·y + b (Foi et al., 2008), lettres inversées.

Les coefficients `a` et `b` se mesurent : courbe de transfert photonique
(PTC d'eoqual-measure) ou recette en vol. Le bruit est faible dans les
zones sombres et fort dans les zones claires : un σ unique ne peut pas
convenir partout.

La **stabilisation de variance** (§5) applique à l'image une fonction
construite à partir de (a, b), qui ramène ce bruit variable à un bruit
d'écart-type **1 partout**. La méthode reçoit alors `σ = 1` :

- le modèle (a, b) a été pris en compte **en amont**, par la
  transformée, puis par son inverse exacte au retour ;
- la méthode fait le même calcul (§4.2), avec une valeur de σ connue
  d'avance ;
- pour les méthodes qui travaillent sur l'échelle 0-255 (NL-Bayes,
  NL-Ridge), σ = 1 est converti avec **la dynamique de l'image
  transformée**, environ 2·√(signal crête en électrons) : σ ≈ 13 sur
  cette échelle à 100 e⁻ en crête ([NL-Bayes](nl-bayes/parametres.md) §7.2).

**σ est imposé, pas fourni.** `vst_denoise` refuse un argument `sigma`
(erreur, et non simple avertissement : un σ dans les unités de l'image
n'a plus de sens après la transformée, et l'ignorer en silence masquerait
une erreur). Le seul écart permis à σ = 1 est explicite : le paramètre
`noise_factor` (§4.5).

| Situation | Stabilisation | Effet mesuré (§5.5) |
|---|---|---|
| Image brute, bruit photonique dominant | **oui** | +4 à +10 dB |
| Bruit de lecture dominant | inutile | ±0,2 dB |
| Bruit déjà stationnaire (simulé, quantification) | non | — |
| Image recalibrée ou rééchantillonnée | à éviter | le modèle (a, b) ne tient plus |

### 4.5 Image démosaïquée : le facteur C

Le démosaïçage interpole chaque couleur à partir des photosites voisins.
Le bruit devient corrélé et sa variance en luminance est multipliée par un
facteur κ_Y propre au démosaïqueur (0,6 à 2,6, §6.3), la forme affine
étant conservée : `var_Y(L) ≈ κ_Y · (a + b·L)`. La recette (§6.4) :

1. stabiliser avec les (a, b) **du capteur**, inchangés ;
2. débruiter avec **σ = C** au lieu de 1, `C = √κ_Y + 0,2` (√κ_Y pour la
   luminance, +0,2 pour la chrominance basse fréquence) ;
3. inverser la stabilisation.

Dans le code, C est le paramètre **`noise_factor`** de `vst_denoise`
(défaut 1). Une valeur hors de [0,5 ; 2,5] déclenche un avertissement :
C va de 0,57 à 1,8 pour les démosaïqueurs mesurés, au-delà il s'agit
probablement d'une erreur d'unités.

Augmenter a et b pour tenir compte du démosaïçage revient presque au même
pour le débruitage, mais **fausse l'inverse exacte** au retour (§6.5) : le
facteur se met sur σ, dans le domaine stabilisé, pas sur (a, b).

### 4.6 Régler σ

σ, comme les autres paramètres, peut se régler par optimisation sur une
image de référence. Le principe : **fixer ce qui se mesure** ((a, b),
l'offset : propriétés du capteur), **optimiser ce qui s'ajuste**
(`sigma_scale`, qui multiplie σ dans le domaine où travaille la méthode ;
c'est le facteur C après démosaïçage, à partir de √κ_Y + 0,2). Détail,
cas par cas, avec bornes, budget et exemple : [optimisation](optimisation.md) §2.

## 5. Bruit poisson-gaussien : stabilisation de variance

**Fichier** : `vst.py` — implémentation originale (numpy, scipy), écrite
depuis les publications, sans code tiers.
**Références** : Anscombe (1948) ; GAT, Murtagh, Starck & Bijaoui
(1995) ; inverses optimales, Mäkitalo & Foi (2011, 2013).

**Le terme VST.** VST signifie *variance-stabilizing transform*,
« transformée stabilisatrice de variance ». C'est le nom d'une famille de
méthodes statistiques : une fonction f est choisie pour que la variance
de f(y) ne dépende plus de la moyenne de y. La transformée d'Anscombe
(1948) est la VST classique de la loi de Poisson ; la transformée
d'Anscombe généralisée (GAT, Murtagh et al. 1995) l'étend au bruit
poisson-gaussien. D'autres VST existent (Freeman-Tukey, transformée de
Fisz). La fonction s'appelle `vst_denoise` d'après le principe
(stabiliser, débruiter, inverser), qui ne dépend pas de la transformée
choisie. Ce module n'implémente que la GAT, et les fonctions de bas
niveau portent son nom (`generalized_anscombe`,
`inverse_generalized_anscombe`).

### 5.1 Modèle

Une image brute de capteur suit, pixel par pixel :

    z = p / K + offset + n,   p ~ Poisson(λ),   n ~ N(0, σ_DN²)

- `gain` = K, gain de conversion en e⁻/DN ;
- `read_noise_dn` = σ_DN, bruit de lecture en DN ;
- `offset_dn` = niveau d'obscurité en DN.

La variance vaut σ²(z) = (z̄ − offset)/K + σ_DN² : elle **croît avec le
signal**. C'est le modèle du §4.4, `σ²(L) = a + b·L`, avec
**a = σ_DN²** (plancher) et **b = 1/K** (pente), L = z̄ − offset : la
droite de la PTC d'eoqual-measure, mêmes lettres. Les grandeurs
physiques se lisent aussi directement sur ses mesures :
`measure_ptc(...).gain`, `.read_noise_dn`, et le niveau moyen de
`measure_dark_frame` pour l'offset.

Un débruiteur gaussien appliqué tel quel reçoit un σ unique (estimé ou
fourni) alors que le vrai σ varie dans l'image. Il sur-lisse les zones
sombres et sous-lisse les zones claires, ou l'inverse.

### 5.2 Transformée directe (GAT)

En électrons, y = K·(z − offset), et avec σ = K·σ_DN :

    f(y) = 2·sqrt(y + 3/8 + σ²)   (0 si l'argument est négatif)

Après f, la variance du bruit vaut ≈ 1 **quel que soit λ**, dès quelques
électrons (vérifié : 0,98 à 1,04 pour λ ≥ 3 e⁻). Le cas pur Poisson
(`gain=1`, `read_noise_dn=0`) redonne l'Anscombe classique,
2·sqrt(z + 3/8).

### 5.3 Inverses

Le débruiteur estime E[f(y)], pas f(E[y]). Comme f est concave, les deux
diffèrent et l'inverse « naïve » est biaisée. Trois inverses sont
proposées :

| `inverse` | Formule | Comportement |
|---|---|---|
| `"exact_unbiased"` (défaut) | inverse numérique de λ ↦ E[f(y) \| λ] | **sans biais à tout flux** |
| `"asymptotic"` | (D/2)² − 1/8 − σ² | juste au-dessus de ~20 e⁻, biaisée en dessous |
| `"algebraic"` | (D/2)² − 3/8 − σ² | biais constant de −0,25 e⁻, et pire à faible flux |

**Calcul de l'inverse exacte.** Pour une grille de ~950 valeurs de λ
dans [0, 400] e⁻ (dense près de 0), E[f(y) | λ] = Σₖ P(k; λ)·E[f(k + n)].
La loi de Poisson est sommée jusqu'à k = λ_max + 12√λ_max + 30, et
l'espérance gaussienne E[f(k + n)] est intégrée par trapèzes sur ±10σ.
La table, monotone, est inversée par interpolation linéaire. Elle est
calculée une fois par valeur de σ (~0,6 s), puis mise en cache. Au-delà
de 400 e⁻, l'inverse asymptotique prend le relais : l'écart au raccord
est < 1e-7 en relatif. Une valeur sous E[f | λ = 0] est ramenée à λ = 0.

**Validation (Monte-Carlo, 4·10⁵ tirages).** Pour σ ∈ {0 ; 0,5 ; 3} e⁻
et λ ∈ {0,3 ; 1 ; 3 ; 10 ; 50 ; 300} e⁻, l'inverse exacte appliquée à la
moyenne empirique de f(y) retrouve λ à moins de 2,1 incertitudes-types.
À λ = 0,3 e⁻ sans bruit de lecture, l'inverse asymptotique donne
0,47 (+57 %) et l'algébrique 0,22 (−26 %). Ces cas sont couverts par
`tests/test_vst.py`, avec une tolérance de 5 incertitudes-types,
stable sur les graines 0 à 8 (`EOQUAL_TEST_SEED`).

### 5.4 Usage

```python
import eoqual.denoising as d
out = d.vst_denoise(raw, "drunet", gain=2.0, read_noise_dn=1.5, offset_dn=100.0)
```

`vst_denoise` applique la GAT, appelle la méthode choisie avec
`sigma = 1`, puis l'inverse exacte. Un argument `sigma` est refusé ; le
seul écart à 1 passe par `noise_factor` (facteur C d'une image
démosaïquée, §4.5 et §6.4). L'image peut être couleur (H, W, 3) pour
`nl_bayes`. Les briques sont aussi exposées
séparément : `generalized_anscombe`, `inverse_generalized_anscombe`,
`expected_gat`, `add_poisson_gaussian_noise` (simulateur du même
modèle).

### 5.5 Résultats (`examples/low_light_vst.py`)

`astronaut` 256×256 ramenée à un flux crête donné, K = 2 e⁻/DN, offset
100 DN. PSNR en dB, référence = signal moyen en DN.

**Bruit photonique dominant** (σ_lecture = 0,25 DN = 0,5 e⁻) :

| Pic | Méthode | Direct | GAT + exacte | GAT + asymptotique | GAT + algébrique |
|---|---|---|---|---|---|
| 2 e⁻ | `nl_means` | 14,47 | **17,60** | 17,28 | 15,06 |
| 2 e⁻ | `bm3d` | 16,18 | **20,31** | 19,81 | 17,06 |
| 2 e⁻ | `drunet` | 11,41 | **21,16** | 20,53 | 17,26 |
| 5 e⁻ | `nl_means` | 16,90 | **21,26** | 21,15 | 19,90 |
| 5 e⁻ | `bm3d` | 18,46 | **23,22** | 23,11 | 21,67 |
| 5 e⁻ | `drunet` | 15,27 | **24,36** | 24,22 | 22,45 |
| 20 e⁻ | `nl_means` | 20,18 | **25,43** | 25,42 | 25,18 |
| 20 e⁻ | `drunet` | 20,84 | **28,66** | 28,64 | 28,16 |

**Bruit de lecture dominant à faible flux** (σ_lecture = 1,5 DN = 3 e⁻) :

| Pic | Méthode | Direct | GAT + exacte |
|---|---|---|---|
| 5 e⁻ | `nl_means` | 17,65 | 17,74 |
| 5 e⁻ | `drunet` | **19,73** | 19,54 |
| 20 e⁻ | `drunet` | 23,78 | **26,41** |
| 100 e⁻ | `bm3d` | 28,98 | **31,04** |
| 100 e⁻ | `drunet` | 28,40 | **32,51** |

Lecture :

- dès que le bruit photonique compte, la GAT apporte **+2 à +10 dB**
  par rapport au débruiteur appliqué directement (le gain est le plus
  fort pour DRUNet, très sensible à un σ mal adapté) ;
- le choix de l'inverse compte **sous ~20 e⁻** : l'exacte gagne 3 à 4 dB
  sur l'algébrique à 2 e⁻, et 0,3 à 0,6 dB sur l'asymptotique ;
- quand le bruit de lecture domine (3 e⁻ contre au plus √5 ≈ 2,2 e⁻ de
  bruit photonique à 5 e⁻ crête), le bruit est presque gaussien et
  stationnaire : la GAT n'apporte rien (±0,2 dB) ;
- `restormer` (aveugle) ne tire pas parti de la GAT (§3.3) : préférer
  une méthode conditionnée par σ.

**Limites** : le modèle suppose des paramètres K, σ_DN et offset
**uniformes** sur l'image. Une PRNU ou une DSNU forte (gain ou offset
variables d'un pixel à l'autre) doit être corrigée avant. La saturation
et la non-linéarité ne sont pas modélisées. L'estimation de K et σ_DN
sur une seule image (sans PTC) n'est pas fournie ; utiliser la PTC
d'eoqual-measure.

## 6. Cas particulier : débruiter une image démosaïquée

### 6.1 Quand ce chapitre s'applique

Une image couleur issue d'un capteur à **matrice de filtres colorés**
(Bayer RGGB ou similaire) a été **démosaïquée** : chaque pixel n'a mesuré
qu'une couleur, les deux autres ont été interpolées. Le bruit n'est alors
plus celui du capteur — il est corrélé entre pixels et entre canaux
(§10.1), et un débruiteur réglé avec le σ du capteur laisse des taches
colorées. Pour une image en niveaux de gris, une image panchromatique ou
multispectrale à un capteur par bande, ou une mosaïque brute non
démosaïquée, ce chapitre ne s'applique pas : §5 (bruit poisson-gaussien)
et §7 suffisent.

Ce chapitre donne la recette côté débruitage ; la théorie détaillée, la
mesure pour chaque démosaïqueur et les études sur orthophoto sont dans la
référence technique d'**eoqual-demosaicing**, chapitre 11.

### 6.2 Modèle de bruit du capteur

Pour une luminance linéaire normalisée ``L ∈ [0, 1]`` (niveau d'obscurité
soustrait) :

```
σ²(L) = a + b · L         a = (σ_r / N_sat)²  (plancher : lecture, obscurité)
                           b = 1 / N_sat        (pente : bruit photonique)
```

avec ``N_sat`` les électrons à pleine échelle et ``σ_r`` le bruit de
lecture en électrons. C'est le modèle du §5.1, écrit en luminance
normalisée, avec les lettres d'eoqual-measure. Correspondance avec les
fonctions du §5 :

```python
from eoqual.denoising.vst import generalized_anscombe, inverse_generalized_anscombe
gain, read_noise = 1.0 / b, a ** 0.5          # σ² = a + b·L
g = generalized_anscombe(L, gain, read_noise)  # bruit d'écart-type ≈ 1
```

### 6.3 Ce que le démosaïçage fait du bruit

Un démosaïqueur calcule chaque canal comme une moyenne pondérée de
photosites voisins (poids ``w``, ``Σ w = 1``). Le bruit de sortie en est
la même combinaison : sa variance par canal vaut ``σ²(L)·Σ w²`` (≤ σ²),
et les poids partagés le **corrèlent** entre pixels et entre canaux. En
luminance ``Y = (R+G+B)/√3``, des canaux corrélés positivement
**augmentent** la variance ; en chrominance (différences de canaux), elle
**diminue**, mais ce qui reste est de basse fréquence. On note ``κ_Y`` le
rapport des variances de Y après / avant démosaïçage (eoqual-demosaicing
§11.3, champ uniforme) :

| Démosaïqueur | κ_Y | √κ_Y + 0.2 |
|---|---|---|
| `bilinear` | 0.58 | 0.96 |
| `ari` | 0.62 | 0.99 |
| `ri`, `mlri` | 0.79 à 0.87 | 1.09 à 1.13 |
| `ha` (Hamilton-Adams) | 1.35 | 1.36 |
| `malvar` | 2.05 | 1.63 |
| `gbtf` | 2.41 | 1.75 |
| `dlmmse` | 2.58 | 1.80 |

### 6.4 Recette

1. **Démosaïquer** (eoqual-demosaicing), en luminance linéaire.
2. **Stabiliser** l'image démosaïquée avec ``(a, b)`` du capteur.
3. **Débruiter avec σ = C**, ``C = √κ_Y + 0.2``, avec un débruiteur
   **couleur** (`nl_bayes` en mode `"yuv"`).
4. **Inverser** la stabilisation (inverse exacte non biaisée, §5.3).

```python
from eoqual.denoising import vst_denoise

C = 1.35 ** 0.5 + 0.2                  # Hamilton-Adams : κ_Y = 1,35
rgb_debruitee = vst_denoise(
    rgb, "nl_bayes",                   # image (H, W, 3)
    gain=1.0 / b, read_noise_dn=a ** 0.5,   # σ² = a + b·L
    noise_factor=C, color_mode="yuv",
)
```

En ligne de commande : `examples/one_method_cli.py demosaiquee.tif
sortie.tif --method nl_bayes --vst-gain … --vst-read-noise … --kappa-y 1.35`.

Sans stabilisation : ``σ = C · σ₀`` avec ``σ₀ = √(a + b·L̄)``, un peu moins
bon (0.05 à 0.2 dB) car le vrai σ varie avec la luminance.

Mode opératoire complet sur données réelles (unités de ``(a, b)``,
valeurs de C à balayer pour Hamilton-Adams et GBTF, critères sans vérité
terrain) : eoqual-demosaicing, §11.11, exemple
`examples/real_data_protocol.py` et notebook `examples/demosaicing_noise.ipynb`.

**Pourquoi √κ_Y + 0.2.** NL-Bayes couleur règle son étape principale sur
le bruit de la luminance, d'écart-type ``√κ_Y`` dans le domaine stabilisé
(d'où ``√κ_Y``) ; le terme ``+0.2``, empirique, efface le bruit de
chrominance basse fréquence, que le débruiteur prend sinon pour du signal.
Validation (19 démosaïqueurs × 3 niveaux de bruit × 4 zones d'une
orthophoto, eoqual-demosaicing §11.4) — perte moyenne par rapport au
meilleur C de chaque cas :

| Règle | bruit fort | moyen | faible |
|---|---|---|---|
| C = 1 (σ du capteur) | 1.12 dB | 0.47 dB | 0.22 dB |
| C = 1.5 fixe (Jin et al. 2020) | 0.31 dB | 0.18 dB | 0.15 dB |
| C = 1.25 fixe | 0.63 dB | 0.20 dB | 0.13 dB |
| **C = √κ_Y + 0.2** | **0.08 dB** | **0.04 dB** | **0.05 dB** |

(Le facteur fixe 1.5 de Jin, Facciolo & Morel 2020 correspond à un
démosaïqueur de κ_Y ≈ 1.7 ; il est trop fort pour les méthodes à faible
κ_Y et trop faible pour GBTF ou DLMMSE.)

### 6.5 Régler C

Avec une image de référence, C se règle par optimisation (dimension
`sigma_scale` dans le domaine stabilisé, départ √κ_Y + 0,2), **en gardant
les (a, b) du capteur** : augmenter (a, b) à la place fausserait
l'inverse exacte au retour (§5.3). Détail : [optimisation](optimisation.md) §2.4.

### 6.6 Deux erreurs à éviter

**Estimer σ sur l'image démosaïquée.** `estimate_noise_sigma` (§7)
mesure la sous-bande d'ondelettes la plus fine, c'est-à-dire les hautes
fréquences du bruit — que le démosaïçage a atténuées. Après
Hamilton-Adams, il trouve **la moitié** du vrai σ (×0.50 à bruit fort,
×0.56 à bruit moyen) : NL-Bayes sous-débruite et laisse des taches
colorées (−2.6 dB, ΔE 5.5 au lieu de 3.3 à bruit fort). Estimer σ — ou
``(a, b)`` — sur la **mosaïque brute** (sous-plans R, G1, G2, B, où le
bruit est encore blanc ; ×1.00 à ×1.15 du vrai σ) ou sur le capteur
(PTC d'eoqual-measure).

**Débruiter la mosaïque avant de démosaïquer.** Sur les quatre
sous-plans en demi-résolution, le bruit est blanc et le modèle exact —
mais la résolution est réduite et la position relative des couleurs
perdue : −0.3 à −0.9 dB par rapport à « démosaïquer puis débruiter avec
C » (eoqual-demosaicing §11.6, Jin et al. 2020).

### 6.7 Résultats indicatifs

Orthophoto (© Toulouse Métropole, Licence Ouverte v2.0), bruit de capteur
simulé, CPSNR (dB) en sRGB, bruit fort / moyen / faible
(eoqual-demosaicing §11.6-11.7) :

| Chaîne | fort | moyen | faible |
|---|---|---|---|
| HA → NL-Bayes, σ du capteur (C = 1), stabilisation avec (a, b) | 30.97 | 34.00 | 35.79 |
| HA → NL-Bayes, C = √κ_Y + 0.2 = 1.36 | 31.54 | 34.13 | 35.72 |
| `gbtf` → NL-Bayes, C = 1.75 | 32.16 | 35.11 | 37.42 |
| `ari` → NL-Bayes, C = 1.0 | 31.46 | 35.22 | 37.92 |
| `ari` → DRUNet couleur (deepinv), 1.5σ₀ (autre tirage, §11.6) | 32.76 | 35.51 | 37.83 |

Le choix du démosaïqueur pèse autant que celui de σ : Hamilton-Adams a
ses propres fausses couleurs, que le débruitage ne corrige pas
(eoqual-demosaicing §11.8). DRUNet demande un facteur plus fort que
NL-Bayes (1.5 à 1.75 avec HA).

**Référence** : Q. Jin, G. Facciolo, J.-M. Morel, « A Review of an Old
Dilemma: Demosaicking First, or Denoising First? », CVPR Workshops 2020
(arXiv 2004.11577).

## 7. Utilisation et conventions

Conventions communes à toutes les méthodes décrites aux §2 et §3.

- **Appel unifié** : toute méthode s'appelle de la même façon,
  `denoise(image, method, **kwargs)`, avec l'image, le nom de la méthode
  (`list_methods()`) et ses arguments propres. Chaque méthode est aussi
  exposée directement (`d.<méthode>_denoise(image, sigma=None, ...)`),
  avec exactement les mêmes arguments. Le même schéma vaut pour l'image
  brute (`vst_denoise(image, method, gain, read_noise_dn, offset_dn,
  **kwargs)`) et pour l'optimisation (`optimize_parameters(noisy,
  reference, method, ...)`, dont `best_kwargs` se repasse à `denoise`).
- **Contrat de chaque méthode** (vérifié par
  `tests/test_registry.py::test_unified_signature`) : signature
  `f(image: np.ndarray, sigma: float | None = None, ...) -> np.ndarray`,
  sortie `float64` de même forme que l'entrée, aucun paramètre nommé
  `method` (réservé au dispatcher). Les arguments propres à une méthode
  sont décrits dans sa docstring et aux §2-3.
- `image` est une image 2D niveaux de gris, de type quelconque (convertie
  en `float64`). Seule `nl_bayes` traite aussi la couleur (H, W, 3),
  selon la méthode de l'article (§2.6) ; pour les autres, appliquer bande
  par bande. Une image contenant des NaN ou des infinis est refusée
  (`ValueError`).
- **Image brute de capteur** (bruit photonique + bruit de lecture) :
  passer par `vst_denoise` (§5), qui stabilise la variance avant
  d'appeler la méthode choisie.
- **`sigma` est toujours exprimé dans les unités de l'image** (comptes
  numériques, réflectance...), pas en fraction de dynamique. Chaque
  couche d'appel le convertit vers l'échelle attendue par sa
  bibliothèque. Toutes les méthodes du socle sont donc **invariantes
  d'échelle** : le même résultat relatif est obtenu sur une image 8 bits,
  12 bits ou en réflectance (vérifié par
  `tests/test_socle.py::test_socle_method_scale_invariant`).
- `sigma=None` → estimation automatique par `estimate_noise_sigma` (voir
  aussi §4.3)
  (estimateur MAD de Donoho & Johnstone sur la sous-bande d'ondelettes
  diagonale la plus fine). Cet estimateur est fiable pour un bruit blanc
  sur une image texturée modérément. Il **surestime** σ sur une image
  très texturée ou à fort contenu haute fréquence : fournir σ dès qu'il
  est connu (caractérisation capteur).
- Les méthodes qui exigent une dynamique fixe (toutes les méthodes
  deepinv : [0, 1] ; `nl_bayes`, `nl_ridge` et leurs versions de
  référence : [0, 255]) acceptent un paramètre `data_range` =
  `(min, max)`. Par défaut, c'est le min/max de l'image, ce qui
  convient pour une image isolée. Pour traiter une série d'images de
  manière cohérente, fixer `data_range` à la dynamique du capteur.
- Aucune méthode ne charge de dépendance lourde ou tierce à l'*import* du
  paquet : torch, deepinv, bm3d, NL-Ridge et l'exécutable NL-Bayes IPOL ne sont chargés ou
  lancés qu'au premier appel de la fonction correspondante, avec un
  message d'erreur explicite (quoi installer, quelle licence) s'ils
  manquent.

## 8. Méthode non incluse

WNLB (Wavelet Non-Local Bayes) : source sans licence, aucune publication
identifiée. Voir `THIRD_PARTY_LICENSES.md` §2.4, et §3 pour l'évaluation
d'une réimplémentation.

## 9. Tests, exemples et performances indicatives

### 9.1 Tests

Images de test : `astronaut` (NASA, domaine public), réduite à 128×128
(niveaux de gris) ou 96×96 (couleur), bruit blanc gaussien σ = 0,1
(fixtures partagées dans `tests/conftest.py`, graine modifiable par
`EOQUAL_TEST_SEED`). Les tolérances des tests statistiques sont tirées
de l'incertitude (5 incertitudes-types), jamais fixées au jugé.

| Fichier | Ce qui est vérifié |
|---|---|
| `test_registry.py` | contrat d'appel unifié de toutes les méthodes (signature, pas de paramètre `method`), dispatch, erreurs, `list_methods` |
| `test_socle.py` | méthodes du socle : gain de PSNR ≥ 3 dB, σ estimé automatiquement, invariance d'échelle (image 12 bits) |
| `test_utils.py` | validation de l'image, estimation de σ, normalisation, générateur de bruit, images de test |
| `test_nl_bayes.py` | niveau de qualité, apport de l'étape 2, paramètres et imbrications de l'article, accord des moteurs, indépendance au nombre de fils, couleur |
| `test_nl_ridge.py` | niveau de qualité, apport de l'étape 2, contrainte affine (poids de somme 1), paramètres des auteurs, accord avec leur code |
| `test_deepinv.py` | méthodes deepinv ; modèles à poids téléchargés seulement si `EOQUAL_TEST_DOWNLOAD=1` |
| `test_external.py` | `bm3d`, `nl_ridge_inria`, `nl_bayes_ipol` (sautés si absents) ; protocole d'appel de l'exécutable IPOL testé avec un faux programme |
| `test_vst.py` | transformée d'Anscombe : cas particuliers, raccord, absence de biais (Monte-Carlo), variance exacte, simulateur, gain à faible flux |
| `test_optimization.py` | optimisation des paramètres (détail : [optimisation](optimisation.md) §7) |
| `test_examples.py` | chaque script de `examples/` lancé en sous-processus avec des tailles réduites |

Les méthodes dont l'extra ou le composant externe manque sont sautées
(`skip`), jamais comptées en échec.

### 9.2 Exemples

| Script | Usage |
|---|---|
| `one_method_cli.py` | débruiter un fichier (niveaux de gris, couleur pour `nl_bayes`, image brute avec `--vst-gain`) ; `--list` |
| `all_methods_overview.py` | comparer les méthodes disponibles (PSNR, SSIM, temps) et enregistrer une planche |
| `low_light_vst.py` | effet de la stabilisation de variance à faible flux (tableaux du §5.5) |
| `optimize_parameters.py` | optimiser les paramètres d'une méthode ([optimisation](optimisation.md)) : journal, meilleurs paramètres, image, figures |
| `benchmark.py` | benchmark principal ([benchmark](benchmark.md) §2-3) |
| `nl_bayes_benchmark.py` | moteurs et couleur de NL-Bayes ([NL-Bayes](nl-bayes/etudes.md) §11.1-11.2) |
| `tau0_study.py` | sensibilité de NL-Bayes au seuil τ₀ ([NL-Bayes](nl-bayes/etudes.md) §11.3) |
| `nl_bayes_sensitivity.py` | sensibilité de NL-Bayes à tous ses paramètres, orthophotos, RGB démosaïqué ([NL-Bayes](nl-bayes/etudes.md) §11.4-11.5) |
| `nl_bayes_figures.py` | figures explicatives de [NL-Bayes](nl-bayes/index.md) |

### 9.3 Performances indicatives

**Le benchmark de référence est [benchmark](benchmark.md)** (astronaut 512×512,
réglage et évaluation séparés, PSNR et SSIM, réglages par défaut et
optimisés). Le tableau ci-dessous, sur une image plus petite, sert de
contrôle rapide de non-régression.

Résultats de `all_methods_overview.py` sur `astronaut` 256×256 en niveaux
de gris, dynamique [0, 1], bruit blanc gaussien (graine 0), σ connu,
paramètres par défaut, CPU — **PSNR en dB** :

| Méthode | σ = 0,05 | σ = 0,1 | σ = 0,2 | Temps (σ = 0,1) |
|---|---|---|---|---|
| *(image bruitée)* | 26,03 | 20,00 | 13,98 | — |
| `bilateral` | 31,08 | 26,77 | 22,09 | 0,07 s |
| `nl_means` | 31,22 | 27,64 | 23,81 | 0,19 s |
| `tv_chambolle` | 31,22 | 27,45 | 23,95 | 0,02 s |
| `tv_bregman` | 30,67 | 26,84 | 22,90 | 0,01 s |
| `wavelet` | 29,80 | 25,81 | 22,61 | 0,03 s |
| `drunet` | **34,66** | **30,70** | **26,85** | 3,0 s |
| `swinir` | 33,00 | 30,69 | 26,18 | 7,3 s |
| `restormer` | 34,64 | 30,68 | 26,79 | 2,0 s |
| `gsdrunet` | 34,44 | 30,52 | 26,67 | 1,4 s |
| `epll` | 32,65 | 28,54 | 24,25 | 3,5 s |
| `tgv` | 31,21 | 27,52 | 24,11 | 3,3 s |
| `wavelet_dict` | 29,91 | 25,84 | 21,98 | 0,13 s |
| `bm3d` | 33,29 | 29,34 | 25,58 | 1,2 s |
| `nl_ridge` | 33,41 | 29,35 | 25,53 | 1,4 s |
| `nl_ridge_inria` | 33,44 | 29,34 | 25,51 | 0,9 s |
| `nl_bayes` | 33,36 | 29,25 | 25,40 | 0,6 s |
| `nl_bayes_ipol` | 33,30 | 29,19 | 25,52 | 0,17 s |

Lecture :

- les méthodes non locales à deux étapes (BM3D, NL-Ridge) gagnent ~2 dB
  sur le meilleur du socle ;
- les réseaux conditionnés ou entraînés pour ce bruit (DRUNet,
  Restormer, GS-DRUNet, SwinIR à σ = 0,1) gagnent encore ~1,3 dB ;
- SwinIR perd 1,7 dB à σ = 0,05, entre deux jeux de poids (§3.2) ;
- EPLL se place entre le socle et BM3D ; TGV et `wavelet_dict` restent
  au niveau de leurs équivalents du socle.

Ces chiffres portent sur une seule image naturelle et ne préjugent pas
du classement sur des images EO. Ils servent de référence de
non-régression, pas de benchmark.

## 10. Pistes

- Réimplémentation MIT de BM3D depuis l'article, sous réserve du brevet
  (évaluation dans `THIRD_PARTY_LICENSES.md` §3).
- **Bruit coloré** (spatialement corrélé) : voir §10.1 ; le cas du
  démosaïçage est traité au §6.
- **DRUNet couleur** : `drunet` est en niveaux de gris ; les poids couleur
  de deepinv (BSD-3-Clause) donnent, après démosaïçage, 0.2 à 0.9 dB de
  plus que NL-Bayes couleur (§6.7).
- **Optimisation** : pistes propres dans [Optimisation des paramètres](optimisation.md), §8.
- **NL-Bayes et NL-Ridge sur GPU** : portage PyTorch du moteur
  vectorisé (mêmes opérations par lots : extraction, `matmul`, `solve`).
- **Couleur pour NL-Ridge** (et les autres méthodes du socle), sur le
  modèle du mode `"yuv"` de NL-Bayes.
- **Recalibrage des valeurs par défaut** de `epll` (`sigma_scale` ≈ 1,2),
  `tv_bregman` (`weight` ≈ 1,5/σ, TV anisotrope) et `bilateral`
  (`sigma_space` ≈ 1,7 px) : l'optimisation gagne 1 à 1,3 dB sur une
  image d'évaluation distincte ([benchmark](benchmark.md), §3). À confirmer sur
  plusieurs images et niveaux de bruit avant de changer les défauts.
- Estimation des paramètres poisson-gaussiens (K, σ_DN) sur une seule
  image, quand aucune PTC n'est disponible.
- Déplacement de `vst.py` vers eoqual-core si un second module en a
  besoin (règle d'eoqual-core : une brique n'y entre qu'avec deux
  utilisateurs). Le module ne dépend que de numpy et scipy pour le
  permettre sans modification.
- SCUNet (deepinv) : poids publiés en couleur seulement, pour du bruit
  réel aveugle ; à évaluer en niveaux de gris (image répliquée sur 3
  canaux) avant intégration.

### 10.1 Bruit coloré

**Définition.** Un bruit **blanc** est indépendant d'un pixel à l'autre :
son spectre de puissance (NPS, *Noise Power Spectrum*) est plat, toutes
les fréquences spatiales en contiennent autant. Un seul nombre, σ, le
décrit entièrement. C'est l'hypothèse de toutes les méthodes du
catalogue. Un bruit **coloré** est corrélé entre pixels voisins : son
NPS n'est plus plat, et σ ne suffit plus à le décrire. Il faut sa
**densité spectrale** complète, S(fx, fy), ou de façon équivalente son
noyau de corrélation spatiale.

**Origines en imagerie EO.** Le bruit d'acquisition est en général blanc
(photonique et de lecture, indépendants d'un pixel à l'autre). Il devient
coloré dès qu'un traitement mélange des pixels :

- **ré-échantillonnage** (orthorectification, zoom, recalage) :
  l'interpolation est un filtre passe-bas, le bruit se concentre aux
  basses fréquences et prend un aspect « grumeleux » ;
- **déconvolution et restauration de MTF** : elles amplifient les hautes
  fréquences, le bruit y devient dominant ;
- **dématriçage** (Bayer) : corrélations spatiales et entre canaux —
  cas traité en détail au §6 ;
- **compression** (JPEG, JPEG 2000, CCSDS) : motifs de blocs ou d'ondelettes
  liés à la quantification ;
- **défauts d'égalisation et rayures** des capteurs pushbroom : bruit
  concentré sur un axe du spectre (lien avec eoqual-destriping).

**Pourquoi les méthodes actuelles échouent.** Un débruiteur réglé pour un
bruit blanc applique la même force à toutes les fréquences. Aux
fréquences où le bruit est fort, il en laisse ; aux fréquences où il est
faible, il retire du signal. Le résidu typique est fait de taches ou de
textures structurées, que l'œil interprète facilement comme des détails
de la scène. Le piège est réel : un σ mesuré globalement peut paraître
correct alors que le spectre du bruit est très inégal.

**Ce qu'il faudrait faire.**

1. **Mesurer la densité spectrale du bruit.** Le NPS d'eoqual-measure
   (`noise/spectrum.py`) en est l'outil naturel, sur une zone uniforme
   ou sur la différence de deux acquisitions. Pour un traitement connu
   (interpolation, filtre de déconvolution), la densité peut aussi se
   calculer : S = σ²·|H|², où H est la réponse en fréquence du
   traitement appliqué à un bruit blanc d'écart-type σ.
2. **La donner au débruiteur à la place de σ.** `bm3d` 4.x accepte
   directement une densité spectrale 2D (`sigma_psd`), et c'est l'objet
   même de Mäkinen, Azzari & Foi (2020). Il suffirait d'exposer ce
   paramètre dans `bm3d_denoise`. ⚠️ La licence non commerciale de
   `bm3d` s'appliquerait aussi à cette évolution.
3. **Pistes sans BM3D.** Blanchir le bruit (filtrer l'image par
   1/√S), débruiter avec n'importe quelle méthode du catalogue, puis
   recolorer. C'est simple, mais instable là où S est proche de zéro (il
   faut régulariser). Autre option : débruiter **avant** le traitement
   qui colore le bruit, chaque fois que la chaîne le permet (par exemple
   débruiter avant l'orthorectification). **Exception mesurée : le
   démosaïçage** — débruiter la mosaïque perd la résolution ; mieux vaut
   débruiter après, avec un σ majoré (§6.6).

**Validation envisagée.** Simuler un bruit blanc, le colorer par un
filtre connu (interpolation bicubique, filtre de Wiener inverse), puis
comparer pour une même méthode : réglage par σ global, blanchiment +
débruitage + recoloration, et `sigma_psd` pour BM3D.

## 11. Choix de conception et variantes écartées

- **Valeurs par défaut du socle**, calibrées par balayage sur astronaut :
  bilatéral `sigma_color` = 2,5σ, TV Chambolle 0,75σ, TV Bregman 1/σ,
  ondelettes `db2`. Débruiteurs deepinv : TGV `weight` = 7,5σ (deepinv
  multiplie le seuil par 0,1 et 0,15), dictionnaire d'ondelettes 1,25σ.
  Le dictionnaire d'ondelettes est un extra séparé, à cause de `ptwt`
  (EUPL-1.2).
- **Comparaison à l'exécutable IPOL de NL-Bayes** : avec
  `data_range=(0, 1)`, tronquer les valeurs bruitées hors de [0, 1] avant
  la quantification 8 bits pénalise l'IPOL de 0,2 à 1,8 dB (comparaison
  256×256). Sans troncature, IPOL et version native sont à égalité
  (±0,1 dB) : c'est ainsi que se fait la comparaison.
- **Règle τ₀ = c·σ²** (seuil du regroupement de NL-Bayes), écartée au
  profit de τ₀ = 4. Ajustée sur 60 cas (gris et stabilisation), elle
  donne c = 0,005 et réduit la perte moyenne face au meilleur τ₀ de 0,09
  à 0,04 dB ; mais en validation indépendante (62 cas : autres images,
  autre graine, couleur `yuv`), elle fait moins bien que τ₀ = 4 en gris
  (−0,024 dB en moyenne, −0,53 au pire) et en couleur (−0,036, −0,59 au
  pire), à peine mieux avec stabilisation (+0,014). La borne haute de τ₀
  dans l'espace d'optimisation de NL-Bayes est 32 (optimum jusqu'à ~30 en
  couleur). Script : `examples/tau0_study.py`.
- **Étape 2 couleur de NL-Bayes canal par canal** : 0,6 à 0,8 dB de moins
  sur le benchmark couleur que les canaux filtrés ensemble, comme le
  tableau 1 de l'article (deux groupes 3D en tout) ; l'écart se voit en
  comparant à l'exécutable IPOL en mode couleur
  ([NL-Bayes](nl-bayes/parametres.md) §8.1, §11.2).
- **`sigma` refusé par `vst_denoise`** (erreur plutôt qu'avertissement) :
  dans le domaine stabilisé, le bruit vaut 1 par construction ; le
  réglage passe par `noise_factor` (σ dans le domaine stabilisé, défaut 1,
  avertissement hors de [0,5 ; 2,5]), qui est le facteur C d'une image
  démosaïquée (§6.5).
