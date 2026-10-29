# Référence technique — eoqual-denoising

Détail de chaque méthode : principe, implémentation, limites connues,
évolutions envisagées. Pour le catalogue résumé et l'installation, voir
`README.md` ; pour l'audit des licences, `THIRD_PARTY_LICENSES.md` ; pour
les citations complètes, `BIBLIOGRAPHY.md`.

## 1. Vue d'ensemble

### 1.1 Modèle de bruit

Toutes les méthodes du catalogue supposent un **bruit blanc gaussien
additif et stationnaire** d'écart-type σ : y = x + n, n ~ N(0, σ²),
indépendant d'un pixel à l'autre et de même variance partout. Deux
écarts fréquents à ce modèle sont traités à part :

- **bruit poisson-gaussien** des images brutes de capteur (variance qui
  croît avec le signal) : stabilisation de variance, §4 ;
- **bruit coloré** (corrélé entre pixels voisins) : non traité, voir
  §9.1.

### 1.2 Les méthodes

| Méthode | Famille | Principe en une ligne | Section |
|---|---|---|---|
| `bilateral` | filtre local | moyenne locale pondérée par la distance spatiale et radiométrique | §2.1 |
| `nl_means` | non local | moyenne des pixels dont les voisinages se ressemblent | §2.2 |
| `tv_chambolle` | variationnel | variation totale (ROF), solveur dual de Chambolle | §2.3 |
| `tv_bregman` | variationnel | variation totale (ROF), solveur split Bregman | §2.4 |
| `wavelet` | transformée | seuillage des coefficients d'ondelettes | §2.5 |
| `nl_bayes` | non local, 2 étapes | estimateur bayésien gaussien sur groupes de patchs ; couleur | §2.6 |
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
limites. L'utilisation commune à toutes les méthodes (appel, unités,
conventions) est décrite au §6, et les performances comparées dans
`BENCHMARK.md`.

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

**Fichiers** : `backends/nl_bayes/native.py` (interface, paramètres,
moteur « boucle ») et `backends/nl_bayes/vectorized.py` (moteur
vectorisé, couleur) — implémentation originale numpy, sans dépendance.
**Références** : Buades, Lebrun & Morel (2013, SIAM) ; article IPOL de
Lebrun, Buades & Morel (2013), **seule source utilisée** (clean-room :
aucun code consulté, voir `THIRD_PARTY_LICENSES.md` §2.1).

#### Algorithme

Deux étapes de même structure :

1. **Étape 1** — pour chaque patch de référence k₁×k₁, on regroupe ses
   N₁ plus proches voisins (distance quadratique moyenne par pixel) dans
   une fenêtre n₁×n₁. Si l'écart-type du groupe est ≤ γσ (zone homogène,
   formule 10), tous les patchs du groupe prennent la valeur moyenne du
   groupe. Sinon, chaque patch est estimé par l'estimateur bayésien
   gaussien Q = P̄ + (C − β₁σ²I)C⁻¹(Q̃ − P̄) (formule 12), avec P̄ et C la
   moyenne et la covariance du groupe.
2. **Étape 2** — le regroupement se fait sur l'estimée u_basic de
   l'étape 1. Tous les patchs à distance ≤ τ = max(τ₀, d_N₂) sont gardés
   (plus nombreux dans les zones homogènes). La covariance vient des
   patchs de u_basic (oracle), la moyenne des patchs bruités, et
   Q = P̄ + C_b(C_b + β₂σ²I)⁻¹(Q̃ − P̄) (formule 15) : un filtre de Wiener
   collaboratif.

Les estimées sont agrégées sans pondération.

#### Les paramètres, leurs imbrications et leurs domaines de recherche

σ s'entend sur l'échelle [0, 255] (l'image y est ramenée par
`data_range`).

| Terme | Rôle | Article (gris) | Domaine de recherche | Effet si trop petit / trop grand |
|---|---|---|---|---|
| k₁, k₂ | côté des patchs (étapes 1 et 2) ; dimension du modèle gaussien d = k² | 5 (σ < 50), 7 au-delà | {3, 5, 7} | petit : peu de structure, sensible au bruit ; grand : groupes moins similaires, covariance d×d mal estimée |
| N₁, N₂ | nombre de patchs du groupe (étape 2 : minimum avant le seuil τ) | 30 / 60 / 90 pour k = 3 / 5 / 7 | suit k (règle de l'article) | petit : covariance singulière ou bruitée ; grand : patchs moins similaires |
| n₁, n₂ (= α·k) | côté de la fenêtre de recherche | α = 7 | `window_factor` α ∈ [3 ; 9] | petit : groupes pauvres ; grand : plus lent, gain faible (tableau 6) |
| γ | seuil du critère des zones homogènes : écart-type du groupe ≤ γσ ⇒ moyenne | 1,05 | [0,9 ; 1,3] | petit : bruit résiduel dans les aplats ; grand : perte de petits détails |
| β₁ | atténuation de l'étape 1, C − β₁σ²I | 1,0 | [0,8 ; 1,2] | < 1 : sous-débruitage ; > 1 : sur-lissage |
| β₂ | atténuation de l'étape 2, C_b + β₂σ²I | 1,2 (σ < 50), 1,0 sinon | [0,8 ; 1,5] | idem, sur le filtre de Wiener |
| τ₀ | seuil minimal de similarité de l'étape 2 (niveaux²) | 4 | [0 ; 16] | 0 : groupes limités à N₂ ; grand : patchs dissemblables agrégés (tableau 7) |
| `use_area1`, `use_area2` | critère homogène à chaque étape | oui / non | {oui, non} | tableau 9 de l'article |
| `sigma_scale` | multiplie σ (pas un terme de l'article) | 1 | [0,5 ; 2] | compense un σ estimé biaisé ; agit sur β·σ² et sur γσ |

**Imbrications** (appliquées par `resolve_parameters`) :

- **N dépend de k** : la covariance d'un groupe est une matrice k²×k²,
  inversible seulement si N > k². D'où la règle N = 30 / 60 / 90 pour
  k = 3 / 5 / 7 (l'article l'impose pour éviter les matrices
  singulières, §4.6). Changer k sans fixer N applique cette règle, et N
  est toujours porté à au moins k² + 1.
- **n dépend de k** : la fenêtre est n = α·k, de sorte que le nombre de
  candidats (n²) croît avec la taille des patchs. Changer k (ou α)
  recalcule n.
- **σ, β et γ agissent ensemble** : β·σ² est la variance de bruit
  retirée à la covariance, et γσ le seuil d'homogénéité. Multiplier σ
  par `sigma_scale` revient à multiplier β par `sigma_scale`² et γ par
  `sigma_scale`. Optimiser à la fois `sigma_scale`, β et γ introduit donc
  une redondance partielle (crêtes dans l'espace de recherche), ce qui
  justifie un budget plus large.
- **k et σ** : l'article fait croître k avec le bruit (paliers à 50 et
  70). Un réglage optimisé à un σ donné ne se transpose pas tel quel à un
  autre niveau de bruit.

Ces 10 dimensions sont initialisées dans
`optimization/spaces/nl_bayes.jsonc` (point de départ = article, σ < 50,
niveaux de gris). Budget conseillé : 60 à 100 évaluations, dont 20 à 30
aléatoires.

#### Deux moteurs

- **`engine="vectorized"`** (défaut) : références sur une **grille** de
  pas `step` (3 par défaut, comme BM3D) au lieu de l'astuce « paste »
  de l'article, qui est séquentielle par nature (un patch utilisé n'est
  plus référence). Pour un lot de références : extraction des fenêtres,
  distances, sélection des voisins, une matrice de filtrage d×d par
  groupe (A = I − β₁σ²C⁻¹ à l'étape 1, A = C_b(C_b + β₂σ²I)⁻¹ à l'étape 2)
  appliquée à tout le groupe par un seul produit, agrégation par
  `bincount` restreinte aux positions touchées. À l'étape 2, les groupes
  sont traités par paquets de taille voisine (médiane 60, mais jusqu'à la
  fenêtre entière dans les aplats). Les lots sont répartis sur
  `n_threads` fils (min(cœurs, 8) par défaut ; résultat indépendant du
  nombre de fils).
- **`engine="loop"`** : version fidèle à l'article (astuce « paste »),
  niveaux de gris ou couleur canal par canal.

Sur 512×256 (`BENCHMARK.md` §8), le moteur vectorisé au pas 3 est
~1,7 fois plus rapide que la boucle à qualité égale (32,42 contre
32,39 dB à σ = 0,1) ; le pas 4 est ~2,8 fois plus rapide pour −0,03 dB ;
le pas 2 gagne +0,06 dB pour un temps comparable à la boucle.

#### Couleur

`color_mode` (image (H, W, 3), σ identique sur les trois canaux) :

- **`"yuv"`** (défaut, choix de l'article, §3.3-3.4 et tableau 8) :
  étape 1 dans l'espace Y₀U₀V₀ rendu orthonormé (σ conservé), groupes
  formés sur Y₀ seul et appliqués à U₀ et V₀, filtrage canal par canal ;
  retour en RGB ; étape 2 en RGB, groupes formés sur la distance moyenne
  des trois canaux de u_basic, filtrage canal par canal ;
- **`"rgb"`** : les deux étapes en RGB, groupes communs aux trois canaux ;
- **`"channel"`** : chaque canal débruité indépendamment, comme une image
  en niveaux de gris.

Paramètres : tableau §4.8 de l'article (patchs de 3 pour σ < 20) pour
`"yuv"` et `"rgb"` ; profil niveaux de gris pour `"channel"`.

Résultat (`BENCHMARK.md` §8, astronaut RGB 512×256) : `"yuv"` est le
meilleur à tous les niveaux de bruit (33,07 dB à σ = 0,1, contre 32,77
pour `"rgb"` et 32,23 pour `"channel"`), et le plus rapide (2,5 fois
plus que `"channel"`). Le regroupement sur la luminance, moins bruitée
car moyenne des trois canaux, donne de meilleurs groupes ; les canaux de
chrominance, plus réguliers, profitent de groupes communs.

#### Choix non fixés par l'article

Fenêtre de recherche rabattue à l'intérieur de l'image près des bords ;
covariance de l'étape 2 centrée sur la moyenne des patchs de u_basic ;
critère homogène de l'étape 2 (optionnel) évalué sur le groupe bruité ;
covariance singulière → patchs laissés inchangés (remarque §3.3). Calcul
en flottants sur l'échelle [0, 255], sans quantification.

#### Résultats et limites

32,42 dB à σ = 0,1 sur le benchmark principal (moteur vectorisé), au
niveau de BM3D (32,38) et de NL-Ridge. Face à l'exécutable IPOL : à
égalité sur 256×256 (±0,1 dB), +0,1 à +0,4 dB sur 512×256. Limite : numpy
sur CPU ; un portage PyTorch (GPU) serait le prochain gain de vitesse
(§9).

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

À noter : à l'étape 1, NL-Ridge a besoin de **n > k** (YᵀY inversible),
l'inverse de NL-Bayes (N > k²) ; ses patchs sont donc grands et ses
groupes petits.

**Validation** : reproduit le code des auteurs à 0,03 dB près sur
astronaut 256×256 (29,35 contre 29,34 dB à σ = 0,1), en temps comparable
(test `test_nl_ridge_native_matches_authors_code`). La variante affine
est 0,02 à 0,06 dB en dessous.

**Non porté** : bruits de Poisson et poisson-gaussien de l'article 2025
(utiliser `vst_denoise`), contraintes coniques et convexes (sans forme
fermée), couleur.

## 3. Méthodes en extra ou composant externe

### 3.1 `drunet` — réseau DRUNet (extra `deep`)

**Fichier** : `backends/drunet/deepinv_impl.py` — `deepinv.models.DRUNet`.
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

### 3.2 `swinir` — transformeur SwinIR (extra `deep`)

**Fichier** : `backends/deepinv_zoo/models.py` — `deepinv.models.SwinIR`.
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

### 3.3 `restormer` — transformeur Restormer (extra `deep`)

**Fichier** : `backends/deepinv_zoo/models.py` — `deepinv.models.Restormer`.
**Référence** : Zamir et al. (2022).

Transformeur dont l'attention porte sur les **canaux** plutôt que sur
les positions, ce qui rend son coût linéaire en nombre de pixels et
permet de traiter de grandes images. Poids niveaux de gris **aveugles** :
un seul réseau, entraîné sur σ ∈ [0, 50]/255 tiré au hasard, qui
n'utilise pas σ (le paramètre est accepté mais ignoré).

**Limites** : hors de son domaine d'entraînement (bruit normalisé
supérieur à 50/255), il échoue nettement, faute de pouvoir être informé
du niveau de bruit : à faible flux, même après stabilisation de variance
(§4), il perd 5 à 8,5 dB face à DRUNet. Ne l'utiliser que dans son domaine
(σ ≤ 0,2 en dynamique [0, 1]).

### 3.4 `gsdrunet` — GS-DRUNet (extra `deep`)

**Fichier** : `backends/deepinv_zoo/models.py` — `deepinv.models.GSDRUNet`.
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
d'appel (bruit blanc uniquement) — voir §9.1.

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
sont pas exposés : passer par `vst_denoise` (§4), commun à toutes les
méthodes.

### 3.10 `nl_bayes_ipol` — NL-Bayes, exécutable IPOL (externe, **GPL-3.0+**)

**Fichier** : `backends/nl_bayes/ipol_external.py`.
**Références** : Buades, Lebrun & Morel (2013) ; implémentation IPOL,
Lebrun, Buades & Morel (2013).

Même méthode que `nl_bayes` (§2.6), par l'implémentation C++ de
référence. Utile pour comparer ; `nl_bayes` est préférable en usage
courant (MIT, sans composant externe, sans quantification, meilleur
PSNR en niveaux de gris : 32,45 contre 32,02 dB).

Pour chaque groupe de patchs similaires, estimateur bayésien (MAP) sous
hypothèse gaussienne : moyenne et covariance du groupe estimées sur
l'image bruitée (étape 1) puis sur l'estimée de l'étape 1 (étape 2), et
filtrage de Wiener du groupe. Le « critère des zones homogènes »
(`use_area1`, `use_area2`) remplace le groupe par sa moyenne quand sa
variance est compatible avec le seul bruit.

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

## 4. Bruit poisson-gaussien : stabilisation de variance

**Fichier** : `vst.py` — implémentation originale (numpy, scipy), écrite
depuis les publications, sans code tiers.
**Références** : Anscombe (1948) ; GAT, Murtagh, Starck & Bijaoui
(1995) ; inverses optimales, Mäkitalo & Foi (2011, 2013).

### 4.1 Modèle

Une image brute de capteur suit, pixel par pixel :

    z = p / K + offset + n,   p ~ Poisson(λ),   n ~ N(0, σ_DN²)

- `gain` = K, gain de conversion en e⁻/DN ;
- `read_noise_dn` = σ_DN, bruit de lecture en DN ;
- `offset_dn` = niveau d'obscurité en DN.

La variance vaut σ²(z) = (z̄ − offset)/K + σ_DN² : elle **croît avec le
signal**. C'est exactement la droite de la PTC, σ²(μ) = a·μ + b avec
a = 1/K et b = σ_DN². Les paramètres se lisent donc directement sur les
mesures d'eoqual-measure : `measure_ptc(...).gain`,
`.read_noise_dn`, et le niveau moyen de `measure_dark_frame` pour
l'offset.

Un débruiteur gaussien appliqué tel quel reçoit un σ unique (estimé ou
fourni) alors que le vrai σ varie dans l'image. Il sur-lisse les zones
sombres et sous-lisse les zones claires, ou l'inverse.

### 4.2 Transformée directe (GAT)

En électrons, y = K·(z − offset), et avec σ = K·σ_DN :

    f(y) = 2·sqrt(y + 3/8 + σ²)   (0 si l'argument est négatif)

Après f, la variance du bruit vaut ≈ 1 **quel que soit λ**, dès quelques
électrons (vérifié : 0,98 à 1,04 pour λ ≥ 3 e⁻). Le cas pur Poisson
(`gain=1`, `read_noise_dn=0`) redonne l'Anscombe classique,
2·sqrt(z + 3/8).

### 4.3 Inverses

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

### 4.4 Usage

```python
import eoqual.denoising as d
out = d.vst_denoise(raw, "drunet", gain=2.0, read_noise_dn=1.5, offset_dn=100.0)
```

`vst_denoise` applique la GAT, appelle la méthode choisie avec
`sigma = 1`, puis l'inverse exacte. Les briques sont aussi exposées
séparément : `generalized_anscombe`, `inverse_generalized_anscombe`,
`expected_gat`, `add_poisson_gaussian_noise` (simulateur du même
modèle).

### 4.5 Résultats (`examples/low_light_vst.py`)

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

## 5. Optimisation des paramètres

**Fichiers** : `optimization/` (`optimizer.py`, `search_spaces.py`,
`plotting.py`, `jsonc.py`) ; script `examples/optimize_parameters.py`.
**Extra** : `optim` (scikit-optimize, BSD-3-Clause).
**Référence** : Shahriari et al. (2016) ; acquisition `gp_hedge`,
Hoffman, Brochu & de Freitas (2011).

### 5.1 Problème et méthode

Les réglages par défaut (§2-3) sont calibrés sur une image naturelle. Sur
une image d'un capteur donné, d'autres valeurs peuvent faire mieux.
`optimize_parameters` cherche, pour une méthode, les paramètres qui
maximisent un score (PSNR ou SSIM) entre l'image débruitée et une
**image de référence** propre (simulation, moyenne de nombreuses
acquisitions, ou scène de calibration).

Chaque évaluation coûte un débruitage complet et la fonction est une
boîte noire (sans gradient, dimensions entières, réelles et
catégorielles mêlées). L'optimisation bayésienne limite le nombre
d'évaluations :

1. un **processus gaussien** est ajusté sur les points déjà évalués ; il
   prédit le score en tout point, avec une incertitude ;
2. une **fonction d'acquisition** choisit le point suivant, entre
   exploitation (près des meilleurs points) et exploration (zones
   incertaines). `gp_hedge` fait concourir Expected Improvement,
   Probability of Improvement et Lower Confidence Bound, et privilégie
   celle qui réussit le mieux au fil des itérations ;
3. le point est évalué, le modèle mis à jour, et ainsi de suite jusqu'à
   `n_calls` évaluations.

Les `n_initial_points` premiers points sont tirés au hasard : le modèle a
besoin d'observations réparties avant de guider la recherche. Le point de
départ (`initial_guess` = réglages par défaut) est toujours évalué en
premier et compte parmi ces points. **L'optimisation ne peut donc pas
faire moins bien que les défauts sur l'image de réglage.**

### 5.2 Espaces de recherche

Chaque méthode réglable a un espace par défaut (`SEARCH_SPACES`) :

| Méthode | Dimensions optimisées |
|---|---|
| `bilateral` | `sigma_color` [0,5 ; 6]·σ, `sigma_space` [0,5 ; 5] px |
| `nl_means` | `h` [0,2 ; 1,5]·σ, `patch_size` 3-11, `patch_distance` 3-21 |
| `tv_chambolle` | `weight` [0,1 ; 3]·σ |
| `tv_bregman` | `weight` [0,1 ; 5]/σ, `isotropic` |
| `wavelet` | `sigma_scale` [0,5 ; 2], ondelette (6), règle, mode |
| `drunet`, `gsdrunet`, `epll` | `sigma_scale` [0,5 ; 2] |
| `swinir` | jeu de poids 15 / 25 / 50 (recherche exhaustive) |
| `tgv` | `weight` [1 ; 20]·σ |
| `wavelet_dict` | `threshold` [0,3 ; 4]·σ, `level` 1-5 |
| `bm3d` | `sigma_scale` [0,5 ; 2], profil (`np`, `refilter`, `vn`, `high`) |
| `nl_ridge`, `nl_ridge_inria` | `sigma_scale`, `p1`, `p2`, `k1`, `k2`, contraintes (linéaire / affine) |
| `nl_bayes` | `sigma_scale`, `k1`, `k2` (3/5/7), `window_factor` [3 ; 9], `gamma` [0,9 ; 1,3], `beta1` [0,8 ; 1,2], `beta2` [0,8 ; 1,5], `tau0` [0 ; 16], `use_area1`, `use_area2` (détail §2.6) |
| `nl_bayes_ipol` | `sigma_scale`, `use_area1`, `use_area2` |
| `restormer` | aucune (réseau aveugle) |

Deux conventions rendent ces espaces **indépendants des unités de
l'image** :

- `"scale": "sigma"` (ou `"inverse_sigma"`) : la borne est un multiple
  de σ (ou de 1/σ). La méthode reçoit `valeur × σ` (ou `valeur / σ`) ;
- `sigma_scale` multiplie le σ transmis à la méthode. C'est l'unique
  réglage des méthodes conditionnées par σ, et il compense aussi une
  estimation de σ biaisée.

**Les espaces par défaut sont des fichiers** :
`optimization/spaces/<méthode>.jsonc`, un par méthode réglable (16 sur
17 ; `restormer` n'a aucun paramètre). Chaque dimension y est commentée
(rôle, valeur de l'article ou calibrée, effet), et `initial_guess` y vaut
les réglages par défaut de la méthode (sauf `swinir`, dont le jeu de
poids est choisi automatiquement selon σ). Ces fichiers sont la source
unique, chargée par `search_spaces.py`. Un autre fichier au même format
(JSON ou JSONC, commentaires et virgules finales tolérés) se passe par
`search_space=` ou `--search-space`. Une contrainte entre dimensions (ex. `patch_distance ≥
patch_size`) se passe par `space_constraint=` : les points invalides sont
rejetés au tirage, sans évaluation. Un espace entièrement catégoriel et
plus petit que le budget (ex. `swinir`, 3 points) est évalué
**exhaustivement**, sans modèle.

### 5.3 Exécution

- **Séquentielle** (`n_jobs=1`) : `skopt.gp_minimize`.
- **Parallèle** (`n_jobs > 1`) : file d'attente glissante sur un
  `ProcessPoolExecutor`. Jusqu'à `n_jobs` évaluations tournent en
  permanence ; chacune est remplacée dès qu'elle se termine (`ask` /
  `tell` un par un), sans attendre un lot. Les images et paramètres fixes
  ne sont envoyés qu'une fois par processus (initialiseur du pool).
  Limite assumée : les points en cours ne sont pas pris en compte par
  `ask`, ce qui peut produire des points voisins.
- **Méthodes PyTorch en parallèle** : chaque processus charge son modèle
  (mémoire × `n_jobs`), et le nombre de fils PyTorch par processus est
  limité à `cœurs // n_jobs` (`threads_per_job`). Sur GPU, rester à
  `n_jobs=1`.
- **Échecs** : une évaluation qui lève une exception, ou renvoie une
  image de mauvaise forme ou non finie, est pénalisée (objectif 1e10) et
  journalisée ; l'optimisation continue.
- **Journal** : `log_csv=` écrit une ligne par évaluation (paramètres,
  score, statut, durée), avec verrou de fichier sous POSIX.
- **Bruit poisson-gaussien** : `vst={"gain", "read_noise_dn",
  "offset_dn"}` optimise la chaîne complète GAT → méthode → inverse
  exacte (§4), avec σ = 1 dans le domaine transformé.

### 5.4 Usage

```python
from eoqual.denoising.optimization import optimize_parameters
res = optimize_parameters(noisy, reference, "bm3d", sigma=12.0, n_calls=25)
res.best_params      # meilleur point, en multiples de σ (ex. sigma_scale, profile)
res.best_kwargs      # arguments en unités de l'image, σ compris
res.gain             # gain sur les défauts, sur l'image de réglage
clean = d.bm3d_denoise(autre_image, **res.best_kwargs)
```

En ligne de commande : `python examples/optimize_parameters.py --method
nl_means --reference ref.tif --noisy bruitee.tif --n-calls 30`. Le script
écrit le journal CSV, les meilleurs paramètres (JSON), l'image débruitée
(TIFF float32) et les figures : convergence, projections de l'objectif
estimé, distribution des points évalués.

`res.best_kwargs` s'utilise tel quel avec `denoise(image, méthode,
**res.best_kwargs)` ou avec la fonction de la méthode
(`d.<méthode>_denoise`).

### 5.5 Budget et précautions

- **Budget** : 15 à 30 évaluations pour 1 à 3 dimensions, 50 à 100
  au-delà de 5 (`nl_ridge`). `n_initial_points` ≈ 20 à 40 % de `n_calls`.
- **Sur-apprentissage** : le gain mesuré sur l'image de réglage est
  optimiste. Il faut le **valider sur une autre image** du même capteur
  (le benchmark sépare réglage et évaluation, voir `BENCHMARK.md`).
- **Référence** : la référence doit être bien plus propre que l'image à
  débruiter. Une référence bruitée biaise le réglage vers le sur-lissage.
- **Score** : le PSNR favorise le lissage des zones homogènes, le SSIM
  la préservation des structures. Choisir le score selon l'usage aval.
- **Dépendance** : le dépôt d'origine de scikit-optimize est archivé
  depuis février 2024 ; les versions 0.10.x (dernière : juin 2024) sont
  publiées par une fourche peu active. Il n'a pas de successeur officiel.
  Le remplaçant le plus solide est **Optuna** (MIT, très actif, v5.0 en
  septembre 2026) : son `GPSampler` fait de l'optimisation bayésienne par
  processus gaussien (entiers, catégories, contraintes, parallélisme), et
  son `TPESampler` par défaut est plus robuste en grande dimension.
  Alternatives : BoTorch / Ax (MIT, Meta, plus lourds, PyTorch), SMAC3
  (BSD, forêts aléatoires, bon sur les espaces mixtes),
  `bayesian-optimization` (MIT, dimensions réelles seulement). Le code
  n'utilise que l'API stable de skopt (`gp_minimize`, `Optimizer`,
  `space_constraint`) ; la migration vers Optuna est notée au §8.

## 6. Utilisation et conventions

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
  passer par `vst_denoise` (§4), qui stabilise la variance avant
  d'appeler la méthode choisie.
- **`sigma` est toujours exprimé dans les unités de l'image** (comptes
  numériques, réflectance...), pas en fraction de dynamique. Chaque
  couche d'appel le convertit vers l'échelle attendue par sa
  bibliothèque. Toutes les méthodes du socle sont donc **invariantes
  d'échelle** : le même résultat relatif est obtenu sur une image 8 bits,
  12 bits ou en réflectance (vérifié par
  `tests/test_socle.py::test_socle_method_scale_invariant`).
- `sigma=None` → estimation automatique par `estimate_noise_sigma`
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

## 7. Méthode non incluse

WNLB (Wavelet Non-Local Bayes) : source sans licence, aucune publication
identifiée. Voir `THIRD_PARTY_LICENSES.md` §2.4, et §3 pour l'évaluation
d'une réimplémentation.

## 8. Tests, exemples et performances indicatives

### 8.1 Tests

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
| `test_optimization.py` | espaces de recherche (fichiers JSONC, cohérence avec les signatures, point de départ = défauts), unités, séquentiel / parallèle, recherche exhaustive, contrainte, SSIM, mode GAT, échecs pénalisés, journal, figures |
| `test_examples.py` | chaque script de `examples/` lancé en sous-processus avec des tailles réduites |

Les méthodes dont l'extra ou le composant externe manque sont sautées
(`skip`), jamais comptées en échec.

### 8.2 Exemples

| Script | Usage |
|---|---|
| `one_method_cli.py` | débruiter un fichier (niveaux de gris, couleur pour `nl_bayes`, image brute avec `--vst-gain`) ; `--list` |
| `all_methods_overview.py` | comparer les méthodes disponibles (PSNR, SSIM, temps) et enregistrer une planche |
| `low_light_vst.py` | effet de la stabilisation de variance à faible flux (tableaux du §4.5) |
| `optimize_parameters.py` | optimiser les paramètres d'une méthode (§5) : journal, meilleurs paramètres, image, figures |
| `benchmark.py` | benchmark principal (`BENCHMARK.md` §2-3) |
| `nl_bayes_benchmark.py` | moteurs et couleur de NL-Bayes (`BENCHMARK.md` §8) |

### 8.3 Performances indicatives

**Le benchmark de référence est `BENCHMARK.md`** (astronaut 512×512,
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

## 9. Évolutions envisagées

- Réimplémentations MIT depuis les publications : NL-Bayes et NL-Ridge
  **faits** (§2.6, §2.7) ; reste éventuellement BM3D (sous réserve du brevet) — évaluation dans
  `THIRD_PARTY_LICENSES.md` §3.
- **Bruit coloré** (spatialement corrélé) : voir §9.1.
- **Moteur d'optimisation** : remplacer scikit-optimize (archivé) par
  Optuna (`GPSampler` ou `TPESampler`), en gardant les mêmes fichiers
  d'espaces de recherche (§5.5).
- **NL-Bayes et NL-Ridge sur GPU** : portage PyTorch du moteur
  vectorisé (mêmes opérations par lots : extraction, `matmul`, `solve`),
  prochain gain de vitesse significatif.
- **Couleur pour NL-Ridge** (et les autres méthodes du socle), sur le
  modèle du mode `"yuv"` de NL-Bayes.
- **Recalibrage des valeurs par défaut** de `epll` (`sigma_scale` ≈ 1,2),
  `tv_bregman` (`weight` ≈ 1,5/σ, TV anisotrope) et `bilateral`
  (`sigma_space` ≈ 1,7 px) : l'optimisation gagne 1 à 1,3 dB sur une
  image d'évaluation distincte (`BENCHMARK.md` §3). À confirmer sur
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

### 9.1 Bruit coloré

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
- **dématriçage** (Bayer) : corrélations spatiales et entre canaux ;
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
   faut régulariser). Autre option, la plus propre : débruiter **avant**
   le traitement qui colore le bruit, chaque fois que la chaîne le
   permet (par exemple débruiter avant l'orthorectification).

**Validation envisagée.** Simuler un bruit blanc, le colorer par un
filtre connu (interpolation bicubique, filtre de Wiener inverse), puis
comparer pour une même méthode : réglage par σ global, blanchiment +
débruitage + recoloration, et `sigma_psd` pour BM3D.


## Annexe A — Historique des choix et corrections

Décisions de conception, calibrations et corrections, dans l'ordre où
elles ont été prises. Le corps du document décrit l'état courant ; cette
annexe en garde la trace. Les versions publiées sont résumées dans
`CHANGELOG.md`.

| Date | Objet | Décision ou correction |
|---|---|---|
| 2026-10-28 | Socle | Méthodes du socle écrites comme couches d'appel vers OpenCV et scikit-image, sans code tiers copié. Valeurs par défaut calibrées par balayage sur astronaut : bilatéral `sigma_color` = 2,5σ, TV Chambolle 0,75σ, TV Bregman 1/σ, ondelettes `db2`. |
| 2026-10-28 | Stabilisation de variance | Inverse exacte non biaisée de la GAT implémentée depuis Mäkitalo & Foi, validée par Monte-Carlo. Test de stabilisation de la variance corrigé : une tolérance fixe de 5 % échouait sur une graine sur 30 à λ = 5 e⁻ ; remplacée par la comparaison à la variance exacte calculée (1,042 à 5 e⁻), à 5 incertitudes-types. |
| 2026-10-28 | Débruiteurs deepinv | Ajout de SwinIR, Restormer, GS-DRUNet, EPLL, TGV et du dictionnaire d'ondelettes. TGV : `weight` = 7,5σ (deepinv multiplie le seuil par 0,1 et 0,15) ; dictionnaire : 1,25σ. Chargement des poids GS-DRUNet par la couche d'appel (clés de métriques d'entraînement refusées par deepinv 0.4.2). Dictionnaire d'ondelettes mis en extra séparé à cause de `ptwt` (EUPL-1.2). |
| 2026-10-28 | Optimisation | Moteur d'optimisation bayésienne repris d'un outil antérieur de l'auteur, sans ses débruiteurs. Conflit corrigé : le paramètre `method` de `wavelet_denoise` empêchait l'appel par `denoise` ; l'optimiseur appelle la fonction de la méthode directement. Point de départ de SwinIR retiré : fixé aux poids « 25 » alors que le choix automatique donne « 15 », il affichait un faux gain de 2,09 dB. |
| 2026-10-28 | NL-Bayes | Réimplémentation clean-room depuis l'article IPOL (moteur boucle). L'appel à l'exécutable IPOL, d'abord nommé `nl_bayes`, devient `nl_bayes_ipol`. Exécutable compilé sous macOS (libpng, fftw, libomp) et testé. |
| 2026-10-29 | Mesures NL-Bayes | Biais de mesure corrigé : avec `data_range=(0, 1)`, le wrapper IPOL tronquait les valeurs bruitées hors [0, 1] avant la quantification 8 bits, ce qui pénalisait l'IPOL de 0,2 à 1,8 dB sur une comparaison 256×256. Sans troncature, IPOL et version native sont à égalité (±0,1 dB). Le benchmark principal, mesuré avec la dynamique par défaut, n'était pas affecté. |
| 2026-10-29 | Appel unifié | Paramètre `method` de `wavelet_denoise` renommé `rule` ; test du contrat de signature ajouté pour toutes les méthodes. |
| 2026-10-29 | NL-Bayes vectorisé | Moteur par lots sur grille de références (pas 3). Étapes d'optimisation : matrice de filtrage d×d par groupe au lieu d'un système par candidat (×2,5), groupes de l'étape 2 traités par paquets de taille voisine, agrégation restreinte aux positions touchées (`np.unique` + `bincount`, au lieu d'un `bincount` sur toute l'image à chaque lot), répartition sur plusieurs fils. Résultat : ~1,7 fois plus rapide que la boucle à qualité égale. |
| 2026-10-29 | NL-Bayes couleur | Modes `yuv` (article), `rgb`, `channel` ; profils de paramètres gris et couleur ; `window_factor` et règle N ≥ k² + 1. |
| 2026-10-29 | NL-Ridge | Réimplémentation clean-room depuis les articles ICIP 2022 et SIAM 2025, à ≤ 0,04 dB du code des auteurs. L'appel au clone AGPL, d'abord nommé `nl_ridge`, devient `nl_ridge_inria`. |
| 2026-10-29 | Espaces de recherche | Passage de dictionnaires Python à un fichier JSONC commenté par méthode ; espace de NL-Bayes étendu à 10 dimensions. |
| 2026-10-29 | Figures d'optimisation | `plot_evaluations` de scikit-optimize échouait pour un espace à une dimension : tracé manuel dans ce cas. |
| 2026-10-29 | Tests et exemples | Tests découpés par thème avec fixtures partagées (`conftest.py`) ; tests des utilitaires, des fichiers d'espaces et des exemples ajoutés ; exemples dotés d'options de taille réduite. |
