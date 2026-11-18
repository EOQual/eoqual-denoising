# Optimisation des paramètres — eoqual-denoising

Comment régler les paramètres d'une méthode de débruitage sur un capteur
ou une chaîne de traitement, à partir d'une image de référence :
principe de l'optimisation bayésienne, réglage de σ et du modèle de
bruit, espaces de recherche, exécution, précautions. Les méthodes
elles-mêmes, le rôle de σ et le modèle de bruit sont décrits dans
`REFERENCE_TECHNIQUE.md` ; les gains mesurés dans `BENCHMARK.md` §3.

**Fichiers** : `optimization/` (`optimizer.py`, `search_spaces.py`,
`plotting.py`, `jsonc.py`) ; script `examples/optimize_parameters.py`.
**Extra** : `optim` (scikit-optimize, BSD-3-Clause).
**Référence** : Shahriari et al. (2016) ; acquisition `gp_hedge`,
Hoffman, Brochu & de Freitas (2011).

## 1. Principe

Les réglages par défaut (`REFERENCE_TECHNIQUE.md` §2-3) sont calibrés
sur une image naturelle. Sur une image d'un capteur donné, d'autres valeurs peuvent faire mieux.
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

## 2. Régler σ et le modèle de bruit

Ce chapitre suppose connus le rôle de σ dans les méthodes, le modèle de
bruit (a, b), la stabilisation de variance et le facteur C d'une image
démosaïquée (`REFERENCE_TECHNIQUE.md` §4 à §6). Il dit ce qu'il faut
régler, et comment.

### 2.1 Principe

**Fixer ce qui se mesure, optimiser ce qui s'ajuste.**

| Grandeur | Nature | Traitement |
|---|---|---|
| (a, b), offset | propriétés du **capteur** | **fixes** : PTC ou recette en vol |
| `sigma_scale` | ajustement du **niveau** de débruitage | **optimisé** ; c'est C après démosaïçage |
| autres paramètres | réglages de la **méthode** | optimisés si besoin, gain souvent faible (§3, `BENCHMARK.md` §3) ; pour NL-Bayes, lesquels régler et pourquoi : [NL_BAYES.md](NL_BAYES.md) §12 |

`sigma_scale` multiplie σ dans le domaine où travaille NL-Bayes : le σ de
l'image sans stabilisation, 1 avec stabilisation. Dans ce second cas,
c'est exactement le `noise_factor` de `vst_denoise` : le meilleur
`sigma_scale` se réutilise tel quel, `vst_denoise(..., noise_factor=C)`. Il compense un σ mal
estimé, une stabilisation imparfaite, ou le gain de bruit du démosaïçage.

### 2.2 Selon le cas

| Cas | `sigma_scale` : départ et bornes | Remarques |
|---|---|---|
| Bruit blanc, σ connu | 1, [0,5 ; 2,5] | l'optimum reste proche de 1 (0,92 sur le benchmark) |
| σ estimé | 1, [0,5 ; 2,5] | corrige le biais de l'estimateur |
| Image brute stabilisée | 1, [0,5 ; 2,5] | (a, b) fixes ; corrige une stabilisation imparfaite |
| Image démosaïquée stabilisée | **√κ_Y + 0,2**, [0,7 ; 1,4] × départ | (a, b) fixes ; `sigma_scale0=` ou `--kappa-y` ; image couleur, mode `yuv` |

### 2.3 Redondances et budget

**Redondance avec les paramètres de la méthode.** Quand une méthode
utilise σ multiplié par un de ses propres paramètres, multiplier σ par
`sigma_scale` revient à changer ce paramètre : les deux ne doivent pas
être optimisés ensemble sans précaution. C'est le cas de NL-Bayes (β·σ²,
γσ) : analyse et recommandation dans [NL_BAYES.md](NL_BAYES.md) §7.3 et
§12.

**Budget.** 15 à 25 évaluations pour `sigma_scale` seul, autant pour
les 3 dimensions de NL-Bayes ([NL_BAYES.md](NL_BAYES.md) §12). Toujours valider le réglage sur une autre
image que celle qui a servi à l'optimiser (§6).

### 2.4 Image démosaïquée : optimiser C, pas (a, b)

Quand une image de référence est disponible (scène de calibration, image
moyennée), C se règle par optimisation, et **seulement C** :

- **(a, b) restent ceux du capteur** (PTC ou recette en vol). Ils
  décrivent le capteur, pas le démosaïqueur.
- **C est la dimension `sigma_scale`** dans le domaine stabilisé, avec
  pour départ `√κ_Y + 0,2` (`sigma_scale0=` ou `--kappa-y`).

Pourquoi ne pas augmenter (a, b) à la place ? Stabiliser avec C²·a et
C²·b revient à diviser l'image transformée par environ C, puis à
débruiter avec σ = 1 : pour le débruitage, c'est presque la même chose
que σ = C. Mais **l'inverse exacte non biaisée**
(`REFERENCE_TECHNIQUE.md` §5.3) repose sur la statistique de Poisson
réelle du capteur, fixée par son vrai gain. Avec
un gain artificiellement divisé par C², elle devient biaisée, surtout
dans les zones sombres. Avec σ = C, la transformée et son inverse restent
exactes ; seul le niveau de débruitage change.

**Exemple : image démosaïquée par Hamilton-Adams (κ_Y = 1,35),
stabilisation avec (a, b) de la recette en vol.**

```python
from eoqual.denoising import vst_denoise
from eoqual.denoising.optimization import optimize_parameters

espace = {
    "search_space": [{"name": "sigma_scale", "type": "float", "low": 0.8, "high": 2.0}],
    "initial_guess": {"sigma_scale": 1.0},
}
res = optimize_parameters(
    rgb_bruite, rgb_reference, "nl_bayes",           # images (H, W, 3), en DN
    search_space=espace,
    vst={"gain": 1 / b, "read_noise_dn": a ** 0.5, "offset_dn": offset},  # σ² = a + b·L
    sigma_scale0=1.35 ** 0.5 + 0.2,                   # départ C = √κ_Y + 0,2
    n_calls=20, n_initial_points=6,
)
C = res.best_params["sigma_scale"]
propre = vst_denoise(autre_rgb, "nl_bayes", gain=1 / b, read_noise_dn=a ** 0.5,
                     offset_dn=offset, noise_factor=C)
```

En ligne de commande : `examples/optimize_parameters.py --method nl_bayes
--vst-gain … --vst-read-noise … --vst-offset … --kappa-y 1.35`.

### 2.5 Pistes non retenues

- **Optimiser (a, b) eux-mêmes** : à réserver à une mesure douteuse, en
  léger ajustement autour d'elle (l'outil ne le propose pas encore,
  §8) ; ce n'est jamais le bon levier pour le démosaïçage (§2.4).
- **τ₀ proportionnel à σ²** (NL-Bayes) : testée et écartée ([NL_BAYES.md](NL_BAYES.md) §11.3).

## 3. Espaces de recherche et contraintes

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
| `nl_bayes` | `sigma_scale` [0,5 ; 2,5], `beta2` [0,4 ; 1,5], `N2` [15 ; 90] ; β₁, γ, τ₀, α, `use_area1`, `use_area2` fixés ([NL_BAYES.md](NL_BAYES.md) §12) |
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
`search_space=` ou `--search-space`.

**Contraintes entre dimensions.** Quand deux dimensions sont liées par
l'algorithme, l'espace déclare une clé `"constraints"` : une liste
d'expressions, toutes requises, portant sur les valeurs de l'espace
(avant conversion par `scale`). Exemple de NL-Ridge :
`"constraints": ["k1 < p1 * p1"]`. Les points invalides sont rejetés au
tirage, sans évaluation ; le point de départ est vérifié au chargement.
Grammaire volontairement restreinte, sans évaluation de code : noms de
dimensions, nombres, `+ - * / // % **`, comparaisons, `and`, `or`,
`not`, parenthèses (`build_constraint`). Une contrainte supplémentaire
peut aussi être passée en Python par `space_constraint=` ; les deux se
cumulent. Un espace entièrement catégoriel et
plus petit que le budget (ex. `swinir`, 3 points) est évalué
**exhaustivement**, sans modèle.

## 4. Exécution

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
  exacte (`REFERENCE_TECHNIQUE.md` §5), avec σ = 1 dans le domaine transformé.

## 5. Usage

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

## 6. Budget et précautions

Pour σ et le modèle de bruit (quoi fixer, quoi optimiser, départ de
`sigma_scale` après démosaïçage), voir §2.

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

## 7. Tests et exemples

`tests/test_optimization.py` : espaces de recherche (fichiers JSONC
commentés, cohérence avec la signature de chaque méthode, point de départ
égal aux réglages par défaut), contraintes déclarées (grammaire, refus
d'expressions non autorisées, cumul avec `space_constraint=`), conversion
des unités, exécution séquentielle et parallèle, recherche exhaustive,
SSIM, image couleur, `sigma_scale0`, mode stabilisation de variance,
échecs pénalisés, journal CSV, figures, reproduction exacte du meilleur
score par `best_kwargs`.

`tests/test_examples.py` lance `examples/optimize_parameters.py` (niveaux
de gris, puis couleur avec `--kappa-y`) avec un budget réduit.

| Script | Usage |
|---|---|
| `examples/optimize_parameters.py` | optimiser une méthode : journal CSV, meilleurs paramètres (JSON), image débruitée, figures |
| `examples/benchmark.py` | réglages par défaut puis optimisés, réglage et évaluation séparés (`BENCHMARK.md` §2-3) |

## 8. Évolutions envisagées

- **Moteur d'optimisation** : remplacer scikit-optimize (archivé) par
  Optuna (`GPSampler` ou `TPESampler`), en gardant les mêmes fichiers
  d'espaces de recherche (§6).
- **Ajustement fin de (a, b)** en dimensions optionnelles, bornées
  autour de la mesure, pour une mesure du capteur douteuse (§2.5).
- **Estimation de (a, b) sur une seule image**, quand aucune PTC n'est
  disponible.
- **Validation croisée intégrée** : optimiser sur une partie des images
  de référence et mesurer sur l'autre, dans `optimize_parameters`
  lui-même.

## Annexe A — Historique des choix et corrections

Choix et corrections propres à l'optimisation ; l'historique du reste du
module est dans `REFERENCE_TECHNIQUE.md`, annexe A.

| Date | Objet | Décision ou correction |
|---|---|---|
| 2026-10-28 | Optimisation | Moteur d'optimisation bayésienne repris d'un outil antérieur de l'auteur, sans ses débruiteurs. Conflit corrigé : le paramètre `method` de `wavelet_denoise` empêchait l'appel par `denoise` ; l'optimiseur appelle la fonction de la méthode directement. Point de départ de SwinIR retiré : fixé aux poids « 25 » alors que le choix automatique donne « 15 », il affichait un faux gain de 2,09 dB. |
| 2026-10-29 | Espaces de recherche | Passage de dictionnaires Python à un fichier JSONC commenté par méthode ; espace de NL-Bayes étendu à 10 dimensions. |
| 2026-10-29 | Figures d'optimisation | `plot_evaluations` de scikit-optimize échouait pour un espace à une dimension : tracé manuel dans ce cas. |
| 2026-11-02 | Contraintes déclaratives (0.1.1) | L'espace de NL-Ridge autorisait k₁ ≥ p₁² (p₁ = 5, k₁ jusqu'à 40) : YᵀY singulière à l'étape 1, PSNR de 1,6 dB, sans erreur signalée. Correction : clé `"constraints"` dans les fichiers d'espaces, analysée sans évaluation de code et appliquée automatiquement par `optimize_parameters` (mécanisme équivalent au `build_space_constraint` de l'outil d'origine, rendu déclaratif) ; contrainte `k1 < p1 * p1` sur `nl_ridge` et `nl_ridge_inria` ; `nl_ridge_denoise` et `nl_ridge_inria_denoise` refusent un tel réglage. |
| 2026-11-16 | Image couleur, départ de C | `optimize_parameters` accepte les images couleur pour `nl_bayes` (PSNR sur les trois canaux, SSIM moyen) ; argument `sigma_scale0` (et `--kappa-y`) pour partir de C = √κ_Y + 0,2 ; borne haute de `sigma_scale` de NL-Bayes portée à 2,5. Correction d'une recommandation antérieure : avec la stabilisation, `sigma_scale` n'est pas superflu, c'est le facteur C. |
| 2026-11-16 | Document séparé | L'optimisation, jusqu'ici chapitre de la référence technique, devient ce document ; le réglage de σ et du modèle de bruit (ex-§4.6 et §6.5 de la référence) y est regroupé au §2. |
