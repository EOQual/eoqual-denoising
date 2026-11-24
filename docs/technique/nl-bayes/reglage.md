# NL-Bayes : réglage par optimisation

## 12. Réglage par optimisation

### 12.1 Ce qu'il faut régler

Classement tiré des [§11.3](etudes.md#113-τ₀--une-règle-fonction-de-σ-testée-et-écartée) à [§11.7](etudes.md#117-n₂-et-zones-plates--taille-réelle-des-groupes), du plus utile au moins utile :

| Rang | Paramètre | Pourquoi | Domaine conseillé |
|---|---|---|---|
| 1 | `sigma_scale` | seul paramètre à effet fort partout (jusqu'à 1 dB et plus) ; compense un σ mal connu, ou donne le niveau réel du bruit après stabilisation | [0,5 ; 2,5] ; départ 1 (image brute ou σ connu), √κ_Y à C après démosaïçage |
| 2 | β₂, avec β₁ = 1 fixé | règle l'étape 2 par rapport à l'étape 1 ; permet de garder σ au niveau réel du bruit ([§11.6](etudes.md#116-σ-et-β₂-après-stabilisation)) | [0,4 ; 1,5], départ 1,2 (article) ; contre la vraie image, 1,2 avec σ = C est au meilleur ([§11.8](etudes.md#118-étape-2-couleur-commune-et-choix-de-la-référence)) |
| 3 | N₂ | groupes plus petits que l'article sur images texturées ; les zones plates gardent de grands groupes par τ₀ ([§11.7](etudes.md#117-n₂-et-zones-plates--taille-réelle-des-groupes)) | [15 ; 90], porté à k₂² + 1 au moins |
| — | `use_area1` | indispensable à faible flux ; ne pas l'optimiser, le garder actif | fixe : oui |
| — | β₁, γ, τ₀, α, `use_area2` | effets faibles, ou redondants avec `sigma_scale` | fixes : 1 ; 1,05 ; 4 ; 7 ; non |
| — | k₁, k₂, N₁ | effets faibles ; la règle de l'article dépend de la couleur (5 en gris, 3 en couleur à faible bruit) | règle de l'article |

Trois dimensions au lieu de sept : 15 à 25 évaluations suffisent. Les
paramètres non optimisés sont fixés explicitement, sauf k₁, k₂ et N₁ :
aucun palier de β ne bascule quand `sigma_scale` fait varier σ ([§6.2](parametres.md#62-les-valeurs-de-larticle-et-σ)).
Les paliers de taille de patch restent (σ = 20 et 50 sur [0, 255]), mais
ils ne concernent que les bruits forts.

**Valeurs à retenir après stabilisation de variance** (σ = 1 transmis,
multiplié par `sigma_scale`) : `sigma_scale` = 1 sur image brute,
C = √κ_Y + 0,2 sur image démosaïquée ; β₁ = 1 ; β₂ = 1,2 ([§11.8](etudes.md#118-étape-2-couleur-commune-et-choix-de-la-référence) ; le
gain d'un réglage fin y est ≤ 0,04 dB contre la vraie image). β₂ multiplie le
même σ² que β₁ : c'est un rapport entre les deux étapes, à `sigma_scale`
donné. Si `sigma_scale` monte, β₂ optimal baisse (crête β₂·σ² ≈
constante, [§11.6](etudes.md#116-σ-et-β₂-après-stabilisation)).

### 12.2 L'espace par défaut

`optimization/spaces/nl_bayes.jsonc` optimise `sigma_scale`,
β₂ et N₂ ; il part de l'article (`sigma_scale` = 1, β₂ = 1,2) avec
N₂ = 30, qui est la valeur de l'article en couleur et la moitié de celle
du gris (60 à k₂ = 5). β₁, γ, τ₀, α, `use_area1` et `use_area2` sont
dans `fixed`. L'espace précédent (7 dimensions : `sigma_scale`, k₁, k₂,
α, τ₀, `use_area1`, `use_area2`) optimisait surtout des paramètres à
effet faible, et laissait de côté β₂ et N₂.

### 12.3 Précautions

- **Valider sur une autre image** que celle du réglage : sur le benchmark,
  un réglage peut gagner 0,15 dB sur l'image de réglage et 0,02 dB
  seulement sur une autre ([benchmark](../benchmark.md) [§3](methode.md#3-la-covariance-dun-groupe)).
- **Images de télédétection** : régler sur des images du capteur, au bon
  niveau de flux ; le classement des paramètres change entre faible et
  fort flux ([§11.5](etudes.md#115-sensibilité-rgb-démosaïqué-zones-sombres)).
- Le réglage de σ et du modèle de bruit (stabilisation, facteur C) suit
  [optimisation](../optimisation.md) [§2](methode.md#2-patchs-groupes-et-distance).

### 12.4 Contraintes et redondances de l'espace de recherche

**Contraintes de la méthode.** L'article ne borne aucun paramètre ; les
seules contraintes viennent du calcul :

| Contrainte | Pourquoi | Traitement |
|---|---|---|
| N₁ ≥ k₁² + 1 | la covariance de l'étape 1 est inversée telle quelle ; avec N ≤ k² elle est singulière (patchs laissés bruités) | N porté à k² + 1 par `resolve_parameters` ([§6.3](parametres.md#63-dépendances-entre-paramètres)) |
| N₂ ≥ k₂² + 1 | sans nécessité : l'étape 2 inverse C_b + β₂σ²I, toujours inversible | appliqué aussi, par symétrie ; borne basse de l'espace : 15 |
| étape 2 couleur | patchs de 3k₂² valeurs (canaux filtrés ensemble, [§8.1](parametres.md#81-trois-traitements-de-la-couleur)), même raison qu'au-dessus : aucune condition sur N₂ | — |
| `nl_bayes_cfa` | inverse U·C·Uᵀ + βσ²I (d × d), toujours inversible : aucune condition sur N | N ≥ 2 |
| fenêtre ≥ patch | n = α·k, α ≥ 1 | α fixé à 7 |
| image ≥ patch | patchs extraits dans l'image | erreur explicite |
| échelle de τ₀ et des paliers | exprimés sur [0, 255] | `data_range` ([§7.2](parametres.md#72-léchelle-0-255)) |

**Redondances de la paramétrisation** (évitables : l'article règle σ
directement, sans `sigma_scale`) :

| Redondance | Cause | Traitement |
|---|---|---|
| `sigma_scale` × β₁ × β₂ × γ | le calcul ne dépend que de β₁σ², β₂σ², γσ ([§7.3](parametres.md#73-redondance-entre-σ-β-et-γ)) | β₁ et γ fixés dans l'espace par défaut ; avertissement d'`optimize_parameters` s'ils sont tous libres |
| `sigma_scale` × β₁ × β₂, sans critère homogène | il ne reste que β₁σ², β₂σ² | idem, avertissement si `use_area1` et `use_area2` sont fixés à faux |

**τ₀ est un plancher, jamais un remplaçant de N.** À l'étape 2, le
groupe contient les patchs à distance ≤ max(τ₀, d_N₂) : au moins N₂
patchs, davantage dans les zones plates (figure du [§7.3](parametres.md#73-redondance-entre-σ-β-et-γ)). Un seuil qui
*remplacerait* N (groupe = patchs à distance ≤ τ, quel que soit N) ferait
de N un paramètre sans effet et pourrait réduire un groupe au seul patch
de référence dans les zones texturées. À l'étape 1, l'article n'a pas de
seuil : N₁ plus proches. C'est ce que fait `nl_bayes`, et ce qu'il faut
vérifier dans toute autre implémentation avant d'en optimiser N.

### 12.5 Correspondance avec nlbayes-ext

Le calcul de `nl_bayes` et `nl_bayes_cfa` est fait par la bibliothèque
**nlbayes-ext** (https://github.com/oam31/nlbayes-ext). Les deux exposent
les mêmes paramètres de la méthode (`k1`, `k2`, `N1`, `N2`, `beta1`,
`beta2`, `gamma`, `tau0`, `window_factor`, `use_area1`, `use_area2`,
`color_mode`, `engine`, `step` ; pour la mosaïque `pattern`, `k`, `N`,
`beta`, `n_iter`, `pilot`, `use_area`…) ; seuls le bruit et le réglage de
σ s'écrivent différemment :

| Besoin | eoqual-denoising | nlbayes-ext |
|---|---|---|
| bruit blanc de σ connu | `denoise(img, "nl_bayes", sigma=s)` | `denoise_gray(img, sigma=s)`, `denoise_rgb(img, sigma=s)` |
| σ estimé | `sigma=None` : estimateur de scikit-image (Donoho & Johnstone), moyenne des canaux en couleur | `sigma=None` : même principe (coefficients de Haar diagonaux), sans scikit-image ; valeurs un peu différentes |
| bruit du capteur var = a + b·L | `vst_denoise(img, "nl_bayes", gain=1/b, read_noise_dn=√a, offset_dn=o)` | `denoise_rgb(img, noise=(a, b), offset=o)` |
| facteur de σ après stabilisation | `noise_factor=C` (`vst_denoise`) | `sigma_scale=C` |
| facteur de σ dans l'optimisation | dimension `sigma_scale` de l'espace | `sigma_scale` (argument) |
| inverse de la stabilisation | `inverse=` de `vst_denoise` | `inverse=` |
| mosaïque de Bayer | `denoise(m, "nl_bayes_cfa", sigma=s, pattern=…)` ; bruit du capteur par `vst_denoise` | `denoise_cfa(m, sigma=s)` ou `denoise_cfa(m, noise=(a, b))` |

À σ donné, ou avec le modèle de bruit (a, b), les sorties sont
identiques au bit près (vérifié, y compris `vst_denoise` contre
`noise=(a, b)`) ; σ estimé diffère un peu, l'estimateur n'étant pas le
même. Un réglage
trouvé par `optimize_parameters` (`best_kwargs`) se transpose donc
directement : `sigma` × `sigma_scale` devient `sigma` (bruit blanc) ou
`sigma_scale` (avec `noise`), les autres paramètres gardent leur nom.
