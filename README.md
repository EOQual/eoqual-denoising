<p align="center">
  <img src="resources/eoqual_rounded.png" alt="EOQual" width="280">
</p>

# eoqual-denoising

Collection de méthodes de débruitage d'images, pensée pour l'observation
de la Terre, avec stabilisation de variance pour le bruit
poisson-gaussien des images brutes de capteur. Voir **[REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md)**
pour le détail de chaque méthode (principe, implémentation, limites,
performances indicatives), **[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)**
pour l'audit complet des licences,
**[BIBLIOGRAPHY.md](BIBLIOGRAPHY.md)** pour les publications de
référence, et **[BENCHMARK.md](BENCHMARK.md)** pour la comparaison
chiffrée des méthodes (réglages par défaut et optimisés).

## Sommaire

1. [Installation](#installation)
2. [Démarrage rapide](#démarrage-rapide)
3. [Catalogue des méthodes](#catalogue-des-méthodes)
4. [Licence](#licence)
5. [Contribuer](#contribuer)
6. [Credits](#credits)

---

## Installation

### 1. Socle (obligatoire)

```bash
pip install git+https://github.com/EOQual/eoqual-denoising.git
```

Installe tout ce qui est nécessaire pour 7 des 17 méthodes du catalogue
(`bilateral`, `nl_means`, `tv_chambolle`, `tv_bregman`, `wavelet`,
`nl_bayes`, `nl_ridge`) et
pour la stabilisation de variance (`vst_denoise`). Les dépendances sont
numpy, scipy, scikit-image, opencv-python, PyWavelets, matplotlib et
rich, toutes sous licence permissive.

### 2. Extras optionnels

| Extra | Commande | Débloque | Pourquoi optionnel |
|---|---|---|---|
| `deep` | `pip install "eoqual-denoising[deep]"` | `drunet`, `swinir`, `restormer`, `gsdrunet`, `epll`, `tgv` (et prérequis de `nl_ridge_inria`, `wavelet_dict`) | Dépendance lourde (PyTorch). Code BSD-3-Clause ; ⚠️ poids de `gsdrunet` et `epll` sans licence déclarée, voir [§2.2](THIRD_PARTY_LICENSES.md#22-extras-pip-optionnels) |
| `optim` | `pip install "eoqual-denoising[optim]"` | optimisation des paramètres (`optimize_parameters`) | scikit-optimize (BSD-3-Clause), inutile pour débruiter |
| `ptwt` | `pip install "eoqual-denoising[deep,ptwt]"` | `wavelet_dict` | ⚠️ `ptwt` est sous **EUPL-1.2** (copyleft) : lire [§2.2](THIRD_PARTY_LICENSES.md#22-extras-pip-optionnels) avant d'installer |
| `bm3d` | `pip install "eoqual-denoising[bm3d]"` | `bm3d` | ⚠️ Paquet `bm3d` de Tampere University sous licence **non commerciale** : **lire [THIRD_PARTY_LICENSES.md §2.2](THIRD_PARTY_LICENSES.md#22-extras-pip-optionnels) avant d'installer** |

### 3. Composants externes (non installés par pip)

Deux méthodes appellent un composant que vous vous procurez vous-même. Il
n'est ni embarqué ni redistribué par ce paquet, à cause de sa licence
copyleft :

| Méthode | Composant | Licence | Mise en place |
|---|---|---|---|
| `nl_ridge_inria` | clone de https://github.com/sherbret/NL-Ridge | **AGPL-3.0** | `export EOQUAL_NLRIDGE_PATH=/chemin/NL-Ridge` + extra `deep` |
| `nl_bayes_ipol` | exécutable `NL_Bayes` compilé depuis [IPOL](https://www.ipol.im/pub/art/2013/16/) | **GPL-3.0+** | `export EOQUAL_NLBAYES_EXE=/chemin/NL_Bayes` |

Ces deux composants ne servent qu'à comparer avec le code de référence :
NL-Bayes et NL-Ridge sont fournis **dans le socle** (`nl_bayes`,
`nl_ridge`), réimplémentés sous MIT depuis les articles, sans dépendance,
avec des résultats équivalents (`BENCHMARK.md`).

Lire les avertissements de [THIRD_PARTY_LICENSES.md §2.3](THIRD_PARTY_LICENSES.md#23-composants-externes-fournis-par-lutilisateur-)
avant de les utiliser dans un logiciel distribué ou un service.

Sans l'extra ou le composant requis, l'appel de la méthode lève une
erreur explicite qui indique quoi installer et sous quelle licence. Aucune
méthode ne plante à l'*import* du paquet.

Pour lister les méthodes, leur licence et leurs restrictions :

```python
from eoqual.denoising.runner import list_methods
list_methods()
```

---

## Démarrage rapide

```python
import eoqual.denoising as d

clean = d.astronaut_gray()                    # image de test, [0, 1]
noisy = d.add_gaussian_noise(clean, sigma=0.1)

out = d.nl_means_denoise(noisy, sigma=0.1)
out = d.tv_chambolle_denoise(noisy)           # sigma estimé automatiquement
out = d.drunet_denoise(noisy, sigma=0.1)      # extra "deep"

# Ou par le dispatcher générique
from eoqual.denoising.runner import denoise
out = denoise(noisy, method="wavelet", sigma=0.1)
```

### Image brute de capteur (bruit poisson-gaussien)

Le bruit d'une image brute n'est pas stationnaire : sa variance croît
avec le signal (bruit photonique). `vst_denoise` stabilise la variance
(transformée d'Anscombe généralisée), applique n'importe quelle méthode
du catalogue, puis revient au signal par l'**inverse exacte non
biaisée** :

```python
# gain K (e⁻/DN) et bruit de lecture (DN) : PTC d'eoqual-measure ;
# offset (DN) : niveau moyen d'obscurité
out = d.vst_denoise(raw, "drunet", gain=2.0, read_noise_dn=1.5, offset_dn=100.0)
```

### Optimiser les paramètres sur une image de référence

```python
from eoqual.denoising.optimization import optimize_parameters

res = optimize_parameters(noisy, reference, "nl_means", n_calls=25)   # extra "optim"
res.best_params, res.best_score, res.gain
clean = d.nl_means_denoise(other_image, **res.best_kwargs)
```

Optimisation bayésienne (scikit-optimize), séquentielle ou parallèle,
avec journal CSV et figures de diagnostic. Elle fonctionne aussi en
chaîne GAT pour une image brute (`vst=`). En ligne de commande :
`examples/optimize_parameters.py`. Voir
[REFERENCE_TECHNIQUE.md §5](REFERENCE_TECHNIQUE.md#5-optimisation-des-paramètres).

À faible flux, le gain sur l'application directe d'un débruiteur
gaussien atteint +2 à +10 dB (voir
[REFERENCE_TECHNIQUE.md §4](REFERENCE_TECHNIQUE.md#4-bruit-poisson-gaussien--stabilisation-de-variance)).

`sigma` est l'écart-type du bruit **dans les unités de l'image** (comptes
numériques, réflectance...). S'il est omis, il est estimé sur l'image.

**Appel unifié** : `denoise(image, method, **kwargs)` pour toutes les
méthodes, où `kwargs` regroupe `sigma` et les arguments propres à la
méthode choisie, par exemple :

```python
denoise(noisy, "nl_bayes", sigma=0.1, use_area2=True)
denoise(noisy, "wavelet", rule="VisuShrink", wavelet="db4")
denoise(noisy, "bm3d", sigma=0.1, profile="refilter")
```

Les arguments de chaque méthode sont dans sa docstring
(`help(d.nl_bayes_denoise)`) et dans
[REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md).

Exemples prêts à l'emploi dans `examples/` :

| Script | Usage |
|---|---|
| `one_method_cli.py` | débruiter un fichier (gris, couleur, image brute) |
| `all_methods_overview.py` | comparer les méthodes disponibles et produire une planche |
| `low_light_vst.py` | stabilisation de variance à faible flux |
| `optimize_parameters.py` | optimiser les paramètres d'une méthode |
| `benchmark.py`, `nl_bayes_benchmark.py` | reproduire `BENCHMARK.md` |

---

## Catalogue des méthodes

Toutes les méthodes attendent une image 2D (niveaux de gris) affectée
d'un bruit blanc gaussien additif ; `nl_bayes` accepte aussi une image
couleur (H, W, 3) (pour une image brute de capteur,
passer par `vst_denoise`).

| Méthode | Extra / composant | Principe | Licence du calcul |
|---|---|---|---|
| `bilateral` | — | Moyenne locale pondérée par la distance spatiale et radiométrique | Apache-2.0 (OpenCV) |
| `nl_means` | — | Moyenne non locale pondérée par la similarité de patchs | BSD-3 (scikit-image) |
| `tv_chambolle` | — | Variation totale (ROF), solveur dual de Chambolle | BSD-3 (scikit-image) |
| `tv_bregman` | — | Variation totale (ROF), solveur split Bregman | BSD-3 (scikit-image) |
| `wavelet` | — | Seuillage des coefficients d'ondelettes (BayesShrink/VisuShrink) | BSD-3 (scikit-image) |
| `nl_bayes` | — | Estimateur bayésien gaussien non local sur groupes de patchs, deux étapes ; **couleur** (Y₀U₀V₀ de l'article, RGB, canal par canal) | MIT (réimplémentation depuis l'article IPOL) |
| `nl_ridge` | — | Combinaisons linéaires de patchs similaires, poids en forme fermée (SURE puis ridge) | MIT (réimplémentation depuis les articles) |
| `drunet` | `deep` | Réseau U-Net résiduel conditionné par σ, poids pré-entraînés | BSD-3 (deepinv) |
| `swinir` | `deep` | Transformeur à fenêtres glissantes, poids par niveau de bruit (15/25/50) | Apache-2.0 (poids) / BSD-3 |
| `restormer` | `deep` | Transformeur à attention entre canaux, poids aveugles (σ ≤ 50/255) | MIT (poids) / BSD-3 |
| `gsdrunet` | `deep` | DRUNet en pas de gradient d'un potentiel appris (plug-and-play convergent) | BSD-3 ; ⚠️ poids sans licence déclarée |
| `epll` | `deep` | A priori de patchs par mélange gaussien appris | BSD-3 ; ⚠️ poids sans licence déclarée |
| `tgv` | `deep` | Variation totale généralisée d'ordre 2 (dégradés sans marches) | BSD-3 (deepinv) |
| `wavelet_dict` | `deep` + `ptwt` | Seuillage ℓ1 dans un dictionnaire de plusieurs bases d'ondelettes | ⚠️ **EUPL-1.2** (`ptwt`) |
| `bm3d` | `bm3d` | Filtrage collaboratif de groupes de blocs, transformée 3D | ⚠️ **non commercial** (TAU) |
| `nl_ridge_inria` | clone externe | Même méthode, code des auteurs | ⚠️ **AGPL-3.0** (Inria) |
| `nl_bayes_ipol` | exécutable externe | Même méthode, implémentation C++ de référence IPOL (quantification 8 bits) | ⚠️ **GPL-3.0+** (IPOL) |

Détail de chaque méthode et performances indicatives :
**[REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md)**.

**Une méthode est absente** (WNLB : source sans licence, aucune
publication identifiée) : voir
**[THIRD_PARTY_LICENSES.md §2.4](THIRD_PARTY_LICENSES.md#24-méthode-non-incluse)**.

---

## Licence

**Le code d'eoqual-denoising est sous licence MIT** (`license.txt`).

Ce dépôt ne contient **aucun code tiers copié**. Il se compose de couches
d'appel vers des bibliothèques et composants qui gardent leur propre
licence :

- **socle** et **stabilisation de variance** : bibliothèques permissives
  (BSD-3, Apache-2.0, MIT) et code original MIT ;
- **`drunet`**, **`swinir`**, **`restormer`**, **`tgv`** : licences
  permissives, sans restriction ;
- **`gsdrunet`**, **`epll`** : code BSD-3-Clause, mais **poids sans
  licence déclarée** ;
- **`wavelet_dict`** : via `ptwt`, **EUPL-1.2** ;
- **`bm3d`** : licence Tampere University, **usage non commercial
  uniquement** ;
- **`nl_ridge_inria`** : **AGPL-3.0** (le `nl_ridge` du socle est MIT) ;
- **`nl_bayes_ipol`** : **GPL-3.0+** (le `nl_bayes` du socle, lui, est
  MIT).

Aucune de ces méthodes n'est installée par défaut. Inventaire
complet, avec vérification de chaque source et évaluation de leur
réimplémentation sous MIT depuis les publications :
**[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)**.

---

## Contribuer

Bug, méthode manquante, erreur de licence constatée : les contributions
sont bienvenues. Voir **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## Credits

- scikit-image [https://scikit-image.org/]
- OpenCV [https://opencv.org/]
- PyWavelets [https://pywavelets.readthedocs.io/]
- DeepInverse [https://deepinv.github.io/] (DRUNet, SwinIR, Restormer, GS-DRUNet, EPLL, TGV, dictionnaire d'ondelettes ; extra `deep`)
- SwinIR [https://github.com/JingyunLiang/SwinIR], Restormer [https://github.com/swz30/Restormer], GSPnP [https://github.com/samuro95/GSPnP] (poids et architectures d'origine)
- PyTorch Wavelet Toolbox [https://github.com/v0lta/PyTorch-Wavelet-Toolbox] (extra `ptwt`)
- bm3d, Tampere University [https://pypi.org/project/bm3d/] (Mäkinen, Azzari & Foi ; extra `bm3d`)
- NL-Ridge [https://github.com/sherbret/NL-Ridge] (Herbreteau & Kervrann ; composant externe)
- NL-Bayes, IPOL [https://www.ipol.im/pub/art/2013/16/] (Lebrun, Buades & Morel ; article source de `nl_bayes`, exécutable de `nl_bayes_ipol`)

Image de test : `astronaut` (Eileen Collins, NASA, domaine public),
fournie par scikit-image.
