# Licences tierces — inventaire et statut

> **Avertissement.** Cet inventaire est tenu au meilleur effort, à partir
> d'une vérification directe des dépôts/paquets sources (fichier LICENSE,
> champ `license` de l'API GitHub, classifieur PyPI, carte du modèle
> Hugging Face), faite le 2026-10-28. Si une erreur y est identifiée
> (source mal vérifiée, statut incorrect, nouvelle dépendance non
> recensée), elle sera corrigée dès que signalée — et si nécessaire,
> l'intégration concernée sera retirée plutôt que sa licence "arrangée"
> a posteriori. Pour signaler un problème, voir `CONTRIBUTING.md`.

Ce document recense, pour chaque méthode de `src/eoqual/denoising/`, le
code qui fait réellement le calcul, sa licence **vérifiée**, et le statut
de compatibilité avec la licence MIT annoncée pour `eoqual-denoising`
(`license.txt`).

**Principe** : **aucun code tiers n'est copié (vendored) dans ce dépôt.**
Le code d'`eoqual-denoising` se limite à des couches d'appel (validation
de l'image, gestion de `sigma` et des unités, conversion de format,
chargement paresseux) écrites pour ce projet. Le calcul est délégué à :

- des bibliothèques permissives installées par pip (socle) ;
- des paquets pip optionnels (extras), jamais installés par défaut ;
- des composants externes que l'utilisateur se procure lui-même (clone
  d'un dépôt, exécutable compilé), jamais installés ni redistribués par
  ce paquet.

**État actuel** : toutes les méthodes du **socle** reposent sur des
bibliothèques BSD-3-Clause / Apache-2.0 / MIT, et la stabilisation de
variance (`vst.py`) est une implémentation originale MIT. Quatre
intégrations ont une licence **restrictive**, signalée à l'utilisateur
à chaque niveau (`list_methods()`, message d'`ImportError`,
`README.md`) : `bm3d` (non commercial), `nl_ridge_inria` (AGPL-3.0),
`nl_bayes_ipol` (GPL-3.0+) et `wavelet_dict` (via `ptwt`, EUPL-1.2).
NL-Bayes et NL-Ridge sont par ailleurs disponibles dans le socle sous
MIT (`nl_bayes`, `nl_ridge`, réimplémentations depuis les articles). Deux
modèles utilisent des **poids sans licence déclarée** : `gsdrunet` et
`epll`. Une méthode
n'est **pas incluse** : `wnlb` (§2.4).

## 1. Légende

- 🔴 **Bloquant** — licence restrictive ou absente, et atteignable sans
  action explicite de l'utilisateur (import du socle). *Aucune entrée
  dans cet état actuellement.*
- 🟡 **Restriction d'usage** — licence non permissive (copyleft,
  non commercial). L'utilisateur ne l'obtient qu'en installant un extra
  ou en fournissant lui-même le composant externe.
- 🟢 **Compatible** — licence permissive (MIT/BSD/Apache), compatible avec
  MIT moyennant la conservation de la notice d'origine.

## 2. Inventaire

### 2.1 Socle — bibliothèques permissives (🟢)

| Méthode | Fichier | Calcul délégué à | Licence vérifiée |
|---|---|---|---|
| `bilateral` | `backends/bilateral/opencv_impl.py` | `cv2.bilateralFilter` (OpenCV) | **Apache-2.0** — classifieur PyPI `opencv-python` |
| `nl_means` | `backends/nl_means/skimage_impl.py` | `skimage.restoration.denoise_nl_means` | **BSD-3-Clause** — champ `license` PyPI `scikit-image` |
| `tv_chambolle` | `backends/tv/skimage_impl.py` | `skimage.restoration.denoise_tv_chambolle` | **BSD-3-Clause** |
| `tv_bregman` | `backends/tv/skimage_impl.py` | `skimage.restoration.denoise_tv_bregman` | **BSD-3-Clause** |
| `wavelet` | `backends/wavelet/skimage_impl.py` | `skimage.restoration.denoise_wavelet` (+ PyWavelets) | **BSD-3-Clause** / **MIT** (PyWavelets) |
| `nl_bayes` | `backends/nl_bayes/native.py` | **Réimplémentation « clean-room »** écrite pour ce projet depuis le seul texte de l'article IPOL (Lebrun, Buades & Morel, 2013 : formules 10-15, §3.3-3.4, tableau §4.8, §6.1.2). Ni le code C++ IPOL (GPL-3.0+), ni le code de l'archive, ni aucune autre implémentation n'ont été consultés pour l'algorithme (seuls le `README.txt` et l'analyse des arguments de `main.cpp` ont été lus, pour l'appel externe de `nl_bayes_ipol`). Valeurs de paramètres reprises de l'article (données publiées) | MIT (auteur du dépôt) |
| `nl_ridge` | `backends/nl_ridge/native.py` | **Réimplémentation « clean-room »** écrite pour ce projet depuis le seul texte des articles (Herbreteau & Kervrann, ICIP 2022 — arXiv:2203.00570 ; SIAM J. Imaging Sci. 2025 — arXiv:2402.13816 : propositions 2.1-2.3, formules 4-6 et 13-17, reprojection §2.4/§2.5, tableau 1). Le code des auteurs (`nlridge.py`, AGPL-3.0) n'a pas été consulté pour l'algorithme : seuls sa signature et sa docstring (noms et valeurs recommandées des paramètres) avaient été lus pour l'appel externe de `nl_ridge_inria`, et il n'a servi ensuite qu'à comparer les résultats | MIT (auteur du dépôt) |
| *(utilitaire)* | `utils.py::estimate_noise_sigma` | `skimage.restoration.estimate_sigma` | **BSD-3-Clause** |
| Optimisation des paramètres | `optimization/` | **Implémentation de l'auteur du dépôt** : moteur d'optimisation bayésienne repris de son outil personnel antérieur (non publié) et adapté au registre de ce paquet ; aucun code tiers repris. Calcul délégué à **scikit-optimize** (extra `optim`), et `concurrent.futures` / `csv` / `json` (bibliothèque standard) | MIT (auteur du dépôt) / **BSD-3-Clause** (scikit-optimize, champ `license` PyPI et API GitHub). Dépôt d'origine `scikit-optimize/scikit-optimize` **archivé** (2024) ; versions 0.10.x publiées par la fourche `holgern/scikit-optimize` (BSD-3-Clause) |
| `vst_denoise` et GAT | `vst.py` | **Implémentation originale** écrite depuis Anscombe (1948), Murtagh, Starck & Bijaoui (1995) et Mäkitalo & Foi (2011, 2013) ; aucun code tiers consulté ni repris (le code MATLAB des auteurs, sous licence non commerciale, n'a pas été utilisé). Loi de Poisson : `scipy.stats` | MIT (auteur du dépôt) / **BSD-3-Clause** (SciPy) |

Autres dépendances du socle (numpy, scipy, matplotlib, rich) : BSD /
BSD / PSF-like (matplotlib) / MIT.

Les valeurs par défaut des paramètres (`sigma_color = 2.5 σ` du bilatéral,
`weight = 0.75 σ` de TV Chambolle, `weight = 1/σ` de TV Bregman) ont été
calibrées pour ce projet (voir `REFERENCE_TECHNIQUE.md` §2) ; celle de
NL-Means (`h = 0.6 σ`) suit la documentation de scikit-image.

### 2.2 Extras pip optionnels

| Méthode | Extra | Fichier | Calcul délégué à | Licence vérifiée | Statut |
|---|---|---|---|---|---|
| `drunet` | `deep` | `backends/drunet/deepinv_impl.py` | `deepinv.models.DRUNet` + poids `drunet_deepinv_gray_finetune_26k.pth` téléchargés par deepinv au premier appel | Code : **BSD-3-Clause** (API GitHub `deepinv/deepinv` ; l'architecture y est reprise de DPIR, K. Zhang, **MIT** — API GitHub `cszn/DPIR`). Poids : **BSD-3-Clause** (carte du modèle `huggingface.co/deepinv/drunet`) | 🟢 permissif — en extra uniquement à cause du poids de PyTorch |
| `swinir` | `deep` | `backends/deepinv_zoo/models.py` | `deepinv.models.SwinIR` + poids `004_grayDN_DFWB_s128w8_SwinIR-M_noise{15,25,50}.pth` téléchargés depuis les *releases* GitHub de `JingyunLiang/SwinIR` | Code : **BSD-3-Clause** (deepinv). Poids : **Apache-2.0** (API GitHub `JingyunLiang/SwinIR` ; poids publiés dans les *releases* du dépôt des auteurs) | 🟢 |
| `restormer` | `deep` | `backends/deepinv_zoo/models.py` | `deepinv.models.Restormer` + poids `gaussian_gray_denoising_blind.pth` (miroir `huggingface.co/deepinv/Restormer`) | Code : **BSD-3-Clause** (deepinv, architecture adaptée de `swz30/Restormer`, **MIT**). Poids : publiés par les auteurs avec `swz30/Restormer` (**MIT**, API GitHub) ; le miroir Hugging Face de deepinv n'a pas de carte de licence | 🟢 permissif par l'origine ; miroir sans carte |
| `gsdrunet` | `deep` | `backends/deepinv_zoo/models.py` | `deepinv.models.GSDRUNet` + poids `GSDRUNet_grayscale_torch.ckpt` (`huggingface.co/deepinv/gradientstep`) | Code : **BSD-3-Clause** (deepinv, d'après `samuro95/GSPnP`, **MIT**). Poids : **licence non déclarée** — le dépôt Hugging Face n'a ni carte ni fichier de licence (vérifié 2026-10-28), et l'origine exacte du fichier (auteurs de GSPnP ou réentraînement deepinv) n'est pas établie | 🟡 **licence des poids à confirmer** |
| `epll` | `deep` | `backends/deepinv_zoo/models.py` | `deepinv.models.EPLLDenoiser` + mélange gaussien `GMM_BSDS_gray2.pt` (`huggingface.co/deepinv/EPLL`) | Code : **BSD-3-Clause** (deepinv). Poids : **licence non déclarée** — dépôt Hugging Face sans carte ni fichier de licence (vérifié 2026-10-28) | 🟡 **licence des poids à confirmer** |
| `tgv` | `deep` | `backends/deepinv_zoo/models.py` | `deepinv.models.TGVDenoiser` (sans poids ; deepinv indique l'avoir adapté du code MATLAB de L. Condat et d'un notebook de D. Smolyakov) | **BSD-3-Clause** (deepinv — la reprise du code amont relève de deepinv) | 🟢 |
| `wavelet_dict` | `deep` + `ptwt` | `backends/deepinv_zoo/models.py` | `deepinv.models.WaveletDictDenoiser`, transformée par `ptwt` (PyTorch Wavelet Toolbox) | deepinv : **BSD-3-Clause**. `ptwt` : **EUPL-1.2** (classifieur PyPI et API GitHub `v0lta/PyTorch-Wavelet-Toolbox`) | 🟡 **voir avertissement ci-dessous** |
| `bm3d` | `bm3d` | `backends/bm3d/tau_wrapper.py` | paquet PyPI `bm3d` 4.x (Y. Mäkinen, L. Azzari, A. Foi — Tampere University) | **Licence propriétaire de Tampere University** — classifieur PyPI `License :: Free for non-commercial use`, fichier `LICENSE` du paquet lu le 2026-10-28 | 🟡 **voir avertissement ci-dessous** |

**⚠️ Avertissement poids `gsdrunet` et `epll`.** Le code est BSD-3-Clause,
mais les fichiers de poids téléchargés n'ont pas de licence déclarée :
juridiquement, l'absence de licence vaut « tous droits réservés ». Usage
d'évaluation et de recherche : risque faible (fichiers publiés
volontairement par l'équipe deepinv pour cet usage). Usage dans un
produit ou un service : demander confirmation à l'équipe deepinv avant.
Ces poids ne sont pas redistribués par `eoqual-denoising`.

**⚠️ Avertissement `ptwt` (EUPL-1.2).** L'EUPL est une licence copyleft
européenne. Elle oblige à publier sous EUPL (ou sous une licence
compatible listée en annexe, dont la GPL) les modifications et œuvres
dérivées de `ptwt` que l'on distribue. Simplement l'importer comme
bibliothèque non modifiée relève, selon l'interprétation de la
Commission européenne, de l'usage et non de la dérivation ; ce point
reste discuté. `eoqual-denoising` ne fait qu'appeler deepinv, qui
appelle `ptwt` : il reste MIT. L'extra `ptwt` n'est jamais installé par
défaut. Si votre produit distribué doit rester sous licence
permissive, ne l'installez pas (`wavelet` couvre le même besoin, avec un
PSNR équivalent).

**⚠️ Avertissement `bm3d`.** La licence TAU autorise reproduction, usage
et modification **uniquement** : (i) à des fins d'information et **non
commerciales** ; (ii) en conservant toutes les notices de copyright ;
(iii) **sans recopier ni publier le logiciel sur un réseau** sans accord
préalable de TAU ; (iv) en **citant les auteurs** dès que des résultats
obtenus avec le logiciel sont rendus publics. `eoqual-denoising` ne
redistribue pas `bm3d` : l'extra ne fait que demander à pip de
l'installer depuis PyPI. L'utilisateur qui installe cet extra accepte
lui-même ces conditions. **Usage commercial (y compris prestation,
produit ou service payant) : ne pas installer cet extra.**

### 2.3 Composants externes fournis par l'utilisateur (🟡)

Ces deux méthodes ne sont **ni des dépendances pip, ni du code embarqué**.
`eoqual-denoising` contient seulement une couche d'appel qui (a) charge un
fichier que l'utilisateur a récupéré lui-même, ou (b) lance un programme
indépendant et échange des fichiers avec lui.

| Méthode | Fichier | Composant externe | Licence vérifiée | Mode d'appel |
|---|---|---|---|---|
| `nl_ridge_inria` | `backends/nl_ridge/external.py` | `nlridge.py` du dépôt https://github.com/sherbret/NL-Ridge (S. Herbreteau, C. Kervrann — Inria), chemin via `nlridge_path=` ou `EOQUAL_NLRIDGE_PATH` | **AGPL-3.0** — fichier `LICENSE` et champ `license` API GitHub. ⚠️ L'en-tête de `nlridge.py` indique en plus « Copyright (C) Inria, All Rights Reserved » : la licence du dépôt (AGPL-3.0) est celle qui s'applique à la copie récupérée | Import Python du fichier, dans le même processus |
| `nl_bayes_ipol` | `backends/nl_bayes/ipol_external.py` | exécutable `NL_Bayes` compilé depuis l'archive IPOL `nl-bayes_20130617.tar.gz` (M. Lebrun), chemin via `nlbayes_exe=` ou `EOQUAL_NLBAYES_EXE` | **GPL-3.0-or-later** — page IPOL de l'article et `README.txt` de l'archive | Programme séparé (sous-processus), échange de fichiers PNG |

**⚠️ Avertissement `nl_ridge_inria` (AGPL-3.0).** Charger `nlridge.py` dans le
même processus Python que votre code revient à combiner les deux
programmes. Si vous **distribuez** un logiciel qui l'utilise, ou si vous
l'exposez comme **service réseau** (clause réseau de l'AGPL), l'ensemble
peut devoir être publié sous AGPL-3.0. Usage local de recherche ou
d'évaluation : pas de contrainte particulière. `eoqual-denoising` reste
MIT car il ne contient ni ne redistribue `nlridge.py`. Les valeurs de
paramètres par palier de σ appliquées par la couche d'appel (`p1`, `p2`,
`k1`, `k2`) sont les valeurs recommandées publiées par les auteurs
(docstring amont et articles) — des données numériques, pas du code
repris.

Pourquoi un chemin plutôt qu'une dépendance pip : le `setup.cfg` amont
déclare `packages = find:` alors que le code est un module isolé
(`nlridge.py`) ; `pip install` du dépôt n'installe donc aucun module
importable (vérifié le 2026-10-28). Il fige aussi `torchvision==0.17`.

**⚠️ Avertissement `nl_bayes_ipol` (GPL-3.0+).** NL-Bayes tourne dans un
processus séparé et ne communique que par fichiers : selon la lecture
usuelle de la GPL (FAQ de la FSF), deux programmes qui communiquent ainsi
restent des œuvres distinctes, et `eoqual-denoising` n'est pas une œuvre
dérivée de NL-Bayes. Si vous **redistribuez** l'exécutable `NL_Bayes`
(par exemple dans une image Docker), vous devez respecter la GPL pour
celui-ci (sources, texte de licence).

### 2.4 Méthode non incluse

| Méthode | Raison |
|---|---|
| **WNLB** (Wavelet Non-Local Bayes) | Seule source identifiée : https://github.com/mago876/wnlb (M. González Olmedo, dernier commit 2016). **Ce dépôt n'a aucun fichier LICENSE** (vérifié via l'API GitHub, 2026-10-28) : tous droits réservés par défaut. Il n'est donc ni copiable ni utilisable comme dépendance. Il n'est de toute façon plus exécutable tel quel avec numpy ≥ 2 (`numpy.lib.pad`, `numpy.matlib`). **Aucune publication de référence** n'a été identifiée pour cette méthode, donc pas de réimplémentation possible d'après un article. Voir §3. |

### 2.5 Dépendances pip — licences déclarées

Licences relevées sur PyPI (champ `license` ou classifieur) pour les
versions disponibles au moment de la vérification. Aucune n'est
embarquée dans le paquet : pip les installe depuis PyPI.

| Paquet | Usage | Licence déclarée |
|---|---|---|
| numpy | socle | BSD-3-Clause (avec composants 0BSD, MIT, Zlib) |
| scipy | socle (loi de Poisson de la GAT) | BSD-3-Clause |
| scikit-image | socle (NL-Means, TV, ondelettes, estimation de σ) | BSD-3-Clause |
| opencv-python | socle (bilatéral, lecture/écriture d'images des exemples) | Apache-2.0 |
| PyWavelets | socle (ondelettes) | MIT (avec composants BSD-3-Clause) |
| matplotlib | socle (figures) | licence matplotlib (type PSF) |
| rich | socle (tableaux de `list_methods`) | MIT |
| torch | extra `deep` | Apache-2.0 (avec exception LLVM pour des composants embarqués) selon PyPI |
| deepinv | extra `deep` | BSD-3-Clause |
| einops | extra `deep` (requis par `nl_ridge_inria`) | MIT |
| ptwt | extra `ptwt` | **EUPL-1.2** (§2.2) |
| bm3d | extra `bm3d` | **non commercial**, Tampere University (§2.2) |
| scikit-optimize | extra `optim` | BSD-3-Clause |

## 3. Réimplémentation depuis les publications — évaluation

Objectif : remplacer à terme les intégrations sous licence restrictive
par des implémentations MIT écrites pour ce projet depuis les seules
publications (voir §4 pour le principe). Les estimations sont des
**ordres de grandeur** pour une implémentation numpy/PyTorch qui atteint
un PSNR à ~0,2 dB de la référence sur des images standard. Elles ne
comptent ni l'optimisation poussée ni la couleur.

| Méthode | Éléments publics disponibles | Difficulté | Effort estimé | Points d'attention |
|---|---|---|---|---|
| **NL-Bayes** | Article IPOL (Lebrun et al., 2013) | Moyenne | — | **Réimplémenté** : `nl_bayes` (socle, MIT, `backends/nl_bayes/`). 32,44 dB à σ = 0,1 sur le benchmark, contre 32,02 pour l'exécutable IPOL et 32,38 pour BM3D ; couleur incluse. |
| **NL-Ridge** | Articles ICIP 2022 et SIAM 2025 (arXiv) | Moyenne-faible | — | **Réimplémenté** : `nl_ridge` (socle, MIT, `backends/nl_ridge/native.py`), bruit gaussien, contraintes linéaire et affine ; reproduit le code des auteurs à ≤ 0,04 dB. Restent : bruits de Poisson / poisson-gaussien, contraintes coniques et convexes. |
| **BM3D** | Dabov et al. (2007) ; analyse détaillée **IPOL** : Lebrun (2012), *An Analysis and Implementation of the BM3D Image Denoising Method*, IPOL 2, 175-213 (texte de l'article seul) | Élevée | 8–12 j | Nombreux détails (transformées 2D/1D, seuillage dur puis Wiener, pondération de l'agrégation, fenêtre de Kaiser) qui pèsent sur le PSNR final. Lent en Python pur : PyTorch ou numba nécessaire. **Brevet signalé — à vérifier avant tout travail** : le source de `xphoto::bm3dDenoising` (OpenCV-contrib, `bm3d_image_denoising.cpp`) n'est compilé qu'avec l'option `OPENCV_ENABLE_NONFREE` et lève sinon l'erreur « *This algorithm is patented and is excluded in this configuration* ». Le brevet lui-même (titulaire, territoires, date d'expiration) n'a pas encore été recherché. |
| **WNLB** | Aucune publication identifiée — seulement l'idée générale (NL-Bayes appliqué aux sous-bandes d'ondelettes) | — | 2–3 j (NL-Bayes natif disponible) | Sans article de référence, ce serait une **méthode nouvelle** de ce projet, à concevoir et valider soi-même (choix de l'ondelette, des sous-bandes traitées, du σ par sous-bande). À ne lancer qu'après NL-Bayes, et seulement si l'intérêt est démontré. |

Débruiteurs deepinv **non intégrés**, et pourquoi :

- **DnCNN** : poids publiés entraînés pour un seul niveau de bruit
  (σ = 2/255), inutilisables comme débruiteur général ;
- **FFDNet** : aucun poids téléchargeable dans deepinv ;
- **SCUNet** : poids couleur seulement, entraînés pour du bruit réel
  aveugle (Apache-2.0, `cszn/SCUNet`) ; à évaluer avant intégration ;
- **BM3D**, **TV**, **ondelettes**, **bilatéral**, **médian** de
  deepinv : doublons du catalogue (BM3D de deepinv appelle d'ailleurs le
  même paquet `bm3d`, non commercial) ;
- modèles de diffusion et de reconstruction générale (DiffUNet, RAM…) :
  hors du périmètre d'un débruiteur à σ connu.

## 4. Principe appliqué : publication vs code

- Le droit d'auteur protège l'**expression** (le code source, le texte et
  les figures d'un article), pas l'**idée** ni la **méthode**. Une
  réimplémentation écrite **depuis la seule publication** (équations,
  pseudo-code, description) n'est pas une œuvre dérivée du code des
  auteurs. Elle peut donc être publiée sous MIT, en citant l'article.
- Reprendre un **code** existant (copie, portage quasi littéral, même
  structure et mêmes noms) reste soumis à la licence de ce code. Un code
  **sans licence** n'est pas réutilisable du tout (tous droits réservés
  par défaut).
- Les **valeurs numériques** publiées (paramètres recommandés, tables)
  sont des données et non une expression protégée. Elles sont reprises
  en citant leur source.
- Deux réserves : (1) un **brevet** peut protéger une méthode
  indépendamment du droit d'auteur (voir BM3D, §3) ; (2) pour qu'une
  réimplémentation soit défendable, elle doit être faite **sans avoir le
  code tiers sous les yeux** (clean-room), en traçant dans l'en-tête du
  module les publications utilisées.

Ce document n'est pas un avis juridique.
