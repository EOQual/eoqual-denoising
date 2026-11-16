#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthodes classiques calculées avec DeepInverse (deepinv, BSD-3-Clause) :
a priori de patchs appris par mélange gaussien (EPLL) et méthodes
variationnelles sans apprentissage (TGV, dictionnaire d'ondelettes).

Les réseaux de neurones (SwinIR, Restormer, GS-DRUNet, DRUNet) sont passés
dans eoqual-neural (version 0.3.0) ; ils restent importables d'ici pour
compatibilité (``backends/neural.py``, extra ``neural``).

Fines couches d'appel : ce paquet n'embarque ni code ni poids. Les poids
sont téléchargés par deepinv au premier appel. Leur origine et leur
licence, vérifiées modèle par modèle, sont dans
``THIRD_PARTY_LICENSES.md`` §2.2.

Extra requis : ``pip install "eoqual-denoising[deep]"``.

Convention commune : ``sigma`` dans les unités de l'image (estimé s'il
est absent) ; l'image est ramenée dans [0, 1] via ``data_range`` (par
défaut son min/max) et ``sigma`` converti en conséquence.

Références (complètes dans ``BIBLIOGRAPHY.md``) :

- EPLL — Zoran & Weiss (2011), ICCV ;
- TGV — Bredies, Kunisch & Pock (2010), SIAM J. Imaging Sci. ; solveur
  primal-dual de Condat (2013), JOTA ;
- dictionnaire d'ondelettes — Starck, Murtagh & Fadili (2010) ; schéma
  proximal parallèle de Combettes & Pesquet (2008, PPXA) ;
- bibliothèque — Tachella et al. (2025), DeepInverse, JOSS 10(115).
"""
from __future__ import annotations

import numpy as np

from ...utils import check_image, from_unit_range, resolve_sigma, to_unit_range
from .._torch import cached_model, import_deepinv, pick_device, run_on_unit_image
from ..neural import gsdrunet_denoise, restormer_denoise, swinir_denoise  # noqa: F401  (compatibilité)

__all__ = [
    "swinir_denoise",
    "restormer_denoise",
    "gsdrunet_denoise",
    "epll_denoise",
    "tgv_denoise",
    "wavelet_dict_denoise",
    "SWINIR_NOISE_LEVELS",
]

#: Niveaux de bruit (échelle [0, 255]) des poids SwinIR niveaux de gris publiés.
SWINIR_NOISE_LEVELS = (15, 25, 50)


def _prepare(image, sigma, data_range, device):
    img = check_image(image)
    s = resolve_sigma(img, sigma)
    unit, lo, scale = to_unit_range(img, data_range)
    return unit, lo, scale, s / scale, pick_device(device)


def epll_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    data_range: tuple[float, float] | None = None,
    device: str | None = None,
    patch_size: int = 6,
    batch_size: int = -1,
) -> np.ndarray:
    """
    Débruite une image par EPLL (Expected Patch Log Likelihood) : a priori
    de patchs 6×6 sous forme de mélange gaussien à 200 composantes appris
    sur BSDS500, optimisé par demi-quadratique sur une suite de poids
    croissants.

    Parameters
    ----------
    image : np.ndarray
        Image 2D bruitée.
    sigma : float, optional
        Écart-type du bruit (unités de l'image). Estimé si absent.
    data_range : tuple(float, float), optional
        Dynamique ``(min, max)`` ramenée à [0, 1]. Par défaut, le min/max
        de l'image.
    device : str, optional
        Périphérique torch. Par défaut ``"cuda"`` si disponible.
    patch_size : int
        Taille des patchs du modèle pré-entraîné. Par défaut ``6``.
    batch_size : int
        Nombre de patchs traités à la fois (``-1`` = tous) ; à réduire en
        cas de mémoire insuffisante, sans effet sur le résultat.

    Returns
    -------
    np.ndarray
        Image débruitée, ``float64``, même forme que l'entrée.
    """
    _, dm = import_deepinv()
    unit, lo, scale, s_unit, device = _prepare(image, sigma, data_range, device)
    model = cached_model(
        ("epll", device, patch_size),
        lambda: dm.EPLLDenoiser(pretrained="download", patch_size=patch_size, channels=1, device=device),
    )
    out = run_on_unit_image(lambda x: model(x, sigma=s_unit, batch_size=batch_size), unit, device)
    return from_unit_range(out, lo, scale)


def tgv_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    weight: float | None = None,
    data_range: tuple[float, float] | None = None,
    device: str | None = None,
    n_it_max: int = 1000,
    crit: float = 1e-5,
) -> np.ndarray:
    """
    Débruite une image par variation totale généralisée d'ordre 2 (TGV²).

    La TGV pénalise à la fois le gradient et la dérivée seconde : elle
    garde les contours francs comme la TV, mais reconstruit les dégradés
    par morceaux affines au lieu de marches d'escalier.

    Parameters
    ----------
    image : np.ndarray
        Image 2D bruitée.
    sigma : float, optional
        Écart-type du bruit (unités de l'image). Estimé si absent. Sert
        uniquement à fixer ``weight`` par défaut.
    weight : float, optional
        Poids de régularisation, en unités de l'image (plus grand = plus
        lisse). deepinv en déduit λ₁ = 0,1·weight (ordre 1) et
        λ₂ = 0,15·weight (ordre 2). Par défaut ``7.5 * sigma`` (calibré
        empiriquement, voir REFERENCE_TECHNIQUE.md §3.6).
    data_range : tuple(float, float), optional
        Dynamique ``(min, max)`` ramenée à [0, 1]. Par défaut, le min/max
        de l'image.
    device : str, optional
        Périphérique torch. Par défaut ``"cuda"`` si disponible.
    n_it_max : int
        Nombre maximal d'itérations Chambolle-Pock. Par défaut ``1000``.
    crit : float
        Critère d'arrêt. Par défaut ``1e-5``.

    Returns
    -------
    np.ndarray
        Image débruitée, ``float64``, même forme que l'entrée.
    """
    _, dm = import_deepinv()
    unit, lo, scale, s_unit, device = _prepare(image, sigma, data_range, device)
    ths = (weight / scale) if weight is not None else 7.5 * s_unit
    model = dm.TGVDenoiser(n_it_max=n_it_max, crit=crit)
    out = run_on_unit_image(lambda x: model(x, ths=ths), unit, device)
    return from_unit_range(out, lo, scale)


def wavelet_dict_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    threshold: float | None = None,
    wavelets: tuple[str, ...] = ("db8", "db4"),
    level: int = 3,
    max_iter: int = 10,
    data_range: tuple[float, float] | None = None,
    device: str | None = None,
) -> np.ndarray:
    """
    Débruite une image par régularisation ℓ1 dans un dictionnaire redondant
    formé de plusieurs bases d'ondelettes : à chaque itération, l'image est
    seuillée dans chaque base, puis les résultats sont moyennés (schéma
    proximal parallèle). La redondance atténue les artefacts propres à une
    base unique (comparer avec ``wavelet``).

    Parameters
    ----------
    image : np.ndarray
        Image 2D bruitée.
    sigma : float, optional
        Écart-type du bruit (unités de l'image). Estimé si absent. Sert
        uniquement à fixer ``threshold`` par défaut.
    threshold : float, optional
        Seuil doux, en unités de l'image. Par défaut ``1.25 * sigma`` (calibré
        empiriquement, voir REFERENCE_TECHNIQUE.md §3.7).
    wavelets : tuple of str
        Bases d'ondelettes du dictionnaire. Par défaut ``("db8", "db4")``.
    level : int
        Niveaux de décomposition. Par défaut ``3``.
    max_iter : int
        Nombre maximal d'itérations (arrêt anticipé sur variation relative
        < 1e-3). Par défaut ``10``.
    data_range : tuple(float, float), optional
        Dynamique ``(min, max)`` ramenée à [0, 1]. Par défaut, le min/max
        de l'image.
    device : str, optional
        Périphérique torch. Par défaut ``"cuda"`` si disponible.

    Returns
    -------
    np.ndarray
        Image débruitée, ``float64``, même forme que l'entrée.
    """
    _, dm = import_deepinv()
    try:
        import ptwt  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "wavelet_dict_denoise nécessite le paquet 'ptwt' (PyTorch Wavelet "
            "Toolbox), sous licence EUPL-1.2 (copyleft) — lire "
            "THIRD_PARTY_LICENSES.md avant installation. Installer avec : "
            'pip install "eoqual-denoising[deep,ptwt]"'
        ) from exc
    unit, lo, scale, s_unit, device = _prepare(image, sigma, data_range, device)
    ths = (threshold / scale) if threshold is not None else 1.25 * s_unit
    model = cached_model(
        ("wavelet_dict", device, tuple(wavelets), level, max_iter),
        lambda: dm.WaveletDictDenoiser(level=level, list_wv=tuple(wavelets), max_iter=max_iter, device=device),
    )
    out = run_on_unit_image(lambda x: model(x, ths=ths), unit, device)
    return from_unit_range(out, lo, scale)
