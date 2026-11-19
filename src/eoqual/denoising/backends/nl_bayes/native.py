#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NL-Bayes (Buades, Lebrun & Morel, 2013) — méthode ``nl_bayes``.

Le calcul est fait par la bibliothèque **nlbayes-ext** (MIT,
https://github.com/oam31/nlbayes-ext), qui implémente NL-Bayes à partir du
seul article IPOL ; ce module ne fait qu'adapter l'appel au contrat
d'eoqual-denoising (``f(image, sigma=None, ...)``, σ estimé par
:func:`~eoqual.denoising.utils.estimate_noise_sigma` s'il est absent).

Méthode, paramètres et études : NL_BAYES.md.
"""
from __future__ import annotations

import numpy as np

from nlbayes_ext import article_parameters as nl_bayes_parameters  # noqa: F401  (réexporté)
from nlbayes_ext import denoise_gray, denoise_rgb, resolve_parameters  # noqa: F401

from ...utils import estimate_noise_sigma, resolve_sigma

__all__ = ["nl_bayes_denoise", "nl_bayes_parameters", "resolve_parameters"]


def nl_bayes_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    data_range: tuple[float, float] | None = None,
    use_area1: bool = True,
    use_area2: bool = False,
    return_basic: bool = False,
    color_mode: str = "yuv",
    engine: str = "vectorized",
    step: int = 3,
    chunk: int = 256,
    n_threads: int | None = None,
    **overrides,
) -> np.ndarray:
    """
    Débruite une image par NL-Bayes (bibliothèque nlbayes-ext).

    Parameters
    ----------
    image : np.ndarray
        Image bruitée : 2D (niveaux de gris) ou (H, W, 3) (couleur RGB).
    sigma : float, optional
        Écart-type du bruit (unités de l'image), identique sur tous les
        canaux. Estimé si absent (moyenne des canaux en couleur).
    data_range : tuple(float, float), optional
        Dynamique ``(min, max)`` ramenée à [0, 255] pour appliquer les
        paramètres de l'article. Par défaut, le min/max de l'image.
    use_area1, use_area2 : bool
        Critère des zones homogènes à la 1re / 2e étape (article : oui / non).
    return_basic : bool
        Renvoyer l'estimée de l'étape 1.
    color_mode : {"yuv", "rgb", "channel"}
        Traitement d'une image couleur (ignoré en niveaux de gris) : voir
        NL_BAYES.md §8.
    engine : {"vectorized", "loop"}
        Moteur par lots (défaut) ou fidèle à l'article (« paste trick »,
        niveaux de gris ou ``color_mode="channel"``).
    step, chunk, n_threads :
        Pas de la grille des références, taille des lots, fils d'exécution.
    **overrides
        Paramètres de l'article à remplacer : ``k1``, ``k2``, ``n1``, ``n2``,
        ``N1``, ``N2``, ``gamma``, ``beta1``, ``beta2``, ``tau0``,
        ``window_factor``.

    Returns
    -------
    np.ndarray
        Image débruitée, ``float64``, même forme que l'entrée.
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    common = dict(data_range=data_range, use_area1=use_area1, use_area2=use_area2, return_basic=return_basic,
                  engine=engine, step=step, chunk=chunk, n_threads=n_threads, **overrides)
    if arr.ndim == 2:
        return denoise_gray(arr, sigma=resolve_sigma(arr, sigma), **common)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        if not np.all(np.isfinite(arr)):
            raise ValueError("L'image contient des valeurs non finies (NaN/inf).")
        s = float(np.mean([estimate_noise_sigma(arr[..., c]) for c in range(3)])) if sigma is None else sigma
        return denoise_rgb(arr, sigma=resolve_sigma(arr[..., 0], s), color_mode=color_mode, **common)
    raise ValueError(f"Image 2D ou (H, W, 3) attendue, reçu {arr.shape}.")
