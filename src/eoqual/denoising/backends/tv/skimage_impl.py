#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Débruitage par variation totale (modèle ROF), deux solveurs.

Fines couches autour de ``skimage.restoration.denoise_tv_chambolle`` et
``skimage.restoration.denoise_tv_bregman`` (scikit-image, BSD-3-Clause).
Aucun code tiers repris.

Références :
    Rudin, L. I., Osher, S., Fatemi, E. (1992). "Nonlinear total variation
    based noise removal algorithms." Physica D 60(1-4), 259-268.
    Chambolle, A. (2004). "An algorithm for total variation minimization
    and applications." J. Math. Imaging Vision 20, 89-97.
    Goldstein, T., Osher, S. (2009). "The split Bregman method for
    L1-regularized problems." SIAM J. Imaging Sci. 2(2), 323-343.
"""
from __future__ import annotations

import numpy as np

from ...utils import check_image, resolve_sigma

__all__ = ["tv_chambolle_denoise", "tv_bregman_denoise"]


def tv_chambolle_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    weight: float | None = None,
    eps: float = 2e-4,
    max_num_iter: int = 200,
) -> np.ndarray:
    """
    Débruite une image par variation totale, solveur dual de Chambolle.

    Parameters
    ----------
    image : np.ndarray
        Image 2D bruitée.
    sigma : float, optional
        Écart-type du bruit (unités de l'image). Estimé si absent. Sert
        uniquement à fixer ``weight`` par défaut.
    weight : float, optional
        Poids de la régularisation TV, dans les unités de l'image (plus
        grand = plus lisse). Par défaut ``0.75 * sigma`` (calibré
        empiriquement, voir REFERENCE_TECHNIQUE.md §2.3).
    eps : float
        Critère d'arrêt relatif. Par défaut ``2e-4``.
    max_num_iter : int
        Nombre maximal d'itérations. Par défaut ``200``.

    Returns
    -------
    np.ndarray
        Image débruitée, ``float64``, même forme que l'entrée.
    """
    from skimage.restoration import denoise_tv_chambolle

    img = check_image(image)
    if weight is None:
        weight = 0.75 * resolve_sigma(img, sigma)
    return denoise_tv_chambolle(img, weight=weight, eps=eps, max_num_iter=max_num_iter, channel_axis=None)


def tv_bregman_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    weight: float | None = None,
    max_num_iter: int = 100,
    eps: float = 1e-3,
    isotropic: bool = True,
) -> np.ndarray:
    """
    Débruite une image par variation totale, solveur split Bregman.

    Parameters
    ----------
    image : np.ndarray
        Image 2D bruitée.
    sigma : float, optional
        Écart-type du bruit (unités de l'image). Estimé si absent. Sert
        uniquement à fixer ``weight`` par défaut.
    weight : float, optional
        Poids de l'**attache aux données** (plus grand = moins lisse —
        sens inverse de ``tv_chambolle_denoise``), en inverse des unités
        de l'image. Par défaut ``1 / sigma`` (calibré empiriquement, voir
        REFERENCE_TECHNIQUE.md §2.4).
    max_num_iter : int
        Nombre maximal d'itérations. Par défaut ``100``.
    eps : float
        Critère d'arrêt. Par défaut ``1e-3``.
    isotropic : bool
        TV isotrope (``True``) ou anisotrope. Par défaut ``True``.

    Returns
    -------
    np.ndarray
        Image débruitée, ``float64``, même forme que l'entrée.
    """
    from skimage.restoration import denoise_tv_bregman

    img = check_image(image)
    if weight is None:
        s = resolve_sigma(img, sigma)
        weight = 1.0 / s if s > 0 else 1e6
    return denoise_tv_bregman(
        img, weight=weight, max_num_iter=max_num_iter, eps=eps, isotropic=isotropic, channel_axis=None
    )
