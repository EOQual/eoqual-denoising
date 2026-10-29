#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-Local Means (Buades, Coll & Morel).

Fine couche autour de ``skimage.restoration.denoise_nl_means``
(scikit-image, BSD-3-Clause). Aucun code tiers repris.

Références :
    Buades, A., Coll, B., Morel, J.-M. (2005). "A non-local algorithm for
    image denoising." Proc. IEEE CVPR, vol. 2, 60-65.
    Darbon, J., Cunha, A., Chan, T. F., Osher, S., Jensen, G. J. (2008).
    "Fast nonlocal filtering applied to electron cryomicroscopy." Proc.
    IEEE ISBI, 1331-1334. (variante ``fast_mode=True`` de scikit-image)
"""
from __future__ import annotations

import numpy as np

from ...utils import check_image, resolve_sigma

__all__ = ["nl_means_denoise"]


def nl_means_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    h: float | None = None,
    patch_size: int = 7,
    patch_distance: int = 11,
    fast_mode: bool = True,
) -> np.ndarray:
    """
    Débruite une image par Non-Local Means.

    Parameters
    ----------
    image : np.ndarray
        Image 2D bruitée.
    sigma : float, optional
        Écart-type du bruit (unités de l'image). Estimé si absent.
    h : float, optional
        Paramètre de filtrage (décroissance des poids). Par défaut
        ``0.6 * sigma`` en mode rapide, ``0.8 * sigma`` sinon
        (recommandation de la documentation scikit-image).
    patch_size : int
        Taille des patchs, en pixels. Par défaut ``7``.
    patch_distance : int
        Demi-taille de la fenêtre de recherche, en pixels. Par défaut ``11``.
    fast_mode : bool
        Algorithme rapide (Darbon et al., 2008). Par défaut ``True``.

    Returns
    -------
    np.ndarray
        Image débruitée, ``float64``, même forme que l'entrée.
    """
    from skimage.restoration import denoise_nl_means

    img = check_image(image)
    s = resolve_sigma(img, sigma)
    if h is None:
        h = (0.6 if fast_mode else 0.8) * s
    return denoise_nl_means(
        img,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h,
        fast_mode=fast_mode,
        sigma=s,
        preserve_range=True,
        channel_axis=None,
    )
