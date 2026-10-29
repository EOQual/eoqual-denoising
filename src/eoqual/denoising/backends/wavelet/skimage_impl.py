#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Débruitage par seuillage des coefficients d'ondelettes (BayesShrink /
VisuShrink).

Fine couche autour de ``skimage.restoration.denoise_wavelet``
(scikit-image, BSD-3-Clause ; transformée PyWavelets, MIT). Aucun code
tiers repris.

Références :
    Chang, S. G., Yu, B., Vetterli, M. (2000). "Adaptive wavelet
    thresholding for image denoising and compression." IEEE Trans. Image
    Process. 9(9), 1532-1546. (BayesShrink)
    Donoho, D. L., Johnstone, I. M. (1994). "Ideal spatial adaptation by
    wavelet shrinkage." Biometrika 81(3), 425-455. (VisuShrink)
"""
from __future__ import annotations

import numpy as np

from ...utils import check_image, resolve_sigma

__all__ = ["wavelet_denoise"]


def wavelet_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    wavelet: str = "db2",
    rule: str = "BayesShrink",
    mode: str = "soft",
    wavelet_levels: int | None = None,
) -> np.ndarray:
    """
    Débruite une image par seuillage de ses coefficients d'ondelettes.

    Parameters
    ----------
    image : np.ndarray
        Image 2D bruitée.
    sigma : float, optional
        Écart-type du bruit (unités de l'image). Estimé si absent.
    wavelet : str
        Ondelette PyWavelets. Par défaut ``"db2"``.
    rule : {"BayesShrink", "VisuShrink"}
        Règle de seuil : adaptative par sous-bande (BayesShrink) ou
        universelle (VisuShrink, plus lissante). Par défaut ``"BayesShrink"``.
    mode : {"soft", "hard"}
        Type de seuillage. Par défaut ``"soft"``.
    wavelet_levels : int, optional
        Nombre de niveaux de décomposition. Par défaut, le maximum
        raisonnable calculé par scikit-image (niveau max - 3).

    Returns
    -------
    np.ndarray
        Image débruitée, ``float64``, même forme que l'entrée.
    """
    from skimage.restoration import denoise_wavelet

    img = check_image(image)
    s = resolve_sigma(img, sigma)
    # denoise_wavelet suppose une image dans [0, 1] (ou [-1, 1]) : on
    # travaille sur une copie centrée-réduite pour rester indépendant des
    # unités, puis on revient à l'échelle d'origine.
    lo, hi = float(img.min()), float(img.max())
    scale = hi - lo if hi > lo else 1.0
    out = denoise_wavelet(
        (img - lo) / scale,
        sigma=s / scale,
        wavelet=wavelet,
        mode=mode,
        wavelet_levels=wavelet_levels,
        method=rule,
        rescale_sigma=True,
        channel_axis=None,
    )
    return out * scale + lo
