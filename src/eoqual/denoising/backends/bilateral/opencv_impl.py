#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtre bilatéral (Tomasi & Manduchi).

Fine couche autour de ``cv2.bilateralFilter`` (OpenCV, Apache-2.0).
Aucun code tiers repris.

Référence :
    Tomasi, C., Manduchi, R. (1998). "Bilateral filtering for gray and
    color images." Proc. ICCV, 839-846.
"""
from __future__ import annotations

import numpy as np

from ...utils import check_image, resolve_sigma

__all__ = ["bilateral_denoise"]


def bilateral_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    sigma_color: float | None = None,
    sigma_space: float = 1.5,
    diameter: int = -1,
) -> np.ndarray:
    """
    Débruite une image par filtre bilatéral.

    Parameters
    ----------
    image : np.ndarray
        Image 2D bruitée.
    sigma : float, optional
        Écart-type du bruit (unités de l'image). Estimé si absent. Sert
        uniquement à fixer ``sigma_color`` par défaut.
    sigma_color : float, optional
        Écart-type du noyau radiométrique (unités de l'image). Par défaut
        ``2.5 * sigma`` (calibré empiriquement, voir
        REFERENCE_TECHNIQUE.md §2.1).
    sigma_space : float
        Écart-type du noyau spatial, en pixels. Par défaut ``1.5``.
    diameter : int
        Diamètre du voisinage en pixels ; ``-1`` le déduit de
        ``sigma_space`` (comportement OpenCV). Par défaut ``-1``.

    Returns
    -------
    np.ndarray
        Image débruitée, ``float64``, même forme que l'entrée.
    """
    import cv2

    img = check_image(image)
    if sigma_color is None:
        sigma_color = 2.5 * resolve_sigma(img, sigma)
    out = cv2.bilateralFilter(img.astype(np.float32), diameter, float(sigma_color), float(sigma_space))
    return out.astype(np.float64)
