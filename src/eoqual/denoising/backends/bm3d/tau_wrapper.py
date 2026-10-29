#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BM3D — filtrage collaboratif par appariement de blocs et transformée 3D
(Dabov et al. ; implémentation de référence de Tampere University).

Fine couche autour du paquet PyPI ``bm3d`` (Mäkinen, Azzari & Foi —
Tampere University). **Licence non libre : usage non commercial
uniquement**, voir ``THIRD_PARTY_LICENSES.md`` §2.2 avant d'installer.
Aucun code de ``bm3d`` n'est embarqué dans ce paquet.

Extra requis : ``pip install "eoqual-denoising[bm3d]"``.

Références :
    Dabov, K., Foi, A., Katkovnik, V., Egiazarian, K. (2007). "Image
    Denoising by Sparse 3-D Transform-Domain Collaborative Filtering."
    IEEE Trans. Image Process. 16(8), 2080-2095.
    Mäkinen, Y., Azzari, L., Foi, A. (2020). "Collaborative Filtering of
    Correlated Noise: Exact Transform-Domain Variance for Improved
    Shrinkage and Patch Matching." IEEE Trans. Image Process. 29,
    8339-8354.
"""
from __future__ import annotations

import numpy as np

from ...utils import check_image, resolve_sigma

__all__ = ["bm3d_denoise"]


def bm3d_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    profile: str = "np",
    hard_threshold_only: bool = False,
) -> np.ndarray:
    """
    Débruite une image par BM3D (bruit blanc gaussien).

    Parameters
    ----------
    image : np.ndarray
        Image 2D bruitée.
    sigma : float, optional
        Écart-type du bruit (unités de l'image). Estimé si absent.
    profile : str
        Profil de paramètres du paquet ``bm3d`` (``"np"`` normal,
        ``"refilter"``, ``"vn"``, ``"high"``, ``"deb"``...). Par défaut
        ``"np"``.
    hard_threshold_only : bool
        N'exécuter que la première étape (seuillage dur), sans le filtrage
        de Wiener. Par défaut ``False`` (les deux étapes).

    Returns
    -------
    np.ndarray
        Image débruitée, ``float64``, même forme que l'entrée.

    Raises
    ------
    ImportError
        Si le paquet ``bm3d`` n'est pas installé — lire l'avertissement de
        licence (non commercial) dans ``THIRD_PARTY_LICENSES.md`` avant.
    """
    try:
        import bm3d
    except ImportError as exc:
        raise ImportError(
            "bm3d_denoise nécessite le paquet 'bm3d' (Tampere University), "
            "sous licence NON COMMERCIALE — lire THIRD_PARTY_LICENSES.md "
            'avant installation. Installer avec : pip install "eoqual-denoising[bm3d]"'
        ) from exc

    img = check_image(image)
    s = resolve_sigma(img, sigma)
    stage = bm3d.BM3DStages.HARD_THRESHOLDING if hard_threshold_only else bm3d.BM3DStages.ALL_STAGES
    return np.asarray(bm3d.bm3d(img, sigma_psd=s, profile=profile, stage_arg=stage), dtype=np.float64)
