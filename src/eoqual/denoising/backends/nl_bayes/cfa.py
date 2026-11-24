#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NL-Bayes CFA — méthode ``nl_bayes_cfa`` : débruitage et démosaïçage
conjoints d'une mosaïque de Bayer (H, W) → image RGB (H, W, 3).

Le calcul est fait par la bibliothèque **nlbayes-ext** (MIT,
https://github.com/oam31/nlbayes-ext, fonction ``denoise_cfa``) ; ce module
adapte l'appel au contrat d'eoqual-denoising (σ estimé sur les quatre
sous-plans du motif par :func:`~eoqual.denoising.utils.estimate_noise_sigma`
s'il est absent). Méthode et études : docs/technique/nl-bayes.md §13.
"""
from __future__ import annotations

import numpy as np

from nlbayes_ext import CFA_PATTERNS, bilinear_demosaic, cfa_channels, channel_map, denoise_cfa  # noqa: F401

from ...utils import check_image, estimate_noise_sigma

__all__ = ["nl_bayes_cfa_denoise", "bilinear_demosaic", "cfa_channels", "channel_map", "CFA_PATTERNS"]

#: Ancien nom, gardé pour les exemples.
_channel_map = channel_map


def _estimate_mosaic_sigma(m: np.ndarray) -> float:
    """σ estimé sur les quatre sous-plans du motif (chacun est une image ordinaire)."""
    subs = [m[i::2, j::2] for i in (0, 1) for j in (0, 1)]
    return float(np.mean([estimate_noise_sigma(s) for s in subs if min(s.shape) >= 8]))


def nl_bayes_cfa_denoise(image: np.ndarray, sigma: float | None = None, **kwargs) -> np.ndarray:
    """
    Débruite et démosaïque une mosaïque de Bayer (``nlbayes_ext.denoise_cfa``).

    Parameters
    ----------
    image : np.ndarray
        Mosaïque bruitée (H, W).
    sigma : float, optional
        Écart-type du bruit de la mosaïque (unités de l'image) ; 1 après
        stabilisation de variance (:func:`~eoqual.denoising.vst_denoise`).
        Estimé sur les quatre sous-plans si absent.
    **kwargs
        Paramètres de ``denoise_cfa`` : ``pattern`` (``"RGGB"``…),
        ``data_range``, ``k`` (3), ``window_factor`` (7), ``N`` (60),
        ``beta`` (1), ``tau0`` (4), ``n_iter`` (8), ``pilot``,
        ``pilot_kwargs``, ``use_area``, ``area_z``, ``area_gamma``,
        ``area_mean``, ``step``, ``chunk``, ``n_threads``.

    Returns
    -------
    np.ndarray
        Image RGB (H, W, 3), ``float64``, unités de la mosaïque.
    """
    if "noise" in kwargs:
        raise ValueError("Pour un modèle de bruit (a, b), passer par vst_denoise (gain, read_noise_dn).")
    mosaic = check_image(image)
    s = _estimate_mosaic_sigma(mosaic) if sigma is None else float(sigma)
    return denoise_cfa(mosaic, sigma=s, **kwargs)
