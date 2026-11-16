#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Débruiteurs appris (DRUNet, SwinIR, Restormer, GS-DRUNet), fournis par
**eoqual-neural** (extra ``neural``).

Depuis la version 0.3.0, les réseaux de neurones d'EOQual sont regroupés
dans eoqual-neural, rangés par rôle. Ce module n'en garde que l'adaptation
au contrat d'eoqual-denoising : image dans les unités du capteur, σ dans
ces unités (estimé s'il est absent), ramenés dans [0, 1] via
``data_range`` (par défaut le min/max de l'image) avant l'appel du réseau.

Extra requis : ``pip install "eoqual-denoising[neural]"``.
"""
from __future__ import annotations

import numpy as np

from ..utils import check_image, from_unit_range, resolve_sigma, to_unit_range

__all__ = ["drunet_denoise", "swinir_denoise", "restormer_denoise", "gsdrunet_denoise"]


def _neural(name: str):
    try:
        from eoqual.neural import get_denoiser
    except ImportError as exc:
        raise ImportError(
            f"La méthode '{name}' est un réseau de neurones fourni par eoqual-neural. "
            'Installer avec : pip install "eoqual-denoising[neural]"'
        ) from exc
    return get_denoiser(name)


def _run(name, image, sigma, data_range, estimate_sigma=True, **kwargs):
    img = check_image(image)
    unit, lo, scale = to_unit_range(img, data_range)
    s_unit = resolve_sigma(img, sigma) / scale if estimate_sigma else None
    return from_unit_range(_neural(name)(unit, s_unit, **kwargs), lo, scale)


def drunet_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    data_range: tuple[float, float] | None = None,
    device: str | None = None,
    pretrained: str = "download",
) -> np.ndarray:
    """
    DRUNet pré-entraîné (niveaux de gris), **conditionné par σ** : un seul
    réseau pour toute la plage de bruit (``eoqual.neural.denoisers.drunet``).

    Parameters
    ----------
    image : np.ndarray
        Image 2D bruitée.
    sigma : float, optional
        Écart-type du bruit (unités de l'image). Estimé si absent.
    data_range : tuple(float, float), optional
        Dynamique ``(min, max)`` ramenée à [0, 1] (domaine d'entraînement du
        réseau). Par défaut, le min/max de l'image.
    device : str, optional
        ``"cpu"``, ``"cuda"``... Par défaut ``"cuda"`` si disponible.
    pretrained : str
        ``"download"`` (poids deepinv) ou chemin d'un fichier ``.pth`` compatible.

    Raises
    ------
    ImportError
        Si l'extra ``neural`` n'est pas installé.
    """
    return _run("drunet", image, sigma, data_range, device=device, pretrained=pretrained)


def swinir_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    noise_level: int | None = None,
    data_range: tuple[float, float] | None = None,
    device: str | None = None,
) -> np.ndarray:
    """
    SwinIR pré-entraîné : **un jeu de poids par niveau de bruit** (σ = 15,
    25, 50 sur [0, 255]) ; le plus proche de ``sigma`` est choisi, sauf si
    ``noise_level`` est imposé (``eoqual.neural.denoisers.swinir``).
    """
    return _run("swinir", image, sigma, data_range, device=device, noise_level=noise_level)


def restormer_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    data_range: tuple[float, float] | None = None,
    device: str | None = None,
) -> np.ndarray:
    """
    Restormer, poids niveaux de gris **aveugles** (entraînés sur σ ∈ [0, 50]
    en échelle [0, 255]) ; ``sigma`` est ignoré
    (``eoqual.neural.denoisers.restormer``).
    """
    return _run("restormer", image, sigma, data_range, estimate_sigma=False, device=device)


def gsdrunet_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    data_range: tuple[float, float] | None = None,
    device: str | None = None,
) -> np.ndarray:
    """
    GS-DRUNet : pas de gradient d'un potentiel appris, ce qui garantit la
    convergence d'un Plug-and-Play (``eoqual.neural.denoisers.gsdrunet``).
    """
    return _run("gsdrunet", image, sigma, data_range, device=device)
