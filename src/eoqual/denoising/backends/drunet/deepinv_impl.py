#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRUNet — réseau débruiteur U-Net à blocs résiduels, conditionné par le
niveau de bruit (Zhang et al.).

Fine couche autour de ``deepinv.models.DRUNet`` (DeepInverse,
BSD-3-Clause). Poids pré-entraînés niveaux de gris téléchargés par
deepinv au premier appel depuis https://huggingface.co/deepinv/drunet
(BSD-3-Clause). Aucun code ni poids embarqué dans ce paquet.

Extra requis : ``pip install "eoqual-denoising[deep]"`` (torch, deepinv).

Référence :
    Zhang, K., Li, Y., Zuo, W., Zhang, L., Van Gool, L., Timofte, R.
    (2022). "Plug-and-Play Image Restoration with Deep Denoiser Prior."
    IEEE Trans. Pattern Anal. Mach. Intell. 44(10), 6360-6376.
"""
from __future__ import annotations

import numpy as np

from ...utils import check_image, from_unit_range, resolve_sigma, to_unit_range
from .._torch import cached_model, import_deepinv, pick_device, run_on_unit_image

__all__ = ["drunet_denoise"]


def drunet_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    data_range: tuple[float, float] | None = None,
    device: str | None = None,
    pretrained: str = "download",
) -> np.ndarray:
    """
    Débruite une image par le réseau DRUNet pré-entraîné (niveaux de gris).

    Parameters
    ----------
    image : np.ndarray
        Image 2D bruitée.
    sigma : float, optional
        Écart-type du bruit (unités de l'image). Estimé si absent.
    data_range : tuple(float, float), optional
        Dynamique ``(min, max)`` du capteur, utilisée pour ramener l'image
        dans [0, 1] (domaine d'entraînement du réseau). Par défaut, le
        min/max de l'image.
    device : str, optional
        ``"cpu"``, ``"cuda"``, ``"mps"``... Par défaut ``"cuda"`` si
        disponible, sinon ``"cpu"``.
    pretrained : str
        ``"download"`` (poids deepinv) ou chemin vers un fichier de poids
        ``.pth`` compatible. Par défaut ``"download"``.

    Returns
    -------
    np.ndarray
        Image débruitée, ``float64``, même forme que l'entrée.

    Raises
    ------
    ImportError
        Si l'extra ``deep`` n'est pas installé.
    """
    _, dm = import_deepinv()
    img = check_image(image)
    s = resolve_sigma(img, sigma)
    device = pick_device(device)
    unit, lo, scale = to_unit_range(img, data_range)
    model = cached_model(
        ("drunet", device, pretrained),
        lambda: dm.DRUNet(in_channels=1, out_channels=1, pretrained=pretrained, device=device),
    )
    out = run_on_unit_image(lambda x: model(x, sigma=s / scale), unit, device)
    return from_unit_range(out, lo, scale)
