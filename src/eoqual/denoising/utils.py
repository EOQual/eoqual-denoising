#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitaires communs à toutes les méthodes de débruitage : validation de
l'image d'entrée, estimation de l'écart-type du bruit, normalisation vers
[0, 1] (pour les backends qui l'exigent), génération d'une image de test.

Convention du paquet : ``sigma`` désigne toujours l'écart-type d'un bruit
blanc gaussien additif, **dans les unités de l'image d'entrée** (pas en
fraction de dynamique). Chaque backend fait lui-même la conversion vers
l'échelle attendue par la bibliothèque appelée.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "check_image",
    "estimate_noise_sigma",
    "resolve_sigma",
    "to_unit_range",
    "from_unit_range",
    "add_gaussian_noise",
    "astronaut_gray",
    "astronaut_rgb",
]


def check_image(image: np.ndarray) -> np.ndarray:
    """
    Valide une image d'entrée et la renvoie en ``float64``.

    Parameters
    ----------
    image : np.ndarray
        Image 2D (niveaux de gris).

    Returns
    -------
    np.ndarray
        Copie ``float64`` de l'image.

    Raises
    ------
    ValueError
        Si l'image n'est pas 2D ou contient des valeurs non finies.
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"Image 2D attendue (niveaux de gris), reçu ndim={arr.ndim}. "
            "Appliquer la méthode bande par bande pour une image multi-bandes."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("L'image contient des valeurs non finies (NaN/inf).")
    return arr


def estimate_noise_sigma(image: np.ndarray) -> float:
    """
    Estime l'écart-type d'un bruit blanc gaussien additif.

    Estimateur robuste de Donoho & Johnstone (1994) : médiane des valeurs
    absolues des coefficients d'ondelettes de la sous-bande diagonale la
    plus fine, divisée par 0.6745 (via
    ``skimage.restoration.estimate_sigma``).

    Parameters
    ----------
    image : np.ndarray
        Image 2D bruitée.

    Returns
    -------
    float
        Écart-type estimé, dans les unités de l'image.
    """
    from skimage.restoration import estimate_sigma

    return float(estimate_sigma(check_image(image), channel_axis=None))


def resolve_sigma(image: np.ndarray, sigma: float | None) -> float:
    """Renvoie ``sigma`` s'il est fourni, sinon son estimation sur ``image``."""
    if sigma is None:
        return estimate_noise_sigma(image)
    if sigma < 0:
        raise ValueError(f"sigma doit être positif ou nul, reçu {sigma}.")
    return float(sigma)


def to_unit_range(
    image: np.ndarray, data_range: tuple[float, float] | None = None
) -> tuple[np.ndarray, float, float]:
    """
    Ramène linéairement une image vers [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Image 2D.
    data_range : tuple(float, float), optional
        Bornes ``(min, max)`` de la dynamique du capteur. Par défaut, le
        min/max de l'image elle-même.

    Returns
    -------
    (np.ndarray, float, float)
        Image normalisée, décalage ``lo`` et échelle ``scale`` tels que
        ``image = normalisée * scale + lo``.
    """
    lo, hi = data_range if data_range is not None else (float(image.min()), float(image.max()))
    scale = hi - lo
    if scale <= 0:
        scale = 1.0
    return (image - lo) / scale, lo, scale


def from_unit_range(image: np.ndarray, lo: float, scale: float) -> np.ndarray:
    """Inverse de :func:`to_unit_range`."""
    return image * scale + lo


def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int | None = 0) -> np.ndarray:
    """
    Ajoute un bruit blanc gaussien additif d'écart-type ``sigma``.

    Parameters
    ----------
    image : np.ndarray
        Image propre.
    sigma : float
        Écart-type du bruit, dans les unités de l'image.
    seed : int, optional
        Graine du générateur (reproductibilité). Par défaut ``0``.

    Returns
    -------
    np.ndarray
        Image bruitée, ``float64``, non tronquée.
    """
    rng = np.random.default_rng(seed)
    return np.asarray(image, dtype=np.float64) + rng.normal(0.0, sigma, size=np.shape(image))


def astronaut_gray(size: int | None = None) -> np.ndarray:
    """
    Image de test « astronaut » (NASA, domaine public) en niveaux de gris,
    valeurs dans [0, 1].

    Fournie par ``skimage.data.astronaut`` (fichier embarqué dans
    scikit-image, aucun téléchargement).

    Parameters
    ----------
    size : int, optional
        Si fourni, l'image est réduite à ``size x size`` (anti-repliement
        activé) — utile pour des tests rapides.

    Returns
    -------
    np.ndarray
        Image 2D ``float64`` dans [0, 1].
    """
    from skimage import data
    from skimage.color import rgb2gray
    from skimage.transform import resize

    img = rgb2gray(data.astronaut())
    if size is not None:
        img = resize(img, (size, size), anti_aliasing=True)
    return img.astype(np.float64)


def astronaut_rgb(size: int | None = None) -> np.ndarray:
    """
    Image de test « astronaut » (NASA, domaine public) en couleur RGB,
    valeurs dans [0, 1], forme (H, W, 3).

    Parameters
    ----------
    size : int, optional
        Si fourni, l'image est réduite à ``size x size`` (anti-repliement).
    """
    from skimage import data
    from skimage.transform import resize

    img = data.astronaut().astype(np.float64) / 255.0
    if size is not None:
        img = resize(img, (size, size, 3), anti_aliasing=True)
    return img
