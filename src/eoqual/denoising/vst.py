#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stabilisation de variance du bruit poisson-gaussien : transformée
d'Anscombe généralisée (GAT) et ses inverses, dont l'**inverse exacte non
biaisée**.

Modèle de capteur (unités : DN pour l'image, e⁻ pour les photo-électrons) ::

    z = p / K + offset + n,   p ~ Poisson(λ),   n ~ N(0, σ_DN²)

où ``K`` est le gain de conversion (e⁻/DN), ``σ_DN`` le bruit de lecture
(DN) et ``offset`` le niveau d'obscurité (DN). Ce sont exactement les
grandeurs fournies par la PTC d'eoqual-measure
(``measure_ptc(...).gain``, ``.read_noise_dn``) et par la mesure
d'obscurité (niveau moyen).

En électrons, y = K·(z − offset) = p + n', n' ~ N(0, σ²) avec σ = K·σ_DN.
La GAT ::

    f(y) = 2·sqrt(y + 3/8 + σ²)   (0 si l'argument est négatif)

rend la variance du bruit approximativement égale à 1, quel que soit le
signal : on peut alors appliquer n'importe quel débruiteur gaussien avec
``sigma = 1``, puis revenir au domaine d'origine.

Le retour est le point délicat. L'inverse algébrique ``(D/2)² − 3/8 − σ²``
est biaisée, car E[f(y)] ≠ f(E[y]). L'**inverse exacte non biaisée**
inverse l'application λ ↦ E[f(y) | λ], calculée numériquement (somme
sur la loi de Poisson, intégrale sur la loi gaussienne). Elle seule reste
juste à faible flux (quelques e⁻ par pixel). Au-delà de ``LAMBDA_MAX``
e⁻, elle se confond avec l'inverse asymptotique ``(D/2)² − 1/8 − σ²``,
utilisée à sa place.

Implémentation originale écrite pour ce projet depuis les publications
(aucun code tiers consulté ni repris) ; numpy et scipy seulement, pour
pouvoir rejoindre eoqual-core sans modification.

Références :
    Anscombe, F. J. (1948). "The transformation of Poisson, binomial and
    negative-binomial data." Biometrika 35(3/4), 246-254.
    Murtagh, F., Starck, J.-L., Bijaoui, A. (1995). "Image restoration
    with noise suppression using a multiresolution support." Astron.
    Astrophys. Suppl. Ser. 112, 179-189. (GAT)
    Mäkitalo, M., Foi, A. (2011). "Optimal inversion of the Anscombe
    transformation in low-count Poisson image denoising." IEEE Trans.
    Image Process. 20(1), 99-109.
    Mäkitalo, M., Foi, A. (2013). "Optimal inversion of the generalized
    Anscombe transformation for Poisson-Gaussian noise." IEEE Trans.
    Image Process. 22(1), 91-103.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = [
    "generalized_anscombe",
    "inverse_generalized_anscombe",
    "expected_gat",
    "add_poisson_gaussian_noise",
    "vst_denoise",
    "INVERSES",
    "LAMBDA_MAX",
]

#: Inverses disponibles pour :func:`inverse_generalized_anscombe`.
INVERSES = ("exact_unbiased", "asymptotic", "algebraic")

#: Flux (e⁻) au-delà duquel l'inverse exacte est remplacée par l'inverse
#: asymptotique (écart relatif < 1e-5 à ce niveau, voir tests).
LAMBDA_MAX = 400.0


def _to_electrons(z, gain, read_noise_dn, offset_dn):
    if gain <= 0:
        raise ValueError(f"gain doit être > 0 (e⁻/DN), reçu {gain}.")
    if read_noise_dn < 0:
        raise ValueError(f"read_noise_dn doit être ≥ 0, reçu {read_noise_dn}.")
    y = gain * (np.asarray(z, dtype=np.float64) - offset_dn)
    return y, gain * read_noise_dn


def _gat_electrons(y, sigma_e):
    return 2.0 * np.sqrt(np.maximum(y + 0.375 + sigma_e**2, 0.0))


def generalized_anscombe(
    z: np.ndarray, gain: float = 1.0, read_noise_dn: float = 0.0, offset_dn: float = 0.0
) -> np.ndarray:
    """
    Transformée d'Anscombe généralisée : ramène un bruit poisson-gaussien
    à un bruit de variance ≈ 1.

    Avec les valeurs par défaut (``gain=1``, ``read_noise_dn=0``,
    ``offset_dn=0``), c'est la transformée d'Anscombe classique d'une image
    de comptages poissoniens : 2·sqrt(z + 3/8).

    Parameters
    ----------
    z : np.ndarray
        Image brute, en DN.
    gain : float
        Gain de conversion K, en e⁻/DN.
    read_noise_dn : float
        Écart-type du bruit gaussien (lecture), en DN.
    offset_dn : float
        Niveau d'obscurité, en DN.

    Returns
    -------
    np.ndarray
        Image transformée (sans unité), bruit de variance ≈ 1.
    """
    y, sigma_e = _to_electrons(z, gain, read_noise_dn, offset_dn)
    return _gat_electrons(y, sigma_e)


def _gaussian_expectation(k: np.ndarray, sigma_e: float) -> np.ndarray:
    """E[f(k + n)], n ~ N(0, σ²), pour des comptages entiers k (trapèzes)."""
    if sigma_e == 0:
        return _gat_electrons(k.astype(np.float64), 0.0)
    t = np.linspace(-10.0, 10.0, 4001)
    w = np.exp(-0.5 * t**2)
    w /= w.sum()
    vals = _gat_electrons(k[:, None] + sigma_e * t[None, :], sigma_e)
    return vals @ w


@lru_cache(maxsize=32)
def _exact_table(sigma_e: float) -> tuple[np.ndarray, np.ndarray]:
    """Table (λ, E[f(y) | λ]) pour λ ∈ [0, LAMBDA_MAX]."""
    from scipy.stats import poisson

    lam = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 2.0, 201),
                np.linspace(2.0, 20.0, 361),
                np.geomspace(20.0, LAMBDA_MAX, 400),
            ]
        )
    )
    k_max = int(np.ceil(LAMBDA_MAX + 12.0 * np.sqrt(LAMBDA_MAX) + 30.0))
    k = np.arange(k_max + 1)
    g = _gaussian_expectation(k, sigma_e)
    pmf = poisson.pmf(k[None, :], lam[:, None])
    expected = pmf @ g
    # Monotonie stricte garantie pour l'interpolation inverse.
    expected = np.maximum.accumulate(expected)
    return lam, expected


def expected_gat(lam: np.ndarray, gain: float = 1.0, read_noise_dn: float = 0.0) -> np.ndarray:
    """
    Espérance E[f(y) | λ] de la GAT pour un flux moyen ``λ`` (e⁻) — la
    courbe que l'inverse exacte non biaisée inverse.

    Parameters
    ----------
    lam : np.ndarray
        Flux moyen, en photo-électrons (≥ 0).
    gain, read_noise_dn : float
        Voir :func:`generalized_anscombe`.

    Returns
    -------
    np.ndarray
        Espérance de la variable transformée.
    """
    sigma_e = gain * read_noise_dn
    lam = np.asarray(lam, dtype=np.float64)
    grid, expected = _exact_table(round(float(sigma_e), 12))
    asym = 2.0 * np.sqrt(np.maximum(lam + 0.125 + sigma_e**2, 0.0))
    return np.where(lam <= LAMBDA_MAX, np.interp(lam, grid, expected), asym)


def inverse_generalized_anscombe(
    d: np.ndarray,
    gain: float = 1.0,
    read_noise_dn: float = 0.0,
    offset_dn: float = 0.0,
    method: str = "exact_unbiased",
) -> np.ndarray:
    """
    Inverse de la GAT, appliquée à une image transformée **débruitée**.

    Parameters
    ----------
    d : np.ndarray
        Image dans le domaine transformé (typiquement sortie d'un
        débruiteur).
    gain, read_noise_dn, offset_dn : float
        Mêmes paramètres que pour la transformée directe.
    method : {"exact_unbiased", "asymptotic", "algebraic"}
        - ``"exact_unbiased"`` (défaut) : inverse de λ ↦ E[f(y) | λ] ;
          sans biais à tout flux. Recommandée.
        - ``"asymptotic"`` : (D/2)² − 1/8 − σ² ; juste à fort flux
          seulement (≳ 20 e⁻).
        - ``"algebraic"`` : (D/2)² − 3/8 − σ² ; inverse exacte de f mais
          biaisée pour une image débruitée. Fournie pour comparaison.

    Returns
    -------
    np.ndarray
        Estimation du signal moyen, en DN.
    """
    if method not in INVERSES:
        raise ValueError(f"method doit valoir {INVERSES}, reçu '{method}'.")
    if gain <= 0:
        raise ValueError(f"gain doit être > 0 (e⁻/DN), reçu {gain}.")
    sigma_e = gain * read_noise_dn
    d = np.asarray(d, dtype=np.float64)
    half_sq = (d / 2.0) ** 2

    if method == "algebraic":
        lam = half_sq - 0.375 - sigma_e**2
    elif method == "asymptotic":
        lam = half_sq - 0.125 - sigma_e**2
    else:
        grid, expected = _exact_table(round(float(sigma_e), 12))
        lam = np.where(
            d <= expected[-1],
            np.interp(d, expected, grid, left=0.0),
            half_sq - 0.125 - sigma_e**2,
        )
        lam = np.maximum(lam, 0.0)
    return lam / gain + offset_dn


def add_poisson_gaussian_noise(
    signal_e: np.ndarray,
    gain: float = 1.0,
    read_noise_dn: float = 0.0,
    offset_dn: float = 0.0,
    seed: int | None = 0,
) -> np.ndarray:
    """
    Simule une image brute selon le modèle poisson-gaussien du module.

    Parameters
    ----------
    signal_e : np.ndarray
        Flux moyen par pixel, en photo-électrons (≥ 0).
    gain, read_noise_dn, offset_dn : float
        Voir :func:`generalized_anscombe`.
    seed : int, optional
        Graine du générateur. Par défaut ``0``.

    Returns
    -------
    np.ndarray
        Image en DN, ``float64``, non quantifiée.
    """
    rng = np.random.default_rng(seed)
    lam = np.clip(np.asarray(signal_e, dtype=np.float64), 0.0, None)
    p = rng.poisson(lam)
    return p / gain + offset_dn + rng.normal(0.0, read_noise_dn, size=lam.shape)


def vst_denoise(
    image: np.ndarray,
    method: str,
    gain: float,
    read_noise_dn: float = 0.0,
    offset_dn: float = 0.0,
    inverse: str = "exact_unbiased",
    **kwargs,
) -> np.ndarray:
    """
    Débruite une image affectée d'un bruit poisson-gaussien :
    GAT → débruiteur gaussien (``sigma = 1``) → inverse.

    Parameters
    ----------
    image : np.ndarray
        Image brute 2D, en DN.
    method : str
        Nom d'une méthode du registre (voir ``list_methods()``).
    gain : float
        Gain de conversion K, en e⁻/DN (PTC).
    read_noise_dn : float
        Bruit de lecture, en DN (PTC).
    offset_dn : float
        Niveau d'obscurité, en DN.
    inverse : {"exact_unbiased", "asymptotic", "algebraic"}
        Inverse utilisée (voir :func:`inverse_generalized_anscombe`).
    **kwargs
        Transmis au débruiteur (``sigma`` est imposé à 1 et ne doit pas
        être fourni).

    Returns
    -------
    np.ndarray
        Image débruitée, en DN.
    """
    from .runner import denoise
    from .utils import check_image

    if "sigma" in kwargs:
        raise ValueError("sigma est imposé à 1 dans le domaine transformé : ne pas le fournir.")
    img = check_image(image)
    d = generalized_anscombe(img, gain, read_noise_dn, offset_dn)
    d_hat = denoise(d, method=method, sigma=1.0, **kwargs)
    return inverse_generalized_anscombe(d_hat, gain, read_noise_dn, offset_dn, method=inverse)
