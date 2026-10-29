#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NL-Bayes — débruitage non local bayésien (Buades, Lebrun & Morel),
implémentation originale numpy, niveaux de gris.

**Réimplémentation « clean-room »** écrite pour ce projet à partir du seul
texte de l'article IPOL (Lebrun, Buades & Morel, 2013) : formules (10) à
(15), description des étapes (§3.3, §3.4), paramètres (tableau §4.8) et
adaptation aux niveaux de gris (§6.1.2). Aucun code tiers n'a été
consulté ni repris (ni le code C++ IPOL, sous GPL, ni aucune autre
implémentation). Licence MIT.

Algorithme (deux étapes identiques dans leur structure) :

1. **Étape 1** — pour chaque patch de référence P̃ (k₁×k₁) non encore
   traité, regrouper les N₁ patchs les plus proches (distance
   quadratique normalisée d² = ‖P̃ − Q̃‖²/k₁²) dans une fenêtre n₁×n₁
   centrée sur P̃. Si l'écart-type du groupe (formule 10) est ≤ γ·σ, la
   zone est jugée homogène et tous les patchs du groupe sont remplacés par
   la moyenne du groupe. Sinon, chaque patch est estimé par (formule 12) ::

       Q_basic = P̄ + (C − β₁σ²I) C⁻¹ (Q̃ − P̄)

   où P̄ et C sont la moyenne et la covariance empiriques du groupe
   (formule 11). Agrégation non pondérée de toutes les estimées → u_basic.
2. **Étape 2** — regroupement sur u_basic (fenêtre n₂, patchs k₂) :
   seuil τ = max(τ₀, d_N₂), tous les patchs à distance ≤ τ sont gardés.
   C_basic = covariance des patchs de u_basic du groupe, P̄ = moyenne des
   patchs **bruités** correspondants, puis (formule 15) ::

       Q_final = P̄ + C_basic (C_basic + β₂σ²I)⁻¹ (Q̃ − P̄)

   Agrégation non pondérée → u_final.

Accélération (« paste ») : un patch utilisé dans un groupe n'est plus pris
comme référence (il peut encore être estimé dans d'autres groupes).

Choix d'implémentation non fixés par l'article (documentés dans
REFERENCE_TECHNIQUE.md §2.6) : fenêtre de recherche rabattue à
l'intérieur de l'image près des bords ; covariance de l'étape 2 centrée
sur la moyenne des patchs de u_basic ; critère homogène de l'étape 2
(optionnel) évalué sur le groupe bruité ; matrice singulière → patchs
laissés inchangés (remarque de l'article, §3.3).

Références :
    Buades, A., Lebrun, M., Morel, J.-M. (2013). "A Nonlocal Bayesian
    Image Denoising Algorithm." SIAM J. Imaging Sci. 6(3), 1665-1688.
    Lebrun, M., Buades, A., Morel, J.-M. (2013). "Implementation of the
    'Non-Local Bayes' (NL-Bayes) Image Denoising Algorithm." Image
    Processing On Line 3, 1-42. https://doi.org/10.5201/ipol.2013.16
"""
from __future__ import annotations

import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ...utils import check_image, from_unit_range, resolve_sigma, to_unit_range

__all__ = ["nl_bayes_denoise", "nl_bayes_parameters"]


_N_FOR_K = {3: 30, 5: 60, 7: 90}  # article §4.6


def nl_bayes_parameters(sigma_255: float, color: bool = False) -> dict:
    """
    Paramètres recommandés par l'article, σ exprimé sur l'échelle [0, 255].

    - ``color=True`` : tableau récapitulatif §4.8 (images couleur) ;
    - ``color=False`` (niveaux de gris, ou couleur canal par canal) : même
      tableau, avec des patchs de 5 là où la couleur utilise 3 (§6.1.2).

    Les autres règles sont communes : N = 30 / 60 / 90 pour k = 3 / 5 / 7
    (§4.6), fenêtre n = 7k (§4.5), γ = 1,05, β₁ = 1, β₂ = 1,2 si σ < 50
    sinon 1, τ₀ = 4.

    Returns
    -------
    dict
        ``k1, k2, n1, n2, N1, N2, gamma, beta1, beta2, tau0``.
    """
    if sigma_255 < 20:
        k1, k2 = (3, 3) if color else (5, 5)
    elif sigma_255 < 50:
        k1, k2 = (5, 3) if color else (5, 5)
    elif sigma_255 < 70:
        k1, k2 = 7, 5
    else:
        k1, k2 = 7, 7
    return {
        "k1": k1,
        "k2": k2,
        "n1": 7 * k1,
        "n2": 7 * k2,
        "N1": _N_FOR_K[k1],
        "N2": _N_FOR_K[k2],
        "gamma": 1.05,
        "beta1": 1.0,
        "beta2": 1.2 if sigma_255 < 50 else 1.0,
        "tau0": 4.0,
    }


def resolve_parameters(sigma_255: float, color: bool, overrides: dict) -> dict:
    """
    Paramètres de l'article, puis surcharges de l'utilisateur.

    Règles d'imbrication appliquées quand une taille de patch ``k`` est
    changée sans fixer le reste : N suit k (§4.6) et la fenêtre n = α·k
    (§4.5), avec α = ``window_factor`` (7 par défaut). N est toujours
    porté à au moins k² + 1, sans quoi la covariance du groupe (k²×k²)
    serait singulière.
    """
    params = nl_bayes_parameters(sigma_255, color)
    allowed = set(params) | {"window_factor"}
    unknown = set(overrides) - allowed
    if unknown:
        raise TypeError(f"Paramètres inconnus : {sorted(unknown)} (autorisés : {sorted(allowed)})")
    alpha = float(overrides.get("window_factor", 7.0))
    params.update({key: v for key, v in overrides.items() if key != "window_factor"})
    for step in ("1", "2"):
        k = int(params["k" + step])
        if "k" + step in overrides and "N" + step not in overrides:
            params["N" + step] = _N_FOR_K.get(k, 2 * k * k + 10)
        if ("k" + step in overrides or "window_factor" in overrides) and "n" + step not in overrides:
            params["n" + step] = int(round(alpha * k))
        params["N" + step] = max(int(params["N" + step]), k * k + 1)
    for key in ("k1", "k2", "n1", "n2", "N1", "N2"):
        params[key] = int(params[key])
    return params


def _patches(img: np.ndarray, k: int) -> np.ndarray:
    """Tous les patchs k×k (coin supérieur gauche), forme (H−k+1, W−k+1, k²)."""
    view = sliding_window_view(img, (k, k))
    return np.ascontiguousarray(view.reshape(view.shape[0], view.shape[1], k * k))


def _window(i: int, j: int, n: int, ph: int, pw: int) -> tuple[slice, slice]:
    """Fenêtre n×n de positions de patchs centrée sur (i, j), rabattue dans l'image."""
    h = n // 2
    i0 = min(max(i - h, 0), max(ph - n, 0))
    j0 = min(max(j - h, 0), max(pw - n, 0))
    return slice(i0, min(i0 + n, ph)), slice(j0, min(j0 + n, pw))


def _fold(acc_p: np.ndarray, cnt_p: np.ndarray, k: int, shape) -> tuple[np.ndarray, np.ndarray]:
    """
    Replie les tampons indexés par position de patch (somme des estimées de
    chaque patch, nombre d'estimées) en tampons image numérateur /
    dénominateur (agrégation non pondérée de l'article).
    """
    h, w = shape
    ph, pw = acc_p.shape[:2]
    num = np.zeros((h, w))
    den = np.zeros((h, w))
    for di in range(k):
        for dj in range(k):
            num[di : di + ph, dj : dj + pw] += acc_p[:, :, di * k + dj]
            den[di : di + ph, dj : dj + pw] += cnt_p
    return num, den


def _step(noisy, guide, sigma, k, n, n_keep, beta, gamma, use_area, tau0, second):
    """Une étape de NL-Bayes. ``guide`` = image servant au regroupement."""
    h, w = noisy.shape
    ph, pw = h - k + 1, w - k + 1
    p_noisy = _patches(noisy, k)
    p_guide = p_noisy if guide is noisy else _patches(guide, k)
    acc_p = np.zeros((ph, pw, k * k))  # somme des estimées, par position de patch
    cnt_p = np.zeros((ph, pw))  # nombre d'estimées, par position de patch
    used = np.zeros((ph, pw), dtype=bool)
    sigma2 = sigma * sigma
    d = k * k
    eye = np.eye(d)

    for i in range(ph):
        for j in range(pw):
            if used[i, j]:
                continue
            si, sj = _window(i, j, n, ph, pw)
            cand = p_guide[si, sj].reshape(-1, d)
            dist = np.mean((cand - p_guide[i, j]) ** 2, axis=1)
            nj = sj.stop - sj.start

            m = min(n_keep, dist.size)
            if second:
                order = np.argsort(dist)
                tau = max(tau0, dist[order[m - 1]])
                sel = np.flatnonzero(dist <= tau)
            else:
                sel = np.argpartition(dist, m - 1)[:m]
            rows, cols = si.start + sel // nj, sj.start + sel % nj
            used[rows, cols] = True
            group = p_noisy[rows, cols]  # (m, d) patchs bruités
            mean = group.mean(axis=0)

            if use_area:
                total = group.size
                var = total / (total - 1) * (np.mean(group**2) - np.mean(group) ** 2)
                if var <= (gamma * sigma) ** 2:
                    np.add.at(acc_p, (rows, cols), group.mean())
                    np.add.at(cnt_p, (rows, cols), 1.0)
                    continue

            centered = group - mean
            try:
                if second:
                    g = p_guide[rows, cols]
                    gc = g - g.mean(axis=0)
                    cov = gc.T @ gc / max(len(sel) - 1, 1)
                    est = mean + (cov @ np.linalg.solve(cov + beta * sigma2 * eye, centered.T)).T
                else:
                    cov = centered.T @ centered / max(len(sel) - 1, 1)
                    est = mean + ((cov - beta * sigma2 * eye) @ np.linalg.solve(cov, centered.T)).T
            except np.linalg.LinAlgError:
                est = group
            np.add.at(acc_p, (rows, cols), est)
            np.add.at(cnt_p, (rows, cols), 1.0)

    num, den = _fold(acc_p, cnt_p, k, (h, w))
    out = noisy.copy()
    covered = den > 0
    out[covered] = num[covered] / den[covered]
    return out


def nl_bayes_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    data_range: tuple[float, float] | None = None,
    use_area1: bool = True,
    use_area2: bool = False,
    return_basic: bool = False,
    color_mode: str = "yuv",
    engine: str = "vectorized",
    step: int = 3,
    chunk: int = 256,
    n_threads: int | None = None,
    **overrides,
) -> np.ndarray:
    """
    Débruite une image par NL-Bayes (réimplémentation MIT depuis l'article).

    Parameters
    ----------
    image : np.ndarray
        Image bruitée : 2D (niveaux de gris) ou (H, W, 3) (couleur RGB).
    sigma : float, optional
        Écart-type du bruit (unités de l'image), identique sur tous les
        canaux. Estimé si absent.
    data_range : tuple(float, float), optional
        Dynamique ``(min, max)`` ramenée à [0, 255] pour appliquer les
        paramètres de l'article. Par défaut, le min/max de l'image. Pas de
        quantification (calcul en flottants).
    use_area1, use_area2 : bool
        Critère des zones homogènes à la 1re / 2e étape. Par défaut
        ``True`` / ``False`` (tableau 9 de l'article).
    return_basic : bool
        Renvoyer l'estimée de l'étape 1. Par défaut ``False``.
    color_mode : {"yuv", "rgb", "channel"}
        Traitement d'une image couleur (ignoré en niveaux de gris) :
        ``"yuv"`` (défaut, choix de l'article : étape 1 en Y₀U₀V₀ groupée
        sur Y₀, étape 2 en RGB), ``"rgb"`` (deux étapes en RGB, groupes
        communs aux 3 canaux), ``"channel"`` (canal par canal, comme 3
        images en niveaux de gris). Voir REFERENCE_TECHNIQUE.md §2.6.
    engine : {"vectorized", "loop"}
        ``"vectorized"`` (défaut) : calcul par lots, références sur une
        grille de pas ``step``. ``"loop"`` : version fidèle à l'article
        (astuce « paste »), niveaux de gris ou ``color_mode="channel"``
        seulement, plus lente.
    step : int
        Pas de la grille des patchs de référence (moteur vectorisé). Par
        défaut ``3``.
    chunk : int
        Nombre de groupes traités par lot (moteur vectorisé) : compromis
        mémoire / vitesse, sans effet sur le résultat. Par défaut ``256``.
    n_threads : int, optional
        Fils d'exécution du moteur vectorisé (les lots de groupes sont
        répartis entre eux). Par défaut ``min(cœurs, 8)``.
    **overrides
        Remplace un paramètre de l'article : ``k1``, ``k2``, ``n1``,
        ``n2``, ``N1``, ``N2``, ``gamma``, ``beta1``, ``beta2``, ``tau0``
        (en niveaux² sur l'échelle [0, 255]), ou ``window_factor`` (α, avec
        n = α·k). Voir :func:`resolve_parameters` pour les imbrications.

    Returns
    -------
    np.ndarray
        Image débruitée, ``float64``, même forme que l'entrée.
    """
    from .vectorized import COLOR_MODES, nl_bayes_vectorized

    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        img = check_image(arr)
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        if not np.all(np.isfinite(arr)):
            raise ValueError("L'image contient des valeurs non finies (NaN/inf).")
        img = arr
    else:
        raise ValueError(f"Image 2D ou (H, W, 3) attendue, reçu {arr.shape}.")
    if color_mode not in COLOR_MODES:
        raise ValueError(f"color_mode doit valoir {COLOR_MODES}, reçu '{color_mode}'.")
    if engine not in ("vectorized", "loop"):
        raise ValueError(f"engine doit valoir 'vectorized' ou 'loop', reçu '{engine}'.")

    if sigma is None:
        sigma = float(np.mean([resolve_sigma(img[..., c], None) for c in range(3)])) if img.ndim == 3 else None
    s = resolve_sigma(img if img.ndim == 2 else img[..., 0], sigma)
    unit, lo, scale = to_unit_range(img, data_range)
    sigma_255 = 255.0 * s / scale
    color = img.ndim == 3 and color_mode != "channel"
    p_gray = resolve_parameters(sigma_255, False, overrides)
    p_color = resolve_parameters(sigma_255, True, overrides)
    kmax = max(p_gray["k1"], p_gray["k2"], p_color["k1"], p_color["k2"])
    if min(img.shape[:2]) < kmax:
        raise ValueError(f"Image trop petite ({img.shape}) pour des patchs de {kmax}.")
    u = 255.0 * unit

    if engine == "loop":
        if color:
            raise ValueError("engine='loop' ne traite la couleur qu'en color_mode='channel'.")
        chans = [u] if u.ndim == 2 else [u[..., c] for c in range(3)]
        outs = []
        for ch in chans:
            basic = _step(ch, ch, sigma_255, p_gray["k1"], p_gray["n1"], p_gray["N1"], p_gray["beta1"],
                          p_gray["gamma"], use_area1, 0.0, second=False)
            if not return_basic:
                basic = _step(ch, basic, sigma_255, p_gray["k2"], p_gray["n2"], p_gray["N2"], p_gray["beta2"],
                              p_gray["gamma"], use_area2, p_gray["tau0"], second=True)
            outs.append(basic)
        out = outs[0] if u.ndim == 2 else np.stack(outs, axis=-1)
    else:
        out = nl_bayes_vectorized(
            u, sigma_255, p_gray, p_color, color_mode, use_area1, use_area2, return_basic, int(step), int(chunk),
            n_threads if n_threads is not None else min(os.cpu_count() or 1, 8),
        )
    return from_unit_range(out / 255.0, lo, scale)
