#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NL-Ridge — débruitage non local par combinaisons linéaires de patchs
(Herbreteau & Kervrann), implémentation originale numpy, niveaux de gris,
bruit blanc gaussien.

**Réimplémentation « clean-room »** écrite pour ce projet à partir du seul
texte des articles : ICIP 2022 (arXiv:2203.00570, Propositions 2.1 à 2.3,
§2.4, tableau 1) et SIAM J. Imaging Sci. 2025 (arXiv:2402.13816, formules
(4)-(6), (13)-(17), §2.5, §4.1). Le code des auteurs (``nlridge.py``,
AGPL-3.0) n'a pas été consulté pour l'algorithme. Licence MIT.

Principe. Un groupe de k patchs similaires de n pixels forme une matrice
Y (n × k, une colonne par patch). Chaque patch est estimé par une
combinaison linéaire des patchs du groupe : X̂ = Y Θ, Θ (k × k). Θ minimise
une estimation du risque quadratique E‖YΘ − X‖² :

- **étape 1** — estimateur sans biais du risque (SURE), en bruit gaussien
  D₁ = nσ²I :  Θ₁ = I − nσ² (YᵀY)⁻¹  (« linéaire ») ;
- **étape 2** — « adaptation interne » : X est remplacé par le groupe X̂₁
  correspondant dans l'image de l'étape 1 (régression ridge multivariée) :
  Θ₂ = I − nσ² (X̂₁ᵀX̂₁ + nσ²I)⁻¹.

Variante ``constraints="affine"`` (article 2025, formules (6) et (15)) :
poids de chaque combinaison de somme 1, Θ = I − (Q⁻¹ − Q⁻¹1(Q⁻¹1)ᵀ /
1ᵀQ⁻¹1) D, avec Q = YᵀY (étape 1) ou X̂₁ᵀX̂₁ + nσ²I (étape 2).

Reprojection pondérée (§2.4 / §2.5) : chaque pixel du patch estimé j reçoit
le poids w_j = 1/‖Θ·,j‖², puis normalisation pixel par pixel.

Accélération : patchs de référence sur une grille de pas δ (4 dans les
articles), fenêtre de recherche κ×κ (37 dans l'article 2025).

Non porté à ce stade : bruits de Poisson et poisson-gaussien de l'article
2025 (utiliser ``vst_denoise``), contraintes coniques et convexes (pas de
forme fermée), couleur.
"""
from __future__ import annotations

import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ...utils import check_image, from_unit_range, resolve_sigma, to_unit_range

__all__ = ["nl_ridge_denoise", "nl_ridge_parameters", "CONSTRAINTS"]

CONSTRAINTS = ("linear", "affine")


def nl_ridge_parameters(sigma_255: float) -> dict:
    """
    Paramètres des auteurs (tableau 1 des deux articles), σ sur [0, 255] :
    taille de patch ``p1``/``p2`` (côté, n = p²), nombre de patchs
    ``k1``/``k2``, fenêtre ``window`` = 37 (article 2025), pas ``step`` = 4.
    En dessous de 10 et au-dessus de 50, le palier le plus proche est
    appliqué (hors du domaine étudié par les auteurs).
    """
    if sigma_255 <= 15:
        p1, p2, k1, k2 = 7, 7, 18, 55
    elif sigma_255 <= 35:
        p1, p2, k1, k2 = 9, 9, 18, 90
    else:
        p1, p2, k1, k2 = 11, 9, 20, 120
    return {"p1": p1, "p2": p2, "k1": k1, "k2": k2, "window": 37, "step": 4}


def _flat(img: np.ndarray, p: int) -> np.ndarray:
    v = sliding_window_view(img, (p, p))
    return np.ascontiguousarray(v.reshape(-1, p * p), dtype=np.float64)


def _grid(ph: int, pw: int, step: int):
    ri = np.unique(np.append(np.arange(0, ph, step), ph - 1))
    rj = np.unique(np.append(np.arange(0, pw, step), pw - 1))
    gi, gj = np.meshgrid(ri, rj, indexing="ij")
    return gi.ravel(), gj.ravel()


def _theta(q: np.ndarray, d: float, affine: bool) -> np.ndarray:
    """Θ = I − Q⁻¹ D (linéaire) ou version affine, D = d·I. q : (b, k, k)."""
    k = q.shape[-1]
    eye = np.eye(k)
    try:
        qinv = np.linalg.inv(q)
    except np.linalg.LinAlgError:
        qinv = np.linalg.pinv(q)
    if affine:
        u = qinv.sum(axis=2)  # Q⁻¹ 1 (b, k)
        s = u.sum(axis=1)  # 1ᵀ Q⁻¹ 1
        qinv = qinv - u[:, :, None] * u[:, None, :] / s[:, None, None]
    return eye - d * qinv


def _step(noisy, pilot, sigma, p, k, window, step, chunk, affine, second, n_threads):
    h, w = noisy.shape
    ph, pw = h - p + 1, w - p + 1
    npos = ph * pw
    n = p * p
    py = _flat(noisy, p)
    pg = py if pilot is None else _flat(pilot, p)
    ni, nj = min(window, ph), min(window, pw)
    off_i, off_j = np.arange(ni), np.arange(nj)
    k = min(k, ni * nj)
    d = n * sigma * sigma
    gi, gj = _grid(ph, pw, step)
    starts = list(range(0, gi.size, chunk))
    n_threads = max(1, min(n_threads, len(starts)))

    def work(part):
        acc = np.zeros((npos, n))
        wsum = np.zeros(npos)
        for s in part:
            ri, rj = gi[s : s + chunk], gj[s : s + chunk]
            b = ri.size
            i0 = np.clip(ri - ni // 2, 0, ph - ni)
            j0 = np.clip(rj - nj // 2, 0, pw - nj)
            lin = ((i0[:, None] + off_i)[:, :, None] * pw + (j0[:, None] + off_j)[:, None, :]).reshape(b, -1)
            ref = ri * pw + rj
            # Regroupement : k plus proches (distance L2) dans la fenêtre,
            # sur l'image bruitée (étape 1) ou l'image pilote (étape 2).
            dist = np.sum((pg[lin] - pg[ref][:, None, :]) ** 2, axis=2)
            sel = np.argpartition(dist, k - 1, axis=1)[:, :k]
            idx = np.take_along_axis(lin, sel, axis=1)  # (b, k)
            y = np.swapaxes(py[idx], 1, 2)  # (b, n, k)
            if second:
                x1 = np.swapaxes(pg[idx], 1, 2)
                q = np.swapaxes(x1, 1, 2) @ x1 + d * np.eye(k)
            else:
                q = np.swapaxes(y, 1, 2) @ y
            theta = _theta(q, d, affine)  # (b, k, k)
            est = np.swapaxes(y @ theta, 1, 2)  # (b, k, n) : patchs estimés
            wts = 1.0 / np.maximum(np.sum(theta * theta, axis=1), 1e-12)  # (b, k), 1/‖Θ·,j‖²
            flat_idx = idx.ravel()
            uniq, inv = np.unique(flat_idx, return_inverse=True)
            inv = inv.ravel()
            we = (est * wts[:, :, None]).reshape(-1, n)
            sums = np.empty((uniq.size, n))
            for col in range(n):
                sums[:, col] = np.bincount(inv, weights=we[:, col], minlength=uniq.size)
            acc[uniq] += sums
            wsum[uniq] += np.bincount(inv, weights=wts.ravel(), minlength=uniq.size)
        return acc, wsum

    parts = [starts[t::n_threads] for t in range(n_threads)]
    if n_threads == 1:
        acc, wsum = work(parts[0])
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            res = list(ex.map(work, parts))
        acc = sum(r[0] for r in res)
        wsum = sum(r[1] for r in res)

    num = np.zeros((h, w))
    den = np.zeros((h, w))
    acc_p = acc.reshape(ph, pw, n)
    w_p = wsum.reshape(ph, pw)
    for di in range(p):
        for dj in range(p):
            num[di : di + ph, dj : dj + pw] += acc_p[:, :, di * p + dj]
            den[di : di + ph, dj : dj + pw] += w_p
    return np.where(den > 0, num / np.maximum(den, 1e-12), noisy)


def nl_ridge_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    data_range: tuple[float, float] | None = None,
    constraints: str = "linear",
    return_basic: bool = False,
    chunk: int = 256,
    n_threads: int | None = None,
    **overrides,
) -> np.ndarray:
    """
    Débruite une image par NL-Ridge (réimplémentation MIT depuis les articles).

    Parameters
    ----------
    image : np.ndarray
        Image 2D bruitée.
    sigma : float, optional
        Écart-type du bruit (unités de l'image). Estimé si absent.
    data_range : tuple(float, float), optional
        Dynamique ``(min, max)`` ramenée à [0, 255] pour appliquer les
        paramètres des auteurs. Par défaut, le min/max de l'image.
    constraints : {"linear", "affine"}
        Combinaisons linéaires libres (défaut des auteurs) ou affines
        (poids de somme 1).
    return_basic : bool
        Renvoyer l'estimée de l'étape 1. Par défaut ``False``.
    chunk : int
        Groupes traités par lot (mémoire / vitesse, sans effet sur le
        résultat). Par défaut ``256``.
    n_threads : int, optional
        Fils d'exécution. Par défaut ``min(cœurs, 8)``.
    **overrides
        Remplace un paramètre de :func:`nl_ridge_parameters` : ``p1``,
        ``p2``, ``k1``, ``k2``, ``window``, ``step``.

    Returns
    -------
    np.ndarray
        Image débruitée, ``float64``, même forme que l'entrée.
    """
    if constraints not in CONSTRAINTS:
        raise ValueError(f"constraints doit valoir {CONSTRAINTS}, reçu '{constraints}'.")
    img = check_image(image)
    s = resolve_sigma(img, sigma)
    unit, lo, scale = to_unit_range(img, data_range)
    sigma_255 = 255.0 * s / scale
    params = nl_ridge_parameters(sigma_255)
    unknown = set(overrides) - set(params)
    if unknown:
        raise TypeError(f"Paramètres inconnus : {sorted(unknown)} (autorisés : {sorted(params)})")
    params.update({key: int(v) for key, v in overrides.items()})
    if min(img.shape) < max(params["p1"], params["p2"]):
        raise ValueError(f"Image trop petite ({img.shape}) pour des patchs de {max(params['p1'], params['p2'])}.")
    threads = n_threads if n_threads is not None else min(os.cpu_count() or 1, 8)
    affine = constraints == "affine"

    u = 255.0 * unit
    basic = _step(u, None, sigma_255, params["p1"], params["k1"], params["window"], params["step"],
                  int(chunk), affine, False, threads)
    if return_basic:
        return from_unit_range(basic / 255.0, lo, scale)
    final = _step(u, basic, sigma_255, params["p2"], params["k2"], params["window"], params["step"],
                  int(chunk), affine, True, threads)
    return from_unit_range(final / 255.0, lo, scale)
