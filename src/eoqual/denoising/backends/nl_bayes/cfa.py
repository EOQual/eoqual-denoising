#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NL-Bayes CFA : débruitage et démosaïçage conjoints d'une mosaïque de Bayer.

Réimplémentation clean-room (MIT) à partir de deux publications seulement :

- l'**estimateur** est celui des estimateurs linéaires par morceaux (PLE) :
  G. Yu, G. Sapiro, S. Mallat, « Solving Inverse Problems With Piecewise
  Linear Estimators: From Gaussian Mixture Models to Structured Sparsity »,
  IEEE Trans. Image Processing 21(5), 2012. Pour une observation
  y = U·x + n d'un patch x ~ N(μ, C), bruit n ~ N(0, σ²I),
  l'espérance a posteriori vaut x̂ = μ + C·Uᵀ·(U·C·Uᵀ + σ²I)⁻¹·(y − U·μ) ;
- l'**a priori** gaussien de chaque patch est celui de son groupe de patchs
  semblables, comme à l'étape 2 de NL-Bayes (Buades, Lebrun & Morel, IPOL
  2013) : groupes formés et moyenne / covariance estimées sur une image
  pilote, seuil τ = max(τ₀, d_N).

Le patch x est la couleur complète d'un bloc k × k (dimension D = 3k²) ; la
mosaïque n'en mesure qu'une valeur par pixel (d = k² mesures, sélection U
fixée par le motif et par la position du patch). L'estimateur rend les
D valeurs : il débruite les valeurs mesurées et calcule les autres. L'image
pilote ne sert qu'à former les groupes et à estimer μ et C ; aucune de ses
valeurs ne passe dans le résultat. Voir NL_BAYES.md §13.
"""

from __future__ import annotations

import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ...utils import check_image, estimate_noise_sigma, from_unit_range, to_unit_range

__all__ = ["nl_bayes_cfa_denoise", "bilinear_demosaic", "cfa_channels", "CFA_PATTERNS"]

CFA_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")
_CHANNEL = {"R": 0, "G": 1, "B": 2}


def cfa_channels(pattern: str) -> np.ndarray:
    """Canal mesuré (0 R, 1 G, 2 B) à chaque position du motif 2 × 2, forme (2, 2)."""
    p = pattern.upper()
    if p not in CFA_PATTERNS:
        raise ValueError(f"Motif inconnu '{pattern}' (attendu : {CFA_PATTERNS}).")
    return np.array([[_CHANNEL[p[0]], _CHANNEL[p[1]]], [_CHANNEL[p[2]], _CHANNEL[p[3]]]])


def _channel_map(h: int, w: int, pattern: str) -> np.ndarray:
    ch = cfa_channels(pattern)
    return np.tile(ch, ((h + 1) // 2, (w + 1) // 2))[:h, :w]


def bilinear_demosaic(mosaic: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Démosaïçage bilinéaire (normalisé par les poids, bords compris) : (H, W) → (H, W, 3)."""
    from scipy import ndimage

    m = np.asarray(mosaic, dtype=np.float64)
    cm = _channel_map(*m.shape, pattern)
    kernel = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])
    out = np.empty(m.shape + (3,))
    for c in range(3):
        sel = (cm == c).astype(np.float64)
        num = ndimage.convolve(m * sel, kernel, mode="mirror")
        den = ndimage.convolve(sel, kernel, mode="mirror")
        out[..., c] = np.where(sel > 0, m, num / np.maximum(den, 1e-12))
    return out


def _estimate_mosaic_sigma(m: np.ndarray) -> float:
    """σ estimé sur les quatre sous-plans du motif (chacun est une image ordinaire)."""
    subs = [m[i::2, j::2] for i in (0, 1) for j in (0, 1)]
    return float(np.mean([estimate_noise_sigma(s) for s in subs if min(s.shape) >= 8]))


def _cfa_pass(y255, pilot, chan, sigma2, k, n, n_keep, beta, tau0, step, chunk, n_threads, area=None):
    """
    Une passe de l'estimateur : mosaïque (H, W) et pilote (H, W, 3), échelle [0, 255].

    ``area`` : None (pas de critère des zones homogènes), ou (``"z"`` ou ``"gamma"``, seuil,
    ``"mosaic"`` ou ``"pilot"`` : valeur donnée aux groupes plats).
    """
    h, w = y255.shape
    ph, pw = h - k + 1, w - k + 1
    npos = ph * pw
    d, D = k * k, 3 * k * k
    # Patchs pilotes : composante (pixel p, canal c) à l'indice 3p + c.
    pp = np.ascontiguousarray(sliding_window_view(pilot, (k, k), axis=(0, 1))
                              .transpose(0, 1, 3, 4, 2).reshape(npos, D), dtype=np.float32)
    py = np.ascontiguousarray(sliding_window_view(y255, (k, k)).reshape(npos, d), dtype=np.float32)
    # Composantes mesurées selon la phase (i mod 2, j mod 2) du coin du patch.
    t_idx = np.empty((4, d), dtype=np.intp)
    for a in (0, 1):
        for b in (0, 1):
            cm = chan[a:a + k, b:b + k].reshape(-1)
            t_idx[2 * a + b] = 3 * np.arange(d) + cm
    pos_phase = ((np.arange(ph)[:, None] % 2) * 2 + (np.arange(pw)[None, :] % 2)).reshape(-1)

    ni, nj = min(n, ph), min(n, pw)
    off_i, off_j = np.arange(ni), np.arange(nj)
    m_win = ni * nj
    n_keep = min(n_keep, m_win)
    eye_d = np.eye(d, dtype=np.float32)
    ri_all = np.unique(np.append(np.arange(0, ph, step), ph - 1))
    rj_all = np.unique(np.append(np.arange(0, pw, step), pw - 1))
    gi, gj = (a.ravel() for a in np.meshgrid(ri_all, rj_all, indexing="ij"))
    starts = list(range(0, gi.size, chunk))
    n_threads = max(1, min(int(n_threads), len(starts)))

    def filter_groups(idx, mask, acc, cnt):
        wts = mask.astype(np.float32)
        msum = wts.sum(axis=1)
        grp = pp[idx]                                                     # (b, K, D)
        mean = np.einsum("bm,bmD->bD", wts, grp) / msum[:, None]
        xc = (grp - mean[:, None, :]) * wts[:, :, None]
        cov = np.swapaxes(xc, 1, 2) @ xc / np.maximum(msum - 1, 1)[:, None, None]
        obs = py[idx]                                                     # (b, K, d)
        ph_idx = pos_phase[idx]
        est = np.zeros_like(grp)
        for f in range(4):
            sel = (ph_idx == f) & mask
            if not sel.any():
                continue
            t = t_idx[f]
            c_t = cov[:, :, t]                                            # (b, D, d) = C·Uᵀ
            s = c_t[:, t, :] + beta * sigma2 * eye_d                      # U·C·Uᵀ + βσ²I
            g_t = np.linalg.solve(s, np.swapaxes(c_t, 1, 2))              # (b, d, D) = Gᵀ
            resid = obs - mean[:, None, t]                                # y − U·μ
            est_f = mean[:, None, :] + resid @ g_t
            est = np.where(sel[:, :, None], est_f, est)
        if area is not None:
            # Critère des zones homogènes, sur les mesures de la mosaïque (bruit blanc σ²),
            # canal par canal : une zone plate est de couleur constante, pas grise.
            ch_obs = t_idx[ph_idx] % 3                                    # (b, K, d) canal mesuré
            obs64 = obs.astype(np.float64)
            ss = np.zeros(idx.shape[0])
            n_tot = np.zeros(idx.shape[0])
            means = np.zeros((idx.shape[0], 3))
            for c in range(3):
                w_c = ((ch_obs == c) & mask[:, :, None]).astype(np.float64)
                n_c = w_c.sum(axis=(1, 2))
                s_c = (obs64 * w_c).sum(axis=(1, 2))
                q_c = (obs64 * obs64 * w_c).sum(axis=(1, 2))
                means[:, c] = s_c / np.maximum(n_c, 1)
                ss += q_c - s_c * means[:, c]
                n_tot += n_c
            dof = np.maximum(n_tot - 3, 1)
            s2 = ss / dof
            if area[0] == "z":
                thr = sigma2 * (1.0 + area[1] * np.sqrt(2.0 / dof))
            else:
                thr = np.full_like(s2, (area[1] ** 2) * sigma2)
            flat = s2 <= thr
            if flat.any():
                if area[2] == "pilot":
                    # moyenne du groupe pilote (limite C → 0 de l'estimateur)
                    est[flat] = mean[flat][:, None, :]
                else:
                    # moyenne des mesures de chaque canal : composante 3p + c = moyenne du canal c
                    est[flat] = np.tile(means[flat], k * k)[:, None, :]
        sel_idx = idx[mask]
        uniq, inv = np.unique(sel_idx, return_inverse=True)
        inv = inv.ravel()
        sel_est = est[mask]
        sums = np.empty((uniq.size, D))
        for col in range(D):
            sums[:, col] = np.bincount(inv, weights=sel_est[:, col], minlength=uniq.size)
        acc[uniq] += sums
        cnt[uniq] += np.bincount(inv, minlength=uniq.size)

    def work(part):
        acc = np.zeros((npos, D))
        cnt = np.zeros(npos)
        for s0 in part:
            ri, rj = gi[s0:s0 + chunk], gj[s0:s0 + chunk]
            b = ri.size
            i0 = np.clip(ri - ni // 2, 0, ph - ni)
            j0 = np.clip(rj - nj // 2, 0, pw - nj)
            lin = ((i0[:, None] + off_i)[:, :, None] * pw + (j0[:, None] + off_j)[:, None, :]).reshape(b, m_win)
            ref = ri * pw + rj
            dist = np.mean((pp[lin] - pp[ref][:, None, :]) ** 2, axis=2)
            d_n = np.partition(dist, n_keep - 1, axis=1)[:, n_keep - 1]
            tau = np.maximum(np.float32(tau0), d_n)
            sizes = (dist <= tau[:, None]).sum(axis=1)
            kb = np.minimum(np.maximum(64, 2 ** np.ceil(np.log2(np.maximum(sizes, 1)))).astype(int), m_win)
            for kval in np.unique(kb):
                rows = np.flatnonzero(kb == kval)
                if kval < m_win:
                    part_idx = np.argpartition(dist[rows], kval - 1, axis=1)[:, :kval]
                else:
                    part_idx = np.broadcast_to(np.arange(m_win), (rows.size, m_win))
                d_sub = np.take_along_axis(dist[rows], part_idx, axis=1)
                idx = np.take_along_axis(lin[rows], part_idx, axis=1)
                filter_groups(idx, d_sub <= tau[rows, None], acc, cnt)
        return acc, cnt

    parts = [starts[t::n_threads] for t in range(n_threads)]
    if n_threads == 1:
        acc, cnt = work(parts[0])
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            res = list(ex.map(work, parts))
        acc = sum(r[0] for r in res)
        cnt = sum(r[1] for r in res)

    acc_p = acc.reshape(ph, pw, d, 3)
    cnt_p = cnt.reshape(ph, pw)
    num = np.zeros((h, w, 3))
    den = np.zeros((h, w))
    for di in range(k):
        for dj in range(k):
            num[di:di + ph, dj:dj + pw] += acc_p[:, :, di * k + dj]
            den[di:di + ph, dj:dj + pw] += cnt_p
    covered = den > 0
    out = pilot.copy()
    out[covered] = num[covered] / den[covered, None]
    return out


def nl_bayes_cfa_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    pattern: str = "RGGB",
    data_range: tuple[float, float] | None = None,
    k: int = 3,
    window_factor: float = 7.0,
    N: int = 60,
    beta: float = 1.0,
    tau0: float = 4.0,
    n_iter: int = 8,
    use_area: bool = False,
    area_z: float = 2.0,
    area_gamma: float | None = None,
    area_mean: str = "mosaic",
    pilot: np.ndarray | str = "bilinear",
    pilot_kwargs: dict | None = None,
    step: int = 3,
    chunk: int = 256,
    n_threads: int | None = None,
) -> np.ndarray:
    """
    Débruite et démosaïque une mosaïque de Bayer (NL-Bayes CFA).

    Parameters
    ----------
    image : np.ndarray
        Mosaïque bruitée (H, W) : une valeur mesurée par pixel, canal donné
        par ``pattern``. Bruit blanc d'écart-type ``sigma`` (après
        stabilisation de variance : 1, voir :func:`~eoqual.denoising.vst_denoise`).
    sigma : float, optional
        Écart-type du bruit de la mosaïque (unités de l'image). Estimé sur
        les quatre sous-plans du motif si absent.
    pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
        Motif 2 × 2, ligne par ligne, à partir du pixel (0, 0).
    data_range : tuple(float, float), optional
        Dynamique ``(min, max)`` ramenée à [0, 255] (échelle de τ₀). Par
        défaut, le min/max de la mosaïque.
    k : int
        Côté des patchs (D = 3k² valeurs couleur, k² mesures). Par défaut 3.
    window_factor : float
        Fenêtre de recherche n = α·k. Par défaut 7 (article NL-Bayes).
    N : int
        Taille minimale des groupes (seuil τ = max(τ₀, d_N)). Par défaut 60 :
        la covariance d'un patch couleur 3 × 3 a D = 27 dimensions, et N = 30
        l'estime mal (−0,2 à −1,6 dB à 4 passes, NL_BAYES.md §13.4).
    beta : float
        Multiplie σ² dans U·C·Uᵀ + βσ²I (rôle de β₂ de NL-Bayes). Par
        défaut 1.
    tau0 : float
        Plancher de similarité, en niveaux² sur [0, 255]. Par défaut 4.
    n_iter : int
        Nombre de passes ; chaque passe prend la sortie de la précédente
        comme pilote (itération à la manière de l'EM de Yu et al.). Par
        défaut 8 : avec N = 60, 8 passes gagnent jusqu'à 1,1 dB sur
        4 passes sur les orthophotos et 2,6 à 3,5 dB sur les scènes à
        grandes zones uniformes ; à 300 e⁻ sur les orthophotos, 4 passes
        valent autant (NL_BAYES.md §13.4).
    use_area : bool
        Critère des zones homogènes : un groupe dont la variance des mesures,
        calculée canal par canal autour de la moyenne de chaque canal, reste
        sous le seuil est remplacé par ces moyennes (couleur constante).
        Désactivé par défaut (coûteux sur une zone parfaitement uniforme) ;
        à activer à faible flux : il ramène l'erreur de luminance des
        ombres au niveau de la chaîne démosaïçage + NL-Bayes, pour 2 à 3 %
        d'erreur de couleur en plus (NL_BAYES.md §13.5).
    area_z : float
        Seuil statistique du critère : s² ≤ σ²·(1 + z·√(2/(n − 3))), n le
        nombre de mesures du groupe (√(2/(n − 3)) : fluctuation relative de s²
        sur du bruit pur). Par défaut 2.
    area_gamma : float, optional
        Seuil fixe s² ≤ (γσ)² de NL-Bayes, à la place du seuil statistique
        (comparaison).
    area_mean : {"mosaic", "pilot"}
        Valeur d'un groupe plat : moyenne des mesures de chaque canal
        (``"mosaic"``) ou moyenne du groupe pilote (``"pilot"``).
    pilot : "bilinear", "none" ou np.ndarray
        Image pilote initiale : ``"bilinear"`` (défaut) = démosaïçage
        bilinéaire puis NL-Bayes couleur ; ``"none"`` = démosaïçage
        bilinéaire seul ; ou une image (H, W, 3) fournie, dans les unités de
        la mosaïque. Un meilleur pilote donne de meilleurs groupes et une
        meilleure covariance : Hamilton-Adams puis NL-Bayes (eoqual-demosaicing)
        accélère la convergence (avec N = 30 : 0,6 à 2,2 dB de plus à
        4 passes, écart nul vers 12 passes).
    pilot_kwargs : dict, optional
        Arguments de :func:`nl_bayes_denoise` pour le pilote ``"bilinear"``
        (σ = ``sigma`` par défaut).
    step, chunk, n_threads :
        Pas de la grille des références, taille des lots, fils d'exécution
        (comme le moteur vectorisé de NL-Bayes).

    Returns
    -------
    np.ndarray
        Image RGB (H, W, 3), ``float64``, unités de la mosaïque.
    """
    from .native import nl_bayes_denoise

    mosaic = check_image(np.asarray(image, dtype=np.float64))
    cfa_channels(pattern)                            # vérifie le motif
    if k < 1 or N < 2 or n_iter < 1 or beta <= 0:
        raise ValueError("k ≥ 1, N ≥ 2, n_iter ≥ 1 et beta > 0 sont requis.")
    h, w = mosaic.shape
    if min(h, w) < k + 1:
        raise ValueError(f"Mosaïque trop petite ({mosaic.shape}) pour des patchs de {k}.")
    s = float(sigma) if sigma is not None else _estimate_mosaic_sigma(mosaic)
    if not s > 0:
        raise ValueError(f"sigma doit être > 0, reçu {s}.")
    unit, lo, scale = to_unit_range(mosaic, data_range)
    y255 = 255.0 * unit
    sigma_255 = 255.0 * s / scale
    n_threads = n_threads if n_threads is not None else min(os.cpu_count() or 1, 8)

    if isinstance(pilot, str):
        if pilot not in ("bilinear", "none"):
            raise ValueError(f"pilot doit valoir 'bilinear', 'none' ou une image, reçu '{pilot}'.")
        cur = bilinear_demosaic(y255, pattern)
        if pilot == "bilinear":
            kw = dict(sigma=sigma_255, data_range=(0.0, 255.0), n_threads=n_threads)
            kw.update(pilot_kwargs or {})
            cur = nl_bayes_denoise(cur, **kw)
    else:
        p = np.asarray(pilot, dtype=np.float64)
        if p.shape != (h, w, 3):
            raise ValueError(f"Pilote (H, W, 3) attendu, de forme {(h, w, 3)}, reçu {p.shape}.")
        cur = 255.0 * (p - lo) / scale

    chan = _channel_map(h + 1, w + 1, pattern)       # +1 : phases des patchs en bord
    n = int(round(window_factor * k))
    area = None
    if use_area:
        if area_mean not in ("mosaic", "pilot"):
            raise ValueError(f"area_mean doit valoir 'mosaic' ou 'pilot', reçu '{area_mean}'.")
        if area_gamma is not None and area_gamma <= 0:
            raise ValueError(f"area_gamma doit être > 0, reçu {area_gamma}.")
        area = (("gamma", float(area_gamma)) if area_gamma is not None else ("z", float(area_z))) + (area_mean,)
    for _ in range(int(n_iter)):
        cur = _cfa_pass(y255, cur, chan, np.float32(sigma_255 ** 2), int(k), n, int(N), float(beta), float(tau0),
                        int(step), int(chunk), n_threads, area)
    return from_unit_range(cur / 255.0, lo, scale)
