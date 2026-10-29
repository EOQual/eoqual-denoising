#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NL-Bayes vectorisé (numpy), niveaux de gris et couleur.

Même méthode et mêmes formules que ``native.py`` (Lebrun, Buades & Morel,
IPOL 2013 — réimplémentation clean-room depuis l'article, MIT), organisée
pour le calcul par lots :

- **références sur une grille** de pas ``step`` (3 par défaut, comme
  BM3D) au lieu de l'astuce « paste » de l'article, qui est séquentielle
  par nature (un patch utilisé n'est plus référence). Toutes les positions
  de bord sont incluses, chaque pixel est donc estimé ;
- pour un lot de références : extraction des fenêtres de recherche,
  distances, sélection des voisins, moyennes et covariances (einsum),
  résolution des systèmes (``np.linalg.solve`` par lots) et agrégation
  (``np.bincount``), sans boucle Python par groupe.

Couleur (article §3.3, §3.4, §5.1.1, tableau 8) — paramètre ``color_mode`` :

- ``"yuv"`` (défaut pour une image 3 canaux, choix de l'article) :
  étape 1 dans l'espace orthonormé Y₀U₀V₀, regroupement sur Y₀ seul, mêmes
  groupes appliqués à U₀ et V₀, filtrage canal par canal ; étape 2 en RGB,
  regroupement sur la distance moyenne des trois canaux de u_basic,
  filtrage canal par canal avec les mêmes groupes ;
- ``"rgb"`` : les deux étapes en RGB, regroupement sur les trois canaux ;
- ``"channel"`` : chaque canal débruité indépendamment comme une image en
  niveaux de gris (groupes différents par canal).

Paramètres : tableau §4.8 de l'article pour la couleur (``"yuv"``,
``"rgb"``), adaptation niveaux de gris §6.1.2 (patchs de 5) pour une
image 2D et le mode ``"channel"``.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = ["nl_bayes_vectorized", "YUV_MATRIX", "COLOR_MODES"]

COLOR_MODES = ("yuv", "rgb", "channel")

#: Transformée Y₀U₀V₀ de l'article (§3.3), normalisée pour être orthonormée :
#: σ est conservé sur chaque canal transformé.
YUV_MATRIX = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, 0.0, -1.0],
        [1.0, -2.0, 1.0],
    ]
) / np.array([[np.sqrt(3.0)], [np.sqrt(2.0)], [np.sqrt(6.0)]])


def _flat_patches(channel: np.ndarray, k: int) -> np.ndarray:
    """Patchs k×k d'un canal, aplatis : (nombre de positions, k²), float32."""
    view = sliding_window_view(channel, (k, k))
    return np.ascontiguousarray(view.reshape(-1, k * k), dtype=np.float32)


def _reference_grid(ph: int, pw: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    ri = np.unique(np.append(np.arange(0, ph, step), ph - 1))
    rj = np.unique(np.append(np.arange(0, pw, step), pw - 1))
    gi, gj = np.meshgrid(ri, rj, indexing="ij")
    return gi.ravel(), gj.ravel()


def _solve(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Résout a x = b par lots. Renvoie (x, ok) ; un système singulier donne
    ok=False (l'appelant laisse alors les patchs inchangés, article §3.3).
    """
    try:
        return np.linalg.solve(a, b), np.ones(a.shape[0], dtype=bool)
    except np.linalg.LinAlgError:
        x = np.zeros_like(b)
        ok = np.ones(a.shape[0], dtype=bool)
        for t in range(a.shape[0]):
            try:
                x[t] = np.linalg.solve(a[t], b[t])
            except np.linalg.LinAlgError:
                ok[t] = False
        return x, ok


def _filter(pn, pg, idx, mask, nc, d, beta, sigma2, eye, gamma, sigma, use_area, second, acc, cnt, npos):
    """
    Filtre bayésien d'un lot de groupes et agrégation.

    ``idx`` (b, K) : positions des patchs de chaque groupe ; ``mask`` (b, K) :
    appartenance effective au groupe (étape 2 : distance ≤ τ).
    """
    wts = mask.astype(np.float32)
    m = wts.sum(axis=1)
    denom = np.maximum(m - 1, 1)[:, None, None]
    sel_idx = idx[mask]
    # Positions touchées par le lot : l'agrégation se fait sur elles seules
    # (coût proportionnel au lot, pas à l'image).
    uniq, pos_inv = np.unique(sel_idx, return_inverse=True)
    pos_inv = pos_inv.ravel()
    nu = uniq.size
    for c in range(nc):
        grp = pn[c][idx]  # (b, K, d)
        mean = np.einsum("bm,bmd->bd", wts, grp) / m[:, None]
        centered = grp - mean[:, None, :]
        # Matrice de filtrage du groupe A (d×d), puis estimées de tous les
        # patchs du groupe par un seul produit : est = P̄ + (Q̃ − P̄) Aᵀ.
        if second:
            # Covariance oracle : patchs de u_basic (guide), même canal.
            gb = pg[c if len(pg) == nc else 0][idx]
            mean_b = np.einsum("bm,bmd->bd", wts, gb) / m[:, None]
            xb = (gb - mean_b[:, None, :]) * wts[:, :, None]
            cov = np.swapaxes(xb, 1, 2) @ xb / denom
            # A = C_b (C_b + β₂σ²I)⁻¹ ; Aᵀ = (C_b + β₂σ²I)⁻¹ C_b (symétriques)
            a_t, ok = _solve(cov + beta * sigma2 * eye, cov)
        else:
            xc = centered * wts[:, :, None]
            cov = np.swapaxes(xc, 1, 2) @ xc / denom
            # A = (C − β₁σ²I) C⁻¹ = I − β₁σ² C⁻¹ (symétrique)
            inv, ok = _solve(cov, np.broadcast_to(eye, cov.shape).copy())
            a_t = eye - beta * sigma2 * inv
        est = mean[:, None, :] + centered @ a_t
        est = np.where(ok[:, None, None], est, grp)
        if use_area:
            # Critère des zones homogènes (formule 10).
            total = m * d
            s1 = np.einsum("bm,bmd->b", wts, grp) / total
            s2 = np.einsum("bm,bmd->b", wts, grp * grp) / total
            var = total / np.maximum(total - 1, 1) * (s2 - s1 * s1)
            flat = var <= (gamma * sigma) ** 2
            if flat.any():
                est[flat] = s1[flat, None, None]
        sel_est = est[mask]
        sums = np.empty((nu, d))
        for col in range(d):
            sums[:, col] = np.bincount(pos_inv, weights=sel_est[:, col], minlength=nu)
        acc[c][uniq] += sums
    cnt[uniq] += np.bincount(pos_inv, minlength=nu)


def _step(
    noisy: np.ndarray,
    guide: np.ndarray,
    sigma: float,
    k: int,
    n: int,
    n_keep: int,
    beta: float,
    gamma: float,
    use_area: bool,
    tau0: float,
    second: bool,
    step: int,
    chunk: int,
    n_threads: int = 1,
) -> np.ndarray:
    """
    Une étape de NL-Bayes, par lots, répartis sur ``n_threads`` fils
    (chacun avec ses propres tampons d'agrégation, sommés à la fin).

    ``noisy`` : (C, H, W) canaux filtrés ; ``guide`` : (G, H, W) canaux
    servant au regroupement (distance moyenne sur les G canaux).
    """
    nc, h, w = noisy.shape
    ph, pw = h - k + 1, w - k + 1
    npos = ph * pw
    d = k * k
    pn = [_flat_patches(noisy[c], k) for c in range(nc)]
    pg = [_flat_patches(guide[g], k) for g in range(guide.shape[0])]
    ni, nj = min(n, ph), min(n, pw)
    off_i, off_j = np.arange(ni), np.arange(nj)
    m_win = ni * nj
    n_keep = min(n_keep, m_win)
    sigma2 = np.float32(sigma * sigma)
    eye = np.eye(d, dtype=np.float32)

    gi, gj = _reference_grid(ph, pw, step)
    starts = list(range(0, gi.size, chunk))
    n_threads = max(1, min(int(n_threads), len(starts)))
    parts = [starts[t::n_threads] for t in range(n_threads)]

    def work(part):
        acc = np.zeros((nc, npos, d))
        cnt = np.zeros(npos)
        for s in part:
            _chunk(s, acc, cnt)
        return acc, cnt

    def _chunk(s, acc, cnt):
        ri, rj = gi[s : s + chunk], gj[s : s + chunk]
        b = ri.size
        i0 = np.clip(ri - ni // 2, 0, ph - ni)
        j0 = np.clip(rj - nj // 2, 0, pw - nj)
        lin = ((i0[:, None] + off_i)[:, :, None] * pw + (j0[:, None] + off_j)[:, None, :]).reshape(b, m_win)
        ref = ri * pw + rj

        dist = np.zeros((b, m_win), dtype=np.float32)
        for p in pg:
            dist += np.mean((p[lin] - p[ref][:, None, :]) ** 2, axis=2)
        dist /= len(pg)

        if second:
            # Seuil τ = max(τ₀, d_N₂) ; tous les candidats sous τ sont gardés.
            d_n = np.partition(dist, n_keep - 1, axis=1)[:, n_keep - 1]
            tau = np.maximum(np.float32(tau0), d_n)
            sizes = (dist <= tau[:, None]).sum(axis=1)
            # Paquets de groupes de taille voisine : chaque paquet est réduit
            # à ses K plus proches candidats (K = puissance de 2 ≥ taille max),
            # ce qui contient exactement les candidats sous τ.
            kb = np.minimum(np.maximum(64, 2 ** np.ceil(np.log2(np.maximum(sizes, 1)))).astype(int), m_win)
            for kval in np.unique(kb):
                rows = np.flatnonzero(kb == kval)
                if kval < m_win:
                    part = np.argpartition(dist[rows], kval - 1, axis=1)[:, :kval]
                else:
                    part = np.broadcast_to(np.arange(m_win), (rows.size, m_win))
                d_sub = np.take_along_axis(dist[rows], part, axis=1)
                idx = np.take_along_axis(lin[rows], part, axis=1)
                mask = d_sub <= tau[rows, None]
                _filter(pn, pg, idx, mask, nc, d, beta, sigma2, eye, gamma, sigma, use_area, True, acc, cnt, npos)
        else:
            sel = np.argpartition(dist, n_keep - 1, axis=1)[:, :n_keep]
            idx = np.take_along_axis(lin, sel, axis=1)
            mask = np.ones(idx.shape, dtype=bool)
            _filter(pn, pg, idx, mask, nc, d, beta, sigma2, eye, gamma, sigma, use_area, False, acc, cnt, npos)

    if n_threads == 1:
        acc, cnt = work(parts[0])
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            results = list(ex.map(work, parts))
        acc = sum(r[0] for r in results)
        cnt = sum(r[1] for r in results)

    out = np.empty_like(noisy, dtype=np.float64)
    for c in range(nc):
        acc_p = acc[c].reshape(ph, pw, d)
        num = np.zeros((h, w))
        den = np.zeros((h, w))
        cnt_p = cnt.reshape(ph, pw)
        for di in range(k):
            for dj in range(k):
                num[di : di + ph, dj : dj + pw] += acc_p[:, :, di * k + dj]
                den[di : di + ph, dj : dj + pw] += cnt_p
        out[c] = np.where(den > 0, num / np.maximum(den, 1e-12), noisy[c])
    return out


def nl_bayes_vectorized(
    u: np.ndarray,
    sigma_255: float,
    params_gray: dict,
    params_color: dict,
    color_mode: str,
    use_area1: bool,
    use_area2: bool,
    return_basic: bool,
    step: int,
    chunk: int,
    n_threads: int = 1,
) -> np.ndarray:
    """
    Cœur vectorisé. ``u`` : image sur l'échelle [0, 255], (H, W) ou
    (H, W, 3). Renvoie le résultat de même forme.
    """
    def run(chans: np.ndarray, guide_first: np.ndarray | None, p: dict):
        g1 = chans if guide_first is None else guide_first
        basic = _step(
            chans, g1, sigma_255, p["k1"], p["n1"], p["N1"], p["beta1"], p["gamma"],
            use_area1, 0.0, False, step, chunk, n_threads,
        )
        return basic

    def second(noisy_c: np.ndarray, basic_c: np.ndarray, p: dict):
        return _step(
            noisy_c, basic_c, sigma_255, p["k2"], p["n2"], p["N2"], p["beta2"], p["gamma"],
            use_area2, p["tau0"], True, step, chunk, n_threads,
        )

    if u.ndim == 2:
        x = u[None]
        basic = run(x, None, params_gray)
        return basic[0] if return_basic else second(x, basic, params_gray)[0]

    rgb = np.moveaxis(u, -1, 0)  # (3, H, W)
    if color_mode == "channel":
        out = np.empty_like(rgb, dtype=np.float64)
        for c in range(rgb.shape[0]):
            x = rgb[c : c + 1]
            basic = run(x, None, params_gray)
            out[c] = basic[0] if return_basic else second(x, basic, params_gray)[0]
        return np.moveaxis(out, 0, -1)

    p = params_color
    if color_mode == "yuv":
        yuv = np.einsum("ij,jhw->ihw", YUV_MATRIX, rgb)
        basic_yuv = run(yuv, yuv[:1], p)  # regroupement sur Y₀ seul
        basic = np.einsum("ij,jhw->ihw", YUV_MATRIX.T, basic_yuv)  # retour RGB
    else:  # "rgb"
        basic = run(rgb, None, p)
    if return_basic:
        return np.moveaxis(basic, 0, -1)
    return np.moveaxis(second(rgb, basic, p), 0, -1)
