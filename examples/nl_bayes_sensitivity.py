#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Étude de sensibilité des paramètres de NL-Bayes sur des images de
télédétection texturées (NL_BAYES.md §9).

Deux volets :

- ``gray`` : luminance, bruit blanc gaussien σ = 2, 5, 10, 20 (échelle
  0-255), 6 tuiles d'orthophotographie plus camera et astronaut ; un
  paramètre à la fois autour des valeurs de l'article, puis le croisement
  ``sigma_scale`` × β₁ × β₂.
- ``rgb``  : chaîne capteur simulée. Flux linéaire (gamma 2,2 retiré),
  mosaïque RGGB, bruit de Poisson + lecture 3 e⁻, flux crête 300, 1 500 et
  7 500 e⁻, démosaïçage Hamilton-Adams (eoqual-demosaicing), puis
  NL-Bayes couleur avec stabilisation de variance (σ = C = √κ_Y + 0,2) ou
  avec un σ global. Référence : démosaïçage de la mosaïque sans bruit.
  Erreur mesurée dans le domaine d'affichage (gamma 1/2,2), séparément
  dans les zones sombres et claires, en luminance et en couleur. Un
  paramètre à la fois, puis les croisements N₂ × k₂, ``sigma_scale`` × N₂,
  γ × critère homogène de l'étape 2, β₁ × β₂.
- ``sigma_beta`` : même chaîne, stabilisation seulement ; croisement du σ
  transmis (0,8 à 1,4 dans le domaine stabilisé) × β₁ × β₂.
- ``groups`` : taille réelle des groupes de l'étape 2 selon N₂, en zones
  sombres et claires (τ₀ agrandit les groupes des zones plates).

Données : extrait « place du Capitole » de l'**Orthophotoplan 2024 de
Toulouse Métropole**, Licence Ouverte v2.0 (Etalab),
https://data.toulouse-metropole.fr/explore/dataset/orthophotoplan-2024/ —
le même que l'exemple ``orthophoto.py`` d'eoqual-resampling. Fenêtre
utilisée : 4 096 × 4 096 pixels de 5 cm à partir de la ligne 3 072,
colonne 2 902 de ``capitole_10240.tif``, moyennée 4 × 4 (pixels de 20 cm,
bruit d'origine réduit).

Usage ::

    python examples/nl_bayes_sensitivity.py capitole_10240.tif --parts gray rgb --out etude_nlb/
    python examples/nl_bayes_sensitivity.py capitole_10240.tif --figures-only --out etude_nlb/

Sorties : ``results_gray.csv``, ``results_rgb.csv``, ``results_sigma_beta.csv``,
``results_groups.csv``, ``rapport.md``
(tableaux) dans ``--out`` ; figures ``docs/figures/nl_bayes/sens_*.png``.
Durée indicative (8 cœurs) : 4 min (``gray``), 8 min (``rgb``), 2 min (``sigma_beta``),
1 min (``groups``).

Dépendances de l'exemple (non requises par le paquet) : rasterio,
pandas, matplotlib, scikit-image ; eoqual-demosaicing pour ``rgb``.
"""
from __future__ import annotations

import argparse
import os
import csv
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures" / "nl_bayes"
WINDOW = (3072, 2902, 4096)       # ligne, colonne, côté (pixels de 5 cm) dans capitole_10240.tif
REDUCE = 4                        # 5 cm → 20 cm
TILE = 128

# --- volet gray -------------------------------------------------------------
G_BASE = dict(k1=5, k2=5, window_factor=7.0, beta1=1.0, beta2=1.2, gamma=1.05, tau0=4.0,
              use_area1=True, use_area2=False)
G_GRID = {
    "sigma_scale": [0.5, 0.7, 0.85, 1.15, 1.3, 1.5, 2.0],
    "k1": [3, 7], "k2": [3, 7], "N1": [30, 40, 90, 120], "N2": [30, 40, 90, 120],
    "window_factor": [3.0, 5.0, 9.0, 11.0], "beta1": [0.5, 0.75, 1.25, 1.5],
    "beta2": [0.6, 0.9, 1.5, 2.0], "gamma": [0.8, 0.95, 1.15, 1.3],
    "use_area1": [False], "use_area2": [True], "tau0": [0.0, 1.0, 2.0, 8.0, 16.0, 32.0],
}
G_SIGMAS = [2.0, 5.0, 10.0, 20.0]

# --- volet rgb --------------------------------------------------------------
PEAKS = [300.0, 1500.0, 7500.0]
READ = 3.0
C0 = float(np.sqrt(1.35) + 0.2)   # facteur C de Hamilton-Adams (eoqual-demosaicing §11.3)
R_BASE = dict(k1=3, k2=3, N1=30, N2=30, window_factor=7.0, beta1=1.0, beta2=1.2, gamma=1.05,
              tau0=4.0, use_area1=True, use_area2=False, color_mode="yuv")

_TILES: dict = {}


def load_scene(path: Path, window) -> np.ndarray:
    """Charge l'orthophoto (RGB), applique la fenêtre si l'image la contient, moyenne 4 × 4."""
    import rasterio

    with rasterio.open(path) as r:
        h, w = r.height, r.width
        if window and h >= window[0] + window[2] and w >= window[1] + window[2]:
            from rasterio.windows import Window

            rgb = r.read(window=Window(window[1], window[0], window[2], window[2]))
        else:
            rgb = r.read()
    rgb = rgb[:3].astype(np.float64).transpose(1, 2, 0)
    n = (min(rgb.shape[:2]) // REDUCE) * REDUCE
    rgb = rgb[:n, :n]
    return rgb.reshape(n // REDUCE, REDUCE, n // REDUCE, REDUCE, 3).mean((1, 3))


def select_tiles(rgb: np.ndarray) -> tuple[dict, dict]:
    """Tuiles « gray » (les plus texturées) et « rgb » (texturées avec 20-60 % de zones sombres)."""
    from scipy import ndimage
    from skimage import color, data

    y = rgb.mean(-1)
    n = y.shape[0]
    cands = []
    for i in range(0, n - TILE + 1, TILE):
        for j in range(0, n - TILE + 1, TILE):
            t = y[i:i + TILE, j:j + TILE]
            cands.append((np.std(ndimage.laplace(t)), i, j))
    cands.sort()
    gray = {}
    for k, q in enumerate((0.99, 0.93, 0.86, 0.79, 0.72, 0.5)):
        _, i, j = cands[int(q * (len(cands) - 1))]
        gray[f"ortho{k + 1}"] = y[i:i + TILE, j:j + TILE]
    gray["camera"] = data.camera()[128:384:2, 128:384:2].astype(np.float64)
    gray["astronaut"] = (color.rgb2gray(data.astronaut()) * 255)[0:256:2, 128:384:2]

    cands = []
    for i in range(0, n - TILE + 1, TILE // 2):
        for j in range(0, n - TILE + 1, TILE // 2):
            t = y[i:i + TILE, j:j + TILE]
            dark = (t < 60).mean()
            if 0.2 <= dark <= 0.6:
                cands.append((np.std(ndimage.laplace(t)), i, j))
    cands.sort(reverse=True)
    picks = []
    for _, i, j in cands:
        if all(abs(i - a) >= TILE or abs(j - b) >= TILE for a, b in picks):
            picks.append((i, j))
        if len(picks) == 6:
            break
    rgbt = {f"rgb{k + 1}": rgb[i:i + TILE, j:j + TILE] for k, (i, j) in enumerate(picks)}
    return gray, rgbt


def _init(tiles):
    _TILES.update(tiles)


def _psnr(ref, x, peak=255.0):
    return float(10 * np.log10(peak ** 2 / np.mean((ref - x) ** 2)))


def run_gray(job):
    from eoqual.denoising.backends.nl_bayes.native import nl_bayes_denoise

    img, sigma, draw, param, value, extra = job
    ref = _TILES[img]
    noisy = ref + np.random.default_rng(1000 * draw + int(sigma)).normal(0, sigma, ref.shape)
    kw = dict(G_BASE)
    kw.update(extra)
    s = sigma
    if param == "sigma_scale":
        s = sigma * value
    elif param not in (None, "cross"):
        kw[param] = value
    if "sigma_scale" in extra:
        s = sigma * kw.pop("sigma_scale")
    out = nl_bayes_denoise(noisy, sigma=s, data_range=(0, 255), n_threads=1, **kw)
    return dict(image=img, sigma=sigma, draw=draw, param=param or "", value=value, change=json.dumps(extra),
                psnr=_psnr(ref, out))


def gray_jobs(names):
    jobs = [(i, s, d, None, None, {}) for i in names for s in G_SIGMAS for d in (0, 1)]
    jobs += [(i, s, d, p, v, {}) for i in names for s in G_SIGMAS for d in (0, 1)
             for p, vals in G_GRID.items() for v in vals]
    ortho = [n for n in names if n.startswith("ortho")]
    for i, s, d, sc, b1, b2 in itertools.product(ortho, G_SIGMAS, (0, 1), [0.8, 0.85, 0.9, 1.0],
                                                [0.6, 0.75, 1.0], [0.6, 0.9, 1.2]):
        jobs.append((i, s, d, "cross", None, {"sigma_scale": sc, "beta1": b1, "beta2": b2}))
    return jobs


def rgb_variants():
    v = [("base", {})]
    one = {"s": [0.6, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0], "k1": [5, 7], "k2": [5, 7],
           "N1": [15, 60, 90], "N2": [15, 60, 90], "window_factor": [3.0, 5.0, 11.0],
           "beta1": [0.6, 0.8, 1.25, 1.5], "beta2": [0.6, 0.9, 1.5, 2.0],
           "gamma": [0.8, 0.95, 1.15, 1.3, 1.6], "use_area1": [False], "use_area2": [True],
           "tau0": [0.0, 1.0, 16.0, 64.0], "color_mode": ["rgb", "channel"]}
    for p, vals in one.items():
        for x in vals:
            ch = {p: x}
            if p in ("k1", "k2"):
                ch["N" + p[1]] = {3: 30, 5: 60, 7: 90}[x]
            v.append((f"1:{p}", ch))
    for k, n in itertools.product([3, 5, 7], [15, 30, 60, 90, 120]):
        if n >= k * k + 1:
            v.append(("x:N2k2", {"k2": k, "N2": n}))
    for s, n in itertools.product([0.8, 1.0, 1.25, 1.5], [15, 30, 60, 90]):
        v.append(("x:sN2", {"s": s, "N2": n}))
    for g, a in itertools.product([0.95, 1.05, 1.3, 1.6], [False, True]):
        v.append(("x:gamma_area2", {"gamma": g, "use_area2": a}))
    for b1, b2 in itertools.product([0.6, 1.0, 1.5], [0.6, 1.2, 2.0]):
        v.append(("x:beta", {"beta1": b1, "beta2": b2}))
    return v


def disp(x, peak):
    """Domaine d'affichage : gamma 1/2,2 sur [0, 255]."""
    return 255.0 * np.clip(x / peak, 0, 1) ** (1 / 2.2)


def rgb_case(img, peak, draw):
    """Référence (démosaïçage sans bruit) et image démosaïquée bruitée, en e⁻."""
    from eoqual.demosaicing.backends.ha.ha import ha_demosaic

    ref = (_TILES[img] / 255.0) ** 2.2 * peak
    rng = np.random.default_rng([int(img[3:]), int(peak), draw])
    mos = np.zeros(ref.shape[:2])
    mos[0::2, 0::2] = ref[0::2, 0::2, 0]
    mos[1::2, 1::2] = ref[1::2, 1::2, 2]
    mos[0::2, 1::2] = ref[0::2, 1::2, 1]
    mos[1::2, 0::2] = ref[1::2, 0::2, 1]
    noisy = rng.poisson(mos) + rng.normal(0, READ, mos.shape)
    # Référence : démosaïçage sans bruit (« ha », défaut : isole l'effet du débruiteur) ou vraie image
    # (« truth » : inclut l'erreur de démosaïçage) ; choisie par --reference (variable d'environnement,
    # héritée par les processus de calcul).
    target = ref if os.environ.get("NLB_REFERENCE", "ha") == "truth" else ha_demosaic(mos).astype(np.float64)
    return target, ha_demosaic(noisy).astype(np.float64)


def rgb_metrics(ref, out, peak):
    e = disp(out, peak) - disp(ref, peak)
    y = disp(ref, peak).mean(-1)
    ey = e.mean(-1)
    ec = e - ey[..., None]
    r = {"psnr": _psnr(ref, out, peak)}
    for zone, m in (("dark", y < 60), ("bright", y > 120)):
        r[f"Y_{zone}"] = float(np.sqrt(np.mean(ey[m] ** 2)))
        r[f"C_{zone}"] = float(np.sqrt(np.mean(np.sum(ec[m] ** 2, -1) / 3)))
    return r


def run_rgb(job):
    from eoqual.denoising.backends.nl_bayes.native import nl_bayes_denoise
    from eoqual.denoising.vst import generalized_anscombe, inverse_generalized_anscombe

    img, peak, draw, mode, name, ch = job
    ref, dem = rgb_case(img, peak, draw)
    kw = dict(R_BASE)
    kw.update({k: v for k, v in ch.items() if k not in ("s", "sig")})
    s = ch.get("s", 1.0)
    if name == "noisy":
        out = dem
    elif mode == "vst":
        g = generalized_anscombe(dem, 1.0, READ, 0.0)
        sig = ch.get("sig", C0 * s)       # σ transmis, en unités du domaine stabilisé (bruit de la mosaïque = 1)
        out = inverse_generalized_anscombe(nl_bayes_denoise(g, sigma=sig, n_threads=1, **kw), 1.0, READ, 0.0)
    else:
        sigma0 = np.sqrt(READ ** 2 + dem.mean())
        out = nl_bayes_denoise(dem, sigma=C0 * s * sigma0, data_range=(0, peak), n_threads=1, **kw)
    return dict(image=img, peak=peak, draw=draw, mode=mode, name=name, change=json.dumps(ch),
                **rgb_metrics(ref, out, peak))


SB_SIG = [0.8, 0.95, 1.1, 1.25, 1.4]      # σ transmis après stabilisation (√κ_Y ≈ 1,16 ; C ≈ 1,36)
SB_BETA1 = [0.6, 0.8, 1.0]
SB_BETA2 = [0.4, 0.6, 0.8, 1.0, 1.2]


def sigma_beta_jobs(names):
    """Croisement σ × β₁ × β₂ après stabilisation (volet ``sigma_beta``)."""
    return [(i, p, d, "vst", "x:sig_beta", {"sig": s, "beta1": b1, "beta2": b2})
            for i, p, d, s, b1, b2 in itertools.product(names, PEAKS, [0, 1], SB_SIG, SB_BETA1, SB_BETA2)]


def rgb_jobs(names):
    cases = list(itertools.product(names, PEAKS, [0, 1], ["vst", "global"]))
    jobs = [(i, p, d, m, "noisy", {}) for i, p, d, m in cases if m == "vst"]
    jobs += [(i, p, d, m, n, c) for i, p, d, m in cases for n, c in rgb_variants()]
    return jobs


def run_spectral(job):
    """Erreur en zones sombres / claires, séparée en basse fréquence (taches) et haute fréquence (grain)."""
    from scipy import ndimage

    from eoqual.denoising.backends.nl_bayes.native import nl_bayes_denoise
    from eoqual.denoising.vst import generalized_anscombe, inverse_generalized_anscombe

    img, peak = job
    ref, dem = rgb_case(img, peak, 0)
    g = generalized_anscombe(dem, 1.0, READ, 0.0)
    out = inverse_generalized_anscombe(nl_bayes_denoise(g, sigma=C0, n_threads=1, **R_BASE), 1.0, READ, 0.0)
    y = disp(ref, peak).mean(-1)
    rows = []
    for zone, m in (("sombre", y < 60), ("clair", y > 120)):
        for what, x in (("bruité", dem), ("débruité", out)):
            e = disp(x, peak) - disp(ref, peak)
            low = np.stack([ndimage.gaussian_filter(e[..., c], 2.0) for c in range(3)], -1)
            high = e - low
            r = dict(image=img, peak=peak, zone=zone, what=what)
            for part, a in (("bas", low), ("haut", high)):
                ay = a.mean(-1)
                r[f"Y_{part}"] = float(np.sqrt(np.mean(ay[m] ** 2)))
                r[f"C_{part}"] = float(np.sqrt(np.mean(((a - ay[..., None])[m] ** 2).sum(-1) / 3)))
            rows.append(r)
    n = dem - ref
    corr = [float(np.corrcoef(n[:, :-1, c].ravel(), n[:, 1:, c].ravel())[0, 1]) for c in range(3)]
    for r in rows:
        r.update(corr_R=corr[0], corr_G=corr[1], corr_B=corr[2])
    return rows


def run_groups(job):
    """
    Taille réelle des groupes de l'étape 2 (patchs à distance ≤ max(τ₀, d_N₂) de l'estimée de base) selon N₂, en
    zones sombres et claires : canal Y₀ = (R + G + B)/√3 du domaine stabilisé, échelle [0, 255], k₂ = 3, n₂ = 21.
    """
    from eoqual.denoising.backends.nl_bayes.native import nl_bayes_denoise
    from eoqual.denoising.utils import to_unit_range
    from eoqual.denoising.vst import generalized_anscombe

    img, peak = job
    k, n, tau0, sig = 3, 21, 4.0, 1.1
    ref, dem = rgb_case(img, peak, 0)
    u, _, sc = to_unit_range(generalized_anscombe(dem, 1.0, READ, 0.0))
    y0 = (255.0 * u).sum(-1) / np.sqrt(3)
    basic = nl_bayes_denoise(y0, sigma=255.0 * sig / sc, data_range=(0.0, 255.0), k1=k, N1=30,
                             return_basic=True, n_threads=1)
    lum = disp(ref, peak).mean(-1)
    h, w = basic.shape
    pat = np.lib.stride_tricks.sliding_window_view(basic, (k, k)).reshape(h - k + 1, w - k + 1, k * k)
    rows = []
    for i in range(0, h - k + 1, 4):
        for j in range(0, w - k + 1, 4):
            v = lum[i + 1, j + 1]
            zone = "sombre" if v < 60 else ("clair" if v > 120 else None)
            if zone is None:
                continue
            win = pat[max(0, i - n // 2):i + n // 2 + 1, max(0, j - n // 2):j + n // 2 + 1].reshape(-1, k * k)
            d = np.sort(np.mean((win - pat[i, j]) ** 2, axis=1))
            for n2 in (15, 30, 60, 90):
                size = int(np.sum(d <= max(tau0, d[min(n2, d.size) - 1])))
                rows.append(dict(image=img, peak=peak, zone=zone, N2=n2, taille=size))
    return rows


def run_all(fn, jobs, tiles, path, n_jobs):
    t0 = time.time()
    with ProcessPoolExecutor(n_jobs, initializer=_init, initargs=(tiles,)) as ex, open(path, "w", newline="") as f:
        w = None
        for k, r in enumerate(ex.map(fn, jobs, chunksize=4)):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(r))
                w.writeheader()
            w.writerow(r)
            if k % 500 == 0:
                print(f"  {path.name} : {k}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"  {path.name} : {len(jobs)} appels, {time.time() - t0:.0f} s")


# --- analyse et figures -----------------------------------------------------
INK, INK2 = "#0b0b0b", "#52514e"


def _diverging():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])


def _md(df, index=True) -> str:
    """Tableau Markdown simple (sans dépendance à tabulate)."""
    d = df.reset_index() if index else df
    cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in d.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in d.itertuples(index=False):
        rows.append("| " + " | ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in r) + " |")
    return "\n".join(rows)


def _heat(ax, data, rows, cols, title, fmt="{:+.2f}", lim=None, annot=None):
    lim = lim or max(abs(np.nanmin(data)), abs(np.nanmax(data)), 1e-9)
    ax.imshow(data, cmap=_diverging(), vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, fontsize=7.5)
    ax.set_yticks(range(len(rows)), rows, fontsize=7.5)
    ax.set_title(title, fontsize=8.5, color=INK)
    annot = data if annot is None else annot
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(annot[i, j]):
                ax.text(j, i, fmt.format(annot[i, j]), ha="center", va="center", fontsize=6.5, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def analyse_gray(out: Path, lines: list):
    import pandas as pd

    df = pd.read_csv(out / "results_gray.csv")
    base = df[df.param.isna()].set_index(["image", "sigma", "draw"]).psnr
    v = df[df.param.notna() & (df.param != "cross")].copy()
    v["d"] = v.psnr.values - base.loc[list(zip(v.image, v.sigma, v.draw))].values
    v["grp"] = np.where(v.image.str.startswith("ortho"), "ortho", "naturel")
    m = v.groupby(["grp", "image", "sigma", "param", "value"]).d.mean().reset_index()
    best = m.groupby(["grp", "sigma", "param", "value"]).d.mean().reset_index()
    lines.append("## Volet gray : gain du meilleur réglage testé (dB, moyenne des tuiles)\n")
    for grp in ("ortho", "naturel"):
        t = best[best.grp == grp].groupby(["param", "sigma"]).d.max().unstack()
        lines.append(f"### {grp}\n\n" + _md(t.round(2)) + "\n")
    t = best[best.grp == "ortho"].pivot_table(index=["param", "value"], columns="sigma", values="d")
    lines.append("### ortho, écart par valeur (dB)\n\n" + _md(t.round(2)) + "\n")
    c = df[df.param == "cross"].copy()
    c["kw"] = c.change.map(json.loads)
    for k in ("sigma_scale", "beta1", "beta2"):
        c[k] = c.kw.map(lambda x, k=k: x[k])
    cm = c.groupby(["sigma", "sigma_scale", "beta1", "beta2"]).psnr.mean().reset_index()
    rows = []
    for s, g in cm.groupby("sigma"):
        ref = g[(g.sigma_scale == 1) & (g.beta1 == 1) & (g.beta2 == 1.2)].psnr.iloc[0]
        only = g[(g.beta1 == 1) & (g.beta2 == 1.2)].psnr.max() - ref
        b = g.loc[g.psnr.idxmax()]
        rows.append(dict(sigma=s, sigma_scale_seul=only, sigma_scale_et_beta=b.psnr - ref,
                         meilleur=f"s={b.sigma_scale}, β₁={b.beta1}, β₂={b.beta2}"))
    lines.append("### Croisement sigma_scale × β₁ × β₂ (tuiles ortho)\n\n"
                 + _md(pd.DataFrame(rows).round(3), index=False) + "\n")
    return best


def fig_gray(best):
    import matplotlib.pyplot as plt

    order = ["sigma_scale", "N2", "k2", "beta2", "beta1", "k1", "window_factor", "gamma",
             "use_area1", "use_area2", "N1", "tau0"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, grp, title in zip(axes, ("ortho", "naturel"), ("Tuiles d'orthophoto texturées", "camera, astronaut")):
        g = best[best.grp == grp]
        lo = g.groupby(["param", "sigma"]).d.min().unstack().reindex(order)
        hi = g.groupby(["param", "sigma"]).d.max().unstack().reindex(order)
        data = np.where(-lo.values > hi.values, lo.values, hi.values)
        _heat(ax, np.clip(data, -3, 3), order, [f"σ={s:g}" for s in lo.columns],
              f"{title}\nécart le plus grand (dB) parmi les valeurs testées (borné à ±3)", lim=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "sens_gray.png", dpi=130)
    plt.close(fig)


def analyse_sigma_beta(out: Path, lines: list):
    """σ × β₁ × β₂ après stabilisation : gain par rapport au réglage de l'article (σ = C, β₁ = 1, β₂ = 1,2)."""
    import pandas as pd

    df = pd.read_csv(out / "results_sigma_beta.csv")
    ch = df.change.map(json.loads)
    for k in ("sig", "beta1", "beta2"):
        df[k] = ch.map(lambda c, k=k: c[k])
    met = ["psnr", "Y_dark", "C_dark", "Y_bright", "C_bright"]
    m = df.groupby(["peak", "sig", "beta1", "beta2"])[met].mean().reset_index()
    rgb = pd.read_csv(out / "results_rgb.csv")
    ref = rgb[(rgb["mode"] == "vst") & (rgb.name == "base")].groupby("peak")[met].mean()
    for c in met[1:]:
        m[c] = 100 * (m[c] / m.peak.map(ref[c]) - 1)
    m["psnr"] = m.psnr - m.peak.map(ref.psnr)
    best = m.loc[m.groupby(["peak", "sig", "beta1"]).psnr.idxmax()]
    lines.append("## σ × β₁ × β₂ après stabilisation : meilleur β₂ par (σ, β₁)\n\n"
                 "Écarts au réglage de l'article (σ = C = 1,36, β₁ = 1, β₂ = 1,2) : PSNR en dB, erreurs en %.\n\n"
                 + _md(best.round(3).set_index(["peak", "sig", "beta1"])) + "\n")
    top = m.loc[m.groupby("peak").psnr.idxmax()]
    lines.append("### Meilleur triplet par flux\n\n" + _md(top.round(3).set_index("peak")) + "\n")
    m.to_csv(out / "sigma_beta_mean.csv", index=False)


def analyse_rgb(out: Path, lines: list):
    import pandas as pd

    df = pd.read_csv(out / "results_rgb.csv")
    key = ["image", "peak", "draw", "mode"]
    met = ["psnr", "Y_dark", "C_dark", "Y_bright", "C_bright"]
    noisy = df[df.name == "noisy"].drop(columns="mode")
    base = df[df.name == "base"]
    b = base.merge(noisy, on=["image", "peak", "draw"], suffixes=("", "_n"))
    t = b.groupby(["mode", "peak"])[[m for m in met] + [m + "_n" for m in met]].mean()
    lines.append("## Volet rgb : erreur avant / après débruitage (niveaux d'affichage, PSNR en dB)\n\n"
                 + _md(t.round(2).T) + "\n")
    v = df[~df.name.isin(["noisy", "base"])].merge(base[key + met], on=key, suffixes=("", "_b"))
    v["dpsnr"] = v.psnr - v.psnr_b
    for m in met[1:]:
        v["d" + m] = 100 * (v[m] / v[m + "_b"] - 1)
    v["label"] = v.change.map(lambda s: ", ".join(f"{k}={x}" for k, x in json.loads(s).items()))
    cols = ["dpsnr", "dY_dark", "dC_dark", "dY_bright"]
    for mode in ("vst", "global"):
        one = v[(v["mode"] == mode) & v.name.str.startswith("1:")]
        t = one.groupby(["label", "peak"])[cols].mean().unstack("peak")
        lines.append(f"### {mode} : un paramètre à la fois (ΔPSNR dB ; Δ erreur en %)\n\n" + _md(t.round(2)) + "\n")
        for x in ("x:N2k2", "x:sN2", "x:gamma_area2", "x:beta"):
            g = v[(v["mode"] == mode) & (v.name == x)].groupby(["label", "peak"])[["dpsnr", "dY_dark", "dC_dark"]].mean()
            lines.append(f"#### {mode} {x}\n\n" + _md(g.unstack("peak").round(2)) + "\n")
    return v


def fig_rgb(v):
    import matplotlib.pyplot as plt

    labels = ["s=0.8", "s=0.9", "s=1.1", "s=1.25", "beta2=0.6", "beta2=1.5", "beta1=0.6", "beta1=1.5",
              "N2=15", "N2=90", "N1=15", "N1=90", "k2=5, N2=60", "k1=5, N1=60", "window_factor=3.0",
              "window_factor=11.0", "gamma=0.8", "gamma=1.3", "use_area1=False", "use_area2=True",
              "tau0=0.0", "tau0=16.0", "color_mode=rgb", "color_mode=channel"]
    one = v[(v["mode"] == "vst") & v.name.str.startswith("1:")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 7.2), sharey=True)
    for ax, (col, title, lim) in zip(axes, (("dpsnr", "ΔPSNR (dB), image entière", 0.6),
                                            ("dY_dark", "Δ erreur de luminance,\nzones sombres (%)", 20),
                                            ("dC_dark", "Δ erreur de couleur,\nzones sombres (%)", 20))):
        t = one.groupby(["label", "peak"])[col].mean().unstack("peak").reindex(labels)
        sign = 1 if col == "dpsnr" else -1          # bleu = mieux dans les trois panneaux
        _heat(ax, sign * t.values, labels, [f"{p:g} e⁻" for p in t.columns], title, lim=lim,
              fmt="{:+.2f}" if col == "dpsnr" else "{:+.0f}", annot=t.values)
    fig.suptitle("Image RGB démosaïquée (HA), stabilisation de variance : effet d'un paramètre à la fois "
                 "(bleu = mieux, orange = moins bien)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "sens_rgb.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    x = v[(v["mode"] == "vst") & (v.name == "x:N2k2")].copy()
    x["k2"] = x.change.map(lambda s: json.loads(s)["k2"])
    x["N2"] = x.change.map(lambda s: json.loads(s)["N2"])
    for ax, peak in zip(axes, (300.0, 1500.0)):
        t = x[x.peak == peak].groupby(["k2", "N2"]).dpsnr.mean().unstack("N2")
        _heat(ax, t.values, [f"k₂={k}" for k in t.index], [f"N₂={n}" for n in t.columns],
              f"ΔPSNR (dB) face à k₂=3, N₂=30 — {peak:g} e⁻", lim=0.4)
    fig.tight_layout()
    fig.savefig(FIG / "sens_rgb_n2k2.png", dpi=130)
    plt.close(fig)


def fig_tiles(gray, rgbt):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 8, figsize=(13, 3.9))
    for a, (k, t) in zip(axes[0], gray.items()):
        a.imshow(t, cmap="gray", vmin=0, vmax=255)
        a.set_title(k, fontsize=8)
    for a, (k, t) in zip(axes[1], rgbt.items()):
        a.imshow(np.clip(t, 0, 255).astype(np.uint8))
        a.set_title(k, fontsize=8)
    for a in axes.ravel():
        a.axis("off")
    fig.text(0.5, 0.005, "Orthophotoplan 2024 © Toulouse Métropole, Licence Ouverte v2.0 (Etalab) ; "
             "camera et astronaut : scikit-image", ha="center", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "sens_tiles.png", dpi=110)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path, help="orthophoto RGB (capitole_10240.tif ou extrait 4096 × 4096)")
    p.add_argument("--parts", nargs="+", choices=("gray", "rgb", "sigma_beta", "groups"), default=["gray", "rgb", "sigma_beta", "groups"])
    p.add_argument("--out", type=Path, default=Path("etude_nl_bayes"))
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--reference", choices=("ha", "truth"), default="ha",
                   help="volets rgb et sigma_beta : référence = démosaïçage sans bruit (ha) ou vraie image (truth)")
    p.add_argument("--figures-only", action="store_true", help="ne recalcule pas, relit les CSV")
    args = p.parse_args()
    os.environ["NLB_REFERENCE"] = args.reference

    import matplotlib

    matplotlib.use("Agg")
    args.out.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    gray, rgbt = select_tiles(load_scene(args.image, WINDOW))
    fig_tiles(gray, rgbt)
    if not args.figures_only:
        if "gray" in args.parts:
            run_all(run_gray, gray_jobs(list(gray)), gray, args.out / "results_gray.csv", args.n_jobs)
        if "rgb" in args.parts:
            run_all(run_rgb, rgb_jobs(list(rgbt)), rgbt, args.out / "results_rgb.csv", args.n_jobs)
            with ProcessPoolExecutor(args.n_jobs, initializer=_init, initargs=(rgbt,)) as ex:
                rows = [r for rs in ex.map(run_spectral, list(itertools.product(rgbt, PEAKS))) for r in rs]
            import pandas as pd

            pd.DataFrame(rows).to_csv(args.out / "results_spectral.csv", index=False)
        if "sigma_beta" in args.parts:
            run_all(run_rgb, sigma_beta_jobs(list(rgbt)), rgbt, args.out / "results_sigma_beta.csv", args.n_jobs)
        if "groups" in args.parts:
            import pandas as pd

            with ProcessPoolExecutor(args.n_jobs, initializer=_init, initargs=(rgbt,)) as ex:
                rows = [r for rs in ex.map(run_groups, list(itertools.product(rgbt, PEAKS))) for r in rs]
            pd.DataFrame(rows).to_csv(args.out / "results_groups.csv", index=False)
    lines = ["# Étude de sensibilité de NL-Bayes\n"]
    if (args.out / "results_gray.csv").exists():
        fig_gray(analyse_gray(args.out, lines))
    if (args.out / "results_rgb.csv").exists():
        fig_rgb(analyse_rgb(args.out, lines))
    if (args.out / "results_spectral.csv").exists():
        import pandas as pd

        s = pd.read_csv(args.out / "results_spectral.csv")
        t = s.groupby(["zone", "peak", "what"])[["Y_bas", "Y_haut", "C_bas", "C_haut"]].mean()
        lines.append("## Volet rgb : erreur basse / haute fréquence (niveaux d'affichage)\n\n" + _md(t.round(2)) + "\n")
        c = s[["corr_R", "corr_G", "corr_B"]].mean()
        lines.append(f"Corrélation du bruit démosaïqué entre voisins horizontaux : R {c.corr_R:.2f}, "
                     f"G {c.corr_G:.2f}, B {c.corr_B:.2f}\n")
    if (args.out / "results_sigma_beta.csv").exists():
        analyse_sigma_beta(args.out, lines)
    if (args.out / "results_groups.csv").exists():
        import pandas as pd

        g = pd.read_csv(args.out / "results_groups.csv")
        g["agrandi"] = g.taille > g.N2
        t = g.groupby(["peak", "zone", "N2"]).agg(taille_moyenne=("taille", "mean"), part_agrandie=("agrandi", "mean"))
        lines.append("## Taille réelle des groupes de l'étape 2 (τ₀ = 4)\n\n" + _md(t.round(2)) + "\n")
    (args.out / "rapport.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"rapport : {args.out / 'rapport.md'} ; figures : {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
