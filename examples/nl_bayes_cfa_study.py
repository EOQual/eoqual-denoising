#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NL-Bayes CFA contre la chaîne « démosaïçage puis NL-Bayes couleur »
(docs/technique/nl-bayes.md §13.1).

Même chaîne capteur que le volet ``rgb`` de ``nl_bayes_sensitivity.py`` :
scène linéarisée (gamma 2,2), mosaïque RGGB, bruit de Poisson + lecture
3 e⁻, flux crête 300, 1 500 et 7 500 e⁻, 2 tirages. Toutes les chaînes
stabilisent la variance (GAT) et inversent à la fin.

Trois jeux de tuiles 128 × 128 (``--tiles``), tous avec 20 à 60 % de zones
sombres :

- ``texture`` : les 6 tuiles d'orthophoto **les plus texturées** (celles de
  ``nl_bayes_sensitivity.py``) ;
- ``lisse``   : les 6 tuiles d'orthophoto **les moins texturées** ;
- ``synth``   : 4 scènes synthétiques de grandes zones uniformes (cellules de
  Voronoï, moitié sombres 15-55, moitié claires 80-220 en niveaux
  d'affichage), texture légère ±2 niveaux.

Volets (``--parts``) :

- ``grid``     : nl_bayes_cfa, pilote bilinéaire, N ∈ {30, 60, 90, 120} ×
  passes ∈ {2, 4, 8} (réglage par défaut de N et du nombre de passes) ;
- ``chains``   : comparaison des chaînes ci-dessous ;
- ``flatness`` : part des blocs sombres plats (s²/σ² des mesures).

Chaînes comparées :

- ``ha+nlb (C)``      : Hamilton-Adams, NL-Bayes couleur en Y₀U₀V₀ (mode de
  l'article), σ = C = √κ_Y + 0,2, β₂ = 1,2 (recette d'eoqual-demosaicing) ;
- ``ha+nlb (réglé)``  : idem, σ = 1,1, β₂ = 0,9 (docs/technique/nl-bayes.md §11.6) ;
- ``cfa bil. n p.``   : nl_bayes_cfa sur la mosaïque stabilisée (σ = 1),
  pilote bilinéaire + NL-Bayes, n passes (1, 2, 4, 8, 12), réglages par défaut ;
- ``cfa bil. β=0,7`` / ``β=1,4`` : sensibilité à β ;
- ``cfa HA n p.``     : pilote = sortie de ``ha+nlb (réglé)`` (le temps
  indiqué n'inclut pas celui du pilote) ;
- ``+zh`` : critère des zones homogènes, seuil statistique z (1, 2, 3) ou
  γ = 1,05 fixe ; groupe plat remplacé par la moyenne des mesures de chaque
  canal, ou par la moyenne du groupe pilote (« pilote »).

Contrôles sans bruit : Hamilton-Adams, bilinéaire, et nl_bayes_cfa
(σ très petit, pilote bilinéaire seul ou Hamilton-Adams) appliqués à la
mosaïque sans bruit.

Erreurs mesurées contre la **vraie** image RGB (pas contre un démosaïçage
sans bruit, qui avantagerait Hamilton-Adams) : PSNR (domaine linéaire),
erreurs de luminance et de couleur dans les zones sombres et claires en
niveaux d'affichage (gamma 1/2,2), et, dans les zones sombres, séparation
grain / taches (filtre gaussien d'écart-type 2 pixels).

Usage ::

    python examples/nl_bayes_cfa_study.py capitole_10240.tif --out etude_cfa/
    python examples/nl_bayes_cfa_study.py capitole_10240.tif --parts grid --tiles lisse synth

Dépendances de l'exemple : eoqual-demosaicing (Hamilton-Adams), rasterio,
pandas. Durée indicative (8 cœurs, trois jeux) : 3 min (``grid``), 10 min
(``chains``).
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nl_bayes_sensitivity as S  # noqa: E402  (chaîne capteur et métriques communes)

PEAKS = S.PEAKS
READ = S.READ
PASSES = (1, 2, 4, 8, 12)
CHAINS = (["ha+nlb (C)", "ha+nlb (réglé)"] + [f"cfa bil. {n} p." for n in PASSES]
          + ["cfa bil. β=0,7", "cfa bil. β=1,4"] + [f"cfa HA {n} p." for n in PASSES])
CONTROLS = ["HA sans bruit", "bilinéaire sans bruit", "cfa bil. sans bruit", "cfa HA sans bruit"]
#: Critère des zones homogènes (réglages par défaut, pilote bilinéaire).
AREA = [("cfa bil. +zh z=1", dict(use_area=True, area_z=1.0)),
        ("cfa bil. +zh z=2", dict(use_area=True, area_z=2.0)),
        ("cfa bil. +zh z=3", dict(use_area=True, area_z=3.0)),
        ("cfa bil. +zh γ=1,05", dict(use_area=True, area_gamma=1.05)),
        ("cfa bil. +zh z=2 pilote", dict(use_area=True, area_z=2.0, area_mean="pilot"))]
CHAINS += [name for name, _ in AREA]


GRID_N = (30, 60, 90, 120)
GRID_PASSES = (2, 4, 8)
SETS = ("texture", "lisse", "synth")


def tile_set(name: str) -> str:
    """Jeu d'une tuile d'après son numéro : rgb1-9 texturées, rgb11-19 lisses, rgb21+ synthétiques."""
    k = int(name[3:])
    return "texture" if k < 11 else ("lisse" if k < 21 else "synth")


def smooth_tiles(rgb: np.ndarray, n: int = 6) -> dict:
    """Tuiles les MOINS texturées (écart-type du laplacien), 20 à 60 % de zones sombres, disjointes."""
    from scipy import ndimage

    tile = S.TILE
    y = rgb.mean(-1)
    cands = []
    for i in range(0, y.shape[0] - tile + 1, tile // 2):
        for j in range(0, y.shape[1] - tile + 1, tile // 2):
            t = y[i:i + tile, j:j + tile]
            if 0.2 <= (t < 60).mean() <= 0.6:
                cands.append((float(np.std(ndimage.laplace(t))), i, j))
    cands.sort()
    picks = []
    for _, i, j in cands:
        if all(abs(i - a) >= tile or abs(j - b) >= tile for a, b in picks):
            picks.append((i, j))
        if len(picks) == n:
            break
    return {f"rgb{11 + k}": rgb[i:i + tile, j:j + tile] for k, (i, j) in enumerate(picks)}


def synthetic_tile(seed: int) -> np.ndarray:
    """Zones uniformes (Voronoï, 14 cellules), moitié sombres, moitié claires ; texture ±2 niveaux d'affichage."""
    from scipy import ndimage

    tile = S.TILE
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, tile, (14, 2))
    yy, xx = np.mgrid[:tile, :tile]
    lab = np.argmin((yy[..., None] - pts[:, 0]) ** 2 + (xx[..., None] - pts[:, 1]) ** 2, axis=-1)
    cols = np.where(rng.random((14, 1)) < 0.5, rng.uniform(15, 55, (14, 3)), rng.uniform(80, 220, (14, 3)))
    tex = ndimage.gaussian_filter(rng.standard_normal((tile, tile, 3)), (1.5, 1.5, 0))
    return np.clip(cols[lab] + 2.0 * tex / tex.std(), 0, 255)


def load_tiles(image: Path, sets) -> dict:
    rgb = S.load_scene(image, S.WINDOW)
    tiles = {}
    if "texture" in sets:
        tiles.update(S.select_tiles(rgb)[1])
    if "lisse" in sets:
        tiles.update(smooth_tiles(rgb))
    if "synth" in sets:
        tiles.update({f"rgb{21 + k}": synthetic_tile(k) for k in range(4)})
    return tiles


def case(img, peak, draw):
    """Vérité, mosaïque sans bruit et mosaïque bruitée (e⁻), mêmes tirages que nl_bayes_sensitivity."""
    truth = (S._TILES[img] / 255.0) ** 2.2 * peak
    rng = np.random.default_rng([int(img[3:]), int(peak), draw])
    mos = np.zeros(truth.shape[:2])
    mos[0::2, 0::2] = truth[0::2, 0::2, 0]
    mos[1::2, 1::2] = truth[1::2, 1::2, 2]
    mos[0::2, 1::2] = truth[0::2, 1::2, 1]
    mos[1::2, 0::2] = truth[1::2, 0::2, 1]
    noisy = rng.poisson(mos) + rng.normal(0, READ, mos.shape)
    return truth, mos, noisy


def dark_split(truth, out, peak):
    """Erreur des zones sombres séparée en taches (basse fréquence) et grain."""
    from scipy import ndimage

    e = S.disp(out, peak) - S.disp(truth, peak)
    m = S.disp(truth, peak).mean(-1) < 60
    low = np.stack([ndimage.gaussian_filter(e[..., c], 2.0) for c in range(3)], -1)
    r = {}
    for part, a in (("taches", low), ("grain", e - low)):
        ay = a.mean(-1)
        r[f"Y_{part}"] = float(np.sqrt(np.mean(ay[m] ** 2)))
        r[f"C_{part}"] = float(np.sqrt(np.mean(((a - ay[..., None])[m] ** 2).sum(-1) / 3)))
    return r


def run(job):
    from eoqual.demosaicing.backends.ha.ha import ha_demosaic
    from eoqual.denoising import nl_bayes_cfa_denoise, nl_bayes_denoise
    from eoqual.denoising.backends.nl_bayes.cfa import bilinear_demosaic
    from eoqual.denoising.vst import generalized_anscombe, inverse_generalized_anscombe

    img, peak, draw = job
    truth, mos, noisy = case(img, peak, draw)
    fwd = lambda x: generalized_anscombe(x, 1.0, READ, 0.0)  # noqa: E731
    inv = lambda x: inverse_generalized_anscombe(x, 1.0, READ, 0.0)  # noqa: E731
    rows = []

    def add(name, out, seconds):
        rows.append(dict(image=img, set=tile_set(img), peak=peak, draw=draw, chain=name, seconds=seconds,
                         **S.rgb_metrics(truth, out, peak), **dark_split(truth, out, peak)))

    g_ha = fwd(ha_demosaic(noisy).astype(np.float64))
    g_mos = fwd(noisy)
    t = time.time()
    add("ha+nlb (C)", inv(nl_bayes_denoise(g_ha, sigma=S.C0, n_threads=1, **S.R_BASE)), time.time() - t)
    t = time.time()
    tuned = nl_bayes_denoise(g_ha, sigma=1.1, n_threads=1, **{**S.R_BASE, "beta2": 0.9})
    add("ha+nlb (réglé)", inv(tuned), time.time() - t)
    variants = [(f"cfa bil. {n} p.", {"n_iter": n}) for n in PASSES]
    variants += [("cfa bil. β=0,7", {"beta": 0.7}), ("cfa bil. β=1,4", {"beta": 1.4})]
    variants += [(f"cfa HA {n} p.", {"pilot": tuned, "n_iter": n}) for n in PASSES]
    variants += AREA
    for name, kw in variants:
        t = time.time()
        out = nl_bayes_cfa_denoise(g_mos, sigma=1.0, n_threads=1, **kw)
        add(name, inv(out), time.time() - t)
    if draw == 0:
        add("HA sans bruit", ha_demosaic(mos).astype(np.float64), 0.0)
        add("bilinéaire sans bruit", bilinear_demosaic(mos), 0.0)
        s0 = 0.01 * peak / 255
        add("cfa bil. sans bruit", nl_bayes_cfa_denoise(mos, sigma=s0, data_range=(0, peak), pilot="none",
                                                         n_threads=1), 0.0)
        add("cfa HA sans bruit", nl_bayes_cfa_denoise(mos, sigma=s0, data_range=(0, peak),
                                                       pilot=ha_demosaic(mos).astype(np.float64), n_threads=1), 0.0)
    return rows


def run_grid(job):
    """nl_bayes_cfa, pilote bilinéaire : grille N × passes (une passe de plus réutilise les précédentes)."""
    from eoqual.denoising import nl_bayes_cfa_denoise
    from eoqual.denoising.vst import generalized_anscombe, inverse_generalized_anscombe

    img, peak, draw = job
    truth, _, noisy = case(img, peak, draw)
    g = generalized_anscombe(noisy, 1.0, READ, 0.0)
    rows = []
    for n_keep in GRID_N:
        cur, done, elapsed = "bilinear", 0, 0.0
        for passes in GRID_PASSES:
            t = time.time()
            cur = nl_bayes_cfa_denoise(g, sigma=1.0, N=n_keep, n_iter=passes - done, pilot=cur, n_threads=1)
            elapsed += time.time() - t
            done = passes
            out = inverse_generalized_anscombe(cur, 1.0, READ, 0.0)
            rows.append(dict(image=img, set=tile_set(img), peak=peak, draw=draw, N=n_keep, passes=passes,
                             seconds=elapsed, **S.rgb_metrics(truth, out, peak), **dark_split(truth, out, peak)))
    return rows


def flatness(rgbt):
    """
    Les zones sombres sont-elles plates ? s²/σ² des mesures de la mosaïque stabilisée, canal par
    canal, sur des blocs 9 × 9 (σ = 1), avec et sans bruit ; part des blocs sous 1,10 (seuil γ² de NL-Bayes).
    """
    from eoqual.denoising.backends.nl_bayes.cfa import _channel_map
    from eoqual.denoising.vst import generalized_anscombe

    rows = []
    for peak in PEAKS:
        for img in rgbt:
            truth, mos, noisy = case(img, peak, 0)
            g, g0 = (generalized_anscombe(x, 1.0, READ, 0.0) for x in (noisy, mos))
            cm = _channel_map(*g.shape, "RGGB")
            y = S.disp(truth, peak).mean(-1)
            for i in range(0, g.shape[0] - 9, 9):
                for j in range(0, g.shape[1] - 9, 9):
                    c = cm[i:i + 9, j:j + 9]
                    v = [sum(((b[c == k] - b[c == k].mean()) ** 2).sum() for k in range(3)) / 78
                         for b in (g[i:i + 9, j:j + 9], g0[i:i + 9, j:j + 9])]
                    rows.append(dict(set=tile_set(img), peak=peak, zone="sombre" if y[i:i + 9, j:j + 9].mean() < 60 else "autre",
                                     s2=v[0], signal=v[1], plat=v[0] <= 1.10))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path, help="orthophoto RGB (capitole_10240.tif ou extrait 4096 × 4096)")
    p.add_argument("--out", type=Path, default=Path("etude_cfa"))
    p.add_argument("--tiles", nargs="+", choices=SETS, default=list(SETS))
    p.add_argument("--parts", nargs="+", choices=("grid", "chains", "flatness"), default=["grid", "chains", "flatness"])
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()
    import pandas as pd

    args.out.mkdir(parents=True, exist_ok=True)
    tiles = load_tiles(args.image, args.tiles)
    S._init(tiles)                      # tuiles aussi dans le processus principal
    jobs = list(itertools.product(tiles, PEAKS, [0, 1]))
    lines = ["# NL-Bayes CFA contre démosaïçage puis NL-Bayes\n",
             f"Jeux : {', '.join(args.tiles)} ({len(tiles)} tuiles), {len(jobs)} cas.\n"]
    sets = [x for x in SETS if x in args.tiles]

    if "grid" in args.parts:
        t0 = time.time()
        with ProcessPoolExecutor(args.n_jobs, initializer=S._init, initargs=(tiles,)) as ex:
            rows = [r for rs in ex.map(run_grid, jobs) for r in rs]
        gdf = pd.DataFrame(rows)
        gdf.to_csv(args.out / "results_grid.csv", index=False)
        lines.append(f"## Grille N × passes (pilote bilinéaire, {time.time() - t0:.0f} s)\n")
        for st in sets:
            t = gdf[gdf.set == st].pivot_table(index=["N", "passes"], columns="peak", values="psnr")
            lines.append(f"### PSNR, jeu {st}\n\n" + S._md(t.round(2)) + "\n")
            t = gdf[gdf.set == st].pivot_table(index=["N", "passes"], columns="peak", values=["Y_dark", "C_dark"])
            lines.append(f"### Zones sombres (luminance, couleur), jeu {st}\n\n" + S._md(t.round(2)) + "\n")
        t = gdf.groupby(["N", "passes"]).seconds.mean()
        lines.append("### Temps par tuile et par fil (s, passes cumulées)\n\n" + S._md(t.round(2).to_frame()) + "\n")

    if "chains" in args.parts:
        t0 = time.time()
        with ProcessPoolExecutor(args.n_jobs, initializer=S._init, initargs=(tiles,)) as ex:
            rows = [r for rs in ex.map(run, jobs) for r in rs]
        df = pd.DataFrame(rows)
        df.to_csv(args.out / "results_cfa.csv", index=False)
        met = ["psnr", "Y_dark", "C_dark", "Y_bright", "C_bright", "Y_taches", "C_taches", "Y_grain", "C_grain",
               "seconds"]
        lines.append(f"## Chaînes ({time.time() - t0:.0f} s)\n")
        for st in sets:
            for peak in PEAKS:
                t = (df[(df.set == st) & (df.peak == peak)].groupby("chain")[met].mean()
                     .reindex(CHAINS + CONTROLS).dropna(how="all"))
                lines.append(f"### Jeu {st}, flux crête {peak:g} e⁻\n\n" + S._md(t.round(3)) + "\n")
        # Gain de PSNR apparié (même tuile, même tirage) sur la chaîne de référence ha+nlb (C).
        ref = df[df.chain == "ha+nlb (C)"].set_index(["image", "peak", "draw"]).psnr
        g = df[df.chain.str.startswith("cfa") & ~df.chain.str.contains("sans bruit")].copy()
        g["gain"] = g.psnr.values - ref.loc[list(zip(g.image, g.peak, g.draw))].values
        for st in sets:
            t = g[g.set == st].pivot_table(index="chain", columns="peak", values="gain", aggfunc=["mean", "min"])
            t = t.reindex([c for c in CHAINS if c.startswith("cfa")]).dropna(how="all")
            lines.append(f"### Gain de PSNR sur ha+nlb (C), apparié (moyenne, minimum), jeu {st}\n\n"
                         + S._md(t.round(2)) + "\n")

    if "flatness" in args.parts:
        f = pd.DataFrame(flatness(tiles))
        t = f.groupby(["set", "peak", "zone"]).agg(blocs=("s2", "size"), s2_mediane=("s2", "median"),
                                                   signal_mediane=("signal", "median"), part_plate=("plat", "mean"))
        lines.append("## Platitude des zones (s²/σ² des mesures, blocs 9 × 9, mosaïque stabilisée)\n\n"
                     + S._md(t.round(2)) + "\n")
    (args.out / "rapport.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"rapport : {args.out / 'rapport.md'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
