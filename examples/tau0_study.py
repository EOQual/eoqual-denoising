#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Étude de sensibilité du seuil τ₀ de NL-Bayes (docs/technique/nl-bayes/etudes.md §11.3).

Pour chaque image et chaque niveau de bruit, débruite avec une grille de
τ₀ (niveaux² sur l'échelle 0-255) et compare au réglage de l'article
(τ₀ = 4) et à une règle relative τ₀ = c·σ². Trois volets :

- ``gray``  : bruit blanc gaussien, σ = 5 à 60 (échelle 0-255) ;
- ``vst``   : bruit poisson-gaussien (gain 2 e⁻/DN, lecture 0,5 DN,
  offset 100 DN), flux crête de 5 à 400 e⁻, via ``vst_denoise`` ; σ = 1
  dans le domaine stabilisé, ramené sur l'échelle 0-255 ;
- ``color`` : images RGB, mode ``yuv``, σ = 10 à 40.

Images : jeu ``fit`` (camera, astronaut, coffee, brick, grass, moon) ou
``validation`` (coins, gravel, rocket, chelsea, text), recadrées en
256×256 au plus (scikit-image, sans téléchargement). Écrit un rapport
Markdown : par cas, PSNR à τ₀ = 4, avec la règle, et au meilleur τ₀ de la
grille ; puis la perte moyenne et maximale de chaque réglage.

Usage ::

    python examples/tau0_study.py --images validation --parts gray vst color --out tau0.md
    python examples/tau0_study.py --images fit --parts gray --c 0.005 --seed 1

Durée indicative en CPU : 5 à 8 min par volet et par jeu d'images.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
from skimage import color, data, util
from skimage.metrics import peak_signal_noise_ratio as psnr

import eoqual.denoising as d

SETS = {
    "fit": ["camera", "astronaut", "coffee", "brick", "grass", "moon"],
    "validation": ["coins", "gravel", "rocket", "chelsea", "text"],
}
TAU = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
K, RN, OFF = 2.0, 0.5, 100.0


def _crop(img, gray, n=256):
    if gray and img.ndim == 3:
        img = color.rgb2gray(img)
    img = util.img_as_float(img)
    h, w = img.shape[:2]
    n = min(n, h, w)
    return img[(h - n) // 2:(h - n) // 2 + n, (w - n) // 2:(w - n) // 2 + n]


def _cases(part, names, seed, size):
    """Génère (libellé, σ sur l'échelle 0-255, fonction τ₀ → PSNR)."""
    for name in names:
        raw_img = getattr(data, name)()
        if part == "color":
            if raw_img.ndim != 3:
                continue
            c = _crop(raw_img, gray=False, n=size)
            for s255 in (10, 20, 40):
                noisy = d.add_gaussian_noise(c, s255 / 255, seed=seed)
                yield f"{name} σ={s255}", s255, lambda t, c=c, n=noisy, s=s255 / 255: psnr(
                    c, d.nl_bayes_denoise(n, sigma=s, data_range=(0, 1), color_mode="yuv", tau0=t), data_range=1)
            continue
        c = _crop(raw_img, gray=True, n=size)
        if part == "gray":
            for s255 in (5, 10, 20, 30, 40, 60):
                noisy = d.add_gaussian_noise(c, s255 / 255, seed=seed)
                yield f"{name} σ={s255}", s255, lambda t, c=c, n=noisy, s=s255 / 255: psnr(
                    c, d.nl_bayes_denoise(n, sigma=s, data_range=(0, 1), tau0=t), data_range=1)
        else:  # vst
            for peak in (5, 20, 100, 400):
                sig = c * peak
                raw = d.add_poisson_gaussian_noise(sig, K, RN, OFF, seed=seed)
                g = d.generalized_anscombe(raw, K, RN, OFF)
                s255 = 255.0 / (g.max() - g.min())
                yield f"{name} {peak} e⁻", s255, lambda t, r=raw, ref=sig / K + OFF, p=peak: psnr(
                    ref, d.vst_denoise(r, "nl_bayes", gain=K, read_noise_dn=RN, offset_dn=OFF, tau0=t),
                    data_range=p / K)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", choices=sorted(SETS), default="validation")
    ap.add_argument("--parts", nargs="+", choices=("gray", "vst", "color"), default=["gray", "vst", "color"])
    ap.add_argument("--c", type=float, default=0.005, help="constante de la règle τ₀ = c·σ²")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="tau0_study.md")
    ap.add_argument("--size", type=int, default=256, help="côté maximal des images recadrées")
    ap.add_argument("--max-cases", type=int, default=None, help="nombre maximal de cas par volet (essai rapide)")
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    lines = [f"# Étude de τ₀ — images « {args.images} », graine {args.seed}, règle τ₀ = {args.c:g}·σ²", ""]
    for part in args.parts:
        lines += [f"## {part}", "", "| Cas | σ (0-255) | τ₀ = 4 | règle | meilleur τ₀ (grille) | PSNR au meilleur |",
                  "|---|---|---|---|---|---|"]
        l4, lr, diff = [], [], []
        for i, (label, s255, run) in enumerate(_cases(part, SETS[args.images], args.seed, args.size)):
            if args.max_cases is not None and i >= args.max_cases:
                break
            grid = {t: run(t) for t in TAU}
            rule = run(args.c * s255 ** 2)
            best_t = max(grid, key=grid.get)
            best = max(grid[best_t], rule)
            l4.append(best - grid[4]); lr.append(best - rule); diff.append(rule - grid[4])
            lines.append(f"| {label} | {s255:.1f} | {grid[4]:.2f} | {rule:.2f} | {best_t} | {grid[best_t]:.2f} |")
            print(lines[-1], flush=True)
        lines += ["", f"Perte face au meilleur : τ₀ = 4 moyenne {np.mean(l4):.3f} / max {max(l4):.3f} dB ; "
                      f"règle moyenne {np.mean(lr):.3f} / max {max(lr):.3f} dB. "
                      f"Règle − τ₀ = 4 : moyenne {np.mean(diff):+.3f} dB, de {min(diff):+.3f} à {max(diff):+.3f}.", ""]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"rapport : {args.out}")


if __name__ == "__main__":
    main()
