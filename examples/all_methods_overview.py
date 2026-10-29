#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparaison de toutes les méthodes disponibles dans l'environnement sur
l'image « astronaut » (NASA, domaine public, fournie par scikit-image),
bruitée par un bruit blanc gaussien connu.

Affiche un tableau PSNR / SSIM / temps par méthode et enregistre une
planche visuelle. Les méthodes dont l'extra ou le composant externe est
absent sont signalées et ignorées.

Usage ::

    python examples/all_methods_overview.py --sigma 0.1 --size 256 --out overview.png
    python examples/all_methods_overview.py --methods nl_means nl_bayes nl_ridge
"""
from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import eoqual.denoising as d
from eoqual.denoising.config import METHODS_CONFIGS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sigma", type=float, default=0.1, help="écart-type du bruit (image dans [0, 1])")
    parser.add_argument("--size", type=int, default=256, help="taille de l'image de test")
    parser.add_argument("--out", default="overview.png", help="planche de sortie")
    parser.add_argument("--methods", nargs="+", default=list(METHODS_CONFIGS), help="méthodes à comparer (défaut : toutes)")
    args = parser.parse_args()

    clean = d.astronaut_gray(args.size)
    noisy = d.add_gaussian_noise(clean, args.sigma, seed=0)

    console = Console()
    table = Table(title=f"astronaut {args.size}x{args.size}, sigma={args.sigma}")
    for col in ("Méthode", "PSNR (dB)", "SSIM", "Temps (s)", "Note"):
        table.add_column(col)
    table.add_row(
        "bruitée",
        f"{peak_signal_noise_ratio(clean, noisy, data_range=1):.2f}",
        f"{structural_similarity(clean, noisy, data_range=1):.3f}",
        "—",
        "",
    )

    results = {"bruitée": noisy}
    for method in args.methods:
        cfg = METHODS_CONFIGS[method]
        t0 = time.perf_counter()
        try:
            out = d.denoise(noisy, method=method, sigma=args.sigma)
        except (ImportError, FileNotFoundError) as exc:
            table.add_row(method, "—", "—", "—", f"indisponible : {str(exc).splitlines()[0][:60]}")
            continue
        dt = time.perf_counter() - t0
        results[method] = out
        table.add_row(
            method,
            f"{peak_signal_noise_ratio(clean, out, data_range=1):.2f}",
            f"{structural_similarity(clean, np.clip(out, 0, 1), data_range=1):.3f}",
            f"{dt:.2f}",
            cfg["restriction"] or "",
        )
    console.print(table)

    n = len(results) + 1
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, img) in zip(axes, [("référence", clean), *results.items()]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=100)
    console.print(f"Planche enregistrée : {args.out}")


if __name__ == "__main__":
    main()
