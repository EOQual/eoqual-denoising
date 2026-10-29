#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark de NL-Bayes : moteurs (boucle fidèle à l'article / vectorisé)
en niveaux de gris, et traitements de la couleur.

Protocole : moitié droite (512×256) d'« astronaut » (NASA, domaine
public), bruit blanc gaussien de σ connu (graine 2), identique sur les
trois canaux en couleur. PSNR (dB) et SSIM ; en couleur, PSNR sur les
trois canaux et SSIM moyen par canal.

Partie A (niveaux de gris) : moteur ``loop`` (astuce « paste ») et moteur
vectorisé pour plusieurs pas de grille, temps compris.
Partie B (couleur) : ``color_mode`` = ``channel`` / ``yuv`` / ``rgb``, et
l'exécutable IPOL en référence s'il est disponible (canal par canal).

Usage ::

    python examples/nl_bayes_benchmark.py --out nl_bayes_benchmark.md
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import eoqual.denoising as d


def _scores(ref, img, color):
    p = float(peak_signal_noise_ratio(ref, img, data_range=1.0))
    s = float(structural_similarity(ref, np.clip(img, 0, 1), data_range=1.0, channel_axis=-1 if color else None))
    return p, s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    ap.add_argument("--out", default="nl_bayes_benchmark.md")
    ap.add_argument("--size", type=int, default=512, help="taille de l'image avant découpe (défaut 512 : 512×256)")
    args = ap.parse_args()

    half = args.size // 2
    gray = d.astronaut_gray(args.size)[:, half:].copy()
    rgb = d.astronaut_rgb(args.size)[:, half:, :].copy()
    lines = ["# Benchmark NL-Bayes — moteurs et couleur", ""]

    lines += ["## A. Niveaux de gris — moteurs", "",
              "| Moteur | " + " | ".join(f"PSNR σ={s:g}" for s in args.sigmas) + " | "
              + " | ".join(f"Temps σ={s:g}" for s in args.sigmas) + " |",
              "|" + "---|" * (2 * len(args.sigmas) + 1)]
    variants = [("boucle (article, « paste »)", dict(engine="loop"))]
    variants += [(f"vectorisé, pas {st}", dict(step=st)) for st in (2, 3, 4)]
    variants += [("vectorisé, pas 3, 1 fil", dict(step=3, n_threads=1))]
    for name, kw in variants:
        ps, ts = [], []
        for s in args.sigmas:
            noisy = d.add_gaussian_noise(gray, s, seed=2)
            t0 = time.perf_counter()
            out = d.nl_bayes_denoise(noisy, sigma=s, data_range=(0, 1), **kw)
            ts.append(f"{time.perf_counter() - t0:.2f} s")
            ps.append(f"{_scores(gray, out, False)[0]:.2f}")
        lines.append(f"| {name} | " + " | ".join(ps + ts) + " |")
        print(lines[-1], flush=True)

    lines += ["", "## B. Couleur (RGB)", "",
              "| Traitement | " + " | ".join(f"PSNR σ={s:g}" for s in args.sigmas) + " | "
              + " | ".join(f"SSIM σ={s:g}" for s in args.sigmas) + " | Temps σ=0.1 |",
              "|" + "---|" * (2 * len(args.sigmas) + 2)]
    noisy_c = {s: d.add_gaussian_noise(rgb, s, seed=2) for s in args.sigmas}
    ps = [f"{_scores(rgb, noisy_c[s], True)[0]:.2f}" for s in args.sigmas]
    ss = [f"{_scores(rgb, noisy_c[s], True)[1]:.3f}" for s in args.sigmas]
    lines.append("| *(image bruitée)* | " + " | ".join(ps + ss) + " | — |")

    def ipol_channels(img, s):
        # Dynamique par défaut (min/max) : pas de troncature avant la
        # quantification 8 bits de l'échange avec l'exécutable.
        return np.stack([d.nl_bayes_ipol_denoise(img[..., c], sigma=s) for c in range(3)], -1)

    modes = [
        ("`channel` — canal par canal (paramètres niveaux de gris)", lambda x, s: d.nl_bayes_denoise(x, sigma=s, data_range=(0, 1), color_mode="channel")),
        ("`yuv` — Y₀U₀V₀ puis RGB (article)", lambda x, s: d.nl_bayes_denoise(x, sigma=s, data_range=(0, 1), color_mode="yuv")),
        ("`rgb` — RGB, groupes communs", lambda x, s: d.nl_bayes_denoise(x, sigma=s, data_range=(0, 1), color_mode="rgb")),
        ("IPOL, canal par canal (référence C++)", ipol_channels),
    ]
    for name, fn in modes:
        try:
            ps, ss, t_mid = [], [], "—"
            for s in args.sigmas:
                t0 = time.perf_counter()
                out = fn(noisy_c[s], s)
                dt = time.perf_counter() - t0
                p, q = _scores(rgb, out, True)
                ps.append(f"{p:.2f}")
                ss.append(f"{q:.3f}")
                if abs(s - 0.1) < 1e-9:
                    t_mid = f"{dt:.2f} s"
            lines.append(f"| {name} | " + " | ".join(ps + ss) + f" | {t_mid} |")
        except (ImportError, FileNotFoundError) as exc:
            lines.append(f"| {name} | indisponible : {str(exc).splitlines()[0][:60]} |")
        print(lines[-1], flush=True)

    text = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
