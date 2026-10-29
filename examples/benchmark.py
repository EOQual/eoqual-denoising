#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark des méthodes de débruitage disponibles dans l'environnement,
avec paramètres par défaut puis paramètres optimisés.

Protocole (reproductible, graines fixées) :

- image : « astronaut » 512×512 en niveaux de gris, dynamique [0, 1]
  (NASA, domaine public, fournie par scikit-image) ;
- **moitié gauche** (512×256) : image de **réglage** (optimisation) ;
  **moitié droite** (512×256) : image d'**évaluation**. Les deux
  moitiés ont un contenu différent (visage / combinaison, drapeau) et
  des tirages de bruit différents (graines 1 et 2) : le gain mesuré sur
  la moitié droite est un gain de généralisation, pas un sur-apprentissage
  de l'image de réglage ;
- bruit blanc gaussien additif, σ connu et transmis aux méthodes ;
- partie A : paramètres par défaut, PSNR / SSIM / temps sur la moitié
  d'évaluation, pour chaque σ de ``--sigmas`` ;
- partie B : à ``--tune-sigma``, optimisation bayésienne (``--n-calls``
  évaluations) sur la moitié de réglage, puis application des meilleurs
  paramètres à la moitié d'évaluation.

Écrit un rapport Markdown (``--out``) et l'affiche.

Usage ::

    python examples/benchmark.py --out benchmark.md
    python examples/benchmark.py --methods nl_means bm3d drunet --n-calls 20
"""
from __future__ import annotations

import argparse
import logging
import platform
import time
from datetime import date

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import eoqual.denoising as d
from eoqual.denoising.config import METHODS_CONFIGS
from eoqual.denoising.runner import _resolve


def _metrics(ref, img):
    return (
        float(peak_signal_noise_ratio(ref, img, data_range=1.0)),
        float(structural_similarity(ref, np.clip(img, 0, 1), data_range=1.0)),
    )


def _available(method, probe):
    try:
        _resolve(method)(probe, sigma=0.1)
        return True, ""
    except (ImportError, FileNotFoundError) as exc:
        return False, str(exc).splitlines()[0][:90]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=list(METHODS_CONFIGS))
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    p.add_argument("--tune-sigma", type=float, default=0.1)
    p.add_argument("--n-calls", type=int, default=25)
    p.add_argument("--n-initial-points", type=int, default=8)
    p.add_argument("--no-optim", action="store_true", help="partie A seulement")
    p.add_argument("--out", default="benchmark.md")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    full = d.astronaut_gray()  # 512×512
    tune_ref, test_ref = full[:, :256].copy(), full[:, 256:].copy()

    probe = d.add_gaussian_noise(d.astronaut_gray(32), 0.1, seed=0)
    methods, missing = [], {}
    for m in args.methods:
        ok, why = _available(m, probe)
        (methods.append(m) if ok else missing.__setitem__(m, why))

    lines = [
        "# Benchmark — eoqual-denoising",
        "",
        f"Généré par `examples/benchmark.py` le {date.today().isoformat()} "
        f"({platform.machine()}, {platform.system()}, CPU).",
        "",
        "## A. Paramètres par défaut — moitié d'évaluation (512×256)",
        "",
    ]
    header = "| Méthode | " + " | ".join(f"PSNR σ={s:g}" for s in args.sigmas) + " | " + " | ".join(
        f"SSIM σ={s:g}" for s in args.sigmas
    ) + f" | Temps σ={args.sigmas[len(args.sigmas)//2]:g} |"
    lines += [header, "|" + "---|" * (2 * len(args.sigmas) + 2)]
    noisy_row = ["*(image bruitée)*"]
    tests = {s: d.add_gaussian_noise(test_ref, s, seed=2) for s in args.sigmas}
    ps, ss = [], []
    for s in args.sigmas:
        a, b = _metrics(test_ref, tests[s])
        ps.append(f"{a:.2f}")
        ss.append(f"{b:.3f}")
    lines.append("| " + " | ".join(noisy_row + ps + ss + ["—"]) + " |")

    defaults = {}
    for m in methods:
        func = _resolve(m)
        ps, ss, t_mid = [], [], None
        for i, s in enumerate(args.sigmas):
            t0 = time.perf_counter()
            out = func(tests[s], sigma=s)
            dt = time.perf_counter() - t0
            a, b = _metrics(test_ref, out)
            defaults[(m, s)] = a
            ps.append(f"{a:.2f}")
            ss.append(f"{b:.3f}")
            if i == len(args.sigmas) // 2:
                t_mid = dt
        lines.append(f"| `{m}` | " + " | ".join(ps + ss) + f" | {t_mid:.2f} s |")
        print(lines[-1], flush=True)

    if missing:
        lines += ["", "Méthodes indisponibles dans cet environnement :", ""]
        lines += [f"- `{m}` : {why}" for m, why in missing.items()]

    if not args.no_optim:
        from eoqual.denoising.optimization import SEARCH_SPACES, optimize_parameters

        s = args.tune_sigma
        tune_noisy = d.add_gaussian_noise(tune_ref, s, seed=1)
        test_noisy = tests.get(s, d.add_gaussian_noise(test_ref, s, seed=2))
        lines += [
            "",
            f"## B. Paramètres optimisés (σ = {s:g}, {args.n_calls} évaluations, réglage sur la moitié gauche)",
            "",
            "| Méthode | PSNR défaut (éval.) | PSNR optimisé (éval.) | Gain éval. | Gain réglage | Durée optim. | Meilleurs paramètres |",
            "|---|---|---|---|---|---|---|",
        ]
        for m in methods:
            if m not in SEARCH_SPACES:
                lines.append(f"| `{m}` | {defaults.get((m, s), float('nan')):.2f} | — | — | — | — | aucun paramètre réglable |")
                continue
            t0 = time.perf_counter()
            res = optimize_parameters(
                tune_noisy, tune_ref, m, sigma=s, n_calls=args.n_calls,
                n_initial_points=args.n_initial_points, random_state=0, data_range=1.0,
            )
            dt = time.perf_counter() - t0
            opt_test, _ = _metrics(test_ref, _resolve(m)(test_noisy, **res.best_kwargs))
            base = defaults.get((m, s)) or _metrics(test_ref, _resolve(m)(test_noisy, sigma=s))[0]
            params = ", ".join(
                f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in res.best_params.items()
            )
            gain_tune = f"{res.gain:+.2f}" if res.gain is not None else "— (exhaustif)"
            lines.append(
                f"| `{m}` | {base:.2f} | {opt_test:.2f} | {opt_test - base:+.2f} | "
                f"{gain_tune} | {dt:.0f} s | {params} |"
            )
            print(lines[-1], flush=True)

    text = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
