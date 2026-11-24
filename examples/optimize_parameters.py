#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recherche des paramètres optimaux d'une méthode de débruitage par
optimisation bayésienne, contre une image de référence.

Sans image fournie, utilise « astronaut » (NASA, domaine public) bruitée
par un bruit blanc gaussien de σ = ``--test-sigma`` — en couleur pour une
méthode couleur (``nl_bayes``).

Sorties dans ``--output-dir`` : journal ``<méthode>_log.csv`` (une ligne
par évaluation), ``<méthode>_best.json`` (meilleurs paramètres et
arguments réutilisables), image débruitée ``<méthode>_denoised.tif``
(float32), et figures de diagnostic (sauf ``--no-plot``).

Exemples ::

    python examples/optimize_parameters.py --method nl_means --n-calls 30
    python examples/optimize_parameters.py --method bm3d --reference ref.tif --noisy noisy.tif --n-jobs 4
    python examples/optimize_parameters.py --method drunet --vst-gain 2 --vst-read-noise 1.5 --vst-offset 100 \\
        --reference ref_dn.tif --noisy raw_dn.tif
    python examples/optimize_parameters.py --method nl_means --search-space mon_espace.jsonc
    # image démosaïquée (Hamilton-Adams, κ_Y = 1,35), couleur, stabilisation avec (a, b) en DN :
    python examples/optimize_parameters.py --method nl_bayes --reference ref_rgb.tif --noisy demos_rgb.tif \\
        --vst-gain 2 --vst-read-noise 1.5 --vst-offset 100 --kappa-y 1.35
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import eoqual.denoising as d
from eoqual.denoising.optimization import SEARCH_SPACES, default_n_jobs, load_search_space, optimize_parameters
from eoqual.denoising.optimization.plotting import plot_images, plot_optimization


def _read(path: Path, color: bool) -> np.ndarray:
    """Lit une image ; couleur (RGB) si ``color`` et si le fichier a 3 canaux, sinon niveaux de gris."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise SystemExit(f"Impossible de lire {path}")
    if img.ndim == 3:
        img = img[..., :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if color else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float64)


def _write(path: Path, img: np.ndarray) -> None:
    out = img.astype(np.float32)
    if out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=sorted(SEARCH_SPACES))
    p.add_argument("--reference", type=Path, help="image de référence (propre)")
    p.add_argument("--noisy", type=Path, help="image bruitée, mêmes dimensions")
    p.add_argument("--test-sigma", type=float, default=0.1, help="σ du bruit simulé sans image fournie")
    p.add_argument("--sigma", type=float, default=None, help="σ du bruit (unités de l'image) ; estimé sinon")
    p.add_argument("--search-space", type=Path, default=None, help="espace de recherche JSON/JSONC")
    p.add_argument("--metric", choices=("psnr", "ssim"), default="psnr")
    p.add_argument("--n-calls", type=int, default=30)
    p.add_argument("--n-initial-points", type=int, default=10)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=1, help=f"évaluations parallèles (machine : {default_n_jobs()} cœurs)")
    p.add_argument("--vst-gain", type=float, default=None, help="gain K (e⁻/DN) : active la GAT")
    p.add_argument("--vst-read-noise", type=float, default=0.0, help="bruit de lecture (DN)")
    p.add_argument("--vst-offset", type=float, default=0.0, help="niveau d'obscurité (DN)")
    p.add_argument("--sigma-scale0", type=float, default=None, help="point de départ de sigma_scale")
    p.add_argument("--kappa-y", type=float, default=None,
                   help="image démosaïquée : κ_Y du démosaïqueur ; départ sigma_scale = √κ_Y + 0.2")
    p.add_argument("--output-dir", type=Path, default=Path("output_optim"))
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from eoqual.denoising.config import COLOR_METHODS

    color = args.method in COLOR_METHODS
    if args.reference and args.noisy:
        reference, noisy = _read(args.reference, color), _read(args.noisy, color)
    elif args.reference or args.noisy:
        raise SystemExit("--reference et --noisy vont ensemble.")
    else:
        reference = d.astronaut_rgb(256) if color else d.astronaut_gray(256)
        noisy = d.add_gaussian_noise(reference, args.test_sigma, seed=0)
    sigma_scale0 = args.sigma_scale0
    if args.kappa_y is not None:
        sigma_scale0 = float(np.sqrt(args.kappa_y) + 0.2)
        print(f"image démosaïquée : départ sigma_scale = √{args.kappa_y:g} + 0.2 = {sigma_scale0:.3f}")
        # Sans VST, σ sert au débruitage ; estimé sur l'image démosaïquée, il
        # vaut environ la moitié du vrai σ (docs/technique/reference.md §6.6).
        if args.sigma is None and args.vst_gain is None:
            logging.warning(
                "image démosaïquée sans --sigma : σ estimé sur l'image, biaisé vers le bas "
                "(environ ×0,5 après Hamilton-Adams). Fournir --sigma (modèle de bruit du "
                "capteur ou estimation sur la mosaïque brute) ou stabiliser avec --vst-gain."
            )

    vst = None
    if args.vst_gain is not None:
        vst = {"gain": args.vst_gain, "read_noise_dn": args.vst_read_noise, "offset_dn": args.vst_offset}

    space = load_search_space(args.search_space) if args.search_space else None
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    log_csv = out / f"{args.method}_log.csv"
    log_csv.unlink(missing_ok=True)

    t0 = time.perf_counter()
    res = optimize_parameters(
        noisy,
        reference,
        args.method,
        sigma=args.sigma,
        search_space=space,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial_points,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        metric=args.metric,
        vst=vst,
        sigma_scale0=sigma_scale0,
        log_csv=log_csv,
    )
    elapsed = time.perf_counter() - t0
    gain = f" (départ {res.initial_score:.4f}, gain {res.gain:+.4f})" if res.initial_score is not None else ""
    print(f"\n{args.method} : meilleur {args.metric} = {res.best_score:.4f}{gain} en {elapsed:.0f} s")
    print(f"paramètres : {res.best_params}")
    print(f"arguments réutilisables : {res.best_kwargs}")

    # Image finale avec les meilleurs paramètres.
    if vst:
        kwargs = dict(res.best_kwargs)
        noise_factor = kwargs.pop("sigma")  # σ dans le domaine stabilisé (facteur C)
        denoised = d.vst_denoise(noisy, args.method, **vst, noise_factor=noise_factor, **kwargs)
        print(f"noise_factor (facteur C) retenu : {noise_factor:.3f}")
    else:
        denoised = d.denoise(noisy, args.method, **res.best_kwargs)
    _write(out / f"{args.method}_denoised.tif", denoised)
    (out / f"{args.method}_best.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "metric": args.metric,
                "best_score": res.best_score,
                "initial_score": res.initial_score,
                "best_params": res.best_params,
                "best_kwargs": res.best_kwargs,
                "n_evaluations": len(res.history),
                "seconds": round(elapsed, 1),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    if not args.no_plot:
        plot_optimization(res, out_dir=out)
        plot_images(reference, noisy, denoised, title=f"{args.method} optimisé", out_path=out / f"{args.method}_images.png")
    print(f"sorties : {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
