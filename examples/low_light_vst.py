#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Débruitage à faible flux : bruit poisson-gaussien, avec et sans
stabilisation de variance (transformée d'Anscombe généralisée).

Simule une image brute (DN) à partir d'« astronaut » ramenée à un flux
crête donné (e⁻), avec gain, bruit de lecture et offset choisis, puis
compare pour chaque débruiteur :

- l'application directe (bruit supposé gaussien stationnaire, σ estimé) ;
- GAT + débruiteur (σ = 1) + inverse exacte non biaisée / asymptotique /
  algébrique.

Usage ::

    python examples/low_light_vst.py --peak 5 --gain 2 --read-noise 0.25 --methods nl_means bm3d drunet
"""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table
from skimage.metrics import peak_signal_noise_ratio

import eoqual.denoising as d
from eoqual.denoising.vst import INVERSES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--peak", type=float, default=5.0, help="flux crête, en e⁻")
    parser.add_argument("--gain", type=float, default=2.0, help="gain K, en e⁻/DN")
    parser.add_argument("--read-noise", type=float, default=0.25, help="bruit de lecture, en DN")
    parser.add_argument("--offset", type=float, default=100.0, help="niveau d'obscurité, en DN")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--methods", nargs="+", default=["nl_means", "tv_chambolle"])
    args = parser.parse_args()

    signal_e = d.astronaut_gray(args.size) * args.peak
    raw = d.add_poisson_gaussian_noise(signal_e, args.gain, args.read_noise, args.offset, seed=0)
    ref = signal_e / args.gain + args.offset

    def score(x):
        return f"{peak_signal_noise_ratio(ref, x, data_range=args.peak / args.gain):.2f}"

    console = Console()
    table = Table(
        title=(
            f"pic {args.peak:g} e⁻, K = {args.gain:g} e⁻/DN, "
            f"σ_lecture = {args.read_noise:g} DN ({args.gain * args.read_noise:g} e⁻) — PSNR (dB)"
        )
    )
    table.add_column("Méthode")
    table.add_column("Direct")
    for inv in INVERSES:
        table.add_column(f"GAT + {inv}")
    table.add_row("(image brute)", score(raw), "", "", "")

    for method in args.methods:
        try:
            row = [score(d.denoise(raw, method=method))]
            for inv in INVERSES:
                out = d.vst_denoise(
                    raw, method, gain=args.gain, read_noise_dn=args.read_noise, offset_dn=args.offset, inverse=inv
                )
                row.append(score(out))
        except (ImportError, FileNotFoundError) as exc:
            row = [f"indisponible : {str(exc).splitlines()[0][:50]}", "", "", ""]
        table.add_row(method, *row)
    console.print(table)


if __name__ == "__main__":
    main()
