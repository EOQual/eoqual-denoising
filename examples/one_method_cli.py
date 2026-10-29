#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Débruite une image (fichier) avec une méthode choisie.

Usage ::

    python examples/one_method_cli.py entree.tif sortie.tif --method nl_bayes
    python examples/one_method_cli.py entree.tif sortie.tif --method bm3d --sigma 12.5
    python examples/one_method_cli.py couleur.png sortie.png --method nl_bayes --color-mode yuv
    python examples/one_method_cli.py brute.tif sortie.tif --method drunet --vst-gain 2 --vst-read-noise 1.5 --vst-offset 100
    python examples/one_method_cli.py --list

- Sans ``--sigma``, l'écart-type du bruit est estimé sur l'image.
- Une image couleur est traitée en couleur par ``nl_bayes`` (mode
  ``--color-mode``) ; pour les autres méthodes, elle est convertie en
  niveaux de gris.
- ``--vst-gain`` (et ``--vst-read-noise``, ``--vst-offset``, en DN) :
  image brute de capteur, débruitée par ``vst_denoise`` (stabilisation de
  variance + méthode + inverse exacte).

Lecture et écriture par OpenCV (TIFF, PNG...). Sortie en ``float32`` pour
un TIFF, sinon tronquée et convertie au type de l'entrée.
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from eoqual.denoising import vst_denoise
from eoqual.denoising.config import METHODS_CONFIGS
from eoqual.denoising.runner import denoise, list_methods

COLOR_METHODS = {"nl_bayes"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--method", default="nl_bayes", choices=list(METHODS_CONFIGS))
    parser.add_argument("--sigma", type=float, default=None, help="écart-type du bruit (unités de l'image)")
    parser.add_argument("--color-mode", default="yuv", choices=("yuv", "rgb", "channel"), help="nl_bayes, image couleur")
    parser.add_argument("--vst-gain", type=float, default=None, help="gain K (e⁻/DN) : active la stabilisation de variance")
    parser.add_argument("--vst-read-noise", type=float, default=0.0, help="bruit de lecture (DN)")
    parser.add_argument("--vst-offset", type=float, default=0.0, help="niveau d'obscurité (DN)")
    parser.add_argument("--list", action="store_true", help="lister les méthodes et quitter")
    args = parser.parse_args()

    if args.list:
        list_methods()
        return 0
    if not (args.input and args.output):
        parser.error("input et output sont requis (ou --list)")

    raw = cv2.imread(args.input, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if raw is None:
        print(f"Impossible de lire {args.input}", file=sys.stderr)
        return 1
    if raw.ndim == 3:
        raw = raw[..., :3]
        if args.method in COLOR_METHODS and args.vst_gain is None:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        else:
            print(f"{args.method} : image couleur convertie en niveaux de gris.")
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    img = raw.astype(np.float64)
    kwargs = {"color_mode": args.color_mode} if img.ndim == 3 else {}

    if args.vst_gain is not None:
        print(f"méthode={args.method} + stabilisation de variance (K={args.vst_gain:g} e⁻/DN)")
        out = vst_denoise(img, args.method, gain=args.vst_gain, read_noise_dn=args.vst_read_noise,
                          offset_dn=args.vst_offset)
    else:
        if args.sigma is not None:
            kwargs["sigma"] = args.sigma
        print(f"méthode={args.method} sigma={args.sigma if args.sigma is not None else 'estimé'}")
        out = denoise(img, args.method, **kwargs)

    if out.ndim == 3:
        out = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_RGB2BGR)
    if args.output.lower().endswith((".tif", ".tiff")):
        cv2.imwrite(args.output, np.asarray(out, dtype=np.float32))
    elif np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        cv2.imwrite(args.output, np.clip(np.rint(out), info.min, info.max).astype(raw.dtype))
    else:
        cv2.imwrite(args.output, np.asarray(out, dtype=raw.dtype))
    print(f"écrit : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
