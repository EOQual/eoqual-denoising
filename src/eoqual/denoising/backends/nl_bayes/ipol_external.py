#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NL-Bayes, implémentation de référence IPOL (Lebrun, 2013), appelée comme
programme externe. Pour l'implémentation intégrée (MIT, sans dépendance),
voir ``native.py`` (méthode ``nl_bayes``).

Appel, comme **programme externe** (sous-processus), de l'exécutable
``NL_Bayes`` de l'implémentation de référence IPOL (Lebrun, 2013,
**GPL-3.0-or-later**), compilé par l'utilisateur — voir
``THIRD_PARTY_LICENSES.md`` §2.3. Aucun code ni binaire NL-Bayes n'est
embarqué dans ce paquet ; les deux programmes communiquent uniquement par
fichiers PNG.

Mise en place ::

    # https://www.ipol.im/pub/art/2013/16/ — archive nl-bayes_20130617.tar.gz
    # dépendances : libpng, fftw3 ; OpenMP optionnel (make OMP=1)
    make
    export EOQUAL_NLBAYES_EXE=/chemin/nl-bayes_20130617/NL_Bayes

Références :
    Buades, A., Lebrun, M., Morel, J.-M. (2013). "A Nonlocal Bayesian
    Image Denoising Algorithm." SIAM J. Imaging Sci. 6(3), 1665-1688.
    Lebrun, M., Buades, A., Morel, J.-M. (2013). "Implementation of the
    'Non-Local Bayes' (NL-Bayes) Image Denoising Algorithm." Image
    Processing On Line 3, 1-42. https://doi.org/10.5201/ipol.2013.16
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ...utils import check_image, from_unit_range, resolve_sigma, to_unit_range

__all__ = ["nl_bayes_ipol_denoise", "ENV_VAR"]

ENV_VAR = "EOQUAL_NLBAYES_EXE"

# Ordre des sorties attendu par la ligne de commande IPOL (main.cpp) :
# image sigma add_noise noisy denoised basic difference bias basic_bias
# diff_bias useArea1 useArea2 computeBias
_OUTPUTS = ("noisy", "denoised", "basic", "diff", "bias", "basic_bias", "diff_bias")


def _find_executable(nlbayes_exe: str | os.PathLike | None) -> str:
    exe = nlbayes_exe or os.environ.get(ENV_VAR) or shutil.which("NL_Bayes")
    if not exe or not Path(exe).is_file():
        raise FileNotFoundError(
            "nl_bayes_ipol_denoise nécessite l'exécutable IPOL 'NL_Bayes' "
            "(GPL-3.0+, non embarqué — voir THIRD_PARTY_LICENSES.md). "
            "Le compiler depuis https://www.ipol.im/pub/art/2013/16/ puis "
            f"fournir nlbayes_exe=... ou définir la variable {ENV_VAR}."
        )
    return str(exe)


def nl_bayes_ipol_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    nlbayes_exe: str | os.PathLike | None = None,
    data_range: tuple[float, float] | None = None,
    use_area1: bool = True,
    use_area2: bool = False,
    return_basic: bool = False,
    timeout: float | None = None,
) -> np.ndarray:
    """
    Débruite une image par NL-Bayes (exécutable IPOL externe).

    L'image est ramenée sur [0, 255] et **quantifiée sur 8 bits** pour
    l'échange (le programme IPOL ne lit que du PNG 8 bits) : l'erreur de
    quantification (écart-type ~0.29 niveau sur 255) s'ajoute au bruit.

    Parameters
    ----------
    image : np.ndarray
        Image 2D bruitée.
    sigma : float, optional
        Écart-type du bruit (unités de l'image). Estimé si absent.
    nlbayes_exe : str or path, optional
        Chemin de l'exécutable ``NL_Bayes``. Par défaut, la variable
        d'environnement ``EOQUAL_NLBAYES_EXE``, puis ``NL_Bayes`` dans le
        ``PATH``.
    data_range : tuple(float, float), optional
        Dynamique ``(min, max)`` du capteur ramenée à [0, 255]. Par
        défaut, le min/max de l'image.
    use_area1, use_area2 : bool
        Critère des zones homogènes à la 1re / 2e étape (voir l'article
        IPOL, §2.3). Par défaut ``True`` / ``False``.
    return_basic : bool
        Renvoyer l'estimée de 1re étape au lieu de l'estimée finale. Par
        défaut ``False``.
    timeout : float, optional
        Durée maximale d'exécution, en secondes.

    Returns
    -------
    np.ndarray
        Image débruitée, ``float64``, même forme que l'entrée.

    Raises
    ------
    FileNotFoundError
        Si l'exécutable est introuvable.
    RuntimeError
        Si le programme externe échoue.
    """
    import cv2

    exe = _find_executable(nlbayes_exe)
    img = check_image(image)
    s = resolve_sigma(img, sigma)
    unit, lo, scale = to_unit_range(img, data_range)
    sigma_255 = 255.0 * s / scale

    with tempfile.TemporaryDirectory(prefix="eoqual_nlbayes_") as tmp:
        tmp = Path(tmp)
        src = tmp / "input.png"
        cv2.imwrite(str(src), np.clip(np.rint(255.0 * unit), 0, 255).astype(np.uint8))
        outs = {name: tmp / f"{name}.png" for name in _OUTPUTS}
        cmd = [
            exe, str(src), f"{sigma_255:.6g}", "0",
            *(str(outs[name]) for name in _OUTPUTS),
            str(int(use_area1)), str(int(use_area2)), "0",
        ]
        proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout, check=False)
        result = outs["basic" if return_basic else "denoised"]
        if proc.returncode != 0 or not result.is_file():
            raise RuntimeError(
                f"NL_Bayes a échoué (code {proc.returncode}) :\n{proc.stdout}\n{proc.stderr}"
            )
        den = cv2.imread(str(result), cv2.IMREAD_GRAYSCALE)

    return from_unit_range(den.astype(np.float64) / 255.0, lo, scale)
