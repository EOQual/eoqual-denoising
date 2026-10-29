#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NL-Ridge, code des auteurs (Inria, AGPL-3.0), appelé depuis un clone
externe. Pour l'implémentation intégrée (MIT, sans dépendance), voir
``native.py`` (méthode ``nl_ridge``).

Appel d'une copie **externe**, fournie par l'utilisateur, du module
``nlridge.py`` du dépôt https://github.com/sherbret/NL-Ridge (Inria,
**AGPL-3.0**) — voir ``THIRD_PARTY_LICENSES.md`` §2.3. Aucun code de
NL-Ridge n'est embarqué ni installé par ce paquet.

Pourquoi un chemin plutôt qu'une dépendance pip : le ``setup.cfg`` amont
déclare ``packages = find:`` alors que le code est un module isolé
(``nlridge.py``) — ``pip install`` du dépôt n'installe aucun module
importable (vérifié le 2026-10-28).

Mise en place ::

    git clone https://github.com/sherbret/NL-Ridge.git /chemin/NL-Ridge
    export EOQUAL_NLRIDGE_PATH=/chemin/NL-Ridge
    pip install "eoqual-denoising[deep]"   # torch + einops

Références :
    Herbreteau, S., Kervrann, C. (2022). "Towards a Unified View of
    Unsupervised Non-Local Methods for Image Denoising: The NL-Ridge
    Approach." Proc. IEEE ICIP, 3376-3380.
    Herbreteau, S., Kervrann, C. (2025). "A Unified Framework of Nonlocal
    Parametric Methods for Image Denoising." SIAM J. Imaging Sci. 18(1),
    89-119.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np

from ...utils import check_image, from_unit_range, resolve_sigma, to_unit_range

__all__ = ["nl_ridge_inria_denoise", "ENV_VAR"]

ENV_VAR = "EOQUAL_NLRIDGE_PATH"

# Paramètres recommandés par les auteurs pour un bruit blanc gaussien,
# par palier de sigma exprimé sur une échelle [0, 255] (docstring de
# NLRidge.forward dans le dépôt amont).
_RECOMMENDED = (
    (15.0, dict(p1=7, p2=7, k1=18, k2=55)),
    (35.0, dict(p1=9, p2=9, k1=18, k2=90)),
    (float("inf"), dict(p1=11, p2=9, k1=20, k2=120)),
)

_MODULE_CACHE: dict[str, object] = {}


def _load_nlridge(nlridge_path: str | os.PathLike | None):
    path = nlridge_path or os.environ.get(ENV_VAR)
    if not path:
        raise ImportError(
            "nl_ridge_inria_denoise nécessite une copie locale de NL-Ridge "
            "(AGPL-3.0, non embarquée — voir THIRD_PARTY_LICENSES.md). "
            "Cloner https://github.com/sherbret/NL-Ridge puis fournir "
            f"nlridge_path=... ou définir la variable {ENV_VAR}."
        )
    path = Path(path)
    file = path / "nlridge.py" if path.is_dir() else path
    if not file.is_file():
        raise ImportError(f"Fichier nlridge.py introuvable : {file}")
    key = str(file.resolve())
    if key not in _MODULE_CACHE:
        try:
            import torch  # noqa: F401
            import einops  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "NL-Ridge nécessite 'torch' et 'einops'. Installer avec : "
                'pip install "eoqual-denoising[deep]"'
            ) from exc
        spec = importlib.util.spec_from_file_location("_eoqual_ext_nlridge", file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE_CACHE[key] = module
    return _MODULE_CACHE[key]


def nl_ridge_inria_denoise(
    image: np.ndarray,
    sigma: float | None = None,
    nlridge_path: str | os.PathLike | None = None,
    data_range: tuple[float, float] | None = None,
    device: str | None = None,
    **nlridge_kwargs,
) -> np.ndarray:
    """
    Débruite une image par NL-Ridge (bruit blanc gaussien homoscédastique).

    Parameters
    ----------
    image : np.ndarray
        Image 2D bruitée.
    sigma : float, optional
        Écart-type du bruit (unités de l'image). Estimé si absent.
    nlridge_path : str or path, optional
        Dossier du clone NL-Ridge (ou chemin direct de ``nlridge.py``).
        Par défaut, la variable d'environnement ``EOQUAL_NLRIDGE_PATH``.
    data_range : tuple(float, float), optional
        Dynamique ``(min, max)`` du capteur, ramenée à [0, 255] pour
        appliquer les paramètres recommandés par les auteurs. Par défaut,
        le min/max de l'image.
    device : str, optional
        Périphérique torch. Par défaut ``"cuda"`` si disponible, sinon
        ``"cpu"``.
    **nlridge_kwargs
        Transmis à ``NLRidge.forward`` (``p1``, ``p2``, ``k1``, ``k2``,
        ``w``, ``s``, ``constraints``...) — remplacent les valeurs
        recommandées choisies selon ``sigma``.

    Returns
    -------
    np.ndarray
        Image débruitée, ``float64``, même forme que l'entrée.

    Raises
    ------
    ImportError
        Si le clone NL-Ridge est introuvable ou si torch/einops manquent.
    """
    module = _load_nlridge(nlridge_path)
    import torch

    img = check_image(image)
    s = resolve_sigma(img, sigma)
    unit, lo, scale = to_unit_range(img, data_range)
    sigma_255 = 255.0 * s / scale

    params = next(p for bound, p in _RECOMMENDED if sigma_255 <= bound)
    params = {**params, "w": 37, "s": 4, **nlridge_kwargs}

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    y = torch.from_numpy((255.0 * unit).astype(np.float32))[None, None].to(device)
    model = module.NLRidge()
    with torch.no_grad():
        den = model(y, sigma=sigma_255, noise_type="gaussian-homoscedastic", **params)
    out = den[0, 0].cpu().numpy().astype(np.float64) / 255.0
    return from_unit_range(out, lo, scale)
