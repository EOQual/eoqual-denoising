#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils communs aux backends PyTorch / deepinv : import paresseux avec
message d'erreur explicite, choix du périphérique, conversion
numpy ↔ tenseur, cache des modèles chargés.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["import_deepinv", "pick_device", "cached_model", "run_on_unit_image"]

_MODELS: dict[tuple, object] = {}


def import_deepinv():
    """Importe ``torch`` et ``deepinv.models``, ou lève une ``ImportError`` explicite."""
    try:
        import torch
        import deepinv.models as dm
    except ImportError as exc:
        raise ImportError(
            "Cette méthode nécessite 'torch' et 'deepinv'. Installer avec : "
            'pip install "eoqual-denoising[deep]"'
        ) from exc
    return torch, dm


def pick_device(device: str | None) -> str:
    """Renvoie ``device`` s'il est fourni, sinon ``"cuda"`` si disponible, sinon ``"cpu"``."""
    torch, _ = import_deepinv()
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def cached_model(key: tuple, factory: Callable[[], object]):
    """
    Renvoie le modèle associé à ``key``, construit une seule fois par
    ``factory`` (évite de recharger les poids à chaque appel).
    """
    if key not in _MODELS:
        model = factory()
        if hasattr(model, "eval"):
            model.eval()
        _MODELS[key] = model
    return _MODELS[key]


def run_on_unit_image(
    model_call: Callable, unit_image: np.ndarray, device: str, needs_grad: bool = False
) -> np.ndarray:
    """
    Applique ``model_call(x)`` à une image 2D dans [0, 1] convertie en
    tenseur ``(1, 1, H, W)`` ``float32``, et renvoie le résultat 2D
    ``float64``.

    ``needs_grad=True`` pour les débruiteurs qui dérivent un potentiel
    (GS-DRUNet) : le calcul se fait alors avec l'autograd activé.
    """
    torch, _ = import_deepinv()
    x = torch.from_numpy(unit_image.astype(np.float32))[None, None].to(device)
    if needs_grad:
        with torch.enable_grad():
            y = model_call(x)
    else:
        with torch.no_grad():
            y = model_call(x)
    return y.detach()[0, 0].cpu().numpy().astype(np.float64)
