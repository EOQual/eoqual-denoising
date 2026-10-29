"""
Point d'entrée unique pour appeler n'importe quelle méthode de débruitage
par son nom, et lister les méthodes disponibles.

Exemple
-------
>>> from eoqual.denoising.runner import denoise, list_methods
>>> list_methods()
>>> clean = denoise(noisy_image, method="nl_means", sigma=0.1)
"""

from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.table import Table

from .config import METHODS_CONFIGS

__all__ = ["denoise", "list_methods"]

# Import différé des fonctions publiques : aucune dépendance lourde ou
# tierce (torch, deepinv, bm3d, NL-Ridge...) n'est chargée avant l'appel.
_DISPATCH = {
    "bilateral": "eoqual.denoising.backends.bilateral.opencv_impl:bilateral_denoise",
    "nl_means": "eoqual.denoising.backends.nl_means.skimage_impl:nl_means_denoise",
    "tv_chambolle": "eoqual.denoising.backends.tv.skimage_impl:tv_chambolle_denoise",
    "tv_bregman": "eoqual.denoising.backends.tv.skimage_impl:tv_bregman_denoise",
    "wavelet": "eoqual.denoising.backends.wavelet.skimage_impl:wavelet_denoise",
    "nl_bayes": "eoqual.denoising.backends.nl_bayes.native:nl_bayes_denoise",
    "nl_ridge": "eoqual.denoising.backends.nl_ridge.native:nl_ridge_denoise",
    "drunet": "eoqual.denoising.backends.drunet.deepinv_impl:drunet_denoise",
    "swinir": "eoqual.denoising.backends.deepinv_zoo.models:swinir_denoise",
    "restormer": "eoqual.denoising.backends.deepinv_zoo.models:restormer_denoise",
    "gsdrunet": "eoqual.denoising.backends.deepinv_zoo.models:gsdrunet_denoise",
    "epll": "eoqual.denoising.backends.deepinv_zoo.models:epll_denoise",
    "tgv": "eoqual.denoising.backends.deepinv_zoo.models:tgv_denoise",
    "wavelet_dict": "eoqual.denoising.backends.deepinv_zoo.models:wavelet_dict_denoise",
    "bm3d": "eoqual.denoising.backends.bm3d.tau_wrapper:bm3d_denoise",
    "nl_ridge_inria": "eoqual.denoising.backends.nl_ridge.external:nl_ridge_inria_denoise",
    "nl_bayes_ipol": "eoqual.denoising.backends.nl_bayes.ipol_external:nl_bayes_ipol_denoise",
}


def _resolve(method: str):
    module_path, func_name = _DISPATCH[method].split(":")
    module = __import__(module_path, fromlist=[func_name])
    return getattr(module, func_name)


def list_methods() -> None:
    """Affiche toutes les méthodes de débruitage et leur statut de licence."""
    console = Console()
    table = Table(title="Méthodes disponibles dans eoqual.denoising", header_style="bold magenta")
    table.add_column("Méthode", style="bold cyan", no_wrap=True)
    table.add_column("Extra")
    table.add_column("Composant externe")
    table.add_column("Licence amont")
    table.add_column("Restriction", style="yellow")
    table.add_column("Référence")

    for name, cfg in METHODS_CONFIGS.items():
        table.add_row(
            name,
            cfg["extra"] or "—",
            cfg["external"] or "—",
            cfg["license"],
            cfg["restriction"] or "—",
            cfg["reference"],
        )

    console.print(table)


def denoise(image: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """
    Applique une méthode de débruitage par son nom.

    Parameters
    ----------
    image : np.ndarray
        Image 2D bruitée.
    method : str
        Nom de la méthode — une des clés de :data:`METHODS_CONFIGS`
        (voir :func:`list_methods`).
    **kwargs
        Transmis à l'implémentation choisie (``sigma`` notamment — voir sa
        docstring).

    Returns
    -------
    np.ndarray
        Image débruitée.

    Raises
    ------
    ValueError
        Si ``method`` n'est pas une méthode connue.
    ImportError, FileNotFoundError
        Si la méthode nécessite un extra ou un composant externe absent —
        le message précise quoi installer.
    """
    if method not in METHODS_CONFIGS:
        raise ValueError(
            f"Méthode '{method}' inconnue. Méthodes disponibles : "
            f"{list(METHODS_CONFIGS.keys())}"
        )
    return _resolve(method)(image, **kwargs)
