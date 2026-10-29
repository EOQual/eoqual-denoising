"""
eoqual.denoising — Collection de méthodes de débruitage d'images
d'observation de la Terre.

Public API : importer les méthodes directement depuis ce paquet.

Examples
--------
>>> import eoqual.denoising as d
>>> clean = d.nl_means_denoise(noisy_image, sigma=0.1)
>>> clean = d.wavelet_denoise(noisy_image)          # sigma estimé

>>> from eoqual.denoising.runner import denoise, list_methods
>>> list_methods()
>>> clean = denoise(noisy_image, method="tv_chambolle", sigma=0.1)

Bruit poisson-gaussien (image brute en DN, paramètres de la PTC) :

>>> clean = d.vst_denoise(raw, "bm3d", gain=2.0, read_noise_dn=1.5, offset_dn=100)

Aucune méthode ne charge de dépendance lourde ou tierce à l'import de ce
paquet — ``drunet_denoise`` (extra ``deep``), ``bm3d_denoise`` (extra
``bm3d``, **non commercial**), ``nl_ridge_inria_denoise`` (clone externe,
AGPL-3.0) et ``nl_bayes_ipol_denoise`` (exécutable externe, GPL-3.0+) ne
chargent leur backend qu'à l'appel. Voir ``README.md`` pour
l'installation et ``THIRD_PARTY_LICENSES.md`` pour l'audit des licences.
"""

__version__ = "0.1.0"
__title__ = "eoqual.denoising"
__description__ = "Collection of Earth observation image denoising methods"
__url__ = "https://github.com/EOQual/eoqual-denoising"
__uri__ = __url__
__doc__ = __description__ + " <" + __uri__ + ">"
__author__ = "Olivier Amram"
__email__ = "olivier.amram@free.fr"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2026 Olivier Amram"

from .config import METHODS_CONFIGS  # noqa: F401

from .backends.bilateral.opencv_impl import bilateral_denoise  # noqa: F401
from .backends.nl_means.skimage_impl import nl_means_denoise  # noqa: F401
from .backends.tv.skimage_impl import tv_chambolle_denoise, tv_bregman_denoise  # noqa: F401
from .backends.wavelet.skimage_impl import wavelet_denoise  # noqa: F401
from .backends.drunet.deepinv_impl import drunet_denoise  # noqa: F401
from .backends.deepinv_zoo.models import (  # noqa: F401
    swinir_denoise,
    restormer_denoise,
    gsdrunet_denoise,
    epll_denoise,
    tgv_denoise,
    wavelet_dict_denoise,
)
from .backends.bm3d.tau_wrapper import bm3d_denoise  # noqa: F401
from .backends.nl_ridge.native import nl_ridge_denoise  # noqa: F401
from .backends.nl_ridge.external import nl_ridge_inria_denoise  # noqa: F401
from .backends.nl_bayes.native import nl_bayes_denoise  # noqa: F401
from .backends.nl_bayes.ipol_external import nl_bayes_ipol_denoise  # noqa: F401

from .utils import add_gaussian_noise, astronaut_gray, astronaut_rgb, estimate_noise_sigma  # noqa: F401
from .vst import (  # noqa: F401
    add_poisson_gaussian_noise,
    generalized_anscombe,
    inverse_generalized_anscombe,
    vst_denoise,
)
from .runner import denoise, list_methods  # noqa: F401

__all__ = [
    "METHODS_CONFIGS",
    "bilateral_denoise",
    "nl_means_denoise",
    "tv_chambolle_denoise",
    "tv_bregman_denoise",
    "wavelet_denoise",
    "drunet_denoise",
    "swinir_denoise",
    "restormer_denoise",
    "gsdrunet_denoise",
    "epll_denoise",
    "tgv_denoise",
    "wavelet_dict_denoise",
    "bm3d_denoise",
    "nl_ridge_denoise",
    "nl_ridge_inria_denoise",
    "nl_bayes_denoise",
    "nl_bayes_ipol_denoise",
    "generalized_anscombe",
    "inverse_generalized_anscombe",
    "vst_denoise",
    "add_poisson_gaussian_noise",
    "add_gaussian_noise",
    "astronaut_gray",
    "astronaut_rgb",
    "estimate_noise_sigma",
    "denoise",
    "list_methods",
]
