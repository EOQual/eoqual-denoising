#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Registre et contrat d'appel unifié : dispatch, signature commune de
toutes les méthodes, erreurs, list_methods.
"""

import numpy as np
import pytest

from _helpers import SIGMA
from eoqual.denoising.config import METHODS_CONFIGS
from eoqual.denoising.runner import denoise


def test_non_2d_raises(noisy):
    with pytest.raises(ValueError):
        denoise(np.stack([noisy, noisy], axis=-1), method="nl_means")


def test_unknown_method_raises(noisy):
    with pytest.raises(ValueError):
        denoise(noisy, method="not_a_real_method")


def test_registry_matches_dispatch():
    """Toutes les méthodes de METHODS_CONFIGS sont bien dispatchables."""
    from eoqual.denoising.runner import _DISPATCH

    assert set(METHODS_CONFIGS.keys()) == set(_DISPATCH.keys())


def test_list_methods_runs(capsys):
    from eoqual.denoising.runner import list_methods

    list_methods()
    assert "nl_bayes" in capsys.readouterr().out


@pytest.mark.parametrize("method", list(METHODS_CONFIGS))
def test_unified_signature(method):
    """
    Toute méthode : f(image, sigma=None, ...), sans paramètre nommé
    « method » (réservé au dispatcher denoise(image, method, **kwargs)).
    """
    import inspect

    from eoqual.denoising.runner import _resolve

    params = list(inspect.signature(_resolve(method)).parameters.values())
    assert params[0].name == "image"
    assert params[1].name == "sigma" and params[1].default is None
    assert all(p.name != "method" for p in params)


def test_dispatcher_forwards_method_specific_args(clean, noisy):
    """Les arguments propres à une méthode passent par denoise() sans ambiguïté."""
    a = denoise(noisy, "wavelet", sigma=SIGMA, rule="VisuShrink", wavelet="db4")
    b = __import__("eoqual.denoising", fromlist=["x"]).wavelet_denoise(noisy, sigma=SIGMA, rule="VisuShrink", wavelet="db4")
    assert np.array_equal(a, b)
