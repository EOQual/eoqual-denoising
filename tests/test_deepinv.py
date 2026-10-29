#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthodes deepinv (extra ``deep``, ``ptwt``). Les modèles à poids
téléchargés ne sont testés que si ``EOQUAL_TEST_DOWNLOAD=1``.
"""
import os

import pytest

from _helpers import SIGMA, check_gain
from eoqual.denoising.runner import denoise

needs_download = pytest.mark.skipif(
    os.environ.get("EOQUAL_TEST_DOWNLOAD") != "1",
    reason="télécharge des poids pré-entraînés : activer avec EOQUAL_TEST_DOWNLOAD=1",
)


@needs_download
@pytest.mark.parametrize("method", ["drunet", "swinir", "restormer", "gsdrunet", "epll"])
def test_pretrained_deepinv(clean, noisy, method):
    """Extra 'deep', modèles à poids téléchargés — skip si torch/deepinv absents."""
    pytest.importorskip("torch")
    pytest.importorskip("deepinv")
    check_gain(clean, noisy, denoise(noisy, method=method, sigma=SIGMA))


def test_tgv(clean, noisy):
    """Extra 'deep', sans poids — skip si torch/deepinv absents."""
    pytest.importorskip("torch")
    pytest.importorskip("deepinv")
    check_gain(clean, noisy, denoise(noisy, method="tgv", sigma=SIGMA))


def test_wavelet_dict(clean, noisy):
    """Extras 'deep' + 'ptwt' (EUPL-1.2) — skip si absents."""
    pytest.importorskip("torch")
    pytest.importorskip("deepinv")
    pytest.importorskip("ptwt")
    check_gain(clean, noisy, denoise(noisy, method="wavelet_dict", sigma=SIGMA))


def test_swinir_picks_closest_weights():
    from eoqual.denoising.backends.deepinv_zoo.models import SWINIR_NOISE_LEVELS

    assert min(SWINIR_NOISE_LEVELS, key=lambda lvl: abs(lvl - 255 * SIGMA)) == 25
