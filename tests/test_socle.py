#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthodes du socle (installées par défaut) : gain de PSNR sur astronaut
bruitée, σ estimé automatiquement, invariance d'échelle (image 12 bits).
"""

import pytest

from _helpers import SIGMA, check_gain
from eoqual.denoising.config import METHODS_CONFIGS
from eoqual.denoising import add_gaussian_noise
from eoqual.denoising.runner import denoise

SOCLE = [m for m, cfg in METHODS_CONFIGS.items() if cfg["extra"] is None and cfg["external"] is None]


@pytest.mark.parametrize("method", SOCLE)
def test_socle_method(clean, noisy, method):
    """Chaque méthode du socle débruite effectivement l'image."""
    check_gain(clean, noisy, denoise(noisy, method=method, sigma=SIGMA))


@pytest.mark.parametrize("method", SOCLE)
def test_socle_method_estimated_sigma(clean, noisy, method):
    """Sans sigma fourni, l'estimation automatique suffit à débruiter."""
    check_gain(clean, noisy, denoise(noisy, method=method))


@pytest.mark.parametrize("method", SOCLE)
def test_socle_method_scale_invariant(clean, method):
    """Le résultat ne dépend pas des unités de l'image (ex. 12 bits)."""
    scale = 4095.0
    noisy12 = add_gaussian_noise(clean * scale, SIGMA * scale, seed=0)
    out = denoise(noisy12, method=method, sigma=SIGMA * scale)
    check_gain(clean, noisy12 / scale, out / scale)
