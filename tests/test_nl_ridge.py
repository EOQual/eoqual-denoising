#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NL-Ridge natif (réimplémentation MIT) : niveau de qualité, deux étapes,
contrainte affine, paramètres des auteurs, accord avec leur code.
"""
import os

import numpy as np
import pytest

from _helpers import SIGMA, psnr
from eoqual.denoising.runner import denoise


@pytest.mark.parametrize("constraints", ["linear", "affine"])
def test_nl_ridge_native_beats_nl_means(clean, noisy, constraints):
    out = denoise(noisy, "nl_ridge", sigma=SIGMA, constraints=constraints)
    ref = denoise(noisy, "nl_means", sigma=SIGMA)
    assert psnr(clean, out, data_range=1.0) > psnr(clean, ref, data_range=1.0) + 1.0


def test_nl_ridge_second_step_improves(clean, noisy):
    basic = denoise(noisy, "nl_ridge", sigma=SIGMA, return_basic=True)
    final = denoise(noisy, "nl_ridge", sigma=SIGMA)
    assert psnr(clean, final, data_range=1.0) > psnr(clean, basic, data_range=1.0)


def test_nl_ridge_affine_weights_sum_to_one():
    """Contrainte affine (article 2025, formule 6) : Θᵀ1 = 1."""
    from eoqual.denoising.backends.nl_ridge.native import _theta

    rng = np.random.default_rng(0)
    y = rng.normal(size=(4, 49, 18))
    q = np.swapaxes(y, 1, 2) @ y
    theta = _theta(q, 49 * 0.5, affine=True)
    assert np.allclose(theta.sum(axis=1), 1.0)


def test_nl_ridge_parameters_follow_article():
    from eoqual.denoising.backends.nl_ridge.native import nl_ridge_parameters

    assert nl_ridge_parameters(12)["p1"] == 7 and nl_ridge_parameters(12)["k2"] == 55
    assert nl_ridge_parameters(25)["p1"] == 9 and nl_ridge_parameters(25)["k2"] == 90
    assert (nl_ridge_parameters(45)["p1"], nl_ridge_parameters(45)["p2"], nl_ridge_parameters(45)["k2"]) == (11, 9, 120)


@pytest.mark.skipif("EOQUAL_NLRIDGE_PATH" not in os.environ, reason="clone NL-Ridge non fourni")
def test_nl_ridge_native_matches_authors_code(clean, noisy):
    """La réimplémentation reproduit le code des auteurs à 0,2 dB près."""
    pytest.importorskip("torch")
    a = denoise(noisy, "nl_ridge", sigma=SIGMA)
    b = denoise(noisy, "nl_ridge_inria", sigma=SIGMA)
    assert abs(psnr(clean, a, data_range=1.0) - psnr(clean, b, data_range=1.0)) < 0.2
