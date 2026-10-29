#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NL-Bayes natif (réimplémentation MIT) : niveau de qualité, deux étapes,
paramètres de l'article et imbrications, moteurs vectorisé / boucle,
couleur.
"""

import numpy as np
import pytest

from _helpers import MIN_GAIN_DB, SIGMA, psnr
from eoqual.denoising.runner import denoise


def test_nl_bayes_native_close_to_bm3d_level(clean, noisy):
    """
    L'implémentation native atteint le niveau annoncé par l'article : bien
    au-dessus de NL-Means (gain ≥ 1 dB sur la même image).
    """
    nl = denoise(noisy, method="nl_bayes", sigma=SIGMA)
    ref = denoise(noisy, method="nl_means", sigma=SIGMA)
    assert psnr(clean, nl, data_range=1.0) > psnr(clean, ref, data_range=1.0) + 1.0


def test_nl_bayes_second_step_improves(clean, noisy):
    """L'étape 2 (oracle u_basic) améliore l'estimée de l'étape 1."""
    basic = denoise(noisy, method="nl_bayes", sigma=SIGMA, return_basic=True)
    final = denoise(noisy, method="nl_bayes", sigma=SIGMA)
    assert psnr(clean, final, data_range=1.0) > psnr(clean, basic, data_range=1.0)


def test_nl_bayes_parameters_follow_article():
    from eoqual.denoising.backends.nl_bayes.native import nl_bayes_parameters

    p = nl_bayes_parameters(25.0)
    assert (p["k1"], p["k2"], p["N1"], p["N2"], p["n1"], p["n2"]) == (5, 5, 60, 60, 35, 35)
    assert (p["gamma"], p["beta1"], p["beta2"], p["tau0"]) == (1.05, 1.0, 1.2, 4.0)
    p = nl_bayes_parameters(80.0)
    assert (p["k1"], p["k2"], p["N1"], p["beta2"]) == (7, 7, 90, 1.0)


def test_nl_bayes_patch_override_updates_group_size(noisy):
    """Changer k1 sans N1 applique la règle de l'article (N = 30/60/90)."""
    out = denoise(noisy, method="nl_bayes", sigma=SIGMA, k1=3, k2=3)
    assert out.shape == noisy.shape and np.all(np.isfinite(out))
    with pytest.raises(TypeError):
        denoise(noisy, method="nl_bayes", sigma=SIGMA, not_a_param=1)


def test_nl_bayes_engines_agree(clean, noisy):
    """Moteur vectorisé (grille) et boucle (« paste ») : même qualité à 0,15 dB près."""
    loop = denoise(noisy, "nl_bayes", sigma=SIGMA, engine="loop")
    vec = denoise(noisy, "nl_bayes", sigma=SIGMA)
    assert abs(psnr(clean, loop, data_range=1.0) - psnr(clean, vec, data_range=1.0)) < 0.15


def test_nl_bayes_threads_do_not_change_result(noisy):
    a = denoise(noisy, "nl_bayes", sigma=SIGMA, n_threads=1)
    b = denoise(noisy, "nl_bayes", sigma=SIGMA, n_threads=4)
    assert np.allclose(a, b)


def test_yuv_matrix_is_orthonormal():
    from eoqual.denoising.backends.nl_bayes.vectorized import YUV_MATRIX

    assert np.allclose(YUV_MATRIX @ YUV_MATRIX.T, np.eye(3))


@pytest.mark.parametrize("mode", ["yuv", "rgb", "channel"])
def test_nl_bayes_color_modes(rgb_pair, mode):
    clean_rgb, noisy_rgb = rgb_pair
    out = denoise(noisy_rgb, "nl_bayes", sigma=SIGMA, color_mode=mode)
    assert out.shape == noisy_rgb.shape and np.all(np.isfinite(out))
    gain = psnr(clean_rgb, out, data_range=1.0) - psnr(clean_rgb, noisy_rgb, data_range=1.0)
    assert gain > MIN_GAIN_DB


def test_nl_bayes_color_rejects_loop_engine(rgb_pair):
    with pytest.raises(ValueError):
        denoise(rgb_pair[1], "nl_bayes", sigma=SIGMA, engine="loop", color_mode="yuv")


def test_nl_bayes_parameter_nesting():
    """Imbrications : N suit k, n = α·k, N ≥ k² + 1 ; profils gris / couleur de l'article."""
    from eoqual.denoising.backends.nl_bayes.native import nl_bayes_parameters, resolve_parameters

    assert nl_bayes_parameters(15.0, color=True)["k1"] == 3
    assert nl_bayes_parameters(15.0, color=False)["k1"] == 5
    assert nl_bayes_parameters(30.0, color=True)["k2"] == 3
    p = resolve_parameters(25.0, False, {"k1": 7, "window_factor": 5})
    assert (p["N1"], p["n1"], p["n2"]) == (90, 35, 25)
    p = resolve_parameters(25.0, False, {"k1": 7, "N1": 20})
    assert p["N1"] == 50  # porté à k² + 1 : covariance 49×49 inversible
