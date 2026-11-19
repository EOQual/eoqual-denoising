#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NL-Bayes CFA (mosaïque de Bayer → RGB) : forme, gain sur le démosaïçage
bilinéaire, motifs, chaîne avec stabilisation de variance, contrôles.
"""

import numpy as np
import pytest

from _helpers import SEED, psnr
from eoqual.denoising import nl_bayes_cfa_denoise, vst_denoise
from eoqual.denoising.backends.nl_bayes.cfa import CFA_PATTERNS, _channel_map, bilinear_demosaic, cfa_channels
from eoqual.denoising.runner import denoise
from eoqual.denoising.utils import astronaut_rgb


@pytest.fixture(scope="module")
def rgb():
    return astronaut_rgb(128)


def mosaic_of(rgb, pattern):
    cm = _channel_map(*rgb.shape[:2], pattern)
    return np.take_along_axis(rgb, cm[..., None], axis=2)[..., 0]


def test_channels_and_bilinear_keep_measured_values(rgb):
    assert cfa_channels("RGGB").tolist() == [[0, 1], [1, 2]]
    assert cfa_channels("gbrg").tolist() == [[1, 2], [0, 1]]
    m = mosaic_of(rgb, "RGGB")
    out = bilinear_demosaic(m, "RGGB")
    cm = _channel_map(*m.shape, "RGGB")
    assert np.allclose(np.take_along_axis(out, cm[..., None], axis=2)[..., 0], m)


@pytest.mark.parametrize("pattern", CFA_PATTERNS)
def test_denoises_and_demosaics_better_than_bilinear(rgb, pattern):
    rng = np.random.default_rng(SEED)
    noisy = mosaic_of(rgb, pattern) + rng.normal(0, 0.05, rgb.shape[:2])
    out = nl_bayes_cfa_denoise(noisy, sigma=0.05, pattern=pattern, data_range=(0, 1), n_threads=1)
    assert out.shape == rgb.shape and np.all(np.isfinite(out))
    base = psnr(rgb, np.clip(bilinear_demosaic(noisy, pattern), 0, 1), data_range=1)
    assert psnr(rgb, np.clip(out, 0, 1), data_range=1) > base + 2.0


def test_wrong_pattern_is_worse(rgb):
    noisy = mosaic_of(rgb, "RGGB")
    good = nl_bayes_cfa_denoise(noisy, sigma=0.005, pattern="RGGB", data_range=(0, 1), n_threads=1)
    bad = nl_bayes_cfa_denoise(noisy, sigma=0.005, pattern="BGGR", data_range=(0, 1), n_threads=1)
    assert psnr(rgb, np.clip(good, 0, 1), data_range=1) > psnr(rgb, np.clip(bad, 0, 1), data_range=1) + 5


def test_noiseless_beats_bilinear_demosaicking(rgb):
    m = mosaic_of(rgb, "RGGB")
    out = nl_bayes_cfa_denoise(m, sigma=0.002, data_range=(0, 1), pilot="none", n_threads=1)
    assert psnr(rgb, np.clip(out, 0, 1), data_range=1) > psnr(rgb, bilinear_demosaic(m), data_range=1)


def test_dispatcher_vst_and_scale(rgb):
    """Chaîne raw : vst_denoise sur la mosaïque en DN, sortie RGB en DN ; invariance d'échelle."""
    from eoqual.denoising.vst import add_poisson_gaussian_noise

    peak, gain, read = 400.0, 1.0, 2.0
    clean_dn = mosaic_of(rgb, "RGGB") * peak
    raw = add_poisson_gaussian_noise(clean_dn, gain=gain, read_noise_dn=read, seed=SEED)
    out = vst_denoise(raw, "nl_bayes_cfa", gain=gain, read_noise_dn=read, pattern="RGGB", n_threads=1)
    assert out.shape == rgb.shape
    base = psnr(rgb * peak, np.clip(bilinear_demosaic(raw), 0, peak), data_range=peak)
    assert psnr(rgb * peak, np.clip(out, 0, peak), data_range=peak) > base + 2.0
    a = denoise(clean_dn, "nl_bayes_cfa", sigma=2.0, n_threads=1)
    b = denoise(clean_dn * 3.0, "nl_bayes_cfa", sigma=6.0, n_threads=1)
    assert np.allclose(a * 3.0, b, rtol=1e-4, atol=1e-3)


def test_given_pilot_and_iterations(rgb):
    rng = np.random.default_rng(SEED)
    noisy = mosaic_of(rgb, "RGGB") + rng.normal(0, 0.05, rgb.shape[:2])
    one = nl_bayes_cfa_denoise(noisy, sigma=0.05, data_range=(0, 1), n_iter=1, n_threads=1)
    two = nl_bayes_cfa_denoise(noisy, sigma=0.05, data_range=(0, 1), n_iter=2, n_threads=1)
    given = nl_bayes_cfa_denoise(noisy, sigma=0.05, data_range=(0, 1), pilot=rgb, n_iter=1, n_threads=1)
    assert not np.allclose(one, two)
    assert psnr(rgb, np.clip(two, 0, 1), data_range=1) > psnr(rgb, np.clip(one, 0, 1), data_range=1)
    # pilote parfait : groupes et covariance exacts, le résultat ne peut pas être pire
    assert psnr(rgb, np.clip(given, 0, 1), data_range=1) >= psnr(rgb, np.clip(one, 0, 1), data_range=1)


def test_estimated_sigma(rgb):
    rng = np.random.default_rng(SEED)
    noisy = mosaic_of(rgb, "RGGB") + rng.normal(0, 0.05, rgb.shape[:2])
    out = nl_bayes_cfa_denoise(noisy, data_range=(0, 1), n_threads=1)
    assert psnr(rgb, np.clip(out, 0, 1), data_range=1) > psnr(rgb, np.clip(bilinear_demosaic(noisy), 0, 1), data_range=1)


@pytest.mark.parametrize("kwargs", [
    {"pattern": "RGBG"}, {"pilot": "ha"}, {"pilot": np.zeros((4, 4, 3))}, {"N": 1}, {"beta": 0.0}, {"sigma": -1.0},
    {"use_area": True, "area_mean": "moyenne"}, {"use_area": True, "area_gamma": 0.0},
])
def test_invalid_arguments(rgb, kwargs):
    with pytest.raises(ValueError):
        nl_bayes_cfa_denoise(mosaic_of(rgb, "RGGB"), **{"sigma": 0.05, **kwargs})


def test_rejects_color_input(rgb):
    with pytest.raises(ValueError):
        nl_bayes_cfa_denoise(rgb, sigma=0.05)


def test_homogeneous_criterion():
    """
    Zone plate colorée : le critère se déclenche et rend la couleur constante
    (moyenne du pilote : au moins aussi bon que sans critère) ; sur une image
    texturée, il ne change presque rien.
    """
    flat = np.ones((64, 64, 3)) * np.array([0.2, 0.5, 0.7])
    rng = np.random.default_rng(SEED)
    m = mosaic_of(flat, "RGGB") + rng.normal(0, 0.05, (64, 64))
    base = nl_bayes_cfa_denoise(m, sigma=0.05, data_range=(0, 1), n_threads=1)
    pil = nl_bayes_cfa_denoise(m, sigma=0.05, data_range=(0, 1), n_threads=1, use_area=True, area_mean="pilot")
    mos = nl_bayes_cfa_denoise(m, sigma=0.05, data_range=(0, 1), n_threads=1, use_area=True)
    assert not np.allclose(pil, base) and not np.allclose(mos, base)
    assert psnr(flat, np.clip(pil, 0, 1), data_range=1) >= psnr(flat, np.clip(base, 0, 1), data_range=1) - 0.1
    assert np.allclose(mos.mean((0, 1)), [0.2, 0.5, 0.7], atol=0.01)
    noisy = mosaic_of(astronaut_rgb(128), "RGGB") + rng.normal(0, 0.05, (128, 128))
    a = nl_bayes_cfa_denoise(noisy, sigma=0.05, data_range=(0, 1), n_threads=1)
    b = nl_bayes_cfa_denoise(noisy, sigma=0.05, data_range=(0, 1), n_threads=1, use_area=True, area_gamma=1.05)
    assert abs(psnr(astronaut_rgb(128), np.clip(a, 0, 1), data_range=1)
               - psnr(astronaut_rgb(128), np.clip(b, 0, 1), data_range=1)) < 0.3
