#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitaires (``utils.py``) : validation de l'image, estimation de σ,
normalisation, générateur de bruit, images de test. Tolérances des tests
statistiques tirées de l'incertitude (5 incertitudes-types).
"""
import numpy as np
import pytest

from _helpers import SEED, SIGMA
from eoqual.denoising import add_gaussian_noise, astronaut_gray, astronaut_rgb, estimate_noise_sigma
from eoqual.denoising.utils import check_image, from_unit_range, resolve_sigma, to_unit_range


def test_check_image_converts_and_validates():
    out = check_image(np.arange(12, dtype=np.uint16).reshape(3, 4))
    assert out.dtype == np.float64 and out.shape == (3, 4)
    with pytest.raises(ValueError, match="2D"):
        check_image(np.zeros((4, 4, 3)))
    with pytest.raises(ValueError, match="non finies"):
        check_image(np.array([[0.0, np.nan], [1.0, 2.0]]))


def test_estimate_noise_sigma(noisy):
    """Estimateur MAD : à 20 % près sur une image modérément texturée."""
    assert abs(estimate_noise_sigma(noisy) - SIGMA) < 0.2 * SIGMA


def test_resolve_sigma():
    img = np.zeros((16, 16))
    assert resolve_sigma(img, 2.5) == 2.5
    with pytest.raises(ValueError):
        resolve_sigma(img, -1.0)


@pytest.mark.parametrize("data_range", [None, (0.0, 4095.0)])
def test_unit_range_round_trip(data_range):
    img = np.random.default_rng(SEED).uniform(100, 3000, size=(8, 8))
    unit, lo, scale = to_unit_range(img, data_range)
    assert np.allclose(from_unit_range(unit, lo, scale), img)
    if data_range is None:
        assert unit.min() == 0.0 and unit.max() == 1.0


def test_constant_image_unit_range():
    unit, lo, scale = to_unit_range(np.full((4, 4), 7.0))
    assert scale == 1.0 and np.all(unit == 0.0)


def test_add_gaussian_noise_statistics():
    """Moyenne nulle et écart-type σ, à 5 incertitudes-types près."""
    n = 200_000
    noise = add_gaussian_noise(np.zeros(n), SIGMA, seed=SEED)
    assert abs(noise.mean()) < 5 * SIGMA / np.sqrt(n)
    assert abs(noise.std() / SIGMA - 1) < 5 / np.sqrt(2 * n)


def test_add_gaussian_noise_is_reproducible():
    img = np.zeros((8, 8))
    assert np.array_equal(add_gaussian_noise(img, 1.0, seed=3), add_gaussian_noise(img, 1.0, seed=3))


def test_test_images():
    g, c = astronaut_gray(64), astronaut_rgb(64)
    assert g.shape == (64, 64) and c.shape == (64, 64, 3)
    assert 0.0 <= g.min() and g.max() <= 1.0 and 0.0 <= c.min() and c.max() <= 1.0
    assert astronaut_gray().shape == (512, 512)
