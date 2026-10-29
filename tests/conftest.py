#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixtures partagées par les tests d'eoqual-denoising.

Image de test : « astronaut » (NASA, domaine public, fournie par
scikit-image), réduite à 128×128 (niveaux de gris) ou 96×96 (couleur),
bruitée par un bruit blanc gaussien de σ connu. La graine est modifiable
par ``EOQUAL_TEST_SEED`` (robustesse des tests statistiques).
"""
import numpy as np
import pytest

from _helpers import SEED, SIGMA
from eoqual.denoising import add_gaussian_noise, astronaut_gray, astronaut_rgb


@pytest.fixture(scope="session")
def clean() -> np.ndarray:
    return astronaut_gray(128)


@pytest.fixture(scope="session")
def noisy(clean) -> np.ndarray:
    return add_gaussian_noise(clean, SIGMA, seed=SEED)


@pytest.fixture(scope="session")
def rgb_pair():
    clean_rgb = astronaut_rgb(96)
    return clean_rgb, add_gaussian_noise(clean_rgb, SIGMA, seed=SEED)
