#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Constantes et assistants communs aux tests (importés par conftest et les tests)."""
import os

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr  # noqa: F401  (réexporté)

#: Écart-type du bruit des images de test (dynamique [0, 1]).
SIGMA = 0.1
#: Gain de PSNR minimal exigé d'une méthode (test de non-régression grossier).
MIN_GAIN_DB = 3.0
#: Graine des tirages aléatoires, modifiable par EOQUAL_TEST_SEED.
SEED = int(os.environ.get("EOQUAL_TEST_SEED", "0"))


def check_gain(clean, noisy, result, min_gain=MIN_GAIN_DB):
    """Même forme, valeurs finies, et gain de PSNR > ``min_gain`` sur l'image bruitée."""
    assert result.shape == noisy.shape
    assert np.all(np.isfinite(result))
    gain = psnr(clean, result, data_range=1.0) - psnr(clean, noisy, data_range=1.0)
    assert gain > min_gain, f"gain PSNR insuffisant : {gain:.2f} dB"
