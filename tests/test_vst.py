#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests de la stabilisation de variance poisson-gaussienne (``vst.py``).

Les tolérances des tests Monte-Carlo sont tirées de l'incertitude
statistique (5 incertitudes-types), pas fixées au jugé. Graine
modifiable par ``EOQUAL_TEST_SEED``.
"""
import os

import numpy as np
import pytest
from skimage.metrics import peak_signal_noise_ratio as psnr

from eoqual.denoising import astronaut_gray, denoise
from eoqual.denoising import vst

SEED = int(os.environ.get("EOQUAL_TEST_SEED", "0"))


def test_classical_anscombe_special_case():
    z = np.array([0.0, 1.0, 10.0, 100.0])
    assert np.allclose(vst.generalized_anscombe(z), 2.0 * np.sqrt(z + 3.0 / 8.0))


def test_units_conversion():
    """La GAT en DN avec gain/offset équivaut à la GAT en électrons."""
    k, rn, off = 2.5, 1.2, 64.0
    z = np.linspace(50, 500, 7)
    y = k * (z - off)
    assert np.allclose(
        vst.generalized_anscombe(z, k, rn, off),
        2.0 * np.sqrt(np.maximum(y + 0.375 + (k * rn) ** 2, 0)),
    )


@pytest.mark.parametrize("sigma_e", [0.0, 3.0])
def test_exact_table_joins_asymptotic(sigma_e):
    """La table exacte rejoint l'inverse asymptotique sans saut à LAMBDA_MAX."""
    _, expected = vst._exact_table(sigma_e)
    asym = 2.0 * np.sqrt(vst.LAMBDA_MAX + 0.125 + sigma_e**2)
    assert abs(expected[-1] - asym) / asym < 1e-5


@pytest.mark.parametrize("sigma_e", [0.0, 0.5, 3.0])
@pytest.mark.parametrize("lam", [0.3, 2.0, 10.0, 80.0])
def test_exact_inverse_is_unbiased(sigma_e, lam):
    """
    Inverse exacte appliquée à la moyenne empirique de f(y) : retrouve λ
    à 5 incertitudes-types près (incertitude propagée par la pente de
    l'inverse).
    """
    rng = np.random.default_rng(SEED)
    n = 200_000
    y = rng.poisson(lam, n) + rng.normal(0.0, sigma_e, n)
    d = vst._gat_electrons(y, sigma_e)
    m, u_m = d.mean(), d.std(ddof=1) / np.sqrt(n)

    def inv(x):
        return vst.inverse_generalized_anscombe(np.array([x]), 1.0, sigma_e)[0]

    h = 1e-4 * max(1.0, m)
    slope = (inv(m + h) - inv(m - h)) / (2 * h)
    assert abs(inv(m) - lam) < 5.0 * slope * u_m


def test_algebraic_inverse_is_biased():
    """L'inverse algébrique sous-estime de ~0,25 e⁻ (témoin du biais corrigé)."""
    lam = 10.0
    d = vst.expected_gat(np.array([lam]))
    biased = vst.inverse_generalized_anscombe(d, method="algebraic")[0]
    exact = vst.inverse_generalized_anscombe(d, method="exact_unbiased")[0]
    assert abs(exact - lam) < 1e-3
    assert biased < lam - 0.2


def _exact_gat_variance(lam: float, sigma_e: float) -> float:
    """Var[f(y)] exacte : somme sur la loi de Poisson, intégrale gaussienne."""
    from scipy.stats import poisson

    k = np.arange(int(lam + 12 * np.sqrt(lam) + 30))
    t = np.linspace(-10.0, 10.0, 4001)
    w = np.exp(-0.5 * t**2)
    w /= w.sum()
    f = vst._gat_electrons(k[:, None] + sigma_e * t[None, :], sigma_e)
    p = poisson.pmf(k, lam)
    m1 = p @ (f @ w)
    m2 = p @ ((f**2) @ w)
    return float(m2 - m1**2)


@pytest.mark.parametrize("lam", [5.0, 30.0, 300.0])
def test_variance_is_stabilized(lam):
    """
    Var[f(y)] exacte proche de 1 dès quelques électrons (propriété de la
    GAT : écart < 6 % à 5 e⁻, < 1 % à 30 e⁻), et variance Monte-Carlo
    conforme à cette valeur exacte à 5 incertitudes-types.
    """
    sigma_e, n = 2.0, 200_000
    exact = _exact_gat_variance(lam, sigma_e)
    assert abs(exact - 1.0) < (0.06 if lam < 10 else 0.01)

    rng = np.random.default_rng(SEED)
    d = vst._gat_electrons(rng.poisson(lam, n) + rng.normal(0.0, sigma_e, n), sigma_e)
    c = d - d.mean()
    var = np.mean(c**2)
    u_var = np.sqrt((np.mean(c**4) - var**2) / n)
    assert abs(var - exact) < 5.0 * u_var


def test_simulator_mean_and_variance():
    """add_poisson_gaussian_noise respecte σ²(μ) = μ_DN/K + σ_DN² (forme PTC)."""
    k, rn, off, lam = 2.0, 1.5, 100.0, 50.0
    z = vst.add_poisson_gaussian_noise(np.full(400_000, lam), k, rn, off, seed=SEED)
    mu = lam / k
    var = lam / k**2 + rn**2
    assert abs(z.mean() - off - mu) < 5 * np.sqrt(var / z.size)
    assert abs(z.var() / var - 1) < 5 * np.sqrt(2 / z.size)


def test_vst_denoise_beats_direct_at_low_flux():
    """
    À faible flux (pic 5 e⁻, bruit photonique dominant), GAT + NL-Means
    bat NL-Means appliqué directement (bruit non stationnaire) d'au moins
    2 dB.
    """
    k, rn, off, peak = 2.0, 0.25, 100.0, 5.0
    signal = astronaut_gray(128) * peak
    raw = vst.add_poisson_gaussian_noise(signal, k, rn, off, seed=SEED)
    ref = signal / k + off
    direct = denoise(raw, method="nl_means")
    stab = vst.vst_denoise(raw, "nl_means", gain=k, read_noise_dn=rn, offset_dn=off)
    gain_db = psnr(ref, stab, data_range=peak / k) - psnr(ref, direct, data_range=peak / k)
    assert stab.shape == raw.shape
    assert gain_db > 2.0, f"gain {gain_db:.2f} dB"


def test_vst_denoise_rejects_sigma():
    with pytest.raises(ValueError, match="sigma"):
        vst.vst_denoise(np.ones((16, 16)), "nl_means", gain=1.0, sigma=2.0)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        vst.generalized_anscombe(np.ones(3), gain=0.0)
    with pytest.raises(ValueError):
        vst.inverse_generalized_anscombe(np.ones(3), method="wrong")
