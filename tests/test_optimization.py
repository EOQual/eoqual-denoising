#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests de l'optimisation bayésienne des paramètres (extra ``optim``) et des
espaces de recherche. Les tests qui lancent une optimisation sont sautés
si scikit-optimize est absent ; ils utilisent des méthodes rapides du
socle sur une petite image.
"""
import csv

import numpy as np

import pytest
from skimage.metrics import peak_signal_noise_ratio as psnr

from eoqual.denoising import add_gaussian_noise, astronaut_gray
from eoqual.denoising.config import METHODS_CONFIGS
from eoqual.denoising.optimization import SEARCH_SPACES, get_search_space, validate_search_space
from eoqual.denoising.optimization.jsonc import loads_jsonc
from eoqual.denoising.optimization.optimizer import _to_kwargs
from eoqual.denoising.runner import _resolve

SIGMA = 0.1


@pytest.fixture(scope="module")
def pair():
    clean = astronaut_gray(96)
    return clean, add_gaussian_noise(clean, SIGMA, seed=0)


# --- Espaces de recherche (sans scikit-optimize) ---------------------------


def test_jsonc_comments_and_trailing_commas():
    text = """
    {
      // commentaire
      "a": "http://pas/un/commentaire", /* bloc */
      "b": [1, 2, 3,],
    }
    """
    assert loads_jsonc(text) == {"a": "http://pas/un/commentaire", "b": [1, 2, 3]}


def test_every_tunable_method_has_a_search_space():
    """Toutes les méthodes du registre ont un espace, sauf restormer (aveugle, sans réglage)."""
    assert set(SEARCH_SPACES) == set(METHODS_CONFIGS) - {"restormer"}


@pytest.mark.parametrize("method", sorted(SEARCH_SPACES))
def test_initial_guess_matches_function_signature(method):
    """Chaque dimension correspond à un paramètre réel de la fonction de la méthode."""
    import inspect

    params = inspect.signature(_resolve(method)).parameters
    accepts_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())
    for dim in SEARCH_SPACES[method]["search_space"]:
        assert dim["name"] == "sigma_scale" or dim["name"] in params or accepts_kwargs, dim["name"]


def test_validate_rejects_out_of_bounds_guess():
    space = get_search_space("tv_chambolle")
    space["initial_guess"]["weight"] = 99.0
    with pytest.raises(ValueError, match="hors bornes"):
        validate_search_space(space)


def test_scaling_to_image_units():
    dims = [
        {"name": "a", "type": "float", "low": 0, "high": 5, "scale": "sigma"},
        {"name": "b", "type": "float", "low": 0, "high": 5, "scale": "inverse_sigma"},
        {"name": "sigma_scale", "type": "float", "low": 0.5, "high": 2},
        {"name": "c", "type": "int", "low": 1, "high": 9},
    ]
    kw = _to_kwargs({"a": 2.0, "b": 2.0, "sigma_scale": 1.5, "c": 3}, dims, sigma=4.0)
    assert kw == {"sigma": 6.0, "a": 8.0, "b": 0.5, "c": 3}


# --- Optimisation (scikit-optimize requis) ---------------------------------

import importlib.util  # noqa: E402

from eoqual.denoising.optimization import optimize_parameters  # noqa: E402  (import paresseux de skopt)


needs_skopt = pytest.mark.skipif(importlib.util.find_spec("skopt") is None, reason="scikit-optimize absent (extra optim)")


@needs_skopt
def test_optimization_improves_or_keeps_initial_and_is_reproducible(pair, tmp_path):
    clean, noisy = pair
    log = tmp_path / "log.csv"
    res = optimize_parameters(noisy, clean, "nl_means", sigma=SIGMA, n_calls=12, n_initial_points=4, log_csv=log)
    assert len(res.history) == 12
    assert res.initial_score is not None and res.best_score >= res.initial_score
    # Les arguments renvoyés reproduisent exactement le meilleur score.
    out = _resolve("nl_means")(noisy, **res.best_kwargs)
    assert psnr(clean, out, data_range=float(clean.max() - clean.min())) == pytest.approx(res.best_score)
    with open(log, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12 and {"h", "patch_size", "psnr", "status"} <= set(rows[0])


@needs_skopt
def test_parallel_rolling_queue(pair):
    clean, noisy = pair
    res = optimize_parameters(noisy, clean, "tv_chambolle", sigma=SIGMA, n_calls=8, n_initial_points=3, n_jobs=2)
    assert len(res.history) == 8
    assert res.best_score >= res.initial_score
    assert len({h["status"] for h in res.history}) == 1  # toutes "ok"


@needs_skopt
def test_exhaustive_search_for_small_categorical_space(pair):
    clean, noisy = pair
    space = {
        "search_space": [
            {"name": "wavelet", "type": "categorical", "categories": ["db1", "db2", "sym4"]},
            {"name": "mode", "type": "categorical", "categories": ["soft", "hard"]},
        ],
        "initial_guess": {"wavelet": "db2", "mode": "soft"},
    }
    res = optimize_parameters(noisy, clean, "wavelet", sigma=SIGMA, search_space=space, n_calls=10)
    assert res.skopt_result is None and len(res.history) == 6


@needs_skopt
def test_space_constraint_is_respected(pair):
    clean, noisy = pair
    res = optimize_parameters(
        noisy,
        clean,
        "nl_means",
        sigma=SIGMA,
        n_calls=10,
        n_initial_points=5,
        space_constraint=lambda p: p[2] >= p[1],  # patch_distance ≥ patch_size
    )
    assert all(h["patch_distance"] >= h["patch_size"] for h in res.history)


@needs_skopt
def test_ssim_metric(pair):
    clean, noisy = pair
    res = optimize_parameters(noisy, clean, "tv_chambolle", sigma=SIGMA, n_calls=6, n_initial_points=3, metric="ssim")
    assert 0 < res.best_score <= 1


@needs_skopt
def test_vst_mode(pair):
    clean, _ = pair
    from eoqual.denoising.vst import add_poisson_gaussian_noise

    signal = clean * 5.0
    raw = add_poisson_gaussian_noise(signal, 2.0, 0.25, 100.0, seed=0)
    res = optimize_parameters(
        raw,
        signal / 2.0 + 100.0,
        "tv_chambolle",
        vst={"gain": 2.0, "read_noise_dn": 0.25, "offset_dn": 100.0},
        n_calls=6,
        n_initial_points=3,
    )
    assert res.best_kwargs["sigma"] == 1.0
    assert res.best_score >= res.initial_score


@needs_skopt
def test_failures_are_penalized_not_fatal(pair):
    """Un point qui fait échouer la méthode est pénalisé, l'optimisation continue."""
    clean, noisy = pair
    space = {
        "search_space": [{"name": "wavelet", "type": "categorical", "categories": ["db2", "pas_une_ondelette"]}],
        "initial_guess": {"wavelet": "db2"},
    }
    res = optimize_parameters(noisy, clean, "wavelet", sigma=SIGMA, search_space=space, n_calls=5)
    statuses = {h["wavelet"]: h["status"] for h in res.history}
    assert statuses["db2"] == "ok" and statuses["pas_une_ondelette"].startswith("echec")
    assert res.best_params == {"wavelet": "db2"}


@needs_skopt
def test_errors(pair):
    clean, noisy = pair
    with pytest.raises(KeyError):
        optimize_parameters(noisy, clean, "restormer")
    with pytest.raises(ValueError):
        optimize_parameters(noisy, clean, "nl_means", metric="mae")
    with pytest.raises(ValueError):
        optimize_parameters(noisy[:50], clean, "nl_means")


# --- Fichiers d'espaces de recherche et figures --------------------------------


def test_every_space_file_is_valid_jsonc_with_comments():
    """Un fichier JSONC commenté par méthode réglable, chargé tel quel par SEARCH_SPACES."""
    from eoqual.denoising.optimization.jsonc import load_jsonc
    from eoqual.denoising.optimization.search_spaces import SPACES_DIR

    files = sorted(SPACES_DIR.glob("*.jsonc"))
    assert {f.stem for f in files} == set(SEARCH_SPACES)
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert "//" in text, f"{f.name} : aucun commentaire"
        assert validate_search_space(load_jsonc(f), f.name) == SEARCH_SPACES[f.stem]


def test_initial_guess_reproduces_method_defaults(pair):
    """
    Le point de départ de chaque méthode du socle redonne exactement ses
    réglages par défaut (l'optimisation part donc des défauts).
    """
    from eoqual.denoising.optimization.optimizer import _to_kwargs

    clean, noisy = pair
    for method in ("bilateral", "nl_means", "tv_chambolle", "tv_bregman", "wavelet", "nl_bayes"):
        space = SEARCH_SPACES[method]
        kwargs = _to_kwargs(space["initial_guess"], space["search_space"], SIGMA)
        a = _resolve(method)(noisy, **kwargs)
        b = _resolve(method)(noisy, sigma=SIGMA)
        assert np.allclose(a, b), method


def test_load_custom_search_space(tmp_path):
    from eoqual.denoising.optimization import load_search_space

    f = tmp_path / "mon_espace.jsonc"
    f.write_text(
        '{\n  // poids TV\n  "search_space": [{"name": "weight", "type": "float", "low": 0.2, "high": 2.0, '
        '"scale": "sigma"},],\n  "initial_guess": {"weight": 0.5},\n}\n',
        encoding="utf-8",
    )
    assert load_search_space(f)["initial_guess"] == {"weight": 0.5}


@needs_skopt
def test_plots_are_written(pair, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from eoqual.denoising.optimization.plotting import plot_images, plot_optimization

    clean, noisy = pair
    res = optimize_parameters(noisy, clean, "tv_chambolle", sigma=SIGMA, n_calls=6, n_initial_points=3)
    files = plot_optimization(res, out_dir=tmp_path)
    assert any(p.name.endswith("_convergence.png") for p in files)
    assert all(p.exists() for p in files)
    out = plot_images(clean, noisy, noisy, out_path=tmp_path / "img.png")
    assert out.exists()
