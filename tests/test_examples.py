#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemples d'utilisation (``examples/``) : chaque script est lancé comme un
utilisateur le ferait (sous-processus), avec des tailles réduites, et ses
sorties sont vérifiées. Garantit que les exemples restent synchrones avec
l'API.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from eoqual.denoising import add_gaussian_noise, astronaut_gray, astronaut_rgb

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def run(script, *args, cwd=None):
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONWARNINGS": "ignore"}
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / script), *map(str, args)],
        capture_output=True, text=True, env=env, cwd=cwd, timeout=600, check=False,
    )
    assert proc.returncode == 0, f"{script} a échoué :\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


@pytest.fixture(scope="module")
def image_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("img")
    gray = np.clip(add_gaussian_noise(astronaut_gray(64), 0.1, seed=0), 0, 1)
    rgb = np.clip(add_gaussian_noise(astronaut_rgb(64), 0.1, seed=0), 0, 1)
    cv2.imwrite(str(d / "gris.png"), np.rint(255 * gray).astype(np.uint8))
    cv2.imwrite(str(d / "couleur.png"), cv2.cvtColor(np.rint(255 * rgb).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return d


def test_one_method_cli_gray(image_files, tmp_path):
    out = tmp_path / "out.tif"
    run("one_method_cli.py", image_files / "gris.png", out, "--method", "nl_means")
    res = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert res.shape == (64, 64) and res.dtype == np.float32


def test_one_method_cli_color_nl_bayes(image_files, tmp_path):
    out = tmp_path / "out.png"
    run("one_method_cli.py", image_files / "couleur.png", out, "--method", "nl_bayes", "--color-mode", "yuv")
    assert cv2.imread(str(out)).shape == (64, 64, 3)


def test_one_method_cli_vst(image_files, tmp_path):
    out = tmp_path / "out.tif"
    run("one_method_cli.py", image_files / "gris.png", out, "--method", "tv_chambolle", "--vst-gain", "2")
    assert out.exists()


def test_one_method_cli_list():
    assert "nl_bayes" in run("one_method_cli.py", "--list")


def test_all_methods_overview(tmp_path):
    out = tmp_path / "planche.png"
    stdout = run("all_methods_overview.py", "--size", 64, "--methods", "nl_means", "tv_chambolle", "--out", out)
    assert out.exists() and "nl_means" in stdout


def test_low_light_vst():
    stdout = run("low_light_vst.py", "--size", 64, "--methods", "nl_means", "tv_chambolle")
    assert "exact_unbiased" in stdout


@pytest.mark.skipif(importlib.util.find_spec("skopt") is None, reason="scikit-optimize absent (extra optim)")
def test_optimize_parameters(tmp_path):
    run("optimize_parameters.py", "--method", "tv_chambolle", "--n-calls", 5, "--n-initial-points", 3,
        "--output-dir", tmp_path, "--no-plot")
    for name in ("tv_chambolle_log.csv", "tv_chambolle_best.json", "tv_chambolle_denoised.tif"):
        assert (tmp_path / name).exists(), name


def test_benchmark_without_optimization(tmp_path):
    out = tmp_path / "bench.md"
    run("benchmark.py", "--methods", "nl_means", "tv_chambolle", "--sigmas", 0.1, "--no-optim", "--out", out)
    text = out.read_text(encoding="utf-8")
    assert "`nl_means`" in text and "Paramètres optimisés" not in text


def test_nl_bayes_benchmark(tmp_path):
    out = tmp_path / "nlb.md"
    run("nl_bayes_benchmark.py", "--size", 128, "--sigmas", 0.1, "--out", out)
    text = out.read_text(encoding="utf-8")
    assert "`yuv`" in text and "boucle" in text
