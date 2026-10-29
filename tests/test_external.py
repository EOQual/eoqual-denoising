#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthodes à licence restrictive ou composant externe : ``bm3d`` (extra
non commercial), ``nl_ridge_inria`` (clone AGPL), ``nl_bayes_ipol``
(exécutable GPL). Sautées si le composant est absent, sauf le protocole
d'appel de NL-Bayes IPOL, testé avec un faux exécutable.
"""
import os
import stat
import sys

import numpy as np
import pytest

from _helpers import SIGMA, check_gain
from eoqual.denoising.runner import denoise


def test_bm3d(clean, noisy):
    """Extra 'bm3d' (licence non commerciale) — skip si absent."""
    pytest.importorskip("bm3d")
    check_gain(clean, noisy, denoise(noisy, method="bm3d", sigma=SIGMA))


@pytest.mark.skipif("EOQUAL_NLRIDGE_PATH" not in os.environ, reason="clone NL-Ridge non fourni")
def test_nl_ridge_inria(clean, noisy):
    pytest.importorskip("torch")
    pytest.importorskip("einops")
    check_gain(clean, noisy, denoise(noisy, method="nl_ridge_inria", sigma=SIGMA))


def test_nl_ridge_inria_missing_path_raises(noisy, monkeypatch):
    monkeypatch.delenv("EOQUAL_NLRIDGE_PATH", raising=False)
    with pytest.raises(ImportError, match="NL-Ridge"):
        denoise(noisy, method="nl_ridge_inria", sigma=SIGMA)


@pytest.mark.skipif("EOQUAL_NLBAYES_EXE" not in os.environ, reason="exécutable NL_Bayes non fourni")
def test_nl_bayes_ipol(clean, noisy):
    check_gain(clean, noisy, denoise(noisy, method="nl_bayes_ipol", sigma=SIGMA))


@pytest.mark.skipif(sys.platform.startswith("win"), reason="script exécutable POSIX")
def test_nl_bayes_ipol_command_line_protocol(noisy, tmp_path):
    """
    Vérifie l'échange avec le programme externe (ordre des arguments,
    fichiers PNG, conversion d'échelle) à l'aide d'un faux NL_Bayes qui
    recopie l'image d'entrée comme sortie débruitée.
    """
    fake = tmp_path / "NL_Bayes"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import shutil, sys\n"
        "a = sys.argv[1:]\n"
        "assert len(a) == 13, a\n"
        "assert a[2] == '0'  # pas de re-bruitage\n"
        "assert abs(float(a[1]) - 25.5) < 1.0, a[1]\n"
        "shutil.copy(a[0], a[4])\n"
        "shutil.copy(a[0], a[5])\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    img = np.clip(noisy, 0, 1)
    out = denoise(img, method="nl_bayes_ipol", sigma=SIGMA, nlbayes_exe=fake, data_range=(0.0, 1.0))
    assert out.shape == img.shape
    # seule différence attendue : la quantification 8 bits de l'échange
    assert np.max(np.abs(out - img)) <= 0.5 / 255 + 1e-9


def test_nl_bayes_ipol_missing_exe_raises(noisy, monkeypatch, tmp_path):
    monkeypatch.delenv("EOQUAL_NLBAYES_EXE", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="NL_Bayes"):
        denoise(noisy, method="nl_bayes_ipol", sigma=SIGMA)
