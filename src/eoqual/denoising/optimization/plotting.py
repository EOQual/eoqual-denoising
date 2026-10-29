#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graphiques de diagnostic d'une optimisation : convergence, projections du
modèle de substitution, distribution des points évalués, et comparaison
visuelle des images. Les évaluations en échec sont exclues.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .optimizer import PENALTY

__all__ = ["plot_optimization", "plot_images"]


def _finish(fig, out_dir, name, show):
    import matplotlib.pyplot as plt

    fig.tight_layout()
    path = None
    if out_dir is not None:
        path = Path(out_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_optimization(
    result: "eoqual.denoising.optimization.optimizer.OptimizationResult",
    out_dir: str | os.PathLike | None = None,
    show: bool = False,
) -> list[Path]:
    """
    Trace la convergence et, si un modèle a été entraîné, les projections
    1D/2D de l'objectif estimé et la distribution des points évalués.

    Parameters
    ----------
    result : eoqual.denoising.optimization.optimizer.OptimizationResult
        Résultat de :func:`optimize_parameters`.
    out_dir : path, optional
        Dossier où enregistrer les figures (PNG).
    show : bool
        Afficher les figures (``plt.show``). Par défaut ``False``.

    Returns
    -------
    list of Path
        Fichiers enregistrés.
    """
    import matplotlib.pyplot as plt

    metric = result.metric
    scores = [h[metric] for h in result.history]
    saved = []

    fig, ax = plt.subplots(figsize=(8, 4.5))
    valid = [(i + 1, s) for i, s in enumerate(scores) if s is not None]
    if valid:
        idx, vals = zip(*valid)
        ax.plot(idx, vals, "o", color="0.6", label="évaluations")
        ax.plot(idx, np.maximum.accumulate(vals), "-", lw=2, label="meilleur cumulé")
        if result.initial_score is not None:
            ax.axhline(result.initial_score, ls="--", color="C3", label="point de départ")
    ax.set_xlabel("évaluation")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Convergence — {result.method}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    saved.append(_finish(fig, out_dir, f"{result.method}_convergence.png", show))

    res = result.skopt_result
    if res is None or not getattr(res, "models", None):
        return [p for p in saved if p]

    from skopt.plots import plot_evaluations, plot_objective
    from skopt.utils import create_result

    vals = np.asarray(res.func_vals, dtype=float)
    bad = np.isclose(vals, PENALTY)
    if (~bad).any():
        vals[bad] = vals[~bad].max() + 0.1 * abs(vals[~bad].max())
    display = create_result(res.x_iters, vals.tolist(), res.space)
    display.models = res.models
    best = int(np.argmin(display.func_vals))
    display.x, display.fun = display.x_iters[best], display.func_vals[best]

    if len(res.space.dimensions) >= 2:
        plot_objective(display)
        saved.append(_finish(plt.gcf(), out_dir, f"{result.method}_objectif.png", show))
        plot_evaluations(display)
        saved.append(_finish(plt.gcf(), out_dir, f"{result.method}_evaluations.png", show))
    else:
        # Une seule dimension : les tracés de skopt ne la gèrent pas
        # (plot_evaluations échoue) ; score en fonction du paramètre.
        name = res.space.dimensions[0].name
        pts = [(h[name], h[metric]) for h in result.history if h[metric] is not None]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "o")
            ax.axvline(result.best_params[name], ls="--", color="C3", label="meilleur")
            ax.legend()
        ax.set_xlabel(name)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Évaluations — {result.method}")
        ax.grid(True, alpha=0.3)
        saved.append(_finish(fig, out_dir, f"{result.method}_evaluations.png", show))
    return [p for p in saved if p]


def plot_images(
    reference: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    title: str = "",
    out_path: str | os.PathLike | None = None,
    show: bool = False,
) -> Path | None:
    """Affiche côte à côte référence, image bruitée et image débruitée."""
    import matplotlib.pyplot as plt

    lo, hi = float(reference.min()), float(reference.max())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (name, img) in zip(axes, (("référence", reference), ("bruitée", noisy), ("débruitée", denoised))):
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(name)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    out_dir, name = (Path(out_path).parent, Path(out_path).name) if out_path else (None, "")
    return _finish(fig, out_dir, name, show)
