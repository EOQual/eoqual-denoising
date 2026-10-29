#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimisation bayésienne des paramètres d'une méthode de débruitage.

Problème : trouver les paramètres d'une méthode qui maximisent un score
(PSNR ou SSIM) entre l'image débruitée et une image de référence. Chaque
évaluation est coûteuse (un débruitage complet) et c'est une boîte noire
(ni forme analytique, ni gradient, dimensions entières, réelles et
catégorielles mêlées). L'optimisation bayésienne minimise le nombre
d'évaluations : un **processus gaussien** modélise le score (moyenne et
incertitude) à partir des points déjà évalués, et une **fonction
d'acquisition** choisit le point suivant en arbitrant entre exploitation
(près des bons points) et exploration (zones incertaines). Les
``n_initial_points`` premiers points sont tirés au hasard ; le point de
départ (``initial_guess``) compte comme l'un d'eux.

Moteur : scikit-optimize (``skopt``, BSD-3-Clause), extra ``optim``.

Exécution :

- séquentielle (``n_jobs=1``) : ``skopt.gp_minimize`` ;
- parallèle (``n_jobs > 1``) : **file d'attente glissante** — jusqu'à
  ``n_jobs`` évaluations en cours dans des processus séparés ; dès qu'une
  se termine, son résultat est transmis au modèle (``tell``) et un
  nouveau point est soumis (``ask``). Aucun cœur n'attend la fin d'un
  lot. Limite assumée : les points en cours ne sont pas pris en compte
  par ``ask`` (pas de stratégie de « menteur constant »).

Les images et paramètres fixes sont transmis **une seule fois** à chaque
processus (initialiseur du pool) ; seuls les paramètres testés circulent
ensuite. La journalisation (module ``logging`` et CSV) a lieu dans le
processus principal uniquement.

Un espace entièrement catégoriel et petit (ex. ``swinir`` : 3 jeux de
poids) est évalué **exhaustivement** plutôt que par processus gaussien.

Référence : Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., de
Freitas, N. (2016). "Taking the Human Out of the Loop: A Review of
Bayesian Optimization." Proc. IEEE 104(1), 148-175.
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import itertools
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..utils import check_image, resolve_sigma
from .search_spaces import get_search_space, validate_search_space

__all__ = ["optimize_parameters", "OptimizationResult", "default_n_jobs", "METRICS", "PENALTY"]

logger = logging.getLogger("eoqual.denoising.optimization")

#: Valeur de l'objectif (minimisé) attribuée à une évaluation en échec.
PENALTY = 1e10

#: Scores disponibles (maximisés).
METRICS = ("psnr", "ssim")


# ---------------------------------------------------------------------------
# Évaluation (exécutée dans les processus de calcul : aucune journalisation)
# ---------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _init_worker(context: dict) -> None:
    """Initialiseur du pool : reçoit images et paramètres fixes une seule fois."""
    _WORKER.clear()
    _WORKER.update(context)
    threads = context.get("threads_per_job")
    if threads:
        try:
            import torch

            torch.set_num_threads(int(threads))
        except ImportError:
            pass


def _to_kwargs(params: dict, dims: list[dict], sigma: float) -> dict:
    """Convertit un point de l'espace en arguments de la méthode."""
    kwargs = {}
    sigma_eff = sigma
    by_name = {d["name"]: d for d in dims}
    for name, value in params.items():
        value = value.item() if isinstance(value, np.generic) else value
        scale = by_name[name].get("scale")
        if name == "sigma_scale":
            sigma_eff = sigma * value
            continue
        if scale == "sigma":
            value = value * sigma
        elif scale == "inverse_sigma":
            value = value / sigma
        kwargs[name] = value
    return {"sigma": sigma_eff, **kwargs}


def _score(reference, image, metric, data_range) -> float:
    if metric == "psnr":
        from skimage.metrics import peak_signal_noise_ratio

        return float(peak_signal_noise_ratio(reference, image, data_range=data_range))
    from skimage.metrics import structural_similarity

    return float(structural_similarity(reference, image, data_range=data_range))


def _denoise_with(ctx: dict, params: dict) -> np.ndarray:
    """
    Débruite l'image du contexte avec un point de l'espace. En mode VST, la
    méthode travaille dans le domaine transformé (σ = 1, ou ``sigma_scale``
    si cette dimension est optimisée).
    """
    from ..runner import _resolve
    from ..vst import generalized_anscombe, inverse_generalized_anscombe

    # Appel direct de la fonction de la méthode (résolue une fois par appel),
    # équivalent à runner.denoise(image, method, **kwargs).
    func = _resolve(ctx["method"])
    kwargs = {**ctx["fixed"], **_to_kwargs(params, ctx["dims"], ctx["sigma"])}
    vst = ctx["vst"]
    if not vst:
        return func(ctx["noisy"], **kwargs)
    g = {k: vst.get(k, 0.0) for k in ("gain", "read_noise_dn", "offset_dn")}
    d = generalized_anscombe(ctx["noisy"], **g)
    d_hat = func(d, **kwargs)
    return inverse_generalized_anscombe(d_hat, **g, method=vst.get("inverse", "exact_unbiased"))


def _evaluate(params: dict) -> tuple[float | None, str, float, int]:
    """Évalue un point : (score ou None, statut, durée, pid)."""
    ctx = _WORKER
    t0 = time.perf_counter()
    try:
        out = _denoise_with(ctx, params)
    except Exception as exc:  # noqa: BLE001 — tout échec est pénalisé, pas fatal
        return None, f"echec: {type(exc).__name__}: {str(exc)[:120]}", time.perf_counter() - t0, os.getpid()
    if out.shape != ctx["reference"].shape or not np.all(np.isfinite(out)):
        return None, "echec: sortie invalide (forme ou valeurs non finies)", time.perf_counter() - t0, os.getpid()
    score = _score(ctx["reference"], out, ctx["metric"], ctx["data_range"])
    return score, "ok", time.perf_counter() - t0, os.getpid()


# ---------------------------------------------------------------------------
# Journal CSV
# ---------------------------------------------------------------------------


def _append_csv(path: Path, row: dict, fieldnames: list[str]) -> None:
    """Ajoute une ligne au journal CSV (verrou de fichier sous POSIX)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        try:
            import fcntl

            fcntl.flock(f, fcntl.LOCK_EX)
        except ImportError:  # Windows : pas de verrou (écriture depuis le seul processus principal)
            fcntl = None
        try:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        finally:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    """
    Résultat d'une optimisation.

    Parameters
    ----------
    method : str
        Méthode optimisée.
    metric : str
        Score maximisé (``"psnr"`` ou ``"ssim"``).
    best_params : dict
        Meilleur point, dans les unités de l'espace de recherche (multiples
        de σ pour les dimensions à échelle).
    best_kwargs : dict
        Arguments effectivement passés à la méthode pour ce point (``sigma``
        compris, valeurs converties en unités de l'image) : réutilisables
        tels quels avec ``denoise(image, method, **best_kwargs)``.
    best_score : float
        Meilleur score obtenu.
    initial_score : float or None
        Score du point de départ (``initial_guess``), pour mesurer le gain.
    history : list of dict
        Une entrée par évaluation : paramètres, score, statut, durée.
    skopt_result : object or None
        Résultat brut de scikit-optimize (``None`` en recherche exhaustive).
    """

    method: str
    metric: str
    best_params: dict
    best_kwargs: dict
    best_score: float
    initial_score: float | None
    history: list = field(default_factory=list)
    skopt_result: Any = None

    @property
    def gain(self) -> float | None:
        """Gain du meilleur point sur le point de départ (même unité que le score)."""
        return None if self.initial_score is None else self.best_score - self.initial_score


def default_n_jobs(threads_per_job: int = 1) -> int:
    """Nombre d'évaluations parallèles sans sursouscrire les cœurs : cœurs // fils par évaluation."""
    return max(1, (os.cpu_count() or 1) // max(1, threads_per_job))


def _build_dimensions(dims: list[dict]):
    from skopt.space import Categorical, Integer, Real

    out = []
    for d in dims:
        if d["type"] == "int":
            out.append(Integer(d["low"], d["high"], name=d["name"]))
        elif d["type"] == "float":
            out.append(Real(d["low"], d["high"], name=d["name"]))
        else:
            out.append(Categorical(d["categories"], name=d["name"]))
    return out


def optimize_parameters(
    noisy: np.ndarray,
    reference: np.ndarray,
    method: str,
    sigma: float | None = None,
    search_space: dict | None = None,
    n_calls: int = 30,
    n_initial_points: int = 10,
    random_state: int | None = 0,
    n_jobs: int = 1,
    threads_per_job: int | None = None,
    metric: str = "psnr",
    data_range: float | None = None,
    fixed: dict | None = None,
    vst: dict | None = None,
    space_constraint: Callable[[list], bool] | None = None,
    log_csv: str | os.PathLike | None = None,
) -> OptimizationResult:
    """
    Cherche les paramètres d'une méthode qui maximisent un score contre une
    image de référence, par optimisation bayésienne.

    Parameters
    ----------
    noisy : np.ndarray
        Image bruitée 2D (unités quelconques ; en DN si ``vst`` est fourni).
    reference : np.ndarray
        Image de référence (propre), même forme et mêmes unités.
    method : str
        Méthode du registre (voir ``list_methods()``).
    sigma : float, optional
        Écart-type du bruit (unités de l'image). Estimé sur ``noisy`` si
        absent. Sert d'unité aux dimensions à échelle (``scale``) et de
        base à ``sigma_scale``. Ignoré si ``vst`` est fourni (σ = 1 dans le
        domaine transformé).
    search_space : dict, optional
        Espace de recherche (format décrit dans
        :mod:`~eoqual.denoising.optimization.search_spaces`). Par défaut,
        celui de la méthode (``SEARCH_SPACES``).
    n_calls : int
        Budget total d'évaluations. Ordre de grandeur : 15-30 pour 1 à 3
        dimensions, 50-100 au-delà de 5. Par défaut ``30``.
    n_initial_points : int
        Points tirés au hasard avant que le modèle guide la recherche (20 à
        40 % de ``n_calls``). Par défaut ``10``.
    random_state : int, optional
        Graine (reproductibilité). Par défaut ``0``.
    n_jobs : int
        Évaluations menées en parallèle (file d'attente glissante). Par
        défaut ``1`` (séquentiel). Voir :func:`default_n_jobs`. Pour les
        méthodes PyTorch, chaque processus charge son propre modèle :
        surveiller la mémoire, et préférer ``n_jobs=1`` sur GPU.
    threads_per_job : int, optional
        Fils PyTorch par processus quand ``n_jobs > 1`` (évite que chaque
        processus occupe tous les cœurs). Par défaut ``cœurs // n_jobs``.
    metric : {"psnr", "ssim"}
        Score maximisé. Par défaut ``"psnr"``.
    data_range : float, optional
        Dynamique utilisée par le score. Par défaut ``max − min`` de la
        référence.
    fixed : dict, optional
        Paramètres fixes transmis à chaque appel (ex. ``nlridge_path``,
        ``device``) ; ils complètent ceux de ``search_space["fixed"]``.
    vst : dict, optional
        Optimiser la chaîne GAT + méthode + inverse (bruit
        poisson-gaussien) : ``{"gain": ..., "read_noise_dn": ...,
        "offset_dn": ..., "inverse": "exact_unbiased"}``.
    space_constraint : callable, optional
        Contrainte sur les points tirés : reçoit une liste de valeurs (ordre
        des dimensions) et renvoie ``True`` si le point est valide. Les
        points invalides sont rejetés au tirage, sans évaluation.
    log_csv : path, optional
        Journal CSV, une ligne par évaluation (ajout en fin de fichier).

    Returns
    -------
    OptimizationResult

    Raises
    ------
    ImportError
        Si scikit-optimize n'est pas installé (extra ``optim``).
    ValueError
        Paramètres incohérents, ou toutes les évaluations en échec.
    """
    try:
        from skopt import Optimizer, gp_minimize
        from skopt.space import Space
    except ImportError as exc:
        raise ImportError(
            "L'optimisation nécessite 'scikit-optimize'. Installer avec : "
            'pip install "eoqual-denoising[optim]"'
        ) from exc

    if metric not in METRICS:
        raise ValueError(f"metric doit valoir {METRICS}, reçu '{metric}'.")
    if n_jobs < 1:
        raise ValueError(f"n_jobs doit être ≥ 1, reçu {n_jobs}.")
    noisy = check_image(noisy)
    reference = check_image(reference)
    if noisy.shape != reference.shape:
        raise ValueError(f"Formes différentes : bruitée {noisy.shape}, référence {reference.shape}.")

    space = validate_search_space(search_space or get_search_space(method), method)
    dims = space["search_space"]
    names = [d["name"] for d in dims]
    sigma_val = 1.0 if vst else resolve_sigma(noisy, sigma)
    if data_range is None:
        data_range = float(reference.max() - reference.min()) or 1.0
    guess = space.get("initial_guess") or {}
    x0 = [[guess[n] for n in names]] if guess else []
    if space_constraint is not None and x0 and not space_constraint(x0[0]):
        raise ValueError("initial_guess ne respecte pas space_constraint.")

    context = {
        "method": method,
        "noisy": noisy,
        "reference": reference,
        "sigma": sigma_val,
        "dims": dims,
        "fixed": {**space.get("fixed", {}), **(fixed or {})},
        "vst": dict(vst) if vst else None,
        "metric": metric,
        "data_range": data_range,
        "threads_per_job": None,
    }
    fieldnames = ["timestamp", "pid", *names, metric, "status", "seconds"]
    history: list[dict] = []
    log_path = Path(log_csv) if log_csv else None

    def record(point, outcome) -> float:
        score, status, seconds, pid = outcome
        params = dict(zip(names, (v.item() if isinstance(v, np.generic) else v for v in point)))
        entry = {**params, metric: score, "status": status, "seconds": round(seconds, 3)}
        history.append(entry)
        if log_path:
            _append_csv(log_path, {"timestamp": time.time(), "pid": pid, **entry}, fieldnames)
        if score is None:
            logger.warning("[%s] %s → %s", method, params, status)
            return PENALTY
        logger.info("[%s] %s → %s = %.4f (%.1f s)", method, params, metric, score, seconds)
        return -score

    sk_dims = _build_dimensions(dims)
    categorical_only = all(d["type"] == "categorical" for d in dims)
    grid_size = int(np.prod([len(d["categories"]) for d in dims])) if categorical_only else None
    skopt_result = None

    if categorical_only and grid_size <= n_calls:
        # Recherche exhaustive : plus sûre et moins coûteuse que le modèle.
        _init_worker(context)
        points = [list(p) for p in itertools.product(*(d["categories"] for d in dims))]
        if space_constraint is not None:
            points = [p for p in points if space_constraint(p)]
        for p in points:
            record(p, _evaluate(dict(zip(names, p))))
    elif n_jobs == 1:
        _init_worker(context)

        def objective(point):
            return record(point, _evaluate(dict(zip(names, point))))

        skopt_result = gp_minimize(
            objective,
            Space(sk_dims),
            x0=x0 or None,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=random_state,
            space_constraint=space_constraint,
        )
    else:
        context["threads_per_job"] = threads_per_job or max(1, (os.cpu_count() or 1) // n_jobs)
        optimizer = Optimizer(
            dimensions=sk_dims,
            n_initial_points=n_initial_points,
            random_state=random_state,
            space_constraint=space_constraint,
        )
        pending = list(x0)
        dispatched = 0
        with cf.ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker, initargs=(context,)) as ex:
            in_flight: dict = {}

            def submit() -> None:
                nonlocal dispatched
                if dispatched >= n_calls:
                    return
                point = pending.pop(0) if pending else optimizer.ask()
                in_flight[ex.submit(_evaluate, dict(zip(names, point)))] = point
                dispatched += 1

            for _ in range(min(n_jobs, n_calls)):
                submit()
            while in_flight:
                done, _ = cf.wait(in_flight, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    point = in_flight.pop(fut)
                    optimizer.tell(point, record(point, fut.result()))
                    submit()
        skopt_result = optimizer.get_result()

    valid = [h for h in history if h[metric] is not None]
    if not valid:
        raise ValueError(
            f"Toutes les évaluations de '{method}' ont échoué. Premier statut : {history[0]['status'] if history else '—'}"
        )
    best = max(valid, key=lambda h: h[metric])
    best_params = {n: best[n] for n in names}
    initial = next(
        (h[metric] for h in history if guess and all(h[n] == guess[n] for n in names)),
        None,
    )
    return OptimizationResult(
        method=method,
        metric=metric,
        best_params=best_params,
        best_kwargs={**context["fixed"], **_to_kwargs(best_params, dims, sigma_val)},
        best_score=best[metric],
        initial_score=initial,
        history=history,
        skopt_result=skopt_result,
    )
