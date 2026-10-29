"""
Optimisation bayésienne des paramètres des méthodes de débruitage
(extra ``optim`` : scikit-optimize).

>>> from eoqual.denoising.optimization import optimize_parameters
>>> res = optimize_parameters(noisy, reference, "nl_means", n_calls=25)
>>> res.best_params, res.best_score, res.gain
>>> clean = denoise(other_image, "nl_means", **res.best_kwargs)

Voir REFERENCE_TECHNIQUE.md §5 pour la méthode, le dimensionnement du
budget et les précautions (sur-apprentissage sur l'image de réglage).
"""
from .optimizer import METRICS, PENALTY, OptimizationResult, default_n_jobs, optimize_parameters
from .search_spaces import SEARCH_SPACES, get_search_space, load_search_space, validate_search_space

__all__ = [
    "optimize_parameters",
    "OptimizationResult",
    "default_n_jobs",
    "METRICS",
    "PENALTY",
    "SEARCH_SPACES",
    "get_search_space",
    "load_search_space",
    "validate_search_space",
]
