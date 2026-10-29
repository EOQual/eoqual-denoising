#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Espaces de recherche des paramètres de chaque méthode.

Un espace de recherche est un dictionnaire ::

    {
      "search_space": [                      # dimensions optimisées
        {"name": "h", "type": "float", "low": 0.2, "high": 1.5, "scale": "sigma"},
        {"name": "patch_size", "type": "int", "low": 3, "high": 11},
        {"name": "wavelet", "type": "categorical", "categories": ["db1", "db2"]},
      ],
      "initial_guess": {"h": 0.6, "patch_size": 7, "wavelet": "db2"},
      "fixed": {"fast_mode": True},          # paramètres fixes (optionnel)
    }

- ``type`` : ``"int"``, ``"float"`` ou ``"categorical"`` ;
- ``scale`` (optionnel) : ``"sigma"`` → la valeur passée à la méthode est
  ``valeur × σ`` ; ``"inverse_sigma"`` → ``valeur / σ``. Les bornes sont
  ainsi exprimées **en multiples de σ**, indépendamment des unités de
  l'image (8 bits, 12 bits, réflectance...) ;
- la dimension spéciale ``sigma_scale`` multiplie le σ transmis à la
  méthode (utile pour les méthodes dont σ est le seul réglage : DRUNet,
  BM3D, EPLL...) ;
- ``initial_guess`` : point de départ, en général les valeurs par défaut
  calibrées de la méthode (voir REFERENCE_TECHNIQUE.md) ; l'optimisation
  ne peut donc pas faire pire que les défauts sur l'image de réglage.

Les espaces par défaut sont les fichiers ``spaces/<méthode>.jsonc``
(commentés dimension par dimension). Un autre fichier au même format se
charge par :func:`load_search_space`, pour adapter les bornes sans toucher
au paquet.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

from .jsonc import load_jsonc

__all__ = ["SEARCH_SPACES", "SPACES_DIR", "get_search_space", "load_search_space", "validate_search_space"]

_SCALES = (None, "sigma", "inverse_sigma")

#: Dossier des espaces de recherche par défaut, un fichier JSONC commenté
#: par méthode (source unique : modifier le fichier, pas le code).
SPACES_DIR = Path(__file__).with_name("spaces")


def _load_default_spaces() -> dict[str, dict]:
    return {path.stem: load_jsonc(path) for path in sorted(SPACES_DIR.glob("*.jsonc"))}


SEARCH_SPACES: dict[str, dict] = _load_default_spaces()
# restormer : aucun paramètre réglable (réseau aveugle, σ ignoré) → pas de fichier.


def validate_search_space(space: dict, name: str = "espace") -> dict:
    """
    Vérifie la cohérence d'un espace de recherche et le renvoie.

    Raises
    ------
    ValueError
        Dimension mal formée, échelle inconnue, ``initial_guess`` incomplet
        ou hors bornes.
    """
    dims = space.get("search_space")
    if not dims:
        raise ValueError(f"{name} : 'search_space' absent ou vide.")
    guess = space.get("initial_guess", {})
    names = set()
    for dim in dims:
        dname = dim.get("name")
        if not dname or dname in names:
            raise ValueError(f"{name} : dimension sans nom ou en double ({dname!r}).")
        names.add(dname)
        dtype = dim.get("type")
        if dtype not in ("int", "float", "categorical"):
            raise ValueError(f"{name} : type inconnu {dtype!r} pour '{dname}'.")
        if dim.get("scale") not in _SCALES:
            raise ValueError(f"{name} : échelle inconnue {dim.get('scale')!r} pour '{dname}'.")
        if dname in guess:
            value = guess[dname]
            if dtype == "categorical":
                if value not in dim["categories"]:
                    raise ValueError(f"{name} : initial_guess['{dname}']={value!r} hors catégories.")
            elif not dim["low"] <= value <= dim["high"]:
                raise ValueError(
                    f"{name} : initial_guess['{dname}']={value} hors bornes [{dim['low']}, {dim['high']}]."
                )
    if guess and set(guess) != names:
        raise ValueError(f"{name} : initial_guess doit couvrir toutes les dimensions ({sorted(names)}).")
    return space


def get_search_space(method: str) -> dict:
    """
    Renvoie (copie de) l'espace de recherche par défaut d'une méthode.

    Raises
    ------
    KeyError
        Si la méthode n'a pas de paramètre réglable (``restormer``) ou est
        inconnue.
    """
    if method not in SEARCH_SPACES:
        raise KeyError(
            f"Pas d'espace de recherche par défaut pour '{method}' "
            "(méthode inconnue, ou sans paramètre réglable comme 'restormer')."
        )
    return copy.deepcopy(SEARCH_SPACES[method])


def load_search_space(path: str | os.PathLike) -> dict:
    """Charge et valide un espace de recherche depuis un fichier JSON/JSONC."""
    return validate_search_space(load_jsonc(path), name=str(path))


for _name, _space in SEARCH_SPACES.items():
    validate_search_space(_space, _name)
