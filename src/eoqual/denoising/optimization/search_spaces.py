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

import ast
import copy
import operator
import os
from pathlib import Path
from typing import Callable

from .jsonc import load_jsonc

__all__ = [
    "SEARCH_SPACES",
    "SPACES_DIR",
    "get_search_space",
    "load_search_space",
    "validate_search_space",
    "build_constraint",
]

_SCALES = (None, "sigma", "inverse_sigma")

#: Dossier des espaces de recherche par défaut, un fichier JSONC commenté
#: par méthode (source unique : modifier le fichier, pas le code).
SPACES_DIR = Path(__file__).with_name("spaces")


def _load_default_spaces() -> dict[str, dict]:
    return {path.stem: load_jsonc(path) for path in sorted(SPACES_DIR.glob("*.jsonc"))}


SEARCH_SPACES: dict[str, dict] = _load_default_spaces()
# restormer : aucun paramètre réglable (réseau aveugle, σ ignoré) → pas de fichier.


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CMPOPS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def _compile_expression(expr: str, names: set[str]):
    """
    Compile une expression de contrainte en fonction ``f(valeurs: dict) -> bool``.

    Grammaire volontairement restreinte, sans ``eval`` : noms de dimensions,
    nombres, ``True``/``False``, ``+ - * / // % **``, moins unaire,
    comparaisons (chaînables), ``and``, ``or``, ``not``, parenthèses.
    Tout autre élément (appel, attribut, indice...) est refusé.
    """
    try:
        tree = ast.parse(expr, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"contrainte invalide {expr!r} : {exc.msg}") from exc

    def check(node):
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError(f"contrainte {expr!r} : dimension inconnue '{node.id}'.")
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, bool)):
                raise ValueError(f"contrainte {expr!r} : constante non numérique.")
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            check(node.left)
            check(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
            check(node.operand)
        elif isinstance(node, ast.Compare) and all(type(o) in _CMPOPS for o in node.ops):
            check(node.left)
            for c in node.comparators:
                check(c)
        elif isinstance(node, ast.BoolOp):
            for v in node.values:
                check(v)
        else:
            raise ValueError(f"contrainte {expr!r} : élément non autorisé ({type(node).__name__}).")

    check(tree)

    def ev(node, env):
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](ev(node.left, env), ev(node.right, env))
        if isinstance(node, ast.UnaryOp):
            v = ev(node.operand, env)
            return -v if isinstance(node.op, ast.USub) else (+v if isinstance(node.op, ast.UAdd) else not v)
        if isinstance(node, ast.Compare):
            left = ev(node.left, env)
            for op, comp in zip(node.ops, node.comparators):
                right = ev(comp, env)
                if not _CMPOPS[type(op)](left, right):
                    return False
                left = right
            return True
        values = (ev(v, env) for v in node.values)  # BoolOp
        return all(values) if isinstance(node.op, ast.And) else any(values)

    return lambda env: bool(ev(tree, env))


def build_constraint(space: dict) -> Callable[[list], bool] | None:
    """
    Construit la fonction de contrainte d'un espace à partir de sa clé
    ``"constraints"`` (liste d'expressions, toutes requises).

    La fonction reçoit un point (une valeur par dimension, dans l'ordre de
    ``search_space``) et renvoie ``True`` s'il est valide. Les expressions
    portent sur les valeurs **de l'espace** (avant conversion par
    ``scale``). Renvoie ``None`` si l'espace n'a pas de contrainte.

    Exemple : ``"constraints": ["k1 < p1 * p1"]`` (NL-Ridge : plus de
    pixels par patch que de patchs par groupe).
    """
    exprs = space.get("constraints") or []
    if not exprs:
        return None
    names = [d["name"] for d in space["search_space"]]
    funcs = [_compile_expression(e, set(names)) for e in exprs]

    def constraint(point: list) -> bool:
        env = dict(zip(names, (v.item() if hasattr(v, "item") else v for v in point)))
        return all(f(env) for f in funcs)

    return constraint


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
    constraints = space.get("constraints", [])
    if not isinstance(constraints, list) or not all(isinstance(c, str) for c in constraints):
        raise ValueError(f"{name} : 'constraints' doit être une liste d'expressions (chaînes).")
    constraint = build_constraint(space)
    if constraint is not None and guess:
        point = [guess[d["name"]] for d in dims]
        if not constraint(point):
            raise ValueError(f"{name} : initial_guess ne respecte pas les contraintes {constraints}.")
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
