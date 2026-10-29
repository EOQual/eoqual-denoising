# Contribuer à eoqual-denoising

Les contributions sont bienvenues, sous toutes leurs formes : signalement
de bug, correction, proposition de nouvelle méthode, implémentation d'une
méthode déjà proposée, amélioration de la documentation.

## Signaler un bug

Ouvrez une issue avec :
- la méthode concernée (`list_methods()` pour voir le registre complet) ;
- un exemple minimal reproduisant le problème (image de test, `sigma`,
  code) ;
- le résultat obtenu vs. attendu.

## Proposer une méthode

Ouvrez une issue décrivant la méthode (nom, référence bibliographique,
ce qu'elle apporte par rapport aux méthodes déjà cataloguées — voir
`REFERENCE_TECHNIQUE.md` pour l'existant). Pas besoin de savoir
l'implémenter pour la proposer.

## Implémenter une méthode

Convention du projet : une méthode = une fonction
`<méthode>_denoise(image: np.ndarray, sigma: float | None = None, ...) -> np.ndarray`,
placée dans `src/eoqual/denoising/backends/<méthode>/<fichier>.py`,
utilisant `utils.check_image` et `utils.resolve_sigma`, avec `sigma`
exprimé dans les unités de l'image, et **sans paramètre nommé `method`**
(réservé au dispatcher `denoise`) ; `test_unified_signature` le
vérifie. Elle est ensuite déclarée dans
`config.py::METHODS_CONFIGS` (extra, composant externe, licence,
restriction, référence), dans le dispatch de `runner.py::_DISPATCH`,
exposée dans `__init__.py`, et sa référence complète est ajoutée à
`BIBLIOGRAPHY.md`. Si elle a des paramètres réglables, lui donner un
espace de recherche dans `optimization/search_spaces.py`
(`SEARCH_SPACES`, bornes en multiples de σ, `initial_guess` = valeurs
par défaut) : le test `test_every_tunable_method_has_a_search_space`
le vérifie.

**Licence — condition de fusion, pas une formalité.**

- **Réimplémentation depuis une publication** : bienvenue sous MIT.
  L'en-tête du module doit citer la ou les publications utilisées et
  indiquer qu'aucun code tiers n'a été consulté ni repris (voir
  `THIRD_PARTY_LICENSES.md` §4).
- **Reprise d'un code publié ailleurs** (copie, portage quasi littéral,
  même structure) : seulement si sa licence *vérifiée* (fichier LICENSE
  ou champ `license` de l'API GitHub/PyPI, pas une supposition) est
  permissive, avec conservation de sa notice. Un code **sans licence**
  n'est pas acceptable.
- **Bibliothèque sous licence restrictive** (copyleft, non commerciale) :
  acceptable seulement en **extra pip optionnel** ou en **composant
  externe** fourni par l'utilisateur, jamais importée ni installée par
  défaut, avec un avertissement explicite dans `THIRD_PARTY_LICENSES.md`
  (voir `bm3d`, `nl_ridge_inria`, `nl_bayes_ipol` comme modèles ; et
  `nl_bayes`, `nl_ridge` comme modèles de réimplémentation clean-room
  depuis un article).
- **Dépendance lourde** (PyTorch, poids de modèle) : en extra même si sa
  licence est permissive, pour garder le socle léger (voir `drunet`). La
  licence des **poids** se vérifie séparément de celle du code.

Une erreur de licence constatée après coup sera corrigée, et
l'intégration retirée si nécessaire.

## Tests

Toute nouvelle méthode doit être testée (voir
`REFERENCE_TECHNIQUE.md` §8.1) :

- une méthode du socle est couverte automatiquement par
  `tests/test_socle.py` (gain de PSNR, σ estimé, invariance d'échelle) ;
- une méthode en extra ou externe ajoute son test dans
  `tests/test_deepinv.py` ou `tests/test_external.py`, avec un `skip`
  propre si le composant est absent ;
- `tests/test_registry.py` vérifie son contrat d'appel, et
  `tests/test_optimization.py` son espace de recherche ;
- utiliser les fixtures de `tests/conftest.py` et `check_gain` de
  `tests/_helpers.py` ; tolérances statistiques tirées de l'incertitude.

Mettre à jour `BENCHMARK.md` avec `examples/benchmark.py`.

## Documentation

Toute méthode nouvelle ou modifiée doit rester reflétée dans `README.md`
(catalogue), `REFERENCE_TECHNIQUE.md` (principe, implémentation,
limites), `THIRD_PARTY_LICENSES.md` (licence) et `BIBLIOGRAPHY.md`
(référence). Les docstrings seules ne suffisent pas.
