# Change Log

Versions publiées et contenu de chacune. L'historique détaillé des choix
de conception et des corrections faites pendant le développement est
dans `REFERENCE_TECHNIQUE.md`, annexe A.

## Version History

0.1.2 (2026-11-04)

:   -   `REFERENCE_TECHNIQUE.md` §10 : **débruiter une image
        démosaïquée** (cas particulier du bruit coloré) — modèle
        ``var = a + b·L`` et correspondance avec la GAT, effet du
        démosaïçage sur le bruit (κ_Y), facteur ``σ = √κ_Y + 0.2``,
        erreurs à éviter (σ estimé après démosaïçage : moitié du vrai),
        résultats indicatifs ; renvois croisés avec eoqual-demosaicing
        (chapitre 11). §9.1 nuancé (débruiter après le démosaïçage).

0.1.1 (2026-11-02)

:   -   **Contraintes déclaratives dans les espaces de recherche** : clé
        `"constraints"` (expressions sur les dimensions, analysées sans
        évaluation de code, `build_constraint`), appliquée
        automatiquement par `optimize_parameters` et cumulable avec
        `space_constraint=`.
    -   **Correction NL-Ridge** : l'espace de recherche autorisait
        k₁ ≥ p₁² (matrice singulière à l'étape 1, résultat inutilisable) ;
        contrainte `k1 < p1 * p1` ajoutée à `nl_ridge` et
        `nl_ridge_inria`, et ces méthodes refusent désormais un tel
        réglage (`ValueError`).

0.1.0 (2026-10-29)

:   -   **17 méthodes de débruitage**, appel unifié
        `denoise(image, method, **kwargs)` (signature commune
        `f(image, sigma=None, ...)`, garantie par test) :
        -   socle (installé par défaut, licences permissives) :
            `bilateral`, `nl_means`, `tv_chambolle`, `tv_bregman`,
            `wavelet`, et deux réimplémentations MIT depuis les articles,
            `nl_bayes` (moteur vectorisé, couleur Y₀U₀V₀ / RGB / canal par
            canal) et `nl_ridge` (contraintes linéaire et affine) ;
        -   extra `deep` (deepinv) : `drunet`, `swinir`, `restormer`,
            `gsdrunet`, `epll`, `tgv` ; extra `ptwt` : `wavelet_dict`
            (**EUPL-1.2**) ; extra `bm3d` : `bm3d` (**non commercial**) ;
        -   composants externes de référence : `nl_ridge_inria` (clone
            **AGPL-3.0**) et `nl_bayes_ipol` (exécutable **GPL-3.0+**).
    -   **Stabilisation de variance** du bruit poisson-gaussien
        (`vst_denoise`) : transformée d'Anscombe généralisée et inverse
        exacte non biaisée, paramètres alignés sur la PTC
        d'eoqual-measure.
    -   **Optimisation bayésienne des paramètres** (extra `optim`) :
        espaces de recherche par défaut en fichiers JSONC commentés
        (bornes en multiples de σ), exécution séquentielle ou parallèle,
        journal CSV, figures, mode stabilisation de variance.
    -   **Benchmark** (`BENCHMARK.md`) : réglages par défaut et optimisés
        avec séparation réglage / évaluation, image brute à faible flux,
        moteurs et couleur de NL-Bayes.
    -   Documentation : `README.md`, `REFERENCE_TECHNIQUE.md`,
        `BENCHMARK.md`, `THIRD_PARTY_LICENSES.md`, `BIBLIOGRAPHY.md`,
        `CONTRIBUTING.md`.
    -   Méthode non incluse : WNLB (source sans licence, aucune
        publication identifiée).

## Évolutions futures

Voir `REFERENCE_TECHNIQUE.md` §9 et `THIRD_PARTY_LICENSES.md` §3.
