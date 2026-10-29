# Change Log

Versions publiées et contenu de chacune. L'historique détaillé des choix
de conception et des corrections faites pendant le développement est
dans `REFERENCE_TECHNIQUE.md`, annexe A.

## Version History

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
