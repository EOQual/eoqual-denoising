# Change Log

Versions publiées et contenu de chacune. L'historique détaillé des choix
de conception et des corrections faites pendant le développement est
dans `REFERENCE_TECHNIQUE.md`, annexe A.

## Version History

0.3.2 (2026-11-17)

:   -   **Notation du modèle de bruit alignée sur eoqual-measure** : **σ² = a·L + b**, `a` pente (1/gain), `b` plancher (bruit de lecture²) ; les `a` et `b` de la PTC d'eoqual-measure s'emploient tels quels. Les versions précédentes notaient `var = a + b·L`, lettres inversées.
        Documentation seulement (`REFERENCE_TECHNIQUE.md` §4.4, §5, §6 ; `OPTIMISATION.md` ; `README.md`) : l'API (`gain`, `read_noise_dn`) ne change pas.

0.3.1 (2026-11-16)

:   -   **`optimize_parameters` vérifie ses arguments avant d'importer
        scikit-optimize** : une erreur d'usage (image couleur pour une
        méthode en niveaux de gris, métrique inconnue, formes différentes,
        méthode sans espace de recherche) est signalée comme telle, même
        sans l'extra `optim`, au lieu d'être masquée par l'`ImportError`.
        Les tests correspondants ne dépendent plus de scikit-optimize.
    -   Extras `neural` et `deep` : eoqual-neural v0.1.1 (correctif
        d'installation par Poetry ; aucun changement de code).

0.3.0 (2026-11-16)

:   -   **Réseaux de neurones déplacés dans eoqual-neural** : DRUNet,
        SwinIR, Restormer et GS-DRUNet sont fournis par le nouveau module
        eoqual-neural (réseaux rangés par rôle, PyTorch et poids en un
        seul endroit). Même appel (`denoise(image, "drunet", …)`,
        `drunet_denoise`, `vst_denoise(..., "drunet")`), mêmes sorties.
        Adaptation dans `backends/neural.py` ; anciens chemins conservés.
    -   **Extra `neural`** (eoqual-neural) ; l'extra `deep` le contient
        pour compatibilité et sert aux méthodes classiques calculées avec
        PyTorch (EPLL, TGV, `wavelet_dict`, `nl_ridge_inria`).
    -   **Espace de recherche de NL-Bayes réduit à 7 dimensions** : β₁,
        β₂ et γ retirés, redondants avec `sigma_scale` (multiplier σ par
        s équivaut exactement à multiplier β₁, β₂ par s² et γ par s ;
        `OPTIMISATION.md` §2.3). Budget indicatif 40-60 évaluations.
    -   `estimate_noise_sigma` : note sur la sous-estimation de σ après
        démosaïçage (≈ ×0,5 après Hamilton-Adams) ;
        `examples/optimize_parameters.py` avertit dans ce cas.

0.2.2 (2026-11-16)

:   -   **Étude de sensibilité de τ₀** (NL-Bayes) : une règle relative
        τ₀ = c·σ² a été testée puis écartée en validation indépendante
        (autres images, autre tirage, couleur) ; τ₀ = 4 de l'article est
        conservé. Résultats dans `REFERENCE_TECHNIQUE.md` §4.2 ; script
        reproductible `examples/tau0_study.py`.
    -   Borne haute de τ₀ dans l'espace de recherche de NL-Bayes portée de
        16 à 32 (optimum jusqu'à ~30 en couleur).

0.2.1 (2026-11-16)

:   -   **`vst_denoise(..., noise_factor=C)`** : σ donné à la méthode
        dans le domaine stabilisé (défaut 1), pour le facteur
        C = √κ_Y + 0,2 d'une image démosaïquée ; avertissement hors de
        [0,5 ; 2,5]. `vst_denoise` accepte aussi une image couleur pour
        `nl_bayes`. Un argument `sigma` reste refusé, avec un message qui
        oriente vers `noise_factor`.
    -   La recette d'une image démosaïquée (`REFERENCE_TECHNIQUE.md` §6.4)
        utilise `vst_denoise` au lieu d'un enchaînement manuel ;
        l'optimiseur et les exemples passent aussi par lui.
    -   `examples/one_method_cli.py` : `--noise-factor` et `--kappa-y`,
        image couleur en mode stabilisation.

0.2.0 (2026-11-16)

:   -   **Optimisation sur image couleur** : `optimize_parameters`
        accepte une image (H, W, 3) pour les méthodes couleur
        (`nl_bayes`), score PSNR sur les trois canaux ou SSIM moyen.
    -   **`sigma_scale0`** (et `--kappa-y` dans
        `examples/optimize_parameters.py`) : point de départ de
        `sigma_scale`, typiquement C = √κ_Y + 0,2 pour une image
        démosaïquée ; borne haute de `sigma_scale` de NL-Bayes portée à
        2,5.
    -   **Documentation réorganisée en deux** :
        -   `REFERENCE_TECHNIQUE.md` (cœur de métier) : nouveau
            chapitre 4 « Le niveau de bruit σ » (rôle dans chaque formule
            de NL-Bayes, seuils τ et τ₀, origine de σ et échelle 0-255,
            modèle (a, b) et stabilisation, facteur C) ; stabilisation en
            §5, image démosaïquée en §6 ; origine du terme VST documentée ;
        -   `OPTIMISATION.md` (nouveau) : optimisation des paramètres,
            dont le réglage de σ et du modèle de bruit (fixer (a, b),
            optimiser C), espaces de recherche, exécution, précautions.

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

Voir `REFERENCE_TECHNIQUE.md` §11 et `THIRD_PARTY_LICENSES.md` §3.
