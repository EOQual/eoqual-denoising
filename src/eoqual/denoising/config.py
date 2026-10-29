"""
Registre central des méthodes de débruitage disponibles dans
``eoqual.denoising``.

Chaque entrée décrit :

- ``extra`` : extra pip requis (``None`` = socle, ``"deep"``, ``"bm3d"``,
  ``"deep + ptwt"``) ;
- ``external`` : composant à fournir par l'utilisateur hors pip (``None``
  si aucun) — clone de dépôt ou exécutable compilé ;
- ``license`` : licence amont vérifiée du code qui fait réellement le
  calcul ;
- ``restriction`` : restriction d'usage à connaître avant d'utiliser la
  méthode (``None`` si aucune au-delà de la conservation des notices) ;
- ``reference`` : référence bibliographique courte (complète dans
  ``BIBLIOGRAPHY.md``).

Voir ``README.md`` (catalogue) et ``THIRD_PARTY_LICENSES.md`` (audit des
licences) pour le détail.
"""

from collections import OrderedDict

METHODS_CONFIGS: OrderedDict = OrderedDict(
    {
        "bilateral": {
            "extra": None,
            "external": None,
            "license": "Apache-2.0 (OpenCV)",
            "restriction": None,
            "reference": "Tomasi & Manduchi (1998), ICCV",
        },
        "nl_means": {
            "extra": None,
            "external": None,
            "license": "BSD-3-Clause (scikit-image)",
            "restriction": None,
            "reference": "Buades, Coll & Morel (2005), CVPR",
        },
        "tv_chambolle": {
            "extra": None,
            "external": None,
            "license": "BSD-3-Clause (scikit-image)",
            "restriction": None,
            "reference": "Chambolle (2004), JMIV 20",
        },
        "tv_bregman": {
            "extra": None,
            "external": None,
            "license": "BSD-3-Clause (scikit-image)",
            "restriction": None,
            "reference": "Goldstein & Osher (2009), SIAM J. Imaging Sci. 2(2)",
        },
        "wavelet": {
            "extra": None,
            "external": None,
            "license": "BSD-3-Clause (scikit-image) / MIT (PyWavelets)",
            "restriction": None,
            "reference": "Chang, Yu & Vetterli (2000), IEEE TIP 9(9)",
        },
        "nl_bayes": {
            "extra": None,
            "external": None,
            "license": "MIT (réimplémentation depuis l'article IPOL)",
            "restriction": None,
            "reference": "Buades, Lebrun & Morel (2013), SIAM J. Imaging Sci. 6(3)",
        },
        "nl_ridge": {
            "extra": None,
            "external": None,
            "license": "MIT (réimplémentation depuis les articles)",
            "restriction": None,
            "reference": "Herbreteau & Kervrann (2022), ICIP",
        },
        "drunet": {
            "extra": "deep",
            "external": None,
            "license": "BSD-3-Clause (deepinv, code et poids)",
            "restriction": None,
            "reference": "Zhang et al. (2022), IEEE TPAMI 44(10)",
        },
        "swinir": {
            "extra": "deep",
            "external": None,
            "license": "Apache-2.0 (poids SwinIR) / BSD-3-Clause (deepinv)",
            "restriction": None,
            "reference": "Liang et al. (2021), ICCV Workshops",
        },
        "restormer": {
            "extra": "deep",
            "external": None,
            "license": "MIT (poids Restormer) / BSD-3-Clause (deepinv)",
            "restriction": None,
            "reference": "Zamir et al. (2022), CVPR",
        },
        "gsdrunet": {
            "extra": "deep",
            "external": None,
            "license": "BSD-3-Clause (deepinv) / MIT (GSPnP amont)",
            "restriction": "Licence des poids non déclarée",
            "reference": "Hurault, Leclaire & Papadakis (2022), ICLR",
        },
        "epll": {
            "extra": "deep",
            "external": None,
            "license": "BSD-3-Clause (deepinv)",
            "restriction": "Licence des poids non déclarée",
            "reference": "Zoran & Weiss (2011), ICCV",
        },
        "tgv": {
            "extra": "deep",
            "external": None,
            "license": "BSD-3-Clause (deepinv)",
            "restriction": None,
            "reference": "Bredies, Kunisch & Pock (2010), SIAM J. Imaging Sci. 3(3)",
        },
        "wavelet_dict": {
            "extra": "deep + ptwt",
            "external": None,
            "license": "BSD-3-Clause (deepinv) + EUPL-1.2 (ptwt)",
            "restriction": "Copyleft EUPL-1.2 (ptwt)",
            "reference": "Starck, Murtagh & Fadili (2010), Cambridge Univ. Press",
        },
        "bm3d": {
            "extra": "bm3d",
            "external": None,
            "license": "Propriétaire Tampere University (paquet bm3d)",
            "restriction": "Usage NON COMMERCIAL uniquement",
            "reference": "Dabov et al. (2007), IEEE TIP 16(8)",
        },
        "nl_ridge_inria": {
            "extra": "deep",
            "external": "clone github.com/sherbret/NL-Ridge (EOQUAL_NLRIDGE_PATH)",
            "license": "AGPL-3.0 (Inria)",
            "restriction": "Copyleft fort, clause réseau",
            "reference": "Herbreteau & Kervrann (2022), ICIP",
        },
        "nl_bayes_ipol": {
            "extra": None,
            "external": "exécutable IPOL NL_Bayes (EOQUAL_NLBAYES_EXE)",
            "license": "GPL-3.0-or-later (IPOL)",
            "restriction": "Copyleft — appel en programme externe uniquement",
            "reference": "Buades, Lebrun & Morel (2013), SIAM J. Imaging Sci. 6(3)",
        },
    }
)
