# NL-Bayes

Ces documents rassemblent tout ce qui concerne NL-Bayes dans eoqual-denoising :
le principe, les mathématiques, chaque paramètre et son rôle, la place de
σ, la couleur, les deux moteurs, les études de sensibilité et le réglage
par optimisation. Les autres documents y renvoient. Les chapitres gardent
une numérotation unique d'un document à l'autre.

| Méthode du registre | Implémentation | Licence |
|---|---|---|
| `nl_bayes` | implémentation numpy écrite depuis l'article, fournie par la bibliothèque **nlbayes-ext** (https://github.com/oam31/nlbayes-ext, `denoise_gray`, `denoise_rgb`), appelée par `backends/nl_bayes/native.py` | MIT |
| `nl_bayes_cfa` | extension aux mosaïques de Bayer, nlbayes-ext (`denoise_cfa`), appelée par `backends/nl_bayes/cfa.py` ([§13](cfa.md)) | MIT |
| `nl_bayes_ipol` | exécutable C++ de l'article IPOL, appelé en composant externe (`backends/nl_bayes/ipol_external.py`, installation : [référence technique](../reference.md) §3.10) | GPL-3.0+ |

**Références.** A. Buades, M. Lebrun, J.-M. Morel, « A Nonlocal Bayesian
Image Denoising Algorithm », *SIAM J. Imaging Sciences* 6(3), 2013 ;
M. Lebrun, A. Buades, J.-M. Morel, « Implementation of the "Non-Local
Bayes" (NL-Bayes) Image Denoising Algorithm », *Image Processing On Line*
3, 2013 (désigné ci-dessous « l'article »). L'article IPOL est la **seule
source** de `nl_bayes` (aucun code consulté, `THIRD_PARTY_LICENSES.md`
§2.1).


Notations : vecteurs et matrices en gras dans le texte (**Q**, **C**),
formules en blocs de code. σ est l'écart-type du bruit, exprimé sur
l'échelle 0-255 de l'article ([§7.2](parametres.md#72-léchelle-0-255)).

| Document | Chapitres | Contenu |
|---|---|---|
| [La méthode](methode.md) | 1-5 | Vue d'ensemble, patchs et groupes, covariance, filtre de Wiener, les deux étapes |
| [Paramètres, σ, couleur, moteurs](parametres.md) | 6-10 | Liste et dépendances des paramètres, place de σ, couleur et images démosaïquées, moteurs, jeux de données |
| [Études](etudes.md) | 11 | Moteurs, couleur, τ₀, sensibilité, σ et β₂ après stabilisation, N₂, étape 2 couleur, gain de l'étape 1 |
| [Réglage par optimisation](reglage.md) | 12 | Ce qu'il faut régler, espace par défaut, précautions, contraintes et redondances, correspondance avec nlbayes-ext |
| [NL-Bayes CFA](cfa.md) | 13 | `nl_bayes_cfa` : débruiter et démosaïquer depuis la mosaïque |
| [Limites et pistes](limites.md) | 14 | Limites constatées et pistes |

```{toctree}
:hidden:
:maxdepth: 1

methode
parametres
etudes
reglage
cfa
limites
```
