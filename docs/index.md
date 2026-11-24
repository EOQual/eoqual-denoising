# eoqual-denoising

17 méthodes de débruitage d'images (filtres, variation totale, ondelettes, non locales dont NL-Bayes en couleur et NL-Ridge réimplémentés sous MIT, réseaux pré-entraînés fournis par eoqual-neural), stabilisation de variance du bruit poisson-gaussien des images brutes (Anscombe généralisée, inverse exacte non biaisée), optimisation bayésienne des paramètres et benchmark — **en cours de développement**.

- [Documents techniques](technique/index.md) : principe, mise en œuvre,
  validation et limites des méthodes.
- [Notes de version](changelog.md) : ce qui change d'une version à l'autre.
- [Référence de l'API](api/index) : générée à partir du code source.

```{toctree}
:hidden:
:maxdepth: 2

technique/index
changelog
api/index
```
