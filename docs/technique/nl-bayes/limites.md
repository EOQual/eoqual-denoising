# NL-Bayes : limites et pistes

## 14. Limites et pistes

- **`nl_bayes_cfa`** ([§13](cfa.md#13-nl_bayes_cfa--débruiter-et-démosaïquer-depuis-la-mosaïque)) : branchement sur `optimize_parameters`
  (sortie RGB d'une entrée mosaïque) ; modèle de bruit par canal (a, b) si
  les gains diffèrent entre R, G et B ; moyenne des groupes plats sur un
  support plus large que le groupe, pour activer le critère des zones
  homogènes par défaut ([§13.5](cfa.md#135-critère-des-zones-homogènes)) ; nombre de passes adapté au flux et au
  contenu ([§13.4](cfa.md#134-comparaison-avec-démosaïçage-puis-nl-bayes)) ; vitesse (8 passes : 6 à 10 fois le coût de
  Hamilton-Adams + NL-Bayes).

- **Bruit corrélé** (image démosaïquée) : remplacer σ²·I par la
  covariance du bruit Σ_n dans les deux filtres (C̃ − β₁Σ_n, puis
  C_b + β₂Σ_n), Σ_n mesurée une fois par démosaïqueur (bruit pur, comme
  κ_Y). Elle garde la chaîne actuelle (démosaïçage puis débruitage) ;
  `nl_bayes_cfa` ([§13](cfa.md#13-nl_bayes_cfa--débruiter-et-démosaïquer-depuis-la-mosaïque)) traite le même défaut à la source ([§8.2](parametres.md#82-image-démosaïquée), [§11.5](etudes.md#115-sensibilité-rgb-démosaïqué-zones-sombres)) ; voir aussi [référence technique](../reference.md)
  [§10.1](parametres.md).
- **Bruit dépendant du signal sans stabilisation** : un σ par groupe,
  calculé à partir du modèle du capteur et de la moyenne du groupe, au
  lieu d'un σ global ; l'étude du [§11.5](etudes.md#115-sensibilité-rgb-démosaïqué-zones-sombres) montre que le σ global est la
  première cause d'erreur dans les zones sombres.
- **Vitesse** : numpy sur CPU ; un portage PyTorch (GPU) serait le
  prochain gain.
