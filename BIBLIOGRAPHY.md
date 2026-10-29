# Bibliographie — eoqual-denoising

Référence bibliographique de chaque méthode du catalogue, en un seul
endroit. `REFERENCE_TECHNIQUE.md`, `config.py` et les docstrings n'en
gardent qu'un rappel court (auteur, année) ; ce document est la source de
vérité pour les citations complètes.

Pour chaque méthode : la **publication d'origine** de la méthode, puis,
le cas échéant, la publication décrivant l'**implémentation** utilisée.
Les références ont été vérifiées sur la page de l'éditeur, la page IPOL,
la fiche PyPI ou le `README` du dépôt des auteurs (2026-10-28).

> Une entrée marquée **réf. incomplète** n'a pas pu être établie avec
> certitude ; seuls les éléments sûrs sont donnés plutôt que d'inventer
> des détails. Signaler via `CONTRIBUTING.md` si vous avez la référence
> exacte.

## Méthodes du socle

| Méthode | Référence |
|---|---|
| `bilateral` | Tomasi, C., & Manduchi, R. (1998). *Bilateral filtering for gray and color images*. Proc. Sixth International Conference on Computer Vision (ICCV), 839-846. |
| `nl_means` | Buades, A., Coll, B., & Morel, J.-M. (2005). *A non-local algorithm for image denoising*. Proc. IEEE CVPR, vol. 2, 60-65. |
| `nl_means` (`fast_mode=True`) | Darbon, J., Cunha, A., Chan, T. F., Osher, S., & Jensen, G. J. (2008). *Fast nonlocal filtering applied to electron cryomicroscopy*. Proc. IEEE ISBI, 1331-1334. |
| `tv_chambolle`, `tv_bregman` (modèle) | Rudin, L. I., Osher, S., & Fatemi, E. (1992). *Nonlinear total variation based noise removal algorithms*. Physica D, 60(1-4), 259-268. |
| `tv_chambolle` (solveur) | Chambolle, A. (2004). *An algorithm for total variation minimization and applications*. Journal of Mathematical Imaging and Vision, 20, 89-97. |
| `tv_bregman` (solveur) | Goldstein, T., & Osher, S. (2009). *The split Bregman method for L1-regularized problems*. SIAM Journal on Imaging Sciences, 2(2), 323-343. |
| `wavelet` (`BayesShrink`) | Chang, S. G., Yu, B., & Vetterli, M. (2000). *Adaptive wavelet thresholding for image denoising and compression*. IEEE Transactions on Image Processing, 9(9), 1532-1546. |
| `wavelet` (`VisuShrink`) ; `estimate_noise_sigma` | Donoho, D. L., & Johnstone, I. M. (1994). *Ideal spatial adaptation by wavelet shrinkage*. Biometrika, 81(3), 425-455. |

## Méthodes en extra ou composant externe

| Méthode | Référence |
|---|---|
| `drunet` | Zhang, K., Li, Y., Zuo, W., Zhang, L., Van Gool, L., & Timofte, R. (2022). *Plug-and-Play Image Restoration with Deep Denoiser Prior*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10), 6360-6376. |
| Méthodes deepinv (`drunet`, `swinir`, `restormer`, `gsdrunet`, `epll`, `tgv`, `wavelet_dict`) — bibliothèque | Tachella, J., et al. (2025). *DeepInverse: A Python package for solving imaging inverse problems with deep learning*. Journal of Open Source Software, 10(115), 8923. https://doi.org/10.21105/joss.08923 (préprint arXiv:2505.20160). |
| `swinir` | Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., & Timofte, R. (2021). *SwinIR: Image Restoration Using Swin Transformer*. Proc. IEEE/CVF ICCV Workshops, 1833-1844. |
| `restormer` | Zamir, S. W., Arora, A., Khan, S., Hayat, M., Khan, F. S., & Yang, M.-H. (2022). *Restormer: Efficient Transformer for High-Resolution Image Restoration*. Proc. IEEE/CVF CVPR, 5728-5739. |
| `gsdrunet` | Hurault, S., Leclaire, A., & Papadakis, N. (2022). *Gradient Step Denoiser for convergent Plug-and-Play*. International Conference on Learning Representations (ICLR). |
| `epll` | Zoran, D., & Weiss, Y. (2011). *From learning models of natural image patches to whole image restoration*. Proc. IEEE ICCV, 479-486. |
| `tgv` | Bredies, K., Kunisch, K., & Pock, T. (2010). *Total Generalized Variation*. SIAM Journal on Imaging Sciences, 3(3), 492-526. |
| `tgv` (solveur) | Condat, L. (2013). *A primal-dual splitting method for convex optimization involving Lipschitzian, proximable and linear composite terms*. Journal of Optimization Theory and Applications, 158(2), 460-479. |
| `wavelet_dict` | Starck, J.-L., Murtagh, F., & Fadili, J. M. (2010). *Sparse Image and Signal Processing: Wavelets, Curvelets, Morphological Diversity*. Cambridge University Press. |
| `wavelet_dict` (schéma) | Combettes, P. L., & Pesquet, J.-C. (2008). *A proximal decomposition method for solving convex variational inverse problems*. Inverse Problems, 24(6), 065014. |
| `wavelet_dict` (`ptwt`) | Wolter, M., Blanke, F., Garcke, J., & Hoyt, C. T. (2024). *ptwt — The PyTorch Wavelet Toolbox*. Journal of Machine Learning Research, 25(80), 1-7. |
| `bm3d` | Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K. (2007). *Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering*. IEEE Transactions on Image Processing, 16(8), 2080-2095. |
| `bm3d` (implémentation `bm3d` 4.x) | Mäkinen, Y., Azzari, L., & Foi, A. (2020). *Collaborative Filtering of Correlated Noise: Exact Transform-Domain Variance for Improved Shrinkage and Patch Matching*. IEEE Transactions on Image Processing, 29, 8339-8354. **Citation exigée par la licence** dès publication de résultats. |
| `bm3d` (analyse détaillée) | Lebrun, M. (2012). *An Analysis and Implementation of the BM3D Image Denoising Method*. Image Processing On Line, 2, 175-213. https://doi.org/10.5201/ipol.2012.l-bm3d |
| `nl_ridge`, `nl_ridge_inria` | Herbreteau, S., & Kervrann, C. (2022). *Towards a Unified View of Unsupervised Non-Local Methods for Image Denoising: The NL-Ridge Approach*. Proc. IEEE ICIP, 3376-3380. https://doi.org/10.1109/ICIP46576.2022.9897992 |
| `nl_ridge` (cadre étendu, contrainte affine) | Herbreteau, S., & Kervrann, C. (2025). *A Unified Framework of Nonlocal Parametric Methods for Image Denoising*. SIAM Journal on Imaging Sciences, 18(1), 89-119. https://doi.org/10.1137/24M1630967 |
| `nl_bayes` | Buades, A., Lebrun, M., & Morel, J.-M. (2013). *A Nonlocal Bayesian Image Denoising Algorithm*. SIAM Journal on Imaging Sciences, 6(3), 1665-1688. |
| `nl_bayes` (source de la réimplémentation), `nl_bayes_ipol` (exécutable) | Lebrun, M., Buades, A., & Morel, J.-M. (2013). *Implementation of the "Non-Local Bayes" (NL-Bayes) Image Denoising Algorithm*. Image Processing On Line, 3, 1-42. https://doi.org/10.5201/ipol.2013.16 |

## Stabilisation de variance (`vst.py`)

| Élément | Référence |
|---|---|
| Transformée d'Anscombe | Anscombe, F. J. (1948). *The transformation of Poisson, binomial and negative-binomial data*. Biometrika, 35(3/4), 246-254. |
| Transformée généralisée (GAT) | Murtagh, F., Starck, J.-L., & Bijaoui, A. (1995). *Image restoration with noise suppression using a multiresolution support*. Astronomy and Astrophysics Supplement Series, 112, 179-189. |
| Inverse exacte non biaisée (Poisson) | Mäkitalo, M., & Foi, A. (2011). *Optimal inversion of the Anscombe transformation in low-count Poisson image denoising*. IEEE Transactions on Image Processing, 20(1), 99-109. |
| Inverse exacte non biaisée (poisson-gaussien) | Mäkitalo, M., & Foi, A. (2013). *Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise*. IEEE Transactions on Image Processing, 22(1), 91-103. |

## Optimisation des paramètres (`optimization/`)

| Élément | Référence |
|---|---|
| Optimisation bayésienne | Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016). *Taking the Human Out of the Loop: A Review of Bayesian Optimization*. Proceedings of the IEEE, 104(1), 148-175. |
| Acquisition `gp_hedge` | Hoffman, M., Brochu, E., & de Freitas, N. (2011). *Portfolio Allocation for Bayesian Optimization*. Proc. Conference on Uncertainty in Artificial Intelligence (UAI), 327-336. |

## Méthode non incluse

| Méthode | Référence |
|---|---|
| WNLB (Wavelet Non-Local Bayes) | Aucune publication identifiée (**réf. incomplète**). Seule trace : dépôt https://github.com/mago876/wnlb (M. González Olmedo, 2016), sans licence. Voir `THIRD_PARTY_LICENSES.md` §2.4. |

## Bibliothèques utilisées

Références à citer pour les bibliothèques qui font le calcul des
méthodes du socle et des extras (leurs licences sont dans
`THIRD_PARTY_LICENSES.md` §2.5).

| Bibliothèque | Référence |
|---|---|
| NumPy | Harris, C. R., et al. (2020). *Array programming with NumPy*. Nature, 585, 357-362. https://doi.org/10.1038/s41586-020-2649-2 |
| SciPy | Virtanen, P., et al. (2020). *SciPy 1.0: fundamental algorithms for scientific computing in Python*. Nature Methods, 17, 261-272. https://doi.org/10.1038/s41592-019-0686-2 |
| scikit-image | van der Walt, S., et al. (2014). *scikit-image: image processing in Python*. PeerJ, 2, e453. https://doi.org/10.7717/peerj.453 |
| OpenCV | Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools. |
| PyWavelets | Lee, G. R., Gommers, R., Waselewski, F., Wohlfahrt, K., & O'Leary, A. (2019). *PyWavelets: A Python package for wavelet analysis*. Journal of Open Source Software, 4(36), 1237. https://doi.org/10.21105/joss.01237 |
| PyTorch | Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. Advances in Neural Information Processing Systems 32 (NeurIPS). |
| DeepInverse | voir la ligne « Méthodes deepinv » ci-dessus. |
| scikit-optimize | Projet scikit-optimize, https://scikit-optimize.github.io (**réf. incomplète** — pas d'article de référence ; citer la version utilisée). |

## Données de test

| Donnée | Source |
|---|---|
| `astronaut` | Portrait de l'astronaute Eileen Collins, NASA (domaine public), embarqué dans scikit-image (`skimage.data.astronaut`). |
