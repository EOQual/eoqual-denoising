#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures explicatives de NL_BAYES.md (principe de la méthode).

    python examples/nl_bayes_figures.py            # écrit docs/figures/nl_bayes/*.png

Image : « astronaut » (NASA, domaine public), niveaux de gris, bruit blanc
σ = 20 sur l'échelle 0-255. Le regroupement et la covariance sont recalculés
ici pour un seul patch, avec les formules de NL_BAYES.md §2 à §4, afin de les
montrer ; le débruitage complet utilise ``eoqual.denoising``.

Dépendances de l'exemple (non requises par le paquet) : matplotlib.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

import eoqual.denoising as d  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "nl_bayes"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SIGMA = 20.0
K = 5            # côté du patch
N = 60           # patchs par groupe (article, k = 5)
WIN = 35         # côté de la fenêtre de recherche (7k)
REF = (118, 150)  # coin haut-gauche du patch de référence (ligne, colonne)

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": "white"})


def _images():
    clean = d.astronaut_gray(256) * 255.0
    noisy = clean + np.random.default_rng(0).normal(0, SIGMA, clean.shape)
    basic = d.denoise(noisy, "nl_bayes", sigma=SIGMA, data_range=(0, 255), return_basic=True)
    final = d.denoise(noisy, "nl_bayes", sigma=SIGMA, data_range=(0, 255))
    return clean, noisy, basic, final


def _group(img):
    """N patchs les plus proches du patch de référence dans la fenêtre (distance moyenne par pixel)."""
    i0, j0 = REF
    ref = img[i0:i0 + K, j0:j0 + K]
    h = WIN // 2
    cands = []
    for i in range(i0 - h, i0 + h + 1):
        for j in range(j0 - h, j0 + h + 1):
            p = img[i:i + K, j:j + K]
            cands.append((np.mean((p - ref) ** 2), i, j))
    cands.sort()
    return cands[:N]


def fig_pipeline(noisy, basic, final):
    fig = plt.figure(figsize=(10, 3.9))
    boxes = [(0.03, "Image bruitée", noisy), (0.38, "Estimée de l'étape 1", basic), (0.73, "Image finale", final)]
    for x, title, im in boxes:
        ax = fig.add_axes([x, 0.33, 0.24, 0.58])
        ax.imshow(im[60:190, 90:220], cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    texts = [(0.325, "Étape 1\ngroupes formés sur l'image bruitée\nfiltre (C̃ − β₁σ²I)·C̃⁻¹\nagrégation"),
             (0.675, "Étape 2\ngroupes formés sur l'étape 1\nfiltre de Wiener C_b·(C_b + β₂σ²I)⁻¹\nappliqué aux patchs bruités")]
    for x, t in texts:
        fig.patches.append(FancyArrowPatch((x - 0.035, 0.62), (x + 0.035, 0.62), transform=fig.transFigure,
                                           arrowstyle="-|>", mutation_scale=18, color=INK2, lw=1.5))
        fig.text(x, 0.03, t, ha="center", va="bottom", fontsize=8, color=INK2)
    fig.savefig(OUT / "pipeline.png", dpi=130)
    plt.close(fig)


def fig_grouping(noisy, group):
    i0, j0 = REF
    h = WIN // 2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    a1.imshow(noisy, cmap="gray", vmin=0, vmax=255)
    a1.set_xlim(j0 - h - 25, j0 + h + 30); a1.set_ylim(i0 + h + 30, i0 - h - 25)
    a1.add_patch(Rectangle((j0 - h - 0.5, i0 - h - 0.5), WIN + K - 1, WIN + K - 1, fill=False, ec=BLUE, lw=1.6, ls="--"))
    for _, i, j in group[1:]:
        a1.add_patch(Rectangle((j - 0.5, i - 0.5), K, K, fill=False, ec=AQUA, lw=0.8))
    a1.add_patch(Rectangle((j0 - 0.5, i0 - 0.5), K, K, fill=False, ec=ORANGE, lw=2.2))
    a1.set_title("Patch de référence (orange), fenêtre de recherche (tirets),\n"
                 f"{N - 1} patchs les plus proches (vert)", fontsize=9, color=INK)
    a1.set_xticks([]); a1.set_yticks([])
    tiles = [noisy[i:i + K, j:j + K] for _, i, j in group]
    cols = 10
    rows = int(np.ceil(len(tiles) / cols))
    mos = np.full((rows * (K + 1) - 1, cols * (K + 1) - 1), np.nan)
    for n, t in enumerate(tiles):
        r, c = divmod(n, cols)
        mos[r * (K + 1):r * (K + 1) + K, c * (K + 1):c * (K + 1) + K] = t
    cmap = plt.get_cmap("gray").copy(); cmap.set_bad("white")
    a2.imshow(mos, cmap=cmap, vmin=0, vmax=255)
    a2.set_title(f"Le groupe : {N} patchs {K}×{K}, vus comme {N} vecteurs de {K*K} valeurs", fontsize=9, color=INK)
    a2.set_xticks([]); a2.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "grouping.png", dpi=130)
    plt.close(fig)


def fig_covariance(noisy, clean, group):
    X = np.array([noisy[i:i + K, j:j + K].ravel() for _, i, j in group])          # N × k²
    Xc = np.array([clean[i:i + K, j:j + K].ravel() for _, i, j in group])
    mu = X.mean(0)
    C = (X - mu).T @ (X - mu) / (len(X) - 1)
    lam, vec = np.linalg.eigh(C)
    lam, vec = lam[::-1], vec[:, ::-1]
    lam_clean = np.sort(np.linalg.eigvalsh(np.cov(Xc.T)))[::-1]
    fig = plt.figure(figsize=(10, 6.3))
    ax = fig.add_axes([0.05, 0.42, 0.28, 0.5])
    lim = np.abs(C).max()
    ax.imshow(C, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_title(f"Covariance C̃ du groupe ({K*K}×{K*K})\ndiagonale ≈ variance d'un pixel", fontsize=9, color=INK)
    ax.set_xlabel("pixel du patch"); ax.set_ylabel("pixel du patch")
    ax2 = fig.add_axes([0.42, 0.45, 0.55, 0.45])
    x = np.arange(1, K * K + 1)
    ax2.semilogy(x, lam, "o-", color=BLUE, ms=4, lw=1.5, label="λ̃ : valeurs propres de C̃ (patchs bruités)")
    ax2.semilogy(x, np.maximum(lam_clean, 1e-1), "s-", color=AQUA, ms=3.5, lw=1.2, label="valeurs propres sans bruit (référence)")
    ax2.axhline(SIGMA ** 2, color=ORANGE, lw=1.5, ls="--")
    ax2.text(K * K, SIGMA ** 2 * 1.15, "σ² (bruit)", color=INK2, ha="right", va="bottom")
    ax2.set_xlabel("direction (motif propre), de la plus forte à la plus faible")
    ax2.set_ylabel("variance (niveaux², échelle log)")
    ax2.legend(frameon=False, loc="upper right")
    ax2.grid(axis="y", color=GRID, lw=0.6)
    ax2.set_title("Au-dessus de σ² : structure ; au niveau de σ² : bruit seul", fontsize=9, color=INK)
    gain1 = 1 - SIGMA ** 2 / lam                    # formule de l'article, non bornée
    shown = list(range(5)) + list(range(K * K - 3, K * K))
    for n, e in enumerate(shown):
        a = fig.add_axes([0.05 + n * 0.115, 0.03, 0.095, 0.19])
        v = vec[:, e].reshape(K, K)
        m = np.abs(v).max()
        a.imshow(v, cmap="RdBu_r", vmin=-m, vmax=m)
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"motif {e+1}\nλ̃/σ² = {lam[e] / SIGMA**2:.2f}\ngain {gain1[e]:+.2f}", fontsize=7.5, color=INK2)
    fig.text(0.05, 0.335, "Motifs propres (vecteurs propres de C̃) et gain de l'étape 1 dans chaque direction : "
             "1 − σ²/λ̃ (β₁ = 1).\nLes 5 plus forts sont gardés ; les 3 plus faibles ont λ̃ < σ² (bruit sous-estimé par le "
             "tirage) : gain négatif, la composante est inversée et parfois amplifiée.", fontsize=8.5, color=INK)
    fig.savefig(OUT / "covariance.png", dpi=130)
    plt.close(fig)


def fig_gains():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    lt = np.logspace(np.log10(0.25), 2, 400)       # λ̃ / σ² : variance mesurée sur les patchs bruités
    r = np.logspace(-2, 2, 400)                     # λ_b / σ² : variance du signal (étape 2)
    for beta, col in ((0.6, AQUA), (1.0, BLUE), (1.5, ORANGE)):
        a1.semilogx(lt, 1 - beta / lt, color=col, lw=2, label=f"β₁ = {beta:g}")
        a2.semilogx(r, r / (r + beta), color=col, lw=2, label=f"β₂ = {beta:g}")
    a1.axhline(0, color=INK2, lw=0.8)
    a1.set_ylim(-2.2, 1.05)
    a1.axvspan(0.25, 1.0, color=GRID, alpha=0.5, lw=0)
    a1.text(0.5, -1.9, "bruit seul :\nλ̃ se disperse\nautour de σ²", ha="center", fontsize=7.5, color=INK2)
    a1.set_title("Étape 1 : gain 1 − β₁σ²/λ̃ (non borné)", fontsize=9, color=INK)
    a1.set_xlabel("variance mesurée dans la direction  λ̃ / σ²")
    a2.set_title("Étape 2 : gain λ_b / (λ_b + β₂σ²), entre 0 et 1", fontsize=9, color=INK)
    a2.set_xlabel("variance du signal dans la direction  λ_b / σ²")
    a2.axvline(1, color=INK2, lw=0.8, ls=":")
    for a in (a1, a2):
        a.grid(color=GRID, lw=0.6)
        a.legend(frameon=False, loc="lower right" if a is a2 else "upper left")
    a1.set_ylabel("gain appliqué à la composante")
    fig.tight_layout()
    fig.savefig(OUT / "gains.png", dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clean, noisy, basic, final = _images()
    group = _group(noisy)
    fig_pipeline(noisy, basic, final)
    fig_grouping(noisy, group)
    fig_covariance(noisy, clean, group)
    fig_gains()
    print(f"figures écrites dans {OUT}")


if __name__ == "__main__":
    main()
