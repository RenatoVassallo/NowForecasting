"""House style for production figures and the PDF report.

Palette and typography taken from renatovassallo.github.io: warm cream paper,
warm near-black ink, terracotta accent, gold and olive as secondary hues.
Spectral (serif) for display text and Karla (sans) for labels, falling back to
Georgia / Helvetica where the webfonts are not installed.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --- the palette --------------------------------------------------------------
INK = "#2a231d"
MUTED = "#6b5f52"
FAINT = "#8c7f6f"
BG = "#f6f0e6"            # page background
SURFACE = "#fbf7f0"       # panel background
BORDER = "#e6dcca"
BORDER_STRONG = "#d3bba9"
ACCENT = "#b0472a"        # terracotta - Peru / headline
ACCENT_TINT = "#f0e0d6"
GOLD = "#9a7b2e"          # China
OLIVE = "#5a7d43"         # commodities / terms of trade
BROWN = "#6b5f52"         # US

BLOCK = {"peru": ACCENT, "china": GOLD, "tot": OLIVE, "usa": BROWN,
         "expectations": FAINT}

SERIF = ["Spectral", "Georgia", "Times New Roman", "serif"]
SANS = ["Karla", "Helvetica Neue", "Arial", "sans-serif"]


def set_style() -> None:
    import logging
    import warnings

    # Spectral/Karla come from the website; Georgia/Helvetica render when they
    # are not installed - the fallback is intended, so silence the chatter.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="findfont")
    mpl.rcParams.update({
        "figure.facecolor": BG, "savefig.facecolor": BG,
        "axes.facecolor": SURFACE, "axes.edgecolor": BORDER_STRONG,
        "axes.labelcolor": MUTED, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": BORDER, "grid.linewidth": 0.7,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "font.family": "sans-serif", "font.sans-serif": SANS, "font.size": 9.5,
        "axes.titlesize": 11, "figure.titlesize": 14,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.solid_capstyle": "round",
    })


def title(ax, text, sub=None):
    y = 1.085 if sub else 1.03
    ax.text(0.0, y, text, transform=ax.transAxes, fontfamily=SERIF,
            fontsize=12.5, color=INK, va="bottom")
    if sub:
        ax.text(0.0, 1.025, sub, transform=ax.transAxes, fontsize=8.2,
                color=MUTED, va="bottom")
    ax.set_title("")


def fan_bands(ax, x, mode, sigma_left, sigma_right, colour, coverages=(0.90, 0.60, 0.30),
              alphas=(0.16, 0.30, 0.50), anchor=None):
    """Nested two-piece-normal bands, darkest at the centre, optional anchor point."""
    from scipy import stats

    mode = np.asarray(mode, dtype=float)
    sl, sr = np.asarray(sigma_left, dtype=float), np.asarray(sigma_right, dtype=float)
    if anchor is not None:                          # fan emanates from the last outturn
        import pandas as pd

        ax0, ay0 = anchor
        x = pd.DatetimeIndex([ax0]).append(pd.DatetimeIndex(x))
        mode = np.r_[ay0, mode]
        sl, sr = np.r_[0.0, sl], np.r_[0.0, sr]
    for cov, a in zip(coverages, alphas):
        z = stats.norm.ppf((1 + cov) / 2)
        ax.fill_between(x, mode - sl * z, mode + sr * z, color=colour, alpha=a, lw=0)
    ax.plot(x, mode, color=colour, lw=1.9, ls=(0, (5, 2)))
    return ax


def history(ax, series, colour=INK, lw=1.9):
    ax.plot(series.index, series.values, color=colour, lw=lw, solid_capstyle="round")


def band_legend(ax, colour, coverages=(0.90, 0.60, 0.30), loc="lower left"):
    import matplotlib.patches as mp

    handles = [mp.Patch(facecolor=colour, alpha=a, label=f"{int(c*100)}%")
               for c, a in zip(coverages, (0.16, 0.30, 0.50))]
    handles.append(plt.Line2D([], [], color=INK, lw=1.8, label="realised"))
    ax.legend(handles=handles, loc=loc, ncol=len(handles), fontsize=8)
