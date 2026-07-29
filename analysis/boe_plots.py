"""Bank of England style fan chart.

Deliberately close to the Inflation/Monetary Policy Report house style: no axis
box, a single hue deepening toward the centre, bands of equal probability, the
forecast starting from the last outturn, and a vertical marker at the forecast
origin. The BoE draws no central line - the eye should read a distribution, not
a point - so the mode is shown only faintly, and the darkest band carries the
central 30%.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ["plot_boe_fan"]

BOE_RED = "#B4325A"


def plot_boe_fan(history: pd.Series, periods, centre, bands, *, title="",
                 ylabel="% YoY", colour=BOE_RED, ax=None, show_mode=True,
                 zero_line=True, mode_shift=None):
    """Draw a BoE-style fan.

    history : realised series indexed by quarterly timestamps
    periods : PeriodIndex of the forecast quarters
    centre  : the mode per forecast quarter (already shifted if desired)
    bands   : {level: (lo, hi)} as returned by ``forecast.boe_fan.tpn_bands``
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(9.6, 4.8))
    hx = pd.DatetimeIndex(history.index)
    fx = pd.PeriodIndex(periods, freq="Q").to_timestamp(how="end")
    # the fan emanates from the last outturn, with zero width there
    x = np.r_[hx[-1:], fx]
    last = float(history.iloc[-1])

    for lv in sorted(bands, reverse=True):          # widest first
        lo, hi = bands[lv]
        alpha = 0.13 + 0.30 * (1 - lv)              # deeper toward the centre
        ax.fill_between(x, np.r_[last, lo], np.r_[last, hi], color=colour,
                        alpha=alpha, lw=0, zorder=2)
    if show_mode:
        ax.plot(x, np.r_[last, centre], color=colour, lw=1.1, alpha=0.75, ls=(0, (4, 2)),
                zorder=3)
    ax.plot(hx, history.values, color="#22242A", lw=1.9, zorder=4)
    if zero_line:
        ax.axhline(0, color="#22242A", lw=0.8, alpha=0.55, zorder=1)
    ax.axvline(hx[-1], color="#8A929E", lw=0.8, ls=":", zorder=1)

    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#22242A")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.grid(axis="y", color="#D8DBE0", lw=0.7)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, rotation=0, ha="left", va="bottom", labelpad=0)
    ax.yaxis.set_label_coords(1.0, 1.02)
    if title:
        ax.set_title(title, loc="left", fontsize=11.5)
    return ax


def plot_decile_fan(ax, history, periods, mode, sigma_left, sigma_right, *,
                    colour="#2C3E7B", title="", ylabel="% YoY", n_bands=9,
                    zero_line=True):
    """BCRP / Inflation-Report style fan: nested bands each carrying 10 percent.

    Unlike the three-band version, every shade adds ten percentage points of
    probability, so the eye reads the density rather than three thresholds. The
    central line is the MODE - which is what a two-piece normal's central
    projection is - and the shading is drawn from the widest band inward.
    """

    import numpy as np
    import pandas as pd
    from scipy import stats

    hx = pd.DatetimeIndex(history.index)
    fx = pd.PeriodIndex(periods, freq="Q").to_timestamp(how="end")
    x = np.r_[hx[-1:], fx]
    last = float(history.iloc[-1])
    mode = np.asarray(mode, dtype=float)
    sl = np.asarray(sigma_left, dtype=float)
    sr = np.asarray(sigma_right, dtype=float)

    covs = np.arange(n_bands, 0, -1) / (n_bands + 1)      # 0.9, 0.8, ... 0.1
    # BCRP-style ramp: the central band is clearly the darkest and the outer ones
    # fade quickly, so the eye reads where the mass is instead of a uniform wash.
    for i, cov in enumerate(covs):
        z = stats.norm.ppf((1.0 + cov) / 2.0)
        lo = np.r_[last, mode - sl * z]
        hi = np.r_[last, mode + sr * z]
        frac = i / max(n_bands - 1, 1)
        ax.fill_between(x, lo, hi, color=colour, alpha=0.05 + 0.62 * frac ** 2.3,
                        lw=0, zorder=2)
    ax.plot(x, np.r_[last, mode], color="#FFFFFF", lw=2.2, zorder=4, alpha=0.85)
    ax.plot(x, np.r_[last, mode], color=colour, lw=1.2, zorder=5)
    ax.plot(hx, history.values, color="#22242A", lw=1.8, zorder=5)
    if zero_line:
        ax.axhline(0, color="#22242A", lw=0.8, alpha=0.5, zorder=1)
    ax.axvline(hx[-1], color="#8A929E", lw=0.8, ls=":", zorder=1)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#22242A")
    ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
    ax.grid(axis="y", color="#D8DBE0", lw=0.7)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, rotation=0, ha="left", va="bottom", labelpad=0)
    ax.yaxis.set_label_coords(1.0, 1.02)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5)
    return ax
