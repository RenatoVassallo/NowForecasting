"""Shared visualization: one soft, professional palette and house style.

Every app and notebook imports its plots from here so the whole project reads as
one visual system. The palette is deliberately low-saturation (easy on the eye,
prints well, colour-blind-considerate); accents are muted, spines are light, and
titles sit left. Call :func:`set_style` once at the top of a notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
# Soft, harmonious, low-saturation categorical palette.
PALETTE = [
    "#4C72A8",  # steel blue
    "#5FA69A",  # teal
    "#C99B6E",  # sand / ochre
    "#8E7CB0",  # dusty violet
    "#8A929E",  # slate grey
    "#B98A80",  # terracotta
    "#7FA97A",  # sage
    "#C7A2C0",  # mauve
    "#6E97B8",  # dusty sky
    "#A79483",  # taupe
]
INK = "#33373D"      # text / strong lines
MUTED = "#9AA0A8"    # secondary / de-emphasised
SPINE = "#C9CCD1"
GRID = "#EAEBEE"

# Semantic colours for economic groups and data providers (fall back to PALETTE).
GROUP_PALETTE = {
    "survey": PALETTE[0], "soft": PALETTE[0],
    "activity": PALETTE[1], "real": PALETTE[1], "sectoral activity": PALETTE[1],
    "prices": PALETTE[2], "international prices": PALETTE[2],
    "prices and financial conditions": PALETTE[2],
    "money": PALETTE[3], "credit": PALETTE[3], "credit and liquidity": PALETTE[3],
    "foreign": PALETTE[4], "foreign sector": PALETTE[4], "other": PALETTE[4],
    "fiscal": PALETTE[5], "fiscal sector": PALETTE[5],
    "property": PALETTE[6], "high-frequency activity": PALETTE[6],
    "external": PALETTE[8], "external trade": PALETTE[8],
    "labor": PALETTE[9], "labor market": PALETTE[9],
    # providers
    "bcrp": PALETTE[0], "fred": PALETTE[5], "inei": PALETTE[1], "nbs": PALETTE[2],
}


def group_color(name: str) -> str:
    return GROUP_PALETTE.get(str(name).strip().lower(), MUTED)


def color_cycle(n: int) -> list[str]:
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


COMBO_MODELS = ("Adaptive-IC",)   # combination models drawn near-black (the product)
COMBO_INK = "#1B1E23"


def model_color_map(models, *, black=COMBO_MODELS) -> dict:
    """Stable ``model -> colour`` so a model keeps ONE colour across every plot
    and subperiod. Colours are assigned to the non-``black`` models in their given
    order, and the combination models in ``black`` are drawn near-black so they
    read as the product rather than another candidate. Because the black models
    are skipped when indexing the palette, adding the Adaptive-IC never shifts a
    member's colour. Pass the SAME model list everywhere (or reuse the returned
    map) to keep colours identical across the horse race and the weight plot."""

    black = set(black)
    palette_models = [m for m in dict.fromkeys(models) if m not in black]
    cmap = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(palette_models)}
    for m in models:
        if m in black:
            cmap[m] = COMBO_INK
    return cmap


def fmt_period(period) -> str:
    """Format an evaluation window as 'YYYY-YYYY' for titles.

    Accepts a (start, end) pair of date-likes, a ready string, or None.
    """
    if period is None:
        return ""
    if isinstance(period, (tuple, list)) and len(period) == 2:
        return f"{pd.Timestamp(period[0]).year}-{pd.Timestamp(period[1]).year}"
    return str(period)


def _title(base: str | None, period=None) -> str:
    p = fmt_period(period)
    if base and p:
        return f"{base} ({p})"
    return base or (f"({p})" if p else "")


def set_style() -> None:
    """Apply the house style to matplotlib (idempotent)."""

    import logging

    import matplotlib as mpl

    # On systems without the preferred fonts, matplotlib logs a "findfont" line
    # per figure; the DejaVu fallback is fine, so keep the log quiet.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.labelsize": 9.5, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.edgecolor": SPINE, "axes.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "legend.frameon": False, "legend.fontsize": 8.5,
        "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
        "lines.linewidth": 1.8, "lines.markersize": 4,
    })


# --------------------------------------------------------------------------- #
# Release-cycle horse race
# --------------------------------------------------------------------------- #
DEFAULT_LEADS = (-120, -105, -90, -75, -60, -45, -30, -15, 0)


def plot_horse_race(
    by_lead: pd.DataFrame,
    summary: pd.DataFrame | None = None,
    *,
    highlight: int = 5,
    order: list[str] | None = None,
    colors: dict | None = None,
    xlim: tuple[int, int] = (-120, 0),
    xticks=DEFAULT_LEADS,
    title: str = "Release-cycle horse race",
    period=None,
    ax=None,
):
    """RMSE across the release cycle. X axis runs -120 -> 0 (approach to release).

    ``by_lead`` is model x lead (see ``core.scoring.rmse_by_lead``). Only models
    that fit are drawn when ``summary`` carries a ``fits`` column; the best
    ``highlight`` are drawn bold, the rest muted. Each model keeps its own stable
    colour (``colors`` overrides); the Adaptive-IC is drawn near-black.
    """

    import matplotlib.pyplot as plt

    df = by_lead.copy()
    if summary is not None and "fits" in summary.columns:
        keep = summary.index[summary["fits"]]
        df = df.loc[[m for m in df.index if m in set(keep)]]
    if order is None:
        if summary is not None:
            key = "rmse_common" if "rmse_common" in summary.columns else "rmse"
            order = [m for m in summary.sort_values(key).index if m in df.index]
        else:
            order = df.mean(axis=1).sort_values().index.tolist()
    cmap = colors or model_color_map(df.index.tolist())

    if ax is None:
        _, ax = plt.subplots(figsize=(8.8, 5.2))
    for i, m in enumerate(order):
        s = df.loc[m].dropna()
        strong = i < highlight or m in COMBO_MODELS
        ax.plot(s.index, s.values, marker="o", color=cmap.get(m, PALETTE[i % len(PALETTE)]),
                lw=2.6 if m in COMBO_MODELS else (2.3 if strong else 1.1),
                alpha=1.0 if strong else 0.5,
                zorder=4 if m in COMBO_MODELS else (3 if strong else 2), label=m)
    ax.set_xlim(*xlim)
    ax.set_xticks(list(xticks))
    ax.set_xlabel("days to publication")
    ax.set_ylabel("RMSE (target units)")
    ax.set_title(_title(title, period))
    ax.legend(ncol=2, loc="upper right")
    return ax


def plot_horse_race_subperiods(
    curves: dict,
    summaries: dict | None = None,
    *,
    highlight: int = 5,
    colors: dict | None = None,
    xlim: tuple[int, int] = (-120, 0),
    xticks=DEFAULT_LEADS,
    suptitle: str = "Release-cycle horse race by subperiod",
    ylabel: str = "RMSE (target units)",
    sharey: bool = False,
):
    """One panel per subperiod. ``curves`` = {label: rmse_by_lead} (see
    ``core.scoring.subperiod_curves``); ``summaries`` = {label: summary} sets
    the model order and drops non-fitting models. Each model keeps its own colour
    across every panel (and matches the other plots when ``colors`` is shared);
    the Adaptive-IC is drawn near-black and bold to stand out as the product."""

    import matplotlib.pyplot as plt

    labels = list(curves)
    fig, axes = plt.subplots(1, len(labels), figsize=(4.9 * len(labels), 4.8), sharey=sharey)
    axes = np.atleast_1d(axes)
    all_models = list(dict.fromkeys(m for c in curves.values() for m in c.index))
    cmap = colors or model_color_map(all_models)

    for ax, label in zip(axes, labels):
        by = curves[label]
        summ = summaries.get(label) if summaries else None
        df = by
        if summ is not None and "fits" in summ.columns:
            df = by.loc[[m for m in by.index if m in set(summ.index[summ["fits"]])]]
        if summ is not None:
            key = "rmse_common" if "rmse_common" in summ.columns else "rmse"
            order = [m for m in summ.sort_values(key).index if m in df.index]
        else:
            order = df.mean(axis=1).sort_values().index.tolist()
        for i, m in enumerate(order):
            s = df.loc[m].dropna()
            combo = m in COMBO_MODELS
            strong = i < highlight or combo
            ax.plot(s.index, s.values, marker="o", color=cmap.get(m, MUTED),
                    lw=2.6 if combo else (2.3 if strong else 1.0),
                    alpha=1.0 if strong else 0.45,
                    zorder=4 if combo else 2, label=m)
        ax.set_xlim(*xlim); ax.set_xticks(list(xticks)); ax.set_xlabel("days to publication")
        ax.set_title(label)
        ax.legend(ncol=1, fontsize=7, loc="upper right")
    axes[0].set_ylabel(ylabel)
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    return fig


def plot_weight_evolution(rc, weights, members, *, colors: dict | None = None,
                          index_col: str = "info_index", n_bins: int = 4,
                          min_quarters: int = 5,
                          xlim: tuple[int, int] = (-120, 0), xticks=DEFAULT_LEADS, ax=None,
                          title: str = "Adaptive-IC member weights across the release cycle"):
    """Stacked-area view of the combination's applied weights over the release
    cycle, in the SAME model colours as the horse race (pass the shared ``colors``
    map) and on the SAME x-axis (``xlim`` / ``xticks``).

    Each origin is mapped to its Information-Index bin, given that quarter's
    learned ``weights`` (columns ``ref_quarter``, ``bin`` and one per member), and
    the weights are averaged across quarters at each ``days_to_publication``.
    Leads where fewer than ``min_quarters`` quarters contribute are dropped: the
    live quarter's weekly grid adds single-quarter leads whose unaveraged weights
    would otherwise draw a saw-tooth on top of the smooth cross-quarter averages.
    """

    import matplotlib.pyplot as plt

    members = list(members)
    cmap = colors or model_color_map(members)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    d = rc.drop_duplicates(["ref_quarter", "days_to_publication"]).copy()
    d["bin"] = np.clip(np.digitize(d[index_col].to_numpy(), edges[1:-1]), 0, n_bins - 1)
    w = d.merge(weights[["ref_quarter", "bin"] + members], on=["ref_quarter", "bin"], how="left")
    counts = w.groupby("days_to_publication")["ref_quarter"].nunique()
    wl = w.groupby("days_to_publication")[members].mean().sort_index()
    wl = wl.loc[counts[counts >= min_quarters].index.intersection(wl.index).sort_values()]

    if ax is None:
        _, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.stackplot(wl.index, *[wl[m].to_numpy() for m in members], labels=members,
                 colors=[cmap.get(m, MUTED) for m in members], alpha=0.9)
    ax.set_xlim(*xlim); ax.set_xticks(list(xticks)); ax.set_ylim(0, 1)
    ax.set_xlabel("days to publication"); ax.set_ylabel("combination weight")
    ax.set_title(title, loc="left")
    ax.legend(ncol=len(members), fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), frameon=False)
    return ax


# --------------------------------------------------------------------------- #
# Dynamic lead-lag correlations
# --------------------------------------------------------------------------- #
def plot_cross_correlation(
    cc_wide: pd.DataFrame,
    *,
    top: int = 10,
    columns=None,
    labels=None,
    ylim: tuple[float, float] | None = None,
    ax=None,
    title: str | None = None,
    period=None,
):
    """Lead-lag correlations, centred at 0. Draws the ``top`` strongest by default.

    ``top`` is customizable; pass explicit ``columns`` to override. The y range is
    fitted to the plotted curves (padded) unless ``ylim`` is given.
    """

    import matplotlib.pyplot as plt

    lags = [int(c.split("_")[1]) for c in cc_wide.columns]
    if columns is None:
        strength = cc_wide.abs().max(axis=1).sort_values(ascending=False)
        columns = strength.head(top).index.tolist()

    if ax is None:
        _, ax = plt.subplots(figsize=(8.8, 5.0))
    vals = []
    for i, col in enumerate(columns):
        if col not in cc_wide.index:
            continue
        y = cc_wide.loc[col].to_numpy(dtype=float)
        vals.append(y)
        name = labels.get(col, col) if labels else col
        ax.plot(lags, y, marker="o", ms=3.2, color=PALETTE[i % len(PALETTE)], label=name)
    ax.axvline(0.0, color=MUTED, lw=0.9, ls="--")
    ax.axhline(0.0, color=MUTED, lw=0.9)
    if ylim is None and vals:
        arr = np.concatenate(vals)
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        pad = max(0.05, 0.08 * (hi - lo))
        ylim = (max(-1.0, lo - pad), min(1.0, hi + pad))
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xticks(lags)
    ax.set_xlabel("indicator lead (periods); positive = indicator leads the target")
    ax.set_ylabel("correlation")
    t = _title(title, period)
    if t:
        ax.set_title(t)
    ax.legend(ncol=2, loc="lower center")
    return ax


# --------------------------------------------------------------------------- #
# Compositions used by the notebooks
# --------------------------------------------------------------------------- #
def plot_series(monthly: pd.DataFrame, *, groups=None, target=None, target_name="target",
                start=None, ncol: int = 3):
    """Alias for :func:`plot_series_grid` (small-multiples of the panel)."""

    return plot_series_grid(monthly, groups=groups, target=target,
                            target_name=target_name, start=start, ncol=ncol)


def plot_information(acc: pd.DataFrame, cc: pd.DataFrame, rank: pd.DataFrame | None = None,
                     *, top: int = 10, period=None, target_label: str = "the target",
                     leads=DEFAULT_LEADS):
    """Two panels: information accumulation (left) + top-N lead-lag correlations (right)."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(acc["days_to_publication"], acc["n_cells"], marker="o", color=PALETTE[0])
    axes[0].set_xlim(min(leads), max(leads))
    axes[0].set_xticks(list(leads))
    axes[0].set_xlabel("days to publication")
    axes[0].set_ylabel("within-quarter obs released")
    axes[0].set_title("Information accumulates through the cycle")
    plot_cross_correlation(cc, top=top, ax=axes[1], period=period,
                           title=f"Lead-lag correlation with {target_label}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Series grid and bars
# --------------------------------------------------------------------------- #
def plot_series_grid(monthly: pd.DataFrame, *, groups: dict | None = None,
                     target: pd.Series | None = None, target_name: str = "target",
                     start: str | None = None, ncol: int = 3):
    """Small-multiples of every series, coloured by group, with an optional target."""

    import matplotlib.pyplot as plt

    cols = list(monthly.columns)
    n = len(cols) + (1 if target is not None else 0)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.5, 2.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        s = monthly[c].loc[start:] if start else monthly[c]
        ax.plot(s.index, s.values, lw=1.1, color=group_color((groups or {}).get(c, "other")))
        ax.set_title(c, fontsize=8)
        ax.tick_params(labelsize=7)
    if target is not None:
        ax = axes[len(cols)]
        t = target.loc[start:] if start else target
        ax.plot(t.index, t.values, lw=1.7, color=INK)
        ax.set_title(f"TARGET: {target_name}", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_group_bar(counts: pd.Series, *, title: str | None = None, horizontal: bool = True, ax=None):
    """A single-series bar chart coloured by the group/category on its index."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, max(3.2, 0.32 * len(counts))))
    colors = [group_color(i) for i in counts.index]
    if horizontal:
        ax.barh(counts.index.astype(str), counts.values, color=colors)
        ax.invert_yaxis()
        ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    else:
        ax.bar(counts.index.astype(str), counts.values, color=colors)
    if title:
        ax.set_title(title)
    return ax


def plot_stacked_bar(frame: pd.DataFrame, *, title: str | None = None, ax=None):
    """Stacked bars (e.g. provider x frequency), using the soft palette."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 4.4))
    frame.plot(kind="bar", stacked=True, ax=ax, color=color_cycle(frame.shape[1]), width=0.72)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y")
    if title:
        ax.set_title(title)
    ax.legend(title=frame.columns.name)
    return ax


# --------------------------------------------------------------------------- #
# Uncertainty / GDPNow-style reporting
# --------------------------------------------------------------------------- #
def plot_nowcast_bands(bands: pd.DataFrame, quarters, *, target_label: str = "target",
                       ncol: int = 4, xlim: tuple[int, int] = (-120, 0), xticks=DEFAULT_LEADS,
                       suptitle: str = "Nowcast through the release cycle (GDPNow-style)"):
    """Small-multiples GDPNow-style panels: the nowcast path across the release
    cycle with 68/90 prediction bands, converging to the realised value.

    ``bands`` is the output of ``MIDAS.release_cycle_bands`` (columns include
    ``ref_quarter, days_to_publication, y_hat, y_true, lo_0.68, hi_0.68,
    lo_0.9, hi_0.9``).
    """

    import matplotlib.pyplot as plt

    qs = [pd.Timestamp(q) for q in quarters]
    nrow = int(np.ceil(len(qs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.9 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, q in zip(axes, qs):
        d = bands[pd.to_datetime(bands["ref_quarter"]) == q].sort_values("days_to_publication")
        if d.empty:
            ax.axis("off"); continue
        x = d["days_to_publication"]
        lvls = sorted((float(c[3:]) for c in d.columns if c.startswith("lo_")), reverse=True)
        for lv, a in zip(lvls, np.linspace(0.12, 0.30, max(len(lvls), 1))):
            lo, hi = f"lo_{lv:g}", f"hi_{lv:g}"
            if lo in d and hi in d:
                ax.fill_between(x, d[lo], d[hi], color=PALETTE[0], alpha=float(a), lw=0)
        ax.plot(x, d["y_hat"], color=PALETTE[0], lw=2, marker="o", ms=3)
        yt = d["y_true"].dropna()
        if len(yt):
            ax.axhline(float(yt.iloc[-1]), color=INK, ls="--", lw=1.2)
        ax.set_xlim(*xlim); ax.set_xticks(list(xticks))
        ax.set_title(f"{q.year}Q{q.quarter}", fontsize=9)
    for ax in axes[len(qs):]:
        ax.axis("off")
    fig.supxlabel("days to publication", fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    return fig


def plot_fan(history: pd.Series, path: pd.DataFrame, bands: pd.DataFrame | None = None, *,
             target_label: str = "target", history_quarters: int = 12,
             title: str | None = None, ax=None):
    """The h = 1..H fan chart: realized history + the live forecast path + bands.

    ``history`` = realized quarterly target (MIDAS quarter dating); ``path`` =
    live-forecast rows for ONE model (columns ``horizon``, ``ref_quarter``,
    ``y_hat``); ``bands`` = per-horizon offsets from ``forecast.fan`` (columns
    ``lo``/``hi`` indexed by horizon), drawn around the path.
    """

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8.8, 4.6))
    h = history.dropna().iloc[-history_quarters:]
    ax.plot(h.index, h.values, color=INK, lw=2.0, marker="s", ms=3.5, label="realized")

    p = path.sort_values("horizon")
    x = pd.to_datetime(p["ref_quarter"])
    xs = [h.index[-1], *x]
    ys = [float(h.iloc[-1]), *p["y_hat"]]
    if bands is not None and len(bands):
        lo = p["y_hat"].to_numpy() + bands["lo"].reindex(p["horizon"]).to_numpy()
        hi = p["y_hat"].to_numpy() + bands["hi"].reindex(p["horizon"]).to_numpy()
        ax.fill_between(x, lo, hi, color=PALETTE[0], alpha=0.18, lw=0, label="90% band")
    ax.plot(xs, ys, color=PALETTE[0], lw=2.0, ls="--", marker="o", ms=4, label="forecast")
    ax.axvline(h.index[-1], color=SPINE, lw=1.0, ls=":")
    ax.annotate(f"{float(p['y_hat'].iloc[-1]):.1f}", (x.iloc[-1], float(p["y_hat"].iloc[-1])),
                textcoords="offset points", xytext=(6, 4), fontsize=9,
                color=PALETTE[0], fontweight="semibold")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel(target_label)
    ax.set_title(title or "Forecast fan", loc="left")
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_current_nowcast(bands_q: pd.DataFrame, quarterly: pd.Series, *,
                         pool=None, target_label: str = "GDP YoY, %",
                         delay_days: int = 51, levels=(0.5, 0.7, 0.9),
                         history_quarters: int = 1, title: str | None = None):
    """The current-quarter nowcast in calendar time + its probability density.

    Panel (a): the last ``history_quarters`` realized quarter(s) (quarterly frequency,
    drawn as flat ink segments over each quarter's span) followed by the **weekly**
    real-time nowcast path of the current target quarter with shaded 50/70/90
    bands; the x-axis runs through the whole release cycle, up to the target's
    publication date (quarter end + ``delay_days``). Panel (b): the probability
    density of the latest nowcast on the same y-scale, shaded at the central
    ``levels``.

    ``bands_q`` = the band frame restricted to ONE model and ONE ref_quarter
    (columns ``origin_date, y_hat, lo_<lv>, hi_<lv>``); ``quarterly`` = the
    realized target series (MIDAS quarter dating); ``pool`` = the residual pool
    behind the latest band (optional, for the empirical density).
    """

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    d = bands_q.dropna(subset=["y_hat"]).sort_values("origin_date").copy()
    if d.empty:
        raise ValueError("bands_q has no finite nowcasts")
    d["origin_date"] = pd.to_datetime(d["origin_date"])
    Q = pd.Period(pd.Timestamp(d["ref_quarter"].iloc[0]), freq="Q")
    q_start, q_end = Q.start_time, Q.end_time.normalize()
    pub = q_end + pd.Timedelta(days=delay_days)

    fig, (ax, axd) = plt.subplots(
        1, 2, figsize=(10.6, 4.3), sharey=True,
        gridspec_kw={"width_ratios": [3.1, 1.0], "wspace": 0.04})

    # ---- panel (a): history (quarterly) + live path (weekly) -----------------
    hist = quarterly.dropna().iloc[-history_quarters:]
    for ts, val in hist.items():
        hq = pd.Period(pd.Timestamp(ts), freq="Q")
        ax.hlines(val, hq.start_time, hq.end_time, color=INK, lw=2.2, zorder=3)
        ax.plot(hq.start_time + (hq.end_time - hq.start_time) / 2, val,
                marker="s", ms=4.5, color=INK, zorder=4)
        ax.annotate(f"{hq.year}Q{hq.quarter}", (hq.start_time + (hq.end_time - hq.start_time) / 2, val),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=8, color=INK)

    ax.axvspan(q_start, q_end, color=MUTED, alpha=0.08, zorder=1)
    ax.axvline(q_end, color=SPINE, lw=1.0, ls="--", zorder=2)
    ax.axvline(pub, color=INK, lw=1.0, ls=":", zorder=2)

    x = d["origin_date"]
    for lv, a in zip(sorted(levels, reverse=True), np.linspace(0.10, 0.30, len(levels))):
        lo, hi = f"lo_{lv:g}", f"hi_{lv:g}"
        if lo in d.columns:
            ax.fill_between(x, d[lo], d[hi], color=PALETTE[0], alpha=float(a), lw=0, zorder=2)
    ax.plot(x, d["y_hat"], color=PALETTE[0], lw=2.0, marker="o", ms=3.5, zorder=5)

    last = d.iloc[-1]
    ax.plot(last["origin_date"], last["y_hat"], marker="o", ms=7,
            mfc="white", mec=PALETTE[0], mew=1.8, zorder=6)
    ax.annotate(f"{last['y_hat']:.2f}", (last["origin_date"], last["y_hat"]),
                textcoords="offset points", xytext=(-9, 9), ha="right", fontsize=9,
                color=PALETTE[0], fontweight="semibold", zorder=6)

    ax.annotate(f"expected release\n~{pub:%b %d}", (pub, ax.get_ylim()[0]),
                textcoords="offset points", xytext=(-4, 6), ha="right", va="bottom",
                fontsize=8, color=MUTED)
    ax.set_xlim(hist.index.min() and pd.Period(pd.Timestamp(hist.index.min()), freq="Q").start_time,
                pub + pd.Timedelta(days=6))
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=(1, 16)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylabel(target_label)
    ax.set_title(title or f"Nowcast of {Q.year}Q{Q.quarter} through the release cycle", loc="left")

    # ---- panel (b): density of the latest nowcast ----------------------------
    axd.set_title("density", loc="left", fontsize=10, color=MUTED)
    if pool is not None and len(pool) >= 5:
        sample = float(last["y_hat"]) + np.asarray(pool, dtype=float)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(sample)
            lo_y = min(sample.min(), float(last["y_hat"])) - 0.6
            hi_y = max(sample.max(), float(last["y_hat"])) + 0.6
            ys = np.linspace(lo_y, hi_y, 240)
            dens = kde(ys)
        except Exception:
            mu, sd = float(np.mean(sample)), float(np.std(sample) + 1e-9)
            ys = np.linspace(mu - 4 * sd, mu + 4 * sd, 240)
            dens = np.exp(-0.5 * ((ys - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        axd.plot(dens, ys, color=PALETTE[0], lw=1.4)
        for lv, a in zip(sorted(levels, reverse=True), np.linspace(0.10, 0.30, len(levels))):
            qlo, qhi = np.quantile(sample, [(1 - lv) / 2, 1 - (1 - lv) / 2])
            m = (ys >= qlo) & (ys <= qhi)
            axd.fill_betweenx(ys[m], 0, dens[m], color=PALETTE[0], alpha=float(a), lw=0)
        axd.axhline(float(last["y_hat"]), color=INK, lw=1.0, ls="-", xmax=0.9)
        axd.annotate(f"{last['y_hat']:.2f}", (dens.max() * 0.97, float(last["y_hat"])),
                     ha="right", va="bottom", fontsize=9, color=INK, fontweight="semibold")
        handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[0], alpha=float(a))
                   for a in np.linspace(0.30, 0.10, len(levels))]
        axd.legend(handles, [f"{int(lv * 100)}%" for lv in sorted(levels)],
                   loc="upper right", fontsize=7, frameon=False, title="probability", title_fontsize=7)
    else:
        axd.text(0.5, 0.5, "no calibration\npool", transform=axd.transAxes,
                 ha="center", fontsize=8, color=MUTED)
    axd.set_xticks([])
    for s in ("top", "right"):
        axd.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.10)
    return fig


def plot_pit(bands: pd.DataFrame, *, exclude_years=(), ax=None, title: str = "PIT histogram"):
    """Probability-integral-transform histogram (uniform if calibrated)."""

    import matplotlib.pyplot as plt

    d = bands.dropna(subset=["pit"])
    if exclude_years:
        d = d[~pd.to_datetime(d["ref_quarter"]).dt.year.isin(list(exclude_years))]
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.hist(d["pit"], bins=10, range=(0, 1), color=PALETTE[0], edgecolor="white")
    ax.axhline(len(d) / 10, color=INK, ls="--", lw=1.0)
    ax.set_xlabel("PIT"); ax.set_ylabel("count"); ax.set_title(title)
    return ax


def coverage_table(bands: pd.DataFrame, subperiods: dict, *, levels=(0.68, 0.90)) -> pd.DataFrame:
    """Realised coverage of each band level per subperiod (target: the nominal level)."""

    rows = []
    for label, spec in subperiods.items():
        start, end, *rest = spec
        excl = rest[0] if rest else ()
        d = bands.copy()
        rq = pd.to_datetime(d["ref_quarter"])
        d = d[(rq >= pd.Timestamp(start)) & (rq <= pd.Timestamp(end))]
        if excl:
            d = d[~pd.to_datetime(d["ref_quarter"]).dt.year.isin(list(excl))]
        row = {"subperiod": label, "n": int(d["y_true"].notna().sum())}
        for lv in levels:
            lo, hi = f"lo_{lv:g}", f"hi_{lv:g}"
            dd = d.dropna(subset=[lo, hi, "y_true"])
            row[f"cov_{int(lv*100)}"] = float(((dd["y_true"] >= dd[lo]) & (dd["y_true"] <= dd[hi])).mean()) if len(dd) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("subperiod")


__all__ = [
    "PALETTE", "INK", "MUTED", "GRID", "SPINE", "GROUP_PALETTE", "DEFAULT_LEADS",
    "set_style", "group_color", "color_cycle",
    "fmt_period",
    "plot_horse_race", "plot_horse_race_subperiods", "plot_cross_correlation", "plot_series_grid",
    "plot_series", "plot_information", "plot_group_bar", "plot_stacked_bar",
    "plot_nowcast_bands", "plot_pit", "coverage_table",
]
