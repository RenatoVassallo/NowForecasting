"""Stage 4: the published figures.

One visual language (pipeline/lib/style.py), no in-figure headline titles - the
Beamer frames carry those. Every builder returns a Figure; ``run`` writes both
PDF (for the report) and PNG (for quick viewing) under ``products/figures``.

The flagship is the release-cycle nowcast: the weekly nowcast path of the
current target quarter with bands that NARROW as information arrives - an
inverted fan - plus the probability density of today's estimate as a sidecar.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.lib import style as S

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# context
# --------------------------------------------------------------------------- #
def load_context(as_of=None, store=None) -> dict:
    """Figure inputs, ALL from the current run: the satellite paths this run's
    chain resolved (run-local or one validated prior bundle), this run's fan,
    and this run's official nowcast artifacts. No global surface is read, and
    nothing is cached across runs (two runs in one process stay independent)."""
    from pipeline.lib.context import resolve_as_of

    as_of = resolve_as_of(None) if as_of is None else __import__("pandas").Timestamp(as_of).normalize()
    if store is None:
        raise RuntimeError(
            "fanchart context requires the run store: figures consume this "
            "run's artifacts, never a global surface")
    import targets
    from targets import china as cn
    from targets import commodities as cmd

    root = Path(store.root)
    blocks = {str(k): Path(v) for k, v in (getattr(store, "blocks", {}) or {}).items()}
    sources = {"usa": blocks.get("usa", root / "blocks" / "us_path_uncertainty.csv"),
               "china": blocks.get("china", root / "blocks" / "china_path_uncertainty.csv"),
               "tot": blocks.get("commodities", root / "blocks" / "tot_path_uncertainty.csv"),
               "peru": root / "peru_gdp_fan.csv"}
    ctx: dict = {}
    for name, fpath in sources.items():
        if not Path(fpath).exists():
            raise FileNotFoundError(
                f"fanchart input missing: {fpath}. Run the forecast stage in "
                "this run (or pass a run that has it) before the figures.")
        d = pd.read_csv(fpath)
        d["period"] = pd.PeriodIndex(d["quarter"], freq="Q")
        ctx[name] = d

    spec = targets.get("peru_gdp")
    _, qq, _ = spec.load_panel()
    y = qq[spec.target].dropna(); y.index = pd.PeriodIndex(y.index, freq="Q")
    ctx["peru_hist"], ctx["peru_spec"] = y, spec

    _, cq, _ = cn.load_panel()
    g = cq[cn.TARGET].dropna(); g.index = pd.PeriodIndex(g.index, freq="Q")
    ctx["china_hist"] = g
    tspec = cmd.SPECS["pe_tot"]
    _, tq, _ = tspec.load_panel()
    t = tq[tspec.target].dropna(); t.index = pd.PeriodIndex(t.index, freq="Q")
    ctx["tot_hist"] = t

    cm, _, _ = cn.load_panel()
    from pipeline.blocks._common import fill_structural_january
    ip = fill_structural_january(cm["ip_cum_yoy"]).dropna()   # merged Jan-Feb
    per = pd.PeriodIndex(ip.index, freq="Q")
    gq = ip.groupby(per)
    ctx["ip_hist"] = gq.mean().where(gq.count() >= 3).dropna()
    both = pd.concat([ctx["ip_hist"].rename("ip"), g.rename("gdp")], axis=1).dropna()
    both = both[(both.index.year >= 2012) & (~both.index.year.isin([2020, 2021]))]
    ctx["bridge"] = np.polyfit(both["gdp"], both["ip"], 1)

    # Peru release-cycle machinery, OFFICIAL artifacts only. The flagship sweep
    # and headline value are read from the files the nowcast stage wrote (one
    # definition everywhere); this stage recomputes NOTHING about the nowcast.
    # The rc frame (members + Adaptive-IC over history) is still needed for the
    # information-bin residual pools behind the band shading; it comes from the
    # run store when this run executed the nowcast stage, else from the newest
    # prior run that did.
    from nowcast.release_cycle import conditional_bands
    from pipeline.config import metadata
    from pipeline.lib.nowcast_artifact import load_official, load_sweep

    aname = metadata.ADAPTIVE["name"]
    rc_all = None
    roots = []
    if store is not None:
        roots.append(Path(store.root))
        roots += sorted((p for p in Path(store.runs_dir).glob("*")
                         if p.is_dir() and not p.is_symlink()), reverse=True)
    for r in roots:
        f = r / "domestic" / "peru_gdp" / "nowcasts.parquet"
        if f.exists():
            rc_all = pd.read_parquet(f)
            break
    if rc_all is None:
        raise FileNotFoundError(
            "no run with a saved Peru nowcast frame; run the nowcast stage "
            "before the figures")
    bands, pools = conditional_bands(rc_all, aname, index_col="info_index",
                                     n_bins=4, levels=(0.30, 0.60, 0.90),
                                     lookback_years=12, exclude_years=(2020, 2021),
                                     min_quarters=6, collect_pools=True)
    ctx["rc_bands"], ctx["rc_pools"], ctx["rc_all"] = bands, pools, rc_all

    sw = load_sweep(expected_as_of=as_of, path=root / "peru_nowcast_sweep.csv")
    sweep = pd.DataFrame({
        "origin_date": pd.to_datetime(sw.origin_date),
        "y_hat": sw.y_hat.to_numpy(dtype=float),
        "dtp": sw.days_to_publication.to_numpy(),
        "ref": pd.to_datetime(sw.ref_quarter),
        "info": sw.info_index.to_numpy(dtype=float)})
    sweep["bin"] = np.clip(np.digitize(sweep["info"], np.linspace(0, 1, 5)[1:-1]), 0, 3)
    ctx["sweep"] = sweep

    official = load_official(expected_as_of=as_of,
                             path=root / "peru_nowcast_official.csv")
    # tolerance: above the artifact's 4-decimal storage rounding, far below the
    # 0.1pp reporting resolution
    if abs(float(official["value"]) - float(sweep["y_hat"].iloc[-1])) > 1e-3:
        raise ValueError(
            "flagship sweep endpoint differs from the official nowcast "
            f"({sweep['y_hat'].iloc[-1]:.4f} vs {official['value']:.4f}); "
            "the stages are not consuming one artifact")
    ctx["official"] = official

    # data frontier inputs: the run's own availability artifact + registry
    avail_path = root / "data_quality" / "availability.csv"
    if not avail_path.exists():
        raise FileNotFoundError(
            f"fanchart input missing: {avail_path}; the preflight writes it "
            "before any estimation stage")
    import json as _json
    reg_path = REPO / "pipeline" / "config" / "data_registry.json"
    ctx["availability"] = pd.read_csv(avail_path)
    ctx["registry"] = _json.loads(reg_path.read_text())
    ctx["as_of"] = as_of
    return ctx


def _x(idx):
    return pd.PeriodIndex(idx, freq="Q").to_timestamp(how="end")


def _pool_for_bin(pools, b):
    cand = [(q, v) for (q, bb), v in pools.items() if bb == int(b)]
    return max(cand, key=lambda t: t[0])[1] if cand else None


def _date_axis(ax, start, end, years_y=-0.16):
    ticks = pd.date_range(start.normalize(), end, freq="14D")
    ax.set_xticks(ticks)
    ax.set_xticklabels([t.strftime("%d-%b").lower() for t in ticks], fontsize=8)
    yrs = sorted({t.year for t in ticks})
    ax.text(0.5, years_y, " / ".join(str(y) for y in yrs), transform=ax.transAxes,
            ha="center", fontsize=9, color=S.MUTED)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=S.BORDER, lw=0.55, alpha=0.8)


def _density_sidecar(axd, values, colour, mode_val):
    """Horizontal probability density with 30/60/90 central regions shaded."""
    kde = stats.gaussian_kde(values)
    lo, hi = np.quantile(values, [0.002, 0.998])
    ys = np.linspace(lo, hi, 300)
    dens = kde(ys)
    axd.fill_betweenx(ys, 0, dens, color=colour, alpha=0.14, lw=0)
    for cov, a in ((0.90, 0.16), (0.60, 0.30), (0.30, 0.52)):
        ql, qh = np.quantile(values, [(1 - cov) / 2, 1 - (1 - cov) / 2])
        m = (ys >= ql) & (ys <= qh)
        axd.fill_betweenx(ys[m], 0, dens[m], color=colour, alpha=a, lw=0)
    axd.plot(dens, ys, color=colour, lw=1.4)
    axd.axhline(mode_val, color=S.INK, lw=1.0, ls=(0, (3, 2)))
    axd.set_xticks([]); axd.grid(False)
    axd.spines["bottom"].set_visible(False)
    axd.tick_params(left=False, labelleft=False)
    axd.set_title("density", fontsize=8, color=S.MUTED, pad=4)


# --------------------------------------------------------------------------- #
# 1. the flagship: nowcast through the release cycle
# --------------------------------------------------------------------------- #
def fig_nowcast_cycle(ctx) -> plt.Figure:
    S.set_style()
    sweep, pools = ctx["sweep"], ctx["rc_pools"]
    spec = ctx["peru_spec"]
    ref_q = pd.Period(pd.Timestamp(sweep["ref"].iloc[0]), freq="Q")
    # observed-release calendar first (evidence for THIS quarter's release),
    # canonical scalar rule otherwise; identical while no history exists
    from pipeline.lib.release_calendar import expected_target_publication
    pub, pub_basis = expected_target_publication(
        ref_q, spec.target_delay_days,
        availability=ctx.get("availability"), code=spec.target)

    fig = plt.figure(figsize=(11.8, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=(4.4, 1.0), wspace=0.04)
    ax = fig.add_subplot(gs[0])
    xs = pd.DatetimeIndex(sweep["origin_date"])
    # the inverted fan: bands from each origin's information-bin pool
    los, his = {c: [] for c in (30, 60, 90)}, {c: [] for c in (30, 60, 90)}
    for _, r in sweep.iterrows():
        pool = _pool_for_bin(pools, r["bin"])
        for cov in (30, 60, 90):
            a = (1 - cov / 100) / 2
            ql, qh = (np.quantile(pool, [a, 1 - a]) if pool is not None else (np.nan, np.nan))
            los[cov].append(r.y_hat + ql); his[cov].append(r.y_hat + qh)
    for cov, alpha in ((90, 0.14), (60, 0.28), (30, 0.48)):
        ax.fill_between(xs, los[cov], his[cov], color=S.ACCENT, alpha=alpha, lw=0)
    ax.plot(xs, sweep["y_hat"], color=S.ACCENT, lw=2.2, marker="o", ms=4.5,
            mec="white", mew=0.8)
    last = sweep.iloc[-1]
    ax.annotate(f"{last.y_hat:.1f}%", (xs[-1], last.y_hat), xytext=(12, 10),
                textcoords="offset points", fontsize=15, fontfamily=S.SERIF,
                color=S.ACCENT, fontweight="bold")
    ax.axvline(pub, color=S.INK, lw=1.1, ls=(0, (4, 3)))
    pub_note = ("expected release" if pub_basis == "canonical_lag"
                else "expected release (observed cadence)")
    ax.annotate(pub_note, (pub, ax.get_ylim()[0]), xytext=(-8, 12),
                textcoords="offset points", rotation=90, fontsize=8, color=S.MUTED,
                ha="right", va="bottom")
    _date_axis(ax, xs.min(), pub + pd.Timedelta(days=7), years_y=-0.13)
    ax.set_ylabel("% YoY")
    ax.set_xlim(xs.min() - pd.Timedelta(days=4), pub + pd.Timedelta(days=10))
    S.band_legend(ax, S.ACCENT, loc="upper left")

    pool = _pool_for_bin(pools, last["bin"])
    axd = fig.add_subplot(gs[1], sharey=ax)
    _density_sidecar(axd, last.y_hat + pool, S.ACCENT, last.y_hat)
    fig.suptitle(f"Nowcast of {ref_q} through the release cycle", x=0.005, ha="left",
                 fontfamily=S.SERIF, fontsize=14, color=S.INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    ctx["headline_nowcast"] = float(last.y_hat)
    ctx["headline_quarter"] = str(ref_q)
    return fig


# --------------------------------------------------------------------------- #
# 2. appendix: the last 8 realised quarters through their cycles
# --------------------------------------------------------------------------- #
def fig_cycle_grid(ctx) -> plt.Figure:
    S.set_style()
    b = ctx["rc_bands"].dropna(subset=["lo_0.3"])
    quarters = sorted(b.ref_quarter.unique())[-8:]
    fig, axes = plt.subplots(2, 4, figsize=(12.6, 5.6), sharex=True)
    for ax, q in zip(axes.ravel(), quarters):
        g = b[b.ref_quarter == q].sort_values("days_to_publication")
        for lv, a in ((0.9, 0.14), (0.6, 0.28), (0.3, 0.48)):
            ax.fill_between(g.days_to_publication, g[f"lo_{lv}"], g[f"hi_{lv}"],
                            color=S.ACCENT, alpha=a, lw=0)
        ax.plot(g.days_to_publication, g.y_hat, color=S.ACCENT, lw=1.7, marker="o", ms=2.6)
        ax.axhline(g.y_true.iloc[0], color=S.INK, lw=1.2, ls=(0, (4, 2)))
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=S.BORDER, lw=0.5, alpha=0.7)
        ax.set_title(str(pd.Period(pd.Timestamp(q), freq="Q")), loc="left",
                     fontsize=9.5, color=S.INK)
    for ax in axes[1]:
        ax.set_xlabel("days to publication", fontsize=8)
    fig.suptitle("The nowcast through past release cycles (dashed line: outturn)",
                 x=0.005, ha="left", fontfamily=S.SERIF, fontsize=13, color=S.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# --------------------------------------------------------------------------- #
# 3. China and terms of trade: current-quarter estimates with densities
# --------------------------------------------------------------------------- #
def _node_with_density(fig, gs_slot, hist, node, colour, label):
    from forecast.fan_mc import tpn_scales

    inner = gs_slot.subgridspec(1, 2, width_ratios=(3.2, 1.0), wspace=0.05)
    ax = fig.add_subplot(inner[0])
    h = hist.iloc[-10:]
    ax.plot(_x(h.index), h.values, color=S.INK, lw=1.8, marker="s", ms=3.4)
    x0 = _x([node["period"]])[0]
    for cov, a, w in ((90, 0.18, 8), (60, 0.34, 8), (30, 0.55, 8)):
        ax.plot([x0, x0], [node[f"lo{cov}"], node[f"hi{cov}"]], color=colour,
                alpha=a, lw=w, solid_capstyle="butt")
    ax.plot([x0], [node["mode"]], marker="o", ms=6, color=colour, mec="white", mew=1.1)
    # label ABOVE the node: to the right it runs into the density gutter
    ax.annotate(f"{node['mode']:.1f}%", (x0, node["hi90"]), xytext=(0, 7),
                textcoords="offset points", fontsize=13, fontfamily=S.SERIF,
                color=colour, fontweight="bold", ha="center", clip_on=False)
    ax.set_title(f"{label} - {node['quarter']}", loc="left", fontsize=10.5,
                 fontfamily=S.SERIF, color=S.INK)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("% YoY")
    # room for the label: one tick per year, margins right and above
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.10)
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_, hi_ + 0.14 * (hi_ - lo_))

    sl, sr = float(node["sigma_left"]), float(node["sigma_right"])
    draws = float(node["mode"]) + np.where(np.random.default_rng(3).random(4000) < sl / (sl + sr),
        -np.abs(np.random.default_rng(4).normal(0, sl, 4000)),
        np.abs(np.random.default_rng(5).normal(0, sr, 4000)))
    axd = fig.add_subplot(inner[1], sharey=ax)
    _density_sidecar(axd, draws, colour, float(node["mode"]))


def fig_china_tot(ctx) -> plt.Figure:
    S.set_style()
    fig = plt.figure(figsize=(12.6, 4.6))
    gs = fig.add_gridspec(1, 2, wspace=0.22)
    _node_with_density(fig, gs[0], ctx["china_hist"], ctx["china"].iloc[0],
                       S.GOLD, "China GDP growth")
    _node_with_density(fig, gs[1], ctx["tot_hist"], ctx["tot"].iloc[0],
                       S.OLIVE, "Peru terms of trade")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 4. conditioning variables
# --------------------------------------------------------------------------- #
def fig_conditioning(ctx) -> plt.Figure:
    S.set_style()
    b1, b0 = ctx["bridge"]
    cn = ctx["china"]
    ip_path = cn.copy()
    ip_path["centre"] = b0 + b1 * cn["centre"]
    ip_path[["sigma_left", "sigma_right"]] = cn[["sigma_left", "sigma_right"]] * abs(b1)

    from targets import usa as usmod
    um, _, _ = usmod.load_panel()
    us_hist = um["us_gdp_yoy_m"].dropna()
    us_hist.index = pd.PeriodIndex(us_hist.index, freq="Q")
    us_hist = us_hist[~us_hist.index.duplicated(keep="last")]
    mm_p, _, _ = ctx["peru_spec"].load_panel()
    exp_hist = mm_p["exp_eco3m"].dropna()
    exp_q = exp_hist.groupby(pd.PeriodIndex(exp_hist.index, freq="Q")).mean()

    panels = [
        ("United States GDP", us_hist.iloc[-14:], ctx["usa"], S.BROWN, None),
        ("China industrial production", ctx["ip_hist"].iloc[-14:], ip_path, S.GOLD, None),
        ("Peru terms of trade", ctx["tot_hist"].iloc[-14:], ctx["tot"], S.OLIVE, None),
        ("Business expectations (3m)", exp_q.iloc[-14:], None, S.FAINT,
         float(exp_hist.iloc[-1])),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 6.9))
    for ax, (label, hist, path, colour, flat) in zip(axes.ravel(), panels):
        ax.plot(_x(hist.index), hist.values, color=S.INK, lw=1.7)
        if path is not None:
            S.fan_bands(ax, _x(path["period"]), path["centre"], path["sigma_left"],
                        path["sigma_right"], colour, coverages=(0.90, 0.60),
                        alphas=(0.18, 0.40),
                        anchor=(_x(hist.index)[-1], float(hist.iloc[-1])))
        else:
            xs = _x(ctx["usa"]["period"])
            ax.plot([_x(hist.index)[-1], *xs], [hist.iloc[-1]] + [flat] * len(xs),
                    color=colour, lw=1.9, ls=(0, (5, 2)))
            ax.annotate("assumption: held flat", (xs[len(xs) // 2], flat),
                        xytext=(0, 8), textcoords="offset points", fontsize=8,
                        color=S.MUTED, ha="center")
        ax.axvline(_x(hist.index)[-1], color=S.BORDER_STRONG, ls=":", lw=1)
        ax.axhline(50 if "expectations" in label.lower() else 0,
                   color=S.BORDER_STRONG, lw=0.9)
        ax.set_title(label, loc="left", fontsize=10.5, fontfamily=S.SERIF, color=S.INK)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("index" if "expectations" in label.lower() else "% YoY",
                      fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 5. the main fan
# --------------------------------------------------------------------------- #
def fig_fan_main(ctx) -> plt.Figure:
    S.set_style()
    fig, ax = plt.subplots(figsize=(8.9, 5.4))
    hist = ctx["peru_hist"].iloc[-14:]
    ax.plot(_x(hist.index), hist.values, color=S.INK, lw=2.0)
    d = ctx["peru"]
    S.fan_bands(ax, _x(d["period"]), d["mode"], d["sigma_left"], d["sigma_right"],
                S.ACCENT, anchor=(_x(hist.index)[-1], float(hist.iloc[-1])))
    ax.annotate(f"{d['mode'].iloc[-1]:.1f}", (_x(d['period'])[-1], d["mode"].iloc[-1]),
                xytext=(8, 0), textcoords="offset points", fontsize=11,
                color=S.ACCENT, fontweight="bold", va="center", fontfamily=S.SERIF)
    ax.axvline(_x(hist.index)[-1], color=S.BORDER_STRONG, ls=":", lw=1)
    ax.axhline(0, color=S.BORDER_STRONG, lw=0.9)
    ax.tick_params(right=True, labelright=True)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("% YoY")
    S.band_legend(ax, S.ACCENT, loc="lower left")
    fig.tight_layout()
    return fig




def fig_data_frontier(ctx):
    """Curated per-variable information clocks in five fixed blocks: GREEN
    through the last released observation, GREY while waiting for the next
    expected release, white beyond. A star marks series whose latest
    observation reached the panel within the seven days before the as-of;
    an amber marker flags releases the provider has already published but
    the panel has not ingested yet (observed-release calendar detection)."""
    from pipeline.config.metadata import FRONTIER_LAYOUT
    from pipeline.lib.data_frontier import frontier_frame

    as_of = ctx["as_of"]
    f = frontier_frame(ctx["availability"], ctx["registry"], as_of,
                       layout=FRONTIER_LAYOUT)
    x0 = as_of - pd.DateOffset(months=6)
    x1 = as_of + pd.DateOffset(months=3)
    AMBER = "#d9a441"

    n_rows = len(f) + f.block.nunique()           # series + block-title rows
    fig, ax = plt.subplots(figsize=(9.8, 0.28 * n_rows + 1.6))
    y = 0
    yticks, ylabels = [], []
    block_rows = []
    for block, rows in reversed([(b, g) for b, g in f.groupby("block", sort=False)]):
        for _, r in rows.iloc[::-1].iterrows():   # bottom-up inside the block
            green_start = pd.Timestamp(x0)
            ax.barh(y, (min(r.obs_end, x1) - green_start).days, left=green_start,
                    height=0.62, color="#79b791", edgecolor="none", zorder=3)
            if r.next_release > r.obs_end:
                ax.barh(y, (min(r.next_release, x1) - max(r.obs_end, green_start)).days,
                        left=max(r.obs_end, green_start), height=0.62,
                        color="#d3d3d0", edgecolor="none", zorder=3)
            if r.released_pending_ingest:
                d = pd.Timestamp(r.detected_release_date)
                ax.plot([d], [y], marker="v", ms=6, color=AMBER, zorder=6)
                ax.annotate(f"out {d.strftime('%d %b').lower()}, pending ingest",
                            (d, y), xytext=(5, -2.5), textcoords="offset points",
                            fontsize=6.5, color=AMBER, zorder=6)
            elif r.continuous:
                ax.annotate("updates ~weekly", (min(r.next_release, x1), y),
                            xytext=(4, -3), textcoords="offset points",
                            fontsize=6.5, color=S.MUTED, zorder=6)
            elif pd.Timestamp(x0) < r.next_release <= x1:
                ax.plot([r.next_release], [y], marker="|", ms=11, color=S.INK,
                        zorder=5)
                if r.days_to_next < 0:
                    # provider running late vs its expected release
                    ax.annotate(f"late {-r.days_to_next}d", (r.next_release, y),
                                xytext=(4, -3), textcoords="offset points",
                                fontsize=6.5, color="#b04a4a", zorder=6)
                elif r.next_release <= x1 - pd.Timedelta(days=6):
                    ax.annotate(f"{r.days_to_next}d", (r.next_release, y),
                                xytext=(4, -3), textcoords="offset points",
                                fontsize=6.5, color=S.MUTED, zorder=6)
            name = r.label if len(r.label) <= 34 else r.label[:33] + "…"
            if r.new_this_run:
                # mathtext star: independent of the text font's glyph coverage
                name = "$\\bigstar$ " + name
            yticks.append(y)
            ylabels.append(name)
            y += 1
        block_rows.append((y, block))             # title row above the block
        y += 1
    for yb, block in block_rows:
        ax.text(pd.Timestamp(x0) - pd.Timedelta(days=3), yb - 0.28, block.upper(),
                fontsize=7.5, color=S.MUTED, fontweight="bold",
                ha="left", va="center", clip_on=False)
    ax.axvline(as_of, color=S.INK, lw=1.1, ls="--", zorder=4)
    ax.annotate(f"as of {as_of.date()}", (as_of, y - 0.6), xytext=(5, 0),
                textcoords="offset points", fontsize=8, color=S.INK)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.8, y - 0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.grid(axis="x", color=S.BORDER, lw=0.5, alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.set_title("Data frontier: released (green), awaiting next release (grey); "
                 "$\\bigstar$ = new in this run, amber = published, pending ingest",
                 loc="left", fontsize=10)
    fig.tight_layout()
    return fig

FIGURES = {
    "nowcast_cycle": fig_nowcast_cycle,
    "cycle_grid": fig_cycle_grid,
    "china_tot": fig_china_tot,
    "conditioning": fig_conditioning,
    "fan_main": fig_fan_main,
    "data_frontier": fig_data_frontier,
}


def run(store, params, panels=None) -> list[str]:
    rctx = getattr(store, "ctx", None)
    ctx = load_context(as_of=rctx.as_of if rctx is not None else None, store=store)
    store.fig_ctx = ctx
    lines = []
    dest = Path(store.root) / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    failures = []
    for name, builder in FIGURES.items():
        try:
            fig = builder(ctx)
            for ext, dpi in (("pdf", None), ("png", 170)):
                tgt = dest / f"{name}.{ext}"
                fig.savefig(tgt, dpi=dpi, bbox_inches="tight")
                if hasattr(store, "_track"):
                    store._track(tgt, "figure")
            plt.close(fig)
            lines.append(f"- figure {name}: ok")
            print(f"  [fanchart] {name}")
        except Exception as exc:
            failures.append(f"{name} ({type(exc).__name__}: {exc})")
            print(f"  [fanchart] {name} FAILED: {exc}")
    if failures:
        # a published report with silently missing figures is worse than a
        # failed run: the stage fails, the run is quarantined, nothing publishes
        raise RuntimeError("fanchart: figure generation failed for "
                           + "; ".join(failures))
    return lines
