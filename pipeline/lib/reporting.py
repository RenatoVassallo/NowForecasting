"""Figures, per-target reports and artifact persistence for a run.

Reuses the shared house style (`analysis.plots`) so the production figures look
exactly like the notebooks'. All markdown is deliberately short and scannable.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from analysis import plots  # noqa: E402

from ..config import metadata  # noqa: E402
from .nowcast_job import Result  # noqa: E402

plots.set_style()
ADAPTIVE = metadata.ADAPTIVE["name"]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def build_figures(result: Result, out_dir) -> dict:
    """Current nowcast, horse race, weight evolution, GDPNow panels. Returns {name: fig}."""

    figs = {}
    cmap = plots.model_color_map(list(result.nowcasts["model"].unique()))

    # The headline figure: the live quarter in calendar time + its density.
    q = result.bands["ref_quarter"].max()
    bq = result.bands[result.bands["ref_quarter"] == q]
    if bq["y_hat"].notna().any():
        last = bq.dropna(subset=["y_hat"]).sort_values("origin_date").iloc[-1]
        pool = result.pools.get((pd.Timestamp(q), int(last["bin"])))
        figs["current"] = plots.plot_current_nowcast(
            bq, result.quarterly[result.target.target], pool=pool,
            target_label=result.target.label, delay_days=result.target.target_delay_days,
            levels=metadata.BANDS["levels"])

    figs["horse_race"] = plots.plot_horse_race_subperiods(
        result.curves, result.summary, highlight=5, colors=cmap,
        suptitle=f"{result.target.label}: release-cycle horse race")

    fig_w, ax = plt.subplots(figsize=(8.8, 4.6))
    plots.plot_weight_evolution(result.nowcasts, result.weights, result.members,
                                colors=cmap, n_bins=metadata.ADAPTIVE["n_bins"], ax=ax)
    figs["weights"] = fig_w

    # Last-8 panels: HISTORICAL (realized) quarters only - the live quarter has
    # its own dedicated figure above.
    realized = result.bands[result.bands["y_true"].notna()]
    recent = sorted(pd.to_datetime(realized["ref_quarter"]).dropna().unique())[-8:]
    if recent:
        figs["gdpnow"] = plots.plot_nowcast_bands(
            realized, recent, target_label=result.target.target, ncol=4,
            suptitle=f"{result.target.label}: Adaptive-IC nowcast + bands (realized quarters)")
    return figs


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_result(store, result: Result, params) -> list:
    """Persist every artifact for one target; returns the saved figure paths."""

    role = result.target.role + "s" if result.target.role == "satellite" else result.target.role
    d = store.target_dir(role, result.target.name)

    if params.SAVE_NOWCASTS:
        store.save_df(d / "nowcasts.parquet", result.nowcasts, kind="nowcasts")
        store.save_df(d / "weights.parquet", result.weights, kind="weights")
        store.save_df(d / "bands.parquet", result.bands, kind="bands")
    store.save_df(d / "latest.parquet", result.latest, kind="latest-nowcast")

    md = store.dir(role, result.target.name, "metrics")
    for lab, s in result.summary.items():
        store.save_df(md / f"summary_{lab.replace(' ', '_')}.csv", s, kind="metrics")

    if params.SAVE_MODELS:
        store.save_pickle(store.dir(role, result.target.name, "models") / "ladder.pkl",
                          result.models, kind="models")

    fig_paths = []
    if params.SAVE_FIGURES:
        fd = store.dir(role, result.target.name, "figures")
        for name, fig in build_figures(result, fd).items():
            fig_paths.append(store.save_fig(fd / f"{name}.png", fig))
            plt.close(fig)

    store.save_text(d / "report.md", target_report(result), kind="report")
    return fig_paths


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def target_report(result: Result) -> str:
    t = result.target
    q = pd.Timestamp(result.latest["ref_quarter"].max())
    qp = pd.Period(q, freq="Q")
    pub = qp.end_time.normalize() + pd.Timedelta(days=t.target_delay_days)
    lines = [f"## {t.label}  \n`{t.name}` &middot; {t.role}", ""]

    ad = result.latest[result.latest["model"] == ADAPTIVE]
    if len(ad):
        r = ad.iloc[0]
        band = ""
        lv = min(metadata.BANDS["levels"])          # headline = the 50% interval
        lo, hi = f"lo_{lv:g}", f"hi_{lv:g}"
        if lo in ad.columns and pd.notna(r.get(lo)):
            band = f" &middot; {int(lv * 100)}% [{r[lo]:.2f}, {r[hi]:.2f}]"
        live = pd.isna(r.get("y_true"))
        status = f"publishes ~{pub:%b %d, %Y}" if live else "published"
        lines += [f"**Nowcast for {qp.year}Q{qp.quarter}: {r['y_hat']:.2f}**{band}  ",
                  f"_Adaptive-IC at origin {pd.Timestamp(r['origin_date']):%b %d} "
                  f"({int(r['days_to_publication'])}d to release; {status})._", ""]

    lines += ["| subperiod | best model | Adaptive-IC rel_" + t.baseline + " | fires |",
              "|---|---|---|---|"]
    for lab, s in result.summary.items():
        best = s.index[0]
        rel, fin = "-", "-"
        if ADAPTIVE in s.index:
            rel = f"{s.loc[ADAPTIVE, 'rel_' + t.baseline]:.2f}"
            fin = f"{s.loc[ADAPTIVE, 'finite']:.0%}"
        lines.append(f"| {lab} | {best} | {rel} | {fin} |")
    lines += ["", f"_Models: {', '.join(result.nowcasts['model'].unique())}._", ""]
    return "\n".join(lines)


def whatsnew_report(name: str, monthly: pd.DataFrame, quarterly: pd.DataFrame, prev_monthly=None) -> str:
    last_m = pd.Timestamp(monthly.dropna(how="all").index.max())
    last_q = pd.Timestamp(quarterly.dropna().index.max())
    lines = [f"### {name}",
             f"- monthly panel through **{last_m.date()}** ({monthly.shape[1]} series)",
             f"- target realized through **{last_q.year}Q{last_q.quarter}**"]
    if prev_monthly is not None:
        prev_last = pd.Timestamp(prev_monthly.dropna(how="all").index.max())
        n_new = int((monthly.dropna(how="all").index > prev_last).sum())
        lines.append(f"- new since last run: **{n_new} month(s)**" if n_new else "- no new months since last run")
    return "\n".join(lines)


def final_report(run_id: str, whatsnew: str, target_reports: list[str], timings: dict) -> str:
    total = sum(timings.values())
    head = [f"# NowForecasting run {run_id}", "",
            f"_Total {total:.0f}s &middot; " + " &middot; ".join(f"{k} {v:.0f}s" for k, v in timings.items()) + "_",
            "", "## 1 &middot; Data", "", whatsnew, ""]
    body = ["## 2-4 &middot; Nowcasts & forecasts", ""] + target_reports
    tail = ["", "---", "_Artifacts (vintage data, all-model nowcasts, weights, bands, metrics, "
            "models, figures) are saved under this run's folder; see `manifest.json`._"]
    return "\n".join(head + body + tail)
