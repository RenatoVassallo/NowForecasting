"""Release-cycle workflow shared by the notebooks and the production pipeline.

One implementation of the h = 0 mechanics, so an experiment in a notebook and a
production run are the same code:

* :func:`run_horse_race` - the two-pass release-cycle backtest on a target's
  common floor (coarse grid + a cheap near-publication pass for off-grid leads).
* :func:`live_path` - the **live** sweep for the current, still-unpublished
  quarter (the backtest harness skips quarters with no realized value); weekly
  origins up to today, same row schema as the backtest.
* :func:`adaptive_combine` - Adaptive-IC combination + the information-index
  propagation onto the combined rows (so bands can condition on it).
* :func:`conditional_bands` - empirical prediction bands conditioned on the
  information index, with an optional **rolling lookback window** (production
  calibrates on the last ``lookback_years`` of errors; pass ``None`` to use the
  full history as in the research notebooks).
* :func:`latest_nowcast` - the newest nowcast per model (+ the adaptive band).
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from MIDAS import add_information_index, combine_release_cycle, run_release_cycle_backtest
from MIDAS.backtest import make_daily_origin_grid, publication_date, quarter_end
from MIDAS.realtime import RealtimeEngine

# Models too expensive to re-run on the fine near-publication grid, where they
# are flat anyway (their -1 value ~ their -15 value).
SLOW_MODELS = ("DFMNowcaster", "GBTreesNowcaster")


# --------------------------------------------------------------------------- #
# Backtest (historical quarters)
# --------------------------------------------------------------------------- #
def run_horse_race(panel, spec, models: dict, *, n_jobs: int = 8,
                   days_before: int = 120, step_days: int = 15,
                   eval_start=None, eval_end=None) -> pd.DataFrame:
    """Release-cycle backtest on ``spec``'s common floor, kept to its selected leads.

    ``eval_start`` / ``eval_end`` override the spec's backtest window (production
    uses a rolling last-N-years window; the research notebooks use the full floor).
    """

    ev0 = spec.backtest_start if eval_start is None else eval_start
    ev1 = spec.backtest_end if eval_end is None else eval_end

    def _bt(db, sd, mods):
        # Capture the harness's raw failure prints and re-emit them as one
        # friendly note (a failed nowcast = the model's own guard returned NaN).
        import contextlib
        import io
        import re

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = run_release_cycle_backtest(
                panel, target=spec.target, models=mods,
                eval_start=ev0, eval_end=ev1,
                days_before=db, days_after=1, step_days=sd, n_jobs=n_jobs, show_progress=False,
            )
        for m in re.findall(r"failed nowcasts by model -> (.+)", buf.getvalue()):
            print(f"      note: some nowcasts unavailable ({m.strip()}); "
                  "counted as missing, the combination renormalizes")
        return out

    leads = set(spec.selected_leads)
    coarse = _bt(days_before, step_days, models)
    frames = [coarse[coarse["days_to_publication"].isin(leads)]]
    off = sorted(l for l in leads if l < 0 and l % step_days != 0)
    if off:
        fast = {k: v for k, v in models.items() if type(v).__name__ not in SLOW_MODELS}
        if fast:
            fine = _bt(-min(off) + 1, 1, fast)
            frames.append(fine[fine["days_to_publication"].isin(off)])
    frames = [f.astype({"y_std": "float64"}, errors="ignore") for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Live sweep (the current, unpublished quarter)
# --------------------------------------------------------------------------- #
def require_exact_origin(origin, as_of, what: str) -> None:
    """The published nowcast's origin must BE the run's as-of date.

    Relabelling an older origin with a newer as-of would claim an information
    set the forecast never saw; consumers refuse instead.
    """
    o, a = pd.Timestamp(origin).normalize(), pd.Timestamp(as_of).normalize()
    if o != a:
        raise ValueError(
            f"{what}: origin {o.date()} does not equal the run as-of "
            f"{a.date()}; the live sweep must include the exact as-of origin "
            "(never relabel an old origin)")


def live_path(panel, spec, models: dict, *, step_days: int = 7,
              days_before: int = 120, min_train: int = 20,
              today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Weekly real-time nowcasts of the next unpublished quarter, up to today.

    Mirrors the backtest harness's per-quarter sweep (masked information sets,
    refit at information changepoints) but targets the quarter the harness skips
    because its realized value does not exist yet. ``y_true`` is NaN.
    """

    today = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today).normalize()
    target = spec.target
    delay = spec.target_delay_days

    last = panel.quarterly[target].dropna().index.max()
    Q = (pd.Period(last, freq="Q") + 1).to_timestamp(how="end").to_period("M").to_timestamp()

    # Extend the quarterly frame with the NaN target row so the engine can mask it.
    p2 = copy.copy(panel)
    p2.quarterly = pd.concat([panel.quarterly,
                              pd.DataFrame({target: [np.nan]}, index=[Q])]).sort_index()

    pub, q_end = publication_date(Q, delay), quarter_end(Q)
    if today > pub:
        # a positive-lead "live" nowcast would be a forecast of a number that
        # should already exist: the target is stale, refuse instead
        raise RuntimeError(
            f"stale target: {target} for {pd.Period(Q, freq='Q')} was due at "
            f"its expected publication date {pub.date()} but is still missing "
            f"from the panel at {today.date()}. Refresh the data (or fix the "
            "release metadata); a live sweep must not run past publication.")
    # the weekly grid gives the historical sweep; the run's EXACT as-of date
    # is always the terminal origin, so the published nowcast's information
    # set is the run's information set (de-duplicated when today is on-grid)
    origins = sorted({o for o in make_daily_origin_grid(pub, days_before=days_before,
                                                        days_after=0, step_days=step_days)
                      if o <= today} | {today})

    engine = RealtimeEngine(p2)
    models = {name: copy.deepcopy(m) for name, m in models.items()}
    rows, cached, last_count = [], {}, None
    for origin in origins:
        info = engine.information_set(origin, target, target_period=Q)
        if info.observed_quarters().size < min_train:
            continue
        count = int(info.monthly.notna().to_numpy().sum() + info.quarterly.notna().to_numpy().sum())
        if count != last_count:
            for name, model in models.items():
                try:
                    res = model.fit(info).nowcast(info)
                    cached[name] = (res.mean, res.std)
                except Exception:
                    cached[name] = (float("nan"), None)
            last_count = count
        for name, (yhat, ystd) in cached.items():
            rows.append({"target": target, "ref_quarter": Q, "origin_date": origin,
                         "days_to_publication": (origin - pub).days,
                         "days_to_quarter_end": (origin - q_end).days,
                         "model": name, "y_true": np.nan, "y_hat": float(yhat),
                         "y_std": None if ystd is None else float(ystd)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Adaptive combination (+ info-index propagation)
# --------------------------------------------------------------------------- #
def adaptive_combine(bt: pd.DataFrame, panel, members: list[str], *,
                     index_col: str = "info_index", n_bins: int = 4, min_train: int = 6,
                     method: str = "inv_mse", name: str = "Adaptive-IC",
                     window_months: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Information index + Adaptive-IC on top of a backtest (or backtest+live) frame.

    Returns ``(rc_all, weights)`` where ``rc_all`` = members + combination, with
    ``info_index`` carried onto the combined rows (``combine_release_cycle`` drops
    it, which would silently collapse the bands to one global bin).
    """

    rc = add_information_index(bt, panel, window_months=window_months)
    members = [m for m in members if m in bt["model"].unique()]
    combo, weights = combine_release_cycle(rc, members, index_col=index_col, n_bins=n_bins,
                                           min_train=min_train, method=method, name=name)
    key = ["ref_quarter", "days_to_publication"]
    combo = combo.drop(columns=index_col, errors="ignore").merge(
        rc.drop_duplicates(key)[key + [index_col]], on=key, how="left")
    return pd.concat([rc, combo], ignore_index=True), weights


# --------------------------------------------------------------------------- #
# Bands with a rolling calibration window
# --------------------------------------------------------------------------- #
def conditional_bands(rc: pd.DataFrame, model: str, *, index_col: str = "info_index",
                      n_bins: int = 4, levels=(0.5, 0.7, 0.9),
                      lookback_years: int | None = None, exclude_years=(),
                      min_quarters: int = 6,
                      collect_pools: bool = False):
    """Real-time empirical bands conditioned on the information index.

    Like ``MIDAS.release_cycle_bands`` but with a **rolling lookback**: at each
    quarter, only residuals from the previous ``lookback_years`` years enter the
    calibration (production uses 10; ``None`` = full history). One residual per
    (past quarter, bin); ``exclude_years`` (e.g. COVID) never enter. With
    ``collect_pools`` also returns ``{(quarter, bin): residuals}`` for density plots.
    """

    d = rc[rc["model"] == model].copy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    d["bin"] = np.clip(np.digitize(d[index_col].to_numpy(), edges[1:-1]), 0, n_bins - 1)
    d["resid"] = d["y_true"] - d["y_hat"]
    perqb = d.groupby(["ref_quarter", "bin"], as_index=False)["resid"].mean()
    perqb["yr"] = pd.to_datetime(perqb["ref_quarter"]).dt.year

    out, pools = [], {}
    for Q in sorted(d["ref_quarter"].unique()):
        lo_cut = pd.Timestamp(Q) - pd.DateOffset(years=lookback_years) if lookback_years else None
        for b, g in d[d["ref_quarter"] == Q].groupby("bin"):
            past = perqb[(perqb["ref_quarter"] < Q) & (perqb["bin"] == b)
                         & (~perqb["yr"].isin(list(exclude_years)))]
            if lo_cut is not None:
                past = past[past["ref_quarter"] >= lo_cut]
            pool = past["resid"].dropna()
            g = g.copy()
            if len(pool) >= min_quarters:
                for lv in levels:
                    a = (1 - lv) / 2
                    g[f"lo_{lv}"] = g["y_hat"] + pool.quantile(a)
                    g[f"hi_{lv}"] = g["y_hat"] + pool.quantile(1 - a)
                g["pit"] = (pool.to_numpy()[None, :] <= g["resid"].to_numpy()[:, None]).mean(axis=1)
                if collect_pools:
                    pools[(pd.Timestamp(Q), int(b))] = pool.to_numpy()
            else:
                for lv in levels:
                    g[f"lo_{lv}"] = np.nan
                    g[f"hi_{lv}"] = np.nan
                g["pit"] = np.nan
            out.append(g)
    bands = pd.concat(out, ignore_index=True)
    return (bands, pools) if collect_pools else bands


# --------------------------------------------------------------------------- #
# Latest nowcast
# --------------------------------------------------------------------------- #
def latest_nowcast(rc_all: pd.DataFrame, bands: pd.DataFrame, adaptive_name: str,
                   levels=(0.5, 0.7, 0.9)) -> pd.DataFrame:
    """Newest nowcast per model (max quarter, max origin) + the adaptive band."""

    q = rc_all["ref_quarter"].max()
    d = rc_all[rc_all["ref_quarter"] == q]
    idx = d.groupby("model")["days_to_publication"].idxmax()
    latest = (d.loc[idx, ["model", "ref_quarter", "origin_date", "days_to_publication",
                          "y_hat", "y_true"]].reset_index(drop=True))
    b = bands[bands["ref_quarter"] == q].dropna(subset=[f"lo_{levels[0]}"])
    if len(b):
        row = b.loc[[b["days_to_publication"].idxmax()]]
        for lv in levels:
            lo, hi = f"lo_{lv}", f"hi_{lv}"
            if lo in row.columns:
                latest.loc[latest["model"] == adaptive_name, lo] = float(row[lo].iloc[0])
                latest.loc[latest["model"] == adaptive_name, hi] = float(row[hi].iloc[0])
    return latest.sort_values("model").reset_index(drop=True)


__all__ = ["run_horse_race", "live_path", "adaptive_combine", "conditional_bands",
           "latest_nowcast", "SLOW_MODELS"]
