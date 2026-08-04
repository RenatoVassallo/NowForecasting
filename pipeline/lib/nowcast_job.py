"""The per-target nowcast job: one function that turns a target + its data into
the full product - the historical release-cycle backtest, the **live weekly
nowcast of the current unpublished quarter**, the Adaptive-IC combination,
rolling-window uncertainty bands and the evaluation metrics.

Pure compute - it returns a ``Result``; persistence and figures are the caller's
job (see ``stages`` and ``reporting``). All mechanics come from
``nowcast.release_cycle`` so notebooks and production share one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core import scoring
from nowcast import release_cycle as rcyc

from ..config import metadata


@dataclass
class Result:
    target: object
    monthly: pd.DataFrame
    quarterly: pd.DataFrame
    nowcasts: pd.DataFrame                 # backtest + live rows, all models + Adaptive-IC
    weights: pd.DataFrame
    bands: pd.DataFrame
    pools: dict                            # (quarter, bin) -> residual pool (for densities)
    summary: dict                          # subperiod -> horse_race_summary (incl. rolling last-10y)
    curves: dict
    members: list
    latest: pd.DataFrame                   # newest nowcast per model (live quarter)
    models: dict = field(default_factory=dict)


def _production_subperiods(as_of: pd.Timestamp) -> dict:
    """Production reports ONE window: the rolling last N years (ex-COVID).

    The academic subperiods (2010-2019, ex-COVID, 2022-2026) live in the research
    notebooks; the monthly production report evaluates on the same rolling window
    that calibrates the bands.
    """

    start = as_of - pd.DateOffset(years=metadata.METRICS_LOOKBACK_YEARS)
    return {f"last {metadata.METRICS_LOOKBACK_YEARS}y ex-COVID":
            (start.strftime("%Y-%m-%d"), as_of.strftime("%Y-%m-%d"), (2020, 2021))}


def run(target, models: dict, *, panel=None, n_jobs: int = 8,
        backtest: pd.DataFrame | None = None, as_of=None) -> Result:
    """Nowcast one target end-to-end. ``models`` = instantiated ladder (see modelset).

    ``backtest``: pass a previous run's member nowcasts to SKIP the backtest (the
    fast monthly-update mode, ``params.RUN_BACKTEST = False``); weights and bands
    are then calibrated on that reused history.
    ``as_of``: the run's canonical date (pipeline.lib.context); today when absent.
    """
    from pipeline.lib.context import resolve_as_of

    as_of = resolve_as_of(None) if as_of is None else pd.Timestamp(as_of).normalize()

    if panel is None:
        monthly, quarterly, mpanel = target.load_panel()
    else:
        monthly, quarterly, mpanel = panel

    # Historical backtest over the rolling production window (or a reused one)
    # + the live weekly sweep of the unpublished quarter.
    if backtest is None:
        eval_start = max(pd.Timestamp(target.backtest_start),
                         as_of - pd.DateOffset(years=metadata.METRICS_LOOKBACK_YEARS))
        bt = rcyc.run_horse_race(mpanel, target, models, n_jobs=n_jobs,
                                 eval_start=eval_start, **metadata.BACKTEST)
    else:
        bt = backtest
    live = rcyc.live_path(mpanel, target, models, step_days=metadata.LIVE["step_days"],
                          days_before=metadata.BACKTEST["days_before"], today=as_of)
    full = (pd.concat([f.astype({"y_std": "float64"}, errors="ignore")
                   for f in (bt, live) if len(f)], ignore_index=True)
            if len(live) else bt)

    rc_all, weights = rcyc.adaptive_combine(
        full, mpanel, metadata.ADAPTIVE_MEMBERS[target.name],
        index_col=target.adaptive_index, n_bins=metadata.ADAPTIVE["n_bins"],
        min_train=metadata.ADAPTIVE["min_train"], method=metadata.ADAPTIVE["method"],
        name=metadata.ADAPTIVE["name"], window_months=metadata.ADAPTIVE["window_months"])

    bands, pools = rcyc.conditional_bands(
        rc_all, metadata.ADAPTIVE["name"], index_col=target.adaptive_index,
        n_bins=metadata.ADAPTIVE["n_bins"], levels=metadata.BANDS["levels"],
        lookback_years=metadata.BANDS["lookback_years"],
        exclude_years=metadata.BANDS["exclude_years"],
        min_quarters=metadata.BANDS["min_quarters"], collect_pools=True)

    subs = _production_subperiods(as_of)
    summary = scoring.subperiod_summary(rc_all, subs, baseline=target.baseline)
    curves = scoring.subperiod_curves(rc_all, subs)
    latest = rcyc.latest_nowcast(rc_all, bands, metadata.ADAPTIVE["name"],
                                 levels=metadata.BANDS["levels"])
    members = [m for m in metadata.ADAPTIVE_MEMBERS[target.name] if m in full["model"].unique()]

    return Result(target=target, monthly=monthly, quarterly=quarterly, nowcasts=rc_all,
                  weights=weights, bands=bands, pools=pools, summary=summary,
                  curves=curves, members=members, latest=latest, models=models)
