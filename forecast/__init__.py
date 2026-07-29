"""Forecast stage (h = 1..8 quarters): workflow shared by notebooks and pipeline.

The h >= 1 counterpart to the nowcast stage. Model *classes* live in ``MIDAS``
(AR/RW/DirectARX/DFM are all driven at any horizon by ``run_horizon_backtest``
via ``info.target_period``); this package holds only the workflow:

* :func:`run_backtest` - the quarterly-origin pseudo-real-time horizon backtest
  on a target spec (thin wrapper over ``MIDAS.run_horizon_backtest``).
* :func:`combine_by_horizon` - real-time inverse-MSE (or equal) combination per
  horizon; at base quarter b and horizon h only errors from base quarters
  b' <= b - h are used (the h-step error of base b' is not known before then).
* :func:`horizon_scoreboard` - matched-sample RMSE by horizon + relatives.
* :func:`live_forecast` - the current forecast path: at today's origin, project
  h = 1..H beyond the last released quarter (the product row of every run).
* :func:`fan` - band offsets per horizon from the model's own real-time errors
  (wrapper over ``MIDAS.horizon_bands``; rolling lookback, COVID excluded).
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from MIDAS import horizon_bands, run_horizon_backtest
from MIDAS.realtime import RealtimeEngine

DEFAULT_HORIZONS = tuple(range(1, 9))


def _qstamp(p: pd.Period) -> pd.Timestamp:
    return p.to_timestamp(how="end").to_period("M").to_timestamp()


# --------------------------------------------------------------------------- #
# Backtest + combination + scoring
# --------------------------------------------------------------------------- #
def run_backtest(panel, spec, models: dict, *, horizons=DEFAULT_HORIZONS,
                 eval_start=None, lookback_years: int | None = None, verbose=False) -> pd.DataFrame:
    """Quarterly-origin horizon backtest on ``spec``'s floor.

    ``lookback_years`` (production) trims the eval window to the rolling last-N
    years; ``eval_start`` overrides explicitly (notebooks use the full floor).
    """

    if eval_start is None:
        eval_start = spec.backtest_start
        if lookback_years is not None:
            eval_start = max(pd.Timestamp(spec.backtest_start),
                             pd.Timestamp.now().normalize() - pd.DateOffset(years=lookback_years))
    return run_horizon_backtest(panel, spec.target, models,
                                eval_start=eval_start, horizons=tuple(horizons), verbose=verbose)


def combine_by_horizon(bt: pd.DataFrame, members: list[str], *, method: str = "inv_mse",
                       min_train: int = 8, name: str = "Combo") -> pd.DataFrame:
    """Append a real-time combination row per (base_quarter, horizon).

    Weights for base quarter b at horizon h come from squared errors of past
    base quarters b' <= b - h (whose h-step outcome had been published by b's
    origin). ``method``: "inv_mse" or "equal". Returns ``bt`` + combined rows.
    """

    members = [m for m in members if m in set(bt["model"])]
    d = bt[bt["model"].isin(members)].copy()
    piv = d.pivot_table(index=["base_quarter", "horizon"], columns="model", values="y_hat")
    truth = d.groupby(["base_quarter", "horizon"])["y_true"].first()
    meta = d.groupby(["base_quarter", "horizon"])[["ref_quarter", "origin_date"]].first()

    rows = []
    bases = sorted(d["base_quarter"].unique())
    order = {b: i for i, b in enumerate(bases)}
    for (b, h), preds in piv.iterrows():
        cutoff = order[b] - int(h)
        past = [bb for bb in bases if order[bb] <= cutoff]
        w = pd.Series(1.0, index=members)
        if method == "inv_mse" and len(past) >= min_train:
            errs = {}
            for m in members:
                e = [(piv.at[(bb, h), m] - truth.get((bb, h), np.nan)) for bb in past
                     if (bb, h) in piv.index]
                e = pd.Series(e).dropna()
                errs[m] = float((e ** 2).mean()) if len(e) >= min_train else np.nan
            inv = pd.Series({m: 1.0 / v if v and np.isfinite(v) and v > 0 else np.nan
                             for m, v in errs.items()})
            if inv.notna().any():
                w = inv
        w = w.where(preds.notna() & w.notna())
        if not w.notna().any():
            continue
        w = w / w.sum()
        rows.append({"target": bt["target"].iloc[0], "base_quarter": b, "horizon": h,
                     "ref_quarter": meta.at[(b, h), "ref_quarter"],
                     "origin_date": meta.at[(b, h), "origin_date"], "model": name,
                     "y_true": truth.get((b, h), np.nan),
                     "y_hat": float((preds * w).sum()), "y_std": None})
    return pd.concat([bt, pd.DataFrame(rows)], ignore_index=True)


def horizon_scoreboard(bt: pd.DataFrame, *, benchmark: str = "AR(1)",
                       start=None, end=None, exclude_years=()) -> pd.DataFrame:
    """Matched-sample RMSE by model x horizon, plus ``rel_<benchmark>`` columns."""

    d = bt.copy()
    bq = pd.to_datetime(d["base_quarter"])
    if start is not None:
        d = d[bq >= pd.Timestamp(start)]
    if end is not None:
        d = d[pd.to_datetime(d["base_quarter"]) <= pd.Timestamp(end)]
    if exclude_years:
        d = d[~pd.to_datetime(d["ref_quarter"]).dt.year.isin(list(exclude_years))]
    piv = d.pivot_table(index=["base_quarter", "horizon"], columns="model", values="y_hat")
    truth = d.groupby(["base_quarter", "horizon"])["y_true"].first()
    common = piv.notna().all(axis=1) & truth.notna()
    rmse = {}
    for m in piv.columns:
        e = (piv.loc[common, m] - truth[common])
        rmse[m] = e.groupby(level="horizon").apply(lambda s: float(np.sqrt((s ** 2).mean())))
    out = pd.DataFrame(rmse)
    if benchmark in out.columns:
        rel = out.div(out[benchmark], axis=0)
        rel.columns = [f"rel_{c}" for c in rel.columns]
        out = pd.concat([out, rel[[c for c in rel.columns if c != f"rel_{benchmark}"]]], axis=1)
    out.index.name = "horizon"
    return out


# --------------------------------------------------------------------------- #
# The live forecast path
# --------------------------------------------------------------------------- #
def live_forecast(panel, spec, models: dict, *, horizons=DEFAULT_HORIZONS,
                  today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Forecast h = 1..H beyond the last released quarter, at today's origin.

    Extends the quarterly frame with NaN rows (the engine masks them) and asks
    each model for every ``target_period`` - the same mechanics as the nowcast
    stage's ``live_path``, at quarterly horizons.
    """

    today = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today)
    target = spec.target
    y = panel.quarterly[target].dropna()
    base = pd.Period(y.index.max(), freq="Q")

    p2 = copy.copy(panel)
    ext = pd.DataFrame({target: np.nan},
                       index=[_qstamp(base + h) for h in range(1, max(horizons) + 1)])
    p2.quarterly = pd.concat([panel.quarterly, ext]).sort_index()

    engine = RealtimeEngine(p2)
    models = {k: copy.deepcopy(m) for k, m in models.items()}
    rows = []
    for h in horizons:
        ref = _qstamp(base + h)
        info = engine.information_set(today, target, target_period=ref)
        for mname, model in models.items():
            try:
                res = model.fit(info).nowcast(info)
                yhat, ystd = float(res.mean), res.std
            except Exception:
                yhat, ystd = float("nan"), None
            rows.append({"target": target, "base_quarter": _qstamp(base), "horizon": int(h),
                         "ref_quarter": ref, "origin_date": today, "model": mname,
                         "y_true": np.nan, "y_hat": yhat,
                         "y_std": None if ystd is None else float(ystd)})
    return pd.DataFrame(rows)


def fan(bt: pd.DataFrame, model: str, *, level: float = 0.90,
        lookback_years: int = 10, exclude_years=(2020, 2021),
        as_of: pd.Timestamp | None = None, delay_days: int = 52) -> pd.DataFrame:
    """Band offsets per horizon from the model's own real-time errors.

    ``exclude_years`` is applied to **both ends** of every forecast: the
    reference quarter (inside ``horizon_bands``) AND the base quarter (here).
    For h >= 1 a forecast can be launched from a COVID-distorted base (China's
    2021 base-effect prints reached +14% YoY) and land its reference in a normal
    year - ref-only exclusion lets those catastrophic errors (up to -13pp)
    contaminate the band pool.

    The raw per-horizon bootstrap quantiles are noisy with a few dozen errors per
    horizon, so the offsets are forced monotone in h (a fan can only widen):
    ``lo`` cumulative-min, ``hi`` cumulative-max over increasing horizon.
    """

    as_of = pd.Timestamp.now().normalize() if as_of is None else pd.Timestamp(as_of)
    d = bt[~pd.to_datetime(bt["base_quarter"]).dt.year.isin(list(exclude_years))]
    b = horizon_bands(d, model, as_of=as_of, by="horizon",
                      lookback_years=lookback_years, exclude_years=exclude_years,
                      level=level, delay_days=delay_days).sort_index()
    b["lo"] = b["lo"].cummin()
    b["hi"] = b["hi"].cummax()
    return b


__all__ = ["DEFAULT_HORIZONS", "run_backtest", "combine_by_horizon",
           "horizon_scoreboard", "live_forecast", "fan"]
