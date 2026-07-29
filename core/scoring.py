"""Scoring a release-cycle horse race (target-agnostic).

Works on the standard backtest frame produced by
``MIDAS.run_release_cycle_backtest``: columns ``model``, ``ref_quarter``,
``days_to_publication``, ``y_true``, ``y_hat``. Reused by every app.

Subperiods: pass ``start`` / ``end`` / ``exclude_years`` to score on any window
(e.g. 2010-2019, 2010-2026 excluding 2020-2021, 2022-2026) from a single
full-sample backtest.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def _rmse(errors) -> float:
    e = pd.Series(errors).dropna().to_numpy()
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan


def filter_period(bt: pd.DataFrame, *, start=None, end=None, exclude_years: Sequence[int] = ()) -> pd.DataFrame:
    """Restrict a backtest frame to a scoring window (by ``ref_quarter``)."""

    d = bt.copy()
    rq = pd.to_datetime(d["ref_quarter"])
    if start is not None:
        d = d[rq >= pd.Timestamp(start)]
    if end is not None:
        d = d[pd.to_datetime(d["ref_quarter"]) <= pd.Timestamp(end)]
    if exclude_years:
        d = d[~pd.to_datetime(d["ref_quarter"]).dt.year.isin(list(exclude_years))]
    return d


def rmse_by_lead(bt: pd.DataFrame, *, start=None, end=None, exclude_years: Sequence[int] = ()) -> pd.DataFrame:
    """RMSE per model x days_to_publication (the honest per-lead comparison)."""

    d = filter_period(bt, start=start, end=end, exclude_years=exclude_years)
    return (d.assign(e=d["y_hat"] - d["y_true"])
              .groupby(["model", "days_to_publication"])["e"].apply(_rmse).unstack())


def horse_race_summary(bt: pd.DataFrame, *, baseline: str = "RW",
                       start=None, end=None, exclude_years: Sequence[int] = ()) -> pd.DataFrame:
    """Per-model scoreboard on a scoring window.

    Columns: ``rmse`` (own finite rows), ``rmse_common`` (sample every fitting
    model shares), ``rel_<baseline>``, ``finite`` (coverage), ``fits`` (False if
    the model is identical to the baseline on every row, i.e. silently falling
    back instead of estimating).
    """

    bt = filter_period(bt, start=start, end=end, exclude_years=exclude_years)
    err = bt.assign(e=bt["y_hat"] - bt["y_true"])
    overall = err.groupby("model")["e"].apply(_rmse).rename("rmse")
    finite = bt.assign(ok=bt["y_hat"].notna()).groupby("model")["ok"].mean().rename("finite")

    keys = ["ref_quarter", "days_to_publication"]
    piv = bt.pivot_table(index=keys, columns="model", values="y_hat")
    yt = bt.pivot_table(index=keys, columns="model", values="y_true")
    base = piv[baseline] if baseline in piv.columns else None
    fits = {m: True if (m == baseline or base is None)
               else (not np.isclose(piv[m], base, equal_nan=True).all())
            for m in piv.columns}

    fit_models = [m for m, ok in fits.items() if ok]
    common_mask = piv[fit_models].notna().all(axis=1) if fit_models else piv.notna().all(axis=1)
    common = pd.Series(
        {m: _rmse(piv.loc[common_mask, m] - yt.loc[common_mask, m]) for m in piv.columns},
        name="rmse_common",
    )

    out = pd.concat([overall, common, finite], axis=1)
    out["rel_" + baseline] = out["rmse_common"] / out["rmse_common"].get(baseline, np.nan)
    out["fits"] = pd.Series(fits)
    return out.sort_values("rmse_common")[["rmse", "rmse_common", "rel_" + baseline, "finite", "fits"]]


def subperiod_curves(bt: pd.DataFrame, subperiods: Mapping[str, tuple]) -> dict[str, pd.DataFrame]:
    """``{label: rmse_by_lead}`` for each subperiod ``label -> (start, end, exclude_years)``."""

    out = {}
    for label, spec in subperiods.items():
        start, end, *rest = spec
        excl = rest[0] if rest else ()
        out[label] = rmse_by_lead(bt, start=start, end=end, exclude_years=excl)
    return out


def subperiod_summary(bt: pd.DataFrame, subperiods: Mapping[str, tuple], *, baseline: str = "RW") -> dict[str, pd.DataFrame]:
    """``{label: horse_race_summary}`` for each subperiod."""

    out = {}
    for label, spec in subperiods.items():
        start, end, *rest = spec
        excl = rest[0] if rest else ()
        out[label] = horse_race_summary(bt, baseline=baseline, start=start, end=end, exclude_years=excl)
    return out


__all__ = [
    "filter_period", "rmse_by_lead", "horse_race_summary",
    "subperiod_curves", "subperiod_summary",
]
