"""Stage 1: information-environment analysis (target-agnostic, reusable).

Before choosing any model, characterise the information environment a
nowcaster faces for a given target and panel:

1. **Availability in time** - publication delays, each variable's
   delay-advantage over the target, and where each variable's within-period
   observations land in the release cycle.
2. **Proxy detection** - whether a high-frequency proxy of the target (e.g. a
   monthly GDP index for quarterly GDP) already carries most of the signal.
3. **Leading indicators** - dynamic (lead-lag) correlations that identify which
   variables lead the target, at what horizon.

The functions take plain frames plus a few column names, so they work for any
target (Peru quarterly GDP, Peru monthly GDP, China GDP, other monthly
targets). The DSAPM ``NowcastApp`` config bundles the arguments per
application. This module is intentionally free of Peru specifics so it can be
exported into a standalone workflow repo.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from MIDAS import align_monthly_to_quarter


def _as_period(obj, freq: str):
    """Return a copy with the index normalised to a PeriodIndex.

    Quarterly frames arrive with mixed conventions across the pipeline (quarter
    start vs quarter end). Collapsing to a PeriodIndex makes ``concat`` and
    ``shift`` align on whole periods regardless of the day-of-month convention.
    """

    out = obj.copy()
    out.index = pd.PeriodIndex(pd.to_datetime(out.index), freq=freq)
    return out


# --------------------------------------------------------------------------- #
# 1. Availability in time
# --------------------------------------------------------------------------- #
def availability_table(
    metadata_map: pd.DataFrame,
    *,
    target_col: str,
    column_col: str = "column",
    freq_col: str = "frequency",
    group_col: str = "group",
    delay_col: str = "publication_delay_days",
    label_col: str = "label",
    panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-variable availability relative to the target.

    ``delay_advantage_days`` is how many days earlier than the target a
    variable is known (target delay minus the variable's delay). Positive means
    the variable arrives before the target; the fast surveys will show large
    positive values, a same-delay monthly proxy will show zero.

    If ``panel`` (a wide frame of the variables) is given, columns for the
    observed history (missing share, first/last observation, count) are added.
    """

    meta = metadata_map.copy()
    target_delay = float(meta.loc[meta[column_col] == target_col, delay_col].iloc[0])

    out = pd.DataFrame(
        {
            "column": meta[column_col].to_numpy(),
            "label": meta[label_col].to_numpy(),
            "group": meta[group_col].to_numpy(),
            "frequency": meta[freq_col].to_numpy(),
            "delay_days": meta[delay_col].to_numpy(dtype=float),
        }
    )
    out["delay_advantage_days"] = target_delay - out["delay_days"]
    out["is_target"] = out["column"] == target_col

    if panel is not None:
        stats = []
        for col in out["column"]:
            if col in panel.columns:
                s = panel[col]
                obs = s.dropna()
                stats.append(
                    {
                        "column": col,
                        "n_obs": int(obs.shape[0]),
                        "missing_share": float(s.isna().mean()),
                        "first_obs": obs.index.min() if not obs.empty else pd.NaT,
                        "last_obs": obs.index.max() if not obs.empty else pd.NaT,
                    }
                )
            else:
                stats.append({"column": col, "n_obs": 0, "missing_share": np.nan, "first_obs": pd.NaT, "last_obs": pd.NaT})
        out = out.merge(pd.DataFrame(stats), on="column", how="left")

    return out.sort_values(["delay_days", "group", "column"]).reset_index(drop=True)


def release_calendar(
    metadata_map: pd.DataFrame,
    *,
    target_col: str,
    reference_quarter: str | pd.Timestamp = "2019-12-31",
    column_col: str = "column",
    freq_col: str = "frequency",
    group_col: str = "group",
    delay_col: str = "publication_delay_days",
    label_col: str = "label",
    months_per_period: int = 3,
) -> pd.DataFrame:
    """When each variable's within-quarter observations arrive, in days-to-publication.

    For a representative ``reference_quarter`` (real calendar, so the day counts
    are exact), compute for every monthly variable and each month of the quarter
    the day-to-target-publication at which that month's value is released
    (a negative number: earlier in the cycle is more negative). This is the raw
    material for a release-calendar figure showing how information accumulates.
    """

    meta = metadata_map.copy()
    target_delay = int(meta.loc[meta[column_col] == target_col, delay_col].iloc[0])

    q = pd.Period(pd.Timestamp(reference_quarter), freq="Q")
    quarter_end = q.end_time.normalize()
    target_pub = quarter_end + pd.Timedelta(days=target_delay)

    month_starts = pd.date_range(q.start_time.normalize().replace(day=1), periods=months_per_period, freq="MS")

    rows: list[dict] = []
    monthly = meta[meta[freq_col] == "M"]
    for _, r in monthly.iterrows():
        delay = int(r[delay_col])
        for m_idx, m_start in enumerate(month_starts, start=1):
            month_end = (m_start + pd.offsets.MonthEnd(1)).normalize()
            release = month_end + pd.Timedelta(days=delay)
            rows.append(
                {
                    "column": r[column_col],
                    "label": r[label_col],
                    "group": r[group_col],
                    "delay_days": delay,
                    "month_of_quarter": m_idx,
                    "release_date": release,
                    "days_to_publication": int((release - target_pub).days),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["days_to_publication", "group", "column"]).reset_index(drop=True)


def information_accumulation(
    calendar: pd.DataFrame,
    *,
    leads: Sequence[int],
) -> pd.DataFrame:
    """Count of within-quarter monthly observations available at each lead.

    Uses the output of :func:`release_calendar`. For each lead L (days to the
    target's publication), count how many variable-month cells are already
    released (``days_to_publication <= L``), overall and per group.
    """

    rows: list[dict] = []
    groups = sorted(calendar["group"].unique())
    for lead in sorted(int(x) for x in leads):
        avail = calendar[calendar["days_to_publication"] <= lead]
        row = {"days_to_publication": lead, "n_cells": int(avail.shape[0])}
        for g in groups:
            row[g] = int((avail["group"] == g).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("days_to_publication").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Proxy detection
# --------------------------------------------------------------------------- #
def _ols_r2_rmse(y: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    """R-squared and in-sample RMSE of an OLS fit of y on [1, X]."""

    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(resid**2)))
    return r2, rmse


def _ar_rmse(y: pd.Series, order: int = 1) -> float:
    """In-sample RMSE of an AR(order) fit (own baseline for the target)."""

    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(y) <= order + 2:
        return np.nan
    lags = pd.concat([y.shift(k) for k in range(1, order + 1)], axis=1)
    df = pd.concat([y.rename("y"), lags], axis=1).dropna()
    yv = df["y"].to_numpy()
    Xv = df.drop(columns="y").to_numpy()
    _, rmse = _ols_r2_rmse(yv, Xv)
    return rmse


def proxy_information_diagnostic(
    target: pd.Series,
    proxy_monthly: pd.Series,
    *,
    target_freq: str = "Q",
    agg: str = "mean",
    proxy_yoy_from_level: bool = False,
    ar_order: int = 1,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    exclude_years: Sequence[int] = (),
    months_per_period: int = 3,
) -> pd.DataFrame:
    """How much of the target a high-frequency proxy already explains.

    Aggregates ``proxy_monthly`` to the target frequency using progressively
    more within-period months (1, then 2, then all ``months_per_period``), fits
    the target on each, and reports R-squared, correlation, and the in-sample
    RMSE against an AR baseline. The rising R-squared across ``months_used``
    shows the proxy sharpening as the period fills in.

    If ``proxy_yoy_from_level`` is True the monthly input is treated as a level:
    it is aggregated to the target frequency and then differenced to a
    year-on-year growth rate, so it lines up with a YoY target (the cleanest
    demonstration that the quarterly average of a monthly GDP index essentially
    is quarterly GDP). Otherwise both series are used as given.
    """

    freq = "Q" if target_freq == "Q" else "M"
    periods_per_year = 4 if freq == "Q" else 12
    tgt = _as_period(pd.to_numeric(target, errors="coerce").dropna(), freq)
    ar_rmse = _ar_rmse(pd.Series(tgt.to_numpy(), index=tgt.index.to_timestamp()), order=ar_order)

    rows: list[dict] = []
    end_months = [1, 2, months_per_period] if target_freq == "Q" else [months_per_period]
    for k in dict.fromkeys(end_months):
        if target_freq == "Q":
            proxy_q = align_monthly_to_quarter(proxy_monthly.to_frame("proxy"), method=agg, end_month=k)["proxy"]
        else:
            proxy_q = proxy_monthly.copy()
        proxy_q = _as_period(pd.to_numeric(proxy_q, errors="coerce"), freq)
        if proxy_yoy_from_level:
            proxy_q = 100.0 * (np.log(proxy_q) - np.log(proxy_q.shift(periods_per_year)))

        df = pd.concat([tgt.rename("y"), proxy_q.rename("proxy")], axis=1).dropna()
        if start is not None:
            df = df[df.index >= pd.Period(pd.Timestamp(start), freq=freq)]
        if end is not None:
            df = df[df.index <= pd.Period(pd.Timestamp(end), freq=freq)]
        if exclude_years:
            df = df[~df.index.year.isin(list(exclude_years))]
        if df.shape[0] < 8:
            rows.append({"months_used": k, "n": int(df.shape[0]), "r2": np.nan, "corr": np.nan, "rmse": np.nan, "rmse_ar": ar_rmse, "rmse_ratio": np.nan})
            continue
        yv = df["y"].to_numpy()
        xv = df["proxy"].to_numpy()
        r2, rmse = _ols_r2_rmse(yv, xv.reshape(-1, 1))
        corr = float(np.corrcoef(yv, xv)[0, 1])
        rows.append(
            {
                "months_used": k,
                "n": int(df.shape[0]),
                "r2": r2,
                "corr": corr,
                "rmse": rmse,
                "rmse_ar": ar_rmse,
                "rmse_ratio": rmse / ar_rmse if ar_rmse and np.isfinite(ar_rmse) else np.nan,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Leading indicators: dynamic (lead-lag) cross-correlation
# --------------------------------------------------------------------------- #
def dynamic_cross_correlation(
    target: pd.Series,
    monthly_panel: pd.DataFrame,
    *,
    target_freq: str = "Q",
    max_lag: int = 4,
    agg: str = "mean",
    min_overlap: int = 16,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    exclude_years: Sequence[int] = (),
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Lead-lag correlations between the target and each indicator.

    Returns a wide frame indexed by indicator column, with one column per lag
    in ``[-max_lag, +max_lag]``. Lag ``k`` is the correlation of the target at
    time ``t`` with the indicator at time ``t - k``; **positive k means the
    indicator leads the target by k periods**. Lag 0 is the contemporaneous
    correlation. Correlations are computed on pairwise-complete observations and
    set to NaN when the overlap is below ``min_overlap``.

    For a quarterly target, monthly indicators are aggregated to quarterly with
    ``agg`` before correlating.
    """

    freq = "Q" if target_freq == "Q" else "M"
    tgt = pd.to_numeric(target, errors="coerce")
    cols = list(columns) if columns is not None else list(monthly_panel.columns)

    if target_freq == "Q":
        ind = align_monthly_to_quarter(monthly_panel[cols], method=agg)
    else:
        ind = monthly_panel[cols].copy()

    tgt = _as_period(tgt, freq)
    ind = _as_period(ind, freq)
    aligned = pd.concat([tgt.rename("__target__"), ind], axis=1).sort_index()
    if start is not None:
        aligned = aligned[aligned.index >= pd.Period(pd.Timestamp(start), freq=freq)]
    if end is not None:
        aligned = aligned[aligned.index <= pd.Period(pd.Timestamp(end), freq=freq)]
    if exclude_years:
        aligned = aligned[~aligned.index.year.isin(list(exclude_years))]

    y = aligned["__target__"]
    lags = list(range(-max_lag, max_lag + 1))
    out = pd.DataFrame(index=cols, columns=[f"lag_{k}" for k in lags], dtype=float)
    out.index.name = "column"
    for col in cols:
        x = aligned[col]
        for k in lags:
            pair = pd.concat([y, x.shift(k)], axis=1).dropna()
            if pair.shape[0] >= min_overlap and pair.iloc[:, 1].std(ddof=0) > 0:
                out.loc[col, f"lag_{k}"] = float(np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1])
    return out


def cross_correlation_ranking(
    cc_wide: pd.DataFrame,
    *,
    metadata_map: pd.DataFrame | None = None,
    column_col: str = "column",
    group_col: str = "group",
    label_col: str = "label",
    delay_col: str = "publication_delay_days",
    sort_by: str = "peak",
) -> pd.DataFrame:
    """Rank indicators by correlation strength, reporting the lead at which they peak.

    ``sort_by`` is ``"peak"`` (strongest absolute correlation over all lags) or
    ``"contemporaneous"`` (lag 0). The ``peak_lead`` column is the lag at which
    the strongest correlation occurs (positive means the indicator leads).
    """

    lags = [int(c.split("_")[1]) for c in cc_wide.columns]
    corr0_col = "lag_0"
    rows: list[dict] = []
    for col, r in cc_wide.iterrows():
        vals = r.to_numpy(dtype=float)
        if np.all(~np.isfinite(vals)):
            continue
        abs_vals = np.abs(vals)
        peak_idx = int(np.nanargmax(abs_vals))
        rows.append(
            {
                "column": col,
                "corr_contemporaneous": float(r.get(corr0_col, np.nan)),
                "peak_corr": float(vals[peak_idx]),
                "peak_lead": lags[peak_idx],
                "abs_peak": float(abs_vals[peak_idx]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    if metadata_map is not None:
        meta = metadata_map[[column_col, group_col, label_col, delay_col]].rename(
            columns={column_col: "column", group_col: "group", label_col: "label", delay_col: "delay_days"}
        )
        out = out.merge(meta, on="column", how="left")

    key = "abs_peak" if sort_by == "peak" else "corr_contemporaneous"
    out["_sort"] = out[key].abs() if key == "corr_contemporaneous" else out[key]
    out = out.sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# App-orchestration wrappers (config-driven, target-agnostic)
# --------------------------------------------------------------------------- #
# These combine the primitives above into the three steps a notebook runs. They
# take explicit arguments (a metadata frame, the target, leads, a window) so the
# same calls work for any application. Plotting lives in ``analysis.plots``.
def availability(metadata_map: pd.DataFrame, target: str, *, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-series timeliness (delay + days earlier than the target), excl. the target."""

    av = availability_table(metadata_map, target_col=target, panel=panel)
    return av[av.column != target].sort_values("delay_advantage_days", ascending=False).reset_index(drop=True)


def accumulation(metadata_map: pd.DataFrame, target: str, leads: Sequence[int],
                 *, reference_quarter: str = "2019-12-31") -> pd.DataFrame:
    """How many within-quarter observations are released at each lead."""

    cal = release_calendar(metadata_map, target_col=target, reference_quarter=reference_quarter)
    return information_accumulation(cal, leads=leads)


def dynamic_correlations(target: pd.Series, monthly: pd.DataFrame, *,
                         start=None, end=None, max_lag: int = 4,
                         metadata_map: pd.DataFrame | None = None):
    """Lead-lag correlation matrix + a ranked leading-indicator table.

    ``start``/``end`` set the estimation window (customizable, and typically wider
    than the horse-race floor). Returns ``(cc_wide, ranking)``.
    """

    cc = dynamic_cross_correlation(target, monthly, target_freq="Q", max_lag=max_lag, start=start, end=end)
    rank = cross_correlation_ranking(cc, metadata_map=metadata_map, sort_by="peak")
    return cc, rank


__all__ = [
    # primitives
    "availability_table", "release_calendar", "information_accumulation",
    "proxy_information_diagnostic", "dynamic_cross_correlation", "cross_correlation_ranking",
    # app-orchestration steps
    "availability", "accumulation", "dynamic_correlations",
]
