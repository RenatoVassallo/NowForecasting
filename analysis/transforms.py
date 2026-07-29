"""Step 1 of the pipeline: cleaning and stationarity transformations.

Every application declares, per series, which transformation takes it to a
comparable stationary growth space. The engine then applies them uniformly, so
Peru (BCRP levels, INEI chained indices) and China (NBS growth rates, several of
them *cumulative*) end up on the same footing before any modelling.

Supported transforms
--------------------
``none``          already stationary (PMI diffusion indices, series published as YoY)
``yoy``           year-on-year log growth of a level, in percent
``mom``/``mom_ann``  month-on-month log growth (optionally annualised)
``qoq_ann``       quarter-on-quarter annualised log growth of a quarterly level
``decum_yoy``     de-cumulate a year-to-date (cumulative) YoY rate into a
                  current-month YoY rate. Chinese NBS publishes several series
                  only as year-to-date growth; using them raw injects a moving
                  average that badly blurs turning points.

The de-cumulation identity
--------------------------
If ``C_m`` is the year-to-date YoY growth through month ``m`` and the base-year
monthly shares are roughly stable, the implied current-month growth is

    g_m ~= m * C_m - (m - 1) * C_{m-1}

with ``g_1 = C_1``. This is the standard approximation for Chinese cumulative
series. It is exact when last year's monthly levels within the year are equal
and a good approximation otherwise; the error grows when monthly seasonality is
extreme, so the result is winsorised by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VALID_TRANSFORMS = ("none", "yoy", "mom", "mom_ann", "qoq_ann", "decum_yoy", "diff")


# --------------------------------------------------------------------------- #
# Growth transforms
# --------------------------------------------------------------------------- #
def yoy(level: pd.Series, periods: int = 12) -> pd.Series:
    """Year-on-year log growth in percent (``periods`` = 12 monthly, 4 quarterly)."""

    s = pd.to_numeric(level, errors="coerce")
    s = s.where(s > 0)
    return 100.0 * (np.log(s) - np.log(s.shift(periods)))


def mom(level: pd.Series, *, annualise: bool = False, periods_per_year: int = 12) -> pd.Series:
    """Month-on-month log growth in percent, optionally annualised."""

    s = pd.to_numeric(level, errors="coerce")
    s = s.where(s > 0)
    g = 100.0 * (np.log(s) - np.log(s.shift(1)))
    return g * periods_per_year if annualise else g


def qoq_ann(level: pd.Series) -> pd.Series:
    """Quarter-on-quarter annualised log growth in percent."""

    return mom(level, annualise=True, periods_per_year=4)


def decumulate_ytd_yoy(cum: pd.Series, *, clip: float | None = 60.0) -> pd.Series:
    """Turn a year-to-date (cumulative) YoY rate into a current-month YoY rate.

    See the module docstring for the identity. ``clip`` winsorises the result to
    +/- ``clip`` percent, because the approximation is unstable in the first
    months of a year when the cumulative base is small (set ``None`` to disable).
    """

    s = pd.to_numeric(cum, errors="coerce")
    if not isinstance(s.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise TypeError("decumulate_ytd_yoy needs a datetime-like index")
    month = pd.Index(s.index).month.to_numpy()
    prev = s.shift(1)
    out = month * s - (month - 1) * prev
    # January (and any month whose predecessor is missing) is already the
    # year-to-date figure for a single month.
    out = out.where(month != 1, s)
    out = out.where(prev.notna() | (month == 1))
    if clip is not None:
        out = out.clip(-clip, clip)
    return out


def apply_transform(series: pd.Series, kind: str, *, periods_per_year: int = 12) -> pd.Series:
    """Apply one named transform (see :data:`VALID_TRANSFORMS`)."""

    k = (kind or "none").strip().lower()
    if k in ("none", "level", ""):
        return pd.to_numeric(series, errors="coerce")
    if k == "yoy":
        return yoy(series, periods=periods_per_year)
    if k == "mom":
        return mom(series, annualise=False, periods_per_year=periods_per_year)
    if k == "mom_ann":
        return mom(series, annualise=True, periods_per_year=periods_per_year)
    if k == "qoq_ann":
        return qoq_ann(series)
    if k == "decum_yoy":
        return decumulate_ytd_yoy(series)
    if k == "diff":
        return pd.to_numeric(series, errors="coerce").diff()
    raise ValueError(f"unknown transform {kind!r}; expected one of {VALID_TRANSFORMS}")


def transform_frame(frame: pd.DataFrame, spec: dict[str, str], *, periods_per_year: int = 12) -> pd.DataFrame:
    """Apply a per-column transform spec to a wide frame."""

    out = {}
    for col in frame.columns:
        out[col] = apply_transform(frame[col], spec.get(col, "none"), periods_per_year=periods_per_year)
    return pd.DataFrame(out, index=frame.index)


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
def suggest_transform(name: str, series: pd.Series | None = None) -> str:
    """Heuristic first guess at the right transform for a series.

    Naming conventions carry most of the information (``*_cum_yoy`` is
    cumulative, ``*_yoy`` is already a growth rate, ``*_pmi`` is a diffusion
    index). When a series is supplied, a positive, trending, high-mean series is
    treated as a level needing ``yoy``.
    """

    n = str(name).lower()
    if "cum_yoy" in n or "cum" in n and "yoy" in n:
        return "decum_yoy"
    if n.endswith("_yoy") or "_yoy" in n or "var_" in n:
        return "none"
    if "pmi" in n or "confidence" in n or "index_sa" in n:
        return "none"
    if series is not None:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if not s.empty and (s > 0).all() and s.mean() > 10 and s.std() / max(abs(s.mean()), 1e-9) > 0.1:
            return "yoy"
    return "none"


def fill_single_month_gap(frame: pd.DataFrame, month: int = 1) -> pd.DataFrame:
    """Fill a structurally missing month (China: NBS merges January into February).

    The missing month is filled with the adjacent published figure so the
    monthly index stays regular. Kept explicit rather than a generic ffill,
    because it is a known publication convention, not a data error.
    """

    out = frame.copy()
    idx_month = pd.Index(out.index).month
    for col in out.columns:
        s = out[col]
        gap = (idx_month == month) & s.isna()
        if gap.any():
            out.loc[gap, col] = s.bfill()[gap]
    return out


def winsorize(frame: pd.DataFrame, *, lower: float = 0.005, upper: float = 0.995) -> pd.DataFrame:
    """Clip extreme values column-wise (guards the factor/ML steps)."""

    out = frame.copy()
    for col in out.columns:
        s = out[col]
        lo, hi = s.quantile(lower), s.quantile(upper)
        out[col] = s.clip(lo, hi)
    return out


__all__ = [
    "VALID_TRANSFORMS",
    "yoy",
    "mom",
    "qoq_ann",
    "decumulate_ytd_yoy",
    "apply_transform",
    "transform_frame",
    "suggest_transform",
    "fill_single_month_gap",
    "winsorize",
]
