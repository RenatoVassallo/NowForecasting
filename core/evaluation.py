"""Evaluation metrics for point and density forecasts.

One module, one vocabulary. Everything here is regime-labelled: the current
backtests use FINAL-VINTAGE values with scalar release lags, which is a
pseudo-real-time exercise, never "genuine real time"; every scoreboard carries
``EVALUATION_REGIME`` so the label cannot fall off in a table.

Selection versus evaluation: all modelling choices made up to 2026-08-03 saw
the full sample (research selection, documented in the audit). The holdout
below is therefore FROZEN FORWARD: results on ``holdout`` rows are honest for
every change made after this date, and nothing after ``SELECTION_END`` may be
used to tune future choices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EVALUATION_REGIME = "pseudo_real_time_final_vintage"
SELECTION_END = pd.Period("2022Q4", freq="Q")       # tuning may look at <= this
HOLDOUT_START = pd.Period("2023Q1", freq="Q")       # frozen forward on 2026-08-03


def sample_label(ref_quarter) -> str:
    q = pd.Period(pd.Timestamp(ref_quarter), freq="Q")
    return "holdout" if q >= HOLDOUT_START else "selection"


# --------------------------------------------------------------------------- #
# point metrics
# --------------------------------------------------------------------------- #
def rmse(e) -> float:
    e = np.asarray(e, dtype=float)
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan


def mae(e) -> float:
    e = np.asarray(e, dtype=float)
    e = e[np.isfinite(e)]
    return float(np.mean(np.abs(e))) if e.size else np.nan


def bias(e) -> float:
    e = np.asarray(e, dtype=float)
    e = e[np.isfinite(e)]
    return float(np.mean(e)) if e.size else np.nan


def finite_share(y_hat) -> float:
    y = np.asarray(y_hat, dtype=float)
    return float(np.mean(np.isfinite(y))) if y.size else np.nan


def directional_accuracy(y_hat, y_true, y_base) -> float:
    """Share of forecasts whose SIGN OF CHANGE from the jump-off is right."""
    h, t, b = (np.asarray(v, dtype=float) for v in (y_hat, y_true, y_base))
    ok = np.isfinite(h) & np.isfinite(t) & np.isfinite(b)
    if not ok.any():
        return np.nan
    return float(np.mean(np.sign(h[ok] - b[ok]) == np.sign(t[ok] - b[ok])))


def revision_size(frame: pd.DataFrame, *, value="y_hat", quarter="ref_quarter",
                  origin="origin_date") -> float:
    """Mean absolute successive revision of the same quarter's forecast."""
    out = []
    for _, g in frame.sort_values(origin).groupby(quarter):
        v = pd.to_numeric(g[value], errors="coerce").dropna()
        if len(v) > 1:
            out.extend(np.abs(np.diff(v)))
    return float(np.mean(out)) if out else np.nan


# --------------------------------------------------------------------------- #
# density metrics (two-piece normal parameterisation)
# --------------------------------------------------------------------------- #
def tpn_pdf(x, mode, sl, sr):
    x = np.asarray(x, dtype=float)
    c = 2.0 / (np.sqrt(2.0 * np.pi) * (sl + sr))
    left = c * np.exp(-0.5 * ((x - mode) / sl) ** 2)
    right = c * np.exp(-0.5 * ((x - mode) / sr) ** 2)
    return np.where(x < mode, left, right)


def log_score(y, mode, sl, sr) -> float:
    d = tpn_pdf(np.asarray([y], dtype=float), float(mode), float(sl), float(sr))[0]
    return float(np.log(d)) if d > 0 else -np.inf


def pit(y, mode, sl, sr) -> float:
    from forecast.fan_mc import tpn_cdf

    return float(tpn_cdf(np.asarray([y], dtype=float), float(mode),
                         float(sl), float(sr))[0])


def coverage(y, lo, hi) -> float:
    y, lo, hi = (np.asarray(v, dtype=float) for v in (y, lo, hi))
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok]))) if ok.any() else np.nan


def coverage_ci(k: int, n: int, level: float = 0.90) -> tuple[float, float]:
    """Binomial (Wilson) interval for an empirical coverage rate."""
    from scipy import stats

    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(centre - half), float(centre + half))


def interval_score(y, lo, hi, cov_level: float) -> float:
    """Winkler interval score for one central interval of coverage ``cov_level``."""
    alpha = 1.0 - cov_level
    y, lo, hi = float(y), float(lo), float(hi)
    s = (hi - lo)
    if y < lo:
        s += (2.0 / alpha) * (lo - y)
    if y > hi:
        s += (2.0 / alpha) * (y - hi)
    return float(s)


def weighted_interval_score(y, mode, intervals: dict) -> float:
    """WIS (Bracher et al. 2021) over central intervals {coverage: (lo, hi)}."""
    K = len(intervals)
    total = 0.5 * abs(float(y) - float(mode))
    for cov_level, (lo, hi) in intervals.items():
        alpha = 1.0 - float(cov_level)
        total += (alpha / 2.0) * interval_score(y, lo, hi, float(cov_level))
    return float(total / (K + 0.5))


# --------------------------------------------------------------------------- #
# equal-accuracy tests
# --------------------------------------------------------------------------- #
def dm(e1, e2, h: int = 1):
    """Diebold-Mariano with Harvey correction (squared loss); from MIDAS."""
    from MIDAS import dm_test

    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    ok = np.isfinite(e1) & np.isfinite(e2)
    if ok.sum() < 10:
        return (np.nan, np.nan)
    return dm_test(e1[ok], e2[ok], h=h)


def cw(y_true, f_restricted, f_unrestricted, h: int = 1):
    """Clark-West for NESTED comparisons (restricted inside unrestricted)."""
    from MIDAS import cw_test

    y, fr, fu = (np.asarray(v, dtype=float) for v in
                 (y_true, f_restricted, f_unrestricted))
    ok = np.isfinite(y) & np.isfinite(fr) & np.isfinite(fu)
    if ok.sum() < 10:
        return (np.nan, np.nan)
    return cw_test(y[ok], fr[ok], fu[ok], h=h)


# --------------------------------------------------------------------------- #
# the scoreboard
# --------------------------------------------------------------------------- #
def scoreboard(df: pd.DataFrame, group_cols: list[str], *,
               y_true="y_true", y_hat="y_hat", y_base=None) -> pd.DataFrame:
    """Point-metric table by any grouping; always carries n, finite share and
    the evaluation regime, so mixed-sample comparisons cannot pass silently."""
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        e = pd.to_numeric(g[y_true], errors="coerce") - \
            pd.to_numeric(g[y_hat], errors="coerce")
        row = dict(zip(group_cols, keys))
        row.update(n=int(len(g)), finite_share=round(finite_share(g[y_hat]), 3),
                   rmse=round(rmse(e), 4), mae=round(mae(e), 4),
                   bias=round(bias(e), 4), regime=EVALUATION_REGIME)
        if y_base is not None and y_base in g:
            row["directional_accuracy"] = round(
                directional_accuracy(g[y_hat], g[y_true], g[y_base]), 3)
        rows.append(row)
    return pd.DataFrame(rows)
