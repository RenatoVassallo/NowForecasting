"""Shared helpers for the production blocks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
COV = (0.30, 0.60, 0.90)
CONTRACT = ("quarter", "h", "source", "centre", "mode", "s", "gamma",
            "sigma_left", "sigma_right")


def arith_to_log_yoy(centre, s=None):
    """Map arithmetic YoY growth (percent) into log YoY growth (percent).

    The Peru model's ``g_tdi`` regressor is 100 times the twelve-month LOG
    difference of the official terms-of-trade index, while the commodity block
    publishes the ARITHMETIC growth of the same index (the BCRP-comparable
    headline). The centre converts exactly, ``100*log1p(x/100)``; a scale
    converts to first order by the delta method, ``s / (1 + x/100)``. The
    asymmetry parameter gamma is a shape ratio and passes through unchanged.
    """
    c = np.asarray(centre, dtype=float)
    finite = c[np.isfinite(c)]
    if finite.size and (finite <= -100.0).any():
        raise ValueError("arithmetic YoY at or below -100 percent cannot be "
                         "log-transformed; the units contract is broken")
    lc = 100.0 * np.log1p(c / 100.0)
    if s is None:
        return lc
    return lc, np.asarray(s, dtype=float) / (1.0 + c / 100.0)


def released_first(grid, released: pd.Series, forecast: pd.Series, name: str,
                   required: bool = True) -> tuple[list, list]:
    """Resolve one conditioning path on the Peru-centered quarter grid.

    Each satellite publishes on its OWN grid: a block whose base quarter is
    already released starts one quarter after Peru's grid does. For every grid
    quarter the rule is data first, model second: a released observation wins
    over a forecast, a forecast fills genuinely future quarters, and a required
    quarter that resolves to neither raises immediately (the silent
    alternative, a NaN custom condition, FREES a condition the model was
    supposed to receive, which is how a released US quarter went missing).

    Returns ``(values, released_mask)``; the mask lets the caller zero out
    path uncertainty for quarters that are data rather than forecast.
    """
    vals, mask, missing = [], [], []
    for p in grid:
        r = float(released.get(p, np.nan)) if p in getattr(released, "index", []) else np.nan
        f = float(forecast.get(p, np.nan)) if p in getattr(forecast, "index", []) else np.nan
        if np.isfinite(r):
            vals.append(r); mask.append(True)
        elif np.isfinite(f):
            vals.append(f); mask.append(False)
        else:
            vals.append(np.nan); mask.append(False); missing.append(str(p))
    if required and missing:
        raise ValueError(
            f"{name}: no released value and no block forecast for {missing}; "
            "refusing to condition on NaN (upstream grid or vintage problem)")
    return vals, mask


def fill_structural_january(obj, columns=("ip_cum_yoy",)):
    """Model-boundary imputation for the NBS combined January-February release.

    Panels keep January structurally missing (correct release timing lives on
    the February cell). CONSUMERS that need a regular monthly lattice (row-wise
    balanced VARs, complete-quarter aggregates) fill January from the February
    value of the SAME (possibly release-masked) series: the cell stays missing
    until February actually releases, so timing never leaks.
    """
    if isinstance(obj, pd.Series):
        out = obj.copy()
        months = pd.DatetimeIndex(out.index).month
        jan = (months == 1) & out.isna()
        out.loc[jan] = out.shift(-1)[jan]
        return out
    out = obj.copy()
    months = pd.DatetimeIndex(out.index).month
    for c in columns:
        if c in out.columns:
            s_ = out[c]
            jan = (months == 1) & s_.isna()
            out.loc[jan, c] = s_.shift(-1)[jan]
    return out


def complete_quarters(monthly: pd.Series):
    """(complete-quarter averages, partial-quarter averages, months in the last)."""
    s = monthly.dropna()
    per = pd.PeriodIndex(s.index, freq="Q")
    g = s.groupby(per)
    avg, cnt = g.mean(), g.count()
    last = per.max()
    return avg.where(cnt >= 3).dropna(), avg.where(cnt < 3).dropna(), int(cnt.get(last, 0))


def fan_frame(periods, centre, fits, source, extra=None) -> pd.DataFrame:
    """Assemble the published contract from per-horizon two-piece-normal fits."""
    from forecast.fan_mc import tpn_shortest_bands

    centre = np.asarray(centre, dtype=float)
    mode = np.array([c + (f.get("mode_shift", 0.0) if f else 0.0)
                     for c, f in zip(centre, fits)])
    sl = np.array([f["sigma_left"] if f else np.nan for f in fits])
    sr = np.array([f["sigma_right"] if f else np.nan for f in fits])
    lo, hi = tpn_shortest_bands(mode, sl, sr, COV)
    out = pd.DataFrame({
        "quarter": [str(p) for p in periods], "h": range(1, len(periods) + 1),
        "source": source if isinstance(source, list) else [source] * len(periods),
        "centre": centre.round(3), "mode": mode.round(3),
        "s": [round(f["s"], 4) if f else np.nan for f in fits],
        "gamma": [round(f["gamma"], 4) if f else np.nan for f in fits],
        "sigma_left": sl.round(4), "sigma_right": sr.round(4)})
    for i, c in enumerate(COV):
        out[f"lo{int(c*100)}"] = lo[:, i].round(3)
        out[f"hi{int(c*100)}"] = hi[:, i].round(3)
    out["width90"] = (out.hi90 - out.lo90).round(3)
    if extra:
        for k, v in extra.items():
            out[k] = v
    return out


def information_stamp(target_spec, current_q, panel=None, as_of=None) -> dict:
    """How far through the release cycle the run sits, for the published fan.

    A fan's first node is only narrow when the current quarter is nearly
    complete; right after a release it is no better informed than the second.
    Stamping the state means a reader never has to guess which one they hold.
    ``as_of`` is the run's canonical date (see pipeline.lib.context); it is
    required in production so every block stamps the same vintage.
    """
    from pipeline.lib.context import resolve_as_of

    as_of = resolve_as_of(None) if as_of is None else pd.Timestamp(as_of).normalize()
    pub = current_q.to_timestamp(how="end") + pd.Timedelta(days=target_spec.target_delay_days)
    days = int((as_of - pub).days)
    return {"as_of": str(as_of.date()), "current_quarter": str(current_q),
            "days_to_publication": days}


def released_last(series_q: pd.Series, delay_days: int, as_of) -> pd.Period:
    """Last quarter of a quarterly series RELEASED at ``as_of``.

    The snapshot may already contain later prints (final-vintage data); the
    base quarter is set by the release rule, never by whatever the file holds.
    """
    s = series_q.dropna()
    per = pd.PeriodIndex(s.index, freq="Q")
    rel = per.to_timestamp(how="end") + pd.Timedelta(days=int(delay_days))
    ok = per[rel <= pd.Timestamp(as_of).normalize()]
    if len(ok) == 0:
        raise RuntimeError(f"no released observation at {pd.Timestamp(as_of).date()} "
                           f"under a {delay_days}-day rule")
    return ok.max()


def _dest(out_dir, fname: str) -> Path:
    """Run-local artifact path. Blocks never write the global surface: pass
    ``store.dir('blocks')`` in production or an explicit directory ad hoc;
    publication to products/ happens only after promotion (pipeline.lib.publish)."""
    if out_dir is None:
        raise ValueError(
            f"out_dir is required for {fname}: blocks write inside the run "
            "directory only (see pipeline.lib.publish for the published surface)")
    return Path(out_dir) / fname


def write(df: pd.DataFrame, path: Path, stamp: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if stamp:
        for k, v in stamp.items():
            df[k] = v
    df.to_csv(path, index=False)
    return path
