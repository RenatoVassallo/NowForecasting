"""Shared helpers for the production blocks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
COV = (0.30, 0.60, 0.90)
CONTRACT = ("quarter", "h", "source", "centre", "mode", "s", "gamma",
            "sigma_left", "sigma_right")


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


def information_stamp(target_spec, current_q, panel=None) -> dict:
    """How far through the release cycle the run sits, for the published fan.

    A fan's first node is only narrow when the current quarter is nearly
    complete; right after a release it is no better informed than the second.
    Stamping the state means a reader never has to guess which one they hold.
    """
    pub = current_q.to_timestamp(how="end") + pd.Timedelta(days=target_spec.target_delay_days)
    days = int((pd.Timestamp.now().normalize() - pub).days)
    return {"as_of": str(pd.Timestamp.now().date()), "current_quarter": str(current_q),
            "days_to_publication": days}


def write(df: pd.DataFrame, path: Path, stamp: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if stamp:
        for k, v in stamp.items():
            df[k] = v
    df.to_csv(path, index=False)
    return path
