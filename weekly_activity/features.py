"""Calendar-week features built only from observations at each cut-off."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd


def cutoff_schedule(month) -> pd.DataFrame:
    """All Sunday closes in a month plus its calendar month-end.

    A month ending Sunday has two labelled rows with the same cut-off.  This is
    intentional: one serves the weekly product and the other the month-end
    evaluation cell.  Values are never interpolated between them.
    """
    p = pd.Period(month, freq="M")
    start = p.start_time.normalize()
    end = p.end_time.normalize()
    first_sunday = start + pd.Timedelta(days=(6 - start.dayofweek) % 7)
    sundays = pd.date_range(first_sunday, end, freq="W-SUN")
    rows = [{"target_month": start, "stage": f"week_{i}", "cutoff_date": d}
            for i, d in enumerate(sundays, start=1)]
    rows.append({"target_month": start, "stage": "month_end", "cutoff_date": end})
    return pd.DataFrame(rows).sort_values(["cutoff_date", "stage"]).reset_index(drop=True)


def _as_daily(frame: pd.DataFrame | None, *, required: Iterable[str]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", *required])
    out = frame.copy()
    if "date" not in out:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index(names="date")
        else:
            raise ValueError("daily source needs a 'date' column or DatetimeIndex")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"]).sort_values("date")
    missing = set(required).difference(out.columns)
    for col in missing:
        out[col] = np.nan
    return out.drop_duplicates("date", keep="last")


def _trend_name(topic) -> str:
    return "trend_" + re.sub(r"[^0-9A-Za-z_]+", "_", str(topic)).strip("_").lower()


def _one_feature_row(month, stage: str, cutoff, electricity: pd.DataFrame,
                     lbtr: pd.DataFrame, trends: pd.DataFrame | None) -> dict:
    p = pd.Period(month, freq="M")
    start, end = p.start_time.normalize(), p.end_time.normalize()
    cutoff = min(pd.Timestamp(cutoff).normalize(), end)
    dates = pd.date_range(start, cutoff, freq="D")
    e = electricity[(electricity.date >= start) & (electricity.date <= cutoff)].copy()
    e = e[pd.to_numeric(e.coes_energy_mwh, errors="coerce").notna()]
    l = lbtr[(lbtr.date >= start) & (lbtr.date <= cutoff)].copy()
    lbtr_cols = ["lbtr_client_value_mn", "lbtr_client_value_me",
                 "lbtr_client_count_mn", "lbtr_client_count_me"]
    l = l.dropna(subset=lbtr_cols, how="all")
    row = {
        "target_month": start, "stage": stage, "cutoff_date": cutoff,
        "month_number": p.month,
        "month_sin": np.sin(2 * np.pi * p.month / 12.0),
        "month_cos": np.cos(2 * np.pi * p.month / 12.0),
        "calendar_days_elapsed": len(dates),
        "elec_observed_days": int(e.date.nunique()),
        "elec_elapsed_days": len(dates),
        "elec_calendar_coverage": float(e.date.nunique() / len(dates)) if len(dates) else np.nan,
        "elec_energy_mwh_mtd": float(e.coes_energy_mwh.sum(min_count=1)),
        "elec_log_mean_mwh": (float(np.log(e.coes_energy_mwh.sum() / len(e)))
                               if len(e) and e.coes_energy_mwh.sum() > 0 else np.nan),
        "elec_last_observation": e.date.max() if len(e) else pd.NaT,
        "lbtr_observed_days": int(l.date.nunique()),
        "lbtr_last_observation": l.date.max() if len(l) else pd.NaT,
    }
    for dow in range(7):
        row[f"elec_observed_dow_{dow}"] = int((e.date.dt.dayofweek == dow).sum()) if len(e) else 0
    for source, feature in (("lbtr_client_value_mn", "value_mn"),
                            ("lbtr_client_value_me", "value_me"),
                            ("lbtr_client_count_mn", "count_mn"),
                            ("lbtr_client_count_me", "count_me")):
        x = pd.to_numeric(l[source], errors="coerce").dropna()
        row[f"lbtr_{feature}_mtd"] = float(x.sum()) if len(x) else np.nan
        row[f"lbtr_log_mean_{feature}"] = (float(np.log1p(x.sum() / len(x)))
                                             if len(x) and x.sum() >= 0 else np.nan)

    if trends is not None and not trends.empty:
        t = trends.copy()
        t["week_end"] = pd.to_datetime(t["week_end"], errors="coerce").dt.normalize()
        t["week_start"] = t.week_end - pd.Timedelta(days=6)
        t = t[(t.week_start >= start) & (t.week_end <= cutoff)]
        row["trend_complete_weeks"] = int(t.week_end.nunique())
        for topic, grp in t.groupby("topic_id"):
            x = pd.to_numeric(grp.value, errors="coerce").dropna()
            row[_trend_name(topic)] = float(x.mean()) if len(x) else np.nan
    return row


def build_feature_frame(
    electricity: pd.DataFrame | None,
    lbtr: pd.DataFrame | None,
    *,
    months: Iterable,
    stages: tuple[str, ...] | None = None,
    trends: pd.DataFrame | None = None,
    include_trends: bool = True,
) -> pd.DataFrame:
    """Build monthly-nowcast features at each Sunday cut-off.

    Every slice is explicitly bounded by ``cutoff_date``.  Monthly source data
    are never expanded to a weekly frequency.  Missing observations remain
    missing and are exposed through coverage columns rather than filled.
    """
    electricity = _as_daily(electricity, required=("coes_energy_mwh",))
    lbtr = _as_daily(lbtr, required=("lbtr_client_value_mn", "lbtr_client_value_me",
                                     "lbtr_client_count_mn", "lbtr_client_count_me"))
    use_trends = trends if include_trends else None
    rows = []
    for month in pd.PeriodIndex(list(months), freq="M").unique().sort_values():
        schedule = cutoff_schedule(month)
        if stages is not None:
            schedule = schedule[schedule.stage.isin(stages)]
        for r in schedule.itertuples(index=False):
            rows.append(_one_feature_row(month, r.stage, r.cutoff_date,
                                         electricity, lbtr, use_trends))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["target_month", "cutoff_date", "stage"]).reset_index(drop=True)


def feature_dictionary(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """A compact machine-readable feature dictionary generated with the data."""
    rows = [
        ("target_month", "key", "Month whose GDP growth is nowcast"),
        ("cutoff_date", "key", "Sunday close or calendar month-end"),
        ("stage", "key", "week_1, week_2, ..., or month_end"),
        ("elec_*", "COES", "Observed daily executed electricity only, aggregated to month-to-date"),
        ("lbtr_*", "BCRP LBTR", "Client payment-system activity, not a consumption measure"),
        ("trend_*", "Google Trends", "Frozen weekly topic index, only complete weeks"),
    ]
    out = pd.DataFrame(rows, columns=["variable", "block", "definition"])
    if frame is not None:
        out["present_in_current_frame"] = out.variable.map(
            lambda x: any(c.startswith(x[:-1]) if x.endswith("*") else c == x
                          for c in frame.columns))
    return out
