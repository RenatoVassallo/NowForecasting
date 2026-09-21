"""Safe hand-off from daily COES data to a *completed-month* GDP nowcast block.

The general NowForecasting engine consumes monthly and quarterly panels only.
This adapter therefore exposes the part of COES that can be represented without
inventing a weekly GDP observation: completed calendar-month electricity growth.
It deliberately does not convert a partial week into a completed monthly value.

For a current-month weekly effect, use the weekly MVP as a separately evaluated
candidate first.  Do not feed its point nowcast into the quarterly model as an
unaccounted-for generated regressor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_ROOT


COES_MONTHLY_COLUMN = "g_coes_yoy"
DEFAULT_COES_DELAY_DAYS = 3


def completed_monthly_coes_block(electricity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a monthly COES growth block using complete calendar months only.

    Returns ``(extra, audit)``.  ``extra`` has the exact ``DatetimeIndex`` and
    one-column shape accepted by ``targets.peru_gdp.load_panel(extra=...)``.
    The annual log difference requires 12 previous complete months.  A missing
    or partial calendar month remains missing, including in its annual change.
    """
    required = {"date", "coes_energy_mwh"}
    missing = required.difference(electricity.columns)
    if missing:
        raise ValueError(f"COES daily data miss {sorted(missing)}")
    d = electricity.loc[:, ["date", "coes_energy_mwh"]].copy()
    d["date"] = pd.to_datetime(d.date, errors="coerce").dt.normalize()
    d["coes_energy_mwh"] = pd.to_numeric(d.coes_energy_mwh, errors="coerce")
    d = d.dropna().drop_duplicates("date", keep="last").sort_values("date")
    if d.empty:
        empty = pd.DataFrame(columns=[COES_MONTHLY_COLUMN], index=pd.DatetimeIndex([], name="date"))
        return empty, pd.DataFrame(columns=["target_month", "observed_days", "calendar_days", "complete_calendar_month"])

    d["target_month"] = d.date.dt.to_period("M").dt.to_timestamp()
    grouped = d.groupby("target_month", as_index=True).agg(
        coes_energy_mwh=("coes_energy_mwh", "sum"),
        observed_days=("date", "nunique"),
    )
    full_index = pd.date_range(grouped.index.min(), grouped.index.max(), freq="MS")
    audit = grouped.reindex(full_index)
    audit.index.name = "target_month"
    audit["calendar_days"] = audit.index.days_in_month
    audit["complete_calendar_month"] = audit.observed_days.eq(audit.calendar_days)
    energy = audit.coes_energy_mwh.where(audit.complete_calendar_month)
    extra = pd.DataFrame({COES_MONTHLY_COLUMN: 100.0 * (np.log(energy) - np.log(energy.shift(12)))},
                         index=full_index)
    extra.index.name = "date"
    return extra, audit.reset_index().rename(columns={"date": "target_month"})


def coes_monthly_metadata(*, delay_days: int = DEFAULT_COES_DELAY_DAYS):
    """Return the explicit scalar-delay metadata required by ``MetadataPanel``.

    The delay is a scenario, not a reconstructed historical first-publication
    calendar.  It starts after the completed reference month.  Production must
    replace it with an observed release calendar or retain the pseudo-real-time
    label.
    """
    if int(delay_days) < 0:
        raise ValueError("delay_days must be non-negative")
    from MIDAS import VariableMeta

    return VariableMeta(column=COES_MONTHLY_COLUMN, frequency="M",
                        group="high-frequency activity",
                        publication_delay_days=int(delay_days))


def load_completed_monthly_coes_block(data_root=DATA_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the cached COES daily artifact and construct the integration block."""
    path = Path(data_root) / "processed" / "coes_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"COES daily cache is missing at {path}; run weekly_activity download first")
    return completed_monthly_coes_block(pd.read_parquet(path))


__all__ = ["COES_MONTHLY_COLUMN", "DEFAULT_COES_DELAY_DAYS",
           "completed_monthly_coes_block", "coes_monthly_metadata",
           "load_completed_monthly_coes_block"]
