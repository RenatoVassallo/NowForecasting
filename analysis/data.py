"""Step 1 helpers: assemble a real-time panel from raw frames + a spec.

Target-agnostic. An application supplies its raw monthly/quarterly frames and a
per-series spec (transforms, publication delays, groups); this returns the
transformed frames and a ``MIDAS.MetadataPanel`` with delays attached. The
app-specific part (which files, which columns) stays in the app's loader.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from MIDAS import MetadataPanel, VariableMeta

from .transforms import fill_single_month_gap, transform_frame


def metadata_map(delays: dict, groups: dict, target: str, target_delay: int,
                 *, target_group: str = "activity", target_label: str = "target",
                 labels: dict | None = None) -> pd.DataFrame:
    """Build the metadata frame the information tools expect, from config dicts."""

    labels = labels or {}
    rows = [{"column": c, "label": labels.get(c, c.replace("_", " ")),
             "group": groups.get(c, "other"), "frequency": "M", "publication_delay_days": d}
            for c, d in delays.items()]
    rows.append({"column": target, "label": labels.get(target, target_label),
                 "group": target_group, "frequency": "Q", "publication_delay_days": target_delay})
    return pd.DataFrame(rows)


def assemble_panel(monthly_raw: pd.DataFrame, quarterly_raw: pd.DataFrame, *,
                   transforms: dict, delays: dict, groups: dict, target: str,
                   target_delay: int, target_group: str = "activity",
                   fill_jan_gap: bool = False, periods_per_year: int = 12):
    """Transform the raw monthly panel and return (monthly, quarterly, MetadataPanel).

    ``fill_jan_gap`` fills a structurally missing January (NBS convention) before
    transforming. Only the columns listed in ``transforms`` are kept.
    """

    cols = [c for c in transforms if c in monthly_raw.columns]
    monthly = monthly_raw[cols]
    if fill_jan_gap:
        monthly = fill_single_month_gap(monthly, month=1)
    monthly = transform_frame(monthly, transforms, periods_per_year=periods_per_year)
    monthly = monthly.replace([np.inf, -np.inf], np.nan)

    quarterly = quarterly_raw[[target]].copy()

    metas = [VariableMeta(column=c, frequency="M", group=groups.get(c, "other"),
                          publication_delay_days=int(delays[c]))
             for c in monthly.columns if c in delays]
    metas.append(VariableMeta(column=target, frequency="Q", group=target_group,
                              publication_delay_days=int(target_delay)))
    panel = MetadataPanel.from_frames(monthly, quarterly, metas)
    return monthly, quarterly, panel


def series_summary(monthly: pd.DataFrame, *, transforms: dict, delays: dict, groups: dict,
                   start=None, end=None) -> pd.DataFrame:
    """Per-series transform / delay / group / coverage / volatility on a window."""

    win = monthly.loc[start:end] if (start or end) else monthly
    return pd.DataFrame({
        "transform": pd.Series(transforms),
        "delay_days": pd.Series(delays),
        "group": pd.Series(groups),
        "missing_share": win.isna().mean().round(3),
        "std": win.std().round(2),
    }).dropna(subset=["missing_share"]).sort_values("delay_days")


__all__ = ["metadata_map", "assemble_panel", "series_summary"]
