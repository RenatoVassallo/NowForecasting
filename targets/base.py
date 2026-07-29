"""The per-target data interface shared by the notebooks and the run pipeline.

A ``Target`` bundles everything the pipeline needs to nowcast one series without
knowing anything country-specific: how to load its (vintage) panel, its target
column and publication delay, the release-cycle leads and the scoring subperiods.
Models are deliberately NOT here - production model specs live in
``pipeline/config/metadata.py`` and experimentation lives in each country's editable
``notebooks/<country>/models.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Target:
    name: str                       # registry key, e.g. "china", "peru_gdp"
    role: str                       # "satellite" | "domestic"
    label: str                      # human label for reports
    target: str                     # target column in the quarterly frame
    target_delay_days: int
    monthly_proxy: str | None
    selected_leads: tuple
    subperiods: dict                # label -> (start, end, exclude_years)
    backtest_start: str
    backtest_end: str
    load_panel: Callable            # () -> (monthly_df, quarterly_df, MetadataPanel)
    metadata_map: Callable          # () -> DataFrame(column,label,group,frequency,delay)
    baseline: str = "RW"
    adaptive_index: str = "info_index"
