"""Application configuration for the general nowcasting framework.

A :class:`NowcastApp` is a declarative description of one nowcasting problem:
the target, its frequency, the preprocessing spec, an optional high-frequency
proxy, the release-cycle lead grid, the evaluation subperiods, and the crisis
window used for crisis-in-estimation treatment.

The object holds data only. It does not load panels or run models, so it stays
free of project-specific imports and can be reused across applications (Peru
quarterly GDP, Peru monthly GDP, later China GDP and other monthly targets).
The concrete Peru instances live at the bottom of this module; a new
application is one more instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class Subperiod:
    """An evaluation window, optionally dropping whole years (e.g. COVID)."""

    label: str
    start: str
    end: str
    exclude_years: tuple[int, ...] = ()

    def mask(self, quarters: Sequence[pd.Timestamp] | pd.Series) -> pd.Series:
        """Boolean mask selecting reference periods inside this subperiod."""

        ts = pd.to_datetime(pd.Series(list(quarters)) if not isinstance(quarters, pd.Series) else quarters)
        keep = (ts >= pd.Timestamp(self.start)) & (ts <= pd.Timestamp(self.end))
        if self.exclude_years:
            keep &= ~ts.dt.year.isin(list(self.exclude_years))
        return keep.reset_index(drop=True) if not isinstance(quarters, pd.Series) else keep


@dataclass(frozen=True)
class NowcastApp:
    """One nowcasting application, described declaratively."""

    name: str
    target: str
    target_freq: str  # "Q" or "M"
    target_label: str
    spec: str = "spec3"
    monthly_proxy: str | None = None
    proxy_label: str | None = None
    survey_group: str | None = None
    eval_start: str = "2005-01-01"
    leads: tuple[int, ...] = tuple(range(-120, 1))
    selected_leads: tuple[int, ...] = (-120, -105, -90, -75, -60, -45, -30, -15, 0)
    subperiods: tuple[Subperiod, ...] = ()
    crisis_window: tuple[str, str] | None = None
    crisis_quarters: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.target_freq not in ("Q", "M"):
            raise ValueError(f"target_freq must be 'Q' or 'M', got {self.target_freq!r}")

    @property
    def primary_subperiod(self) -> Subperiod:
        """The first subperiod; for Peru this is 2005-2019 (run first)."""

        if not self.subperiods:
            raise ValueError(f"app {self.name!r} has no subperiods defined")
        return self.subperiods[0]

    def subperiod(self, label: str) -> Subperiod:
        for sp in self.subperiods:
            if sp.label == label:
                return sp
        raise KeyError(f"no subperiod {label!r} in app {self.name!r}")


# The three Peru evaluation windows. 2005-2019 is run first; the crisis window
# only bites on the later two, so it is deferred until we extend past 2019.
_PERU_SUBPERIODS = (
    Subperiod("2005-2019", "2005-01-01", "2019-12-31"),
    Subperiod("2005-2026 ex-COVID", "2005-01-01", "2026-12-31", exclude_years=(2020, 2021)),
    Subperiod("2022-2026", "2022-01-01", "2026-12-31"),
)

# Peru release cycle: quarterly GDP is published about 120 days into the
# window we trace, so the lead grid runs -120 to 0. The selected leads sweep
# every 15 days, which is where the horse race is scored first.
_PERU_SELECTED_LEADS = (-120, -105, -90, -75, -60, -45, -30, -15, 0)
_PERU_FULL_LEADS = tuple(range(-120, 1))

# Crisis-in-estimation window (unified target for every model once we cross
# into the post-COVID subperiods). Stored in both the month-window and
# quarter-tuple forms the current estimators expect.
_PERU_CRISIS_WINDOW = ("2020-01", "2021-12")
_PERU_CRISIS_QUARTERS = (
    "2020Q1", "2020Q2", "2020Q3", "2020Q4",
    "2021Q1", "2021Q2", "2021Q3", "2021Q4",
)


PERU_GDP_Q = NowcastApp(
    name="peru_gdp_q",
    target="g_pbiq",
    target_freq="Q",
    target_label="Real GDP, YoY %",
    spec="spec3",
    monthly_proxy="g_pbim",
    proxy_label="Monthly GDP index",
    survey_group="Business surveys",
    eval_start="2005-01-01",
    leads=_PERU_FULL_LEADS,
    selected_leads=_PERU_SELECTED_LEADS,
    subperiods=_PERU_SUBPERIODS,
    crisis_window=_PERU_CRISIS_WINDOW,
    crisis_quarters=_PERU_CRISIS_QUARTERS,
)


__all__ = ["Subperiod", "NowcastApp", "PERU_GDP_Q"]
