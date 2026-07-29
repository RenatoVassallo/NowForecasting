"""Trend-cycle and structural layer (STUB).

Decompose a level series into trend (potential) and cycle (gap): HP / BK
filters and unobserved-components / state-space models, in-sample and
out-of-sample (real-time-consistent). Feeds the diagram's layer D and the
"Trend-Cycle & Output Gap" product.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TrendCycle:
    """A trend/cycle decomposition of one series."""

    trend: pd.Series      # potential level
    cycle: pd.Series      # gap (level minus trend, or % of trend)


def hp_filter(level: pd.Series, lamb: float = 1600.0) -> TrendCycle:
    """Hodrick-Prescott decomposition (default quarterly lambda)."""
    raise NotImplementedError("Wrap statsmodels.tsa.filters.hp_filter")


def uc_output_gap(level: pd.Series, **kwargs) -> TrendCycle:
    """Unobserved-components / state-space potential GDP and output gap."""
    raise NotImplementedError("Fit a UC model (statsmodels UnobservedComponents)")
