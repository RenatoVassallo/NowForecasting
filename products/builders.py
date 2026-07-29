"""Product builders (STUB).

Turn model output into the monthly publications in the diagram: the GDP growth
nowcast, the 8-quarter fan chart, the trend-cycle / output-gap product, and the
scenario builder. Each function takes fitted results / backtest frames and
returns a tidy, publication-ready table plus (optionally) a figure, so the
website / API layer can consume them uniformly.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


def gdp_nowcast(backtest: pd.DataFrame, *, model: str, app: Any) -> pd.DataFrame:
    """Current-quarter nowcast (early / mid / late) for publication."""
    raise NotImplementedError("Assemble the headline nowcast table")


def fan_chart(forecasts: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    """8-quarter point forecast + prediction-interval bands (fan chart data)."""
    raise NotImplementedError("Assemble the fan-chart table from forecasts + bands")


def output_gap(trend_cycle: Any) -> pd.DataFrame:
    """Potential GDP and output-gap series for publication."""
    raise NotImplementedError("Assemble the output-gap table")


def scenario(app: Any, assumptions: Mapping[str, float]) -> pd.DataFrame:
    """Re-run the engine under user assumptions on exogenous inputs.

    Exogenous inputs are the satellite outputs (China growth, commodity prices,
    US growth / Fed path) and domestic policy variables; returns the implied GDP,
    components, and output-gap paths.
    """
    raise NotImplementedError("Wire exogenous assumptions through the engine")
