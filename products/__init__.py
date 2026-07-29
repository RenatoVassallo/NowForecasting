"""Products layer: turn model output into publishable objects (STUB).

The assembly stage between the modelling engine and the eventual website / API.
``builders`` turns backtest and forecast frames into the headline nowcast table,
the 8-quarter fan chart and the scenario table; ``trend_cycle`` derives the
output-gap product. Each builder returns a tidy, publication-ready frame so the
reporting layer can consume every app's output uniformly.

Kept deliberately thin: plotting stays in :mod:`analysis.plots`; this layer only
composes the objects a reader consumes.
"""

from .builders import fan_chart, gdp_nowcast, output_gap, scenario
from .trend_cycle import TrendCycle, hp_filter, uc_output_gap

__all__ = [
    "fan_chart",
    "gdp_nowcast",
    "output_gap",
    "scenario",
    "TrendCycle",
    "hp_filter",
    "uc_output_gap",
]
