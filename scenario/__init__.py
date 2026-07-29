"""Scenario stage: conditional forecasts (STUB).

The conditional counterpart to :mod:`forecast`. Given an assumed path for a set
of conditioning variables (or a shock), produce the target's conditional
forecast and its band. Builds on the same :mod:`core` engine and ``NowcastApp``
config; the conditional-forecast machinery itself is a ``MIDAS`` primitive, so
this package holds only the workflow: define the conditioning set, run the
conditional path, and hand the result to :mod:`products` for the scenario table.

Interface mirrors the other stages: ``build_scenarios`` (declare the conditioning
paths) then ``run_scenarios`` (produce the conditional forecasts).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd


def build_scenarios(app, definitions: Mapping[str, Mapping[str, Sequence[float]]]) -> dict:
    """Materialise scenario definitions into conditioning paths.

    ``definitions`` maps a scenario name to {conditioning_variable: path}. Returns
    the runnable conditioning objects the engine consumes. Mirrors
    ``utils.models.build_models`` in the nowcast/forecast stages.
    """
    raise NotImplementedError("Materialise conditioning paths from config.SCENARIOS")


def run_scenarios(panel: pd.DataFrame, scenarios: dict, *, target: str, horizons: Sequence[int]) -> pd.DataFrame:
    """Conditional forecasts of ``target`` under each scenario's conditioning path.

    Returns a tidy frame indexed by (scenario, horizon) with the conditional
    forecast and interval columns, ready for :func:`products.scenario`.
    """
    raise NotImplementedError("Run conditional forecasts given the conditioning paths")


__all__ = ["build_scenarios", "run_scenarios"]
