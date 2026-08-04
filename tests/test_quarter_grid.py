"""Task 3: one Peru-centered quarter grid for every conditioning path.

Satellite blocks publish on their own grids (a block whose base quarter is
already released starts one quarter later than Peru's). The Peru interface
must resolve every grid quarter as RELEASED data first, block forecast second,
and must refuse to run when a required quarter resolves to nothing. The old
behavior (leading NaN surviving a forward fill and silently freeing a known
condition) is the bug under test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.blocks._common import released_first


def _q(label):
    return pd.Period(label, freq="Q")


GRID = [_q("2026Q2") + k for k in range(4)]


def test_released_fills_before_forecast():
    released = pd.Series({_q("2026Q1"): 2.0, _q("2026Q2"): 1.9})
    forecast = pd.Series({_q("2026Q3"): 1.7, _q("2026Q4"): 2.1, _q("2027Q1"): 2.0})
    vals, mask = released_first(GRID, released, forecast, "us_gdp_yoy_m")
    assert vals == [1.9, 1.7, 2.1, 2.0]
    assert mask == [True, False, False, False]


def test_released_wins_when_both_exist():
    released = pd.Series({_q("2026Q2"): 1.9})
    forecast = pd.Series({_q("2026Q2"): 5.0, _q("2026Q3"): 1.7,
                          _q("2026Q4"): 2.1, _q("2027Q1"): 2.0})
    vals, mask = released_first(GRID, released, forecast, "x")
    assert vals[0] == 1.9 and mask[0] is True


def test_unresolved_required_quarter_raises_with_names():
    released = pd.Series(dtype=float)
    forecast = pd.Series({_q("2026Q3"): 1.7, _q("2026Q4"): 2.1, _q("2027Q1"): 2.0})
    with pytest.raises(ValueError, match=r"us_gdp_yoy_m.*2026Q2"):
        released_first(GRID, released, forecast, "us_gdp_yoy_m")


def test_interior_gap_is_an_error_not_a_fill():
    released = pd.Series({_q("2026Q2"): 1.9})
    forecast = pd.Series({_q("2026Q4"): 2.1, _q("2027Q1"): 2.0})   # 2026Q3 missing
    with pytest.raises(ValueError, match="2026Q3"):
        released_first(GRID, released, forecast, "x")


def test_optional_variable_returns_nan_without_raising():
    released = pd.Series(dtype=float)
    forecast = pd.Series({_q("2026Q3"): 1.7})
    vals, mask = released_first(GRID, released, forecast, "x", required=False)
    assert np.isnan(vals[0]) and vals[1] == 1.7
    assert np.isnan(vals[2]) and np.isnan(vals[3])


def test_nan_in_sources_is_treated_as_missing():
    released = pd.Series({_q("2026Q2"): np.nan})
    forecast = pd.Series({_q("2026Q2"): 3.0, _q("2026Q3"): 1.7,
                          _q("2026Q4"): 2.1, _q("2027Q1"): 2.0})
    vals, mask = released_first(GRID, released, forecast, "x")
    assert vals[0] == 3.0 and mask[0] is False
