"""Task 14: exact-chain harness primitives and the no-lookahead guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.lib.exact_chain import (BENCH_MEMBERS, ORIGIN_DAY, default_bases,
                                      no_lookahead_checks, origin_for,
                                      released_by_rule)


def test_origin_sits_at_day_thirty_of_the_cycle():
    t = origin_for(pd.Period("2026Q1", freq="Q"))
    # current quarter 2026Q2 ends June 30; day 30 of the cycle is July 30
    assert t == pd.Timestamp("2026-07-30")
    assert ORIGIN_DAY == 30


def test_released_by_rule_boundary():
    s = pd.Series([1.0, 2.0], index=pd.PeriodIndex(["2026Q1", "2026Q2"], freq="Q"))
    got = released_by_rule(s, 52, "2026-08-03")
    assert list(got.index.astype(str)) == ["2026Q1"]          # Q2 releases Aug 21
    got2 = released_by_rule(s, 52, "2026-08-22")
    assert list(got2.index.astype(str)) == ["2026Q1", "2026Q2"]


def test_no_lookahead_guard_raises_on_every_violation():
    t = pd.Timestamp("2024-07-30")
    ok = dict(ladder_max_origin=pd.Timestamp("2024-07-29"),
              spf_release=pd.Timestamp("2024-05-15"), weo_round="2024-04",
              frame_releases={"peru_gdp": pd.Timestamp("2024-05-22")})
    checks = no_lookahead_checks(t, **ok)
    assert checks["weo_round"] == "2024-04" and "ladder_max_origin" in checks

    with pytest.raises(AssertionError, match="ladder"):
        no_lookahead_checks(t, **{**ok, "ladder_max_origin": pd.Timestamp("2024-08-05")})
    with pytest.raises(AssertionError, match="SPF"):
        no_lookahead_checks(t, **{**ok, "spf_release": pd.Timestamp("2024-08-15")})
    with pytest.raises(AssertionError, match="WEO"):
        no_lookahead_checks(t, **{**ok, "weo_round": "2024-10"})
    with pytest.raises(AssertionError, match="peru_gdp"):
        no_lookahead_checks(t, **{**ok,
                                  "frame_releases": {"peru_gdp": pd.Timestamp("2024-09-01")}})


def test_default_bases_and_members():
    bases = default_bases()
    assert bases[0] == pd.Period("2019Q1", freq="Q")
    assert bases[-1] == pd.Period("2025Q3", freq="Q")
    assert BENCH_MEMBERS == ("S1-chain", "RW", "AR(2)", "BVAR-unconditional")
