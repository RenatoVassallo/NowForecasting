"""F10: historical as-of selections cannot see future-dated information.

These tests use deliberately FUTURE-CONTAMINATED synthetic inputs: the
snapshot contains observations whose release dates fall after the as-of, and
every selection rule must ignore them. This proves release-rule masking over
final-vintage values (the pseudo_real_time_final_vintage regime), never true
historical vintages.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.blocks._common import released_last


def _q(stamp):
    return pd.Timestamp(stamp)


def test_released_last_ignores_future_release():
    # 2026Q2 ends Jun 30 and releases ~Aug 21 under the 52-day rule; before
    # that date the base must still be 2026Q1 even though the snapshot
    # already carries the Q2 print
    s = pd.Series([3.1, 2.8, 2.5],
                  index=[_q("2025-12-01"), _q("2026-03-01"), _q("2026-06-01")])
    assert str(released_last(s, 52, "2026-08-04")) == "2026Q1"
    assert str(released_last(s, 52, "2026-08-25")) == "2026Q2"
    with pytest.raises(RuntimeError, match="no released"):
        released_last(s, 52, "2020-01-01")


def test_peru_expectations_path_uses_released_observation_only():
    from pipeline.blocks.peru import _released_expectations

    idx = pd.date_range("2026-01-01", periods=7, freq="MS")
    exp = pd.Series([52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 99.0], index=idx)
    # July (released ~Aug 15 under the 15-day rule) must be invisible Aug 4
    assert _released_expectations(exp, pd.Timestamp("2026-08-04")) == 57.0
    assert _released_expectations(exp, pd.Timestamp("2026-08-20")) == 99.0


def test_us_spf_selection_respects_the_survey_release_rule():
    from pipeline.blocks.usa import _released_spf

    spf = pd.DataFrame(
        {"spf_gdp_h0": [2.0, 2.1], "spf_gdp_h1": [2.2, 2.3]},
        index=pd.PeriodIndex(["2026Q2", "2026Q3"], freq="Q"))
    # the Q3 survey (first month July, released ~mid-August under the 45-day
    # rule) must be invisible on August 4; the Q2 survey is the newest usable
    assert _released_spf(spf, pd.Timestamp("2026-08-04")).index.max() == \
        pd.Period("2026Q2", freq="Q")
    assert _released_spf(spf, pd.Timestamp("2026-09-20")).index.max() == \
        pd.Period("2026Q3", freq="Q")


def test_availability_events_after_as_of_are_invisible():
    from pipeline.lib.data_availability import load_events

    ev = pd.DataFrame({
        "internal_code": ["x", "x"],
        "attempted_at": ["2026-08-01T10:00:00", "2026-09-01T10:00:00"],
        "status": ["successfully_updated", "ingestion_failure"],
        "detail": ["", "future event"]})
    p = None
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "events.jsonl"
        p.write_text("\n".join(ev.iloc[[i]].to_json(orient="records", lines=True).strip()
                               for i in range(len(ev))))
        got = load_events(p, as_of=pd.Timestamp("2026-08-04"))
    assert len(got) == 1
    assert got.status.iloc[0] == "successfully_updated", (
        "an event recorded AFTER the as-of drove a historical dashboard")
