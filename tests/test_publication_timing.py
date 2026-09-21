"""H5 (P1): ONE canonical publication-date convention everywhere.

`PeriodIndex.to_timestamp(how="end")` lands on 23:59:59.999999999; adding a
delay and flooring `.days` then disagrees by one day with the normalized
MIDAS convention used by `live_path`. Run 2026-08-04__final published
days-to-publication -16 (official) and -17 (fan and PDF) for the same
quarter. The canonical rule: publication = NORMALIZED quarter end + delay;
for Peru 2026Q2 (delay 51) that is 2026-08-20 and the as-of 2026-08-04 sits
16 days before it, on every surface.

Documented distinctions kept: `fan_calibration.PERU_DELAY = 52` is a
deliberately BUFFERED knowable-before constant (one day beyond the 51-day
publication metadata), and `day_in_cycle` measures days since the quarter
end on the same canonical clock (its previous value undercounted by one day
through the same flooring).
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.blocks._common import expected_publication, information_stamp


class _Spec:
    target_delay_days = 51


def test_canonical_publication_date_is_normalized():
    pub = expected_publication(pd.Period("2026Q2", freq="Q"), 51)
    assert pub == pd.Timestamp("2026-08-20")
    assert pub == pub.normalize(), "no nanosecond quarter-end residue"
    # month-stamp input (the panel convention) gives the identical date
    assert expected_publication(pd.Timestamp("2026-06-01"), 51) == pub


def test_information_stamp_matches_the_live_path_convention():
    stamp = information_stamp(_Spec, pd.Period("2026Q2", freq="Q"),
                              as_of="2026-08-04")
    assert stamp["days_to_publication"] == -16, (
        "the fan stamp must agree with the official artifact (-16), not "
        "floor to -17 through a nanosecond quarter end")


def test_quarter_boundary_and_on_publication_day():
    q = pd.Period("2026Q2", freq="Q")
    on_day = information_stamp(_Spec, q, as_of="2026-08-20")
    assert on_day["days_to_publication"] == 0
    q4 = pd.Period("2025Q4", freq="Q")               # year boundary
    assert expected_publication(q4, 51) == pd.Timestamp("2026-02-20")


def test_released_last_includes_the_quarter_on_its_publication_day():
    from pipeline.blocks._common import released_last

    s = pd.Series([2.5], index=[pd.Timestamp("2026-06-01")])
    assert str(released_last(s, 51, "2026-08-20")) == "2026Q2"
    with pytest.raises(RuntimeError):
        released_last(s, 51, "2026-08-19")


def test_stale_target_exactly_one_day_after_publication():
    from tests.test_live_asof import _Spec as LiveSpec  # noqa: F401
    from tests.test_live_asof import _sweep

    live = _sweep("2026-08-21")                      # fixture delay 52: pub day
    assert live.origin_date.max() == pd.Timestamp("2026-08-21")
    with pytest.raises(RuntimeError, match="expected publication"):
        _sweep("2026-08-22")                          # one day past: stale


def test_knowable_buffer_is_distinct_and_documented():
    from pipeline.lib import fan_calibration as fc
    from targets.peru_gdp import TARGET_DELAY_DAYS

    assert TARGET_DELAY_DAYS == 51
    assert fc.PERU_DELAY == 52, (
        "the knowable-before rule keeps its deliberate one-day buffer over "
        "the publication metadata; if this changes, recalibrate deliberately")
