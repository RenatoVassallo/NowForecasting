"""G1 (P0): the live sweep's terminal origin IS the run's as-of date.

`live_path` built a weekly grid anchored on the expected publication date and
filtered it to origins no later than today, so an as-of between grid points
produced an official nowcast from a STALE origin (run 2026-08-04__140712:
as-of Aug 4, Peru official origin Jul 29, with July releases in the Aug 4
information set that the Jul 29 origin could not see). The contract now: the
exact normalized as-of is the terminal origin (weekly history preserved,
de-duplicated, sorted); a target still unpublished after its expected
publication date fails closed; and the official artifact refuses an
origin/as-of mismatch instead of relabelling an old origin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from MIDAS import BridgeNowcaster, MetadataPanel, RandomWalkNowcaster, VariableMeta
from nowcast import release_cycle as rc


def _panel(x_delay=1):
    """Quarterly target through 2026Q1; next quarter 2026Q2 publishes ~Aug 21
    (52-day rule). The June x observation (INSIDE the target quarter) carries
    a surprise; with ``x_delay=33`` it releases Aug 2, BETWEEN the Jul 30 and
    Aug 6 weekly origins, so only the exact-as-of terminal origin sees it."""
    months = pd.date_range("2015-01-01", "2026-06-01", freq="MS")
    rng = np.random.default_rng(7)
    x = pd.Series(rng.normal(2.0, 0.3, len(months)), index=months, name="x")
    x.iloc[-1] = 9.0                       # the June surprise
    qs = pd.date_range("2015-03-01", "2026-03-01", freq="3MS")
    y = pd.Series(2.0 + 0.5 * rng.normal(size=len(qs)), index=qs, name="tgt")
    metas = [VariableMeta(column="x", frequency="M", group="g",
                          publication_delay_days=x_delay),
             VariableMeta(column="tgt", frequency="Q", group="g",
                          publication_delay_days=52)]
    return MetadataPanel.from_frames(x.to_frame(), y.to_frame(), metas)


class _Spec:
    target = "tgt"
    target_delay_days = 52


MODELS = {"RW": RandomWalkNowcaster(),
          "Bridge(x)": BridgeNowcaster(indicators=["x"], _name="Bridge(x)")}


def _sweep(today, x_delay=1):
    return rc.live_path(_panel(x_delay), _Spec, dict(MODELS), step_days=7,
                        today=pd.Timestamp(today))


def test_terminal_origin_is_the_exact_as_of():
    live = _sweep("2026-08-04")
    assert len(live)
    assert live.origin_date.max() == pd.Timestamp("2026-08-04"), (
        "the as-of between weekly grid points must itself be an origin")


def test_weekly_history_preserved_and_sorted_unique():
    live = _sweep("2026-08-04")
    origins = pd.DatetimeIndex(sorted(live.origin_date.unique()))
    assert pd.Timestamp("2026-07-30") in origins       # the weekly point
    assert list(origins) == sorted(set(origins))
    diffs = np.diff(origins.asi8)
    assert (diffs > 0).all()                           # strictly increasing


def test_release_between_grid_points_reaches_the_terminal_origin():
    live = _sweep("2026-08-04", x_delay=33).set_index(["origin_date", "model"]).y_hat
    at_grid = live[(pd.Timestamp("2026-07-30"), "Bridge(x)")]
    at_asof = live[(pd.Timestamp("2026-08-04"), "Bridge(x)")]
    # June's x=9.0 (a target-quarter month) released Aug 2: invisible on
    # Jul 30, visible only at the exact as-of terminal origin
    assert np.isfinite(at_asof)
    assert not np.isclose(at_grid, at_asof), (
        "the terminal origin must be a FRESH fit on the as-of information set")


def test_no_duplicate_origin_when_as_of_sits_on_the_grid():
    live = _sweep("2026-07-30")
    per_model = live[live.model == "RW"].origin_date
    assert per_model.is_unique
    assert per_model.max() == pd.Timestamp("2026-07-30")


def test_stale_target_after_expected_publication_fails_closed():
    with pytest.raises(RuntimeError, match="expected publication"):
        _sweep("2026-08-25")               # 2026Q2 was due ~Aug 21, still NaN


def test_official_artifact_refuses_a_stale_origin():
    from pipeline.lib.nowcast_artifact import official_from

    rows = []
    for m, v in (("RW", 2.0), ("Bridge(leaders)", 2.2), ("Adaptive-IC", 2.1)):
        rows.append({"target": "g_pbiq", "ref_quarter": "2026-06-01",
                     "origin_date": "2026-07-29", "days_to_publication": -23,
                     "model": m, "y_true": np.nan, "y_hat": v, "y_std": None,
                     "info_index": 0.9})
    nowcasts = pd.DataFrame(rows)
    weights = pd.DataFrame([{"ref_quarter": "2026-06-01", "bin": 3,
                             "RW": 0.2, "Bridge(leaders)": 0.5,
                             "P-MIDAS(leaders)": 0.3}])
    with pytest.raises(ValueError, match="origin"):
        official_from(nowcasts, weights, {("2026-06-01", 3): [0.1, -0.1, 0.2]},
                      members=["RW", "Bridge(leaders)", "P-MIDAS(leaders)"],
                      adaptive_name="Adaptive-IC", as_of="2026-08-04")


def test_require_exact_origin_helper():
    rc.require_exact_origin("2026-08-04", "2026-08-04", "test")   # no raise
    with pytest.raises(ValueError, match="China"):
        rc.require_exact_origin("2026-08-01", "2026-08-04", "China nowcast")
