"""Task 6: the information frontier and the China January-February release.

Project-level acceptance for the corrected MIDAS information index (pinned
0.2.3) and for the panel assembly rule that keeps January structurally missing
while February carries the combined NBS release at its true timing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from MIDAS import MetadataPanel, VariableMeta
from nowcast.release_cycle import add_information_index


def _panel(monthly: pd.DataFrame, delays: dict) -> MetadataPanel:
    quarterly = pd.DataFrame({"y": [1.0]}, index=[pd.Timestamp("2025-12-01")])
    metas = [VariableMeta(column=c, frequency="M", group="g",
                          publication_delay_days=int(d)) for c, d in delays.items()]
    metas.append(VariableMeta(column="y", frequency="Q", group="g",
                              publication_delay_days=52))
    return MetadataPanel.from_frames(monthly, quarterly, metas)


def _rc(ref: str, origins) -> pd.DataFrame:
    return pd.DataFrame({"ref_quarter": pd.Timestamp(ref),
                         "origin_date": pd.to_datetime(list(origins))})


def test_live_incomplete_quarter_stays_below_one():
    idx = pd.date_range("2024-01-01", "2025-04-01", freq="MS")
    panel = _panel(pd.DataFrame({"fast": 1.0, "slow": 1.0}, index=idx),
                   {"fast": 1, "slow": 45})
    rc = add_information_index(_rc("2025-06-01", ["2025-05-20"]), panel,
                               window_months=6)
    assert rc.info_index.iloc[0] < 1.0


def test_denominator_is_invariant_to_snapshot_truncation():
    full = pd.DataFrame({"x": 1.0},
                        index=pd.date_range("2024-01-01", "2025-06-01", freq="MS"))
    live = full.loc[:"2025-03-01"]
    origins = ["2025-03-10", "2025-04-10", "2025-05-10"]
    a = add_information_index(_rc("2025-06-01", origins), _panel(full, {"x": 15}))
    b = add_information_index(_rc("2025-06-01", origins), _panel(live, {"x": 15}))
    np.testing.assert_allclose(a.info_index, b.info_index)


def test_combined_release_arrives_only_at_its_actual_date():
    """February carries the combined Jan-Feb figure; nothing about it can be
    visible before the mid-March release (the old January copy leaked it a
    month early)."""
    idx = pd.date_range("2020-01-01", "2025-03-01", freq="MS")
    vals = pd.Series(1.0, index=idx)
    vals[idx.month == 1] = np.nan                     # structurally missing
    panel = _panel(pd.DataFrame({"cn": vals}), {"cn": 15})
    rc = add_information_index(
        _rc("2025-03-01", ["2025-02-20", "2025-03-20", "2025-04-20"]),
        panel, window_months=2)
    jan_feb_window = rc.set_index(rc.origin_date.dt.strftime("%m-%d"))["info_index"]
    # window is Jan..Mar; expected cells are Feb and Mar only. On Feb 20 nothing
    # has been released; on Mar 20 the combined (February) figure is out.
    assert jan_feb_window.loc["02-20"] == 0.0
    assert jan_feb_window.loc["03-20"] == pytest.approx(0.5)
    assert jan_feb_window.loc["04-20"] == pytest.approx(1.0)


def test_assemble_panel_keeps_january_structural_but_february_complete():
    from analysis.data import assemble_panel

    idx = pd.date_range("2023-01-01", "2024-06-01", freq="MS")
    cum = pd.Series(np.linspace(4, 6, len(idx)), index=idx)
    cum[idx.month == 1] = np.nan                      # NBS combined release
    plain = pd.Series(2.0, index=idx)                 # publishes January normally
    raw = pd.DataFrame({"cn_cum_yoy": cum, "plain_yoy": plain})
    monthly, _, _ = assemble_panel(
        raw, pd.DataFrame({"y": [1.0]}, index=[pd.Timestamp("2024-03-01")]),
        transforms={"cn_cum_yoy": "decum_yoy", "plain_yoy": "none"},
        delays={"cn_cum_yoy": 15, "plain_yoy": 15}, groups={}, target="y",
        target_delay=52, fill_jan_gap=True)
    jan = monthly.index.month == 1
    assert monthly.loc[jan, "cn_cum_yoy"].isna().all(), \
        "January must stay structurally missing after the fill-and-blank rule"
    assert monthly.loc[~jan, "cn_cum_yoy"].notna().all(), \
        "February and later months must survive de-cumulation"
    feb = monthly.index[(monthly.index.month == 2)]
    for f in feb:                                     # equal-base-share identity
        assert monthly.loc[f, "cn_cum_yoy"] == pytest.approx(float(cum.loc[f]))
    assert monthly.loc[jan, "plain_yoy"].notna().all(), \
        "columns that genuinely publish January must keep it"
