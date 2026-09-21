"""Data-frontier figure: per-variable timelines of what the run actually knew.

For every production-relevant series the run's availability artifact yields
three spans on a monthly axis: GREEN through the last released observation,
GREY from there to the next expected release (the waiting window), WHITE
beyond. Series whose latest observation was released within the last seven
days before the as-of are flagged NEW for this run (the Avance Coyuntural
batch, monthly PMIs on the 1st, and so on). The frame is computed run-locally
from availability.csv plus the registry; no cross-run reads.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.lib.data_frontier import frontier_frame


def _availability():
    return pd.DataFrame([
        {"internal_code": "g_pbim", "variable_name": "Peru monthly GDP proxy",
         "frequency": "M", "status": "not_yet_released",
         "last_observation": "2026-05", "next_expected_release": "2026-08-20"},
        {"internal_code": "ip_cum_yoy", "variable_name": "China IP",
         "frequency": "M", "status": "successfully_updated",
         "last_observation": "2026-06", "next_expected_release": "2026-08-15"},
        {"internal_code": "consumo_cemento", "variable_name": "Cemento",
         "frequency": "M", "status": "successfully_updated",
         "last_observation": "2026-06", "next_expected_release": "2026-09-01"},
        {"internal_code": "not_a_model_input", "variable_name": "Other",
         "frequency": "M", "status": "successfully_updated",
         "last_observation": "2026-06", "next_expected_release": "2026-09-01"},
    ])


REGISTRY = {"series": [
    {"internal_code": "g_pbim", "publication_lag_days": 51,
     "required_for_publication": True,
     "monitor": {"target": "peru_gdp"}},
    {"internal_code": "ip_cum_yoy", "publication_lag_days": 15,
     "required_for_publication": True,
     "monitor": {"target": "china"}},
    {"internal_code": "consumo_cemento", "publication_lag_days": 32,
     "required_for_publication": False,
     "monitor": {"target": "peru_gdp"}},
    {"internal_code": "not_a_model_input", "publication_lag_days": 15,
     "required_for_publication": False,
     "monitor": {"target": "peru_gdp"}},
]}


def test_frontier_spans_and_grouping():
    f = frontier_frame(_availability(), REGISTRY, as_of="2026-08-06",
                       include=["g_pbim", "ip_cum_yoy", "consumo_cemento"])
    assert list(f.internal_code) != []
    assert "not_a_model_input" not in set(f.internal_code)
    row = f.set_index("internal_code").loc["g_pbim"]
    assert row.obs_end == pd.Timestamp("2026-05-31")       # green through May
    assert row.next_release == pd.Timestamp("2026-08-20")  # grey until here
    assert row.group == "peru_gdp"
    assert row.required


def test_new_this_run_flag_uses_the_release_rule():
    f = frontier_frame(_availability(), REGISTRY, as_of="2026-08-06",
                       include=["g_pbim", "ip_cum_yoy", "consumo_cemento"])
    by = f.set_index("internal_code")
    # cemento June released Jun 30 + 32d = Aug 1: within 7 days of Aug 6 -> NEW
    assert bool(by.loc["consumo_cemento", "new_this_run"])
    # China IP June released Jul 15: outside the window -> not new
    assert not bool(by.loc["ip_cum_yoy", "new_this_run"])
    # Peru proxy May released Jul 21 (51d): outside -> not new
    assert not bool(by.loc["g_pbim", "new_this_run"])


def test_waiting_days_and_as_of_inside_grey_window():
    f = frontier_frame(_availability(), REGISTRY, as_of="2026-08-06",
                       include=["g_pbim"])
    row = f.iloc[0]
    assert row.days_to_next == 14                 # Aug 6 -> Aug 20
    assert row.obs_end < pd.Timestamp("2026-08-06") <= row.next_release


# --------------------------------------------------------------------------- #
# curated layout: fixed blocks in declared order, curated labels, loud failure
# --------------------------------------------------------------------------- #
LAYOUT = (
    ("Domestic", (
        ("consumo_cemento", "Cement consumption"),
        ("g_pbim", "Monthly GDP"),
    )),
    ("Foreign real", (
        ("ip_cum_yoy", "China industrial production"),
    )),
)


def test_layout_blocks_order_and_labels():
    f = frontier_frame(_availability(), REGISTRY, as_of="2026-08-06", layout=LAYOUT)
    assert list(f.internal_code) == ["consumo_cemento", "g_pbim", "ip_cum_yoy"]
    assert list(f.block) == ["Domestic", "Domestic", "Foreign real"]
    assert list(f.label) == ["Cement consumption", "Monthly GDP",
                             "China industrial production"]


def test_layout_missing_code_fails_loud():
    bad = (("Domestic", (("no_such_series", "Ghost"),)),)
    with pytest.raises(KeyError, match="no_such_series"):
        frontier_frame(_availability(), REGISTRY, as_of="2026-08-06", layout=bad)


def test_weo_round_cadence():
    avail = pd.DataFrame([{
        "internal_code": "weo_usa", "variable_name": "WEO", "frequency": "A",
        "status": "successfully_updated", "last_observation": "2026-08-06",
        "next_expected_release": None}])
    reg = {"series": [{"internal_code": "weo_usa", "publication_lag_days": 0,
                       "required_for_publication": False,
                       "monitor": {"target": "usa"}}]}
    lay = (("Surveys and external", (
        ("weo_usa", "IMF WEO", {"cadence": "weo_rounds"}),)),)
    f = frontier_frame(avail, reg, as_of="2026-08-06", layout=lay)
    row = f.iloc[0]
    # July update round (round months 1/4/7/10; updates late, full editions mid)
    assert row.obs_end == pd.Timestamp("2026-07-25")
    assert row.next_release == pd.Timestamp("2026-10-15")
    assert not row.new_this_run                    # round is 12 days old


def test_continuous_cadence():
    avail = pd.DataFrame([{
        "internal_code": "us_gdpnow", "variable_name": "GDPNow", "frequency": "Q",
        "status": "successfully_updated", "last_observation": "2026-07-01",
        "next_expected_release": None}])
    reg = {"series": [{"internal_code": "us_gdpnow", "publication_lag_days": 1,
                       "required_for_publication": False,
                       "monitor": {"target": "usa"}}]}
    lay = (("Surveys and external", (
        ("us_gdpnow", "Atlanta Fed GDPNow",
         {"cadence": "continuous", "interval_days": 7}),)),)
    f = frontier_frame(avail, reg, as_of="2026-08-06", layout=lay)
    row = f.iloc[0]
    assert bool(row.continuous)
    assert row.next_release == pd.Timestamp("2026-08-13")    # as_of + interval


def test_quarterly_start_anchored_series_spf_style():
    avail = pd.DataFrame([{
        "internal_code": "spf_gdp_h0", "variable_name": "SPF", "frequency": "Q",
        "status": "not_yet_released", "last_observation": "2026-04-01",
        "next_expected_release": None}])
    reg = {"series": [{"internal_code": "spf_gdp_h0", "publication_lag_days": 45,
                       "required_for_publication": False,
                       "release_calendar": {"rule": "period_start_plus_lag",
                                            "source": "t"},
                       "monitor": {"target": "usa"}}]}
    lay = (("Surveys and external", (("spf_gdp_h0", "SPF real GDP"),)),)
    f = frontier_frame(avail, reg, as_of="2026-08-06", layout=lay)
    row = f.iloc[0]
    assert row.obs_end == pd.Timestamp("2026-06-30")         # Q2 information edge
    assert row.next_release == pd.Timestamp("2026-08-15")    # Q3 start + 45
    assert not row.new_this_run                              # released May 16


def test_detected_release_passthrough_marks_pending_ingest():
    avail = _availability()
    avail["release_detected"] = [False, False, True, False]  # cemento row
    avail["detected_release_date"] = [None, None, "2026-08-05", None]
    avail["calendar_expected_release"] = [None, None, "2026-08-05", None]
    lay = (("Domestic", (("consumo_cemento", "Cement consumption"),)),)
    f = frontier_frame(avail, REGISTRY, as_of="2026-08-06", layout=lay)
    row = f.iloc[0]
    assert bool(row.released_pending_ingest)
    assert row.next_release == pd.Timestamp("2026-08-05")    # grey ends at detection
    assert row.days_to_next == -1
