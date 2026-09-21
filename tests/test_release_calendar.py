"""Observed-release calendar: expected release dates learned from evidence.

The registry's scalar ``publication_lag_days`` is a prior, not a calendar:
BCRP published the July 2026 expectations block on August 5 while the scalar
rule said August 7. The calendar layer keeps an append-only store of OBSERVED
release dates (first date each observation was seen on the provider), and per
series derives:

- ``expected``: the next release date. Registry scalar until at least
  ``MIN_HISTORY`` observed releases exist, then the observed median lag.
- ``check_from``: when to START probing the provider. Two days before the
  expected date with no history; from the earliest lag ever observed once
  there is any evidence.
- detection: an API probe (or any refresh) that sees the next period records
  an observed release, which flips the availability row to "released,
  pending ingest" for the frontier WITHOUT touching the preflight gate.

As-of safety: reads filter ``first_seen <= as_of``; probes only run when the
as-of IS the wall-clock today, so historical runs can never leak the future.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.lib import release_calendar as rc


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #
def test_record_and_load_keep_earliest_first_seen(tmp_path):
    p = tmp_path / "observed.parquet"
    n1 = rc.record_observed(p, [{"internal_code": "exp_eco3m", "period": "2026-07",
                                 "first_seen": "2026-08-05", "source": "operator"}])
    n2 = rc.record_observed(p, [{"internal_code": "exp_eco3m", "period": "2026-07",
                                 "first_seen": "2026-08-06", "source": "api_probe"}])
    obs = rc.load_observed(p)
    assert (n1, n2) == (1, 0)                      # second sighting is not new
    assert len(obs) == 1
    assert pd.Timestamp(obs.first_seen.iloc[0]) == pd.Timestamp("2026-08-05")


def test_load_observed_filters_by_as_of(tmp_path):
    p = tmp_path / "observed.parquet"
    rc.record_observed(p, [
        {"internal_code": "a", "period": "2026-06", "first_seen": "2026-07-05", "source": "s"},
        {"internal_code": "a", "period": "2026-07", "first_seen": "2026-08-05", "source": "s"},
    ])
    obs = rc.load_observed(p, as_of="2026-07-31")
    assert list(obs.period) == ["2026-06"]         # the August sighting is the future


# --------------------------------------------------------------------------- #
# expectation rule
# --------------------------------------------------------------------------- #
def _obs(rows):
    return pd.DataFrame(rows, columns=["internal_code", "period", "first_seen", "source"])


def test_no_history_falls_back_to_scalar_with_probe_margin():
    e = rc.expected_next_release("exp_eco3m", "2026-06", 7, _obs([]))
    assert e["period"] == "2026-07"
    assert e["expected"] == pd.Timestamp("2026-08-07")       # Jul 31 + 7
    assert e["check_from"] == pd.Timestamp("2026-08-05")     # expected - 2d margin
    assert e["basis"] == "registry_lag"


def test_full_history_uses_observed_median_and_min_lag():
    rows = [{"internal_code": "exp_eco3m", "period": p, "first_seen": d, "source": "s"}
            for p, d in [("2026-04", "2026-05-05"),          # lag 5
                         ("2026-05", "2026-06-06"),          # lag 6
                         ("2026-06", "2026-07-05")]]         # lag 5
    e = rc.expected_next_release("exp_eco3m", "2026-06", 7, _obs(rows))
    assert e["expected"] == pd.Timestamp("2026-08-05")       # Jul 31 + median 5
    assert e["check_from"] == pd.Timestamp("2026-08-05")     # Jul 31 + min 5
    assert e["basis"] == "observed_median(n=3)"


def test_partial_history_widens_probe_window_but_keeps_scalar_expected():
    rows = [{"internal_code": "exp_eco3m", "period": "2026-06",
             "first_seen": "2026-07-05", "source": "s"}]     # single lag-5 sighting
    e = rc.expected_next_release("exp_eco3m", "2026-06", 7, _obs(rows))
    assert e["expected"] == pd.Timestamp("2026-08-07")       # scalar until MIN_HISTORY
    assert e["check_from"] == pd.Timestamp("2026-08-05")     # but probe from min lag
    assert e["basis"] == "registry_lag"


def test_start_anchored_quarterly_rule_spf_style():
    # SPF: quarter stamped by its FIRST month, released lag days after the
    # quarter STARTS (Q3 2026 survey: Jul 1 + 45 = Aug 15)
    e = rc.expected_next_release("spf_gdp_h0", "2026Q2", 45, _obs([]),
                                 freq="Q", rule="period_start_plus_lag")
    assert e["period"] == "2026Q3"
    assert e["expected"] == pd.Timestamp("2026-08-15")
    assert e["check_from"] == pd.Timestamp("2026-08-13")


def test_due_for_probe_window_and_already_detected():
    obs0 = _obs([])
    assert not rc.due_for_probe("x", "2026-06", 7, obs0, as_of="2026-08-04")
    assert rc.due_for_probe("x", "2026-06", 7, obs0, as_of="2026-08-05")
    assert rc.due_for_probe("x", "2026-06", 7, obs0, as_of="2026-08-20")  # late: keep probing
    seen = _obs([{"internal_code": "x", "period": "2026-07",
                  "first_seen": "2026-08-05", "source": "s"}])
    assert not rc.due_for_probe("x", "2026-06", 7, seen, as_of="2026-08-06")


# --------------------------------------------------------------------------- #
# BCRP probe parser
# --------------------------------------------------------------------------- #
def test_probe_release_parses_bcrp_json():
    from sources.bcrp import probe_release

    def fake(url):
        # three-month window: single-month requests trip BCRP's anti-bot layer
        assert "/api/PD38045AM/json/2026-5/2026-7" in url
        return json.dumps({"periods": [{"name": "Jul.2026", "values": ["62.2676"]}]})

    assert probe_release("PD38045AM", "2026-07", fetcher=fake) is True


def test_probe_release_rejects_missing_or_nd_values():
    from sources.bcrp import probe_release
    nd = json.dumps({"periods": [{"name": "Jul.2026", "values": ["n.d."]}]})
    other = json.dumps({"periods": [{"name": "Jun.2026", "values": ["61.0"]}]})
    empty = json.dumps({"periods": []})
    assert probe_release("X", "2026-07", fetcher=lambda u: nd) is False
    assert probe_release("X", "2026-07", fetcher=lambda u: other) is False
    assert probe_release("X", "2026-07", fetcher=lambda u: empty) is False


def test_probe_release_names_the_anti_bot_challenge():
    from sources.bcrp import BCRPProbeError, probe_release
    with pytest.raises(BCRPProbeError, match="challenge"):
        probe_release("X", "2026-07",
                      fetcher=lambda u: "<!DOCTYPE html>\n<html>...</html>")


# --------------------------------------------------------------------------- #
# BCRP probe: quarterly series (the API labels them "T1.26"/"Q1.26", and the
# quarterly window is expressed in quarter ordinals, not months)
# --------------------------------------------------------------------------- #
def test_parse_period_name_handles_both_frequencies():
    from sources.bcrp import _parse_period_name
    assert _parse_period_name("Ene.2026") == pd.Period("2026-01", freq="M")
    assert _parse_period_name("T2.26") == pd.Period("2026Q2", freq="Q")   # Spanish endpoint
    assert _parse_period_name("Q2.26") == pd.Period("2026Q2", freq="Q")   # /ing endpoint
    assert _parse_period_name("T1.93") == pd.Period("1993Q1", freq="Q")   # century pivot
    assert _parse_period_name("nope") is None
    assert _parse_period_name("T5.26") is None


def test_probe_release_handles_quarterly_series():
    from sources.bcrp import probe_release

    def fake(url):
        # three QUARTERS ending at the target, in quarter ordinals
        assert "/api/PN02533AQ/json/2025-4/2026-2" in url, url
        return json.dumps({"periods": [{"name": "T2.26", "values": ["34824.8675"]}]})

    assert probe_release("PN02533AQ", "2026Q2", fetcher=fake) is True


def test_probe_release_quarterly_is_strict_about_the_period():
    from sources.bcrp import probe_release
    other = json.dumps({"periods": [{"name": "T1.26", "values": ["31399.65"]}]})
    nd = json.dumps({"periods": [{"name": "T2.26", "values": ["n.d."]}]})
    assert probe_release("X", "2026Q2", fetcher=lambda u: other) is False
    assert probe_release("X", "2026Q2", fetcher=lambda u: nd) is False


_REG_Q = {"series": [
    {"internal_code": "g_invq_m", "frequency": "Q", "publication_lag_days": 51,
     "provider_code": "PN02533AQ",
     "source_url": "https://estadisticas.bcrp.gob.pe/estadisticas/series/api",
     "release_calendar": {"rule": "period_end_plus_lag", "source": "t"},
     "monitor": {"type": "target_panel", "target": "peru_gdp"}},
]}


def _table_q():
    return pd.DataFrame([
        {"internal_code": "g_invq_m", "frequency": "Q", "status": "stale_observation",
         "last_observation": "2026-03-01", "next_expected_release": "2026-08-20"},
    ])


def test_probe_due_series_probes_quarterly_series(tmp_path):
    """Regression: quarterly series were filtered out and never probed, so a
    published quarter stayed invisible to the observed-release calendar."""
    p = tmp_path / "observed.parquet"
    calls = []

    def prober(provider_code, period):        # two positional args, as before
        calls.append((provider_code, str(period)))
        return True

    table = rc.annotate_expectations(_table_q(), _REG_Q, as_of="2026-09-21", store_path=p)
    hits, errors = rc.probe_due_series(table, _REG_Q, as_of="2026-09-21",
                                       store_path=p, prober=prober)
    assert calls == [("PN02533AQ", "2026Q2")]     # the quarter, at quarterly frequency
    assert hits == ["g_invq_m"] and errors == []
    obs = rc.load_observed(p)
    assert list(obs.period) == ["2026Q2"]


# --------------------------------------------------------------------------- #
# availability annotation (the preflight join)
# --------------------------------------------------------------------------- #
_REG = {"series": [
    {"internal_code": "exp_eco3m", "frequency": "M", "publication_lag_days": 7,
     "provider_code": "PD38045AM",
     "source_url": "https://estadisticas.bcrp.gob.pe/estadisticas/series/api",
     "release_calendar": {"rule": "period_end_plus_lag", "source": "t"},
     "monitor": {"type": "target_panel", "target": "peru_gdp"}},
    {"internal_code": "ip_cum_yoy", "frequency": "M", "publication_lag_days": 15,
     "provider_code": "CN_IP",
     "source_url": "https://www.stats.gov.cn",
     "release_calendar": {"rule": "period_end_plus_lag", "source": "t"},
     "monitor": {"type": "tabular_file", "target": "china"}},
]}


def _table():
    return pd.DataFrame([
        {"internal_code": "exp_eco3m", "frequency": "M", "status": "not_yet_released",
         "last_observation": "2026-06-01", "next_expected_release": "2026-08-07"},
        {"internal_code": "ip_cum_yoy", "frequency": "M", "status": "not_yet_released",
         "last_observation": "2026-06-01", "next_expected_release": "2026-08-15"},
    ])


def test_annotate_marks_detected_release_and_keeps_scalar_elsewhere(tmp_path):
    p = tmp_path / "observed.parquet"
    rc.record_observed(p, [{"internal_code": "exp_eco3m", "period": "2026-07",
                            "first_seen": "2026-08-05", "source": "api_probe"}])
    t = rc.annotate_expectations(_table(), _REG, as_of="2026-08-06", store_path=p)
    by = t.set_index("internal_code")
    assert bool(by.loc["exp_eco3m", "release_detected"])
    assert pd.Timestamp(by.loc["exp_eco3m", "detected_release_date"]) == pd.Timestamp("2026-08-05")
    assert pd.Timestamp(by.loc["exp_eco3m", "calendar_expected_release"]) == pd.Timestamp("2026-08-05")
    assert by.loc["exp_eco3m", "calendar_basis"] == "detected"
    assert by.loc["exp_eco3m", "calendar_next_period"] == "2026-07"
    assert not bool(by.loc["ip_cum_yoy", "release_detected"])
    assert by.loc["ip_cum_yoy", "calendar_basis"] == "registry_lag"
    assert pd.Timestamp(by.loc["ip_cum_yoy", "calendar_expected_release"]) == pd.Timestamp("2026-08-15")
    # the preflight gate column is untouched: detection never blocks a run
    assert by.loc["exp_eco3m", "status"] == "not_yet_released"


def test_annotate_is_as_of_safe(tmp_path):
    p = tmp_path / "observed.parquet"
    rc.record_observed(p, [{"internal_code": "exp_eco3m", "period": "2026-07",
                            "first_seen": "2026-08-05", "source": "api_probe"}])
    t = rc.annotate_expectations(_table(), _REG, as_of="2026-08-04", store_path=p)
    row = t.set_index("internal_code").loc["exp_eco3m"]
    assert not bool(row.release_detected)          # sighting lies after this as-of


def test_probe_due_series_records_hits_and_survives_failures(tmp_path):
    p = tmp_path / "observed.parquet"
    calls = []

    def prober(provider_code, period):
        calls.append((provider_code, str(period)))
        if provider_code == "PD38045AM":
            return True
        raise RuntimeError("network down")

    table = rc.annotate_expectations(_table(), _REG, as_of="2026-08-06", store_path=p)
    hits, errors = rc.probe_due_series(table, _REG, as_of="2026-08-06",
                                       store_path=p, prober=prober)
    assert hits == ["exp_eco3m"]
    # ip_cum_yoy is not a BCRP series: never probed, no error either
    assert calls == [("PD38045AM", "2026-07")]
    assert errors == []
    obs = rc.load_observed(p)
    assert list(obs.internal_code) == ["exp_eco3m"]
    assert pd.Timestamp(obs.first_seen.iloc[0]) == pd.Timestamp("2026-08-06")
    assert obs.source.iloc[0] == "api_probe"


def test_probe_skips_series_outside_window(tmp_path):
    p = tmp_path / "observed.parquet"
    table = rc.annotate_expectations(_table(), _REG, as_of="2026-08-03", store_path=p)
    hits, errors = rc.probe_due_series(table, _REG, as_of="2026-08-03",
                                       store_path=p,
                                       prober=lambda c, per: pytest.fail("must not probe"))
    assert hits == [] and errors == []


# --------------------------------------------------------------------------- #
# target publication date for the nowcast figure's dashed line
# --------------------------------------------------------------------------- #
def test_target_publication_falls_back_to_canonical_rule():
    from pipeline.blocks._common import expected_publication
    avail = pd.DataFrame([{"internal_code": "g_pbiq",
                           "calendar_next_period": "2026Q2",
                           "calendar_expected_release": "2026-08-20",
                           "calendar_basis": "registry_lag"}])
    q = pd.Period("2026Q2", freq="Q")
    d, basis = rc.expected_target_publication(q, 52, availability=avail, code="g_pbiq")
    assert d == expected_publication(q, 52)        # scalar basis: canonical wins
    assert basis == "canonical_lag"


def test_target_publication_uses_observed_evidence_for_matching_quarter():
    avail = pd.DataFrame([{"internal_code": "g_pbiq",
                           "calendar_next_period": "2026Q2",
                           "calendar_expected_release": "2026-08-14",
                           "calendar_basis": "observed_median(n=4)"}])
    q = pd.Period("2026Q2", freq="Q")
    d, basis = rc.expected_target_publication(q, 52, availability=avail, code="g_pbiq")
    assert d == pd.Timestamp("2026-08-14")
    assert basis == "observed_median(n=4)"
    # a different reference quarter must NOT inherit that date
    d2, basis2 = rc.expected_target_publication(pd.Period("2026Q3", freq="Q"), 52,
                                                availability=avail, code="g_pbiq")
    assert basis2 == "canonical_lag"
