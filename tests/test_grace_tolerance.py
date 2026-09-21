"""Graceful lateness: disclose and proceed inside a grace window, block beyond.

The hard gate treated a provider running two days late (BCRP June terms of
trade, US CPI released the afternoon of its due date) identically to a
cache rotting for a month, forcing a manual override for routine slippage.
Now REQUIRED series that are ``stale_observation`` are TOLERATED while
their lateness is within a grace window (default 7 days; widened by the
observed-release history when the series has one), loudly disclosed in the
console, the availability artifact, the frontier and the report. Error
statuses (ingestion, validation, source unavailable) and staleness beyond
grace still hard-block. Nothing silent: tolerance is disclosure, not a
free pass.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.lib import release_calendar as rc
from pipeline.lib.data_availability import build_availability
from pipeline.lib.preflight import evaluate


def _registry_rows():
    base = {
        "source_institution": "T", "source_url": "t", "unit": "u",
        "transformation": "none", "geographic_coverage": "PE",
        "start_date": "2000-01-01", "revision_policy": "r",
        "seasonal_adjustment_status": "s", "vintage_availability": "v",
        "expected_update_frequency": "monthly",
        "current_availability_status": None, "ingestion_script": "t",
        "downstream_models": ["m"], "fallback_source": None,
        "validation_rules": {}, "last_successful_refresh": None,
        "known_issues": [], "provider_code": "X",
    }
    return [
        {**base, "internal_code": "tot_like", "variable_name": "ToT",
         "frequency": "M", "publication_lag_days": 40,
         "release_calendar": {"rule": "period_end_plus_lag", "source": "t"},
         "monitor": {"type": "manual"}, "required_for_publication": True},
        {**base, "internal_code": "cpi_like", "variable_name": "CPI",
         "frequency": "M", "publication_lag_days": 12,
         "release_calendar": {"rule": "period_end_plus_lag", "source": "t"},
         "monitor": {"type": "manual"}, "required_for_publication": True},
    ]


def test_build_availability_reports_days_late():
    obs = pd.DataFrame([
        {"internal_code": "tot_like", "last_observation": pd.Timestamp("2026-05-31"),
         "last_value": 1.0, "n_observations": 10, "cache_modified_at": pd.NaT,
         "collector_status": None, "collector_detail": ""},
        {"internal_code": "cpi_like", "last_observation": pd.Timestamp("2026-06-30"),
         "last_value": 1.0, "n_observations": 10, "cache_modified_at": pd.NaT,
         "collector_status": None, "collector_detail": ""},
    ])
    table = build_availability({"series": _registry_rows()}, obs, as_of="2026-08-12")
    by = table.set_index("internal_code")
    # June ToT was expected Jun 30 + 40 = Aug 9: three days late
    assert by.loc["tot_like", "status"] == "stale_observation"
    assert int(by.loc["tot_like", "days_late"]) == 3
    # July CPI was expected Jul 31 + 12 = Aug 12: due today, zero days late
    assert by.loc["cpi_like", "status"] == "stale_observation"
    assert int(by.loc["cpi_like", "days_late"]) == 0


def _table(rows):
    return pd.DataFrame([{"internal_code": c, "status": s, "detail": d,
                          "days_late": dl} for c, s, d, dl in rows])


def _reg(required):
    return {"series": [{"internal_code": c, "required_for_publication": True}
                       for c in required]}


def test_evaluate_tolerates_within_grace_blocks_beyond():
    reg = _reg(["tot_like", "dead_like"])
    table = _table([("tot_like", "stale_observation", "late", 3),
                    ("dead_like", "stale_observation", "rotten", 9)])
    offenders, waived, tolerated, unknown = evaluate(table, reg, grace_days=7)
    assert tolerated == ["tot_like"]
    assert len(offenders) == 1 and "dead_like" in offenders[0]
    assert waived == [] and unknown == []


def test_evaluate_error_statuses_never_tolerated():
    reg = _reg(["x"])
    table = _table([("x", "ingestion_failure", "boom", 0)])
    offenders, _, tolerated, _ = evaluate(table, reg, grace_days=7)
    assert tolerated == [] and len(offenders) == 1


def test_evaluate_per_series_grace_overrides_default():
    reg = _reg(["slow"])
    table = _table([("slow", "stale_observation", "late", 10)])
    offenders, _, tolerated, _ = evaluate(table, reg, grace_days=7,
                                          grace_by_code={"slow": 13})
    assert tolerated == ["slow"] and offenders == []


def test_evaluate_manual_override_still_waives():
    reg = _reg(["m2"])
    table = _table([("m2", "stale_observation", "portal outage", 30)])
    offenders, waived, tolerated, _ = evaluate(
        table, reg, overrides={"m2": {"author": "rvs", "reason": "outage"}},
        grace_days=7)
    assert waived == ["m2"] and offenders == [] and tolerated == []


def test_grace_days_learned_from_observed_history():
    assert rc.grace_days_for("x", 40, pd.DataFrame(columns=["internal_code", "period", "first_seen", "source"])) == rc.DEFAULT_GRACE_DAYS
    rows = [{"internal_code": "x", "period": p, "first_seen": d, "source": "s"}
            for p, d in [("2026-03", "2026-05-10"),   # lag 40
                         ("2026-04", "2026-06-10"),   # lag 41
                         ("2026-05", "2026-07-22")]]  # lag 52 (worst)
    g = rc.grace_days_for("x", 40, pd.DataFrame(rows))
    # median lag 41, worst 52: tolerate the historical worst plus margin
    assert g == max(rc.DEFAULT_GRACE_DAYS, 52 - 41 + 2)
