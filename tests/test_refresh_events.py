"""Task 9: the refresh event log is append-only, typed, and dashboard-visible."""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.lib import refresh_events as ev
from pipeline.lib.data_availability import build_availability


def test_events_append_and_never_rewrite(tmp_path):
    p = tmp_path / "events.jsonl"
    ev.record("x", "successfully_updated", as_of="2026-08-03", detail="first",
              n_rows=100, first_observation="2000-01-01",
              last_observation="2026-06-01", path=p)
    ev.record("x", "ingestion_failure", as_of="2026-08-04", detail="second",
              path=p)
    d = ev.read(p)
    assert len(d) == 2
    assert list(d.status) == ["successfully_updated", "ingestion_failure"]
    assert d.detail.tolist() == ["first", "second"]
    assert d.n_rows.iloc[0] == 100 and d.last_observation.iloc[0] == "2026-06-01"


def test_invalid_status_and_undocumented_override_are_rejected(tmp_path):
    p = tmp_path / "events.jsonl"
    with pytest.raises(ValueError, match="invalid event status"):
        ev.record("x", "kind_of_ok", as_of="2026-08-03", path=p)
    with pytest.raises(ValueError, match="author and a reason"):
        ev.record("x", "manually_overridden", as_of="2026-08-03", path=p)
    e = ev.record("x", "manually_overridden", as_of="2026-08-03",
                  override_author="analyst", override_reason="source outage",
                  override_effective_from="2026-08-03",
                  override_effective_to="2026-08-10", path=p)
    assert e["override_author"] == "analyst"


def test_error_events_drive_the_dashboard_status(tmp_path):
    registry = {"series": [{
        "variable_name": "x", "internal_code": "x", "provider_code": "X",
        "source_institution": "t", "source_url": "u", "frequency": "M",
        "unit": "i", "transformation": "none", "geographic_coverage": "t",
        "start_date": "2020-01-01",
        "release_calendar": {"rule": "period_end_plus_lag", "source": "t"},
        "publication_lag_days": 15, "revision_policy": "t",
        "seasonal_adjustment_status": "t", "vintage_availability": "t",
        "expected_update_frequency": "monthly",
        "current_availability_status": None, "ingestion_script": "t",
        "downstream_models": ["m"], "fallback_source": None,
        "validation_rules": {}, "last_successful_refresh": None,
        "known_issues": [], "monitor": {"type": "manual"},
        "required_for_publication": True,
    }]}
    obs = pd.DataFrame([{"internal_code": "x",
                         "last_observation": pd.Timestamp("2026-07-01"),
                         "last_value": 1.0, "n_observations": 1,
                         "cache_modified_at": pd.Timestamp("2026-08-01"),
                         "collector_status": None, "collector_detail": ""}])
    p = tmp_path / "events.jsonl"
    ev.record("x", "source_unavailable", as_of="2026-08-03",
              detail="HTTP 503", path=p)
    table = build_availability(registry, obs, events=ev.read(p), as_of="2026-08-03")
    assert table.status.iloc[0] == "source_unavailable"
    assert "503" in table.detail.iloc[0]


def test_codes_for_maps_providers_and_targets():
    registry = {"series": [
        {"internal_code": "a", "ingestion_script": "sources/commodities.py",
         "monitor": {"type": "target_panel", "target": "pe_tot"}},
        {"internal_code": "b", "ingestion_script": "targets/peru_gdp.py",
         "monitor": {"type": "target_panel", "target": "peru_gdp"}},
        {"internal_code": "c", "ingestion_script": "sources/imf.py",
         "monitor": {"type": "tabular_file"}},
    ]}
    assert ev.codes_for(registry, provider="commodities") == ["a"]
    assert ev.codes_for(registry, provider="imf") == ["c"]
    assert ev.codes_for(registry, target="peru_gdp") == ["b"]
