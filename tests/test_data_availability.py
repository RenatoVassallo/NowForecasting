from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline.lib.data_availability import (
    DEFAULT_REGISTRY,
    build_availability,
    collect_observations,
    load_events,
    load_registry,
    render_markdown,
)


def _series(code, *, lag=15, rule="period_end_plus_lag", monitor=None):
    return {
        "variable_name": code,
        "internal_code": code,
        "provider_code": code.upper(),
        "source_institution": "Test source",
        "source_url": "https://example.test/api",
        "frequency": "M",
        "unit": "index",
        "transformation": "none",
        "geographic_coverage": "test",
        "start_date": "2020-01-01",
        "release_calendar": {"rule": rule, "source": "test", "stale_after_days": 10},
        "publication_lag_days": lag,
        "revision_policy": "test",
        "seasonal_adjustment_status": "not adjusted",
        "vintage_availability": "test",
        "expected_update_frequency": "monthly",
        "current_availability_status": None,
        "ingestion_script": "test",
        "downstream_models": ["test model"],
        "fallback_source": None,
        "validation_rules": {
            "unique_periods": True,
            "monotonic_periods": True,
            "allow_all_missing": False,
            "plausible_range": [-100, 100],
        },
        "last_successful_refresh": None,
        "known_issues": [],
        "monitor": monitor or {"type": "manual", "manual_override": False},
        "required_for_publication": True,
    }


def _observations(rows):
    defaults = {
        "last_observation": pd.NaT,
        "last_value": 1.0,
        "n_observations": 1,
        "cache_modified_at": pd.Timestamp("2026-08-03"),
        "collector_status": None,
        "collector_detail": "",
    }
    return pd.DataFrame([{"internal_code": code, **defaults, **values}
                         for code, values in rows.items()])


def test_initial_registry_is_complete_and_unique():
    registry = load_registry(DEFAULT_REGISTRY)
    codes = [row["internal_code"] for row in registry["series"]]
    assert len(codes) >= 20
    assert len(codes) == len(set(codes))
    assert {"g_pbiq", "g_pbim_yoy", "gdp_yoy", "g_pe_tot", "weo_usa_ngdp_rpch"} <= set(codes)


def test_period_calendar_distinguishes_success_not_due_and_stale():
    registry = {
        "series": [
            _series("success", lag=1),
            _series("not_due", lag=15),
            _series("stale", lag=15),
        ]
    }
    obs = _observations({
        "success": {"last_observation": pd.Timestamp("2026-07-01")},
        "not_due": {"last_observation": pd.Timestamp("2026-06-01")},
        "stale": {"last_observation": pd.Timestamp("2026-05-01")},
    })
    result = build_availability(registry, obs, as_of="2026-08-03").set_index("internal_code")
    assert result.loc["success", "status"] == "successfully_updated"
    assert result.loc["not_due", "status"] == "not_yet_released"
    assert result.loc["stale", "status"] == "stale_observation"
    assert result.loc["not_due", "latest_expected_observation"] == pd.Timestamp("2026-06-01")


def test_failure_and_manual_statuses_have_explicit_precedence():
    registry = {
        "series": [
            _series("source"),
            _series("ingest"),
            _series("invalid"),
            _series("manual", monitor={"type": "manual", "manual_override": True}),
        ]
    }
    obs = _observations({
        "source": {"last_observation": pd.Timestamp("2026-06-01")},
        "ingest": {"collector_status": "ingestion_failure", "collector_detail": "parser failed"},
        "invalid": {"collector_status": "validation_failure", "collector_detail": "range failed"},
        "manual": {},
    })
    events = pd.DataFrame([{
        "internal_code": "source",
        "attempted_at": pd.Timestamp("2026-08-03"),
        "status": "source_unavailable",
        "detail": "HTTP 503",
    }])
    result = build_availability(registry, obs, events=events,
                                as_of="2026-08-03").set_index("internal_code")
    assert result.loc["source", "status"] == "source_unavailable"
    assert result.loc["ingest", "status"] == "ingestion_failure"
    assert result.loc["invalid", "status"] == "validation_failure"
    assert result.loc["manual", "status"] == "manually_overridden"


def test_age_threshold_can_use_cache_refresh_time():
    row = _series(
        "flash",
        rule="age_threshold",
        monitor={"type": "manual", "manual_override": False, "freshness_basis": "cache_mtime"},
    )
    obs = _observations({
        "flash": {
            "last_observation": pd.Timestamp("2026-07-01"),
            "cache_modified_at": pd.Timestamp("2026-08-02"),
        }
    })
    result = build_availability({"series": [row]}, obs, as_of="2026-08-03")
    assert result.loc[0, "status"] == "successfully_updated"


def test_collector_validates_a_tabular_cache(tmp_path):
    cache = tmp_path / "series.csv"
    pd.DataFrame({
        "date": ["2026-05-01", "2026-06-01"],
        "value": [2.0, 3.0],
        "country": ["PE", "PE"],
    }).to_csv(cache, index=False)
    row = _series("file_series", monitor={
        "type": "tabular_file",
        "path": cache.name,
        "column": "value",
        "date_column": "date",
        "filters": {"country": "PE"},
    })
    observed = collect_observations({"series": [row]}, repo_root=tmp_path)
    assert observed.loc[0, "last_observation"] == pd.Timestamp("2026-06-01")
    assert observed.loc[0, "n_observations"] == 2
    assert observed.loc[0, "collector_status"] is None


def test_event_loader_and_markdown_output(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({
        "internal_code": "x",
        "attempted_at": "2026-08-03T08:00:00",
        "status": "successfully_updated",
        "detail": "ok",
    }) + "\n")
    events = load_events(path)
    assert events.loc[0, "status"] == "successfully_updated"

    table = pd.DataFrame([{
        "internal_code": "x",
        "variable_name": "x",
        "source_institution": "Test source",
        "frequency": "M",
        "status": "successfully_updated",
        "last_observation": pd.Timestamp("2026-07-01"),
        "latest_expected_observation": pd.Timestamp("2026-07-01"),
        "next_expected_release": pd.Timestamp("2026-09-01"),
        "last_value": 1.0,
        "n_observations": 1,
        "cache_modified_at": pd.Timestamp("2026-08-03"),
        "last_successful_refresh": pd.Timestamp("2026-08-03"),
        "vintage_consistent": True,
        "detail": "ok",
    }])
    markdown = render_markdown(table, as_of="2026-08-03")
    assert "Data availability dashboard as of 2026-08-03" in markdown
    assert "successfully_updated" in markdown

