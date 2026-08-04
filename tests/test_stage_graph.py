"""F6: the stage configuration is validated BEFORE any output exists.

Stages form a dependency graph (report needs fanchart, fanchart needs forecast
and nowcast, forecast needs nowcast). An incoherent selection (the old default
enabled report alone) refuses up front instead of failing deep inside a stage.
The only sanctioned way to re-render a report without recomputing is the
EXPLICIT report-from-run mode, which names a promoted source run; nothing is
ever found implicitly.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.lib.stagegraph import (StageConfigError, resolve_report_source,
                                     validate_stages)


def test_full_run_is_valid():
    validate_stages({"data": True, "nowcast": True, "forecast": True,
                     "fanchart": True, "report": True})


def test_report_only_is_refused():
    with pytest.raises(StageConfigError, match="report.*fanchart"):
        validate_stages({"data": False, "nowcast": False, "forecast": False,
                         "fanchart": False, "report": True})


def test_fanchart_without_forecast_is_refused():
    with pytest.raises(StageConfigError, match="fanchart"):
        validate_stages({"data": False, "nowcast": True, "forecast": False,
                         "fanchart": True, "report": False})


def test_estimation_only_selections_are_valid():
    validate_stages({"data": True, "nowcast": False, "forecast": False,
                     "fanchart": False, "report": False})
    validate_stages({"data": False, "nowcast": True, "forecast": True,
                     "fanchart": False, "report": False})


def test_main_refuses_incoherent_config_before_any_output(tmp_path, monkeypatch):
    from pipeline.config import params

    monkeypatch.setattr(params, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(params, "STAGES",
                        {"data": False, "nowcast": False, "forecast": False,
                         "fanchart": False, "report": True})
    from pipeline.main import main

    with pytest.raises(StageConfigError):
        main(as_of="2026-08-04", run_id="2026-08-04__bad")
    assert not any(tmp_path.rglob("*")), "no output may exist for an invalid config"


def _promoted_source(tmp_path, rid="2026-08-01__src", status="success"):
    run = tmp_path / rid
    run.mkdir(parents=True)
    (run / "_SUCCESS").write_text("t")
    (run / "manifest.json").write_text(json.dumps(
        {"run_id": rid, "as_of": "2026-08-01", "status": status,
         "code_version": "src1"}))
    return run


def test_report_source_resolution(tmp_path):
    with pytest.raises(StageConfigError, match="no promoted run"):
        resolve_report_source(tmp_path, "2026-08-01__missing")

    run = _promoted_source(tmp_path)
    (run / "_SUCCESS").unlink()
    with pytest.raises(StageConfigError, match="_SUCCESS"):
        resolve_report_source(tmp_path, run.name)

    (run / "_SUCCESS").write_text("t")
    src = resolve_report_source(tmp_path, run.name)
    assert src.root == run
    assert str(src.as_of.date()) == "2026-08-01"


def test_report_from_run_wiring(tmp_path, monkeypatch):
    from pipeline.config import params
    import pipeline.stages.report as rep

    monkeypatch.setattr(params, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(params, "UPDATE_LATEST_SYMLINK", False)
    monkeypatch.setattr(params, "PUBLISH_PRODUCTS", False)
    src = _promoted_source(tmp_path)

    def fake_report(store, params_, whatsnew, lines, timings):
        (store.root / "report.md").write_text("rendered")
        store._track(store.root / "report.md", "report")
        store.require("report.md")

    monkeypatch.setattr(rep, "run", fake_report)
    from pipeline.main import main

    root = main(as_of="ignored", run_id="2026-08-04__rerender",
                report_from=src.name)
    man = json.loads((root / "manifest.json").read_text())
    assert man["source_run"] == src.name
    assert man["as_of"] == "2026-08-01", "the report must carry the SOURCE as_of"
    assert (root / "report.md").read_text() == "rendered"
