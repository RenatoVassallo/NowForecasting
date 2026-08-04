"""Task 23: synthetic pipeline smoke run.

Exercises the production entry point end to end on the run-lifecycle level:
context creation, staged store, manifest (environment, hashes, seeds,
calibration inputs, statuses), success marker, atomic promotion, and the
``latest`` pointer. Stage bodies are disabled, so the smoke run works on a
public clone without the private data layer; stage logic has its own tests.
"""

from __future__ import annotations

import json

import pytest


def test_main_lifecycle_smoke(tmp_path, monkeypatch):
    from pipeline.config import params

    monkeypatch.setattr(params, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(params, "STAGES", {k: False for k in params.STAGES})
    monkeypatch.setattr(params, "UPDATE_LATEST_SYMLINK", True)

    from pipeline.main import main

    root = main(as_of="2026-08-04", run_id="2026-08-04__smoke")

    assert root == tmp_path / "2026-08-04__smoke"
    assert (root / "_SUCCESS").exists()
    assert not (tmp_path / ".staging" / "2026-08-04__smoke").exists()
    assert (tmp_path / "latest").resolve() == root.resolve()

    man = json.loads((root / "manifest.json").read_text())
    assert man["status"] == "success"
    assert man["as_of"] == "2026-08-04"
    assert set(man["stage_status"]) == {"data", "preflight", "nowcast",
                                        "forecast", "fanchart", "report"}
    assert all(v == "skipped" for v in man["stage_status"].values())
    assert man["environment"]["python"].startswith("3.11")
    assert man["registry_sha"] not in (None, "", "absent")
    assert "seeds" in man and "calibration_inputs" in man


def test_failed_run_is_quarantined(tmp_path, monkeypatch):
    from pipeline.config import params
    import pipeline.lib.preflight as pf

    monkeypatch.setattr(params, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(params, "STAGES",
                        {k: False for k in params.STAGES} | {"report": True})
    monkeypatch.setattr(params, "UPDATE_LATEST_SYMLINK", True)
    monkeypatch.setattr(pf, "run_preflight",
                        lambda store, params: (_ for _ in ()).throw(
                            RuntimeError("synthetic preflight failure")))

    from pipeline.main import main

    with pytest.raises(RuntimeError, match="synthetic preflight"):
        main(as_of="2026-08-04", run_id="2026-08-04__fail")

    assert not (tmp_path / "2026-08-04__fail").exists()
    assert not (tmp_path / "latest").exists()
    staged = tmp_path / ".staging" / "2026-08-04__fail"
    assert (staged / "_FAILED").exists()
    man = json.loads((staged / "manifest.json").read_text())
    assert man["status"] == "failed"
    assert man["stage_status"]["preflight"] == "failed"
