"""Task 21: the run lifecycle is atomic and fully recorded.

A production run must stage its artifacts in a temporary directory, record
environment/hashes/seeds/timings in the manifest, write a success marker, and
only then be promoted (atomically) into ``runs/<run_id>``, with ``latest``
updated last. A failed run must never appear as a promoted run and must never
be used as a prior-bundle fallback.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.lib.context import RunContext
from pipeline.lib.store import RunStore


def _ctx(run_id="2026-08-04__test"):
    return RunContext(as_of=pd.Timestamp("2026-08-04"), run_id=run_id,
                      code_version="testver")


def test_staged_run_promotes_atomically_and_records_provenance(tmp_path):
    store = RunStore(tmp_path, ctx=_ctx(), staged=True)
    assert store.root != store.final_root
    assert not store.final_root.exists()

    p = store.dir("domestic", "peru_gdp") / "fan.csv"
    pd.DataFrame({"a": [1]}).to_csv(p)
    store._track(p, "table")
    store.set_meta(timings={"data": 1.0}, stage_status={"data": "ok"})
    store.write_manifest()
    store.mark_success()
    final = store.promote()

    assert final == tmp_path / store.run_id
    assert (final / "_SUCCESS").exists()
    assert not (tmp_path / ".staging" / store.run_id).exists()

    man = json.loads((final / "manifest.json").read_text())
    assert man["code_version"] == "testver"
    assert man["environment"]["python"].startswith("3.")
    assert "pandas" in man["environment"]["packages"]
    assert man["registry_sha"] not in (None, "", "absent")
    assert man["stage_status"] == {"data": "ok"}
    assert man["timings"]["data"] == 1.0
    assert "seeds" in man
    assert "calibration_inputs" in man
    files = {f["path"]: f for f in man["files"]}
    entry = files["domestic/peru_gdp/fan.csv"]
    assert len(entry["sha256"]) == 64
    assert entry["bytes"] > 0


def test_manifest_validation_fails_closed_on_missing_artifact(tmp_path):
    store = RunStore(tmp_path, ctx=_ctx(), staged=True)
    ghost = store.dir("domestic") / "ghost.csv"
    store._track(ghost, "table")                      # tracked but never written
    with pytest.raises(RuntimeError, match="ghost.csv"):
        store.write_manifest()
    store.write_manifest(strict=False)                # the failure path may record


def test_promote_requires_the_success_marker(tmp_path):
    store = RunStore(tmp_path, ctx=_ctx(), staged=True)
    store.write_manifest()
    with pytest.raises(RuntimeError, match="_SUCCESS"):
        store.promote()
    assert not store.final_root.exists()


def test_promote_refuses_to_overwrite_a_promoted_run(tmp_path):
    (tmp_path / "2026-08-04__test").mkdir(parents=True)
    store = RunStore(tmp_path, ctx=_ctx(), staged=True)
    store.write_manifest()
    store.mark_success()
    with pytest.raises(RuntimeError, match="exists"):
        store.promote()


def test_aborted_run_is_quarantined_not_promoted(tmp_path):
    store = RunStore(tmp_path, ctx=_ctx(), staged=True)
    store.abort("boom: stage forecast failed")
    assert not store.final_root.exists()
    assert not (tmp_path / "latest").exists()
    marker = tmp_path / ".staging" / store.run_id / "_FAILED"
    assert marker.exists()
    assert "boom" in marker.read_text()


def test_latest_updates_only_after_promotion(tmp_path):
    store = RunStore(tmp_path, ctx=_ctx(), staged=True)
    store.write_manifest()
    with pytest.raises(RuntimeError, match="promote"):
        store.update_latest_symlink()
    store.mark_success()
    store.promote()
    store.update_latest_symlink()
    assert (tmp_path / "latest").resolve() == (tmp_path / store.run_id).resolve()


def test_prior_bundle_fallback_ignores_unpromoted_runs(tmp_path):
    """A bundle inside a run without _SUCCESS must be invisible to fallback."""
    from pipeline.lib.bundle import BundleError, load_prior_bundle

    run = tmp_path / "2026-08-01__000000"
    (run / "blocks").mkdir(parents=True)
    (run / "blocks" / "bundle.json").write_text(json.dumps({"schema": 1}))

    with pytest.raises(BundleError) as err:
        load_prior_bundle(tmp_path)
    assert "2026-08-01__000000" not in str(err.value)   # never even considered

    (run / "_SUCCESS").write_text("t")                  # now promoted: considered
    with pytest.raises(BundleError) as err:
        load_prior_bundle(tmp_path)
    assert "2026-08-01__000000" in str(err.value)


def test_staging_directory_is_never_scanned_for_bundles(tmp_path):
    from pipeline.lib.bundle import BundleError, load_prior_bundle

    ghost = tmp_path / ".staging" / "2026-08-02__000000" / "blocks"
    ghost.mkdir(parents=True)
    (ghost / "bundle.json").write_text(json.dumps({"schema": 1}))
    (ghost.parent / "_SUCCESS").write_text("t")         # even marked: still staged
    with pytest.raises(BundleError) as err:
        load_prior_bundle(tmp_path)
    assert "no prior run has a bundle.json" in str(err.value)
