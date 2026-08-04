"""F3: every final artifact is tracked, and _SUCCESS means the contract held.

Strict manifest mode compares the ACTUAL run tree against the declared
artifacts: untracked files, missing or empty tracked files, missing REQUIRED
artifacts, duplicate manifest paths, and paths outside the run directory all
fail the run before it can be promoted. Figure and report-PDF failures fail
their stages instead of degrading silently.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.lib.context import RunContext
from pipeline.lib.store import RunStore


def _store(tmp_path, rid="2026-08-04__c"):
    ctx = RunContext(as_of=pd.Timestamp("2026-08-04"), run_id=rid,
                     code_version="t")
    return RunStore(tmp_path, ctx=ctx, staged=True)


def test_untracked_final_artifact_fails_strict_manifest(tmp_path):
    store = _store(tmp_path)
    p = store.dir("domestic") / "tracked.csv"
    pd.DataFrame({"a": [1]}).to_csv(p)
    store._track(p, "table")
    (store.root / "stray_final_output.csv").write_text("orphan")
    with pytest.raises(RuntimeError, match="untracked"):
        store.write_manifest()
    store.write_manifest(strict=False)          # failure path may still record


def test_duplicate_manifest_paths_are_rejected(tmp_path):
    store = _store(tmp_path)
    p = store.dir("d") / "x.csv"
    pd.DataFrame({"a": [1]}).to_csv(p)
    store._track(p, "table")
    store._track(p, "table")
    with pytest.raises(RuntimeError, match="duplicate"):
        store.write_manifest()


def test_tracking_outside_the_run_directory_is_impossible(tmp_path):
    store = _store(tmp_path)
    outside = tmp_path / "elsewhere.csv"
    outside.write_text("x")
    with pytest.raises(ValueError):
        store._track(outside, "table")


def test_missing_required_artifact_fails_strict_manifest(tmp_path):
    store = _store(tmp_path)
    store.require("peru_gdp_fan.csv")
    with pytest.raises(RuntimeError, match="required.*peru_gdp_fan"):
        store.write_manifest()
    p = store.root / "peru_gdp_fan.csv"
    pd.DataFrame({"a": [1]}).to_csv(p)
    store._track(p, "fan")
    store.write_manifest()                       # satisfied now
    man = json.loads((store.root / "manifest.json").read_text())
    assert "peru_gdp_fan.csv" in man["required"]


def test_figure_failure_fails_the_fanchart_stage(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    import pipeline.stages.fanchart as fc

    store = _store(tmp_path)

    def good(ctx):
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        return fig

    def bad(ctx):
        raise ValueError("synthetic figure failure")

    monkeypatch.setattr(fc, "FIGURES", {"good": good, "bad": bad})
    monkeypatch.setattr(fc, "load_context", lambda as_of=None, store=None: {})

    class P:
        pass

    with pytest.raises(RuntimeError, match="bad"):
        fc.run(store, P)
    assert (store.root / "figures" / "good.pdf").exists()   # built before the raise


def test_report_pdf_mode_fails_without_a_compiler(tmp_path, monkeypatch):
    import shutil as _sh

    from pipeline.stages.report import _compile_pdf

    monkeypatch.setattr(_sh, "which", lambda cmd: None)
    with pytest.raises(RuntimeError, match="PDF"):
        _compile_pdf(tmp_path, require_pdf=True)
    assert _compile_pdf(tmp_path, require_pdf=False) is False
