"""F9: fast-mode backtest reuse accepts only coherent promoted evidence.

``RUN_BACKTEST=False`` reuses a prior run's member nowcast frame. The old
selector took the newest directory with the right filename; it now requires a
PROMOTED run (``_SUCCESS``), an as-of no later than the current run, matching
registry hash, code version, evaluation regime, target, member set and
schema, and an artifact whose sha256 equals the prior manifest's record. The
reused run and hash are stamped into the new manifest.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from pipeline.lib.context import RunContext
from pipeline.lib.store import RunStore
from pipeline.stages._models import _previous_backtest

MEMBERS = ("RW", "Q-AR", "Bridge(leaders)", "P-MIDAS(leaders)")


def _frame(members=MEMBERS, target="g_pbiq"):
    rows = []
    for m in list(members) + ["Adaptive-IC"]:
        rows.append({"target": target, "ref_quarter": "2026-03-01",
                     "origin_date": "2026-04-01", "days_to_publication": -30,
                     "model": m, "y_true": 2.0, "y_hat": 2.1, "y_std": None})
        rows.append({"target": target, "ref_quarter": "2026-06-01",
                     "origin_date": "2026-07-01", "days_to_publication": -30,
                     "model": m, "y_true": None, "y_hat": 2.2, "y_std": None})
    return pd.DataFrame(rows)


def _prior_run(tmp_path, rid, *, as_of="2026-08-01", code="cur", reg="REG",
               promoted=True, frame=None, manifest_sha=True):
    run = tmp_path / rid
    d = run / "domestic" / "peru_gdp"
    d.mkdir(parents=True)
    f = d / "nowcasts.parquet"
    (frame if frame is not None else _frame()).to_parquet(f)
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    man = {"run_id": rid, "as_of": as_of, "status": "success",
           "code_version": code, "registry_sha": reg,
           "evaluation_regime": "pseudo_real_time_final_vintage",
           "files": [{"path": "domestic/peru_gdp/nowcasts.parquet",
                      "sha256": sha if manifest_sha else "0" * 64}]}
    (run / "manifest.json").write_text(json.dumps(man))
    if promoted:
        (run / "_SUCCESS").write_text("t")
    return run


def _store(tmp_path, monkeypatch, as_of="2026-08-04", code="cur", reg="REG"):
    import pipeline.stages._models as M

    monkeypatch.setattr(M, "_current_registry_sha", lambda: reg)
    ctx = RunContext(as_of=pd.Timestamp(as_of), run_id=f"{as_of}__cur",
                     code_version=code)
    return RunStore(tmp_path, ctx=ctx, staged=True)


def _call(store):
    return _previous_backtest(store, "domestic", "peru_gdp",
                              expected_target="g_pbiq", members=set(MEMBERS))


def test_valid_prior_run_is_reused_and_stamped(tmp_path, monkeypatch):
    _prior_run(tmp_path, "2026-08-01__ok")
    store = _store(tmp_path, monkeypatch)
    bt, rid = _call(store)
    assert rid == "2026-08-01__ok"
    assert set(bt.model) == set(MEMBERS)          # members only, realized only
    assert bt.y_true.notna().all()
    stamp = store._manifest["reused_backtests"]["domestic/peru_gdp"]
    assert stamp["run_id"] == rid and len(stamp["artifact_sha256"]) == 64


def test_unpromoted_and_future_runs_are_rejected(tmp_path, monkeypatch):
    _prior_run(tmp_path, "2026-08-02__unpromoted", promoted=False)
    _prior_run(tmp_path, "2026-09-01__future", as_of="2026-09-01")
    store = _store(tmp_path, monkeypatch)
    bt, rid = _call(store)
    assert bt is None and rid is None


def test_incompatible_runs_are_rejected(tmp_path, monkeypatch):
    _prior_run(tmp_path, "2026-08-01__code", code="OTHER")
    _prior_run(tmp_path, "2026-07-31__reg", reg="OTHERREG")
    _prior_run(tmp_path, "2026-07-30__members",
               frame=_frame(members=("RW", "Q-AR")))
    _prior_run(tmp_path, "2026-07-29__target", frame=_frame(target="other"))
    _prior_run(tmp_path, "2026-07-28__tamper", manifest_sha=False)
    store = _store(tmp_path, monkeypatch)
    bt, rid = _call(store)
    assert bt is None and rid is None


def test_newest_valid_wins_over_newer_invalid(tmp_path, monkeypatch):
    _prior_run(tmp_path, "2026-08-02__bad", code="OTHER")
    _prior_run(tmp_path, "2026-08-01__good")
    store = _store(tmp_path, monkeypatch)
    bt, rid = _call(store)
    assert rid == "2026-08-01__good"
