"""Task 5: fail closed on incoherent fallback.

A failed satellite must never make Peru read global product files of unknown
vintage. Either the current run published every required block, or ONE
complete prior bundle passes every compatibility check, or publication stops.
Mixing blocks across runs is impossible by construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pipeline.lib.bundle import (BundleError, REQUIRED_BLOCKS, load_prior_bundle,
                                 registry_sha, resolve_block_paths, write_bundle)

FNAMES = {"usa": "us_path_uncertainty.csv",
          "china": "china_path_uncertainty.csv",
          "commodities": "tot_path_uncertainty.csv"}


def _ctx(run_id="2026-08-01__100000", code="abc123"):
    return SimpleNamespace(as_of=pd.Timestamp("2026-08-01"), run_id=run_id,
                           code_version=code)


def _contract_frame(n=8, start="2026Q3"):
    q = pd.period_range(start, periods=n, freq="Q")
    return pd.DataFrame({
        "quarter": [str(p) for p in q], "h": range(1, n + 1), "source": "test",
        "centre": 1.0, "mode": 1.0, "s": 0.5, "gamma": 0.0,
        "sigma_left": 0.5, "sigma_right": 0.5})


def _make_run(runs_dir: Path, run_id: str, code="abc123", frame=None) -> Path:
    blocks_dir = runs_dir / run_id / "blocks"
    blocks_dir.mkdir(parents=True)
    published = {}
    for name, fname in FNAMES.items():
        df = frame if frame is not None else _contract_frame()
        df.to_csv(blocks_dir / fname, index=False)
        published[name] = blocks_dir / fname
    assert write_bundle(blocks_dir, published, _ctx(run_id, code)) is not None
    # T21: only PROMOTED runs (carrying _SUCCESS) are fallback candidates
    (runs_dir / run_id / "_SUCCESS").write_text("t")
    return runs_dir / run_id


def test_write_bundle_refuses_incomplete_sets(tmp_path):
    blocks_dir = tmp_path / "r" / "blocks"
    blocks_dir.mkdir(parents=True)
    df = _contract_frame()
    df.to_csv(blocks_dir / FNAMES["usa"], index=False)
    assert write_bundle(blocks_dir, {"usa": blocks_dir / FNAMES["usa"]}, _ctx()) is None
    assert not (blocks_dir / "bundle.json").exists()


def test_valid_bundle_roundtrip(tmp_path):
    _make_run(tmp_path, "2026-08-01__100000")
    got = load_prior_bundle(tmp_path, current_code_version="abc123")
    meta = got.pop("__meta__")
    assert set(got) == set(REQUIRED_BLOCKS)
    assert meta["run_id"] == "2026-08-01__100000"
    assert meta["registry_sha"] == registry_sha()


def test_tampered_artifact_is_rejected(tmp_path):
    run = _make_run(tmp_path, "2026-08-01__100000")
    p = run / "blocks" / FNAMES["china"]
    df = pd.read_csv(p); df.loc[0, "centre"] = 99.0; df.to_csv(p, index=False)
    with pytest.raises(BundleError, match="hash mismatch"):
        load_prior_bundle(tmp_path, current_code_version="abc123")


def test_nan_in_required_columns_is_rejected(tmp_path):
    frame = _contract_frame()
    frame.loc[3, "s"] = float("nan")
    _make_run(tmp_path, "2026-08-01__100000", frame=frame)
    with pytest.raises(BundleError, match="NaN"):
        load_prior_bundle(tmp_path, current_code_version="abc123")


def test_code_version_mismatch_is_rejected(tmp_path):
    _make_run(tmp_path, "2026-08-01__100000", code="oldcode")
    with pytest.raises(BundleError, match="code version"):
        load_prior_bundle(tmp_path, current_code_version="newcode")


def test_registry_change_is_rejected(tmp_path):
    run = _make_run(tmp_path, "2026-08-01__100000")
    bfile = run / "blocks" / "bundle.json"
    man = json.loads(bfile.read_text())
    man["registry_sha"] = "0" * 64
    bfile.write_text(json.dumps(man))
    with pytest.raises(BundleError, match="registry"):
        load_prior_bundle(tmp_path, current_code_version="abc123")


def test_newest_valid_bundle_wins_over_newer_broken_one(tmp_path):
    _make_run(tmp_path, "2026-07-25__090000")
    newer = _make_run(tmp_path, "2026-08-01__100000")
    (newer / "blocks" / FNAMES["usa"]).unlink()          # newer bundle broken
    got = load_prior_bundle(tmp_path, current_code_version="abc123")
    assert got["__meta__"]["run_id"] == "2026-07-25__090000"


def test_current_run_is_never_its_own_fallback(tmp_path):
    _make_run(tmp_path, "2026-08-01__100000")
    with pytest.raises(BundleError):
        load_prior_bundle(tmp_path, exclude_run_id="2026-08-01__100000",
                          current_code_version="abc123")


def test_resolver_passes_through_a_complete_current_run(tmp_path):
    run = _make_run(tmp_path, "2026-08-01__100000")
    blocks = {n: run / "blocks" / f for n, f in FNAMES.items()}
    paths, meta = resolve_block_paths(blocks, tmp_path, ctx=_ctx())
    assert meta is None and set(paths) == set(REQUIRED_BLOCKS)


def test_resolver_never_mixes_runs(tmp_path):
    prior = _make_run(tmp_path, "2026-07-25__090000")
    current = _make_run(tmp_path, "2026-08-01__100000")
    # current run only published two of three blocks
    blocks = {"usa": current / "blocks" / FNAMES["usa"],
              "china": current / "blocks" / FNAMES["china"]}
    paths, meta = resolve_block_paths(blocks, tmp_path, ctx=_ctx("2026-08-01__100000"))
    assert meta["run_id"] == "2026-07-25__090000"
    for name in REQUIRED_BLOCKS:                       # ALL from the prior bundle
        assert str(paths[name]).startswith(str(prior))


def test_resolver_aborts_when_nothing_coherent_exists(tmp_path):
    current = _make_run(tmp_path, "2026-08-01__100000")
    blocks = {"usa": current / "blocks" / FNAMES["usa"]}
    with pytest.raises(BundleError, match="[Pp]ublication"):
        resolve_block_paths(blocks, tmp_path, ctx=_ctx("2026-08-01__100000"))
