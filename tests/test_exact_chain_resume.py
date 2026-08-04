"""F5: the exact-chain harness resumes only under an identical configuration.

Every artifact carries a fingerprint (code, dependencies, registry, frozen
calibration inputs, model specification, release rules, seeds, draw counts,
origin day). Resuming with ANY difference refuses instead of silently merging
incompatible rows; a legacy artifact without a fingerprint refuses too; the
checks file must list exactly the parquet's completed origins. The harness's
Monte-Carlo settings are reduced relative to production, so the claim is named
``exact_rule_reduced_mc`` and can never be described as computationally
identical to the live rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.lib import exact_chain as ec


def _art(tmp_path, bases=("2019Q1", "2019Q2"), fp=None, checks_bases=None):
    out = tmp_path / "exact_chain.parquet"
    rows = [{"base": b, "ref": b, "h": 1, "model": "S1-chain",
             "y_hat": 1.0, "origin": "2019-05-01"} for b in bases]
    pd.DataFrame(rows).to_parquet(out)
    if fp is not None:
        out.with_name("exact_chain_fingerprint.json").write_text(
            json.dumps(fp, indent=2, default=str))
    cb = bases if checks_bases is None else checks_bases
    out.with_name("exact_chain_checks.json").write_text(
        json.dumps({b: {"origin": "x"} for b in cb}))
    return out


def test_fingerprint_names_the_reduced_mc_claim():
    fp = ec.fingerprint(bases=["2019Q1"], panel_sha="test")
    assert fp["claim"] == "exact_rule_reduced_mc"
    for key in ("code_version", "dependencies", "registry_sha",
                "calibration_assets", "model_spec", "release_rules", "seeds",
                "draws", "origin_day", "evaluation_regime"):
        assert key in fp, f"fingerprint lacks {key}"
    assert "exact_chain.parquet" not in str(fp["calibration_assets"]), (
        "the artifact's own frozen copy must not enter its input fingerprint")


def test_legacy_artifact_without_fingerprint_refuses_resume(tmp_path):
    out = _art(tmp_path)
    fp = ec.fingerprint(bases=["2019Q3"], panel_sha="test")
    with pytest.raises(ec.ExactChainConfigError, match="fingerprint"):
        ec._load_resume_state(out, fp)


def test_differing_fingerprint_refuses_and_names_the_key(tmp_path):
    fp = ec.fingerprint(bases=["2019Q1", "2019Q2"], panel_sha="test")
    stale = json.loads(json.dumps(fp, default=str))
    stale["seeds"] = {"tot_chains": [99]}
    out = _art(tmp_path, fp=stale)
    with pytest.raises(ec.ExactChainConfigError, match="seeds"):
        ec._load_resume_state(out, fp)


def test_matching_fingerprint_resumes_with_done_set(tmp_path):
    fp = ec.fingerprint(bases=["2019Q1", "2019Q2"], panel_sha="test")
    out = _art(tmp_path, fp=json.loads(json.dumps(fp, default=str)))
    frames, done = ec._load_resume_state(out, fp)
    assert done == {"2019Q1", "2019Q2"}
    assert len(frames) == 1


def test_requested_bases_may_extend_without_refusal(tmp_path):
    fp_old = ec.fingerprint(bases=["2019Q1", "2019Q2"], panel_sha="test")
    out = _art(tmp_path, fp=json.loads(json.dumps(fp_old, default=str)))
    fp_new = ec.fingerprint(bases=["2019Q1", "2019Q2", "2019Q3"],
                            panel_sha="test")
    frames, done = ec._load_resume_state(out, fp_new)   # extension is fine
    assert done == {"2019Q1", "2019Q2"}


def test_checks_parquet_incoherence_refuses(tmp_path):
    fp = ec.fingerprint(bases=["2019Q1", "2019Q2"], panel_sha="test")
    out = _art(tmp_path, fp=json.loads(json.dumps(fp, default=str)),
               checks_bases=("2019Q1",))                # missing 2019Q2
    with pytest.raises(ec.ExactChainConfigError, match="checks"):
        ec._load_resume_state(out, fp)


def test_supersede_moves_the_old_artifact_aside(tmp_path):
    fp = ec.fingerprint(bases=["2019Q1"], panel_sha="test")
    out = _art(tmp_path, fp=json.loads(json.dumps(fp, default=str)))
    moved = ec.supersede(out)
    assert not out.exists()
    assert moved.exists() and "superseded" in moved.name
    assert not out.with_name("exact_chain_checks.json").exists()
