"""G4 (P1): publication is transactional, versioned, and hash-verified.

The old publish step copied files into the flat products/ tree one by one, so
an interruption left mixed vintages and stale figures survived forever. Now:
the complete declared surface is staged into a temporary directory, every
file's size and sha256 are verified against the PROMOTED run's manifest, a
publication manifest is written, the directory is atomically renamed to
``products/published/<run_id>/``, and only then does the single authoritative
``products/latest`` pointer switch. Any failure leaves the previous
publication untouched. Flat generated paths are deprecated; source code under
products/ is never touched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.lib import publish as pub


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _run(tmp_path, rid="2026-08-04__A", *, promoted=True, tamper=None,
         drop=None):
    run = tmp_path / rid
    (run / "blocks").mkdir(parents=True)
    (run / "figures").mkdir()
    files = {"blocks/us_path_uncertainty.csv": b"us,1\n",
             "blocks/bundle.json": b"{\"schema\": 1}\n",
             "peru_gdp_fan.csv": b"fan,2\n",
             "report.pdf": b"%PDF fake\n",
             "report.md": b"# report\n",
             "figures/fan_main.pdf": b"%PDF fig\n"}
    man = {"run_id": rid, "as_of": "2026-08-04", "code_version": "vX",
           "status": "success", "required": ["peru_gdp_fan.csv", "report.pdf"],
           "files": []}
    for rel, content in files.items():
        p = run / rel
        p.write_bytes(content)
        man["files"].append({"path": rel, "sha256": _sha(content),
                             "bytes": len(content)})
    if tamper:
        (run / tamper).write_bytes(b"TAMPERED AFTER PROMOTION")
    if drop:
        (run / drop).unlink()
    (run / "manifest.json").write_text(json.dumps(man))
    if promoted:
        (run / "_SUCCESS").write_text("t")
    return run


def test_success_publishes_versioned_dir_and_switches_pointer(tmp_path):
    run = _run(tmp_path)
    products = tmp_path / "products"
    out = pub.publish_run(run, products_dir=products)
    assert out == products / "published" / run.name
    assert (out / "peru_gdp_fan.csv").read_bytes() == b"fan,2\n"
    assert (products / "latest").resolve() == out.resolve()
    pman = json.loads((out / "publication_manifest.json").read_text())
    assert pman["run_id"] == run.name
    assert pman["as_of"] == "2026-08-04"
    assert pman["code_version"] == "vX"
    assert pman["files"]["peru_gdp_fan.csv"] == _sha(b"fan,2\n")


def test_unpromoted_run_refuses(tmp_path):
    run = _run(tmp_path, promoted=False)
    with pytest.raises(RuntimeError, match="_SUCCESS"):
        pub.publish_run(run, products_dir=tmp_path / "products")


def test_hash_mismatch_refuses_and_pointer_survives(tmp_path):
    products = tmp_path / "products"
    old = pub.publish_run(_run(tmp_path, "2026-08-03__old"), products_dir=products)
    run = _run(tmp_path, "2026-08-04__bad", tamper="peru_gdp_fan.csv")
    with pytest.raises(RuntimeError, match="sha256|hash"):
        pub.publish_run(run, products_dir=products)
    assert (products / "latest").resolve() == old.resolve()
    assert not (products / "published" / "2026-08-04__bad").exists()


def test_missing_declared_artifact_refuses(tmp_path):
    run = _run(tmp_path, drop="report.pdf")
    with pytest.raises(RuntimeError, match="report.pdf"):
        pub.publish_run(run, products_dir=tmp_path / "products")


def test_injected_copy_failure_leaves_previous_latest(tmp_path, monkeypatch):
    products = tmp_path / "products"
    old = pub.publish_run(_run(tmp_path, "2026-08-03__old"), products_dir=products)

    calls = {"n": 0}
    real = pub._copy_verified

    def flaky(src, dest, expected):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk full (injected)")
        return real(src, dest, expected)

    monkeypatch.setattr(pub, "_copy_verified", flaky)
    with pytest.raises(OSError, match="injected"):
        pub.publish_run(_run(tmp_path, "2026-08-04__new"), products_dir=products)
    assert (products / "latest").resolve() == old.resolve()
    assert not (products / "published" / "2026-08-04__new").exists()
    stray = [p for p in (products / "published").iterdir()
             if p.name.startswith(".tmp")]
    assert not stray, "failed staging must not linger"


def test_new_release_does_not_inherit_obsolete_files(tmp_path):
    products = tmp_path / "products"
    old = pub.publish_run(_run(tmp_path, "2026-08-03__old"), products_dir=products)
    (old / "figures" / "legacy_only.png").write_bytes(b"old fig")
    new = pub.publish_run(_run(tmp_path, "2026-08-04__new"), products_dir=products)
    assert not (new / "figures" / "legacy_only.png").exists()
    assert (products / "latest" / "figures" / "fan_main.pdf").exists()


def test_republishing_the_same_run_id_refuses(tmp_path):
    products = tmp_path / "products"
    run = _run(tmp_path, "2026-08-04__A")
    pub.publish_run(run, products_dir=products)
    with pytest.raises(RuntimeError, match="already"):
        pub.publish_run(run, products_dir=products)
