"""Task 22: production never imports notebook code or reads notebook caches.

Two enforcement layers:

1. A source scan: no string literal in production packages may reference
   ``notebooks`` (docstrings exempt; comments are invisible to the AST). This
   kills both ``sys.path.insert(REPO / "notebooks/...")`` imports and
   ``rglob`` reads of notebook output caches.
2. Frozen calibration assets: the research outputs production consumes live in
   ``calibration/`` with sha256 pinned in the tracked MANIFEST.json; the loader
   verifies the hash so a lab rerun can never silently change production.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PRODUCTION = ("pipeline", "forecast", "core", "targets", "nowcast")


def _non_docstring_strings(tree: ast.AST):
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docs.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docs:
            yield node.value


def test_production_code_never_references_notebooks():
    offenders = []
    for pkg in PRODUCTION:
        for f in sorted((REPO / pkg).rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            for s in _non_docstring_strings(ast.parse(f.read_text())):
                if "notebooks" in s:
                    offenders.append(f"{f.relative_to(REPO)}: {s!r}")
    assert not offenders, \
        "production references notebook paths:\n" + "\n".join(offenders)


def test_asset_loader_verifies_the_frozen_hash(tmp_path):
    from pipeline.lib.calibration_assets import ASSETS, asset_path

    name, fname = next(iter(ASSETS.items()))
    (tmp_path / fname).write_bytes(b"frozen bytes")
    sha = hashlib.sha256(b"frozen bytes").hexdigest()
    (tmp_path / "MANIFEST.json").write_text(json.dumps(
        {"assets": {fname: {"sha256": sha, "bytes": 12}}}))

    assert asset_path(name, root=tmp_path) == tmp_path / fname

    (tmp_path / fname).write_bytes(b"tampered bytes")     # lab rerun, say
    with pytest.raises(RuntimeError, match="hash mismatch"):
        asset_path(name, root=tmp_path)

    (tmp_path / fname).unlink()
    with pytest.raises(RuntimeError, match="missing"):
        asset_path(name, root=tmp_path)


def test_every_declared_asset_is_frozen_and_intact():
    from pipeline.lib.calibration_assets import ASSETS, ROOT, asset_path

    present = [f for f in ASSETS.values() if (ROOT / f).exists()]
    if not (ROOT / "MANIFEST.json").exists() or not present:
        # a public clone tracks the manifest but not the binaries
        pytest.skip("calibration binaries not present (public clone)")
    # once ANY binary is present the freeze must be complete and intact
    for name in ASSETS:
        p = asset_path(name)                    # raises on mismatch or absence
        assert p.stat().st_size > 0
