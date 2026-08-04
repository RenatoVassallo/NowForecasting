"""F2: the staged run directory is the ONLY production information surface.

Blocks and stages write inside ``store.root`` and consume run-local paths;
nothing in the current run reads or writes the global ``products/`` surface.
Publication is one controlled step AFTER promotion (``pipeline/lib/publish.py``),
so a failed or concurrent run can never partially update published outputs.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]

# every module that participates in a production run; publish.py is the ONE
# sanctioned owner of the products/ surface
SCANNED = [p for p in [*(REPO / "pipeline" / "blocks").glob("*.py"),
                       *(REPO / "pipeline" / "stages").glob("*.py"),
                       *(REPO / "pipeline" / "lib").glob("*.py")]
           if p.name != "publish.py" and "__pycache__" not in p.parts]


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


def test_no_run_participant_touches_the_products_surface():
    offenders = []
    for f in SCANNED:
        for s in _non_docstring_strings(ast.parse(f.read_text())):
            if "products" in s:
                offenders.append(f"{f.relative_to(REPO)}: {s!r}")
    assert not offenders, (
        "run participants reference the global products/ surface:\n"
        + "\n".join(offenders))


def _fake_builder(tag):
    def build(ctx=None, out_dir=None, **_):
        if out_dir is None:
            raise ValueError("out_dir is required")
        df = pd.DataFrame({"quarter": ["2026Q3"], "h": [1], "source": [tag],
                           "centre": [1.0], "mode": [1.0], "s": [0.5],
                           "gamma": [0.0], "sigma_left": [0.5],
                           "sigma_right": [0.5]})
        p = Path(out_dir) / f"{tag}_path_uncertainty.csv"
        df.to_csv(p, index=False)
        return df, [f"- {tag} ok"], p
    return build


def test_interleaved_runs_never_share_block_paths(tmp_path, monkeypatch):
    import pipeline.stages.chain as chain
    from pipeline.lib.context import RunContext
    from pipeline.lib.store import RunStore

    fakes = {n: (_fake_builder(n), ("peru",)) for n in ("usa", "china", "commodities")}
    monkeypatch.setattr(chain, "BLOCKS", fakes)
    monkeypatch.setattr(chain, "ORDER", tuple(fakes))

    stores = {}
    for rid in ("2026-08-04__A", "2026-08-04__B"):
        ctx = RunContext(as_of=pd.Timestamp("2026-08-04"), run_id=rid,
                         code_version="t")
        st = RunStore(tmp_path, ctx=ctx, staged=True)

        class P:  # minimal params stub
            CHAIN_BLOCKS = ()
        chain.run(st, P)
        stores[rid] = st

    for rid, st in stores.items():
        assert set(st.blocks) == {"usa", "china", "commodities"}
        for name, path in st.blocks.items():
            path = Path(path)
            assert str(path).startswith(str(st.root)), (
                f"{rid}: {name} block path {path} is outside the run root")
            assert pd.read_csv(path)["source"].iloc[0] == name
    a, b = stores.values()
    assert {str(p) for p in a.blocks.values()}.isdisjoint(
        {str(p) for p in b.blocks.values()})


def test_publication_never_writes_the_deprecated_flat_surface(tmp_path):
    """G4: the authoritative surface is products/published/<run_id> plus the
    products/latest pointer; the old flat generated paths are deprecated and
    a publication writes NOTHING at the products root."""
    import hashlib as _h
    import json as _j

    from pipeline.lib.publish import publish_run

    run = tmp_path / "2026-08-04__flat"
    run.mkdir()
    content = b"fan\n"
    (run / "peru_gdp_fan.csv").write_bytes(content)
    (run / "manifest.json").write_text(_j.dumps({
        "run_id": run.name, "as_of": "2026-08-04", "code_version": "v",
        "required": [], "files": [{"path": "peru_gdp_fan.csv",
                                   "sha256": _h.sha256(content).hexdigest(),
                                   "bytes": len(content)}]}))
    (run / "_SUCCESS").write_text("t")

    products = tmp_path / "products"
    products.mkdir()
    (products / "builders.py").write_text("SOURCE = True\n")   # package code
    (products / "peru_gdp_fan.csv").write_text("LEGACY FLAT COPY")

    out = publish_run(run, products_dir=products)
    root_entries = {p.name for p in products.iterdir()}
    assert root_entries == {"builders.py", "peru_gdp_fan.csv",
                            "published", "latest"}
    assert (products / "peru_gdp_fan.csv").read_text() == "LEGACY FLAT COPY", (
        "the deprecated flat copy must be left frozen, not rewritten")
    assert (products / "builders.py").read_text() == "SOURCE = True\n"
    assert (products / "latest" / "peru_gdp_fan.csv").read_bytes() == content
    assert out.name == run.name
