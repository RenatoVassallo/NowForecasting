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
    # lifecycle smoke only: an all-off run has no publishable surface
    monkeypatch.setattr(params, "PUBLISH_PRODUCTS", False)

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


def _fake_stage_run(tmp_path, monkeypatch, *, publish=True, break_publish=None):
    """Drive the REAL main() lifecycle with faked stage bodies producing a
    minimal but contract-complete artifact surface."""
    import pipeline.lib.inputs as inputs
    import pipeline.lib.preflight as pf
    import pipeline.stages.chain as s_chain
    import pipeline.stages.domestic as s_dom
    import pipeline.stages.fanchart as s_fan
    import pipeline.stages.report as s_rep
    import pipeline.stages.satellites as s_sat
    from pipeline.config import params

    products = tmp_path / "products"
    monkeypatch.setattr(params, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(params, "STAGES",
                        {"data": False, "nowcast": True, "forecast": True,
                         "fanchart": True, "report": True})
    monkeypatch.setattr(params, "UPDATE_LATEST_SYMLINK", True)
    monkeypatch.setattr(params, "PUBLISH_PRODUCTS", publish)
    monkeypatch.setattr(params, "PUBLISH_DIR", products, raising=False)
    # belt AND braces: even if main ever ignores PUBLISH_DIR again, the
    # module default must point at the sandbox, never the real products/
    import pipeline.lib.publish as _pub
    monkeypatch.setattr(_pub, "PRODUCTS_DIR", products)
    monkeypatch.setattr(pf, "run_preflight", lambda store, params: None)
    monkeypatch.setattr(inputs, "pin_inputs",
                        lambda root=None: {"fake.parquet": {"sha256": "0" * 64,
                                                            "bytes": 1}})
    monkeypatch.setattr(inputs, "verify_inputs", lambda pins, root=None: None)

    def _w(store, rel, content, kind, require=True):
        p = store.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        store._track(p, kind)
        if require:
            store.require(rel)

    def sat(store, params_, panels):
        return ["- satellites ok"]

    def dom(store, params_, panels):
        _w(store, "peru_nowcast_official.csv", "as_of,value\n2026-08-04,2.7\n",
           "official-nowcast")
        _w(store, "peru_nowcast_sweep.csv", "as_of\n2026-08-04\n",
           "official-nowcast")
        return ["- domestic ok"]

    def chain(store, params_, panels):
        _w(store, "blocks/us_path_uncertainty.csv", "quarter\n2026Q3\n", "block")
        _w(store, "blocks/bundle.json", "{\"schema\": 1}", "bundle")
        store.blocks = {"usa": store.root / "blocks/us_path_uncertainty.csv"}
        return ["- chain ok"]

    def fan(store, params_):
        _w(store, "peru_gdp_fan.csv", "quarter,mode\n2026Q2,2.7\n", "fan")
        _w(store, "peru_gdp_model_paths.csv",
           "quarter,model,forecast\n2026Q2,S1-chain,2.7\n2026Q2,S2 exp-AR1,2.7\n",
           "prospective-forecast")
        return ["- fan ok"]

    def figs(store, params_, panels):
        _w(store, "figures/fan_main.pdf", "%PDF fig", "figure", require=False)
        store.fig_ctx = {}
        return ["- figures ok"]

    def rep(store, params_, whatsnew, lines, timings):
        _w(store, "report.tex", "\\tex", "report")
        _w(store, "report.md", "# r", "report")
        _w(store, "report.pdf", "%PDF report", "report")

    monkeypatch.setattr(s_sat, "run", sat)
    monkeypatch.setattr(s_dom, "run", dom)
    monkeypatch.setattr(s_chain, "run", chain)
    monkeypatch.setattr(s_dom, "run_peru_fan", fan)
    monkeypatch.setattr(s_fan, "run", figs)
    monkeypatch.setattr(s_rep, "run", rep)
    if break_publish is not None:
        import pipeline.lib.publish as pub
        monkeypatch.setattr(pub, "publish_run", break_publish)

    from pipeline.main import main

    return main(as_of="2026-08-04", run_id="2026-08-04__integration"), products


def test_normal_path_with_automatic_publication(tmp_path, monkeypatch):
    """H2: the DEFAULT production path (PUBLISH_PRODUCTS=True) must complete
    without raising after a successful publication."""
    root, products = _fake_stage_run(tmp_path, monkeypatch, publish=True)
    assert (root / "_SUCCESS").exists()
    assert (tmp_path / "runs" / "latest").resolve() == root.resolve()
    pub_dir = products / "published" / "2026-08-04__integration"
    assert (products / "latest").resolve() == pub_dir.resolve()
    pman = json.loads((pub_dir / "publication_manifest.json").read_text())
    assert pman["run_id"] == "2026-08-04__integration"
    assert "peru_gdp_model_paths.csv" in pman["files"]
    assert len(pman["files"]) >= 6


def test_publication_failure_after_promotion_preserves_both_pointers(
        tmp_path, monkeypatch):
    from pipeline.lib.publish import PublicationError

    def boom(root, products_dir=None):
        raise PublicationError("synthetic publication failure")

    with pytest.raises(PublicationError, match="synthetic"):
        _fake_stage_run(tmp_path, monkeypatch, publish=True, break_publish=boom)
    run = tmp_path / "runs" / "2026-08-04__integration"
    assert (run / "_SUCCESS").exists(), "the promoted run must be preserved"
    assert (tmp_path / "runs" / "latest").resolve() == run.resolve()
    assert not (tmp_path / "products" / "latest").exists()


def test_failed_run_is_quarantined(tmp_path, monkeypatch):
    from pipeline.config import params
    import pipeline.lib.preflight as pf

    monkeypatch.setattr(params, "RUNS_DIR", tmp_path)
    # a VALID selection (nowcast has no dependencies) whose preflight fails:
    # the quarantine path, not the config-validation path
    monkeypatch.setattr(params, "STAGES",
                        {k: False for k in params.STAGES} | {"nowcast": True})
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
