"""F4: production fan calibration reads ONE frozen, hash-verified input.

`production_fits` previously branched on ``output/backtests/exact_chain.parquet``
(ignored, mutable, unverified): published bands changed with local file
presence. The exact-chain errors are now a frozen calibration asset with full
provenance in calibration/MANIFEST.json; the silent existence branch is gone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]

from pipeline.lib.calibration_assets import ROOT, ASSETS  # noqa: E402


def test_exact_chain_is_a_declared_asset_with_provenance():
    assert "exact_chain" in ASSETS, "exact_chain must be a frozen calibration asset"
    man = json.loads((ROOT / "MANIFEST.json").read_text())["assets"]
    meta = man.get(ASSETS["exact_chain"])
    assert meta is not None, "exact_chain missing from calibration/MANIFEST.json"
    for field in ("sha256", "bytes", "producer", "evaluation_regime"):
        assert meta.get(field), f"exact_chain manifest lacks {field}"


def test_no_silent_existence_branch_remains():
    import pipeline.lib.fan_calibration as fc

    assert not hasattr(fc, "CHAIN"), (
        "fan_calibration still carries the mutable CHAIN path; production "
        "must read the frozen asset only")
    src = Path(fc.__file__).read_text()
    assert "CHAIN.exists" not in src


def test_production_fits_identical_without_output_backtests():
    if not (REPO / "input").exists():
        pytest.skip("private data layer absent (public clone)")
    if not (ROOT / ASSETS.get("exact_chain", "exact_chain.parquet")).exists():
        pytest.skip("exact_chain asset not frozen yet")
    from pipeline.lib.fan_calibration import production_fits

    d1a, d30a = production_fits("2026-08-03", H=7)

    bt_dir = REPO / "output" / "backtests"
    moved = bt_dir.with_name("backtests_MOVED_BY_TEST")
    if not bt_dir.exists():
        pytest.skip("output/backtests absent; nothing to prove")
    bt_dir.rename(moved)
    try:
        d1b, d30b = production_fits("2026-08-03", H=7)
    finally:
        moved.rename(bt_dir)
    for a, b in ((d1a, d1b), (d30a, d30b)):
        assert set(a) == set(b)
        for h in a:
            assert a[h]["sigma1"] == pytest.approx(b[h]["sigma1"], abs=1e-12), (
                "production fan parameters depend on the presence of the "
                "mutable output/backtests directory")
