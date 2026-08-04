"""Task 4: the China block must recompute its live profile, never restamp one.

The defect: production read ``china_profile_fan.csv`` (a notebook cache), gave
it the current date and WEO round, and published it as the live path. The
contract now: the centre path is recomputed each run by
``pipeline.blocks._china_model.live_profile`` and required inputs raise when
missing instead of degrading into a stale product.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_no_cached_profile_is_read_anywhere_in_production():
    offenders = []
    for f in [*(REPO / "pipeline").rglob("*.py")]:
        if "china_profile_fan" in f.read_text():
            offenders.append(str(f.relative_to(REPO)))
    assert not offenders, (
        f"production reads the cached notebook profile again: {offenders}")


def test_missing_required_input_raises_instead_of_restamping(tmp_path):
    """The block's inputs are frozen assets: absence or tampering is fatal.

    T22 replaced the ad-hoc ``_require`` with the hash-verified calibration
    asset loader; the fail-closed contract lives there now.
    """
    import json

    from pipeline.lib.calibration_assets import asset_path

    with pytest.raises(RuntimeError, match="manifest missing"):
        asset_path("china_ladder_full", root=tmp_path)

    (tmp_path / "MANIFEST.json").write_text(json.dumps(
        {"assets": {"china_ladder_full.parquet": {"sha256": "0" * 64}}}))
    with pytest.raises(RuntimeError, match="missing"):
        asset_path("china_ladder_full", root=tmp_path)

    (tmp_path / "china_ladder_full.parquet").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        asset_path("china_ladder_full", root=tmp_path)


def test_provenance_contract_fields_are_declared():
    """The block's stamp must let a reader reconstruct WHAT produced the path."""
    import inspect

    from pipeline.blocks import _china_model as m

    src = inspect.getsource(m.live_profile)
    for key in ("weo_round", "model_members", "blend_alpha", "ladder_cache_sha",
                "horse_cache_sha", "panel_sha", "tilt_ess_share"):
        assert f'"{key}"' in src, f"provenance key {key} disappeared"
