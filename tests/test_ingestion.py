"""Task 10: ingestion defects stay fixed.

Covered: the BCRP adapter's call signature (the old positional form could
never run); the commodity refresh never replacing a complete cache with a
degraded frame; atomic cache writes; required-column validation; and the NBS
refresh path actually bypassing response caches.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]

if not (REPO / "sources").exists():
    pytest.skip("private data layer absent (public clone)",
                allow_module_level=True)


# --------------------------------------------------------------------------- #
# BCRP adapter contract
# --------------------------------------------------------------------------- #
def test_bcrp_call_matches_macropy_signature():
    MacroPy = pytest.importorskip("MacroPy")
    sig = inspect.signature(MacroPy.get_bcrp_data)
    # the repaired call binds cleanly
    sig.bind(["PN01654XM"], frequency="m", names=["gold"], start_period="1994-1")
    # the OLD positional pattern (codes, names, frequency) cannot bind: the
    # names list lands in ``frequency`` and start_period collides
    with pytest.raises(TypeError):
        sig.bind(["PN01654XM"], ["gold"], "m", start_period="1994-1",
                 frequency="m")


def test_bcrp_provider_uses_named_arguments_only():
    src = (REPO / "sources" / "bcrp.py").read_text()
    assert "get_bcrp_data(list(mapping), **kw_call)" in src
    assert "list(mapping), list(mapping.values())" not in src


# --------------------------------------------------------------------------- #
# atomic writes and validation
# --------------------------------------------------------------------------- #
def test_atomic_write_replaces_only_on_success(tmp_path):
    from sources.base import atomic_write_parquet

    p = tmp_path / "cache.parquet"
    atomic_write_parquet(pd.DataFrame({"a": [1.0]}), p)
    assert pd.read_parquet(p)["a"].iloc[0] == 1.0

    class Broken:
        def to_parquet(self, _):
            raise RuntimeError("disk full mid-write")

    with pytest.raises(RuntimeError):
        atomic_write_parquet(Broken(), p)
    assert pd.read_parquet(p)["a"].iloc[0] == 1.0     # prior cache intact


def test_validate_cache_blocks_degraded_frames():
    from sources.base import validate_cache

    prior = pd.DataFrame({"p_copper": [1.0], "pe_tot": [2.0]})
    fresh_ok = pd.DataFrame({"p_copper": [1.1], "pe_tot": [2.1]})
    assert validate_cache(fresh_ok, ("p_copper", "pe_tot"), prior=prior) == []

    missing = pd.DataFrame({"p_copper": [1.1]})
    probs = validate_cache(missing, ("p_copper", "pe_tot"), prior=prior)
    assert any("pe_tot" in p and "missing" in p for p in probs)

    emptied = pd.DataFrame({"p_copper": [1.1], "pe_tot": [np.nan]})
    probs = validate_cache(emptied, ("p_copper", "pe_tot"), prior=prior)
    assert any("entirely missing" in p for p in probs)


# --------------------------------------------------------------------------- #
# commodity refresh keeps the prior cache on partial failure
# --------------------------------------------------------------------------- #
@pytest.fixture()
def _commodities(tmp_path, monkeypatch):
    cmod = pytest.importorskip("sources.commodities")
    monkeypatch.setattr(cmod, "CACHE_DIR", tmp_path)
    idx = pd.date_range("2024-01-01", periods=6, freq="MS")
    full = pd.DataFrame({c: 1.0 for c in cmod.REQUIRED_COLUMNS}, index=idx)
    full.to_parquet(tmp_path / cmod._FILE)
    return cmod, tmp_path, full


def test_bcrp_outage_never_overwrites_the_cache(_commodities, monkeypatch):
    cmod, tmp_path, full = _commodities

    def boom():
        raise RuntimeError("BCRP 503")

    monkeypatch.setattr(cmod, "fetch_bcrp", boom)
    monkeypatch.setattr(cmod, "_fred", lambda m: full[["p_copper", "p_gold", "p_wti"]])
    msgs = cmod.refresh()
    assert any("prior cache kept" in m for m in msgs)
    kept = pd.read_parquet(tmp_path / cmod._FILE)
    assert list(kept.columns) == list(full.columns)   # nothing was lost


def test_degraded_frame_is_rejected_by_validation(_commodities, monkeypatch):
    cmod, tmp_path, full = _commodities
    fred_only = full[["p_copper", "p_gold", "p_wti"]]
    bcrp_empty = full[["pe_px", "pe_pm", "pe_tot"]].copy()
    bcrp_empty[:] = np.nan                             # reachable but empty

    monkeypatch.setattr(cmod, "_fred", lambda m: fred_only)
    monkeypatch.setattr(cmod, "fetch_bcrp", lambda: bcrp_empty)
    msgs = cmod.refresh()
    assert any("REJECTED" in m for m in msgs)
    kept = pd.read_parquet(tmp_path / cmod._FILE)
    assert kept["pe_tot"].notna().all()                # prior values intact


def test_successful_refresh_is_written_atomically(_commodities, monkeypatch):
    cmod, tmp_path, full = _commodities
    newer = full.copy()
    newer.loc[newer.index[-1], "p_copper"] = 9.9

    monkeypatch.setattr(cmod, "_fred", lambda m: newer[["p_copper", "p_gold", "p_wti"]])
    monkeypatch.setattr(cmod, "fetch_bcrp", lambda: newer[["pe_px", "pe_pm", "pe_tot"]])
    monkeypatch.setattr(cmod, "fetch_daily",
                        lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    msgs = cmod.refresh()
    assert not any("REJECTED" in m or "prior cache kept" in m for m in msgs)
    assert pd.read_parquet(tmp_path / cmod._FILE)["p_copper"].iloc[-1] == 9.9
    assert not (tmp_path / (cmod._FILE + ".tmp")).exists()


# --------------------------------------------------------------------------- #
# NBS refresh path
# --------------------------------------------------------------------------- #
def test_china_refresh_bypasses_response_caches():
    src = (REPO / "targets" / "china.py").read_text()
    assert '_fetch_nbs("M", refresh=True)' in src
    assert '_fetch_nbs("Q", refresh=True)' in src


def test_nbs_client_supports_age_invalidation():
    import sources.nbs as nbs

    sig = inspect.signature(nbs.get_nbs_data)
    assert "max_age_days" in sig.parameters
    client = nbs._NBSChinaClient(cache_dir=None, refresh=False, max_age_days=7)
    assert client.max_age_days == 7


# --------------------------------------------------------------------------- #
# commodity panel must not truncate faster series (preflight finding)
# --------------------------------------------------------------------------- #
def test_commodity_panel_keeps_faster_series_current():
    if not (REPO / "input" / "commodities").exists():
        pytest.skip("private data caches are not present")
    import sys

    sys.path.insert(0, str(REPO))
    import targets
    from targets import commodities as cmd

    us_m, _, _ = targets.get("usa").load_panel()
    pm, _, _ = cmd.SPECS["pe_tot"].load_panel()
    for col in ("us_vix", "g_us_indpro"):
        upstream = us_m[col].dropna().index.max()
        in_panel = pm[col].dropna().index.max()
        assert in_panel >= upstream, (
            f"{col}: commodity panel ends {in_panel} but the source has "
            f"{upstream}; the panel is truncating faster series again")
