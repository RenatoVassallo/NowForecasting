"""Task 8: the migrated registry is complete, valid, and free of conflicts.

The registry is the single static description of every ACTIVE series: the
catalog feeds it, the target modules' delay dictionaries were reconciled into
it, and the production-critical set is flagged ``required_for_publication``.
Dynamic refresh outcomes never live here (they go to the append-only event
log), which these tests also pin.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "pipeline" / "config" / "data_registry.json"
SCHEMA = REPO / "pipeline" / "config" / "data_registry.schema.json"
CATALOG = REPO / "sources" / "catalog.csv"


@pytest.fixture(scope="module")
def registry():
    return json.loads(REGISTRY.read_text())


def test_registry_validates_against_the_schema(registry):
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(registry, json.loads(SCHEMA.read_text()))


def test_every_active_catalog_series_is_declared(registry):
    if not CATALOG.exists():
        pytest.skip("private catalog is not present")
    cat = pd.read_csv(CATALOG)
    active = cat[cat["active"] == 1]
    codes = {s["internal_code"] for s in registry["series"]}
    provider_codes = {str(s["provider_code"]) for s in registry["series"]}
    missing = [sid for sid, pc in zip(active["series_id"], active["provider_code"])
               if sid not in codes and str(pc) not in provider_codes]
    assert not missing, f"active catalog series absent from the registry: {missing}"


def test_no_duplicate_codes_and_static_only(registry):
    codes = [s["internal_code"] for s in registry["series"]]
    assert len(codes) == len(set(codes))
    # static metadata only: refresh outcomes must never be persisted here
    for s in registry["series"]:
        assert s["current_availability_status"] is None, s["internal_code"]


def test_required_set_is_the_curated_production_core(registry):
    required = {s["internal_code"] for s in registry["series"]
                if s.get("required_for_publication")}
    assert {"g_pbiq", "g_pbim_yoy", "g_tdi", "gdp_yoy", "ip_cum_yoy",
            "g_pe_tot", "us_gdp_yoy_m", "spf_gdp_h0",
            "weo_usa_ngdp_rpch"} <= required
    assert 25 <= len(required) <= 35, "the required set changed size unexpectedly"


def test_catalog_lags_agree_where_both_declare(registry):
    if not CATALOG.exists():
        pytest.skip("private catalog is not present")
    cat = pd.read_csv(CATALOG).set_index("series_id")
    problems = []
    for s in registry["series"]:
        sid = s["internal_code"]
        if sid in cat.index and pd.notna(cat.loc[sid, "publication_delay_days"]):
            cat_lag = int(cat.loc[sid, "publication_delay_days"])
            reg_lag = s["publication_lag_days"]
            if reg_lag is not None and int(reg_lag) < cat_lag:
                problems.append(f"{sid}: registry {reg_lag} < catalog {cat_lag}")
    assert not problems, ("registry claims an EARLIER release than the catalog "
                          f"(information-leak direction): {problems}")


def test_terms_of_trade_lag_is_forty_everywhere(registry):
    lags = {s["internal_code"]: s["publication_lag_days"] for s in registry["series"]}
    assert lags.get("g_tdi") == 40 and lags.get("g_pe_tot") == 40
    if not (Path(__file__).resolve().parents[1] / "input").exists():
        pytest.skip("private data layer absent: registry side verified only")
    import targets.commodities as cmd

    assert cmd.DELAYS["g_pe_tot"] == 40 and cmd.DELAYS["g_pe_px"] == 40
    assert cmd.TARGETS["pe_tot"][2] == 40
