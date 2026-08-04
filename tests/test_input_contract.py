"""F7: every series production consumes maps to ONE monitored registry entry.

The declared inputs are extracted from the model specifications themselves
(nowcast ladders, the China production system, the commodity VAR, the Peru S1
system and its bolted-on columns, and the report-critical US assumptions), so
a spec change that consumes a new series fails this test until the registry
declares it. Publication-critical series must be monitored and marked
``required_for_publication``; making one optional to quiet the preflight is
exactly the failure mode this contract exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def consumed_series() -> dict[str, set]:
    """Column names production consumes, by consumer, from the specs."""
    from pipeline.blocks import _china_model as cm
    from pipeline.blocks import _peru_panel as pp
    from pipeline.blocks import commodities as cmd
    from pipeline.blocks import peru as pblock
    from pipeline.config import metadata

    consumed: dict[str, set] = {}

    def add(consumer, cols):
        consumed.setdefault(consumer, set()).update(cols)

    for target, specs in metadata.MODELS.items():
        for name, (_cls, kw) in specs.items():
            add(f"nowcast:{target}", kw.get("indicators", []))
            add(f"nowcast:{target}", kw.get("monthly_vars", []))

    add("china:system", [c for c in cm.SYSTEM if c != "gdp_yoy"])
    add("china:us_block", cm.US_COLS)
    add("china:leaders", metadata.CHINA_LEADERS)
    add("china:activity", metadata.CHINA_ACTIVITY)

    add("commodities:var", cmd.VARS)

    add("peru:s1", pblock.SYSTEM)
    add("peru:panel_extras", ["g_invq_m", "us_gdp_saar_m", "us_fedfunds",
                              "us_vix"] + [f"spf_gdp_h{i}" for i in range(5)])
    add("peru:target", ["g_pbiq", "g_pbim"])
    add("report:consensus", ["us_gdpnow"])
    return consumed


# model column -> registry internal_code, where the names legitimately differ;
# every alias must be documented in docs/audit/registry_reconciliation.md
ALIASES = {
    "g_us_indpro": "g_us_indpro",
}


def test_every_consumed_series_has_exactly_one_registry_entry():
    reg = json.loads((REPO / "pipeline/config/data_registry.json").read_text())
    codes = [s["internal_code"] for s in reg["series"]]
    assert len(codes) == len(set(codes)), "registry codes are not unique"
    known = set(codes)

    missing = {}
    for consumer, cols in consumed_series().items():
        for col in sorted(cols):
            code = ALIASES.get(col, col)
            if code not in known:
                missing.setdefault(col, []).append(consumer)
    assert not missing, (
        "series consumed by production without a registry entry:\n"
        + "\n".join(f"  {c}: {v}" for c, v in sorted(missing.items())))


def test_publication_critical_series_are_required_and_monitored():
    reg = json.loads((REPO / "pipeline/config/data_registry.json").read_text())
    by_code = {s["internal_code"]: s for s in reg["series"]}
    problems = []
    for consumer, cols in consumed_series().items():
        for col in sorted(cols):
            s = by_code.get(ALIASES.get(col, col))
            if s is None:
                continue                      # the other test reports absence
            if not s.get("required_for_publication"):
                problems.append(f"{col} ({consumer}): not required_for_publication")
            mon = s.get("monitor") or {}
            if mon.get("type") in (None, "none"):
                problems.append(f"{col} ({consumer}): not monitored")
    assert not problems, (
        "publication-critical series with weak registry status:\n"
        + "\n".join(sorted(set(problems))))


def test_fed_alias_is_reconciled():
    """`fed` (Peru spec3 panel) and `us_fedfunds` (US block) are the same
    upstream FRED series ingested twice; both entries must say so."""
    reg = json.loads((REPO / "pipeline/config/data_registry.json").read_text())
    by_code = {s["internal_code"]: s for s in reg["series"]}
    for code, other in (("fed", "us_fedfunds"), ("us_fedfunds", "fed")):
        entry = by_code.get(code)
        assert entry is not None, f"{code} missing from the registry"
        note = json.dumps(entry).lower()
        assert other in note, (
            f"registry entry {code} does not cross-reference its alias {other}")
