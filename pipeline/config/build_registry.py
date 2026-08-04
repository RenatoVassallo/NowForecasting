"""Build the full machine-readable data registry from its sources of truth.

    PYTHONPATH=../MIDAS/src:../MacroPy/src python3.11 -m pipeline.config.build_registry

Inputs, in order of authority:

1. the EXISTING registry entries (the production-critical set is hand-curated;
   the builder never rewrites them, it only injects newly required fields);
2. `sources/catalog.csv` active rows (provider, codes, units, lags, transforms);
3. the target modules' ``DELAYS`` dictionaries, which hold the publication lags
   the models ACTUALLY use (they fill the catalog's missing NBS lags and are
   cross-checked against the catalog everywhere both exist);
4. the local panels, to locate each series' monitored column and start date.

Outputs: `pipeline/config/data_registry.json` (every active series; static
economic metadata only, never refresh outcomes) and
`docs/audit/registry_reconciliation.md` (every lag or transform conflict found
and how it was resolved). The registry is generated but COMMITTED; rerun this
builder after editing the catalog or the curated entries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "pipeline" / "config" / "data_registry.json"
CATALOG = REPO / "sources" / "catalog.csv"
RECON = REPO / "docs" / "audit" / "registry_reconciliation.md"

SOURCE_URLS = {
    "bcrp": "https://estadisticas.bcrp.gob.pe/estadisticas/series/api",
    "fred": "https://api.stlouisfed.org/fred/series/observations",
    "nbs": "https://data.stats.gov.cn/dg/website/page.html#/pc/national/home",
    "inei": "https://www.inei.gob.pe/biblioteca-virtual/boletines/",
}
INSTITUTIONS = {
    "bcrp": "Banco Central de Reserva del Peru",
    "fred": "Federal Reserve Bank of St. Louis (FRED)",
    "nbs": "National Bureau of Statistics of China",
    "inei": "Instituto Nacional de Estadistica e Informatica",
}
TRANSFORM_TEXT = {
    "yoy": "100 times the twelve-period ratio change of the level",
    "mom": "one-period change of the level",
    "mom_ann": "annualized one-month change",
    "qoq_ann": "annualized one-quarter change",
    "decum_yoy": "de-cumulated from year-to-date growth",
    "diff": "first difference",
    "none": "none, used as published",
}
# Deliberate lag resolutions where the catalog and the model code disagreed.
# Format: internal_code -> (chosen_lag, why). The catalog is authoritative when
# it has a value; over-claiming timeliness is the dangerous direction.
LAG_OVERRIDES: dict[str, tuple[int, str]] = {}


def _delay_maps():
    import targets.china as cn
    import targets.commodities as cmd
    import targets.usa as us

    return {"china": dict(cn.DELAYS), "commodities": dict(cmd.DELAYS),
            "usa": dict(us.DELAYS)}


def _panels():
    """(name -> monthly frame) for column mapping; loaded once, offline."""
    import targets
    from targets import commodities as cmd

    out = {}
    for name in ("peru_gdp", "china", "usa"):
        try:
            m, _, _ = targets.get(name).load_panel()
            out[name] = m
        except Exception:
            out[name] = pd.DataFrame()
    try:
        m, _, _ = cmd.SPECS["pe_tot"].load_panel()
        out["pe_tot"] = m
    except Exception:
        out["pe_tot"] = pd.DataFrame()
    return out


PANEL_PATHS = {
    "peru_gdp": "input/peru/monthly_panel_spec3.parquet",
    "china": "input/china/nbs_monthly_history.csv",
    "usa": "input/us/us_monthly.parquet",
    "pe_tot": "input/commodities/commodities_monthly.parquet",
}
PANEL_MODELS = {
    "peru_gdp": ["Peru nowcast ladder", "Peru research panel"],
    "china": ["China nowcast ladder", "China research panel"],
    "usa": ["United States block", "US research panel"],
    "pe_tot": ["Terms-of-trade monthly BVAR", "commodity research panel"],
}


def _find_column(sid: str, panels: dict) -> tuple[str, str] | None:
    cands = [sid, f"g_{sid}", f"{sid}_yoy", f"g_{sid}_yoy"]
    for pname, frame in panels.items():
        for c in cands:
            if c in frame.columns:
                return pname, c
    return None


def build() -> dict:
    existing = json.loads(REGISTRY.read_text())
    curated = {row["internal_code"]: row for row in existing["series"]}
    for row in curated.values():                      # inject the new field
        row.setdefault("required_for_publication", True)

    cat = pd.read_csv(CATALOG)
    active = cat[cat["active"] == 1].copy()
    delays = _delay_maps()
    panels = _panels()

    conflicts: list[dict] = []
    series: list[dict] = list(curated.values())
    curated_codes = set(curated)
    curated_provider_codes = {str(r.get("provider_code")) for r in curated.values()}

    for _, r in active.iterrows():
        sid = str(r["series_id"])
        if sid in curated_codes or str(r["provider_code"]) in curated_provider_codes:
            continue                                   # already hand-curated
        provider = str(r["provider"])
        cat_lag = r["publication_delay_days"]
        cat_lag = None if pd.isna(cat_lag) else int(cat_lag)

        hit = _find_column(sid, panels)
        pname, col = hit if hit else (None, None)
        code_lag = None
        for scope, dd in delays.items():
            if col in dd:
                code_lag = int(dd[col]); break
            if sid in dd:
                code_lag = int(dd[sid]); break

        if sid in LAG_OVERRIDES:
            lag, why = LAG_OVERRIDES[sid]
            conflicts.append(dict(series=sid, catalog=cat_lag, code=code_lag,
                                  chosen=lag, rule=f"override: {why}"))
        elif cat_lag is not None and code_lag is not None and cat_lag != code_lag:
            lag = max(cat_lag, code_lag)
            conflicts.append(dict(series=sid, catalog=cat_lag, code=code_lag,
                                  chosen=lag, rule="conflict: the LARGER lag wins "
                                  "(over-claiming timeliness is the dangerous direction)"))
        elif cat_lag is None and code_lag is not None:
            lag = code_lag
            conflicts.append(dict(series=sid, catalog=None, code=code_lag,
                                  chosen=lag, rule="catalog lag missing; model code value adopted"))
        else:
            lag = cat_lag

        tr = str(r["transform"]) if pd.notna(r["transform"]) else "none"
        sa = ("requires seasonal adjustment (project X13 preprocess)"
              if r.get("need_sa") == 1 else
              "not seasonally adjusted" if r.get("need_sa") == 0 else
              "undeclared in the catalog")
        start = None
        if col is not None and col in panels[pname].columns:
            fv = panels[pname][col].first_valid_index()
            start = str(pd.Timestamp(fv).date()) if fv is not None else None

        monitor = ({"type": "target_panel", "target": pname,
                    "frame": "monthly", "column": col,
                    "path": PANEL_PATHS[pname]}
                   if col is not None else {"type": "none"})
        series.append({
            "variable_name": str(r["label"]) if pd.notna(r["label"]) else sid,
            "internal_code": sid,
            "provider_code": str(r["provider_code"]),
            "source_institution": INSTITUTIONS[provider],
            "source_url": SOURCE_URLS[provider],
            "frequency": str(r["frequency"]),
            "unit": str(r["unit"]) if pd.notna(r["unit"]) else "undeclared",
            "transformation": TRANSFORM_TEXT.get(tr, tr),
            "geographic_coverage": str(r["country"]),
            "start_date": start,
            "release_calendar": {"rule": "period_end_plus_lag",
                                 "source": "sources/catalog.csv reconciled with "
                                           "target-module DELAYS"},
            "publication_lag_days": lag,
            "revision_policy": "latest-revised snapshot; no source-vintage archive",
            "seasonal_adjustment_status": sa,
            "vintage_availability": "run snapshots only",
            "expected_update_frequency": {"M": "monthly", "Q": "quarterly",
                                          "A": "annual", "D": "daily"}.get(
                                              str(r["frequency"]), "unknown"),
            "current_availability_status": None,
            "ingestion_script": f"sources/{provider}.py",
            "downstream_models": (PANEL_MODELS[pname] if pname else ["catalog only"]),
            "fallback_source": None,
            "validation_rules": {"unique_periods": True, "monotonic_periods": True,
                                 "allow_all_missing": False, "plausible_range": None},
            "last_successful_refresh": None,
            "known_issues": ([] if pd.isna(r.get("notes")) or not str(r.get("notes")).strip()
                             else [str(r["notes"])]),
            "monitor": monitor,
            "required_for_publication": False,
        })

    out = dict(existing)
    out["schema_version"] = "0.2.0"
    out["scope"] = ("Every active series in sources/catalog.csv plus the hand-curated "
                    "production-critical set. Static economic metadata only; refresh "
                    "outcomes live in the append-only event log "
                    "(output/data_quality/refresh_events.jsonl).")
    out["series"] = series
    return out, conflicts


def write_reconciliation(conflicts: list[dict]):
    lines = ["# Registry reconciliation", "",
             "Publication-lag conflicts between `sources/catalog.csv` and the",
             "target modules' `DELAYS` dictionaries, and how each was resolved.",
             "Rule: where both declare a value and disagree, the LARGER lag wins,",
             "because claiming a release earlier than it happens is the direction",
             "that leaks information. Hand overrides live in",
             "`pipeline/config/build_registry.py:LAG_OVERRIDES`.", "",
             "| series | catalog | model code | chosen | rule |", "|---|---|---|---|---|"]
    for c in sorted(conflicts, key=lambda x: str(x["series"])):
        lines.append(f"| {c['series']} | {c['catalog']} | {c['code']} | "
                     f"{c['chosen']} | {c['rule']} |")
    lines += ["", f"{len(conflicts)} rows reconciled.", ""]
    RECON.parent.mkdir(parents=True, exist_ok=True)
    RECON.write_text("\n".join(lines))


def main():
    registry, conflicts = build()
    REGISTRY.write_text(json.dumps(registry, indent=2))
    write_reconciliation(conflicts)
    n = len(registry["series"])
    req = sum(1 for s in registry["series"] if s.get("required_for_publication"))
    print(f"registry: {n} series ({req} required for publication); "
          f"{len(conflicts)} conflicts reconciled -> {RECON.name}")


if __name__ == "__main__":
    main()
