"""Availability preflight: the data-quality gate before any estimation.

Joins the registry, the local caches and the append-only event log at the
run's as-of date, saves the dashboard under the run directory, and BLOCKS the
run when any series flagged ``required_for_publication`` is stale, failed,
invalid or unreachable, unless a documented override waives it.

Overrides live in ``params.AVAILABILITY_OVERRIDES``::

    AVAILABILITY_OVERRIDES = {
        "m2_yoy": {"author": "rvs", "reason": "PBoC portal outage, value judged
                    unchanged; expires with the next release"},
    }

Every waiver is recorded as a ``manually_overridden`` event in the append-only
log (author and reason are mandatory) and listed in the run manifest, so an
override can never be silent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BLOCKING = ("stale_observation", "ingestion_failure",
            "validation_failure", "source_unavailable")


class PreflightError(RuntimeError):
    """Required inputs are not fit for publication."""


def evaluate(table: pd.DataFrame, registry: dict,
             overrides: dict | None = None) -> tuple[list[str], list[str], list[str]]:
    """(offenders, waived, unknown_overrides) for one availability table."""
    overrides = overrides or {}
    required = {s["internal_code"] for s in registry["series"]
                if s.get("required_for_publication")}
    bad = table[table.status.isin(BLOCKING) & table.internal_code.isin(required)]
    offenders, waived = [], []
    for _, r in bad.iterrows():
        code = str(r.internal_code)
        ov = overrides.get(code)
        if isinstance(ov, dict) and ov.get("author") and ov.get("reason"):
            waived.append(code)
        else:
            offenders.append(f"{code} [{r.status}] {r.detail}")
    unknown = [c for c in overrides if c not in set(table.internal_code)]
    return offenders, waived, unknown


def run_preflight(store, params) -> pd.DataFrame:
    """Build, persist and enforce the availability dashboard for this run."""
    from pipeline.lib import refresh_events as ev
    from pipeline.lib.context import resolve_as_of
    from pipeline.lib.data_availability import (build_availability,
                                                collect_observations,
                                                load_registry, render_markdown)

    ctx = getattr(store, "ctx", None)
    as_of = resolve_as_of(ctx)
    registry = load_registry()
    observations = collect_observations(registry)
    events = ev.read()
    table = build_availability(registry, observations, events=events, as_of=as_of)

    d = store.dir("data_quality")
    store.save_text(d / "availability_dashboard.md",
                    render_markdown(table, as_of=as_of), kind="data-quality")
    store.save_df(d / "availability.csv", table.set_index("internal_code"),
                  kind="data-quality")

    overrides = getattr(params, "AVAILABILITY_OVERRIDES", {}) or {}
    offenders, waived, unknown = evaluate(table, registry, overrides)
    for code in waived:
        ov = overrides[code]
        ev.record(code, "manually_overridden", as_of=as_of,
                  detail=f"preflight waiver: {ov['reason']}",
                  override_author=ov["author"], override_reason=ov["reason"],
                  override_effective_from=str(as_of.date()))
    store.set_meta(availability={"blocking": offenders, "waived": waived,
                                 "unknown_overrides": unknown})
    if unknown:
        print(f"    [preflight] WARNING: overrides for unknown series: {unknown}")
    if waived:
        print(f"    [preflight] waived by documented override: {', '.join(waived)}")
    if offenders:
        raise PreflightError(
            "required inputs are not fit for publication: "
            + "; ".join(offenders)
            + ". Fix the data, or add a documented override to "
              "params.AVAILABILITY_OVERRIDES (author + reason).")
    return table
