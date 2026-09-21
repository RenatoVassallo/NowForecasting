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
             overrides: dict | None = None, *, grace_days: int = 7,
             grace_by_code: dict | None = None,
             ) -> tuple[list[str], list[str], list[str], list[str]]:
    """(offenders, waived, tolerated, unknown_overrides) for one table.

    A REQUIRED series that is merely ``stale_observation`` and whose
    ``days_late`` sits within its grace window is TOLERATED: the run
    proceeds, with the lateness disclosed everywhere (console, artifact,
    frontier, report). Error statuses and staleness beyond grace block as
    before; documented manual overrides waive anything.
    """
    overrides = overrides or {}
    grace_by_code = grace_by_code or {}
    required = {s["internal_code"] for s in registry["series"]
                if s.get("required_for_publication")}
    bad = table[table.status.isin(BLOCKING) & table.internal_code.isin(required)]
    offenders, waived, tolerated = [], [], []
    for _, r in bad.iterrows():
        code = str(r.internal_code)
        ov = overrides.get(code)
        if isinstance(ov, dict) and ov.get("author") and ov.get("reason"):
            waived.append(code)
            continue
        days_late = pd.to_numeric(r.get("days_late"), errors="coerce")
        grace = int(grace_by_code.get(code, grace_days))
        if (str(r.status) == "stale_observation" and pd.notna(days_late)
                and int(days_late) <= grace):
            tolerated.append(code)
        else:
            offenders.append(f"{code} [{r.status}] {r.detail}")
    unknown = [c for c in overrides if c not in set(table.internal_code)]
    return offenders, waived, tolerated, unknown


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
    if len(events) and "attempted_at" in events.columns:
        # only events KNOWN by the as-of drive a (possibly historical) gate
        cutoff = pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1)
        events = events[pd.to_datetime(events["attempted_at"]) < cutoff]
    table = build_availability(registry, observations, events=events, as_of=as_of)

    # observed-release calendar: annotate evidence-based expected dates and,
    # on a live as-of only, probe providers whose check window is open. The
    # status column (the gate) stays scalar-driven: detection is disclosure.
    from pipeline.lib import release_calendar as rc
    table = rc.annotate_expectations(table, registry, as_of=as_of)
    probe_hits, probe_errors = [], []
    if getattr(params, "RELEASE_PROBE", True) and rc.is_live_as_of(as_of):
        probe_hits, probe_errors = rc.probe_due_series(table, registry, as_of=as_of)
        if probe_hits:
            table = rc.annotate_expectations(table, registry, as_of=as_of)
            print("    [preflight] release probe: published on provider, "
                  "pending ingest: " + ", ".join(probe_hits))
        for w in probe_errors:
            print(f"    [preflight] release probe warning: {w}")

    d = store.dir("data_quality")
    store.save_text(d / "availability_dashboard.md",
                    render_markdown(table, as_of=as_of), kind="data-quality")
    # the CSV is saved ONCE below, after the tolerance verdicts are marked
    # (the strict manifest rejects duplicate artifact paths)

    overrides = getattr(params, "AVAILABILITY_OVERRIDES", {}) or {}
    grace_default = int(getattr(params, "AVAILABILITY_GRACE_DAYS", 7))
    observed = rc.load_observed(as_of=as_of)
    reg_by_code = {s["internal_code"]: s for s in registry["series"]}
    grace_by_code = {
        code: rc.grace_days_for(
            code, int(meta.get("publication_lag_days") or 0), observed,
            freq=str(meta.get("frequency") or "M"),
            rule=(meta.get("release_calendar") or {}).get("rule",
                                                          "period_end_plus_lag"))
        for code, meta in reg_by_code.items()
        if meta.get("required_for_publication")
    }
    grace_by_code = {c: max(g, grace_default) for c, g in grace_by_code.items()}
    offenders, waived, tolerated, unknown = evaluate(
        table, registry, overrides, grace_days=grace_default,
        grace_by_code=grace_by_code)
    table["tolerated_late"] = table.internal_code.isin(tolerated)
    store.save_df(d / "availability.csv", table.set_index("internal_code"),
                  kind="data-quality")
    for code in waived:
        ov = overrides[code]
        ev.record(code, "manually_overridden", as_of=as_of,
                  detail=f"preflight waiver: {ov['reason']}",
                  override_author=ov["author"], override_reason=ov["reason"],
                  override_effective_from=str(as_of.date()))
    store.set_meta(availability={"blocking": offenders, "waived": waived,
                                 "tolerated_late": tolerated,
                                 "unknown_overrides": unknown,
                                 "release_probe": {"hits": probe_hits,
                                                   "errors": probe_errors}})
    if unknown:
        print(f"    [preflight] WARNING: overrides for unknown series: {unknown}")
    if waived:
        print(f"    [preflight] waived by documented override: {', '.join(waived)}")
    if tolerated:
        by = table.set_index("internal_code")
        for code in tolerated:
            r = by.loc[code]
            print(f"    [preflight] TOLERATED LATE ({int(r.days_late)}d <= "
                  f"{grace_by_code.get(code, grace_default)}d grace): {code}; "
                  f"{r.detail}")
    if offenders:
        raise PreflightError(
            "required inputs are not fit for publication: "
            + "; ".join(offenders)
            + ". Fix the data, or add a documented override to "
              "params.AVAILABILITY_OVERRIDES (author + reason).")
    return table
