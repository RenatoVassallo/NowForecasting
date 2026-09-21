"""The data frontier: what each production variable's clock looked like.

One row per production-relevant series, computed RUN-LOCALLY from the run's
availability artifact (written by the preflight) plus the tracked registry:

- ``obs_end``: end of the last released observation (the GREEN span's edge);
- ``next_release``: the next expected release date (GREY runs until here).
  Preference order: a detected release (the observed-release calendar saw
  the next period on the provider), then the preflight's calendar-informed
  expectation, then the availability scalar, then period arithmetic;
- ``released_pending_ingest``: the provider has published the next
  observation but the panel has not ingested it yet (disclosure flag);
- ``new_this_run``: the latest observation's own release date (canonical
  rule: normalized period end plus the registry lag) falls within the seven
  days up to the run's as-of, so this run is the first weekly cycle that
  could have seen it;
- ``days_to_next``: as-of to the next expected release;
- ``group``/``block``: the registry monitor target, or the curated report
  block when a ``layout`` is passed.

A ``layout`` is a tuple of ``(block_title, entries)`` where each entry is
``(internal_code, display_label)`` or ``(internal_code, display_label,
opts)``. Options: ``{"cadence": "continuous", "interval_days": 7}`` for
sources that update quasi-continuously (Atlanta Fed GDPNow), and
``{"cadence": "weo_rounds"}`` for the IMF WEO round calendar (rounds in
January, April, July, October; updates late in the month for the January
and July editions, mid-month for the full April and October editions). A
layout code missing from the availability table raises ``KeyError``: a
curated report row must never silently vanish.

Quarterly convention: the availability artifact stamps a quarter by its
FINAL month (2026Q1 appears as 2026-03), so monthly period arithmetic lands
on the true quarter end.

No cross-run reads and no wall clock: everything derives from the as-of and
the run's own artifacts, so the figure is reproducible from the run alone.
"""

from __future__ import annotations

import pandas as pd

NEW_WINDOW_DAYS = 7

# WEO round months and the approximate day-of-month of each edition
_WEO_ROUNDS = {1: 25, 4: 15, 7: 25, 10: 15}


def _weo_dates(as_of: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = [pd.Timestamp(year=y, month=m, day=d)
             for y in (as_of.year - 1, as_of.year, as_of.year + 1)
             for m, d in _WEO_ROUNDS.items()]
    past = [d for d in dates if d <= as_of]
    future = [d for d in dates if d > as_of]
    return max(past), min(future)


def _row(a: pd.Series, meta: dict, as_of: pd.Timestamp, *,
         label: str | None = None, block: str | None = None,
         opts: dict | None = None) -> dict | None:
    code = str(a["internal_code"])
    opts = opts or {}
    cadence = opts.get("cadence")
    lag = int(meta.get("publication_lag_days", 15) or 15)
    rule = (meta.get("release_calendar") or {}).get("rule", "period_end_plus_lag")
    freq = str(a.get("frequency") or "M")
    freq = freq if freq in ("M", "Q") else "M"
    continuous = False
    detected = bool(a.get("release_detected", False))
    det_date = pd.to_datetime(a.get("detected_release_date"), errors="coerce")

    if cadence == "weo_rounds":
        obs_end, next_release = _weo_dates(as_of)
        last_release = obs_end
        detected = False
        det_date = pd.NaT
    else:
        last = a.get("last_observation")
        if pd.isna(last) or not str(last):
            return None
        per = pd.Period(pd.Timestamp(str(last)), freq=freq)
        obs_end = per.to_timestamp(how="end").normalize()
        anchor = (per.to_timestamp(how="start").normalize()
                  if rule == "period_start_plus_lag" else obs_end)
        last_release = anchor + pd.Timedelta(days=lag)
        if cadence == "continuous":
            # quasi-continuously updated source (GDPNow): current by nature
            continuous = True
            obs_end = min(obs_end, as_of)
            interval = int(opts.get("interval_days", 7))
            next_release = as_of + pd.Timedelta(days=interval)
            detected = False
            det_date = pd.NaT
        else:
            cal = pd.to_datetime(a.get("calendar_expected_release"), errors="coerce")
            nxt = a.get("next_expected_release")
            if detected and pd.notna(det_date):
                next_release = pd.Timestamp(det_date).normalize()
            elif pd.notna(cal):
                next_release = pd.Timestamp(cal).normalize()
            elif pd.notna(nxt) and str(nxt):
                next_release = pd.Timestamp(nxt).normalize()
            else:
                nxt_anchor = (per + 1).to_timestamp(
                    how="start" if rule == "period_start_plus_lag" else "end"
                ).normalize()
                next_release = nxt_anchor + pd.Timedelta(days=lag)

    pending = bool(detected and pd.notna(det_date)
                   and pd.Timestamp(det_date).normalize() <= as_of)
    return {
        "internal_code": code,
        "label": str(label if label is not None else a.get("variable_name", code)),
        "block": block,
        "group": (meta.get("monitor") or {}).get("target", "other"),
        "required": bool(meta.get("required_for_publication", False)),
        "status": str(a.get("status", "")),
        "obs_end": obs_end,
        "last_release": last_release,
        "next_release": next_release,
        "days_to_next": int((next_release - as_of).days),
        "continuous": continuous,
        "released_pending_ingest": pending,
        "detected_release_date": det_date,
        "new_this_run": bool(
            as_of - pd.Timedelta(days=NEW_WINDOW_DAYS) < last_release <= as_of),
    }


def frontier_frame(availability: pd.DataFrame, registry: dict, as_of,
                   include: list[str] | None = None,
                   layout: tuple | None = None) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of).normalize()
    reg = {s["internal_code"]: s for s in registry.get("series", [])}
    rows = []

    if layout is not None:
        avail = availability.set_index("internal_code", drop=False)
        for block_title, entries in layout:
            for entry in entries:
                code, label = entry[0], entry[1]
                opts = entry[2] if len(entry) > 2 else None
                if code not in avail.index:
                    raise KeyError(
                        f"frontier layout code {code!r} is absent from the "
                        "availability artifact; fix the layout or the registry")
                r = _row(avail.loc[code], reg.get(code, {}), as_of,
                         label=label, block=block_title, opts=opts)
                if r is None:
                    raise KeyError(
                        f"frontier layout code {code!r} has no usable "
                        "last observation in the availability artifact")
                rows.append(r)
        return pd.DataFrame(rows)

    for _, a in availability.iterrows():
        code = str(a["internal_code"])
        if include is not None and code not in include:
            continue
        r = _row(a, reg.get(code, {}), as_of)
        if r is not None:
            rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        order = {"peru_gdp": 0, "china": 1, "usa": 2, "pe_tot": 3}
        out["_g"] = out.group.map(order).fillna(9)
        out = out.sort_values(["_g", "internal_code"]).drop(columns="_g")
    return out.reset_index(drop=True)
