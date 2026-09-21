"""Observed-release calendar: expected release dates learned from evidence.

The registry's scalar ``publication_lag_days`` is a PRIOR, not a calendar.
Providers move: BCRP published the July 2026 expectations block on August 5
while the scalar rule said August 7. This module keeps an append-only store
of OBSERVED release dates and derives, per series:

- ``expected``: the next release date. The registry scalar until at least
  ``MIN_HISTORY`` releases have been observed, then the observed MEDIAN lag
  from period end to first sighting.
- ``check_from``: when to start probing the provider for the next
  observation. ``PROBE_EARLY_DAYS`` before the expected date with no
  history; from the EARLIEST lag ever observed once there is any evidence.
- detection: a probe (or any refresh) that sees the next period records an
  observed release. The availability row is then annotated "released,
  pending ingest" for the data frontier and the report.

Contract with the preflight gate: the ``status`` column is NEVER touched.
Gating stays on the conservative registry scalar so an early provider
release can never flip a required series to ``stale_observation`` and block
the run before the panel rebuild (which is X13-gated) can possibly ingest
it. Detection is disclosure, not a gate.

As-of safety: every read filters ``first_seen <= as_of``, and the probe
channel must only be invoked when the as-of IS the wall-clock today
(``is_live_as_of``), so a historical rerun can never leak later knowledge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE = REPO_ROOT / "input" / "calendars" / "observed_releases.parquet"

MIN_HISTORY = 3          # observed releases before the median replaces the scalar
PROBE_EARLY_DAYS = 2     # probe window opens this many days before the scalar date
DEFAULT_GRACE_DAYS = 7   # tolerated lateness for a required series before the
                         # preflight hard-blocks (see grace_days_for)

_COLS = ["internal_code", "period", "first_seen", "source"]

# providers with a machine-probeable presence API (monthly and quarterly)
_PROBEABLE_HOSTS = ("estadisticas.bcrp.gob.pe/estadisticas/series/api",)


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #
def load_observed(path: str | Path | None = None, as_of=None) -> pd.DataFrame:
    """The observed-release table, restricted to sightings KNOWN by ``as_of``."""
    p = Path(path) if path is not None else DEFAULT_STORE
    if not p.exists():
        return pd.DataFrame(columns=_COLS)
    obs = pd.read_parquet(p)
    obs["first_seen"] = pd.to_datetime(obs["first_seen"])
    if as_of is not None:
        obs = obs[obs["first_seen"] <= pd.Timestamp(as_of).normalize()]
    return obs.reset_index(drop=True)


def record_observed(path: str | Path | None, rows) -> int:
    """Upsert sightings; the EARLIEST first_seen per (code, period) wins.

    Returns the number of genuinely new (code, period) pairs. Atomic write.
    """
    p = Path(path) if path is not None else DEFAULT_STORE
    new = pd.DataFrame(list(rows), columns=_COLS) if not isinstance(rows, pd.DataFrame) \
        else rows[_COLS].copy()
    if new.empty:
        return 0
    new["first_seen"] = pd.to_datetime(new["first_seen"])
    new["period"] = new["period"].astype(str)
    old = load_observed(p)
    before = {(c, per) for c, per in zip(old.internal_code, old.period)}
    n_added = sum((c, per) not in before
                  for c, per in zip(new.internal_code, new.period))
    both = new if old.empty else pd.concat([old, new], ignore_index=True)
    both = (both.sort_values("first_seen")
                .drop_duplicates(["internal_code", "period"], keep="first")
                .sort_values(["internal_code", "period"]).reset_index(drop=True))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.parquet")
    both.to_parquet(tmp, index=False)
    tmp.replace(p)
    return int(n_added)


# --------------------------------------------------------------------------- #
# expectation rule
# --------------------------------------------------------------------------- #
def _anchor(period: pd.Period, rule: str = "period_end_plus_lag") -> pd.Timestamp:
    """The date the publication lag counts from (period end, or start for
    start-anchored calendars such as the quarterly SPF)."""
    how = "start" if rule == "period_start_plus_lag" else "end"
    return period.to_timestamp(how=how).normalize()


def _observed_lags(observed: pd.DataFrame, code: str, freq: str,
                   rule: str = "period_end_plus_lag") -> pd.Series:
    block = observed[observed.internal_code == code]
    if block.empty:
        return pd.Series(dtype=float)
    ends = pd.DatetimeIndex(block.period.map(
        lambda s: _anchor(pd.Period(s, freq=freq), rule)))
    seen = pd.DatetimeIndex(pd.to_datetime(block.first_seen)).normalize()
    return pd.Series((seen - ends).days, dtype=float)


def expected_next_release(code: str, last_obs_period, lag_days: int,
                          observed: pd.DataFrame, freq: str = "M",
                          rule: str = "period_end_plus_lag") -> dict:
    """Expected date and probe window for the NEXT unseen observation."""
    nxt = pd.Period(str(last_obs_period), freq=freq) + 1
    end = _anchor(nxt, rule)
    scalar = end + pd.Timedelta(days=int(lag_days))
    lags = _observed_lags(observed, code, freq, rule)
    if len(lags) >= MIN_HISTORY:
        med = int(round(float(np.median(lags))))
        expected = end + pd.Timedelta(days=max(0, med))
        check_from = end + pd.Timedelta(days=max(0, int(lags.min())))
        basis = f"observed_median(n={len(lags)})"
    elif len(lags) >= 1:
        expected = scalar
        check_from = min(scalar - pd.Timedelta(days=PROBE_EARLY_DAYS),
                         end + pd.Timedelta(days=max(0, int(lags.min()))))
        basis = "registry_lag"
    else:
        expected = scalar
        check_from = scalar - pd.Timedelta(days=PROBE_EARLY_DAYS)
        basis = "registry_lag"
    return {"period": str(nxt), "expected": expected,
            "check_from": min(check_from, expected), "basis": basis}


def due_for_probe(code: str, last_obs_period, lag_days: int,
                  observed: pd.DataFrame, as_of, freq: str = "M",
                  rule: str = "period_end_plus_lag") -> bool:
    """Probe when the window has opened and the period is not yet detected.

    A late release stays due indefinitely: we keep checking until it shows.
    """
    e = expected_next_release(code, last_obs_period, lag_days, observed, freq, rule)
    seen = ((observed.internal_code == code) & (observed.period == e["period"])).any()
    return (not seen) and pd.Timestamp(as_of).normalize() >= e["check_from"]


def is_live_as_of(as_of) -> bool:
    """True only when the run's as-of is the wall-clock today (probe allowed)."""
    return pd.Timestamp(as_of).normalize() == pd.Timestamp.now().normalize()


def grace_days_for(code: str, lag_days: int, observed: pd.DataFrame,
                   freq: str = "M", rule: str = "period_end_plus_lag") -> int:
    """Tolerated lateness for one series before the preflight hard-blocks.

    Providers slip a few days routinely; the scalar lag is a prior with
    noise. With enough observed releases the window covers the series' own
    HISTORICAL WORST slippage past its typical lag (plus a two-day margin),
    never less than ``DEFAULT_GRACE_DAYS``. Structural failures are not
    covered by grace at all; this only softens ``stale_observation``.
    """
    lags = _observed_lags(observed, code, freq, rule)
    if len(lags) >= MIN_HISTORY:
        med = float(np.median(lags))
        worst = float(lags.max())
        return max(DEFAULT_GRACE_DAYS, int(round(worst - med)) + 2)
    return DEFAULT_GRACE_DAYS


# --------------------------------------------------------------------------- #
# availability annotation (called by the preflight, consumed by the frontier)
# --------------------------------------------------------------------------- #
def annotate_expectations(table: pd.DataFrame, registry: dict, *, as_of,
                          store_path: str | Path | None = None) -> pd.DataFrame:
    """Join calendar evidence onto the availability table as NEW columns.

    Adds ``calendar_next_period``, ``calendar_expected_release``,
    ``calendar_check_from``, ``calendar_basis``, ``release_detected`` and
    ``detected_release_date``. The ``status`` column is left untouched.
    """
    as_of = pd.Timestamp(as_of).normalize()
    observed = load_observed(store_path, as_of=as_of)
    reg = {s["internal_code"]: s for s in registry.get("series", [])}
    out = table.copy()
    periods, expecteds, checks, bases = [], [], [], []
    detected, det_dates = [], []
    for _, row in out.iterrows():
        code = str(row["internal_code"])
        meta = reg.get(code, {})
        freq = str(row.get("frequency") or meta.get("frequency") or "M")
        lag = meta.get("publication_lag_days")
        rule = (meta.get("release_calendar") or {}).get("rule")
        last = row.get("last_observation")
        if (freq not in ("M", "Q") or lag is None or pd.isna(last)
                or rule not in ("period_end_plus_lag", "period_start_plus_lag")):
            periods.append(None); expecteds.append(pd.NaT); checks.append(pd.NaT)
            bases.append(None); detected.append(False); det_dates.append(pd.NaT)
            continue
        last_p = pd.Period(pd.Timestamp(last), freq=freq)
        e = expected_next_release(code, last_p, int(lag), observed, freq, rule)
        hit = observed[(observed.internal_code == code)
                       & (observed.period == e["period"])]
        if len(hit):
            d = pd.Timestamp(hit.first_seen.iloc[0]).normalize()
            periods.append(e["period"]); expecteds.append(d); checks.append(e["check_from"])
            bases.append("detected"); detected.append(True); det_dates.append(d)
        else:
            periods.append(e["period"]); expecteds.append(e["expected"])
            checks.append(e["check_from"]); bases.append(e["basis"])
            detected.append(False); det_dates.append(pd.NaT)
    out["calendar_next_period"] = periods
    out["calendar_expected_release"] = expecteds
    out["calendar_check_from"] = checks
    out["calendar_basis"] = bases
    out["release_detected"] = detected
    out["detected_release_date"] = det_dates
    return out


def probe_due_series(table: pd.DataFrame, registry: dict, *, as_of,
                     store_path: str | Path | None = None,
                     prober=None) -> tuple[list[str], list[str]]:
    """Probe probeable providers for series whose check window is open.

    Monthly AND quarterly series of providers in ``_PROBEABLE_HOSTS`` (today:
    the BCRP series API) are probed. Every failure is caught and reported as a
    warning string: the probe channel is strictly optional and can never
    fail a run. Hits are recorded with ``first_seen = as_of``.

    Callers must ensure ``is_live_as_of(as_of)`` before invoking this with a
    real network prober; probing the live API for a historical as-of would
    leak future information into the run.
    """
    as_of = pd.Timestamp(as_of).normalize()
    if prober is None:
        from sources.bcrp import probe_release as prober   # pragma: no cover
    observed = load_observed(store_path, as_of=as_of)
    reg = {s["internal_code"]: s for s in registry.get("series", [])}
    hits, errors = [], []
    for _, row in table.iterrows():
        code = str(row["internal_code"])
        meta = reg.get(code, {})
        url = str(meta.get("source_url") or "")
        pcode = meta.get("provider_code")
        lag = meta.get("publication_lag_days")
        last = row.get("last_observation")
        freq = str(row.get("frequency") or meta.get("frequency") or "M")
        if (not any(h in url for h in _PROBEABLE_HOSTS) or not pcode
                or freq not in ("M", "Q") or lag is None or pd.isna(last)):
            continue
        if bool(row.get("release_detected", False)):
            continue
        last_p = pd.Period(pd.Timestamp(last), freq=freq)
        if not due_for_probe(code, last_p, int(lag), observed, as_of, freq):
            continue
        # ``e["period"]`` carries its own frequency ("2026-07" vs "2026Q3"), so
        # the prober infers it without a signature change
        e = expected_next_release(code, last_p, int(lag), observed, freq)
        try:
            if prober(str(pcode), e["period"]):
                record_observed(store_path, [{
                    "internal_code": code, "period": e["period"],
                    "first_seen": as_of, "source": "api_probe"}])
                hits.append(code)
        except Exception as exc:
            errors.append(f"{code}: {type(exc).__name__}: {exc}")
    return hits, errors


# --------------------------------------------------------------------------- #
# panel auto-ingest: probe upstream, then force the rebuild on detection
# --------------------------------------------------------------------------- #
def _panel_series(registry: dict, monthly: pd.DataFrame, target: str):
    """(internal_code, meta, column, last_period) for monitored panel columns."""
    out = []
    for s in registry.get("series", []):
        mon = s.get("monitor") or {}
        if (mon.get("type") == "target_panel" and mon.get("target") == target
                and mon.get("frame", "monthly") == "monthly"):
            col = mon.get("column") or s["internal_code"]
            if col in monthly.columns:
                last = monthly[col].dropna().index.max()
                if pd.notna(last):
                    out.append((s["internal_code"], s, col,
                                pd.Period(pd.Timestamp(last), freq="M")))
    return out


def detected_pending_for_panel(registry: dict, monthly: pd.DataFrame, *, as_of,
                               store_path: str | Path | None = None,
                               target: str = "peru_gdp") -> list[str]:
    """Store sightings NEWER than the panel: released upstream, not ingested.

    Each entry reads ``<code> <period> (seen <date>)`` and is meant to force
    a panel rebuild in the same run (the data stage passes the list into the
    target's ``refresh``).
    """
    as_of = pd.Timestamp(as_of).normalize()
    observed = load_observed(store_path, as_of=as_of)
    pending = []
    for code, _s, _col, last_p in _panel_series(registry, monthly, target):
        rows = observed[observed.internal_code == code]
        for r in rows.itertuples(index=False):
            if pd.Period(str(r.period), freq="M") > last_p:
                pending.append(f"{code} {r.period} "
                               f"(seen {pd.Timestamp(r.first_seen).date()})")
    return sorted(pending)


def probe_panel_releases(registry: dict, monthly: pd.DataFrame, *, as_of,
                         store_path: str | Path | None = None,
                         prober=None,
                         target: str = "peru_gdp") -> tuple[list[str], list[str]]:
    """Probe BCRP panel series whose check window is open, BEFORE the
    rebuild decision, so an early release is ingested in the same run.

    Callers must gate on ``is_live_as_of``; the table is derived from the
    loaded panel itself, so this needs no availability artifact.
    """
    series = _panel_series(registry, monthly, target)
    table = pd.DataFrame([
        {"internal_code": code, "frequency": "M",
         "last_observation": last_p.to_timestamp(how="start")}
        for code, _s, _col, last_p in series
    ])
    if table.empty:
        return [], []
    return probe_due_series(table, registry, as_of=as_of,
                            store_path=store_path, prober=prober)


# --------------------------------------------------------------------------- #
# target publication date (the nowcast figure's dashed line)
# --------------------------------------------------------------------------- #
def expected_target_publication(ref_q: pd.Period, delay_days: int, *,
                                availability: pd.DataFrame | None = None,
                                code: str = "g_pbiq"):
    """(date, basis) for the reference quarter's publication.

    Observed evidence wins ONLY when the availability row's calendar entry
    refers to this exact reference quarter and its basis is empirical
    (``detected`` or ``observed_median``). Otherwise the canonical rule
    ``expected_publication(ref_q, delay_days)`` applies, so with an empty
    calendar the dashed line is exactly what it always was.
    """
    from pipeline.blocks._common import expected_publication

    if availability is not None and "calendar_basis" in availability.columns:
        rows = availability[availability.internal_code == code]
        if len(rows):
            row = rows.iloc[0]
            basis = str(row.get("calendar_basis") or "")
            same_q = str(row.get("calendar_next_period") or "") == str(ref_q)
            date = pd.to_datetime(row.get("calendar_expected_release"), errors="coerce")
            if same_q and pd.notna(date) and (
                    basis == "detected" or basis.startswith("observed_median")):
                return pd.Timestamp(date).normalize(), basis
    return expected_publication(ref_q, delay_days), "canonical_lag"
