"""Build a pre-run data availability view from the machine-readable registry.

The registry is static economic metadata. Mutable refresh outcomes belong in an
append-only event table and are joined only when this view is built. This keeps a
failed refresh from rewriting economic definitions and makes the state at any
forecast vintage reproducible.

Typical use::

    python -m pipeline.lib.data_availability \
        --as-of 2026-08-03 \
        --output output/data_quality/availability_dashboard.md

An optional event file may be JSON, JSON Lines, or CSV and must contain
``internal_code``, ``attempted_at``, ``status``, and optionally ``detail``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "pipeline" / "config" / "data_registry.json"

STATUSES = (
    "not_yet_released",
    "source_unavailable",
    "ingestion_failure",
    "validation_failure",
    "stale_observation",
    "manually_overridden",
    "successfully_updated",
)

REQUIRED_FIELDS = (
    "variable_name",
    "internal_code",
    "provider_code",
    "source_institution",
    "source_url",
    "frequency",
    "unit",
    "transformation",
    "geographic_coverage",
    "start_date",
    "release_calendar",
    "publication_lag_days",
    "revision_policy",
    "seasonal_adjustment_status",
    "vintage_availability",
    "expected_update_frequency",
    "current_availability_status",
    "ingestion_script",
    "downstream_models",
    "fallback_source",
    "validation_rules",
    "last_successful_refresh",
    "known_issues",
    "monitor",
    "required_for_publication",
)

ERROR_STATUSES = {
    "source_unavailable",
    "ingestion_failure",
    "validation_failure",
}


class RegistryError(ValueError):
    """The data registry is incomplete or internally inconsistent."""


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    """Load and validate the parts of the registry needed by the monitor."""

    registry = json.loads(Path(path).read_text())
    if not isinstance(registry.get("series"), list):
        raise RegistryError("registry.series must be a list")
    rows = registry["series"]
    codes = [row.get("internal_code") for row in rows]
    dupes = sorted({c for c in codes if c is not None and codes.count(c) > 1})
    if dupes:
        raise RegistryError(f"duplicate internal_code values: {dupes}")
    for i, row in enumerate(rows):
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise RegistryError(f"series row {i} is missing fields: {missing}")
        status = row["current_availability_status"]
        if status is not None and status not in STATUSES:
            raise RegistryError(f"{row['internal_code']}: invalid current status {status!r}")
        cal = row["release_calendar"]
        if not isinstance(cal, dict) or "rule" not in cal or "source" not in cal:
            raise RegistryError(f"{row['internal_code']}: release_calendar needs rule and source")
        if not isinstance(row["downstream_models"], list) or not row["downstream_models"]:
            raise RegistryError(f"{row['internal_code']}: downstream_models must be non-empty")
    declared = set(registry.get("status_definitions", {}))
    missing_statuses = sorted(set(STATUSES) - declared)
    if missing_statuses:
        raise RegistryError(f"status definitions missing: {missing_statuses}")
    return registry


def _read_tabular(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported monitored file type: {path.suffix}")


def _validate_series(series: pd.Series, rules: dict[str, Any]) -> str | None:
    if rules.get("unique_periods", False) and series.index.has_duplicates:
        return "duplicate reference periods"
    if rules.get("monotonic_periods", False) and not series.index.is_monotonic_increasing:
        return "reference periods are not sorted"
    values = pd.to_numeric(series, errors="coerce")
    if not rules.get("allow_all_missing", False) and values.notna().sum() == 0:
        return "series is entirely missing"
    bounds = rules.get("plausible_range")
    if bounds is not None and values.notna().any():
        # This is an operational refresh check. Known historical events such as
        # 2020 base effects must not make every future run fail forever. A caller
        # can request a wider tail with validation_window_periods.
        window = int(rules.get("validation_window_periods", 1))
        checked = values.dropna().iloc[-window:]
        lo, hi = map(float, bounds)
        bad = checked[(checked < lo) | (checked > hi)]
        if not bad.empty:
            return f"{len(bad)} values outside plausible range [{lo:g}, {hi:g}]"
    return None


def _cache_mtime(path: Path) -> pd.Timestamp | pd.NaT:
    if not path.exists():
        return pd.NaT
    return pd.Timestamp(datetime.fromtimestamp(path.stat().st_mtime))


def collect_observations(
    registry: dict[str, Any] | Iterable[dict[str, Any]],
    *,
    repo_root: str | Path = REPO_ROOT,
) -> pd.DataFrame:
    """Inspect current caches without network access.

    Each failure becomes a row-level status so the dashboard itself can still be
    rendered when an optional source, private cache, or parser is absent.
    """

    rows = registry["series"] if isinstance(registry, dict) else list(registry)
    root = Path(repo_root)
    panel_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame] | Exception] = {}
    out: list[dict[str, Any]] = []

    for meta in rows:
        code = meta["internal_code"]
        monitor = meta["monitor"]
        record: dict[str, Any] = {
            "internal_code": code,
            "last_observation": pd.NaT,
            "last_value": np.nan,
            "n_observations": 0,
            "cache_modified_at": pd.NaT,
            "collector_status": None,
            "collector_detail": "",
        }
        try:
            kind = monitor["type"]
            path_value = monitor.get("path")
            path = root / path_value if path_value else None
            if path is not None:
                record["cache_modified_at"] = _cache_mtime(path)

            if kind == "none":
                # static metadata only; the series is declared but not monitored
                continue

            if kind == "manual":
                if monitor.get("manual_override", False):
                    record["collector_status"] = "manually_overridden"
                    record["collector_detail"] = "registry declares an active manual override"
                out.append(record)
                continue

            if kind == "target_panel":
                target = monitor["target"]
                if target not in panel_cache:
                    try:
                        import targets

                        monthly, quarterly, _ = targets.get(target).load_panel()
                        panel_cache[target] = (monthly, quarterly)
                    except Exception as exc:  # dashboard must survive missing private data
                        panel_cache[target] = exc
                cached = panel_cache[target]
                if isinstance(cached, Exception):
                    raise cached
                frame = cached[0] if monitor["frame"] == "monthly" else cached[1]
            elif kind == "tabular_file":
                if path is None or not path.exists():
                    raise FileNotFoundError(path or "monitor.path is missing")
                frame = _read_tabular(path)
                for key, expected in monitor.get("filters", {}).items():
                    if key not in frame.columns:
                        raise KeyError(f"filter column {key!r} is absent")
                    frame = frame[frame[key] == expected]
                date_col = monitor.get("date_column")
                if date_col is not None:
                    if date_col not in frame.columns:
                        raise KeyError(f"date column {date_col!r} is absent")
                    frame = frame.set_index(pd.to_datetime(frame[date_col]))
            else:
                raise ValueError(f"unknown monitor type {kind!r}")

            column = monitor.get("column")
            if column not in frame.columns:
                record["collector_status"] = "validation_failure"
                record["collector_detail"] = f"monitored column {column!r} is absent"
                out.append(record)
                continue
            series = frame[column].copy()
            series.index = pd.to_datetime(series.index)
            validation = _validate_series(series, meta["validation_rules"])
            if validation:
                record["collector_status"] = "validation_failure"
                record["collector_detail"] = validation
            valid = pd.to_numeric(series, errors="coerce").dropna()
            if not valid.empty:
                record["last_observation"] = pd.Timestamp(valid.index.max()).normalize()
                record["last_value"] = float(valid.sort_index().iloc[-1])
                record["n_observations"] = int(valid.shape[0])
        except FileNotFoundError as exc:
            record["collector_status"] = "ingestion_failure"
            record["collector_detail"] = f"local cache is missing: {exc}"
        except Exception as exc:
            record["collector_status"] = "ingestion_failure"
            record["collector_detail"] = f"{type(exc).__name__}: {exc}"
        out.append(record)
    return pd.DataFrame(out)


def load_events(path: str | Path | None, as_of=None) -> pd.DataFrame:
    """Read an optional append-only refresh event table.

    ``as_of`` filters to events KNOWN by that date: a historical dashboard
    must not be driven by refresh attempts recorded later.
    """

    if path is None:
        return pd.DataFrame(columns=["internal_code", "attempted_at", "status", "detail"])
    p = Path(path)
    if p.suffix.lower() == ".csv":
        events = pd.read_csv(p)
    elif p.suffix.lower() in {".jsonl", ".ndjson"}:
        events = pd.read_json(p, lines=True)
    else:
        payload = json.loads(p.read_text())
        events = pd.DataFrame(payload if isinstance(payload, list) else payload.get("events", []))
    required = {"internal_code", "attempted_at", "status"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"event table is missing columns: {missing}")
    if not events["status"].isin(STATUSES).all():
        bad = sorted(events.loc[~events["status"].isin(STATUSES), "status"].unique())
        raise ValueError(f"event table has invalid statuses: {bad}")
    if "detail" not in events:
        events["detail"] = ""
    events["attempted_at"] = pd.to_datetime(events["attempted_at"])
    if as_of is not None:
        cutoff = pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1)
        events = events[events["attempted_at"] < cutoff]
    return events


def _candidate_period(as_of: pd.Timestamp, frequency: str, rule: str) -> pd.Period:
    current = pd.Period(as_of, freq=frequency)
    if rule == "period_start_plus_lag":
        return current
    end = current.end_time.normalize()
    return current if end <= as_of.normalize() else current - 1


def _release_date(period: pd.Period, rule: str, lag: int) -> pd.Timestamp:
    anchor = period.start_time.normalize() if rule == "period_start_plus_lag" \
        else period.end_time.normalize()
    return anchor + pd.Timedelta(days=int(lag))


def _latest_due_period(
    as_of: pd.Timestamp,
    frequency: str,
    rule: str,
    lag: int,
) -> tuple[pd.Period, pd.Period]:
    candidate = _candidate_period(as_of, frequency, rule)
    due = candidate
    for _ in range(40):
        if _release_date(due, rule, lag) <= as_of.normalize():
            return candidate, due
        due -= 1
    raise ValueError(f"could not resolve latest due {frequency} period")


def _latest_event(events: pd.DataFrame, code: str) -> pd.Series | None:
    block = events[events["internal_code"] == code]
    if block.empty:
        return None
    return block.sort_values("attempted_at").iloc[-1]


def _last_success(events: pd.DataFrame, code: str, static_value: Any) -> pd.Timestamp | pd.NaT:
    block = events[(events["internal_code"] == code)
                   & (events["status"] == "successfully_updated")]
    if not block.empty:
        return pd.Timestamp(block["attempted_at"].max())
    return pd.to_datetime(static_value) if static_value else pd.NaT


def build_availability(
    registry: dict[str, Any] | Iterable[dict[str, Any]],
    observations: pd.DataFrame,
    *,
    events: pd.DataFrame | None = None,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Classify each series at one explicit forecast vintage."""

    rows = registry["series"] if isinstance(registry, dict) else list(registry)
    as_of_ts = pd.Timestamp.now().normalize() if as_of is None else pd.Timestamp(as_of).normalize()
    events = events if events is not None else pd.DataFrame(
        columns=["internal_code", "attempted_at", "status", "detail"])
    if not events.empty:
        events = events.copy()
        events["attempted_at"] = pd.to_datetime(events["attempted_at"])
    obs = observations.set_index("internal_code", drop=False)
    output: list[dict[str, Any]] = []

    for meta in rows:
        code = meta["internal_code"]
        if meta.get("monitor", {}).get("type") == "none":
            continue                    # declared but unmonitored: no status row
        row = obs.loc[code] if code in obs.index else pd.Series(dtype=object)
        last_obs = pd.to_datetime(row.get("last_observation"), errors="coerce")
        cache_mtime = pd.to_datetime(row.get("cache_modified_at"), errors="coerce")
        collector_status = row.get("collector_status")
        detail = str(row.get("collector_detail") or "")
        latest_event = _latest_event(events, code)
        last_success = _last_success(events, code, meta["last_successful_refresh"])
        status: str
        latest_expected = pd.NaT
        next_release = pd.NaT

        monitor = meta["monitor"]
        if monitor.get("manual_override", False) or collector_status == "manually_overridden":
            status = "manually_overridden"
        elif collector_status in ERROR_STATUSES:
            status = str(collector_status)
        elif latest_event is not None and latest_event["status"] in ERROR_STATUSES:
            status = str(latest_event["status"])
            detail = str(latest_event.get("detail") or detail)
        elif pd.isna(last_obs):
            status = "ingestion_failure"
            detail = detail or "no usable observation was collected"
        elif last_obs.normalize() > as_of_ts:
            status = "validation_failure"
            detail = f"last observation {last_obs.date()} is after the as-of date"
        else:
            calendar = meta["release_calendar"]
            rule = calendar["rule"]
            if rule == "age_threshold":
                basis = cache_mtime if monitor.get("freshness_basis") == "cache_mtime" else last_obs
                stale_after = int(calendar["stale_after_days"])
                age = (as_of_ts - pd.Timestamp(basis).normalize()).days if pd.notna(basis) else np.inf
                if age > stale_after:
                    status = "stale_observation"
                    detail = f"freshness age {age:g} days exceeds {stale_after} days"
                else:
                    status = "successfully_updated"
            elif rule in {"period_end_plus_lag", "period_start_plus_lag"}:
                lag = meta["publication_lag_days"]
                if lag is None:
                    status = "validation_failure"
                    detail = "period-based calendar has no publication lag"
                else:
                    candidate, due = _latest_due_period(as_of_ts, meta["frequency"], rule, int(lag))
                    last_period = pd.Period(last_obs, freq=meta["frequency"])
                    latest_expected = due.start_time.normalize()
                    next_release = _release_date(due + 1, rule, int(lag))
                    if last_period > candidate:
                        status = "validation_failure"
                        detail = f"reference period {last_period} is after candidate {candidate}"
                    elif last_period < due:
                        status = "stale_observation"
                        detail = f"latest due period is {due}, cache ends at {last_period}"
                    elif last_period < candidate:
                        status = "not_yet_released"
                        detail = f"next period {last_period + 1} is expected on or after {next_release.date()}"
                    else:
                        status = "successfully_updated"
            elif rule == "manual":
                status = "manually_overridden"
            else:
                status = "validation_failure"
                detail = f"unknown release-calendar rule {rule!r}"

        output.append({
            "internal_code": code,
            "variable_name": meta["variable_name"],
            "source_institution": meta["source_institution"],
            "frequency": meta["frequency"],
            "status": status,
            "last_observation": last_obs,
            "latest_expected_observation": latest_expected,
            "next_expected_release": next_release,
            "last_value": row.get("last_value", np.nan),
            "n_observations": int(row.get("n_observations", 0) or 0),
            "cache_modified_at": cache_mtime,
            "last_successful_refresh": last_success,
            "vintage_consistent": status != "validation_failure",
            "detail": detail,
        })
    return pd.DataFrame(output)


def _date_text(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(ts) else str(pd.Timestamp(ts).date())


def _reference_text(value: Any, frequency: str) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    if frequency in {"M", "Q"}:
        return str(pd.Period(ts, freq=frequency))
    return str(pd.Timestamp(ts).date())


def render_markdown(table: pd.DataFrame, *, as_of: str | pd.Timestamp) -> str:
    """Render a compact, diff-friendly monitoring table."""

    as_of_text = str(pd.Timestamp(as_of).date())
    order = {
        "validation_failure": 0,
        "ingestion_failure": 1,
        "source_unavailable": 2,
        "stale_observation": 3,
        "manually_overridden": 4,
        "not_yet_released": 5,
        "successfully_updated": 6,
    }
    view = table.copy()
    view["_order"] = view["status"].map(order).fillna(99)
    view = view.sort_values(["_order", "source_institution", "internal_code"])
    counts = view["status"].value_counts().reindex(STATUSES, fill_value=0)
    summary = ", ".join(f"{name}: {int(count)}" for name, count in counts.items() if count)
    lines = [
        f"# Data availability dashboard as of {as_of_text}",
        "",
        f"Status counts: {summary or 'no series'}.",
        "",
        "| status | code | source | freq | last observation | latest expected | next release | cache modified | detail |",
        "|---|---|---|:---:|---|---|---|---|---|",
    ]
    for row in view.itertuples(index=False):
        detail = str(row.detail).replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {row.status} | `{row.internal_code}` | {row.source_institution} | "
            f"{row.frequency} | {_reference_text(row.last_observation, row.frequency)} | "
            f"{_reference_text(row.latest_expected_observation, row.frequency)} | "
            f"{_date_text(row.next_expected_release)} | "
            f"{_date_text(row.cache_modified_at)} | {detail} |"
        )
    lines.extend([
        "",
        "The dashboard is a pre-run control, not proof of real-time vintage integrity. "
        "A scalar publication lag cannot replace dated release calendars or historical source vintages.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--events", default=None)
    parser.add_argument("--as-of", default=str(pd.Timestamp.now().date()))
    parser.add_argument("--output", default=None, help="Markdown output path; stdout if omitted")
    args = parser.parse_args(argv)

    registry = load_registry(args.registry)
    observations = collect_observations(registry)
    events = load_events(args.events, as_of=args.as_of)
    table = build_availability(registry, observations, events=events, as_of=args.as_of)
    text = render_markdown(table, as_of=args.as_of)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
