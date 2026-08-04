"""Append-only per-series refresh events.

The registry describes what a series IS; this log records what HAPPENED each
time we tried to refresh it. One JSON line per (series, attempt), never
rewritten: a failed refresh must not be able to alter economic metadata, and
the state at any past vintage must be reconstructible by replaying the log up
to a cutoff.

Event statuses are attempt-level: ``successfully_updated``,
``source_unavailable``, ``ingestion_failure``, ``validation_failure``,
``manually_overridden``. The state-level readings (``not_yet_released``,
``stale_observation``) are produced by the availability dashboard when it
joins the registry, the caches and this log at one as-of date; they are valid
in an event only for manual entries that assert them explicitly.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .data_availability import STATUSES

REPO = Path(__file__).resolve().parents[2]
EVENTS_PATH = REPO / "output" / "data_quality" / "refresh_events.jsonl"

FIELDS = ("internal_code", "attempted_at", "as_of", "status", "detail",
          "source_url", "raw_response_sha256", "parser_version",
          "n_rows", "n_cols", "first_observation", "last_observation",
          "artifact_sha256", "override_author", "override_reason",
          "override_effective_from", "override_effective_to")


def file_sha(path: Path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def describe_series(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n_rows": 0, "first_observation": None, "last_observation": None}
    return {"n_rows": int(s.size),
            "first_observation": str(pd.Timestamp(s.index.min()).date()),
            "last_observation": str(pd.Timestamp(s.index.max()).date())}


def record(internal_code: str, status: str, *, as_of, detail: str = "",
           source_url: str | None = None, raw_response_sha256: str | None = None,
           parser_version: str | None = None, n_rows: int | None = None,
           n_cols: int | None = None, first_observation: str | None = None,
           last_observation: str | None = None, artifact_sha256: str | None = None,
           override_author: str | None = None, override_reason: str | None = None,
           override_effective_from: str | None = None,
           override_effective_to: str | None = None,
           attempted_at=None, path: Path = EVENTS_PATH) -> dict:
    """Append one immutable event line; returns the recorded payload."""
    if status not in STATUSES:
        raise ValueError(f"invalid event status {status!r}; expected one of {STATUSES}")
    if status == "manually_overridden" and not (override_author and override_reason):
        raise ValueError("manual overrides must carry an author and a reason")
    event = {
        "internal_code": str(internal_code),
        "attempted_at": (datetime.now().isoformat(timespec="seconds")
                         if attempted_at is None else str(attempted_at)),
        "as_of": str(pd.Timestamp(as_of).date()),
        "status": status,
        "detail": str(detail)[:500],
        "source_url": source_url,
        "raw_response_sha256": raw_response_sha256,
        "parser_version": parser_version,
        "n_rows": None if n_rows is None else int(n_rows),
        "n_cols": None if n_cols is None else int(n_cols),
        "first_observation": first_observation,
        "last_observation": last_observation,
        "artifact_sha256": artifact_sha256,
        "override_author": override_author,
        "override_reason": override_reason,
        "override_effective_from": override_effective_from,
        "override_effective_to": override_effective_to,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(event, default=str) + "\n")
    return event


def read(path: Path = EVENTS_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=list(FIELDS))
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    out = pd.DataFrame(rows)
    if len(out):
        out["attempted_at"] = pd.to_datetime(out["attempted_at"])
    return out


def codes_for(registry: dict, *, provider: str | None = None,
              target: str | None = None) -> list[str]:
    """Registry series covered by one ingestion script or one target panel."""
    out = []
    for s in registry["series"]:
        if provider is not None and f"sources/{provider}" in str(s.get("ingestion_script", "")):
            out.append(s["internal_code"])
        elif target is not None and s.get("monitor", {}).get("target") == target:
            out.append(s["internal_code"])
    return out


def record_batch(codes: list[str], status: str, *, as_of, detail: str = "",
                 parser_version: str | None = None,
                 path: Path = EVENTS_PATH) -> int:
    """One attempt covering many series (a provider-level fetch)."""
    for code in codes:
        record(code, status, as_of=as_of, detail=detail,
               parser_version=parser_version, path=path)
    return len(codes)
