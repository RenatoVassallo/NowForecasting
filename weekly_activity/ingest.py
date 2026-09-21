"""Public-source ingestion for the Peru weekly-activity MVP.

This module intentionally uses only documented public endpoints.  COES exposes
executed demand by area and date; the BCRP API exposes four daily client-LBTR
series.  Google Trends has no credential-free stable public API in this MVP, so
it is accepted only as a user-supplied, frozen raw snapshot with provenance.
"""

from __future__ import annotations

import concurrent.futures as futures
import hashlib
import json
import logging
import time
from datetime import date
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

COES_EXECUTED_URL = "https://appserver.coes.org.pe/waMediciones/api/Demanda?fechaInicio={date}"
BCRP_API = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/{codes}/json/{start}/{end}/ing"
USER_AGENT = "Mozilla/5.0 (NowForecasting weekly-activity research MVP)"
LBTR_CODES = {
    "PD38073DD": "lbtr_client_value_mn",
    "PD38075DD": "lbtr_client_value_me",
    "PD38077DD": "lbtr_client_count_mn",
    "PD38079DD": "lbtr_client_count_me",
}
# COES returns these three geographic areas continuously in the verified
# 2015-2026 archive.  The separate Electroandes subarea is intermittently
# blank, especially during 2023.  The MVP therefore models this stable core,
# not a falsely complete national total assembled by imputing that subarea.
COES_CORE_AREA_CODES = frozenset(("3004", "3005", "3006"))
_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
           "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
           "ene": 1, "abr": 4, "ago": 8, "set": 9, "dic": 12}


class WeeklyActivitySourceError(RuntimeError):
    """A public endpoint did not return a valid source response."""


def _data_root(data_root=None) -> Path:
    if data_root is not None:
        return Path(data_root)
    from weekly_activity.config import DATA_ROOT
    return DATA_ROOT


def _request_json(url: str, *, retries: int = 2, timeout: int = 30):
    last = None
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT,
                                        "Accept": "application/json"})
            with urlopen(req, timeout=timeout) as response:
                payload = response.read()
            text = payload.decode("utf-8", errors="replace")
            if text.lstrip().lower().startswith("<html"):
                raise WeeklyActivitySourceError("HTML challenge page instead of JSON")
            return json.loads(text), payload
        except (URLError, TimeoutError, json.JSONDecodeError, WeeklyActivitySourceError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise WeeklyActivitySourceError(f"request failed after {retries + 1} attempts: {url}: {last}")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Atomically replace a local processed cache without project-private imports."""
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def _append_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _merge_latest_by_date(prior: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
    """Merge a refreshed daily batch, deterministically preferring fresh rows.

    Sorting before ``drop_duplicates(keep='last')`` is unsafe because an
    unstable sort can retain a stale cached row.  Concatenation order is the
    actual update contract here.
    """
    out = fresh.copy() if prior is None or prior.empty else pd.concat([prior, fresh], ignore_index=True)
    out["date"] = pd.to_datetime(out.date).dt.normalize()
    return out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def _hourly_columns(row: dict) -> list[str]:
    by_lower = {str(k).lower(): k for k in row}
    keys = []
    for i in range(1, 49):
        key = by_lower.get(f"h{i}")
        if key is None:
            raise ValueError(f"COES record has no h{i} interval")
        keys.append(key)
    return keys


def clean_coes_payload(payload, *, source_url: str, retrieved_at=None) -> pd.DataFrame:
    """Convert one COES daily response into area-level energy observations.

    Values are MW readings at half-hour intervals.  The daily energy is thus
    the sum of 48 values times 0.5 hours.  Partial area-days are retained with
    an explicit flag and no energy value, so their source quality is visible
    and they can never enter an aggregate silently.
    """
    if not isinstance(payload, list) or not payload:
        raise ValueError("COES returned no executed-demand area records")
    rows = []
    retrieved_at = pd.Timestamp(retrieved_at or pd.Timestamp.now(tz="UTC")).isoformat()
    for raw in payload:
        if not isinstance(raw, dict):
            raise ValueError("COES response contains a non-object record")
        hcols = _hourly_columns(raw)
        values = pd.to_numeric(pd.Series([raw[c] for c in hcols]), errors="coerce")
        is_complete = bool(values.notna().sum() == 48)
        date_value = raw.get("MEDIFECHA", raw.get("Medifecha"))
        point = raw.get("PTOMEDICODI", raw.get("Ptomedicodi"))
        name = raw.get("PTOMEDIELENOMB", raw.get("NombreUbicacion", raw.get("NombreEquipo")))
        rows.append({
            "date": pd.Timestamp(date_value).normalize(),
            "area_code": str(point), "area_name": str(name),
            "energy_mwh": float(values.sum() * 0.5) if is_complete else np.nan,
            "interval_count": int(values.notna().sum()), "is_complete": is_complete,
            "source_url": source_url, "retrieved_at": retrieved_at,
        })
    out = pd.DataFrame(rows).sort_values(["date", "area_code"])
    if out.duplicated(["date", "area_code"]).any():
        raise ValueError("COES response has duplicate area-date observations")
    return out.reset_index(drop=True)


def aggregate_coes_daily(area_data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the stable COES core areas and retain all quality diagnostics."""
    if area_data.empty:
        return pd.DataFrame(columns=["date", "coes_energy_mwh", "area_count",
                                     "area_complete_count", "core_area_count",
                                     "core_complete", "area_signature"])
    d = area_data.copy()
    d["date"] = pd.to_datetime(d.date).dt.normalize()
    sig = d.groupby("date").area_code.apply(lambda x: "|".join(sorted(map(str, x))))
    d["is_core_area"] = d.area_code.astype(str).isin(COES_CORE_AREA_CODES)
    core = d[d.is_core_area & d.is_complete].copy()
    core_energy = core.groupby("date").energy_mwh.sum()
    core_count = core.groupby("date").area_code.nunique()
    out = (d.groupby("date", as_index=False)
             .agg(area_count=("area_code", "nunique"),
                  area_complete_count=("is_complete", "sum"),
                  core_area_returned=("is_core_area", "sum"),
                  interval_count_min=("interval_count", "min")))
    out["core_area_count"] = out.date.map(core_count).fillna(0).astype(int)
    out["core_complete"] = out.core_area_count.eq(len(COES_CORE_AREA_CODES))
    out["coes_energy_mwh"] = out.date.map(core_energy).where(out.core_complete)
    out["area_signature"] = out.date.map(sig)
    return out.sort_values("date").reset_index(drop=True)


def _coes_day(day, raw_dir: Path, *, refresh: bool) -> tuple[pd.DataFrame, dict | None]:
    day = pd.Timestamp(day).normalize()
    path = raw_dir / f"{day.date()}.json"
    url = COES_EXECUTED_URL.format(date=day.date())
    if path.exists() and not refresh:
        payload = json.loads(path.read_text())
        return clean_coes_payload(payload, source_url=url, retrieved_at=path.stat().st_mtime), None
    payload, raw = _request_json(url)
    _atomic_bytes(path, raw)
    event = {"source": "COES", "url": url, "raw_path": str(path),
             "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
             "sha256": hashlib.sha256(raw).hexdigest(), "observation_date": str(day.date())}
    return clean_coes_payload(payload, source_url=url, retrieved_at=event["retrieved_at"]), event


def download_coes(
    start,
    end=None,
    *,
    data_root=None,
    refresh: bool = False,
    workers: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch COES executed demand one calendar date at a time with a raw cache.

    The public aggregate endpoint accepts only ``fechaInicio`` and returns one
    date's areas.  The per-day raw archive makes a long first bootstrap
    resumable and prevents re-querying already frozen observations.
    """
    root = _data_root(data_root)
    raw_dir = root / "raw" / "coes" / "daily"
    days = pd.date_range(pd.Timestamp(start).normalize(),
                         pd.Timestamp(end or pd.Timestamp.today()).normalize(), freq="D")
    good, events, failures, statuses = [], [], [], []
    with futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        pending = {pool.submit(_coes_day, day, raw_dir, refresh=refresh): day for day in days}
        for future in futures.as_completed(pending):
            day = pending[future]
            try:
                data, event = future.result()
                if not data.date.eq(pd.Timestamp(day).normalize()).all():
                    raise WeeklyActivitySourceError("COES response date differs from request")
                good.append(data)
                statuses.append({"date": pd.Timestamp(day).normalize(),
                                 "status": "received_pending_core_validation", "error": "",
                                 "checked_at": pd.Timestamp.now(tz="UTC").isoformat()})
                if event:
                    events.append(event)
            except Exception as exc:
                failure = {"date": str(pd.Timestamp(day).date()), "error": str(exc)}
                failures.append(failure)
                statuses.append({**failure, "status": "invalid_or_unavailable",
                                 "checked_at": pd.Timestamp.now(tz="UTC").isoformat()})
    _append_manifest(root / "raw" / "coes" / "manifest.jsonl", events)
    # A raw response can contain a partial non-core area.  The stable three-area
    # core is the validity rule; raw-file presence alone is never evidence that
    # a day is usable in the model.
    status_path = root / "processed" / "coes_download_status.parquet"
    if not good:
        raise WeeklyActivitySourceError(f"COES returned no valid days; failures: {failures[:3]}")
    area = pd.concat(good, ignore_index=True).sort_values(["date", "area_code"])
    daily = aggregate_coes_daily(area)
    quality = daily.set_index("date").core_complete
    status = pd.DataFrame(statuses)
    status["date"] = pd.to_datetime(status.date).dt.normalize()
    pending = status.status.eq("received_pending_core_validation")
    valid = status.date.map(quality).fillna(False)
    status.loc[pending & valid, "status"] = "valid"
    status.loc[pending & ~valid, "status"] = "invalid_core_area"
    status.loc[pending & ~valid, "error"] = "one or more stable COES core areas lack 48 intervals"
    if status_path.exists():
        status = pd.concat([pd.read_parquet(status_path), status], ignore_index=True)
    status["date"] = pd.to_datetime(status.date).dt.normalize()
    status = status.sort_values("checked_at").drop_duplicates("date", keep="last")
    atomic_write_parquet(status, status_path)
    previous_path = root / "processed" / "coes_daily.parquet"
    if previous_path.exists():
        prior = pd.read_parquet(previous_path)
        daily = _merge_latest_by_date(prior, daily)
    atomic_write_parquet(daily, previous_path)
    failures = pd.DataFrame(failures, columns=["date", "error"])
    return daily.reset_index(drop=True), failures


def _parse_bcrp_date(value: str) -> pd.Timestamp:
    parts = str(value).split(".")
    if len(parts) != 3:
        raise ValueError(f"unparseable BCRP daily period: {value!r}")
    day, mon, yr = parts
    month = _MONTHS.get(mon.lower()[:3])
    year = int(yr)
    if year < 100:
        year += 2000
    if month is None:
        raise ValueError(f"unparseable BCRP month: {value!r}")
    return pd.Timestamp(year=year, month=month, day=int(day))


def clean_lbtr_payload(payload: dict) -> pd.DataFrame:
    """Parse the four requested BCRP daily client-LBTR series."""
    if not isinstance(payload, dict) or "periods" not in payload:
        raise ValueError("BCRP LBTR response lacks a periods array")
    rows = []
    for obs in payload["periods"]:
        values = obs.get("values") or []
        if len(values) != len(LBTR_CODES):
            raise ValueError("BCRP LBTR response does not contain the four requested series")
        row = {"date": _parse_bcrp_date(obs.get("name", ""))}
        for name, value in zip(LBTR_CODES.values(), values):
            row[name] = pd.to_numeric(str(value).replace(",", ""), errors="coerce")
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("date")
    if out.duplicated("date").any():
        raise ValueError("BCRP LBTR response has duplicate dates")
    for col in LBTR_CODES.values():
        if (out[col].dropna() < 0).any():
            raise ValueError(f"BCRP LBTR contains negative values in {col}")
    return out.reset_index(drop=True)


def download_lbtr(start, end=None, *, data_root=None, refresh: bool = False) -> pd.DataFrame:
    """Fetch and cache all four public daily client-LBTR series in one request."""
    root = _data_root(data_root)
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end or pd.Timestamp.today()).normalize()
    if end < start:
        raise ValueError("LBTR end date precedes start date")
    stamp = f"{start.date()}_{end.date()}"
    raw_path = root / "raw" / "bcrp" / f"lbtr_{stamp}.json"
    url = BCRP_API.format(codes="-".join(LBTR_CODES),
                          start=f"{start.year}-{start.month}-{start.day}",
                          end=f"{end.year}-{end.month}-{end.day}")
    if raw_path.exists() and not refresh:
        payload = json.loads(raw_path.read_text())
        raw = raw_path.read_bytes()
    else:
        payload, raw = _request_json(url)
        _atomic_bytes(raw_path, raw)
        _append_manifest(root / "raw" / "bcrp" / "manifest.jsonl", [{
            "source": "BCRP", "url": url, "raw_path": str(raw_path),
            "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "codes": list(LBTR_CODES),
        }])
    out = clean_lbtr_payload(payload)
    processed = root / "processed" / "lbtr_daily.parquet"
    if processed.exists():
        out = _merge_latest_by_date(pd.read_parquet(processed), out)
    atomic_write_parquet(out, processed)
    return out.reset_index(drop=True)


def load_google_trends_snapshot(path: str | Path) -> pd.DataFrame:
    """Validate a frozen, provenance-rich Google Trends weekly snapshot.

    The official Google Trends API is currently alpha and access-controlled.
    The MVP therefore does not scrape its consumer interface or depend on
    pytrends.  A user who has legitimate API access may supply this exact raw
    snapshot contract, preserving its query window and download timestamp.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Google Trends snapshot absent: {path}. The MVP continues without "
            "Trends rather than scraping an unstable undocumented endpoint."
        )
    out = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {"week_end", "topic_id", "value", "geo", "downloaded_at", "window_start", "window_end"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Google Trends snapshot misses provenance columns: {sorted(missing)}")
    out["week_end"] = pd.to_datetime(out.week_end, errors="coerce").dt.normalize()
    out["downloaded_at"] = pd.to_datetime(out.downloaded_at, errors="coerce", utc=True)
    if out.week_end.isna().any() or out.downloaded_at.isna().any():
        raise ValueError("Google Trends snapshot has invalid dates")
    if out.duplicated(["week_end", "topic_id", "geo"]).any():
        raise ValueError("Google Trends snapshot has duplicate topic-week rows")
    return out.sort_values(["week_end", "topic_id"]).reset_index(drop=True)


__all__ = [
    "COES_EXECUTED_URL", "BCRP_API", "LBTR_CODES", "WeeklyActivitySourceError",
    "clean_coes_payload", "aggregate_coes_daily", "download_coes",
    "clean_lbtr_payload", "download_lbtr", "load_google_trends_snapshot",
]
