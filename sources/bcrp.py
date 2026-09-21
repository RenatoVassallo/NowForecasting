"""BCRP provider (Peru central bank) via MacroPy, behind the uniform contract.

Also home to the release PROBE used by the observed-release calendar
(``pipeline.lib.release_calendar``): the BCRP portal has no machine-readable
release calendar (the "Última actualización" column on the browse pages is
rendered client-side), but the series API itself is a reliable oracle, since
requesting exactly one period returns that period's value the moment it is
published.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

from .base import SourceError, to_wide

PROVIDER = "bcrp"
FREQUENCIES = ("M", "Q")

# Private investment: the quarterly BCRP series the Peru block conditions on.
# ``pipeline/blocks/_peru_panel.py`` reads the cache below as ``g_invq`` (the
# year-on-year % change) and maps it onto the monthly panel.
PRIVATE_INVESTMENT_CODE = "PN02533AQ"
PRIVATE_INVESTMENT_CACHE = "input/bcrp/private_investment.parquet"
_MAX_REVISION_PP = 5.0

BCRP_API = ("https://estadisticas.bcrp.gob.pe/estadisticas/series/api/"
            "{code}/json/{start}/{end}")
_UA = {"User-Agent": "Mozilla/5.0 (NowForecasting release monitor)"}
_TIMEOUT = 20

_MONTHS = {
    # Spanish and English three-letter abbreviations as the API emits them
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7,
    "ago": 8, "sep": 9, "set": 9, "oct": 10, "nov": 11, "dic": 12,
    "jan": 1, "apr": 4, "aug": 8, "dec": 12,
}


class BCRPProbeError(RuntimeError):
    """The probe response could not be interpreted."""


_PROBE_SPACING_S = 1.6      # the portal rate-limits rapid API calls
_last_probe_at = 0.0


def _probe_fetch(url: str) -> str:
    """Paced fetch with one retry: burst probing trips the portal's
    anti-bot challenge (an HTML interstitial instead of JSON)."""
    import time

    global _last_probe_at
    wait = _PROBE_SPACING_S - (time.monotonic() - _last_probe_at)
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        body = r.read().decode("utf-8", errors="replace")
    if body.lstrip().startswith("<"):
        time.sleep(5.0)
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            body = r.read().decode("utf-8", errors="replace")
    _last_probe_at = time.monotonic()
    return body


def _resolve_year(year: int) -> int:
    """Expand the API's two-digit years ("T1.93" -> 1993, "T2.26" -> 2026).

    Monthly responses carry four-digit years ("Ene.2026"), but the quarterly
    ones are abbreviated, so a century pivot is needed: map into the 2000s and
    step back a century when that lands in the future.
    """
    if year >= 100:
        return year
    year += 2000
    return year - 100 if year > pd.Timestamp.today().year + 1 else year


def _parse_period_name(name: str) -> pd.Period | None:
    """Parse a BCRP period label into a monthly or quarterly ``Period``.

    The API labels periods by frequency AND language: monthly as ``Ene.2026``
    and quarterly as ``T1.26`` (Spanish, the default endpoint) or ``Q1.26``
    (the ``/ing`` endpoint). Returning ``None`` means "not a period we know".
    """
    parts = str(name).strip().split(".")
    if len(parts) != 2:
        return None
    head = parts[0].strip().lower()
    try:
        year = _resolve_year(int(parts[1]))
    except ValueError:
        return None
    if head[:1] in ("t", "q") and head[1:].isdigit():   # T1.26 / Q1.26
        quarter = int(head[1:])
        return pd.Period(year=year, quarter=quarter, freq="Q") if 1 <= quarter <= 4 else None
    mon = _MONTHS.get(head[:3])
    return pd.Period(year=year, month=mon, freq="M") if mon else None


def probe_release(provider_code: str, period, fetcher=None, freq: str | None = None) -> bool:
    """True when ``period`` of ``provider_code`` is published.

    Handles monthly AND quarterly series: the frequency is taken from ``freq``
    when given, otherwise inferred from ``period`` (``"2026-07"`` -> monthly,
    ``"2026Q3"`` -> quarterly), and the API window is expressed in that
    frequency's ordinals (``YYYY-M`` or ``YYYY-Q``).

    Strict on both ends: the returned period name must map to the requested
    period AND its value must parse as a number ("n.d." placeholders do not
    count as a release). The request spans three periods ending at the
    target; single-period windows have been observed to trip the portal's
    anti-bot challenge.
    """
    p = pd.Period(str(period), freq=freq) if freq else pd.Period(str(period))
    start = p - 2
    ordinal = (lambda q: q.quarter) if p.freqstr.startswith("Q") else (lambda m: m.month)
    url = BCRP_API.format(code=provider_code,
                          start=f"{start.year}-{ordinal(start)}",
                          end=f"{p.year}-{ordinal(p)}")
    raw = (fetcher or _probe_fetch)(url)
    if str(raw).lstrip().startswith("<"):
        raise BCRPProbeError(
            f"{provider_code}: anti-bot challenge page instead of JSON "
            "(rate limited); the probe retries once with pacing, then skips")
    try:
        payload = json.loads(raw)
        entries = payload.get("periods", [])
    except (json.JSONDecodeError, AttributeError) as exc:
        raise BCRPProbeError(f"{provider_code}: unparseable probe response") from exc
    for item in entries:
        if _parse_period_name(item.get("name", "")) != p:
            continue
        values = item.get("values") or []
        try:
            float(str(values[0]).replace(",", ""))
            return True
        except (ValueError, IndexError):
            continue
    return False


def available(frequency: str | None = None, **kw) -> pd.DataFrame:
    """BCRP has no browsable catalogue here: the registry is the source of truth."""
    from .registry import load_catalog

    cat = load_catalog()
    sel = cat[cat["provider"] == PROVIDER]
    if frequency:
        sel = sel[sel["frequency"] == frequency]
    return sel[["series_id", "provider", "label", "frequency", "unit"]].reset_index(drop=True)


def fetch(series=None, *, frequency: str = "M", start=None, end=None, refresh: bool = False,
          **kw) -> pd.DataFrame:
    """Wide frame of BCRP series. ``series`` is a list of codes or {code: name}."""
    from MacroPy import get_bcrp_data

    if series is None:
        raise SourceError("`series` is required for the BCRP provider")
    mapping = series if isinstance(series, dict) else {s: s for s in series}
    # named arguments only: the old positional call put the names list into
    # ``frequency`` and then collided on ``start_period`` (TypeError), so the
    # central BCRP path could never run (tests/test_ingestion.py pins this)
    kw_call = dict(frequency=frequency.lower(), names=list(mapping.values()))
    if start is not None:
        kw_call["start_period"] = start
    raw = get_bcrp_data(list(mapping), **kw_call)
    return to_wide(raw, start=start, end=end)


def private_investment_path() -> Path:
    return Path(__file__).resolve().parents[1] / PRIVATE_INVESTMENT_CACHE


def refresh_private_investment(path: str | Path | None = None, *, start: str = "1993-1",
                               max_revision_pp: float = _MAX_REVISION_PP) -> dict:
    """Refresh the ``g_invq`` cache that the Peru panel conditions on.

    ``g_invq`` is the year-on-year % change of the private investment LEVEL,
    indexed (as the provider returns it) at the first day of the quarter's last
    month, which is what ``_peru_panel._q2m`` expects. Nothing else in the
    pipeline wrote this file, so it silently went stale and the availability
    preflight blocked the run once the quarter it lacked came due.

    Fail-closed on the cache: an existing observation must never disappear and
    revisions must stay within ``max_revision_pp``, so a truncated or garbled
    provider response raises instead of destroying history. When a cache
    exists its start is preserved, so a refresh only ever extends the series
    forward (the estimation sample does not silently grow backwards).

    Returns ``{path, rows, added, last_period, last_value, max_revision_pp}``.
    """
    target = Path(path) if path is not None else private_investment_path()
    previous = None
    if target.exists():
        previous = pd.read_parquet(target)["g_invq"]

    level = fetch(series={PRIVATE_INVESTMENT_CODE: "invq"}, frequency="Q", start=start)["invq"]
    level = pd.to_numeric(level, errors="coerce").dropna().sort_index()
    if level.empty:
        raise SourceError(f"BCRP returned no data for {PRIVATE_INVESTMENT_CODE}")
    frame = ((level / level.shift(4) - 1.0) * 100.0).dropna().rename("g_invq").to_frame()

    revision = float("nan")
    if previous is not None and len(previous):
        frame = frame.loc[frame.index >= previous.index.min()]
        frame.index.name = previous.index.name
        lost = previous.index.difference(frame.index)
        if len(lost):
            raise SourceError(
                f"refusing to write {target.name}: the refresh would drop "
                f"{len(lost)} cached observation(s), e.g. {lost[0].date()}")
        revision = float((previous - frame["g_invq"].reindex(previous.index)).abs().max())
        if revision > max_revision_pp:
            raise SourceError(
                f"refusing to write {target.name}: a cached value moved by "
                f"{revision:.2f}pp (> {max_revision_pp}pp), which looks like a "
                "provider or parsing fault rather than a revision")

    added = [t for t in frame.index if previous is None or t not in previous.index]
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target)
    last = frame.index.max()
    return {"path": str(target), "rows": len(frame), "added": [str(t.date()) for t in added],
            "last_period": str(pd.Period(last, freq="Q")),
            "last_value": float(frame["g_invq"].iloc[-1]),
            "max_revision_pp": revision}
