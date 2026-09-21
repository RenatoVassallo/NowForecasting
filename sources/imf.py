"""IMF WEO vintages: the external-judgment provider.

The World Economic Outlook is ingested as a **vintage database**: every
April/October round (plus the September rounds of the early 2010s) is stored
with its release date, so backtests can ask "what did the IMF believe at this
origin?" exactly like the SPF pattern in :mod:`sources.us`. First use: annual
real-GDP-growth paths as entropic-tilting targets for the China (and later US,
euro, latam) forecast profiles.

imf.org sits behind bot protection, so the rounds are fetched from **DBnomics**
(https://db.nomics.world), which mirrors each WEO edition as a dataset
``IMF/WEO:{YYYY-MM}`` with series ``{ISO}.{SUBJECT}.{unit}``. We keep the two
judgment subjects (real GDP growth ``NGDP_RPCH``, CPI inflation ``PCPIPCH``)
for ALL countries - the loader is country-agnostic by design.

Release dates are approximated as the 15th of the round month (actual WEO
launches cluster mid-month). At the platform's quarterly origins (day 1 of a
quarter) the induced vintage mapping is exact either way: an April round is
never available on April 1, an October round never on October 1.

Cache: ``input/imf/weo_vintages.parquet`` (tidy: round, release_date, iso,
subject, year, value). ``refresh()`` follows the platform convention: never
raises, always reports, fetches only missing rounds.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .base import SourceError

PROVIDER = "imf"
REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "input" / "imf"
CACHE_FILE = "weo_vintages.parquet"

API = "https://api.db.nomics.world/v22/series/IMF/WEO:{round}"
AGG_API = "https://api.db.nomics.world/v22/series/IMF/WEOAGG:{round}"
SUBJECTS = ("NGDP_RPCH", "PCPIPCH")
# WEO's commodity price ASSUMPTIONS live in the aggregates file, under the World
# entity (code 001): annual average price levels, history + projections. These
# are the only forward-looking commodity numbers the WEO publishes, and they are
# what the whole forecast round is built on, so they are the right anchor for a
# copper or oil tilt. Stored with iso="WLD" so `path()` reaches them unchanged.
AGG_SUBJECTS = ("PCOPP", "POILWTI", "POILAPSP")
START_YEAR = 2010

# DBnomics lags the newest rounds (its editions stop at 2025-04 as of Jul 2026;
# the IMF now distributes fresh database files only through data.imf.org, which
# is not scriptable). Until DBnomics catches up, the CURRENT round's headline
# paths are curated here by hand and merged into ``load()`` output; ``refresh``
# reports when a manual round has been superseded by the real feed. Keep these
# to the numbers actually used (tilting anchors), nothing more.
MANUAL_ROUNDS: dict[str, dict[str, dict[str, dict[int, float]]]] = {}

# The CURRENT round comes from the IMF's own SDMX 3.0 API, which publishes each
# WEO update as soon as it is released (DBnomics lags by a year or more). It has
# no vintage history - it always serves the latest round - so it complements the
# DBnomics vintages rather than replacing them: history for backtests, this for
# today's anchor. Stored under a round label taken from the retrieval date.
SDMX_DATA = ("https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.RES/WEO/9.0.0/"
             "{countries}.{subject}.A")
SDMX_COUNTRIES = ("USA", "CHN", "PER", "CHL", "BRA", "MEX", "COL")


def _round_candidates(year: int) -> list[str]:
    return [f"{year}-04", f"{year}-10"]


def _fetch_round(rnd: str) -> pd.DataFrame | None:
    """One WEO edition from DBnomics -> tidy frame, or None if it doesn't exist."""

    dims = json.dumps({"weo-subject": list(SUBJECTS)})
    rows, offset = [], 0
    while True:
        url = (API.format(round=rnd)
               + f"?observations=1&limit=1000&offset={offset}"
               + "&dimensions=" + urllib.request.quote(dims))
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                payload = json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
        series = payload.get("series", {})
        docs = series.get("docs", [])
        for d in docs:
            iso, subject = d["series_code"].split(".")[:2]
            for per, val in zip(d["period"], d["value"]):
                try:
                    v = float(val)          # DBnomics encodes missing as "NA"
                except (TypeError, ValueError):
                    continue
                rows.append({"iso": iso, "subject": subject,
                             "year": int(per), "value": v})
        offset += len(docs)
        if offset >= series.get("num_found", 0) or not docs:
            break
    if not rows:
        return None
    out = pd.DataFrame(rows)
    out.insert(0, "round", rnd)
    y, m = rnd.split("-")
    out.insert(1, "release_date", pd.Timestamp(int(y), int(m), 15))
    return out


def _fetch_agg_round(rnd: str) -> pd.DataFrame | None:
    """WEO commodity price assumptions (World entity) for one edition."""

    # NOTE: filtering on weo-country in the query returns nothing (the dimension
    # keys are not the numeric entity codes), so we filter the World entity
    # ("001.") out of the returned series codes instead.
    dims = json.dumps({"weo-subject": list(AGG_SUBJECTS)})
    url = (AGG_API.format(round=rnd) + "?observations=1&limit=1000&dimensions="
           + urllib.request.quote(dims))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            payload = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    rows = []
    for d in payload.get("series", {}).get("docs", []):
        if not d["series_code"].startswith("001."):
            continue                      # World entity carries the assumptions
        subject = d["series_code"].split(".")[1]
        for per, val in zip(d["period"], d["value"]):
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            rows.append({"iso": "WLD", "subject": subject, "year": int(per), "value": v})
    if not rows:
        return None
    out = pd.DataFrame(rows)
    out.insert(0, "round", rnd)
    y, m = rnd.split("-")
    out.insert(1, "release_date", pd.Timestamp(int(y), int(m), 15))
    return out


def fetch_current(subject: str = "NGDP_RPCH",
                  countries: tuple = SDMX_COUNTRIES) -> pd.DataFrame | None:
    """Latest WEO round straight from the IMF SDMX 3.0 API (no vintage history)."""

    url = (SDMX_DATA.format(countries="+".join(countries), subject=subject)
           + "?c%5BTIME_PERIOD%5D=ge:1990-12-31+le:2035-12-31&detail=full&limit=2000")
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        payload = json.load(r)
    st = payload["data"]["structures"][0]
    years = [v.get("id") or v.get("value") for v in st["dimensions"]["observation"][0]["values"]]
    dims = [x["values"] for x in st["dimensions"]["series"]]
    rows = []
    for key, blk in payload["data"]["dataSets"][0]["series"].items():
        idx = [int(i) for i in key.split(":")]
        iso = dims[0][idx[0]]["id"]
        subj = dims[1][idx[1]]["id"]
        for k, obs in blk["observations"].items():
            try:
                v = float(obs[0])
            except (TypeError, ValueError):
                continue
            rows.append({"iso": iso, "subject": subj, "year": int(years[int(k)]), "value": v})
    if not rows:
        return None
    out = pd.DataFrame(rows)
    today = pd.Timestamp.now().normalize()
    rnd = f"{today.year}-{today.month:02d}-live"
    out.insert(0, "round", rnd)
    out.insert(1, "release_date", today)
    return out


def refresh_current(subjects=("NGDP_RPCH", "PCPIPCH")) -> list[str]:
    """Refresh the live round from the IMF API and merge it into the cache."""

    f = CACHE_DIR / CACHE_FILE
    old = pd.read_parquet(f) if f.exists() else None
    parts, msgs = [], []
    for subj in subjects:
        try:
            p_ = fetch_current(subject=subj)
        except Exception as exc:
            msgs.append(f"IMF live {subj}: FAILED ({type(exc).__name__})")
            continue
        if p_ is not None:
            parts.append(p_)
    if not parts:
        return msgs or ["IMF live: nothing fetched"]
    live = pd.concat(parts, ignore_index=True)
    keep = old[~old["round"].str.endswith("-live")] if old is not None else None
    out = pd.concat([keep, live], ignore_index=True) if keep is not None else live
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    from .base import atomic_write_parquet
    atomic_write_parquet(out, f)
    msgs.append(f"IMF live round {live['round'].iloc[0]}: {live['iso'].nunique()} countries, "
                f"{live['subject'].nunique()} subjects, through {live['year'].max()}")
    return msgs


def load(block: str = "weo") -> pd.DataFrame:
    if block != "weo":
        raise SourceError(f"unknown IMF block '{block}' (only 'weo').")
    f = CACHE_DIR / CACHE_FILE
    if not f.exists():
        raise SourceError("IMF WEO cache missing; run sources.imf.refresh() once online.")
    weo = pd.read_parquet(f)
    have = set(weo["round"].unique())
    manual = [{"round": rnd, "release_date": pd.Timestamp(int(rnd[:4]), int(rnd[5:7]), 15),
               "iso": iso, "subject": sub, "year": yr, "value": val}
              for rnd, isos in MANUAL_ROUNDS.items() if rnd not in have
              for iso, subs in isos.items() for sub, path_ in subs.items()
              for yr, val in path_.items()]
    if manual:
        weo = pd.concat([weo, pd.DataFrame(manual)], ignore_index=True)
    return weo


def rounds() -> list[str]:
    return sorted(load()["round"].unique())


def path(iso: str, as_of, subject: str = "NGDP_RPCH") -> tuple[pd.Series, str]:
    """The annual path (year -> value) from the latest round released by ``as_of``.

    Returns ``(series, round)``; raises if no round is available that early.
    The series covers history + projections as published in that round; the
    round's own calendar year is the "current-year" projection.
    """

    weo = load()
    as_of = pd.Timestamp(as_of)
    ok = weo[(weo["release_date"] <= as_of) & (weo["iso"] == iso)
             & (weo["subject"] == subject)]
    if ok.empty:
        raise SourceError(f"no WEO round released by {as_of.date()} (cache starts "
                          f"{weo['release_date'].min().date()}).")
    rnd = ok.loc[ok["release_date"].idxmax(), "round"]
    sel = ok[ok["round"] == rnd].set_index("year")["value"].sort_index()
    return sel, rnd


def available(frequency: str | None = None, **kw) -> pd.DataFrame:
    cat = pd.DataFrame([{"series_id": f"weo_{s.lower()}", "provider": PROVIDER,
                         "label": {"NGDP_RPCH": "WEO real GDP growth",
                                   "PCPIPCH": "WEO CPI inflation"}[s],
                         "frequency": "A", "unit": "% yoy"} for s in SUBJECTS])
    return cat if frequency in (None, "A") else cat.iloc[0:0]


def fetch(series=None, *, refresh: bool = False, **kw) -> pd.DataFrame:
    if refresh:
        globals()["refresh"]()
    return load()


def refresh() -> list[str]:
    """Fetch rounds missing from the cache (never raises, always reports)."""

    f = CACHE_DIR / CACHE_FILE
    old = pd.read_parquet(f) if f.exists() else None
    have = set(old["round"].unique()) if old is not None else set()
    today = pd.Timestamp.now()

    wanted = []
    for year in range(START_YEAR, today.year + 1):
        for rnd in _round_candidates(year):
            y, m = map(int, rnd.split("-"))
            if pd.Timestamp(y, m, 15) <= today and rnd not in have:
                wanted.append(rnd)

    got, failed = [], []
    frames = [] if old is None else [old]
    for rnd in wanted:
        try:
            part = _fetch_round(rnd)
        except Exception as exc:
            failed.append(f"{rnd} ({type(exc).__name__})")
            continue
        if part is None:
            failed.append(f"{rnd} (not on DBnomics)")
            continue
        frames.append(part)
        try:
            agg = _fetch_agg_round(rnd)
        except Exception:
            agg = None
        if agg is not None:
            frames.append(agg)
        got.append(rnd)
        time.sleep(0.3)

    msgs = []
    if got:
        out = pd.concat(frames, ignore_index=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        from .base import atomic_write_parquet
        atomic_write_parquet(out, f)
        msgs.append(f"IMF WEO: added rounds {', '.join(got)} "
                    f"({out['round'].nunique()} cached)")
    elif old is not None:
        msgs.append(f"IMF WEO: no new rounds ({old['round'].nunique()} cached, "
                    f"latest {max(have)})")
    else:
        msgs.append("IMF WEO: fetch FAILED for every round")
    if failed:
        msgs.append(f"IMF WEO: unavailable: {', '.join(failed)}")

    active_manual = [r for r in MANUAL_ROUNDS if r not in have.union(got)]
    superseded = [r for r in MANUAL_ROUNDS if r in have.union(got)]
    if active_manual:
        msgs.append(f"IMF WEO: manual overlay active for {', '.join(sorted(active_manual))} "
                    "(update sources.imf.MANUAL_ROUNDS each new round)")
    if superseded:
        msgs.append(f"IMF WEO: manual rounds superseded by the feed: "
                    f"{', '.join(sorted(superseded))} - remove from MANUAL_ROUNDS")
    try:
        cur, rnd = path("CHN", pd.Timestamp.now())
        y0 = int(rnd.split("-")[0])
        msgs.append(f"WEO {rnd} China: {y0} {cur.get(y0, float('nan')):.1f}, "
                    f"{y0+1} {cur.get(y0+1, float('nan')):.1f}")
    except Exception:
        pass
    return msgs


__all__ = ["available", "fetch", "load", "path", "rounds", "refresh",
           "SUBJECTS", "AGG_SUBJECTS", "CACHE_DIR"]
