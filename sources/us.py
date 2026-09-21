"""US external block: the "buy, don't build" provider.

The US is not modelled in-house - GDPNow, the Philadelphia Fed SPF and a small
FRED macro block are ingested as data and snapshotted per run, so the runs/ store
accumulates a real-time vintage database of what the consensus believed each
month. Two blocks:

* **FRED** (via the existing ``sources.fred`` loader): GDPNow (Atlanta Fed,
  ``GDPNOW``), realized real GDP (``GDPC1``), and monthly conditions
  (``INDPRO``, ``UNRATE``, ``T10Y3M``, ``DTWEXBGS``, ``NFCI``).
* **Philadelphia Fed SPF** (direct download): median real-GDP-growth forecasts
  by horizon - the h>=1 anchor. The published file carries the full survey
  history, i.e. true real-time vintages for backtesting.

Everything is cached under ``input/us`` (git-ignored); ``refresh()`` follows the
platform convention: never raises, always reports (including an explicit
"no new releases").
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .base import SourceError

PROVIDER = "us"
REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "input" / "us"

# FRED code -> our column name
FRED_MONTHLY = {
    "INDPRO": "us_indpro",          # industrial production index
    "UNRATE": "us_unrate",          # unemployment rate
    "T10Y3M": "us_curve_10y3m",     # yield-curve slope
    "DTWEXBGS": "us_dollar_broad",  # broad dollar index
    "NFCI": "us_nfci",              # Chicago Fed financial conditions
    "VIXCLS": "us_vix",             # VIX (monthly average)
    "CPIAUCSL": "us_cpi_index",     # CPI level (YoY derived in targets.usa)
    "FEDFUNDS": "us_fedfunds",      # effective federal funds rate
}
FRED_QUARTERLY = {
    "GDPC1": "us_gdp_level",        # real GDP level (for YoY growth)
    "GDPNOW": "us_gdpnow",          # Atlanta Fed GDPNow (qoq SAAR, current vintage)
}

# Philadelphia Fed SPF: median growth forecasts (qoq SAAR) by survey horizon.
SPF_URL = ("https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
           "survey-of-professional-forecasters/historical-data/medianGrowth.xlsx")
# DRGDP2 = the survey (current) quarter, DRGDP3..DRGDP6 = 1..4 quarters ahead.
SPF_COLUMNS = {"DRGDP2": "spf_gdp_h0", "DRGDP3": "spf_gdp_h1", "DRGDP4": "spf_gdp_h2",
               "DRGDP5": "spf_gdp_h3", "DRGDP6": "spf_gdp_h4"}


def _load_env() -> None:
    """Make sure FRED_KEY from the repo .env is in the environment (tolerates
    spaces around '=' as in a hand-written file)."""

    if os.getenv("FRED_KEY"):
        return
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def _fred(mapping: dict, frequency: str) -> pd.DataFrame:
    from . import fred

    _load_env()
    out = fred.fetch(dict(mapping), frequency=frequency)
    return out.rename(columns=mapping)


def fetch_fred_monthly() -> pd.DataFrame:
    return _fred(FRED_MONTHLY, "M")


def fetch_fred_quarterly() -> pd.DataFrame:
    return _fred(FRED_QUARTERLY, "Q")


def fetch_spf() -> pd.DataFrame:
    """Median SPF real-GDP-growth paths, indexed by survey quarter (full history).

    The published workbook has one sheet per variable (we read ``RGDP``) and
    malformed docProps timestamps that strict openpyxl rejects, so the metadata
    is stripped in-memory before parsing.
    """

    import io
    import re
    import urllib.request
    import warnings
    import zipfile

    req = urllib.request.Request(SPF_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    zin = zipfile.ZipFile(io.BytesIO(raw))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "docProps/core.xml":
                data = re.sub(rb"<dcterms:(created|modified)[^>]*>[^<]*</dcterms:\1>", b"", data)
            zout.writestr(item, data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(io.BytesIO(buf.getvalue()), sheet_name="RGDP")
    df.columns = [str(c).strip().upper() for c in df.columns]
    if not {"YEAR", "QUARTER"}.issubset(df.columns):
        raise SourceError(f"SPF file schema changed: {list(df.columns)[:8]}")
    idx = pd.PeriodIndex(year=df["YEAR"].astype(int), quarter=df["QUARTER"].astype(int),
                         freq="Q").to_timestamp(how="start")
    keep = {c: n for c, n in SPF_COLUMNS.items() if c in df.columns}
    out = df[list(keep)].rename(columns=keep)
    out.index = idx
    out.index.name = "survey_quarter"
    return out.apply(pd.to_numeric, errors="coerce")


# --------------------------------------------------------------------------- #
# Uniform contract
# --------------------------------------------------------------------------- #
_FILES = {"monthly": "us_monthly.parquet", "quarterly": "us_quarterly.parquet",
          "spf": "spf_median_gdp_growth.parquet"}


def load(block: str) -> pd.DataFrame:
    """Read one cached block ('monthly' | 'quarterly' | 'spf')."""

    f = CACHE_DIR / _FILES[block]
    if not f.exists():
        raise SourceError(f"US cache missing ({f.name}); run the pipeline data "
                          "refresh (or sources.us.refresh()) once online.")
    return pd.read_parquet(f)


def available(frequency: str | None = None, **kw) -> pd.DataFrame:
    rows = ([{"series_id": v, "provider": PROVIDER, "label": k, "frequency": "M",
              "unit": ""} for k, v in FRED_MONTHLY.items()]
            + [{"series_id": v, "provider": PROVIDER, "label": k, "frequency": "Q",
                "unit": ""} for k, v in FRED_QUARTERLY.items()]
            + [{"series_id": v, "provider": PROVIDER, "label": k, "frequency": "Q",
                "unit": "% qoq saar"} for k, v in SPF_COLUMNS.items()])
    cat = pd.DataFrame(rows)
    return cat[cat["frequency"] == frequency] if frequency else cat


def fetch(series=None, *, frequency: str = "M", start=None, end=None,
          refresh: bool = False, **kw) -> pd.DataFrame:
    if refresh:
        globals()["refresh"]()
    block = load("monthly" if frequency == "M" else "quarterly")
    spf = load("spf")
    if frequency != "M":
        block = block.join(spf, how="outer")
    if series:
        block = block[[c for c in series if c in block.columns]]
    return block.loc[start:end]


# --------------------------------------------------------------------------- #
# Refresh (the platform convention: never raises, always reports)
# --------------------------------------------------------------------------- #
def _snap(name: str, fetcher) -> tuple[pd.DataFrame | None, str]:
    old = None
    f = CACHE_DIR / _FILES[name]
    if f.exists():
        old = pd.read_parquet(f)
    try:
        fresh = fetcher()
    except Exception as exc:
        last = "" if old is None else f"; cache through {old.dropna(how='all').index.max():%Y-%m}"
        return old, f"US {name}: fetch FAILED ({type(exc).__name__}: {exc}){last}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    from .base import atomic_write_parquet
    atomic_write_parquet(fresh, f)
    last = fresh.dropna(how="all").index.max()
    if old is not None and last <= old.dropna(how="all").index.max():
        return fresh, f"US {name}: no new releases (through {last:%Y-%m})"
    return fresh, f"US {name}: updated through {last:%Y-%m}"


def refresh() -> list[str]:
    msgs = []
    m, msg = _snap("monthly", fetch_fred_monthly)
    msgs.append(msg)
    q, msg = _snap("quarterly", fetch_fred_quarterly)
    msgs.append(msg)
    spf, msg = _snap("spf", fetch_spf)
    msgs.append(msg)

    # Headline lines for the report: what the consensus currently says.
    try:
        if q is not None and q["us_gdpnow"].dropna().size:
            g = q["us_gdpnow"].dropna()
            qp = pd.Period(g.index.max(), freq="Q")
            msgs.append(f"GDPNow: {g.iloc[-1]:.1f}% (qoq saar) for {qp.year}Q{qp.quarter}")
        if spf is not None and len(spf):
            row = spf.dropna(how="all").iloc[-1]
            qp = pd.Period(spf.dropna(how="all").index.max(), freq="Q")
            path = ", ".join(f"h{i}={row.get(f'spf_gdp_h{i}', float('nan')):.1f}"
                             for i in range(5) if pd.notna(row.get(f"spf_gdp_h{i}")))
            msgs.append(f"SPF ({qp.year}Q{qp.quarter} survey): {path}")
    except Exception:
        pass
    return msgs


__all__ = ["available", "fetch", "load", "refresh", "fetch_spf",
           "fetch_fred_monthly", "fetch_fred_quarterly",
           "FRED_MONTHLY", "FRED_QUARTERLY", "SPF_COLUMNS", "CACHE_DIR"]
