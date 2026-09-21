"""Commodity prices: Peru's terms-of-trade block (copper first).

v1 backend is **FRED** (IMF Primary Commodity Prices, monthly averages) via the
existing loader - one dependency, auto-refreshed, delay ~2 weeks. The World Bank
Pink Sheet (monthly xlsx + semiannual CMO forecasts) and exchange futures curves
are the planned phase-2 backends; the module keeps their hook.

Cache: ``input/commodities`` (git-ignored). ``refresh()`` follows the platform
convention: never raises, always reports.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import SourceError

PROVIDER = "commodities"
REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "input" / "commodities"

# FRED (IMF PCPS) code -> our column name; monthly average USD prices.
FRED_SERIES = {
    "PCOPPUSDM": "p_copper",     # copper, $/mt
    "PZINCUSDM": "p_zinc",       # zinc, $/mt
    "POILWTIUSDM": "p_wti",      # WTI crude, $/bbl
}
_FILE = "commodities_monthly.parquet"
_DAILY_FILE = "commodities_daily.parquet"

# BCRP daily quotations: the monthly averages above publish ~2 weeks after month
# end, but the dailies run to within a day or two of today. They are used to
# build FLASH months - month-to-date averages for months the monthly feed lacks
# - whose YoY growth is statistically identical to the FRED-based series
# (correlation 0.9998+, mean gap under 0.35pp on 25 years of overlap).
DAILY_SERIES = {"PD04701XD": "copper", "PD04704XD": "gold", "PD04705XD": "wti"}

# BCRP (same source family Peru uses for its own terms of trade): the LME/spot
# quotations FRED's IMF block does not carry (gold), and Peru's trade price
# indices. Monthly averages, index base 2007 = 100 for the three indices.
BCRP_SERIES = {
    "PN01654XM": "p_gold",       # gold, LME, $/troy oz
    "PN01652XM": "p_copper_lme", # copper, LME, cents/lb (cross-check on FRED)
    "PN01655XM": "p_silver",     # silver, $/troy oz
    "PN38915BM": "pe_px",        # Peru export price index
    "PN38919BM": "pe_pm",        # Peru import price index
    "PN38923BM": "pe_tot",       # Peru terms of trade
}


def _fred(mapping: dict) -> pd.DataFrame:
    from . import fred, us

    us._load_env()
    out = fred.fetch(dict(mapping), frequency="M")
    return out.rename(columns=mapping)


def fetch_prices() -> pd.DataFrame:
    """FRED (IMF) prices joined with the BCRP quotations and Peru trade indices."""

    out = _fred(FRED_SERIES)
    try:
        bc = fetch_bcrp()
    except Exception:
        return out                      # FRED block alone still usable
    return out.join(bc.reindex(out.index.union(bc.index)), how="outer")


def fetch_bcrp() -> pd.DataFrame:
    """Gold/silver quotations and Peru's export, import and terms-of-trade indices."""

    from MacroPy import get_bcrp_data

    raw = get_bcrp_data(list(BCRP_SERIES), frequency="M", start_period="1994-1")
    if raw is None or not len(raw):
        raise SourceError("BCRP returned no rows for the commodity block.")
    raw = raw.copy()
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw["date"]))
    keep = {c: n for c, n in BCRP_SERIES.items() if c in raw.columns}
    out = raw[list(keep)].rename(columns=keep).apply(pd.to_numeric, errors="coerce")
    out.index.name = None
    return out


def load() -> pd.DataFrame:
    f = CACHE_DIR / _FILE
    if not f.exists():
        raise SourceError("commodity cache missing; run the pipeline data refresh "
                          "(or sources.commodities.refresh()) once online.")
    return pd.read_parquet(f)


def fetch_daily() -> pd.DataFrame:
    """Daily quotations from BCRP, cached; ~27 years of history."""

    from MacroPy import get_bcrp_data

    raw = get_bcrp_data(list(DAILY_SERIES), frequency="D", start_period="2000-01-01")
    if raw is None or not len(raw):
        raise SourceError("BCRP returned no daily commodity rows.")
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw["date"]))
    out = raw[list(DAILY_SERIES)].rename(columns=DAILY_SERIES).apply(
        pd.to_numeric, errors="coerce")
    out.index.name = None
    return out


def load_daily() -> pd.DataFrame:
    f = CACHE_DIR / _DAILY_FILE
    if not f.exists():
        raise SourceError("daily commodity cache missing; run refresh() once online.")
    return pd.read_parquet(f)


def flash_growth(min_days: int = 10) -> pd.DataFrame:
    """YoY growth of month(-to-date) averages from the dailies.

    Returns a monthly frame with columns g_copper/g_gold/g_wti covering every
    month with at least ``min_days`` daily observations - including the current,
    incomplete month. Growth is computed within the BCRP series (same units both
    ends), so the cUS$/lb vs $/mt difference against FRED never enters.
    """

    d = load_daily()
    out = {}
    for col in d.columns:
        s = d[col].dropna()
        g = s.groupby(pd.PeriodIndex(s.index, freq="M"))
        avg = g.mean().where(g.count() >= min_days)
        out[f"g_{col}"] = 100.0 * (avg / avg.shift(12) - 1.0)
    res = pd.DataFrame(out).dropna(how="all")
    res.index = pd.DatetimeIndex([p.to_timestamp() for p in res.index])
    return res


def available(frequency: str | None = None, **kw) -> pd.DataFrame:
    cat = pd.DataFrame([{"series_id": v, "provider": PROVIDER, "label": k,
                         "frequency": "M", "unit": "USD"} for k, v in FRED_SERIES.items()])
    return cat if frequency in (None, "M") else cat.iloc[0:0]


def fetch(series=None, *, frequency: str = "M", start=None, end=None,
          refresh: bool = False, **kw) -> pd.DataFrame:
    if refresh:
        globals()["refresh"]()
    block = load()
    if series:
        block = block[[c for c in series if c in block.columns]]
    return block.loc[start:end]


# Columns that must survive every refresh: the ToT model's drivers and the
# official Peru trade indices. A refresh that would lose any of them keeps the
# prior cache and reports a failure instead (the audit found a BCRP outage
# could silently overwrite the full cache with a FRED-only frame).
REQUIRED_COLUMNS = ("p_copper", "p_gold", "p_wti", "pe_px", "pe_pm", "pe_tot")


def refresh() -> list[str]:
    from .base import atomic_write_parquet, validate_cache

    old = None
    f = CACHE_DIR / _FILE
    if f.exists():
        old = pd.read_parquet(f)
    try:
        # strict path: BOTH sources must deliver (fetch_prices degrades to
        # FRED-only on a BCRP failure, which is fine for ad-hoc use but must
        # never replace the production cache)
        fred_part = _fred(FRED_SERIES)
        bcrp_part = fetch_bcrp()
        fresh = fred_part.join(bcrp_part.reindex(fred_part.index.union(bcrp_part.index)),
                               how="outer")
    except Exception as exc:
        last = "" if old is None else f"; cache through {old.dropna(how='all').index.max():%Y-%m}"
        return [f"Commodities: fetch FAILED ({type(exc).__name__}: {exc}); "
                f"prior cache kept{last}"]
    problems = validate_cache(fresh, REQUIRED_COLUMNS, prior=old)
    if problems:
        return [f"Commodities: refresh REJECTED, prior cache kept ({'; '.join(problems)})"]
    atomic_write_parquet(fresh, f)
    try:
        daily = fetch_daily()
        atomic_write_parquet(daily, CACHE_DIR / _DAILY_FILE)
        dmsg = f"Commodities daily: through {daily.dropna(how='all').index.max():%Y-%m-%d}"
    except Exception as exc:
        dmsg = f"Commodities daily: FAILED ({type(exc).__name__}); flash months unavailable"
    last = fresh.dropna(how="all").index.max()
    msgs = [dmsg]
    if old is not None and last <= old.dropna(how="all").index.max():
        msgs.append(f"Commodities: no new releases (through {last:%Y-%m})")
    else:
        msgs.append(f"Commodities: updated through {last:%Y-%m}")
    for col, label, fmt in (("p_copper", "Copper", "{:,.0f} $/mt"),
                            ("p_gold", "Gold", "{:,.0f} $/oz"),
                            ("p_wti", "WTI", "{:,.0f} $/bbl"),
                            ("pe_tot", "Peru terms of trade", "{:,.1f} (2007=100)")):
        if col not in fresh.columns:
            continue
        s_ = fresh[col].dropna()
        if len(s_) >= 13:
            msgs.append(f"{label}: {fmt.format(s_.iloc[-1])} "
                        f"({100 * (s_.iloc[-1] / s_.iloc[-13] - 1):+.1f}% YoY)")
    return msgs


__all__ = ["available", "fetch", "load", "refresh", "fetch_prices", "fetch_bcrp",
           "FRED_SERIES", "BCRP_SERIES", "CACHE_DIR"]
