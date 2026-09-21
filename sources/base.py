"""Uniform contract shared by every data provider.

Each provider module (``fred``, ``bcrp``, ``inei``, ``nbs``) exposes the same
two entry points, so ingestion code never needs to know which provider it is
talking to:

``available(frequency=None, **kw) -> DataFrame``
    Catalogue of what the provider offers, with :data:`CATALOG_COLUMNS`.

``fetch(series=None, *, frequency=None, start=None, end=None, refresh=False, **kw) -> DataFrame``
    A **wide** frame: DatetimeIndex (period start) x one column per series id.

Provider-native functions (``get_inei_series``, ``get_nbs_data``, ...) stay
available for power use; these wrappers are the common surface the registry and
the automation build on.
"""

from __future__ import annotations

import pandas as pd

PROVIDERS = ("fred", "bcrp", "inei", "nbs")

#: Standard catalogue schema returned by every ``available()``.
CATALOG_COLUMNS = ["series_id", "provider", "label", "frequency", "unit", "source_url"]


class SourceError(RuntimeError):
    """Any failure while listing or retrieving data from a provider."""


def as_catalog(
    frame: pd.DataFrame,
    *,
    provider: str,
    id_col: str | None = None,
    label_col: str | None = None,
    unit_col: str | None = None,
    freq_col: str | None = None,
    frequency: str | None = None,
) -> pd.DataFrame:
    """Normalise a provider's native listing into :data:`CATALOG_COLUMNS`."""

    df = frame.copy()

    def _first_present(explicit, candidates):
        if explicit and explicit in df.columns:
            return explicit
        return next((c for c in candidates if c in df.columns), None)

    id_col = _first_present(
        id_col, ("series_id", "indicator_key", "indicator", "code", "key", "clean_name", "name", "variable")
    )
    ids = df[id_col] if id_col else pd.Series(df.index, index=df.index)
    ids = pd.Series(ids).astype(str).to_numpy()

    label_col = _first_present(label_col, ("label", "description", "title"))
    unit_col = _first_present(unit_col, ("unit", "units"))
    freq_col = _first_present(freq_col, ("frequency", "freq"))

    out = pd.DataFrame(
        {
            "series_id": ids,
            "provider": provider,
            "label": df[label_col].to_numpy() if label_col else ids,
            "frequency": df[freq_col].to_numpy() if freq_col else frequency,
            "unit": df[unit_col].to_numpy() if unit_col else None,
            "source_url": df["source_url"].to_numpy() if "source_url" in df.columns else None,
        }
    )
    return out.reset_index(drop=True)


def to_wide(frame: pd.DataFrame, *, start=None, end=None) -> pd.DataFrame:
    """Standardise a retrieved frame: DatetimeIndex, sorted, optionally sliced."""

    df = frame.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "period" in df.columns:
            df = df.set_index("period")
        elif "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.PeriodIndex(df.index, freq=_infer_freq(df.index)).to_timestamp(how="start") \
            if not isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(df.index)
    df = df.sort_index()
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def _infer_freq(index) -> str:
    sample = str(index[0]) if len(index) else ""
    return "Q" if "Q" in sample.upper() else "M"

def atomic_write_parquet(df, path):
    """Write a cache atomically: tmp file in the same directory, then replace.

    A crash mid-write must never leave a truncated cache; the prior file stays
    intact until the new one is fully on disk.
    """
    import os
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)
    return path


def atomic_write_csv(df, path, **to_csv_kw):
    """CSV twin of :func:`atomic_write_parquet`."""
    import os
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, **to_csv_kw)
    os.replace(tmp, path)
    return path


def validate_cache(fresh, required_columns, prior=None):
    """Problems that must BLOCK a cache replacement; empty list means safe.

    Checks: every required column present and not entirely missing, and no
    column that the prior cache carried disappears or empties (a partial
    upstream outage must degrade into a recorded failure, never into a
    silently narrower cache).
    """
    problems = []
    for c in required_columns:
        if c not in fresh.columns:
            problems.append(f"required column {c!r} is missing")
        elif fresh[c].notna().sum() == 0:
            problems.append(f"required column {c!r} is entirely missing")
    if prior is not None:
        for c in prior.columns:
            if prior[c].notna().sum() == 0:
                continue
            if c not in fresh.columns:
                problems.append(f"column {c!r} present in the prior cache disappeared")
            elif fresh[c].notna().sum() == 0:
                problems.append(f"column {c!r} present in the prior cache became empty")
    return problems
