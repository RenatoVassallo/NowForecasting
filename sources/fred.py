"""FRED provider (foreign / US series) via MacroPy, behind the uniform contract."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .base import SourceError, to_wide

PROVIDER = "fred"
FREQUENCIES = ("M", "Q")


def _api_key(explicit: str | None = None) -> str:
    key = explicit or os.getenv("FRED_KEY")
    if not key:
        raise SourceError("FRED_KEY not set. Put it in the repo .env (git-ignored).")
    return key


def available(frequency: str | None = None, **kw) -> pd.DataFrame:
    """FRED has no browsable catalogue here: the registry is the source of truth."""
    from .registry import load_catalog

    cat = load_catalog()
    sel = cat[cat["provider"] == PROVIDER]
    if frequency:
        sel = sel[sel["frequency"] == frequency]
    return sel[["series_id", "provider", "label", "frequency", "unit"]].reset_index(drop=True)


def fetch(series=None, *, frequency: str = "M", start=None, end=None, refresh: bool = False,
          api_key: str | None = None, **kw) -> pd.DataFrame:
    """Wide frame of FRED series. ``series`` is a list of codes or {code: name}."""
    from MacroPy import get_fred_data

    if series is None:
        raise SourceError("`series` is required for the FRED provider")
    mapping = series if isinstance(series, dict) else {s: s for s in series}
    raw = get_fred_data(list(mapping), list(mapping.values()), frequency.lower(),
                        _api_key(api_key), start_period=start)
    return to_wide(raw, start=start, end=end)
