"""Data ingestion: every provider behind one uniform interface.

Providers: ``fred`` and ``bcrp`` (via MacroPy), ``inei`` (Peru INEI flash
reports) and ``nbs`` (China NBS / OECD / DBnomics).

**One control point.** ``sources/catalog.csv`` declares every series the project
ingests (id, provider, provider code, country, frequency, group, publication
delay, transform, active). Add a row and the monthly refresh picks it up::

    from sources import registry
    cat = registry.load_catalog()
    manifest = registry.ingest(cat)            # fetch everything + write raw store

**Uniform retrieval.** Every provider exposes the same two calls, so nothing
downstream needs to know the provider::

    import sources
    sources.available("nbs")                    # catalogue of what a provider offers
    sources.fetch("nbs", ["gdp_real_yoy"], frequency="Q", start="2005-01")
    sources.fetch("inei", ["pbi_global"])

Provider-native functions stay available for power use
(``sources.inei.get_inei_series``, ``sources.nbs.get_china_data``, ...).
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import CATALOG_COLUMNS, PROVIDERS, SourceError

# Peru INEI loaders (eager: light dependencies, and widely used directly). The
# INEI/NBS retrieval modules are kept private (provider-data licensing), so the
# import is guarded: the framework still imports without them, only the INEI
# native helpers become unavailable.
_INEI_NATIVES = [
    "INEIError",
    "available_inei_indicators",
    "build_inei_history",
    "classify_title",
    "clean_indicators",
    "download_report",
    "get_inei_series",
    "list_reports",
    "parse_report",
    "rank_indicator_consistency",
    "select_reports_for_history",
    "update_inei_latest",
]
try:
    from .inei import (
        INEIError,
        available_inei_indicators,
        build_inei_history,
        classify_title,
        clean_indicators,
        download_report,
        get_inei_series,
        list_reports,
        parse_report,
        rank_indicator_consistency,
        select_reports_for_history,
        update_inei_latest,
    )
except ImportError:
    _INEI_NATIVES = []


def provider(name: str):
    """Import and return a provider module by name (``fred``/``bcrp``/``inei``/``nbs``)."""

    if name not in PROVIDERS:
        raise SourceError(f"unknown provider {name!r}; expected one of {PROVIDERS}")
    return importlib.import_module(f"{__name__}.{name}")


def available(name: str, frequency: str | None = None, **kw) -> Any:
    """Catalogue of what a provider offers, in the standard schema."""

    return provider(name).available(frequency=frequency, **kw)


def fetch(name: str, series=None, **kw) -> Any:
    """Retrieve series from a provider as a wide frame (DatetimeIndex x series_id)."""

    return provider(name).fetch(series, **kw)


__all__ = [
    # uniform interface
    "PROVIDERS",
    "CATALOG_COLUMNS",
    "SourceError",
    "provider",
    "available",
    "fetch",
    # INEI natives (present only when the private loader is available)
    *_INEI_NATIVES,
]
