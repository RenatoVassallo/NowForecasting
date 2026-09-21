"""The single control point for data ingestion.

One catalogue file (``sources/catalog.csv``) declares **every** series the
project ingests, whatever the provider. Adding a row is all it takes for the
monthly refresh to pick a series up; nothing else needs editing.

Schema (one row per series):

===========================  ====================================================
``series_id``                canonical name used everywhere downstream
``provider``                 ``fred`` | ``bcrp`` | ``inei`` | ``nbs``
``provider_code``            the provider's own code (FRED id, BCRP code, ...)
``country``                  ``PE`` | ``CN`` | ``US`` | ``GLOBAL``
``frequency``                ``M`` | ``Q``
``group``                    economic block (used by the DFM and reporting)
``label`` / ``unit``         human-readable name / units
``publication_delay_days``   release lag; drives the real-time ragged edge
``need_sa``                  1 if the series needs seasonal adjustment
``transform``                default transform (``yoy``/``mom_ann``/``none``/...)
``active``                   1 to include in the refresh
``notes``                    free text
===========================  ====================================================

Typical use::

    from sources import registry
    cat = registry.load_catalog()
    panels, manifest = registry.fetch_catalog(cat, providers=["nbs"])

``fetch_catalog`` dispatches to each provider's uniform ``fetch()``, returns one
wide frame per frequency, and records an ingestion manifest (rows, last
observation, status) so the automated monthly run can be monitored.
"""

from __future__ import annotations

import datetime as _dt
import importlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

CATALOG_PATH = Path(__file__).resolve().parent / "catalog.csv"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "input" / "_ingestion_manifest.csv"

CATALOG_COLUMNS = [
    "series_id", "provider", "provider_code", "country", "frequency", "group",
    "label", "unit", "publication_delay_days", "need_sa", "transform", "active", "notes",
]
REQUIRED = ["series_id", "provider", "provider_code", "frequency"]


def load_catalog(path: str | Path | None = None, *, active_only: bool = True) -> pd.DataFrame:
    """Read and validate the catalogue."""

    p = Path(path) if path is not None else CATALOG_PATH
    if not p.exists():
        raise FileNotFoundError(f"catalogue not found at {p}; run registry.seed_catalog() first")
    cat = pd.read_csv(p)
    missing = [c for c in REQUIRED if c not in cat.columns]
    if missing:
        raise ValueError(f"catalogue is missing required columns: {missing}")
    dupes = cat.loc[cat["series_id"].duplicated(), "series_id"].tolist()
    if dupes:
        raise ValueError(f"duplicate series_id in catalogue: {dupes[:5]}")
    for col in CATALOG_COLUMNS:
        if col not in cat.columns:
            cat[col] = pd.NA
    if active_only and "active" in cat.columns:
        cat = cat[cat["active"].fillna(1).astype(int) == 1]
    return cat.reset_index(drop=True)


def provider_module(name: str):
    """Import a provider module by catalogue name."""

    return importlib.import_module(f"{__package__}.{name}")


def fetch_catalog(
    catalog: pd.DataFrame | None = None,
    *,
    providers: Sequence[str] | None = None,
    frequency: str | None = None,
    start: str | None = None,
    refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Fetch every catalogued series, grouped by provider then frequency.

    Returns ``(panels, manifest)`` where ``panels`` maps frequency -> wide frame
    (columns are ``series_id``) and ``manifest`` logs one row per provider/
    frequency batch with status, row count and last observation.
    """

    cat = load_catalog() if catalog is None else catalog
    if providers:
        cat = cat[cat["provider"].isin(list(providers))]
    if frequency:
        cat = cat[cat["frequency"] == frequency]

    panels: dict[str, list[pd.DataFrame]] = {}
    log: list[dict] = []
    for (prov, freq), block in cat.groupby(["provider", "frequency"], dropna=False):
        mapping = dict(zip(block["provider_code"].astype(str), block["series_id"].astype(str)))
        row = {"provider": prov, "frequency": freq, "n_requested": len(mapping),
               "fetched_at": _dt.datetime.now().isoformat(timespec="seconds")}
        try:
            mod = provider_module(str(prov))
            got = mod.fetch(mapping, frequency=str(freq), start=start, refresh=refresh)
            panels.setdefault(str(freq), []).append(got)
            row.update(status="ok", n_returned=got.shape[1], n_obs=len(got),
                       last_obs=str(got.index.max().date()) if len(got) else "")
        except Exception as exc:  # keep one bad provider from killing the run
            row.update(status=f"FAILED: {type(exc).__name__}", n_returned=0, n_obs=0,
                       last_obs="", error=str(exc)[:200])
        log.append(row)

    merged = {f: pd.concat(frames, axis=1).sort_index() for f, frames in panels.items()}
    manifest = pd.DataFrame(log)
    return merged, manifest


def ingest(
    catalog: pd.DataFrame | None = None,
    *,
    out_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    **kw,
) -> pd.DataFrame:
    """Fetch everything and write the raw store + the ingestion manifest.

    This is the entry point the monthly automation should call.
    """

    panels, manifest = fetch_catalog(catalog, **kw)
    out = Path(out_dir) if out_dir is not None else MANIFEST_PATH.parent / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for freq, frame in panels.items():
        frame.to_parquet(out / f"panel_{freq}.parquet")
    mp = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
    mp.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(mp, index=False)
    return manifest


def coverage(panels: Mapping[str, pd.DataFrame], catalog: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-series availability: first/last observation, count and missing share."""

    cat = load_catalog() if catalog is None else catalog
    meta = cat.set_index("series_id")
    rows = []
    for freq, frame in panels.items():
        for col in frame.columns:
            s = frame[col].dropna()
            info = meta.loc[col] if col in meta.index else {}
            rows.append({
                "series_id": col,
                "provider": info.get("provider") if hasattr(info, "get") else None,
                "country": info.get("country") if hasattr(info, "get") else None,
                "group": info.get("group") if hasattr(info, "get") else None,
                "frequency": freq,
                "n_obs": int(s.shape[0]),
                "first_obs": s.index.min() if not s.empty else pd.NaT,
                "last_obs": s.index.max() if not s.empty else pd.NaT,
                "missing_share": float(frame[col].isna().mean()),
                "publication_delay_days": info.get("publication_delay_days") if hasattr(info, "get") else None,
            })
    return pd.DataFrame(rows).sort_values(["frequency", "provider", "series_id"]).reset_index(drop=True)
