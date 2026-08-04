"""Stage 1: refresh (optional) + snapshot the data vintage, report what's new.

The vintage saved here is exactly the data every later stage uses. If
``params.REFRESH_DATA`` is on, fresh data is pulled first (needs keys + the private
loaders); otherwise the committed caches are the vintage.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

import targets

from ..lib import reporting


def _previous_run(store) -> Path | None:
    runs = sorted(p for p in store.runs_dir.glob("*")
                  if p.is_dir() and p.name != store.run_id and not p.is_symlink())
    return runs[-1] if runs else None


def _prev_monthly(prev: Path | None, name: str):
    if prev is None:
        return None
    f = prev / "data" / name / "monthly.parquet"
    return pd.read_parquet(f) if f.exists() else None


# Providers with no target of their own: the WEO judgment feed and the commodity
# block. They must be refreshed explicitly or the satellite chain silently runs
# on a stale vintage.
PROVIDERS = ("imf", "commodities")


def _events_context(store):
    """(registry, as_of, parser_version) for per-series event recording."""
    from pipeline.lib.context import resolve_as_of
    from pipeline.lib.data_availability import load_registry

    ctx = getattr(store, "ctx", None)
    try:
        registry = load_registry()
    except Exception:
        registry = {"series": []}
    return (registry, resolve_as_of(ctx),
            getattr(ctx, "code_version", None))


def _refresh_providers(store=None) -> list[str]:
    import importlib

    from pipeline.lib import refresh_events as ev

    registry, as_of, parser = _events_context(store)
    out = []
    for name in PROVIDERS:
        codes = ev.codes_for(registry, provider=name)
        try:
            mod = importlib.import_module(f"sources.{name}")
            fn = getattr(mod, "refresh_current", None) or getattr(mod, "refresh", None)
            msgs = fn() if fn else [f"{name}: no refresh hook"]
            # the IMF feed has both a vintage archive and a live round
            if name == "imf" and hasattr(mod, "refresh"):
                msgs = list(msgs) + list(mod.refresh())
            out += [f"{name}: {m}" if not m.lower().startswith(name) else m for m in msgs]
            ev.record_batch(codes, "successfully_updated", as_of=as_of,
                            detail="; ".join(msgs)[:400], parser_version=parser)
        except Exception as exc:
            out.append(f"{name}: refresh FAILED ({type(exc).__name__}: {exc}); using cache")
            status = ("source_unavailable"
                      if any(k in str(exc).lower() for k in
                             ("timeout", "connection", "http", "unreachable", "503", "502"))
                      else "ingestion_failure")
            ev.record_batch(codes, status, as_of=as_of,
                            detail=f"{type(exc).__name__}: {exc}", parser_version=parser)
    return out


def _refresh_target(name: str, store=None) -> list[str]:
    """Run the target's own refresh hook (see targets/<name>.py:refresh)."""
    import importlib

    from pipeline.lib import refresh_events as ev

    registry, as_of, parser = _events_context(store)
    codes = ev.codes_for(registry, target=name)
    try:
        mod = importlib.import_module(f"targets.{name}")
        fn = getattr(mod, "refresh", None)
        if fn is None:
            return ["no refresh hook; using committed data"]
        msgs = fn()
        ev.record_batch(codes, "successfully_updated", as_of=as_of,
                        detail="; ".join(str(m) for m in msgs)[:400],
                        parser_version=parser)
        return msgs
    except Exception as exc:
        ev.record_batch(codes, "ingestion_failure", as_of=as_of,
                        detail=f"{type(exc).__name__}: {exc}", parser_version=parser)
        return [f"refresh FAILED ({type(exc).__name__}: {exc}); using committed data"]


def run(store, params) -> tuple[dict, str]:
    t0 = time.time()
    enabled = [n for n in targets.REGISTRY if params.TARGETS.get(n)]

    refresh_msgs: dict[str, list[str]] = {}
    if getattr(params, "REFRESH_DATA", False):
        print("    [data] refreshing providers (IMF WEO, commodities) ...", flush=True)
        refresh_msgs["providers"] = _refresh_providers(store)
        for m in refresh_msgs["providers"]:
            print(f"      {m}", flush=True)
    if params.REFRESH_DATA:
        for name in enabled:
            print(f"    [data] refreshing {name} ...", flush=True)
            refresh_msgs[name] = _refresh_target(name, store)
            for m in refresh_msgs[name]:
                print(f"      - {m}", flush=True)

    prev = _previous_run(store)
    panels, sections = {}, []
    for name in enabled:
        t = targets.get(name)
        monthly, quarterly, panel = t.load_panel()      # AFTER refresh: the vintage
        panels[name] = (monthly, quarterly, panel)

        d = store.dir("data", name)
        store.save_df(d / "monthly.parquet", monthly, kind="vintage")
        store.save_df(d / "quarterly.parquet", quarterly.dropna(), kind="vintage")

        section = reporting.whatsnew_report(t.label, monthly, quarterly, _prev_monthly(prev, name))
        if refresh_msgs.get(name):
            section += "\n" + "\n".join(f"- {m}" for m in refresh_msgs[name])
        store.save_text(d / "whats_new.md", section)
        sections.append(section)

    text = "\n\n".join(sections)
    store.log_stage("data", {"targets": enabled, "refreshed": bool(params.REFRESH_DATA),
                             "refresh_messages": refresh_msgs,
                             "seconds": round(time.time() - t0, 1)})
    return panels, text
