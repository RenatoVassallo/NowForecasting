"""Shared logic for the model stages (satellites and domestic are identical
except for which targets they cover)."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

import targets

from ..lib import modelset, nowcast_job, reporting
from ..config import metadata


def _previous_backtest(store, role: str, name: str) -> tuple[pd.DataFrame | None, str | None]:
    """Newest previous run's member nowcasts (realized quarters only), for the
    fast RUN_BACKTEST=False mode. Returns (frame, run_id) or (None, None)."""

    runs = sorted((p for p in store.runs_dir.glob("*")
                   if p.is_dir() and not p.is_symlink() and p.name != store.run_id),
                  reverse=True)
    for r in runs:
        f = r / role / name / "nowcasts.parquet"
        if f.exists():
            rc = pd.read_parquet(f)
            bt = rc[(rc["model"] != metadata.ADAPTIVE["name"])      # members only
                    & rc["y_true"].notna()]                          # drop the old live quarter
            drop = [c for c in ("info_index",) if c in bt.columns]
            return bt.drop(columns=drop), r.name
    return None, None


def run_targets(store, params, panels, names, stage_name) -> list[str]:
    t0 = time.time()
    reports = []
    enabled = [n for n in names if params.TARGETS.get(n)]
    for name in enabled:
        if name not in metadata.MODELS:
            kind = ("forecast-stage target - no release-cycle nowcast"
                    if name in metadata.FORECAST
                    else "external/consensus target - no in-house model (data + report only)")
            print(f"    [{stage_name}] {name}: {kind}")
            continue
        t = targets.get(name)
        role = "satellites" if t.role == "satellite" else t.role

        bt = None
        if not getattr(params, "RUN_BACKTEST", True):
            bt, run_id = _previous_backtest(store, role, name)
            if bt is not None:
                print(f"    [{stage_name}] {name}: RUN_BACKTEST=False -> reusing the "
                      f"backtest from run {run_id} ({bt['ref_quarter'].nunique()} quarters); "
                      "computing only the live nowcast", flush=True)
            else:
                print(f"    [{stage_name}] {name}: RUN_BACKTEST=False but no previous "
                      "run found -> running the backtest once", flush=True)

        print(f"    [{stage_name}] {name} ...", flush=True)
        models = modelset.build(metadata.MODELS[name])
        result = nowcast_job.run(t, models, panel=panels.get(name),
                                 n_jobs=params.N_JOBS, backtest=bt)
        reporting.save_result(store, result, params)
        reports.append(reporting.target_report(result))
    store.log_stage(stage_name, {"targets": enabled, "seconds": round(time.time() - t0, 1)})
    return reports
