"""Shared logic for the model stages (satellites and domestic are identical
except for which targets they cover)."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

import targets

from ..lib import modelset, nowcast_job, reporting
from ..config import metadata


def _current_registry_sha() -> str:
    from pipeline.lib.bundle import registry_sha

    return registry_sha()


_BT_SCHEMA = {"target", "ref_quarter", "origin_date", "days_to_publication",
              "model", "y_true", "y_hat"}


def _previous_backtest(store, role: str, name: str, *, expected_target: str,
                       members: set) -> tuple[pd.DataFrame | None, str | None]:
    """Newest COHERENT promoted run's member nowcasts, or (None, None).

    A prior backtest is evidence about today's model only when the run is
    promoted, no later than this as-of, and matches code version, registry,
    evaluation regime, target, member set and schema, with the artifact hash
    equal to the prior manifest's record. Anything less is skipped with a
    reason; the reused run and hash are stamped into the new manifest.
    """
    import hashlib
    import json

    ctx = getattr(store, "ctx", None)
    reg_now = _current_registry_sha()
    runs = sorted((p for p in store.runs_dir.glob("*")
                   if p.is_dir() and not p.is_symlink()
                   and not p.name.startswith(".")
                   and p.name != store.run_id),
                  reverse=True)
    for r in runs:
        f = r / role / name / "nowcasts.parquet"
        if not f.exists():
            continue

        def _skip(why: str):
            print(f"    [reuse] {r.name}: {why}; skipped")

        if not (r / "_SUCCESS").exists():
            _skip("not promoted")
            continue
        mpath = r / "manifest.json"
        if not mpath.exists():
            _skip("no manifest")
            continue
        try:
            man = json.loads(mpath.read_text())
        except Exception as exc:
            _skip(f"unreadable manifest ({exc})")
            continue
        if ctx is not None and man.get("as_of") \
                and pd.Timestamp(man["as_of"]) > ctx.as_of:
            _skip(f"as_of {man['as_of']} is after this run")
            continue
        if ctx is not None and man.get("code_version") != ctx.code_version:
            _skip(f"code version {man.get('code_version')} differs")
            continue
        if man.get("registry_sha") not in (None, reg_now):
            _skip("registry hash changed")
            continue
        regime = man.get("evaluation_regime")
        if regime not in (None, "pseudo_real_time_final_vintage"):
            _skip(f"evaluation regime {regime!r} differs")
            continue
        rel = f"{role}/{name}/nowcasts.parquet"
        recorded = {e["path"]: e.get("sha256") for e in man.get("files", [])}
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        if recorded.get(rel) is not None and recorded[rel] != sha:
            _skip("artifact hash differs from the prior manifest")
            continue
        rc = pd.read_parquet(f)
        if not _BT_SCHEMA <= set(rc.columns):
            _skip(f"schema lacks {sorted(_BT_SCHEMA - set(rc.columns))}")
            continue
        if str(rc["target"].iloc[0]) != str(expected_target):
            _skip(f"target {rc['target'].iloc[0]!r} differs")
            continue
        got = set(rc["model"].unique()) - {metadata.ADAPTIVE["name"]}
        if got != set(members):
            _skip(f"member set {sorted(got)} differs from {sorted(members)}")
            continue

        bt = rc[(rc["model"] != metadata.ADAPTIVE["name"])      # members only
                & rc["y_true"].notna()]                          # drop the old live quarter
        drop = [c for c in ("info_index",) if c in bt.columns]
        if hasattr(store, "_manifest"):
            store._manifest.setdefault("reused_backtests", {})[f"{role}/{name}"] = {
                "run_id": r.name, "artifact_sha256": sha}
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
            bt, run_id = _previous_backtest(
                store, role, name, expected_target=t.target,
                members=set(metadata.MODELS[name]))
            if bt is not None:
                print(f"    [{stage_name}] {name}: RUN_BACKTEST=False -> reusing the "
                      f"backtest from run {run_id} ({bt['ref_quarter'].nunique()} quarters); "
                      "computing only the live nowcast", flush=True)
            else:
                print(f"    [{stage_name}] {name}: RUN_BACKTEST=False but no previous "
                      "run found -> running the backtest once", flush=True)

        print(f"    [{stage_name}] {name} ...", flush=True)
        models = modelset.build(metadata.MODELS[name])
        ctx = getattr(store, "ctx", None)
        result = nowcast_job.run(t, models, panel=panels.get(name),
                                 n_jobs=params.N_JOBS, backtest=bt,
                                 as_of=ctx.as_of if ctx is not None else None)
        reporting.save_result(store, result, params)
        store.nowcast_results = getattr(store, "nowcast_results", {})
        store.nowcast_results[name] = result
        reports.append(reporting.target_report(result))
    store.log_stage(stage_name, {"targets": enabled, "seconds": round(time.time() - t0, 1)})
    return reports
