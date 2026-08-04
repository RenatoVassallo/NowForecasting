"""NowForecasting production pipeline - the single entry point.

    python -m pipeline.main

Five stages, each skippable via ``pipeline/config/params.py``:

1. **data**      - refresh every source (NBS, INEI, BCRP, FRED, SPF, IMF WEO,
                   commodities), snapshot the vintage, report what's new.
2. **nowcast**   - current-quarter nowcasts: China satellite and Peru domestic
                   (release-cycle ladders + Adaptive-IC + conditional bands).
3. **forecast**  - the block chain in dependency order (US -> China ->
                   commodities), then the Peru conditional BVAR and its fan
                   numbers. Pure Python; notebooks are development only.
4. **fanchart**  - the five published figures, one house style.
5. **report**    - report.pdf (cover, figures, tables) + report.md.

Every published fan carries an information-state stamp (days to publication,
information index), because the first node is only narrow when the run happens
late in the release cycle.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="A worker stopped while some jobs")
warnings.filterwarnings("ignore", category=FutureWarning, module="joblib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import params                        # noqa: E402
from pipeline.lib.context import RunContext               # noqa: E402
from pipeline.lib.store import RunStore                   # noqa: E402


def main(as_of=None, run_id: str | None = None) -> Path:
    ctx = RunContext.create(as_of=as_of, run_id=run_id)
    store = RunStore(params.RUNS_DIR, ctx=ctx, staged=True)
    from core.evaluation import EVALUATION_REGIME
    store.set_meta(evaluation_regime=EVALUATION_REGIME,
                   params={k: getattr(params, k) for k in
                           ("STAGES", "TARGETS", "REFRESH_DATA", "SAVE_MODELS",
                            "SAVE_FIGURES", "N_JOBS")})
    print(f"[run] {store.run_id}  as_of={ctx.as_of.date()}  "
          f"code={ctx.code_version}  ->  staging {store.root}")

    timings: dict = {}
    status = {s: "skipped" for s in
              ("data", "preflight", "nowcast", "forecast", "fanchart", "report")}
    panels: dict = {}
    whatsnew = ""
    lines: list[str] = []

    def _stage(name, fn):
        t = time.time()
        try:
            out = fn()
        except BaseException:
            status[name] = "failed"
            timings[name] = time.time() - t
            raise
        status[name] = "ok"
        timings[name] = time.time() - t
        return out

    try:
        if params.STAGES.get("data"):
            from pipeline.stages import data as s_data
            panels, whatsnew = _stage("data", lambda: s_data.run(store, params))
            print(f"[1/5] data ({timings['data']:.0f}s)")

        if any(params.STAGES.get(s) for s in
               ("nowcast", "forecast", "fanchart", "report")):
            from pipeline.lib.preflight import run_preflight
            _stage("preflight", lambda: run_preflight(store, params))
            print(f"[gate] availability preflight ({timings['preflight']:.0f}s) ok")

        if params.STAGES.get("nowcast"):
            from pipeline.stages import domestic as s_dom
            from pipeline.stages import satellites as s_sat
            lines += _stage("nowcast",
                            lambda: s_sat.run(store, params, panels)
                            + s_dom.run(store, params, panels))
            print(f"[2/5] nowcast ({timings['nowcast']:.0f}s)")

        if params.STAGES.get("forecast"):
            from pipeline.stages import chain as s_chain
            from pipeline.stages import domestic as s_dom
            lines += _stage("forecast",
                            lambda: s_chain.run(store, params, panels)
                            + s_dom.run_peru_fan(store, params))
            print(f"[3/5] forecast ({timings['forecast']:.0f}s)")

        if params.STAGES.get("fanchart"):
            from pipeline.stages import fanchart as s_fan
            lines += _stage("fanchart", lambda: s_fan.run(store, params, panels))
            print(f"[4/5] fanchart ({timings['fanchart']:.0f}s)")

        if params.STAGES.get("report"):
            from pipeline.stages import report as s_report
            _stage("report",
                   lambda: s_report.run(store, params, whatsnew, lines, timings))
            print(f"[5/5] report ({timings['report']:.0f}s) -> "
                  f"{store.root / 'report.pdf'}")
    except BaseException as exc:
        # record what happened, quarantine the staging dir, and re-raise: the
        # failed run must never be promoted, linked, or fallen back to
        store.set_meta(status="failed", timings=timings, stage_status=status,
                       error=f"{type(exc).__name__}: {exc}")
        store.write_manifest(strict=False)
        store.abort(f"{type(exc).__name__}: {exc}")
        print(f"[failed] staging kept for forensics: {store.root}")
        raise

    store.set_meta(status="success", timings=timings, stage_status=status)
    store.write_manifest()                           # hashes + validates artifacts
    store.mark_success()
    root = store.promote()                           # atomic rename into runs/
    if params.UPDATE_LATEST_SYMLINK:                 # only after promotion
        store.update_latest_symlink()
    print(f"[done] {root}")
    return root


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NowForecasting production run")
    parser.add_argument("--as-of", default=None,
                        help="release-selection cutoff (YYYY-MM-DD); today if omitted")
    parser.add_argument("--run-id", default=None,
                        help="explicit run identifier; derived from as-of if omitted")
    args = parser.parse_args()
    main(as_of=args.as_of, run_id=args.run_id)
