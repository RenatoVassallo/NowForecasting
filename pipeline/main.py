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
from pipeline.lib.store import RunStore                   # noqa: E402
from pipeline.stages import chain as s_chain              # noqa: E402
from pipeline.stages import data as s_data                # noqa: E402
from pipeline.stages import domestic as s_dom             # noqa: E402
from pipeline.stages import fanchart as s_fan             # noqa: E402
from pipeline.stages import report as s_report            # noqa: E402
from pipeline.stages import satellites as s_sat           # noqa: E402


def main() -> Path:
    store = RunStore(params.RUNS_DIR)
    store.set_meta(params={k: getattr(params, k) for k in
                           ("STAGES", "TARGETS", "REFRESH_DATA", "SAVE_MODELS",
                            "SAVE_FIGURES", "N_JOBS")})
    print(f"[run] {store.run_id}  ->  {store.root}")

    timings: dict = {}
    panels: dict = {}
    whatsnew = ""
    lines: list[str] = []

    if params.STAGES.get("data"):
        t = time.time()
        panels, whatsnew = s_data.run(store, params)
        timings["data"] = time.time() - t
        print(f"[1/5] data ({timings['data']:.0f}s)")

    if params.STAGES.get("nowcast"):
        t = time.time()
        lines += s_sat.run(store, params, panels)      # China (and other satellites)
        lines += s_dom.run(store, params, panels)      # Peru
        timings["nowcast"] = time.time() - t
        print(f"[2/5] nowcast ({timings['nowcast']:.0f}s)")

    if params.STAGES.get("forecast"):
        t = time.time()
        lines += s_chain.run(store, params, panels)    # usa -> china -> commodities
        lines += s_dom.run_peru_fan(store, params)     # the Peru numbers
        timings["forecast"] = time.time() - t
        print(f"[3/5] forecast ({timings['forecast']:.0f}s)")

    if params.STAGES.get("fanchart"):
        t = time.time()
        lines += s_fan.run(store, params, panels)
        timings["fanchart"] = time.time() - t
        print(f"[4/5] fanchart ({timings['fanchart']:.0f}s)")

    if params.STAGES.get("report"):
        t = time.time()
        s_report.run(store, params, whatsnew, lines, timings)
        timings["report"] = time.time() - t
        print(f"[5/5] report ({timings['report']:.0f}s) -> {store.root / 'report.pdf'}")

    store.write_manifest()
    if params.UPDATE_LATEST_SYMLINK:
        store.update_latest_symlink()
    print(f"[done] {store.root}")
    return store.root


if __name__ == "__main__":
    main()
