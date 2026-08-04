"""Run switches: the on/off buttons for the production pipeline.

Flip these and press run (``python -m pipeline.main``). Nothing here is model-specific
- model specs and hyperparameters live in ``metadata.py``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "output" / "runs"     # one output/ tree; each run = output/runs/<id>

# --- which stages run (in order) ---------------------------------------------
# The graph is validated up front (pipeline/lib/stagegraph.py): report needs
# fanchart, fanchart needs forecast+nowcast, forecast needs nowcast. The safe
# default is the full coherent run; to re-render a report without recomputing,
# use the EXPLICIT mode: python -m pipeline.main --report-from RUN_ID
STAGES = {
    "data":     True,    # 1. refresh + snapshot every source, report what's new
    "nowcast":  True,    # 2. current-quarter nowcasts (China satellite, Peru domestic)
    "forecast": True,    # 3. block chain (usa -> china -> commodities) + Peru BVAR fan
    "fanchart": True,    # 4. the five published figures
    "report":   True,    # 5. report.pdf + report.md
}

# --- which targets run (must exist in targets.REGISTRY) ----------------------
TARGETS = {
    "china":    True,
    "peru_gdp": True,
    "usa":      True,     # external consensus target: data + report only, no in-house model
}

# --- data stage ---------------------------------------------------------------
REFRESH_DATA = True    # pull fresh NBS / INEI data and report new releases; the
                       # committed caches are used (with a message) when a source
                       # fails or has nothing new.

# --- backtesting --------------------------------------------------------------
RUN_BACKTEST = True    # True : re-run the release-cycle backtest over the last
                       #        10 years and report that horse race.
                       # False: fast monthly update - reuse the newest previous
                       #        run's backtest for weights/bands and only compute
                       #        the live weekly nowcast of the current quarter.

# --- what to persist per run --------------------------------------------------
SAVE_NOWCASTS = True   # all models' release-cycle nowcasts (not just the best)
SAVE_MODELS   = True   # pickle the configured model ladder
SAVE_FACTORS  = True   # intermediate factors (DFM) where available
SAVE_FIGURES  = True   # the report figures

# --- compute ------------------------------------------------------------------
N_JOBS = 8

# --- housekeeping -------------------------------------------------------------
UPDATE_LATEST_SYMLINK = True   # output/runs/latest -> newest run
PUBLISH_PRODUCTS = True        # after promotion ONLY: copy the promoted run's
                               # product surface to products/ (pipeline.lib.publish)
REPORT_PDF = True              # report stage MUST produce report.pdf; False is
                               # the explicit TeX-only artifact contract

# Restrict the chain to a subset while iterating, e.g. ("commodities", "peru").
# Empty or None runs every block in dependency order.
CHAIN_BLOCKS = ()

# --- availability preflight overrides (documented waivers only) --------------
# {internal_code: {"author": ..., "reason": ...}}; every waiver is recorded as a
# manually_overridden event in the append-only log and in the run manifest.
AVAILABILITY_OVERRIDES: dict = {}
