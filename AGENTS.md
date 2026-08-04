# AGENTS.md — NowForecasting

Guide for AI agents (and humans) working in this repo. Read this first to recap
and resume.

## What this is

A **monorepo of macro nowcasting apps**. One reusable engine + shared analysis
pipeline drive a main **domestic** product (Peru GDP) and upstream **satellite**
models (China GDP, later commodities/US) that feed it, all the way to
publishable products (nowcast, fan chart, output gap, scenarios).

## Architecture (layers × lifecycle stages × apps)

Three orthogonal axes. Data-layers move data toward products; lifecycle stages
(nowcast, forecast, scenario) are model families that all rest on one engine;
apps pick a target and compose the engine.

```
NowForecasting/
├── sources/      DATA INGESTION: one loader per provider (fred, bcrp, inei, nbs)
│                 behind ONE uniform interface, driven by sources/catalog.csv.
├── analysis/     PREPARE + ANALYZE (target-agnostic): data (assemble_panel,
│                 series_summary, metadata_map), transforms (yoy/decum_yoy/...),
│                 information (availability, correlations), plots (house style),
│                 factors (PCA activity/property/inflation).
├── core/         STAGE-AGNOSTIC ENGINE: appconfig (NowcastApp), panel, preprocess,
│                 backtest harness, scoring (subperiods), robustness, set_app_root.
│                 Everything below builds on core; core imports nothing above it.
├── nowcast/      STAGE h=0: release-cycle ladder + monthly_target.        [built]
├── forecast/     STAGE h=1..8: run_backtest/combine_by_horizon/live_forecast/fan
│                 over MIDAS run_horizon_backtest; models.py = project-local
│                 DirectARXNowcaster (BaseNowcaster contract).              [built]
├── scenario/     STAGE: conditional forecasts under assumed paths.        [scaffold]
├── products/     ASSEMBLY: nowcast table, fan chart, output gap, scenario  [scaffold]
│                 table + the cross-app output contract (output/<app>/...).
├── targets/      PER-TARGET DATA INTERFACE (the registry): china, peru_gdp,
│                 usa (EXTERNAL: consensus ingested, never modelled - no
│                 metadata.MODELS entry, the skip hook bypasses model stages;
│                 targets.usa.us_block joins the US/SPF columns to any panel),
│                 copper (forecast-stage only). NO models here.
├── pipeline/     PRODUCTION RUN: `python -m pipeline.main` -> data snapshot ->
│                 satellites -> domestic -> report, saving a vintage of every
│                 artifact under runs/<id>/. config/params.py (switches),
│                 config/metadata.py (model specs), lib/ (job, reporting,
│                 store, modelset), stages/, main.py (the entry point).
├── notebooks/    EXPERIMENTATION per country (china/, peru/): 01/02/03 notebooks +
│                 an editable models.py, importing targets/<c> + the shared packages.
│                 Cross-cutting research notebooks live at notebooks/ root.
├── input/ output/ runs/ tests/ · pyproject.toml · README.md
```

Dependency rule: `sources → analysis → core → {nowcast, forecast, scenario} →
products`; `targets/` sits on core+analysis (data only); `pipeline/` orchestrates
targets+metadata into runs/. Model *classes* (RW, Bridge, DFM, and future
BVAR/FAVAR) live in the `MIDAS` package; production model *specs* live in
`pipeline/config/metadata.py`; the per-country `notebooks/<c>/models.py` are for experimentation.
Keep plotting/data-prep in `analysis`, never in `core`.

## The country pattern (notebook-first, replicable)

A country is `targets/<name>.py` (the data interface: loaders, transforms, delays,
groups, the common floor + SUBPERIODS) + a thin experimentation folder:

```
notebooks/<country>/
├── 01_data.ipynb        data + information environment (timeliness, correlations,
│                        candidate screening)
├── 02_backtesting.ipynb the horse race, scored on tgt.SUBPERIODS
├── 03_adaptive_and_reporting.ipynb  Adaptive-IC, bands, GDPNow-style report
├── 04_forecast.ipynb    the h=1..8 ladder, scoreboard vs RW, DM/CW, fan
│   (later: 05_forecast_product, 06_scenarios mirror the pattern)
├── models.py            the EDITABLE experimentation ladder (never read by runner)
└── output/ figures/     local artifacts (git-ignored)
```

Notebook bootstrap:
```python
HERE = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "models.py").exists())
REPO = HERE.parents[1]; sys.path[:0] = [str(REPO), str(HERE)]
from targets import china as tgt                       # the data interface
from analysis import information as info, plots        # prepare/analyze steps
from core import scoring                               # engine: subperiod scoring
import models                                          # the editable ladder
plots.set_style()
```

Backtest/adaptive/band mechanics come from ``nowcast.release_cycle``
(`run_horse_race`, `live_path`, `adaptive_combine`, `conditional_bands`) - ONE
implementation shared by the notebooks and the production runner. The forecast
stage mirrors this: ``forecast`` (`run_backtest`, `combine_by_horizon`,
`live_forecast`, `fan`) over `MIDAS.run_horizon_backtest`; production specs in
`pipeline/config/metadata.py:FORECAST` (models/members/benchmark per target).
EMPIRICAL RULES learned so far: China's benchmark at h=1-2 is RW (too smooth to
beat just after a release); D-ARX(leaders) wins h=4-8 among simple models; the
MacroPy BVAR (Minnesota prior -> RW shrinkage, Lenza-Primiceri COVID scaling,
wrapped as forecast/models.py:BVARNowcaster) is best at h=4 alone and lifts the
Combo at h=8; **AR(1) is OUT of China's Combo** (mean-reverts to the pre-2012
boom, real-time bias to -2.8pp - it made the fan lean far below the path); the
US block HURTS China at every horizon (notebooks/china/04); for the copper YoY
target AR beats RW everywhere and the CHINA block wins h>=3 (the dollar hurts).
Fan bands lean down for China even with good members: EVERY regression
over-predicts through a secular slowdown (bias persistent 2016-2026, so the
rolling band lookback is fair) - that lean is information, not a bug.
**h>0 band rule: COVID exclusion must apply to BOTH ends of a forecast** -
`forecast.fan` drops errors whose BASE quarter is in exclude_years in addition
to horizon_bands' ref-quarter filter; a forecast launched from a 2021 base-effect
print (+14% YoY) lands its ref in 2022 and would otherwise contaminate the pool
with -13pp errors (this was the "huge lower band" bug).

## Conventions (do not break)

- **Fan calibration matches the information state.** Published fan scales come
  from real-time errors measured at origins that MATCH the publication timing
  (Peru: day-1 and day-30 origin sets in `02_bt_S1*.parquet`, interpolated by
  the day the run publishes). The backtest harness's `base_quarter` is the
  just-ended UNPUBLISHED quarter, so fan node k maps to harness horizon k−1.
  For conditioned models whose error profile is flat in h with a real
  short-horizon dip, smooth scales with `fit_tpn_smooth(smoothing="monotone")`,
  never the power law (it moves width from the short horizons to the middle).
  The live model must receive every condition its backtest counterpart had:
  the current-quarter nowcast is served explicitly (`make_fn` in
  `pipeline/blocks/peru.py`) because the ladder lookup only covers backtested
  quarters.
- **Common evaluation floor:** ONE full-sample backtest
  (`tgt.BACKTEST_START/END`, 2010-2026, expanding window), scored on several
  **subperiods** (`tgt.SUBPERIODS`, label -> (start, end, exclude_years)):
  2010-2019, 2010-2026 excluding 2020-2021, and 2022-2026 (research), plus a
  ROLLING "last 10y ex-COVID" window in production. Selected leads
  `-120,...,-15,-1,0`, RMSE in target units. **Production band calibration is
  DYNAMIC**: `pipeline/config/metadata.py:BANDS` uses the last `lookback_years=10`
  of real-time errors (see `nowcast.release_cycle.conditional_bands`), levels
  50/70/90. Post-2020 windows
  keep COVID in the *estimation* window (a real-time forecaster saw it) but drop
  2020-2021 from *scoring* in the ex-COVID window. Lead 0 is the publication
  instant and is auto-excluded by the backtest; never use a positive lead.
- **Correlation window** is separate and wider (`tgt.CORR_START/END`, default
  2000-2019) and customizable; the window shows in the plot title via
  `plots.plot_*(..., period=(start, end))`.
- **All plots** come from `analysis.plots` (one soft palette + `set_style()`),
  never MIDAS `use_aer_style` or ad-hoc colors. **All horse-race scoring** comes
  from `core.scoring` (`horse_race_summary`, `rmse_by_lead`,
  `subperiod_summary`, `subperiod_curves`).
- **Where code goes:** MIXED-FREQUENCY model classes → `MIDAS`; plain-quarterly
  forecast classes → `forecast/models.py` (project-local: DirectARXNowcaster lives
  there, implementing the MIDAS BaseNowcaster contract); stage-agnostic engine → `core`;
  the h=0 workflow → `nowcast` (h≥1 → `forecast`, conditional → `scenario`);
  prepare/analyze → `analysis`; ingestion → `sources`; per-country data → `targets/`;
  experimentation → `notebooks/<country>/`.
  `core` must import nothing from a stage or an app.
- **Quarterly panels must use MIDAS's convention: a quarter is dated at the FIRST
  DAY OF ITS END MONTH** (Q1 -> March 1, months [3,6,9,12], day 1). Getting this
  wrong (e.g. quarter-start Jan 1) makes `BridgeNowcaster` silently fall back to
  the last value and mis-times the release cycle by two months. China's loader
  fixes this in `targets/china.py:load_raw`.
- **`summary.fits == False`** means a model returned values identical to the
  baseline on every row (a silent fallback). Its numbers are invalid; ignore it.
- **Bridge validity:** the Bell-2014 *proxy* bridge (`PooledBridgeNowcaster`,
  `MonthlyARNowcaster`) assumes the monthly proxy time-aggregates to the target as
  an accounting identity. TRUE for Peru (monthly GDP averages to quarterly GDP),
  FALSE for China (IP grows faster than GDP). For targets without a coherent
  monthly proxy, use bridge/MIDAS *regressions* (`BridgeNowcaster`,
  `PooledMIDASNowcaster`, `ADLMIDASNowcaster`) with the target on the left.

## Data ingestion

`sources/catalog.csv` is the single control point (series_id, provider,
provider_code, country, frequency, group, publication_delay_days, transform,
active). `sources.available(p)` / `sources.fetch(p, series)` are uniform across
providers; `sources.registry.ingest(cat)` is the monthly-automation entry point.
Open metadata gap: publication delays are missing for the INEI/NBS blocks.

## Environment

- The published wheel is pinned in `pyproject.toml` (`MIDAS 0.2.2`, `MacroPy`).
  Run `uv sync`. To develop against a local MIDAS checkout, uncomment
  `[tool.uv.sources]`.
- **Agent dev shortcut** (the repo `.venv` may be stale): run scripts with
  `PYTHONPATH=<local MIDAS checkout>/src python3.11` (pandas 2.x). Verify
  notebooks by executing their code cells as a script; **do not run the .ipynb
  files** (edit and let the user run them).
- Never commit `.env` (FRED key). **Publish policy (the repo is a public
  framework showcase):** the committed surface is the modeling framework and the
  pipeline stages (`analysis/`, `core/`, `nowcast/`, `forecast/`, `scenario/`,
  `targets/`, `pipeline/`, `products/*.py`, `tests/`). The entire data layer
  (`sources/`, `input/`, `output/`, caches, database files), the development
  `notebooks/` and every generated artifact (`products/**/*.csv`, figures, the
  weekly report) are git-ignored. Each `targets/<block>.py` documents the series
  a block consumes, so the data interface stays transparent without the
  retrieval code. A fresh clone therefore does NOT run end-to-end (no data);
  results live on the website.

## Status (2026-07-24)

Both apps built end-to-end on the identical framework (01 data + information,
02 subperiod horse race, 03 Adaptive-IC + uncertainty + GDPNow report). Ready for
the first commit.

- **China satellite:** no timely monthly-GDP proxy (PMIs 0d, GDP 18d), so
  indicator-driven. The winner **flips by regime**: RW ties in the calm 2010-2019,
  the activity/leaders bridges crush it post-COVID. Adaptive-IC is the robust
  product (near-best every subperiod, fires the whole cycle, calibrated bands).
- **Peru domestic:** HAS a timely monthly-GDP proxy (`g_pbim_yoy`; proxy and GDP
  both ~51d), so the proxy bridge beats RW by ~34% in *every* regime, but only
  fires the second half of the cycle (`finite ~0.47`). Adaptive-IC fires the whole
  cycle and is best on the shared sample every subperiod (`rel_RW ~0.61`).
- Longer memory of the whole project history lives in the agent's own notes.
