# NowForecasting

Real-time nowcasting, forecasting and scenario analysis of macroeconomic
aggregates. One reusable engine drives a family of country apps &mdash; a main
**domestic** product (Peru GDP) and upstream **satellite** models (China, and
later the US and commodities) that feed it &mdash; from raw data all the way to
publishable products.

> **Coming soon.** The live nowcasts, fan charts and scenarios will be published
> on a companion website. This repository is the open **framework and
> methodology**; the underlying data and results are not included here.

## What's inside

The pipeline runs along three axes: data layers move data toward products,
lifecycle **stages** (nowcast &rarr; forecast &rarr; scenario) are model families
that all rest on one engine, and **apps** pick a target and compose the engine.

```
sources/     data ingestion, one loader per provider, driven by the registry   [private]
analysis/    prepare + analyze: transforms, panel assembly, information, plots
core/        stage-agnostic engine: config, backtest harness, scoring, evaluation
nowcast/     stage h=0    release-cycle horse race + the Adaptive-IC combination
forecast/    stage h=1..8 conditional models + combination + two-piece-normal fans
scenario/    stage        conditional forecasts under assumed paths             [next]
products/    assembly: nowcast bands, fan chart, output gap, scenario table
targets/     per-target data interface (china, peru_gdp, usa external, commodities)
pipeline/    production run: python -m pipeline.main -> promoted vintage runs
calibration/ frozen research artifacts production consumes; MANIFEST.json tracked
notebooks/   experimentation per country (china/, peru/) + research             [private]
```

The repository publishes the modeling framework and the production pipeline.
The data-retrieval layer (`sources/`), local databases and caches, development
notebooks and generated artifacts (fans, figures, the weekly report) are kept
out of version control; each `targets/<block>.py` documents the series a block
consumes, so the data interface is fully transparent even though the retrieval
code is not.

## Production runs

```bash
python -m pipeline.main --as-of 2026-08-04
```

One canonical `as_of` date is fixed at the start and every stage selects
releases against it (`pipeline/lib/context.py`; a source-scan test keeps any
other wall-clock read out of the pipeline). Five stages run behind an
availability preflight gate: **data** (refresh + vintage snapshot), **nowcast**
(China satellite, Peru domestic), **forecast** (US &rarr; China &rarr;
commodities chain, then the Peru conditional BVAR and its fan), **fanchart**,
**report**.

Each run is staged in `output/runs/.staging/<run_id>/`, its artifacts hashed
and validated into `manifest.json` (git commit + dirty-diff digest, Python and
package versions, registry hash, model seeds, frozen-calibration hashes, stage
timings and statuses), marked `_SUCCESS`, then **atomically promoted** to
`output/runs/<run_id>/`; `latest` moves only after promotion. A failed run is
quarantined in the staging area with a `_FAILED` marker and is invisible to
everything downstream, including the coherent-bundle fallback
(`pipeline/lib/bundle.py`), which only ever serves one complete, hash-verified,
same-code-version prior bundle.

Published uncertainty is honest by construction: fans are calibrated
sequentially on the exact production rule's own errors, using only outcomes
published before each origin (`pipeline/lib/fan_calibration.py`), and every
evaluation artifact is stamped `pseudo_real_time_final_vintage` because the
input panels are final-vintage reconstructions, not true historical vintages.

## Reproducibility boundary

Public (this repository): the framework, the pipeline, the data registry
(`pipeline/config/data_registry.json`), the calibration manifest
(`calibration/MANIFEST.json`, sha256 per frozen artifact), and the test suite.
Private (local only, by data policy): provider retrieval code, raw caches and
panels, notebooks, calibration binaries, and run outputs.

Production never imports notebook code and never reads notebook caches;
research outputs enter production only as **frozen calibration assets**,
hash-verified on every read (`pipeline/lib/calibration_assets.py`). Both rules
are enforced by tests. On a public clone the suite runs the full contract
surface and a synthetic pipeline smoke run, with data-dependent tests skipping
themselves with an explicit reason; CI (`.github/workflows/ci.yml`) executes
exactly that boundary on every push.

Each app is deliberately thin &mdash; a declarative `config.py` plus notebooks
(`01_data`, `02_backtesting`, `03_adaptive_and_reporting`) over shared code. The
mixed-frequency model classes live in the companion
[`MIDAS`](https://github.com/RenatoVassallo/MIDAS) package; the stage folders here
are workflow only.

## Method

One expanding-window backtest per target, evaluated **across the release cycle**
(leads &minus;120 &hellip; 0) and scored on several subperiods. Models climb a
ladder from a random walk to bridge/MIDAS regressions and factor models; the
robust product is an **Adaptive-IC** combination re-weighted in real time by how
much of the panel has been released, with honest empirical prediction bands that
narrow as data arrive. See `AGENTS.md` for the architecture and conventions in
full.

## Setup

```bash
uv sync
uv run pytest -q
```

Built on the `MIDAS` and `MacroPy` packages (pinned to published wheels in
`pyproject.toml`; `uv.lock` is the reproducible environment). Secrets (a FRED
key) load from a git-ignored `.env`; the China and Peru data loaders are not
distributed. `docs/pipeline.md` documents the production path and run
lifecycle in full; `docs/audit/` records the 2026-08 technical audit and its
implementation log.
