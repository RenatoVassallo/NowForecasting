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
sources/    data ingestion, one loader per provider, driven by catalog.csv  [private]
analysis/   prepare + analyze: transforms, panel assembly, information, plots
core/       stage-agnostic engine: config, backtest harness, scoring, robustness
nowcast/    stage h=0    release-cycle horse race + the Adaptive-IC combination   [built]
forecast/   stage h=1..8 direct-ARX ladder + combination + fan charts             [built]
scenario/   stage        conditional forecasts under assumed paths               [next]
products/   assembly: nowcast bands, fan chart, output gap, scenario table
targets/    per-target data interface (china, peru_gdp, usa external, commodities)
pipeline/   production run: python -m pipeline.main -> vintage runs/ + reports
notebooks/  experimentation per country (china/, peru/) + cross-cutting research [private]
```

The repository publishes the modeling framework and the production pipeline.
The data-retrieval layer (`sources/`), local databases and caches, development
notebooks and generated artifacts (fans, figures, the weekly report) are kept
out of version control; each `targets/<block>.py` documents the series a block
consumes, so the data interface is fully transparent even though the retrieval
code is not.

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
```

Built on the `MIDAS` package (pinned to a published wheel in `pyproject.toml`).
Secrets (a FRED key) load from a git-ignored `.env`; the China and Peru data
loaders are not distributed.
