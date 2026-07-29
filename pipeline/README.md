# pipeline — the production run

One entry point runs the whole monthly update and saves a full, reproducible
**vintage** of everything it touched.

```bash
python -m pipeline.main
```

Stages run in order, each writing a short report:

1. **data** — refresh (optional) + snapshot the data vintage; report what's new.
2. **satellites** — nowcast each satellite target (China; USA/commodities to come).
3. **domestic** — nowcast the domestic target (Peru GDP).
4. **forecast** — h = 1..8 for every target with a `FORECAST` spec (China,
   copper): rolling backtest, real-time Combo per horizon, live path at today's
   origin, fan chart. The US is never modelled — its external consensus
   (GDPNow/SPF) is ingested in the data stage and vintaged per run.
5. **report** — assemble the figures + final `report.md`.

## Configure — the two knobs

- **`config/params.py`** — the on/off buttons: which stages, which targets,
  `REFRESH_DATA` (pull fresh NBS/INEI data, with explicit "no new releases"
  messages), `RUN_BACKTEST` (True = re-run the last-10-years horse race; False =
  fast monthly update reusing the previous run's backtest, live nowcast only),
  what to persist (`SAVE_MODELS`, `SAVE_FIGURES`, ...), `N_JOBS`.
- **`config/metadata.py`** — the model specs: each target's ladder as
  `name → (MIDAS class, kwargs)`, the Adaptive-IC members, the band levels. This is
  the single source of truth for *what* runs in production (the app `models.py`
  files are for experimentation only and are never read here).

## What every run saves

```
output/runs/<run_id>/
  manifest.json                 params, timings, and a typed index of every file
  report.md                     the final report
  data/<target>/                monthly.parquet, quarterly.parquet, whats_new.md   (the vintage)
  satellites/<target>/          domestic/<target>/
    nowcasts.parquet              all models + Adaptive-IC, full release cycle + info index
    weights.parquet  bands.parquet  latest.parquet
    metrics/summary_<subperiod>.csv
    models/ladder.pkl             the configured ladder
    figures/{current,horse_race,weights,gdpnow}.png
    forecast/{forecasts.parquet, horizon_scoreboard.csv, fan.png}
    report.md
  output/runs/latest -> <run_id>
```

## Extending

- **New target** (USA, commodities): add `targets/<name>.py` (a `Target` with
  `load_panel`), list it in `targets/__init__.py`, add its model block to
  `config/metadata.py`, and switch it on in `config/params.py`.
- **New model / spec**: add a row to `config/metadata.py`; nothing else changes.
- **New stage** (e.g. `forecast`): add `stages/<name>.py` and wire it in `run.py`.

The reusable engine lives in `core/`, `pipeline/`, `nowcast/` (and the coming
`forecast/`); `targets/` is the per-country data interface; `notebooks/<country>/`
is for experimentation and visualization only.

## The block chain (stage 5)

`pipeline/stages/chain.py` runs the notebook blocks in dependency order:

```
usa ----------\
china ---------> peru/forecast
commodities --/
```

Each upstream block publishes one CSV per horizon quarter holding a central path
and the two-piece-normal parameters fitted to its own real-time errors; Peru
imports them rather than re-deriving anything. The stage executes each notebook
in place (the notebook is the source of truth, so a run also refreshes the
analysis a human reads), validates the contract at every hand-off, and copies the
artifacts into `runs/<id>/chain/`.

A failing block does not abort the run: the chain reports it and says explicitly
that downstream will use the previous vintage, which is safer than a
half-updated forecast that looks current.

Configure with `CHAIN_BLOCKS` in `pipeline/config/params.py` - a subset while
iterating on one block, empty to run everything (~4 minutes, of which Peru's
Monte-Carlo fan is ~3).
