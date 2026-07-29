# core

The stage-agnostic engine every app and every lifecycle stage rests on. Nothing
here is specific to nowcasting, forecasting, a single target, or a single app.

```
appconfig.py   NowcastApp / Subperiod: the declarative app descriptor
panel.py       metadata -> MIDAS panel plumbing
preprocess.py  transforms + X13 seasonal adjustment + processed-cache artifacts
backtest.py    expanding-window harness, YoY conversion, competition scoring
scoring.py     subperiod-aware horse-race scoring (rmse_by_lead, horse_race_summary,
               subperiod_summary, subperiod_curves)
robustness.py  cross-run robustness grids and relative-path tables
```

An app imports `core` and calls `core.set_app_root(HERE)` from its `config.py` so
every read and write lands in that app's own `input/` and `output/`. The stage
packages (`nowcast/`, `forecast/`, `scenario/`) build on top of this; the reverse
never happens (`core` imports nothing from a stage or an app).
