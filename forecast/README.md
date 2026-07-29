# forecast (stage — built)

The h = 1..8 counterpart to the nowcast stage. Same `core`/`targets` foundations,
quarterly horizons. Mixed-frequency model classes live in `MIDAS` (`ARNowcaster`,
`DFMNowcaster`); the plain-quarterly `DirectARXNowcaster` is project-local in
`forecast/models.py` (no wheel publish needed to evolve it) — all driven at any
horizon by `run_horizon_backtest` via `info.target_period`.

```
run_backtest(panel, spec, models, horizons, lookback_years)   quarterly-origin backtest
combine_by_horizon(bt, members)     real-time inv-MSE Combo per horizon
                                    (base b, horizon h uses only bases b' <= b-h)
horizon_scoreboard(bt, benchmark)   matched-sample RMSE by horizon + relatives
live_forecast(panel, spec, models)  the h=1..H path at today's origin
fan(bt, model)                      per-horizon band offsets (monotone-widened)
```

Production specs: `pipeline/config/metadata.py:FORECAST` (models / Combo members /
benchmark per target — China and copper today). Development:
`notebooks/<country>/04_forecast.ipynb`. The **US is never modelled**: its
external consensus (GDPNow / SPF) is ingested by the data stage and joined to
other targets' panels via `targets.usa.us_block`.

Empirical findings live in AGENTS.md ("EMPIRICAL RULES") and the 04 notebooks.
Next: BVAR second pass (also the scenario-stage engine), futures-curve baseline
for copper, wiring the China/copper paths into the Peru forecast.
