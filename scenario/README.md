# scenario (stage — scaffold)

Conditional forecasts: the target's path under an assumed path for a set of
conditioning variables (or a shock). The conditional counterpart to `forecast/`.

**Status:** interface stub. `build_scenarios(app, definitions)` declares the
conditioning paths (from `config.SCENARIOS`); `run_scenarios(panel, scenarios,
target, horizons)` produces the conditional forecasts. The conditional-forecast
machinery itself is a `MIDAS` primitive; this package holds only the workflow and
hands results to `products.scenario`.

**To fill:** add the conditional-forecast capability to `MIDAS`, implement the two
functions, then give each app a `06_scenarios.ipynb`.
