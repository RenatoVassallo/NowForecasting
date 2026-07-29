# products (scaffold)

The assembly layer between the modelling engine and the eventual website / API.
Turns backtest and forecast frames into publishable objects.

```
builders.py     gdp_nowcast · fan_chart · output_gap · scenario  (tidy tables)
trend_cycle.py  hp_filter · uc_output_gap -> the output-gap product
```

**Status:** interface stub (functions raise `NotImplementedError`). Kept
deliberately thin: plotting stays in `pipeline.plots`; this layer only composes
the objects a reader consumes, so every app publishes the same shapes. This is
also where the **cross-app output contract** lands — each app writes its
standardized nowcast/forecast frames to `output/<app>/` for downstream apps
(satellites -> domestic) to consume.
