"""Stage 4: h = 1..8 forecasts for every target with a FORECAST spec.

Backtest (rolling window) -> real-time Combo per horizon -> live path at today's
origin -> rolling-window scoreboard vs the benchmark -> fan chart. Artifacts land
under ``<role>/<target>/forecast/`` in the run store.
"""

from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import forecast as fc  # noqa: E402
import targets  # noqa: E402
from analysis import plots  # noqa: E402

from ..config import metadata  # noqa: E402
from ..lib import modelset  # noqa: E402


def _target_section(t, live_combo: pd.DataFrame, sb: pd.DataFrame, benchmark: str) -> str:
    lines = [f"### {t.label} - forecast (h = 1..{max(metadata.HORIZONS)})", ""]
    if len(live_combo):
        path = " | ".join(
            f"{pd.Period(pd.Timestamp(r.ref_quarter), freq='Q')}: {r.y_hat:.1f}"
            for r in live_combo.sort_values("horizon").itertuples() if r.horizon <= 4)
        lines.append(f"**{metadata.FORECAST_COMBO} path:** {path}  ")
    rel_col = f"rel_{metadata.FORECAST_COMBO}"
    if rel_col in sb.columns:
        h4 = sb[rel_col].get(4)
        if pd.notna(h4):
            lines.append(f"_Combo vs {benchmark} at h=4: {h4:.2f} "
                         f"(last {metadata.METRICS_LOOKBACK_YEARS}y, ex-COVID)._")
    lines.append("")
    return "\n".join(lines)


def run(store, params, panels) -> list[str]:
    t0 = time.time()
    sections, done = [], []
    for name, spec_cfg in metadata.FORECAST.items():
        if not params.TARGETS.get(name):
            continue
        t = targets.get(name)
        role = "satellites" if t.role == "satellite" else t.role
        print(f"    [forecast] {name} ...", flush=True)

        monthly, quarterly, panel = panels.get(name) or t.load_panel()
        models = modelset.build(spec_cfg["models"])

        bt = fc.run_backtest(panel, t, models, horizons=metadata.HORIZONS,
                             lookback_years=metadata.FORECAST_LOOKBACK_YEARS)
        live = fc.live_forecast(panel, t, models, horizons=metadata.HORIZONS)
        rc = fc.combine_by_horizon(pd.concat([bt, live], ignore_index=True),
                                   spec_cfg["members"], name=metadata.FORECAST_COMBO)

        benchmark = spec_cfg.get("benchmark", t.baseline)
        start = pd.Timestamp.now().normalize() - pd.DateOffset(years=metadata.METRICS_LOOKBACK_YEARS)
        sb = fc.horizon_scoreboard(rc, benchmark=benchmark, start=start,
                                   exclude_years=metadata.FAN["exclude_years"])

        d = store.dir(role, name, "forecast")
        store.save_df(d / "forecasts.parquet", rc, kind="forecasts")
        store.save_df(d / "horizon_scoreboard.csv", sb, kind="metrics")

        combo_live = rc[(rc["model"] == metadata.FORECAST_COMBO)
                        & rc["y_true"].isna() & rc["y_hat"].notna()]
        if params.SAVE_FIGURES and len(combo_live):
            bands = fc.fan(rc, metadata.FORECAST_COMBO, level=metadata.FAN["level"],
                           lookback_years=metadata.FAN["lookback_years"],
                           exclude_years=metadata.FAN["exclude_years"],
                           delay_days=t.target_delay_days)
            fig, ax = plt.subplots(figsize=(8.8, 4.6))
            plots.plot_fan(quarterly[t.target], combo_live, bands,
                           target_label=t.label,
                           title=f"{t.label}: {max(metadata.HORIZONS)}-quarter forecast fan",
                           ax=ax)
            store.save_fig(d / "fan.png", fig, kind="figure")
            plt.close(fig)

        sections.append(_target_section(t, combo_live, sb, benchmark))
        done.append(name)
    store.log_stage("forecast", {"targets": done, "seconds": round(time.time() - t0, 1)})
    return sections
