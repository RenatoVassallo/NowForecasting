"""Producers for the frozen China calibration assets (tracked, deterministic).

Regenerates, in dependency order, the three parquets frozen under
``calibration/``:

1. ``china_ladder_full``: the release-cycle nowcast ladder (the four members in
   ``pipeline.config.metadata.MODELS['china']``) on the target's common floor.
2. ``china_horizon_2012floor``: the h=1..8 horizon horse race of the five
   production members in ``pipeline.blocks._china_model`` (the conditional
   members receive the real-time nowcast function built FROM the ladder above,
   exactly as the live rule does).
3. ``china_tilt_weo``: the WEO tilt backtest: at each historical origin the two
   production BVAR systems' predictive draws are entropically tilted to the
   WEO round available that day (vintage-correct), with the documented
   untilted fallback when the solver declines.

Everything runs on the CURRENT panel rule (January structurally blank; the
combined January-February NBS release carried by February at its true
mid-March release date) and the exact production specifications. Seeds are the
production constants; the outputs are deterministic given the input caches.

Run (regenerate into a scratch directory, then freeze deliberately):

    PYTHONPATH=../MIDAS/src:../MacroPy/src python3.11 -m pipeline.lib.china_assets \
        --out output/china_assets_regen [--only ladder,horizon,tilt] [--freeze]

``--freeze`` copies the regenerated files into ``calibration/`` and rewrites
their MANIFEST entries (sha256, producer, panel rule, generation metadata).
"""

from __future__ import annotations

import contextlib
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

PANEL_RULE = "january_blank_combined_feb_release"
# the frozen predecessor's evaluation window (first base 2016Q3), pinned so a
# regeneration isolates the panel-rule change instead of shifting the sample
HORIZON_EVAL_START = "2016-07-29"
HORIZONS = tuple(range(1, 9))


def build_ladder(n_jobs: int = 8) -> pd.DataFrame:
    """The China release-cycle nowcast ladder on the corrected panel."""
    from nowcast import release_cycle
    from pipeline.config import metadata
    from pipeline.lib import modelset
    from targets import china as tgt

    _, _, panel = tgt.load_panel()
    models = modelset.build(metadata.MODELS["china"])
    return release_cycle.run_horse_race(
        panel, tgt.SPEC, models, n_jobs=n_jobs,
        days_before=metadata.BACKTEST["days_before"],
        step_days=metadata.BACKTEST["step_days"])


def build_horizon(ladder: pd.DataFrame) -> pd.DataFrame:
    """The five-member production horizon horse race (h = 1..8)."""
    import forecast
    from pipeline.blocks import _china_model as cm
    from pipeline.config import metadata
    from targets import china as tgt

    _, _, panel = cm.load_panel_with_us()
    ladder_hist = ladder[ladder.y_true.notna()]
    nowcast_fn = cm._realtime_nowcast_fn(ladder_hist,
                                         metadata.ADAPTIVE_MEMBERS["china"])
    models = cm.build_members(nowcast_fn)
    with contextlib.redirect_stdout(io.StringIO()):
        return forecast.run_backtest(panel, tgt.SPEC, models,
                                     horizons=HORIZONS,
                                     eval_start=HORIZON_EVAL_START)


def build_tilt(ladder: pd.DataFrame, horizon_bt: pd.DataFrame,
               as_of: pd.Timestamp) -> pd.DataFrame:
    """The WEO tilt backtest at every feasible historical origin.

    Bases: every horizon-backtest base whose h=8 outcome is released by
    ``as_of``. Origin per base: the first day after the base quarter's end
    month (the horizon harness's convention). Rounds and realized history are
    selected AT the origin; the tilt operates on the same stacked draws as the
    live rule (7-variable conditional system + the 3-variable BVAR).
    """
    import copy as _copy

    from MacroPy import BayesianVAR as _BV
    from MIDAS.realtime import RealtimeEngine
    from forecast import tilt as ftilt
    from pipeline.blocks import _china_model as cm
    from pipeline.config import metadata
    from sources import imf
    from targets import china as tgt

    as_of = pd.Timestamp(as_of).normalize()
    _, quarterly, panel = cm.load_panel_with_us()
    qser = quarterly[tgt.TARGET].dropna()
    qidx = pd.PeriodIndex(qser.index, freq="Q")
    delay = int(tgt.TARGET_DELAY_DAYS)
    release = qidx.to_timestamp(how="end") + pd.Timedelta(days=delay)

    ladder_hist = ladder[ladder.y_true.notna()]
    nowcast_fn = cm._realtime_nowcast_fn(ladder_hist,
                                         metadata.ADAPTIVE_MEMBERS["china"])

    bases = sorted(pd.PeriodIndex(pd.to_datetime(
        horizon_bt.base_quarter.unique()), freq="Q"))
    rows: list[dict] = []
    for B in bases:
        ref8 = B + 8
        if ref8.to_timestamp(how="end") + pd.Timedelta(days=delay) > as_of:
            continue                      # h=8 outcome not yet released
        origin = (B.asfreq("M", how="end") + 1).to_timestamp()
        t0 = time.time()

        p2 = _copy.copy(panel)
        ext = [(B + h).asfreq("M", how="end").to_timestamp()
               for h in range(1, 9)]
        ext = [e for e in ext if e not in p2.quarterly.index]
        if ext:
            p2.quarterly = pd.concat(
                [panel.quarterly,
                 pd.DataFrame({tgt.TARGET: np.nan}, index=ext)]).sort_index()
        info = RealtimeEngine(p2).information_set(
            origin, tgt.TARGET,
            target_period=(B + 8).asfreq("M", how="end").to_timestamp())

        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            mt = cm.build_members(nowcast_fn)["Cond-BVAR"]
            mt.fit(info)
            mt.nowcast(info)
            if not hasattr(mt, "_model"):
                print(f"  {B}: Cond-BVAR undefined at {origin.date()} "
                      "(min_train); skipped")
                continue
            dr7, _ = mt._model.conditional_forecast(
                conditions=mt._last_conditions, fhor=9,
                plot_forecast=False, shock_uncertainty=True)
            q7 = mt._last_system
            q3 = q7[["ip_cum_yoy", "m2_yoy", tgt.TARGET]]
            bv_kw = dict(lags=2, prior_type=2, post_draws=800, burnin=0.5,
                         fhor=9, seed=cm.BVAR_SEED, prior_params=dict(cm.TIGHT))
            if pd.Timestamp(q3.index[-1]) >= pd.Timestamp("2020-03-01"):
                bv_kw.update(covid_window=("2020Q1", "2021Q4"),
                             covid_mode="lenza-primiceri")
            b3 = _BV(q3, **bv_kw)
            b3.sample_posterior()
            dr3 = np.asarray(b3.forecast(fhor=9, plot_forecast=False)["forecast_draws"])
        draws = np.vstack([np.asarray(dr7)[:, :, list(q7.columns).index(tgt.TARGET)],
                           dr3[:, :, list(q3.columns).index(tgt.TARGET)]])

        grid = pd.period_range(B, periods=9, freq="Q")
        weo, rnd = imf.path("CHN", origin)
        realized = qser[release <= origin]
        cons = ftilt.annual_constraints(grid, weo, rnd, realized)
        tilted = ftilt.tilt_path(draws, cons)
        if tilted is None:
            path, ess, flag = draws.mean(axis=0), float(len(draws)), "declined"
        else:
            path, _, ess = tilted
            flag = "tilted"

        for h in range(1, 9):
            ref = (B + h).asfreq("M", how="end").to_timestamp()
            yt = qser.get(ref, np.nan)
            rows.append({
                "base_quarter": B.asfreq("M", how="end").to_timestamp(),
                "origin_date": origin, "horizon": h, "ref_quarter": ref,
                "round": rnd, "flag": flag, "ess": float(ess),
                "n_constraints": int(len(cons)),
                "y_untilted": float(draws[:, h].mean()),
                "y_tilted": float(path[h]), "y_true": float(yt)})
        print(f"  {B}: round {rnd}, {flag}, ess {100 * ess / len(draws):.0f}% "
              f"({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate China calibration assets")
    parser.add_argument("--out", default=str(REPO / "output" / "china_assets_regen"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--only", default="ladder,horizon,tilt")
    parser.add_argument("--freeze", action="store_true",
                        help="deliberately freeze the regenerated files into "
                             "calibration/ and update MANIFEST.json")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now().normalize()
    steps = set(args.only.split(","))

    def _write(df, name):
        p = out / f"{name}.parquet"
        df.to_parquet(p)
        print(f"[china-assets] wrote {p} ({len(df)} rows)")
        return p

    ladder_p = out / "china_ladder_full.parquet"
    if "ladder" in steps:
        t0 = time.time()
        ladder = build_ladder()
        print(f"[china-assets] ladder built in {time.time() - t0:.0f}s")
        _write(ladder, "china_ladder_full")
    else:
        ladder = pd.read_parquet(ladder_p)

    horizon_p = out / "china_horizon_2012floor.parquet"
    if "horizon" in steps:
        t0 = time.time()
        horizon = build_horizon(ladder)
        print(f"[china-assets] horizon race built in {time.time() - t0:.0f}s")
        _write(horizon, "china_horizon_2012floor")
    else:
        horizon = pd.read_parquet(horizon_p)

    if "tilt" in steps:
        t0 = time.time()
        tilt = build_tilt(ladder, horizon, as_of)
        print(f"[china-assets] tilt backtest built in {time.time() - t0:.0f}s")
        _write(tilt, "china_tilt_weo")

    if args.freeze:
        from pipeline.blocks._china_model import MEMBERS
        from pipeline.lib.calibration_assets import freeze_asset
        from pipeline.lib.context import _code_version

        extra = {"panel_rule": PANEL_RULE,
                 "generated_on": str(as_of.date()),
                 "code_version": _code_version(),
                 "evaluation_regime": "pseudo_real_time_final_vintage"}
        for name, meta in (
            ("china_ladder_full", {}),
            ("china_horizon_2012floor",
             {"eval_start": HORIZON_EVAL_START, "members": MEMBERS}),
            ("china_tilt_weo", {"seed": 7}),
        ):
            entry = freeze_asset(name, out / f"{name}.parquet",
                                 producer="pipeline/lib/china_assets.py",
                                 extra={**extra, **meta})
            print(f"[china-assets] froze {name}: sha {entry['sha256'][:12]}")


if __name__ == "__main__":
    main()
