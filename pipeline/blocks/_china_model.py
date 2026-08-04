"""The China production forecast model, promoted from notebooks/china/forecast/03.

This module holds the SELECTED specification (members, priors, blend rule) as
tracked code and recomputes everything live at each run:

- node 1: the Adaptive-IC nowcast of the current unpublished quarter, from a
  FRESH weekly live sweep at the run's as-of date (never a cached live row);
- nodes 2..8: half the five-member Combo live forecast, half the WEO-consistent
  path obtained by entropic tilting of the live conditional-BVAR draws to the
  WEO round available at the as-of date.

Two frozen calibration assets remain INPUTS (see calibration/MANIFEST.json),
hash-verified on read and hashed into the provenance: the horse-race backtest
(member forecasts used for combination weights and error fits) and the nowcast
ladder backtest (historical rows for adaptive weights and band pools). The
published centre path itself is never read from a cache; if a required input
is missing or altered this module raises instead of restamping anything.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ._common import REPO

SAMPLE_START = "2012-01-01"       # regime floor for the VARs (notebooks/china/03)
US_COLS = ["us_vix", "us_gdp_saar_m", "us_gdp_yoy_m", "us_cpi_yoy", "us_fedfunds",
           "spf_gdp_h0", "spf_gdp_h1", "spf_gdp_h2", "spf_gdp_h3", "spf_gdp_h4"]
SYSTEM = ["us_vix", "us_gdp_yoy_m", "us_cpi_yoy", "us_fedfunds", "ip_cum_yoy", "m2_yoy"]
TIGHT = {"mn_mean": 1, "lamda1": 0.2, "lamda2": 0.5, "lamda3": 1, "lamda4": 1e5}
MEMBERS = ["RW", "D-ARX(leaders)", "BVAR(3v)", "Cond-BVAR", "Cond-BVAR+SS"]
ALPHA = 0.5                       # published weight on the WEO-tilted path
HORIZONS = tuple(range(1, 9))
BVAR_SEED = 7                     # every BVAR draw in this block


def file_sha(path: Path, n: int = 12) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:n]


def _panel_sha(*frames, n: int = 12) -> str:
    h = hashlib.sha256()
    for f in frames:
        h.update(pd.util.hash_pandas_object(f.fillna(-9e9), index=True).to_numpy().tobytes())
    return h.hexdigest()[:n]


def load_panel_with_us():
    from targets import china as tgt
    from targets import usa

    monthly, quarterly, _ = tgt.load_panel()
    blk, metas = usa.us_block(monthly.index, columns=US_COLS)
    _, _, panel = tgt.load_panel(extra=blk, extra_meta=metas)
    return monthly, quarterly, panel


def _realtime_nowcast_fn(ladder_hist: pd.DataFrame, nc_members: list[str]):
    """Real-time nowcast of any quarter as of any origin, from the ladder rows."""
    g0 = ladder_hist[ladder_hist.model.isin(nc_members) & ladder_hist.y_hat.notna()].copy()
    g0["origin_date"] = pd.to_datetime(g0.origin_date)
    g0["ref_quarter"] = pd.to_datetime(g0.ref_quarter)

    def fn(info, period):
        stamp = period.to_timestamp(how="end").to_period("M").to_timestamp()
        g = g0[(g0.ref_quarter == stamp) & (g0.origin_date <= pd.Timestamp(info.origin))]
        if g.empty:
            return None
        return float(g[g.origin_date == g.origin_date.max()].y_hat.mean())
    return fn


def build_members(nowcast_fn):
    """The five production members, exactly as selected in the notebook."""
    from MIDAS import RandomWalkNowcaster
    from forecast.models import (BVARNowcaster, ConditionalBVARNowcaster,
                                 DirectARXNowcaster)
    from pipeline.config.metadata import CHINA_LEADERS
    from sources import imf as _imf
    from targets import china as tgt

    def cond_kwargs():
        return dict(variables=SYSTEM, spf_var="us_gdp_yoy_m", spf_mode="yoy",
                    fixed_vars=["us_fedfunds", "us_vix", "us_cpi_yoy"],
                    partial_vars=["ip_cum_yoy", "m2_yoy"],
                    quarterly_placed=["us_gdp_saar_m", "us_gdp_yoy_m"],
                    nowcast_fn=nowcast_fn, min_train=28,
                    lags=2, post_draws=800, shock_uncertainty=False)

    WEO_MAP = {tgt.TARGET: ("CHN", "NGDP_RPCH"), "us_gdp_yoy_m": ("USA", "NGDP_RPCH"),
               "us_cpi_yoy": ("USA", "PCPIPCH")}

    def ss_spec(q, info, judgment_sd=0.3):
        m, sd = {}, {}
        for col, (iso, subj) in WEO_MAP.items():
            if col not in q.columns:
                continue
            try:
                path_, rnd = _imf.path(iso, as_of=info.origin, subject=subj)
                yy = max(int(y) for y in path_.index if int(y) >= int(rnd[:4]))
            except Exception:
                continue
            m[col], sd[col] = float(path_.loc[yy]), judgment_sd
        for c in q.columns:
            if c not in m:
                m[c], sd[c] = float(q[c].iloc[-20:].mean()), 1.0 if c == tgt.TARGET else 2.0
        return {**TIGHT, "ss_mean": m, "ss_sd": sd}

    return {
        "RW": RandomWalkNowcaster(),
        "D-ARX(leaders)": DirectARXNowcaster(indicators=CHINA_LEADERS, min_train=24,
                                             _name="D-ARX(leaders)"),
        "BVAR(3v)": BVARNowcaster(variables=["ip_cum_yoy", "m2_yoy"], lags=2,
                                  post_draws=800, sample_start=SAMPLE_START,
                                  min_train=28, _name="BVAR(3v)"),
        "Cond-BVAR": ConditionalBVARNowcaster(
            **cond_kwargs(), sample_start=SAMPLE_START,
            prior_params=TIGHT, glp_select=(), _name="Cond-BVAR"),
        "Cond-BVAR+SS": ConditionalBVARNowcaster(
            **cond_kwargs(), sample_start=None,
            prior_params=ss_spec, glp_select=(), _name="Cond-BVAR+SS"),
    }


def live_profile(as_of: pd.Timestamp):
    """Recompute the published China profile at ``as_of``.

    Returns ``(prof, pool, provenance)`` where ``prof`` has one row per node
    (period, y_hat, source), ``pool`` is the nowcast node's information-bin
    residual pool, and ``provenance`` records rounds, members and input hashes.
    """
    import contextlib
    import io

    import forecast
    from forecast import tilt as ftilt
    from nowcast import release_cycle as rcyc
    from nowcast.release_cycle import (add_information_index, combine_release_cycle,
                                       conditional_bands)
    from pipeline.config import metadata
    from pipeline.lib import modelset
    from sources import imf
    from targets import china as tgt

    import copy as _copy0

    as_of = pd.Timestamp(as_of).normalize()
    monthly, quarterly, panel = load_panel_with_us()

    # The base quarter is set by the RELEASE RULE at as_of, never by whatever
    # the final snapshot happens to contain: later prints are masked out of the
    # working panel so the same code is exact at historical origins
    # (values remain the final snapshot: pseudo_real_time_final_vintage).
    delay = int(getattr(tgt, "TARGET_DELAY_DAYS", 18))
    qser = quarterly[tgt.TARGET].dropna()
    qidx = pd.PeriodIndex(qser.index, freq="Q")
    released_mask = (qidx.to_timestamp(how="end") + pd.Timedelta(days=delay)) <= as_of
    if not released_mask.any():
        raise RuntimeError(f"no released China GDP at {as_of.date()}")
    base_rule = qidx[released_mask].max()
    panel = _copy0.copy(panel)
    cut = base_rule.to_timestamp(how="end")
    panel.quarterly = panel.quarterly[panel.quarterly.index <= cut]
    quarterly = quarterly[quarterly.index <= cut]

    from pipeline.lib.calibration_assets import asset_path

    ladder_path = asset_path("china_ladder_full")
    horse_path = asset_path("china_horizon_2012floor")

    # ---- node 1: fresh live nowcast sweep + adaptive combination ------------
    nc_members = metadata.ADAPTIVE_MEMBERS["china"]
    ladder = pd.read_parquet(ladder_path)
    ladder = ladder[pd.to_datetime(ladder.origin_date) <= as_of]   # no future rows
    ladder_hist = ladder[ladder.y_true.notna()]           # NEVER reuse cached live rows
    nc_models = modelset.build({k: metadata.MODELS["china"][k] for k in nc_members})
    live_nc = rcyc.live_path(panel, tgt.SPEC, nc_models,
                             step_days=metadata.LIVE["step_days"], today=as_of)
    if not len(live_nc):
        raise RuntimeError("China live nowcast sweep produced no rows; refusing "
                           "to publish without a current node 1")
    rc_nc = add_information_index(pd.concat([ladder_hist, live_nc], ignore_index=True),
                                  panel, window_months=6)
    combo, _w = combine_release_cycle(rc_nc, nc_members, index_col="info_index",
                                      n_bins=4, min_train=6, method="inv_mse",
                                      name="Adaptive-IC")
    k = ["ref_quarter", "days_to_publication"]
    combo = combo.drop(columns="info_index", errors="ignore").merge(
        rc_nc.drop_duplicates(k)[k + ["info_index"]], on=k, how="left")
    rc_all = pd.concat([rc_nc, combo], ignore_index=True)
    _, pools = conditional_bands(rc_all, "Adaptive-IC", index_col="info_index",
                                 n_bins=4, levels=metadata.BANDS["levels"],
                                 lookback_years=metadata.BANDS["lookback_years"],
                                 exclude_years=metadata.BANDS["exclude_years"],
                                 min_quarters=metadata.BANDS["min_quarters"],
                                 collect_pools=True)
    live_combo = combo[combo.y_true.isna() & combo.y_hat.notna()].copy()
    live_combo["origin_date"] = pd.to_datetime(live_combo.origin_date)
    newest = live_combo.sort_values("origin_date").iloc[-1]
    nc_hat = float(newest.y_hat)
    info_idx = float(newest.get("info_index", np.nan))
    bin_now = int(np.clip(np.digitize([info_idx], np.linspace(0, 1, 5)[1:-1]), 0, 3)[0])
    pool_cands = [(q_, v) for (q_, b_), v in pools.items() if b_ == bin_now]
    if not pool_cands:
        raise RuntimeError(f"no nowcast residual pool for information bin {bin_now}")
    _, pool = max(pool_cands, key=lambda t: t[0])

    # ---- nodes 2..8: five-member Combo, tilted half-way to the WEO ----------
    nowcast_fn = _realtime_nowcast_fn(ladder_hist, nc_members)
    models = build_members(nowcast_fn)
    bt = pd.read_parquet(horse_path)
    bt = bt[bt.model.isin(MEMBERS)]
    # the cache supplies PAST bases only (combination weights); the current
    # base must come from the live recomputation, and at historical origins a
    # cached realized row for the same base would otherwise shadow the live
    # one inside the combine (quarter stamps are FIRST DAY OF THE END MONTH,
    # so compare against exactly that stamp)
    base_stamp = base_rule.asfreq("M", how="end").to_timestamp()
    bt = bt[pd.to_datetime(bt.base_quarter) < base_stamp]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        live = forecast.live_forecast(panel, tgt.SPEC, models,
                                      horizons=HORIZONS, today=as_of)
    rc2 = forecast.combine_by_horizon(pd.concat([bt, live], ignore_index=True),
                                      MEMBERS, name="Combo")
    combo_live = rc2[rc2.y_true.isna() & rc2.y_hat.notna()
                     & (rc2.model == "Combo")].set_index("horizon")
    if not set(HORIZONS) <= set(combo_live.index):
        raise RuntimeError("China Combo live forecast is incomplete: horizons "
                           f"{sorted(set(HORIZONS) - set(combo_live.index))} missing")

    # WEO-consistent path: tilt the live predictive draws of the two BVAR systems
    base_t = pd.Period(quarterly[tgt.TARGET].dropna().index.max(), freq="Q")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import copy as _copy

        from MacroPy import BayesianVAR as _BV
        from MIDAS.realtime import RealtimeEngine

        mt = build_members(nowcast_fn)["Cond-BVAR"]
        ext = [(base_t + h).to_timestamp(how="end").to_period("M").to_timestamp()
               for h in range(1, 9)]
        p2 = _copy.copy(panel)
        p2.quarterly = pd.concat([panel.quarterly,
                                  pd.DataFrame({tgt.TARGET: np.nan}, index=ext)]).sort_index()
        info_t = RealtimeEngine(p2).information_set(as_of, tgt.TARGET, target_period=ext[-1])
        mt.fit(info_t); mt.nowcast(info_t)
        if not hasattr(mt, "_model"):
            raise RuntimeError(
                "the production China rule is UNDEFINED at this origin: the "
                "2012-floored conditional BVAR has fewer than min_train "
                "quarters (feasible from roughly 2019Q1). An exact-chain "
                "replay cannot substitute a different specification.")
        dr7, _ = mt._model.conditional_forecast(conditions=mt._last_conditions, fhor=9,
                                                plot_forecast=False, shock_uncertainty=True)
        q7 = mt._last_system
        q3 = q7[["ip_cum_yoy", "m2_yoy", tgt.TARGET]]
        bv_kw = dict(lags=2, prior_type=2, post_draws=800, burnin=0.5, fhor=9,
                     seed=BVAR_SEED, prior_params=dict(TIGHT))
        if pd.Timestamp(q3.index[-1]) >= pd.Timestamp("2020-03-01"):
            bv_kw.update(covid_window=("2020Q1", "2021Q4"),
                         covid_mode="lenza-primiceri")
        b3 = _BV(q3, **bv_kw)
        b3.sample_posterior()
        dr3 = np.asarray(b3.forecast(fhor=9, plot_forecast=False)["forecast_draws"])
    draws = np.vstack([np.asarray(dr7)[:, :, list(q7.columns).index(tgt.TARGET)],
                       dr3[:, :, list(q3.columns).index(tgt.TARGET)]])
    grid_t = pd.period_range(base_t, periods=9, freq="Q")
    weo, rnd = imf.path("CHN", as_of)
    cons = ftilt.annual_constraints(grid_t, weo, rnd, quarterly[tgt.TARGET].dropna())
    tilted = ftilt.tilt_path(draws, cons)
    if tilted is None:
        # the entropic solver declined (no feasible constraint or collapsed
        # ESS): the documented fallback is the UNTILTED model path, flagged
        path_t, ess, tilt_declined = draws.mean(axis=0), float(len(draws)), True
    else:
        path_t, _, ess = tilted
        tilt_declined = False
    weo_line = pd.Series(path_t[1:], index=pd.PeriodIndex(grid_t[1:], freq="Q"))

    nodes = [{"period": base_t + 1, "y_hat": nc_hat, "source": "nowcast"}]
    for h in range(2, 9):
        per = pd.Period(pd.Timestamp(combo_live.loc[h, "ref_quarter"]), freq="Q")
        nodes.append({"period": per,
                      "y_hat": (1 - ALPHA) * float(combo_live.loc[h, "y_hat"])
                      + ALPHA * float(weo_line.loc[per]),
                      "source": f"blend a={ALPHA}"})
    prof = pd.DataFrame(nodes)

    provenance = {
        "weo_round": rnd,
        "weo_current_year": float(weo.get(as_of.year, np.nan)),
        "model_members": "+".join(MEMBERS),
        "blend_alpha": ALPHA,
        "nowcast_members": "+".join(nc_members),
        "nowcast_information_bin": bin_now,
        "tilt_ess_share": round(float(ess) / len(draws), 3),
        "tilt_declined": tilt_declined,
        "ladder_cache_sha": file_sha(ladder_path),
        "horse_cache_sha": file_sha(horse_path),
        "panel_sha": _panel_sha(monthly, quarterly),
    }
    return prof, np.asarray(pool, dtype=float), provenance
