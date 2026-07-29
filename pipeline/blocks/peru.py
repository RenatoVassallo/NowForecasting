"""Peru GDP: nowcast + conditional BVAR on imported paths, with the BoE fan."""

from __future__ import annotations

import contextlib
import copy as _c
import io

import numpy as np
import pandas as pd

from ._common import REPO, complete_quarters, fan_frame, information_stamp, write

SYSTEM = ["us_gdp_yoy_m", "ip_cum_yoy", "g_tdi", "exp_eco3m", "g_invq_m"]
H = 8


def _panel(spec):
    import sys
    d = str(REPO / "notebooks/peru/forecast")
    if d not in sys.path:
        sys.path.insert(0, d)
    from common import build_panel, make_cond, nowcast_lookup
    return build_panel(spec), make_cond, nowcast_lookup


def build(blocks: dict | None = None, **_) -> tuple[pd.DataFrame, list[str], object]:
    import forecast
    import targets
    from forecast.boe_fan import fit_tpn_smooth
    from forecast.fan_mc import fit_tpn_mle
    from MIDAS.realtime import RealtimeEngine
    from nowcast.release_cycle import (add_information_index, combine_release_cycle,
                                       conditional_bands)
    from targets import china as cn

    spec = targets.get("peru_gdp")
    (mm, quarterly, panel), make_cond, nowcast_lookup = _panel(spec)
    base = pd.Period(quarterly[spec.target].dropna().index.max(), freq="Q")
    grid = [base + h for h in range(1, H + 1)]
    blocks = blocks or {}

    def load(name, fname):
        p = blocks.get(name) or next((REPO / "products/blocks").glob(fname), None)
        if p is None:
            raise FileNotFoundError(f"{name} block has not published {fname}")
        d = pd.read_csv(p); d["period"] = pd.PeriodIndex(d["quarter"], freq="Q")
        return d.set_index("period")

    us, china, tot = (load("usa", "us_path_uncertainty.csv"),
                      load("china", "china_path_uncertainty.csv"),
                      load("commodities", "tot_path_uncertainty.csv"))

    # China arrives as GDP; the system uses industrial production
    cn_m, cn_q, _ = cn.load_panel()
    ipq, _, _ = complete_quarters(cn_m["ip_cum_yoy"])
    gq = cn_q[cn.TARGET].dropna(); gq.index = pd.PeriodIndex(gq.index, freq="Q")
    both = pd.concat([ipq.rename("ip"), gq.rename("gdp")], axis=1).dropna()
    both = both[(both.index.year >= 2012) & (~both.index.year.isin([2020, 2021]))]
    b1, b0 = np.polyfit(both["gdp"], both["ip"], 1)

    ip_obs, _, ip_seen = complete_quarters(cn_m["ip_cum_yoy"])
    paths = {
        "us_gdp_yoy_m": [float(us["centre"].get(p, np.nan)) for p in grid],
        "g_tdi": [float(tot["centre"].get(p, np.nan)) for p in grid],
        "ip_cum_yoy": [float(ip_obs[p]) if p in ip_obs.index
                       else float(b0 + b1 * china["centre"].get(p, np.nan)) for p in grid],
        "exp_eco3m": [float(mm["exp_eco3m"].dropna().iloc[-1])] * H,
    }
    for k, v in paths.items():
        paths[k] = list(pd.Series(v).ffill().to_numpy())

    calib = {}
    for var, src, scale in (("us_gdp_yoy_m", us, 1.0), ("g_tdi", tot, 1.0),
                            ("ip_cum_yoy", china, abs(b1))):
        calib[var] = {h: ({"s": float(src["s"].get(p, np.nan)) * scale,
                           "gamma": float(src["gamma"].get(p, 0.0))}
                          if p in src.index and np.isfinite(src["s"].get(p, np.nan)) else None)
                      for h, p in enumerate(grid, start=1)}
    if ip_seen >= 3:                       # the current quarter is data, not a forecast
        calib["ip_cum_yoy"][1] = {"s": 1e-6, "gamma": 0.0}

    # live model + simulation
    fn, _ = nowcast_lookup(panel)
    ext = [(base + h).to_timestamp(how="end").to_period("M").to_timestamp() for h in range(1, H + 1)]
    p2 = _c.copy(panel)
    p2.quarterly = pd.concat([panel.quarterly,
                              pd.DataFrame({spec.target: np.nan}, index=ext)]).sort_index()
    info = RealtimeEngine(p2).information_set(pd.Timestamp.now().normalize(), spec.target,
                                              target_period=ext[-1])
    from forecast.fan_mc import simulate_var_fan
    sims = simulate_var_fan(lambda cu: make_cond(SYSTEM, nowcast_fn=fn, custom=cu, name="mc"),
                            info, paths, calib, target=spec.target, horizons=H,
                            n_scenarios=150, draws_per_scenario=8, seed=11)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        lv = forecast.live_forecast(panel, spec,
                                    {"S1": make_cond(SYSTEM, nowcast_fn=fn, custom=paths,
                                                     name="S1", draws=3000)},
                                    horizons=tuple(range(1, H + 1))).set_index("horizon")

    # nowcast node
    nc_bt = pd.read_parquet(next((REPO / "notebooks/peru").rglob("ladder_full.parquet")))
    MEM = ["Bridge(leaders)", "P-MIDAS(leaders)"]
    rc = add_information_index(nc_bt, panel, window_months=6)
    combo, _w = combine_release_cycle(rc, MEM, index_col="info_index", n_bins=4,
                                      min_train=6, method="inv_mse", name="Adaptive-IC")
    k_ = ["ref_quarter", "days_to_publication"]
    combo = combo.drop(columns="info_index", errors="ignore").merge(
        rc.drop_duplicates(k_)[k_ + ["info_index"]], on=k_, how="left")
    rc_all = pd.concat([rc, combo], ignore_index=True)
    _, pools = conditional_bands(rc_all, "Adaptive-IC", index_col="info_index", n_bins=4,
                                 levels=(0.30, 0.60, 0.90), lookback_years=12,
                                 exclude_years=(2020, 2021), min_quarters=6,
                                 collect_pools=True)
    cur = base + 1
    pub = cur.to_timestamp(how="end") + pd.Timedelta(days=spec.target_delay_days)
    dtp = (pd.Timestamp.now().normalize() - pub).days
    near = rc_all.iloc[(rc_all["days_to_publication"] - dtp).abs().argsort()[:150]]
    info_idx = float(near["info_index"].median())
    bin_now = int(np.clip(np.digitize([info_idx], np.linspace(0, 1, 5)[1:-1]), 0, 3)[0])
    _q, pool = max([(q_, v) for (q_, b_), v in pools.items() if b_ == bin_now],
                   key=lambda t: t[0])
    from MIDAS import BridgeNowcaster, PooledMIDASNowcaster
    from pipeline.config.metadata import PERU_LEADERS
    lead = [c for c in PERU_LEADERS if c in mm.columns]
    with contextlib.redirect_stdout(io.StringIO()):
        nc_live = forecast.live_forecast(panel, spec, {
            "Bridge": BridgeNowcaster(indicators=lead),
            "PMIDAS": PooledMIDASNowcaster(monthly_vars=lead)}, horizons=(1,))
    nc_hat = float(nc_live["y_hat"].mean())

    fits = [fit_tpn_mle(nc_hat + pool)] + [fit_tpn_mle(sims[:, i]) for i in range(1, H)]
    for f in fits:
        if f:
            f["mode_shift"] = 0.0
    centre = [nc_hat] + [float(lv.loc[h, "y_hat"]) for h in range(2, H + 1)]
    periods = [cur] + [base + h for h in range(2, H + 1)]
    src = ["nowcast"] + ["conditional BVAR (S1)"] * (H - 1)
    df = fan_frame(periods, centre, fits, src)

    # the outturn-based reading, for comparison
    s1 = next((REPO / "notebooks/peru").rglob("02_bt_S1.parquet"), None)
    if s1 is not None:
        d = pd.read_parquet(s1)
        d = d[(d.model == "S1 as-specified") & d.y_true.notna()]
        d = d[~pd.to_datetime(d.ref_quarter).dt.year.isin([2020, 2021])]
        d = d[~pd.to_datetime(d.base_quarter).dt.year.isin([2020, 2021])]
        emp = fit_tpn_smooth({h: (d[d.horizon == h].y_true - d[d.horizon == h].y_hat).to_numpy()
                              for h in range(1, 9)})
        df["outturn_sigma_left"] = [df.sigma_left.iloc[0]] + [emp[h]["sigma1"] for h in range(2, H + 1)]
        df["outturn_sigma_right"] = [df.sigma_right.iloc[0]] + [emp[h]["sigma2"] for h in range(2, H + 1)]

    stamp = information_stamp(spec, cur)
    stamp.update(information_index=round(info_idx, 3), information_bin=bin_now,
                 nowcast_pool_n=len(pool))
    out = write(df, REPO / "products/peru_gdp_fan.csv", stamp)
    lines = [f"- **Peru GDP**: " + ", ".join(f"{q} {v:.1f}%" for q, v in
                                             zip(df.quarter.head(4), df["mode"].head(4))),
             f"  - information state: {dtp:+d} days to publication, index {info_idx:.2f} "
             f"(bin {bin_now}) - the first node is {'well informed' if info_idx > 0.75 else 'early-cycle'}",
             f"  - 90% band {df.width90.iloc[0]:.2f}pp at the nowcast, "
             f"{df.width90.iloc[-1]:.2f}pp at h=8"]
    return df, lines, out
