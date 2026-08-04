"""Peru GDP: nowcast + conditional BVAR on imported paths, with the BoE fan."""

from __future__ import annotations

import contextlib
import copy as _c
import io

import numpy as np
import pandas as pd

from ._common import (REPO, _dest, complete_quarters, fan_frame,
                      fill_structural_january, information_stamp,
                      released_first, released_last, write)

SYSTEM = ["us_gdp_yoy_m", "ip_cum_yoy", "g_tdi", "exp_eco3m", "g_invq_m"]
H = 8
FAN_MC = {"n_scenarios": 150, "draws_per_scenario": 8, "seed": 11}
EXP_DELAY_DAYS = 15         # business expectations: mid-next-month release


def _released_expectations(series_m: pd.Series, as_of) -> float:
    """Last expectations observation RELEASED at ``as_of`` (15-day rule)."""
    s = series_m.dropna()
    ends = pd.DatetimeIndex(s.index) + pd.offsets.MonthEnd(0)
    ok = s[ends + pd.Timedelta(days=EXP_DELAY_DAYS)
           <= pd.Timestamp(as_of).normalize()]
    if ok.empty:
        raise RuntimeError(f"no released expectations at {pd.Timestamp(as_of).date()}")
    return float(ok.iloc[-1])


def _empirical_fits(H, as_of):
    """Published fan scales: the Phase 4 ADOPTED rule.

    Sequential-symmetric calibration from errors KNOWABLE BEFORE ``as_of``
    (legacy day-1/day-30 S1 backtests plus the exact-chain errors as they
    accrue), fitted in ``pipeline.lib.fan_calibration.production_fits``. The
    harness's base quarter is the just-ended unpublished one (our node 1), so
    its horizon k-1 is our node k; day-1 and day-30 anchors interpolate by
    the day this run publishes, as before.
    """
    from pipeline.lib.fan_calibration import production_fits

    return production_fits(as_of, H=H - 1)


def _panel(spec):
    from pipeline.blocks._peru_panel import (build_panel, make_cond,
                                             nowcast_lookup)
    return build_panel(spec), make_cond, nowcast_lookup


def build(blocks: dict | None = None, ctx=None, out_dir=None,
          official_path=None, **_) -> tuple[pd.DataFrame, list[str], object]:
    import forecast
    import targets
    from forecast.fan_mc import fit_tpn_mle
    from MIDAS.realtime import RealtimeEngine
    from pipeline.lib.context import resolve_as_of
    from targets import china as cn

    spec = targets.get("peru_gdp")
    (mm, quarterly, panel), make_cond, nowcast_lookup = _panel(spec)
    today = resolve_as_of(ctx)                 # the run's canonical as-of date
    # the base quarter is set by the RELEASE RULE at as-of, never by whatever
    # the final snapshot happens to contain (a historical replay would
    # otherwise anchor on a print that was not out yet)
    base = released_last(quarterly[spec.target], spec.target_delay_days, today)
    grid = [base + h for h in range(1, H + 1)]
    blocks = blocks or {}

    # Satellite inputs come from THIS run, or from ONE validated prior bundle,
    # never from unversioned global product files (pipeline.lib.bundle enforces
    # as-of, hash, grid, registry and code-version coherence; partial current
    # runs are set aside whole rather than mixed with old vintages).
    from pipeline.config.params import RUNS_DIR
    from pipeline.lib.bundle import resolve_block_paths
    paths_map, bundle_meta = resolve_block_paths(blocks, RUNS_DIR, ctx=ctx)

    def load(name):
        d = pd.read_csv(paths_map[name])
        d["period"] = pd.PeriodIndex(d["quarter"], freq="Q")
        return d.set_index("period")

    us, china, tot = load("usa"), load("china"), load("commodities")

    # Units contract: the ToT block publishes ARITHMETIC YoY of the official
    # index, while this model's g_tdi regressor is LOG YoY of the same index
    # (they differ by up to 6pp at current growth rates). Convert the centre
    # exactly and the scale by the delta method BEFORE conditioning, so the
    # live condition has the identical definition the model was trained on.
    # The backtests never had this problem: they condition g_tdi from the
    # panel itself, which is already in log units.
    declared = str(tot["units"].iloc[0]) if "units" in tot.columns else "pct_yoy_arithmetic"
    if declared != "pct_yoy_arithmetic":
        raise ValueError(f"ToT block declares units {declared!r}; the Peru "
                         "interface only accepts pct_yoy_arithmetic")
    from ._common import arith_to_log_yoy
    tot = tot.copy()
    tot["centre"], tot["s"] = arith_to_log_yoy(
        tot["centre"].to_numpy(dtype=float), tot["s"].to_numpy(dtype=float))

    # China arrives as GDP; the system uses industrial production
    cn_m, cn_q, _ = cn.load_panel()
    ipq, _, _ = complete_quarters(fill_structural_january(cn_m["ip_cum_yoy"]))
    gq = cn_q[cn.TARGET].dropna(); gq.index = pd.PeriodIndex(gq.index, freq="Q")
    both = pd.concat([ipq.rename("ip"), gq.rename("gdp")], axis=1).dropna()
    both = both[(both.index.year >= 2012) & (~both.index.year.isin([2020, 2021]))]
    b1, b0 = np.polyfit(both["gdp"], both["ip"], 1)

    ip_obs, _, _seen = complete_quarters(fill_structural_january(cn_m["ip_cum_yoy"]))

    # One Peru-centered grid. Each satellite publishes on its OWN grid (a block
    # whose base quarter is already released starts one quarter after ours), so
    # every condition resolves as released data FIRST, block forecast second,
    # and a hole raises instead of silently conditioning on NaN. "Released"
    # honours the run's as-of date through each series' publication delay.
    def released_by(sq: pd.Series, delay_days: int) -> pd.Series:
        sq = sq.dropna()
        rel = pd.PeriodIndex(sq.index, freq="Q").to_timestamp(how="end") \
            + pd.Timedelta(days=delay_days)
        return sq[rel <= today]

    us_rel = mm["us_gdp_yoy_m"].dropna()
    us_rel.index = pd.PeriodIndex(us_rel.index, freq="Q")
    us_rel = released_by(us_rel[~us_rel.index.duplicated(keep="last")], 30)
    tdi_rel = released_by(complete_quarters(mm["g_tdi"])[0], 40)
    ip_rel = released_by(ip_obs, 15)
    ip_fc = pd.Series(b0 + b1 * china["centre"].astype(float).to_numpy(),
                      index=china.index)

    us_path, us_mask = released_first(grid, us_rel, us["centre"], "us_gdp_yoy_m")
    tdi_path, tdi_mask = released_first(grid, tdi_rel, tot["centre"], "g_tdi")
    ip_path, ip_mask = released_first(grid, ip_rel, ip_fc, "ip_cum_yoy")
    paths = {"us_gdp_yoy_m": us_path, "g_tdi": tdi_path, "ip_cum_yoy": ip_path,
             "exp_eco3m": [_released_expectations(mm["exp_eco3m"], today)] * H}
    masks = {"us_gdp_yoy_m": us_mask, "g_tdi": tdi_mask, "ip_cum_yoy": ip_mask}

    calib = {}
    for var, src, scale in (("us_gdp_yoy_m", us, 1.0), ("g_tdi", tot, 1.0),
                            ("ip_cum_yoy", china, abs(b1))):
        calib[var] = {h: (None if masks[var][h - 1]      # released quarters are data
                          else {"s": float(src["s"].get(p, np.nan)) * scale,
                                "gamma": float(src["gamma"].get(p, 0.0))}
                          if p in src.index and np.isfinite(src["s"].get(p, np.nan))
                          else None)
                      for h, p in enumerate(grid, start=1)}

    # nowcast node FIRST: the OFFICIAL artifact written by the nowcast stage is
    # the single definition of the current quarter (value, ensemble, weights,
    # information state, node TPN). The fan neither re-averages members nor
    # refits the node; it validates the artifact's as-of and consumes it.
    from pipeline.lib.nowcast_artifact import load_official

    cur = base + 1
    if official_path is None:
        raise ValueError(
            "official_path is required: pass the current run's "
            "peru_nowcast_official.csv (the fan never reads a global surface)")
    official = load_official(expected_as_of=today, path=official_path)
    if str(official["quarter"]) != str(cur):
        raise ValueError(f"official nowcast is for {official['quarter']} but the "
                         f"fan's first node is {cur}; refresh the nowcast stage")
    nc_hat = float(official["value"])
    dtp = int(official["days_to_publication"])
    info_idx = float(official["information_index"])
    bin_now = int(official["information_bin"])
    node_fit = {"s": float(official["s"]), "gamma": float(official["gamma"]),
                "sigma_left": float(official["sigma_left"]),
                "sigma_right": float(official["sigma_right"]), "mode_shift": 0.0}

    # live model + simulation. The VAR must see the nowcast it is published
    # with: the ladder lookup only covers backtested quarters, so the current
    # quarter's condition is served explicitly - without it the model launches
    # from its own unconditioned reading of the quarter and the whole path
    # inherits the gap (that was the kink between nodes 1 and 2).
    fn, _ = nowcast_lookup(panel)

    def make_fn(shift=0.0):
        def f(info_, period):
            if pd.Period(period, freq="Q") == cur:
                return nc_hat + shift
            return fn(info_, period)
        return f

    ext = [(base + h).to_timestamp(how="end").to_period("M").to_timestamp() for h in range(1, H + 1)]
    p2 = _c.copy(panel)
    p2.quarterly = pd.concat([panel.quarterly,
                              pd.DataFrame({spec.target: np.nan}, index=ext)]).sort_index()
    info = RealtimeEngine(p2).information_set(today, spec.target, target_period=ext[-1])
    from forecast.fan_mc import simulate_var_fan, tpn_ppf
    jit = np.random.default_rng(13)

    def factory(cu):
        # each scenario jitters the jump-off by the nowcast node's own TPN
        eps = float(tpn_ppf(jit.uniform(), 0.0,
                            node_fit["sigma_left"], node_fit["sigma_right"]))
        return make_cond(SYSTEM, nowcast_fn=make_fn(eps), custom=cu, name="mc")

    sims = simulate_var_fan(factory, info, paths, calib, target=spec.target, horizons=H,
                            **FAN_MC)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        lv = forecast.live_forecast(panel, spec,
                                    {"S1": make_cond(SYSTEM, nowcast_fn=make_fn(), custom=paths,
                                                     name="S1", draws=3000)},
                                    horizons=tuple(range(1, H + 1)),
                                    today=today).set_index("horizon")

    # published scales: the model's own real-time errors AT THE MATCHED
    # information state, interpolated between the day-1 and day-30 origin sets
    # by where in the cycle this run publishes (clamped outside the range).
    # This extends the nowcast node's information-state logic to every node.
    emp1, emp30 = _empirical_fits(H, today)
    day_in_cycle = int((today - cur.to_timestamp(how="end")).days)
    w = 0.0 if emp30 is None else float(np.clip((day_in_cycle - 1) / 29.0, 0.0, 1.0))
    fits = [node_fit]
    for k in range(2, H + 1):
        a = emp1[k - 1]
        b = emp30[k - 1] if emp30 is not None else a
        sl = (1 - w) * a["sigma1"] + w * b["sigma1"]
        sr = (1 - w) * a["sigma2"] + w * b["sigma2"]
        s = float(np.sqrt((sl * sl + sr * sr) / 2.0))
        fits.append({"mode_shift": 0.0, "sigma_left": float(sl), "sigma_right": float(sr),
                     "s": s, "gamma": float((sr * sr - sl * sl) / (2 * s * s))})
    centre = [nc_hat] + [float(lv.loc[h, "y_hat"]) for h in range(2, H + 1)]
    periods = [cur] + [base + h for h in range(2, H + 1)]
    src = ["nowcast"] + ["conditional BVAR (S1)"] * (H - 1)
    df = fan_frame(periods, centre, fits, src)

    # the structural Monte-Carlo reading, for comparison
    mc = [fit_tpn_mle(sims[:, i]) for i in range(1, H)]
    df["structural_sigma_left"] = [node_fit["sigma_left"]] + \
        [m["sigma_left"] if m else np.nan for m in mc]
    df["structural_sigma_right"] = [node_fit["sigma_right"]] + \
        [m["sigma_right"] if m else np.nan for m in mc]

    stamp = information_stamp(spec, cur, as_of=today)
    stamp.update(information_index=round(info_idx, 3), information_bin=bin_now,
                 nowcast_pool_n=int(official["pool_n"]), nowcast_conditioned=True,
                 day_in_cycle=day_in_cycle, calibration_weight_day30=round(w, 2),
                 tot_condition_units="log_yoy_pct_converted_from_arithmetic",
                 evaluation_regime="pseudo_real_time_final_vintage",
                 conditions_released="us={},tot={},ip={}".format(
                     sum(us_mask), sum(tdi_mask), sum(ip_mask)))
    if bundle_meta is not None:
        stamp.update(fallback_bundle_run=bundle_meta["run_id"],
                     fallback_bundle_as_of=bundle_meta["as_of"])
        lines_prefix = (f"  - WARNING: satellites from prior bundle "
                        f"{bundle_meta['run_id']} (as of {bundle_meta['as_of']})")
    else:
        lines_prefix = None
    out = write(df, _dest(out_dir, "peru_gdp_fan.csv"), stamp)
    lines = [f"- **Peru GDP**: " + ", ".join(f"{q} {v:.1f}%" for q, v in
                                             zip(df.quarter.head(4), df["mode"].head(4))),
             f"  - information state: {dtp:+d} days to publication, index {info_idx:.2f} "
             f"(bin {bin_now}) - the first node is {'well informed' if info_idx > 0.75 else 'early-cycle'}",
             f"  - 90% band {df.width90.iloc[0]:.2f}pp at the nowcast, "
             f"{df.width90.iloc[-1]:.2f}pp at h=8"]
    if lines_prefix:
        lines.insert(1, lines_prefix)
    return df, lines, out
