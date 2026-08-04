"""Sequential exact-rule fan calibration, with its comparisons.

    PYTHONPATH=../MIDAS/src:../MacroPy/src python3.11 -m pipeline.lib.fan_calibration

Task 18 of the audit: intervals for origin ``t`` may use only forecast errors
whose OUTCOME was published before ``t``. Two knowable-before error sources
feed the pool: the legacy S1 day-30 backtest (the pre-chain rule, labelled
prior) and the exact-chain errors as they accrue. Each variant fits a
two-piece normal per horizon (pooled standardized skew, per-horizon RMS
scale, monotone smoothing across horizons) and is scored OUT OF SAMPLE on the
chain's own realized errors: 30/60/90 coverage with Wilson intervals, the
weighted interval score, the log score, and PIT.

Variants cover tasks 19 and 20 on a fixed evaluation window (ex-COVID both
ends, so calibration choices vary while the yardstick does not): symmetric
versus skewed, zero mode shift versus rolling horizon bias correction,
ref-only versus both-ends COVID exclusion, COVID included and downweighted,
plus two LOOKAHEAD-FLAGGED references fitted on today's full information (the
published production scales and the published structural scales). References
are context, never contenders: a sequential variant is adopted only if it
beats the incumbent on the pre-specified scores.

Not run here, documented as the extension hook: per-origin structural
simulation with joint satellite-error covariance and generated-regressor
uncertainty requires storing the Monte-Carlo draws in the exact chain
(rerun the harness with ``store_sims=True`` once implemented).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.lib.calibration_assets import asset_path

REPO = Path(__file__).resolve().parents[2]
CHAIN = REPO / "output" / "backtests" / "exact_chain.parquet"
PRIOR = "peru_s1_day30"           # frozen calibration asset (day-30 prior pool)
LEVELS = (0.30, 0.60, 0.90)
MIN_POOL = 30          # pooled errors needed before an origin becomes evaluable
MIN_H = 6              # per-horizon errors needed for that horizon's own scale
PERU_DELAY = 52

VARIANTS = {
    "sequential (primary)": dict(exclude="both_ends", skew=True, bias=False, covid_w=0.0),
    "sequential symmetric": dict(exclude="both_ends", skew=False, bias=False, covid_w=0.0),
    "sequential bias-corrected": dict(exclude="both_ends", skew=True, bias=True, covid_w=0.0),
    "sequential ref-only exclusion": dict(exclude="ref_only", skew=True, bias=False, covid_w=0.0),
    "sequential COVID included": dict(exclude="none", skew=True, bias=False, covid_w=1.0),
    "sequential COVID downweighted": dict(exclude="none", skew=True, bias=False, covid_w=0.25),
}
REFERENCES = ("published production scales [lookahead]",
              "published structural scales [lookahead]")


def _covid(frame: pd.DataFrame, cols=("base", "ref")) -> pd.Series:
    hit = pd.Series(False, index=frame.index)
    for c in cols:
        hit |= frame[c].astype(str).str.startswith(("2020", "2021"))
    return hit


def knowable_before(errors: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Errors whose outcome was PUBLISHED before origin ``t`` (release rule)."""
    ref_end = pd.PeriodIndex(errors["ref"], freq="Q").to_timestamp(how="end")
    return errors[ref_end + pd.Timedelta(days=PERU_DELAY) <= t]


def _weights(frame: pd.DataFrame, cfg: dict) -> np.ndarray:
    w = np.ones(len(frame))
    if cfg["exclude"] == "both_ends":
        w[_covid(frame, ("base", "ref")).to_numpy()] = 0.0
    elif cfg["exclude"] == "ref_only":
        w[_covid(frame, ("ref",)).to_numpy()] = 0.0
    else:
        w[_covid(frame, ("base", "ref")).to_numpy()] = cfg["covid_w"]
    return w


def _fit_variant(avail: pd.DataFrame, cfg: dict, horizons) -> dict | None:
    """Per-horizon TPN params from weighted knowable errors; None if starved."""
    from forecast.boe_fan import tpn_quantile
    from scipy.optimize import least_squares

    w = _weights(avail, cfg)
    use = w > 0
    if w[use].sum() < MIN_POOL:
        return None
    e_all, w_all = avail["err"].to_numpy()[use], w[use]
    hs = avail["h"].to_numpy()[use]

    scale, shift = {}, {}
    for h in horizons:
        m = hs == h
        if w_all[m].sum() >= MIN_H:
            scale[h] = float(np.sqrt(np.average(e_all[m] ** 2, weights=w_all[m])))
            shift[h] = float(np.average(e_all[m], weights=w_all[m]))
    if not scale:
        return None
    # monotone in h over the horizons that have data, extended flat outside
    ordered = sorted(scale)
    run = dict(zip(ordered, np.maximum.accumulate([scale[h] for h in ordered])))
    full_scale = {h: run.get(h, run[max(k for k in ordered if k <= h)]
                             if any(k <= h for k in ordered) else run[ordered[0]])
                  for h in horizons}
    full_shift = {h: (shift.get(h, 0.0) if cfg["bias"] else 0.0) for h in horizons}

    if cfg["skew"]:
        std = np.concatenate([(e_all[hs == h] / scale[h])[np.newaxis].ravel()
                              for h in ordered])
        wts = np.concatenate([w_all[hs == h] for h in ordered])
        probs = np.array([0.10, 0.25, 0.75, 0.90])
        idx = np.argsort(std)
        cw = np.cumsum(wts[idx]) / wts.sum()
        target = np.interp(probs, cw, std[idx])

        def resid(theta):
            s1, s2 = theta
            if min(s1, s2) <= 1e-6:
                return np.full(len(probs), 1e3)
            return tpn_quantile(probs, 0.0, s1, s2) - target

        sol = least_squares(resid, x0=[1.0, 1.0], bounds=([1e-3, 1e-3], [10, 10]))
        s1u, s2u = float(sol.x[0]), float(sol.x[1])
    else:
        s1u = s2u = 1.0

    return {h: {"sigma_left": s1u * full_scale[h], "sigma_right": s2u * full_scale[h],
                "shift": full_shift[h]} for h in horizons}


def _score_row(y, mode, sl, sr) -> dict:
    from core.evaluation import log_score, pit, weighted_interval_score
    from forecast.fan_mc import tpn_shortest_bands

    lo, hi = tpn_shortest_bands(np.array([mode]), np.array([sl]),
                                np.array([sr]), LEVELS)
    intervals = {lv: (float(lo[0, j]), float(hi[0, j])) for j, lv in enumerate(LEVELS)}
    out = {"wis": weighted_interval_score(y, mode, intervals),
           "log_score": log_score(y, mode, sl, sr),
           "pit": pit(y, mode, sl, sr)}
    for lv, (l_, h_) in intervals.items():
        out[f"hit{int(lv*100)}"] = float(l_ <= y <= h_)
        out[f"width{int(lv*100)}"] = h_ - l_
    return out


def load_errors() -> tuple[pd.DataFrame, pd.DataFrame]:
    import targets

    chain = pd.read_parquet(CHAIN)
    chain = chain[chain.model == "S1-chain"].copy()
    _, qq, _ = targets.get("peru_gdp").load_panel()
    y = qq["g_pbiq"].dropna()
    y.index = pd.PeriodIndex(y.index, freq="Q").astype(str)
    chain["y_true"] = chain.ref.map(y.to_dict())
    chain["err"] = chain.y_true - chain.y_hat
    chain["origin"] = pd.to_datetime(chain.origin)

    p = pd.read_parquet(asset_path(PRIOR))
    p = p[(p.model == "S1 as-specified") & p.y_true.notna()].copy()
    # harness base is the just-ended quarter: fan node k = harness h = k-1
    # is already the convention used live; here horizons align one-to-one
    # with the chain's ref = base + h definition
    prior = pd.DataFrame({
        "base": pd.PeriodIndex(pd.to_datetime(p.base_quarter), freq="Q").astype(str),
        "ref": pd.PeriodIndex(pd.to_datetime(p.ref_quarter), freq="Q").astype(str),
        "h": p.horizon.astype(int), "err": (p.y_true - p.y_hat).astype(float)})
    return chain, prior


def run(out_dir: Path | None = None):
    out_dir = Path(out_dir) if out_dir else REPO / "output" / "backtests"
    chain, prior = load_errors()
    horizons = sorted(chain.h.unique())
    origins = sorted(chain.origin.unique())

    # today's published scales, as LOOKAHEAD references
    fan = pd.read_csv(REPO / "products" / "peru_gdp_fan.csv")
    ref_scales = {
        REFERENCES[0]: {int(r.h): {"sigma_left": r.sigma_left,
                                   "sigma_right": r.sigma_right, "shift": 0.0}
                        for r in fan.itertuples()},
        REFERENCES[1]: {int(r.h): {"sigma_left": r.structural_sigma_left,
                                   "sigma_right": r.structural_sigma_right,
                                   "shift": 0.0}
                        for r in fan.itertuples()},
    }

    rows, diag = [], []
    for name, cfg in VARIANTS.items():
        for t in origins:
            fc = chain[chain.origin == t]
            avail = pd.concat([
                knowable_before(prior.assign(origin=pd.NaT), t)[["base", "ref", "h", "err"]],
                knowable_before(chain[chain.origin < t], t)[["base", "ref", "h", "err"]],
            ], ignore_index=True).dropna(subset=["err"])
            fit = _fit_variant(avail, cfg, horizons)
            if fit is None:
                continue
            diag.append({"variant": name, "origin": t, "pool_n": len(avail)})
            for r in fc.itertuples():
                if not np.isfinite(r.y_true):
                    continue
                pr = fit[int(r.h)]
                mode = float(r.y_hat) + pr["shift"]
                rows.append({"variant": name, "origin": t, "base": r.base,
                             "ref": r.ref, "h": int(r.h), "y_true": float(r.y_true),
                             "mode": mode,
                             **_score_row(float(r.y_true), mode,
                                          pr["sigma_left"], pr["sigma_right"])})
    for name in REFERENCES:
        for t in origins:
            for r in chain[chain.origin == t].itertuples():
                if not np.isfinite(r.y_true) or int(r.h) not in ref_scales[name]:
                    continue
                pr = ref_scales[name][int(r.h)]
                rows.append({"variant": name, "origin": t, "base": r.base,
                             "ref": r.ref, "h": int(r.h), "y_true": float(r.y_true),
                             "mode": float(r.y_hat),
                             **_score_row(float(r.y_true), float(r.y_hat),
                                          pr["sigma_left"], pr["sigma_right"])})

    res = pd.DataFrame(rows)
    # ONE fixed evaluation window for every variant: ex-COVID both ends
    res = res[~_covid(res, ("base", "ref"))]
    res.to_parquet(out_dir / "fan_calibration_cells.parquet")
    pd.DataFrame(diag).to_csv(out_dir / "fan_calibration_pools.csv", index=False)
    return res


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    from core.evaluation import coverage_ci
    from scipy import stats

    out = []
    for name, g in res.groupby("variant"):
        row = {"variant": name, "n": len(g),
               "wis": round(float(g.wis.mean()), 3),
               "log_score": round(float(g.log_score.mean()), 3)}
        for lv in LEVELS:
            k = int(g[f"hit{int(lv*100)}"].sum())
            lo, hi = coverage_ci(k, len(g))
            row[f"cov{int(lv*100)}"] = round(k / len(g), 3)
            row[f"cov{int(lv*100)}_ci"] = f"[{lo:.2f}, {hi:.2f}]"
            row[f"width{int(lv*100)}"] = round(float(g[f"width{int(lv*100)}"].mean()), 2)
        row["pit_ks_p"] = round(float(stats.kstest(g.pit, "uniform").pvalue), 3)
        out.append(row)
    return pd.DataFrame(out).sort_values("wis")


def production_fits(as_of, H: int = 8):
    """The ADOPTED production calibration (Phase 4 decision, 2026-08-03).

    Sequential-symmetric fits from errors knowable before ``as_of``:
    - day-30 anchor: legacy day-30 backtest rows plus exact-chain errors;
    - day-1 anchor: legacy day-1 backtest rows;
    both knowable-before filtered, RMS scale per horizon, monotone in h, no
    skew (small knowable pools estimate noise: the skewed variant scored
    worse out of sample), zero mode shift (rolling bias correction collapsed
    90 percent coverage to 0.75).

    Returns ``(day1_fits, day30_fits)`` shaped like the legacy
    ``fit_tpn_smooth`` output ({h: {"sigma1": s, "sigma2": s}}) so the Peru
    block's day-in-cycle interpolation is unchanged.
    """
    as_of = pd.Timestamp(as_of).normalize()
    cfg = dict(exclude="both_ends", skew=False, bias=False, covid_w=0.0)
    horizons = list(range(1, H + 1))

    def _legacy(asset):
        # frozen, hash-verified prior pools: a missing asset is a hard error
        # (silently dropping the prior would change published widths)
        d = pd.read_parquet(asset_path(asset))
        d = d[(d.model == "S1 as-specified") & d.y_true.notna()]
        return pd.DataFrame({
            "base": pd.PeriodIndex(pd.to_datetime(d.base_quarter), freq="Q").astype(str),
            "ref": pd.PeriodIndex(pd.to_datetime(d.ref_quarter), freq="Q").astype(str),
            "h": d.horizon.astype(int), "err": (d.y_true - d.y_hat).astype(float)})

    day1 = knowable_before(_legacy("peru_s1_day1"), as_of)
    d30 = _legacy(PRIOR)
    if CHAIN.exists():
        ch = pd.read_parquet(CHAIN)
        ch = ch[(ch.model == "S1-chain")].copy()
        import targets
        _, qq, _ = targets.get("peru_gdp").load_panel()
        y = qq["g_pbiq"].dropna()
        y.index = pd.PeriodIndex(y.index, freq="Q").astype(str)
        ch["err"] = ch.ref.map(y.to_dict()) - ch.y_hat
        d30 = pd.concat([d30, ch[["base", "ref", "h", "err"]].dropna()],
                        ignore_index=True)
    day30 = knowable_before(d30, as_of)

    def _shape(avail):
        fit = _fit_variant(avail.dropna(subset=["err"]), cfg, horizons)
        if fit is None:
            raise RuntimeError(
                f"fan calibration starved at {as_of.date()}: fewer than "
                f"{MIN_POOL} knowable errors; cannot publish honest scales")
        return {h: {"sigma1": fit[h]["sigma_left"], "sigma2": fit[h]["sigma_right"]}
                for h in horizons}

    return _shape(day1), _shape(day30)


if __name__ == "__main__":
    res = run()
    print(summarize(res).to_string(index=False))
