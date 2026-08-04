"""Exact-chain historical backtest: replay the LIVE rule at past origins.

    PYTHONPATH=../MIDAS/src:../MacroPy/src python3.11 -m pipeline.lib.exact_chain

For every base quarter the origin sits at day ORIGIN_DAY of the release cycle
(the current publication timing). At each origin the harness rebuilds exactly
what production builds: the official Adaptive-IC nowcast condition, the
SPF/WEO United States path at the origin's vintage, the recomputed China
profile (recursive everything, WEO round of the day), the masked monthly ToT
BVAR with its ragged edge and reconstructed daily flash months, the recursive
GDP-to-IP bridge, and the S1 conditional BVAR with the same custom paths,
transforms and hyperparameters as the live block. Simple benchmarks run at
the same origins on the same information sets.

Regime: values are the FINAL data snapshot masked by scalar release rules,
i.e. ``core.evaluation.EVALUATION_REGIME`` (pseudo_real_time_final_vintage).
SPF rows and WEO rounds are genuine vintages. Every origin records a
no-lookahead check block and the run refuses to store an origin that fails
one.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "backtests" / "exact_chain.parquet"
ORIGIN_DAY = 30
H = 8
SYSTEM = ["us_gdp_yoy_m", "ip_cum_yoy", "g_tdi", "exp_eco3m", "g_invq_m"]
BENCH_MEMBERS = ("S1-chain", "RW", "AR(2)", "BVAR-unconditional")

# The harness replays the exact production RULE but with REDUCED Monte-Carlo
# settings (ToT BVAR 2 chains x 3000 draws vs the published 4 x 6000), so its
# results are exact-rule-reduced-MC, never "computationally identical" to the
# live run. The measured chain-to-chain wobble of the ToT centre is documented
# in pipeline/blocks/commodities.py; quantify per regeneration at one origin.
CLAIM = "exact_rule_reduced_mc"
TOT_CHAINS = (7, 17)
TOT_DRAWS = 3000


class ExactChainConfigError(RuntimeError):
    """The stored artifact was generated under a different configuration."""


def _dep_versions() -> dict:
    import importlib.metadata as md

    out = {}
    for pkg in ("pandas", "numpy", "scipy", "midas", "macropy"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            try:
                mod = __import__({"midas": "MIDAS", "macropy": "MacroPy"}.get(pkg, pkg))
                out[pkg] = getattr(mod, "__version__", "local-src")
            except Exception:
                out[pkg] = "unavailable"
    return out


def fingerprint(bases, *, panel_sha: str, tot_chains=TOT_CHAINS,
                tot_draws=TOT_DRAWS) -> dict:
    """Everything that must be IDENTICAL for two artifact rows to be mergeable.

    ``requested_bases`` is recorded but excluded from the equality check:
    extending the chain to new bases under the same configuration is the
    purpose of resuming.
    """
    from core.evaluation import EVALUATION_REGIME
    from pipeline.blocks import _china_model as cm
    from pipeline.blocks import _peru_panel as pp
    from pipeline.blocks.peru import FAN_MC
    from pipeline.lib.bundle import registry_sha
    from pipeline.lib.calibration_assets import manifest_hashes
    from pipeline.lib.context import _code_version

    assets = {k: v for k, v in manifest_hashes().items()
              if not k.endswith("exact_chain.parquet")}   # output lineage, not input
    return {
        "schema": 1,
        "claim": CLAIM,
        "evaluation_regime": EVALUATION_REGIME,
        "code_version": _code_version(),
        "dependencies": _dep_versions(),
        "registry_sha": registry_sha(),
        "calibration_assets": assets,
        "panel_sha": panel_sha,
        "model_spec": {"system": pp.CANDIDATE_SYSTEMS["S1 as-specified"],
                       "prior": pp.TIGHT, "floor": pp.FLOOR,
                       "china_members": list(cm.MEMBERS),
                       "blend_alpha": cm.ALPHA},
        "release_rules": {"peru_gdp": 52, "china_gdp": 18, "ip_cum_yoy": 15,
                          "g_tdi": 40, "us_gdp": 30},
        "seeds": {"tot_chains": list(tot_chains), "bvar": cm.BVAR_SEED,
                  "fan_mc": dict(FAN_MC)},
        "draws": {"tot_draws": tot_draws, "bvar_post_draws": 800},
        "origin_day": ORIGIN_DAY,
        "requested_bases": [str(b) for b in bases],
    }


_FP_COMPARE_EXCLUDE = {"requested_bases"}


def _load_resume_state(out: Path, fp: dict):
    """Rows + completed bases of a resumable artifact, or refuse loudly."""
    out = Path(out)
    if not out.exists():
        return [], set()
    fpath = out.with_name("exact_chain_fingerprint.json")
    if not fpath.exists():
        raise ExactChainConfigError(
            f"{out} exists without a fingerprint: its configuration is "
            "unknown and rows must never be merged. Start a new artifact "
            "(supersede() or --new) or delete it deliberately.")
    stored = json.loads(fpath.read_text())
    current = json.loads(json.dumps(fp, default=str))
    diff = [k for k in sorted(set(stored) | set(current))
            if k not in _FP_COMPARE_EXCLUDE and stored.get(k) != current.get(k)]
    if diff:
        raise ExactChainConfigError(
            "exact-chain configuration changed since the stored artifact was "
            f"generated (differing keys: {', '.join(diff)}). Refusing to "
            "resume: rows from different configurations must never merge. "
            "Start a new artifact (supersede() or --new).")
    prev = pd.read_parquet(out)
    done = set(prev.base.unique())
    cpath = out.with_name("exact_chain_checks.json")
    checks = set(json.loads(cpath.read_text())) if cpath.exists() else set()
    if checks != done:
        raise ExactChainConfigError(
            "exact-chain checks file is out of step with the parquet "
            f"(parquet bases {sorted(done)}, checks {sorted(checks)}); the "
            "artifact is partial or corrupt. Supersede it and regenerate.")
    return [prev], done


def supersede(out: Path = OUT) -> Path:
    """Move an existing artifact (and sidecars) aside; returns the new home."""
    out = Path(out)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    moved = out.with_name(f"exact_chain_superseded_{stamp}.parquet")
    if out.exists():
        out.rename(moved)
    for side in ("exact_chain_checks.json", "exact_chain_fingerprint.json"):
        p = out.with_name(side)
        if p.exists():
            p.rename(out.with_name(f"{Path(side).stem}_superseded_{stamp}.json"))
    return moved


def origin_for(base: pd.Period) -> pd.Timestamp:
    cur = base + 1
    return (cur.to_timestamp(how="end") + pd.Timedelta(days=ORIGIN_DAY)).normalize()


def default_bases(start="2019Q1", end="2025Q3"):
    # the live China rule (2012-floor conditional BVARs, min_train 28)
    # is undefined before roughly 2019Q1; the chain replays the rule
    # only where the rule exists
    return list(pd.period_range(start, end, freq="Q"))


def released_by_rule(series: pd.Series, delay_days: int, as_of,
                     freq: str = "Q") -> pd.Series:
    s = series.dropna()
    idx = pd.PeriodIndex(s.index, freq=freq)
    rel = idx.to_timestamp(how="end") + pd.Timedelta(days=int(delay_days))
    return pd.Series(s.to_numpy(), index=idx)[rel <= pd.Timestamp(as_of)]


def no_lookahead_checks(t: pd.Timestamp, *, ladder_max_origin, spf_release,
                        weo_round: str, frame_releases: dict) -> dict:
    """Assert nothing used at origin ``t`` postdates it; return the evidence."""
    checks = {"origin": str(t.date())}
    if pd.notna(ladder_max_origin):
        assert pd.Timestamp(ladder_max_origin) <= t, \
            f"ladder row from {ladder_max_origin} used at origin {t.date()}"
        checks["ladder_max_origin"] = str(pd.Timestamp(ladder_max_origin).date())
    if spf_release is not None:
        assert pd.Timestamp(spf_release) <= t, \
            f"SPF survey released {spf_release} used at origin {t.date()}"
        checks["spf_release"] = str(pd.Timestamp(spf_release).date())
    ym = weo_round[:7]
    assert pd.Timestamp(ym + "-01") <= t, \
        f"WEO round {weo_round} used at origin {t.date()}"
    checks["weo_round"] = weo_round
    for name, rel in frame_releases.items():
        if rel is None:
            continue
        assert pd.Timestamp(rel) <= t, \
            f"{name}: cell released {rel} used at origin {t.date()}"
        checks[f"{name}_last_release"] = str(pd.Timestamp(rel).date())
    return checks


class ChainContext:
    """Everything loadable once and shared across origins."""

    def __init__(self):
        import targets
        from pipeline.config import metadata

        from nowcast.release_cycle import add_information_index, combine_release_cycle
        from pipeline.blocks._peru_panel import (build_panel, make_cond,
                                                 nowcast_lookup)

        self.make_cond = make_cond
        self.spec = targets.get("peru_gdp")
        self.mm, self.qq, self.panel = build_panel(self.spec)
        self.fn_ladder, _ = nowcast_lookup(self.panel)

        # the official nowcast rule over the full ladder history: the combined
        # rows are real-time by construction (weights use earlier quarters in
        # the same information bin only)
        from pipeline.lib.calibration_assets import asset_path
        ladder = pd.read_parquet(asset_path("peru_ladder_full"))
        self.ladder = ladder
        members = metadata.ADAPTIVE_MEMBERS["peru_gdp"]
        rc = add_information_index(ladder, self.panel, window_months=6)
        combo, _ = combine_release_cycle(rc, [m for m in members
                                              if m in ladder.model.unique()],
                                         index_col="info_index", n_bins=4,
                                         min_train=6, method="inv_mse",
                                         name="Adaptive-IC")
        combo["origin_date"] = pd.to_datetime(combo.origin_date)
        combo["ref_quarter"] = pd.to_datetime(combo.ref_quarter)
        self.combo = combo

        from targets import usa as us_t
        um, uq, _ = us_t.load_panel()
        self.spf = um[[f"spf_gdp_h{i}" for i in range(5)]].dropna(how="all")
        self.spf.index = pd.PeriodIndex(self.spf.index, freq="Q")
        self.spf = self.spf[~self.spf.index.duplicated(keep="last")]
        saar = um["us_gdp_saar_m"].dropna()
        saar.index = pd.PeriodIndex(saar.index, freq="Q")
        self.saar = saar
        usy = um["us_gdp_yoy_m"].dropna()
        usy.index = pd.PeriodIndex(usy.index, freq="Q")
        self.us_realised = usy[~usy.index.duplicated(keep="last")]

        from targets import china as cn
        cm, cq, _ = cn.load_panel()
        self.cn_ip = cm["ip_cum_yoy"]
        gq = cq[cn.TARGET].dropna()
        gq.index = pd.PeriodIndex(gq.index, freq="Q")
        self.cn_gdp = gq

        self.y = self.qq[self.spec.target].dropna()
        self.y.index = pd.PeriodIndex(self.y.index, freq="Q")


def official_nowcast_at(ctx: ChainContext, cur: pd.Period, t: pd.Timestamp):
    stamp = cur.to_timestamp(how="end").to_period("M").to_timestamp()
    g = ctx.combo[(ctx.combo.ref_quarter == stamp) & (ctx.combo.origin_date <= t)
                  & ctx.combo.y_hat.notna()]
    if g.empty:
        return None, pd.NaT
    row = g.sort_values("origin_date").iloc[-1]
    return float(row.y_hat), row.origin_date


def us_path_at(ctx: ChainContext, t: pd.Timestamp, grid):
    from pipeline.blocks.usa import _blend_weight, _spf_yoy, weo_at
    from sources import imf

    # newest survey whose approximate release (quarter start + 45d) has passed
    rel = ctx.spf.index.to_timestamp(how="start") + pd.Timedelta(days=45)
    avail = ctx.spf[rel <= t]
    survey_q = avail.index.max() if len(avail) else None
    spf_release = (survey_q.to_timestamp(how="start") + pd.Timedelta(days=45)) \
        if survey_q is not None else None
    saar_hist = released_by_rule(ctx.saar, 30, t)
    sp = _spf_yoy(avail, saar_hist, survey_q) if survey_q is not None else {}
    weo, rnd = imf.path("USA", t)
    us_rel = released_by_rule(ctx.us_realised, 30, t)
    vals = []
    for k, q in enumerate(grid, start=1):
        if q in us_rel.index:
            vals.append(float(us_rel[q])); continue
        a = _blend_weight(k)
        s_ = sp.get(q, np.nan)
        w_ = weo_at(weo, q.year)
        v = ((1 - a) * s_ + a * w_ if np.isfinite(s_) and np.isfinite(w_)
             else s_ if np.isfinite(s_) else w_)
        vals.append(float(v))
    if not np.isfinite(vals).all():
        raise RuntimeError(f"US path unresolved at {t.date()}: {vals}")
    return vals, rnd, spf_release


def run_origin(ctx: ChainContext, base: pd.Period, *, tot_chains=(7, 17),
               tot_draws=3000, s1_draws=1500) -> tuple[list[dict], dict]:
    import copy as _copy

    import forecast
    from forecast.models import BVARNowcaster
    from pipeline.blocks._china_model import live_profile
    from pipeline.blocks._common import arith_to_log_yoy, released_first
    from pipeline.blocks.commodities import tot_path

    t = origin_for(base)
    cur = base + 1
    grid = [base + h for h in range(1, H + 1)]

    y_rel = released_by_rule(ctx.y, int(ctx.spec.target_delay_days), t)
    assert y_rel.index.max() == base, \
        f"base {base} vs released Peru GDP {y_rel.index.max()} at {t.date()}"

    nc_hat, nc_origin = official_nowcast_at(ctx, cur, t)
    if nc_hat is None:
        raise RuntimeError(f"no ladder nowcast for {cur} at {t.date()}")

    us_vals, weo_rnd, spf_release = us_path_at(ctx, t, grid)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        prof, _pool, prov = live_profile(t)
    china_centre = pd.Series(prof.y_hat.to_numpy(), index=pd.PeriodIndex(prof.period, freq="Q"))

    periods_tot, centre_tot, _fits, seen_tot, base_tot = tot_path(
        t, chains=tot_chains, draws=tot_draws)
    tot_centre = pd.Series(centre_tot, index=pd.PeriodIndex(periods_tot, freq="Q"))
    tot_centre_log = pd.Series(arith_to_log_yoy(tot_centre.to_numpy(dtype=float)),
                               index=tot_centre.index)

    # recursive GDP-to-IP bridge on data released by the rule at t
    from pipeline.blocks._common import fill_structural_january

    ip_rel_m = ctx.cn_ip.copy()
    ends = pd.DatetimeIndex(ip_rel_m.index) + pd.offsets.MonthEnd(0)
    ip_rel_m[ends + pd.Timedelta(days=15) > t] = np.nan
    ip_rel_m = fill_structural_january(ip_rel_m).dropna()
    per = pd.PeriodIndex(ip_rel_m.index, freq="Q")
    gqm = ip_rel_m.groupby(per)
    ip_rel_q = gqm.mean().where(gqm.count() >= 3).dropna()
    cn_gdp_rel = released_by_rule(ctx.cn_gdp, 18, t)
    both = pd.concat([ip_rel_q.rename("ip"), cn_gdp_rel.rename("gdp")], axis=1).dropna()
    both = both[(both.index.year >= 2012) & (~both.index.year.isin([2020, 2021]))]
    if len(both) < 8:
        raise RuntimeError(f"bridge sample too short at {t.date()}")
    b1, b0 = np.polyfit(both["gdp"], both["ip"], 1)
    ip_fc = pd.Series(b0 + b1 * china_centre.to_numpy(), index=china_centre.index)

    tdi_rel = released_by_rule(
        ctx.mm["g_tdi"].groupby(pd.PeriodIndex(ctx.mm.index, freq="Q")).mean()
        .where(ctx.mm["g_tdi"].groupby(pd.PeriodIndex(ctx.mm.index, freq="Q")).count() >= 3),
        40, t)
    us_path = us_vals
    tdi_path, _ = released_first(grid, tdi_rel, tot_centre_log, "g_tdi")
    ip_path, _ = released_first(grid, ip_rel_q, ip_fc, "ip_cum_yoy")
    exp_rel = ctx.mm["exp_eco3m"].dropna()
    ends_e = pd.DatetimeIndex(exp_rel.index) + pd.offsets.MonthEnd(0)
    exp_rel = exp_rel[ends_e + pd.Timedelta(days=5) <= t]
    paths = {"us_gdp_yoy_m": us_path, "g_tdi": tdi_path, "ip_cum_yoy": ip_path,
             "exp_eco3m": [float(exp_rel.iloc[-1])] * H}

    checks = no_lookahead_checks(
        t, ladder_max_origin=nc_origin, spf_release=spf_release, weo_round=weo_rnd,
        frame_releases={
            "peru_gdp": y_rel.index.max().to_timestamp(how="end")
            + pd.Timedelta(days=int(ctx.spec.target_delay_days)),
            "china_ip": (pd.PeriodIndex(ip_rel_m.index, freq="M").max()
                         .to_timestamp(how="end") + pd.Timedelta(days=15)),
            "tot_target": (pd.Period(base_tot, freq="Q").to_timestamp(how="end")
                           + pd.Timedelta(days=40)),
        })

    # ---- the S1 chain forecast, and the benchmarks, on one panel ------------
    panel = _copy.copy(ctx.panel)
    cutq = base.to_timestamp(how="end")
    panel.quarterly = panel.quarterly[panel.quarterly.index <= cutq]

    def make_fn():
        def f(info_, period):
            if pd.Period(period, freq="Q") == cur:
                return nc_hat
            return ctx.fn_ladder(info_, period)
        return f

    models = {
        "S1-chain": ctx.make_cond(SYSTEM, nowcast_fn=make_fn(), custom=paths,
                                  name="S1-chain", draws=s1_draws),
        "BVAR-unconditional": BVARNowcaster(
            variables=["ip_cum_yoy", "g_tdi", "exp_eco3m", "g_invq_m"],
            lags=2, post_draws=800, sample_start="2003-01-01", min_train=28,
            _name="BVAR-unconditional"),
    }
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        lv = forecast.live_forecast(panel, ctx.spec, models,
                                    horizons=tuple(range(1, H + 1)), today=t)
    piv = lv.pivot_table(index="horizon", columns="model", values="y_hat")

    yy = y_rel.to_numpy(dtype=float)
    l1, l2 = float(yy[-1]), float(yy[-2])
    A = np.column_stack([yy[1:-1], yy[:-2], np.ones(len(yy) - 2)])
    coef, *_ = np.linalg.lstsq(A, yy[2:], rcond=None)
    ar_path = []
    a, b = l1, l2
    for _h in range(H):
        p_ = coef[0] * a + coef[1] * b + coef[2]
        ar_path.append(float(p_)); a, b = p_, a

    rows = []
    for h in range(1, H + 1):
        ref = base + h
        preds = {
            "S1-chain": float(piv.loc[h, "S1-chain"]) if h in piv.index else np.nan,
            "BVAR-unconditional": float(piv.loc[h, "BVAR-unconditional"])
            if h in piv.index and "BVAR-unconditional" in piv.columns else np.nan,
            "RW": l1,
            "AR(2)": ar_path[h - 1],
            "Nowcast-node" if h == 1 else "_skip": nc_hat if h == 1 else np.nan,
        }
        preds.pop("_skip", None)
        member_vals = [preds[m] for m in BENCH_MEMBERS]
        preds["Mean(4)"] = float(np.nanmean(member_vals))
        preds["Median(4)"] = float(np.nanmedian(member_vals))
        for model, hat in preds.items():
            rows.append(dict(base=str(base), origin=str(t.date()), h=h,
                             ref=str(ref), model=model, y_hat=hat,
                             nc_hat=nc_hat, weo_round=weo_rnd,
                             tot_months_observed=int(seen_tot[0]),
                             bridge_b1=round(float(b1), 4)))
    return rows, checks


def run_all(bases=None, out: Path = OUT, tot_chains=TOT_CHAINS,
            tot_draws=TOT_DRAWS, **kw) -> pd.DataFrame:
    from pipeline.blocks._china_model import _panel_sha

    ctx = ChainContext()
    bases = bases or default_bases()
    fp = fingerprint(bases, panel_sha=_panel_sha(ctx.mm, ctx.qq),
                     tot_chains=tot_chains, tot_draws=tot_draws)
    frames, done = _load_resume_state(out, fp)      # refuses on any mismatch

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fpath = out.with_name("exact_chain_fingerprint.json")
    if fpath.exists():                               # extend requested bases
        stored = json.loads(fpath.read_text())
        fp["requested_bases"] = sorted(set(stored.get("requested_bases", []))
                                       | set(fp["requested_bases"]))
    fpath.write_text(json.dumps(fp, indent=2, default=str))
    cpath = out.with_name("exact_chain_checks.json")

    for base in bases:
        if str(base) in done:
            continue
        t0 = time.time()
        try:
            rows, checks = run_origin(ctx, base, tot_chains=tot_chains,
                                      tot_draws=tot_draws, **kw)
        except Exception as exc:
            print(f"[exact-chain] {base}  FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        frames.append(pd.DataFrame(rows))
        cur = pd.concat(frames, ignore_index=True)
        tmp = out.with_suffix(".tmp")
        cur.to_parquet(tmp)
        tmp.replace(out)
        # checks move in lockstep with the parquet (same cadence, so a crash
        # between origins never leaves the two out of step)
        old = json.loads(cpath.read_text()) if cpath.exists() else {}
        old[str(base)] = checks
        cpath.write_text(json.dumps(old, indent=2))
        print(f"[exact-chain] {base}  ok ({time.time()-t0:.0f}s)", flush=True)
    return pd.read_parquet(out) if out.exists() else pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019Q1")
    parser.add_argument("--end", default="2025Q3")
    parser.add_argument("--new", action="store_true",
                        help="supersede any existing artifact and start fresh")
    args = parser.parse_args()
    if args.new:
        moved = supersede(OUT)
        print(f"[exact-chain] superseded previous artifact -> {moved.name}")
    run_all(default_bases(args.start, args.end))
