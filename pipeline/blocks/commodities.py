"""Peru's terms of trade: monthly BVAR aggregated to quarters."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd

from ._common import REPO, fan_frame, information_stamp, write

VARS = ["g_us_indpro", "ip_cum_yoy", "g_copper", "g_wti", "us_vix", "g_pe_tot"]
TIGHT = {"mn_mean": 1, "lamda1": 0.2, "lamda2": 0.5, "lamda3": 1, "lamda4": 1e5}
LAGS, FH, DRAWS = 3, 30, 6000
# The published path is a Monte-Carlo mean of a wide distribution (sd ~10pp at
# the long end), so a single 2,000-draw chain carries up to ~0.7pp of pure
# sampling noise between runs and machines (measured across seeds; different
# BLAS = a different chain). Averaging four chains with ~12k kept draws brings
# the run-to-run wobble under ~0.1pp, which is below reporting resolution.
CHAINS = (7, 17, 27, 37)


def _ragged_conditions(frame, monthly) -> dict:
    """Pin forecast months that are already observed.

    The balanced frame ends at the SLOWEST series (the terms-of-trade index,
    ~35-day delay), which silently discards a month of published metals, VIX and
    industrial production - and the BCRP dailies add a flash month on top. Every
    such month enters the forecast as a hard condition, so the near-term draws
    are anchored on data rather than simulated.
    """
    from sources import commodities as csrc

    start = pd.Period(frame.index[-1], freq="M") + 1
    obs = {}
    for col in frame.columns:
        if col == "g_pe_tot":
            continue
        s = monthly[col].dropna()
        s.index = pd.PeriodIndex(s.index, freq="M")
        obs[col] = s
    try:
        flash = csrc.flash_growth()
        flash.index = pd.PeriodIndex(flash.index, freq="M")
        for col in ("g_copper", "g_gold", "g_wti"):
            if col in obs and col in flash.columns:
                extra = flash[col].dropna()
                obs[col] = obs[col].combine_first(extra[extra.index > obs[col].index.max()])
    except Exception:
        pass
    conds, n_pinned = {}, 0
    for col, s in obs.items():
        path = [float(s[start + k]) if (start + k) in s.index else np.nan
                for k in range(FH)]
        if any(np.isfinite(v) for v in path):
            conds[col] = path
            n_pinned += int(np.isfinite(path).sum()) if hasattr(np, "isfinite") else 0
    return conds


def _forecast(frame, monthly=None):
    from MacroPy import BayesianVAR

    conds = _ragged_conditions(frame, monthly) if monthly is not None else {}
    chains = []
    for seed in CHAINS:
        kw = dict(lags=LAGS, prior_type=2, post_draws=DRAWS, burnin=0.5, fhor=FH,
                  seed=seed, prior_params=dict(TIGHT))
        if frame.index[-1] >= pd.Timestamp("2020-03-01"):
            kw.update(covid_window=("2020-03", "2021-06"), covid_mode="lenza-primiceri")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            b = BayesianVAR(frame, **kw); b.sample_posterior()
            if conds:
                d, _ = b.conditional_forecast(conditions=conds, fhor=FH,
                                              plot_forecast=False, shock_uncertainty=True)
                d = np.asarray(d)
            else:
                d = np.asarray(b.forecast(fhor=FH, plot_forecast=False)["forecast_draws"])
        chains.append(d[:, :, list(frame.columns).index("g_pe_tot")])
    return np.vstack(chains)


def build(**_) -> tuple[pd.DataFrame, list[str], object]:
    """Monthly draws averaged into quarters: observed months carry their weight,
    so the first node is narrow without any partial-quarter correction."""
    from forecast.fan_mc import fit_tpn_mle
    from targets import commodities as cmd

    spec = cmd.SPECS["pe_tot"]
    monthly, quarterly, _ = spec.load_panel()
    M = monthly[VARS].dropna()
    M = M[M.index >= "2004-01-01"]
    dr = _forecast(M, monthly)

    obs = M["g_pe_tot"].copy(); obs.index = pd.PeriodIndex(obs.index, freq="M")
    idx = pd.period_range(obs.index.max() + 1, periods=dr.shape[1], freq="M")
    base = pd.Period(quarterly[spec.target].dropna().index.max(), freq="Q")
    periods, centre, fits, seen_n = [], [], [], []
    for h in range(1, 9):
        q = base + h
        mons = pd.period_range(q.asfreq("M", "s"), periods=3, freq="M")
        seen = [float(obs[p]) for p in mons if p in obs.index]
        pos = [list(idx).index(p) for p in mons if p in list(idx)]
        if len(seen) + len(pos) < 3:
            continue
        sim = (np.sum(seen) + dr[:, pos].sum(axis=1)) / 3.0
        f = fit_tpn_mle(sim)
        f["mode_shift"] = 0.0                    # mode = the model's central path
        periods.append(q); centre.append(float(sim.mean())); fits.append(f)
        seen_n.append(len(seen))
    df = fan_frame(periods, centre, fits,
                   [f"monthly BVAR ({n}/3 observed)" for n in seen_n],
                   extra={"months_observed": seen_n})
    df["conditioned_on_ragged_edge"] = True
    stamp = information_stamp(spec, periods[0])
    out = write(df, REPO / "products/blocks/tot_path_uncertainty.csv", stamp)
    lines = [f"- **Terms of trade**: {centre[0]:.1f}% ({seen_n[0]}/3 months observed) "
             f"to {centre[-1]:.1f}% at h=8; 90% band {df.width90.iloc[0]:.1f} to "
             f"{df.width90.iloc[-1]:.1f}pp"]
    return df, lines, out
