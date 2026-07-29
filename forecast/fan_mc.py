"""Monte-Carlo fan for a conditional VAR (Vega 2003, adapted).

The single-equation version of this method (Vega 2003; see the "How to build a
fan chart" report) draws paths for each exogenous driver from its own two-piece
normal, pushes them through the equation, and fits a two-piece normal to the
resulting sample at each horizon. The density of the target is the pushforward
of the input uncertainty.

Two things change when the model is a **VAR**:

1. **Shocks are correlated.** A single equation adds one scalar residual; a VAR
   adds a vector with covariance ``Sigma``. Drawing them independently would
   understate uncertainty when the target loads on several correlated variables,
   and overstate it when they offset. Here the VAR's own posterior draws supply
   the shocks, so ``Sigma`` (and parameter uncertainty) come along for free.
2. **Conditioning is not exogeneity.** In the VAR the conditioned variables are
   endogenous, held to a path by structural shocks (Waggoner-Zha). Perturbing an
   assumed path therefore feeds back into every other variable, which is exactly
   the transmission we want to measure.

So the simulation is: draw a *scenario* (a perturbed path for each conditioning
variable, from that variable's own two-piece normal), run the conditional
forecast, keep posterior draws, repeat. The resulting sample per horizon mixes
scenario uncertainty with correlated VAR shock uncertainty, and a two-piece
normal is fitted to it by maximum likelihood.

Parameterisation follows the report: scale ``s`` and skew ``gamma`` in (-1, 1),
with ``sigma_left = s sqrt(1-gamma)``, ``sigma_right = s sqrt(1+gamma)``.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
from scipy import optimize, stats

__all__ = ["tpn_scales", "tpn_cdf", "tpn_ppf", "tpn_moments", "tpn_shortest_bands",
           "fit_tpn_mle", "calibrate_from_errors", "calibrate_from_bands",
           "simulate_var_fan"]


# --------------------------------------------------------------------------- #
# two-piece normal (Vega's parameterisation)
# --------------------------------------------------------------------------- #
def tpn_scales(s, gamma):
    s = np.asarray(s, dtype=float)
    gamma = np.clip(np.asarray(gamma, dtype=float), -0.98, 0.98)
    return s * np.sqrt(1.0 - gamma), s * np.sqrt(1.0 + gamma)


def tpn_cdf(x, mode, sl, sr):
    x, mode = np.asarray(x, dtype=float), np.asarray(mode, dtype=float)
    den = sl + sr
    left = 2.0 * sl / den * stats.norm.cdf((x - mode) / sl)
    right = (sl - sr + 2.0 * sr * stats.norm.cdf((x - mode) / sr)) / den
    return np.where(x <= mode, left, right)


def tpn_ppf(p, mode, sl, sr):
    p = np.atleast_1d(np.asarray(p, dtype=float))
    split = sl / (sl + sr)
    out = np.empty_like(p)
    lo = p <= split
    out[lo] = mode + sl * stats.norm.ppf(np.clip(p[lo] * (sl + sr) / (2 * sl), 1e-12, 1 - 1e-12))
    hi = ~lo
    out[hi] = mode + sr * stats.norm.ppf(
        np.clip(0.5 + (p[hi] * (sl + sr) - sl) / (2 * sr), 1e-12, 1 - 1e-12))
    return out if out.size > 1 else float(out[0])


def tpn_moments(mode, sl, sr):
    mean = mode + np.sqrt(2.0 / np.pi) * (sr - sl)
    var = (1.0 - 2.0 / np.pi) * (sr - sl) ** 2 + sl * sr
    return mean, var


def tpn_shortest_bands(mode, sl, sr, coverages):
    """BoE shortest intervals containing the mode (report, section 4)."""

    mode, sl, sr = (np.asarray(v, dtype=float) for v in (mode, sl, sr))
    z = stats.norm.ppf((1.0 + np.asarray(coverages, dtype=float)) / 2.0)
    return mode[:, None] - sl[:, None] * z[None, :], mode[:, None] + sr[:, None] * z[None, :]


def fit_tpn_mle(sample):
    """Maximum-likelihood (mode, s, gamma) for one sample."""

    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return None
    s0 = max(float(np.std(x)), 1e-6)

    def negll(theta):
        mode, log_s, z = theta
        s = np.exp(log_s)
        gamma = np.tanh(z)
        sl, sr = tpn_scales(s, gamma)
        sig = np.where(x <= mode, sl, sr)
        return float(np.sum(((x - mode) / sig) ** 2) / 2 + x.size * np.log(sl + sr))

    best, val = None, np.inf
    for g0 in (-0.4, 0.0, 0.4):
        r = optimize.minimize(negll, [float(np.median(x)), np.log(s0), np.arctanh(g0)],
                              method="Nelder-Mead",
                              options={"maxiter": 4000, "fatol": 1e-8, "xatol": 1e-8})
        if r.fun < val:
            best, val = r, r.fun
    mode, log_s, z = best.x
    s, gamma = float(np.exp(log_s)), float(np.tanh(z))
    sl, sr = tpn_scales(s, gamma)
    return {"mode": float(mode), "s": s, "gamma": gamma,
            "sigma_left": float(sl), "sigma_right": float(sr), "n": int(x.size)}


# --------------------------------------------------------------------------- #
# calibrating each conditioning variable's own uncertainty
# --------------------------------------------------------------------------- #
def calibrate_from_errors(errors_by_h: dict, min_obs: int = 12) -> dict:
    """(s, gamma) per horizon from a block's own real-time forecast errors."""

    out = {}
    for h, e in errors_by_h.items():
        e = np.asarray(e, dtype=float)
        e = e[np.isfinite(e)]
        if e.size < min_obs:
            out[h] = None
            continue
        fit = fit_tpn_mle(e) if e.size >= 20 else None
        if fit is None:
            out[h] = {"s": float(np.std(e)), "gamma": 0.0}
        else:
            out[h] = {"s": fit["s"], "gamma": fit["gamma"]}
    return out


def calibrate_from_bands(centre, lo, hi, coverage: float = 0.90) -> dict:
    """(s, gamma) per horizon from a published fan's own bands.

    Inverts the BoE shortest-band formula: ``lo = mode - sigma_left z`` and
    ``hi = mode + sigma_right z``, so each side's scale is read straight off the
    published interval. Use when a block gives us a fan rather than an error
    sample (the China profile, for instance).
    """

    z = stats.norm.ppf((1.0 + coverage) / 2.0)
    out = {}
    for i, (c, l, u) in enumerate(zip(np.asarray(centre), np.asarray(lo), np.asarray(hi)), start=1):
        if not np.isfinite([c, l, u]).all():
            out[i] = None
            continue
        sl, sr = max((c - l) / z, 1e-6), max((u - c) / z, 1e-6)
        s = np.sqrt((sl ** 2 + sr ** 2) / 2.0)
        gamma = float(np.clip((sr ** 2 - sl ** 2) / (sr ** 2 + sl ** 2), -0.98, 0.98))
        out[i] = {"s": float(s), "gamma": gamma}
    return out


# --------------------------------------------------------------------------- #
# the simulation
# --------------------------------------------------------------------------- #
def simulate_var_fan(model_factory, info, base_paths: dict, calib: dict, *,
                     target: str, horizons: int = 8, n_scenarios: int = 150,
                     draws_per_scenario: int = 8, seed: int = 11):
    """Push conditioning-path uncertainty through the conditional VAR.

    ``model_factory(custom)`` must return a fitted-and-ready nowcaster taking a
    ``custom_conditions`` dict; ``base_paths`` holds each conditioned variable's
    central path (list per variable, one entry per horizon), and ``calib`` the
    (s, gamma) of that variable's own uncertainty per horizon. Variables absent
    from ``calib`` are treated as known with certainty - which is what a genuinely
    deterministic assumption means.

    Returns an (n_scenarios * draws_per_scenario, horizons) array of target paths.
    """

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_scenarios):
        custom = {}
        for var, path in base_paths.items():
            p = list(path)
            cal = calib.get(var)
            if cal:
                for i in range(min(len(p), horizons)):
                    c = cal.get(i + 1)
                    if c is None or not np.isfinite(p[i]):
                        continue
                    sl, sr = tpn_scales(c["s"], c["gamma"])
                    p[i] = float(tpn_ppf(rng.uniform(), p[i], sl, sr))
            custom[var] = p
        m = model_factory(custom)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                m.fit(info)
                m.nowcast(info)
                mod, conds = m._model, m._last_conditions
                sysdf = m._last_system
                off = 1 if pd.Period(sysdf.index[-1], freq="Q") < pd.Period(
                    info.y.dropna().index[-1], freq="Q") else 0
                fh = off + horizons
                dr, _ = mod.conditional_forecast(conditions=conds, fhor=fh,
                                                 plot_forecast=False, shock_uncertainty=True)
            arr = np.asarray(dr)[:, off:off + horizons, list(sysdf.columns).index(target)]
        except Exception:
            continue
        pick = rng.choice(arr.shape[0], size=min(draws_per_scenario, arr.shape[0]), replace=False)
        out.append(arr[pick])
    if not out:
        return None
    return np.vstack(out)
