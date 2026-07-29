"""Bank of England fan chart: the two-piece normal density (Britton, Fisher &
Whitley 1998, "The Inflation Report projections: understanding the fan chart").

The BoE summarises a forecast by three numbers per horizon - the **mode** (the
single most likely outcome, which is what the central line of a fan chart
actually is), the **uncertainty**, and the **skew** - and represents them with a
two-piece normal: two half-normals of different standard deviations joined at
the mode,

    f(x) = A exp(-(x-mu)^2 / 2 sigma1^2)   x <= mu
    f(x) = A exp(-(x-mu)^2 / 2 sigma2^2)   x >  mu ,    A = sqrt(2/pi)/(sigma1+sigma2)

so the density is continuous, unimodal, and asymmetric whenever sigma1 != sigma2.
The mean sits at mu + sqrt(2/pi)(sigma2 - sigma1), i.e. skew moves the mean away
from the mode without moving the mode - which is exactly the property the BoE
wants when it says "risks are to the downside" while keeping its central view.

Here the two sigmas are **estimated from the model's own real-time forecast
errors**, one pair per horizon, by matching the empirical error quantiles. So the
skew is measured, not asserted.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

__all__ = ["tpn_quantile", "fit_tpn", "fit_tpn_smooth", "tpn_bands"]


def tpn_quantile(p, mode: float, sigma1: float, sigma2: float):
    """Quantile function of the two-piece normal (sigma1 = left, sigma2 = right)."""

    p = np.atleast_1d(np.asarray(p, dtype=float))
    s = sigma1 + sigma2
    split = sigma1 / s
    out = np.empty_like(p)
    lo = p <= split
    out[lo] = mode + sigma1 * norm.ppf(np.clip(p[lo] * s / (2 * sigma1), 1e-12, 1 - 1e-12))
    hi = ~lo
    out[hi] = mode + sigma2 * norm.ppf(
        np.clip(0.5 + (p[hi] * s - sigma1) / (2 * sigma2), 1e-12, 1 - 1e-12))
    return out if out.size > 1 else float(out[0])


def fit_tpn(errors, probs=(0.10, 0.25, 0.75, 0.90)):
    """Fit (mode_shift, sigma1, sigma2) to a sample of forecast errors.

    Errors are (realised - forecast), so the fitted density describes where the
    outturn falls relative to the central projection. ``mode_shift`` is added to
    the point forecast to get the mode of the fan: it is normally small, and is
    what stops a skewed error distribution from silently re-centring the chart.
    """

    e = np.asarray(errors, dtype=float)
    e = e[np.isfinite(e)]
    if e.size < 12:
        return None
    target = np.quantile(e, probs)
    sd = float(np.std(e)) or 1.0

    def resid(theta):
        shift, s1, s2 = theta
        if s1 <= 1e-6 or s2 <= 1e-6:
            return np.full(len(probs), 1e3)
        return tpn_quantile(np.asarray(probs), shift, s1, s2) - target

    sol = least_squares(resid, x0=[float(np.median(e)), sd, sd],
                        bounds=([-5 * sd, 1e-3, 1e-3], [5 * sd, 10 * sd, 10 * sd]))
    shift, s1, s2 = sol.x
    return {"mode_shift": float(shift), "sigma1": float(s1), "sigma2": float(s2),
            "skew": float(s2 - s1), "n": int(e.size)}


def tpn_bands(centre, params, levels=(0.30, 0.60, 0.90)):
    """Central-probability bands from fitted two-piece normal parameters.

    ``centre`` is the point forecast per horizon and ``params`` the per-horizon
    fit. Returns {level: (lo, hi)} arrays. Levels follow the BoE convention of
    equal-probability bands around the mode rather than a 50/70/90 split.
    """

    centre = np.asarray(centre, dtype=float)
    out = {}
    for lv in levels:
        lo, hi = np.empty(len(centre)), np.empty(len(centre))
        a = (1 - lv) / 2
        for i, c in enumerate(centre):
            pr = params[i]
            if pr is None:
                lo[i] = hi[i] = np.nan
                continue
            m = c + pr["mode_shift"]
            lo[i] = tpn_quantile(a, m, pr["sigma1"], pr["sigma2"])
            hi[i] = tpn_quantile(1 - a, m, pr["sigma1"], pr["sigma2"])
        out[lv] = (lo, hi)
    return out


def fit_tpn_smooth(errors_by_h: dict, probs=(0.10, 0.25, 0.75, 0.90)):
    """Stable per-horizon parameters, the way the BoE actually builds a fan.

    Fitting (mode, sigma1, sigma2) independently at each horizon on ~35 errors
    overfits: the downside sigma collapses to zero at one horizon and the skew
    flips sign at the next, which is sampling noise dressed as judgment. The BoE
    instead treats **uncertainty** as a smooth function of horizon (from
    historical forecast-error variances) and **skew** as a single stance carried
    across the profile. This does the same, both measured:

    1. standardise each horizon's errors by that horizon's standard deviation and
       pool them, then fit one two-piece normal with the mode fixed at zero. That
       gives the skew as a ratio, estimated on the whole sample rather than a
       thirtieth of it;
    2. estimate the scale per horizon and smooth it with a power law in h, so
       uncertainty grows monotonically as a fan should.

    The mode stays at the model's central projection: no shift is applied, so the
    published number is the mode and the skew moves only the *mean*, which is the
    BoE's convention.
    """

    hs = sorted(errors_by_h)
    sd = {}
    pooled = []
    for h in hs:
        e = np.asarray(errors_by_h[h], dtype=float)
        e = e[np.isfinite(e)]
        if e.size < 12:
            continue
        s_ = float(np.std(e))
        if s_ <= 0:
            continue
        sd[h] = s_
        pooled.append(e / s_)
    if not pooled:
        return {h: None for h in hs}
    pooled = np.concatenate(pooled)

    target = np.quantile(pooled, probs)

    def resid(theta):
        s1, s2 = theta
        if min(s1, s2) <= 1e-6:
            return np.full(len(probs), 1e3)
        return tpn_quantile(np.asarray(probs), 0.0, s1, s2) - target

    sol = least_squares(resid, x0=[1.0, 1.0], bounds=([1e-3, 1e-3], [10, 10]))
    s1u, s2u = sol.x

    # smooth the scale across horizons: c * h^gamma, gamma clipped so the fan
    # always opens but never explodes
    hh = np.array([h for h in hs if h in sd], dtype=float)
    ss = np.array([sd[h] for h in hs if h in sd], dtype=float)
    if hh.size < 2:
        # a single horizon (e.g. the nowcast node) has nothing to smooth over
        g, c = 0.0, float(ss[0]) / float(hh[0]) ** 0.0
    else:
        g = float(np.polyfit(np.log(hh), np.log(ss), 1)[0])
        g = min(max(g, 0.0), 1.0)
        c = float(np.exp(np.mean(np.log(ss) - g * np.log(hh))))

    out = {}
    for h in hs:
        k = c * h ** g
        out[h] = {"mode_shift": 0.0, "sigma1": s1u * k, "sigma2": s2u * k,
                  "skew": (s2u - s1u) * k, "n": int(pooled.size),
                  "scale": k, "gamma": g}
    return out
