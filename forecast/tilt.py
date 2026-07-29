"""Entropic tilting: minimum-relative-entropy incorporation of external judgment.

Robertson, Tallman & Whiteman (2005); Krueger, Clark & Ravazzolo (2017);
Altavilla, Giacomini & Ragusa (2017). Given predictive draws and moment
constraints (here: IMF WEO annual growth = the mean of the quarterly YoY draws
over a calendar year, the same averaging identity used for the SPF SAAR
conversion), reweight the draws as little as possible (in KL divergence) so
the constraints hold exactly. The tilted density keeps the model's dynamics,
covariances and asymmetries; judgment enters only where it speaks.

The effective sample size (ESS) is the honesty meter: low ESS means the
judgment sits deep in the model's tail. Callers should fall back to the
untilted forecast (and say so) rather than accept a handful of draws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["entropic_weights", "annual_constraints", "tilt_path"]


def entropic_weights(G: np.ndarray, targets: np.ndarray,
                     min_ess_frac: float = 0.01):
    """Exponential-tilting weights so that ``E_w[G] = targets``.

    Solves the convex dual ``min_l log mean(exp((G - t) @ l))`` (BFGS on a
    logsumexp, translation-safe). Returns ``(w, lam, ess)`` or ``None`` when
    the solver fails, the constraint lies outside the draws' support, or the
    ESS lands below ``min_ess_frac`` of the draws.
    """

    from scipy import optimize

    G = np.atleast_2d(np.asarray(G, dtype=float))
    t = np.asarray(targets, dtype=float)
    if G.shape[0] < 10 or not np.isfinite(G).all() or not np.isfinite(t).all():
        return None
    # a target outside the draws' hull is unattainable by reweighting
    if (t <= G.min(axis=0)).any() or (t >= G.max(axis=0)).any():
        return None

    def dual(l):
        z = (G - t) @ l
        m = z.max()
        return m + np.log(np.mean(np.exp(z - m)))

    res = optimize.minimize(dual, np.zeros(G.shape[1]), method="BFGS")
    if not np.isfinite(res.fun):
        return None
    z = (G - t) @ res.x
    w = np.exp(z - z.max())
    w /= w.sum()
    if not np.allclose(w @ G, t, atol=0.05):
        return None
    ess = 1.0 / float((w ** 2).sum())
    if ess < min_ess_frac * len(w):
        return None
    return w, res.x, ess


def annual_constraints(ref_quarters, weo: pd.Series, round_label: str,
                       realized: pd.Series, max_years: int = 2):
    """WEO annual constraints expressed on a quarterly draw grid.

    For each calendar year with a WEO projection (the round's own year and the
    next, ``max_years`` total), whose four quarters are all either realized at
    the origin or on the ``ref_quarters`` grid, the constraint is

        mean(draws over the grid quarters) = (4 * weo_year - sum(realized)) / n_grid

    i.e. annual growth ~ the average of the four quarterly YoY prints (exact up
    to intra-year weights). Returns a list of ``(mask, target)`` with ``mask``
    a boolean selector over ``ref_quarters``.
    """

    ref = pd.PeriodIndex(ref_quarters, freq="Q")
    realized_q = {pd.Period(ix, freq="Q"): float(v)
                  for ix, v in realized.dropna().items()}
    y0 = int(round_label.split("-")[0])

    out = []
    for year in range(y0, y0 + max_years):
        if year not in weo.index:
            continue
        quarters = [pd.Period(f"{year}Q{k}") for k in (1, 2, 3, 4)]
        # a grid quarter that is ALREADY released (the live origin's base) is
        # data, not a free draw: it belongs on the realized side of the identity
        mask = np.array([p.year == year and p not in realized_q for p in ref])
        n_grid = int(mask.sum())
        done = [q for q in quarters if q in realized_q]
        if n_grid == 0 or n_grid + len(done) < 4:
            continue                      # year not fully covered: no constraint
        target = (4.0 * float(weo.loc[year])
                  - sum(realized_q[q] for q in done)) / n_grid
        out.append((mask, target))
    return out


def tilt_path(draws: np.ndarray, constraints, quantiles=(0.05, 0.16, 0.84, 0.95)):
    """Tilt ``draws`` (n x fhor) to the constraints; None if tilting declined.

    Returns ``(path, bands, ess)`` where ``path`` is the weighted mean per
    horizon and ``bands`` the weighted quantiles (dict level -> array).
    """

    if not constraints:
        return None
    G = np.column_stack([draws[:, m].mean(axis=1) for m, _ in constraints])
    t = np.array([tv for _, tv in constraints])
    res = entropic_weights(G, t)
    if res is None:
        return None
    w, _, ess = res
    order = np.argsort(draws, axis=0)
    path = w @ draws
    bands = {}
    for q in quantiles:
        col = np.empty(draws.shape[1])
        for h in range(draws.shape[1]):
            x = draws[order[:, h], h]
            cw = np.cumsum(w[order[:, h]])
            col[h] = float(np.interp(q, cw, x))
        bands[q] = col
    return path, bands, ess
