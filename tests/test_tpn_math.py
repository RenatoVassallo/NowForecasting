"""Task 17: the two-piece normal is mathematically pinned down.

Normalization, CDF/PPF inversion, moment formulas (verified by quadrature,
not by trusting the formula), continuity at the mode, shortest-interval
probability mass and the equal-density property, the equal-tailed versus
shortest semantic split, MLE recovery on simulated data, and the
table-equals-plot contract of the published fan frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from core.evaluation import tpn_pdf
from forecast.boe_fan import tpn_equal_tailed_bands, tpn_quantile
from forecast.fan_mc import (fit_tpn_mle, tpn_cdf, tpn_ppf, tpn_scales,
                             tpn_shortest_bands)

MODE, SL, SR = 1.3, 0.9, 1.7


def test_density_normalizes_and_is_continuous_at_the_mode():
    total, _ = quad(lambda x: tpn_pdf(x, MODE, SL, SR), -20, 25)
    assert total == pytest.approx(1.0, abs=1e-8)
    eps = 1e-9
    left = tpn_pdf(MODE - eps, MODE, SL, SR)
    right = tpn_pdf(MODE + eps, MODE, SL, SR)
    assert left == pytest.approx(right, rel=1e-6)


def test_cdf_ppf_round_trip():
    ps = np.linspace(0.01, 0.99, 25)
    xs = np.array([tpn_ppf(p, MODE, SL, SR) for p in ps])
    back = tpn_cdf(xs, MODE, SL, SR)
    np.testing.assert_allclose(back, ps, atol=1e-8)


def test_moments_match_quadrature():
    mean_q, _ = quad(lambda x: x * tpn_pdf(x, MODE, SL, SR), -20, 25)
    mean_formula = MODE + np.sqrt(2 / np.pi) * (SR - SL)
    assert mean_q == pytest.approx(mean_formula, abs=1e-6)
    var_q, _ = quad(lambda x: (x - mean_q) ** 2 * tpn_pdf(x, MODE, SL, SR), -20, 25)
    var_formula = (1 - 2 / np.pi) * (SR - SL) ** 2 + SL * SR
    assert var_q == pytest.approx(var_formula, abs=1e-6)


def test_scales_parameterisation_round_trip():
    s, gamma = 1.2, 0.35
    sl, sr = tpn_scales(s, gamma)
    assert sl == pytest.approx(s * np.sqrt(1 - gamma))
    assert sr == pytest.approx(s * np.sqrt(1 + gamma))


def test_shortest_bands_have_the_nominal_mass_and_equal_density():
    covs = (0.30, 0.60, 0.90)
    lo, hi = tpn_shortest_bands(np.array([MODE]), np.array([SL]),
                                np.array([SR]), covs)
    for j, cov in enumerate(covs):
        mass, _ = quad(lambda x: tpn_pdf(x, MODE, SL, SR), lo[0, j], hi[0, j])
        assert mass == pytest.approx(cov, abs=1e-6), f"cov {cov}"
        # shortest = equal density at both edges
        assert tpn_pdf(lo[0, j], MODE, SL, SR) == pytest.approx(
            tpn_pdf(hi[0, j], MODE, SL, SR), rel=1e-6)


def test_equal_tailed_and_shortest_disagree_for_skewed():
    params = [{"mode_shift": 0.0, "sigma1": SL, "sigma2": SR}]
    et = tpn_equal_tailed_bands(np.array([MODE]), params, levels=(0.90,))
    lo_et, hi_et = float(et[0.90][0][0]), float(et[0.90][1][0])
    mass, _ = quad(lambda x: tpn_pdf(x, MODE, SL, SR), lo_et, hi_et)
    assert mass == pytest.approx(0.90, abs=1e-6)      # same coverage...
    lo_s, hi_s = tpn_shortest_bands(np.array([MODE]), np.array([SL]),
                                    np.array([SR]), (0.90,))
    # ...different interval, and the shortest one is strictly shorter
    assert (hi_s[0, 0] - lo_s[0, 0]) < (hi_et - lo_et)
    # equal tails means equal tail mass, not equal edge density
    left_tail, _ = quad(lambda x: tpn_pdf(x, MODE, SL, SR), -20, lo_et)
    assert left_tail == pytest.approx(0.05, abs=1e-6)


def test_mle_recovers_simulated_parameters():
    rng = np.random.default_rng(11)
    draws = np.array([tpn_ppf(u, MODE, SL, SR) for u in rng.uniform(size=4000)])
    fit = fit_tpn_mle(draws)
    assert fit is not None
    sl_hat, sr_hat = fit["sigma_left"], fit["sigma_right"]
    assert sl_hat == pytest.approx(SL, rel=0.12)
    assert sr_hat == pytest.approx(SR, rel=0.12)
    assert fit["gamma"] == pytest.approx((SR**2 - SL**2) / (SL**2 + SR**2), abs=0.08)


def test_fan_frame_table_equals_the_band_formula():
    from pipeline.blocks._common import COV, fan_frame

    fits = [{"mode_shift": 0.0, "s": 1.2, "gamma": 0.3,
             "sigma_left": SL, "sigma_right": SR}] * 3
    periods = pd.period_range("2026Q2", periods=3, freq="Q")
    df = fan_frame(periods, [2.0, 2.5, 3.0], fits, "test")
    lo, hi = tpn_shortest_bands(df["mode"].to_numpy(),
                                df["sigma_left"].to_numpy(),
                                df["sigma_right"].to_numpy(), COV)
    for j, c in enumerate(COV):
        np.testing.assert_allclose(df[f"lo{int(c*100)}"], np.round(lo[:, j], 3))
        np.testing.assert_allclose(df[f"hi{int(c*100)}"], np.round(hi[:, j], 3))
