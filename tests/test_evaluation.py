"""Task 16: the evaluation metric set, against hand-computed examples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import evaluation as E


def test_point_metrics_hand_computed():
    e = [1.0, -1.0, 2.0, np.nan]
    assert E.rmse(e) == pytest.approx(np.sqrt(2.0))
    assert E.mae(e) == pytest.approx(4.0 / 3.0)
    assert E.bias(e) == pytest.approx(2.0 / 3.0)
    assert E.finite_share([1.0, np.nan, 2.0, np.nan]) == 0.5


def test_directional_accuracy_signs():
    # base 2.0: forecasts up/up/down, truth up/down/down -> 2 of 3
    acc = E.directional_accuracy([3.0, 3.0, 1.0], [4.0, 1.0, 1.5], [2.0, 2.0, 2.0])
    assert acc == pytest.approx(2.0 / 3.0)


def test_revision_size_mean_absolute_step():
    f = pd.DataFrame({
        "ref_quarter": ["2025Q1"] * 3 + ["2025Q2"] * 2,
        "origin_date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01",
                                       "2025-04-01", "2025-05-01"]),
        "y_hat": [2.0, 2.5, 2.3, 4.0, 3.0]})
    # |0.5| + |-0.2| for Q1, |-1.0| for Q2 -> mean 1.7/3
    assert E.revision_size(f) == pytest.approx(1.7 / 3.0)


def test_tpn_density_normalizes_and_scores():
    from scipy.integrate import quad

    mode, sl, sr = 1.0, 0.8, 1.4
    total, _ = quad(lambda x: E.tpn_pdf(x, mode, sl, sr), -15, 20)
    assert total == pytest.approx(1.0, abs=1e-6)
    # log score at the mode equals log of the shared density constant
    c = 2.0 / (np.sqrt(2 * np.pi) * (sl + sr))
    assert E.log_score(mode, mode, sl, sr) == pytest.approx(np.log(c))


def test_pit_is_uniform_under_the_true_model():
    from scipy import stats

    from forecast.fan_mc import tpn_ppf

    rng = np.random.default_rng(5)
    mode, sl, sr = 0.5, 1.0, 2.0
    draws = np.array([tpn_ppf(u, mode, sl, sr) for u in rng.uniform(size=400)])
    pits = np.array([E.pit(d, mode, sl, sr) for d in draws])
    assert stats.kstest(pits, "uniform").pvalue > 0.01


def test_interval_and_weighted_scores_hand_computed():
    # inside the interval: score is the width
    assert E.interval_score(0.0, -1.0, 1.0, 0.90) == pytest.approx(2.0)
    # outside by 0.5 below with alpha 0.1: width + 2/0.1 * 0.5 = 12
    assert E.interval_score(-1.5, -1.0, 1.0, 0.90) == pytest.approx(12.0)
    # WIS with one interval, y at the mode inside: (0.5*0 + 0.05*2)/(1.5)
    wis = E.weighted_interval_score(0.0, 0.0, {0.90: (-1.0, 1.0)})
    assert wis == pytest.approx(0.1 / 1.5)


def test_coverage_and_wilson_interval():
    assert E.coverage([0, 5, 10], [-1, 4, 11], [1, 6, 12]) == pytest.approx(2 / 3)
    lo, hi = E.coverage_ci(27, 30, 0.90)
    assert lo < 0.9 < hi and 0 < lo < hi < 1


def test_scoreboard_carries_regime_and_samples():
    df = pd.DataFrame({
        "model": ["A"] * 3 + ["B"] * 3,
        "y_true": [1.0, 2.0, 3.0] * 2,
        "y_hat": [1.1, 2.1, np.nan, 0.9, 1.9, 2.9]})
    sb = E.scoreboard(df, ["model"])
    assert set(sb.columns) >= {"model", "n", "finite_share", "rmse", "regime"}
    assert (sb.regime == E.EVALUATION_REGIME).all()
    assert sb.set_index("model").loc["A", "finite_share"] == pytest.approx(2 / 3, abs=1e-3)


def test_sample_labels_at_every_boundary():
    assert E.sample_label("2022-12-01") == "selection"                 # 2022Q4
    assert E.sample_label("2023-03-01") == "inspected_post_selection"  # 2023Q1
    assert E.sample_label("2026-03-01") == "inspected_post_selection"  # 2026Q1
    assert E.sample_label("2026-06-01") == "prospective"               # 2026Q2
    assert E.sample_label("2030-03-01") == "prospective"   # future stays out
    # the historical boundary constant survives, as a boundary only
    assert E.HOLDOUT_START == pd.Period("2023Q1", freq="Q")
    assert E.HOLDOUT_START == pd.Period("2023Q1", freq="Q")
