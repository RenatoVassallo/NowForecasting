"""DirectARXNowcaster: recovers a known h-step relation and forecasts beyond sample."""

import numpy as np
import pandas as pd

from forecast.models import DirectARXNowcaster
from MIDAS.base import InformationSet


def _panel(T=120, seed=0):
    """Monthly x leads the quarterly target by 2 quarters: y_{t+2} = 0.8 x_t + e."""

    rng = np.random.default_rng(seed)
    q_periods = pd.period_range("1995Q1", periods=T, freq="Q")
    q_idx = pd.DatetimeIndex([p.to_timestamp(how="end").to_period("M").to_timestamp()
                              for p in q_periods])
    x_q = rng.normal(0, 1, T)
    y = np.full(T, np.nan)
    y[2:] = 0.8 * x_q[:-2] + 0.1 * rng.normal(0, 1, T - 2)

    m_idx = pd.date_range("1995-01-01", periods=T * 3, freq="MS")
    x_m = np.repeat(x_q, 3) + 0.01 * rng.normal(0, 1, T * 3)
    monthly = pd.DataFrame({"x": x_m}, index=m_idx)
    quarterly = pd.DataFrame({"tgt": y}, index=q_idx)
    return monthly, quarterly, q_periods


def test_recovers_two_step_relation_and_beats_noise():
    monthly, quarterly, qp = _panel()
    info = InformationSet(monthly=monthly, quarterly=quarterly, target="tgt",
                          origin=pd.Timestamp("2024-12-31"),
                          target_period=quarterly.index[-1])
    # pretend the last 2 quarters of the target are unreleased -> h=2 forecast
    q_masked = quarterly.copy()
    q_masked.iloc[-2:, 0] = np.nan
    info = InformationSet(monthly=monthly, quarterly=q_masked, target="tgt",
                          origin=pd.Timestamp("2024-12-31"),
                          target_period=quarterly.index[-1])

    model = DirectARXNowcaster(indicators=["x"], ar_lags=1, min_train=20)
    res = model.fit(info).nowcast(info)
    truth = quarterly["tgt"].iloc[-1]
    assert np.isfinite(res.mean)
    assert abs(res.mean - truth) < 0.5          # x explains y two quarters ahead
    assert res.std is not None and res.std < 0.5


def test_beyond_sample_target_period():
    monthly, quarterly, qp = _panel()
    beyond = (qp[-1] + 3).to_timestamp(how="end").to_period("M").to_timestamp()
    info = InformationSet(monthly=monthly, quarterly=quarterly, target="tgt",
                          origin=pd.Timestamp("2025-06-30"), target_period=beyond)
    res = DirectARXNowcaster(indicators=["x"], min_train=20).fit(info).nowcast(info)
    assert np.isfinite(res.mean)                # h=3 direct projection exists


def test_nan_when_too_little_history():
    monthly, quarterly, qp = _panel(T=20)
    beyond = (qp[-1] + 1).to_timestamp(how="end").to_period("M").to_timestamp()
    info = InformationSet(monthly=monthly, quarterly=quarterly, target="tgt",
                          origin=pd.Timestamp("2000-06-30"), target_period=beyond)
    res = DirectARXNowcaster(indicators=["x"], min_train=30).fit(info).nowcast(info)
    assert np.isnan(res.mean)                   # honest refusal, no silent fallback
