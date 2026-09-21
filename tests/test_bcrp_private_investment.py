"""The BCRP private-investment cache refresher (``g_invq``).

``pipeline/blocks/_peru_panel.py`` conditions the Peru block on this cache but
nothing used to WRITE it, so it silently went stale until the availability
preflight blocked a run on ``g_invq_m``. The refresher closes that hole, and
must be fail-closed: a truncated or garbled provider response has to raise
rather than destroy cached history.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sources import bcrp                      # noqa: E402
from sources.base import SourceError          # noqa: E402


def _levels(periods, values) -> pd.DataFrame:
    """A frame shaped like ``bcrp.fetch``: levels indexed at the quarter's LAST month."""
    idx = pd.to_datetime([f"{p.year}-{p.quarter * 3:02d}-01" for p in periods])
    return pd.DataFrame({"invq": list(values)}, index=idx)


def _compounding(start="2019Q1", end="2026Q2", rate=1.05):
    periods = pd.period_range(start, end, freq="Q")
    return periods, [100.0 * rate ** i for i in range(len(periods))]


def test_refresh_writes_yoy_at_the_quarter_end_month(tmp_path, monkeypatch):
    periods, values = _compounding()
    monkeypatch.setattr(bcrp, "fetch", lambda **kw: _levels(periods, values))

    target = tmp_path / "private_investment.parquet"
    out = bcrp.refresh_private_investment(target)

    frame = pd.read_parquet(target)
    assert list(frame.columns) == ["g_invq"]
    # index convention _peru_panel._q2m relies on: Q2 -> June 1st
    assert frame.index.max() == pd.Timestamp("2026-06-01")
    # g_invq is the YoY % of the level: four quarters of +5% => 1.05**4 - 1
    assert frame["g_invq"].iloc[-1] == pytest.approx((1.05 ** 4 - 1) * 100, abs=1e-9)
    assert out["last_period"] == "2026Q2"
    assert out["rows"] == len(frame)


def test_refresh_appends_only_the_new_quarter(tmp_path, monkeypatch):
    periods, values = _compounding()
    target = tmp_path / "private_investment.parquet"

    monkeypatch.setattr(bcrp, "fetch",
                        lambda **kw: _levels(periods[:-1], values[:-1]))
    first = bcrp.refresh_private_investment(target)
    assert first["last_period"] == "2026Q1"

    monkeypatch.setattr(bcrp, "fetch", lambda **kw: _levels(periods, values))
    second = bcrp.refresh_private_investment(target)
    assert second["added"] == ["2026-06-01"]          # exactly one new quarter
    assert second["last_period"] == "2026Q2"
    assert second["rows"] == first["rows"] + 1


def test_refresh_is_idempotent(tmp_path, monkeypatch):
    periods, values = _compounding()
    monkeypatch.setattr(bcrp, "fetch", lambda **kw: _levels(periods, values))
    target = tmp_path / "private_investment.parquet"
    bcrp.refresh_private_investment(target)
    again = bcrp.refresh_private_investment(target)
    assert again["added"] == []


def test_refresh_refuses_to_drop_cached_history(tmp_path, monkeypatch):
    periods, values = _compounding()
    target = tmp_path / "private_investment.parquet"
    monkeypatch.setattr(bcrp, "fetch", lambda **kw: _levels(periods, values))
    bcrp.refresh_private_investment(target)
    before = pd.read_parquet(target)

    # provider hiccup: the tail is missing
    monkeypatch.setattr(bcrp, "fetch",
                        lambda **kw: _levels(periods[:-3], values[:-3]))
    with pytest.raises(SourceError, match="would drop"):
        bcrp.refresh_private_investment(target)
    pd.testing.assert_frame_equal(pd.read_parquet(target), before)   # untouched


def test_refresh_refuses_an_implausible_revision(tmp_path, monkeypatch):
    periods, values = _compounding()
    target = tmp_path / "private_investment.parquet"
    monkeypatch.setattr(bcrp, "fetch", lambda **kw: _levels(periods, values))
    bcrp.refresh_private_investment(target)
    before = pd.read_parquet(target)

    garbled = [v * (10.0 if i > 4 else 1.0) for i, v in enumerate(values)]
    monkeypatch.setattr(bcrp, "fetch", lambda **kw: _levels(periods, garbled))
    with pytest.raises(SourceError, match="pp"):
        bcrp.refresh_private_investment(target)
    pd.testing.assert_frame_equal(pd.read_parquet(target), before)


def test_refresh_rejects_an_empty_provider_response(tmp_path, monkeypatch):
    monkeypatch.setattr(bcrp, "fetch",
                        lambda **kw: pd.DataFrame({"invq": []},
                                                  index=pd.to_datetime([])))
    with pytest.raises(SourceError):
        bcrp.refresh_private_investment(tmp_path / "private_investment.parquet")
