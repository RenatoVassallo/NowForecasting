"""F1 side-finding: the quarterly BVAR must tolerate the merged NBS release.

Under the corrected panel rule January is structurally blank for combined
January-February NBS series. `ConditionalBVARNowcaster` already tolerates this
by design (2 of 3 months per quarter). `BVARNowcaster` required complete
quarters, which under the corrected panel silently DROPPED every Q1 row and
then let the VAR treat Q4 -> Q2 as adjacent lags: a sample-selection and
lag-splicing defect, not a timing leak. The China production member now passes
``min_months=2``; the class default stays 3 (the generic complete-quarter
contract for targets without a merged release).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from MIDAS import MetadataPanel, VariableMeta
from MIDAS.realtime import RealtimeEngine


def _panel(n_years=14, jan_blank=True):
    months = pd.date_range("2010-01-01", periods=12 * n_years, freq="MS")
    rng = np.random.default_rng(3)
    m = pd.DataFrame({"a": rng.normal(5, 1, len(months)),
                      "b": rng.normal(8, 1, len(months))}, index=months)
    if jan_blank:
        m.loc[m.index.month == 1, "a"] = np.nan       # merged Jan-Feb release
    qs = pd.date_range("2010-03-01", periods=4 * n_years, freq="3MS")
    y = pd.Series(rng.normal(4, 1, len(qs)), index=qs, name="tgt")
    metas = [VariableMeta(column="a", frequency="M", group="g", publication_delay_days=15),
             VariableMeta(column="b", frequency="M", group="g", publication_delay_days=15),
             VariableMeta(column="tgt", frequency="Q", group="g", publication_delay_days=18)]
    return MetadataPanel.from_frames(m, y.to_frame(), metas)


def _system(min_months):
    from forecast.models import BVARNowcaster

    panel = _panel()
    info = RealtimeEngine(panel).information_set(
        pd.Timestamp("2024-06-30"), "tgt", target_period=pd.Timestamp("2024-06-01"))
    m = BVARNowcaster(variables=["a", "b"], min_train=8, min_months=min_months,
                      _name="t")
    return m._system(info)


def test_tolerant_rule_keeps_q1_and_never_splices():
    q = _system(min_months=2)
    quarters = pd.PeriodIndex(q.index, freq="Q")
    assert 1 in quarters.quarter, "Q1 rows must survive the merged Jan release"
    # contiguous quarters: the VAR's lag structure is real, never spliced
    assert (np.diff(quarters.asi8) == 1).all()


def test_complete_quarter_default_drops_q1_documenting_the_defect():
    q = _system(min_months=3)
    quarters = pd.PeriodIndex(q.index, freq="Q")
    assert 1 not in quarters.quarter          # the old behavior, kept as default
    assert (np.diff(quarters.asi8) != 1).any()   # spliced adjacency exists


def test_china_production_member_declares_the_tolerance():
    from pipeline.blocks import _china_model

    members = _china_model.build_members(nowcast_fn=None)
    assert getattr(members["BVAR(3v)"], "min_months", 3) == 2, (
        "the China BVAR(3v) production spec must tolerate the merged "
        "January-February release (min_months=2)")
