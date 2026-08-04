"""Task 7: one official Peru nowcast artifact.

The headline Adaptive-IC and the fan's first node published different
ensembles under one label. The artifact is now generated once by the nowcast
stage and consumed by the fan, figures and report; these tests pin its
contract: value identity with the newest live Adaptive-IC origin, recorded
members and weights, the documented missing-member renormalization, and the
as-of coherence check on the consumer side. Under the exact-origin
contract (G1) the artifact's origin equals the as-of, so fixtures pass the
newest live origin as as_of.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.lib.nowcast_artifact import load_official, official_from

MEMBERS = ["RW", "Bridge(leaders)", "P-MIDAS(leaders)"]
ANAME = "Adaptive-IC"


def _inputs(rw_live=2.2):
    origins = pd.to_datetime(["2026-07-16", "2026-07-23", "2026-07-30"])
    ref = pd.Timestamp("2026-06-01")                      # 2026Q2 end-month stamp
    rows = []
    for o, dtp in zip(origins, [-36, -29, -22]):
        for m, v in [("RW", rw_live), ("Bridge(leaders)", 2.6),
                     ("P-MIDAS(leaders)", 2.8), (ANAME, 2.64)]:
            rows.append(dict(ref_quarter=ref, origin_date=o, days_to_publication=dtp,
                             model=m, y_true=np.nan, y_hat=v, info_index=0.9))
    hist = [dict(ref_quarter=pd.Timestamp("2026-03-01"), origin_date=pd.Timestamp("2026-05-01"),
                 days_to_publication=-21, model=m, y_true=3.5, y_hat=3.4, info_index=0.9)
            for m in MEMBERS + [ANAME]]
    nowcasts = pd.DataFrame(rows + hist)
    weights = pd.DataFrame([{"ref_quarter": ref, "bin": 3, "info_lo": 0.75, "info_hi": 1.0,
                             "RW": 0.2, "Bridge(leaders)": 0.5, "P-MIDAS(leaders)": 0.3}])
    pools = {(pd.Timestamp("2026-03-01"), 3): np.random.default_rng(3).normal(0, 0.5, 40)}
    return nowcasts, weights, pools


def test_value_is_the_newest_live_adaptive_origin():
    nowcasts, weights, pools = _inputs()
    art = official_from(nowcasts, weights, pools, members=MEMBERS,
                        adaptive_name=ANAME, as_of="2026-07-30").iloc[0]
    assert art["value"] == pytest.approx(2.64)
    assert art["quarter"] == "2026Q2"
    assert art["origin_date"] == "2026-07-30"
    assert art["members"] == "+".join(MEMBERS)
    assert art["information_bin"] == 3
    assert art["sigma_left"] > 0 and art["sigma_right"] > 0
    realized = dict(kv.split(":") for kv in art["weights_realized"].split("|"))
    assert sum(float(v) for v in realized.values()) == pytest.approx(1.0, abs=2e-3)


def test_missing_member_weight_renormalizes_proportionally():
    nowcasts, weights, pools = _inputs(rw_live=np.nan)
    art = official_from(nowcasts, weights, pools, members=MEMBERS,
                        adaptive_name=ANAME, as_of="2026-07-30").iloc[0]
    assert art["missing_members"] == "RW"
    realized = {k: float(v) for k, v in
                (kv.split(":") for kv in art["weights_realized"].split("|"))}
    assert realized["RW"] == 0.0
    # learned 0.5 / 0.3 renormalized over the surviving 0.8
    assert realized["Bridge(leaders)"] == pytest.approx(0.5 / 0.8, abs=1e-3)
    assert realized["P-MIDAS(leaders)"] == pytest.approx(0.3 / 0.8, abs=1e-3)


def test_consumer_rejects_a_stale_as_of(tmp_path):
    nowcasts, weights, pools = _inputs()
    frame = official_from(nowcasts, weights, pools, members=MEMBERS,
                          adaptive_name=ANAME, as_of="2026-07-30")
    p = tmp_path / "official.csv"
    frame.to_csv(p, index=False)
    row = load_official(expected_as_of="2026-07-30", path=p)
    assert float(row["value"]) == pytest.approx(2.64)
    with pytest.raises(ValueError, match="information set"):
        load_official(expected_as_of="2026-08-03", path=p)
    with pytest.raises(FileNotFoundError):
        load_official(expected_as_of="2026-07-30", path=tmp_path / "absent.csv")
