"""Task 12: the availability preflight blocks unfit required inputs."""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.lib.preflight import BLOCKING, evaluate


def _registry(required, optional=()):
    mk = lambda c, req: {"internal_code": c, "required_for_publication": req}
    return {"series": [mk(c, True) for c in required] + [mk(c, False) for c in optional]}


def _table(rows):
    return pd.DataFrame([{"internal_code": c, "status": s, "detail": d}
                         for c, s, d in rows])


def test_required_bad_states_block_and_optional_do_not():
    reg = _registry(required=["g_pbim_yoy"], optional=["credme"])
    for status in BLOCKING:
        table = _table([("g_pbim_yoy", status, "boom"),
                        ("credme", "stale_observation", "old")])
        offenders, waived, unknown = evaluate(table, reg)
        assert len(offenders) == 1 and "g_pbim_yoy" in offenders[0]
        assert not waived and not unknown


def test_healthy_states_pass():
    reg = _registry(required=["g_pbim_yoy", "gdp_yoy"])
    table = _table([("g_pbim_yoy", "successfully_updated", ""),
                    ("gdp_yoy", "not_yet_released", "next in Oct")])
    offenders, waived, unknown = evaluate(table, reg)
    assert offenders == [] and waived == [] and unknown == []


def test_documented_override_waives_and_undocumented_does_not():
    reg = _registry(required=["m2_yoy"])
    table = _table([("m2_yoy", "stale_observation", "PBoC lag")])
    ok = {"m2_yoy": {"author": "rvs", "reason": "portal outage, judged unchanged"}}
    offenders, waived, _ = evaluate(table, reg, ok)
    assert offenders == [] and waived == ["m2_yoy"]

    bad = {"m2_yoy": "just skip it"}                   # no author, no reason
    offenders, waived, _ = evaluate(table, reg, bad)
    assert len(offenders) == 1 and waived == []


def test_unknown_override_codes_are_reported():
    reg = _registry(required=["g_pbim_yoy"])
    table = _table([("g_pbim_yoy", "successfully_updated", "")])
    _, _, unknown = evaluate(table, reg, {"tyop_code": {"author": "a", "reason": "r"}})
    assert unknown == ["tyop_code"]
