"""Task 1: one canonical run context.

Two guarantees. First, RunContext is immutable, normalized, and recorded in the
run manifest. Second, no pipeline block or stage reads the wall clock on its
own: the only clock read in production code lives in pipeline.lib.context and
fires only when no context is supplied (ad-hoc interactive use).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]

WALL_CLOCK = re.compile(
    r"Timestamp\.now\(|Timestamp\.today\(|datetime\.now\(|date\.today\(")

SCANNED = sorted([
    *(REPO / "pipeline" / "blocks").glob("*.py"),
    *(REPO / "pipeline" / "stages").glob("*.py"),
    REPO / "pipeline" / "lib" / "nowcast_job.py",
])


def test_pipeline_has_single_wall_clock_source():
    offenders = {}
    for f in SCANNED:
        hits = [i + 1 for i, line in enumerate(f.read_text().splitlines())
                if WALL_CLOCK.search(line)]
        if hits:
            offenders[str(f.relative_to(REPO))] = hits
    assert not offenders, (
        "wall-clock reads outside pipeline.lib.context break as-of "
        f"reproducibility: {offenders}")


def test_run_context_is_immutable_and_normalized():
    from pipeline.lib.context import RunContext

    ctx = RunContext.create(as_of="2026-07-15 13:45")
    assert ctx.as_of == pd.Timestamp("2026-07-15")          # normalized to a date
    assert ctx.run_id.startswith("2026-07-15__")            # vintage visible in the id
    assert isinstance(ctx.code_version, str) and ctx.code_version
    with pytest.raises(Exception):
        ctx.as_of = pd.Timestamp("2030-01-01")              # frozen


def test_run_context_explicit_run_id_wins():
    from pipeline.lib.context import RunContext

    ctx = RunContext.create(as_of="2026-07-15", run_id="replay-001")
    assert ctx.run_id == "replay-001"


def test_resolve_as_of_prefers_context():
    from pipeline.lib.context import RunContext, resolve_as_of

    ctx = RunContext.create(as_of="2020-05-04")
    assert resolve_as_of(ctx) == pd.Timestamp("2020-05-04")
    assert resolve_as_of(None) == pd.Timestamp.now().normalize()


def test_information_stamp_uses_the_context_date():
    from pipeline.blocks._common import information_stamp

    class Spec:
        target_delay_days = 52

    q = pd.Period("2026Q2", freq="Q")
    stamp = information_stamp(Spec(), q, as_of=pd.Timestamp("2026-08-03"))
    assert stamp["as_of"] == "2026-08-03"
    # publication anchor is the period END TIMESTAMP (2026-06-30 23:59:59) plus
    # 52 days; the floor division therefore lands one below the naive count.
    # This is the project-wide dtp convention the band calibration is built on.
    assert stamp["days_to_publication"] == -19
    earlier = information_stamp(Spec(), q, as_of=pd.Timestamp("2026-07-04"))
    assert earlier["days_to_publication"] == -49
    # a 30-day shift in as_of shifts the stamp by exactly 30 days
    assert earlier["days_to_publication"] - stamp["days_to_publication"] == -30


def test_run_store_records_the_context(tmp_path):
    from pipeline.lib.context import RunContext
    from pipeline.lib.store import RunStore

    ctx = RunContext.create(as_of="2026-07-15", run_id="ctx-test")
    store = RunStore(tmp_path, ctx=ctx)
    assert store.run_id == "ctx-test"
    assert store.ctx is ctx
    store.write_manifest()
    manifest = json.loads((tmp_path / "ctx-test" / "manifest.json").read_text())
    assert manifest["as_of"] == "2026-07-15"
    assert manifest["run_id"] == "ctx-test"
    assert manifest["code_version"] == ctx.code_version
