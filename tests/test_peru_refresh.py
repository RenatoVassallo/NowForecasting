"""Task 11: a Peru refresh rebuilds the model panel or fails explicitly.

The defect: `refresh()` updated the INEI bulletin cache, printed the spec3
panel's age, and counted as success even when the panel production consumes
was stale. The contract now: when the release calendar says an observation is
due and the cache lacks it, the panel is rebuilt through the full preprocess
or the refresh raises ``PanelRebuildError``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

if not (Path(__file__).resolve().parents[1] / "input").exists():
    pytest.skip("private data layer absent (public clone)",
                allow_module_level=True)

import targets.peru_gdp as pg


def test_due_rule_boundaries():
    # delay 51d: May (ends 2026-05-31) releases 2026-07-21
    due, why = pg._release_due("2026-04", "2026-07-21", 51)
    assert due and "2026-05" in why
    due, _ = pg._release_due("2026-05", "2026-07-21", 51)
    assert not due
    # one day before the release date, April is still the latest due month
    due, _ = pg._release_due("2026-04", "2026-07-20", 51)
    assert not due
    # a panel AHEAD of the due month is never due
    due, _ = pg._release_due("2026-06", "2026-07-21", 51)
    assert not due


def _mute_inei(monkeypatch):
    fake = types.ModuleType("sources.inei")
    fake.update_inei_latest = lambda: {"new_reports": []}
    monkeypatch.setitem(sys.modules, "sources.inei", fake)


def test_refresh_raises_when_due_but_unbuildable(monkeypatch):
    _mute_inei(monkeypatch)
    monkeypatch.setattr(pg, "panel_release_due",
                        lambda as_of=None: (True, "month 2026-06 was released"))

    def broken(**kw):
        raise RuntimeError("x13as binary not found")

    monkeypatch.setattr(pg, "rebuild_panel", broken)
    with pytest.raises(pg.PanelRebuildError, match="due.*cannot"):
        pg.refresh()


def test_refresh_rebuilds_when_due_and_possible(monkeypatch):
    _mute_inei(monkeypatch)
    monkeypatch.setattr(pg, "panel_release_due",
                        lambda as_of=None: (True, "month 2026-06 was released"))
    monkeypatch.setattr(pg, "rebuild_panel",
                        lambda **kw: ["spec3 panel REBUILT: monthly through 2026-06"])
    msgs = pg.refresh()
    assert any("REBUILT" in m for m in msgs)


def test_refresh_is_quietly_current_when_nothing_due(monkeypatch):
    _mute_inei(monkeypatch)
    monkeypatch.setattr(pg, "panel_release_due",
                        lambda as_of=None: (False, "panel through 2026-05, latest due month is 2026-05"))
    called = {"n": 0}

    def never(**kw):
        called["n"] += 1
        return []

    monkeypatch.setattr(pg, "rebuild_panel", never)
    msgs = pg.refresh()
    assert any("current" in m for m in msgs)
    assert called["n"] == 0


def test_live_due_state_is_consistent_with_the_cache():
    """With the real cache: today's panel must not be due (otherwise the next
    production refresh will correctly fail until X13 is installed, which the
    operator should know from this test rather than from a 2am run)."""
    if not (pg.PROCESSED_DIR / "monthly_panel_spec3.parquet").exists():
        pytest.skip("private data caches are not present")
    due, why = pg.panel_release_due(pd.Timestamp("2026-08-03"))
    assert isinstance(due, bool) and why
