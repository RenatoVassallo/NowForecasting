"""Automatic ingestion of early releases into the spec3 panel.

The panel rebuild used to trigger ONLY on the slow monthly-GDP schedule
(g_pbim, 51 days), so a BCRP series released early (the July expectations
block, CPI on the 1st) sat on the provider while the panel stayed stale
until the next g_pbim wave: the frontier showed "published, pending
ingest" and the user reasonably asked why the run did not just ingest it.

Now the data stage probes the provider for panel series whose check
window is open, and any store sighting NEWER than the panel column forces
the rebuild in the SAME run: probe -> detected-pending -> rebuild.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.lib import release_calendar as rc

REG = {"series": [
    {"internal_code": "exp_eco3m", "frequency": "M", "publication_lag_days": 7,
     "provider_code": "PD38045AM",
     "source_url": "https://estadisticas.bcrp.gob.pe/estadisticas/series/api",
     "release_calendar": {"rule": "period_end_plus_lag", "source": "t"},
     "monitor": {"type": "target_panel", "target": "peru_gdp",
                 "frame": "monthly", "column": "exp_eco3m"}},
    {"internal_code": "ip_cum_yoy", "frequency": "M", "publication_lag_days": 15,
     "provider_code": "CN_IP", "source_url": "https://www.stats.gov.cn",
     "release_calendar": {"rule": "period_end_plus_lag", "source": "t"},
     "monitor": {"type": "tabular_file", "target": "china",
                 "column": "ip_cum_yoy"}},
]}


def _panel(exp_through="2026-06"):
    idx = pd.period_range("2025-01", exp_through, freq="M").to_timestamp()
    return pd.DataFrame({"exp_eco3m": range(len(idx))}, index=idx)


def test_detected_pending_lists_store_sightings_newer_than_panel(tmp_path):
    p = tmp_path / "obs.parquet"
    rc.record_observed(p, [{"internal_code": "exp_eco3m", "period": "2026-07",
                            "first_seen": "2026-08-05", "source": "api_probe"}])
    pend = rc.detected_pending_for_panel(REG, _panel(), as_of="2026-08-06",
                                         store_path=p)
    assert len(pend) == 1 and "exp_eco3m" in pend[0] and "2026-07" in pend[0]
    # already ingested -> nothing pending
    assert rc.detected_pending_for_panel(REG, _panel("2026-07"),
                                         as_of="2026-08-06", store_path=p) == []
    # as-of safety: a sighting from the future never counts
    assert rc.detected_pending_for_panel(REG, _panel(),
                                         as_of="2026-08-04", store_path=p) == []


def test_probe_panel_releases_probes_only_due_bcrp_panel_series(tmp_path):
    p = tmp_path / "obs.parquet"
    calls = []

    def prober(code, period):
        calls.append((code, str(period)))
        return True

    hits, errors = rc.probe_panel_releases(REG, _panel(), as_of="2026-08-06",
                                           store_path=p, prober=prober)
    # exp window opened Aug 5 (Jul 31 + 7 - 2d margin); china is not BCRP
    assert calls == [("PD38045AM", "2026-07")]
    assert hits == ["exp_eco3m"] and errors == []
    obs = rc.load_observed(p)
    assert list(obs.period) == ["2026-07"]


def test_refresh_rebuilds_on_pending_upstream(monkeypatch):
    from targets import peru_gdp

    calls = []
    monkeypatch.setattr(peru_gdp, "rebuild_panel",
                        lambda **kw: calls.append(1) or ["REBUILT"])
    monkeypatch.setattr(peru_gdp, "update_inei_stub", None, raising=False)
    monkeypatch.setattr(peru_gdp, "panel_release_due",
                        lambda as_of=None: (False, "panel through 2026-05"))
    import sources.inei as inei
    monkeypatch.setattr(inei, "update_inei_latest",
                        lambda **kw: {"n_new_reports": 0, "added": [], "failures": []})

    msgs = peru_gdp.refresh("2026-08-06",
                            pending_upstream=["exp_eco3m 2026-07 (seen 2026-08-05)"])
    assert calls == [1]
    assert any("exp_eco3m" in m for m in msgs)

    calls.clear()
    msgs = peru_gdp.refresh("2026-08-06", pending_upstream=[])
    assert calls == []
    assert any("current" in m for m in msgs)
