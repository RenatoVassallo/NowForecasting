"""Prospective S1/S2 forecasts are mandatory, immutable run artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def test_recursive_expectations_fit_is_origin_vintaged_and_finite():
    from pipeline.blocks._peru_panel import recursive_expectations_fit

    idx = pd.date_range("2015-01-01", periods=48, freq="MS")
    history = pd.Series(np.linspace(48.0, 58.0, len(idx)), index=idx)
    fit = recursive_expectations_fit(history, last=58.0, H=8)

    assert len(fit["path"]) == 8
    assert fit["path"][0] == 58.0
    assert np.isfinite(fit["path"]).all()
    assert 0.0 <= fit["phi"] <= 0.98
    assert 45.0 <= fit["long_run_mean"] <= 65.0
    assert fit["n_quarters"] == 16


def test_prospective_frame_records_both_models_and_common_nowcast():
    from pipeline.blocks.peru import _prospective_frame

    periods = list(pd.period_range("2026Q2", periods=8, freq="Q"))
    frame = _prospective_frame(
        periods=periods,
        s1=[2.7, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6],
        s2=[2.7, 2.9, 3.0, 3.15, 3.2, 3.25, 3.3, 3.35],
        fits=[{"mode_shift": 0.0, "sigma_left": 0.5 + 0.1 * h,
               "sigma_right": 0.5 + 0.1 * h,
               "s": 0.5 + 0.1 * h, "gamma": 0.0} for h in range(8)],
        base=pd.Period("2026Q1", freq="Q"),
        as_of=pd.Timestamp("2026-08-06"),
        run_id="2026-08-06__test",
        target_delay_days=52,
        exp_fit={"phi": 0.8, "long_run_mean": 54.0,
                 "n_quarters": 40, "fallback_flat": False},
    )

    assert len(frame) == 16
    assert set(frame.model) == {"S1-chain", "S2 exp-AR1"}
    assert frame.groupby("model").fan_node.apply(
        lambda values: list(values) == list(range(1, 9))).all()
    assert frame[frame.fan_node.eq(1)].forecast.eq(2.7).all()
    assert frame[frame.model.eq("S1-chain")].is_published_center.all()
    assert not frame[frame.model.eq("S2 exp-AR1")].is_published_center.any()
    assert frame.as_of.eq("2026-08-06").all()
    assert frame.run_id.eq("2026-08-06__test").all()
    assert frame.outcome_release_date.notna().all()
    assert frame.y_true.isna().all(), "outcomes must not be read when forecasts are frozen"
    widths = frame.pivot(index="fan_node", columns="model", values="width90")
    assert np.allclose(widths["S1-chain"], widths["S2 exp-AR1"])
    assert frame[frame.model.eq("S2 exp-AR1")].band_rule.str.contains("shared").all()


def test_peru_fan_stage_tracks_and_requires_candidate_archive(tmp_path, monkeypatch):
    import pipeline.blocks as blocks
    from pipeline.stages.domestic import run_peru_fan

    fan = tmp_path / "peru_gdp_fan.csv"
    candidates = tmp_path / "peru_gdp_model_paths.csv"

    def fake_build(**_):
        pd.DataFrame({"quarter": ["2026Q2"], "mode": [2.7]}).to_csv(fan, index=False)
        pd.DataFrame({"model": ["S1-chain", "S2 exp-AR1"]}).to_csv(candidates, index=False)
        return pd.DataFrame(), [], fan

    monkeypatch.setattr(blocks, "build_peru", fake_build)

    class Store:
        root = tmp_path
        ctx = None
        blocks = {}

        def __init__(self):
            self.tracked, self.required = [], []

        def _track(self, path, kind):
            self.tracked.append((Path(path).name, kind))

        def require(self, *paths):
            self.required.extend(paths)

    store = Store()
    run_peru_fan(store, object())
    assert ("peru_gdp_model_paths.csv", "prospective-forecast") in store.tracked
    assert "peru_gdp_model_paths.csv" in store.required


def test_candidate_archive_is_part_of_the_versioned_publication_surface():
    from pipeline.lib.publish import SURFACE

    assert "peru_gdp_model_paths.csv" in SURFACE
