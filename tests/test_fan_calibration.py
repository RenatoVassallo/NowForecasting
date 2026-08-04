"""Task 18: sequential calibration fits only on knowable-before errors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.lib.fan_calibration import (MIN_POOL, _fit_variant,
                                          knowable_before)


def _errors(n_per_h=12, horizons=(1, 2, 3), base_year=2015, err=1.0):
    rows = []
    for h in horizons:
        for i in range(n_per_h):
            base = pd.Period(f"{base_year + i // 4}Q{i % 4 + 1}", freq="Q")
            rows.append(dict(base=str(base), ref=str(base + h), h=h,
                             err=err * (1 if i % 2 else -1) * (1 + 0.1 * h)))
    return pd.DataFrame(rows)


def test_knowable_before_honours_the_release_rule():
    e = pd.DataFrame({"base": ["2026Q1"], "ref": ["2026Q2"], "h": [1], "err": [0.5]})
    # 2026Q2 publishes ~Aug 21 (52d): unknowable on Aug 3, knowable on Aug 22
    assert len(knowable_before(e, pd.Timestamp("2026-08-03"))) == 0
    assert len(knowable_before(e, pd.Timestamp("2026-08-22"))) == 1


def test_fit_is_symmetric_monotone_and_starves_honestly():
    cfg = dict(exclude="both_ends", skew=False, bias=False, covid_w=0.0)
    fit = _fit_variant(_errors(), cfg, horizons=[1, 2, 3, 4])
    assert fit is not None
    # symmetric: left equals right at every horizon
    for h in (1, 2, 3, 4):
        assert fit[h]["sigma_left"] == pytest.approx(fit[h]["sigma_right"])
        assert fit[h]["shift"] == 0.0
    # monotone in h, and h=4 (no data) extends flat from h=3
    ss = [fit[h]["sigma_left"] for h in (1, 2, 3, 4)]
    assert all(b >= a - 1e-12 for a, b in zip(ss, ss[1:]))
    assert fit[4]["sigma_left"] == pytest.approx(fit[3]["sigma_left"])

    tiny = _errors(n_per_h=3)               # below MIN_POOL
    assert 3 * 3 < MIN_POOL
    assert _fit_variant(tiny, cfg, horizons=[1, 2, 3]) is None


def test_covid_weights_shape_the_fit():
    covid = _errors(base_year=2020, err=8.0)          # huge COVID-era errors
    calm = _errors(base_year=2016, err=1.0)
    both = pd.concat([calm, covid], ignore_index=True)
    excl = _fit_variant(both, dict(exclude="both_ends", skew=False, bias=False,
                                   covid_w=0.0), horizons=[1, 2, 3])
    incl = _fit_variant(both, dict(exclude="none", skew=False, bias=False,
                                   covid_w=1.0), horizons=[1, 2, 3])
    down = _fit_variant(both, dict(exclude="none", skew=False, bias=False,
                                   covid_w=0.25), horizons=[1, 2, 3])
    assert excl[1]["sigma_left"] < down[1]["sigma_left"] < incl[1]["sigma_left"]


def test_production_fits_interface_and_no_lookahead():
    from pipeline.lib import fan_calibration as fc
    from pipeline.lib.calibration_assets import ASSETS, ROOT

    if not (ROOT / ASSETS["exact_chain"]).exists():
        pytest.skip("frozen exact-chain asset not present")
    day1, day30 = fc.production_fits("2026-08-03", H=7)
    for fits in (day1, day30):
        assert set(fits) == set(range(1, 8))
        for h in fits:
            assert fits[h]["sigma1"] == pytest.approx(fits[h]["sigma2"])
            assert fits[h]["sigma1"] > 0
    # an early as-of has strictly fewer knowable errors, never more width
    # information: the fit at 2020 must not silently equal today's
    early1, _ = fc.production_fits("2020-06-30", H=7)
    assert any(abs(early1[h]["sigma1"] - day1[h]["sigma1"]) > 1e-9
               for h in range(1, 8))
