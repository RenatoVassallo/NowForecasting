"""Task 2: one canonical terms-of-trade transformation at the Peru interface.

The Peru BVAR regressor ``g_tdi`` is 100 times the twelve-month LOG difference
of the official BCRP terms-of-trade index. The commodity block publishes the
ARITHMETIC year-over-year growth of the same index (the BCRP-comparable
headline unit). Passing one into a model trained on the other is a scale error
of up to six percentage points. The contract: the commodity block keeps its
arithmetic headline units and declares them; the Peru interface converts the
centre exactly and the scale by the delta method before conditioning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def test_log_and_arithmetic_yoy_identity():
    rng = np.random.default_rng(7)
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 60))))
    arith = 100 * (price / price.shift(12) - 1)
    logg = 100 * np.log(price / price.shift(12))
    converted = 100 * np.log1p(arith / 100)
    pd.testing.assert_series_equal(converted.dropna(), logg.dropna(),
                                   check_names=False, atol=1e-10, rtol=0)


def test_converter_centre_and_delta_method_scale():
    from pipeline.blocks._common import arith_to_log_yoy

    centre = np.array([17.2, 0.0, -25.0])
    s = np.array([2.0, 2.0, 2.0])
    lc, ls = arith_to_log_yoy(centre, s)
    assert np.allclose(lc, 100 * np.log1p(centre / 100))
    # delta method: sigma shrinks by 1/(1+c/100) above zero, grows below
    assert np.allclose(ls, s / (1 + centre / 100))
    assert ls[0] < 2.0 < ls[2]
    # gamma is a shape ratio and passes through unchanged by construction


def test_converter_rejects_impossible_arithmetic_growth():
    from pipeline.blocks._common import arith_to_log_yoy

    with pytest.raises(ValueError):
        arith_to_log_yoy(np.array([-100.0]))
    with pytest.raises(ValueError):
        arith_to_log_yoy(np.array([np.nan, -120.0]))


@pytest.mark.skipif(not (REPO / "input" / "commodities").exists(),
                    reason="private data caches are not present")
def test_g_tdi_equals_log_yoy_of_the_official_index():
    """Overlap check on real data: the two series are the SAME index under two
    transforms. The raw gap is material; the converted gap is numerical zero."""
    import sys

    sys.path.insert(0, str(REPO))
    import targets
    from targets import commodities as cmd

    mm, _, _ = targets.get("peru_gdp").load_panel()
    cm, _, _ = cmd.SPECS["pe_tot"].load_panel()
    j = pd.concat([mm["g_tdi"].rename("tdi"), cm["g_pe_tot"].rename("tot")],
                  axis=1).dropna()
    assert len(j) > 200
    raw_rmse = float(np.sqrt(((j.tot - j.tdi) ** 2).mean()))
    conv = 100 * np.log1p(j.tot / 100) - j.tdi
    conv_rmse = float(np.sqrt((conv ** 2).mean()))
    assert raw_rmse > 0.5, "the raw units no longer differ; retire the converter"
    assert conv_rmse < 1e-6, f"conversion does not reproduce g_tdi (rmse {conv_rmse})"
