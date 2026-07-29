"""US external target: bought, not built.

The US is the one target with no in-house model (deliberate: GDPNow / NY Fed /
SPF are the free frontier and very hard to beat). This target exists so that

* the pipeline's data stage snapshots the US block per run - accumulating a
  real-time vintage database of what the consensus believed each month, and
* other targets (China, copper) can join the US block - including the SPF
  forecast *paths*, which are time-t observables and therefore legitimate
  regressors for h >= 1 - via :func:`us_block`.

It has NO entry in ``pipeline/config/metadata.py:MODELS``, so the model stages
skip it by construction. Data caching lives in :mod:`sources.us`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from MIDAS import MetadataPanel, VariableMeta
from .base import Target

REPO_ROOT = Path(__file__).resolve().parents[1]

TARGET = "us_gdp_yoy"
TARGET_DELAY_DAYS = 30          # BEA advance estimate ~30 days after quarter end
TARGET_GROUP = "US activity"

# Publication delays (days after the reference month) for the monthly block.
# ``us_indpro`` / ``us_dollar_broad`` are level indices: modelling uses their
# YoY growth (``g_``), added in ``load_panel`` - levels are not valid regressors.
DELAYS = {
    "us_indpro": 15, "us_unrate": 7, "us_curve_10y3m": 1,
    "us_dollar_broad": 1, "us_nfci": 7,
    "g_us_indpro": 15, "g_us_dollar": 1,
    "us_vix": 1, "us_cpi_yoy": 12, "us_fedfunds": 1,
    "us_gdp_saar_m": 30,   # advance estimate ~30d after quarter end (on the end month)
    "us_gdp_yoy_m": 30,    # GDPC1 percent change from year ago, same release
}
# SPF: surveyed mid-quarter; values sit on the survey quarter's first month.
SPF_DELAY_DAYS = 45
LABELS = {
    "us_indpro": "US industrial production", "us_unrate": "US unemployment",
    "us_curve_10y3m": "US yield-curve slope", "us_dollar_broad": "US broad dollar",
    "us_nfci": "US financial conditions", "us_gdpnow": "Atlanta Fed GDPNow",
    "spf_gdp_h0": "SPF GDP growth, current Q", "spf_gdp_h1": "SPF GDP growth, +1Q",
    "spf_gdp_h2": "SPF GDP growth, +2Q", "spf_gdp_h3": "SPF GDP growth, +3Q",
    "spf_gdp_h4": "SPF GDP growth, +4Q",
}

SELECTED_LEADS = (-120, -105, -90, -75, -60, -45, -30, -15, -1, 0)
BACKTEST_START, BACKTEST_END = "2010-01-01", "2026-12-31"
SUBPERIODS = {
    "2010-2019":          ("2010-01-01", "2019-12-31", ()),
    "2010-2026 ex-COVID": ("2010-01-01", "2026-12-31", (2020, 2021)),
    "2022-2026":          ("2022-01-01", "2026-12-31", ()),
}


def _spf_monthly(index: pd.DatetimeIndex) -> pd.DataFrame:
    """SPF survey paths as monthly columns (survey quarter's first month)."""

    from sources import us as src

    spf = src.load("spf")
    spf.index = pd.DatetimeIndex(spf.index)
    return spf.reindex(index)


def load_panel():
    """US monthly block (+ SPF paths) and quarterly realized GDP growth."""

    from sources import us as src

    monthly = src.load("monthly").copy()
    monthly.index = pd.DatetimeIndex(monthly.index)
    monthly["g_us_indpro"] = 100.0 * (monthly["us_indpro"] / monthly["us_indpro"].shift(12) - 1.0)
    monthly["g_us_dollar"] = 100.0 * (monthly["us_dollar_broad"] / monthly["us_dollar_broad"].shift(12) - 1.0)
    if "us_cpi_index" in monthly.columns:
        monthly["us_cpi_yoy"] = 100.0 * (monthly["us_cpi_index"] / monthly["us_cpi_index"].shift(12) - 1.0)
    monthly = monthly.join(_spf_monthly(monthly.index))

    q_raw = src.load("quarterly").copy()
    q_raw.index = pd.DatetimeIndex(q_raw.index)
    lvl0 = q_raw["us_gdp_level"]
    saar = 400.0 * (np.log(lvl0) - np.log(lvl0.shift(1)))
    end_month = pd.PeriodIndex(saar.index, freq="Q").to_timestamp(how="end").to_period("M").to_timestamp()
    monthly["us_gdp_saar_m"] = pd.Series(saar.to_numpy(), index=end_month).reindex(monthly.index)
    yoy = 100.0 * (lvl0 / lvl0.shift(4) - 1.0)      # GDPC1, percent change from year ago
    monthly["us_gdp_yoy_m"] = pd.Series(yoy.to_numpy(), index=end_month).reindex(monthly.index)

    q = src.load("quarterly").copy()
    q.index = pd.DatetimeIndex(q.index)
    # MIDAS convention: quarter dated at the first day of its end month.
    q.index = pd.PeriodIndex(q.index, freq="Q").to_timestamp(how="end").to_period("M").to_timestamp()
    lvl = q["us_gdp_level"]
    quarterly = (100.0 * (lvl / lvl.shift(4) - 1.0)).rename(TARGET).to_frame()

    metas = [VariableMeta(column=c, frequency="M", group="US activity",
                          publication_delay_days=int(d)) for c, d in DELAYS.items()
             if c in monthly.columns]
    metas += [VariableMeta(column=c, frequency="M", group="US consensus",
                           publication_delay_days=SPF_DELAY_DAYS)
              for c in monthly.columns if c.startswith("spf_")]
    metas.append(VariableMeta(column=TARGET, frequency="Q", group=TARGET_GROUP,
                              publication_delay_days=TARGET_DELAY_DAYS))
    return monthly, quarterly, MetadataPanel.from_frames(monthly, quarterly, metas)


def metadata_map() -> pd.DataFrame:
    rows = [{"column": c, "label": LABELS.get(c, c), "group": "US activity",
             "frequency": "M", "publication_delay_days": d} for c, d in DELAYS.items()]
    rows += [{"column": f"spf_gdp_h{i}", "label": LABELS[f"spf_gdp_h{i}"],
              "group": "US consensus", "frequency": "M",
              "publication_delay_days": SPF_DELAY_DAYS} for i in range(5)]
    rows.append({"column": TARGET, "label": "US real GDP, YoY", "group": TARGET_GROUP,
                 "frequency": "Q", "publication_delay_days": TARGET_DELAY_DAYS})
    return pd.DataFrame(rows)


MODEL_COLUMNS = ["g_us_indpro", "us_unrate", "us_curve_10y3m", "g_us_dollar", "us_nfci",
                 "us_vix", "us_cpi_yoy", "us_fedfunds", "us_gdp_saar_m", "us_gdp_yoy_m",
                 "spf_gdp_h0", "spf_gdp_h1", "spf_gdp_h2", "spf_gdp_h3", "spf_gdp_h4"]


def us_block(index: pd.DatetimeIndex, columns=MODEL_COLUMNS) -> tuple[pd.DataFrame, list]:
    """The US block aligned to another target's monthly index, ready for
    ``MetadataPanel`` promotion (mirrors the Peru candidates pattern).

    Defaults to the STATIONARY model columns (growth rates, rates, spreads and
    the SPF paths) - never the raw level indices."""

    monthly, _, _ = load_panel()
    block = monthly.reindex(index)[[c for c in columns if c in monthly.columns]]
    metas = [VariableMeta(column=c, frequency="M",
                          group="US consensus" if c.startswith("spf_") else "US activity",
                          publication_delay_days=SPF_DELAY_DAYS if c.startswith("spf_")
                          else int(DELAYS.get(c, 15)))
             for c in block.columns]
    return block, metas


def refresh() -> list[str]:
    from sources import us as src

    return src.refresh()


SPEC = Target(
    name="usa", role="satellite", label="US real GDP (external consensus)",
    target=TARGET, target_delay_days=TARGET_DELAY_DAYS, monthly_proxy=None,
    selected_leads=SELECTED_LEADS, subperiods=SUBPERIODS,
    backtest_start=BACKTEST_START, backtest_end=BACKTEST_END,
    load_panel=load_panel, metadata_map=metadata_map,
)
