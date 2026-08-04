"""The commodity and terms-of-trade block: copper, gold, WTI, and Peru's export
price index and terms of trade.

One shared panel serves all five targets, because they share the same drivers:
global demand (China industrial production and money), the US cycle and policy
stance (industrial production, inflation, the funds rate, financial conditions),
the dollar, and global risk appetite (VIX). Each target is the **YoY growth of
the quarterly average** of its price or index, so the target is stationary and
comparable across the block.

Peru's export price index and terms of trade are the point of the exercise: they
are the channel through which the global cycle reaches the Peruvian economy, and
they are far more forecastable than the individual metal prices because they are
weighted averages (idiosyncratic price noise partly cancels).

Note the deliberate duplication with :mod:`targets.copper`: that module is what
the production pipeline runs, and it keeps its leaner panel. This module is the
research block for the whole commodity family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from MIDAS import MetadataPanel, VariableMeta
from .base import Target

TARGET_GROUP = "Commodity prices"

# monthly price/index columns -> the YoY driver name we model with
PRICES = {"p_copper": "g_copper", "p_gold": "g_gold", "p_wti": "g_wti",
          "p_silver": "g_silver", "pe_px": "g_pe_px", "pe_pm": "g_pe_pm",
          "pe_tot": "g_pe_tot"}

# the five targets: column in the monthly frame -> (target name, label, delay)
TARGETS = {
    "p_copper": ("g_copper_q", "Copper price, quarterly avg YoY %", 15),
    "p_gold":   ("g_gold_q",   "Gold price, quarterly avg YoY %", 15),
    "p_wti":    ("g_wti_q",    "WTI crude, quarterly avg YoY %", 15),
    "pe_px":    ("g_pe_px_q",  "Peru export price index, quarterly avg YoY %", 40),
    "pe_tot":   ("g_pe_tot_q", "Peru terms of trade, quarterly avg YoY %", 40),
}

# publication delays (days after the reference month)
DELAYS = {"g_copper": 15, "g_gold": 15, "g_wti": 15, "g_silver": 15,
          "g_pe_px": 40, "g_pe_pm": 40, "g_pe_tot": 40,
          "ip_cum_yoy": 15, "m2_yoy": 13,
          "g_us_indpro": 15, "us_cpi_yoy": 12, "us_fedfunds": 1, "us_vix": 1,
          "g_us_dollar": 1, "us_nfci": 7, "us_gdp_yoy_m": 30, "us_gdp_saar_m": 30,
          "spf_gdp_h0": 45, "spf_gdp_h1": 45, "spf_gdp_h2": 45}

US_COLS = ["g_us_indpro", "us_cpi_yoy", "us_fedfunds", "us_vix", "g_us_dollar",
           "us_nfci", "us_gdp_yoy_m", "us_gdp_saar_m",
           "spf_gdp_h0", "spf_gdp_h1", "spf_gdp_h2"]
CHINA_COLS = ["ip_cum_yoy", "m2_yoy"]

SELECTED_LEADS = (-120, -105, -90, -75, -60, -45, -30, -15, -1, 0)
BACKTEST_START, BACKTEST_END = "2005-01-01", "2026-12-31"
SUBPERIODS = {
    "2010-2019":          ("2010-01-01", "2019-12-31", ()),
    "2010-2026 ex-COVID": ("2010-01-01", "2026-12-31", (2020, 2021)),
    "2022-2026":          ("2022-01-01", "2026-12-31", ()),
}


def _quarter_stamp(p: pd.Period) -> pd.Timestamp:
    return p.to_timestamp(how="end").to_period("M").to_timestamp()


def _monthly_block() -> pd.DataFrame:
    """Every monthly driver in the block, all stationary (YoY / rates / spreads)."""

    from sources import commodities as src
    from targets import china as cn
    from targets import usa

    prices = src.load()
    prices.index = pd.DatetimeIndex(prices.index)
    cn_m, _, _ = cn.load_panel()
    us_m, _, _ = usa.load_panel()

    # The panel index is the UNION of all blocks. Anchoring on the price index
    # alone silently truncated every faster series to the FRED metals' last
    # month, so a published July VIX or US IP never reached the ToT model's
    # ragged-edge conditioning (found by the availability preflight).
    idx = prices.index.union(cn_m.index).union(us_m.index)
    monthly = pd.DataFrame(index=idx)
    for raw, g in PRICES.items():
        if raw in prices.columns:
            monthly[g] = (100.0 * (prices[raw] / prices[raw].shift(12) - 1.0)
                          ).reindex(idx)
    monthly = monthly.join(cn_m[CHINA_COLS].reindex(idx))
    monthly = monthly.join(us_m[[c for c in US_COLS if c in us_m.columns]]
                           .reindex(idx))
    return monthly


def _quarterly(price_col: str, target: str) -> pd.DataFrame:
    from sources import commodities as src

    prices = src.load()
    prices.index = pd.DatetimeIndex(prices.index)
    s = prices[price_col].dropna()
    # COMPLETE quarters only. Sources run to different months (FRED's IMF prices
    # publish a month ahead of BCRP's), so the newest quarter can hold one or two
    # months; averaging those against a full three-month base year produces a
    # spurious YoY move in exactly the observation the live forecast anchors on.
    g = s.groupby(pd.PeriodIndex(s.index, freq="Q"))
    q_avg = g.mean().where(g.count() >= 3)
    g_q = 100.0 * (q_avg / q_avg.shift(4) - 1.0)
    idx = pd.DatetimeIndex([_quarter_stamp(p) for p in g_q.index])
    return pd.DataFrame({target: g_q.to_numpy()}, index=idx).dropna()


def make_load_panel(price_col: str, target: str):
    def load_panel(extra: pd.DataFrame | None = None, extra_meta: list | None = None):
        monthly = _monthly_block()
        if extra is not None:
            monthly = monthly.join(extra.reindex(monthly.index), rsuffix="_x")
        quarterly = _quarterly(price_col, target)
        metas = [VariableMeta(column=c, frequency="M", group=TARGET_GROUP,
                              publication_delay_days=int(d))
                 for c, d in DELAYS.items() if c in monthly.columns]
        if extra_meta:
            metas += list(extra_meta)
        metas.append(VariableMeta(column=target, frequency="Q", group=TARGET_GROUP,
                                  publication_delay_days=int(TARGETS[price_col][2])))
        return monthly, quarterly, MetadataPanel.from_frames(monthly, quarterly, metas)
    return load_panel


def make_metadata_map(price_col: str, target: str):
    def metadata_map() -> pd.DataFrame:
        rows = [{"column": c, "label": c, "group": TARGET_GROUP, "frequency": "M",
                 "publication_delay_days": d} for c, d in DELAYS.items()]
        rows.append({"column": target, "label": TARGETS[price_col][1],
                     "group": TARGET_GROUP, "frequency": "Q",
                     "publication_delay_days": TARGETS[price_col][2]})
        return pd.DataFrame(rows)
    return metadata_map


def _spec(price_col: str) -> Target:
    target, label, delay = TARGETS[price_col]
    name = target[2:-2] if target.startswith("g_") else target      # g_copper_q -> copper
    return Target(
        name=name, role="satellite", label=label, target=target,
        target_delay_days=delay, monthly_proxy=PRICES.get(price_col),
        selected_leads=SELECTED_LEADS, subperiods=SUBPERIODS,
        backtest_start=BACKTEST_START, backtest_end=BACKTEST_END,
        load_panel=make_load_panel(price_col, target),
        metadata_map=make_metadata_map(price_col, target))


SPECS: dict[str, Target] = {_spec(c).name: _spec(c) for c in TARGETS}


def refresh() -> list[str]:
    from sources import commodities as src

    return src.refresh()


__all__ = ["SPECS", "TARGETS", "PRICES", "DELAYS", "US_COLS", "CHINA_COLS", "refresh"]
