"""Copper price target: the first commodity satellite (feeds Peru's terms of trade).

Target = **quarterly average copper price, YoY %** (`g_copper_q`), built from the
IMF/FRED monthly average. Forecasting bar (honest, per the literature): futures
and the random walk are rarely beaten beyond ~2 quarters - the goal is to match
RW at long horizons and beat it at h = 1..2 through the **China demand channel**
(China IP and money lead the copper cycle), with the real value being scenario
coherence: a China path implies a copper path implies Peru's terms of trade.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from MIDAS import MetadataPanel, VariableMeta
from .base import Target

REPO_ROOT = Path(__file__).resolve().parents[1]

TARGET = "g_copper_q"
TARGET_DELAY_DAYS = 15          # IMF monthly averages land ~2 weeks after month end
TARGET_GROUP = "Commodity prices"

DELAYS = {"g_copper": 15, "g_zinc": 15, "g_wti": 15,
          "ip_cum_yoy": 15, "m2_yoy": 13, "g_us_dollar": 1}
LABELS = {"g_copper": "Copper price, YoY", "g_zinc": "Zinc price, YoY",
          "g_wti": "WTI oil price, YoY", "ip_cum_yoy": "China IP (YoY)",
          "m2_yoy": "China M2", "g_us_dollar": "US broad dollar, YoY"}

SELECTED_LEADS = (-120, -105, -90, -75, -60, -45, -30, -15, -1, 0)
BACKTEST_START, BACKTEST_END = "2005-01-01", "2026-12-31"
SUBPERIODS = {
    "2010-2019":          ("2010-01-01", "2019-12-31", ()),
    "2010-2026 ex-COVID": ("2010-01-01", "2026-12-31", (2020, 2021)),
    "2022-2026":          ("2022-01-01", "2026-12-31", ()),
}


def load_panel():
    """Monthly drivers (copper YoY + China demand + dollar) and the quarterly target."""

    from sources import commodities as src
    from targets import china as cn

    prices = src.load()
    prices.index = pd.DatetimeIndex(prices.index)
    monthly = pd.DataFrame(index=prices.index)
    for raw, g in (("p_copper", "g_copper"), ("p_zinc", "g_zinc"), ("p_wti", "g_wti")):
        monthly[g] = 100.0 * (prices[raw] / prices[raw].shift(12) - 1.0)

    # China demand block + the dollar, aligned to the price index.
    cn_m, _, _ = cn.load_panel()
    monthly = monthly.join(cn_m[["ip_cum_yoy", "m2_yoy"]].reindex(monthly.index))
    from targets import usa

    us_m, _, _ = usa.load_panel()
    monthly = monthly.join(us_m[["g_us_dollar"]].reindex(monthly.index))

    # Quarterly target: YoY growth of the quarterly average price.
    q_avg = prices["p_copper"].groupby(pd.PeriodIndex(prices.index, freq="Q")).mean()
    g_q = 100.0 * (q_avg / q_avg.shift(4) - 1.0)
    q_idx = pd.DatetimeIndex([p.to_timestamp(how="end").to_period("M").to_timestamp()
                              for p in g_q.index])
    quarterly = pd.DataFrame({TARGET: g_q.to_numpy()}, index=q_idx)

    metas = [VariableMeta(column=c, frequency="M", group=TARGET_GROUP,
                          publication_delay_days=int(d)) for c, d in DELAYS.items()
             if c in monthly.columns]
    metas.append(VariableMeta(column=TARGET, frequency="Q", group=TARGET_GROUP,
                              publication_delay_days=TARGET_DELAY_DAYS))
    return monthly, quarterly, MetadataPanel.from_frames(monthly, quarterly, metas)


def metadata_map() -> pd.DataFrame:
    rows = [{"column": c, "label": LABELS.get(c, c), "group": TARGET_GROUP,
             "frequency": "M", "publication_delay_days": d} for c, d in DELAYS.items()]
    rows.append({"column": TARGET, "label": "Copper price, quarterly avg YoY",
                 "group": TARGET_GROUP, "frequency": "Q",
                 "publication_delay_days": TARGET_DELAY_DAYS})
    return pd.DataFrame(rows)


def refresh() -> list[str]:
    from sources import commodities as src

    return src.refresh()


SPEC = Target(
    name="copper", role="satellite", label="Copper price, quarterly avg YoY %",
    target=TARGET, target_delay_days=TARGET_DELAY_DAYS, monthly_proxy="g_copper",
    selected_leads=SELECTED_LEADS, subperiods=SUBPERIODS,
    backtest_start=BACKTEST_START, backtest_end=BACKTEST_END,
    load_panel=load_panel, metadata_map=metadata_map,
)
