"""Peru domestic target: quarterly real GDP growth, YoY (`g_pbiq`).

Peru HAS a timely monthly GDP proxy (`g_pbim_yoy`, YoY of the SA monthly index),
so the proxy bridge is the workhorse; business surveys (7-day delay) add the
early signal. The heavy preprocessing (BCRP/FRED download, X13 SA, spec3
transforms) is cached under ``input/peru``; here we read those
caches and add the proxy.

This module is the reusable data interface used by ``notebooks/peru`` and the pipeline. Production models live in ``pipeline/config/metadata.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from MIDAS import MetadataPanel, VariableMeta
from core.backtest import yoy_from_levels
from .base import Target

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "input" / "peru"

TARGET = "g_pbiq"
TARGET_DELAY_DAYS = 51
TARGET_GROUP = "Real activity"
MONTHLY_PROXY = "g_pbim_yoy"

# Delays + groups + labels come from the committed spec3 metadata (data-driven).
_mm = pd.read_csv(PROCESSED_DIR / "metadata_map_spec3.csv")
_m = _mm[_mm.frequency == "M"]
DELAYS = {c: int(d) for c, d in zip(_m.column, _m.publication_delay_days)}
DELAYS["g_pbim_yoy"] = 51
GROUPS = dict(zip(_m.column, _m.group))
GROUPS["g_pbim_yoy"] = "Sectoral activity"
LABELS = dict(zip(_mm.column, _mm.label))

SELECTED_LEADS = (-120, -105, -90, -75, -60, -45, -30, -15, -1, 0)
EVAL_START, EVAL_END = "2010-01-01", "2019-12-31"
CORR_START, CORR_END = "2005-01-01", "2019-12-31"
BACKTEST_START, BACKTEST_END = "2010-01-01", "2026-12-31"
SUBPERIODS = {
    "2010-2019":          ("2010-01-01", "2019-12-31", ()),
    "2010-2026 ex-COVID": ("2010-01-01", "2026-12-31", (2020, 2021)),
    "2022-2026":          ("2022-01-01", "2026-12-31", ()),
}


def load_processed():
    return (pd.read_parquet(PROCESSED_DIR / "monthly_panel_spec3.parquet"),
            pd.read_parquet(PROCESSED_DIR / "quarterly_sa_spec3.parquet"),
            pd.read_parquet(PROCESSED_DIR / "monthly_sa_spec3.parquet"),
            pd.read_csv(PROCESSED_DIR / "metadata_map_spec3.csv"))


def load_panel(extra: pd.DataFrame | None = None, extra_meta: list | None = None):
    """Transformed monthly panel (+ YoY proxy) and quarterly YoY target.

    ``extra`` / ``extra_meta`` promote screened candidate variables (see
    ``notebooks/peru/candidates.py``) into the pool for backtesting.
    """

    monthly, q_sa, m_sa, mm = load_processed()
    monthly = monthly.copy()
    lvl = m_sa["pbim"]
    monthly["g_pbim_yoy"] = 100.0 * (np.log(lvl) - np.log(lvl.shift(12)))
    quarterly = yoy_from_levels(q_sa["pbiq"]).rename(TARGET).to_frame()

    metas = [VariableMeta(column=c, frequency="M", group=GROUPS.get(c, "other"),
                          publication_delay_days=int(DELAYS[c]))
             for c in monthly.columns if c in DELAYS]
    if extra is not None:
        monthly = pd.concat([monthly, extra.reindex(monthly.index)], axis=1)
        metas.extend(extra_meta or [])
    metas.append(VariableMeta(column=TARGET, frequency="Q", group=TARGET_GROUP,
                              publication_delay_days=TARGET_DELAY_DAYS))
    return monthly, quarterly, MetadataPanel.from_frames(monthly, quarterly, metas)


def metadata_map() -> pd.DataFrame:
    rows = [{"column": c, "label": LABELS.get(c, c.replace("_", " ")),
             "group": GROUPS.get(c, "other"), "frequency": "M",
             "publication_delay_days": d} for c, d in DELAYS.items()]
    rows.append({"column": TARGET, "label": "Real GDP YoY", "group": TARGET_GROUP,
                 "frequency": "Q", "publication_delay_days": TARGET_DELAY_DAYS})
    return pd.DataFrame(rows)


def series_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    win = monthly.loc[EVAL_START:EVAL_END]
    return pd.DataFrame({
        "delay_days": pd.Series(DELAYS), "group": pd.Series(GROUPS),
        "missing_share": win.isna().mean().round(3), "std": win.std().round(2),
    }).dropna(subset=["missing_share"]).sort_values("delay_days")


def refresh() -> list[str]:
    """Pull fresh Peru data where automated; report the rest honestly.

    * **INEI** bulletins: live idempotent update into ``input/inei`` (new bulletins
      are ingested as vintages and the derived panels rebuilt).
    * **BCRP/FRED spec3 caches** (the modelling panel): NOT rebuilt here - the
      spec3 pipeline (download + X13 seasonal adjustment + transforms) is the
      legacy preprocess and needs its own environment. The message reports how
      current the caches are so a stale panel is never silent.

    Never raises: failures become messages and the committed data stays in place.
    """

    msgs = []
    try:
        from sources.inei import update_inei_latest
        res = update_inei_latest()
        new = res.get("new_reports", res.get("new", res)) if isinstance(res, dict) else res
        if isinstance(new, (list, tuple)):
            msgs.append(f"INEI: {len(new)} new bulletin(s) ingested" if len(new)
                        else "INEI: no new bulletins")
        else:
            msgs.append(f"INEI update: {new}")
    except ImportError:
        msgs.append("INEI: loader not available in this checkout; using cached data")
    except Exception as exc:
        msgs.append(f"INEI: update FAILED ({type(exc).__name__}: {exc}); using cached data")

    try:
        monthly, q_sa, m_sa, _ = load_processed()
        m_last = monthly.dropna(how="all").index.max()
        q_last = q_sa["pbiq"].dropna().index.max()
        qp = pd.Period(q_last, freq="Q")
        import datetime as _dt
        cache_age = (_dt.date.today()
                     - _dt.date.fromtimestamp((PROCESSED_DIR / "monthly_panel_spec3.parquet").stat().st_mtime)).days
        msgs.append(f"BCRP/FRED spec3 caches: monthly panel through {m_last:%Y-%m}, "
                    f"GDP through {qp.year}Q{qp.quarter} (cache built {cache_age}d ago; "
                    "refresh requires the legacy X13 preprocess, not run automatically)")
    except Exception as exc:
        msgs.append(f"spec3 caches: status check failed ({type(exc).__name__}: {exc})")
    return msgs


SPEC = Target(
    name="peru_gdp", role="domestic", label="Peru real GDP, YoY %",
    target=TARGET, target_delay_days=TARGET_DELAY_DAYS, monthly_proxy=MONTHLY_PROXY,
    selected_leads=SELECTED_LEADS, subperiods=SUBPERIODS,
    backtest_start=BACKTEST_START, backtest_end=BACKTEST_END,
    load_panel=load_panel, metadata_map=metadata_map,
)
