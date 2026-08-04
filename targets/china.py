"""China satellite target: quarterly real GDP growth, YoY (`gdp_yoy`).

Structural note: China's monthly GDP reference lags GDP itself (95 vs 18 days), so
there is no timely monthly GDP proxy - the nowcast is indicator-driven (PMIs 0d,
prices/money 9-13d, activity/property 15d). Four NBS series publish only as
year-to-date growth and are de-cumulated. The quarterly index is re-dated to the
MIDAS convention (first day of the end month) inside ``load_raw``.

This module is the reusable data interface used by ``notebooks/china`` and the pipeline. Production models live in ``pipeline/config/metadata.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis import data as pdata
from .base import Target

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "input" / "china"
MONTHLY_RAW = RAW_DIR / "nbs_monthly_history.csv"
QUARTERLY_RAW = RAW_DIR / "china_quarterly_modeling.csv"

TARGET = "gdp_yoy"
TARGET_DELAY_DAYS = 18
TARGET_GROUP = "GDP"
MONTHLY_PROXY = None            # deliberately none: monthly GDP ref lags GDP

TRANSFORMS = {
    "ip_yoy": "none", "ip_cum_yoy": "decum_yoy", "retail_sales_yoy": "none",
    "electricity_output_yoy": "none", "manufacturing_pmi": "none",
    "non_manufacturing_pmi": "none", "fixed_asset_investment_cum_yoy": "decum_yoy",
    "real_estate_investment_cum_yoy": "decum_yoy", "property_sales_floor_cum_yoy": "decum_yoy",
    "cpi_index_yoy": "none", "ppi_index_yoy": "none", "m2_yoy": "none",
}
DELAYS = {
    "manufacturing_pmi": 0, "non_manufacturing_pmi": 0, "ppi_index_yoy": 9,
    "cpi_index_yoy": 12, "m2_yoy": 13, "ip_yoy": 15, "ip_cum_yoy": 15,
    "retail_sales_yoy": 15, "electricity_output_yoy": 15,
    "fixed_asset_investment_cum_yoy": 15, "real_estate_investment_cum_yoy": 15,
    "property_sales_floor_cum_yoy": 15,
}
GROUPS = {
    "ip_yoy": "activity", "ip_cum_yoy": "activity", "retail_sales_yoy": "activity",
    "electricity_output_yoy": "activity", "manufacturing_pmi": "survey",
    "non_manufacturing_pmi": "survey", "fixed_asset_investment_cum_yoy": "property",
    "real_estate_investment_cum_yoy": "property", "property_sales_floor_cum_yoy": "property",
    "cpi_index_yoy": "prices", "ppi_index_yoy": "prices", "m2_yoy": "money",
}

SELECTED_LEADS = (-120, -105, -90, -75, -60, -45, -30, -15, -1, 0)
EVAL_START, EVAL_END = "2010-01-01", "2019-12-31"
CORR_START, CORR_END = "2000-01-01", "2019-12-31"
BACKTEST_START, BACKTEST_END = "2010-01-01", "2026-12-31"
SUBPERIODS = {
    "2010-2019":          ("2010-01-01", "2019-12-31", ()),
    "2010-2026 ex-COVID": ("2010-01-01", "2026-12-31", (2020, 2021)),
    "2022-2026":          ("2022-01-01", "2026-12-31", ()),
}


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Committed monthly NBS panel and quarterly file, re-dated to MIDAS quarters."""

    m = pd.read_csv(MONTHLY_RAW)
    m["period"] = pd.PeriodIndex(m["period"].astype(str), freq="M").to_timestamp(how="start")
    m = m.set_index("period").sort_index()

    q = pd.read_csv(QUARTERLY_RAW)
    q["period"] = pd.to_datetime(q["period"])
    q = q.set_index("period").sort_index()
    q.index = pd.PeriodIndex(q.index, freq="Q").to_timestamp(how="end").to_period("M").to_timestamp()
    return m, q


def load_panel(extra: pd.DataFrame | None = None, extra_meta: list | None = None):
    """Transformed monthly + quarterly frames and the MIDAS panel with delays.

    ``extra`` / ``extra_meta`` promote additional monthly columns (e.g. the US
    external block from ``targets.usa.us_block``) into the panel, mirroring the
    Peru candidates pattern.
    """

    monthly_raw, quarterly_raw = load_raw()
    monthly, quarterly, panel = pdata.assemble_panel(
        monthly_raw, quarterly_raw, transforms=TRANSFORMS, delays=DELAYS, groups=GROUPS,
        target=TARGET, target_delay=TARGET_DELAY_DAYS, fill_jan_gap=True,
    )
    if extra is not None:
        from MIDAS import MetadataPanel, VariableMeta

        monthly = pd.concat([monthly, extra.reindex(monthly.index)], axis=1)
        metas = [VariableMeta(column=c, frequency="M", group=GROUPS.get(c, "other"),
                              publication_delay_days=int(DELAYS[c]))
                 for c in monthly.columns if c in DELAYS]
        metas.extend(extra_meta or [])
        metas.append(VariableMeta(column=TARGET, frequency="Q", group=TARGET_GROUP,
                                  publication_delay_days=TARGET_DELAY_DAYS))
        panel = MetadataPanel.from_frames(monthly, quarterly, metas)
    return monthly, quarterly, panel


def metadata_map() -> pd.DataFrame:
    return pdata.metadata_map(DELAYS, GROUPS, TARGET, TARGET_DELAY_DAYS, target_label="Real GDP YoY")


def series_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    return pdata.series_summary(monthly, transforms=TRANSFORMS, delays=DELAYS,
                                groups=GROUPS, start=EVAL_START, end=EVAL_END)


# Catalog concept -> modelling-CSV column. The catalogue names some series
# differently from the committed panels; without this translation a freshly
# published release is fetched and then silently dropped as "unmapped".
CONCEPT_TO_COLUMN = {
    # quarterly
    "gdp_real_yoy": "gdp_yoy",
    "gdp_real_qoq_sa": "gdp_qoq_sa",
    # monthly
    "cpi_yoy": "cpi_index_yoy",
    "ppi_yoy": "ppi_index_yoy",
    "industrial_value_added_yoy": "ip_yoy",
    "industrial_value_added_cum_yoy": "ip_cum_yoy",
    "fixed_asset_investment_yoy": "fixed_asset_investment_cum_yoy",  # NBS FAI is YTD-cumulative
}


def _fetch_nbs(frequency: str, refresh: bool = False) -> pd.DataFrame | None:
    """Fetch the active NBS series for one frequency via the uniform interface,
    renamed to the modelling-CSV column names. Duplicate concepts (several
    backends per series) are combined, first-listed backend winning per cell.
    Returns None when nothing usable."""

    import sources
    from sources import registry

    cat = registry.load_catalog()
    rows = cat[(cat["provider"] == "nbs") & (cat["frequency"] == frequency)]
    if "active" in rows.columns:
        rows = rows[rows["active"].astype(bool)]
    if rows.empty:
        return None
    fresh = sources.fetch("nbs", list(rows["series_id"]), frequency=frequency,
                          refresh=refresh)
    if fresh is None or fresh.empty:
        return None
    if "concept" in rows.columns:
        fresh = fresh.rename(columns=dict(zip(rows["series_id"], rows["concept"])))
    fresh = fresh.rename(columns=CONCEPT_TO_COLUMN)
    # several backends can serve one column: combine them (first listed wins per cell)
    combined = {}
    for c in dict.fromkeys(fresh.columns):
        block = fresh.loc[:, fresh.columns == c]
        combined[c] = block.bfill(axis=1).iloc[:, 0] if block.shape[1] > 1 else block.iloc[:, 0]
    fresh = pd.DataFrame(combined)
    fresh.index = pd.PeriodIndex(fresh.index, freq=frequency).to_timestamp(how="start") \
        if not isinstance(fresh.index, pd.DatetimeIndex) else pd.DatetimeIndex(fresh.index)
    return fresh


def _merge_csv(path, fresh: pd.DataFrame, *, period_fmt: str) -> tuple[int, int, list[str]]:
    """Merge fresh (canonical-named) data into a committed CSV; fresh values win.

    Returns (new_rows, revised_cells, unmatched_columns)."""

    old = pd.read_csv(path)
    old["period"] = pd.to_datetime(
        pd.PeriodIndex(old["period"].astype(str), freq="M").to_timestamp(how="start")
        if period_fmt == "M" else old["period"])
    old = old.set_index("period").sort_index()

    matched = [c for c in fresh.columns if c in old.columns]
    unmatched = [c for c in fresh.columns if c not in old.columns]
    if not matched:
        return 0, 0, unmatched

    f = fresh[matched].copy()
    f.index = pd.PeriodIndex(f.index, freq=period_fmt).to_timestamp(how="start")
    f = f[~f.index.duplicated()].sort_index()
    new_rows = int((~f.index.isin(old.index) & f.notna().any(axis=1)).sum())
    both = f.reindex(old.index)[matched]
    revised = int(((both.notna()) & (old[matched].notna())
                   & (~np.isclose(both, old[matched], equal_nan=True))).to_numpy().sum())

    upd = f.combine_first(old).sort_index()
    upd = upd[old.columns]                      # preserve the file's column order
    upd.index.name = "period"
    out = upd.reset_index()
    out["period"] = (out["period"].dt.to_period("M").astype(str) if period_fmt == "M"
                     else out["period"].dt.strftime("%Y-%m-%d"))
    from sources.base import atomic_write_csv
    atomic_write_csv(out, path, index=False)
    return new_rows, revised, unmatched


def refresh() -> list[str]:
    """Pull fresh NBS data and update the committed snapshots; report what's new.

    Never raises: on any fetch failure the committed snapshot stays in place and
    the failure is reported as a message.
    """

    msgs = []
    m_old, q_old = load_raw()
    old_m_last = m_old.dropna(how="all").index.max()
    old_gdp_last = q_old[TARGET].dropna().index.max()

    try:
        fresh_m = _fetch_nbs("M", refresh=True)
        if fresh_m is not None:
            n_new, n_rev, unmatched = _merge_csv(MONTHLY_RAW, fresh_m, period_fmt="M")
            m_new, _ = load_raw()
            last = m_new.dropna(how="all").index.max()
            if n_new or n_rev:
                msgs.append(f"NBS monthly: +{n_new} new month-rows, {n_rev} revised cells "
                            f"(panel now through {last:%Y-%m})")
            else:
                msgs.append(f"NBS monthly: no new releases (panel through {last:%Y-%m})")
            if unmatched:
                msgs.append(f"NBS monthly: {len(unmatched)} fetched series are not in the "
                            "modelling panel (skipped)")
        else:
            msgs.append(f"NBS monthly: nothing fetched; snapshot through {old_m_last:%Y-%m}")
    except Exception as exc:
        msgs.append(f"NBS monthly: fetch FAILED ({type(exc).__name__}: {exc}); "
                    f"using snapshot through {old_m_last:%Y-%m}")

    try:
        fresh_q = _fetch_nbs("Q", refresh=True)
        if fresh_q is not None:
            n_new, n_rev, unmatched = _merge_csv(QUARTERLY_RAW, fresh_q, period_fmt="Q")
            _, q_new = load_raw()
            gdp_last = q_new[TARGET].dropna().index.max()
            if gdp_last > old_gdp_last:
                qp = pd.Period(gdp_last, freq="Q")
                msgs.append(f"China GDP: NEW release {qp.year}Q{qp.quarter} = "
                            f"{q_new[TARGET].dropna().iloc[-1]:.1f} YoY")
            else:
                qp = pd.Period(old_gdp_last, freq="Q")
                msgs.append(f"China GDP: fetched OK, but the source has no release newer "
                            f"than {qp.year}Q{qp.quarter}")
        else:
            qp = pd.Period(old_gdp_last, freq="Q")
            msgs.append(f"China GDP: nothing fetched; latest {qp.year}Q{qp.quarter}")
    except Exception as exc:
        qp = pd.Period(old_gdp_last, freq="Q")
        msgs.append(f"China GDP: fetch FAILED ({type(exc).__name__}: {exc}); "
                    f"latest remains {qp.year}Q{qp.quarter}")
    return msgs


SPEC = Target(
    name="china", role="satellite", label="China real GDP, YoY %",
    target=TARGET, target_delay_days=TARGET_DELAY_DAYS, monthly_proxy=MONTHLY_PROXY,
    selected_leads=SELECTED_LEADS, subperiods=SUBPERIODS,
    backtest_start=BACKTEST_START, backtest_end=BACKTEST_END,
    load_panel=load_panel, metadata_map=metadata_map,
)
