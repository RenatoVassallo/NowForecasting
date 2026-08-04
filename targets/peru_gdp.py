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


class PanelRebuildError(RuntimeError):
    """A Peru release is due but the model panel cannot be rebuilt."""


def _release_due(last_obs_month, as_of, delay_days: int) -> tuple[bool, str]:
    """Pure due rule: the latest month whose release date has passed at
    ``as_of`` must be present in the panel."""
    as_of = pd.Timestamp(as_of).normalize()
    last = pd.Period(last_obs_month, freq="M")
    cand = pd.Period(as_of, freq="M")
    due = cand
    for _ in range(48):
        if (due.to_timestamp(how="end").normalize()
                + pd.Timedelta(days=int(delay_days))) <= as_of:
            break
        due -= 1
    if last >= due:
        return False, f"panel through {last}, latest due month is {due}"
    return True, f"month {due} was released by {as_of.date()} but the panel ends at {last}"


def panel_release_due(as_of=None) -> tuple[bool, str]:
    """Is a spec3 observation due at ``as_of`` that the cache lacks?

    The gate watches the raw monthly GDP proxy (``g_pbim``) in the processed
    snapshot, the slowest required series and the one the whole ladder hangs
    off; its delay equals the target's 51 days.
    """
    as_of = pd.Timestamp.now().normalize() if as_of is None else pd.Timestamp(as_of)
    monthly, _, _, _ = load_processed()
    col = "g_pbim" if "g_pbim" in monthly.columns else MONTHLY_PROXY
    last = monthly[col].dropna().index.max()
    return _release_due(last, as_of, TARGET_DELAY_DAYS)


def rebuild_panel(*, refresh_downloads: bool = True) -> list[str]:
    """Rebuild the EXACT spec3 panel production consumes: raw download, X13
    seasonal adjustment, transforms, snapshot export into ``input/peru``."""
    from core import preprocess

    preprocess.locate_x13_binary()          # fail fast with an actionable error
    art = preprocess.build_processed_artifacts(
        spec="spec3", refresh_downloads=refresh_downloads, refresh_sa=True)
    preprocess.export_processed_snapshot(art, output_dir=PROCESSED_DIR)
    m_last = art.panel.monthly.dropna(how="all").index.max()
    return [f"spec3 panel REBUILT (X13 + transforms): monthly through {m_last:%Y-%m}"]


def refresh(as_of=None) -> list[str]:
    """Pull fresh Peru data; rebuild the model panel when a release is due.

    * **INEI** bulletins: live idempotent update into ``input/inei``.
    * **spec3 caches** (the modelling panel production actually consumes):
      when the release calendar says a new observation is DUE and the cache
      lacks it, the panel is rebuilt through the full preprocess (download,
      X13, transforms). If that rebuild is impossible the refresh RAISES
      ``PanelRebuildError``: a refresh that only updated an unrelated INEI
      cache must never count as success while the model panel is stale.
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

    due, why = panel_release_due(as_of)
    if not due:
        msgs.append(f"spec3 panel: current ({why})")
        return msgs
    try:
        msgs += rebuild_panel()
    except Exception as exc:
        raise PanelRebuildError(
            f"a Peru release is due ({why}) but the spec3 model panel cannot "
            f"be rebuilt: {type(exc).__name__}: {exc}. Install the X13 binary "
            "(core.preprocess.locate_x13_binary) or rebuild manually; a refresh "
            "that only touched INEI is NOT a successful Peru refresh.") from exc
    return msgs


SPEC = Target(
    name="peru_gdp", role="domestic", label="Peru real GDP, YoY %",
    target=TARGET, target_delay_days=TARGET_DELAY_DAYS, monthly_proxy=MONTHLY_PROXY,
    selected_leads=SELECTED_LEADS, subperiods=SUBPERIODS,
    backtest_start=BACKTEST_START, backtest_end=BACKTEST_END,
    load_panel=load_panel, metadata_map=metadata_map,
)
