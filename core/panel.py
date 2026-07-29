"""Active DSAPM panel helpers.

This module defines the current data conventions for the Peru GDP nowcasting
project. It translates the metadata catalogue into canonical panel column names
and loads either the legacy Excel database or freshly processed monthly and
quarterly frames into a :class:`MIDAS.MetadataPanel`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from MIDAS import MetadataPanel, VariableMeta


def read_metadata(metadata_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the monthly and quarterly metadata sheets."""

    path = Path(metadata_path)
    monthly = pd.read_excel(path, sheet_name="Monthly")
    quarterly = pd.read_excel(path, sheet_name="Quarterly")
    return monthly, quarterly


def read_database_sheet(path: str | Path, sheet: str, *, date_col: str = "date") -> pd.DataFrame:
    """Read a dated Excel sheet into a sorted date-indexed frame."""

    df = pd.read_excel(path, sheet_name=sheet)
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col).sort_index()


def canonical_column(variable: str, transform: str) -> str:
    """Column name used by the active processed panel.

    Growth transforms keep the historical `g_` prefix convention, so the public
    model configuration remains stable even when the underlying transformation
    changes from YoY to annualized MoM or QoQ.
    """

    var = str(variable)
    tr = str(transform).lower()
    if tr in {"yoy", "mom", "mom_ann", "qoq_ann"} and not var.startswith("g_"):
        return f"g_{var}"
    return var


def metadata_with_columns(
    metadata_path: str | Path,
    *,
    spec: str = "spec3",
    active_only: bool = True,
) -> pd.DataFrame:
    """Metadata catalogue augmented with canonical active panel columns."""

    monthly, quarterly = read_metadata(metadata_path)
    meta = pd.concat([monthly.assign(sheet="monthly"), quarterly.assign(sheet="quarterly")], ignore_index=True)
    if active_only:
        meta = meta.loc[meta["active"].fillna(1).astype(int).eq(1)].copy()
    meta["transform"] = meta[spec].astype(str).str.lower()
    meta["column"] = [canonical_column(v, t) for v, t in zip(meta["variable"], meta["transform"])]
    meta["frequency"] = meta["frequency"].astype(str).str.upper()
    return meta


def metadata_to_variable_meta(meta: pd.DataFrame) -> list[VariableMeta]:
    """Convert the catalogue to the metadata objects consumed by MIDAS."""

    rows: list[VariableMeta] = []
    for row in meta.itertuples(index=False):
        rows.append(
            VariableMeta(
                column=str(row.column),
                frequency=str(row.frequency).upper(),
                group=str(row.group),
                publication_delay_days=int(row.publication_delay_days),
                label=None if pd.isna(getattr(row, "label", None)) else str(row.label),
                unit=None if pd.isna(getattr(row, "unit", None)) else str(row.unit),
            )
        )
    return rows


def panel_from_processed_frames(
    monthly: pd.DataFrame,
    quarterly: pd.DataFrame,
    meta: pd.DataFrame,
) -> MetadataPanel:
    """Build a :class:`MetadataPanel` from processed frames and the active catalogue."""

    metas = metadata_to_variable_meta(meta)
    return MetadataPanel.from_frames(monthly.sort_index(), quarterly.sort_index(), metas)


def load_database_panel(
    database_path: str | Path,
    metadata_path: str | Path,
    *,
    spec: str = "spec3",
    active_only: bool = True,
) -> MetadataPanel:
    """Load the legacy harmonized Excel database under the active naming rules."""

    meta = metadata_with_columns(metadata_path, spec=spec, active_only=active_only)
    monthly_db = read_database_sheet(database_path, "monthly")
    quarterly_db = read_database_sheet(database_path, "quarterly")
    return panel_from_processed_frames(monthly_db, quarterly_db, meta)


def benchmark_indicators(
    panel: MetadataPanel,
    *,
    monthly_gdp: str = "g_pbim",
) -> list[str]:
    """Monthly GDP plus the business-survey block, in the current panel."""

    surveys = [c for c in panel.columns_by_group().get("Business surveys", []) if c != monthly_gdp]
    return [monthly_gdp] + surveys


__all__ = [
    "benchmark_indicators",
    "canonical_column",
    "load_database_panel",
    "metadata_to_variable_meta",
    "metadata_with_columns",
    "panel_from_processed_frames",
    "read_database_sheet",
    "read_metadata",
]
