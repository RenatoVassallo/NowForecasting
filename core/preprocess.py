"""Active DSAPM preprocessing helpers.

This module rebuilds the Peru monthly and quarterly panel directly from the
provider APIs, applies seasonal adjustment when required by the metadata, and
transforms the data according to the chosen preprocessing specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd

from MIDAS import MetadataPanel, VariableMeta

HERE = Path(__file__).resolve().parents[1]
INPUT_DIR = HERE / "input"
OUTPUT_DIR = HERE / "output" / "processed"
CACHE_DIR = HERE / ".cache" / "preprocess"
LOCAL_X13 = HERE / ".cache" / "x13" / "bin" / "x13as"

MONTHLY_START = "1994-01-01"
SPEC_COLUMNS = ("spec1", "spec2", "spec3")
VALID_TRANSFORMS = {"none", "yoy", "mom", "mom_ann", "qoq_ann"}
TARGET_CANONICAL = "g_pbiq"
MONTHLY_GDP_CANONICAL = "g_pbim"
BENCHMARK_MONTHLY_GDP_LEVEL = "pbim_level"
SEED = 42


@dataclass(frozen=True)
class ProcessedPanelArtifacts:
    """A fully transformed active panel plus the preprocessing side products."""

    spec: str
    panel: MetadataPanel
    metadata_map: pd.DataFrame
    monthly_raw: pd.DataFrame
    quarterly_raw: pd.DataFrame
    monthly_sa: pd.DataFrame
    quarterly_sa: pd.DataFrame
    target_column: str
    target_transform: str
    target_level_variable: str
    monthly_gdp_column: str
    monthly_gdp_level_variable: str
    benchmark_vars: list[str]
    warnings: list[str]


def _opt_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _date_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    if out["date"].duplicated().any():
        dup = out.loc[out["date"].duplicated(), "date"].astype(str).tolist()
        raise ValueError(f"duplicate dates detected: {dup[:5]}")
    return out.set_index("date")


def _canonical_name(variable: str, transform: str) -> str:
    var = str(variable)
    if transform in {"yoy", "mom", "mom_ann", "qoq_ann"} and not var.startswith("g_"):
        return f"g_{var}"
    return var


def _group_is_financial(row: pd.Series) -> bool:
    text = " ".join(
        [
            _opt_text(row.get("group")),
            _opt_text(row.get("label")),
            _opt_text(row.get("description")),
        ]
    ).lower()
    tokens = ("rate", "yield", "spread", "exchange", "currency", "policy")
    return any(tok in text for tok in tokens)


def _load_metadata(metadata_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = INPUT_DIR / "metadata.xlsx" if metadata_path is None else Path(metadata_path)
    monthly = pd.read_excel(path, sheet_name="Monthly")
    quarterly = pd.read_excel(path, sheet_name="Quarterly")
    return monthly, quarterly


def audit_metadata(metadata_path: str | Path | None = None) -> pd.DataFrame:
    """Return a compact issue table for the metadata catalogue."""

    monthly, quarterly = _load_metadata(metadata_path)
    frames = [monthly.assign(sheet="Monthly"), quarterly.assign(sheet="Quarterly")]
    meta = pd.concat(frames, ignore_index=True)
    issues: list[dict[str, Any]] = []
    key_cols = [
        "source_code",
        "variable",
        "frequency",
        "group",
        "need_sa",
        "publication_delay_days",
        "source",
    ]

    for sheet_name, frame in (("Monthly", monthly), ("Quarterly", quarterly)):
        missing = frame[key_cols + list(SPEC_COLUMNS)].isna().sum()
        for col, count in missing.items():
            if int(count) > 0:
                issues.append(
                    {
                        "severity": "high",
                        "sheet": sheet_name,
                        "variable": "*",
                        "issue": "missing metadata",
                        "detail": f"{col}: {int(count)} missing values",
                        "recommendation": f"Fill the missing `{col}` entries before the exercise.",
                    }
                )

        for key in ("variable", "source_code"):
            dup = frame.loc[frame[key].duplicated(keep=False), key].dropna().unique().tolist()
            for value in dup:
                issues.append(
                    {
                        "severity": "high",
                        "sheet": sheet_name,
                        "variable": str(value),
                        "issue": "duplicate entry",
                        "detail": f"`{key}` appears more than once in the {sheet_name.lower()} sheet.",
                        "recommendation": f"Keep a single authoritative row per `{key}`.",
                    }
                )

    for row in meta.itertuples(index=False):
        variable = str(row.variable)
        sheet = getattr(row, "sheet")
        for spec in SPEC_COLUMNS:
            value = str(getattr(row, spec)).lower()
            if value not in VALID_TRANSFORMS:
                issues.append(
                    {
                        "severity": "high",
                        "sheet": sheet,
                        "variable": variable,
                        "issue": "unknown transformation",
                        "detail": f"{spec} = {value!r} is not one of {sorted(VALID_TRANSFORMS)}.",
                        "recommendation": "Use one of `none`, `yoy`, `mom_ann`, or `qoq_ann`.",
                    }
                )

        if int(row.publication_delay_days) < 0:
            issues.append(
                {
                    "severity": "high",
                    "sheet": sheet,
                    "variable": variable,
                    "issue": "negative publication delay",
                    "detail": f"publication_delay_days = {int(row.publication_delay_days)}.",
                    "recommendation": "Publication delays should be non-negative.",
                }
            )

        if int(row.publication_delay_days) == 0:
            issues.append(
                {
                    "severity": "low",
                    "sheet": sheet,
                    "variable": variable,
                    "issue": "zero publication delay",
                    "detail": "The release is assumed to be available immediately at period end.",
                    "recommendation": "Check whether a same-day release is really intended.",
                }
            )

        if str(row.variable).startswith("g_") and int(row.transformation_code) == 1:
            issues.append(
                {
                    "severity": "medium",
                    "sheet": sheet,
                    "variable": variable,
                    "issue": "naming inconsistency",
                    "detail": "Variable name already carries a growth prefix, but the metadata says the raw source still needs a growth transformation.",
                    "recommendation": "Consider renaming the raw variable or documenting this exception explicitly.",
                }
            )

        if _group_is_financial(pd.Series(row._asdict())) and str(row.spec1).lower() == "yoy":
            issues.append(
                {
                    "severity": "medium",
                    "sheet": sheet,
                    "variable": variable,
                    "issue": "economically questionable transformation",
                    "detail": "A year-on-year growth transform is assigned to a financial or price-like variable.",
                    "recommendation": "Check whether a level, spread, first difference, or depreciation measure is more defensible.",
                }
            )

        text = " ".join(
            [
                _opt_text(getattr(row, "label", None)),
                _opt_text(getattr(row, "description", None)),
                _opt_text(getattr(row, "notes", None)),
            ]
        ).lower()
        if "seasonally adjusted" in text and int(row.need_sa) == 1:
            issues.append(
                {
                    "severity": "medium",
                    "sheet": sheet,
                    "variable": variable,
                    "issue": "possible double seasonal adjustment",
                    "detail": "Metadata text suggests the series may already be seasonally adjusted, but `need_sa == 1`.",
                    "recommendation": "Verify whether the official source is already seasonally adjusted.",
                }
            )

    out = pd.DataFrame(issues)
    if out.empty:
        return pd.DataFrame(columns=["severity", "sheet", "variable", "issue", "detail", "recommendation"])
    sev_order = {"high": 0, "medium": 1, "low": 2}
    out["severity_order"] = out["severity"].map(sev_order).fillna(9)
    return out.sort_values(["severity_order", "sheet", "variable", "issue"]).drop(columns="severity_order").reset_index(drop=True)


def metadata_report_markdown(issues: pd.DataFrame) -> str:
    """Readable markdown summary for the notebook."""

    if issues.empty:
        return (
            "### Metadata review\n"
            "\n"
            "- No missing values or duplicated rows were detected in the core metadata columns.\n"
            "- No immediately problematic transformation or delay entries were flagged.\n"
        )

    top = issues.head(8)
    lines = [
        "### Metadata review",
        "",
        f"- The audit flagged **{len(issues)}** recommendations. The most important ones are summarized below.",
    ]
    if (issues["severity"] == "high").any():
        lines.append(
            f"- High-priority issues: **{int((issues['severity'] == 'high').sum())}**."
        )
    if (issues["severity"] == "medium").any():
        lines.append(
            f"- Medium-priority issues: **{int((issues['severity'] == 'medium').sum())}**."
        )
    lines.append("- None of these metadata issues are modified automatically in the notebook.")
    lines.append("")
    lines.append("**Main recommendations**")
    lines.append("")
    for row in top.itertuples(index=False):
        lines.append(
            f"- `{row.variable}` ({row.sheet}): {row.issue}. {row.recommendation}"
        )
    return "\n".join(lines)


def _load_fred_key() -> str:
    from dotenv import load_dotenv
    import os

    load_dotenv(HERE / ".env")
    key = os.getenv("FRED_KEY")
    if not key:
        raise RuntimeError(f"FRED_KEY not found. Add it to {HERE / '.env'}.")
    return key


def download_raw_inputs(
    metadata_path: str | Path | None = None,
    *,
    start_period: str = "1994-1",
    refresh: bool = False,
    cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download or load cached raw monthly and quarterly inputs."""

    from MacroPy import get_bcrp_data, get_fred_data

    monthly_meta, quarterly_meta = _load_metadata(metadata_path)
    cache = CACHE_DIR if cache_dir is None else Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    monthly_path = cache / "monthly_raw.parquet"
    quarterly_path = cache / "quarterly_raw.parquet"

    if monthly_path.exists() and quarterly_path.exists() and not refresh:
        return pd.read_parquet(monthly_path), pd.read_parquet(quarterly_path)

    m_bcrp = (
        monthly_meta.loc[monthly_meta["source"].eq("BCRP") & monthly_meta["active"].eq(1)]
        .dropna(subset=["source_code", "variable"])
        .set_index("source_code")["variable"]
        .to_dict()
    )
    m_fred = (
        monthly_meta.loc[monthly_meta["source"].eq("FRED") & monthly_meta["active"].eq(1)]
        .dropna(subset=["source_code", "variable"])
        .set_index("source_code")["variable"]
        .to_dict()
    )
    q_bcrp = (
        quarterly_meta.loc[quarterly_meta["source"].eq("BCRP") & quarterly_meta["active"].eq(1)]
        .dropna(subset=["source_code", "variable"])
        .set_index("source_code")["variable"]
        .to_dict()
    )

    monthly_bcrp = get_bcrp_data(m_bcrp, frequency="M", start_period=start_period)
    monthly = _date_index(monthly_bcrp)
    if m_fred:
        api_key = _load_fred_key()
        fred_codes = list(m_fred.keys())
        fred_names = list(m_fred.values())
        monthly_fred = get_fred_data(fred_codes, fred_names, "m", api_key, start_period=start_period)
        monthly = monthly.join(_date_index(monthly_fred), how="left")

    quarterly_bcrp = get_bcrp_data(q_bcrp, frequency="Q", start_period=start_period)
    quarterly = _date_index(quarterly_bcrp)

    monthly.to_parquet(monthly_path)
    quarterly.to_parquet(quarterly_path)
    return monthly, quarterly


def locate_x13_binary() -> Path:
    """Find a usable X13 binary."""

    import shutil
    import os

    candidates = [
        LOCAL_X13,
        Path("x13as"),
        Path("/opt/homebrew/bin/x13as"),
        Path("/usr/local/bin/x13as"),
    ]
    env_path = os.getenv("X13PATH")
    if env_path:
        candidates.insert(0, Path(env_path))
    found = shutil.which("x13as")
    if found:
        candidates.insert(0, Path(found))
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        "X13 binary not found. Expected one of: local DSAPM cache, PATH, /opt/homebrew/bin/x13as, /usr/local/bin/x13as."
    )


def _series_with_freq(series: pd.Series, freq: str) -> pd.Series:
    s = series.astype(float).copy()
    if freq == "M":
        return s.asfreq("MS")
    if freq == "Q":
        idx = pd.DatetimeIndex(s.index, freq="QS-MAR")
        s.index = idx
        return s
    raise ValueError(f"unsupported frequency {freq!r}")


def _official_sa_hint(row: pd.Series) -> bool:
    text = " ".join(
        [
            _opt_text(row.get("label")),
            _opt_text(row.get("description")),
            _opt_text(row.get("notes")),
        ]
    ).lower()
    return "seasonally adjusted" in text


def _seasonally_adjust_series(
    series: pd.Series,
    *,
    freq: str,
    x13_path: Path,
) -> pd.Series:
    from statsmodels.tsa.x13 import x13_arima_analysis

    s = _series_with_freq(series, freq)
    first = s.first_valid_index()
    last = s.last_valid_index()
    if first is None or last is None:
        return s
    work = s.loc[first:last].copy()
    if work.isna().any():
        raise ValueError("X13 input has internal missing values.")
    min_obs = 36 if freq == "M" else 16
    if work.notna().sum() < min_obs:
        raise ValueError(f"too few observations for X13 ({int(work.notna().sum())} < {min_obs})")
    if (work <= 0).any():
        res = x13_arima_analysis(work, x12path=str(x13_path), prefer_x13=True, log=None)
    else:
        res = x13_arima_analysis(work, x12path=str(x13_path), prefer_x13=True, log=None)
    out = s.copy()
    out.loc[res.seasadj.index] = res.seasadj.astype(float)
    return out


def seasonal_adjust_frames(
    monthly_raw: pd.DataFrame,
    quarterly_raw: pd.DataFrame,
    metadata_path: str | Path | None = None,
    *,
    refresh: bool = False,
    cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply X13 where `need_sa == 1`, otherwise keep the series unchanged."""

    monthly_meta, quarterly_meta = _load_metadata(metadata_path)
    cache = CACHE_DIR if cache_dir is None else Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    monthly_path = cache / "monthly_sa.parquet"
    quarterly_path = cache / "quarterly_sa.parquet"
    log_path = cache / "seasonal_adjustment_log.parquet"
    if monthly_path.exists() and quarterly_path.exists() and log_path.exists() and not refresh:
        return pd.read_parquet(monthly_path), pd.read_parquet(quarterly_path), pd.read_parquet(log_path)

    x13_path = locate_x13_binary()
    logs: list[dict[str, Any]] = []

    def adjust_frame(frame: pd.DataFrame, meta: pd.DataFrame, freq: str) -> pd.DataFrame:
        out = frame.copy()
        active = meta.loc[meta["active"].eq(1)].copy()
        for row in active.itertuples(index=False):
            variable = str(row.variable)
            if variable not in out.columns:
                logs.append(
                    {
                        "frequency": freq,
                        "variable": variable,
                        "action": "missing",
                        "warning": "variable absent from raw download",
                    }
                )
                continue
            if int(row.need_sa) == 0:
                logs.append({"frequency": freq, "variable": variable, "action": "kept_as_is", "warning": ""})
                continue
            row_s = pd.Series(row._asdict())
            if _official_sa_hint(row_s):
                logs.append({"frequency": freq, "variable": variable, "action": "official_sa_kept", "warning": ""})
                continue
            try:
                out[variable] = _seasonally_adjust_series(out[variable], freq=freq, x13_path=x13_path)
                logs.append({"frequency": freq, "variable": variable, "action": "x13_adjusted", "warning": ""})
            except Exception as exc:
                logs.append(
                    {
                        "frequency": freq,
                        "variable": variable,
                        "action": "fallback_raw",
                        "warning": f"{type(exc).__name__}: {exc}",
                    }
                )
        return out

    monthly_sa = adjust_frame(monthly_raw, monthly_meta, "M")
    quarterly_sa = adjust_frame(quarterly_raw, quarterly_meta, "Q")
    log = pd.DataFrame(logs)
    monthly_sa.to_parquet(monthly_path)
    quarterly_sa.to_parquet(quarterly_path)
    log.to_parquet(log_path)
    return monthly_sa, quarterly_sa, log


def _log_growth(series: pd.Series, lag: int, scale: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if (s.dropna() <= 0).any():
        warnings.warn(
            "non-positive observations detected; falling back from log growth to simple percentage growth",
            RuntimeWarning,
            stacklevel=2,
        )
        return scale * (s / s.shift(lag) - 1.0)
    return scale * (np.log(s) - np.log(s.shift(lag)))


def _apply_transform(series: pd.Series, transform: str, freq: str) -> pd.Series:
    if transform == "none":
        return pd.to_numeric(series, errors="coerce")
    if transform == "yoy":
        lag = 12 if freq == "M" else 4
        return _log_growth(series, lag, 100.0)
    if transform == "mom":
        if freq != "M":
            raise ValueError("mom is only defined for monthly series")
        return _log_growth(series, 1, 100.0)
    if transform == "mom_ann":
        if freq != "M":
            raise ValueError("mom_ann is only defined for monthly series")
        return _log_growth(series, 1, 1200.0)
    if transform == "qoq_ann":
        if freq != "Q":
            raise ValueError("qoq_ann is only defined for quarterly series")
        return _log_growth(series, 1, 400.0)
    raise ValueError(f"unknown transform {transform!r}")


def build_spec_artifacts(
    spec: str,
    *,
    metadata_path: str | Path | None = None,
    refresh_downloads: bool = False,
    refresh_sa: bool = False,
    start_period: str = "1994-1",
) -> ProcessedPanelArtifacts:
    """Rebuild the panel under one transformation specification."""

    if spec not in SPEC_COLUMNS:
        raise ValueError(f"spec must be one of {SPEC_COLUMNS}, got {spec!r}")

    monthly_meta, quarterly_meta = _load_metadata(metadata_path)
    monthly_raw, quarterly_raw = download_raw_inputs(
        metadata_path,
        start_period=start_period,
        refresh=refresh_downloads,
    )
    monthly_sa, quarterly_sa, sa_log = seasonal_adjust_frames(
        monthly_raw,
        quarterly_raw,
        metadata_path,
        refresh=refresh_sa,
    )

    warnings_list = [
        f"{row.variable}: {row.warning}"
        for row in sa_log.loc[sa_log["warning"].astype(str).str.len() > 0].itertuples(index=False)
    ]

    monthly_out = pd.DataFrame(index=monthly_sa.index)
    quarterly_out = pd.DataFrame(index=quarterly_sa.index)
    records: list[dict[str, Any]] = []

    def transform_frame(frame: pd.DataFrame, meta: pd.DataFrame, freq: str, out: pd.DataFrame) -> None:
        for row in meta.loc[meta["active"].eq(1)].itertuples(index=False):
            variable = str(row.variable)
            if variable not in frame.columns:
                warnings_list.append(f"{variable}: raw series missing from downloaded data")
                continue
            transform = str(getattr(row, spec)).lower()
            name = _canonical_name(variable, transform)
            series = frame[variable]
            if transform != "none" and (pd.to_numeric(series, errors="coerce").dropna() <= 0).any():
                warnings_list.append(
                    f"{variable}: non-positive observations forced a simple percentage-growth fallback under `{transform}`"
                )
            try:
                out[name] = _apply_transform(series, transform, freq)
            except Exception as exc:
                warnings_list.append(f"{variable}: failed {transform} transform ({type(exc).__name__}: {exc})")
                out[name] = np.nan
            records.append(
                {
                    "source_variable": variable,
                    "column": name,
                    "frequency": freq,
                    "group": str(row.group),
                    "publication_delay_days": int(row.publication_delay_days),
                    "label": _opt_text(getattr(row, "label", "")),
                    "unit": _opt_text(getattr(row, "unit", "")),
                    "transform": transform,
                    "source": str(row.source),
                    "need_sa": int(row.need_sa),
                }
            )

    transform_frame(monthly_sa, monthly_meta, "M", monthly_out)
    transform_frame(quarterly_sa, quarterly_meta, "Q", quarterly_out)

    metadata_map = pd.DataFrame(records)
    if metadata_map["column"].duplicated().any():
        dup = metadata_map.loc[metadata_map["column"].duplicated(), "column"].tolist()
        raise ValueError(f"duplicate transformed column names detected: {dup}")

    metas = [
        VariableMeta(
            column=str(row.column),
            frequency=str(row.frequency),
            group=str(row.group),
            publication_delay_days=int(row.publication_delay_days),
            label=row.label or None,
            unit=row.unit or None,
        )
        for row in metadata_map.itertuples(index=False)
    ]
    panel = MetadataPanel.from_frames(monthly_out, quarterly_out, metas)

    target_rows = metadata_map.loc[(metadata_map["frequency"] == "Q") & (metadata_map["column"] == TARGET_CANONICAL)]
    if target_rows.empty:
        raise ValueError(f"target column {TARGET_CANONICAL!r} not found under {spec}")
    target_row = target_rows.iloc[0]
    monthly_gdp_rows = metadata_map.loc[(metadata_map["frequency"] == "M") & (metadata_map["column"] == MONTHLY_GDP_CANONICAL)]
    if monthly_gdp_rows.empty:
        raise ValueError(f"monthly GDP proxy {MONTHLY_GDP_CANONICAL!r} not found under {spec}")
    monthly_gdp_row = monthly_gdp_rows.iloc[0]

    benchmark_vars = [
        MONTHLY_GDP_CANONICAL,
        *[
            c
            for c in panel.columns_by_group().get("Business surveys", [])
            if c != MONTHLY_GDP_CANONICAL
        ],
    ]
    return ProcessedPanelArtifacts(
        spec=spec,
        panel=panel,
        metadata_map=metadata_map,
        monthly_raw=monthly_raw,
        quarterly_raw=quarterly_raw,
        monthly_sa=monthly_sa,
        quarterly_sa=quarterly_sa,
        target_column=TARGET_CANONICAL,
        target_transform=str(target_row["transform"]).lower(),
        target_level_variable=str(target_row["source_variable"]),
        monthly_gdp_column=MONTHLY_GDP_CANONICAL,
        monthly_gdp_level_variable=str(monthly_gdp_row["source_variable"]),
        benchmark_vars=benchmark_vars,
        warnings=warnings_list,
    )

def build_processed_artifacts(
    *,
    spec: str = "spec3",
    metadata_path: str | Path | None = None,
    refresh_downloads: bool = False,
    refresh_sa: bool = False,
    start_period: str = "1994-1",
) -> ProcessedPanelArtifacts:
    """Named wrapper for the active preprocessing regime."""

    return build_spec_artifacts(
        spec,
        metadata_path=metadata_path,
        refresh_downloads=refresh_downloads,
        refresh_sa=refresh_sa,
        start_period=start_period,
    )


def build_benchmark_artifacts(
    artifacts: ProcessedPanelArtifacts,
    *,
    monthly_gdp_column: str = BENCHMARK_MONTHLY_GDP_LEVEL,
) -> ProcessedPanelArtifacts:
    """Build the benchmark view used by monthly-GDP proxy models.

    The richer models should keep the active transformed panel of the requested
    spec. The simple benchmark family, however, is more coherent when monthly
    GDP remains in seasonally adjusted levels and only the quarterly target is
    transformed to the requested growth measure.
    """

    monthly = pd.DataFrame(index=artifacts.monthly_sa.index)
    quarterly = pd.DataFrame(index=artifacts.quarterly_sa.index)
    records: list[dict[str, Any]] = []

    gdp_rows = artifacts.metadata_map.loc[
        (artifacts.metadata_map["frequency"] == "M")
        & (artifacts.metadata_map["source_variable"] == artifacts.monthly_gdp_level_variable)
    ]
    if gdp_rows.empty:
        raise ValueError(f"monthly GDP source {artifacts.monthly_gdp_level_variable!r} not found in the metadata map")
    gdp_row = gdp_rows.iloc[0]
    monthly[monthly_gdp_column] = pd.to_numeric(
        artifacts.monthly_sa[artifacts.monthly_gdp_level_variable],
        errors="coerce",
    )
    records.append(
        {
            "source_variable": artifacts.monthly_gdp_level_variable,
            "column": monthly_gdp_column,
            "frequency": "M",
            "group": str(gdp_row["group"]),
            "publication_delay_days": int(gdp_row["publication_delay_days"]),
            "label": _opt_text(gdp_row.get("label", "")),
            "unit": _opt_text(gdp_row.get("unit", "")),
            "transform": "level",
            "source": str(gdp_row["source"]),
            "need_sa": int(gdp_row["need_sa"]),
        }
    )

    survey_rows = artifacts.metadata_map.loc[
        (artifacts.metadata_map["frequency"] == "M")
        & (artifacts.metadata_map["group"] == "Business surveys")
    ].copy()
    for row in survey_rows.itertuples(index=False):
        src = str(row.source_variable)
        col = str(row.column)
        monthly[col] = pd.to_numeric(artifacts.monthly_sa[src], errors="coerce")
        records.append(
            {
                "source_variable": src,
                "column": col,
                "frequency": "M",
                "group": str(row.group),
                "publication_delay_days": int(row.publication_delay_days),
                "label": _opt_text(getattr(row, "label", "")),
                "unit": _opt_text(getattr(row, "unit", "")),
                "transform": "none",
                "source": str(row.source),
                "need_sa": int(row.need_sa),
            }
        )

    quarterly[artifacts.target_column] = _apply_transform(
        artifacts.quarterly_sa[artifacts.target_level_variable],
        artifacts.target_transform,
        "Q",
    )
    target_rows = artifacts.metadata_map.loc[
        (artifacts.metadata_map["frequency"] == "Q")
        & (artifacts.metadata_map["column"] == artifacts.target_column)
    ]
    if target_rows.empty:
        raise ValueError(f"target column {artifacts.target_column!r} not found in the metadata map")
    target_row = target_rows.iloc[0]
    records.append(
        {
            "source_variable": artifacts.target_level_variable,
            "column": artifacts.target_column,
            "frequency": "Q",
            "group": str(target_row["group"]),
            "publication_delay_days": int(target_row["publication_delay_days"]),
            "label": _opt_text(target_row.get("label", "")),
            "unit": _opt_text(target_row.get("unit", "")),
            "transform": artifacts.target_transform,
            "source": str(target_row["source"]),
            "need_sa": int(target_row["need_sa"]),
        }
    )

    metadata_map = pd.DataFrame(records)
    metas = [
        VariableMeta(
            column=str(row.column),
            frequency=str(row.frequency),
            group=str(row.group),
            publication_delay_days=int(row.publication_delay_days),
            label=row.label or None,
            unit=row.unit or None,
        )
        for row in metadata_map.itertuples(index=False)
    ]
    panel = MetadataPanel.from_frames(monthly, quarterly, metas)
    benchmark_vars = [monthly_gdp_column] + [str(row.column) for row in survey_rows.itertuples(index=False)]

    return ProcessedPanelArtifacts(
        spec=f"{artifacts.spec}_benchmark",
        panel=panel,
        metadata_map=metadata_map,
        monthly_raw=artifacts.monthly_raw,
        quarterly_raw=artifacts.quarterly_raw,
        monthly_sa=artifacts.monthly_sa,
        quarterly_sa=artifacts.quarterly_sa,
        target_column=artifacts.target_column,
        target_transform=artifacts.target_transform,
        target_level_variable=artifacts.target_level_variable,
        monthly_gdp_column=monthly_gdp_column,
        monthly_gdp_level_variable=artifacts.monthly_gdp_level_variable,
        benchmark_vars=benchmark_vars,
        warnings=list(artifacts.warnings),
    )


def export_processed_snapshot(
    artifacts: ProcessedPanelArtifacts,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write the processed panel artefacts to disk for downstream notebooks."""

    out_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "monthly_raw": out_dir / f"monthly_raw_{artifacts.spec}.parquet",
        "quarterly_raw": out_dir / f"quarterly_raw_{artifacts.spec}.parquet",
        "monthly_sa": out_dir / f"monthly_sa_{artifacts.spec}.parquet",
        "quarterly_sa": out_dir / f"quarterly_sa_{artifacts.spec}.parquet",
        "monthly_panel": out_dir / f"monthly_panel_{artifacts.spec}.parquet",
        "quarterly_panel": out_dir / f"quarterly_panel_{artifacts.spec}.parquet",
        "metadata_map": out_dir / f"metadata_map_{artifacts.spec}.csv",
        "warnings": out_dir / f"preprocess_warnings_{artifacts.spec}.csv",
    }
    artifacts.monthly_raw.to_parquet(paths["monthly_raw"])
    artifacts.quarterly_raw.to_parquet(paths["quarterly_raw"])
    artifacts.monthly_sa.to_parquet(paths["monthly_sa"])
    artifacts.quarterly_sa.to_parquet(paths["quarterly_sa"])
    artifacts.panel.monthly.to_parquet(paths["monthly_panel"])
    artifacts.panel.quarterly.to_parquet(paths["quarterly_panel"])
    artifacts.metadata_map.to_csv(paths["metadata_map"], index=False)
    pd.DataFrame({"warning": artifacts.warnings}).to_csv(paths["warnings"], index=False)
    return paths


__all__ = [
    "CACHE_DIR",
    "INPUT_DIR",
    "BENCHMARK_MONTHLY_GDP_LEVEL",
    "MONTHLY_GDP_CANONICAL",
    "OUTPUT_DIR",
    "ProcessedPanelArtifacts",
    "SEED",
    "SPEC_COLUMNS",
    "TARGET_CANONICAL",
    "audit_metadata",
    "build_processed_artifacts",
    "build_benchmark_artifacts",
    "build_spec_artifacts",
    "download_raw_inputs",
    "export_processed_snapshot",
    "locate_x13_binary",
    "metadata_report_markdown",
    "seasonal_adjust_frames",
]
