"""Recursive standardization + PCA factor construction (target-agnostic).

Pure functions over pandas objects: no data retrieval and no models. Grew out of
the China satellite (validation, recursive real-time standardization,
monthly-to-quarterly aggregation, and PCA activity/property/inflation factors);
the block conveniences default to the China column names but every function takes
explicit column lists.

Conventions
-----------
- Monthly and quarterly frames are indexed by a ``DatetimeIndex`` (monthly = month start,
  quarterly = quarter start), matching the China data notebook. A ``PeriodIndex`` is also
  accepted and normalised internally.
- Monthly columns use the canonical names in ``ACTIVITY_VARS`` / ``PROPERTY_VARS`` /
  ``INFLATION_VARS``. The data notebook renames the retrieved series to these names; the
  NBS deep-history panel is mapped with ``NBS_HISTORY_RENAME``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------------------
# Canonical monthly variable groups (factor blocks).
# --------------------------------------------------------------------------------------
ACTIVITY_VARS: list[str] = [
    "ip", "retail", "electricity", "pmi", "exports", "imports",
    "gdp_monthly", "cli", "business_confidence",
]
PROPERTY_VARS: list[str] = ["fai", "property_investment", "property_sales"]
INFLATION_VARS: list[str] = ["cpi", "ppi", "m2"]

# Map the committed NBS deep-history panel (input/nbs_monthly_history.csv) column names
# to the canonical names used here. CPI is taken from OECD (a clean percent YoY), not the
# NBS index level, so it is intentionally not mapped from the NBS panel.
NBS_HISTORY_RENAME: dict[str, str] = {
    "ip_yoy": "ip",
    "retail_sales_yoy": "retail",
    "electricity_output_yoy": "electricity",
    "manufacturing_pmi": "pmi",
    "non_manufacturing_pmi": "non_mfg_pmi",
    "fixed_asset_investment_cum_yoy": "fai",
    "real_estate_investment_cum_yoy": "property_investment",
    "property_sales_floor_cum_yoy": "property_sales",
    "ppi_index_yoy": "ppi",
    "m2_yoy": "m2",
}


# --------------------------------------------------------------------------------------
# Index helpers
# --------------------------------------------------------------------------------------
def _infer_freq(index: pd.Index) -> str | None:
    if isinstance(index, pd.PeriodIndex):
        return index.freqstr
    try:
        return pd.infer_freq(index)
    except (ValueError, TypeError):
        return None


def _to_period_index(index: pd.Index, freq: str) -> pd.PeriodIndex:
    if isinstance(index, pd.PeriodIndex):
        return index.asfreq(freq)
    return pd.DatetimeIndex(index).to_period(freq)


# --------------------------------------------------------------------------------------
# 1. Validation
# --------------------------------------------------------------------------------------
def validate_china_data(monthly: pd.DataFrame, quarterly: pd.DataFrame) -> pd.DataFrame:
    """Validate the monthly and quarterly China panels.

    Checks duplicated dates, missing values, sample start/end, and frequency per dataset,
    and returns a small per-variable summary DataFrame. Does not print anything.
    """
    rows: list[dict] = []
    for dataset, df, expected in (("monthly", monthly, "M"), ("quarterly", quarterly, "Q")):
        index = df.index
        n_duplicates = int(pd.Index(index).duplicated().sum())
        inferred = _infer_freq(index)
        freq_ok = inferred is not None and inferred.upper().lstrip("-").startswith(expected)
        for column in df.columns:
            series = df[column]
            observed = series.dropna()
            rows.append({
                "dataset": dataset,
                "variable": column,
                "freq": inferred,
                "freq_ok": bool(freq_ok),
                "n_obs": int(series.notna().sum()),
                "n_missing": int(series.isna().sum()),
                "pct_missing": round(float(series.isna().mean()), 4),
                "start": None if observed.empty else observed.index.min(),
                "end": None if observed.empty else observed.index.max(),
                "duplicate_dates": n_duplicates,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 1b. Chinese January (Spring Festival) gap
# --------------------------------------------------------------------------------------
def fill_china_january_gap(
    monthly: pd.DataFrame,
    interpolate_residual: bool = True,
) -> pd.DataFrame:
    """Fill the systematic January gap in Chinese monthly indicators.

    NBS does not publish separate January figures for most hard indicators (industrial
    production, retail, fixed-asset and real-estate investment, property sales,
    electricity): January and February are reported together to remove the moving Spring
    Festival. The January cell is therefore missing while February carries the combined
    reading. This sets ``January := February`` (the official combined value) and, if
    ``interpolate_residual``, time-interpolates any remaining *internal* gaps so a dense
    monthly panel can feed MIDAS-style lag structures. Trailing (ragged-edge) gaps are
    left untouched.

    Real-time caveat: the Jan-Feb combined figure is released together in mid-March, so a
    scalar per-variable publication delay makes the filled January value look available
    about two weeks early at an end-February origin (only Q1, only that vintage). A dated
    release calendar would remove this; the effect is negligible for pre-COVID h=0 work.
    """
    out = monthly.copy()
    is_jan = out.index.month == 1
    february = out[out.index.month == 2].copy()
    february.index = february.index - pd.DateOffset(months=1)  # move Feb onto the Jan slot
    for column in out.columns:
        jan_missing = is_jan & out[column].isna()
        out.loc[jan_missing, column] = february[column].reindex(out.index)[jan_missing]
    if interpolate_residual:
        out = out.interpolate(method="time", limit_area="inside")
    return out


# --------------------------------------------------------------------------------------
# 2. Recursive standardization (expanding, no look-ahead)
# --------------------------------------------------------------------------------------
def recursive_standardize(
    df: pd.DataFrame | pd.Series,
    min_periods: int = 2,
) -> pd.DataFrame | pd.Series:
    """Standardize each column using only past-and-current information.

    At time ``t`` the value is ``(x_t - mean(x_1..x_t)) / std(x_1..x_t)`` where the mean
    and standard deviation expand over observed values up to ``t``. This has no look-ahead
    bias: no future observation enters the scaling at ``t``. The first ``min_periods``
    observations of each series (and any period with zero dispersion) are ``NaN``.
    """
    is_series = isinstance(df, pd.Series)
    frame = df.to_frame() if is_series else df
    mean = frame.expanding(min_periods=min_periods).mean()
    std = frame.expanding(min_periods=min_periods).std()  # ddof=1
    std = std.where(std > 0)  # zero dispersion -> NaN, avoids divide-by-zero
    standardized = (frame - mean) / std
    if is_series:
        return standardized.iloc[:, 0].rename(df.name)
    return standardized


# --------------------------------------------------------------------------------------
# 3. Monthly -> quarterly aggregation
# --------------------------------------------------------------------------------------
def aggregate_monthly_to_quarterly(
    monthly: pd.DataFrame | pd.Series,
    method_map: dict[str, str] | None = None,
    default_method: str = "mean",
) -> pd.DataFrame:
    """Aggregate a monthly frame to quarterly frequency.

    ``method_map`` maps a column name to one of ``"mean"``, ``"last"``, or ``"sum"``.
    Columns without an entry use ``default_method``. ``"last"`` takes the last observed
    (non-missing) value in the quarter; ``"sum"`` returns ``NaN`` for an all-missing
    quarter. The result is indexed by quarter-start timestamps.
    """
    if isinstance(monthly, pd.Series):
        monthly = monthly.to_frame()
    method_map = dict(method_map or {})
    quarters = _to_period_index(monthly.index, "M").asfreq("Q")

    aggregated: dict[str, pd.Series] = {}
    for column in monthly.columns:
        method = method_map.get(column, default_method)
        grouped = monthly[column].groupby(quarters)
        if method == "mean":
            aggregated[column] = grouped.mean()
        elif method == "sum":
            aggregated[column] = grouped.sum(min_count=1)
        elif method == "last":
            aggregated[column] = grouped.apply(
                lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan
            )
        else:
            raise ValueError(
                f"Unsupported aggregation method '{method}' for column '{column}'. "
                "Use 'mean', 'last', or 'sum'."
            )

    result = pd.DataFrame(aggregated)
    result.index = result.index.to_timestamp(how="start")
    result.index.name = monthly.index.name
    return result.sort_index()


# --------------------------------------------------------------------------------------
# 4. PCA factors
# --------------------------------------------------------------------------------------
class FactorResult(NamedTuple):
    """One PCA factor. Unpacks as ``factor, loadings, explained_variance`` and also
    exposes the same names as attributes."""

    factor: pd.Series
    loadings: pd.Series
    explained_variance: float


def _pca_factor(
    monthly: pd.DataFrame,
    variables: list[str],
    name: str,
    sign_reference: str,
    *,
    min_obs_frac: float = 0.5,
    std_min_periods: int = 2,
) -> FactorResult:
    """Build a one-component PCA factor from recursively standardized inputs.

    Only the ``variables`` present in ``monthly`` are used. Inputs are recursively
    standardized (no look-ahead), rows are kept when at least ``min_obs_frac`` of the
    variables are observed, and remaining gaps are filled with the standardized mean
    (zero) so the ragged edge is preserved. The factor sign is fixed so it correlates
    positively with ``sign_reference``.
    """
    present = [c for c in variables if c in monthly.columns]
    standardized = recursive_standardize(monthly[present], min_periods=std_min_periods)
    standardized = standardized.dropna(axis=1, how="all")
    columns = list(standardized.columns)
    if len(columns) < 2:
        raise ValueError(
            f"Factor '{name}' needs at least 2 observed variables from {variables}; "
            f"found {columns}."
        )

    min_count = max(2, int(np.ceil(min_obs_frac * len(columns))))
    enough = standardized.notna().sum(axis=1) >= min_count
    standardized = standardized.loc[enough]
    if standardized.empty:
        raise ValueError(
            f"Factor '{name}': no periods with at least {min_count} observed variables."
        )

    filled = standardized.fillna(0.0)
    pca = PCA(n_components=1)
    scores = pca.fit_transform(filled.values)[:, 0]
    factor = pd.Series(scores, index=standardized.index, name=name)
    loadings = pd.Series(pca.components_[0], index=columns, name="loading")
    explained_variance = float(pca.explained_variance_ratio_[0])

    # Deterministic sign: positive correlation with the reference variable.
    if sign_reference in standardized.columns:
        reference = standardized[sign_reference]
        mask = reference.notna()
        if mask.sum() >= 2:
            corr = np.corrcoef(factor[mask].to_numpy(), reference[mask].to_numpy())[0, 1]
            if np.isfinite(corr) and corr < 0:
                factor = -factor
                loadings = -loadings

    return FactorResult(factor=factor, loadings=loadings, explained_variance=explained_variance)


def build_activity_factor(monthly: pd.DataFrame, **kwargs) -> FactorResult:
    """Activity factor (IP, retail, electricity, PMI, trade, monthly GDP, CLI, confidence)."""
    return _pca_factor(monthly, ACTIVITY_VARS, "activity", "ip", **kwargs)


def build_property_factor(monthly: pd.DataFrame, **kwargs) -> FactorResult:
    """Property factor (fixed-asset investment, real-estate investment, property sales)."""
    return _pca_factor(monthly, PROPERTY_VARS, "property", "fai", **kwargs)


def build_inflation_factor(monthly: pd.DataFrame, **kwargs) -> FactorResult:
    """Inflation factor (CPI, PPI, M2). Sign anchored to CPI (no IP/FAI in this block)."""
    return _pca_factor(monthly, INFLATION_VARS, "inflation", "cpi", **kwargs)


def build_all_factors(
    monthly: pd.DataFrame,
    **kwargs,
) -> tuple[pd.DataFrame, dict[str, FactorResult]]:
    """Convenience: build all three factors. Returns (monthly factor frame, results dict)."""
    results = {
        "activity": build_activity_factor(monthly, **kwargs),
        "property": build_property_factor(monthly, **kwargs),
        "inflation": build_inflation_factor(monthly, **kwargs),
    }
    factors = pd.DataFrame({key: result.factor for key, result in results.items()})
    return factors.sort_index(), results


# --------------------------------------------------------------------------------------
# 5. Quarterly factors
# --------------------------------------------------------------------------------------
def aggregate_factors_to_quarterly(factors_monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the monthly factors to quarterly frequency using quarterly means."""
    return aggregate_monthly_to_quarterly(factors_monthly, default_method="mean")
