"""Robustness helpers for the refreshed DSAPM workflow.

The goal is to keep notebook 04 compact while still making the long-running
robustness exercises resumable and easy to audit.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from MIDAS import PALETTE, add_information_index, combine_release_cycle

from .backtest import (
    attach_relative_rmse,
    build_competition_runs,
    build_main_models,
    run_competition_backtest,
    run_yoy_backtest,
    selected_leads_metrics,
)
from .preprocess import ProcessedPanelArtifacts, build_processed_artifacts

HERE = Path(__file__).resolve().parents[1]
ROBUSTNESS_DIR = HERE / "output" / "robustness"
ROBUSTNESS_BACKTEST_DIR = ROBUSTNESS_DIR / "backtests"
ROBUSTNESS_TABLE_DIR = ROBUSTNESS_DIR / "tables"
ROBUSTNESS_FIG_DIR = HERE / "figures" / "robustness"


def ensure_robustness_dirs() -> dict[str, Path]:
    """Create the robustness output directories and return them."""

    for path in (ROBUSTNESS_DIR, ROBUSTNESS_BACKTEST_DIR, ROBUSTNESS_TABLE_DIR, ROBUSTNESS_FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": ROBUSTNESS_DIR,
        "backtests": ROBUSTNESS_BACKTEST_DIR,
        "tables": ROBUSTNESS_TABLE_DIR,
        "figures": ROBUSTNESS_FIG_DIR,
    }


def slugify(label: str) -> str:
    """Filesystem-safe label."""

    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def selected_cache_path(target: str, spec: str, label: str) -> Path:
    """Cache path for a selected-leads robustness backtest."""

    ensure_robustness_dirs()
    return ROBUSTNESS_BACKTEST_DIR / f"selected_leads_{target}_{spec}_{slugify(label)}.parquet"


def competition_cache_path(target: str, spec: str, label: str = "competition") -> Path:
    """Cache path for the coherent five-model selected-leads competition."""

    ensure_robustness_dirs()
    return ROBUSTNESS_BACKTEST_DIR / f"selected_leads_{target}_{spec}_{slugify(label)}.parquet"


def load_or_run_selected_backtest(
    *,
    artifacts: ProcessedPanelArtifacts,
    models: Mapping[str, Any],
    cache_path: str | Path,
    leads: Sequence[int],
    eval_start: str | pd.Timestamp,
    eval_end: str | pd.Timestamp | None,
    refresh: bool = False,
    min_train: int = 20,
    n_jobs: int | None = None,
    verbose: bool = True,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    """Load a cached selected-leads run or build it."""

    path = Path(cache_path)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    bt = run_yoy_backtest(
        artifacts,
        models=models,
        leads=leads,
        eval_start=eval_start,
        eval_end=eval_end,
        min_train=min_train,
        n_jobs=n_jobs,
        verbose=verbose,
        progress_desc=progress_desc,
    )
    bt.to_parquet(path)
    return bt


def load_or_run_competition_backtest(
    *,
    artifacts: ProcessedPanelArtifacts,
    cache_path: str | Path,
    leads: Sequence[int],
    eval_start: str | pd.Timestamp,
    eval_end: str | pd.Timestamp | None,
    refresh: bool = False,
    min_train: int = 20,
    n_jobs: int | None = None,
    verbose: bool = True,
    progress_desc: str | None = None,
    dfm_factors: int = 3,
    dfm_factor_orders: int = 2,
    dfm_maxiter: int = 200,
    dfm_tolerance: float = 1e-6,
    dfm_suppress_fit_warnings: bool = True,
    sglasso_n_lambda: int = 20,
    covid_window: tuple[str, str] | None = None,
    covid_quarters: Sequence[str] | None = None,
    simple_exclude_month_window: tuple[str, str] | None = None,
    simple_exclude_quarter_window: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Load or run the spec-aware benchmark plus main-model competition."""

    path = Path(cache_path)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    bt = run_competition_backtest(
        artifacts,
        leads=leads,
        eval_start=eval_start,
        eval_end=eval_end,
        min_train=min_train,
        n_jobs=n_jobs,
        verbose=verbose,
        progress_desc=progress_desc,
        dfm_factors=dfm_factors,
        dfm_factor_orders=dfm_factor_orders,
        dfm_maxiter=dfm_maxiter,
        dfm_tolerance=dfm_tolerance,
        dfm_suppress_fit_warnings=dfm_suppress_fit_warnings,
        sglasso_n_lambda=sglasso_n_lambda,
        covid_window=covid_window,
        covid_quarters=covid_quarters,
        simple_exclude_month_window=simple_exclude_month_window,
        simple_exclude_quarter_window=simple_exclude_quarter_window,
    )
    bt.to_parquet(path)
    return bt


def refresh_competition_models(
    *,
    artifacts: ProcessedPanelArtifacts,
    cache_path: str | Path,
    model_names: Sequence[str],
    leads: Sequence[int],
    eval_start: str | pd.Timestamp,
    eval_end: str | pd.Timestamp | None,
    min_train: int = 20,
    n_jobs: int | None = None,
    verbose: bool = True,
    progress_desc: str | None = None,
    dfm_factors: int = 3,
    dfm_factor_orders: int = 2,
    dfm_maxiter: int = 200,
    dfm_tolerance: float = 1e-6,
    dfm_suppress_fit_warnings: bool = True,
    sglasso_n_lambda: int = 20,
    covid_window: tuple[str, str] | None = None,
    covid_quarters: Sequence[str] | None = None,
    simple_exclude_month_window: tuple[str, str] | None = None,
    simple_exclude_quarter_window: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Refresh only the requested competition models inside an existing cache.

    This is meant for targeted repairs such as rerunning only the corrected DFM
    rows after a bug fix, without touching the benchmark or sparse-model rows.
    """

    path = Path(cache_path)
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    wanted = list(dict.fromkeys(str(name) for name in model_names))
    refreshed_parts: list[pd.DataFrame] = []

    bundles = build_competition_runs(
        artifacts=artifacts,
        dfm_factors=dfm_factors,
        dfm_factor_orders=dfm_factor_orders,
        dfm_maxiter=dfm_maxiter,
        dfm_tolerance=dfm_tolerance,
        dfm_suppress_fit_warnings=dfm_suppress_fit_warnings,
        sglasso_n_lambda=sglasso_n_lambda,
        covid_window=covid_window,
        covid_quarters=covid_quarters,
        simple_exclude_month_window=simple_exclude_month_window,
        simple_exclude_quarter_window=simple_exclude_quarter_window,
    )
    for bundle in bundles:
        subset = {name: model for name, model in bundle["models"].items() if name in wanted}
        if not subset:
            continue
        rerun = run_yoy_backtest(
                bundle["artifacts"],
                models=subset,
                leads=leads,
                eval_start=eval_start,
                eval_end=eval_end,
                min_train=min_train,
                n_jobs=n_jobs,
                verbose=verbose,
                progress_desc=progress_desc,
            )
        if not rerun.empty:
            rerun["panel_view"] = bundle["label"]
        refreshed_parts.append(rerun)

    refreshed = pd.concat(refreshed_parts, ignore_index=True) if refreshed_parts else pd.DataFrame()
    if existing.empty:
        out = refreshed
    else:
        out = pd.concat(
            [existing.loc[~existing["model"].isin(wanted)].copy(), refreshed],
            ignore_index=True,
            sort=False,
        )
    if not out.empty:
        out = out.sort_values(["ref_quarter", "days_to_publication", "model"]).reset_index(drop=True)
    out.to_parquet(path)
    return out


def load_artifacts_for_spec(
    *,
    spec: str,
    metadata_path: str | Path,
    start_period: str,
    refresh_downloads: bool = False,
    refresh_sa: bool = False,
) -> ProcessedPanelArtifacts:
    """Thin wrapper used by the robustness notebook for spec-level runs."""

    return build_processed_artifacts(
        spec=spec,
        metadata_path=metadata_path,
        refresh_downloads=refresh_downloads,
        refresh_sa=refresh_sa,
        start_period=start_period,
    )


def metrics_for_run(
    df: pd.DataFrame,
    *,
    baseline: str,
    label: str,
    exclude_years: Sequence[int] = (),
) -> pd.DataFrame:
    """Selected-leads metrics plus relative RMSE and a run label."""

    metrics = attach_relative_rmse(
        selected_leads_metrics(df, exclude_years=exclude_years),
        baseline=baseline,
    )
    metrics["run_label"] = label
    return metrics


def collect_run_metrics(
    runs: Mapping[str, pd.DataFrame],
    *,
    baseline: str,
    exclude_years: Sequence[int] = (),
) -> pd.DataFrame:
    """Metrics for several cached runs keyed by label."""

    parts = [
        metrics_for_run(frame, baseline=baseline, label=label, exclude_years=exclude_years)
        for label, frame in runs.items()
    ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def attach_relative_to_reference(
    metrics: pd.DataFrame,
    *,
    reference_model: str,
    metric: str = "rmse",
) -> pd.DataFrame:
    """Relative metric against ``reference_model`` within each lead and run."""

    out = metrics.copy()
    num = out[out["days_to_publication"] != "overall"].copy()
    ref = (
        num[num["model"] == reference_model][["run_label", "days_to_publication", metric]]
        .rename(columns={metric: f"reference_{metric}"})
    )
    num = num.merge(ref, on=["run_label", "days_to_publication"], how="left")
    num[f"relative_{metric}_to_ref"] = num[metric] / num[f"reference_{metric}"]

    overall = out[out["days_to_publication"] == "overall"].copy()
    ref_all = (
        overall[overall["model"] == reference_model][["run_label", metric]]
        .rename(columns={metric: f"reference_{metric}"})
    )
    overall = overall.merge(ref_all, on="run_label", how="left")
    overall[f"relative_{metric}_to_ref"] = overall[metric] / overall[f"reference_{metric}"]
    return pd.concat([overall, num], ignore_index=True)


def build_dfm_family_models(
    *,
    benchmark_vars: Sequence[str],
    monthly_gdp: str,
    dfm_factors: int,
    dfm_factor_orders: int,
    dfm_maxiter: int,
    dfm_tolerance: float,
    dfm_suppress_fit_warnings: bool,
    sglasso_n_lambda: int,
    covid_window: tuple[str, str] | None,
    covid_quarters: Sequence[str] | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Anchors plus the baseline DFM and three alternative DFM specifications."""

    anchors = build_main_models(
        benchmark_vars=benchmark_vars,
        monthly_gdp=monthly_gdp,
        dfm_factors=dfm_factors,
        dfm_factor_orders=dfm_factor_orders,
        dfm_maxiter=dfm_maxiter,
        dfm_tolerance=dfm_tolerance,
        dfm_suppress_fit_warnings=dfm_suppress_fit_warnings,
        sglasso_n_lambda=sglasso_n_lambda,
        covid_window=covid_window,
        covid_quarters=covid_quarters,
    )
    models = {k: anchors[k] for k in ("Q-AR", "M-AR", "Bridge")}
    specs = [
        {"model": "DFM", "label": "baseline", "family": "dfm", "dfm_kwargs": {"name": "DFM"}},
        {"model": "DFM-FO1", "label": "factor order 1", "family": "dfm", "dfm_kwargs": {"factor_orders": 1, "name": "DFM-FO1"}},
        {"model": "DFM-F2", "label": "2 factors", "family": "dfm", "dfm_kwargs": {"factors": 2, "name": "DFM-F2"}},
        {"model": "DFM-Groups", "label": "group factors", "family": "dfm", "dfm_kwargs": {"factors": "groups", "name": "DFM-Groups"}},
    ]
    for spec in specs:
        models[spec["model"]] = build_main_models(
            benchmark_vars=benchmark_vars,
            monthly_gdp=monthly_gdp,
            dfm_factors=dfm_factors,
            dfm_factor_orders=dfm_factor_orders,
            dfm_maxiter=dfm_maxiter,
            dfm_tolerance=dfm_tolerance,
            dfm_suppress_fit_warnings=dfm_suppress_fit_warnings,
            dfm_kwargs=spec["dfm_kwargs"],
            sglasso_n_lambda=sglasso_n_lambda,
            covid_window=covid_window,
            covid_quarters=covid_quarters,
        )["DFM"]
    return models, pd.DataFrame(specs)


def build_sglasso_family_models(
    *,
    benchmark_vars: Sequence[str],
    monthly_gdp: str,
    dfm_factors: int,
    dfm_factor_orders: int,
    dfm_maxiter: int,
    dfm_tolerance: float,
    dfm_suppress_fit_warnings: bool,
    sglasso_n_lambda: int,
    covid_window: tuple[str, str] | None,
    covid_quarters: Sequence[str] | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Anchors plus the baseline sg-LASSO and three alternative specifications."""

    anchors = build_main_models(
        benchmark_vars=benchmark_vars,
        monthly_gdp=monthly_gdp,
        dfm_factors=dfm_factors,
        dfm_factor_orders=dfm_factor_orders,
        dfm_maxiter=dfm_maxiter,
        dfm_tolerance=dfm_tolerance,
        dfm_suppress_fit_warnings=dfm_suppress_fit_warnings,
        sglasso_n_lambda=sglasso_n_lambda,
        covid_window=covid_window,
        covid_quarters=covid_quarters,
    )
    models = {k: anchors[k] for k in ("Q-AR", "M-AR", "Bridge")}
    specs = [
        {"model": "sg-LASSO", "label": "baseline", "family": "sglasso", "sglasso_kwargs": {"name": "sg-LASSO"}},
        {"model": "sg-L24", "label": "24 lags", "family": "sglasso", "sglasso_kwargs": {"n_lags": 24, "name": "sg-L24"}},
        {"model": "sg-D3", "label": "degree 3", "family": "sglasso", "sglasso_kwargs": {"degree": 3, "name": "sg-D3"}},
        {"model": "sg-LS", "label": "squared loss", "family": "sglasso", "sglasso_kwargs": {"loss": "squared", "name": "sg-LS"}},
    ]
    for spec in specs:
        models[spec["model"]] = build_main_models(
            benchmark_vars=benchmark_vars,
            monthly_gdp=monthly_gdp,
            dfm_factors=dfm_factors,
            dfm_factor_orders=dfm_factor_orders,
            dfm_maxiter=dfm_maxiter,
            dfm_tolerance=dfm_tolerance,
            dfm_suppress_fit_warnings=dfm_suppress_fit_warnings,
            sglasso_n_lambda=sglasso_n_lambda,
            sglasso_kwargs=spec["sglasso_kwargs"],
            covid_window=covid_window,
            covid_quarters=covid_quarters,
        )["sg-LASSO"]
    return models, pd.DataFrame(specs)


def run_adaptive_grid(
    rc: pd.DataFrame,
    panel,
    *,
    variants: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute a family of Adaptive-IC combinations from one release-cycle file."""

    base = rc.copy()
    if "info_index" not in base.columns:
        base = add_information_index(base, panel)

    combo_rows: list[pd.DataFrame] = []
    weight_rows: list[pd.DataFrame] = []
    variant_rows: list[dict[str, Any]] = []
    for spec in variants:
        combo, weights = combine_release_cycle(
            base,
            spec["members"],
            index_col=spec.get("index_col", "info_index"),
            n_bins=int(spec.get("n_bins", 4)),
            min_train=int(spec.get("min_train", 6)),
            method=str(spec.get("method", "inv_mse")),
            power=float(spec.get("power", 1.0)),
            name=str(spec["name"]),
        )
        combo["variant_family"] = str(spec.get("family", "adaptive"))
        combo_rows.append(combo)
        weights["model"] = str(spec["name"])
        weights["variant_family"] = str(spec.get("family", "adaptive"))
        weight_rows.append(weights)
        variant_rows.append(
            {
                "model": str(spec["name"]),
                "family": str(spec.get("family", "adaptive")),
                "label": str(spec.get("label", spec["name"])),
                "method": str(spec.get("method", "inv_mse")),
                "n_bins": int(spec.get("n_bins", 4)),
                "members": " + ".join(spec["members"]),
            }
        )
    combos = pd.concat(combo_rows, ignore_index=True) if combo_rows else pd.DataFrame()
    weights = pd.concat(weight_rows, ignore_index=True) if weight_rows else pd.DataFrame()
    variants_df = pd.DataFrame(variant_rows)
    return base, combos, weights, variants_df


def adaptive_variant_specs() -> list[dict[str, Any]]:
    """Default Adaptive-IC robustness grid."""

    return [
        {
            "name": "Adaptive-IC",
            "label": "baseline",
            "family": "baseline",
            "members": ["DFM", "Bridge", "sg-LASSO"],
            "n_bins": 4,
            "method": "inv_mse",
        },
        {
            "name": "Adaptive-IC-B3",
            "label": "3 bins",
            "family": "bins",
            "members": ["DFM", "Bridge", "sg-LASSO"],
            "n_bins": 3,
            "method": "inv_mse",
        },
        {
            "name": "Adaptive-IC-B5",
            "label": "5 bins",
            "family": "bins",
            "members": ["DFM", "Bridge", "sg-LASSO"],
            "n_bins": 5,
            "method": "inv_mse",
        },
        {
            "name": "Adaptive-IC-EQ",
            "label": "equal weights",
            "family": "method",
            "members": ["DFM", "Bridge", "sg-LASSO"],
            "n_bins": 4,
            "method": "equal",
        },
        {
            "name": "Adaptive-IC-BEST",
            "label": "hard best model",
            "family": "method",
            "members": ["DFM", "Bridge", "sg-LASSO"],
            "n_bins": 4,
            "method": "best",
        },
        {
            "name": "Adaptive-IC-noBridge",
            "label": "drop Bridge",
            "family": "members",
            "members": ["DFM", "sg-LASSO"],
            "n_bins": 4,
            "method": "inv_mse",
        },
        {
            "name": "Adaptive-IC-noDFM",
            "label": "drop DFM",
            "family": "members",
            "members": ["Bridge", "sg-LASSO"],
            "n_bins": 4,
            "method": "inv_mse",
        },
        {
            "name": "Adaptive-IC-noSG",
            "label": "drop sg-LASSO",
            "family": "members",
            "members": ["DFM", "Bridge"],
            "n_bins": 4,
            "method": "inv_mse",
        },
    ]


def plot_relative_paths(
    metrics: pd.DataFrame,
    *,
    reference_model: str,
    ax=None,
    title: str = "",
    models: Sequence[str] | None = None,
    column: str = "relative_rmse_to_ref",
    legend_ncol: int = 2,
):
    """Simple line plot for variant paths across selected leads."""

    import matplotlib.pyplot as plt

    data = metrics.loc[metrics["days_to_publication"] != "overall"].copy()
    if models is not None:
        data = data[data["model"].isin(list(models))]
    data = data[data["model"] != reference_model].copy()
    order = sorted(data["days_to_publication"].unique())
    if ax is None:
        _, ax = plt.subplots(figsize=(7.8, 4.5))

    palette = [
        PALETTE.get("myblue", "#29466E"),
        PALETTE.get("mygreen", "#1f9e89"),
        PALETTE.get("myred", "#c0392b"),
        PALETTE.get("myyellow", "#B48A2C"),
        "#6c757d",
        "#7b6fd0",
    ]
    for i, model in enumerate(dict.fromkeys(data["model"])):
        s = data[data["model"] == model].sort_values("days_to_publication")
        ax.plot(
            s["days_to_publication"],
            s[column],
            marker="o",
            lw=2,
            color=palette[i % len(palette)],
            label=model,
        )
    ax.axhline(1.0, color="black", lw=1, ls="--")
    ax.set_xticks(order)
    ax.set_ylabel("relative RMSE")
    ax.set_xlabel("days to publication")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, ncol=legend_ncol)
    return ax


__all__ = [
    "ROBUSTNESS_BACKTEST_DIR",
    "ROBUSTNESS_DIR",
    "ROBUSTNESS_FIG_DIR",
    "ROBUSTNESS_TABLE_DIR",
    "adaptive_variant_specs",
    "attach_relative_to_reference",
    "build_dfm_family_models",
    "build_sglasso_family_models",
    "competition_cache_path",
    "collect_run_metrics",
    "ensure_robustness_dirs",
    "load_artifacts_for_spec",
    "refresh_competition_models",
    "load_or_run_competition_backtest",
    "load_or_run_selected_backtest",
    "metrics_for_run",
    "plot_relative_paths",
    "run_adaptive_grid",
    "selected_cache_path",
    "slugify",
]
