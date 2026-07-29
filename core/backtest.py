"""Active DSAPM backtest helpers.

These helpers keep the real-time evaluation logic local to the DSAPM project
while reusing the estimators implemented in :mod:`MIDAS`.
"""

from __future__ import annotations

from copy import deepcopy
import gc
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

from MIDAS import (
    ARNowcaster,
    DFMNowcaster,
    InformationSet,
    MetadataPanel,
    MonthlyLevelARNowcaster,
    MonthlyARNowcaster,
    PALETTE,
    PooledLevelBridgeNowcaster,
    PooledBridgeNowcaster,
    RealtimeEngine,
    SparseMIDASNowcaster,
    publication_date,
)
from MIDAS.base import BaseNowcaster

from .preprocess import ProcessedPanelArtifacts, build_benchmark_artifacts

HERE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = HERE / "output" / "backtests"


def _progress_bar(iterable, *, enabled: bool, total: int, desc: str):
    """Return a tqdm progress bar when available, else the raw iterable."""

    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, leave=False, dynamic_ncols=True)
    except Exception:
        return iterable


def _information_signature(info: InformationSet) -> tuple[int, int]:
    """Compact signature of the released information available at an origin.

    In this project the real-time masking only changes through newly released
    observations, so the pair of non-missing cell counts is enough to detect
    when the information set has genuinely changed as the origin moves forward.
    """

    return (
        int(info.monthly.notna().to_numpy().sum()),
        int(info.quarterly.notna().to_numpy().sum()),
    )


def _resolve_n_jobs(
    requested: int | None,
    *,
    n_quarters: int,
    n_leads: int,
) -> int:
    """Choose a sensible worker count for the backtest.

    Dense release-cycle sweeps are expensive enough to benefit from process
    parallelism, while short selected-lead exercises are usually faster in a
    single process.
    """

    total_windows = n_quarters * max(1, n_leads)
    max_workers = max(1, min(os.cpu_count() or 1, n_quarters))
    small_job = total_windows < 500 or n_leads <= 8
    medium_job = total_windows < 1500

    if requested == -1:
        if small_job or max_workers == 1:
            return 1
        if medium_job:
            return min(2, max_workers)
        return max_workers

    if requested is not None:
        req = max(1, min(int(requested), max_workers))
        return req

    if small_job or max_workers == 1:
        return 1
    if medium_job:
        return min(2, max_workers)
    return max_workers


def _run_yoy_backtest_quarter(
    quarter: pd.Timestamp,
    *,
    panel: MetadataPanel,
    target: str,
    level_sa: pd.Series,
    target_transform: str,
    monthly_gdp_column: str,
    spec: str,
    models: Mapping[str, BaseNowcaster],
    leads: Sequence[int],
    min_train: int,
    reuse_models_within_quarter: bool,
) -> tuple[list[dict[str, Any]], dict[str, int], int, int]:
    """Evaluate one quarter across all requested leads.

    The expensive part is model refitting. Along a daily release-cycle grid many
    adjacent dates share the same information set, so we only refit at those
    changepoints and reuse the cached predictions in between.
    """

    engine = RealtimeEngine(panel)
    delay = panel.delay_of(target)
    pub = publication_date(quarter, delay)
    sorted_leads = sorted(int(lead) for lead in leads)
    state = {name: deepcopy(proto) for name, proto in models.items()} if reuse_models_within_quarter else {}

    rows: list[dict[str, Any]] = []
    failures: dict[str, int] = {}
    attempted_windows = 0
    refit_events = 0
    last_signature: tuple[int, int] | None = None
    cached_rows: dict[str, dict[str, Any]] = {}

    y_true_raw = float(panel.quarterly.at[quarter, target])
    y_true = actual_target_yoy(quarter, level_sa=level_sa)

    for lead in sorted_leads:
        origin = pub + pd.Timedelta(days=int(lead))
        info = engine.information_set(origin, target, target_period=quarter)
        if pd.notna(info.quarterly.at[quarter, target]):
            continue
        if info.observed_quarters().size < min_train:
            continue

        attempted_windows += 1
        signature = _information_signature(info)
        if signature != last_signature or not cached_rows:
            std_info, m_scalers, _ = recursive_standardize(info, target=target)
            current_rows: dict[str, dict[str, Any]] = {}
            for name, proto in models.items():
                model = state[name] if reuse_models_within_quarter else deepcopy(proto)
                model_info = info if getattr(model, "requires_unstandardized_info", False) else std_info
                try:
                    res = model.fit(model_info).nowcast(model_info)
                    y_hat_raw = float(res.mean)
                    if isinstance(model, (MonthlyARNowcaster, PooledBridgeNowcaster)):
                        y_hat_raw = unscale_proxy_prediction(y_hat_raw, m_scalers.get(monthly_gdp_column))
                    y_hat = convert_prediction_to_yoy(
                        y_hat_raw,
                        quarter=quarter,
                        target_transform=target_transform,
                        level_sa=level_sa,
                    )
                    extra = dict(res.extra or {})
                except Exception as exc:
                    y_hat_raw, y_hat = float("nan"), float("nan")
                    extra = {"error": f"{type(exc).__name__}: {exc}"}
                    failures[name] = failures.get(name, 0) + 1

                row = {
                    "spec": spec,
                    "target": target,
                    "target_transform": target_transform,
                    "y_true_raw": y_true_raw,
                    "y_hat_raw": y_hat_raw,
                    "y_true": y_true,
                    "y_hat": y_hat,
                }
                for key, value in extra.items():
                    if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                        row[str(key)] = value
                current_rows[name] = row

            cached_rows = current_rows
            last_signature = signature
            refit_events += 1

        for name, cached in cached_rows.items():
            rows.append(
                {
                    **cached,
                    "ref_quarter": pd.Timestamp(quarter),
                    "origin_date": pd.Timestamp(origin),
                    "days_to_publication": int(lead),
                    "model": name,
                }
            )

    return rows, failures, attempted_windows, refit_events


def build_main_models(
    *,
    benchmark_vars: Sequence[str],
    monthly_gdp: str = "g_pbim",
    dfm_factors: int = 3,
    dfm_factor_orders: int = 2,
    dfm_maxiter: int = 200,
    dfm_tolerance: float = 1e-6,
    dfm_suppress_fit_warnings: bool = True,
    dfm_kwargs: Mapping[str, Any] | None = None,
    sglasso_n_lambda: int = 20,
    sglasso_kwargs: Mapping[str, Any] | None = None,
    covid_window: tuple[str, str] | None = None,
    covid_quarters: Sequence[str] | None = None,
    simple_exclude_month_window: tuple[str, str] | None = None,
    simple_exclude_quarter_window: tuple[str, str] | None = None,
) -> dict[str, BaseNowcaster]:
    """Core horse-race models for the refreshed DSAPM project."""

    dfm_params: dict[str, Any] = {
        "factors": dfm_factors,
        "factor_orders": dfm_factor_orders,
        "maxiter": dfm_maxiter,
        "tolerance": dfm_tolerance,
        "suppress_fit_warnings": dfm_suppress_fit_warnings,
        "covid_window": covid_window,
    }
    if dfm_kwargs:
        dfm_params.update(dict(dfm_kwargs))

    sglasso_params: dict[str, Any] = {
        "loss": "huber",
        "n_lambda": sglasso_n_lambda,
        "covid_quarters": covid_quarters,
    }
    if sglasso_kwargs:
        sglasso_params.update(dict(sglasso_kwargs))

    return {
        "Q-AR": ARNowcaster(order=1, exclude_quarter_window=simple_exclude_quarter_window),
        "M-AR": MonthlyARNowcaster(monthly_gdp=monthly_gdp, order=2),
        "Bridge": PooledBridgeNowcaster(monthly_gdp=monthly_gdp, monthly_vars=list(benchmark_vars)),
        "DFM": DFMNowcaster(**dfm_params),
        "sg-LASSO": SparseMIDASNowcaster(**sglasso_params),
    }


def build_competition_runs(
    *,
    artifacts: ProcessedPanelArtifacts,
    dfm_factors: int = 3,
    dfm_factor_orders: int = 2,
    dfm_maxiter: int = 200,
    dfm_tolerance: float = 1e-6,
    dfm_suppress_fit_warnings: bool = True,
    dfm_kwargs: Mapping[str, Any] | None = None,
    sglasso_n_lambda: int = 20,
    sglasso_kwargs: Mapping[str, Any] | None = None,
    covid_window: tuple[str, str] | None = None,
    covid_quarters: Sequence[str] | None = None,
    simple_exclude_month_window: tuple[str, str] | None = None,
    simple_exclude_quarter_window: tuple[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Spec-aware competition bundles.

    `DFM`, `sg-LASSO`, and `Q-AR` use the active transformed panel. The monthly
    GDP proxy benchmarks use a benchmark-specific panel that keeps monthly GDP in
    levels so the implied quarter aggregation stays coherent under every spec.
    """

    benchmark_artifacts = build_benchmark_artifacts(artifacts)
    main_models = {
        "Q-AR": ARNowcaster(order=1, exclude_quarter_window=simple_exclude_quarter_window),
        "DFM": DFMNowcaster(
            factors=dfm_factors,
            factor_orders=dfm_factor_orders,
            maxiter=dfm_maxiter,
            tolerance=dfm_tolerance,
            suppress_fit_warnings=dfm_suppress_fit_warnings,
            covid_window=covid_window,
            **({} if dfm_kwargs is None else dict(dfm_kwargs)),
        ),
        "sg-LASSO": SparseMIDASNowcaster(
            loss="huber",
            n_lambda=sglasso_n_lambda,
            covid_quarters=covid_quarters,
            **({} if sglasso_kwargs is None else dict(sglasso_kwargs)),
        ),
    }
    benchmark_models = {
        "M-AR": MonthlyLevelARNowcaster(
            monthly_gdp=benchmark_artifacts.monthly_gdp_column,
            target_transform=benchmark_artifacts.target_transform,
            order=2,
            exclude_month_window=simple_exclude_month_window,
        ),
        "Bridge": PooledLevelBridgeNowcaster(
            monthly_gdp=benchmark_artifacts.monthly_gdp_column,
            monthly_vars=list(benchmark_artifacts.benchmark_vars),
            target_transform=benchmark_artifacts.target_transform,
            exclude_month_window=simple_exclude_month_window,
        ),
    }
    return [
        {"label": "main", "artifacts": artifacts, "models": main_models},
        {"label": "benchmarks", "artifacts": benchmark_artifacts, "models": benchmark_models},
    ]


def run_competition_backtest(
    artifacts: ProcessedPanelArtifacts,
    *,
    leads: Sequence[int],
    eval_start: str | pd.Timestamp,
    eval_end: str | pd.Timestamp | None = None,
    min_train: int = 20,
    reuse_models_within_quarter: bool = True,
    n_jobs: int | None = None,
    verbose: bool = True,
    progress_desc: str | None = None,
    dfm_factors: int = 3,
    dfm_factor_orders: int = 2,
    dfm_maxiter: int = 200,
    dfm_tolerance: float = 1e-6,
    dfm_suppress_fit_warnings: bool = True,
    dfm_kwargs: Mapping[str, Any] | None = None,
    sglasso_n_lambda: int = 20,
    sglasso_kwargs: Mapping[str, Any] | None = None,
    covid_window: tuple[str, str] | None = None,
    covid_quarters: Sequence[str] | None = None,
    simple_exclude_month_window: tuple[str, str] | None = None,
    simple_exclude_quarter_window: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Run the coherent five-model competition for one preprocessing spec."""

    frames: list[pd.DataFrame] = []
    for bundle in build_competition_runs(
        artifacts=artifacts,
        dfm_factors=dfm_factors,
        dfm_factor_orders=dfm_factor_orders,
        dfm_maxiter=dfm_maxiter,
        dfm_tolerance=dfm_tolerance,
        dfm_suppress_fit_warnings=dfm_suppress_fit_warnings,
        dfm_kwargs=dfm_kwargs,
        sglasso_n_lambda=sglasso_n_lambda,
        sglasso_kwargs=sglasso_kwargs,
        covid_window=covid_window,
        covid_quarters=covid_quarters,
        simple_exclude_month_window=simple_exclude_month_window,
        simple_exclude_quarter_window=simple_exclude_quarter_window,
    ):
        sub = run_yoy_backtest(
            bundle["artifacts"],
            models=bundle["models"],
            leads=leads,
            eval_start=eval_start,
            eval_end=eval_end,
            min_train=min_train,
            reuse_models_within_quarter=reuse_models_within_quarter,
            n_jobs=n_jobs,
            verbose=verbose,
            progress_desc=progress_desc if progress_desc else f"{artifacts.spec} {bundle['label']}",
        )
        if not sub.empty:
            sub["panel_view"] = bundle["label"]
        frames.append(sub)
        gc.collect()
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ref_quarter", "days_to_publication", "model"]).reset_index(drop=True)


def recursive_standardize(
    info: InformationSet,
    *,
    target: str,
) -> tuple[InformationSet, dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    """Standardize predictors using only the data released by the origin."""

    monthly = info.monthly.copy()
    quarterly = info.quarterly.copy()
    m_scalers: dict[str, tuple[float, float]] = {}
    q_scalers: dict[str, tuple[float, float]] = {}

    for col in monthly.columns:
        obs = monthly[col].dropna()
        if obs.empty:
            m_scalers[col] = (0.0, 1.0)
            continue
        mu = float(obs.mean())
        sd = float(obs.std(ddof=0))
        if not np.isfinite(sd) or sd == 0.0:
            sd = 1.0
        monthly[col] = (monthly[col] - mu) / sd
        m_scalers[col] = (mu, sd)

    for col in quarterly.columns:
        if col == target:
            continue
        obs = quarterly[col].dropna()
        if obs.empty:
            q_scalers[col] = (0.0, 1.0)
            continue
        mu = float(obs.mean())
        sd = float(obs.std(ddof=0))
        if not np.isfinite(sd) or sd == 0.0:
            sd = 1.0
        quarterly[col] = (quarterly[col] - mu) / sd
        q_scalers[col] = (mu, sd)

    standardized = InformationSet(
        monthly=monthly,
        quarterly=quarterly,
        target=info.target,
        origin=info.origin,
        target_period=info.target_period,
        metadata=info.metadata,
    )
    return standardized, m_scalers, q_scalers


def yoy_from_levels(levels: pd.Series) -> pd.Series:
    s = pd.to_numeric(levels, errors="coerce")
    return 100.0 * (np.log(s) - np.log(s.shift(4)))


def convert_prediction_to_yoy(
    pred: float,
    *,
    quarter: pd.Timestamp,
    target_transform: str,
    level_sa: pd.Series,
) -> float:
    if not np.isfinite(pred):
        return float("nan")
    if target_transform == "yoy":
        return float(pred)
    if target_transform != "qoq_ann":
        raise ValueError(f"unsupported target transform {target_transform!r}")
    prev_q = pd.Timestamp((pd.Period(quarter, freq="Q") - 1).end_time).normalize().replace(day=1)
    lag4 = level_sa.shift(4).get(quarter, np.nan)
    prev = level_sa.get(prev_q, np.nan)
    if not np.isfinite(prev) or not np.isfinite(lag4):
        return float("nan")
    log_hat = np.log(prev) + pred / 400.0
    return float(100.0 * (log_hat - np.log(lag4)))


def actual_target_yoy(
    quarter: pd.Timestamp,
    *,
    level_sa: pd.Series,
) -> float:
    return float(yoy_from_levels(level_sa).get(quarter, np.nan))


def unscale_proxy_prediction(pred: float, scaler: tuple[float, float] | None) -> float:
    if scaler is None or not np.isfinite(pred):
        return float(pred)
    mu, sd = scaler
    return float(mu + sd * pred)


def run_yoy_backtest(
    artifacts: ProcessedPanelArtifacts,
    *,
    models: Mapping[str, BaseNowcaster],
    leads: Sequence[int],
    eval_start: str | pd.Timestamp,
    eval_end: str | pd.Timestamp | None = None,
    min_train: int = 20,
    reuse_models_within_quarter: bool = True,
    n_jobs: int | None = None,
    verbose: bool = True,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    """Pseudo-real-time backtest reported uniformly in GDP YoY growth terms.

    Parameters
    ----------
    n_jobs:
        Number of quarter workers. Use ``-1`` for all available cores. When
        left as ``None`` the helper stays sequential for short selected-lead
        runs and switches to all available cores for dense release-cycle sweeps.
    """

    panel = artifacts.panel
    target = artifacts.target_column
    level_sa = artifacts.quarterly_sa[artifacts.target_level_variable].copy()

    q_index = panel.quarterly.index
    start = pd.Timestamp(eval_start)
    end = q_index[-1] if eval_end is None else pd.Timestamp(eval_end)
    quarters = [q for q in q_index if start <= q <= end]
    rows: list[dict[str, Any]] = []
    failures: dict[str, int] = {}
    attempted_windows = 0
    refit_events = 0
    resolved_n_jobs = _resolve_n_jobs(n_jobs, n_quarters=len(quarters), n_leads=len(leads))

    desc = progress_desc or f"{target} backtest"
    if verbose:
        print(
            f"Running {target} backtest | spec={artifacts.spec} | quarters={len(quarters)} | "
            f"leads={len(leads)} | models={len(models)} | workers={resolved_n_jobs}"
        )

    worker_kwargs = dict(
        panel=panel,
        target=target,
        level_sa=level_sa,
        target_transform=artifacts.target_transform,
        monthly_gdp_column=artifacts.monthly_gdp_column,
        spec=artifacts.spec,
        models=models,
        leads=tuple(int(lead) for lead in leads),
        min_train=min_train,
        reuse_models_within_quarter=reuse_models_within_quarter,
    )

    if resolved_n_jobs == 1:
        quarter_iter = _progress_bar(quarters, enabled=verbose, total=len(quarters), desc=desc)
        for i, quarter in enumerate(quarter_iter, start=1):
            if verbose and hasattr(quarter_iter, "set_postfix_str"):
                qlabel = f"{pd.Timestamp(quarter).year}Q{pd.Timestamp(quarter).quarter}"
                quarter_iter.set_postfix_str(qlabel)
            r, f, a, refits = _run_yoy_backtest_quarter(quarter, **worker_kwargs)
            rows.extend(r)
            attempted_windows += a
            refit_events += refits
            for key, value in f.items():
                failures[key] = failures.get(key, 0) + value

            if verbose and not hasattr(quarter_iter, "set_postfix_str") and (i == 1 or i % 10 == 0 or i == len(quarters)):
                print(f"  processed {i}/{len(quarters)} quarters")
        if hasattr(quarter_iter, "close"):
            quarter_iter.close()
    else:
        try:
            from joblib import Parallel, delayed
        except ModuleNotFoundError:
            warnings.warn("joblib not installed; falling back to sequential execution.", RuntimeWarning, stacklevel=2)
            return run_yoy_backtest(
                artifacts,
                models=models,
                leads=leads,
                eval_start=eval_start,
                eval_end=eval_end,
                min_train=min_train,
                reuse_models_within_quarter=reuse_models_within_quarter,
                n_jobs=1,
                verbose=verbose,
                progress_desc=progress_desc,
            )

        parallel_kwargs = {
            "n_jobs": resolved_n_jobs,
            "verbose": 10 if verbose else 0,
            "prefer": "processes",
            "batch_size": 1,
            "pre_dispatch": "n_jobs",
        }

        def _tasks():
            for quarter in quarters:
                yield delayed(_run_yoy_backtest_quarter)(quarter, **worker_kwargs)

        try:
            result_iter = Parallel(return_as="generator", **parallel_kwargs)(_tasks())
        except TypeError:
            result_iter = Parallel(**parallel_kwargs)(_tasks())

        result_iter = _progress_bar(result_iter, enabled=verbose, total=len(quarters), desc=desc)
        for i, (r, f, a, refits) in enumerate(result_iter, start=1):
            rows.extend(r)
            attempted_windows += a
            refit_events += refits
            for key, value in f.items():
                failures[key] = failures.get(key, 0) + value
            if i % max(1, resolved_n_jobs) == 0:
                gc.collect()
        if hasattr(result_iter, "close"):
            result_iter.close()

    out = pd.DataFrame(rows)
    if failures:
        msg = ", ".join(f"{k}: {v}" for k, v in sorted(failures.items()))
        warnings.warn(f"run_yoy_backtest encountered fit failures -> {msg}", RuntimeWarning, stacklevel=2)
    if out.empty:
        if verbose:
            print("Backtest finished with no valid rows.")
        return out
    out = out[np.isfinite(out["y_true"])].copy()
    out = out.sort_values(["ref_quarter", "days_to_publication", "model"]).reset_index(drop=True)
    if verbose:
        reuse_count = max(0, attempted_windows - refit_events)
        print(
            f"Finished {target} backtest | attempted quarter-lead windows={attempted_windows} | "
            f"unique information sets={refit_events} | reused windows={reuse_count} | "
            f"rows={len(out)} | valid quarters={out['ref_quarter'].nunique()}"
        )
    return out


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["error"] = out["y_hat"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    out["sq_error"] = out["error"] ** 2
    return out


def selected_leads_metrics(
    df: pd.DataFrame,
    *,
    models: Sequence[str] | None = None,
    exclude_years: Sequence[int] = (),
) -> pd.DataFrame:
    """Overall and lead-by-lead metrics for the selected leads."""

    if df.empty:
        return pd.DataFrame()
    data = add_error_columns(df)
    data = data[np.isfinite(data["y_true"]) & np.isfinite(data["y_hat"])].copy()
    if models is not None:
        data = data[data["model"].isin(list(models))]
    if exclude_years:
        data = data[~data["ref_quarter"].dt.year.isin(list(exclude_years))]
    if data.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (model, lead), g in data.groupby(["model", "days_to_publication"]):
        err = g["error"].to_numpy()
        y = g["y_true"].to_numpy()
        yhat = g["y_hat"].to_numpy()
        rows.append(
            {
                "model": model,
                "days_to_publication": int(lead),
                "n": int(len(g)),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "mae": float(np.mean(np.abs(err))),
                "bias": float(np.mean(err)),
                "corr": float(np.corrcoef(y, yhat)[0, 1]) if len(g) > 1 else np.nan,
            }
        )

    out = pd.DataFrame(rows).sort_values(["days_to_publication", "rmse", "model"])
    overall = (
        data.groupby("model")
        .apply(
            lambda g: pd.Series(
                {
                    "n": int(len(g)),
                    "rmse": float(np.sqrt(np.mean((g["y_hat"] - g["y_true"]) ** 2))),
                    "mae": float(np.mean(np.abs(g["y_hat"] - g["y_true"]))),
                    "bias": float(np.mean(g["y_hat"] - g["y_true"])),
                    "corr": float(np.corrcoef(g["y_true"], g["y_hat"])[0, 1]) if len(g) > 1 else np.nan,
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    overall["days_to_publication"] = "overall"
    return pd.concat([overall, out], ignore_index=True)


def attach_relative_rmse(metrics: pd.DataFrame, *, baseline: str) -> pd.DataFrame:
    out = metrics.copy()
    numeric = out[out["days_to_publication"] != "overall"].copy()
    if numeric.empty:
        return out
    base = numeric[numeric["model"] == baseline][["days_to_publication", "rmse"]].rename(columns={"rmse": "baseline_rmse"})
    numeric = numeric.merge(base, on="days_to_publication", how="left")
    numeric["relative_rmse"] = numeric["rmse"] / numeric["baseline_rmse"]

    overall = out[out["days_to_publication"] == "overall"].copy()
    base_all = overall.loc[overall["model"] == baseline, "rmse"]
    if len(base_all):
        overall["relative_rmse"] = overall["rmse"] / float(base_all.iloc[0])
    return pd.concat([overall, numeric], ignore_index=True)


def metric_table(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    table = (
        metrics.loc[metrics["days_to_publication"] != "overall", ["model", "days_to_publication", metric]]
        .pivot(index="model", columns="days_to_publication", values=metric)
        .sort_index()
    )
    table.columns.name = "days_to_publication"
    return table


def ranking_table(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics.loc[metrics["days_to_publication"] != "overall", ["model", "days_to_publication", "rmse"]].copy()
    base["rank"] = base.groupby("days_to_publication")["rmse"].rank(method="dense")
    return (
        base.sort_values(["days_to_publication", "rank", "model"])
        .pivot(index="model", columns="days_to_publication", values="rank")
        .sort_index()
    )


def plot_rmse_horse_race(
    metrics: pd.DataFrame,
    *,
    label: str,
    ax=None,
):
    import matplotlib.pyplot as plt

    data = metrics.loc[metrics["days_to_publication"] != "overall"].copy()
    leads = sorted(data["days_to_publication"].unique())
    models = list(dict.fromkeys(data["model"]))
    width = 0.18
    x = np.arange(len(models))
    if ax is None:
        _, ax = plt.subplots(figsize=(8.8, 4.8))

    colors = [PALETTE["myblue"], PALETTE["myred"], PALETTE["mygreen"], PALETTE.get("myyellow", "#B48A2C")]
    for i, lead in enumerate(leads):
        vals: list[float] = []
        for model in models:
            hit = data.loc[(data["model"] == model) & (data["days_to_publication"] == lead), "rmse"]
            vals.append(float(hit.iloc[0]) if len(hit) else np.nan)
        ax.bar(x + (i - (len(leads) - 1) / 2) * width, vals, width=width, label=f"{lead}d", color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("RMSE in GDP YoY points")
    ax.set_title(label, loc="left")
    ax.legend(frameon=False, ncol=min(len(leads), 4), title="Origin")
    return ax


def common_sample_rmse(df: pd.DataFrame) -> pd.DataFrame:
    """RMSEs after restricting every model to the common non-missing sample."""

    keys = ["ref_quarter", "days_to_publication"]
    hits = (
        df.assign(valid=np.isfinite(df["y_hat"]))
        .pivot_table(index=keys, columns="model", values="valid", aggfunc="first")
        .fillna(False)
    )
    common_keys = hits[hits.all(axis=1)].index
    common = df.set_index(keys).loc[common_keys].reset_index()
    metrics = selected_leads_metrics(common)
    metrics["sample"] = "common"
    return metrics


__all__ = [
    "OUTPUT_DIR",
    "add_error_columns",
    "actual_target_yoy",
    "attach_relative_rmse",
    "build_competition_runs",
    "build_main_models",
    "common_sample_rmse",
    "convert_prediction_to_yoy",
    "metric_table",
    "plot_rmse_horse_race",
    "ranking_table",
    "recursive_standardize",
    "run_competition_backtest",
    "run_yoy_backtest",
    "selected_leads_metrics",
    "unscale_proxy_prediction",
    "yoy_from_levels",
]
