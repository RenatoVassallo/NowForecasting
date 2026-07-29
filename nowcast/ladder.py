"""Stage 2: the config-driven model ladder.

Given a :class:`NowcastApp` and the processed artifacts, assemble the ladder
(simple to complex) and run it through the release-cycle YoY backtest:

* **Q-AR**  - quarterly AR(1) on the target (naive floor).
* **M-AR**  - the time-aggregated monthly-proxy AR (rides the proxy).
* **Bridge** - pooled single-indicator bridge over proxy + survey block.
* **DFM family** - small (1 structural factor), medium (two-block real vs soft),
  full (3 global factors, 2 lags over the whole panel).
* **GB-Trees** - gradient-boosted trees on interpretable per-indicator features.

sg-LASSO is intentionally dropped. Crisis-in-estimation is one window threaded
to every model through its own mechanism (off by default, since it is inert on
the pre-COVID 2005-2019 window that is scored first).

This module is workflow, not modelling, so it lives in the project layer and is
written to be reused for other applications (a China app is another
`NowcastApp`). It adds functions rather than editing the existing
`build_main_models` / `run_competition_backtest`, so the current notebooks keep
running.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from MIDAS import (
    ARNowcaster,
    DFMNowcaster,
    GBTreesNowcaster,
    MonthlyLevelARNowcaster,
    PooledLevelBridgeNowcaster,
)
from MIDAS.base import BaseNowcaster

from core.appconfig import NowcastApp
from core.preprocess import ProcessedPanelArtifacts, build_benchmark_artifacts
from core.backtest import run_yoy_backtest

# Metadata groups treated as the real-activity block and the survey/soft block
# of the medium DFM. Overridable per application.
REAL_GROUPS = ("Sectoral activity", "High-frequency activity", "External trade")
SOFT_GROUPS = ("Business surveys",)
# GDP-component series near-collinear with the monthly proxy; dropped from the
# DFM blocks to keep the EM's factor covariance positive definite.
COLLINEAR_WITH_PROXY = ("g_pbim_manuf", "g_pbim_const", "g_pbim_prim", "g_pbim_nonprim", "g_dd")


def real_soft_blocks(
    metadata_map: pd.DataFrame,
    columns: Sequence[str],
    *,
    real_groups: Sequence[str] = REAL_GROUPS,
    soft_groups: Sequence[str] = SOFT_GROUPS,
    drop: Sequence[str] = COLLINEAR_WITH_PROXY,
) -> tuple[list[str], list[str]]:
    """Split panel columns into a real-activity block and a survey/soft block."""

    grp = metadata_map.set_index("column")["group"].to_dict()
    drop_set = set(drop)
    real = [c for c in columns if grp.get(c) in set(real_groups) and c not in drop_set]
    soft = [c for c in columns if grp.get(c) in set(soft_groups)]
    return real, soft


def default_small_block(columns: Sequence[str], proxy: str | None) -> list[str]:
    """A small, diverse, non-collinear structural block for the 1-factor DFM."""

    wanted = [proxy] if proxy else []
    wanted += ["cem", "g_elec", "g_p_cu", "g_x_t"]
    return [c for c in dict.fromkeys(wanted) if c and c in set(columns)]


def build_ladder_runs(
    app: NowcastApp,
    artifacts: ProcessedPanelArtifacts,
    *,
    crisis: bool = False,
    dfm_maxiter: int = 200,
    dfm_factors_full: int = 3,
    dfm_factor_orders_full: int = 2,
    small_block: Sequence[str] | None = None,
    gb_vars: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Two spec-aware bundles (transformed-panel models and levels benchmarks).

    Mirrors ``build_competition_runs`` but swaps sg-LASSO for the DFM family and
    GB-Trees. Crisis-in-estimation is applied to every model only when
    ``crisis`` is True (via the app's ``crisis_window`` / ``crisis_quarters``).
    """

    benchmark_artifacts = build_benchmark_artifacts(artifacts)
    cols = list(artifacts.panel.monthly.columns)
    real, soft = real_soft_blocks(artifacts.metadata_map, cols)
    small = list(small_block) if small_block is not None else default_small_block(cols, app.monthly_proxy)
    gb = list(gb_vars) if gb_vars is not None else real + soft

    # Crisis-in-estimation parameters (unified window, per-model mechanism).
    month_window = app.crisis_window if crisis else None
    quarter_window = _month_to_quarter_window(app.crisis_window) if crisis else None
    covid_quarters = list(app.crisis_quarters) if (crisis and app.crisis_quarters) else None

    main_models: dict[str, BaseNowcaster] = {
        "Q-AR": ARNowcaster(order=1, exclude_quarter_window=quarter_window),
        "DFM-S": DFMNowcaster(
            factors=1, factor_orders=1, monthly_vars=small, idiosyncratic_ar1=False,
            maxiter=dfm_maxiter, covid_window=month_window, name="DFM-S",
        ),
        "DFM-M": DFMNowcaster(
            factors={"Real": real, "Soft": soft}, factor_orders=1, monthly_vars=real + soft,
            idiosyncratic_ar1=False, maxiter=dfm_maxiter, covid_window=month_window, name="DFM-M",
        ),
        "DFM-F": DFMNowcaster(
            factors=dfm_factors_full, factor_orders=dfm_factor_orders_full,
            maxiter=dfm_maxiter, covid_window=month_window, name="DFM-F",
        ),
        "GB-Trees": GBTreesNowcaster(monthly_vars=gb, covid_quarters=covid_quarters),
    }
    benchmark_models: dict[str, BaseNowcaster] = {
        "M-AR": MonthlyLevelARNowcaster(
            monthly_gdp=benchmark_artifacts.monthly_gdp_column,
            target_transform=benchmark_artifacts.target_transform,
            order=2, exclude_month_window=month_window,
        ),
        "Bridge": PooledLevelBridgeNowcaster(
            monthly_gdp=benchmark_artifacts.monthly_gdp_column,
            monthly_vars=list(benchmark_artifacts.benchmark_vars),
            target_transform=benchmark_artifacts.target_transform,
            exclude_month_window=month_window,
        ),
    }
    return [
        {"label": "main", "artifacts": artifacts, "models": main_models},
        {"label": "benchmarks", "artifacts": benchmark_artifacts, "models": benchmark_models},
    ]


def run_ladder_backtest(
    app: NowcastApp,
    artifacts: ProcessedPanelArtifacts,
    *,
    leads: Sequence[int],
    eval_start: str | pd.Timestamp,
    eval_end: str | pd.Timestamp | None = None,
    exclude_years: Sequence[int] = (),
    crisis: bool = False,
    min_train: int = 20,
    n_jobs: int | None = None,
    verbose: bool = True,
    dfm_maxiter: int = 200,
) -> pd.DataFrame:
    """Run the full ladder for one application and return the tidy YoY frame."""

    bundles = build_ladder_runs(app, artifacts, crisis=crisis, dfm_maxiter=dfm_maxiter)
    frames: list[pd.DataFrame] = []
    for bundle in bundles:
        sub = run_yoy_backtest(
            bundle["artifacts"], models=bundle["models"], leads=leads,
            eval_start=eval_start, eval_end=eval_end, min_train=min_train,
            n_jobs=n_jobs, verbose=verbose, progress_desc=f"{app.name} {bundle['label']}",
        )
        if not sub.empty:
            sub["panel_view"] = bundle["label"]
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if exclude_years and not out.empty:
        out = out[~out["ref_quarter"].dt.year.isin(list(exclude_years))]
    return out.sort_values(["ref_quarter", "days_to_publication", "model"]).reset_index(drop=True)


def _month_to_quarter_window(month_window: tuple[str, str] | None) -> tuple[str, str] | None:
    """Convert a ('YYYY-MM','YYYY-MM') window to ('YYYYQq','YYYYQq')."""

    if month_window is None:
        return None
    lo, hi = month_window
    return (str(pd.Period(lo, freq="Q")), str(pd.Period(hi, freq="Q")))


__all__ = [
    "REAL_GROUPS",
    "SOFT_GROUPS",
    "real_soft_blocks",
    "default_small_block",
    "build_ladder_runs",
    "run_ladder_backtest",
]
