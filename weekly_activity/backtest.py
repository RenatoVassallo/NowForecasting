"""Expanding-window evaluation for weekly information-set GDP nowcasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.evaluation import mae, rmse

from .config import (EVALUATION_REGIME, EVALUATION_START, MIN_TRAIN_MONTHS,
                     TEST_START, VALIDATION_END)
from .models import ModelUnavailable, candidate_names, fit_candidate, predict_candidate


def _period_group(month) -> str:
    y = pd.Timestamp(month).year
    if y < 2020:
        return "pre_2020"
    if y <= 2021:
        return "2020_2021"
    return "post_2020"


def _sample(month) -> str:
    p = pd.Timestamp(month)
    if p >= pd.Timestamp(TEST_START):
        return "test"
    if p <= pd.Timestamp(VALIDATION_END):
        return "validation"
    return "gap"


def run_backtest(
    features: pd.DataFrame,
    target: pd.DataFrame,
    *,
    stages: tuple[str, ...] | None = None,
    min_train_months: int = MIN_TRAIN_MONTHS,
    evaluation_start: str = EVALUATION_START,
) -> pd.DataFrame:
    """One expanding-window nowcast per target month, stage and model.

    Training labels must have their explicit assumed release date no later than
    the feature cut-off.  Predictor data are already bounded by their
    observation date in :func:`weekly_activity.features.build_feature_frame`.
    The latter is labelled pseudo-real-time because source publication
    timestamps are not publicly reconstructible.
    """
    req = {"target_month", "stage", "cutoff_date"}
    missing = req.difference(features.columns)
    if missing:
        raise ValueError(f"feature frame misses keys: {sorted(missing)}")
    target_req = {"target_month", "pbi_mom_pct", "target_release_date"}
    missing = target_req.difference(target.columns)
    if missing:
        raise ValueError(f"target frame misses keys: {sorted(missing)}")
    f = features.copy()
    t = target.copy()
    for d in (f, t):
        d["target_month"] = pd.to_datetime(d.target_month).dt.to_period("M").dt.to_timestamp()
    f["cutoff_date"] = pd.to_datetime(f.cutoff_date).dt.normalize()
    t["target_release_date"] = pd.to_datetime(t.target_release_date).dt.normalize()
    panel = f.merge(t[["target_month", "pbi_mom_pct", "target_release_date"]],
                    on="target_month", how="left")
    if stages is not None:
        panel = panel[panel.stage.isin(stages)]
    panel = panel[panel.target_month >= pd.Timestamp(evaluation_start)]
    panel = panel.sort_values(["stage", "target_month", "cutoff_date"])
    rows = []
    for current in panel.itertuples(index=False):
        row = pd.Series(current._asdict())
        if not np.isfinite(pd.to_numeric(pd.Series([row.pbi_mom_pct]), errors="coerce")).iloc[0]:
            continue
        train = panel[(panel.stage == row.stage)
                      & (panel.target_month < row.target_month)
                      & (panel.target_release_date <= row.cutoff_date)].copy()
        train = train.dropna(subset=["pbi_mom_pct"]).sort_values("target_month")
        if len(train) < int(min_train_months):
            continue
        jump = float(train.pbi_mom_pct.iloc[-1])
        last_month = pd.Timestamp(train.target_month.iloc[-1])
        last_release = pd.Timestamp(train.target_release_date.iloc[-1])
        for model in candidate_names():
            try:
                fit = fit_candidate(model, train, row)
                y_hat = predict_candidate(fit, row.target_month, row)
                n_train = int(getattr(fit, "n_train", len(train)))
                alpha = getattr(fit, "alpha", np.nan)
            except ModelUnavailable as exc:
                y_hat, n_train, alpha = np.nan, 0, np.nan
                issue = str(exc)
            else:
                issue = ""
            rows.append({
                "target_month": row.target_month, "cutoff_date": row.cutoff_date,
                "stage": row.stage, "model": model, "y_true": float(row.pbi_mom_pct),
                "y_hat": y_hat, "y_jump_off": jump, "n_train": n_train,
                "ridge_alpha": alpha, "model_issue": issue,
                "train_last_target_month": last_month,
                "train_last_release_date": last_release,
                "outcome_release_date": pd.Timestamp(row.target_release_date),
                "period_group": _period_group(row.target_month),
                "sample": _sample(row.target_month),
                "evaluation_regime": EVALUATION_REGIME,
            })
    return pd.DataFrame(rows)


def _directional(g: pd.DataFrame) -> float:
    d = g.dropna(subset=["y_true", "y_hat", "y_jump_off"])
    if d.empty:
        return np.nan
    return float(np.mean(np.sign(d.y_hat - d.y_jump_off) == np.sign(d.y_true - d.y_jump_off)))


def score_backtest(backtest: pd.DataFrame) -> pd.DataFrame:
    """Own and common-sample MAE/RMSE by model, cut-off and time regime."""
    if backtest.empty:
        return pd.DataFrame()
    keys = ["stage", "sample", "period_group"]
    rows = []
    for group, g in backtest.groupby(keys, dropna=False):
        pivot = g.pivot_table(index="target_month", columns="model", values="y_hat")
        # A common comparison means every pre-specified candidate emits at the
        # same origin.  ``pivot_table`` otherwise drops a model that is all-NaN
        # in an early block and silently enlarges the comparison sample.
        pivot = pivot.reindex(columns=list(candidate_names()))
        complete = pivot.notna().all(axis=1)
        for model, m in g.groupby("model"):
            own = m.dropna(subset=["y_true", "y_hat"])
            common = m[m.target_month.isin(pivot.index[complete])].dropna(subset=["y_true", "y_hat"])
            e = own.y_hat - own.y_true
            ec = common.y_hat - common.y_true
            rows.append({
                **dict(zip(keys, group)), "model": model, "n": len(own),
                "rmse": rmse(e), "mae": mae(e), "bias": float(e.mean()) if len(e) else np.nan,
                "directional_accuracy": _directional(own), "common_n": len(common),
                "rmse_common": rmse(ec), "mae_common": mae(ec),
                "evaluation_regime": EVALUATION_REGIME,
            })
    return pd.DataFrame(rows).sort_values(keys + ["rmse_common", "model"]).reset_index(drop=True)
