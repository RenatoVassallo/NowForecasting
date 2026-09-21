"""Small, auditable models for the weekly-activity MVP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import RIDGE_ALPHAS

CALENDAR_FEATURES = ("month_sin", "month_cos", "calendar_days_elapsed")
ELECTRICITY_FEATURES = ("elec_log_mean_mwh", "elec_observed_days", "elec_elapsed_days",
                        "elec_calendar_coverage",
                        "elec_observed_dow_0", "elec_observed_dow_1", "elec_observed_dow_2",
                        "elec_observed_dow_3", "elec_observed_dow_4", "elec_observed_dow_5",
                        "elec_observed_dow_6")
LBTR_FEATURES = ("lbtr_log_mean_value_mn", "lbtr_log_mean_value_me",
                 "lbtr_log_mean_count_mn", "lbtr_log_mean_count_me",
                 "lbtr_observed_days")


class ModelUnavailable(RuntimeError):
    """A candidate cannot be fitted on its required, observed feature set."""


def candidate_names() -> tuple[str, ...]:
    return ("AR(1)", "Ridge electricity", "Ridge multiblock")


def _blocks(feature: str) -> str:
    if feature.startswith("elec_"):
        return "COES electricity"
    if feature.startswith("lbtr_"):
        return "BCRP LBTR"
    if feature.startswith("trend_"):
        return "Google Trends"
    return "calendar"


def _as_numeric(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


@dataclass
class AR1Fit:
    intercept: float
    phi: float
    last_value: float
    last_month: pd.Timestamp
    n_train: int

    def predict(self, target_month) -> float:
        target = pd.Timestamp(target_month).to_period("M")
        last = pd.Timestamp(self.last_month).to_period("M")
        h = max(1, int(target.ordinal - last.ordinal))
        value = float(self.last_value)
        for _ in range(h):
            value = self.intercept + self.phi * value
        return float(value)


def fit_ar1(train: pd.DataFrame, *, target_col: str = "pbi_mom_pct") -> AR1Fit:
    d = train[["target_month", target_col]].dropna().sort_values("target_month")
    if len(d) < 12:
        raise ModelUnavailable("AR(1) needs at least 12 released monthly GDP observations")
    y = pd.to_numeric(d[target_col], errors="coerce").to_numpy(dtype=float)
    x, yy = y[:-1], y[1:]
    intercept, phi = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), yy, rcond=None)[0]
    # A simple benchmark must not produce an explosive multi-step forecast when
    # its short early sample is noisy.  The cap is fixed before evaluation.
    phi = float(np.clip(phi, -0.95, 0.95))
    return AR1Fit(float(intercept), phi, float(y[-1]), pd.Timestamp(d.target_month.iloc[-1]), len(d))


@dataclass
class RidgeFit:
    features: list[str]
    mean: np.ndarray
    scale: np.ndarray
    estimator: Ridge
    alpha: float
    n_train: int

    def predict(self, row: pd.Series | pd.DataFrame) -> float:
        frame = row.to_frame().T if isinstance(row, pd.Series) else row
        x = _as_numeric(frame, self.features)
        if x.shape[0] != 1 or not np.isfinite(x).all():
            return np.nan
        return float(self.estimator.predict((x - self.mean) / self.scale)[0])

    def importance(self) -> pd.DataFrame:
        out = pd.DataFrame({"feature": self.features,
                            "standardized_coefficient": self.estimator.coef_})
        out["block"] = out.feature.map(_blocks)
        return (out.assign(abs_coefficient=out.standardized_coefficient.abs())
                   .groupby("block", as_index=False).abs_coefficient.sum()
                   .rename(columns={"abs_coefficient": "importance"})
                   .sort_values("importance", ascending=False))


def _ridge_alpha(X: np.ndarray, y: np.ndarray, alphas=RIDGE_ALPHAS) -> float:
    """Choose Ridge alpha with chronological folds entirely inside training."""
    n = len(y)
    if n < 30:
        return float(alphas[1])  # pre-specified conservative default, no tiny-CV noise
    # Three expanding validation blocks, all strictly earlier than the final
    # production fit.  No random folds or test-period tuning.
    starts = [max(12, n // 2), max(16, (2 * n) // 3), max(20, (3 * n) // 4)]
    scores = {float(a): [] for a in alphas}
    for start in sorted(set(s for s in starts if s < n - 2)):
        train_x, test_x = X[:start], X[start:min(n, start + max(3, n // 8))]
        train_y, test_y = y[:start], y[start:min(n, start + max(3, n // 8))]
        mean, scale = train_x.mean(axis=0), train_x.std(axis=0, ddof=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        for alpha in alphas:
            fit = Ridge(alpha=float(alpha)).fit((train_x - mean) / scale, train_y)
            scores[float(alpha)].append(float(np.mean((fit.predict((test_x - mean) / scale) - test_y) ** 2)))
    means = {a: np.mean(v) if v else np.inf for a, v in scores.items()}
    return min(means, key=means.get)


def fit_ridge(train: pd.DataFrame, features: list[str], *, target_col: str = "pbi_mom_pct") -> RidgeFit:
    if not features:
        raise ModelUnavailable("Ridge candidate has no observed feature columns")
    cols = [*features, target_col]
    d = train.dropna(subset=cols).sort_values("target_month")
    if len(d) < 24:
        raise ModelUnavailable("Ridge needs at least 24 complete released observations")
    X = _as_numeric(d, features)
    y = pd.to_numeric(d[target_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ModelUnavailable("Ridge received non-finite data after complete-case filtering")
    mean, scale = X.mean(axis=0), X.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    alpha = _ridge_alpha(X, y)
    estimator = Ridge(alpha=alpha).fit((X - mean) / scale, y)
    return RidgeFit(list(features), mean, scale, estimator, alpha, len(d))


def ridge_features(name: str, train: pd.DataFrame, row: pd.Series) -> list[str]:
    """Required blocks by candidate, with Trends only if genuinely available."""
    if name == "Ridge electricity":
        cols = list(CALENDAR_FEATURES + ELECTRICITY_FEATURES)
        missing = [c for c in cols if c not in train.columns or pd.isna(row.get(c))]
        if missing:
            raise ModelUnavailable(f"{name} lacks observed features at cutoff: {missing}")
        return cols
    if name != "Ridge multiblock":
        raise KeyError(name)
    cols = list(CALENDAR_FEATURES + ELECTRICITY_FEATURES + LBTR_FEATURES)
    trend_cols = [c for c in train.columns if c.startswith("trend_") and c != "trend_complete_weeks"]
    # A Trends column joins only if it is completely observed in the training
    # sample and at this cut-off.  This avoids an implicit fill or a future data
    # availability test.
    for c in trend_cols:
        if pd.notna(row.get(c)) and train[c].notna().all():
            cols.append(c)
    missing = [c for c in cols if c not in train.columns or pd.isna(row.get(c))]
    if missing:
        raise ModelUnavailable(f"{name} lacks observed features at cutoff: {missing}")
    return cols


def fit_candidate(name: str, train: pd.DataFrame, row: pd.Series):
    """Fit a pre-specified candidate using only one origin's released history."""
    if name == "AR(1)":
        return fit_ar1(train)
    return fit_ridge(train, ridge_features(name, train, row))


def predict_candidate(fit, target_month, row: pd.Series) -> float:
    return fit.predict(target_month) if isinstance(fit, AR1Fit) else fit.predict(row)


def candidate_importance(fit) -> pd.DataFrame:
    if isinstance(fit, RidgeFit):
        return fit.importance()
    return pd.DataFrame({"block": ["lagged GDP"], "importance": [1.0]})
