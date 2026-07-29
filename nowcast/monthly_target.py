"""Monthly-target nowcasting helpers for DSAPM notebook 13.

These utilities are intentionally local to ``notebooks/DSAPM``. They mirror the
package's pseudo-real-time discipline for a *monthly* target without changing the
public ``MIDAS`` library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV

from MIDAS.backtest import publication_date as quarter_publication_date
from MIDAS.metadata import MetadataPanel
from MIDAS.realtime import RealtimeEngine


# ---------------------------------------------------------------------------
# Core monthly information-set plumbing
# ---------------------------------------------------------------------------
@dataclass
class MonthlyInformationSet:
    """Everything knowable at one forecast origin for a monthly target."""

    monthly: pd.DataFrame
    target: str
    origin: pd.Timestamp
    target_period: pd.Timestamp
    metadata: MetadataPanel | None = None

    @property
    def y(self) -> pd.Series:
        return self.monthly[self.target]

    def released_target(self) -> pd.Series:
        return self.y.dropna()


@dataclass
class MonthlyNowcastResult:
    """Point nowcast plus optional diagnostics."""

    mean: float
    model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseMonthlyNowcaster(ABC):
    """Minimal monthly-target interface for the local DSAPM experiments."""

    _name: str | None = None
    train_rmse_: float | None = None

    @property
    def name(self) -> str:
        return self._name or type(self).__name__

    @abstractmethod
    def fit(self, info: MonthlyInformationSet) -> "BaseMonthlyNowcaster":
        """Estimate the model on the released history in ``info``."""

    @abstractmethod
    def nowcast(self, info: MonthlyInformationSet) -> MonthlyNowcastResult:
        """Produce a nowcast for ``info.target_period``."""


def month_publication_date(target_period: pd.Timestamp, delay_days: int) -> pd.Timestamp:
    """Publication date of a monthly target: month end plus ``delay_days``."""

    return pd.Timestamp(target_period) + pd.offsets.MonthEnd(1) + pd.Timedelta(days=int(delay_days))


class MonthlyRealtimeEngine:
    """Mask not-yet-released monthly values using the panel's delays."""

    def __init__(self, panel: MetadataPanel) -> None:
        self.panel = panel
        self._delays = panel.delays()

    def _mask_frame(self, frame: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        out = frame.copy()
        period_end = out.index + pd.offsets.MonthEnd(1)
        for col in out.columns:
            release = period_end + pd.Timedelta(days=int(self._delays[col]))
            unreleased = pd.Series(release > origin, index=out.index)
            if unreleased.any():
                out.loc[unreleased, col] = np.nan
        return out

    def information_set(
        self,
        origin: str | pd.Timestamp,
        target: str,
        *,
        target_period: str | pd.Timestamp,
    ) -> MonthlyInformationSet:
        origin = pd.Timestamp(origin)
        target_period = pd.Timestamp(target_period)
        monthly = self._mask_frame(self.panel.monthly, origin)
        return MonthlyInformationSet(
            monthly=monthly,
            target=target,
            origin=origin,
            target_period=target_period,
            metadata=self.panel,
        )


# ---------------------------------------------------------------------------
# Small linear-algebra helpers
# ---------------------------------------------------------------------------
def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def _ar_design(y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.ones((len(y) - p, p + 1))
    for k in range(1, p + 1):
        X[:, k] = y[p - k : len(y) - k]
    return X, y[p:]


def _is_stationary(ar_coefs: np.ndarray, tol: float = 0.999) -> bool:
    p = len(ar_coefs)
    if p == 0:
        return True
    comp = np.zeros((p, p))
    comp[0, :] = ar_coefs
    if p > 1:
        comp[1:, :-1] = np.eye(p - 1)
    return float(np.max(np.abs(np.linalg.eigvals(comp)))) < tol


def _fit_stable_ar(series: pd.Series, order: int = 2, min_obs: int = 12) -> tuple[np.ndarray, int] | None:
    s = series.dropna()
    if len(s) < max(min_obs, order + 3):
        return None
    y = s.to_numpy(dtype=float)
    p = min(order, len(y) - 2)
    beta: np.ndarray | None = None
    while p >= 1:
        X, yt = _ar_design(y, p)
        beta, _ = _ols(X, yt)
        if _is_stationary(beta[1:]):
            return beta, p
        p -= 1
    return None


def _extend_series_with_ar(
    series: pd.Series,
    end_month: pd.Timestamp,
    *,
    order: int = 2,
    min_obs: int = 12,
) -> pd.Series:
    """Extend a released series to ``end_month`` with a stable AR or flat carry."""

    released = series.dropna()
    if released.empty:
        return pd.Series(dtype=float)
    idx = pd.date_range(released.index[0], pd.Timestamp(end_month), freq="MS")
    arr = series.reindex(idx).to_numpy(dtype=float).copy()
    fitted = _fit_stable_ar(series, order=order, min_obs=min_obs)
    if fitted is None:
        return pd.Series(arr, index=idx).ffill()
    beta, p = fitted
    for i in range(len(arr)):
        if np.isfinite(arr[i]):
            continue
        if i < p:
            continue
        lags = np.array([arr[i - k] for k in range(1, p + 1)])
        if not np.all(np.isfinite(lags)):
            if i > 0 and np.isfinite(arr[i - 1]):
                arr[i] = arr[i - 1]
            continue
        arr[i] = beta[0] + beta[1:] @ lags
    return pd.Series(arr, index=idx).ffill()


def _relative_rmse_group(df: pd.DataFrame, *, baseline: str) -> pd.DataFrame:
    out = df.copy()
    base = out[out["model"] == baseline][["days_to_publication", "rmse"]].rename(columns={"rmse": "baseline_rmse"})
    out = out.merge(base, on="days_to_publication", how="left")
    out["relative_rmse"] = out["rmse"] / out["baseline_rmse"]
    return out


def _months_in_quarter(quarter: pd.Timestamp) -> list[pd.Timestamp]:
    period = pd.Period(pd.Timestamp(quarter), freq="Q")
    start = pd.Timestamp(period.start_time).normalize().replace(day=1)
    return list(pd.date_range(start, periods=3, freq="MS"))


def quarterly_mean_from_monthly_series(series: pd.Series) -> pd.Series:
    """Quarterly mean of a monthly series under the panel's date convention."""

    s = series.dropna().copy()
    if s.empty:
        return pd.Series(dtype=float)
    q = s.groupby(s.index.to_period("Q")).mean()
    q.index = q.index.to_timestamp(how="end").normalize().map(lambda ts: ts.replace(day=1))
    q.index.name = "ref_quarter"
    return q.astype(float)


# ---------------------------------------------------------------------------
# Monthly target models
# ---------------------------------------------------------------------------
@dataclass
class MonthlyARNowcaster(BaseMonthlyNowcaster):
    """Monthly AR on the target with recursive ragged-edge extension."""

    target_lags: tuple[int, ...] = (1, 2, 12)
    min_obs: int = 120
    _name: str = "M-AR"

    def fit(self, info: MonthlyInformationSet) -> "MonthlyARNowcaster":
        y = info.released_target()
        self._released = y
        self._max_lag = max(self.target_lags)
        self._model = None
        self.train_rmse_ = None
        if len(y) < self.min_obs:
            return self
        D = pd.DataFrame({"y": y})
        for k in self.target_lags:
            D[f"y{k}"] = y.shift(k)
        D = D.dropna()
        if len(D) < self.min_obs:
            return self
        X = D[[f"y{k}" for k in self.target_lags]].to_numpy()
        yt = D["y"].to_numpy()
        self._model = LinearRegression().fit(X, yt)
        resid = yt - self._model.predict(X)
        self.train_rmse_ = float(np.sqrt(np.mean(resid**2)))
        return self

    def _extended_target(self, target_month: pd.Timestamp) -> pd.Series:
        y = self._released
        idx = pd.date_range(y.index[0], pd.Timestamp(target_month), freq="MS")
        arr = y.reindex(idx).to_numpy(dtype=float).copy()
        if self._model is None:
            return pd.Series(arr, index=idx).ffill()
        lag_list = list(self.target_lags)
        for i in range(len(arr)):
            if np.isfinite(arr[i]):
                continue
            row = []
            for lag in lag_list:
                if i - lag < 0 or not np.isfinite(arr[i - lag]):
                    row = []
                    break
                row.append(arr[i - lag])
            if not row:
                if i > 0 and np.isfinite(arr[i - 1]):
                    arr[i] = arr[i - 1]
                continue
            arr[i] = float(self._model.predict(np.asarray(row, dtype=float).reshape(1, -1))[0])
        return pd.Series(arr, index=idx).ffill()

    def nowcast(self, info: MonthlyInformationSet) -> MonthlyNowcastResult:
        if self._released.empty:
            return MonthlyNowcastResult(mean=float("nan"), model=self.name)
        ext = self._extended_target(info.target_period)
        yhat = ext.get(info.target_period, np.nan)
        return MonthlyNowcastResult(mean=float(yhat), model=self.name)


@dataclass
class MonthlyARXNowcaster(BaseMonthlyNowcaster):
    """Parsimonious monthly ARX with recursive ragged-edge projection."""

    predictors: Sequence[str]
    predictor_lags: Mapping[str, int] | int = 0
    target_lags: tuple[int, ...] = (1, 2, 12)
    predictor_ar_order: int = 2
    min_obs: int = 120
    standardize: bool = True
    _name: str = "ARX-M"

    def _lag_of(self, var: str) -> int:
        if isinstance(self.predictor_lags, Mapping):
            return int(self.predictor_lags.get(var, 0))
        return int(self.predictor_lags)

    def fit(self, info: MonthlyInformationSet) -> "MonthlyARXNowcaster":
        y = info.released_target()
        self._released = y
        self._target = info.target
        self._vars = [v for v in self.predictors if v in info.monthly.columns and v != info.target]
        self._feature_names: list[str] = []
        self._predictor_feature_names: list[str] = []
        self._model = None
        self._x_mean = None
        self._x_scale = None
        self.coef_ = pd.Series(dtype=float)
        self.coef_raw_ = pd.Series(dtype=float)
        self.intercept_raw_ = np.nan
        self.train_rmse_ = None
        if len(y) < self.min_obs or not self._vars:
            return self

        D = pd.DataFrame({"y": y})
        feature_names: list[str] = []
        predictor_feature_names: list[str] = []
        for lag in self.target_lags:
            name = f"y_lag{lag}"
            D[name] = y.shift(lag)
            feature_names.append(name)
        for var in self._vars:
            lag = self._lag_of(var)
            name = f"x__{var}__lag{lag}"
            D[name] = info.monthly[var].shift(lag)
            feature_names.append(name)
            predictor_feature_names.append(name)

        D = D.replace([np.inf, -np.inf], np.nan).dropna()
        if len(D) < self.min_obs:
            return self

        X_raw = D[feature_names].to_numpy(dtype=float)
        yt = D["y"].to_numpy(dtype=float)
        x_mean = X_raw.mean(axis=0)
        x_scale = X_raw.std(axis=0, ddof=0)
        x_scale = np.where(x_scale <= 0.0, 1.0, x_scale)
        X = (X_raw - x_mean) / x_scale if self.standardize else X_raw

        model = LinearRegression().fit(X, yt)
        fitted = model.predict(X)
        resid = yt - fitted
        self.train_rmse_ = float(np.sqrt(np.mean(resid**2)))
        self._model = model
        self._x_mean = x_mean
        self._x_scale = x_scale
        self._feature_names = feature_names
        self._predictor_feature_names = predictor_feature_names
        self.coef_ = pd.Series(model.coef_, index=feature_names, dtype=float)
        if self.standardize:
            coef_raw = model.coef_ / x_scale
            intercept_raw = float(model.intercept_ - np.dot(x_mean / x_scale, model.coef_))
        else:
            coef_raw = model.coef_
            intercept_raw = float(model.intercept_)
        self.coef_raw_ = pd.Series(coef_raw, index=feature_names, dtype=float)
        self.intercept_raw_ = intercept_raw
        return self

    def _target_tail(self, info: MonthlyInformationSet, end_month: pd.Timestamp) -> pd.Series:
        order = 2 if len(self.target_lags) < 2 else min(4, max(2, len(self.target_lags)))
        return _extend_series_with_ar(info.y, end_month, order=order, min_obs=24)

    def _predictor_tail(
        self,
        info: MonthlyInformationSet,
        *,
        var: str,
        end_month: pd.Timestamp,
    ) -> pd.Series:
        return _extend_series_with_ar(
            info.monthly[var],
            end_month,
            order=self.predictor_ar_order,
            min_obs=24,
        )

    def nowcast(self, info: MonthlyInformationSet) -> MonthlyNowcastResult:
        if self._model is None or self._x_mean is None or self._x_scale is None:
            return MonthlyNowcastResult(mean=float("nan"), model=self.name)

        target_month = pd.Timestamp(info.target_period)
        observed = info.monthly[self._target].get(target_month, np.nan)
        if np.isfinite(observed):
            return MonthlyNowcastResult(mean=float(observed), model=self.name)

        y_tail = self._target_tail(info, target_month)
        predictor_tails: dict[str, pd.Series] = {}
        row: list[float] = []
        raw_features: dict[str, float] = {}

        for lag in self.target_lags:
            name = f"y_lag{lag}"
            val = y_tail.get(target_month - pd.DateOffset(months=lag), np.nan)
            if not np.isfinite(val):
                return MonthlyNowcastResult(mean=float("nan"), model=self.name)
            row.append(float(val))
            raw_features[name] = float(val)

        for var in self._vars:
            lag = self._lag_of(var)
            look_month = target_month - pd.DateOffset(months=lag)
            tail = predictor_tails.get(var)
            if tail is None:
                tail = self._predictor_tail(info, var=var, end_month=look_month)
                predictor_tails[var] = tail
            val = tail.get(look_month, np.nan)
            if not np.isfinite(val):
                return MonthlyNowcastResult(mean=float("nan"), model=self.name)
            name = f"x__{var}__lag{lag}"
            row.append(float(val))
            raw_features[name] = float(val)

        x_raw = np.asarray(row, dtype=float)
        x_fit = (x_raw - self._x_mean) / self._x_scale if self.standardize else x_raw
        if not np.all(np.isfinite(x_fit)):
            return MonthlyNowcastResult(mean=float("nan"), model=self.name)

        yhat = float(self._model.predict(x_fit.reshape(1, -1))[0])
        contrib = pd.Series(x_fit * self.coef_.reindex(self._feature_names).to_numpy(dtype=float), index=self._feature_names)
        predictor_contrib = contrib.reindex(self._predictor_feature_names).dropna()
        extra: dict[str, Any] = {
            "n_predictors": int(len(self._vars)),
            "train_rmse": float(self.train_rmse_) if self.train_rmse_ is not None else np.nan,
            "top_predictor": predictor_contrib.abs().idxmax() if not predictor_contrib.empty else "",
        }
        if len(self._predictor_feature_names) == 1:
            feature = self._predictor_feature_names[0]
            extra["coef_x"] = float(self.coef_raw_.get(feature, np.nan))
            extra["coef_x_std"] = float(self.coef_.get(feature, np.nan))
            extra["coef_sign"] = int(np.sign(extra["coef_x_std"])) if np.isfinite(extra["coef_x_std"]) else 0
            extra["x_value"] = float(raw_features.get(feature, np.nan))
        return MonthlyNowcastResult(mean=yhat, model=self.name, extra=extra)


@dataclass
class MonthlyPooledBridgeNowcaster(BaseMonthlyNowcaster):
    """Pooled single-indicator monthly bridge on the monthly GDP target."""

    target_lags: tuple[int, ...] = (1, 2)
    monthly_vars: Sequence[str] | None = None
    indicator_ar_order: int = 2
    min_obs: int = 120
    _name: str = "Bridge-M"

    def fit(self, info: MonthlyInformationSet) -> "MonthlyPooledBridgeNowcaster":
        y = info.released_target()
        self._y = y
        self._vars = [v for v in (self.monthly_vars or list(info.monthly.columns)) if v != info.target]
        self._members: list[dict[str, Any]] = []
        pooled_train: list[pd.Series] = []
        for var in self._vars:
            D = pd.DataFrame({"y": y, "x": info.monthly[var]})
            for lag in self.target_lags:
                D[f"y{lag}"] = y.shift(lag)
            D = D.dropna()
            if len(D) < self.min_obs:
                continue
            cols = [f"y{lag}" for lag in self.target_lags] + ["x"]
            X = np.column_stack([np.ones(len(D))] + [D[c].to_numpy() for c in cols])
            beta, resid = _ols(X, D["y"].to_numpy())
            rmse = float(np.sqrt(np.mean(resid**2)))
            self._members.append(
                {
                    "var": var,
                    "beta": beta,
                    "rmse": rmse,
                    "x": info.monthly[var],
                }
            )
            fitted = pd.Series(X @ beta, index=D.index, name=var)
            pooled_train.append(fitted)
        self.train_rmse_ = None
        if pooled_train:
            panel = pd.concat(pooled_train, axis=1, sort=False)
            weights = np.asarray([1.0 / max(mem["rmse"], 1e-6) for mem in self._members], dtype=float)
            weights = weights / weights.sum()
            common = panel.dropna(how="all").index.intersection(y.index)
            pooled = []
            truth = []
            for ts in common:
                vals = panel.loc[ts].to_numpy(dtype=float)
                ok = np.isfinite(vals)
                if not ok.any():
                    continue
                w = weights[ok]
                w = w / w.sum()
                pooled.append(float(np.dot(w, vals[ok])))
                truth.append(float(y.loc[ts]))
            if pooled:
                err = np.asarray(pooled) - np.asarray(truth)
                self.train_rmse_ = float(np.sqrt(np.mean(err**2)))
        return self

    def _target_tail(self, info: MonthlyInformationSet, end_month: pd.Timestamp) -> pd.Series:
        return _extend_series_with_ar(info.y, end_month, order=2, min_obs=24)

    def nowcast(self, info: MonthlyInformationSet) -> MonthlyNowcastResult:
        if not self._members:
            return MonthlyNowcastResult(mean=float("nan"), model=self.name)
        target_month = pd.Timestamp(info.target_period)
        y_tail = self._target_tail(info, target_month)
        preds: list[float] = []
        weights: list[float] = []
        for mem in self._members:
            x_tail = _extend_series_with_ar(
                mem["x"],
                target_month,
                order=self.indicator_ar_order,
                min_obs=24,
            )
            row = [1.0]
            ok = True
            for lag in self.target_lags:
                val = y_tail.get(target_month - pd.DateOffset(months=lag), np.nan)
                if not np.isfinite(val):
                    ok = False
                    break
                row.append(float(val))
            x0 = x_tail.get(target_month, np.nan)
            if not ok or not np.isfinite(x0):
                continue
            row.append(float(x0))
            preds.append(float(np.dot(np.asarray(row, dtype=float), mem["beta"])))
            weights.append(1.0 / max(mem["rmse"], 1e-6))
        if not preds:
            return MonthlyNowcastResult(mean=float("nan"), model=self.name)
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        return MonthlyNowcastResult(
            mean=float(np.dot(w, preds)),
            model=self.name,
            extra={"n_members": len(preds)},
        )


@dataclass
class MonthlyBlockRidgeNowcaster(BaseMonthlyNowcaster):
    """Factor-augmented monthly GDP nowcaster built from timely indicator blocks."""

    blocks: Mapping[str, Sequence[str]]
    target_lags: tuple[int, ...] = (1, 2, 12)
    n_pcs_per_block: int = 1
    predictor_ar_order: int = 2
    min_obs: int = 120
    _name: str = "Block-Ridge"

    def fit(self, info: MonthlyInformationSet) -> "MonthlyBlockRidgeNowcaster":
        y = info.released_target()
        self._y = y
        self._target = info.target
        self._block_models: dict[str, dict[str, Any]] = {}
        self._feature_names: list[str] = []
        self.train_rmse_ = None
        if len(y) < self.min_obs:
            self._ridge = None
            return self

        train_end = y.index[-1]
        for block, vars_ in self.blocks.items():
            hist = info.monthly[list(vars_)].loc[:train_end].copy()
            hist = hist.loc[:, hist.notna().sum() >= max(36, self.min_obs // 3)]
            if hist.shape[1] == 0:
                continue
            mu = hist.mean()
            sd = hist.std(ddof=0).replace(0.0, 1.0)
            Z = ((hist - mu) / sd).fillna(0.0)
            k = int(min(self.n_pcs_per_block, Z.shape[1]))
            pca = PCA(n_components=k).fit(Z.to_numpy())
            pcs = pd.DataFrame(
                pca.transform(Z.to_numpy()),
                index=Z.index,
                columns=[f"{block}_pc{i+1}" for i in range(k)],
            )
            self._block_models[block] = {
                "vars": list(hist.columns),
                "mu": mu,
                "sd": sd,
                "pca": pca,
                "pcs": pcs,
            }

        rows: list[np.ndarray] = []
        targets: list[float] = []
        for ts in y.index:
            row = self._training_feature_row(ts)
            if row is None:
                continue
            rows.append(row)
            targets.append(float(y.loc[ts]))

        if len(rows) < self.min_obs:
            self._ridge = None
            return self
        X = np.vstack(rows)
        yt = np.asarray(targets, dtype=float)
        self._ridge = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X, yt)
        fitted = self._ridge.predict(X)
        self.train_rmse_ = float(np.sqrt(np.mean((yt - fitted) ** 2)))
        return self

    def _training_feature_row(self, ts: pd.Timestamp) -> np.ndarray | None:
        row: list[float] = []
        for lag in self.target_lags:
            val = self._y.shift(lag).get(ts, np.nan)
            if not np.isfinite(val):
                return None
            row.append(float(val))
        feature_names = [f"y_lag{lag}" for lag in self.target_lags]

        for block, obj in self._block_models.items():
            pcs = obj["pcs"]
            cur = pcs.get(pcs.columns, pd.Series(dtype=float))
            for col in pcs.columns:
                for lag in (0, 1):
                    val = pcs[col].shift(lag).get(ts, np.nan)
                    if not np.isfinite(val):
                        return None
                    row.append(float(val))
                    feature_names.append(f"{col}_lag{lag}")
        self._feature_names = feature_names
        return np.asarray(row, dtype=float)

    def _project_block(
        self,
        info: MonthlyInformationSet,
        *,
        block: str,
        ts: pd.Timestamp,
    ) -> np.ndarray:
        obj = self._block_models[block]
        vals = []
        for var in obj["vars"]:
            tail = _extend_series_with_ar(
                info.monthly[var],
                ts,
                order=self.predictor_ar_order,
                min_obs=24,
            )
            vals.append(float(tail.get(ts, np.nan)))
        x = pd.Series(vals, index=obj["vars"], dtype=float)
        z = ((x - obj["mu"]) / obj["sd"]).fillna(0.0).to_numpy(dtype=float)
        return np.asarray(obj["pca"].transform(z.reshape(1, -1))[0], dtype=float)

    def _target_tail(self, info: MonthlyInformationSet, end_month: pd.Timestamp) -> pd.Series:
        return _extend_series_with_ar(info.y, end_month, order=2, min_obs=24)

    def nowcast(self, info: MonthlyInformationSet) -> MonthlyNowcastResult:
        if getattr(self, "_ridge", None) is None:
            return MonthlyNowcastResult(mean=float("nan"), model=self.name)

        target_month = pd.Timestamp(info.target_period)
        y_tail = self._target_tail(info, target_month)
        row: list[float] = []
        feature_names: list[str] = [f"y_lag{lag}" for lag in self.target_lags]
        for lag in self.target_lags:
            val = y_tail.get(target_month - pd.DateOffset(months=lag), np.nan)
            if not np.isfinite(val):
                return MonthlyNowcastResult(mean=float("nan"), model=self.name)
            row.append(float(val))

        for block, obj in self._block_models.items():
            for lag in (0, 1):
                ts = target_month - pd.DateOffset(months=lag)
                pc_vals = self._project_block(info, block=block, ts=ts)
                row.extend(pc_vals.tolist())
                feature_names.extend([f"{col}_lag{lag}" for col in obj["pcs"].columns])

        x = np.asarray(row, dtype=float)
        if not np.all(np.isfinite(x)):
            return MonthlyNowcastResult(mean=float("nan"), model=self.name)

        yhat = float(self._ridge.predict(x.reshape(1, -1))[0])
        contrib = pd.Series(x * self._ridge.coef_, index=feature_names)
        grouped = pd.Series(
            {
                "AR": float(contrib[[c for c in contrib.index if c.startswith("y_lag")]].sum()),
                **{
                    block: float(contrib[[c for c in contrib.index if c.startswith(f"{block}_pc")]].sum())
                    for block in self._block_models
                },
            }
        )
        grouped["Intercept"] = float(self._ridge.intercept_)
        grouped = grouped.sort_values(key=lambda s: s.abs(), ascending=False)
        return MonthlyNowcastResult(
            mean=yhat,
            model=self.name,
            extra={
                "alpha": float(self._ridge.alpha_),
                "feature_contributions": contrib.sort_values(key=lambda s: s.abs(), ascending=False),
                "block_contributions": grouped,
                "n_train": int(len(self._y)),
            },
        )


@dataclass
class MonthlyAdaptiveComboNowcaster(BaseMonthlyNowcaster):
    """Simple adaptive combination of monthly AR, Bridge, and block-factor models."""

    members: Mapping[str, BaseMonthlyNowcaster]
    _name: str = "Adaptive-Combo"

    def fit(self, info: MonthlyInformationSet) -> "MonthlyAdaptiveComboNowcaster":
        self._fitted: dict[str, BaseMonthlyNowcaster] = {}
        rmse_proxy: dict[str, float] = {}
        for name, model in self.members.items():
            fitted = deepcopy(model).fit(info)
            self._fitted[name] = fitted
            score = getattr(fitted, "train_rmse_", None)
            if score is not None and np.isfinite(score) and score > 0:
                rmse_proxy[name] = float(score)
        if rmse_proxy:
            w = {k: 1.0 / v for k, v in rmse_proxy.items()}
            total = float(sum(w.values()))
            self._weights = {k: v / total for k, v in w.items()}
            self.train_rmse_ = float(
                np.average(
                    [rmse_proxy[k] for k in self._weights],
                    weights=[self._weights[k] for k in self._weights],
                )
            )
        else:
            eq = 1.0 / max(len(self._fitted), 1)
            self._weights = {k: eq for k in self._fitted}
            self.train_rmse_ = None
        return self

    def nowcast(self, info: MonthlyInformationSet) -> MonthlyNowcastResult:
        preds: dict[str, float] = {}
        for name, model in self._fitted.items():
            res = model.nowcast(info)
            if np.isfinite(res.mean):
                preds[name] = float(res.mean)
        if not preds:
            return MonthlyNowcastResult(mean=float("nan"), model=self.name)
        weights = {k: self._weights.get(k, 0.0) for k in preds}
        if sum(weights.values()) == 0:
            eq = 1.0 / len(preds)
            weights = {k: eq for k in preds}
        else:
            total = float(sum(weights.values()))
            weights = {k: v / total for k, v in weights.items()}
        yhat = float(sum(weights[k] * preds[k] for k in preds))
        return MonthlyNowcastResult(
            mean=yhat,
            model=self.name,
            extra={"weights": weights, "member_preds": preds},
        )


@dataclass
class MonthlyAverageComboNowcaster(BaseMonthlyNowcaster):
    """Equal-weight average of a set of monthly nowcasters."""

    members: Mapping[str, BaseMonthlyNowcaster]
    _name: str = "Combo50"

    def fit(self, info: MonthlyInformationSet) -> "MonthlyAverageComboNowcaster":
        self._fitted = {name: deepcopy(model).fit(info) for name, model in self.members.items()}
        scores = [
            float(model.train_rmse_)
            for model in self._fitted.values()
            if getattr(model, "train_rmse_", None) is not None and np.isfinite(model.train_rmse_)
        ]
        self.train_rmse_ = float(np.mean(scores)) if scores else None
        return self

    def nowcast(self, info: MonthlyInformationSet) -> MonthlyNowcastResult:
        preds: dict[str, float] = {}
        for name, model in self._fitted.items():
            res = model.nowcast(info)
            if np.isfinite(res.mean):
                preds[name] = float(res.mean)
        if not preds:
            return MonthlyNowcastResult(mean=float("nan"), model=self.name)
        yhat = float(np.mean(list(preds.values())))
        weights = {name: 1.0 / len(preds) for name in preds}
        return MonthlyNowcastResult(
            mean=yhat,
            model=self.name,
            extra={"weights": weights, "member_preds": preds},
        )


# ---------------------------------------------------------------------------
# Backtesting and evaluation
# ---------------------------------------------------------------------------
def _month_lead_rows(
    month: pd.Timestamp,
    *,
    panel: MetadataPanel,
    target: str,
    leads: Sequence[int],
    models: Mapping[str, BaseMonthlyNowcaster],
    min_train: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    engine = MonthlyRealtimeEngine(panel)
    y_true = panel.monthly[target].get(month, np.nan)
    if not np.isfinite(y_true):
        return [], {}

    rows: list[dict[str, Any]] = []
    failures: dict[str, int] = {}
    delay = panel.delay_of(target)
    pub = month_publication_date(month, delay)
    for lead in leads:
        origin = pub + pd.Timedelta(days=int(lead))
        info = engine.information_set(origin, target, target_period=month)
        if pd.notna(info.monthly.at[month, target]):
            continue
        if info.released_target().size < min_train:
            continue
        for name, model in models.items():
            try:
                res = deepcopy(model).fit(info).nowcast(info)
                y_hat = float(res.mean)
                extra = dict(res.extra or {})
            except Exception as exc:  # pragma: no cover
                y_hat = float("nan")
                extra = {"error": f"{type(exc).__name__}: {exc}"}
                failures[name] = failures.get(name, 0) + 1
            if not np.isfinite(y_hat):
                failures[name] = failures.get(name, 0) + 1
            row = {
                "target": target,
                "ref_month": pd.Timestamp(month),
                "origin_date": pd.Timestamp(origin),
                "days_to_publication": int(lead),
                "model": name,
                "y_true": float(y_true),
                "y_hat": float(y_hat),
            }
            for key, value in extra.items():
                if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                    row[str(key)] = value
            rows.append(row)
    return rows, failures


def run_monthly_selected_leads_backtest(
    panel: MetadataPanel,
    *,
    target: str,
    models: Mapping[str, BaseMonthlyNowcaster],
    leads: Sequence[int],
    eval_start: str | pd.Timestamp,
    eval_end: str | pd.Timestamp | None = None,
    min_train: int = 120,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Pseudo-real-time backtest for a monthly target at a few selected leads."""

    months = panel.monthly.index
    start = pd.Timestamp(eval_start)
    end = months[-1] if eval_end is None else pd.Timestamp(eval_end)
    eval_months = [m for m in months if start <= m <= end]

    if n_jobs == 1:
        results = [
            _month_lead_rows(
                m,
                panel=panel,
                target=target,
                leads=leads,
                models=models,
                min_train=min_train,
            )
            for m in eval_months
        ]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_month_lead_rows)(
                m,
                panel=panel,
                target=target,
                leads=leads,
                models=models,
                min_train=min_train,
            )
            for m in eval_months
        )

    rows: list[dict[str, Any]] = []
    failures: dict[str, int] = {}
    for rs, fs in results:
        rows.extend(rs)
        for k, v in fs.items():
            failures[k] = failures.get(k, 0) + v

    if failures:
        print(
            "[run_monthly_selected_leads_backtest] failed nowcasts by model -> "
            + ", ".join(f"{k}: {v}" for k, v in failures.items())
        )
    return pd.DataFrame(rows).sort_values(["ref_month", "days_to_publication", "model"]).reset_index(drop=True)


def monthly_selected_leads_metrics(
    df: pd.DataFrame,
    *,
    models: Sequence[str] | None = None,
    exclude_years: Sequence[int] = (2020, 2021),
) -> pd.DataFrame:
    """Overall and lead-specific RMSE/MAE summaries for monthly backtests."""

    if df.empty:
        return pd.DataFrame()
    data = df.copy()
    if models is not None:
        data = data[data["model"].isin(list(models))]
    if exclude_years:
        data = data[~data["ref_month"].dt.year.isin(list(exclude_years))]
    data = data[np.isfinite(data["y_true"]) & np.isfinite(data["y_hat"])].copy()
    if data.empty:
        return pd.DataFrame()

    data["error"] = data["y_hat"] - data["y_true"]
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


def attach_monthly_relative_rmse(metrics: pd.DataFrame, *, baseline: str) -> pd.DataFrame:
    """Add RMSE relative to a monthly baseline model."""

    out = metrics.copy()
    numeric = out[out["days_to_publication"] != "overall"].copy()
    if numeric.empty:
        return out
    numeric = _relative_rmse_group(numeric, baseline=baseline)
    overall = out[out["days_to_publication"] == "overall"].copy()
    base_all = overall.loc[overall["model"] == baseline, "rmse"]
    if len(base_all):
        overall["relative_rmse"] = overall["rmse"] / float(base_all.iloc[0])
    return pd.concat([overall, numeric], ignore_index=True)


def quarterly_mean_from_monthly_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Quarterly means implied by a monthly-target backtest."""

    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["ref_quarter"] = (
        out["ref_month"].dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize().map(lambda ts: ts.replace(day=1))
    )
    agg = (
        out.groupby(["ref_quarter", "days_to_publication", "model"], as_index=False)
        .agg(y_true=("y_true", "mean"), y_hat=("y_hat", "mean"), n_months=("ref_month", "size"))
    )
    return agg


def run_quarterly_bridge_from_monthly_model(
    panel: MetadataPanel,
    *,
    quarterly_target: str,
    monthly_target: str,
    monthly_model: BaseMonthlyNowcaster,
    leads: Sequence[int],
    eval_start: str | pd.Timestamp,
    eval_end: str | pd.Timestamp | None = None,
    min_monthly_train: int = 120,
    min_quarterly_train: int = 24,
    bridge_target_lags: tuple[int, ...] = (1,),
) -> pd.DataFrame:
    """Quarterly GDP nowcast from monthly-target model plus a small bridge step."""

    q_index = panel.quarterly.index
    q_start = pd.Timestamp(eval_start)
    q_end = q_index[-1] if eval_end is None else pd.Timestamp(eval_end)
    eval_quarters = [q for q in q_index if q_start <= q <= q_end and np.isfinite(panel.quarterly[quarterly_target].get(q, np.nan))]
    if not eval_quarters:
        return pd.DataFrame()

    monthly_engine = MonthlyRealtimeEngine(panel)
    quarterly_engine = RealtimeEngine(panel)
    q_delay = panel.delay_of(quarterly_target)
    rows: list[dict[str, Any]] = []

    for quarter in eval_quarters:
        pub = quarter_publication_date(quarter, q_delay)
        months = _months_in_quarter(quarter)
        for lead in leads:
            origin = pub + pd.Timedelta(days=int(lead))
            m_info = monthly_engine.information_set(origin, monthly_target, target_period=months[-1])
            if m_info.released_target().size < min_monthly_train:
                continue
            fitted_monthly = deepcopy(monthly_model).fit(m_info)

            month_values: list[float] = []
            observed_months = 0
            for month in months:
                observed = m_info.monthly[monthly_target].get(month, np.nan)
                if np.isfinite(observed):
                    month_values.append(float(observed))
                    observed_months += 1
                    continue
                month_info = MonthlyInformationSet(
                    monthly=m_info.monthly,
                    target=monthly_target,
                    origin=pd.Timestamp(origin),
                    target_period=pd.Timestamp(month),
                    metadata=panel,
                )
                month_pred = fitted_monthly.nowcast(month_info).mean
                if not np.isfinite(month_pred):
                    month_values = []
                    break
                month_values.append(float(month_pred))
            if len(month_values) != 3:
                continue
            monthly_proxy = float(np.mean(month_values))

            q_info = quarterly_engine.information_set(origin, quarterly_target, target_period=quarter)
            q_hist = q_info.quarterly[quarterly_target].dropna()
            q_proxy_hist = quarterly_mean_from_monthly_series(m_info.monthly[monthly_target])
            train = pd.DataFrame({"y": q_hist, "mproxy": q_proxy_hist.reindex(q_hist.index)})
            for lag_q in bridge_target_lags:
                train[f"y_lag{lag_q}"] = q_hist.shift(lag_q)
            train = train.dropna()
            if len(train) < min_quarterly_train:
                continue

            cols = [f"y_lag{lag_q}" for lag_q in bridge_target_lags] + ["mproxy"]
            X = np.column_stack([np.ones(len(train))] + [train[c].to_numpy(dtype=float) for c in cols])
            beta, resid = _ols(X, train["y"].to_numpy(dtype=float))
            bridge_rmse = float(np.sqrt(np.mean(resid**2)))

            row = [1.0]
            ok = True
            for lag_q in bridge_target_lags:
                val = q_hist.get(quarter - pd.DateOffset(months=3 * lag_q), np.nan)
                if not np.isfinite(val):
                    ok = False
                    break
                row.append(float(val))
            if not ok:
                continue
            row.append(monthly_proxy)
            y_hat = float(np.dot(np.asarray(row, dtype=float), beta))
            rows.append(
                {
                    "target": quarterly_target,
                    "ref_quarter": pd.Timestamp(quarter),
                    "origin_date": pd.Timestamp(origin),
                    "days_to_publication": int(lead),
                    "model": monthly_model.name + "-QBridge",
                    "y_true": float(panel.quarterly[quarterly_target].loc[quarter]),
                    "y_hat": y_hat,
                    "monthly_proxy": monthly_proxy,
                    "n_observed_months": int(observed_months),
                    "n_nowcast_months": int(3 - observed_months),
                    "bridge_train_rmse": bridge_rmse,
                    "bridge_beta_proxy": float(beta[-1]),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["ref_quarter", "days_to_publication", "model"]).reset_index(drop=True)


def screen_monthly_indicators(
    panel: MetadataPanel,
    *,
    target: str,
    baseline_backtest: pd.DataFrame,
    baseline_model: str = "M-AR",
    leads: Sequence[int],
    eval_start: str | pd.Timestamp,
    eval_end: str | pd.Timestamp | None = None,
    candidate_vars: Sequence[str] | None = None,
    predictor_lags: Sequence[int] = (0, 1, 2),
    min_train: int = 120,
    exclude_years: Sequence[int] = (2020, 2021),
    recent_start: str | pd.Timestamp = "2023-01-01",
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Real-time candidate screen for one-variable monthly ARX models."""

    target_delay = panel.delay_of(target)
    candidates = list(candidate_vars) if candidate_vars is not None else [
        c for c in panel.monthly.columns if c != target and panel.monthly[c].notna().sum() >= min_train
    ]
    baseline = baseline_backtest.copy()
    baseline = baseline[baseline["model"] == baseline_model].copy()
    if exclude_years:
        baseline = baseline[~baseline["ref_month"].dt.year.isin(list(exclude_years))]
    baseline = baseline[np.isfinite(baseline["y_hat"]) & np.isfinite(baseline["y_true"])].copy()

    periods: dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]] = {
        "full_excl_covid": (None, None),
        "pre_2020": (None, pd.Timestamp("2019-12-01")),
        "recent": (pd.Timestamp(recent_start), None),
    }

    def _period_filter(df: pd.DataFrame, lo: pd.Timestamp | None, hi: pd.Timestamp | None) -> pd.DataFrame:
        out = df
        if lo is not None:
            out = out[out["ref_month"] >= lo]
        if hi is not None:
            out = out[out["ref_month"] <= hi]
        return out

    def _one_candidate(var: str, lag: int) -> list[dict[str, Any]]:
        model_name = f"{var}[L{lag}]"
        model = MonthlyARXNowcaster(
            predictors=[var],
            predictor_lags={var: lag},
            min_obs=min_train,
            _name=model_name,
        )
        bt = run_monthly_selected_leads_backtest(
            panel,
            target=target,
            models={model_name: model},
            leads=leads,
            eval_start=eval_start,
            eval_end=eval_end,
            min_train=min_train,
            n_jobs=1,
        )
        if bt.empty:
            return []
        if exclude_years:
            bt = bt[~bt["ref_month"].dt.year.isin(list(exclude_years))]
        bt = bt[np.isfinite(bt["y_hat"]) & np.isfinite(bt["y_true"])].copy()
        if bt.empty:
            return []

        coef_series = bt["coef_x_std"] if "coef_x_std" in bt.columns else pd.Series(dtype=float)
        coef_series = coef_series.replace([np.inf, -np.inf], np.nan).dropna()
        pos_share = float((coef_series > 0).mean()) if len(coef_series) else np.nan
        neg_share = float((coef_series < 0).mean()) if len(coef_series) else np.nan
        sign_share = float(np.nanmax([pos_share, neg_share])) if len(coef_series) else np.nan
        coef_mean_std = float(coef_series.mean()) if len(coef_series) else np.nan
        coef_mean_raw = float(bt["coef_x"].replace([np.inf, -np.inf], np.nan).dropna().mean()) if "coef_x" in bt.columns else np.nan

        rows: list[dict[str, Any]] = []
        for period_label, (lo, hi) in periods.items():
            cand_period = _period_filter(bt, lo, hi)
            base_period = _period_filter(baseline, lo, hi)
            merged = cand_period.merge(
                base_period[["ref_month", "days_to_publication", "y_hat"]].rename(columns={"y_hat": "y_hat_base"}),
                on=["ref_month", "days_to_publication"],
                how="inner",
            )
            if merged.empty:
                continue
            for lead in sorted(set(merged["days_to_publication"])):
                g = merged[merged["days_to_publication"] == lead].copy()
                if g.empty:
                    continue
                err = g["y_hat"] - g["y_true"]
                err_base = g["y_hat_base"] - g["y_true"]
                rmse = float(np.sqrt(np.mean(err**2)))
                base_rmse = float(np.sqrt(np.mean(err_base**2)))
                rows.append(
                    {
                        "variable": var,
                        "predictor_lag": int(lag),
                        "delay_days": int(panel.delay_of(var)),
                        "delay_advantage_days": int(target_delay - panel.delay_of(var)),
                        "group": panel.group_of(var),
                        "model": model_name,
                        "period": period_label,
                        "days_to_publication": int(lead),
                        "n": int(len(g)),
                        "rmse": rmse,
                        "baseline_rmse": base_rmse,
                        "relative_rmse": rmse / base_rmse if base_rmse > 0 else np.nan,
                        "coef_mean_std": coef_mean_std,
                        "coef_mean_raw": coef_mean_raw,
                        "sign_share": sign_share,
                    }
                )
        return rows

    specs = [(var, int(lag)) for var in candidates for lag in predictor_lags]
    if n_jobs == 1:
        rows = [_one_candidate(var, lag) for var, lag in specs]
    else:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_one_candidate)(var, lag) for var, lag in specs
        )
    flat = [row for block in rows for row in block]
    if not flat:
        return pd.DataFrame()
    return pd.DataFrame(flat).sort_values(
        ["period", "days_to_publication", "relative_rmse", "delay_days", "variable", "predictor_lag"]
    ).reset_index(drop=True)


def select_stable_monthly_indicators(
    screen: pd.DataFrame,
    *,
    prefer_periods: Sequence[str] = ("pre_2020", "recent"),
    prefer_leads: Sequence[int] = (-30, -7, -1),
    max_features: int = 5,
    max_per_group: int = 1,
    min_sign_share: float = 0.75,
    max_delay_days: int | None = None,
) -> pd.DataFrame:
    """Select a small stable ARX block from the real-time screen."""

    if screen.empty:
        return pd.DataFrame()
    data = screen.copy()
    data = data[data["period"].isin(list(prefer_periods)) & data["days_to_publication"].isin(list(prefer_leads))].copy()
    if max_delay_days is not None:
        data = data[data["delay_days"] <= max_delay_days]
    if data.empty:
        return pd.DataFrame()

    grouped = (
        data.groupby(["variable", "predictor_lag", "group", "delay_days", "delay_advantage_days"], as_index=False)
        .agg(
            screen_score=("relative_rmse", "mean"),
            best_rel_rmse=("relative_rmse", "min"),
            helpful_share=("relative_rmse", lambda s: float(np.mean(np.asarray(s, dtype=float) < 1.0))),
            min_n=("n", "min"),
            sign_share=("sign_share", "max"),
            coef_mean_std=("coef_mean_std", "mean"),
            coef_mean_raw=("coef_mean_raw", "mean"),
        )
    )
    grouped["stability_penalty"] = grouped["sign_share"].map(lambda s: max(0.0, min_sign_share - float(s)) if np.isfinite(s) else 1.0)
    grouped["delay_penalty"] = grouped["delay_days"] / 1000.0
    grouped["selection_score"] = grouped["screen_score"] + 0.25 * grouped["stability_penalty"] + grouped["delay_penalty"]
    grouped["passes"] = (
        (grouped["sign_share"] >= min_sign_share)
        & (grouped["helpful_share"] >= 0.5)
        & (grouped["best_rel_rmse"] < 1.0)
    )
    grouped = grouped.sort_values(
        ["passes", "selection_score", "screen_score", "delay_days", "variable", "predictor_lag"],
        ascending=[False, True, True, True, True, True],
    ).reset_index(drop=True)

    chosen: list[int] = []
    per_group: dict[str, int] = {}
    for idx, row in grouped.iterrows():
        if len(chosen) >= max_features:
            break
        group = str(row["group"])
        if per_group.get(group, 0) >= max_per_group:
            continue
        chosen.append(idx)
        per_group[group] = per_group.get(group, 0) + 1
    out = grouped.copy()
    out["selected"] = False
    out.loc[chosen, "selected"] = True
    return out


__all__ = [
    "BaseMonthlyNowcaster",
    "MonthlyAdaptiveComboNowcaster",
    "MonthlyAverageComboNowcaster",
    "MonthlyARNowcaster",
    "MonthlyARXNowcaster",
    "MonthlyBlockRidgeNowcaster",
    "MonthlyInformationSet",
    "MonthlyNowcastResult",
    "MonthlyPooledBridgeNowcaster",
    "MonthlyRealtimeEngine",
    "attach_monthly_relative_rmse",
    "month_publication_date",
    "monthly_selected_leads_metrics",
    "quarterly_mean_from_monthly_series",
    "quarterly_mean_from_monthly_backtest",
    "run_quarterly_bridge_from_monthly_model",
    "run_monthly_selected_leads_backtest",
    "screen_monthly_indicators",
    "select_stable_monthly_indicators",
]
