"""Project-local forecast model classes.

``DirectARXNowcaster`` lives HERE rather than in the MIDAS package: it is a plain
quarterly OLS projection (no mixed-frequency machinery), and keeping it in the
project means no wheel publish is needed to evolve it. It implements the MIDAS
``BaseNowcaster`` contract, so the horizon backtest harness drives it like any
packaged model.

For a target quarter h steps beyond the last released observation, the direct
projection

    y_{t+h} = a_h + rho_h(L) y_t + b_h' X_t + e_{t+h}

is re-estimated per horizon on the origin's masked information set (no
look-ahead): ``X_t`` are quarterly means of the observed monthly indicator
values in quarter t, anchored at the last quarter with a released target value.
Forecast-path regressors (e.g. the SPF's h-step-ahead consensus, a time-t
observable) enter like any other indicator column.

Direct vs iterated: the ladder deliberately carries both. ``ARNowcaster``
(MIDAS) is the iterated/recursive side - efficient when the one-step dynamics
are well specified, but it compounds any misspecification h times and cannot use
indicator regressors without a model for their future values. The direct form
trades some efficiency for robustness and needs only time-t regressors
(Marcellino, Stock & Watson 2006).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from MIDAS.base import BaseNowcaster, InformationSet, NowcastResult


def _quarter_stamp(p: pd.Period) -> pd.Timestamp:
    """MIDAS convention: a quarter is dated at the first day of its end month."""

    return p.to_timestamp(how="end").to_period("M").to_timestamp()


def _quarterly_means(monthly: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Quarterly mean of the OBSERVED monthly values per indicator (ragged-safe)."""

    cols = [c for c in columns if c in monthly.columns]
    if not cols:
        return pd.DataFrame()
    m = monthly[cols]
    q = m.groupby(pd.PeriodIndex(m.index, freq="Q")).mean()
    q.index = pd.DatetimeIndex([_quarter_stamp(p) for p in q.index])
    return q


@dataclass
class DirectARXNowcaster(BaseNowcaster):
    """Direct h-step projection of the target on its own lags + indicators."""

    indicators: Sequence[str] = field(default_factory=list)
    ar_lags: int = 1
    min_train: int = 24
    sample_start: str | None = None   # regime-consistent estimation (e.g. "2012-01-01")
    _name: str = "Direct-ARX"

    def fit(self, info: InformationSet) -> "DirectARXNowcaster":
        # All estimation happens in ``nowcast`` because the regression depends on
        # the horizon, which only ``info.target_period`` reveals. OLS is cheap.
        self._info = info
        return self

    # ------------------------------------------------------------------ #
    def _design(self, info: InformationSet, h: int):
        """Aligned (y_{t+h}, [1, y_t.., X_t]) training pairs + the base-quarter row."""

        y = info.y.dropna()
        if self.sample_start is not None:
            y = y.loc[self.sample_start:]
        if y.empty:
            return None
        Xq = _quarterly_means(info.monthly, self.indicators)

        qidx = pd.PeriodIndex(y.index, freq="Q")
        rows, targets = [], []
        for i in range(self.ar_lags - 1, len(y)):
            p_t = qidx[i]
            t_h = _quarter_stamp(p_t + h)
            if t_h not in y.index:
                continue
            ar_part = [float(y.iloc[i - j]) for j in range(self.ar_lags)]
            x_part = []
            ok = True
            for c in self.indicators:
                ts = _quarter_stamp(p_t)
                v = Xq.at[ts, c] if (len(Xq) and ts in Xq.index and c in Xq.columns) else np.nan
                if pd.isna(v):
                    ok = False
                    break
                x_part.append(float(v))
            if not ok:
                continue
            rows.append([1.0] + ar_part + x_part)
            targets.append(float(y.loc[t_h]))

        # base-quarter regressor row (the forecast input)
        p0 = qidx[-1]
        base_ar = [float(y.iloc[-1 - j]) for j in range(self.ar_lags)] if len(y) >= self.ar_lags else None
        base_x = []
        for c in self.indicators:
            ts = _quarter_stamp(p0)
            v = Xq.at[ts, c] if (len(Xq) and ts in Xq.index and c in Xq.columns) else np.nan
            if pd.isna(v):
                return y, None, None, None
            base_x.append(float(v))
        if base_ar is None:
            return y, None, None, None
        return y, np.asarray(rows), np.asarray(targets), np.asarray([1.0] + base_ar + base_x)

    def nowcast(self, info: InformationSet) -> NowcastResult:
        y = info.y.dropna()
        if y.empty or info.target_period is None:
            return NowcastResult(mean=float("nan"), model=self.name)

        last = pd.Period(y.index[-1], freq="Q")
        h = (pd.Period(info.target_period, freq="Q") - last).n
        if h <= 0:
            if info.target_period in y.index:
                return NowcastResult(mean=float(y.loc[info.target_period]), model=self.name)
            return NowcastResult(mean=float(y.iloc[-1]), model=self.name)

        out = self._design(info, h)
        if out is None:
            return NowcastResult(mean=float("nan"), model=self.name)
        y, X, t, x0 = out
        if X is None or len(t) < self.min_train:
            return NowcastResult(mean=float("nan"), model=self.name)

        beta, *_ = np.linalg.lstsq(X, t, rcond=None)
        resid = t - X @ beta
        dof = max(len(t) - len(beta), 1)
        std = float(np.sqrt(resid @ resid / dof))
        return NowcastResult(mean=float(x0 @ beta), std=std, model=self.name)




@dataclass
class BVARNowcaster(BaseNowcaster):
    """Quarterly Bayesian VAR (MacroPy >= 0.1.8) behind the BaseNowcaster contract.

    The system is [target] + quarterly means of ``variables`` (complete quarters
    only - this is a pure quarterly-frequency forecaster for the h >= 1 stage).
    The Minnesota-family prior shrinks toward a random walk, which matters for
    targets in secular slowdown where regressions over-predict. COVID quarters
    are treated with the Lenza-Primiceri volatility scaling by default.

    The posterior is re-sampled once per information set (cached on the target
    fingerprint), so the horizon harness's per-horizon calls stay cheap.
    """

    variables: Sequence[str] = field(default_factory=list)
    lags: int = 2
    prior_type: int = 2                 # Normal-Wishart (Minnesota moments)
    post_draws: int = 2000
    covid_window: tuple | None = ("2020Q1", "2021Q4")
    covid_mode: str | None = "lenza-primiceri"
    min_train: int = 40
    min_months: int = 3               # months per quarter for a "complete" row;
                                      # 2 tolerates a merged Jan-Feb release
                                      # (dropping the row would splice Q4->Q2
                                      # into adjacent VAR lags)
    sample_start: str | None = None   # regime-consistent estimation
    prior_params: dict | None = None  # e.g. Banbura soc/dio keys (MacroPy >= 0.1.9)
    seed: int = 7
    _name: str = "BVAR"

    def fit(self, info: InformationSet) -> "BVARNowcaster":
        self._info = info
        return self

    def _system(self, info: InformationSet) -> pd.DataFrame | None:
        y = info.y.dropna()
        if len(y) < self.min_train:
            return None
        q = y.to_frame()
        Xq = _quarterly_means(info.monthly, self.variables)
        # keep only quarters whose months are all released (balanced system);
        # min_months=2 tolerates a structurally merged month (China Jan-Feb)
        counts = info.monthly[list(self.variables)].notna().groupby(
            pd.PeriodIndex(info.monthly.index, freq="Q")).sum()
        complete = counts.index[(counts >= self.min_months).all(axis=1)]
        keep = pd.DatetimeIndex([_quarter_stamp(p) for p in complete])
        for c in Xq.columns:
            q[c] = Xq[c].reindex(q.index).where(q.index.isin(keep))
        q = q.dropna()
        if self.sample_start is not None:
            q = q.loc[self.sample_start:]
        return q

    def _posterior(self, q: pd.DataFrame):
        key = (q.index[-1], q.shape, float(np.nansum(q.to_numpy())))
        if getattr(self, "_cache_key", None) == key:
            return self._model
        from MacroPy import BayesianVAR

        kw = dict(lags=self.lags, prior_type=self.prior_type, post_draws=self.post_draws,
                  burnin=0.5, fhor=9, seed=self.seed)
        if self.prior_params is not None:
            kw["prior_params"] = dict(self.prior_params)
        if self.covid_mode and q.index[-1] >= pd.Timestamp("2020-03-01"):
            kw.update(covid_window=self.covid_window, covid_mode=self.covid_mode)
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model = BayesianVAR(q, **kw)
            model.sample_posterior()
        self._cache_key, self._model = key, model
        return model

    def nowcast(self, info: InformationSet) -> NowcastResult:
        y = info.y.dropna()
        if y.empty or info.target_period is None:
            return NowcastResult(mean=float("nan"), model=self.name)
        q = self._system(info)
        if q is None or len(q) < self.min_train:
            return NowcastResult(mean=float("nan"), model=self.name)

        last = pd.Period(q.index[-1], freq="Q")
        h = (pd.Period(info.target_period, freq="Q") - last).n
        if h <= 0:
            if info.target_period in y.index:
                return NowcastResult(mean=float(y.loc[info.target_period]), model=self.name)
            return NowcastResult(mean=float(y.iloc[-1]), model=self.name)

        try:
            model = self._posterior(q)
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                ff = model.forecast_frame(fhor=int(h), quantiles=(0.16, 0.84))
        except Exception:
            return NowcastResult(mean=float("nan"), model=self.name)

        tgt = info.target
        if isinstance(ff.index, pd.MultiIndex):
            row = ff.xs(tgt, level=0).iloc[int(h) - 1]
        else:
            sel = ff[ff["variable"] == tgt] if "variable" in ff.columns else ff
            row = sel.iloc[int(h) - 1]
        std = float((row["q84"] - row["q16"]) / 2) if {"q84", "q16"} <= set(row.index) else None
        return NowcastResult(mean=float(row["mean"]), std=std, model=self.name)




@dataclass
class ConditionalBVARNowcaster(BaseNowcaster):
    """Conditional BVAR: forecasts the target GIVEN real-time-observable paths.

    The user-specified exercise: BVAR(2) on [US VIX, US GDP growth, US
    inflation, fed funds, China IP, China M2, China GDP], Normal-Wishart prior
    with GLP-selected tightness, Waggoner-Zha conditional forecast with
    ``shock_uncertainty=False``. All conditions are time-t observables, so the
    model is honestly backtestable:

    * ``spf_map``: the system's US growth variable is conditioned on the LATEST
      SPF survey observed at the origin (the vintage, not today's survey). The
      US growth variable is qoq SAAR (``us_gdp_saar_m``) so the SPF conditions
      map one-to-one - conditioning a YoY variable on a SAAR path would be
      inconsistent.
    * ``fixed_vars``: held flat at their last observed value over the horizon
      (fed funds, VIX, inflation - the "other fixed variables").
    * ``partial_vars``: conditioned at h=1 on the already-released months of the
      first forecast quarter (China IP/M2 arrive within 15 days).

    ``quarterly_placed`` marks columns that live on one month per quarter (the
    SAAR and SPF columns), whose quarterly value needs 1 observed month, not 3.
    """

    variables: Sequence[str] = field(default_factory=list)   # monthly cols, system order
    spf_var: str | None = None            # system var conditioned on the SPF path
    spf_mode: str = "saar"                # "saar": direct; "yoy": SPF SAAR converted to
                                          # implied YoY via the 4-quarter identity
                                          # yoy_{t+h} ~ mean(saar_{t+h-3..t+h}) using
                                          # realized SAAR history (us_gdp_saar_m)
    nowcast_fn: Callable | None = None     # (info, period) -> nowcast of the current
                                          # unreleased quarter. When given, the model is
                                          # told what we already know about the near term
                                          # (last released print + that nowcast), so its
                                          # path is internally consistent with the
                                          # nowcast stage instead of re-deriving it.
    custom_conditions: dict | None = None  # {var: [path...]} overrides (e.g. FOMC SEP);
                                          # applied LAST, NaN = leave free at that horizon
    sample_start: str | None = None       # trim the estimation sample (regime choice):
                                          # a VAR's long-run mean is its sample mean, so
                                          # estimating across China's 2012 growth-regime
                                          # break pulls forecasts toward the old boom
    fixed_vars: Sequence[str] = field(default_factory=list)
    partial_vars: Sequence[str] = field(default_factory=list)
    quarterly_placed: Sequence[str] = field(default_factory=list)
    lags: int = 2
    prior_type: int = 2                   # Normal-Wishart
    prior_params: dict | Callable | None = None   # dict, or callable(q, info) -> dict
                                          # (per-origin priors: Villani ss anchors);
                                          # lamda1 keeps small-COVID-sample draws stable -
                                          # per-origin GLP there selects loose tightness,
                                          # triggering the sampler's stability-rejection
                                          # storm, ~100s/origin)
    glp_select: tuple = ("lamda1", "lamda3")
    post_draws: int = 800
    covid_window: tuple | None = ("2020Q1", "2021Q4")
    covid_mode: str | None = "lenza-primiceri"
    min_train: int = 40
    seed: int = 7
    shock_uncertainty: bool = False
    _name: str = "Cond-BVAR"

    def fit(self, info: InformationSet) -> "ConditionalBVARNowcaster":
        self._info = info
        return self

    # ------------------------------------------------------------------ #
    def _qvalue(self, info, col: str, period: pd.Period) -> float:
        """Quarterly value of ``col`` at ``period`` from observed months only."""

        m = info.monthly[col] if col in info.monthly.columns else None
        if m is None:
            return float("nan")
        w = m[pd.PeriodIndex(m.index, freq="Q") == period].dropna()
        # quarterly-placed columns carry one value per quarter; true monthlies
        # need 2 of 3 months (tolerates China's merged January releases).
        need = 1 if col in self.quarterly_placed else 2
        return float(w.mean()) if len(w) >= need else float("nan")

    def _system(self, info: InformationSet) -> pd.DataFrame | None:
        y = info.y.dropna()
        if len(y) < self.min_train:
            return None
        q = pd.DataFrame(index=y.index)
        periods = pd.PeriodIndex(y.index, freq="Q")
        for col in self.variables:
            q[col] = [self._qvalue(info, col, p) for p in periods]
        q[info.target] = y
        q = q.dropna()
        if self.sample_start is not None:
            q = q.loc[self.sample_start:]
        return q

    def _posterior(self, q: pd.DataFrame):
        key = (q.index[-1], q.shape, float(np.nansum(q.to_numpy())))
        if getattr(self, "_cache_key", None) == key:
            return self._model
        import contextlib
        import io

        from MacroPy import BayesianVAR

        kw = dict(lags=self.lags, prior_type=self.prior_type, post_draws=self.post_draws,
                  burnin=0.5, fhor=9, seed=self.seed)
        if self.prior_params is not None:
            # a callable is evaluated per origin: real-time priors (e.g. the
            # Villani steady state anchored on the WEO vintage of the day)
            pp = (self.prior_params(q, getattr(self, "_info", None))
                  if callable(self.prior_params) else self.prior_params)
            kw["prior_params"] = dict(pp)
        if self.covid_mode and q.index[-1] >= pd.Timestamp("2020-03-01"):
            kw.update(covid_window=self.covid_window, covid_mode=self.covid_mode)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model = BayesianVAR(q, **kw)
            if self.glp_select:
                try:
                    model.select_hyperparameters(select=self.glp_select, verbose=False)
                except Exception:
                    pass                     # fall back to default tightness
            model.sample_posterior()
        self._cache_key, self._model = key, model
        return model

    def _spf_conditions(self, info: InformationSet, base: pd.Period, h: int) -> list | None:
        """US growth path for h'=1..h from the latest SPF survey observed at origin.

        ``spf_mode="yoy"`` converts the survey's qoq-SAAR path into the implied
        percent-change-from-year-ago via the log identity
        yoy_{t+h} ~ (saar_{t+h} + saar_{t+h-1} + saar_{t+h-2} + saar_{t+h-3}) / 4,
        filling pre-forecast quarters with the realized SAAR history.
        """

        spf_cols = [c for c in info.monthly.columns if c.startswith("spf_gdp_h")]
        if not spf_cols or self.spf_var is None:
            return None
        obs = info.monthly[spf_cols].dropna(how="all")
        if obs.empty:
            return None
        srow = obs.iloc[-1]
        survey_q = pd.Period(obs.index[-1], freq="Q")
        cal = {survey_q + j: float(srow.get(f"spf_gdp_h{j}"))
               for j in range(5) if pd.notna(srow.get(f"spf_gdp_h{j}"))}

        if self.spf_mode != "yoy":
            return [cal.get(base + hp, np.nan) for hp in range(1, h + 1)]

        saar_hist = info.monthly.get("us_gdp_saar_m")
        if saar_hist is None:
            return None
        hist = saar_hist.dropna()
        hist_q = {pd.Period(ts, freq="Q"): float(v) for ts, v in hist.items()}
        path = []
        for hp in range(1, h + 1):
            quarters = [base + hp - j for j in range(4)]
            vals = [cal.get(qq, hist_q.get(qq, np.nan)) for qq in quarters]
            path.append(float(np.mean(vals)) if all(pd.notna(v) for v in vals) else np.nan)
        return path

    def nowcast(self, info: InformationSet) -> NowcastResult:
        y = info.y.dropna()
        if y.empty or info.target_period is None:
            return NowcastResult(mean=float("nan"), model=self.name)
        q = self._system(info)
        if q is None or len(q) < self.min_train:
            return NowcastResult(mean=float("nan"), model=self.name)

        base = pd.Period(q.index[-1], freq="Q")
        h = (pd.Period(info.target_period, freq="Q") - base).n
        if h <= 0:
            if info.target_period in y.index:
                return NowcastResult(mean=float(y.loc[info.target_period]), model=self.name)
            return NowcastResult(mean=float(y.iloc[-1]), model=self.name)

        conditions: dict[str, list] = {}
        spf_path = self._spf_conditions(info, base, h)
        # only condition a variable the system actually contains: a system built
        # without the US block still has spf_var set, and passing a condition for
        # an absent column makes the sampler reject the whole call (silently
        # returning NaN for every horizon)
        if (spf_path is not None and self.spf_var in q.columns
                and any(pd.notna(v) for v in spf_path)):
            conditions[self.spf_var] = spf_path
        for col in self.fixed_vars:
            if col in q.columns:
                conditions[col] = [float(q[col].iloc[-1])] * h
        for col in self.partial_vars:
            m = info.monthly.get(col)
            if m is None:
                continue
            w = m[pd.PeriodIndex(m.index, freq="Q") == base + 1].dropna()
            if len(w):
                path = [np.nan] * h
                path[0] = float(w.mean())
                conditions[col] = path
        # Ingest the near term we already know. The balanced system ends one
        # quarter before the last GDP print (the US block is not out at China's
        # release timing), so both the print and the current-quarter nowcast are
        # future steps from the model's point of view - and conditioning on them
        # is what makes the published path continue from the nowcast rather than
        # re-derive that quarter on its own.
        if self.nowcast_fn is not None:
            y_last = pd.Period(y.index[-1], freq="Q")
            i_real = (y_last - base).n - 1
            path = [np.nan] * h
            if 0 <= i_real < h:
                path[i_real] = float(y.iloc[-1])
            if 0 <= i_real + 1 < h:
                try:
                    v = self.nowcast_fn(info, y_last + 1)
                except Exception:
                    v = None
                if v is not None and np.isfinite(v):
                    path[i_real + 1] = float(v)
            if any(pd.notna(v) for v in path):
                conditions[info.target] = path

        if self.custom_conditions:
            for col, path in self.custom_conditions.items():
                if col in q.columns:
                    padded = list(path)[:h] + [np.nan] * max(0, h - len(path))
                    conditions[col] = [float(v) if pd.notna(v) else np.nan for v in padded]

        try:
            model = self._posterior(q)
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                cf = model.conditional_forecast_frame(
                    conditions=conditions if conditions else None, fhor=int(h),
                    quantiles=(0.05, 0.16, 0.84, 0.95),
                    shock_uncertainty=self.shock_uncertainty)
        except Exception:
            return NowcastResult(mean=float("nan"), model=self.name)

        tgt = info.target
        row = (cf.xs(tgt, level=0).iloc[int(h) - 1] if isinstance(cf.index, pd.MultiIndex)
               else cf.iloc[int(h) - 1])
        std = float((row["q84"] - row["q16"]) / 2) if {"q84", "q16"} <= set(row.index) else None
        res = NowcastResult(mean=float(row["mean"]), std=std, model=self.name)
        self._last_frame = cf                     # full conditional path, for fans
        self._last_conditions = conditions
        self._last_system = q
        return res

    def scenario_table(self, n_hist: int = 4) -> pd.DataFrame:
        """Last ``n_hist`` realized observations + the conditioned future paths.

        Rows = system variables; columns = the last realized quarters then
        h1..hH. NaN in a future column means the variable is left FREE (solved
        by the model). Call after ``nowcast``.
        """

        q = self._last_system
        cond = self._last_conditions
        base = pd.Period(q.index[-1], freq="Q")
        h = max((len(v) for v in cond.values()), default=0)
        hist = q.iloc[-n_hist:].T
        hist.columns = [str(pd.Period(c, freq="Q")) for c in hist.columns]
        fut = pd.DataFrame(index=hist.index,
                           columns=[f"{base + i} (h{i})" for i in range(1, h + 1)],
                           dtype=float)
        for col, path in cond.items():
            for i, v in enumerate(path, start=1):
                if pd.notna(v):
                    fut.loc[col, f"{base + i} (h{i})"] = float(v)
        return pd.concat([hist, fut], axis=1)


__all__ = ["DirectARXNowcaster", "BVARNowcaster", "ConditionalBVARNowcaster"]
