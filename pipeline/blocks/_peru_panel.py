"""Peru panel construction and the conditional-BVAR factory (production).

Promoted from ``notebooks/peru/forecast/common.py`` (T22): the Peru block, the
exact-chain replay, and the notebooks all build the estimation panel and the
S1 conditional model HERE, so a difference between research and production
information sets is impossible by construction. The notebooks import this
module through a thin shim left at the old location.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from MIDAS import MetadataPanel, VariableMeta

REPO = Path(__file__).resolve().parents[2]

# candidate systems raced in the backtest notebooks; the winner runs live
CANDIDATE_SYSTEMS = {
    "S1 as-specified":  ["us_gdp_yoy_m", "ip_cum_yoy", "g_tdi", "exp_eco3m", "g_invq_m"],
    "S2 evidence":      ["ip_cum_yoy", "g_tdi", "exp_eco3m"],
    "S3 evidence+inv":  ["ip_cum_yoy", "g_tdi", "exp_eco3m", "g_invq_m"],
    "S4 evidence+risk": ["ip_cum_yoy", "g_tdi", "exp_eco3m", "embig_lac"],
}
TIGHT = {"mn_mean": 1, "lamda1": 0.2, "lamda2": 0.5, "lamda3": 1, "lamda4": 1e5}
FLOOR = "2003-01-01"
LAB = {"us_gdp_yoy_m": "US GDP (% YoY)", "ip_cum_yoy": "China industrial production (% YoY)",
       "g_tdi": "Peru terms of trade (% YoY)", "exp_eco3m": "Business expectations, 3m",
       "g_invq_m": "Private investment (% YoY)", "embig_lac": "EMBIG LatAm spread (bp)",
       "g_pbiq": "Peru GDP (% YoY)"}

# publication delays for the columns we bolt on (days after the reference month)
_DELAYS = {"g_invq_m": 51, "us_gdp_yoy_m": 30, "us_gdp_saar_m": 30, "us_fedfunds": 1,
           "us_vix": 1, "ip_cum_yoy": 15, "m2_yoy": 13,
           **{f"spf_gdp_h{i}": 45 for i in range(5)}}


def _q2m(s: pd.Series, index: pd.DatetimeIndex, name: str) -> pd.Series:
    out = pd.Series(np.nan, index=index, name=name)
    for ix, v in s.dropna().items():
        st = pd.Period(ix, freq="Q").to_timestamp(how="end").to_period("M").to_timestamp()
        if st in out.index:
            out.loc[st] = float(v)
    return out


def build_panel(spec):
    """Peru's own panel + private investment + the US and China blocks."""

    from targets import china as cn
    from targets import usa

    monthly, quarterly, _ = spec.load_panel()
    inv = pd.read_parquet(REPO / "input/bcrp/private_investment.parquet")["g_invq"]
    ext = pd.DataFrame(index=monthly.index)
    ext["g_invq_m"] = _q2m(inv, monthly.index, "g_invq_m")

    us_m, _, _ = usa.load_panel()
    for c in ["us_gdp_yoy_m", "us_gdp_saar_m", "us_fedfunds", "us_vix"] + \
             [f"spf_gdp_h{i}" for i in range(5)]:
        if c in us_m.columns and c not in monthly.columns:
            ext[c] = us_m[c].reindex(monthly.index)
    cn_m, _, _ = cn.load_panel()
    for c in ["ip_cum_yoy", "m2_yoy"]:
        if c not in monthly.columns:
            ext[c] = cn_m[c].reindex(monthly.index)

    mm = monthly.join(ext)
    metas = []
    for c in mm.columns:
        if c in _DELAYS:
            d = _DELAYS[c]
        elif c.startswith("g_pbim") or c.startswith("g_inv"):
            d = 51
        elif c in ("g_tdi", "g_ipx", "g_ipm"):
            d = 40
        else:
            d = 15
        metas.append(VariableMeta(column=c, frequency="M", group="Peru",
                                  publication_delay_days=int(d)))
    metas.append(VariableMeta(column=spec.target, frequency="Q", group="Peru",
                              publication_delay_days=int(spec.target_delay_days)))
    return mm, quarterly, MetadataPanel.from_frames(mm, quarterly, metas)


def nowcast_ladder_path() -> Path:
    """The frozen Peru nowcast ladder (hash-verified calibration asset)."""

    from pipeline.lib.calibration_assets import asset_path

    return asset_path("peru_ladder_full")


def nowcast_lookup(panel):
    """Real-time Peru nowcast of any quarter, as of any origin (for `nowcast_fn`)."""

    nc = pd.read_parquet(nowcast_ladder_path())
    nc = nc[nc.model.isin(["Bridge(leaders)", "P-MIDAS(leaders)"]) & nc.y_hat.notna()].copy()
    nc["origin_date"] = pd.to_datetime(nc.origin_date)
    nc["ref_quarter"] = pd.to_datetime(nc.ref_quarter)

    def fn(info, period):
        st = period.to_timestamp(how="end").to_period("M").to_timestamp()
        g = nc[(nc.ref_quarter == st) & (nc.origin_date <= pd.Timestamp(info.origin))]
        if g.empty:
            return None
        return float(g[g.origin_date == g.origin_date.max()].y_hat.mean())
    return fn, nc


def make_cond(system, *, nowcast_fn=None, custom=None, name="Cond-BVAR",
              draws=800):
    from forecast.models import ConditionalBVARNowcaster

    return ConditionalBVARNowcaster(
        variables=list(system), spf_var="us_gdp_yoy_m", spf_mode="yoy",
        fixed_vars=[c for c in ("us_fedfunds", "us_vix") ],
        partial_vars=[c for c in system if c in ("g_tdi", "exp_eco3m", "embig_lac",
                                                 "ip_cum_yoy")],
        quarterly_placed=["us_gdp_yoy_m", "us_gdp_saar_m", "g_invq_m"],
        nowcast_fn=nowcast_fn, custom_conditions=custom,
        sample_start=FLOOR, min_train=28, prior_params=TIGHT, glp_select=(),
        lags=2, post_draws=draws, shock_uncertainty=False, _name=name)
