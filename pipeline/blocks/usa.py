"""US GDP growth: SPF at the short end, IMF WEO beyond, calibrated on real-time errors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import REPO, fan_frame, information_stamp, write

SAMPLE_FROM = 2010           # WEO vintages start here; a 10y window is WIDER (tested)
ORIGIN_DAY = 50              # the SPF for the current quarter is out by day ~50


def weo_at(weo: pd.Series, year: int) -> float:
    """WEO annual value with the flat medium-term tail: a year beyond the
    round's horizon takes the last projected year (the IMF convention); a
    year before the first projection stays missing."""
    if year in weo.index:
        return float(weo.loc[year])
    if len(weo.index) and year > int(max(weo.index)):
        return float(weo.loc[max(weo.index)])
    return float("nan")


def _blend_weight(h: int) -> float:
    """SPF where it is decisively better, handing over to the WEO as it runs out."""
    return {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.5, 5: 0.5}.get(h, 1.0)


def _spf_yoy(spf, saar, survey_q, hmax=8) -> dict:
    """YoY path implied by one SPF survey: a YoY rate is the mean of four SAARs."""
    if survey_q not in spf.index:
        return {}
    row = spf.loc[survey_q]
    fc = {survey_q + i: float(row[f"spf_gdp_h{i}"]) for i in range(5)
          if pd.notna(row.get(f"spf_gdp_h{i}"))}
    hist = saar[saar.index < survey_q]
    out = {}
    for k in range(hmax + 1):
        tgt = survey_q + k
        vals = []
        for w in [tgt - j for j in range(4)]:
            if w in fc:
                vals.append(fc[w])
            elif w in hist.index:
                vals.append(float(hist.loc[w]))
            else:
                vals = None
                break
        if vals:
            out[tgt] = float(np.mean(vals))
    return out


def build(ctx=None, **_) -> tuple[pd.DataFrame, list[str]]:
    from forecast.fan_mc import fit_tpn_mle
    from pipeline.lib.context import resolve_as_of
    from sources import imf
    from targets import usa

    as_of = resolve_as_of(ctx)

    m, q, _ = usa.load_panel()
    realised = q[usa.TARGET].dropna(); realised.index = pd.PeriodIndex(realised.index, freq="Q")
    saar = m["us_gdp_saar_m"].dropna(); saar.index = pd.PeriodIndex(saar.index, freq="Q")
    spf = m[[f"spf_gdp_h{i}" for i in range(5)]].dropna(how="all")
    spf.index = pd.PeriodIndex(spf.index, freq="Q")
    spf = spf[~spf.index.duplicated(keep="last")]

    # real-time errors of the rule, one origin per quarter at day ~50
    rows = []
    for o in pd.period_range("2000Q1", realised.index.max(), freq="Q"):
        sp = _spf_yoy(spf, saar, o)
        try:
            weo, _ = imf.path("USA", as_of=o.to_timestamp(how="start") + pd.Timedelta(days=ORIGIN_DAY))
        except Exception:
            weo = pd.Series(dtype=float)
        for k in range(1, 9):
            tgt = o + (k - 1)
            if tgt not in realised.index:
                continue
            s_ = sp.get(tgt, np.nan)
            w_ = weo_at(weo, tgt.year)
            a = _blend_weight(k)
            rule = ((1 - a) * s_ + a * w_ if np.isfinite(s_) and np.isfinite(w_)
                    else s_ if np.isfinite(s_) else w_)
            if np.isfinite(rule):
                rows.append({"h": k, "target": tgt, "err": float(realised.loc[tgt]) - rule})
    e = pd.DataFrame(rows)
    e = e[(~e.target.dt.year.isin([2020, 2021])) & (e.target.dt.year >= SAMPLE_FROM)]
    fits = {}
    for h, g in e.groupby("h"):
        arr = g.err.to_numpy()
        f = fit_tpn_mle(arr)
        if f is None:
            sd = float(np.std(arr))
            f = {"mode": 0.0, "s": sd, "gamma": 0.0, "sigma_left": sd, "sigma_right": sd}
        f["mode_shift"] = 0.0                     # the mode is the central projection
        fits[h] = f

    # the run's live path
    weo_now, rnd = imf.path("USA", as_of)
    sp_now = _spf_yoy(spf, saar, spf.index.max())
    last_q = realised.index.max()
    grid = [last_q + k for k in range(1, 9)]
    centre, src = [], []
    for k, tgt in enumerate(grid, start=1):
        a = _blend_weight(k)
        s_ = sp_now.get(tgt, np.nan)
        w_ = weo_at(weo_now, tgt.year)
        if np.isfinite(s_) and np.isfinite(w_):
            centre.append((1 - a) * s_ + a * w_); src.append(f"SPF/WEO a={a:.2f}")
        elif np.isfinite(s_):
            centre.append(s_); src.append("SPF")
        else:
            centre.append(w_); src.append("WEO")
    df = fan_frame(grid, centre, [fits.get(h) for h in range(1, 9)], src)
    stamp = information_stamp(usa.SPEC, grid[0], as_of=as_of)
    stamp["weo_round"] = rnd
    out = write(df, REPO / "products/blocks/us_path_uncertainty.csv", stamp)
    lines = [f"- **US**: {centre[0]:.2f}% next quarter, {centre[-1]:.2f}% at h=8 "
             f"(WEO {rnd}); 90% band {df.width90.iloc[0]:.2f} to {df.width90.iloc[-1]:.2f}pp"]
    return df, lines, out
