"""China GDP: the published profile (nowcast, then model/WEO blend) as a fan.

The centre path is RECOMPUTED at every run by ``pipeline.blocks._china_model``
(fresh nowcast sweep, live five-member Combo, live WEO tilt at the run's as-of
date). Research caches enter only as calibration data: the horse-race backtest
supplies combination weights and the blend's historical errors for the
two-piece-normal fits, and both cache hashes are stamped into the product. A
missing input raises; nothing is ever restamped from a cached profile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.lib.calibration_assets import asset_path

from ._common import REPO, _dest, fan_frame, information_stamp, write
from ._china_model import ALPHA, MEMBERS, live_profile


def build(ctx=None, out_dir=None, **_) -> tuple[pd.DataFrame, list[str], object]:
    import forecast
    from forecast.fan_mc import fit_tpn_mle
    from pipeline.lib.context import resolve_as_of
    from targets import china as tgt

    as_of = resolve_as_of(ctx)

    # ---- error calibration: blend backtest errors -> TPN per horizon --------
    cache = asset_path("china_horizon_2012floor")
    bt = pd.read_parquet(cache)
    bt = bt[bt.model.isin(MEMBERS)]
    rc = forecast.combine_by_horizon(bt, MEMBERS, name="Combo")
    combo_bt = rc[(rc.model == "Combo") & rc.y_true.notna()]

    tilt = asset_path("china_tilt_weo")
    td = pd.read_parquet(tilt)

    fits = {}
    cb = combo_bt[["base_quarter", "horizon", "y_hat"]].rename(columns={"y_hat": "combo"})
    j = td.merge(cb, on=["base_quarter", "horizon"], how="inner")
    j = j[~pd.to_datetime(j.ref_quarter).dt.year.isin([2020, 2021])]
    j = j[~pd.to_datetime(j.base_quarter).dt.year.isin([2020, 2021])]
    j["blend"] = (1 - ALPHA) * j["combo"] + ALPHA * j["y_tilted"]
    for h, g in j.groupby("horizon"):
        e = (g.y_true - g.blend).dropna().to_numpy()
        f = fit_tpn_mle(e) if e.size >= 20 else None
        if f is None and e.size:
            sd = float(np.std(e)); f = {"s": sd, "gamma": 0.0,
                                        "sigma_left": sd, "sigma_right": sd}
        if f:
            f["mode_shift"] = 0.0
            fits[int(h)] = f

    # ---- the live profile, recomputed from the current snapshot -------------
    prof, pool, prov = live_profile(as_of)
    periods = pd.PeriodIndex(prof["period"], freq="Q")
    centre = prof["y_hat"].to_numpy(dtype=float)
    src = prof["source"].tolist()

    # first node keeps the nowcast's own information-conditional uncertainty
    node1 = fit_tpn_mle(np.asarray(pool, dtype=float))
    if node1 is None:
        sd = float(np.std(pool))
        node1 = {"s": sd, "gamma": 0.0, "sigma_left": sd, "sigma_right": sd}
    node1["mode_shift"] = 0.0
    nfits = [node1] + [fits.get(i + 1) for i in range(1, len(periods))]

    df = fan_frame(periods, centre, nfits, src)
    stamp = information_stamp(tgt.SPEC, periods[0], as_of=as_of)
    stamp.update(prov)
    out = write(df, _dest(out_dir, "china_path_uncertainty.csv"), stamp)
    lines = [f"- **China**: {centre[0]:.2f}% nowcast, {centre[-1]:.2f}% at h=8 "
             f"(WEO {prov['weo_round']}: {prov['weo_current_year']:.1f}%); "
             f"90% band {df.width90.iloc[0]:.2f} to {df.width90.iloc[-1]:.2f}pp"]
    return df, lines, out
