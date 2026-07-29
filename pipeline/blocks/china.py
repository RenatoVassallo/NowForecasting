"""China GDP: the published profile (nowcast, then model/WEO blend) as a fan."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import REPO, fan_frame, information_stamp, write

ALPHA = 0.5          # half the model, half the WEO-consistent path
FLOOR = "2012-01-01"


def build(**_) -> tuple[pd.DataFrame, list[str], object]:
    """Rebuild China's published path from the cached backtest + live models.

    The heavy horse race lives in the notebook; production reuses its cache and
    recomputes only the live path, which is what actually changes between runs.
    """
    import forecast
    from forecast.fan_mc import fit_tpn_mle
    from sources import imf
    from targets import china as tgt

    cache = next((REPO / "notebooks").rglob("horizon_2012floor.parquet"), None)
    if cache is None:
        raise FileNotFoundError("China horse-race cache missing; run the china notebook once")
    bt = pd.read_parquet(cache)
    MEMBERS = ["RW", "D-ARX(leaders)", "BVAR(3v)", "Cond-BVAR", "Cond-BVAR+SS"]
    bt = bt[bt.model.isin(MEMBERS)]
    rc = forecast.combine_by_horizon(bt, MEMBERS, name="Combo")
    combo_bt = rc[(rc.model == "Combo") & rc.y_true.notna()]

    tilt = next((REPO / "notebooks").rglob("tilt_weo.parquet"), None)
    td = pd.read_parquet(tilt) if tilt is not None else None

    # blend errors -> two-piece normal per horizon
    fits = {}
    if td is not None:
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

    prof = next((REPO / "notebooks").rglob("china_profile_fan.csv"), None)
    if prof is None:
        raise FileNotFoundError("China profile missing; run the china notebook once")
    p = pd.read_csv(prof)
    periods = pd.PeriodIndex(p.iloc[:, 0], freq="Q")
    centre = p["y_hat"].to_numpy()
    src = ["nowcast"] + [f"blend a={ALPHA}"] * (len(periods) - 1)

    # first node keeps the nowcast's own (tighter) uncertainty
    nfits = []
    for i in range(len(periods)):
        if i == 0:
            w = float(p["q90_hi_s"].iloc[0] - p["q90_lo_s"].iloc[0]) / (2 * 1.6449)
            nfits.append({"s": w, "gamma": 0.0, "sigma_left": w, "sigma_right": w,
                          "mode_shift": 0.0})
        else:
            nfits.append(fits.get(i + 1))
    df = fan_frame(periods, centre, nfits, src)
    weo, rnd = imf.path("CHN", pd.Timestamp.now())
    stamp = information_stamp(tgt.SPEC, periods[0]); stamp["weo_round"] = rnd
    out = write(df, REPO / "products/blocks/china_path_uncertainty.csv", stamp)
    lines = [f"- **China**: {centre[0]:.2f}% nowcast, {centre[-1]:.2f}% at h=8 "
             f"(WEO {rnd}: {weo.get(pd.Timestamp.now().year, float('nan')):.1f}%); "
             f"90% band {df.width90.iloc[0]:.2f} to {df.width90.iloc[-1]:.2f}pp"]
    return df, lines, out
