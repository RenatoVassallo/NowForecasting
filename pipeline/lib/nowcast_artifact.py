"""One OFFICIAL current-quarter nowcast per run, consumed everywhere.

The audit found three surfaces publishing "the nowcast" under one label with
two definitions: the headline Adaptive-IC combines RW, Bridge and P-MIDAS
(``metadata.ADAPTIVE_MEMBERS``), while the fan's first node and the flagship
figure averaged only Bridge and P-MIDAS. The rule now:

- the nowcast stage derives ONE artifact from its own result (the newest live
  Adaptive-IC origin), records members, learned bin weights, realized weights,
  the information state, and the node's two-piece-normal parameters fitted to
  the information-bin residual pool;
- the fan, figures and report CONSUME this artifact; none recomputes its own
  ensemble;
- missing members follow the MIDAS combine convention, recorded here as the
  artifact contract: the learned bin weights renormalize PROPORTIONALLY over
  the members whose live nowcast is finite. Both the learned and the realized
  weights are stored, plus the list of missing members.

The artifact is one CSV row: ``products/peru_nowcast_official.csv`` plus a
copy in the run directory. Consumers must check ``as_of`` against their run
context; a mismatch means the stages were run on different information sets
and publication must stop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OFFICIAL_PATH = REPO / "products" / "peru_nowcast_official.csv"
SWEEP_PATH = REPO / "products" / "peru_nowcast_sweep.csv"


def official_from(nowcasts: pd.DataFrame, weights: pd.DataFrame, pools: dict,
                  *, members: list[str], adaptive_name: str, as_of,
                  n_bins: int = 4) -> pd.DataFrame:
    """Build the one-row official artifact from the nowcast job's outputs."""
    from forecast.fan_mc import fit_tpn_mle

    as_of = pd.Timestamp(as_of).normalize()
    live = nowcasts[(nowcasts.model == adaptive_name)
                    & nowcasts.y_true.isna() & nowcasts.y_hat.notna()].copy()
    if live.empty:
        raise RuntimeError("no live Adaptive-IC nowcast row; the nowcast stage "
                           "must run before an official artifact can exist")
    live["origin_date"] = pd.to_datetime(live.origin_date)
    newest = live.sort_values("origin_date").iloc[-1]
    quarter = pd.Period(pd.Timestamp(newest.ref_quarter), freq="Q")
    value = float(newest.y_hat)
    info_idx = float(newest.get("info_index", np.nan))
    bin_now = int(np.clip(np.digitize([info_idx], np.linspace(0, 1, n_bins + 1)[1:-1]),
                          0, n_bins - 1)[0])

    # learned weights for (live quarter, bin); realized = renormalized over the
    # members whose nowcast at the SAME origin is finite (the combine rule)
    wrow = weights[(pd.to_datetime(weights.ref_quarter) == pd.Timestamp(newest.ref_quarter))
                   & (weights.bin == bin_now)]
    learned = {m: (float(wrow[m].iloc[0]) if len(wrow) and m in wrow else np.nan)
               for m in members}
    same_origin = nowcasts[(pd.to_datetime(nowcasts.origin_date) == newest.origin_date)
                           & (pd.to_datetime(nowcasts.ref_quarter)
                              == pd.Timestamp(newest.ref_quarter))]
    member_vals = {m: float(same_origin[same_origin.model == m].y_hat.iloc[0])
                   if len(same_origin[same_origin.model == m]) else np.nan
                   for m in members}
    finite = {m: np.isfinite(member_vals[m]) for m in members}
    missing = sorted(m for m in members if not finite[m])
    wsum = sum(learned[m] for m in members if finite[m] and np.isfinite(learned[m]))
    realized = {m: (learned[m] / wsum if finite[m] and np.isfinite(learned[m]) and wsum > 0
                    else 0.0) for m in members}

    pool_cands = [(q, v) for (q, b), v in pools.items() if b == bin_now]
    if not pool_cands:
        raise RuntimeError(f"no residual pool for information bin {bin_now}")
    _, pool = max(pool_cands, key=lambda t: t[0])
    pool = np.asarray(pool, dtype=float)
    fit = fit_tpn_mle(value + pool)
    if fit is None:
        sd = float(np.std(pool))
        fit = {"s": sd, "gamma": 0.0, "sigma_left": sd, "sigma_right": sd}

    return pd.DataFrame([{
        "target": "peru_gdp", "quarter": str(quarter),
        "as_of": str(as_of.date()),
        "origin_date": str(pd.Timestamp(newest.origin_date).date()),
        "days_to_publication": int(newest.days_to_publication),
        "value": round(value, 4),
        "adaptive_name": adaptive_name,
        "members": "+".join(members),
        "weights_learned": "|".join(f"{m}:{learned[m]:.3f}" for m in members),
        "weights_realized": "|".join(f"{m}:{realized[m]:.3f}" for m in members),
        "missing_members": "+".join(missing) if missing else "",
        "member_values": "|".join(f"{m}:{member_vals[m]:.3f}" for m in members),
        "information_index": round(info_idx, 4),
        "information_bin": bin_now,
        "pool_n": int(pool.size),
        "s": round(float(fit["s"]), 4), "gamma": round(float(fit["gamma"]), 4),
        "sigma_left": round(float(fit["sigma_left"]), 4),
        "sigma_right": round(float(fit["sigma_right"]), 4),
    }])


def sweep_from(nowcasts: pd.DataFrame, *, adaptive_name: str, as_of) -> pd.DataFrame:
    """The official release-cycle sweep: every live Adaptive-IC origin.

    Persisted next to the one-row artifact so the flagship figure plots the
    SAME path the headline comes from, instead of recomputing an ensemble of
    its own from research caches.
    """
    live = nowcasts[(nowcasts.model == adaptive_name)
                    & nowcasts.y_true.isna() & nowcasts.y_hat.notna()].copy()
    live["origin_date"] = pd.to_datetime(live.origin_date)
    live = live.sort_values("origin_date")
    out = pd.DataFrame({
        "origin_date": live.origin_date.dt.date.astype(str),
        "y_hat": live.y_hat.to_numpy(dtype=float),
        "days_to_publication": live.days_to_publication.to_numpy(),
        "ref_quarter": pd.to_datetime(live.ref_quarter).dt.date.astype(str),
        "info_index": live["info_index"].to_numpy(dtype=float)})
    out["as_of"] = str(pd.Timestamp(as_of).normalize().date())
    return out


def write_official(store, result, *, as_of) -> Path:
    from pipeline.config import metadata

    frame = official_from(result.nowcasts, result.weights, result.pools,
                          members=metadata.ADAPTIVE_MEMBERS["peru_gdp"],
                          adaptive_name=metadata.ADAPTIVE["name"], as_of=as_of)
    sweep = sweep_from(result.nowcasts,
                       adaptive_name=metadata.ADAPTIVE["name"], as_of=as_of)
    OFFICIAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OFFICIAL_PATH, index=False)
    sweep.to_csv(SWEEP_PATH, index=False)
    for name, obj in ((OFFICIAL_PATH.name, frame), (SWEEP_PATH.name, sweep)):
        run_copy = Path(store.root) / name
        obj.to_csv(run_copy, index=False)
        if hasattr(store, "_track"):
            store._track(run_copy, "official-nowcast")
    return OFFICIAL_PATH


def load_sweep(expected_as_of, path: Path | None = None) -> pd.DataFrame:
    p = Path(path) if path is not None else SWEEP_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"official nowcast sweep missing ({p}); run the nowcast stage "
            "in this as-of before the figures")
    d = pd.read_csv(p)
    expected = str(pd.Timestamp(expected_as_of).date())
    if str(d["as_of"].iloc[0]) != expected:
        raise ValueError(
            f"official sweep is as of {d['as_of'].iloc[0]} but this run is as "
            f"of {expected}; stages must share one information set")
    return d


def load_official(expected_as_of, path: Path | None = None) -> pd.Series:
    """Read and validate the official nowcast for a consumer stage."""
    p = Path(path) if path is not None else OFFICIAL_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"official nowcast artifact missing ({p}); run the nowcast stage "
            "in this as-of before the fan or report")
    row = pd.read_csv(p).iloc[0]
    expected = str(pd.Timestamp(expected_as_of).date())
    if str(row["as_of"]) != expected:
        raise ValueError(
            f"official nowcast is as of {row['as_of']} but this run is as of "
            f"{expected}; stages must share one information set")
    return row
