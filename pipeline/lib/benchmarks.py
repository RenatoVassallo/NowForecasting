"""Simple nowcast-family benchmarks on the release-cycle ladder.

Everything the audit asked to see next to Adaptive-IC, computed on the SAME
rows with common-sample discipline: equal-weight mean, median, previous-best
member, and a release-stage switch, plus the Bridge versus P-MIDAS comparison
restricted to their common finite sample. Every table carries n and the
finite share, so materially different samples cannot be compared silently.

All combinations here are REAL TIME by construction: previous-best and the
switch use only quarters whose outcome had been published before the origin,
and mean/median use only the members' contemporaneous values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.evaluation import EVALUATION_REGIME, rmse, sample_label

MEMBERS = ("RW", "Bridge(leaders)", "P-MIDAS(leaders)")
STAGE_EDGES = (0.0, 0.5, 0.75, 1.0)          # information-index stages


def _wide(ladder: pd.DataFrame) -> pd.DataFrame:
    d = ladder[ladder.model.isin(MEMBERS)].copy()
    d["origin_date"] = pd.to_datetime(d.origin_date)
    d["ref_quarter"] = pd.to_datetime(d.ref_quarter)
    keys = ["ref_quarter", "origin_date", "days_to_publication"]
    if "info_index" in d.columns:
        keys.append("info_index")
    wide = d.pivot_table(index=keys, columns="model", values="y_hat",
                         aggfunc="first")
    truth = d.pivot_table(index=keys, values="y_true", aggfunc="first")
    out = wide.join(truth).reset_index()
    return out


def add_benchmarks(ladder: pd.DataFrame) -> pd.DataFrame:
    """Long frame with the benchmark ensembles appended row-per-origin."""
    w = _wide(ladder)
    rows = []

    # publication date per quarter: first origin where y_true is knowable is
    # approximated by the release rule embedded in days_to_publication <= 0
    pub_known = {}
    for q, g in w.groupby("ref_quarter"):
        if g.y_true.notna().any():
            pub_known[q] = q            # outcome exists (final vintage)

    w = w.sort_values(["origin_date"])
    quarters = sorted(w.ref_quarter.unique())
    for _, r in w.iterrows():
        vals = {m: r.get(m, np.nan) for m in MEMBERS}
        fin = {m: v for m, v in vals.items() if np.isfinite(v)}
        base = dict(ref_quarter=r.ref_quarter, origin_date=r.origin_date,
                    days_to_publication=r.days_to_publication,
                    info_index=r.get("info_index", np.nan), y_true=r.y_true)
        rows.append({**base, "model": "Mean(members)",
                     "y_hat": float(np.mean(list(fin.values()))) if fin else np.nan})
        rows.append({**base, "model": "Median(members)",
                     "y_hat": float(np.median(list(fin.values()))) if fin else np.nan})

        # previous-best: the member with the lowest RMSE over quarters whose
        # outcome was PUBLISHED before this origin (release rule: publication
        # is ~52 days after quarter end)
        past = w[(w.ref_quarter + pd.Timedelta(days=82) < r.origin_date)
                 & w.y_true.notna()]
        best, best_rmse = None, np.inf
        for m in MEMBERS:
            e = (past.y_true - past[m]).dropna() if m in past else pd.Series(dtype=float)
            if len(e) >= 8 and rmse(e) < best_rmse:
                best, best_rmse = m, rmse(e)
        rows.append({**base, "model": "Previous-best",
                     "y_hat": vals.get(best, np.nan) if best else np.nan})

        # release-stage switch: RW early, P-MIDAS mid, Bridge late
        idx = r.get("info_index", np.nan)
        if np.isfinite(idx):
            pick = ("RW" if idx < STAGE_EDGES[1]
                    else "P-MIDAS(leaders)" if idx < STAGE_EDGES[2]
                    else "Bridge(leaders)")
        else:
            pick = "RW"
        rows.append({**base, "model": "Stage-switch",
                     "y_hat": vals.get(pick, np.nan)})
    long = pd.DataFrame(rows)
    keep = ladder[ladder.model.isin(MEMBERS + ("Adaptive-IC",))].copy()
    keep["origin_date"] = pd.to_datetime(keep.origin_date)
    keep["ref_quarter"] = pd.to_datetime(keep.ref_quarter)
    cols = ["ref_quarter", "origin_date", "days_to_publication", "model",
            "y_hat", "y_true"]
    if "info_index" in keep.columns:
        cols.append("info_index")
    return pd.concat([keep[cols], long], ignore_index=True)


def nowcast_scoreboard(rc: pd.DataFrame, *, exclude_years=(2020, 2021)) -> pd.DataFrame:
    """Common-sample scoreboard by model and sample split, plus the
    Bridge/P-MIDAS head-to-head restricted to their common finite rows."""
    d = rc[rc.y_true.notna()].copy()
    d = d[~pd.to_datetime(d.ref_quarter).dt.year.isin(list(exclude_years))]
    d["sample"] = [sample_label(q) for q in d.ref_quarter]

    piv = d.pivot_table(index=["ref_quarter", "origin_date"], columns="model",
                        values="y_hat", aggfunc="first")
    truth = d.pivot_table(index=["ref_quarter", "origin_date"], values="y_true",
                          aggfunc="first").y_true
    labels = pd.Series([sample_label(q) for q, _ in piv.index], index=piv.index)

    rows = []
    full = piv.dropna(how="any")            # the strict common sample
    for sample in ("selection", "holdout", "all"):
        m = slice(None) if sample == "all" else (labels == sample)
        sub = full[m] if sample != "all" else full
        tru = truth.reindex(sub.index)
        for model in sub.columns:
            e = tru - sub[model]
            rows.append(dict(sample=sample, comparison="common_sample_all_models",
                             model=model, n=len(sub), finite_share=1.0,
                             rmse=round(rmse(e), 3), regime=EVALUATION_REGIME))
    # Bridge vs P-MIDAS on THEIR common finite rows (wider than the strict set)
    bp = piv[["Bridge(leaders)", "P-MIDAS(leaders)"]].dropna(how="any")
    tru = truth.reindex(bp.index)
    for model in bp.columns:
        rows.append(dict(sample="all", comparison="bridge_vs_pmidas_common",
                         model=model, n=len(bp), finite_share=1.0,
                         rmse=round(rmse(tru - bp[model]), 3),
                         regime=EVALUATION_REGIME))
    # each model on ITS full sample, finite share reported (NOT comparable)
    for model in piv.columns:
        e = truth - piv[model]
        rows.append(dict(sample="all", comparison="own_sample_not_comparable",
                         model=model, n=int(piv[model].notna().sum()),
                         finite_share=round(float(piv[model].notna().mean()), 3),
                         rmse=round(rmse(e), 3), regime=EVALUATION_REGIME))
    return pd.DataFrame(rows)
