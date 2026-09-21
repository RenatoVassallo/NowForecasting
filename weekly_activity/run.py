"""Command-line workflow for the weekly Peru activity nowcast MVP.

Examples:
    python -m weekly_activity.run download --end 2026-08-17
    python -m weekly_activity.run build
    python -m weekly_activity.run evaluate
    python -m weekly_activity.run nowcast --as-of 2026-08-18
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import run_backtest, score_backtest
from .config import (COES_START, DATA_ROOT, EVALUATION_REGIME, LBTR_START,
                     MIN_TRAIN_MONTHS, OUTPUT_ROOT, STAGES_TO_REPORT)
from .features import build_feature_frame, feature_dictionary
from .models import (ModelUnavailable, candidate_importance, candidate_names,
                     fit_candidate, predict_candidate)
from .target import load_monthly_target, write_monthly_target


def _paths(data_root=DATA_ROOT, output_root=OUTPUT_ROOT) -> dict[str, Path]:
    data, out = Path(data_root), Path(output_root)
    return {
        "coes": data / "processed" / "coes_daily.parquet",
        "lbtr": data / "processed" / "lbtr_daily.parquet",
        "features": data / "processed" / "weekly_features.parquet",
        "dictionary": data / "processed" / "feature_dictionary.csv",
        "target": data / "processed" / "target_monthly.parquet",
        "backtest": out / "backtest.parquet",
        "scoreboard": out / "backtest_scoreboard.csv",
        "coverage": out / "source_coverage.csv",
        "nowcast": out / "latest_nowcast.csv",
        "importance": out / "latest_block_importance.csv",
        "plot": out / "nowcasts_vs_actual.png",
        "assessment": out / "mvp_assessment.md",
    }


def _load_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} is missing at {path}; run the preceding pipeline step")
    return pd.read_parquet(path)


def source_coverage(electricity: pd.DataFrame | None, lbtr: pd.DataFrame | None,
                    *, trends: pd.DataFrame | None = None,
                    coes_status: pd.DataFrame | None = None) -> pd.DataFrame:
    """Coverage table that distinguishes observed dates from release timestamps."""
    rows = []
    for name, frame, freq, timing, caveat in (
        ("COES executed demand", electricity, "daily, 48 half-hour readings", "not published historically",
         "Historical simulations use observation dates; first-publication timestamps unavailable."),
        ("BCRP client LBTR", lbtr, "daily business-day observations", "not published historically",
         "Historical simulations use observation dates; BCRP revisions and first-seen times are not archived."),
        ("Google Trends", trends, "weekly, frozen topic index", "snapshot-specific",
         "Disabled unless a provenance-rich official API snapshot is supplied."),
    ):
        if frame is None or frame.empty:
            start = end = pd.NaT
            n, status = 0, "unavailable"
        else:
            date_col = "date" if "date" in frame else "week_end"
            x = pd.to_datetime(frame[date_col], errors="coerce")
            start, end, n, status = x.min(), x.max(), len(frame), "available"
        rows.append({"source": name, "effective_frequency": freq, "rows": n,
                     "first_observation": start, "last_observation": end,
                     "historical_release_timestamp": timing, "status": status,
                     "known_limitation": caveat})
    out = pd.DataFrame(rows)
    if electricity is not None and not electricity.empty:
        if {"area_complete_count", "area_count"}.issubset(electricity.columns):
            partial = (pd.to_numeric(electricity.area_complete_count, errors="coerce")
                       < pd.to_numeric(electricity.area_count, errors="coerce"))
            out.loc[out.source.eq("COES executed demand"), "partial_optional_area_dates"] = int(partial.fillna(False).sum())
        if "core_complete" in electricity:
            core_ok = electricity.core_complete
            out.loc[out.source.eq("COES executed demand"), "invalid_core_area_dates"] = int((~pd.Series(core_ok).fillna(False).astype(bool)).sum())
    if coes_status is not None and not coes_status.empty:
        bad = coes_status[coes_status.status.ne("valid")]
        out.loc[out.source.eq("COES executed demand"), "invalid_or_unavailable_dates"] = len(bad)
        out.loc[out.source.eq("COES executed demand"), "latest_invalid_date"] = (
            pd.to_datetime(bad.date).max() if len(bad) else pd.NaT)
    return out


def build(data_root=DATA_ROOT, output_root=OUTPUT_ROOT, *, trends_path: str | None = None) -> dict:
    """Build target and all historical weekly-cutoff feature rows from cached raw data."""
    paths = _paths(data_root, output_root)
    electricity = _load_required(paths["coes"], "COES processed daily data")
    lbtr = _load_required(paths["lbtr"], "BCRP LBTR processed daily data")
    trends = None
    if trends_path:
        from .ingest import load_google_trends_snapshot
        trends = load_google_trends_snapshot(trends_path)
    target = load_monthly_target()
    target_path = write_monthly_target(target, data_root=data_root)
    months = pd.PeriodIndex(target.target_month, freq="M")
    features = build_feature_frame(electricity, lbtr, months=months, trends=trends,
                                   include_trends=trends is not None)
    paths["features"].parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(paths["features"], index=False)
    feature_dictionary(features).to_csv(paths["dictionary"], index=False)
    Path(output_root).mkdir(parents=True, exist_ok=True)
    status_path = Path(data_root) / "processed" / "coes_download_status.parquet"
    coes_status = pd.read_parquet(status_path) if status_path.exists() else None
    coverage = source_coverage(electricity, lbtr, trends=trends, coes_status=coes_status)
    coverage.to_csv(paths["coverage"], index=False)
    return {"target": target_path, "features": paths["features"],
            "coverage": paths["coverage"], "n_features": len(features)}


def evaluate(data_root=DATA_ROOT, output_root=OUTPUT_ROOT) -> dict:
    """Run the fixed expanding-window horse race and write auditable tables."""
    paths = _paths(data_root, output_root)
    features = _load_required(paths["features"], "weekly feature frame")
    target = _load_required(paths["target"], "monthly GDP target")
    backtest = run_backtest(features, target, min_train_months=MIN_TRAIN_MONTHS)
    if backtest.empty:
        raise RuntimeError("backtest has no eligible origins; inspect source coverage and train window")
    paths["backtest"].parent.mkdir(parents=True, exist_ok=True)
    backtest.to_parquet(paths["backtest"], index=False)
    score = score_backtest(backtest)
    score.to_csv(paths["scoreboard"], index=False)
    _write_assessment(score, paths["assessment"])
    _plot_backtest(backtest, paths["plot"])
    return {"backtest": paths["backtest"], "scoreboard": paths["scoreboard"],
            "plot": paths["plot"], "assessment": paths["assessment"],
            "n_predictions": len(backtest)}


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    d = frame.loc[:, [c for c in columns if c in frame]].copy()
    for c in d.select_dtypes(include="number"):
        d[c] = d[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    head = "| " + " | ".join(d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in d.itertuples(index=False, name=None)]
    return [head, sep, *body]


def _write_assessment(score: pd.DataFrame, path: Path) -> None:
    """Freeze a transparent V1 decision from pre-specified test comparisons."""
    required = score[(score["sample"] == "test") & (score.period_group == "post_2020")
                     & score.stage.isin(STAGES_TO_REPORT)].copy()
    rows = []
    wins_all = True
    for stage in STAGES_TO_REPORT:
        d = required[required.stage == stage].set_index("model")
        needed = {"AR(1)", "Ridge electricity", "Ridge multiblock"}
        if not needed.issubset(d.index):
            wins_all = False
            rows.append({"stage": stage, "decision": "no common test sample"})
            continue
        multi = d.loc["Ridge multiblock", "rmse_common"]
        competitor = d.loc[["AR(1)", "Ridge electricity"], "rmse_common"].min()
        passed = bool(pd.notna(multi) and pd.notna(competitor) and multi < competitor)
        wins_all &= passed
        rows.append({"stage": stage, "multiblock_rmse_common": multi,
                     "best_baseline_rmse_common": competitor,
                     "decision": "wins" if passed else "does not win"})
    decision = ("Do not proceed to a more complex V2. The multiblock model does not "
                "beat both pre-specified baselines at all required test cut-offs. "
                "Retain electricity as the research leader and keep LBTR diagnostic."
                if not wins_all else
                "The multiblock model clears the pre-specified test RMSE gate. This still "
                "does not establish genuine real-time validity because the target and source "
                "vintages remain incomplete.")
    lines = ["# Weekly-activity MVP assessment", "",
             f"Evaluation regime: `{EVALUATION_REGIME}`.", "",
             "## Required test comparison", "",
             *_markdown_table(pd.DataFrame(rows), ["stage", "multiblock_rmse_common",
                                                    "best_baseline_rmse_common", "decision"]),
             "", "## Decision", "", decision, "",
             "## Interpretation limits", "",
             "- Google Trends was unavailable and was not silently substituted.",
             "- LBTR starts in 2019, so pre-2020 cannot identify its contribution.",
             "- GDP and seasonal adjustment are final-vintage; source publication timestamps "
             "are not reconstructed historically.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _plot_backtest(backtest: pd.DataFrame, path: Path) -> None:
    """Small realised-versus-nowcast plot, kept separate from estimation."""
    # Production and CI have no macOS GUI event loop.  Select the file backend
    # before importing pyplot so report generation cannot abort after a valid
    # backtest has already been written.
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from analysis import plots

    plots.set_style()
    d = backtest[(backtest.stage == "month_end") & backtest.y_hat.notna()].copy()
    if d.empty:
        return
    latest = d.target_month.max() - pd.DateOffset(months=36)
    d = d[d.target_month >= latest]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    actual = d.drop_duplicates("target_month").sort_values("target_month")
    ax.plot(actual.target_month, actual.y_true, color=plots.INK, lw=2.4, label="PBI mensual realizado")
    for model, g in d.groupby("model"):
        ax.plot(g.target_month, g.y_hat, marker="o", ms=3, alpha=.85, label=model)
    ax.set(title="Nowcasts al cierre de mes y PBI mensual realizado",
           ylabel="Variación mensual SA (%)", xlabel="")
    ax.legend(ncol=2, loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _last_sunday(as_of) -> pd.Timestamp:
    d = pd.Timestamp(as_of).normalize()
    return d - pd.Timedelta(days=(d.dayofweek - 6) % 7)


def _interval_from_prior_errors(backtest: pd.DataFrame, *, model: str, stage: str,
                                as_of: pd.Timestamp, point: float) -> tuple[float, float, int]:
    d = backtest[(backtest.model == model) & (backtest.stage == stage)
                 & (pd.to_datetime(backtest.outcome_release_date) <= as_of)].dropna(subset=["y_true", "y_hat"])
    errors = (d.y_true - d.y_hat).to_numpy(dtype=float)
    if len(errors) < 12:
        return np.nan, np.nan, int(len(errors))
    lo, hi = np.quantile(errors, [0.10, 0.90])
    return float(point + lo), float(point + hi), int(len(errors))


def nowcast(data_root=DATA_ROOT, output_root=OUTPUT_ROOT, *, as_of=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce a weekly estimate for the current month from cached source data.

    The requested date is rounded down to the latest Sunday.  If a block lacks
    source observations at that cut-off, its candidate is reported unavailable
    rather than filled or silently replaced with another model.
    """
    paths = _paths(data_root, output_root)
    requested = pd.Timestamp(as_of or pd.Timestamp.today()).normalize()
    cutoff = _last_sunday(requested)
    month = cutoff.to_period("M")
    electricity = _load_required(paths["coes"], "COES processed daily data")
    lbtr = _load_required(paths["lbtr"], "BCRP LBTR processed daily data")
    target = _load_required(paths["target"], "monthly GDP target")
    features = _load_required(paths["features"], "weekly feature frame")
    current = build_feature_frame(electricity, lbtr, months=[month])
    row = current[current.cutoff_date.eq(cutoff)]
    if row.empty:
        raise RuntimeError(f"no Sunday feature row for {cutoff.date()}")
    row = row.iloc[0]
    historical = features.copy()
    historical.target_month = pd.to_datetime(historical.target_month).dt.to_period("M").dt.to_timestamp()
    target.target_month = pd.to_datetime(target.target_month).dt.to_period("M").dt.to_timestamp()
    target.target_release_date = pd.to_datetime(target.target_release_date).dt.normalize()
    train = historical.merge(target[["target_month", "pbi_mom_pct", "target_release_date"]],
                             on="target_month", how="inner")
    train = train[(train.stage == row.stage) & (train.target_release_date <= cutoff)
                  & (train.target_month < month.start_time)].sort_values("target_month")
    try:
        backtest = _load_required(paths["backtest"], "backtest results")
    except FileNotFoundError:
        backtest = pd.DataFrame()
    rows, importance = [], []
    role = {"AR(1)": "benchmark",
            "Ridge electricity": "research_leader_test_only",
            "Ridge multiblock": "not_selected_multiblock_gate_failed"}
    for model in candidate_names():
        try:
            fit = fit_candidate(model, train, row)
            point = predict_candidate(fit, month.start_time, row)
            model_train_n = int(getattr(fit, "n_train", len(train)))
            lo80, hi80, n_error = _interval_from_prior_errors(
                backtest, model=model, stage=row.stage, as_of=cutoff, point=point) if not backtest.empty else (np.nan, np.nan, 0)
            issue = ""
            imp = candidate_importance(fit).assign(model=model, cutoff_date=cutoff)
            importance.append(imp)
        except ModelUnavailable as exc:
            point, lo80, hi80, n_error, model_train_n, issue = (np.nan, np.nan, np.nan,
                                                                  0, 0, str(exc))
        rows.append({
            "requested_as_of": requested, "cutoff_date": cutoff, "target_month": month.start_time,
            "stage": row.stage, "model": model, "nowcast_pbi_mom_pct": point,
            "lo80": lo80, "hi80": hi80, "interval_error_n": n_error,
            "n_released_train_labels": len(train), "n_model_train": model_train_n,
            "electricity_last_observation": row.elec_last_observation,
            "lbtr_last_observation": row.lbtr_last_observation, "model_issue": issue,
            "model_role": role[model], "publication_status": "research_mvp_not_production",
            "evaluation_regime": EVALUATION_REGIME,
        })
    out = pd.DataFrame(rows)
    imp = pd.concat(importance, ignore_index=True) if importance else pd.DataFrame()
    Path(output_root).mkdir(parents=True, exist_ok=True)
    out.to_csv(paths["nowcast"], index=False)
    imp.to_csv(paths["importance"], index=False)
    return out, imp


def download(data_root=DATA_ROOT, *, start_coes=COES_START, start_lbtr=LBTR_START,
             end=None, refresh=False, workers=4) -> dict:
    """Download the two viable public high-frequency blocks into raw caches."""
    from .ingest import download_coes, download_lbtr

    end = pd.Timestamp(end or pd.Timestamp.today()).normalize()
    coes, coes_failures = download_coes(start_coes, end, data_root=data_root,
                                        refresh=refresh, workers=workers)
    lbtr = download_lbtr(start_lbtr, end, data_root=data_root, refresh=refresh)
    return {"coes_rows": len(coes), "coes_failures": len(coes_failures),
            "lbtr_rows": len(lbtr), "end": str(end.date())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "build", "evaluate", "nowcast"))
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--end")
    parser.add_argument("--as-of")
    parser.add_argument("--start-coes", default=COES_START)
    parser.add_argument("--start-lbtr", default=LBTR_START)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--trends-path")
    args = parser.parse_args(argv)
    if args.command == "download":
        result = download(args.data_root, start_coes=args.start_coes, start_lbtr=args.start_lbtr,
                          end=args.end, refresh=args.refresh, workers=args.workers)
    elif args.command == "build":
        result = build(args.data_root, args.output_root, trends_path=args.trends_path)
    elif args.command == "evaluate":
        result = evaluate(args.data_root, args.output_root)
    else:
        result, _ = nowcast(args.data_root, args.output_root, as_of=args.as_of)
        result = result.to_dict(orient="records")
    print(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
