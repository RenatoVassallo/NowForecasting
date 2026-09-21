"""Exploratory, common-sample screen of completed-month COES in Peru's quarterly nowcast.

This does not alter production metadata or the published Adaptive-IC ensemble.
It answers a narrower question: after a completed month is available, does
COES improve the existing Peru quarterly candidates on the same forecast cells?

Run:
    PYTHONPATH=../MIDAS/src:../MacroPy/src python -m weekly_activity.quarterly_screen
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_ROOT, OUTPUT_ROOT
from .quarterly import (COES_MONTHLY_COLUMN, DEFAULT_COES_DELAY_DAYS,
                        coes_monthly_metadata, load_completed_monthly_coes_block)


LEADERS = ("g_pbim_yoy", "exp_sec3m", "exp_eco3m", "cem")
PERIODS = {
    "2017-2019": ("2017-01-01", "2019-12-31", ()),
    "2022-2026": ("2022-01-01", "2026-12-31", ()),
    "full ex-COVID": ("2017-01-01", "2026-12-31", (2020, 2021)),
    "near publication ex-COVID": ("2017-01-01", "2026-12-31", (2020, 2021)),
}


def _models():
    from MIDAS import BridgeNowcaster, PooledMIDASNowcaster, RandomWalkNowcaster

    plus_coes = [*LEADERS, COES_MONTHLY_COLUMN]
    return {
        "RW": RandomWalkNowcaster(),
        "Bridge(leaders)": BridgeNowcaster(indicators=list(LEADERS)),
        "P-MIDAS(leaders)": PooledMIDASNowcaster(monthly_vars=list(LEADERS)),
        "Bridge(leaders+COES)": BridgeNowcaster(indicators=plus_coes),
        "P-MIDAS(leaders+COES)": PooledMIDASNowcaster(monthly_vars=plus_coes),
        "Bridge(COES)": BridgeNowcaster(indicators=[COES_MONTHLY_COLUMN]),
    }


def common_sample_score(backtest: pd.DataFrame) -> pd.DataFrame:
    """Score all candidates only where every candidate produced a point estimate."""
    rows = []
    for label, (start, end, exclude) in PERIODS.items():
        d = backtest[(backtest.ref_quarter >= pd.Timestamp(start))
                     & (backtest.ref_quarter <= pd.Timestamp(end))].copy()
        if label.startswith("near publication"):
            d = d[d.days_to_publication >= -30]
        if exclude:
            d = d[~pd.to_datetime(d.ref_quarter).dt.year.isin(exclude)]
        wide = d.pivot_table(index=["ref_quarter", "days_to_publication"], columns="model",
                             values=["y_hat", "y_true"], aggfunc="first")
        if wide.empty or "y_hat" not in wide:
            continue
        hats = wide["y_hat"].dropna()
        if hats.empty or "y_true" not in wide:
            continue
        y = wide["y_true"].iloc[:, 0].reindex(hats.index)
        rw = hats.get("RW")
        for model in hats:
            errors = y - hats[model]
            fallback_share = (np.isclose(hats[model], rw, equal_nan=False).mean()
                              if rw is not None else np.nan)
            rows.append({"period": label, "model": model, "common_n": len(errors),
                         "common_quarters": hats.index.get_level_values("ref_quarter").nunique(),
                         "rmse_common": float(np.sqrt(np.mean(errors ** 2))),
                         "mae_common": float(np.mean(np.abs(errors))),
                         "rw_identical_share": float(fallback_share)})
    return pd.DataFrame(rows).sort_values(["period", "rmse_common", "model"]).reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small report table without adding an optional dependency."""
    if frame.empty:
        return "No comparable candidate cells."
    d = frame.copy()
    for col in d.select_dtypes(include="number"):
        d[col] = d[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    header = "| " + " | ".join(d.columns) + " |"
    rule = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |"
            for row in d.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def _report(score: pd.DataFrame, *, delay_days: int, audit: pd.DataFrame) -> str:
    focus = score[score.model.isin(["Bridge(leaders)", "Bridge(leaders+COES)"])]
    lines = ["# COES completed-month quarterly screen", "",
             "Status: exploratory, not a production-selection exercise.", "",
             f"Availability scenario: completed month plus {delay_days} calendar day(s).",
             "Historical COES first-publication timestamps are unavailable, so this is a ",
             "pseudo-real-time final-vintage screen.", "",
             "## Common-sample results", "",
             _markdown_table(score), "",
             "## Integration rule", "",
             "Do not add COES to `pipeline/config/metadata.py` or to Adaptive-IC from this screen alone.",
             "Promotion requires a pre-specified, delay-sensitivity-stable improvement over",
             "`Bridge(leaders)` in an exact-origin backtest, a non-trivial non-fallback share,",
             "and prospective weekly-vintage evidence.", "",
             "## Data contract", "",
             f"- Complete COES months in cache: {int(audit.complete_calendar_month.sum())}",
             f"- Incomplete months excluded before transform: {int((~audit.complete_calendar_month).sum())}",
             f"- Monthly feature: `{COES_MONTHLY_COLUMN}`, annual log change of completed-month energy.",
             "- Current-month partial electricity is intentionally outside this screen.", "",
             "## Decision aid", "",
             _markdown_table(focus),
             ""]
    return "\n".join(lines)


def run_screen(*, data_root=DATA_ROOT, delay_days: int = DEFAULT_COES_DELAY_DAYS, output_root=OUTPUT_ROOT,
               eval_start: str = "2017-01-01", eval_end: str = "2026-12-31",
               n_jobs: int = 1) -> dict[str, Path]:
    """Run one scenario without touching production objects or artifacts."""
    from nowcast.release_cycle import run_horse_race
    from targets import peru_gdp

    extra, audit = load_completed_monthly_coes_block(data_root=data_root)
    _, _, panel = peru_gdp.load_panel(extra=extra,
                                      extra_meta=[coes_monthly_metadata(delay_days=delay_days)])
    backtest = run_horse_race(panel, peru_gdp.SPEC, _models(), n_jobs=n_jobs,
                              eval_start=eval_start, eval_end=eval_end)
    score = common_sample_score(backtest)
    out = Path(output_root) / "quarterly_coes_screen"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"delay_{int(delay_days)}d"
    paths = {"backtest": out / f"{stem}_nowcasts.parquet",
             "scoreboard": out / f"{stem}_scoreboard.csv",
             "report": out / f"{stem}_report.md",
             "audit": out / f"{stem}_monthly_audit.csv"}
    backtest.to_parquet(paths["backtest"], index=False)
    score.to_csv(paths["scoreboard"], index=False)
    audit.to_csv(paths["audit"], index=False)
    paths["report"].write_text(_report(score, delay_days=delay_days, audit=audit))
    return paths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--delay-days", type=int, default=DEFAULT_COES_DELAY_DAYS)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--eval-start", default="2017-01-01")
    parser.add_argument("--eval-end", default="2026-12-31")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    for name, path in run_screen(data_root=args.data_root, delay_days=args.delay_days, output_root=args.output_root,
                                 eval_start=args.eval_start, eval_end=args.eval_end,
                                 n_jobs=args.n_jobs).items():
        print(f"{name}: {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
