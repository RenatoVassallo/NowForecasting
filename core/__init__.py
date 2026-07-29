"""Stage-agnostic engine for NowForecasting.

The shared harness every lifecycle stage (nowcast, forecast, scenario) and every
application rests on: the app descriptor (:class:`NowcastApp`), the data plumbing
(``panel``, ``preprocess``), the expanding-window backtest harness (``backtest``),
subperiod scoring (``scoring``) and the cross-run robustness tooling
(``robustness``). None of it is specific to a single stage or target.

An application imports ``core`` and calls :func:`set_app_root` from its config so
every read and write lands in that app's own ``input/`` and ``output/`` folders.
The stage packages (``nowcast``, ``forecast``, ``scenario``) build on top of this.
"""

from .appconfig import NowcastApp, Subperiod, PERU_GDP_Q
from .backtest import (
    OUTPUT_DIR as BACKTEST_OUTPUT_DIR,
    add_error_columns,
    actual_target_yoy,
    attach_relative_rmse,
    build_competition_runs,
    build_main_models,
    common_sample_rmse,
    convert_prediction_to_yoy,
    metric_table,
    plot_rmse_horse_race,
    ranking_table,
    recursive_standardize,
    run_competition_backtest,
    run_yoy_backtest,
    selected_leads_metrics,
    unscale_proxy_prediction,
    yoy_from_levels,
)
from .panel import (
    benchmark_indicators,
    canonical_column,
    load_database_panel,
    metadata_to_variable_meta,
    metadata_with_columns,
    panel_from_processed_frames,
    read_database_sheet,
    read_metadata,
)
from .preprocess import (
    BENCHMARK_MONTHLY_GDP_LEVEL,
    CACHE_DIR as PREPROCESS_CACHE_DIR,
    INPUT_DIR as PREPROCESS_INPUT_DIR,
    MONTHLY_GDP_CANONICAL,
    OUTPUT_DIR as PROCESSED_OUTPUT_DIR,
    ProcessedPanelArtifacts,
    SEED as PREPROCESS_SEED,
    SPEC_COLUMNS,
    TARGET_CANONICAL,
    audit_metadata,
    build_benchmark_artifacts,
    build_processed_artifacts,
    build_spec_artifacts,
    download_raw_inputs,
    export_processed_snapshot,
    locate_x13_binary,
    metadata_report_markdown,
    seasonal_adjust_frames,
)
from .robustness import (
    ROBUSTNESS_BACKTEST_DIR,
    ROBUSTNESS_DIR,
    ROBUSTNESS_FIG_DIR,
    ROBUSTNESS_TABLE_DIR,
    adaptive_variant_specs,
    attach_relative_to_reference,
    build_dfm_family_models,
    build_sglasso_family_models,
    competition_cache_path,
    collect_run_metrics,
    ensure_robustness_dirs,
    load_artifacts_for_spec,
    load_or_run_competition_backtest,
    load_or_run_selected_backtest,
    metrics_for_run,
    plot_relative_paths,
    refresh_competition_models,
    run_adaptive_grid,
    selected_cache_path,
    slugify,
)
from . import scoring  # noqa: F401  (also usable as ``from core import scoring``)


def set_app_root(path) -> None:
    """Point the engine's data/output paths at one application's directory.

    The modules default their input/output/cache paths to the package location;
    an application calls this from its config so every read and write lands in
    that app's own ``input/`` and ``output/`` folders. This is what makes the
    engine reusable across applications (Peru, China, ...).
    """
    from pathlib import Path as _Path
    from . import preprocess as _pp, backtest as _bt, robustness as _rb

    root = _Path(path).resolve()
    _pp.INPUT_DIR = root / "input"
    _pp.OUTPUT_DIR = root / "output" / "processed"
    _pp.CACHE_DIR = root / ".cache" / "preprocess"
    _pp.LOCAL_X13 = root / ".cache" / "x13" / "bin" / "x13as"
    _bt.OUTPUT_DIR = root / "output" / "backtests"
    _rb.ROBUSTNESS_DIR = root / "output" / "robustness"
    _rb.ROBUSTNESS_BACKTEST_DIR = _rb.ROBUSTNESS_DIR / "backtests"
    _rb.ROBUSTNESS_TABLE_DIR = _rb.ROBUSTNESS_DIR / "tables"
    _rb.ROBUSTNESS_FIG_DIR = root / "figures" / "robustness"


__all__ = [
    "set_app_root",
    "scoring",
    "NowcastApp",
    "Subperiod",
    "PERU_GDP_Q",
    "BACKTEST_OUTPUT_DIR",
    "BENCHMARK_MONTHLY_GDP_LEVEL",
    "MONTHLY_GDP_CANONICAL",
    "PREPROCESS_CACHE_DIR",
    "PREPROCESS_INPUT_DIR",
    "PREPROCESS_SEED",
    "PROCESSED_OUTPUT_DIR",
    "ProcessedPanelArtifacts",
    "SPEC_COLUMNS",
    "TARGET_CANONICAL",
    "ROBUSTNESS_BACKTEST_DIR",
    "ROBUSTNESS_DIR",
    "ROBUSTNESS_FIG_DIR",
    "ROBUSTNESS_TABLE_DIR",
    "add_error_columns",
    "adaptive_variant_specs",
    "actual_target_yoy",
    "attach_relative_to_reference",
    "attach_relative_rmse",
    "audit_metadata",
    "benchmark_indicators",
    "build_competition_runs",
    "build_main_models",
    "build_dfm_family_models",
    "build_benchmark_artifacts",
    "build_processed_artifacts",
    "build_sglasso_family_models",
    "build_spec_artifacts",
    "canonical_column",
    "collect_run_metrics",
    "competition_cache_path",
    "common_sample_rmse",
    "convert_prediction_to_yoy",
    "download_raw_inputs",
    "ensure_robustness_dirs",
    "export_processed_snapshot",
    "load_database_panel",
    "load_artifacts_for_spec",
    "load_or_run_competition_backtest",
    "load_or_run_selected_backtest",
    "locate_x13_binary",
    "metadata_report_markdown",
    "metadata_to_variable_meta",
    "metadata_with_columns",
    "metric_table",
    "metrics_for_run",
    "panel_from_processed_frames",
    "plot_rmse_horse_race",
    "plot_relative_paths",
    "refresh_competition_models",
    "ranking_table",
    "read_database_sheet",
    "read_metadata",
    "recursive_standardize",
    "run_competition_backtest",
    "run_yoy_backtest",
    "run_adaptive_grid",
    "seasonal_adjust_frames",
    "selected_leads_metrics",
    "selected_cache_path",
    "slugify",
    "unscale_proxy_prediction",
    "yoy_from_levels",
]
