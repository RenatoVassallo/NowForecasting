"""Nowcast stage (h = 0): release-cycle workflow.

The within-quarter nowcast problem, built on the :mod:`core` engine. ``ladder``
composes the simple-to-complex model ladder and runs the release-cycle horse
race for a quarterly target; ``monthly_target`` is the monthly-frequency
counterpart (nowcasting the monthly GDP proxy at parity). The adaptive-IC
combination and the release-cycle bands live in the ``MIDAS`` package and are
wired in from the apps' ``03`` notebooks.

Forecasts (h = 1..H) live in :mod:`forecast`; scenarios in :mod:`scenario`; both
rest on the same :mod:`core` engine and the same ``NowcastApp`` config.
"""

from .ladder import (
    build_ladder_runs,
    default_small_block,
    real_soft_blocks,
    run_ladder_backtest,
)
from .release_cycle import (
    adaptive_combine,
    conditional_bands,
    latest_nowcast,
    live_path,
    run_horse_race,
)
from .monthly_target import (
    BaseMonthlyNowcaster,
    MonthlyAdaptiveComboNowcaster,
    MonthlyARNowcaster,
    MonthlyARXNowcaster,
    MonthlyAverageComboNowcaster,
    MonthlyBlockRidgeNowcaster,
    MonthlyInformationSet,
    MonthlyNowcastResult,
    MonthlyPooledBridgeNowcaster,
    MonthlyRealtimeEngine,
    attach_monthly_relative_rmse,
    month_publication_date,
    monthly_selected_leads_metrics,
    quarterly_mean_from_monthly_backtest,
    quarterly_mean_from_monthly_series,
    run_monthly_selected_leads_backtest,
    run_quarterly_bridge_from_monthly_model,
    screen_monthly_indicators,
    select_stable_monthly_indicators,
)

__all__ = [
    "build_ladder_runs",
    "default_small_block",
    "real_soft_blocks",
    "run_ladder_backtest",
    "adaptive_combine",
    "conditional_bands",
    "latest_nowcast",
    "live_path",
    "run_horse_race",
    "BaseMonthlyNowcaster",
    "MonthlyAdaptiveComboNowcaster",
    "MonthlyARNowcaster",
    "MonthlyARXNowcaster",
    "MonthlyAverageComboNowcaster",
    "MonthlyBlockRidgeNowcaster",
    "MonthlyInformationSet",
    "MonthlyNowcastResult",
    "MonthlyPooledBridgeNowcaster",
    "MonthlyRealtimeEngine",
    "attach_monthly_relative_rmse",
    "month_publication_date",
    "monthly_selected_leads_metrics",
    "quarterly_mean_from_monthly_backtest",
    "quarterly_mean_from_monthly_series",
    "run_monthly_selected_leads_backtest",
    "run_quarterly_bridge_from_monthly_model",
    "screen_monthly_indicators",
    "select_stable_monthly_indicators",
]
