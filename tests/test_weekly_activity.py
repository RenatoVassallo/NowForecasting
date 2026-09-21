"""Contracts for the Peru weekly-activity MVP.

The tests deliberately use tiny synthetic data.  They assert information-set
rules, not an in-sample relationship with GDP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_weekly_schedule_has_sunday_cutoffs_and_a_month_end_row():
    from weekly_activity.features import cutoff_schedule

    out = cutoff_schedule(pd.Period("2026-02", freq="M"))
    weekly = out[out.stage.str.startswith("week_")]

    assert (weekly.cutoff_date.dt.dayofweek == 6).all()
    assert weekly.cutoff_date.dt.month.eq(2).all()
    assert out[out.stage.eq("month_end")].cutoff_date.iloc[0] == pd.Timestamp("2026-02-28")


def test_features_do_not_use_observations_after_the_cutoff():
    from weekly_activity.features import build_feature_frame

    dates = pd.date_range("2024-01-01", "2024-01-14", freq="D")
    electricity = pd.DataFrame({
        "date": dates,
        "coes_energy_mwh": [100.0] * 7 + [9999.0] * 7,
        "area_count": 4,
        "interval_count": 48,
    })
    lbtr = pd.DataFrame({
        "date": dates,
        "lbtr_client_value_mn": 10.0,
        "lbtr_client_value_me": 2.0,
        "lbtr_client_count_mn": 20.0,
        "lbtr_client_count_me": 3.0,
    })

    out = build_feature_frame(
        electricity, lbtr, months=[pd.Period("2024-01", freq="M")],
        stages=("week_1",), include_trends=False,
    )
    row = out.iloc[0]
    assert row.cutoff_date == pd.Timestamp("2024-01-07")
    assert row.elec_energy_mwh_mtd == 700.0
    assert row.elec_observed_days == 7
    assert row.elec_calendar_coverage == 1.0
    assert row.lbtr_value_mn_mtd == 70.0


def test_target_availability_gate_is_explicit_scalar_assumption():
    from weekly_activity.target import target_release_date, target_is_known

    month = pd.Period("2026-05", freq="M")
    assert target_release_date(month, lag_days=51) == pd.Timestamp("2026-07-21")
    assert not target_is_known(month, "2026-07-20", lag_days=51)
    assert target_is_known(month, "2026-07-21", lag_days=51)


def test_coes_energy_integrates_half_hourly_mw_without_silent_gaps():
    from weekly_activity.ingest import (_merge_latest_by_date, aggregate_coes_daily,
                                        clean_coes_payload)

    payload = [{"MEDIFECHA": "2026-08-10T00:00:00", "PTOMEDICODI": 3004,
                "PTOMEDIELENOMB": "AREA NORTE", **{f"h{i}": 2.0 for i in range(1, 49)}}]
    out = clean_coes_payload(payload, source_url="https://example.test", retrieved_at="2026-08-11")
    assert out.energy_mwh.iloc[0] == 48.0
    assert out.interval_count.iloc[0] == 48

    incomplete = [dict(payload[0], h48=None)]
    partial = clean_coes_payload(incomplete, source_url="https://example.test", retrieved_at="2026-08-11")
    assert not partial.is_complete.iloc[0]
    assert np.isnan(partial.energy_mwh.iloc[0])

    all_areas = pd.concat([
        out.assign(area_code="3004"), out.assign(area_code="3005"),
        out.assign(area_code="3006"), partial.assign(area_code="3009"),
    ], ignore_index=True)
    daily = aggregate_coes_daily(all_areas)
    assert daily.core_complete.iloc[0]
    assert daily.coes_energy_mwh.iloc[0] == 144.0

    old = pd.DataFrame({"date": [pd.Timestamp("2026-08-10")], "value": [1.0]})
    fresh = pd.DataFrame({"date": [pd.Timestamp("2026-08-10")], "value": [2.0]})
    assert _merge_latest_by_date(old, fresh).value.iloc[0] == 2.0


def test_backtest_never_trains_on_target_not_released_at_cutoff():
    from weekly_activity.backtest import run_backtest

    months = pd.period_range("2019-01", "2024-12", freq="M")
    rows = []
    for month in months:
        for stage, cutoff in [("week_1", pd.Timestamp(month.start_time) + pd.Timedelta(days=6)),
                              ("month_end", pd.Timestamp(month.end_time).normalize())]:
            rows.append({"target_month": month.start_time, "stage": stage,
                         "cutoff_date": cutoff, "elec_log_mean_mwh": float(month.ordinal % 11),
                         "elec_observed_days": 7.0, "elec_elapsed_days": 7.0,
                         "month_sin": 0.0, "month_cos": 1.0,
                         "lbtr_log_mean_value_mn": float(month.ordinal % 7),
                         "lbtr_log_mean_value_me": 1.0,
                         "lbtr_log_mean_count_mn": 2.0,
                         "lbtr_log_mean_count_me": 1.0,
                         "lbtr_observed_days": 5.0})
    features = pd.DataFrame(rows)
    target = pd.DataFrame({"target_month": months.to_timestamp(),
                           "pbi_mom_pct": np.sin(np.arange(len(months)) / 5),
                           "target_release_date": [
                               pd.Timestamp(m.end_time).normalize() + pd.Timedelta(days=51)
                               for m in months]})
    out = run_backtest(features, target, min_train_months=24)
    assert not out.empty
    assert (pd.to_datetime(out.train_last_release_date) <= pd.to_datetime(out.cutoff_date)).all()
    assert (pd.to_datetime(out.train_last_target_month) < pd.to_datetime(out.target_month)).all()


def test_electricity_gdp_diagnostics_keep_gdp_at_monthly_frequency():
    from weekly_activity.diagnostics import prepare_diagnostic_data

    dates = pd.date_range("2018-01-01", "2020-12-31", freq="D")
    electricity = pd.DataFrame({"date": dates, "coes_energy_mwh": 100.0})
    months = pd.period_range("2018-01", "2020-12", freq="M").to_timestamp()
    target = pd.DataFrame({"target_month": months,
                           "pbi_sa_index": np.linspace(90.0, 110.0, len(months))})

    data = prepare_diagnostic_data(electricity, target)
    gdp = data["gdp"]
    monthly = data["monthly"]

    assert len(gdp) == len(months)
    assert gdp.pbi_plot_date.equals(gdp.target_month + pd.Timedelta(days=14))
    assert monthly.target_month.is_unique
    assert monthly.observed_days.eq(monthly.calendar_days).all()


def test_completed_monthly_coes_block_rejects_incomplete_months_and_preserves_gaps():
    from weekly_activity.quarterly import COES_MONTHLY_COLUMN, completed_monthly_coes_block

    # Jan-2020 is deliberately incomplete.  The later annual transform must
    # remain unavailable rather than bridge across that missing calendar month.
    dates = pd.date_range("2020-01-02", "2021-02-28", freq="D")
    daily = pd.DataFrame({"date": dates, "coes_energy_mwh": 100.0})
    extra, audit = completed_monthly_coes_block(daily)

    jan_2020 = audit.loc[audit.target_month.eq(pd.Timestamp("2020-01-01"))].iloc[0]
    assert not jan_2020.complete_calendar_month
    assert pd.isna(extra.loc[pd.Timestamp("2021-01-01"), COES_MONTHLY_COLUMN])
    assert np.isfinite(extra.loc[pd.Timestamp("2021-02-01"), COES_MONTHLY_COLUMN])
