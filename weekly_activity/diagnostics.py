"""Descriptive electricity and monthly-GDP diagnostics for the weekly MVP.

These figures deliberately keep the two observation frequencies visible.  COES
is plotted at weekly frequency; monthly GDP is plotted only once per month,
anchored at day 15 and joined only to its own monthly observations.  No monthly
GDP value is interpolated into weeks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_ROOT, OUTPUT_ROOT


WINDOWS = (
    ("2015-2026, incluida la pandemia", "2015-01-01", "2026-12-31"),
    ("2015-2019, antes de la pandemia", "2015-01-01", "2019-12-31"),
    ("2022-2026, recuperación reciente", "2022-01-01", "2026-12-31"),
)
SCATTER_WINDOWS = (
    ("2015-2019", "2015-01-01", "2019-12-31"),
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2022-2026", "2022-01-01", "2026-12-31"),
)


def _as_daily_electricity(electricity: pd.DataFrame) -> pd.DataFrame:
    """Return one valid COES core-area total per observed day."""
    required = {"date", "coes_energy_mwh"}
    missing = required.difference(electricity.columns)
    if missing:
        raise ValueError(f"electricity frame misses {sorted(missing)}")
    out = electricity.loc[:, ["date", "coes_energy_mwh"]].copy()
    out["date"] = pd.to_datetime(out.date, errors="coerce").dt.normalize()
    out["coes_energy_mwh"] = pd.to_numeric(out.coes_energy_mwh, errors="coerce")
    return out.dropna().drop_duplicates("date", keep="last").sort_values("date")


def _as_monthly_gdp(target: pd.DataFrame) -> pd.DataFrame:
    """Keep observed monthly GDP at one point per reference month."""
    required = {"target_month", "pbi_sa_index"}
    missing = required.difference(target.columns)
    if missing:
        raise ValueError(f"target frame misses {sorted(missing)}")
    out = target.loc[:, ["target_month", "pbi_sa_index"]].copy()
    out["target_month"] = pd.to_datetime(out.target_month, errors="coerce").dt.to_period("M").dt.to_timestamp()
    out["pbi_sa_index"] = pd.to_numeric(out.pbi_sa_index, errors="coerce")
    out = out.dropna().drop_duplicates("target_month", keep="last").sort_values("target_month")
    # A mid-month marker improves temporal legibility on a weekly x axis.  It
    # is a plotting coordinate only, not a weekly GDP estimate.
    out["pbi_plot_date"] = out.target_month + pd.Timedelta(days=14)
    out["pbi_yoy_pct"] = 100.0 * np.log(out.pbi_sa_index).diff(12)
    return out


def prepare_diagnostic_data(electricity: pd.DataFrame, target: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Construct weekly and genuine monthly descriptive series.

    Monthly electricity is an aggregation of daily observations.  It is kept
    only when all calendar days of that month are present, so an incomplete
    current month cannot masquerade as a complete observation.
    """
    e = _as_daily_electricity(electricity)
    g = _as_monthly_gdp(target)

    weekly = (e.set_index("date").resample("W-SUN").agg(
        elec_energy_mwh=("coes_energy_mwh", "sum"),
        observed_days=("coes_energy_mwh", "size"),
    ).reset_index(names="week_end"))
    weekly = weekly[weekly.observed_days.eq(7)].copy()
    weekly["elec_daily_mean_mwh"] = weekly.elec_energy_mwh / 7.0
    weekly["elec_daily_mean_13w"] = weekly.elec_daily_mean_mwh.rolling(13, min_periods=13).mean()

    e["target_month"] = e.date.dt.to_period("M").dt.to_timestamp()
    monthly = e.groupby("target_month", as_index=False).agg(
        elec_energy_mwh=("coes_energy_mwh", "sum"),
        observed_days=("date", "nunique"),
    )
    monthly["calendar_days"] = monthly.target_month.dt.days_in_month
    monthly = monthly[monthly.observed_days.eq(monthly.calendar_days)].copy()
    monthly["elec_yoy_pct"] = 100.0 * np.log(monthly.elec_energy_mwh).diff(12)
    monthly["plot_date"] = monthly.target_month + pd.Timedelta(days=14)
    monthly = monthly.merge(g, on="target_month", how="inner")

    base_e = weekly.loc[weekly.week_end.dt.year.eq(2019), "elec_daily_mean_13w"].mean()
    base_g = g.loc[g.target_month.dt.year.eq(2019), "pbi_sa_index"].mean()
    if not np.isfinite(base_e) or not np.isfinite(base_g):
        raise ValueError("2019 base is unavailable for one of the descriptive series")
    weekly["elec_index_2019"] = 100.0 * weekly.elec_daily_mean_mwh / base_e
    weekly["elec_trend_index_2019"] = 100.0 * weekly.elec_daily_mean_13w / base_e
    g["pbi_index_2019"] = 100.0 * g.pbi_sa_index / base_g
    return {"weekly": weekly, "gdp": g, "monthly": monthly}


def _window(frame: pd.DataFrame, date_col: str, start: str, end: str) -> pd.DataFrame:
    d = pd.to_datetime(frame[date_col])
    return frame[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))].copy()


def _date_axis(ax) -> None:
    import matplotlib.dates as mdates

    locator = mdates.YearLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def plot_level_paths(data: dict[str, pd.DataFrame], path: Path) -> None:
    """Weekly electricity levels versus discrete monthly GDP level markers."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from analysis import plots

    plots.set_style()
    weekly, gdp = data["weekly"], data["gdp"]
    fig, axes = plt.subplots(3, 1, figsize=(10.4, 9.2), sharey=True)
    for ax, (title, start, end) in zip(axes, WINDOWS):
        w = _window(weekly, "week_end", start, end)
        g = _window(gdp, "pbi_plot_date", start, end)
        ax.plot(w.week_end, w.elec_index_2019, color=plots.PALETTE[1], alpha=.18, lw=.65,
                label="Electricidad semanal, promedio diario")
        ax.plot(w.week_end, w.elec_trend_index_2019, color=plots.PALETTE[1], lw=2.0,
                label="Electricidad, media móvil trailing de 13 semanas")
        ax.plot(g.pbi_plot_date, g.pbi_index_2019, color=plots.PALETTE[0], lw=1.35,
                marker="o", ms=3.2, label="PBI mensual SA, punto al día 15")
        ax.axhline(100, color=plots.MUTED, lw=.8, ls=":")
        ax.set(title=title, ylabel="Índice, promedio 2019 = 100")
        _date_axis(ax)
    axes[0].legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.25))
    axes[-1].set_xlabel("Fecha de referencia. El PBI no se interpola entre meses.")
    fig.suptitle("Electricidad semanal y PBI mensual: niveles comparables", x=.125, ha="left",
                 fontsize=13, fontweight="semibold")
    fig.savefig(path)
    plt.close(fig)


def plot_growth_paths(data: dict[str, pd.DataFrame], path: Path) -> None:
    """Annual changes from genuinely monthly electricity aggregates and GDP."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from analysis import plots

    plots.set_style()
    monthly = data["monthly"].dropna(subset=["elec_yoy_pct", "pbi_yoy_pct"])
    fig, axes = plt.subplots(3, 1, figsize=(10.4, 9.2), sharex=False, sharey=False)
    for ax, (title, start, end) in zip(axes, WINDOWS):
        d = _window(monthly, "plot_date", start, end)
        ax.axhline(0, color=plots.MUTED, lw=.8, ls=":")
        ax.plot(d.plot_date, d.elec_yoy_pct, color=plots.PALETTE[1], lw=1.9,
                marker="o", ms=2.9, label="Electricidad mensual, variación anual")
        ax.plot(d.plot_date, d.pbi_yoy_pct, color=plots.PALETTE[0], lw=1.7,
                marker="o", ms=2.9, label="PBI mensual SA, variación anual")
        ax.set(title=title, ylabel="Variación anual (%)")
        _date_axis(ax)
    axes[0].legend(ncol=2, loc="upper left", bbox_to_anchor=(0, 1.25))
    axes[-1].set_xlabel("Mes de referencia, representado por un punto al día 15")
    fig.suptitle("Co-movimiento mensual: crecimiento anual de electricidad y PBI", x=.125,
                 ha="left", fontsize=13, fontweight="semibold")
    fig.text(.125, .005, "Las escalas verticales son específicas a cada panel para hacer visible el período sin COVID.",
             fontsize=8.5, color=plots.MUTED)
    fig.savefig(path)
    plt.close(fig)


def _fit_line(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 3 or d.x.nunique() < 2:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(d.x, d.y, 1)
    return float(slope), float(intercept), float(d.x.corr(d.y))


def plot_growth_scatter(data: dict[str, pd.DataFrame], path: Path) -> pd.DataFrame:
    """Regime-specific scatterplots to prevent COVID-driven full-sample claims."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from analysis import plots

    plots.set_style()
    monthly = data["monthly"].dropna(subset=["elec_yoy_pct", "pbi_yoy_pct"])
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    summary = []
    for i, (ax, (title, start, end)) in enumerate(zip(axes, SCATTER_WINDOWS)):
        d = _window(monthly, "plot_date", start, end)
        slope, intercept, corr = _fit_line(d.elec_yoy_pct, d.pbi_yoy_pct)
        ax.axhline(0, color=plots.MUTED, lw=.8, ls=":")
        ax.axvline(0, color=plots.MUTED, lw=.8, ls=":")
        ax.scatter(d.elec_yoy_pct, d.pbi_yoy_pct, color=plots.PALETTE[i], s=25, alpha=.8)
        if np.isfinite(slope):
            xx = np.linspace(d.elec_yoy_pct.min(), d.elec_yoy_pct.max(), 100)
            ax.plot(xx, intercept + slope * xx, color=plots.INK, lw=1.25)
        ax.set(title=title, xlabel="Electricidad mensual, variación anual (%)")
        if i == 0:
            ax.set_ylabel("PBI mensual SA, variación anual (%)")
        ax.text(.03, .96, f"n = {len(d)}\nr = {corr:.2f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=9, color=plots.INK)
        summary.append({"period": title, "start": start, "end": end, "n": len(d),
                        "pearson_correlation": corr, "ols_slope": slope})
    fig.suptitle("El co-movimiento contemporáneo cambia por régimen", x=.125, ha="left",
                 fontsize=13, fontweight="semibold")
    fig.savefig(path)
    plt.close(fig)
    return pd.DataFrame(summary)


def render_diagnostics(data_root=DATA_ROOT, output_root=OUTPUT_ROOT) -> dict[str, Path]:
    """Write the three reproducible descriptive figures and their numeric summary."""
    data_root, output_root = Path(data_root), Path(output_root)
    electricity = pd.read_parquet(data_root / "processed" / "coes_daily.parquet")
    target = pd.read_parquet(data_root / "processed" / "target_monthly.parquet")
    data = prepare_diagnostic_data(electricity, target)
    out = output_root / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "levels": out / "electricity_pbi_levels.png",
        "growth": out / "electricity_pbi_growth.png",
        "scatter": out / "electricity_pbi_growth_scatter.png",
        "summary": out / "electricity_pbi_growth_correlation.csv",
    }
    plot_level_paths(data, paths["levels"])
    plot_growth_paths(data, paths["growth"])
    plot_growth_scatter(data, paths["scatter"]).to_csv(paths["summary"], index=False)
    return paths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args(argv)
    result = render_diagnostics(args.data_root, args.output_root)
    print("\n".join(f"{name}: {path}" for name, path in result.items()))


if __name__ == "__main__":  # pragma: no cover
    main()
