"""Build and refresh the one-off deep-history NBS monthly panel.

The powerful monthly activity indicators for China (industrial production, PMI, retail,
investment, real estate, electricity, money) are only available with deep history from
the NBS ``stream/esData`` endpoint. They are registered in ``data_china.py`` under the
``portal_stream`` backend, so a single pull returns the full history AND the latest
month; there is no separate short-window update step.

This module is the "build once, refresh monthly" tool:

- ``build_monthly_history()`` pulls the whole panel and returns a wide DataFrame.
- Running the module (``python -m utils.nbs_history``) writes it to
  ``input/nbs_monthly_history.csv``, the committed historical dataset. Re-running it is
  the monthly refresh: the stream pull always includes the newest print.

If the NBS endpoint is ever unreachable, the committed CSV remains as the historical
spine, and only the latest month needs to be appended from any working source.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .nbs import FROZEN_CSV_DIR, get_china_data

# Friendly column name -> registered indicator code. All are portal_stream (deep history).
NBS_MONTHLY_DEEP: dict[str, str] = {
    "ip_yoy": "industrial_value_added_yoy",
    "ip_cum_yoy": "industrial_value_added_cum_yoy",
    "cpi_index_yoy": "cpi_yoy",
    "ppi_index_yoy": "ppi_yoy",
    "manufacturing_pmi": "manufacturing_pmi",
    "non_manufacturing_pmi": "non_manufacturing_pmi",
    "retail_sales_yoy": "retail_sales_yoy",
    "fixed_asset_investment_cum_yoy": "fixed_asset_investment_yoy",
    "real_estate_investment_cum_yoy": "real_estate_investment_cum_yoy",
    "property_sales_floor_cum_yoy": "property_sales_floor_cum_yoy",
    "electricity_output_yoy": "electricity_output_yoy",
    "m2_yoy": "m2_yoy",
}

DEFAULT_OUTPUT = FROZEN_CSV_DIR / "nbs_monthly_history.csv"


def build_monthly_history(
    start_period: str = "1990M01",
    cache_dir: str | Path | None = None,
    refresh: bool = True,
) -> pd.DataFrame:
    """Pull the full deep-history NBS monthly panel as a wide, period-indexed frame."""
    panel = get_china_data(
        NBS_MONTHLY_DEEP,
        frequency="monthly",
        start_period=start_period,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    panel["period"] = panel["period"].astype(str)
    return panel.set_index("period").sort_index()


def save_monthly_history(
    output_path: str | Path = DEFAULT_OUTPUT,
    start_period: str = "1990M01",
    cache_dir: str | Path | None = None,
) -> Path:
    """Build the panel and write it to a committed CSV. Returns the path written."""
    frame = build_monthly_history(start_period=start_period, cache_dir=cache_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path)
    return output_path


if __name__ == "__main__":
    path = save_monthly_history()
    written = pd.read_csv(path, index_col="period")
    print(f"Wrote {path} ({written.shape[0]} months x {written.shape[1]} indicators)")
    coverage = pd.DataFrame(
        {
            "first": written.apply(lambda c: c.dropna().index.min()),
            "last": written.apply(lambda c: c.dropna().index.max()),
            "n": written.notna().sum(),
        }
    )
    print(coverage.to_string())
