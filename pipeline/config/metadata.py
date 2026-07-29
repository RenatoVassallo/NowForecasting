"""Model metadata: the production model specs, hyperparameters and combination.

The single source of truth for WHAT the pipeline estimates (the app ``models.py``
files are for experimentation only). A spec is ``(MIDAS class name, kwargs)``; the
dict key is the model's name in every artifact. Add a model by adding a row; add a
target by adding a block.
"""

from __future__ import annotations

# --- indicator blocks (from the step-2 information analysis) ------------------
PERU_LEADERS = ["g_pbim_yoy", "exp_sec3m", "exp_eco3m", "cem"]     # proxy + 3m expectations + cement
CHINA_ACTIVITY = ["ip_cum_yoy", "retail_sales_yoy", "electricity_output_yoy"]
CHINA_LEADERS = ["ip_cum_yoy", "m2_yoy", "property_sales_floor_cum_yoy"]

# --- model ladders (name -> (MIDAS class, kwargs)) ---------------------------
MODELS: dict[str, dict[str, tuple]] = {
    "china": {
        "RW":               ("RandomWalkNowcaster", {}),
        "Q-AR":             ("ARNowcaster", {"order": 1}),
        "Bridge(activity)": ("BridgeNowcaster", {"indicators": CHINA_ACTIVITY}),
        "P-MIDAS(leaders)": ("PooledMIDASNowcaster", {"monthly_vars": CHINA_LEADERS}),
    },
    "peru_gdp": {
        "RW":               ("RandomWalkNowcaster", {}),
        "Q-AR":             ("ARNowcaster", {"order": 1}),
        "Bridge(leaders)":  ("BridgeNowcaster", {"indicators": PERU_LEADERS}),
        "P-MIDAS(leaders)": ("PooledMIDASNowcaster", {"monthly_vars": PERU_LEADERS}),
        #"DFM-F":            ("DFMNowcaster", {"factors": 3, "factor_orders": 2,
        #                                      "idiosyncratic_ar1": True, "maxiter": 200}),
    },
}

# --- Adaptive-IC combination (the headline product) --------------------------
ADAPTIVE_MEMBERS: dict[str, list[str]] = {
    "china":    ["P-MIDAS(leaders)", "Bridge(activity)"],
    "peru_gdp": ["RW", "Bridge(leaders)", "P-MIDAS(leaders)"],
}
ADAPTIVE = {"name": "Adaptive-IC", "n_bins": 4, "min_train": 6,
            "method": "inv_mse", "window_months": 6}

# --- uncertainty bands --------------------------------------------------------
# Production calibration is DYNAMIC: the empirical error bands use only the last
# ``lookback_years`` of real-time errors (rolling), unlike the fixed academic
# subperiods used in the research notebooks. COVID years never enter.
BANDS = {"levels": (0.50, 0.70, 0.90), "lookback_years": 10,
         "exclude_years": (2020, 2021), "min_quarters": 6}

# --- backtest grid ------------------------------------------------------------
BACKTEST = {"days_before": 120, "step_days": 15}

# --- live sweep of the current, unpublished quarter ---------------------------
LIVE = {"step_days": 7}          # weekly nowcast path up to today

# --- dynamic evaluation window for the production report ----------------------
METRICS_LOOKBACK_YEARS = 10      # adds a rolling "last 10y ex-COVID" scoreboard

# --- forecast stage (h = 1..8) ------------------------------------------------
# Lean production ladder per target: the US block is deliberately NOT in China's
# spec (tested in notebooks/china/04: it worsens every horizon); the US external
# consensus is still vintaged every run and feeds the commodity/scenario stages.
HORIZONS = tuple(range(1, 9))
FORECAST: dict[str, dict] = {
    "china": {
        # Member evidence (notebooks/china/04 + the BVAR eval): AR(1) mean-reverts
        # to the pre-2012 boom and is OUT of the Combo (kept as a scoreboard
        # reference); the MacroPy BVAR (Minnesota prior shrinks toward RW,
        # Lenza-Primiceri COVID scaling) is best at h=4 alone and lifts the
        # Combo at h=8; biases roughly halve without AR.
        "models": {
            "RW":             ("RandomWalkNowcaster", {}),
            "AR(1)":          ("ARNowcaster", {"order": 1}),
            "D-ARX(leaders)": ("DirectARXNowcaster", {"indicators": CHINA_LEADERS,
                                                      "min_train": 24}),
            "BVAR":           ("BVARNowcaster", {"variables": ["ip_cum_yoy", "m2_yoy"],
                                                 "lags": 2, "post_draws": 800}),
        },
        "members": ["RW", "D-ARX(leaders)", "BVAR"],
        "benchmark": "RW",
    },
    "copper": {
        # Notebook findings (commodities/01): AR beats RW everywhere in YoY space
        # (mean reversion); the CHINA demand block wins h>=3; the dollar HURTS.
        "models": {
            "RW":           ("RandomWalkNowcaster", {}),
            "AR(1)":        ("ARNowcaster", {"order": 1}),
            "D-ARX(china)": ("DirectARXNowcaster", {"indicators": ["ip_cum_yoy", "m2_yoy"],
                                                    "min_train": 24}),
        },
        "members": ["AR(1)", "D-ARX(china)"],   # RW stays as the benchmark only
        "benchmark": "RW",
    },
}
FORECAST_COMBO = "Combo"
FORECAST_LOOKBACK_YEARS = 12     # backtest window feeding weights/fans
FAN = {"level": 0.90, "lookback_years": 10, "exclude_years": (2020, 2021)}
