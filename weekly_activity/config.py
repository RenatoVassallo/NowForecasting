"""Central configuration for the weekly-activity MVP.

All paths are repo-relative.  Input and output data are deliberately outside
version control under the project-wide data policy.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("WEEKLY_ACTIVITY_DATA_DIR",
                                REPO_ROOT / "input" / "weekly_activity"))
OUTPUT_ROOT = Path(os.environ.get("WEEKLY_ACTIVITY_OUTPUT_DIR",
                                  REPO_ROOT / "output" / "weekly_activity"))

# Public source coverage, verified 2026-08-18.  COES was queried at 2015-01-02
# and BCRP's client LBTR series begin 2019-01-02.  These are not availability
# timestamps for historical observations.
COES_START = "2015-01-01"
LBTR_START = "2019-01-02"
EVALUATION_START = COES_START  # no AR-only scoring before the first MVP block exists

# The existing Peru target registry uses this conservative scalar.  It is an
# explicit approximation, not a reconstructed historical release calendar.
TARGET_RELEASE_LAG_DAYS = 51

MIN_TRAIN_MONTHS = 24
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
STAGES_TO_REPORT = ("week_1", "week_2", "month_end")
VALIDATION_END = "2023-12-01"
TEST_START = "2024-01-01"

# Source data use observation dates in historical simulations, because neither
# provider exposes a public, historic first-publication timestamp.  The label
# prevents those exercises from being presented as genuine real-time backtests.
EVALUATION_REGIME = "pseudo_real_time_final_vintage_observation_clock"
