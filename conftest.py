"""Make the repository root importable however pytest is invoked.

Local runs use ``python -m pytest`` (which inserts the cwd); CI uses
``uv run pytest`` (which does not). The packages here (pipeline, core,
forecast, targets, nowcast) are imported from the repo root by design, so pin
that on sys.path once, at collection time.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
