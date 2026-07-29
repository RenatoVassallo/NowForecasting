"""Production block builders: pure Python, no notebook execution.

Each block is a function that rebuilds one published object from source data and
returns ``(DataFrame, report_lines)``. The notebooks under ``notebooks/`` remain
the development and documentation surface; these modules are what runs nightly.
They deliberately share the same library code the notebooks import
(``targets``, ``forecast``, ``sources``), so the two cannot drift in method -
only in presentation.

Every upstream block publishes the same contract:

    quarter, h, source, centre, mode, s, gamma, sigma_left, sigma_right, lo*, hi*

where ``(s, gamma)`` are the two-piece-normal scale and skew fitted to that
block's own real-time errors.
"""

from .usa import build as build_usa
from .china import build as build_china
from .commodities import build as build_commodities
from .peru import build as build_peru

__all__ = ["build_usa", "build_china", "build_commodities", "build_peru", "BLOCKS"]

# name -> (builder, feeds)
BLOCKS = {
    "usa": (build_usa, ("peru",)),
    "china": (build_china, ("peru",)),
    "commodities": (build_commodities, ("peru",)),
    "peru": (build_peru, ()),
}
