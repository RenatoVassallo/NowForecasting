"""Stage 2: satellite nowcasts (China; USA / commodities to come)."""

from __future__ import annotations

import targets

from ._models import run_targets


def run(store, params, panels) -> list[str]:
    return run_targets(store, params, panels, targets.SATELLITES, "satellites")
