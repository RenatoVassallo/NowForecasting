"""Stage 3: domestic GDP nowcast (Peru)."""

from __future__ import annotations

import targets

from ._models import run_targets


def run(store, params, panels) -> list[str]:
    lines = run_targets(store, params, panels, targets.DOMESTIC, "domestic")
    result = getattr(store, "nowcast_results", {}).get("peru_gdp")
    if result is not None:
        from pipeline.lib.context import resolve_as_of
        from pipeline.lib.nowcast_artifact import write_official

        path = write_official(store, result,
                              as_of=resolve_as_of(getattr(store, "ctx", None)))
        lines.append(f"- official Peru nowcast artifact written ({path.name})")
    return lines


def run_peru_fan(store, params) -> list[str]:
    """The BoE fan for Peru: the terminal block of the chain.

    Imports whatever the satellite blocks published in this run (falling back to
    the last vintage if one failed) and writes ``products/peru_gdp_fan.csv``
    plus the figure.
    """
    import shutil
    from pathlib import Path

    from pipeline.blocks import build_peru

    blocks = getattr(store, "blocks", {}) or {}
    df, lines, path = build_peru(blocks=blocks, ctx=getattr(store, "ctx", None))
    dest = Path(store.root)
    shutil.copy2(path, dest / path.name)
    return lines
