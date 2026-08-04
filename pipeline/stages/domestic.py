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
        if hasattr(store, "require"):
            store.require("peru_nowcast_official.csv", "peru_nowcast_sweep.csv")
        lines.append(f"- official Peru nowcast artifact written ({path.name})")
    return lines


def run_peru_fan(store, params) -> list[str]:
    """The BoE fan for Peru: the terminal block of the chain.

    Consumes the satellite blocks THIS run published (or one validated prior
    bundle if a block failed) plus this run's official nowcast artifact, and
    writes the fan inside the run directory. Publication to products/ happens
    only after promotion.
    """
    from pathlib import Path

    from pipeline.blocks import build_peru

    blocks = getattr(store, "blocks", {}) or {}
    official = Path(store.root) / "peru_nowcast_official.csv"
    df, lines, path = build_peru(blocks=blocks, ctx=getattr(store, "ctx", None),
                                 out_dir=Path(store.root), official_path=official)
    if hasattr(store, "_track"):
        store._track(Path(path), "fan")
    if hasattr(store, "require"):
        store.require("peru_gdp_fan.csv")
    return lines
