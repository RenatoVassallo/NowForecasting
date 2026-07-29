"""Instantiate a target's model ladder from the declarative specs in metadata.

A spec is ``(class_name, kwargs)``; the dict key names the model in every
artifact (``run_release_cycle_backtest`` labels by the dict key, so the model's
own ``name`` is cosmetic - we still pass it where the class accepts it).
"""

from __future__ import annotations

from MIDAS import (
    ADLMIDASNowcaster, ARNowcaster, BridgeNowcaster, DFMNowcaster,
    GBTreesNowcaster, PooledMIDASNowcaster, RandomWalkNowcaster,
)
from forecast.models import BVARNowcaster, DirectARXNowcaster

CLASSES = {c.__name__: c for c in [
    RandomWalkNowcaster, ARNowcaster, BridgeNowcaster, PooledMIDASNowcaster,
    ADLMIDASNowcaster, BVARNowcaster, DFMNowcaster, DirectARXNowcaster, GBTreesNowcaster,
]}


def build(specs: dict[str, tuple]) -> dict:
    """``{name: (class_name, kwargs)}`` -> ``{name: instance}``."""

    out = {}
    for name, (cls_name, kwargs) in specs.items():
        cls = CLASSES[cls_name]
        for name_kw in ("_name", "name"):          # pass the label where supported
            try:
                out[name] = cls(**kwargs, **{name_kw: name})
                break
            except TypeError:
                continue
        else:
            out[name] = cls(**kwargs)
    return out
