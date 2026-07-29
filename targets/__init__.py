"""Registry of nowcast targets.

Each target is a reusable, country-agnostic data interface (see ``base.Target``).
The run pipeline (``pipeline``) and the notebooks both consume these; add a new
target (USA, commodities, ...) by writing a module and listing it here.
"""

from __future__ import annotations

from . import china, commodities, copper, peru_gdp, usa
from .base import Target

# `copper` stays bound to targets.copper (the production pipeline's leaner
# panel); the commodity research block adds the rest of the family.
REGISTRY: dict[str, Target] = {t.name: t for t in (china.SPEC, peru_gdp.SPEC,
                                                   usa.SPEC, copper.SPEC)}
REGISTRY.update({n: s for n, s in commodities.SPECS.items() if n not in REGISTRY})

SATELLITES = [name for name, t in REGISTRY.items() if t.role == "satellite"]
DOMESTIC = [name for name, t in REGISTRY.items() if t.role == "domestic"]


def get(name: str) -> Target:
    if name not in REGISTRY:
        raise KeyError(f"unknown target {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


__all__ = ["Target", "REGISTRY", "SATELLITES", "DOMESTIC", "get"]
