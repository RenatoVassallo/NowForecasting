"""Shared analysis pipeline: the target-agnostic data -> information -> plots steps.

Kept separate from ``nowcast`` (the modelling engine) so any application (China,
Peru, USA, ...) reuses the same three steps with only a thin, app-specific loader
and config. Import the submodules directly:

    from analysis import data, information, plots, transforms
"""

from . import data, information, plots, transforms
from .plots import set_style

__all__ = ["data", "information", "plots", "transforms", "set_style"]
