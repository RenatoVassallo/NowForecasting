"""Run provenance: the seeds and frozen calibration inputs behind a run.

Everything here feeds the run manifest. Seeds are read from the modules that
actually use them (no duplicated literals); calibration inputs are the frozen
artifacts production consumes, hashed so a manifest pins the exact vintage.
"""

from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# seed/spec constants, read from where they live so they cannot drift
_SEED_PROBES = {
    "core.preprocess.SEED": ("core.preprocess", "SEED"),
    "china_bvar.seed": ("pipeline.blocks._china_model", "BVAR_SEED"),
    "commodities.chains": ("pipeline.blocks.commodities", "CHAINS"),
    "commodities.draws": ("pipeline.blocks.commodities", "DRAWS"),
    "peru.fan_mc": ("pipeline.blocks.peru", "FAN_MC"),
}


def declared_seeds() -> dict:
    out: dict = {}
    for key, (mod, attr) in _SEED_PROBES.items():
        try:
            out[key] = getattr(import_module(mod), attr)
        except Exception as exc:            # record the gap, never crash a run
            out[key] = f"unavailable ({type(exc).__name__})"
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_inputs() -> dict:
    """sha256 of every frozen calibration artifact production can read.

    Production consumes calibration data ONLY through the frozen assets
    (including the exact-chain errors); nothing under output/ is probed.
    """
    try:
        from pipeline.lib.calibration_assets import manifest_hashes
        return dict(manifest_hashes())
    except Exception as exc:
        return {"calibration_assets": f"unavailable ({type(exc).__name__})"}
