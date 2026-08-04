"""Frozen calibration assets: the only research outputs production may read.

Production never reaches into ``notebooks/``. When a research backtest becomes
a production input (combination weights, error pools, tilt paths), the lab
FREEZES it: the file is copied into ``calibration/`` and its sha256 recorded in
``calibration/MANIFEST.json`` (tracked in git, unlike the binaries, which stay
local under the repository's data policy). Production reads the file only
through :func:`asset_path`, which verifies the hash, so a casual notebook rerun
can never silently change published numbers. Refreshing an asset is a
deliberate, reviewable act: replace the file, update the manifest, commit both
in one change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "calibration"

# logical name -> frozen file (all documented in calibration/MANIFEST.json)
ASSETS = {
    "peru_ladder_full": "peru_ladder_full.parquet",
    "peru_s1_day1": "peru_s1_day1.parquet",
    "peru_s1_day30": "peru_s1_day30.parquet",
    "china_ladder_full": "china_ladder_full.parquet",
    "china_horizon_2012floor": "china_horizon_2012floor.parquet",
    "china_tilt_weo": "china_tilt_weo.parquet",
    "exact_chain": "exact_chain.parquet",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest(root: Path) -> dict:
    mf = Path(root) / "MANIFEST.json"
    if not mf.exists():
        raise RuntimeError(
            f"calibration manifest missing: {mf}. The frozen calibration "
            "assets are a declared production input; see calibration/README.md.")
    return json.loads(mf.read_text()).get("assets", {})


def asset_path(name: str, root: Path | None = None, verify: bool = True) -> Path:
    """The verified path of a frozen calibration asset; raises when unusable."""
    if name not in ASSETS:
        raise ValueError(f"unknown calibration asset {name!r}; "
                         f"known: {sorted(ASSETS)}")
    root = Path(root) if root is not None else ROOT
    fname = ASSETS[name]
    meta = _manifest(root).get(fname)
    if meta is None:
        raise RuntimeError(
            f"calibration asset {fname} is not in {root / 'MANIFEST.json'}; "
            "freeze it deliberately before production can read it.")
    path = root / fname
    if not path.exists():
        raise RuntimeError(
            f"frozen calibration asset missing: {path}. Restore the frozen "
            "file; production never falls back to the lab tree.")
    if verify and _sha(path) != meta.get("sha256"):
        raise RuntimeError(
            f"calibration asset hash mismatch: {path} changed since it was "
            "frozen. If the change is deliberate, re-freeze it (update "
            "calibration/MANIFEST.json in the same commit); otherwise restore "
            "the frozen file.")
    return path


def freeze_asset(name: str, src: Path, *, producer: str,
                 extra: dict | None = None, root: Path | None = None) -> dict:
    """Deliberately (re)freeze an asset: copy the file, rewrite its manifest entry.

    This is the ONLY sanctioned way to change a frozen asset. The existing
    ``role`` is preserved; ``producer`` and any ``extra`` provenance fields
    (panel rule, generation date, code version, seeds, windows) are recorded so
    the manifest states exactly how the frozen vintage was made.
    """
    if name not in ASSETS:
        raise ValueError(f"unknown calibration asset {name!r}")
    root = Path(root) if root is not None else ROOT
    fname = ASSETS[name]
    data = Path(src).read_bytes()
    if not data:
        raise RuntimeError(f"refusing to freeze empty file {src}")
    (root / fname).write_bytes(data)
    mf = root / "MANIFEST.json"
    man = json.loads(mf.read_text())
    entry = man.setdefault("assets", {}).get(fname, {})
    entry.update({"sha256": hashlib.sha256(data).hexdigest(),
                  "bytes": len(data),
                  "frozen_from": str(Path(src)),
                  "producer": producer, **(extra or {})})
    man["assets"][fname] = entry
    mf.write_text(json.dumps(man, indent=2, default=str) + "\n")
    return entry


def manifest_hashes(root: Path | None = None) -> dict:
    """Current sha256 of every frozen asset, for the run manifest."""
    root = Path(root) if root is not None else ROOT
    if not (root / "MANIFEST.json").exists():
        return {"calibration/": "absent"}
    out: dict = {}
    for fname in sorted(_manifest(root)):
        p = root / fname
        out[f"calibration/{fname}"] = _sha(p) if p.exists() else "missing"
    return out
