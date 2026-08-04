"""Input pinning: verified immutability of everything a run consumes.

Downstream blocks load target panels from local caches at stage time, so a
cache rewritten between the data stage and a later stage would silently change
the run. Rather than re-plumbing every loader (an architecture rewrite), the
run PINS the sha256 of every production input file at start and RE-VERIFIES
before ``_SUCCESS``: any drift fails the run, so no publication can mix input
vintages. The pins live in the manifest, which also makes every input
reconstructible byte-for-byte after the fact.

``sources_code_sha`` complements the git commit: ``sources/`` is git-ignored
(provider licensing), so the commit-based code version cannot capture loader
changes; the content hash over the private loader code is recorded in the run
manifest and in refresh events.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# every glob production reads at stage time (targets/*.load_panel and the
# direct parquet reads in the blocks); extend when a loader gains an input
PRODUCTION_INPUTS = (
    "peru/*.parquet", "peru/*.csv",
    "bcrp/*.parquet",
    "us/*.parquet",
    "china/*.csv", "china/*.parquet",
    "imf/*.parquet",
    "commodities/*.parquet",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin_inputs(root: Path | None = None) -> dict:
    """sha256 + size of every production input file, keyed by relative path."""
    root = REPO / "input" if root is None else Path(root)
    if not root.is_dir():
        raise RuntimeError(
            f"input pinning: {root} does not exist; an estimation run needs "
            "the private data layer")
    pins: dict = {}
    for pattern in PRODUCTION_INPUTS:
        for p in sorted(root.glob(pattern)):
            if p.is_file():
                pins[str(p.relative_to(root))] = {"sha256": _sha(p),
                                                  "bytes": p.stat().st_size}
    if not pins:
        raise RuntimeError(f"input pinning: no production inputs under {root}")
    return pins


def verify_inputs(pins: dict, root: Path | None = None) -> None:
    """Fail closed when any pinned input changed, appeared or vanished."""
    root = REPO / "input" if root is None else Path(root)
    drift: list[str] = []
    for rel, meta in pins.items():
        p = root / rel
        if not p.exists():
            drift.append(f"{rel}: deleted during the run")
        elif _sha(p) != meta["sha256"]:
            drift.append(f"{rel}: content changed during the run")
    if drift:
        raise RuntimeError(
            "input caches mutated between run start and completion; the run "
            "cannot publish on a mixed information set: " + "; ".join(drift))


def sources_code_sha(root: Path | None = None) -> str:
    """Content hash of the private source-loader code (git cannot see it)."""
    root = REPO / "sources" if root is None else Path(root)
    if not root.is_dir():
        return "absent"
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()
