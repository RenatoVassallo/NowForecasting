"""Transactional publication: one versioned, hash-verified surface per run.

The staged run directory is the only information surface during a run; this
module is the ONE place allowed to publish. A publication is a TRANSACTION:

1. the promoted run (``_SUCCESS`` present) declares its surface;
2. every artifact is copied into a temporary directory and its size and
   sha256 are verified against the promoted run's manifest;
3. a publication manifest (run id, as-of, code version, per-file sha256) is
   written into the staging directory;
4. the directory is atomically renamed to ``products/published/<run_id>/``;
5. only then does the single authoritative ``products/latest`` pointer
   switch (atomic symlink replace).

Any failure before the final rename leaves the previous publication and the
pointer untouched, and each release is built fresh, so obsolete files can
never linger on the authoritative surface. The OLD flat generated paths
directly under products/ are DEPRECATED: nothing writes them any more, and
downstream consumers must read ``products/latest/``. Source code that lives
under products/ (the assembly package) is never touched by publication.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PRODUCTS_DIR = REPO / "products"

# the declared product surface, relative to a run root; ``required`` entries
# in the run manifest must exist, everything else is published when present
SURFACE = (
    "blocks/us_path_uncertainty.csv",
    "blocks/china_path_uncertainty.csv",
    "blocks/tot_path_uncertainty.csv",
    "blocks/bundle.json",
    "peru_gdp_fan.csv",
    "peru_nowcast_official.csv",
    "peru_nowcast_sweep.csv",
    "report.pdf",
    "report.md",
)
FIGURE_DIR = "figures"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_verified(src: Path, dest: Path, expected: dict | None) -> str:
    """Copy one artifact and verify bytes + sha256 against the run manifest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    got = _sha(dest)
    if expected is not None:
        if expected.get("sha256") not in (None, got) \
                or (expected.get("bytes") is not None
                    and expected["bytes"] != dest.stat().st_size):
            raise RuntimeError(
                f"publish: {src} does not match the promoted manifest "
                f"(sha256/bytes changed after promotion); refusing to publish "
                "a tampered surface")
    return got


def publish_run(run_root: Path, products_dir: Path | None = None) -> Path:
    """Publish one promoted run transactionally; returns the versioned dir."""
    run_root = Path(run_root)
    products_dir = PRODUCTS_DIR if products_dir is None else Path(products_dir)
    if not (run_root / "_SUCCESS").exists():
        raise RuntimeError(
            f"publish: {run_root} carries no _SUCCESS marker; only a promoted "
            "run may publish")
    man = json.loads((run_root / "manifest.json").read_text())
    recorded = {e["path"]: e for e in man.get("files", [])}

    published_root = products_dir / "published"
    final = published_root / run_root.name
    if final.exists():
        raise RuntimeError(
            f"publish: {final} already exists; a run id publishes once "
            "(remove the old publication deliberately to republish)")
    published_root.mkdir(parents=True, exist_ok=True)
    tmp = published_root / f".tmp-{run_root.name}"
    if tmp.exists():
        shutil.rmtree(tmp)

    try:
        targets = list(SURFACE)
        figs = run_root / FIGURE_DIR
        if figs.is_dir():
            targets += [f"{FIGURE_DIR}/{f.name}" for f in sorted(figs.iterdir())
                        if f.is_file() and f.stat().st_size > 0]
        required = [r for r in man.get("required", []) if r in SURFACE]
        missing = [r for r in required
                   if not (run_root / r).exists()
                   or (run_root / r).stat().st_size == 0]
        if missing:
            raise RuntimeError(
                "publish: declared artifacts missing from the promoted run: "
                + ", ".join(missing))

        hashes: dict[str, str] = {}
        for rel in targets:
            src = run_root / rel
            if not src.exists() or src.stat().st_size == 0:
                continue
            hashes[rel] = _copy_verified(src, tmp / rel, recorded.get(rel))
        if not hashes:
            raise RuntimeError(
                "publish: the run produced none of the declared surface; "
                "nothing to publish")
        tmp.mkdir(parents=True, exist_ok=True)

        (tmp / "publication_manifest.json").write_text(json.dumps({
            "run_id": man.get("run_id", run_root.name),
            "as_of": man.get("as_of"),
            "code_version": man.get("code_version"),
            "published_at": datetime.now().isoformat(timespec="seconds"),
            "files": dict(sorted(hashes.items())),
        }, indent=2))

        os.replace(tmp, final)                    # the transaction commits here
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)    # nothing published, pointer intact
        raise

    # atomic pointer switch, strictly after the rename above
    link = products_dir / "latest"
    tmp_link = products_dir / ".latest-tmp"
    try:
        if tmp_link.is_symlink() or tmp_link.exists():
            tmp_link.unlink()
        os.symlink(Path("published") / final.name, tmp_link)
        os.replace(tmp_link, link)
    except OSError:
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(Path("published") / final.name, link)
    return final
