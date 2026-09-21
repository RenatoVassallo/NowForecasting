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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PRODUCTS_DIR = REPO / "products"


class PublicationError(RuntimeError):
    """Publication failed; the promoted run and prior pointers are intact."""


@dataclass(frozen=True)
class PublicationResult:
    """Return contract of :func:`publish_run`.

    ``path`` is the versioned publication directory; ``files`` maps each
    published relative path to its sha256 (the publication manifest itself is
    not part of its own mapping).
    """
    path: Path
    files: dict = field(default_factory=dict)

    @property
    def n_files(self) -> int:
        return len(self.files)

# the declared product surface, relative to a run root; ``required`` entries
# in the run manifest must exist, everything else is published when present
SURFACE = (
    "blocks/us_path_uncertainty.csv",
    "blocks/china_path_uncertainty.csv",
    "blocks/tot_path_uncertainty.csv",
    "blocks/bundle.json",
    "peru_gdp_fan.csv",
    "peru_gdp_model_paths.csv",
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
            raise PublicationError(
                f"publish: {src} does not match the promoted manifest "
                f"(sha256/bytes changed after promotion); refusing to publish "
                "a tampered surface")
    return got


def publish_run(run_root: Path,
                products_dir: Path | None = None) -> PublicationResult:
    """Publish one promoted run transactionally.

    Returns a :class:`PublicationResult` (versioned directory + published
    file hashes); raises :class:`PublicationError` on any refusal, leaving
    the previous publication and the ``latest`` pointer untouched.
    """
    run_root = Path(run_root)
    products_dir = PRODUCTS_DIR if products_dir is None else Path(products_dir)
    if not (run_root / "_SUCCESS").exists():
        raise PublicationError(
            f"publish: {run_root} carries no _SUCCESS marker; only a promoted "
            "run may publish")
    man = json.loads((run_root / "manifest.json").read_text())
    if str(man.get("run_id", "")) != run_root.name:
        raise PublicationError(
            f"publish: manifest run_id {man.get('run_id')!r} does not match "
            f"the run directory {run_root.name!r}")
    if man.get("status") != "success":
        raise PublicationError(
            f"publish: manifest status is {man.get('status')!r}, not "
            "'success'; only a successful run publishes")
    recorded = {e["path"]: e for e in man.get("files", [])}

    published_root = products_dir / "published"
    final = published_root / run_root.name
    if final.exists():
        raise PublicationError(
            f"publish: {final} already exists; a run id publishes once "
            "(remove the old publication deliberately to republish)")
    published_root.mkdir(parents=True, exist_ok=True)
    tmp = published_root / f".tmp-{run_root.name}"
    if tmp.exists():
        shutil.rmtree(tmp)

    try:
        # the expected surface derives from the PROMOTED MANIFEST, never from
        # surviving files: manifest-recorded SURFACE entries plus every
        # manifest entry of kind "figure" under figures/. A file on disk that
        # the manifest does not record is never published; a recorded file
        # that is missing or altered fails the publication.
        fig_expected = sorted(
            str(e["path"]) for e in man.get("files", [])
            if e.get("kind") == "figure"
            and str(e.get("path", "")).startswith(f"{FIGURE_DIR}/"))
        targets = [s for s in SURFACE if s in recorded] + fig_expected
        required = [r for r in man.get("required", []) if r in SURFACE]
        missing = [r for r in required if r not in targets]
        if missing:
            raise PublicationError(
                "publish: required artifacts absent from the manifest surface: "
                + ", ".join(missing))

        hashes: dict[str, str] = {}
        for rel in targets:
            src = run_root / rel
            if not src.exists() or src.stat().st_size == 0:
                raise PublicationError(
                    f"publish: {rel} is recorded in the promoted manifest but "
                    "missing or empty on disk; the run surface was altered "
                    "after promotion")
            hashes[rel] = _copy_verified(src, tmp / rel, recorded.get(rel))
        if not hashes:
            raise PublicationError(
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

    # transactional pointer switch, strictly after the rename above: a unique
    # temporary symlink is created FIRST, then one atomic replace. On ANY
    # failure the previous pointer is untouched, the temporary link is
    # removed, and the unreferenced release is quarantined under
    # published/.unreferenced-<run_id>-<stamp> so the run id can simply be
    # republished once the cause is fixed (see products/PUBLISHING.md). The
    # old fallback that unlinked the valid pointer is gone.
    link = products_dir / "latest"
    stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    tmp_link = products_dir / f".latest-tmp-{os.getpid()}-{stamp}"
    try:
        os.symlink(Path("published") / final.name, tmp_link)
        os.replace(tmp_link, link)
    except OSError as exc:
        try:
            if tmp_link.is_symlink() or tmp_link.exists():
                tmp_link.unlink()
        except OSError:
            pass
        quarantine = published_root / f".unreferenced-{final.name}-{stamp}"
        os.replace(final, quarantine)
        raise PublicationError(
            f"publish: could not switch the latest pointer ({exc}). The "
            "previous publication remains authoritative; the new release was "
            f"quarantined at published/{quarantine.name}. Fix the cause and "
            "republish the run (the run id is free again).") from exc
    return PublicationResult(path=final, files=dict(sorted(hashes.items())))
