"""One immutable identity per production run.

Every stage and block must select releases against the SAME as-of date. Before
this module each component read the wall clock on its own, so a run crossing
midnight (or replayed later) could mix information sets and could not be
reproduced. The rule now is:

- ``RunContext.create(as_of=...)`` is built once, in ``pipeline.main``.
- Everything downstream receives the context (or the store carrying it) and
  resolves dates through it.
- The ONLY wall-clock read in pipeline code lives here, as the fallback when no
  context is supplied (ad-hoc interactive calls). ``tests/test_run_context.py``
  scans the blocks and stages to keep it that way.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def dirty_inventory(repo_root: Path = REPO) -> dict[str, str]:
    """Relative path -> content sha256 for every deviation from HEAD.

    Covers staged and unstaged changes to tracked files AND untracked
    non-ignored files (``git status --porcelain -uall`` respects .gitignore),
    so a new production module cannot change silently. Deleted files map to
    the literal ``"deleted"``. Recorded in the run manifest so two differing
    code fingerprints can be diagnosed file by file. A hash DETECTS
    difference; it does not reconstruct an uncommitted tree, so a clean
    commit remains the preferred operator state for an external release.
    """
    out = subprocess.run(["git", "status", "--porcelain", "-uall"],
                         cwd=repo_root, capture_output=True, text=True,
                         timeout=30, check=True).stdout
    inv: dict[str, str] = {}
    for line in out.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:                       # rename: hash the new side
            path = path.split(" -> ", 1)[1]
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        p = Path(repo_root) / path
        inv[path] = (hashlib.sha256(p.read_bytes()).hexdigest()
                     if p.is_file() else "deleted")
    return dict(sorted(inv.items()))


def _code_version(repo_root: Path = REPO) -> str:
    """Short commit id, plus a digest over EVERY working-tree deviation.

    The digest is a sha256 over the sorted (path, content-sha) inventory of
    staged, unstaged and untracked non-ignored files, so untracked production
    modules are part of the code identity (the old ``git diff HEAD`` digest
    was blind to them).
    """
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=repo_root, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
        inv = dirty_inventory(repo_root)
        if inv:
            h = hashlib.sha256()
            for path, sha in inv.items():        # already sorted: deterministic
                h.update(path.encode())
                h.update(b"\0")
                h.update(sha.encode())
                h.update(b"\n")
            head += "+dirty." + h.hexdigest()[:8]
        return head
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class RunContext:
    as_of: pd.Timestamp          # release-selection cutoff (normalized date)
    run_id: str
    code_version: str

    @classmethod
    def create(cls, as_of=None, run_id: str | None = None,
               repo_root: Path = REPO) -> "RunContext":
        now = pd.Timestamp.now()
        as_of_ts = (now if as_of is None else pd.Timestamp(as_of)).normalize()
        # the vintage date leads the id; the wall-clock suffix only disambiguates
        # two runs of the same vintage on the same machine
        rid = run_id or f"{as_of_ts:%Y-%m-%d}__{now:%H%M%S}"
        return cls(as_of=as_of_ts, run_id=rid, code_version=_code_version(repo_root))


def resolve_as_of(ctx: RunContext | None) -> pd.Timestamp:
    """The run's as-of date, or today when called without a run (interactive)."""
    return ctx.as_of if ctx is not None else pd.Timestamp.now().normalize()
