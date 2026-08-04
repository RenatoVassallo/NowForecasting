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


def _code_version(repo_root: Path = REPO) -> str:
    """Short commit id, plus a digest of the uncommitted diff when dirty."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=repo_root, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=repo_root,
                              capture_output=True, text=True, timeout=30,
                              check=True).stdout
        if diff:
            head += "+dirty." + hashlib.sha256(diff.encode()).hexdigest()[:8]
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
