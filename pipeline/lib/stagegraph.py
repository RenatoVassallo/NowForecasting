"""Stage dependency graph: validate the configuration BEFORE anything runs.

The stages are not independent switches. The forecast stage consumes the
official nowcast artifact; the fanchart consumes the forecast blocks, the fan
and the nowcast artifacts; the report consumes the fanchart context and
figures. The old committed default enabled ``report`` alone, which failed deep
inside the stage after outputs already existed. The rule now: an enabled stage
requires every stage it depends on, validated up front, and the ONLY way to
render a report without recomputing is the explicit report-from-run mode
naming a promoted source run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# stage -> stages it consumes artifacts from (within one run)
DEPS: dict[str, set] = {
    "data": set(),
    "nowcast": set(),
    "forecast": {"nowcast"},
    "fanchart": {"nowcast", "forecast"},
    "report": {"nowcast", "forecast", "fanchart"},
}


class StageConfigError(RuntimeError):
    """The selected stages cannot produce a coherent run."""


def validate_stages(stages: dict) -> None:
    enabled = {s for s, on in stages.items() if on}
    unknown = enabled - set(DEPS)
    if unknown:
        raise StageConfigError(f"unknown stages: {sorted(unknown)}")
    for s in sorted(enabled):
        missing = DEPS[s] - enabled
        if missing:
            raise StageConfigError(
                f"stage '{s}' requires {sorted(missing)} in the same run "
                "(artifacts are run-local; nothing is found implicitly). "
                "Enable the missing stages, or use the explicit "
                "report-from-run mode (--report-from RUN_ID).")


@dataclass(frozen=True)
class ReportSource:
    """A promoted run whose artifacts a re-rendered report consumes."""
    root: Path
    run_id: str
    as_of: pd.Timestamp
    manifest: dict

    @property
    def runs_dir(self) -> Path:
        return self.root.parent

    @property
    def blocks(self) -> dict:
        out = {}
        for name, fname in (("usa", "us_path_uncertainty.csv"),
                            ("china", "china_path_uncertainty.csv"),
                            ("commodities", "tot_path_uncertainty.csv")):
            p = self.root / "blocks" / fname
            if p.exists():
                out[name] = p
        return out


def resolve_report_source(runs_dir: Path, run_id: str) -> ReportSource:
    """The NAMED promoted run a report re-render consumes, or refuse."""
    run = Path(runs_dir) / str(run_id)
    if not run.is_dir():
        raise StageConfigError(
            f"report-from-run: no promoted run named {run_id!r} under "
            f"{runs_dir} (nothing is ever found implicitly)")
    if not (run / "_SUCCESS").exists():
        raise StageConfigError(
            f"report-from-run: {run_id} carries no _SUCCESS marker; an "
            "unpromoted or failed run cannot source a report")
    mpath = run / "manifest.json"
    if not mpath.exists():
        raise StageConfigError(f"report-from-run: {run_id} has no manifest")
    man = json.loads(mpath.read_text())
    if man.get("status") not in (None, "success"):
        raise StageConfigError(
            f"report-from-run: {run_id} manifest status is {man.get('status')!r}")
    as_of = man.get("as_of")
    if not as_of:
        raise StageConfigError(f"report-from-run: {run_id} manifest lacks as_of")
    return ReportSource(root=run, run_id=str(run_id),
                        as_of=pd.Timestamp(as_of), manifest=man)
