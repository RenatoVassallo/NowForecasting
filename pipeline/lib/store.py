"""RunStore: one directory per run (vintage), staged, validated, then promoted.

Lifecycle (``staged=True``, the production path)::

    runs/.staging/<run_id>/     all artifacts written and hashed here
      manifest.json             environment, hashes, seeds, timings, statuses
      _SUCCESS                  written only when every stage finished
    runs/<run_id>/              the staging dir, promoted by one atomic rename
    runs/latest -> <run_id>     updated only after promotion

A run that fails keeps its staging directory (with a ``_FAILED`` marker) for
forensics and is invisible to everything that scans ``runs/``: it is never
promoted, never pointed to by ``latest``, and never eligible as a prior-bundle
fallback. ``staged=False`` preserves the old direct layout for ad-hoc use.

Layout inside a run::

    manifest.json                 what ran: params, versions, hashes, timings
    report.md                     the final report
    data/<target>/                the data vintage used (+ what's-new.md)
    satellites/<target>/          nowcasts, weights, bands, metrics, models, figures
    domestic/<target>/            (same)

Everything is addressed through this object so the layout lives in one place.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _environment() -> dict:
    """Interpreter and installed-package versions, for the manifest."""
    import importlib.metadata as md

    pkgs: dict[str, str] = {}
    for dist in md.distributions():
        name = (dist.metadata.get("Name") or "").strip().lower()
        if name:
            pkgs[name] = dist.version
    return {"python": platform.python_version(),
            "platform": platform.platform(),
            "packages": dict(sorted(pkgs.items()))}


class RunStore:
    def __init__(self, runs_dir: Path, run_id: str | None = None, ctx=None,
                 staged: bool = False):
        self.runs_dir = Path(runs_dir)
        self.ctx = ctx
        if ctx is not None:
            self.run_id = run_id or ctx.run_id
        else:
            self.run_id = run_id or datetime.now().strftime("%Y-%m-%d__%H%M%S")
        self.staged = staged
        self.final_root = self.runs_dir / self.run_id
        self.root = (self.runs_dir / ".staging" / self.run_id) if staged \
            else self.final_root
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest: dict[str, Any] = {"run_id": self.run_id, "files": [],
                                          "stages": {}}
        if ctx is not None:
            self._manifest.update(as_of=str(ctx.as_of.date()),
                                  code_version=ctx.code_version)

    # -- paths -----------------------------------------------------------------
    def dir(self, *parts) -> Path:
        d = self.root.joinpath(*[str(p) for p in parts])
        d.mkdir(parents=True, exist_ok=True)
        return d

    def target_dir(self, role: str, name: str) -> Path:
        return self.dir(role, name)

    def _rel(self, path: Path) -> str:
        return str(Path(path).relative_to(self.root))

    def _track(self, path: Path, kind: str):
        self._manifest["files"].append({"path": self._rel(path), "kind": kind})

    # -- savers ----------------------------------------------------------------
    def save_df(self, path: Path, df: pd.DataFrame, kind: str = "table") -> Path:
        path = Path(path)
        df.to_parquet(path) if path.suffix == ".parquet" else df.to_csv(path, index=True)
        self._track(path, kind)
        return path

    def save_fig(self, path: Path, fig, kind: str = "figure") -> Path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
        self._track(path, kind)
        return path

    def save_pickle(self, path: Path, obj, kind: str = "model") -> Path:
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)
        self._track(path, kind)
        return path

    def save_text(self, path: Path, text: str, kind: str = "report") -> Path:
        Path(path).write_text(text)
        self._track(path, kind)
        return path

    # -- manifest --------------------------------------------------------------
    def log_stage(self, stage: str, info: dict):
        self._manifest["stages"][stage] = info

    def set_meta(self, **kw):
        self._manifest.update(kw)

    def write_manifest(self, strict: bool = True):
        """Hash and validate every tracked artifact, then write manifest.json.

        ``strict=True`` (the success path) fails closed when a tracked artifact
        is missing or empty; the failure path uses ``strict=False`` so a broken
        run can still record what it managed to produce.
        """
        from pipeline.lib.bundle import registry_sha
        from pipeline.lib.provenance import calibration_inputs, declared_seeds

        man = self._manifest
        man["environment"] = _environment()
        man["registry_sha"] = registry_sha()
        man.setdefault("seeds", declared_seeds())
        man.setdefault("calibration_inputs", calibration_inputs())

        problems: list[str] = []
        for entry in man["files"]:
            p = self.root / entry["path"]
            if p.exists() and p.stat().st_size > 0:
                entry["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
                entry["bytes"] = p.stat().st_size
            else:
                entry["sha256"] = "missing"
                problems.append(entry["path"])
        if strict and problems:
            raise RuntimeError(
                "manifest validation: tracked artifacts missing or empty: "
                + ", ".join(problems))
        (self.root / "manifest.json").write_text(
            json.dumps(man, indent=2, default=str))

    # -- lifecycle ---------------------------------------------------------------
    def mark_success(self):
        """A run without this marker must never be promoted or fallen back to."""
        (self.root / "_SUCCESS").write_text(datetime.now().isoformat(timespec="seconds"))

    def abort(self, reason: str = ""):
        """Quarantine a failed staged run: marker written, never promoted."""
        try:
            (self.root / "_FAILED").write_text(
                datetime.now().isoformat(timespec="seconds") + "\n" + reason)
        except OSError:
            pass

    def promote(self) -> Path:
        """Atomically move the staged run into its final place.

        Requires the ``_SUCCESS`` marker; refuses to overwrite an existing
        promoted run (a replayed run id must be resolved by the operator, not
        by silent replacement).
        """
        if not self.staged or self.root == self.final_root:
            return self.root                      # unstaged: already in place
        if not (self.root / "_SUCCESS").exists():
            raise RuntimeError("promote: refusing without the _SUCCESS marker")
        if self.final_root.exists():
            raise RuntimeError(
                f"promote: {self.final_root} already exists; refusing to "
                "overwrite a promoted run")
        self.final_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.root, self.final_root)    # atomic on one filesystem
        self.root = self.final_root
        return self.root

    def update_latest_symlink(self):
        if self.staged and self.root != self.final_root:
            raise RuntimeError("latest: promote the run before updating latest")
        link = self.runs_dir / "latest"
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(self.run_id)
        except OSError:
            pass  # symlinks may be unavailable; not fatal
