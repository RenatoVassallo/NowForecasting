"""RunStore: one directory per run (vintage), everything saved and indexed.

Layout::

    runs/<run_id>/
      manifest.json                 what ran, params, versions, timings, files
      report.md                     the final report
      data/<target>/                the data vintage used (+ what's-new.md)
      satellites/<target>/          nowcasts, weights, bands, metrics, models, factors, figures, report
      domestic/<target>/            (same)
    runs/latest -> <run_id>

Everything is addressed through this object so the layout lives in one place.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class RunStore:
    def __init__(self, runs_dir: Path, run_id: str | None = None):
        self.runs_dir = Path(runs_dir)
        self.run_id = run_id or datetime.now().strftime("%Y-%m-%d__%H%M%S")
        self.root = self.runs_dir / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest: dict[str, Any] = {"run_id": self.run_id, "files": [], "stages": {}}

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

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self._manifest, indent=2, default=str))

    def update_latest_symlink(self):
        link = self.runs_dir / "latest"
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(self.run_id)
        except OSError:
            pass  # symlinks may be unavailable; not fatal
