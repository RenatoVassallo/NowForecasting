"""Coherent satellite bundles: the only sanctioned fallback after a block fails.

The old behavior let the Peru block glob ``products/blocks/*.csv`` when a
satellite failed, silently mixing today's successful blocks with files of
unknown age, grid, and code version. The rule now:

- A run whose REQUIRED blocks all succeed writes ``blocks/bundle.json`` next to
  the copied block CSVs: one as-of date, run id, code version, registry hash,
  and a sha256 per artifact. The bundle file is written only on full success,
  so its existence IS the chain-stage success marker.
- When the current run cannot publish all required blocks, the domestic stage
  may use ONE complete prior bundle, whole, after it passes every compatibility
  check below. Mixing quarters or files across bundles is impossible by
  construction: the loader returns all paths from a single validated bundle or
  raises.
- The global ``products/blocks`` directory is a publication convenience and is
  never read as a fallback.

Compatibility checks: bundle schema version; one shared as_of / run_id /
code_version; registry hash equal to the current registry; every artifact
present with matching sha256; required contract columns present and non-NaN;
per-block quarter grid recorded (first, last, count) and internally coherent.
Code-version equality is deliberately strict: a bundle produced by different
code is not evidence about today's model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

BUNDLE_SCHEMA = 1
REQUIRED_BLOCKS = ("usa", "china", "commodities")
CONTRACT = ("quarter", "h", "source", "centre", "mode", "s", "gamma",
            "sigma_left", "sigma_right")
REGISTRY_PATH = Path(__file__).resolve().parents[1] / "config" / "data_registry.json"


class BundleError(RuntimeError):
    """No coherent satellite bundle is available; publication must stop."""


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registry_sha(path: Path = REGISTRY_PATH) -> str:
    return file_sha(path) if Path(path).exists() else "absent"


def _grid(df: pd.DataFrame) -> dict:
    q = list(df["quarter"].astype(str))
    return {"first": q[0], "last": q[-1], "n": len(q)}


def write_bundle(blocks_dir: Path, published: dict[str, Path], ctx) -> Path | None:
    """Record a coherent bundle; returns None (and writes nothing) if incomplete."""
    blocks_dir = Path(blocks_dir)
    if set(published) < set(REQUIRED_BLOCKS):
        return None
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "as_of": str(ctx.as_of.date()) if ctx is not None else None,
        "run_id": ctx.run_id if ctx is not None else None,
        "code_version": ctx.code_version if ctx is not None else None,
        "registry_sha": registry_sha(),
        "blocks": {},
    }
    for name in REQUIRED_BLOCKS:
        src = Path(published[name])
        dest = blocks_dir / src.name
        if not dest.exists():
            raise BundleError(f"bundle write: {dest} was not copied into the run")
        df = pd.read_csv(dest)
        manifest["blocks"][name] = {"file": src.name, "sha256": file_sha(dest),
                                    "grid": _grid(df)}
    out = blocks_dir / "bundle.json"
    out.write_text(json.dumps(manifest, indent=2))
    return out


def _check(cond: bool, run_name: str, why: str, problems: list[str]):
    if not cond:
        problems.append(f"{run_name}: {why}")


def load_prior_bundle(runs_dir: Path, *, exclude_run_id: str | None = None,
                      current_code_version: str | None = None) -> dict:
    """Return the newest COMPLETE, COHERENT prior bundle, or raise BundleError.

    The result maps block name to CSV path, all from one bundle, plus
    ``__meta__`` with the bundle manifest for stamping.
    """
    runs_dir = Path(runs_dir)
    problems: list[str] = []
    # only PROMOTED runs are candidates: never the staging area, never a run
    # that lacks the _SUCCESS marker (a failed run is not evidence)
    candidates = sorted((p for p in runs_dir.glob("*") if p.is_dir()
                         and not p.is_symlink() and p.name != exclude_run_id
                         and not p.name.startswith(".")
                         and (p / "_SUCCESS").exists()),
                        reverse=True)
    reg_now = registry_sha()
    for run in candidates:
        bfile = run / "blocks" / "bundle.json"
        if not bfile.exists():
            continue
        name = run.name
        try:
            man = json.loads(bfile.read_text())
        except Exception as exc:
            _check(False, name, f"unreadable bundle.json ({exc})", problems)
            continue
        ok: list[str] = []
        _check(man.get("schema") == BUNDLE_SCHEMA, name, "bundle schema mismatch", ok)
        _check(bool(man.get("as_of")) and bool(man.get("run_id")), name,
               "bundle lacks as_of or run_id", ok)
        _check(man.get("registry_sha") == reg_now, name,
               "registry hash changed since the bundle was written", ok)
        if current_code_version is not None:
            _check(man.get("code_version") == current_code_version, name,
                   f"code version {man.get('code_version')} differs from "
                   f"{current_code_version}", ok)
        blocks = man.get("blocks", {})
        _check(set(blocks) >= set(REQUIRED_BLOCKS), name, "bundle is incomplete", ok)
        paths: dict[str, Path] = {}
        if not ok:
            for bname in REQUIRED_BLOCKS:
                meta = blocks.get(bname, {})
                p = run / "blocks" / str(meta.get("file"))
                if not p.exists():
                    _check(False, name, f"{bname}: artifact missing", ok)
                    continue
                if file_sha(p) != meta.get("sha256"):
                    _check(False, name, f"{bname}: artifact hash mismatch", ok)
                    continue
                try:
                    df = pd.read_csv(p)
                except Exception as exc:
                    _check(False, name, f"{bname}: unreadable ({exc})", ok)
                    continue
                missing = [c for c in CONTRACT if c not in df.columns]
                _check(not missing, name, f"{bname}: missing columns {missing}", ok)
                if not missing:
                    _check(not df[["centre", "s", "gamma"]].isna().any().any(),
                           name, f"{bname}: NaN in required columns", ok)
                    _check(_grid(df) == meta.get("grid"), name,
                           f"{bname}: grid changed since the bundle was written", ok)
                paths[bname] = p
        if not ok and len(paths) == len(REQUIRED_BLOCKS):
            paths["__meta__"] = man
            return paths
        problems.extend(ok)
    raise BundleError(
        "no coherent prior satellite bundle: "
        + ("; ".join(problems) if problems else "no prior run has a bundle.json")
        + ". Publication must not proceed on mixed or unverified vintages.")


def resolve_block_paths(blocks: dict, runs_dir: Path, ctx=None) -> tuple[dict, dict | None]:
    """The ONLY sanctioned way to obtain satellite paths for the domestic stage.

    A complete current run passes through untouched. Anything less falls back
    to ONE validated prior bundle, whole (today's partial successes are set
    aside rather than mixed), or raises ``BundleError`` so publication stops.
    """
    if set(blocks or {}) >= set(REQUIRED_BLOCKS):
        return {n: Path(blocks[n]) for n in REQUIRED_BLOCKS}, None
    prior = load_prior_bundle(
        runs_dir,
        exclude_run_id=getattr(ctx, "run_id", None),
        current_code_version=getattr(ctx, "code_version", None))
    meta = prior.pop("__meta__")
    return prior, meta
