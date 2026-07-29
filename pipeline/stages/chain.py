"""Satellite chain: US, China and commodities, in dependency order.

Pure Python - the notebooks under ``notebooks/`` are the development surface and
are never executed in production. Each block rebuilds its published fan from
source data and writes ``products/blocks/<name>_path_uncertainty.csv`` carrying
a central path plus the two-piece-normal parameters fitted to that block's own
real-time errors, stamped with the information state of the run.

    usa ----------\\
    china ---------> peru (domestic stage)
    commodities --/

A failing block never aborts the run: it is reported, and the stage says
explicitly that the domestic stage will fall back to the previous vintage -
safer than a half-updated forecast that looks current.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pandas as pd

from pipeline.blocks import BLOCKS
from pipeline.blocks._common import CONTRACT, REPO

ORDER = ("usa", "china", "commodities")


def _validate(path: Path) -> str:
    df = pd.read_csv(path)
    missing = [c for c in CONTRACT if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} breaks the contract, missing {missing}")
    if df[["centre", "s", "gamma"]].isna().any().any():
        raise ValueError(f"{path.name} has NaN in the contract columns")
    return f"{len(df)} quarters"


def run(store, params, panels=None) -> list[str]:
    only = set(getattr(params, "CHAIN_BLOCKS", ()) or ORDER)
    lines, published = [], {}
    for name in ORDER:
        if name not in only:
            lines.append(f"- **{name}**: skipped")
            continue
        builder, feeds = BLOCKS[name]
        t0 = time.time()
        try:
            _df, block_lines, path = builder()
            _validate(path)
            published[name] = path
            lines += block_lines
            dest = Path(store.root) / "blocks"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest / path.name)
            print(f"  [chain] {name:12s} {time.time()-t0:6.0f}s  ok")
        except Exception as exc:
            lines.append(f"- **{name}**: FAILED ({type(exc).__name__}: {exc})")
            lines.append(f"  - downstream {', '.join(feeds)} will use the previous vintage")
            print(f"  [chain] {name:12s} {time.time()-t0:6.0f}s  FAILED {exc}")
    store.blocks = published
    return lines
