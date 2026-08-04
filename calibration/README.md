# Frozen calibration assets

Research backtests that production consumes as inputs: combination weights,
error pools for fan calibration, and the WEO tilt path. The binaries stay
local (the repository's data policy keeps parquet out of git), but
`MANIFEST.json` is tracked, so every clone knows exactly which frozen
artifacts production expects and their sha256.

Rules, enforced by `pipeline/lib/calibration_assets.py` and
`tests/test_no_notebook_deps.py`:

- Production reads these files ONLY through `asset_path(name)`, which fails
  closed on a missing file or a hash mismatch. Production never reads
  `notebooks/`.
- A notebook rerun that rewrites its own outputs changes nothing here.
  Refreshing an asset is deliberate: copy the new file in, update
  `MANIFEST.json` (hash, bytes, provenance), and commit both together.
- The run manifest records the hash of every asset used, so any published
  number can be traced to the exact frozen vintage behind it.

| asset | role |
| --- | --- |
| `peru_ladder_full.parquet` | Peru release-cycle nowcast ladder; Adaptive-IC and exact-chain replay |
| `peru_s1_day1.parquet` | legacy S1 backtest, day-1 information state; fan-calibration prior pool |
| `peru_s1_day30.parquet` | legacy S1 backtest, day-30 information state; fan-calibration prior pool |
| `china_ladder_full.parquet` | China nowcast ladder; Adaptive-IC weights |
| `china_horizon_2012floor.parquet` | China horse race; Combo weights (past bases only) + TPN calibration |
| `china_tilt_weo.parquet` | WEO-tilted China path backtest; published blend weight ALPHA |
