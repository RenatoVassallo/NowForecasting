# The production pipeline

The single entry point:

```bash
python -m pipeline.main --as-of 2026-08-04 [--run-id 2026-08-04__120000]
```

Switches live in `pipeline/config/params.py` (stages, targets, persistence,
jobs); model specifications live in `pipeline/config/metadata.py`; the series
catalog is `pipeline/config/data_registry.json` (validated against
`data_registry.schema.json`, hashed into every run and bundle).

## One information set per run

`RunContext.create(as_of=...)` is built once in `pipeline.main` and threaded
through every stage and block. All release selection resolves against this one
date; the only wall-clock fallback lives in `pipeline/lib/context.py` for
ad-hoc interactive calls, and `tests/test_run_context.py` scans the blocks and
stages to keep it that way. The code version recorded per run is the short git
commit plus a sha256 digest of the uncommitted diff when dirty.

## Stages

| stage | what it does |
| --- | --- |
| data | refresh every source, snapshot the vintage into the run, report what's new; refresh outcomes are appended to an event log with content hashes |
| preflight | availability gate: required inputs must be fit for the as-of date or the run stops (documented waivers only, recorded in the manifest) |
| nowcast | China satellite and Peru domestic release-cycle ladders + Adaptive-IC; ONE official Peru nowcast artifact + its persisted sweep |
| forecast | block chain in dependency order (usa -> china -> commodities), then the Peru conditional BVAR; every published fan is a two-piece normal with the shortest-interval band convention |
| fanchart | the five published figures, one house style |
| report | report.pdf + report.md, with information-state stamps |

Satellite hand-off is coherent by construction: a complete current run writes
`blocks/bundle.json` (one as-of, run id, code version, registry hash, sha256
per artifact); anything less falls back to ONE validated prior bundle, whole,
or publication stops (`pipeline/lib/bundle.py`).

## Run lifecycle (atomic)

1. artifacts are written under `output/runs/.staging/<run_id>/`
2. `manifest.json` records: as-of, code version (commit + dirty digest),
   Python and package versions, registry sha256, model seeds (read from the
   modules that use them), frozen-calibration hashes, per-file sha256 and
   size for every tracked artifact, stage timings and statuses
3. manifest writing validates that every tracked artifact exists and is
   non-empty (the success path fails closed)
4. `_SUCCESS` is written, the run is promoted by one atomic rename into
   `output/runs/<run_id>/`, and only then does `runs/latest` move
5. a failed run keeps its staging directory with a `_FAILED` marker; it is
   never promoted, never `latest`, and never a bundle-fallback candidate

## Publication (transactional, versioned)

Publication is a transaction owned by `pipeline/lib/publish.py`: the promoted
run's declared surface is staged into a temporary directory, every file is
verified (size and sha256) against the run manifest, a publication manifest
(run id, as-of, code version, per-file hashes) is written, the directory is
atomically renamed to `products/published/<run_id>/`, and only then does the
single authoritative `products/latest` pointer switch. Any failure leaves the
previous publication and the pointer untouched, and each release is built
fresh, so stale figures cannot linger. The OLD flat generated paths directly
under `products/` are DEPRECATED (frozen as of their last flat publication);
downstream consumers read `products/latest/`. The assembly package's source
files under `products/` are never touched.

## Frozen calibration assets

Research backtests that production consumes (combination weights, fan error
pools, the WEO tilt) are frozen into `calibration/` with sha256 pinned in the
tracked `MANIFEST.json`. Production reads them only through
`pipeline.lib.calibration_assets.asset_path`, which fails closed on absence or
hash mismatch; a notebook rerun cannot silently change published numbers.
`tests/test_no_notebook_deps.py` additionally scans production packages so no
string literal can reference the notebook tree at all.

## Evaluation honesty

Every evaluation artifact carries `pseudo_real_time_final_vintage`: origins
replay the exact live rule at day-30-of-cycle information states, but the
panels are final-vintage reconstructions, so results are not claims about
genuine real-time performance. Fan widths are calibrated sequentially from the
exact chain's own errors using only outcomes published before each origin
(`pipeline/lib/fan_calibration.py`), with COVID excluded at both ends of every
forecast; the selection/holdout split is frozen at 2022Q4/2023Q1.

## CI and the public/private boundary

`.github/workflows/ci.yml` builds the locked Python 3.11 environment with
`uv sync --frozen` and runs, as named steps: registry validation, transform
contracts, quarter-grid contracts, two-piece-normal invariants, run-lifecycle
and notebook-independence contracts, the synthetic pipeline smoke run, then
the full public suite. Tests that need the private data layer skip themselves
with an explicit reason, so CI is the executable definition of what a public
clone can reproduce.
