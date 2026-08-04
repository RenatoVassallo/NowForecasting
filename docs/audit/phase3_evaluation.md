# Phase 3 evaluation: the exact chain and its benchmarks

Regime: **pseudo_real_time_final_vintage** (final-snapshot values under scalar
release rules; SPF surveys and IMF WEO rounds are genuine vintages). Selection
sample ends 2022Q4. The 2023Q1 to 2026Q1 window (sample label
`inspected_post_selection`) was frozen forward on 2026-08-03;
it has NOW BEEN INSPECTED (here and in the follow-up audit) and therefore
cannot serve as an untouched holdout for any later modelling choice. The
prospective evaluation window starts at outcomes realized AFTER 2026-08-04:
2026Q2 and later prints score a record fixed before their release
(`core.evaluation.PROSPECTIVE_START`).

Regenerated 2026-08-04 from the frozen artifacts (`calibration/
exact_chain.parquet`, sha 02273cbad954; `output/backtests/
nowcast_benchmarks.csv`) after the China calibration assets were rebuilt
under the corrected January-February timing rule. The chain claim is
`exact_rule_reduced_mc`: the replay uses reduced ToT Monte-Carlo settings
(2 chains x 3000 draws vs the published 4 x 6000); the measured effect at a
probe origin is 0.003pp mean, 0.007pp max across horizons, far below the
0.1pp reporting resolution, but the run is NOT computationally identical to
production.

## The exact chain

27 origins, 2019Q1 to 2025Q3, one per base quarter at day 30 of the release
cycle (today's publication timing). The production China rule (2012-floor
conditional BVARs, min_train 28) is undefined before roughly 2019Q1, so the
chain starts there: a replay must not substitute a different specification.
Every origin recomputes the official nowcast condition, the vintaged SPF/WEO
United States path, the full China profile (recursive bridge, WEO round of the
day, tilt with the documented untilted fallback), the masked monthly ToT BVAR
with reconstructed daily flash months, and the S1 conditional BVAR, then runs
the benchmarks on the same information set. Per-origin no-lookahead evidence
is stored in `exact_chain_checks.json`; the harness refuses to store an origin
that fails a check, refuses to resume under a changed configuration
(`exact_chain_fingerprint.json`), and keeps the checks file in lockstep with
the parquet.

Scoring: both-ends COVID exclusion (base AND target outside 2020-2021),
strict common sample across all core models. After the exclusions the usable
cells are mostly in the inspected post-selection window (12-13 per horizon)
with only 2-6 selection cells at short horizons, so this table is close to a
pure post-selection reading; the
CELLS are release-origin rows, and each horizon has at most one cell per
quarter here, so cells equal independent quarters in this table.

### RMSE by horizon (percentage points)

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| AR(2) | 1.58 | 2.03 | 2.37 | 2.50 | 2.36 | 2.10 | 1.62 | 0.93 |
| Mean(4) | 0.87 | 1.22 | 1.65 | 2.01 | 2.06 | 1.97 | 1.73 | 1.48 |
| Median(4) | 0.96 | 1.14 | 1.71 | 2.27 | 2.44 | 2.43 | 2.16 | 1.76 |
| Nowcast-node | 0.56 |  |  |  |  |  |  |  |
| RW | 1.22 | 1.67 | 2.34 | 2.95 | 3.18 | 3.23 | 3.12 | 2.91 |
| S1-chain | 0.56 | 0.84 | 1.12 | 1.49 | 1.65 | 1.69 | 1.76 | 2.12 |

### Relative to the random walk

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| AR(2) | 1.29 | 1.21 | 1.01 | 0.85 | 0.74 | 0.65 | 0.52 | 0.32 |
| Mean(4) | 0.71 | 0.73 | 0.71 | 0.68 | 0.65 | 0.61 | 0.55 | 0.51 |
| Median(4) | 0.78 | 0.68 | 0.73 | 0.77 | 0.77 | 0.75 | 0.69 | 0.61 |
| Nowcast-node | 0.46 |  |  |  |  |  |  |  |
| RW | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| S1-chain | 0.46 | 0.50 | 0.48 | 0.50 | 0.52 | 0.52 | 0.56 | 0.73 |

### Cells per horizon (common ex-COVID sample; one cell = one quarter)

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| all core models | 18 | 17 | 15 | 13 | 12 | 11 | 10 | 9 |

### Equal-accuracy tests (p-values; n of 9 to 18: EXPLORATORY only)

| h | n | DM S1-chain | DM AR(2) | DM Mean(4) | DM Median(4) | CW AR2 nests RW |
|---|---|---|---|---|---|---|
| 1 | 18 | 0.003 | 0.184 | 0.051 | 0.064 | 0.245 |
| 2 | 17 | 0.119 | 0.650 | 0.310 | 0.278 | 0.232 |
| 3 | 15 | 0.034 | 0.978 | 0.219 | 0.291 | 0.197 |
| 4 | 13 | 0.007 | 0.672 | 0.046 | 0.231 | 0.102 |
| 5 | 12 | degenerate | degenerate | degenerate | 0.000 | 0.000 |
| 6 | 11 | degenerate | degenerate | degenerate | degenerate | degenerate |
| 7 | 10 | degenerate | degenerate | degenerate | degenerate | degenerate |
| 8 | 9 |  |  |  |  |  |

DM uses the Harvey correction at each h; Clark-West covers the nested
AR(2)-vs-RW pair. "degenerate" marks cells where the HAC long-run variance of
the loss differential collapses toward zero in these tiny samples and the
statistic explodes; such cells carry NO evidential weight. With 9 to 18
quarters per horizon, every entry in this table is exploratory: the consistent
SIGN of the RMSE gains across horizons is the stronger evidence, and no
selection decision may rest on these p-values.

### Findings

- The full satellite-conditioned rule roughly HALVES the random-walk error at
  every horizon (rel-RW 0.46 to 0.73), and node 1 equals the official nowcast
  by construction (0.56pp RMSE at h=1).
- The equal Mean(4) is the best simple combination (0.51 to 0.71 rel-RW);
  the unconditional BVAR confirms that most of the value is CONDITIONING,
  consistent with the research notebooks.
- S1 bias GROWS with horizon (-0.5pp at h=4 to -2.0pp at h=8 in the
  forecast-minus-realized convention: launches from the 2022-2023 slowdown
  under-predicted the 2024-2025 recovery). Phase 4 tested a live bias
  correction and REJECTED it (coverage collapsed); the bias is regime-driven
  and not exploitable in real time.
- Rebuilding the China calibration assets under the corrected January timing
  moved these chain cells by at most 0.08pp (mean 0.009pp): material for the
  China block's own weights and bands, second order HERE, by measurement.

## Nowcast-family benchmarks (release-cycle ladder, common samples)

Rows are release-cycle origins; the effective number of INDEPENDENT
observations is the distinct quarters column, not n.

Full common sample (n = 269 rows over 65 quarters):

| model | n rows | quarters | rmse |
|---|---|---|---|
| Adaptive-IC | 269 | 65 | 0.809 |
| Mean(members) | 269 | 65 | 0.815 |
| Stage-switch | 269 | 65 | 0.836 |
| Bridge(leaders) | 269 | 65 | 0.836 |
| Previous-best | 269 | 65 | 0.869 |
| Median(members) | 269 | 65 | 0.872 |
| P-MIDAS(leaders) | 269 | 65 | 1.016 |
| RW | 269 | 65 | 1.323 |

Inspected post-selection window, 2023Q1 through 2026Q1 (n = 62 rows over 13
quarters; the sample label is `inspected_post_selection`, never an untouched
holdout):

| model | n rows | quarters | rmse |
|---|---|---|---|
| Mean(members) | 62 | 13 | 0.457 |
| Median(members) | 62 | 13 | 0.619 |
| Adaptive-IC | 62 | 13 | 0.631 |
| Bridge(leaders) | 62 | 13 | 0.720 |
| Previous-best | 62 | 13 | 0.720 |
| Stage-switch | 62 | 13 | 0.720 |
| P-MIDAS(leaders) | 62 | 13 | 0.763 |
| RW | 62 | 13 | 1.069 |

Bridge versus P-MIDAS on their common finite rows (n = 273 rows):

| model | n rows | rmse |
|---|---|---|
| Bridge(leaders) | 273 | 0.850 |
| P-MIDAS(leaders) | 273 | 1.018 |

### Findings, stated with the discipline the samples deserve

- **Adaptive-IC has NOT established superiority over the equal mean.** On the
  full common sample the gap is under one percent (0.809 vs 0.815); on the
  13-quarter inspected window the equal mean wins decisively (0.457 vs
  0.631). The
  equal mean is therefore the MANDATORY benchmark: no future nowcast change
  ships without beating it on the prospective window.
- **Adaptive-IC is NOT being replaced on this evidence either.** Thirteen
  quarters of inspected rows (sharing within-quarter shocks) cannot support a
  production swap by themselves; replacing the headline ensemble on this one
  analysis would be exactly the selection-on-inspected-data practice this report
  now forbids. The decision is deferred to the prospective window, with the
  equal mean published alongside internally.
- Bridge beats P-MIDAS head-to-head on their common sample (0.850 vs 1.018,
  n=273 rows); own-sample rows in the CSV carry finite shares and are
  labelled not comparable.

## Status of the published fan calibration

The symmetric sequential rule adopted in Phase 4 remains in production and is
**provisional**: horizon-level coverage estimates rest on 9 to 18 cells per
horizon with shared origin shocks, so their Wilson intervals are wide (the
pooled 90 percent coverage is 0.895 [0.84, 0.93] on the corrected chain) and
the per-horizon numbers cannot yet discriminate between nearby calibration
rules. Recalibration on the corrected chain left the published widths
unchanged to 0.001. The rule stands until the prospective window accumulates
enough independent quarters to test it.

## Boundaries found and codified while building the chain

- The tilt solver can DECLINE (entropic weights infeasible); the documented
  fallback is the untilted model path, flagged `tilt_declined` in provenance.
- A WEO round's horizon can end before the grid does; the annual path now
  extends flat at its last projected year (the IMF medium-term convention),
  in the live block and the chain alike.
- COVID-base origins (2019Q4 to 2021Q2) RUN, and their rows exist in the
  artifact, but the both-ends rule excludes them from every table above.

Artifacts: `calibration/exact_chain.parquet` (frozen, hash-verified; the ONLY
chain-error input to production fan calibration),
`output/backtests/exact_chain_checks.json`,
`exact_chain_fingerprint.json`, `fan_calibration_cells.parquet`,
`nowcast_benchmarks.csv`. Final-vintage results (everything above) are never
mixed with genuine-vintage results; the only genuine vintages in the chain
are the SPF survey rows and the IMF WEO rounds, used as inputs.
