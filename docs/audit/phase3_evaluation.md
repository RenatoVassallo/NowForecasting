# Phase 3 evaluation: the exact chain and its benchmarks

Regime: **pseudo_real_time_final_vintage** (final-snapshot values under scalar
release rules; SPF surveys and IMF WEO rounds are genuine vintages). Selection
sample ends 2022Q4; the holdout (2023Q1 onward) was frozen on 2026-08-03 and
is honest FORWARD (all pre-existing modelling choices saw the full sample).

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
that fails a check.

Scoring: both-ends COVID exclusion (base AND target outside 2020-2021),
strict common sample across all core models. After the exclusions the usable
cells are mostly holdout (12-13 per horizon) with only 2-6 selection cells at
short horizons, so this table is close to a pure frozen-holdout reading.

### RMSE by horizon (percentage points)

|  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| AR(2) | 1.58 | 2.03 | 2.37 | 2.50 | 2.36 | 2.10 | 1.62 | 0.93 |
| Mean(4) | 0.87 | 1.22 | 1.65 | 2.01 | 2.06 | 1.97 | 1.73 | 1.48 |
| Median(4) | 0.96 | 1.14 | 1.71 | 2.27 | 2.43 | 2.43 | 2.16 | 1.76 |
| Nowcast-node | 0.56 |  |  |  |  |  |  |  |
| RW | 1.22 | 1.67 | 2.34 | 2.95 | 3.18 | 3.23 | 3.12 | 2.91 |
| S1-chain | 0.56 | 0.84 | 1.13 | 1.49 | 1.64 | 1.69 | 1.76 | 2.14 |

### Relative to the random walk

|  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| AR(2) | 1.29 | 1.21 | 1.01 | 0.85 | 0.74 | 0.65 | 0.52 | 0.32 |
| Mean(4) | 0.71 | 0.73 | 0.71 | 0.68 | 0.65 | 0.61 | 0.55 | 0.51 |
| Median(4) | 0.78 | 0.68 | 0.73 | 0.77 | 0.77 | 0.75 | 0.69 | 0.61 |
| Nowcast-node | 0.46 |  |  |  |  |  |  |  |
| RW | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| S1-chain | 0.46 | 0.50 | 0.48 | 0.50 | 0.52 | 0.52 | 0.56 | 0.74 |

### Cells per horizon

|  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| AR(2) | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} |
| Mean(4) | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} |
| Median(4) | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} |
| Nowcast-node | {:.0f} |  |  |  |  |  |  |  |
| RW | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} |
| S1-chain | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} |

### Equal-accuracy tests (p-values; n is SMALL, 9 to 18 cells)

|  | n | DM S1-chain | DM AR(2) | DM Mean(4) | DM Median(4) | CW AR2 nests RW |
|---|---|---|---|---|---|---|
| 1 | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |
| 2 | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |
| 3 | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |
| 4 | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |
| 5 | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |
| 6 | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |
| 7 | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |
| 8 | {:.3f} |  |  |  |  |  |

DM is used for non-nested comparisons against the RW with the Harvey
correction at each h; Clark-West covers the nested AR(2)-vs-RW pair. Read the
h >= 5 zeros with the sample sizes in view.

### Findings

- The full satellite-conditioned rule roughly HALVES the random-walk error at
  every horizon (rel-RW 0.46 to 0.74), and node 1 equals the official nowcast
  by construction (0.56pp RMSE at h=1).
- The equal Mean(4) is the best simple combination (0.51 to 0.73 rel-RW);
  the unconditional BVAR confirms that most of the value is CONDITIONING,
  consistent with the research notebooks.
- S1 bias grows with horizon (+0.5pp at h=4 to +2.0pp at h=8 on this
  window): forecasts launched from the 2022-2023 slowdown under-predicted the
  2024-2025 recovery. Phase 4 must test horizon-dependent bias correction
  against a zero mode shift (audit task 20).
- Directional accuracy of the chain: 0.67 to 1.00 by horizon (RW predicts no
  change by definition and is excluded from that comparison).

## Nowcast-family benchmarks (release-cycle ladder, common samples)

|  | n | rmse |
|---|---|---|
| Adaptive-IC | {:.3f} | {:.3f} |
| Mean(members) | {:.3f} | {:.3f} |
| Bridge(leaders) | {:.3f} | {:.3f} |
| Stage-switch | {:.3f} | {:.3f} |
| Previous-best | {:.3f} | {:.3f} |
| Median(members) | {:.3f} | {:.3f} |
| P-MIDAS(leaders) | {:.3f} | {:.3f} |
| RW | {:.3f} | {:.3f} |

Frozen holdout (2023Q1 onward):

|  | n | rmse |
|---|---|---|
| Mean(members) | {:.3f} | {:.3f} |
| Median(members) | {:.3f} | {:.3f} |
| Adaptive-IC | {:.3f} | {:.3f} |
| Bridge(leaders) | {:.3f} | {:.3f} |
| Previous-best | {:.3f} | {:.3f} |
| Stage-switch | {:.3f} | {:.3f} |
| P-MIDAS(leaders) | {:.3f} | {:.3f} |
| RW | {:.3f} | {:.3f} |

Bridge versus P-MIDAS on their common finite rows:

|  | n | rmse |
|---|---|---|
| Bridge(leaders) | {:.3f} | {:.3f} |
| P-MIDAS(leaders) | {:.3f} | {:.3f} |

### Findings

- On the full common sample Adaptive-IC (0.809) beats the equal mean (0.815)
  by under one percent. On the FROZEN HOLDOUT the equal mean (0.457) beats
  Adaptive-IC (0.631) decisively. The audit's concern stands: the adaptive
  complexity is NOT currently earning its keep, and the equal mean is the
  benchmark every future nowcast change must beat.
- Bridge beats P-MIDAS head-to-head on their common sample (0.850 vs 1.018,
  n=273); own-sample rows in the CSV carry finite shares and are labelled
  not-comparable.

## Boundaries found and codified while building the chain

- The tilt solver can DECLINE (entropic weights infeasible); the documented
  fallback is the untilted model path, flagged `tilt_declined` in provenance.
- A WEO round's horizon can end before the grid does; the annual path now
  extends flat at its last projected year (the IMF medium-term convention),
  in the live block and the chain alike.
- COVID-base origins (2019Q4 to 2021Q2) now RUN, and their rows exist in the
  artifact, but the both-ends rule excludes them from every table above.

Artifacts: `output/backtests/exact_chain.parquet`, `exact_chain_checks.json`,
`exact_chain_scoreboard.csv`, `exact_chain_dm.csv`, `nowcast_benchmarks.csv`.
Density metrics (coverage, WIS, PIT, log score) are implemented and tested in
`core/evaluation.py`; their application to the exact-rule intervals is Phase 4
by design (calibration must not precede these corrected errors).
