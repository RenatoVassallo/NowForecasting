# Phase 4: fan calibration on exact-rule errors

Regime: pseudo_real_time_final_vintage. Evaluation window fixed across every
variant: the exact chain's realized cells, COVID excluded at both ends
(n = 105 origin-horizon cells; cells within one origin share shocks, so the
pooled binomial intervals and the PIT test are optimistic about independence
and are read as descriptive).

## Task 17: the mathematics, pinned

Eight invariant tests (`tests/test_tpn_math.py`): density normalization and
continuity at the mode by quadrature; CDF/PPF round trip; mean and variance
formulas verified numerically; the (s, gamma) parameterisation round trip;
shortest intervals carry exactly their nominal mass with equal edge density;
equal-tailed and shortest intervals provably differ for skewed distributions
(the shortest is strictly shorter at the same coverage); MLE recovery on
simulated draws; and the published fan frame's table equals the band formula.
The equal-tailed API is now named ``tpn_equal_tailed_bands`` (old name kept
as an alias) with a docstring that forbids mixing conventions in one table.

## Task 18: sequential exact-rule calibration, and tasks 19-20 variants

For every chain origin, interval parameters were fitted ONLY on errors whose
outcome had been published before that origin (legacy S1 day-1/day-30
backtests as the prior pool, exact-chain errors as they accrue), then scored
out of sample. Pre-specified scores: WIS primary; log score and 90 percent
coverage secondary.

|  | n | wis | log_score | cov30 | cov30_ci | width30 | cov60 | cov60_ci | width60 | cov90 | cov90_ci | width90 | pit_ks_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| published production scales [lookahead] | 105 | 0.714 | -1.623 | 0.343 | [0.27, 0.42] | 1.07 | 0.648 | [0.57, 0.72] | 2.34 | 0.924 | [0.87, 0.96] | 4.57 | 0.001 |
| published structural scales [lookahead] | 105 | 0.719 | -1.641 | 0.352 | [0.28, 0.43] | 1.14 | 0.638 | [0.56, 0.71] | 2.49 | 0.933 | [0.88, 0.96] | 4.88 | 0.001 |
| sequential symmetric | 105 | 0.737 | -1.715 | 0.39 | [0.32, 0.47] | 1.1 | 0.705 | [0.63, 0.77] | 2.41 | 0.886 | [0.82, 0.93] | 4.71 | 0.001 |
| sequential (primary) | 105 | 0.762 | -1.814 | 0.381 | [0.31, 0.46] | 1.09 | 0.686 | [0.61, 0.75] | 2.39 | 0.848 | [0.78, 0.90] | 4.66 | 0.0 |
| sequential COVID downweighted | 105 | 0.848 | -1.988 | 0.552 | [0.47, 0.63] | 1.77 | 0.771 | [0.70, 0.83] | 3.86 | 0.962 | [0.92, 0.98] | 7.54 | 0.0 |
| sequential bias-corrected | 105 | 1.027 | -2.218 | 0.314 | [0.25, 0.39] | 1.09 | 0.61 | [0.53, 0.68] | 2.39 | 0.752 | [0.68, 0.81] | 4.66 | 0.0 |
| sequential ref-only exclusion | 105 | 1.027 | -2.1 | 0.59 | [0.51, 0.67] | 2.69 | 0.857 | [0.79, 0.90] | 5.88 | 0.99 | [0.96, 1.00] | 11.49 | 0.0 |
| sequential COVID included | 105 | 1.579 | -2.745 | 0.838 | [0.77, 0.89] | 4.93 | 1.0 | [0.97, 1.00] | 10.77 | 1.0 | [0.97, 1.00] | 21.05 | 0.0 |

### The adopted rule, by horizon (sequential symmetric)

|  | n | cov30 | cov60 | cov90 | wis |
|---|---|---|---|---|---|
| 1 | 18 | 0.72 [0.53,0.86] | 0.94 [0.79,0.99] | 1.00 [0.87,1.00] | 0.375 |
| 2 | 17 | 0.41 [0.24,0.61] | 0.76 [0.57,0.89] | 1.00 [0.86,1.00] | 0.49 |
| 3 | 15 | 0.27 [0.13,0.48] | 0.80 [0.59,0.92] | 0.93 [0.75,0.98] | 0.628 |
| 4 | 13 | 0.38 [0.20,0.61] | 0.69 [0.46,0.85] | 0.85 [0.63,0.95] | 0.822 |
| 5 | 12 | 0.42 [0.22,0.64] | 0.67 [0.43,0.84] | 0.75 [0.51,0.90] | 0.9 |
| 6 | 11 | 0.45 [0.24,0.68] | 0.64 [0.39,0.82] | 0.82 [0.57,0.94] | 0.93 |
| 7 | 10 | 0.20 [0.07,0.46] | 0.50 [0.27,0.73] | 0.70 [0.44,0.87] | 0.996 |
| 8 | 9 | 0.00 [-0.00,0.23] | 0.33 [0.14,0.60] | 0.89 [0.62,0.97] | 1.249 |

## Decisions (each against the pre-specified scores)

1. **ADOPTED: sequential symmetric.** It costs about 3 percent of WIS against
   the LOOKAHEAD references (0.737 vs 0.714, intervals overlapping) while
   removing the lookahead entirely; 90 percent coverage 0.89 [0.82, 0.93].
   Production (`pipeline/blocks/peru.py`) now calibrates through
   ``fan_calibration.production_fits``: knowable-before errors only, RMS
   scale per horizon, monotone in h, zero skew, zero mode shift; the day-1 /
   day-30 information-state interpolation is unchanged. Published 90 percent
   widths narrow from 4.4 / 4.6 / 5.3 / 5.5 to 3.5 / 3.8 / 4.5 / 5.0 at
   h = 2..8 because the chain's conditioned errors are genuinely smaller
   than the legacy backtest's.
2. **REJECTED: estimated skew.** Worse out of sample (0.762) than symmetric;
   small knowable pools estimate noise, consistent with the near-symmetric
   ex-COVID error record.
3. **REJECTED: horizon bias correction.** The Phase 3 in-sample bias (+0.5 to
   +2.0pp by h=8) is regime-driven; correcting it in real time collapsed 90
   percent coverage to 0.75 and pushed WIS to 1.027. Zero mode shift stands.
4. **COVID: both-ends exclusion confirmed.** Ref-only exclusion overcovers
   (0.99 at 90 percent) with 11.5pp widths; inclusion is unusable (21pp);
   downweighting (w=0.25) overcovers at every level. The production rule was
   already both-ends; it now has the sensitivity table the audit asked for.

## Not run, with the hook

Per-origin structural simulation with joint satellite-error covariance and
generated-regressor uncertainty needs the Monte-Carlo draws stored inside the
exact chain (a ``store_sims`` extension of the harness); today's published
STRUCTURAL scales were evaluated as a fixed lookahead reference instead
(0.719 WIS, statistically indistinguishable from the empirical references).
Until the stored-sims comparison exists, the empirical rule remains the
baseline by the audit's own default.

Artifacts: `output/backtests/fan_calibration_cells.parquet`,
`fan_calibration_pools.csv`; the adopted rule lives in
`pipeline/lib/fan_calibration.py:production_fits` and is consumed by the
Peru block.

## Follow-up audit addendum (2026-08-04)

- The exact-chain errors behind `production_fits` are now a FROZEN,
  hash-verified calibration asset (`calibration/exact_chain.parquet`); the
  silent `if CHAIN.exists()` branch is gone, so published bands can no longer
  depend on the presence of a mutable local file.
- The chain was regenerated after the China calibration assets were rebuilt
  under the corrected January-February timing rule: the 105 ex-COVID cells
  moved by at most 0.08pp (mean 0.009pp) and the recalibrated variant table
  reproduces this page's ordering (adopted rule WIS 0.736, cov90 0.895
  [0.84, 0.93]); the published day-30 sigmas changed by at most 0.0013.
- STATUS: the symmetric sequential rule is PROVISIONAL. Per-horizon coverage
  rests on 9 to 18 cells with shared origin shocks; those estimates cannot
  yet discriminate between nearby calibration rules. The 2023Q1+ window has
  been inspected and is no longer an untouched holdout; the rule faces the
  prospective window (outcomes from 2026Q2 on, frozen 2026-08-04) before any
  refinement is considered.
