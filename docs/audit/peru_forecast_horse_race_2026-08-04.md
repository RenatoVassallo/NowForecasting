# Peru forecast modeling experiment, 2026-08-04

## Production correction

The legacy day-30 forecast backtest and the exact-chain backtest used different
raw horizon origins.  The legacy raw horizon 1 forecasts fan node 2, while the
exact-chain raw horizon 1 is the official nowcast at fan node 1.  Production now
normalizes both sources onto the same contract:

- fan node 1: official information-conditional nowcast only;
- fan nodes 2 to 8: legacy raw horizons 1 to 7 plus exact-chain raw horizons 2
  to 8;
- exact-chain raw horizon 1 is excluded from medium-term calibration.

The corrected 2026-08-04 run leaves every centre unchanged and widens the
symmetric scale at nodes 2 to 8 by 0.046 to 0.105 percentage points.  The 90
percent widths now move from 2.030 at node 1 to 3.717 at node 2 and 5.274 at
node 8.  Run `2026-08-04__fan-horizon-fix-prod` completed all stages and was
promoted.

## Exact-origin development race

Notebook 02 replays the day-30 production information set at 27 origins from
2019Q1 through 2025Q3.  It holds the official nowcast and US, China, and terms
of trade paths fixed across candidates.  The scored sample excludes fan node 1
and applies the both-ends COVID rule.  The focused development candidates are:

- S1 with flat expectations, the incumbent;
- S2 with recursive AR(1) expectations;
- a strictly sequential inverse-MSE ensemble of S1 and S2.

RW and AR(2) are valid benchmarks.  The unconditional BVAR does not fire under
the exact information set and is reported but excluded from matched scoring.

The reduced-draw screening race selects S2 with recursive AR(1) expectations.
Its average RMSE is 21.0 percent below S1 across fan nodes 2 to 8.  It improves
S1 at six of seven nodes, cuts mean absolute bias from 0.900 to 0.348 percentage
points, and has WIS 0.637 versus 0.915 for S1.  Its empirical 90 percent
coverage is 0.841, with a Wilson interval of 0.752 to 0.903.  S2 is the only one
of the three candidates whose interval contains the nominal rate.

The sequential S1/S2 ensemble improves S1 at every node and lowers mean RMSE by
13.6 percent, but it is inferior to S2 alone.  Its WIS is 0.745 and its 90
percent coverage is 0.762, whose Wilson upper bound is only 0.838.  An
equal-weight sensitivity is also inferior to S2.  The ensemble therefore does
not diversify a useful independent error source; it partly reintroduces S1's
negative long-horizon bias.

## Decision status

Notebook `02b_full_draw_stability.ipynb` completed all 27 origins with the live
draw settings: four terms-of-trade chains with 6,000 draws each and 3,000 draws
for every conditional BVAR.  Each origin was cached separately with the
exact-chain no-lookahead evidence and a manifest that fingerprints code,
dependencies, registry, frozen inputs, the panel, model rules, seeds and draws.

The full-draw centre confirms the screening result.  S2's mean horizon RMSE is
22.1 percent below S1, it wins six of seven nodes, and its worst node is only
0.6 percent worse.  Mean absolute bias falls from 0.899 to 0.327 percentage
points.  Sequential WIS is 0.625 versus 0.928 for S1, and S2's empirical 90
percent coverage is 0.841 with a Wilson interval of 0.752 to 0.903.  The
inverse-MSE ensemble lowers mean RMSE by 14.2 percent but remains inferior to
S2 and its coverage interval still excludes 0.90.

The result is not stable enough for production promotion under the rule fixed
before inspecting the full-draw output:

- the inspected-post-selection segment strongly favors S2, with mean horizon
  relative RMSE 0.783, but the selection segment has only six matched cells and
  gives 1.269;
- every leave-one-origin-out estimate favors S2, ranging from 0.753 to 0.845;
- omitting the 2023 launch year raises S2's mean horizon relative RMSE to 1.075;
- an eight-origin circular block bootstrap has median 0.782 but its 90th
  percentile is 1.073;
- the rolling-20-quarter AR(1), persistence-cap sensitivity, and flat-path
  decomposition all remain better than S1, so the instability is not an
  isolated AR(1) parameter artifact.

The recorded decision is therefore `RETAIN_S1_PRODUCTION_SHADOW_S2`.  S2
remains the primary development challenger and should be stored prospectively
at each publication date.  There is no untouched post-selection outcome in the
current record, and the first prospective reference quarter is 2026Q2.  No fan
width should be recalibrated from the same sample used to select S2.

Remaining work:

1. reconsider the centre after prospective outcomes accumulate;
2. if S2 is eventually promoted, change the central path first and retain the
   conservative production fan calibration until independent S2 errors exist;
3. store structural simulations and run one-factor-off US, China, and terms of
   trade uncertainty experiments before placing caps or floors on satellite
   bands.

Prospective storage is now implemented. Every successful forecast run must
produce and publish `peru_gdp_model_paths.csv`, containing both S1 and S2 for
all eight fan nodes. Node 1 is the common official nowcast. Both candidates
carry the width curve known at that publication date, with S2 explicitly
labelled as a shared-width center-shift counterfactual. The artifact stores the
run id, as-of, expected outcome release date, AR(1) diagnostics, and a blank
outcome field; the strict manifest fails the run if this record is absent.
