# Phase 1 implementation log

Implementation lead session, started 2026-08-03. Baseline before any change:
`pytest -q` gave 22 passed, 0 skipped (the brief expected 21 passed, 1 skipped;
the difference is the network-guarded INEI test, which skips offline and passed
online here). Worktree was already dirty in `.gitignore`, `AGENTS.md`,
`forecast/boe_fan.py`, `pipeline/blocks/peru.py`, `pipeline/config/params.py`;
all preserved, none reverted.

Conventions: every task follows reproduce, failing test, smallest fix, focused
tests, full suite, log. Audit verdicts are stated as confirmed, narrowed, or
rejected.

## Task 1: canonical run context

**Audit claim**: run date is hidden global state; repeated `pd.Timestamp.now()`
across blocks, stages and report can mix information sets inside one run and
make reruns irreproducible. **Verdict: CONFIRMED.** Grep found wall-clock reads
in `pipeline/blocks/_common.py` (twice), `peru.py`, `usa.py`, `china.py`,
`pipeline/stages/report.py` (twice), `pipeline/stages/forecast.py`,
`pipeline/lib/nowcast_job.py` (twice), plus implicit defaults in
`forecast.live_forecast` and `nowcast.release_cycle.live_path` that production
callers did not override.

**Changes**
- NEW `pipeline/lib/context.py`: frozen `RunContext(as_of, run_id,
  code_version)`; `RunContext.create()` (as_of normalized; run_id defaults to
  `<as_of>__<HHMMSS>` so the vintage leads the id; code_version = short git
  commit plus `+dirty.<sha8>` of the uncommitted diff); `resolve_as_of(ctx)` is
  the single sanctioned wall-clock fallback for context-less interactive use.
- `pipeline/main.py`: builds the context once (`main(as_of=None, run_id=None)`
  plus CLI `--as-of` / `--run-id`), passes it into `RunStore`.
- `pipeline/lib/store.py`: accepts `ctx`; run_id from ctx; manifest now records
  `as_of` and `code_version`.
- Blocks `usa/china/commodities/peru`: `build(ctx=None, ...)`; all IMF round
  selection (`imf.path`), information stamps, dtp, day_in_cycle, information
  sets and both Peru `live_forecast` calls use the context as_of.
- `pipeline/blocks/_common.py`: `information_stamp(..., as_of=None)`.
- Stages: `chain` and `domestic` pass `ctx` into builders; `_models` and
  `nowcast_job.run(as_of=...)` thread it into the horse race window and
  `live_path(today=as_of)`; `fanchart.load_context(as_of)` and its weekly
  sweep; `report` uses one as_of for the DATE token and the Peru WEO lookup;
  the generic `forecast` stage likewise.
- Data-refresh timestamps (`sources/registry.py`, `sources/nbs.py`) still use
  the wall clock deliberately: they record when a fetch HAPPENED (provenance),
  not which releases are selected.

**Tests added** (`tests/test_run_context.py`, 6): source scan proving no
wall-clock read remains in `pipeline/blocks/*`, `pipeline/stages/*`,
`pipeline/lib/nowcast_job.py`; context immutability and normalization;
explicit run_id; `resolve_as_of` preference; `information_stamp` under two
as-of dates (documents the period-end-timestamp dtp convention); RunStore
manifest recording. All six failed before the change, all pass after.

**Functional acceptance**: `usa.build` under as_of 2026-08-03 selects WEO round
`2026-08-live`; under 2026-05-10 selects `2025-04`; stamps match the as_of; the
same as_of twice gives byte-identical output (sha checked). Product CSV backed
up and restored during the check.

**Behavior change**: none at default (as_of today reproduces previous behavior
modulo the midnight race); historical as_of now selects historical WEO rounds
and stamps consistently.

**Unresolved / scoped out**: an historical as_of selects releases through the
DELAY RULES but reads today's cache SNAPSHOT (latest-revised values, current
SPF file). True vintage reconstruction is Phase 3 by design. The commodity
block's ragged-edge pins and daily flash months are not yet as-of-filtered;
that lands in Task 3 where the conditions are rebuilt on the master grid.

**Suite after task**: 28 passed (22 baseline + 6 new).

## Task 2: terms-of-trade transformation contract

**Audit claim**: the commodity block supplies arithmetic YoY ToT growth into a
Peru BVAR trained on log YoY (`g_tdi`). **Verdict: CONFIRMED, and NARROWED.**
Reproduced on the 353-month overlap: raw gap RMSE 1.339pp, max 6.135pp,
January 2026 gap 5.78pp (matching the audit's numbers); `100*log1p(x/100)` of
the block series reproduces `g_tdi` to machine precision, proving both are the
same official index under two transforms. Narrowing: the S1 BACKTESTS are NOT
contaminated. In backtests the Cond-BVAR derives its `g_tdi` conditions from
the panel itself (log units throughout, via `partial_vars`); only the LIVE
custom path carried the wrong units. Therefore no backtest rebuild is needed,
contrary to the audit's suggested follow-up.

**Canonical definition (documented decision)**: the model-side canonical ToT
transform is LOG YoY, because that is what S1 was estimated on and log growth
aggregates additively. The commodity block keeps PUBLISHING arithmetic YoY (the
BCRP-comparable headline unit) and now declares it (`units =
pct_yoy_arithmetic` stamped in the CSV). The Peru interface converts: centre
exactly via `100*log1p(x/100)`, scale to first order by the delta method
`s/(1+x/100)`; gamma is a shape ratio and passes through. The interface
REFUSES any declared units other than arithmetic.

**Changes**: `pipeline/blocks/_common.py:arith_to_log_yoy` (with a domain
guard at -100 percent); `pipeline/blocks/commodities.py` stamps `units`;
`pipeline/blocks/peru.py` validates the declaration, converts centre and s
before building paths and calib, and stamps
`tot_condition_units=log_yoy_pct_converted_from_arithmetic`.

**Tests added** (`tests/test_transform_contract.py`, 4): the log/arithmetic
identity; converter centre + delta-method scale; domain rejection; and a
data-gated overlap test asserting BOTH that the raw units still differ (>0.5pp
RMSE, so the converter stays necessary) and that the conversion reproduces
`g_tdi` within 1e-6.

**Live effect** (Peru block rebuilt, product restored afterwards): centres move
by -0.01 to -0.04pp and widths are unchanged; the small size is expected since
the measured short-horizon ToT pass-through is weak, but the condition now has
the trained definition. Regeneration of the published products happens once at
the Phase 1 gate.

**Suite after task**: 32 passed.

## Task 3: master quarter grid

**Audit claim**: US and China products start 2026Q3 while Peru and ToT start
2026Q2; the Peru custom US path therefore begins with NaN even though US Q2 is
released, and the leading NaN survives the forward fill. **Verdict:
CONFIRMED.** Reproduced from the current products: usa and china grids
2026Q3..2028Q2, commodities and peru 2026Q2..2028Q1; the reconstructed US
custom path was ``{2026Q2: None, 2026Q3: 1.72, ...}``. Because custom
conditions override the model's internal SPF conditions even when NaN, the
released US quarter was effectively FREED, not conditioned.

**Changes**
- `pipeline/blocks/_common.py:released_first(grid, released, forecast, name,
  required=True)`: resolves every Peru grid quarter as released data first,
  block forecast second; raises with the variable name and the missing
  quarters when a required condition cannot be resolved (no more silent fill).
- `pipeline/blocks/peru.py`: builds all three imported paths through the
  resolver. "Released" honours the run's as-of date through each series'
  publication delay (US 30d; g_tdi 40d, the catalog value, chosen over the
  35d used elsewhere because over-claiming release timing is the dangerous
  direction; China IP 15d). The released mask zeroes path uncertainty for data
  quarters, which also replaces the old `ip_seen >= 3` special case with the
  general rule. The ffill hack is gone. The stamp now records
  ``conditions_released`` (e.g. ``us=1,tot=0,ip=1``).

**Tests added** (`tests/test_quarter_grid.py`, 6): released-before-forecast,
released-wins-overlap, required-hole raises with names, interior gap raises,
optional variable returns NaN without raising, NaN sources treated as missing.

**Live effect** (rebuild at as_of 2026-08-03, product restored): released
conditions us=1, ip=1, tot=0 (ToT Q2 has 2 of 3 months published, so the block
forecast correctly remains). Centres move at most 0.07pp (2026Q3 3.04 to
2.98) as the released US print replaces the previously freed condition;
published widths unchanged (they are the state-matched empirical scales).

**Note for later tasks**: the NaN-override semantics inside
`ConditionalBVARNowcaster.custom_conditions` remains a footgun; it is now
unreachable from production (the resolver refuses NaN), and a model-side guard
is worth adding when that class is next touched.

**Suite after task**: 38 passed.

## Task 4: recompute the China live forecast

**Audit claim**: `pipeline/blocks/china.py` republished `china_profile_fan.csv`
(a notebook cache) and stamped it with the current date and WEO round.
**Verdict: CONFIRMED** from the code (the CSV supplied the centre path; only
the stamp was current).

**Changes**
- NEW `pipeline/blocks/_china_model.py`: the SELECTED production specification
  promoted verbatim from notebooks/china/forecast/03: five-member Combo (RW,
  D-ARX(leaders), BVAR(3v), Cond-BVAR, Cond-BVAR+SS with the WEO-anchored
  steady-state prior), 2012 regime floor, TIGHT Minnesota prior, blend
  alpha=0.5 to the entropically WEO-tilted live conditional-BVAR draws
  (`forecast.tilt`), and node 1 from a FRESH weekly nowcast sweep at the run's
  as-of combined with Adaptive-IC weights. Cached LIVE ladder rows are dropped
  before combining (only y_true-realized history is reused), so a stale live
  nowcast can never be republished.
- `pipeline/blocks/china.py` rewritten: error calibration (blend backtest
  errors to TPN per horizon) unchanged; the centre path now comes from
  `live_profile(as_of)`; the node-1 band is fitted to the information-bin
  residual pool instead of the cached q90 columns; provenance stamped into the
  product: weo_round, weo_current_year, model_members, blend_alpha,
  nowcast_members, nowcast_information_bin, tilt_ess_share, ladder_cache_sha,
  horse_cache_sha, panel_sha. Missing required inputs raise (`_require`);
  nothing is restamped.
- Research caches remain INPUTS as calibration data (combination weights,
  historical blend errors), declared and hashed. That boundary is documented
  in the module docstring.

**Tests added** (`tests/test_china_block.py`, 3): production-wide source scan
proving `china_profile_fan` is never read again; `_require` raises with a
refusing-to-restamp message; provenance keys contract.

**Functional acceptance** (products restored afterwards)
- as_of 2026-08-03: recomputed path [5.07, 4.53, 4.46, 4.58, 4.65, 4.73,
  4.85, 4.95] vs the old restamped cache [5.07, 4.51, 4.45, ...]: the promoted
  code reproduces the published rule (differences of 0.01-0.02pp reflect the
  fresher panel), in 12 seconds.
- Controlled input change: as_of 2026-07-20 selects WEO round 2026-04 and
  moves the blend by up to 0.21pp; as_of 2026-08-03 selects 2026-08-live.
- Determinism: repeated 2026-08-03 builds give identical centres.
- Fail-closed replay: as_of 2026-05-10 RAISES ("live nowcast sweep produced no
  rows") because the current snapshot already contains the July GDP release; a
  deep historical as-of on a future-contaminated snapshot now refuses instead
  of publishing an incoherent node. True replay needs Phase 3 vintages.

**Note**: `metadata.FORECAST["china"]` (three-member spec for the unused
generic stage) is now visibly stale next to the promoted five-member profile;
left untouched here, flagged for the documentation task.

**Suite after task**: 41 passed.

## Task 5: coherent fallback policy

**Audit claim**: a failed block lets Peru glob `products/blocks/*.csv` with no
vintage checks, so a report can mix current US data with stale China or ToT
assumptions. **Verdict: CONFIRMED** (`peru.py` used
``blocks.get(name) or glob(products/blocks/...)``; chain.py advertised the
"previous vintage" without any check).

**Changes**
- NEW `pipeline/lib/bundle.py`: `write_bundle` (chain records blocks/bundle.json
  with one as_of, run_id, code_version, registry sha256 and per-artifact sha256
  plus grid, ONLY when every required block succeeded, so the file itself is
  the chain success marker); `load_prior_bundle` (newest COMPLETE prior bundle
  passing schema, as_of/run_id presence, registry hash, code-version equality,
  artifact existence + hash, contract columns, no required NaNs, recorded
  grid); `resolve_block_paths` (complete current run passes through; anything
  less uses ONE whole validated prior bundle or raises `BundleError`).
  Code-version equality is deliberately strict: a bundle from different code is
  not evidence about today's model.
- `pipeline/stages/chain.py` writes the bundle after the block loop and reports
  an incomplete chain explicitly.
- `pipeline/blocks/peru.py` obtains satellite paths ONLY through
  `resolve_block_paths`; the global-products glob is gone. A fallback stamps
  `fallback_bundle_run` and `fallback_bundle_as_of` into the published fan and
  adds a WARNING line to the report.
- `RunStore.mark_success()` writes `_SUCCESS`; `pipeline.main` writes it after
  the manifest and moves the `latest` symlink update AFTER the marker (the
  full atomic lifecycle remains Phase 5).

**Tests added** (`tests/test_bundle.py`, 11): incomplete bundles are never
written; valid roundtrip; tampered artifact, NaN, code-version and registry
changes rejected; newest valid bundle wins over a newer broken one; the
current run is never its own fallback; the resolver passes through a complete
run, NEVER mixes runs (all three paths from the prior bundle even when two
current blocks succeeded), and aborts when nothing coherent exists.

**Behavior change**: a run with a failed satellite now publishes Peru only
from one coherent prior bundle (prominently stamped) or not at all. Ad-hoc
`build()` calls without a blocks dict now require a prior bundle too; pass
explicit paths for interactive work.

**Suite after task**: 52 passed.

## Task 6: information-index frontier and the China January-February release

**Audit claims**: (a) the information-index denominator counts only cells
present in the current panel slice, so live incomplete quarters reach 1.0;
(b) `fill_single_month_gap` copies February into January BEFORE release
masking, dating February information a month early. **Verdict: both
CONFIRMED** (`MIDAS/adaptive.py` used `rel[sub.notna()]`;
`targets/china.py` assembles with `fill_jan_gap=True`).

**Changes (MIDAS, version 0.2.2 to 0.2.3)**
- `add_information_index` now builds the frontier from the release CALENDAR:
  every started column, at every calendar month it structurally publishes, in
  the window, whether or not the cell exists in the snapshot. Numerator =
  cells that EXIST and whose release date has passed; denominator = the
  expected lattice. Structurally missing calendar months are inferred from
  history (a month observed at least three times and never non-missing) with
  an explicit `structural_missing={column: {months}}` override. The inference
  also handles quarterly-placed monthly columns automatically. Historical and
  live rules are now identical by construction, and an expected cell that
  never materialises keeps the index below 1 (the honest reading; my first
  implementation counted scheduled dates in the numerator and the new test
  caught it).
- MIDAS tests added (`tests/test_adaptive_frontier.py`, 5): incomplete
  quarter below 1 with closure once the snapshot fills; denominator invariant
  to truncation; inferred structural January; explicit override; series-start
  limits. MIDAS suite: 39 passed.

**Changes (NowForecasting)**
- `analysis/data.py:assemble_panel`: with `fill_jan_gap=True`, January is
  filled ONLY for the de-cumulation identity (February = 2*cum_feb - cum_jan)
  and blanked afterwards at the originally missing positions. February alone
  carries the combined NBS release, at its true reference month and its true
  mid-March release timing; no January-dated cell exists to leak in February.
  Columns that genuinely publish January are untouched.
- `pyproject.toml`: MIDAS pin moved to the 0.2.3 wheel URL.

**Tests added** (`tests/test_information_frontier.py`, 4): live incomplete
quarter below 1; denominator invariance; the combined release is invisible at
Feb 20, present at Mar 20 (step at the actual release date); the
fill-and-blank panel rule (January structural, February equals the cumulative
figure by the equal-base-share identity, normal columns keep January).

**Empirical effect** (real China ladder + panel): the maximum corrected index
across all origins more than 30 days before publication is 0.860 (the old
rule reached 1.000 at 78 days out); recent quarters close at 0.896 to 0.983,
the shortfall being cells that never materialised.

**Unresolved / user actions**
- The MIDAS 0.2.3 WHEEL must be built and published (and `uv.lock` refreshed)
  before any uv-synced deployment; the dev PYTHONPATH path already runs the
  fixed source. Until then a wheel-based run uses the old index semantics.
- China research caches (ladder_full, horse race) were generated under the
  old panel rule (January cells present). The live sweep now runs on the new
  panel; historical member rows still embed the old January cells. The drift
  is second order (January cells only) but the notebooks should be re-run
  once to regenerate the caches before the next publication.

**Suite after task**: 56 passed (NowForecasting) + 39 (MIDAS).

## Task 7: one official Peru nowcast artifact

**Audit claim**: the configured headline Adaptive-IC combines RW, Bridge and
P-MIDAS while the fan's first node and the flagship figure averaged only
Bridge and P-MIDAS live values. **Verdict: CONFIRMED, and slightly WIDER than
audited**: the fanchart's band machinery also combined only the two members,
a third variant of "the nowcast" under one label.

**Changes**
- NEW `pipeline/lib/nowcast_artifact.py`: `official_from` derives ONE artifact
  from the nowcast job's own outputs: the newest live Adaptive-IC origin
  (value, quarter, origin date, days to publication), the official member
  list, LEARNED bin weights and REALIZED weights (the documented
  missing-member rule: learned weights renormalize proportionally over
  members with a finite live nowcast, the MIDAS combine convention), member
  values, information index/bin, and the node two-piece normal fitted to the
  information-bin residual pool. `sweep_from` persists the full weekly
  Adaptive-IC live path. `write_official` publishes
  `products/peru_nowcast_official.csv` and `products/peru_nowcast_sweep.csv`
  plus run copies; `load_official`/`load_sweep` validate the consumer's as-of
  and refuse a stale artifact.
- `pipeline/stages/domestic.py` writes the artifact right after the Peru
  nowcast job; `_models.py` exposes results on the store.
- `pipeline/blocks/peru.py`: the fan's first node now CONSUMES the artifact
  (value, TPN scales, information state); the block's private nowcast
  machinery (ladder read, two-member combine, pools, live Bridge+P-MIDAS
  averaging, PERU_LEADERS import) is deleted. A quarter or as-of mismatch
  raises.
- `pipeline/stages/fanchart.py`: the flagship sweep is read from the
  persisted official sweep (no recomputation of any ensemble); the rc frame
  behind the band pools comes from the newest run's saved nowcast frame; a
  coherence assertion compares the sweep endpoint with the official value
  (tolerance 1e-3, above the artifact's 4-decimal storage rounding). During
  verification this assertion correctly caught the fallback path recombining
  from the stale ladder cache (2.7057 vs 2.7019), which motivated persisting
  the sweep.

**Tests added** (`tests/test_nowcast_artifact.py`, 3): value = newest live
Adaptive-IC origin with members, bin, TPN fields and realized weights summing
to one; missing-member renormalization (RW missing: 0.5/0.3 learned weights
become 0.625/0.375); consumers reject a stale as-of and a missing file.

**Functional acceptance** (full pipeline at as_of 2026-08-03)
- official nowcast 2.7019 (2026Q2, bin 3, index 0.917 under the corrected
  frontier, dtp -22, pool 31), realized weights RW 0.058, Bridge 0.940,
  P-MIDAS 0.002;
- fan node 1 mode 2.7020 (4-decimal rounding), sweep endpoint 2.7019, 15
  weekly origins: ONE number on every surface;
- note the definitions genuinely differed before: the two-member average was
  publishing ~2.64 next to a ~2.70 configured headline.

**Suite after task**: 59 passed (NowForecasting) + 39 (MIDAS).

## Phase 1 review gate

Full pipeline verified end to end at as_of 2026-08-03 (nowcast 19s, chain:
usa 4s + china 11s recomputed + commodities 69s, Peru fan 270s, figures,
Beamer report compiled; `_SUCCESS` written, `latest` updated after it).

| gate check | status | evidence |
|---|---|---|
| transform contract | PASS | raw ToT gap 1.34pp RMSE vs converted 0.000; interface converts and stamps units; 4 tests |
| master quarter grid | PASS | `conditions_released us=1,tot=0,ip=1`; released US Q2 conditions node 1; holes raise; 6 tests |
| coherent fallback | PASS | bundle.json + validator; resolver never mixes runs; 11 tests, all checks exercised |
| information index | PASS | live max 0.860 at >30d pre-release (was 1.000); historical = live rule; 5 MIDAS + 4 project tests |
| nowcast identity | PASS | official 2.7019 = fan node 2.7020 = sweep endpoint 2.7019; one ensemble, weights recorded |
| run context | PASS | one as-of everywhere; WEO round selection follows as-of; wall-clock scan enforced |
| China recompute | PASS | live path recomputed in 11s with provenance hashes; controlled WEO change moves it; restamping impossible |

Test totals: NowForecasting 59 passed (was 22 at baseline; 37 added across 8
files); MIDAS 39 passed (5 added). No baseline test modified except the
documented dtp-convention expectation inside the new test file itself.

**Unresolved issues carried forward**
1. The MIDAS 0.2.3 wheel must be BUILT AND PUBLISHED and `uv.lock` refreshed;
   until then only PYTHONPATH-based runs use the corrected information index.
2. China research caches (ladder_full, horizon_2012floor) predate the
   January-blank panel rule and the corrected index; regenerate the notebooks
   once before the next external publication (second-order drift meanwhile).
3. Historical as-of replays remain release-RULE accurate but read the current
   cache snapshot (latest-revised values); true vintage reconstruction is
   Phase 3, and the China block now fails closed on deep replays rather than
   fabricating a node.
4. `metadata.FORECAST["china"]` (unused generic stage) is stale next to the
   promoted five-member profile; reconcile in the documentation task.
5. The Peru official artifact currently records a bin-3 information index of
   0.917 under the corrected frontier (previously 0.97 under the flawed one);
   band recalibration stays FROZEN per instructions until Phases 2 and 3
   deliver the exact-chain errors.

# Phase 2 implementation log

Phase 2 started after the user published and installed the MIDAS 0.2.3 wheel
(verified: pyproject pin and uv.lock resolve the 0.2.3 wheel with hash).
Baseline re-confirmed: 59 passed.

## Task 8: registry migration

**Changes**
- NEW `pipeline/config/build_registry.py` (tracked, idempotent): merges the
  hand-curated production-critical entries (preserved verbatim), the active
  rows of `sources/catalog.csv`, the target modules' `DELAYS` dictionaries
  (they fill the catalog's 27 missing NBS lags), and the local panels (for
  monitored columns and start dates). Registry now covers 113 series: 29
  required for publication, 85 monitored, 28 declared-only
  (`monitor.type=none`, static metadata without an operational status, so the
  fixed seven-status vocabulary is never diluted).
- Schema 0.2.0: `required_for_publication` (required boolean) and monitor
  type `none`; registry validates under jsonschema.
- Conflict policy: where the catalog and model code both declare a lag and
  disagree, the LARGER lag wins (claiming an earlier release than reality is
  the information-leak direction); resolutions in
  `docs/audit/registry_reconciliation.md`. Three NBS rows adopted code lags.
- Hand-resolved cross-code conflict: the official ToT index appeared as
  `tdi` (40d), `g_tdi` (35d) and `g_pe_tot` (35d); aligned to 40 everywhere:
  `targets/commodities.py` (DELAYS and TARGETS), the notebook plumbing
  fallback, and the curated registry entries. `pipeline/blocks/peru.py`
  already used 40 for its released-quarter gate (Task 3).
- Static/dynamic separation enforced by test: `current_availability_status`
  must be null in the registry for every series.

**Tests added** (`tests/test_registry.py`, 6): schema validation; every
active catalog series declared; unique codes and static-only; the required
set contains the production core and its size is pinned; registry never
claims an EARLIER release than the catalog; the ToT lag is 40 in registry
and code.

**Observation**: the expanded dashboard immediately surfaced 12 genuinely
stale series (e.g. `credme`, `credmn` ending 2026-05 with June due), which is
exactly the visibility the migration was for; none are in the required set.

**Suite after task**: 65 passed.

## Task 9: append-only per-series refresh events

**Changes**: NEW `pipeline/lib/refresh_events.py`: one JSON line per (series,
attempt) at `output/data_quality/refresh_events.jsonl`, never rewritten, with
the full field set (internal_code, attempted_at, as_of, status, detail,
source_url, raw_response_sha256, parser_version, n_rows/n_cols, first/last
observation, artifact_sha256, and the manual-override block with author,
reason, effective dates). Event statuses are attempt-level; the state-level
readings (`not_yet_released`, `stale_observation`) remain the dashboard's
job when it joins registry + caches + events at one as-of. An undocumented
manual override (no author or reason) is rejected. The data stage now records
events for every provider and target refresh, per covered registry series
(`codes_for` maps ingestion scripts and monitor targets to codes), with
source_unavailable inferred from network-flavoured exceptions and
parser_version stamped from the run's code_version.

**Tests** (`tests/test_refresh_events.py`, 4): append-only roundtrip with
observation metadata; invalid status and undocumented override rejected;
error events drive the dashboard status (precedence verified end to end);
provider/target code mapping.

## Task 10: ingestion defects

**Verdicts**: all four audit claims CONFIRMED, one amplified: the BCRP
positional call is not merely wrong, it cannot BIND (`names` lands in
`frequency`, then `start_period` collides), so the central BCRP path never
ran at all.

**Changes**
- `sources/bcrp.py`: named-argument call only.
- `sources/base.py`: `atomic_write_parquet`, `atomic_write_csv` (tmp file in
  the same directory, then `os.replace`) and `validate_cache` (required
  columns present and non-empty; no prior column disappears or empties).
- `sources/commodities.py`: `REQUIRED_COLUMNS`; `refresh` now uses the STRICT
  path (FRED and BCRP must both deliver; the lenient degrade in
  `fetch_prices` stays for ad-hoc use only), validates against the prior
  cache, and writes atomically. A BCRP outage or a degraded frame keeps the
  prior cache and reports failure.
- `sources/us.py`, `sources/imf.py`: cache writes atomic.
- `targets/china.py`: the refresh hook now calls the NBS provider with
  `refresh=True` (cached responses could previously suppress new releases
  forever); the modelling-snapshot merge writes atomically.
- `sources/nbs.py`: `max_age_days` threaded from `get_nbs_data` to the client
  cache check as interim age-based invalidation (release-aware invalidation
  noted as follow-up).

**Tests** (`tests/test_ingestion.py`, 9): signature bind contract both ways;
named-arguments source scan; atomic write leaves the prior file intact on a
mid-write crash; validation blocks missing and emptied required columns;
BCRP outage never overwrites; degraded frame rejected with prior values
intact; successful refresh written atomically with no tmp residue; China
refresh bypasses response caches; NBS age invalidation exists.

**Suite after task**: 78 passed.

## Task 11: Peru production refresh

**Audit claim**: a refresh-enabled run can publish from stale core Peru inputs
because `refresh()` only updates INEI and reports the spec3 cache's age.
**Verdict: CONFIRMED** (the docstring itself said the panel is "NOT rebuilt
here").

**Changes** (`targets/peru_gdp.py`)
- `_release_due(last_obs_month, as_of, delay)`: the pure due rule (the latest
  month whose release date has passed must be in the panel).
- `panel_release_due(as_of)`: applies it to the spec3 snapshot's raw monthly
  GDP proxy (`g_pbim`, 51-day delay), the slowest required series.
- `rebuild_panel()`: the REAL rebuild through `core.preprocess`
  (`locate_x13_binary` fails fast with an actionable message, then
  `build_processed_artifacts(spec3, refresh_downloads, refresh_sa)` and
  `export_processed_snapshot` into `input/peru`).
- `refresh(as_of=None)`: INEI update as before; then nothing due means an
  explicit "panel current" message, and due-but-unbuildable RAISES
  `PanelRebuildError` ("a refresh that only touched INEI is NOT a successful
  Peru refresh"). The data stage records the raise as an ingestion_failure
  event and the preflight blocks on the stale panel, so the failure is
  doubly closed.
- `core/preprocess.export_processed_snapshot` writes stay as-is for now (the
  builder is executed manually until an X13 binary is installed; the export
  path is exercised then).

**Tests** (`tests/test_peru_refresh.py`, 5): due-rule boundaries (including
release-day edge and an ahead-of-due panel); due-but-unbuildable raises;
due-and-possible rebuilds; nothing-due never invokes the builder; live
consistency check against the real cache.

**Live state at 2026-08-03**: panel through 2026-05 and 2026-05 is the latest
due month (June g_pbim releases about August 20), so today's refresh is
legitimately current, and the August 20 due date will exercise the
rebuild-or-fail path. NOTE for the operator: no X13 binary is currently on
this machine (`locate_x13_binary` fails), so the first due refresh will fail
closed until X13 is installed. That is the intended behavior, but install it
before August 20.

## Task 12: availability preflight gate

**Changes**
- NEW `pipeline/lib/preflight.py`: builds the dashboard from registry +
  caches + event log at the run's as-of, saves markdown and CSV under
  `runs/<id>/data_quality/`, and raises `PreflightError` when any
  `required_for_publication` series is stale, failed, invalid or unreachable.
  Overrides come from `params.AVAILABILITY_OVERRIDES` and must carry an
  author and a reason; every waiver is recorded as a `manually_overridden`
  event in the append-only log and in the run manifest; override entries for
  unknown codes are warned about. `pipeline/main.py` runs the gate after the
  data stage and before ANY estimation stage.
- `pipeline/config/params.py`: `AVAILABILITY_OVERRIDES = {}` documented.

**Tests** (`tests/test_preflight.py`, 4): every blocking status blocks a
required series while optional series never block; healthy states pass;
documented overrides waive and undocumented ones do not; unknown override
codes are reported.

**Live firefight (the gate working as designed).** The first live preflight
BLOCKED on m2_yoy and us_vix; the instrumented refresh then caught an
IndentationError I had introduced into `sources/imf.py` during the atomic
write conversion (recorded as ingestion_failure by MY OWN event log, which
is exactly the point). Diagnoses and honest fixes:
- `us_vix` was NOT stale upstream (July was in the US cache): the commodity
  panel anchored its index on the FRED price index and TRUNCATED every faster
  series to June, so the published July VIX never reached the ToT model's
  ragged-edge conditioning. Fixed in `targets/commodities.py:_monthly_block`
  (union index) with a regression test comparing the panel against the
  upstream source.
- `m2_yoy` is genuinely late AT THE SOURCE: a fresh NBS fetch returns M2 only
  through April; the NBS portal republishes PBoC M2 roughly a quarter late,
  so the registry's 13-day lag was wrong metadata from the start. Corrected
  to the measured 95 days with the PBoC-direct loader noted as the
  timeliness fix; the models have always handled this lag through their
  balanced-system logic, so this is metadata repair, not gate gaming.
- `sources/imf.py` indentation repaired; refresh verified live (32 rounds
  cached, live round 2026-08 through 2031); the WEO block cleared once a
  SUCCESSFUL attempt was recorded through the instrumented path (error
  events correctly keep precedence until then).

**Preflight after repairs: PASSED.** Ten NON-required research-panel series
remain flagged (credit, liquidity, EMBIG, some BCRP prices lag their declared
catalog lags by about a month); they are correctly non-blocking and now
visible for a later metadata sweep.

## Phase 2 review gate

Full pipeline at as_of 2026-08-03 WITH the availability gate in `main`:
preflight passed and its dashboard saved under `runs/<id>/data_quality/`;
nowcast 19s; chain usa 4s + china 11s + commodities 68s with the bundle
recorded; Peru fan 269s; figures; report compiled; `_SUCCESS` written.
Identity held (official 2.7019 = fan node 2.7020); released conditions
us=1, tot=0, ip=1; the fan is numerically unchanged except a 0.01pp wiggle
at 2026Q4 from the commodity panel's union index.

| gate check | status | evidence |
|---|---|---|
| registry migration | PASS | 113 series, jsonschema-valid, 29 required, conflicts reconciled and documented; 6 tests |
| refresh events | PASS | 80+ live events recorded incl. two real failures; error precedence verified; 4 tests |
| ingestion defects | PASS | BCRP bind contract, strict commodity refresh, atomic writes, NBS bypass; 10 tests |
| Peru refresh | PASS | due-gate live-consistent; rebuild-or-raise; 5 tests; X13 install needed before Aug 20 |
| preflight gate | PASS | blocked on real staleness AND on my own injected bug, then passed after honest fixes; 4 tests |

Test totals: NowForecasting 88 passed (37 Phase 1 + 29 Phase 2 additions over
the 22 baseline); MIDAS 39 passed via the published 0.2.3 wheel.

**Unresolved / operator actions after Phase 2**
1. Install an X13 binary (`core.preprocess.locate_x13_binary` documents the
   search paths) BEFORE the June Peru print is due (~2026-08-20), or the next
   refresh will fail closed by design.
2. Ten non-required series have optimistic catalog lags (about a month);
   sweep them in a later metadata pass.
3. A PBoC-direct M2 loader would recover ~80 days of timeliness that the NBS
   portal republication loses.
4. Dated release calendars (audit task 2.3) remain future work; the registry
   still runs on scalar lags, now honestly reconciled.
5. China research caches still predate the January-blank rule (carried from
   Phase 1).

# Phase 3 implementation log

## Task 16: evaluation metrics and the selection split (built first, foundation)

NEW `core/evaluation.py`: EVALUATION_REGIME constant; frozen split
(SELECTION_END 2022Q4, HOLDOUT_START 2023Q1, frozen 2026-08-03 and honest
FORWARD, since all earlier choices saw the full sample); point metrics (rmse,
mae, bias, finite_share, directional_accuracy, revision_size); density
metrics for the TPN parameterisation (pdf, log score, PIT, coverage with
Wilson intervals, Winkler interval score, weighted interval score); DM and
CW wrappers over MIDAS with finite-sample guards (the CW wrapper was fixed to
MIDAS's (y, f_restricted, f_unrestricted) signature); a scoreboard that
always carries n, finite share and the regime label so mixed-sample
comparisons cannot pass silently. Tests (`tests/test_evaluation.py`, 9): all
hand-computed, including TPN normalization by quadrature, PIT uniformity
under the true model, and WIS on a worked example.

## Task 13: relabelling

The regime label is stamped into the published fan CSV
(`evaluation_regime` column), the run manifest, the report's information
paragraph ("pseudo real time on final-vintage data ... not genuine real
time"), and every scoreboard row. Nothing calls the exercise real time.

## Task 14: exact-chain historical backtest

NEW `pipeline/lib/exact_chain.py` (runnable module, resumable, atomic
writes): 27 origins 2019Q1..2025Q3 at day 30 of the cycle replaying the LIVE
rule end to end: official Adaptive-IC nowcast condition from the ladder
(real-time combine), vintaged SPF/WEO US path with released-first grid fill,
the full recomputed China profile per origin, the masked monthly ToT BVAR
(delay-rule masking, daily-history flash reconstruction, reduced 2x3000
chains), the recursive GDP-to-IP bridge (slope moves 2.72 pre-COVID to 1.21
now), and the S1 conditional BVAR with the same transforms and
hyperparameters. Per-origin no-lookahead assertions stored in
`exact_chain_checks.json`; failing origins are refused, not stored.

Boundaries surfaced and codified while building it (each a live-rule fix):
- the production China rule is UNDEFINED before ~2019Q1 (min_train), now an
  explicit error; the chain replays the rule only where the rule exists;
- the entropic tilt can DECLINE; documented fallback = untilted model path,
  flagged in provenance (live block and chain);
- a WEO round's horizon can end before the grid; flat medium-term tail
  adopted in `usa.py` and the chain;
- Phase 1's January-blank rule broke the commodity model's row-wise balanced
  frame and the ip complete-quarter aggregates at Q1: fixed by
  `fill_structural_january` (model-boundary imputation from the MASKED
  February value, so release timing still cannot leak), applied in the
  commodity model, the Peru block and the chain;
- a same-base cached row could shadow the live China Combo inside the
  combine (quarter stamps are first-day-of-end-month); the cache now
  supplies strictly earlier bases only.

## Task 15: simple benchmarks

In the chain: RW, recursive AR(2), unconditional BVAR, Mean(4), Median(4),
plus the nowcast node at h=1. NEW `pipeline/lib/benchmarks.py` for the
nowcast family on the ladder: equal mean, median, previous-best (real-time),
release-stage switch, and the Bridge/P-MIDAS head-to-head on their common
finite rows; every table carries n and finite share, own-sample rows are
labelled not comparable.

## Phase 3 results (full tables in docs/audit/phase3_evaluation.md)

- Exact chain, common sample, ex-COVID both ends: S1 rel-RW 0.46 / 0.50 /
  0.48 / 0.50 / 0.52 / 0.52 / 0.56 / 0.74 at h=1..8; DM vs RW significant at
  h=1,3,4 and (with n caveats, 9-18 cells) h=5..7. Node 1 equals the official
  nowcast (0.56pp RMSE). Mean(4) is the best simple combination.
- S1 bias GROWS with horizon (+0.5pp at h=4 to +2.0pp at h=8): launches from
  the 2022-2023 slowdown under-predicted the 2024-2025 recovery. Feeds
  Phase 4's bias-correction sensitivity.
- Nowcast family: Adaptive-IC 0.809 vs equal mean 0.815 on the full common
  sample, but the mean WINS the frozen holdout decisively (0.457 vs 0.631):
  the audit's complexity concern is CONFIRMED on recent data. The equal mean
  is now the benchmark every nowcast change must beat.
- Bridge beats P-MIDAS 0.850 vs 1.018 on their common 273 rows.

**Suite after phase**: 101 tests (100 passed + 1 network-guarded skip when
offline).

# Phase 4 implementation log

Full evidence in docs/audit/phase4_calibration.md.

## Task 17: TPN mathematics and API semantics

`tests/test_tpn_math.py` (8): normalization + mode continuity by quadrature;
CDF/PPF round trip; mean/variance formulas verified numerically; (s, gamma)
round trip; shortest intervals carry exact nominal mass with equal edge
density; equal-tailed vs shortest provably differ for skewed inputs (shortest
strictly shorter at equal coverage); MLE recovery; fan-frame table equals the
band formula. `forecast/boe_fan.py` renamed the equal-tailed API to
`tpn_equal_tailed_bands` (alias kept) with a never-mix-conventions docstring.
Every invariant passed first run: the audit's "formula is not the defect"
reading confirmed, now enforced.

## Tasks 18-20: sequential calibration, comparisons, sensitivity

NEW `pipeline/lib/fan_calibration.py`: knowable-before rule (outcome
published before the origin under the 52-day release rule), weighted
per-horizon RMS scales, monotone in h, optional pooled quantile-matched skew
and rolling bias shift, COVID weighting variants; runner scores every
variant OUT OF SAMPLE on the exact chain's 105 ex-COVID cells (WIS primary,
log score + coverage with Wilson intervals secondary; pooled cells share
origin shocks, so independence-based tests are read as descriptive).

Results (WIS / cov90): lookahead references 0.714-0.719 / 0.92-0.93;
sequential symmetric 0.737 / 0.886; sequential skewed 0.762 / 0.848;
COVID downweighted 0.848 / 0.962 (over-wide); bias-corrected 1.027 / 0.752;
ref-only exclusion 1.027 / 0.990 (11.5pp widths); COVID included 1.579
(21pp widths).

**Decisions, against the pre-specified scores**
- ADOPTED sequential symmetric: honest (no lookahead) at a statistically
  invisible cost to the full-information references.
  `pipeline/blocks/peru.py` now calibrates through
  `fan_calibration.production_fits` (day-1 and day-30 anchors both
  knowable-before; interpolation unchanged). Published 90 percent widths
  NARROW from 4.4/4.6/5.3/5.5 to 3.5/3.8/4.5/5.0 at h=2..8: the chain's
  conditioned errors are genuinely smaller than the legacy backtest's.
- REJECTED estimated skew (worse out of sample; small pools fit noise).
- REJECTED horizon bias correction (Phase 3's +0.5 to +2.0pp bias is
  regime-driven; correcting it live collapsed cov90 to 0.75).
- COVID both-ends exclusion CONFIRMED against included / downweighted /
  ref-only, now with the sensitivity table the audit required.

**Boundary honestly stated**: joint satellite-error covariance and
generated-regressor propagation need Monte-Carlo draws stored in the chain
(documented `store_sims` hook); today's structural scales entered as a fixed
lookahead reference (0.719, indistinguishable from empirical) rather than a
per-origin contender.

**Suite after phase**: 113 tests green (112 + 1 network-guarded skip when
offline).

# Phase 5 implementation log

## Task 21: atomic run lifecycle

`pipeline/lib/store.py` rebuilt around staging: artifacts are written under
`runs/.staging/<run_id>/`, `write_manifest` hashes every tracked file (sha256
+ bytes) and fails closed on a missing or empty artifact, `_SUCCESS` is
written, `promote()` moves the run into `runs/<run_id>` with one atomic
`os.replace` (refusing to overwrite a promoted run), and `latest` may only
move after promotion. A failed run keeps its staging directory with a
`_FAILED` marker. The manifest now records: code version (commit + dirty-diff
digest, from RunContext), Python/platform and every installed package version,
registry sha256, model seeds read live from the modules that use them
(`pipeline/lib/provenance.py`; literals lifted to `FAN_MC` in peru and
`BVAR_SEED` in the China block so nothing is duplicated), frozen-calibration
hashes, stage timings and statuses. `pipeline/main.py` wraps every stage,
records failure into the manifest, quarantines, and re-raises; stage imports
became lazy so the lifecycle works without the private data layer.

**Latent hazard found and closed while testing**: `load_prior_bundle` scanned
every directory under `runs/`, so a HALF-WRITTEN run whose chain stage had
already produced a valid `bundle.json` could have served as the domestic
stage's fallback. Candidates now require the `_SUCCESS` marker and dot-dirs
are excluded; regression tests pin both.

Tests: `tests/test_run_lifecycle.py` (8, written failing first) covering
promotion, marker requirement, no-overwrite, quarantine, latest ordering,
manifest validation, and the bundle gate; `tests/test_bundle.py` fixtures mark
their fake runs promoted. Wiring verified live: an all-stages-off run promoted
with a complete manifest, and a synthetic preflight failure left
`.staging/<id>/` with `_FAILED`, no promoted dir, no `latest`. Verdict: audit
finding confirmed (runs were written in place with no failure isolation).

## Task 22: production notebook dependencies removed

Inventory (all verified in code, wider than the audit's list): notebook CODE
imported via `sys.path.insert` in peru block, exact chain, and fanchart stage
(`notebooks/peru/forecast/common.py`); notebook CACHES read via `rglob` in
five modules (Peru nowcast ladder, day-1/day-30 S1 prior pools, China nowcast
ladder, horse race, WEO tilt); one stale FRED-key message.

Fixes, smallest defensible:
- `common.py` PROMOTED to `pipeline/blocks/_peru_panel.py` (panel builder,
  conditional-BVAR factory, nowcast lookup); a shim at the old path keeps the
  notebooks importing the production implementation, so research and
  production share one information set by construction. The fanchart insert
  was vestigial (`nowcast/` is a root-level production package) and is gone.
- Six research parquets FROZEN into `calibration/` with sha256, size, source
  and role pinned in the tracked `calibration/MANIFEST.json`. Production
  reads them only through `pipeline/lib/calibration_assets.asset_path`,
  which fails closed on absence or hash mismatch: a lab rerun can no longer
  silently change published numbers, and the missing-prior fallback that
  would have quietly narrowed fan pools is now a hard error.
- Enforcement: `tests/test_no_notebook_deps.py` AST-scans production
  packages (pipeline, forecast, core, targets, nowcast) so no non-docstring
  string literal may reference the notebook tree, plus loader tamper tests.

Reproduction identity: `production_fits("2026-08-03")` through the frozen
assets equals the published fan's sigmas to 1e-4 (nodes 2..8:
1.0663/1.1563/1.3640/1.5085 flat), and ChainContext builds through the
promoted panel. No published number changed. Verdict: confirmed, and broader
than written.

## Task 23: CI and documentation

CI (`.github/workflows/ci.yml`): locked Python 3.11 environment via
`uv sync --frozen`, then named steps for registry validation, transform
contracts, quarter-grid contracts, TPN invariants, run-lifecycle +
notebook-independence contracts, the synthetic pipeline smoke run
(`tests/test_smoke_pipeline.py`: all-stages-off promotion and the quarantine
path, runnable without private data), and the full public suite.

The boundary was made executable and PROVEN: a simulated public clone
(tracked files only, no sources/input/output/notebooks/binaries) runs
92 passed + 8 skipped, every skip carrying an explicit boundary reason
(module-level guards added to test_ingestion and test_peru_refresh, a partial
guard in test_registry, and the calibration-asset test skips only when NO
binary is present so a broken freeze still fails). A root `conftest.py` pins
repo-root imports because CI's `uv run pytest` does not insert the cwd the
way `python -m pytest` does; verified with `python -P -m pytest` (safe_path).

Documentation: README gained Production runs and Reproducibility boundary
sections; `docs/pipeline.md` documents the entry point, the one-information-
set rule, stages, the atomic lifecycle, frozen calibration assets, the
evaluation-honesty stamp, and CI. Verdict: confirmed (no CI existed; README
described the research layout, not the production path).

**Residuals, deliberately out of scope**: `targets/peru_gdp.py` reads a
private CSV at import time (guarded in tests, ugly but harmless in
production); `uv lock --check` could not run locally (uv not on PATH here),
so lock/pyproject coherence is asserted by CI's `--frozen` sync itself.

**Suite after phase**: 126 locally (24 files); 92 + 8 explicit skips on a
public clone. Every phase-gate baseline preserved.

# Phase 5 review gate

- T21 atomic lifecycle: staging, validation, hashes/versions/seeds recorded,
  `_SUCCESS`, atomic promote, `latest` last, failed runs quarantined and
  excluded from bundle fallback. CONFIRMED and closed.
- T22 notebook independence: code promoted, caches frozen and hash-verified,
  AST-scan enforcement, zero numeric drift. CONFIRMED (broader than the
  audit's inventory) and closed.
- T23 CI + docs: boundary-true workflow, synthetic smoke, README +
  docs/pipeline.md describing the actual production path. CONFIRMED and
  closed.

This completes the audit's five phases. Carried operator actions: install the
X13 binary before the ~2026-08-20 Peru June refresh (fails closed by design);
regenerate the China research caches and re-freeze deliberately when desired
(current frozen vintages predate the January-rule fix, matching what
production consumed all along); PBoC-direct M2 loader; dated release
calendars; the `store_sims` structural-propagation hook in the exact chain;
ten non-required registry series with optimistic catalog lags.
