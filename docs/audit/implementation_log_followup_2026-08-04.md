# Follow-up audit implementation log (2026-08-04)

Implementation lead and statistical auditor session on commit `c83b4cf`
(branch main, worktree CLEAN at start). Method per task: reproduce, failing
test, smallest fix, focused tests, full suite, log, verdict.

**Baseline record**
- Commit c83b4cf (main, clean). Python 3.11.15. MIDAS 0.2.3 and MacroPy
  0.1.10 via LOCAL SOURCE CHECKOUTS (`PYTHONPATH=../MIDAS/src:../MacroPy/src`).
- `uv` NOT installed on this machine and `.venv` is a stale non-native build:
  the LOCKED path (`uv sync --frozen; uv run pytest`) could NOT be executed
  here. Every suite result below is the LOCAL-CHECKOUT environment; the locked
  wheels are exercised only by CI. X13 absent (Peru rebuild would fail closed;
  nothing was due). pdflatex present.
- Baseline suite (local checkout): 126 passed.

## F1 (P0): China calibration assets regenerated under the corrected timing rule

**Reproduction and quantification.** The frozen ladder embedded the OLD rule:
at every probed Q1 pre-release origin (Feb 10 to Mar 14), `Bridge(activity)`
had a finite nowcast built from the leaked January cell while the corrected
panel correctly abstains (e.g. 2023Q1 at Feb 17: old 3.97, corrected: no
forecast). Leak footprint in the frozen ladder: 34 of 34 finite Bridge rows
in that window; 0 of 34 after regeneration.

**Side defect found and fixed first (failing test, then fix).**
`BVARNowcaster._system` required complete 3-month quarters, so under the
corrected panel every Q1 dropped out of the BVAR(3v) system and the VAR
spliced Q4 onto Q2 as adjacent lags. `ConditionalBVARNowcaster` already
tolerated the merged release by design (2 of 3 months). Fix: `min_months`
field (default 3 preserves the generic contract); the China production
member passes `min_months=2`. Tests: `tests/test_bvar_system.py` (3).

**Regeneration.** NEW tracked producer `pipeline/lib/china_assets.py`
rebuilds, in dependency order: the nowcast ladder (the four
`metadata.MODELS['china']` members via `nowcast.release_cycle.run_horse_race`),
the five-member horizon horse race (members from
`pipeline.blocks._china_model.build_members`, with the real-time nowcast
function built FROM the new ladder; eval window pinned to the frozen
predecessor's 2016-07-29 so the comparison isolates the timing rule), and the
WEO tilt backtest (per-origin two-BVAR draw stacks tilted to the WEO round of
the origin; realized history release-masked at the origin; the producer was
reconstructed from the live-rule semantics because the original notebook cell
no longer exists). Alignment validated against the frozen predecessor on a
2019Q1 probe: WEO round and constraint count match exactly.

**Old vs new (the required comparison).**
- Ladder: 2340 to 2376 rows; Bridge(activity) finite share 0.556 to 0.448
  (the honest firing rate); own-sample ex-COVID RMSE Bridge 0.433 to 0.524
  and P-MIDAS 0.766 to 0.850 (the old numbers were flattered by leaked
  information); RW and Q-AR essentially unchanged (0.892/0.888, 1.137/1.148).
- Horizon race: identical shape (1420 rows, 39 bases, same finite shares);
  common-cell deltas mean 0.15 to 0.21pp, max 1.35pp; matched RMSE slightly
  BETTER at every horizon (h1 1.254 to 1.167, h8 1.928 to 1.877); bias
  roughly unchanged.
- Tilt: rounds reproduce exactly (share 1.00); ex-COVID y_tilted deltas mean
  0.59pp, max 3.09pp; 7 bases now DECLINE the tilt (6 COVID-era plus 2022Q3)
  under the documented untilted fallback.
- Adaptive weights (latest learned, P-MIDAS/Bridge): bin2 0.37/0.63 to
  0.08/0.92, bin3 0.39/0.61 to 0.14/0.87. P-MIDAS's mid/late-cycle credit
  was substantially leak-financed.
- China band parameters (blend TPN s by horizon): h1 1.435 to 1.376; h5
  0.649 to 0.963; h6 0.582 to 0.997; h7 0.668 to 0.918; h8 0.841 to 1.086.
  The old narrow long-horizon China bands were partly a leak artifact.

**Freeze.** All three assets frozen via the new deliberate
`calibration_assets.freeze_asset` with producer
`pipeline/lib/china_assets.py`, panel rule
`january_blank_combined_feb_release`, generation date, code version and
member/seed metadata. Tests: `tests/test_china_assets.py` (3: engine-level
no-leak probe, frozen-ladder reproduction from the corrected panel,
manifest provenance).

**Downstream.** Exact chain regenerated (see gate). The Peru chain moved by
at most 0.083pp (mean 0.009pp) across the 105 ex-COVID cells: the corrected
assets are FIRST order for the China block's own weights and bands and,
by measurement rather than assertion, second order for the Peru chain.
**Verdict: CONFIRMED; materiality now quantified on both sides.**

## F2 (P0): the staged run directory is the only information surface

Reproduced: four blocks wrote `products/**` mid-run; `store.blocks` kept
global paths after the chain copied files into staging; fanchart wrote
figures to `products/figures` AND read all four product CSVs; the report
copied `report.pdf` into products before promotion; the Peru fan's first node
read the GLOBAL official-nowcast artifact; `fan_calibration.run()` read the
published fan from products.

Fixes: blocks take a REQUIRED `out_dir` (refuse None); the chain passes
`store.dir('blocks')` and keeps run-local paths in `store.blocks`; the
official nowcast and sweep are written ONLY into the run and their loaders
take explicit run-local paths (no global default); fanchart writes only into
the run and builds its context from `store.blocks` plus run-local artifacts
(module-level context cache DELETED); the report consumes the run's figures
and never touches products; NEW `pipeline/lib/publish.py` is the ONE owner of
the products surface and copies a PROMOTED run's declared surface atomically
(`_SUCCESS` required), wired after promotion in `pipeline.main`
(`params.PUBLISH_PRODUCTS`). The fanchart ip-history context also gained the
`fill_structural_january` treatment (its complete-quarter rule silently
dropped every Q1 after the January correction).

Tests: `tests/test_run_surface.py` (4): an AST scan banning `products`
literals in every run participant except publish.py; interleaved synthetic
runs whose block paths never cross; publish refusing an unpromoted run; a
stale published file surviving a failed run and being replaced only on
success. **Verdict: CONFIRMED and closed.**

## F3 (P0): every final artifact tracked; _SUCCESS means the contract held

`RunStore.write_manifest(strict=True)` now rejects: untracked files in the
run tree (allowlist: lifecycle markers and LaTeX build residue), duplicate
manifest paths, tracked-but-missing-or-empty artifacts, and missing REQUIRED
artifacts (`store.require`); `_track` already refuses paths outside the run.
Stages declare their contracts: chain (each published block CSV plus
bundle.json when complete), nowcast (official artifact and sweep), the fan
(peru_gdp_fan.csv), report (report.tex, report.md, and report.pdf when
`params.REPORT_PDF`). Figure failures now FAIL the fanchart stage (all
figures attempted, then one error naming the failures); report-PDF mode
fails closed without a compiler, with `REPORT_PDF=False` as the explicit
TeX-only contract. report.tex/md/pdf are tracked.

Tests: `tests/test_artifact_contract.py` (6). **Verdict: CONFIRMED and
closed.**

## F4 (P0): production fan calibration frozen

The silent `if CHAIN.exists()` branch in `fan_calibration` is GONE:
`production_fits` (and `load_errors` by default) read the exact-chain errors
ONLY through the frozen, hash-verified `calibration/exact_chain.parquet`
(manifest provenance: producer, claim, regime, code version, registry sha,
dependency versions, panel sha, model spec, seeds, draw counts, origin day,
Monte-Carlo error measurement, base range, generation date). The runner's
lookahead reference now reads the published fan from `runs/latest`, not
products. Tests: `tests/test_fan_frozen.py` (3), including the proof that
`production_fits` is IDENTICAL with `output/backtests/` renamed away.
**Verdict: CONFIRMED and closed.**

## F5 (P0): exact chain fingerprinted, coherent, honestly named

`run_all` now writes `exact_chain_fingerprint.json` (code version,
dependency versions, registry sha, frozen-calibration hashes excluding the
chain's own frozen copy, panel sha, model spec, release rules, seeds, draw
counts, origin day, requested bases) and REFUSES to resume when anything but
the requested-bases extension differs, when the artifact predates
fingerprints, or when the checks file disagrees with the parquet (checks now
written in lockstep per origin, closing the crash window). `supersede()` and
`--new` move an old artifact aside deliberately. The claim is renamed
`exact_rule_reduced_mc`; the reduced-vs-production Monte-Carlo error was
MEASURED at a probe origin (2024Q2, 4x6000 vs 2x3000): mean 0.003pp, max
0.007pp across h=1..8, far below the 0.1pp reporting resolution, and the
artifact can no longer be described as computationally identical to
production. Tests: `tests/test_exact_chain_resume.py` (7).
**Verdict: CONFIRMED and closed.**

## Phase 1 gate

- Exact chain regenerated from scratch under the new fingerprint on the
  corrected China assets (27 origins, all no-lookahead checks stored).
  S1 ex-COVID RMSE by horizon changed at the third decimal at most.
- `calibration/exact_chain.parquet` re-frozen with full provenance
  (sha 02273cbad954); `production_fits` day-30 sigmas moved by at most
  0.0013, so the published Peru widths are STABLE under the corrected
  assets.
- Fan-calibration variant table regenerated: ordering identical to Phase 4
  (adopted sequential symmetric WIS 0.736, cov90 0.895 [0.84, 0.93]); no
  calibration decision reopened.
- One full production pipeline run at as-of 2026-08-04 from the final
  working tree: see the final acceptance section of the session report.

## F6 (P1): stage dependency graph

NEW `pipeline/lib/stagegraph.py`: report needs fanchart, fanchart needs
forecast and nowcast, forecast needs nowcast; `validate_stages` runs BEFORE
any output exists (the old committed default, report alone, now refuses up
front; reproduced against the F2 surface it previously failed deep inside).
The default `params.STAGES` is the full coherent run. The explicit
re-render mode `python -m pipeline.main --report-from RUN_ID` consumes a
NAMED promoted run (marker, manifest, status, as-of validated), copies its
figures, renders under the SOURCE as-of, records `source_run` in the new
manifest, and never publishes. Tests: `tests/test_stage_graph.py` (7).

## F7 (P1): required-input registry completed

`tests/test_input_contract.py` extracts every consumed series from the model
specifications (nowcast ladders, China system and US block, commodity VAR,
Peru S1 and its bolted-on columns, report consensus) and requires one
monitored registry entry each; publication-critical series must be required
and monitored. Gaps found and fixed: `us_fedfunds` and `us_cpi_yoy` (both
NEW entries, required, monitored on the US panel, lags from targets/usa.py)
and `g_pbim` (the raw proxy column the refresh due-gate watches). The
`fed`/`us_fedfunds` alias is reconciled with cross-references in both
entries and in `registry_reconciliation.md`. Registry: 116 series, 32
required. Nothing was made optional to pass the preflight.

## F8 (P1): verified input immutability + private-loader code hash

NEW `pipeline/lib/inputs.py`: `pin_inputs` hashes every production input
file at run start into the manifest (`input_hashes`); `verify_inputs` runs
before `_SUCCESS` and FAILS the run on any drift (mutation, deletion), so no
publication can mix input vintages; a mutation-after-pin test proves it.
Full snapshot re-plumbing of every loader was deliberately not done (the
brief forbids architecture rewrites); detection-plus-abort gives the same
publication guarantee and the manifest makes inputs reconstructible.
`sources_code_sha` (content hash over the git-ignored `sources/*.py`) is
recorded in every run manifest and every refresh event, closing the
code-version blind spot. Tests: `tests/test_input_pinning.py` (5).

## F9 (P1): coherent previous-backtest reuse

`_previous_backtest` accepts only PROMOTED runs no later than the current
as-of whose manifest matches code version, registry sha and evaluation
regime, whose artifact hash equals the prior manifest's record, and whose
frame matches schema, target and the configured member set; every rejection
prints its reason, and a reuse stamps `reused_backtests[{role/name}] =
{run_id, artifact_sha256}` into the new manifest. Tests:
`tests/test_backtest_reuse.py` (4, including future-run and five
incompatibility rejections).

## F10 (P1): historical as-of safety

Fixed with release-rule masking over the final-vintage snapshot (never a
claim of true vintages): the Peru base quarter (was snapshot max, now
`released_last` under the target's release rule), the business-expectations
path (was the snapshot's last observation, now the last observation RELEASED
at as-of under the 15-day rule), the US live path (the newest RELEASED SPF
survey under the 45-day survey rule and the last RELEASED US quarter under
the 30-day rule, instead of snapshot maxima), and availability events
(filtered to events recorded by the as-of in both the preflight and the CLI
dashboard). The fanchart context cache was already removed in F2; the
sources-hash cache was removed so two sequential runs in one process share
no state. Tests: `tests/test_asof_safety.py` (4, future-contaminated
synthetic snapshots).

## F11 + F12: statistical reporting repaired

`docs/audit/phase3_evaluation.md` was REGENERATED from the frozen artifacts
(the literal brace-format placeholder tables are gone; a scan test keeps
every audit document free of them). The interpretation now states: the
holdout equal-mean result (0.457 vs Adaptive-IC 0.631) does NOT establish
either superiority of the mean or grounds to replace Adaptive-IC on one
13-quarter inspected sample; the equal mean is the mandatory benchmark; the
2023Q1+ window is INSPECTED and no longer an untouched holdout
(`core.evaluation.HOLDOUT_INSPECTED_ON`); prospective evaluation starts at
outcomes from 2026Q2 (`PROSPECTIVE_START`), scoring a record frozen
2026-08-04; effective independent quarters are reported next to row counts
(269 rows = 65 quarters; 62 rows = 13 quarters); DM/CW entries at n of 9 to
18 are labelled exploratory and degenerate HAC cells carry no weight; the
symmetric sequential fan is labelled PROVISIONAL in both phase documents;
final-vintage and genuine-vintage inputs remain explicitly separated.

## Final acceptance (2026-08-04)

**Suites.** Local-checkout environment (PYTHONPATH to MIDAS 0.2.3 and
MacroPy 0.1.10 source): 177 passed (126 at baseline; 51 added across 12 new
test files). The LOCKED environment could not be run on this machine (`uv`
not installed; `.venv` is a stale non-native build): locked-wheel results
come only from CI (`uv sync --frozen`), and no local result is claimed as
proof of the locked path.

**Production run.** `python -m pipeline.main --as-of 2026-08-04`, twice:
- Attempt 1 FAILED AT THE PREFLIGHT on the newly added `g_pbim` registry
  entry (my initial plausible-range guess treated a growth column as a
  level). The failure was quarantined in staging; `latest` and every
  published file stayed untouched: the new machinery caught its author's own
  bad metadata before anything else could.
- Attempt 2 (range corrected from the data, COVID trough -363 in scope):
  all five stages, 18 input files pinned and verified, report.pdf via
  latexmk, promoted to `runs/2026-08-04__140712`, `latest` moved after
  promotion, then ONE publish step copied 19 artifacts to products/.

All thirteen acceptance checks passed: code version equals commit plus dirty
digest; manifest status success; all six stage statuses present; every
tracked output hashed; inputs pinned with hashes; sources code sha recorded;
required artifacts declared and present; bundle as-of and run id match the
run; official nowcast 2.7019 equals fan node 1 (2.702) within rounding; ONE
as-of (2026-08-04) across the manifest, bundle, blocks, fan and official
artifact; report.pdf present; evaluation regime stamped.

**Numerical changes to published outputs (vs the 2026-08-03 publication).**
- Peru fan: modes moved at most 0.02pp and 90 percent widths at most 0.01pp
  (2.03/3.51/3.80/4.48/4.96 flat): the corrected China assets are
  measurably second order here, consistent with the chain evidence.
- China block: centres +0.00 to +0.07pp; the node-1 band NARROWED (s 0.542
  to 0.394: honest weights and pools) and the h=3..8 bands WIDENED by 0.15
  to 0.42 (s at h=6: 0.582 to 0.997): the previous long-horizon narrowness
  was partly financed by the January leak. These are the intended,
  explained consequences of the regeneration, not drift.
- The re-render mode was exercised live (`--report-from 2026-08-04__140712`):
  promoted with `source_run` stamped, source as-of carried, nothing
  published, `latest` unchanged. The live exercise caught one wiring gap the
  unit test's monkeypatch had masked (report.run ignored `report_source`);
  fixed and re-verified.

**Source scans.** No run participant references products/ (AST-scan test);
no unresolved brace-format placeholders in docs (scan test); no uncontrolled
wall-clock reads in blocks/stages (existing scan test); production reads
nothing from output/backtests/ (the frozen asset is the only chain-error
input; the research runner WRITES its artifacts there, which is its job).
`git diff --check` clean.

**Open limitations after this session.**
1. The locked-environment suite is unverified on this machine (uv absent);
   CI is the authority for the wheels.
2. The regime remains pseudo_real_time_final_vintage everywhere; nothing
   here claims genuine real-time performance.
3. The 2023Q1+ window is inspected; all forward-looking claims now rest on
   the prospective window (outcomes from 2026Q2, record frozen 2026-08-04).
4. The tilt backtest producer was reconstructed from the live-rule semantics
   (the original notebook cell no longer exists); its vintage selection
   reproduces the frozen predecessor exactly, but the historical y_untilted
   values differ through the corrected panel, as documented in F1.
5. Carried operator actions: install X13 before the ~2026-08-20 Peru June
   refresh; PBoC-direct M2 loader; dated release calendars; the store_sims
   structural-propagation hook; stale-lag sweep of ten non-required series.

# Final follow-up session (G-series, 2026-08-04)

Same working tree (commit c83b4cf plus the uncommitted audit work).
Baseline suite before this session: 177 passed (local checkout). `uv`
remains unavailable: the locked-wheel path is still CI-only and no local
result below claims otherwise.

## G1 (P0): live nowcasts use the exact run as-of date. CLOSED.

**Finding and root cause.** `nowcast.release_cycle.live_path` filtered a
weekly origin grid (anchored on the expected publication date) to origins no
later than today, and never appended today itself: run 2026-08-04__140712
published the Peru official nowcast from a 2026-07-29 origin and the China
node from 2026-08-01 while claiming as-of 2026-08-04. The as-of information
set contained July releases (tpm 4.25, fed 3.63) the July 29 origin could
not see: the information-set contract was violated even though those series
are not direct nowcast members.

**Failing tests first** (`tests/test_live_asof.py`, 7): terminal origin
equals an as-of between grid points; weekly history preserved, sorted,
unique; a synthetic June observation released between grid points (Aug 2)
moves ONLY the terminal-origin nowcast; no duplicate when the as-of sits on
the grid; a target still unpublished after its expected publication date
raises a stale-target error instead of producing positive-lead forecasts;
`official_from` refuses an origin/as-of mismatch; the shared
`require_exact_origin` helper.

**Implementation.** `live_path`: origins are now the union of the weekly
grid and the normalized today (sorted, de-duplicated), with a fail-closed
stale-target guard when today passes the expected publication;
`require_exact_origin` is enforced in `official_from` (Peru) and
`_china_model.live_profile` (China), so an old origin can never be
relabelled with a newer as-of. Fixture updates in
`tests/test_nowcast_artifact.py` reflect the contract (artifact origin
equals as-of).

**Before/after at 2026-08-04** (vs run 140712):
- Peru official: origin 2026-07-29 to 2026-08-04, days-to-publication -22
  to -16; value 2.7019 IDENTICAL; information index 0.9173 IDENTICAL;
  realized weights (RW 0.058, Bridge 0.940, P-MIDAS 0.002) IDENTICAL; node
  TPN (s 0.6624, gamma -0.6773) IDENTICAL. No member-relevant release fell
  between the two origins at this date, so the correction is contract-only
  here; the synthetic test proves the value moves when one does.
- China: node 4.30, bin 2, full path byte-identical; same reading.

## G2 (P1): the code fingerprint covers untracked production files. CLOSED.

**Root cause.** `_code_version` hashed `git diff HEAD`, blind to untracked
files: `pipeline/lib/inputs.py`, `publish.py`, `stagegraph.py`,
`china_assets.py` could change without changing the recorded code identity,
and the exact-chain fingerprint inherited the blind spot.

**Failing tests first** (`tests/test_code_fingerprint.py`, 6, on a temporary
git repository): clean-tree stability; tracked edits; staged changes;
untracked additions AND edits; .gitignore exclusion; creation-order
determinism.

**Implementation.** `dirty_inventory()` walks `git status --porcelain
-uall` (which respects .gitignore) and hashes every deviating file's
content; `_code_version` digests the sorted (path, sha) inventory, so
staged, unstaged and untracked non-ignored files all enter the identity.
The run manifest records the inventory (`code_inventory`) so two differing
fingerprints are diagnosable path by path. The exact-chain fingerprint
consumes `_code_version` and therefore the corrected identity automatically;
the ignored source loaders remain covered by `sources_code_sha` and frozen
assets by their manifest hashes. Documented: a hash detects difference but
does not reconstruct an uncommitted tree; a clean commit remains the
preferred operator condition for an external release (nothing was committed
in this task). OPERATOR NOTE: the stored exact-chain artifact carries the
OLD-format code version; the next resume attempt will refuse until the
artifact is deliberately superseded, which is the designed behavior.

## G3 (P1): inspected and prospective samples are separated. CLOSED.

`sample_label` now returns `selection` (through 2022Q4),
`inspected_post_selection` (2023Q1 to 2026Q1) and `prospective` (2026Q2
onward, outcomes unseen at the 2026-08-04 freeze; new rows land there by
construction and cannot merge into the inspected sample). `HOLDOUT_START`
survives as a HISTORICAL boundary with an explicit non-holdout docstring.
`benchmarks.nowcast_scoreboard` loops over the three labels plus the `all`
aggregate row (empty prospective samples are skipped, never mislabelled).
Boundary tests at 2022Q4, 2023Q1, 2026Q1, 2026Q2 and a far-future quarter.
`output/backtests/nowcast_benchmarks.csv` was regenerated through the full
combination path and reproduces every prior value exactly under the new
labels (Adaptive-IC 0.809/0.631/0.856; Mean 0.815/0.457/0.895);
`docs/audit/phase3_evaluation.md` no longer calls the inspected period a
holdout anywhere.

## G4 (P1): publication is transactional, versioned and hash-verified. CLOSED.

**Consumer survey.** No production or research code READS the flat
`products/` paths (docstring mentions only); `products/` also contains the
assembly package's SOURCE files, which publication must never touch. The
authoritative surface is therefore `products/published/<run_id>/` plus one
`products/latest` pointer; the flat generated paths are DEPRECATED, frozen
at their last flat publication, and documented in `products/PUBLISHING.md`.

**Failing tests first** (`tests/test_publish_versioned.py`, 8, plus the
deprecation test in `tests/test_run_surface.py`): success publishes the
versioned directory, writes the publication manifest (run id, as-of, code
version, per-file sha256) and switches the pointer only afterwards;
unpromoted runs refuse; a sha mismatch against the promoted manifest
refuses and leaves the previous pointer; a missing declared artifact
refuses; an injected copy failure leaves the previous `latest` and no
staging residue; a new release inherits nothing from an older one;
republishing a run id refuses; publication writes NOTHING at the products
root (source files and frozen flat copies untouched).

**Implementation.** `publish_run` stages into
`products/published/.tmp-<run_id>`, copies and verifies every declared
artifact (size and sha256 against the promoted run manifest), writes
`publication_manifest.json`, commits with one atomic rename, then swaps the
`latest` symlink atomically; any failure removes the staging directory and
leaves the pointer untouched; an empty surface refuses.

## G5 (P1): the report publishes the provisional-calibration qualification. CLOSED.

`pipeline/stages/report.py:calibration_disclosure()` is the ONE source for
both outputs, and reads `PROSPECTIVE_START` and the regime label from
`core.evaluation` (a monkeypatch test proves no hardcoded literal). The
wording: "Fan calibration is provisional: horizon-specific coverage is
estimated from 9 to 18 pseudo-real-time observations on final-vintage data
(pseudo_real_time_final_vintage). The post-2023 evaluation window has been
inspected; prospective validation begins with the 2026Q2 GDP release." It
lands in report.md (a Disclosures section, next to the retained final-
vintage regime disclosure) and in the TeX/PDF through the information
paragraph token. The template's two unqualified "real-time errors" claims
now read "pseudo-real-time forecast errors under release-date rules", with
a test forbidding any unqualified real-time line in the template. PDF mode
still fails closed without a compiler (pinned in
tests/test_artifact_contract.py).

## G-series validation and production acceptance

**Suites.** Local checkout (MIDAS 0.2.3 / MacroPy 0.1.10 source paths):
177 to **200 passed** (23 new tests across 5 files; fixture updates in
test_nowcast_artifact, test_run_surface, test_smoke_pipeline,
test_evaluation reflect the new contracts). The LOCKED environment remains
unavailable on this machine (`uv` not installed); locked-wheel results come
only from CI and no local run is presented as locked proof.
`git diff --check` clean.

**Production run `2026-08-04__final`** (as-of 2026-08-04, all stages,
publication disabled during the run, then exercised as its own step). All
twelve acceptance checks PASSED:

1. every configured stage `ok`; 2. all artifacts unique, in-tree, hashed,
required set present; 3. 18 pinned inputs verified through success;
4. Peru official origin 2026-08-04 == as-of (was 2026-07-29 in 140712);
5. China live origin 2026-08-04 == as-of (was 2026-08-01); 6. official
2.7019 equals fan node 1 (2.702) under the declared 1e-3 storage-rounding
contract; 7. fan node 1 TPN metadata equals the official artifact; 8. one
as-of across manifest, bundle, blocks, fan, official artifact; 9. report.md
and report.tex both carry the final-vintage AND provisional-calibration
disclosures, report.pdf compiled; 10. the `products/latest` pointer switched
only on the successful publication (failure paths pinned by tests);
11. `products/published/2026-08-04__final/` contains exactly the declared
surface (19 verified files plus the publication manifest); 12. the prior
promoted run and the deprecated flat surface are byte-untouched.

**Numerical before/after (final run vs 2026-08-04__140712).** Peru fan and
China path are BYTE-IDENTICAL (max mode, width and band-parameter deltas
0.0000) while the origin metadata corrected from 2026-07-29/2026-08-01 to
2026-08-04: the G1 violation was real but carried zero numerical distortion
at this date, exactly as the focused before/after predicted.

**Unresolved limitations.** Locked-environment verification is CI-only;
everything remains pseudo_real_time_final_vintage; the fan calibration is
provisional pending the prospective window (now publicly disclosed in the
report); the stored exact-chain artifact must be superseded before its next
extension (old-format code version in its fingerprint, by design).

**Operator actions.** Unchanged from the follow-up list (X13 before the
~2026-08-20 Peru refresh; PBoC-direct M2; dated calendars; store_sims hook;
stale-lag sweep), plus: point the website at `products/latest/` (the flat
paths are frozen), and prefer a clean commit before the next external
release so the code fingerprint is a bare commit id.

## Status table

| finding | status | evidence |
|---|---|---|
| P0 exact-as-of integrity | closed | tests/test_live_asof.py (7); acceptance checks 4, 5, 6, 7; official origin 2026-08-04; values byte-identical, contract restored |
| P1 code fingerprint | closed | tests/test_code_fingerprint.py (6); manifest `code_inventory`; run banner digest covers untracked modules |
| P1 prospective sample labels | closed | boundary tests in tests/test_evaluation.py; regenerated nowcast_benchmarks.csv (values identical, labels honest); docs relabelled |
| P1 transactional publication | closed | tests/test_publish_versioned.py (8) + deprecation test; acceptance checks 10, 11, 12; publication manifest with 19 verified hashes |
| P1 public fan disclosure | closed | tests/test_report_disclosure.py (4); acceptance check 9: disclosure present in report.md and report.tex/pdf, sourced from core.evaluation |
