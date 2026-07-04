# FRAMEWORK_UPDATE.md

## 2026-07-04 C2C S0 Semantic Enrichment Index Rebuild

- Fixed S0 C2C semantic enrichment ordering: enriched `code_chunks` now rebuild `implementation_surface_map` and `code_retrieval_index` before artifacts/static bundle are written.
- `retrieve_code_chunks()` now scores `retrieval_keywords`, `mechanism_tags`, and `semantic_summary`, so semantic enrichment affects precomputed retrieval results.
- Code retrieval results and implementation surface items now carry semantic fields forward for S1/S2 consumers.
- Added regression coverage proving enriched code semantics appear in static bundle, surface map, retrieval index match reasons, and S0 gate still passes.

## 2026-07-04 C2C S1 Discriminating Evidence And Direction Scorecard

- Upgraded S1a evidence request plans with competing `candidate_direction_hypotheses`, `uncertainty_axes`, `discriminating_evidence_requests`, and `must_have_before_direction`.
- S1a still cannot choose a direction or emit evidence bundles; the new fields only make deterministic S1b retrieval ask sharper, direction-separating questions.
- Added deterministic S1b code-neighborhood expansion from parsed code edges/same-file chunks, with `code_neighborhood_expansions` and coverage contributors recorded in the retrieval trace.
- Added `literature/c2c/direction_candidate_scorecard.json` so S1c records candidate directions, selected direction, scores, evidence/counterevidence refs, and `why_not_selected` for alternatives.
- S1 gate, stage contracts, and C2C artifact audit now require/schema-check the new direction candidate scorecard.
- Added regression coverage for discriminating request validation, code-neighborhood expansion, C2C S1 follow-up, C2C pipeline artifact creation, stage contracts, and validators.

## 2026-07-04 C2C S1 Shared Evidence/Direction Session

- Changed C2C S1 two-phase defaults so `evidence_request_agent` and `direction_agent` share one Codex resume session by default.
- S1b remains deterministic and non-GPT: it inserts the retrieved evidence bundle between the two Codex turns.
- Updated the S1c prompt to explicitly continue the same evidence-on-demand session while still forbidding refs outside the deterministic bundle.
- Added a same-session S1c follow-up evidence loop: when S1c returns `status=needs_more_evidence`, the system runs deterministic retrieval for the requested extra cards, merges them into the bundle, and resumes S1c again.
- Added regression coverage that the S1c direction-agent call uses `codex resume` after S1a and after a follow-up retrieval.

## 2026-07-04 C2C Real Smoke Record And Replay Hardening

- Added `meta/c2c_real_smoke_record.json` plus schema/report/validator coverage.
- Ran the first real `c2c_real_smoke_001` smoke through readiness, real S1/S2/S2.5/S3 hooks, audit, replay, and report.
- Fixed real-run blockers exposed by the smoke: S1c direction payload normalization, S1 code evidence coverage, C2C S2 expected-file surface filtering, S3 proxy artifact manifest registration, route decision archival, replay-from-stage isolation, and same-stage audit stale handling.
- Final smoke status: readiness `warn`, S3 proxy decision `proxy_repairable`, route `route_to_s2`, audit `pass`, replay `match`, final pipeline status `failed` at S2 planner gate after the S3 route.

See `FRAMEWORK_STAGE_UPDATES.md` for the full implementation and validation log.

## 2026-07-04 C2C Proxy Repair Routing And Adaptive Selector Guard

- Reclassified `proxy_repairable` / `effect_first_proxy_repair` from method-level proxy failure to S2.5 patch-only repair, even when a legacy proxy decision still carries `route_hint=return_s2`.
- Prevented repairable proxy events from consuming same-direction proxy failure budget or writing method memory.
- Updated S2 feedback aggregation to normalize repairable proxy failures as implementation failures, recompute counters from attempt records when present, and dedupe the same failure event across route/proxy/main-results/performance/ledger sources.
- Allowed implementation-repair mode to reuse the same integration point/fingerprint without being blocked by adaptive force-new-integration constraints.
- Aligned the default C2C same-direction method-level proxy failure budget to 5 attempts across project config, route policy fallback, and S2 adaptive feedback context.
- Added regression tests for repairable proxy routing, attempt ledger budget accounting, S2 feedback dedupe, and implementation repair gate behavior.
