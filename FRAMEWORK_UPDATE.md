# FRAMEWORK_UPDATE.md

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
