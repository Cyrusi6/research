# FRAMEWORK_UPDATE.md

## 2026-07-04 C2C Real Smoke Record And Replay Hardening

- Added `meta/c2c_real_smoke_record.json` plus schema/report/validator coverage.
- Ran the first real `c2c_real_smoke_001` smoke through readiness, real S1/S2/S2.5/S3 hooks, audit, replay, and report.
- Fixed real-run blockers exposed by the smoke: S1c direction payload normalization, S1 code evidence coverage, C2C S2 expected-file surface filtering, S3 proxy artifact manifest registration, route decision archival, replay-from-stage isolation, and same-stage audit stale handling.
- Final smoke status: readiness `warn`, S3 proxy decision `proxy_repairable`, route `route_to_s2`, audit `pass`, replay `match`, final pipeline status `failed` at S2 planner gate after the S3 route.

See `FRAMEWORK_STAGE_UPDATES.md` for the full implementation and validation log.
