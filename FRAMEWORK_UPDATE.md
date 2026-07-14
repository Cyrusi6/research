# FRAMEWORK_UPDATE.md

## 2026-07-14 M1.1.2 Authoritative Attempt / S3 Transaction Closure

- Upgraded the breaking authority layer to Event v3, AttemptRecord v3, TrialSpec v2, ExecutionObservation v2, TrialResult v3, and RouteOutcome v3; prior workspaces are rejected and must restart from S1.
- `reserve_attempt()` now validates and freezes the complete TrialSpec, then derives protocol, sample-manifest, acceptance-contract, and attempt-input hashes inside the Ledger transaction.
- Public attempt transitions can only advance execution states. Structured FailureEvidence exclusively derives implementation repair, resource pause, and integrity block; ResumeEvidence exclusively restores `RESOURCE_PAUSED -> READY` without changing attempt or variant identity.
- S3 precommit, Ledger commit, and postcommit Gate share one validator over frozen TrialSpec, exact observation/result artifacts, activation/proxy/readiness evidence, roles, seeds, phases, constraints, identity, and budget.
- The deterministic classifier emits ConstraintResult records. Missing required ablation/control/coverage evidence is non-evaluable and zero-write; complete hard-constraint failures are rejected outcomes; only fully passing accepted outcomes can become aggregate best.
- Orchestrator control is read back from the committed SQLite event. Result routes are diagnostic-only and must canonically equal the event-bound RouteOutcome.
- Project-global `execution_width=1`, five-slot budget invariants, lifecycle-scoped idempotency, late replay, corrupt/future projection recovery, and sixth-reservation rejection are reducer invariants.
- Generic and C2C simulated standard runs complete five semantic variants; C2C simulation now emits explicit matched-control, coverage, ablation, activation, and full-readiness evidence rather than relying on mutable summaries.

## 2026-07-14 M1.1 Authoritative State And S3 Transaction Hardening

- Replaced JSON event files with one SQLite-WAL Event v2 store using transactional sequence allocation, strict type-specific payload validation, event/hash-chain verification, and deterministic rebuilds.
- Split direction and variant semantic identities from complete specification hashes; IDs and lineage no longer create fake scientific novelty.
- Constrained the attempt state machine, froze identities/budget fields, retained repair/pause reservations, and required explicit abandonment before release.
- Changed S3 to typed observations, pure pre-commit validation, and one atomic finalization event containing TrialResult, terminal attempt state, budget update, RouteOutcome, and optional five-result aggregate.
- Enforced `consumed + reserved <= 5`, `execution_width=1`, one standard outcome per variant semantic hash, closed-direction rejection, and bootstrap/standard identity separation.
- Added concurrency, tamper, crash-before-projection, idempotency, phase-consistency, bootstrap-to-standard, abandonment, and sixth-outcome adversarial coverage.

## 2026-07-13 C2C Bootstrap GPT-5.6 Terra XHigh

- Configured `c2c_bootstrap_s3_20260713_1` to use `gpt-5.6-terra` with `reasoning_effort: xhigh` for the global LLM client, all configured reasoning agents, S1 evidence/novelty agents, and S2.5 code patching.
- Added explicit `model_reasoning_effort` forwarding to the S1 evidence-direction and S2 directional-planner Codex CLI subprocesses.
- Verified the effective merged project config resolves every configured agent model to `gpt-5.6-terra`, while S1 and code patch reasoning both resolve to `xhigh`.
- Readiness remains `pass` with no warnings or blocking reasons.

## 2026-07-13 Bootstrap Cached-S0-Only Mode

- Added `orchestration.bootstrap.cached_s0_only`, enabled by default for the bootstrap profile.
- Bootstrap now requires a compatible `intake/c2c/static_bundle.json`; if it is absent, stale, or force-refresh is requested, S0 blocks explicitly instead of calling DeepSeek enrichment or MinerU PDF parsing.
- Readiness treats DeepSeek credentials as unnecessary when the cached bundle is present and compatible, and exposes `cached_s0_only_ready` as a blocking environment check.
- The prepared project `c2c_bootstrap_s3_20260713_1` now passes readiness with zero warnings and zero blocking reasons without DeepSeek or MinerU credentials.

## 2026-07-13 C2C Symlinked Dataset Readiness

- Updated the C2C execution-hook dataset probe to traverse symlinked dataset directories while tracking resolved directories to avoid cycles.
- This supports the existing Hugging Face snapshot layout under `/home/lijunsi/projects/KVcache/datasets/c2c`, where dataset names are symbolic links to cache snapshots.
- Added regression coverage proving a valid JSON sample inside a symlinked dataset snapshot satisfies `dataset_one_example_loadable`.
- Created and validated the real bootstrap project `c2c_bootstrap_s3_20260713_1`; readiness now has no blocking reasons and recommends `run_c2c`.

## 2026-07-13 Bootstrap S0-S3 Proxy Path

- Added an explicit `orchestration.profile: bootstrap` mode for the narrow first milestone: complete one `S0 -> S1 -> S2 -> S2.5 -> S3` traversal and obtain a cheap proxy metric.
- Added `run-c2c --profile bootstrap`; it forces one iteration, one candidate, and `stop_after_stage: S3_experiment`. Standard mode remains the default.
- Applied bootstrap behavior as a runtime configuration overlay rather than permanently expanding relaxed values into project config, so switching back to `standard` restores normal gates.
- Kept S1 schemas, bundle-grounded references, must-resolve requests, counterevidence, implementation coverage, and direction contracts strict. Only paper/code evidence-count shortfalls can be recorded as explicit bootstrap quality debt.
- Kept S2/S2.5 patch generation, whitelist/frozen-patch safety, `py_compile`, and targeted tests. Bootstrap disables config-activation hard gating, mechanism self-review, runtime training smoke, wiring smoke, and forward probes.
- Added an S3 proxy-only terminal path: a completed cheap proxy metric records `bootstrap_proxy_complete`, skips activation/readiness/full train/eval/posthoc/failure rerouting, and does not mislabel proxy metrics as full experiment metrics.
- Added `experiment/results/bootstrap_proxy_completion.json` and profile-sensitive S3 stage contracts. Bootstrap still validates S2.5 artifact locks and requires a real proxy mean metric.
- Added regression coverage for CLI/config overlays, profile compatibility, negative cached proxy completion, and S3 bootstrap gate pass/retry behavior.

## 2026-07-10 S1 Evidence Quality And S2 Falsifiability Contracts

- Removed the deterministic S1 retriever's positive score for source-type matches alone; paper/code/rebuttal type is now a filter, and a request requires lexical/query relevance before it can satisfy `must_resolve`.
- Added `source_only_match` rejection traces so irrelevant same-type candidates remain observable without being admitted into the evidence bundle.
- Marked compatibility-generated direction evidence as explicit placeholders and added configurable S1 Gate enforcement through `ideation.contract_quality.reject_placeholder_evidence`.
- Changed C2C evidence quality to require resolved counterevidence and to exclude framework placeholders from counterevidence counts.
- Added explicit novelty quality debt for unavailable or disabled audits and configurable strict enforcement through `ideation.contract_quality.require_novelty_audit`.
- Upgraded newly generated variant contracts to `auto_research_variant_contract_v2` with structured intervention, null/alternative hypotheses, minimum effect, mechanism predictions, falsification conditions, treatment/fixed/nuisance variables, forbidden simultaneous changes, replicate bounds, and an early-stop rule.
- Kept legacy v1 contracts readable while making S2 Gate enforce the additional scientific fields whenever a v2 contract is declared.
- Enabled strict placeholder and novelty enforcement in the repository's default real-run config.
- Added regression coverage for source-only retrieval rejection, placeholder evidence rejection, disabled novelty audits, resolved counterevidence, and v2 falsifiability/variable-control fields.

## 2026-07-05 C2C Smoke Audit And Execution Probe Hardening

- Added stage-aware `audit-c2c --scope completed|up-to-current|full`; default `completed` now skips unreached stages and records `expected_stages` / `skipped_stages`.
- Hardened artifact manifest validation so enabled hash validation fails on missing manifest entries, missing manifest `sha256`, and hash mismatches.
- Added `meta/c2c_execution_hooks_report.json` with cheap real-run probes for env python, target repo importability, eval entrypoint/help, dataset sample readability, output writability, and timeout configuration.
- `doctor-c2c` and real C2C preflight now write execution hooks before readiness; readiness consumes the hooks gate and smoke records expose `execution_hooks_gate`.
- Expanded `smoke-c2c` with bootstrap/override flags for topic, target repo, ref paper/rebuttal, env python, S0 cache behavior, audit scope, and prepare-only mode.
- Added regression tests for audit scope, strict manifest hash, execution hooks, and smoke CLI overrides.

## 2026-07-05 C2C Real Smoke CLI Entry

- Added `auto-research smoke-c2c --project-id <id>` as a one-command real C2C smoke regression entrypoint.
- The command now runs the stable sequence `doctor-c2c -> run-c2c(max_iterations=1, stop_after_stage=S3_experiment) -> audit-c2c -> replay-c2c -> report --json` and rewrites `meta/c2c_real_smoke_record.json` at the end.
- Readiness `fail` now short-circuits before the real run and still writes the final smoke record for debugging.
- Added CLI regression coverage for the readiness-fail short circuit, smoke sequence order, fixed real-smoke run overrides, parser registration, and final record writeback.

## 2026-07-05 C2C S0 Cache Validity And Evidence Brief Hardening

- Added a validity fingerprint to C2C S0 `static_bundle.json` covering reference inputs, editable repo surface, allowed edit policy, baseline/datasets, PDF ingest config, and semantic enrichment config.
- S0 cached bundle reuse now rejects stale bundles when the current input/config fingerprint differs from the saved bundle fingerprint.
- Cached S0 reuse now refreshes and rewrites `static_bundle.json` after merging shared method failure memory, so S1 reads the same updated context as the sidecar artifacts.
- Rebuilt C2C `evidence_brief` on cache reuse and fixed field mapping to preserve `editable_surface`, `protocol_constraints`, retrieval `questions`, and nested follow-up `cross_source_targets`.
- Strengthened the S0 gate so `evidence_brief.json` must contain compact repo surface and retrieval-target context, not just exist.
- Added regression coverage for stale cache rejection, shared-memory writeback into cached bundles, and current evidence brief field mapping.

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
# 2026-07-14 S1-S3 Authoritative Contracts / Five-Variant Reducer Breaking Change

- Added strict `DirectionSpec v2`, `VariantSpec v3`, `AttemptRecord v1`, `TrialResult v1`, and `RouteOutcome v1` contracts with Draft 2020-12 schemas, fixed versions, closed objects, and recomputed identities.
- Replaced split attempt/route counters with an immutable atomic event ledger and deterministic `ResearchEventLedger` reducer. `meta/research_state.json` and per-attempt files are derived snapshots; duplicate event IDs and crash/resume are idempotent.
- Standard profile now reserves and sequentially executes exactly five method-evaluable variants under one direction. Success does not stop outcomes 1–4; implementation/resource failures release reservations; the fifth outcome finishes or starts a new direction and a sixth reservation is rejected.
- Bootstrap remains a single cached-S0 cheap-proxy traversal with `consumes_direction_budget=false` and `bootstrap_proxy_complete`, isolated from the standard budget.
- Feedback attribution now separates planner feedback, implementation repair history, and method tried history. Only `AttemptCompleted` with `method_evaluable=true` enters scientific anti-repeat history.
- S1/S2/S3 producers, consumers, Gates, reporting, C2C replay/audit, and tests now use only `literature/direction.json`, `plan/variant.json`, `experiment/results/trial_result.json`, and unified route outcomes.
- Removed legacy loaders, route policy modules, direction-to-idea conversion, legacy C2C debate selection, old attempt/route facts, and inference from ideas or `plan.yaml`. Old projects must rerun from S1.
- Preserved cached-S0 bootstrap, GPT-5.6 Terra `xhigh` propagation, persistent Codex patch/repair, activation smoke, and C2C proxy/full execution hooks.

Validation:

```text
TMPDIR=$PWD/.tmp uv run pytest -q
326 passed, 2 skipped
```

## 2026-07-14 M1.1.1 Event v2 对抗封口

- Attempt 增加 `lifecycle_generation`。implementation revision 与 resource resume 进入新世代；transition/disposition/finalization 的幂等键绑定世代、implementation/input identity、expected state、operation/failure 与 phase。
- 业务事件只能通过受约束的 Ledger domain API 创建；公开低层 `append()` 仅允许 `AuditMarker`。Disposition/Finalization 事件只存事实，状态、预算、RouteOutcome、方向关闭和 aggregate 全由 reducer 推导。
- late replay 返回原事件对应的 attempt/route/aggregate，不再返回全局 `last_route_outcome`。
- profile × attempt kind × reservation/budget 映射在 schema、domain API、reducer 和 invariant 四层校验；standard 不存在免预算 method outcome。
- Variant 科学语义 hash 排除 ID、lineage、iteration、nonce 与展示 coordinate metadata；只有 intervention/configuration/operations/controls/ablation/surfaces/metric/hypothesis/falsification 等真实方法内容参与去重。
- S3 precommit 与 postcommit Gate 共用纯验证器；Ledger 在 `BEGIN IMMEDIATE` 内再次校验 TrialResult、typed observations、artifact hash、phase/dataset/seed/role coverage、proxy/bootstrap evidence 和预算后才原子提交。
- Projection 写入加独占锁并在锁内重读 SQLite 最新 sequence，阻止乱序写回旧 snapshot。
- 新增 lifecycle replay、伪造事件、profile/kind、语义 nonce、phase、TrialResult 防伪、Gate-before-commit、路由权威、三轮 repair 和并发 execution-width 对抗测试。
