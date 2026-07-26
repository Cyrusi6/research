# FRAMEWORK_UPDATE.md

## 2026-07-26 M1.1.5.3.1 Frozen Decoder, Exact-Set & Recovery Acceptance Closure

- Promoted the breaking runtime authority to Event/AttemptRecord/ResearchState v10 and TrialSpec v9. PhaseCommandPlan v4, PhaseCommand v5, EvidenceDerivationPlan v2, EvidenceDerivationManifest v3, DecoderDescriptor v2, and ReadinessCheckPlan v2 carry the changed authority shapes; TrialResult v6, PhaseRunReceipt v5, EvidenceManifest v6, ActivationEvidence v4, FullS3Readiness v4, ProxyDecisionPolicy v2, and ProxyOutcome v4 remain structurally unchanged.
- Replaced descriptor-only decoder identity with `DecoderProgram v1` inside a content-addressed `DecoderImplementationBundle v1`. The bundle freezes the real declarative transformation program, runtime ABI, entrypoint, dependencies, canonical JSON rules, measurement mapping, numeric rules, pairing, activation/readiness semantics, and output identity. Historical replay executes this immutable program and never consults the current decoder registry or source file for semantics.
- Moved the producing decoder behind `journal.run_once()`. A new derive command starts before the decoder runs; completed replay returns the historical receipt; durable-receipt recovery performs semantic validation and commits only the missing Completed event; Started without a trustworthy receipt commits one replayable `BLOCK_INTEGRITY` route instead of silently rederiving.
- Added one pure semantic validator for derive-receipt precommit, EvidenceManifest binding, reducer/rebuild, state/query, replay, and S3 Gate. It recomputes normalized bytes from the frozen plan, immutable decoder artifact, exact completed physical receipts, and raw outputs before `PhaseCommandCompleted`, so a hash-consistent but scientifically wrong derivation cannot enter the ledger.
- Froze explicit physical authority roles and readiness check IDs. Readiness and activation inputs are ordered exact sets: missing, extra, duplicate, reordered, cross-Attempt, cross-phase, wrong-generation, wrong-producer, wrong-command/output, or wrong-authority inputs fail closed. Synthetic activation now emits explicit observed measurements rather than copying expected surfaces.
- Added M1.1.5.3.1 P0, hash-consistent attack, and seven-point derive/readiness crash suites under `tests/test_m11531_*.py`. The tests distinguish producing decoder invocations from validator recomputation and verify stable receipts, manifests, routes, budgets, and read-only snapshots across cold restart.
- Collected 781 tests and completed all four local full-suite environments with `779 passed, 2 skipped, 0 failed`: normal (`1:05:20`), proxy-cleared (`1:05:19`), empty HOME/HF cache with PATH excluding Codex (`1:04:14`), and CPU-only with PATH excluding `nvidia-smi` (`1:04:19`). The two skips are the unchanged optional torch/transformers forward probes; no skip or xfail was added.
- Commit `1fd4e84` established the M1.1.5.3 derivation/readiness migration chain but is not the final acceptance point. Production-component local subprocess tests remain engineering authority checks only; Native Unified S1, real external Codex S1, M2, real GPU training, and scientific metric experiments remain unstarted.

## 2026-07-26 M1.1.5.3 Immutable Derivation & Readiness Authority Closure

- Raised the breaking authority contracts to Event/AttemptRecord/ResearchState v9 and TrialSpec v8. TrialResult v6, PhaseCommandPlan v3, PhaseCommand v4, PhaseRunReceipt v5, EvidenceManifest v6, EvidenceDerivationManifest v2, ActivationEvidence v4, FullS3Readiness v4, ProxyDecisionPolicy v2, and ProxyOutcome v4 reflect the changed authoritative shapes. `EvidenceDerivationPlan v1`, `DecoderDescriptor v1`, and `ReadinessCheckPlan v1` are new frozen contracts; replaced schemas/readers are deleted with no dual read, fallback, or automatic migration.
- Established one immutable phase derivation chain: frozen EvidenceDerivationPlan → completed physical command receipts/raw ContractRefs → constrained Core decoder → one EvidenceDerivationManifest v2 → the derive PhaseRunReceipt v5 `derivation_ref/hash` → the exact same reference in EvidenceManifest v6 → ProxyOutcome or TrialResult. The former direct/self derivation sourced from the derive command's own normalized outputs is removed.
- Made immutable validation strictly read-only. Precommit, Ledger transaction, reducer/rebuild, state/query, restart replay, and S3 Gate use the same validator to reread frozen decoder and physical raw bytes, deterministically recompute normalized evidence in memory, and compare every manifest, receipt, output reference, hash, and byte sequence. Missing or damaged authority fails closed without CAS/SQLite writes, blob reconstruction, source-file fallback, or command rerun.
- Froze receipt-level activation and readiness semantics. Exit code zero proves command completion only; ActivationEvidence v4 is recomputed from enabled/disabled measurements, observed surfaces, frozen coverage, and delta threshold. FullS3Readiness v4 is derived from ReadinessCheckPlan v1 predicates and exact physical receipt bindings. Expected surfaces cannot self-certify observed coverage, and producer-authored `ready`, PASS lists, thresholds, summaries, or decisions have no authority.
- Closed deterministic semantic blocking: valid activation or readiness `BLOCKED` maps to `REPAIR_IMPLEMENTATION`, retains the Attempt reservation, starts no full subprocess, creates no method-history/result entry, and consumes no direction outcome. Missing, extra, duplicate, stale, cross-Attempt, or corrupted derivation/readiness evidence remains an integrity failure; only a committed reducer-owned `RUN_FULL` authorizes full execution.
- Added immutable-derivation, read-only audit, exact physical-source-set, cold-restart, orphan-blob, activation-surface, readiness BLOCKED/replay, and zero-write inventory attacks under `tests/test_m1153_derivation_authority.py`, `tests/test_m1153_derivation_recovery.py`, and `tests/test_m1153_readiness_authority.py`; the final combined attack group reports `42 passed`.
- Completed the four-environment full acceptance with `747 passed, 2 skipped, 0 failed` in each environment: normal (`44:35`), proxy-cleared (`43:32`), empty HOME/HF cache with PATH excluding Codex (`44:23`), and CPU-only with PATH excluding `nvidia-smi` (`43:49`). The two skips are unchanged optional torch/transformers checks.
- Production-component checks use real local subprocess fixtures through the production executors, frozen commands, receipts, CAS, Core derivation, Ledger/rebuild, and Gate. Synthetic checks are explicitly separate. Native Unified S1 Producer/Core, real external Codex S1 smoke, real Codex availability, and real GPU scientific training are not part of this milestone and are not claimed as completed.

## 2026-07-16 M1.1.5.2 Raw Execution Provenance and Recovery Closure

- Completed the breaking authority migration to Event/AttemptRecord/ResearchState v8 and TrialSpec v7. PhaseExecutionManifest v3 remains current; PhaseCommandPlan v2, PhaseCommand v3, PhaseRunReceipt v4, EvidenceManifest v5, EvidenceDerivationManifest v1, SampleManifest v4, FailureEvidence v6, CoreResourceProbeReceipt v2, and C2CCheckpointManifest v1 close the new raw-execution boundary. CompletionEvidence v3, ResumeEvidence v5, ResourceProbe v4, TrialResult v5, and RouteOutcome v4 retain their existing shapes. Replaced schemas/readers are deleted rather than dual-read or migrated, so older workspaces must restart from S1.
- Physical commands now execute from the frozen typed contract: exact `argv[]`, `cwd`, environment overrides, inherited-environment allowlist, source snapshot, dependencies, conditions, expected outputs, and retry/resource/resume policies. `ExperimentRunner` uses `subprocess.Popen(..., shell=False, env=...)`; display strings never participate in execution.
- Scientific provenance now follows `authorized command -> immutable raw outputs in PhaseRunReceipt v4 -> EvidenceDerivationManifest v1 -> normalized ContractRefs -> EvidenceManifest v5 -> classifier/finalization`. Mutable result directories, caller-authored observations, and post-hoc aggregate relabeling cannot create TrialResult, budget, route, or aggregate authority.
- Real SampleManifest v4 entries bind the selected sample bytes and derive ordered sample identities from those bytes. Receipt-derived failure classification and constrained resource measurements replace caller-authored command-result or resource receipts. Missing GPU tooling or insufficient capacity atomically routes `PAUSE_RESOURCE`, leaves the Attempt recoverable, preserves its reservation, and consumes no outcome.
- Closed all seven restart boundaries: Started before receipt; durable receipt before Completed; Completed before evidence commit; ProxyEvidenceCommitted before FullPhaseStarted; FullPhaseStarted before the first full command; completed full commands before AttemptFinalized; and AttemptFinalized before RouteOutcome delivery. Recovery reuses durable receipts/events and does not repeat subprocess side effects or budget consumption.
- The normal, proxy-cleared, empty-HOME/empty-HF-cache/PATH-without-Codex, and CPU-only/PATH-without-`nvidia-smi` full suites each report `705 passed, 2 skipped, 0 failed`. The two skips remain the pre-existing optional torch/transformers dynamic skips.
- Non-simulated production-component checks execute real local subprocesses through frozen commands, receipts, CAS, decoders, Ledger, rebuild, and Gate. Synthetic runs prove deterministic state and budget semantics only. Neither category is real GPU scientific training or a scientific-success claim; Native Unified S1 Producer/Core and real external Codex S1 smoke remain unstarted.

## 2026-07-15 M1.1.5.1 Production Phase Authority Closure

- Raised the breaking authority contracts to Event/AttemptRecord/ResearchState v7 and TrialSpec v6. PhaseExecutionManifest v3, PhaseCommand v2, PhaseRunReceipt v3, EvidenceManifest v4, SampleManifest v3, CompletionEvidence v3, FailureEvidence/ResumeEvidence v5, and ResourceProbe v4 are the current phase-authority contracts; old workspaces must restart from S1.
- Wired the production phase boundary around four explicit executors: `C2CProxyPhaseExecutor`, `C2CFullPhaseExecutor`, `GenericExternalPhaseExecutor`, and `SyntheticPhaseExecutor`. Adapter callbacks execute only while holding a SQLite-derived executor capability, and authority is rechecked before and after the callback.
- Added content-addressed `PhaseCommandPlan v1`. Each command freezes its spec ID, phase, ordinal/dependencies, exact argv, safe cwd, source snapshot, expected outputs, retry/resume policy, and condition. `PhaseCommandStarted` verifies the command against the current plan hash rather than accepting a merely self-consistent runtime command.
- Upgraded the command journal so `PhaseRunReceipt v3` binds the Started event, command plan, Attempt generation, phase execution, producer, exit status, durable stdout/stderr ContractRefs, and exact immutable output ContractRefs. Completed replay reconstructs command results and inventories from the receipt instead of process-local memory.
- Added receipt-bound evidence lineage. Authoritative evidence must be a declared immutable output of a completed current-phase command. For current adapters the frozen `collect-evidence` command is the deterministic decoder node: its DAG dependencies require the physical measurement/probe commands to be completed first, and its immutable outputs are the only bytes accepted by proxy commit or finalization. Orphan, stale-generation, cross-Attempt, wrong-phase, extra, duplicate, or unregistered evidence is rejected.
- Preserved the physical proxy barrier: proxy commands and immutable evidence commit before reducer `RUN_FULL`; only then may `FullPhaseStarted` and full commands occur. Proxy rejection creates no full command or full artifact and consumes no standard method outcome.
- Kept crash recovery fail closed: durable receipt before DB completion is reconciled without rerunning; completed commands replay from immutable receipt outputs; Started without a trustworthy receipt or attachable external job becomes unknown outcome; finalization remains idempotent after DB commit/projection loss.
- Resource pause/resume is specified through receipt-bound FailureEvidence v5, ResumeEvidence v5, and ResourceProbe v4 while preserving Attempt identity and reservation. The remaining production black-box resource-resume/OOM recovery verification is still in progress and is not reported as accepted.
- This entry records the current M1.1.5.1 implementation state, not final milestone acceptance. Native Unified S1 Producer/Core, real external Codex S1 smoke, and real GPU scientific training have not started; local subprocess and synthetic runs are engineering checks only.

## 2026-07-14 M1.1.2 Authoritative Attempt / S3 Transaction Closure

- Upgraded the breaking authority layer to Event v5, AttemptRecord v5, TrialSpec v4, ExecutionObservation v4, TrialResult v5, EvidenceManifest v3, ConstraintResult v2, and RouteOutcome v4; prior workspaces are rejected and must restart from S1.
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

## 2026-07-14 M1.1.3 Scientific Evidence Provenance Closure

- Upgraded Event, Attempt, and State to v4; TrialSpec to v3; TrialResult to v4; ExecutionObservation to v3; EvidenceManifest and ConstraintResult to v2.
- Replaced caller-authored TrialResult finalization with CompletionEvidence containing only the current Attempt's explicit staged inventory.
- Added immutable attempt-scoped content-addressed evidence paths and strict schemas for quantitative rows, activation, proxy policies/reports, readiness, bootstrap completion, failure, and resource/resume probes.
- Ledger finalization now reads each evidence artifact once, verifies bytes/hash/schema/identity, deterministically decodes observations and paired constraints, and commits result/budget/route/aggregate atomically.
- Removed fixed-path evidence discovery and arbitrary-artifact observation binding; stale files from another Attempt cannot authorize a result.
- Made non-simulated Codex CLI absence a typed provider-unavailable failure and kept simulation explicitly synthetic.

### M1.1.3 final migration closure

- Completion finalization stores a fingerprint binding Attempt/generation, frozen TrialSpec, implementation/input identity, producer runs, EvidenceManifest, and immutable artifact hashes. Exact replay returns the historical operation result without writes; conflicting, missing, or corrupted evidence fails with zero state/budget change.
- Bootstrap finalization and S3 Gate now use only committed Attempt-scoped proxy evidence. Repeated Gate audits are read-only; bootstrap leaves the standard budget at `target=5, reserved=0, consumed=0`, does not enter method history, and creates no direction aggregate.
- Generic and C2C simulated standard pipelines each commit five unique semantic variants from strict row evidence, finish with `target=5, reserved=0, consumed=5`, and reject a sixth variant before execution.
- Historical M1.1.3 checkpoint validation recorded `537 passed, 2 skipped` in the normal, proxy-cleared, and empty-HOME/no-Codex environments. This count predates the breaking Event v6/TrialSpec v5 migration and is not M1.1.5-Final acceptance.

## 2026-07-14 M1.1.4 Authoritative Phase Transactions

- Upgraded the breaking authority layer to Event v5, AttemptRecord v5, ResearchState v5, TrialSpec v4, EvidenceManifest v3, ExecutionObservation v4, TrialResult v5, RouteOutcome v4, FailureEvidence v3, and ResumeEvidence v3. Replaced schema readers were removed; older workspaces must restart from S1.
- Added authoritative `ProxyPhaseStarted`, `ProxyEvidenceCommitted`, and `FullPhaseStarted` transactions. The reducer derives proxy delta, `ProxyOutcome`, route, phase state, and full eligibility; callers cannot author proxy decisions or enter completed/full states through generic transitions.
- Bound every evidence artifact and quantitative row to lifecycle generation, implementation/input hashes, phase execution ID, phase-start event, and producer run. Repair/resume invalidates uncommitted prior-generation evidence.
- Real reservations now verify both sample and evaluator manifests from content-addressed bytes; evaluator source/config/dependency identity cannot be replaced by a caller-supplied digest after planning.
- Unified immutable evidence reads through the fail-closed `EvidenceStore`, including ancestor/leaf symlink rejection, content-addressed exact paths, single-read hashing/decoding, and atomic exclusive writes.
- Added explicit non-simulated external-manifest execution for Generic projects and strict current-run C2C evidence inventories. Bootstrap now emits proxy-only evidence and finishes without standard budget/history/aggregate effects.
- Native Unified S1 remains intentionally out of scope; local fake commands validate production components but are not reported as real scientific or GPU results.

The M1.1.4 entries above describe the migration checkpoint that introduced phase-scoped evidence. M1.1.5-Final replaces its Event/Attempt/State/TrialSpec and phase-execution contracts; current acceptance must be read against the section below rather than the earlier checkpoint versions.

## 2026-07-15 M1.1.5-Final Authoritative Integration

- Upgraded the sole runtime authority to Event v6, AttemptRecord v6, ResearchState v6, and TrialSpec v5. TrialResult v5, ExecutionObservation v4, EvidenceManifest v3, and RouteOutcome v4 remain current because their shapes did not require a mechanical version bump.
- Split preregistered proxy science from execution identity. `ProxyDecisionPolicy v1` is frozen in TrialSpec and contains metrics, objective, paired aggregation, exact coverage, thresholds, activation/readiness requirements, evidence kinds, mode, and route semantics. `ProxyEvaluationBinding v1` is generated by `ProxyPhaseStarted` and binds that policy to the current Attempt generation, implementation/input hashes, phase execution, producer, command plan, ContractRefs, and provenance.
- Connected one pure proxy classifier to precommit, `ProxyEvidenceCommitted`, reducer/rebuild, and Gate audit. Producer effective/calibration policy, threshold, decision, constraints, summary, and route are diagnostic only and cannot enter the authoritative evidence set.
- Enforced the physical barrier `ProxyPhaseStarted → proxy command events → ProxyEvidenceCommitted → RUN_FULL → FullPhaseStarted → full command events → AttemptFinalized`. A proxy science rejection commits `PROPOSE_NEXT_VARIANT`, releases the reservation, consumes no outcome, and never invokes full execution.
- Integrated `ResearchLedgerPhaseAuthority` with the production phase executors. Every side-effecting command rechecks an exact SQLite-derived authorization; `None`, booleans, caller mappings, stale contexts, and forged full authorization fail closed before runner invocation.
- Replaced mutable sample/evaluator authority with content-addressed `ContractRef v1`, `SampleManifest v2`, and `EvaluatorManifest v2`. Reservation verifies immutable bytes through `ContractStore`; mutable plan projections and self-declared digest strings are not accepted as provenance.
- Moved command lifecycle into Event v6 through `PhaseCommandStarted`, `PhaseCommandCompleted`, and `PhaseCommandUnknownOutcome`. `PhaseRunReceipt v2` binds the Started event, command, phase/generation, output hashes, external job, and provenance. A durable receipt can be reconciled without rerunning; an unprovable started command becomes unknown outcome rather than being repeated.
- Unified Failure/Resume v4 validation across public transaction, reducer/rebuild, and audit. Implementation/activation failures require a current nonzero-exit command result; pause requires an insufficient resource probe; resume requires an available matching probe and the committed pause event.
- Defined recovery semantics: ordinary process crashes preserve generation and phase execution; implementation repair invalidates proxy authority and reruns proxy; proxy resource resume resets proxy; full resource resume preserves the committed proxy outcome and resumes from `PROXY_COMPLETED` with full pending.
- Removed runtime readers for Event v5, AttemptRecord v5, ResearchState v5, TrialSpec v4, PhaseExecutionManifest v1, FailureEvidence v3, ResumeEvidence v3, ResourceProbe v2, ProxyDecisionContract v1, ProxyOutcome v1/v2, EffectiveProxyPolicy v3, and ProxyCalibrationPolicy v3. No dual-read, fallback, or migration path is provided.

### Historical M1.1.5-Final acceptance checkpoint

- The pre-v7 M1.1.5-Final checkpoint recorded three green environments. That historical count does not validate the breaking M1.1.5.1 Event v7 / TrialSpec v6 migration; current results must be reported only in the final delivery.
- The empty-cache run initially exposed four C2C production-component tests that were silently borrowing the host Hugging Face dataset cache. Their fixture now creates an explicit temporary fake cache before Agent construction; production preflight remains strict and unchanged.
- Non-simulated local subprocess coverage proves physical proxy/full ordering, immutable phase evidence, bootstrap isolation, five unique standard outcomes, and sixth-attempt rejection. These commands are test executables, not GPU training or scientific success claims.
- Native Unified S1 Producer/Core and real external Codex S1 smoke remain outside this milestone.

## 2026-07-16 M1.1.5.1 Production Phase Authority Documentation Correction

- ExperimentAgent production phase dispatch is defined through exactly four executors: `C2CProxyPhaseExecutor`, `C2CFullPhaseExecutor`, `GenericExternalPhaseExecutor`, and `SyntheticPhaseExecutor`. Internal callbacks are executor-owned adapter hooks, not alternate phase authorities.
- `PhaseCommandPlan v1` freezes the exact command DAG. `PhaseCommandStarted` is valid only when command spec ID, phase, ordering/dependencies, argv, cwd, source snapshot, expected outputs, policies, and command-plan hash match the frozen entry.
- Event v7 SQLite remains the sole authority for phase authorization, command lifecycle, replay, budget, route, and result state. JSON state, Attempt, TrialResult, RouteOutcome, proxy outcome, and aggregate files are rebuildable projections.
- `PhaseCommand v2` and `PhaseRunReceipt v3` bind started/completed command events to durable stdout, stderr, and declared immutable output references. Completed-command replay reconstructs from receipt references rather than process-local holders; an unproven started command is not silently rerun.
- `EvidenceManifest v4` enforces direct linkage to a completed current-Attempt/current-generation/current-phase receipt output. Current Generic/C2C adapters use the frozen collector command as the deterministic decoder node, with command-DAG dependencies and current-generation receipt checks closing the upstream physical-command prerequisite.
- Bootstrap remains proxy-terminal and isolated from the standard five-outcome budget/history/aggregate. Standard remains sequential, project-wide `execution_width=1`, and consumes budget only through a verified committed method result.
- Current breaking versions are Event/AttemptRecord/ResearchState v7, TrialSpec v6, PhaseExecutionManifest v3, PhaseCommand v2, PhaseRunReceipt v3, EvidenceManifest v4, SampleManifest v3, CompletionEvidence v3, FailureEvidence/ResumeEvidence v5, and ResourceProbe v4. Replaced v6/v5/v4/v3 readers and schemas are removed from runtime; there is no dual read, fallback, or automatic migration.
- Final M1.1.5.1 acceptance: normal, proxy-cleared, and empty-HOME/empty-HF-cache/PATH-without-Codex full suites each report `669 passed, 2 skipped, 0 failed`. The two skips are the existing optional torch/transformers skips. Native Unified S1 Producer/Core, real external Codex S1 smoke, and real GPU scientific training have not started.
- Generic/C2C local subprocess paths validate non-simulated production-component authority and evidence wiring. Synthetic five-variant paths remain explicitly synthetic. Neither path is represented as real GPU training or scientific success.
- Native Unified S1 Producer/Core and real external Codex S1 smoke remain outside this milestone.
