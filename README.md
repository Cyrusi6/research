# Auto-Research System v2

This repository turns the design in `prompts.md` into a runnable Python CLI for a staged idea-to-paper workflow.

## What It Does

- Creates isolated project workspaces with strict artifact management.
- Runs a 5-stage pipeline: literature, planning, experiments, writing, review.
- Downloads paper PDFs into a project-local `references/papers/` directory.
- Tracks every stage output with `stage_manifest.json`.
- Persists pipeline state in `meta/registry.yaml`.
- Supports `init`, `start`, `resume`, `status`, `report`, `doctor-c2c`, `audit-c2c`, `replay-c2c`, `enrich-s0`, `review`, and `catchup`.

## Quick Start

```bash
uv run auto-research init --topic "multimodal retrieval with hard negatives"
uv run auto-research start --project-id <project_id> --simulate
uv run auto-research status --project-id <project_id>
uv run auto-research report --project-id <project_id>
```

`--simulate` makes the pipeline produce deterministic mock results for validation and tests. Without it, the experiment stage refuses to invent results and will block if no executable experiment plan is available.

Optional S0 semantic enrichment runs during C2C S0 intake after deterministic chunking. The default config uses a small DeepSeek sample first (`limit: 6`) so cost and quality can be inspected before switching to full enrichment. It can also be run manually on an existing S0 bundle:

```bash
uv run auto-research enrich-s0 --project-id <project_id> --limit 6 --dry-run
uv run auto-research enrich-s0 --project-id <project_id> --limit 6
```

The command writes `intake/c2c/semantic_enrichment_sample.json` and `.jsonl`, including token usage and projected full-run cost. During S0 intake, enriched chunks keep their deterministic `chunk_id`, `source_path`, and line/section anchors; DeepSeek only adds semantic fields.

For C2C projects, the E2E guardrail commands can be run without restarting the whole loop:

```bash
uv run auto-research doctor-c2c --project-id <project_id>
uv run auto-research audit-c2c --project-id <project_id>
uv run auto-research replay-c2c --project-id <project_id> --from-stage S3_experiment
```

`doctor-c2c` writes readiness and runtime-health reports under `meta/`, `audit-c2c` checks C2C stage artifacts, schemas, manifest hashes, and stale route-invalidated files, and `replay-c2c` rebuilds the deterministic event-derived research state without rerunning S1/S2 LLM calls.

To first prove that a real project can traverse S0-S2.5 and produce one cheap S3 proxy metric, use the explicit bootstrap profile:

```bash
uv run auto-research run-c2c \
  --project-id <project_id> \
  --profile bootstrap
```

Bootstrap forces one S0→S1→S2→S2.5→S3 cheap-proxy traversal and stops after `bootstrap_proxy_complete`. It requires a compatible cached S0 bundle and will block rather than fall back to DeepSeek semantic enrichment or MinerU PDF parsing. Its attempt has `consumes_direction_budget=false`, so it never enters or consumes the standard five-variant loop.

## Authoritative S1-S3 Model

The breaking S1-S3 contract has one canonical artifact for each scientific identity:

- `literature/direction.json`: `DirectionSpec v3`, with separate scientific `direction_semantic_hash` and complete `direction_spec_hash` identities.
- `plan/variant.json`: `VariantSpec v4`, with separate method `variant_semantic_hash` and complete `variant_spec_hash` identities.
- `plan/attempts/<attempt_id>/trial_spec/<trial_spec_hash>.json`: the immutable `TrialSpec v7` projection frozen by the reservation; `plan/trial_spec.json` is diagnostic only.
- `meta/attempts/<attempt_id>.json`: `AttemptRecord v8`, a rebuildable projection reused across crash/resume, resource resume, and implementation repair.
- `experiment/results/trial_result.json`: `TrialResult v5`, the latest committed result deterministically decoded from immutable row-level evidence.
- `meta/route_outcome.json`: `RouteOutcome v4`, the latest deterministic route projection.
- `meta/research_events.sqlite3`: the sole authoritative Event v8 SQLite-WAL store; `ResearchState v8`, attempts, results, routes, phase outcomes, command lifecycle, and aggregates are rebuildable projections.

The `standard` profile executes five sequential method-evaluable variants under one unchanged direction with `execution_width=1` and `stop_on_success=false`. Patch/static/activation failures, resource pauses, and OOM retries do not consume an outcome; their reservation remains until repair or explicit abandonment. Only an atomically committed, typed, verified TrialResult consumes one slot. Outcomes 1–4 route to `PROPOSE_NEXT_VARIANT`; outcome 5 creates an exact five-result aggregate and closes the direction. A sixth reservation, patch, run, or outcome is rejected by the reducer.

For `proxy_full`, execution is physically ordered by authority: `ProxyPhaseStarted` → `PhaseCommandStarted/Completed` → immutable proxy evidence → reducer-derived `ProxyOutcome`/route → `FullPhaseStarted` → full command events → finalization. Every side-effecting command rechecks an exact SQLite-derived `PhaseAuthorization`; `True`, `None`, caller mappings, and stale authorizations fail closed. Full commands cannot run before a committed `RUN_FULL`, proxy evidence never consumes the five-outcome budget, and an implementation repair must execute proxy again. Bootstrap uses `terminal_bootstrap` policy, emits proxy-only evidence, and never receives full authorization.

`TrialSpec v7` freezes a scientific `ProxyDecisionPolicy v1` and exact `PhaseCommandPlan v2` before execution. The command plan fixes typed `argv[]`, `cwd`, environment overrides, inherited-environment allowlists, dependencies, conditions, expected raw outputs, and retry/resource policy. `ExperimentRunner` executes the frozen invocation with `subprocess.Popen(..., shell=False, env=...)`; display strings never participate in execution. The proxy policy contains metrics, objective, paired aggregation, exact dataset/seed/role coverage, thresholds, activation/readiness requirements, evidence kinds, and deterministic route semantics—but no Attempt, generation, implementation, producer, or phase identity. `ProxyPhaseStarted` derives a separate `ProxyEvaluationBinding v1` that binds the policy to the current Attempt generation, implementation/input hashes, phase execution, command plan, sample/evaluator contracts, and provenance mode. Producer policy reports cannot change thresholds or routes.

Sample and evaluator provenance are stored through a symlink-safe content-addressed `ContractStore`. In real mode, `SampleManifest v4` binds the selected samples' actual immutable bytes, ordering, record boundaries, source revision, and Core-recomputed sample identities; metadata-derived IDs are not accepted. Reservation validates immutable `ContractRef` bytes before writing `AttemptReserved`; mutable sample/evaluator projections are not canonical inputs. The four production paths enter through `C2CProxyPhaseExecutor`, `C2CFullPhaseExecutor`, `GenericExternalPhaseExecutor`, or `SyntheticPhaseExecutor`; internal adapter callbacks are not independent production entry points.

Every command must match the frozen `PhaseCommandPlan v2` DAG and its typed invocation before `PhaseCommandStarted`. `PhaseCommand v3` and `PhaseRunReceipt v4` bind the command, authorization, stdout/stderr, and each physical raw output to the current Attempt/generation/phase. `EvidenceDerivationManifest v1` then binds those receipt outputs and the decoder identity to normalized evidence; `EvidenceManifest v5`, the classifier, reducer, rebuild, and Gate consume only that receipt-backed derivation chain. Completed receipts reconstruct command results without process-local state; a started command with no trustworthy receipt becomes an explicit unknown outcome rather than being silently repeated.

Planner pools, scorecards, patch diagnostics, producer proxy-policy reports, mutable result summaries, and human-readable projections remain diagnostic artifacts only. Runtime stages do not infer identity, observations, budgets, authorization, or routing from those files. Projects without DirectionSpec v3, VariantSpec v4, TrialSpec v7, and Event/Attempt/ResearchState v8 authority must restart from S1; no dual-read or migration exists. See `docs/authoritative_s1_s3_contracts.md` and `docs/regression_test_migration.md`.

Attempt lifecycle operations are generation-scoped. Implementation repair invalidates prior uncommitted phase authority and requires proxy re-execution. Failure classification and routing use receipt-derived `FailureEvidence v6`; resource pause/resume retains the same Attempt and reservation and requires immutable, receipt-bound resource measurements plus `ResumeEvidence v5`/`ResourceProbe v4`. With no `nvidia-smi` or insufficient GPU capacity, a GPU-requiring Attempt deterministically enters `RESOURCE_PAUSED` with `PAUSE_RESOURCE`, does not consume an outcome, and does not remain quarantined or permanently running.

Exactly-once recovery is verified at seven authority boundaries: after command start but before a trustworthy receipt; after durable receipt but before command completion; after completion but before evidence commit; after proxy commit but before full start; after full start but before its first command; after full commands but before finalization; and after finalization but before route delivery. Recovery reuses committed receipts/events, never repeats a proven side effect, and never consumes the budget twice.

S3 runs one shared pure validation layer before commit and again as a post-commit audit. Invalid identity, TrialSpec, phase/dataset/seed/role coverage, proxy/bootstrap evidence, or artifact hash produces zero authoritative writes. Projection writers lock and reread the newest SQLite state before writing, so a delayed writer cannot roll JSON views back to an older sequence.

## Layout

```text
workspace/<project_id>/
├── references/
│   ├── papers/
│   └── bib/
├── literature/
├── plan/
├── experiment/
├── paper/
├── review/
└── meta/
```

## Environment

- `OPENAI_API_KEY`: required for S1/S2 GPT reasoning when `llm.reasoning_provider: openai`.
- `OPENAI_BASE_URL`: optional, for OpenAI-compatible third-party endpoints. Host-only values such as `https://api-cdn.owlai.tech` are normalized to the SDK API root; full `/v1/responses` values are also accepted.
- `OPENAI_REASONING_EFFORT`: optional reasoning effort, for example `xhigh`.
- `OPENAI_DEFAULT_HEADERS`: optional JSON object or `key=value;key2=value2` string for non-auth proxy headers.
- `OPENAI_ORGANIZATION`: optional, forwarded to the OpenAI client.
- `OPENAI_PROJECT`: optional, forwarded to the OpenAI client.
- `.env` or `.env.local` in the repo root are loaded automatically if present.
- `DEEPSEEK_API_KEY`: optional, used by `auto-research enrich-s0` for DeepSeek S0 semantic enrichment.
- `SEMANTIC_SCHOLAR_API_KEY`: optional, improves literature retrieval rate limits.
- `SERPAPI_API_KEY`: reserved for future use.
- `MM_ROOT`: optional, points to a local multimodal research assets directory when reusing datasets, codebases, or checkpoints.

## Notes

- The system keeps output management strict: stage files are committed atomically and recorded in manifests.
- Required stage inputs are checked before the agent runs or writes any artifact.
- S2.5 persistent Codex sessions keep metadata/events in `workspace/<project_id>/plan/code_worktrees/`, but new Git worktree repos are stored outside the repo by default via `code_patch.worktree_storage_root` or `AUTO_RESEARCH_WORKTREE_ROOT`.
- It is designed to be honest. If a stage lacks the data or execution hooks needed to proceed, the registry is marked blocked or failed instead of fabricating artifacts.

## Authoritative Scientific Evidence

M1.1.3 makes S3 evidence Attempt-scoped and content-addressed. Experiment adapters submit only the explicit immutable inventory produced by the current command. The SQLite ledger reads each artifact once, verifies SHA-256/schema/identity, decodes row-level measurements, computes constraints and outcome, then atomically commits TrialResult, budget, RouteOutcome, and aggregate. Caller-authored observations, summaries, or outcomes are diagnostic only and cannot authorize a method result.

### Authoritative S3 phase order

For `proxy_full` attempts, SQLite commits proxy evidence and a reducer-derived `RUN_FULL` route before any full train/eval/ablation command may start. Proxy rejection releases the reservation without consuming one of the five standard outcomes. Bootstrap uses the same strict proxy evidence path but never starts full execution and never enters standard method history. Evidence and command receipts are attempt-, generation-, implementation-, and phase-scoped; mutable result summaries are diagnostic only.

M1.1.5.2 closes raw execution provenance and recovery for the production-component paths. The complete suite passes in all four required environments with `705 passed, 2 skipped`: normal (`54:27`), all upper/lower-case proxy variables cleared (`54:18`), empty HOME/Hugging Face cache with no Codex on PATH (`58:01`), and CPU-only/no-`nvidia-smi` (`54:24`). The two skips remain the existing optional torch/transformers skips.

Production-component acceptance uses real local subprocesses, frozen typed `argv/env/cwd`, `shell=False`, immutable receipt outputs, deterministic derivation, SQLite Ledger, rebuild, and Gate. Synthetic tests separately prove deterministic state, budget, and evidence semantics. Neither path is reported as real scientific success: Native Unified S1, real external Codex S1, and real GPU scientific training have not started.
