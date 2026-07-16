# Authoritative S1-S3 Contracts

## Canonical Authority

| Concept | Version | Canonical location | Meaning |
|---|---|---|---|
| Direction | `DirectionSpec v3` | `literature/direction.json` | Complete research direction with separate semantic and full-spec identities |
| Variant | `VariantSpec v4` | `plan/variant.json` | One scientific intervention bound to the current DirectionSpec |
| Trial | `TrialSpec v7` | frozen in `AttemptReserved`; projected under `plan/attempts/<attempt_id>/trial_spec/<trial_spec_hash>.json` | Preregistered protocol, phase contracts, acceptance, proxy policy, immutable sample/evaluator references, and command-plan references |
| Event | `Event v8` | `meta/research_events.sqlite3` | Sole S1-S3 authority: SQLite WAL, continuous sequence, and hash chain |
| State | `ResearchState v8` | rebuilt from `meta/research_events.sqlite3` | Deterministic state, budget, active-attempt, command, route, and aggregate reduction |
| Attempt | `AttemptRecord v8` | `meta/attempts/<attempt_id>.json` | Rebuildable lifecycle, reservation, phase, implementation-revision, command, and receipt projection |
| Result | `TrialResult v5` | diagnostic projection under `experiment/results/` | Reducer-generated result decoded from immutable evidence |
| Observation | `ExecutionObservation v4` | embedded in TrialResult | Deterministically decoded row-level measurement with evidence and phase identity |
| Evidence manifest | `EvidenceManifest v5` | event-bound/projected | Exact receipt- and derivation-linked evidence set for one authoritative transaction |
| Evidence derivation | `EvidenceDerivationManifest v1` | immutable `ContractStore` object | Deterministic raw-receipt-output to normalized-evidence lineage |
| Route | `RouteOutcome v4` | `meta/route_outcome.json` | Reducer-derived control projection bound to a source event and sequence |
| Proxy outcome | `ProxyOutcome v3` | event-bound/projected | Reducer-derived proxy classification and authorization decision |
| Aggregate | `DirectionOutcomeAggregate v1` | `meta/direction_outcome_aggregate.json` | Exactly five verified standard outcomes |

`meta/research_state.json`, Attempt JSON, TrialResult JSON, RouteOutcome JSON, proxy reports, aggregates, Gate reports, scorecards, and summaries are projections or diagnostics. Runtime decisions read SQLite through `ResearchEventLedger`; damaged, missing, future-sequence, or stale projections can be isolated and rebuilt. A workspace using an older Event/Attempt/State/TrialSpec contract is rejected with `BreakingSchemaError` and must rerun from S1. There is no dual read or automatic migration.

## Scientific and Contract Identity

- `direction_semantic_hash` excludes IDs, names, iteration, and run lineage; it identifies the scientific direction.
- `direction_spec_hash` locks the complete authoritative DirectionSpec.
- `variant_semantic_hash` is derived from intervention operations/configuration, controls, ablation, implementation surfaces, metric expectations, hypotheses, and falsification. ID/lineage/nonce-only changes do not create a new method.
- `variant_spec_hash` locks the complete VariantSpec and its DirectionSpec lineage.
- `implementation_hash` binds the frozen patch, resulting files, and implementation manifest.
- `trial_spec_hash` locks TrialSpec v7, including phase contracts, acceptance constraints, immutable ContractRefs, ProxyDecisionPolicy, and PhaseCommandPlan references.
- `attempt_input_hash` binds implementation, frozen TrialSpec, runtime configuration, evaluator identity, seeds, and sample provenance.

Canonical JSON sorts object keys and rejects non-finite numbers. Attempt and TrialResult bind complete spec hashes, not only semantic hashes.

## Frozen Trial and ContractStore

TrialSpec v7 freezes before reservation:

- protocol, datasets, metrics, primary metric, objective, aggregation, statistical testing, and seeds;
- required roles, acceptance constraints, required artifacts, and exact evidence requirements;
- per-phase datasets, seeds, roles, metrics, evidence kinds, terminal semantics, and budget semantics;
- `ProxyDecisionPolicy v1` when a proxy phase is required;
- a `ContractRef v1` to immutable `SampleManifest v4` bytes;
- content-addressed `PhaseCommandPlan v2` references for applicable phases;
- evaluator provenance resolved from immutable evaluator source/config/dependency bytes.

`ContractStore` uses content-addressed files and safe path resolution. Reservation rereads the referenced bytes inside the authoritative transaction and verifies path, hash, schema, kind, source revision, ordered sample identities, evaluator file hashes, configuration, dependencies, and provenance. In real mode, `SampleManifest v4` references the actual selected record/shard bytes; Core recomputes record boundaries, ordering, sample IDs, counts, shard digests, and aggregate digests from those bytes. Missing raw refs, metadata-derived IDs, changed ordering, or changed sample bytes fail before reservation. Ancestor/leaf symlinks, path escape, hard-link substitution, and hash drift also fail closed. Mutable sample/evaluator projections are not authority.

Synthetic sample/evaluator provenance is explicit. Local non-simulated subprocess tests are intended to validate production-component wiring and evidence handling; final results are reported in the delivery. They are not real GPU training or scientific success.

## Frozen Proxy Policy and Runtime Binding

`ProxyDecisionPolicy v1` is scientific policy frozen in TrialSpec v7. It contains:

- primary metric, objective, paired aggregation;
- exact datasets, seeds, metrics, and roles;
- aggregate improvement and per-dataset maximum-regression thresholds;
- required activation surfaces and readiness check IDs;
- the exact authoritative evidence-kind set;
- `gate_to_full` or `terminal_bootstrap` mode;
- deterministic science-reject, integrity-failure, and resource-failure semantics;
- a canonical policy hash.

It deliberately excludes Attempt, generation, implementation, producer, and phase-execution identity.

`ProxyEvaluationBinding v1` is generated by the Ledger in the `ProxyPhaseStarted` transaction. It binds the frozen policy hash to the current Attempt, direction/variant/trial identities, lifecycle generation, implementation/input hashes, phase execution and start event, producer run, command-plan and phase-contract hashes, sample/evaluator ContractRefs, provenance mode, and expected evidence kinds. It is embedded in `PhaseExecutionManifest v3` and independently rederived during reducer/rebuild.

The single pure proxy classifier is shared by proxy precommit, `ProxyEvidenceCommitted`, reducer/rebuild, and Gate audit. It reads only the frozen policy, Ledger binding, exact EvidenceManifest, and immutable receipt-bound evidence bytes. It rejects missing, duplicate, extra, or aggregate-expanded rows; validates exact dataset × seed × metric × role coverage; computes paired deltas and per-dataset regression; and derives activation/readiness from the registered evidence bytes. Producer-authored effective policy, calibration, threshold, decision, constraints, summary, or route has no authority and cannot enter the authoritative evidence set.

A complete scientific proxy miss commits `ProxyEvidenceCommitted`, routes `PROPOSE_NEXT_VARIANT`, releases the reservation, consumes no five-variant outcome, and never starts full execution. Missing, malformed, damaged, or identity-conflicting evidence is an integrity failure, not a scientific rejection.

## Physical Phase Transactions

A `proxy_full` Attempt follows this authority order:

```text
AttemptReserved / READY
→ ProxyPhaseStarted
→ PhaseCommandStarted(proxy)
→ PhaseCommandCompleted(proxy)
→ ProxyEvidenceCommitted
→ reducer-derived ProxyOutcome and RouteOutcome
→ FullPhaseStarted only when decision=RUN_FULL
→ PhaseCommandStarted(full)
→ PhaseCommandCompleted(full)
→ AttemptFinalized
```

`AuthoritativePhaseContext` and `PhaseAuthorization` are derived from current SQLite state. Production executors reject `None`, booleans, caller mappings, stale generations, stale implementation/input hashes, mismatched command plans, and forged proxy authorization. Every side-effecting command revalidates authority immediately before execution.

`PhaseExecutionManifest v3` freezes phase identity, the content-addressed command-plan reference/hash, adapter/provenance mode, expected evidence kinds, ProxyEvaluationBinding, and—only for full after proxy—a reference to the committed proxy authorization. Generic full-only execution may omit proxy authorization only when the frozen protocol says it is full-only. Bootstrap uses `terminal_bootstrap` and cannot receive `FullPhaseStarted`.

The executors are phase-specific: `C2CProxyPhaseExecutor`, `C2CFullPhaseExecutor`, `GenericExternalPhaseExecutor`, and `SyntheticPhaseExecutor`. The production flow does not use a phase-agnostic C2C runner and does not recursively rerun the whole ExperimentAgent pipeline.

## Ledger Command Lifecycle

Command execution is part of Event v8 authority:

- `PhaseCommandStarted` validates `PhaseCommand v3` against the current `PhaseCommandPlan v2`: command-spec ID, typed `argv[]`, `cwd`, environment overrides, inherited-environment allowlist, source snapshot, phase, ordinal/dependencies, expected raw outputs, policies, condition, authorization, and idempotency identity are frozen before a side effect.
- `ExperimentRunner` executes the exact frozen invocation with `subprocess.Popen(argv, shell=False, env=...)`; shell strings and display rendering are never execution inputs.
- `PhaseRunReceipt v4` is content-addressed and binds the Started event ID/hash, Attempt/generation/phase identity, command and command-plan identity, timestamps, exit status, durable stdout/stderr ContractRefs, external job identity, and the exact immutable physical raw-output ContractRefs.
- `PhaseCommandCompleted` commits the verified receipt reference.
- `PhaseCommandUnknownOutcome` records a started command whose result cannot be attached or proven; the command is not silently rerun.

Recovery rules are fail closed:

1. A completed command reuses its committed receipt and does not run again.
2. A durable receipt written after the side effect but before the Completed event is verified and reconciled without rerunning.
3. A started command with an attachable external job resumes by job identity.
4. A started command without a trusted receipt or attachable job becomes an unknown outcome.
5. Only the absence of a Started event permits first execution.

A normal process crash preserves generation, phase execution ID, and producer run. A committed resource pause/resume advances lifecycle generation. Completed replay reconstructs the typed result and inventory from receipt bytes rather than process-local callback state.

## Evidence, Finalization, and Reducer Closure

Adapters return an explicit `PhaseArtifactInventory`; they do not scan fixed result directories. Every authoritative normalized item has one precise path/hash/kind and a lineage through `EvidenceDerivationManifest v1` to completed current-phase `PhaseRunReceipt v4` raw outputs. The manifest binds source command IDs, receipt hashes, raw output refs/hashes, decoder ID/version/source hash, frozen evaluator/sample/protocol identities, and normalized output refs/hashes. Decoders read only immutable receipt outputs; mutable comparison state, result directories, producer summaries, and caller dictionaries cannot supply scientific values. The inventory kind set must exactly equal the frozen phase contract: required kinds cannot be missing, optional-but-unregistered kinds cannot be added, and each kind appears at most once.

Quantitative rows are Attempt-, generation-, implementation-, input-, phase-, phase-execution-, producer-, dataset-, metric-, role-, and seed-scoped. Observations are decoded by Common Core from the same immutable bytes used for hashing and schema validation. Callers cannot submit observations, constraints, primary summaries, pass/fail, outcome, budget, or route.

`CompletionEvidence v3` carries only authoritative identity and immutable inventory. The Ledger rereads the current Attempt and frozen TrialSpec, validates `EvidenceManifest v5`, receipt/output/derivation lineage, exact evidence and phase semantics, decodes observations, recomputes constraints, completeness, summary, hard-pass state, and outcome, and atomically commits `AttemptFinalized`, TrialResult, reservation release, budget change, RouteOutcome, and any fifth-outcome aggregate. Reducer/rebuild and Gate repeat the same physical-raw-byte and derivation semantics; mutating a receipt output, derivation manifest, normalized evidence, event-derived observations, constraints, outcome, summary, or manifest causes `IntegrityError`.

Exact completion replay revalidates the committed fingerprint and immutable bytes, then returns the historical Attempt/TrialResult/Route/Aggregate without a new event. The same event ID with a different request or a different completion fingerprint is an integrity conflict.

## Failure and Resource Authority

`FailureEvidence v6`, `ResumeEvidence v5`, and `ResourceProbe v4` are immutable, content-addressed, receipt-bound transaction evidence. Public APIs, reducer/rebuild, and audit call the same raw-byte, command-lineage, and resource-identity validators. Failure class, exit status, stdout/stderr hashes, and route are derived from the committed receipt plus frozen policy; caller-authored command-result evidence has no authority.

- Implementation and activation failures require a current-phase nonzero-exit command receipt and class-specific evidence.
- Resource pause requires `probe_status=insufficient` and `observed_capacity < required_capacity` for the current Attempt/resource/phase.
- Resume requires `probe_status=available`, sufficient capacity, the same resource identity, and the committed pause event.
- Arbitrary logs, zero-exit failures, cross-Attempt probes, altered bytes, or mismatched generation/implementation/input identity are rejected.

When a GPU-requiring command cannot find `nvidia-smi` or observes insufficient capacity, the trusted resource path atomically records the measurement and reducer-derived `PAUSE_RESOURCE`; the Attempt becomes `RESOURCE_PAUSED`, keeps its reservation, and consumes no method outcome. It must not fall into quarantine or remain permanently `PROXY_RUNNING`/`FULL_RUNNING`.

Implementation repair preserves Attempt and Variant identity, changes implementation/input hashes, advances generation, clears old proxy authorization, and reruns proxy. Proxy resource resume advances generation and resets proxy to pending. Full resource resume preserves the committed proxy outcome and restores `PROXY_COMPLETED` with proxy completed/full pending, so full can restart without rerunning proxy. Full-only resume returns ready with full pending.

## Budget, Routing, and Bootstrap

The reducer maintains `0 <= consumed`, `0 <= reserved`, and `consumed + reserved <= target == 5`, with project-wide execution width one across executable standard and bootstrap Attempts.

- Only a verified standard method-evaluable TrialResult consumes one outcome.
- Outcomes 1–4 route `PROPOSE_NEXT_VARIANT` even when accepted.
- Outcome 5 creates an aggregate containing exactly five unique standard outcomes and closes or exhausts the direction.
- A sixth reservation, patch, command, artifact creation, or outcome is rejected before execution.
- Proxy science rejection, implementation/activation failure, resource pause, integrity block, and explicit abandonment do not consume a method outcome.
- Bootstrap is `bootstrap + bootstrap_proxy`, uses proxy-terminal evidence, routes `FINISH_RUN` only after verified completion, consumes no standard budget, creates no standard tried-history entry, and creates no direction aggregate.

RouteOutcome is reducer-owned and bound to its source event, sequence, Attempt, identity, budget snapshot, artifacts, and idempotency key. Orchestrator queries the committed Ledger operation result; diagnostic result/route JSON cannot override SQLite control authority.

## Seven-Point Crash Matrix

| Crash point | Authoritative recovery |
|---|---|
| 1. After `PhaseCommandStarted`, before a trustworthy receipt | Attach by frozen external-job identity or record a typed unknown/integrity outcome; never silently rerun |
| 2. After side effect and durable receipt, before `PhaseCommandCompleted` | Validate the receipt and immutable outputs, then reconcile Completed without rerunning |
| 3. After `PhaseCommandCompleted`, before proxy/full evidence commit | Rebuild inventory and derivation from the committed receipt; do not rerun the command |
| 4. After `ProxyEvidenceCommitted`, before `FullPhaseStarted` | Rebuild the same reducer-owned `RUN_FULL` authorization and create at most one FullPhaseStarted |
| 5. After `FullPhaseStarted`, before the first full command | Resume the frozen full command DAG from SQLite and start each command at most once |
| 6. After full command completion, before `AttemptFinalized` | Rebuild receipt outputs and derivations, finalize once, and consume at most one budget slot |
| 7. After `AttemptFinalized`, before RouteOutcome reaches the caller | Return the historical Attempt, TrialResult, RouteOutcome, and aggregate without duplicate-method validation or new writes |

## Deleted Runtime Contracts

The current runtime authority is Event/AttemptRecord/ResearchState v8, TrialSpec v7, PhaseExecutionManifest v3, PhaseCommandPlan v2, PhaseCommand v3, PhaseRunReceipt v4, EvidenceManifest v5, EvidenceDerivationManifest v1, SampleManifest v4, CompletionEvidence v3, FailureEvidence v6, ResumeEvidence v5, and ResourceProbe v4. Replaced readers must not be restored as dual-read or migration paths.

It also does not use mutable sample/evaluator canonical paths, fixed result/evidence discovery, arbitrary hash glob lookup, producer-authored proxy decisions, caller-authored TrialResult/failure/resume outcomes, a phase-agnostic production C2C runner, validation-only phase spoofing, legacy direction/variant/route readers, or compatibility fallback. Historical documentation may mention these names only as removed designs.

## Scope Boundary

M1.1.5.2 closes raw execution provenance and recovery for the four PhaseExecutor production-component entries, typed command DAGs, durable raw-output receipts, deterministic evidence derivation, SQLite replay/rebuild, bootstrap/standard isolation, receipt-derived failure/resource authority, and S3 Gate. The complete suite passes in all four required environments with `705 passed, 2 skipped`: normal (`54:27`), upper/lower-case proxy variables cleared (`54:18`), empty HOME/Hugging Face cache with no Codex on PATH (`58:01`), and CPU-only/no-`nvidia-smi` (`54:24`). The two skips are the existing optional torch/transformers skips.

Production-component validation runs real local subprocesses through frozen typed `argv/env/cwd`, `shell=False`, immutable raw outputs, `PhaseRunReceipt v4`, `EvidenceDerivationManifest v1`, normalized evidence, Ledger, rebuild, and Gate. Synthetic tests separately validate deterministic state, budget, and evidence semantics. These are engineering acceptance paths, not claims of scientific success: Native Unified S1 Producer/Core, real external Codex S1, and real GPU scientific training have not started.
