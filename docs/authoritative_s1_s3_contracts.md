# Authoritative S1-S3 Contracts

## Canonical Authority

| Concept | Version | Canonical location | Meaning |
|---|---|---|---|
| Direction | `DirectionSpec v3` | `literature/direction.json` | Complete research direction with separate semantic and full-spec identities |
| Variant | `VariantSpec v4` | `plan/variant.json` | One scientific intervention bound to the current DirectionSpec |
| Trial | `TrialSpec v6` | frozen in `AttemptReserved`; projected under `plan/attempts/<attempt_id>/trial_spec/<trial_spec_hash>.json` | Preregistered protocol, phase contracts, acceptance, proxy policy, immutable sample/evaluator references, and command-plan references |
| Event | `Event v7` | `meta/research_events.sqlite3` | Sole S1-S3 authority: SQLite WAL, continuous sequence, and hash chain |
| Attempt | `AttemptRecord v7` | `meta/attempts/<attempt_id>.json` | Rebuildable lifecycle, reservation, phase, implementation-revision, command, and receipt projection |
| Result | `TrialResult v5` | diagnostic projection under `experiment/results/` | Reducer-generated result decoded from immutable evidence |
| Observation | `ExecutionObservation v4` | embedded in TrialResult | Deterministically decoded row-level measurement with evidence and phase identity |
| Evidence manifest | `EvidenceManifest v4` | event-bound/projected | Exact receipt-linked evidence set for one authoritative transaction |
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
- `trial_spec_hash` locks TrialSpec v6, including phase contracts, acceptance constraints, immutable ContractRefs, ProxyDecisionPolicy, and PhaseCommandPlan references.
- `attempt_input_hash` binds implementation, frozen TrialSpec, runtime configuration, evaluator identity, seeds, and sample provenance.

Canonical JSON sorts object keys and rejects non-finite numbers. Attempt and TrialResult bind complete spec hashes, not only semantic hashes.

## Frozen Trial and ContractStore

TrialSpec v6 freezes before reservation:

- protocol, datasets, metrics, primary metric, objective, aggregation, statistical testing, and seeds;
- required roles, acceptance constraints, required artifacts, and exact evidence requirements;
- per-phase datasets, seeds, roles, metrics, evidence kinds, terminal semantics, and budget semantics;
- `ProxyDecisionPolicy v1` when a proxy phase is required;
- a `ContractRef v1` to immutable `SampleManifest v3` bytes;
- content-addressed `PhaseCommandPlan v1` references for applicable phases;
- evaluator provenance resolved from immutable evaluator source/config/dependency bytes.

`ContractStore` uses content-addressed files and safe path resolution. Reservation rereads the referenced bytes inside the authoritative transaction and verifies path, hash, schema, kind, source revision, ordered sample identities, evaluator file hashes, configuration, dependencies, and provenance. Ancestor/leaf symlinks, path escape, hard-link substitution, and hash drift fail closed. Mutable `plan/sample_manifest.json` and `plan/evaluator_manifest.json` are not authority.

Synthetic sample/evaluator provenance is explicit. Local non-simulated subprocess tests are intended to validate production-component wiring and evidence handling; final results are reported in the delivery. They are not real GPU training or scientific success.

## Frozen Proxy Policy and Runtime Binding

`ProxyDecisionPolicy v1` is scientific policy frozen in TrialSpec v6. It contains:

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

Command execution is part of Event v7 authority:

- `PhaseCommandStarted` validates `PhaseCommand v2` against the current `PhaseCommandPlan v1`: command-spec ID, argv, cwd, source snapshot, phase, ordinal/dependencies, expected outputs, policies, condition, authorization, and idempotency identity are frozen before a side effect.
- `PhaseRunReceipt v3` is content-addressed and binds the Started event ID/hash, Attempt/generation/phase identity, command and command-plan identity, timestamps, exit status, durable stdout/stderr ContractRefs, external job identity, and exact immutable output ContractRefs.
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

Adapters return an explicit `PhaseArtifactInventory`; they do not scan fixed result directories. Each authoritative item has one precise path/hash/kind and a completed-command `PhaseRunReceipt v3` output reference. For current Generic/C2C adapters, the preregistered collector command is the deterministic decoder node in the frozen command DAG: all physical measurement/probe dependencies must be completed first, current-generation receipts are rechecked, and the collector's immutable outputs are the canonical evidence. The inventory kind set must exactly equal the frozen phase contract: required kinds cannot be missing, optional-but-unregistered kinds cannot be added, and each kind appears at most once.

Quantitative rows are Attempt-, generation-, implementation-, input-, phase-, phase-execution-, producer-, dataset-, metric-, role-, and seed-scoped. Observations are decoded by Common Core from the same immutable bytes used for hashing and schema validation. Callers cannot submit observations, constraints, primary summaries, pass/fail, outcome, budget, or route.

`CompletionEvidence v3` carries only authoritative identity and immutable inventory. The Ledger rereads the current Attempt and frozen TrialSpec, validates exact evidence and phase semantics, decodes observations, recomputes constraints, completeness, summary, hard-pass state, and outcome, and atomically commits `AttemptFinalized`, TrialResult, reservation release, budget change, RouteOutcome, and any fifth-outcome aggregate. Reducer/rebuild and Gate repeat the same raw-byte semantics; mutating event-derived observations, constraints, outcome, summary, manifest, or evidence bytes causes `IntegrityError`.

Exact completion replay revalidates the committed fingerprint and immutable bytes, then returns the historical Attempt/TrialResult/Route/Aggregate without a new event. The same event ID with a different request or a different completion fingerprint is an integrity conflict.

## Failure and Resume v4

`FailureEvidence v5`, `ResumeEvidence v5`, and `ResourceProbe v4` are immutable, content-addressed, receipt-bound transaction evidence. Public APIs, reducer/rebuild, and audit call the same raw-byte and command-lineage validator.

- Implementation and activation failures require a current-phase nonzero-exit command receipt and class-specific evidence.
- Resource pause requires `probe_status=insufficient` and `observed_capacity < required_capacity` for the current Attempt/resource/phase.
- Resume requires `probe_status=available`, sufficient capacity, the same resource identity, and the committed pause event.
- Arbitrary logs, zero-exit failures, cross-Attempt probes, altered bytes, or mismatched generation/implementation/input identity are rejected.

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

## Crash Matrix

| Crash point | Authoritative recovery |
|---|---|
| After phase start, before command start | No Started event exists; the authorized command may start once |
| After `PhaseCommandStarted`, before side effect outcome is known | Attach by external job identity or record unknown outcome; never silently rerun |
| After side effect and receipt write, before `PhaseCommandCompleted` | Verify and reconcile the content-addressed receipt; do not rerun |
| After proxy commit, before full start | Rebuild committed ProxyOutcome/route; create at most one FullPhaseStarted |
| After full evidence ingest, before finalization | Reuse immutable orphaned evidence and complete once after validation |
| After SQLite finalization, before projection | Rebuild exactly one TrialResult, route, aggregate, and budget state |
| During full resource pause/resume | Keep committed proxy authorization; new generation restarts only full |
| During implementation repair | Invalidate old generation authority and evidence; rerun proxy |

## Deleted Runtime Contracts

The current runtime authority is Event/AttemptRecord/ResearchState v7, TrialSpec v6, PhaseExecutionManifest v3, PhaseCommand v2, PhaseRunReceipt v3, EvidenceManifest v4, SampleManifest v3, CompletionEvidence v3, FailureEvidence/ResumeEvidence v5, and ResourceProbe v4. Replaced readers must not be restored as dual-read or migration paths.

It also does not use mutable sample/evaluator canonical paths, fixed result/evidence discovery, arbitrary hash glob lookup, producer-authored proxy decisions, caller-authored TrialResult/failure/resume outcomes, a phase-agnostic production C2C runner, validation-only phase spoofing, legacy direction/variant/route readers, or compatibility fallback. Historical documentation may mention these names only as removed designs.

## Scope Boundary

M1.1.5.1 integrates the four PhaseExecutor production entries, frozen command DAGs, durable command receipts, receipt-output evidence binding through the frozen collector decoder, SQLite replay/rebuild, bootstrap/standard isolation, failure/resume contracts, and S3 Gate. Final normal, proxy-cleared, and empty-HOME/empty-HF-cache/PATH-without-Codex suites each pass with `669 passed, 2 skipped`; the production-component E2E group passes `6` tests. Native Unified S1 Producer/Core, real external Codex S1 smoke, and real GPU scientific training have not started.
