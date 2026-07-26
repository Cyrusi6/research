# Authoritative S1-S3 Contracts

## Canonical Authority

| Concept | Version | Canonical location | Meaning |
|---|---|---|---|
| Direction | `DirectionSpec v3` | `literature/direction.json` | Complete research direction with separate semantic and full-spec identities |
| Variant | `VariantSpec v4` | `plan/variant.json` | One scientific intervention bound to the current DirectionSpec |
| Trial | `TrialSpec v9` | frozen in `AttemptReserved`; projected under `plan/attempts/<attempt_id>/trial_spec/<trial_spec_hash>.json` | Preregistered protocol, phase contracts, proxy/readiness policy, derivation plans, acceptance, and immutable contract references |
| Event | `Event v10` | `meta/research_events.sqlite3` | Sole S1-S3 authority: SQLite WAL, continuous sequence, and hash chain |
| State | `ResearchState v10` | rebuilt from `meta/research_events.sqlite3` | Deterministic state, budget, active-attempt, command, route, and aggregate reduction |
| Attempt | `AttemptRecord v10` | `meta/attempts/<attempt_id>.json` | Rebuildable lifecycle, reservation, phase, implementation-revision, command, receipt, and frozen derivation projection |
| Result | `TrialResult v6` | diagnostic projection under `experiment/results/` | Reducer-generated result decoded from one immutable receipt-derived evidence chain |
| Observation | `ExecutionObservation v4` | embedded in TrialResult | Deterministically decoded row-level measurement with evidence and phase identity |
| Derivation plan | `EvidenceDerivationPlan v2` | frozen in each TrialSpec phase contract and content-addressed | Ordered physical source bindings and roles, immutable decoder identity, canonicalization/coverage, cross-phase bindings, and exact normalized output set |
| Decoder | `DecoderDescriptor v2` + `DecoderImplementationBundle v1` + `DecoderProgram v1` | frozen in the derivation/readiness plan and content-addressed | Decoder identity plus the actual declarative executable semantics, runtime ABI, entrypoint, dependencies, and implementation hash |
| Evidence derivation | `EvidenceDerivationManifest v3` | immutable `ContractStore` object referenced by the derive receipt | The sole deterministic physical-receipt/raw-output to normalized-evidence lineage for one phase |
| Evidence manifest | `EvidenceManifest v6` | event-bound/projected | Exact normalized evidence set; every entry points to the same derive-receipt derivation reference/hash |
| Readiness plan | `ReadinessCheckPlan v2` | frozen in the proxy phase contract and content-addressed | Ordered authority-role/check bindings, predicates, thresholds, coverage, decoder identity, and BLOCKED route semantics |
| Route | `RouteOutcome v4` | `meta/route_outcome.json` | Reducer-derived control projection bound to a source event and sequence |
| Proxy outcome | `ProxyOutcome v4` | event-bound/projected | Reducer-derived scientific, activation, readiness, and authorization decision |
| Aggregate | `DirectionOutcomeAggregate v1` | `meta/direction_outcome_aggregate.json` | Exactly five verified standard outcomes |

`meta/research_state.json`, Attempt JSON, TrialResult JSON, RouteOutcome JSON, proxy reports, aggregates, Gate reports, scorecards, and summaries are projections or diagnostics. Runtime decisions read SQLite through `ResearchEventLedger`; damaged, missing, future-sequence, or stale projections can be isolated and rebuilt. A workspace using an older Event/Attempt/State/TrialSpec contract is rejected with `BreakingSchemaError` and must rerun from S1. There is no dual read or automatic migration.

## Scientific and Contract Identity

- `direction_semantic_hash` excludes IDs, names, iteration, and run lineage; it identifies the scientific direction.
- `direction_spec_hash` locks the complete authoritative DirectionSpec.
- `variant_semantic_hash` is derived from intervention operations/configuration, controls, ablation, implementation surfaces, metric expectations, hypotheses, and falsification. ID/lineage/nonce-only changes do not create a new method.
- `variant_spec_hash` locks the complete VariantSpec and its DirectionSpec lineage.
- `implementation_hash` binds the frozen patch, resulting files, and implementation manifest.
- `trial_spec_hash` locks TrialSpec v9, including phase contracts, acceptance constraints, immutable ContractRefs, ProxyDecisionPolicy, ReadinessCheckPlan, EvidenceDerivationPlan, and PhaseCommandPlan references.
- `attempt_input_hash` binds implementation, frozen TrialSpec, runtime configuration, evaluator identity, seeds, and sample provenance.

Canonical JSON sorts object keys and rejects non-finite numbers. Attempt and TrialResult bind complete spec hashes, not only semantic hashes.

## Frozen Trial and ContractStore

TrialSpec v9 freezes before reservation:

- protocol, datasets, metrics, primary metric, objective, aggregation, statistical testing, and seeds;
- required roles, acceptance constraints, required artifacts, and exact evidence requirements;
- per-phase datasets, seeds, roles, metrics, evidence kinds, terminal semantics, and budget semantics;
- `ProxyDecisionPolicy v2` and `ReadinessCheckPlan v2` when a proxy phase is required;
- one `EvidenceDerivationPlan v2` per executable phase, including its immutable `DecoderDescriptor v2`, ordered physical source bindings and authority roles, canonicalization and coverage rules, cross-phase bindings, and ordered normalized evidence exact-set;
- the descriptor's immutable `DecoderImplementationBundle v1` and `DecoderProgram v1`, which freeze the actual declarative transformation algorithm, runtime ABI, entrypoint, dependencies, canonical JSON semantics, measurement mapping, finite-number rules, pairing, activation/readiness derivation, and output identity;
- a `ContractRef v1` to immutable `SampleManifest v4` bytes;
- content-addressed `PhaseCommandPlan v4` references for applicable phases;
- evaluator provenance resolved from immutable evaluator source/config/dependency bytes.

`ContractStore` uses content-addressed files and safe path resolution. Reservation rereads the referenced bytes inside the authoritative transaction and verifies path, hash, schema, kind, source revision, ordered sample identities, evaluator file hashes, configuration, dependencies, and provenance. In real mode, `SampleManifest v4` references the actual selected record/shard bytes; Core recomputes record boundaries, ordering, sample IDs, counts, shard digests, and aggregate digests from those bytes. Missing raw refs, metadata-derived IDs, changed ordering, or changed sample bytes fail before reservation. Ancestor/leaf symlinks, path escape, hard-link substitution, and hash drift also fail closed. Mutable sample/evaluator projections are not authority.

Synthetic sample/evaluator provenance is explicit. Local non-simulated subprocess tests are intended to validate production-component wiring and evidence handling; final results are reported in the delivery. They are not real GPU training or scientific success.

## Frozen Proxy Policy and Runtime Binding

`ProxyDecisionPolicy v2` is scientific policy frozen in TrialSpec v9. It contains:

- primary metric, objective, paired aggregation;
- exact datasets, seeds, metrics, and roles;
- aggregate improvement and per-dataset maximum-regression thresholds;
- required activation surfaces, activation-delta threshold, readiness check IDs, and the immutable `ReadinessCheckPlan v2` reference/hash;
- the exact authoritative evidence-kind set;
- `gate_to_full` or `terminal_bootstrap` mode;
- deterministic science-reject, integrity-failure, and resource-failure semantics;
- a canonical policy hash.

It deliberately excludes Attempt, generation, implementation, producer, and phase-execution identity.

`ProxyEvaluationBinding v1` is generated by the Ledger in the `ProxyPhaseStarted` transaction. It binds the frozen policy hash to the current Attempt, direction/variant/trial identities, lifecycle generation, implementation/input hashes, phase execution and start event, producer run, command-plan and phase-contract hashes, sample/evaluator ContractRefs, provenance mode, and expected evidence kinds. It is embedded in `PhaseExecutionManifest v3` and independently rederived during reducer/rebuild.

The single pure proxy classifier is shared by proxy precommit, `ProxyEvidenceCommitted`, reducer/rebuild, and Gate audit. It reads only the frozen policy, Ledger binding, exact EvidenceManifest, immutable receipt-bound evidence bytes, and frozen ReadinessCheckPlan. It rejects missing, duplicate, extra, or aggregate-expanded rows; validates exact dataset × seed × metric × role coverage; computes paired deltas and per-dataset regression; recomputes activation from enabled/disabled measurements plus observed surfaces; and evaluates readiness predicates over independently bound raw receipt outputs. Producer-authored effective policy, calibration, threshold, PASS list, decision, constraints, summary, or route has no authority and cannot enter the authoritative evidence set.

Command completion, activation, readiness, and scientific proxy acceptance are separate facts. Exit code zero proves only that a command completed. `ActivationEvidence v4` requires explicit enabled/disabled and observed-surface measurements; expected surfaces cannot be copied into observed coverage. `FullS3Readiness v4` records the deterministic results of `ReadinessCheckPlan v2`; each check has an ordered exact set of receipt authority roles/check IDs, predicate, comparator, threshold, and coverage rule. Missing, extra, duplicate, reordered, or cross-authority inputs are integrity failures.

The reducer applies this priority: malformed or incomplete receipt/derivation/coverage → `BLOCK_INTEGRITY`; authoritative resource insufficiency → `PAUSE_RESOURCE`; valid activation or readiness `BLOCKED` → `REPAIR_IMPLEMENTATION`; readiness PASS plus a scientific proxy miss → `PROPOSE_NEXT_VARIANT`; readiness PASS plus scientific proxy acceptance → `RUN_FULL`; verified bootstrap proxy completion → `FINISH_RUN`. A `REPAIR_IMPLEMENTATION` proxy outcome preserves the reservation, creates no TrialResult or method-history entry, consumes no direction outcome, and never authorizes `FullPhaseStarted`.

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

Command execution is part of Event v10 authority:

- `PhaseCommandStarted` validates `PhaseCommand v5` against the current `PhaseCommandPlan v4`: command-spec ID, typed `argv[]`, `cwd`, environment overrides, inherited-environment allowlist, source snapshot, phase, ordinal/dependencies, authority role, readiness check binding, physical/derivation output contracts, policies, condition, authorization, and idempotency identity are frozen before a side effect.
- `ExperimentRunner` executes the exact frozen invocation with `subprocess.Popen(argv, shell=False, env=...)`; shell strings and display rendering are never execution inputs.
- A physical `PhaseRunReceipt v5` is content-addressed and binds the Started event ID/hash, Attempt/generation/phase identity, command and command-plan identity, timestamps, exit status, durable stdout/stderr ContractRefs, external job identity, and the exact immutable physical raw-output ContractRefs.
- The derive `PhaseRunReceipt v5` additionally commits one structured `derivation_ref/hash` and the exact normalized output ContractRefs. The reference is transaction data, not a value parsed from stdout.
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

Adapters return an explicit `PhaseArtifactInventory`; they do not scan fixed result directories. The only authoritative derivation is:

```text
frozen EvidenceDerivationPlan v2
→ completed physical PhaseCommand receipts and immutable raw outputs
→ journal-owned execution of DecoderProgram v1 over those exact bytes
→ one EvidenceDerivationManifest v3
→ derive PhaseRunReceipt v5.derivation_ref/hash
→ EvidenceManifest v6 entries using that identical ref/hash
→ ProxyOutcome v4 or TrialResult v6
→ RouteOutcome
```

The derivation plan and manifest bind the ordered/exact physical command and raw-output set, explicit authority roles, completed-event and receipt identities, decoder descriptor and actual immutable implementation bundle/program, frozen evaluator/sample/protocol identities, coverage/canonicalization, cross-phase bindings, and ordered normalized outputs. A derive command cannot list itself as a physical source. Missing, extra, duplicate, reordered, cross-Attempt, cross-generation, cross-phase, cross-producer, wrong-command/output, wrong-decoder, or wrong-authority sources fail closed. The former direct/self derivation builder, which created a second manifest from already normalized derive outputs, is deleted.

`validate_immutable_derivation()` is the shared read-only authority used by derive-receipt precommit, `PhaseCommandCompleted`, EvidenceManifest binding, Ledger transaction, reducer/rebuild, state/query, restart replay, and S3 Gate. It obtains the derivation reference only from the durable/completed derive receipt, loads DecoderProgram v1 from the immutable implementation bundle, rereads the exact physical raw bytes, runs the constrained declarative VM in memory, and byte-compares recomputed normalized outputs with the derivation manifest, derive receipt outputs, EvidenceManifest entries, and content-addressed evidence blobs. It does not call `put_json`, `put_bytes`, write SQLite/projections, consult the current decoder registry/source file for historical semantics, reconstruct missing blobs, or rerun physical/producing derive commands.

The producing decoder executes only inside `journal.run_once()` after a successful `PhaseCommandStarted`. Completed replay returns the historical receipt. Durable receipt without Completed performs the same semantic validation and commits only the missing Completed event. Started without a trustworthy receipt does not rederive; it commits one typed, queryable, replayable `BLOCK_INTEGRITY` control route. Orphan normalized or manifest blobs never become recovery authority.

Every authoritative normalized item has one precise path/hash/kind and the same phase-level derivation reference. Decoders read only immutable receipt outputs; staging, mutable comparison state, result directories, producer summaries, caller dictionaries, and semantically similar manifests cannot supply scientific values. The inventory kind set and order must exactly equal the frozen derivation/phase contract: required kinds cannot be missing, optional-but-unregistered kinds cannot be added, and each kind appears at most once.

Quantitative rows are Attempt-, generation-, implementation-, input-, phase-, phase-execution-, producer-, dataset-, metric-, role-, and seed-scoped. Observations are decoded by Common Core from the same immutable bytes used for hashing and schema validation. Callers cannot submit observations, constraints, primary summaries, pass/fail, outcome, budget, or route.

`CompletionEvidence v3` carries only authoritative identity and immutable inventory. The Ledger rereads the current Attempt and frozen TrialSpec, validates `EvidenceManifest v6`, receipt/output/derivation lineage, exact evidence and phase semantics, decodes observations, recomputes constraints, completeness, summary, hard-pass state, and outcome, and atomically commits `AttemptFinalized`, TrialResult v6, reservation release, budget change, RouteOutcome, and any fifth-outcome aggregate. Reducer/rebuild and Gate repeat the same physical-raw-byte and derivation semantics; mutating a receipt output, derivation manifest, normalized evidence, event-derived observations, constraints, outcome, summary, or manifest causes `IntegrityError`.

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

Derivation recovery adds two constrained boundaries inside item 3:

- If the unique derive receipt and immutable outputs are durable but `PhaseCommandCompleted` is absent, restart validates that receipt, commits only the missing Completed event, and does not rerun physical commands or Core derivation.
- If the derive command is completed but `ProxyEvidenceCommitted`/`AttemptFinalized` is absent, restart reuses the same `derivation_ref/hash` and normalized outputs. Unreferenced orphan derivation blobs never acquire recovery authority.

Read-only state, rebuild, query, and Gate operations do not perform recovery writes. Their CAS/event/receipt/evidence/projection snapshots must be byte-for-byte unchanged on both success and failure.

## Deleted Runtime Contracts

The current runtime authority is Event/AttemptRecord/ResearchState v10, TrialSpec v9, TrialResult v6, PhaseExecutionManifest v3, PhaseCommandPlan v4, PhaseCommand v5, PhaseRunReceipt v5, EvidenceManifest v6, EvidenceDerivationManifest v3, EvidenceDerivationPlan v2, DecoderDescriptor v2, DecoderImplementationBundle v1, DecoderProgram v1, ReadinessCheckPlan v2, ActivationEvidence v4, FullS3Readiness v4, ProxyDecisionPolicy v2, ProxyOutcome v4, SampleManifest v4, CompletionEvidence v3, FailureEvidence v6, ResumeEvidence v5, and ResourceProbe v4. Replaced readers must not be restored as dual-read or migration paths.

It also does not use direct/self derivation, a second derivation manifest synthesized from normalized outputs, stdout-only derivation authority, `Path(__file__)` decoder authority, validator-time CAS repair, mutable sample/evaluator canonical paths, fixed result/evidence discovery, arbitrary hash glob lookup, producer-authored activation/readiness/proxy decisions, caller-authored TrialResult/failure/resume outcomes, a phase-agnostic production C2C runner, validation-only phase spoofing, legacy direction/variant/route readers, or compatibility fallback. Historical documentation may mention these names only as removed designs.

## Scope Boundary

Commit `1fd4e84` established the M1.1.5.3 physical derivation/readiness chain but is a migration checkpoint, not final acceptance. M1.1.5.3.1 freezes the executable decoder semantics, moves producing derivation behind journal idempotency, validates raw-to-normalized semantics at Completed, enforces ordered exact activation/readiness authority sets, and verifies seven derivation/readiness cold-restart boundaries. Earlier `747 passed, 2 skipped` results certify only the replaced v9/v8 contracts and are not reused as current evidence.

The current v10/v9 acceptance collects 781 tests and reports `779 passed, 2 skipped, 0 failed` in normal, proxy-cleared, empty-HOME/no-Codex, and CPU-only/no-`nvidia-smi` local environments. The two skips are unchanged optional torch/transformers probes; no skip or xfail was added.

Production-component validation runs real local subprocess fixtures through frozen typed `argv/env/cwd`, `shell=False`, immutable physical raw outputs, `PhaseRunReceipt v5`, `EvidenceDerivationManifest v3`, normalized evidence, Ledger, rebuild, and Gate. Synthetic tests separately validate deterministic state, budget, and evidence semantics. These are local engineering attestations, not claims of scientific success: Native Unified S1 Producer/Core, real external Codex S1, M2, and real GPU scientific training have not started.
