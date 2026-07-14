# Authoritative S1-S3 Contracts

## Canonical Artifacts

| Concept | Version | Canonical path | Authority |
|---|---|---|---|
| Direction | `DirectionSpec v3` | `literature/direction.json` | Complete research-direction specification with separate semantic/spec hashes |
| Variant | `VariantSpec v4` | `plan/variant.json` | One scientific intervention bound to the complete DirectionSpec |
| Event | `Event v2` | `meta/research_events.sqlite3` | The only authoritative S1-S3 transaction store; SQLite WAL with a hash chain |
| Attempt | `AttemptRecord v2` | `meta/attempts/<attempt_id>.json` | Rebuildable attempt lifecycle projection |
| Result | `TrialResult v2` | `experiment/results/trial_result.json` | Rebuildable projection of the latest committed verified result |
| Route | `RouteOutcome v2` | `meta/route_outcome.json` | Rebuildable deterministic route projection |
| Aggregate | `DirectionOutcomeAggregate v1` | `meta/direction_outcome_aggregate.json` | Exactly five verified standard outcomes and deterministic selection status |

`meta/research_state.json`, attempt views, TrialResult, RouteOutcome, aggregate, scorecards, Gate reports, and human-readable result files are projections or diagnostics. Deleting them does not delete authority; `ResearchEventLedger.rebuild()` reconstructs them from SQLite. Event v1 workspaces are rejected with a breaking-schema error and must restart from S1.

## Identity Boundaries

- `direction_semantic_hash` identifies the scientific direction without IDs, display names, run IDs, or iteration metadata.
- `direction_spec_hash` locks every authoritative DirectionSpec field except the hash fields themselves.
- `variant_semantic_hash` identifies the scientific method from intervention operations/configuration, real variation values, controls, ablation, implementation surfaces, expected metrics, hypotheses, and falsification conditions. Changing only IDs or lineage cannot create a new method.
- `variant_spec_hash` locks the complete VariantSpec. `VariantSpec.lineage.direction_spec_hash` must equal the current complete DirectionSpec hash.
- `implementation_hash` binds the frozen patch, actual file contents, and implementation manifest.
- `attempt_input_hash` binds implementation, protocol, sample manifest, seeds, runtime config, and evaluator identity.

Canonical JSON sorts object keys and rejects NaN/Inf. A repair keeps direction, variant, and attempt identity fixed; only a new implementation revision and its derived input hash may change.

## Event Transactions

Every event uses `Event v2` with a strict event-type enum, type-specific payload schema, validated ID, continuous sequence, `previous_event_hash`, and `event_hash`. Append performs duplicate-ID checking, sequence allocation, schema/transition validation, reduction, invariant checking, and durable insertion under one SQLite `BEGIN IMMEDIATE` transaction.

Rebuild rejects schema mismatch, sequence gaps/duplicates, duplicate IDs, hash-chain damage, event-hash damage, illegal transitions, and invalid state invariants with `IntegrityError`. The same event ID and same type/payload is idempotent; a conflicting payload is an integrity failure.

Attempt finalization is one domain transaction: verified TrialResult, terminal attempt state, reservation release, budget consumption, deterministic route, and the fifth-outcome aggregate are committed in one event. A crash after database commit but before projection writes rebuilds one result and one route without double counting.

## Attempt Lifecycle

The constrained states are `PLANNED`, `IMPLEMENTING`, `IMPLEMENTATION_REPAIR`, `READY`, `PROXY_RUNNING`, `PROXY_COMPLETED`, `FULL_RUNNING`, `METHOD_COMPLETED`, `METHOD_FAILED`, `RESOURCE_PAUSED`, `INTEGRITY_BLOCKED`, and `ABANDONED`. Each legal edge has explicit preconditions; terminal states cannot transition out.

Reservation occurs only after the frozen patch, TrialSpec/protocol, sample manifest, runtime config, seeds, and evaluator are known. Repair and resource pause retain their reservation. Releasing it requires explicit `AttemptAbandoned`, and an abandoned attempt can never submit an outcome. Crash/resume reuses the same attempt ID. Bootstrap and standard identities include profile and attempt kind, so they cannot collide.

## S3 Commit Order

S3 uses: prepare/reserve → typed draft observations → uncommitted TrialResult draft → pure pre-commit validation → atomic ledger commit → projection generation. Gate/pre-commit rejection writes no event and changes no budget, method history, route, snapshot, or canonical TrialResult. Invalid drafts may be written only under `experiment/quarantine/`.

`ExecutionObservation v1` records phase, command status, dataset, metric, finite value, sample/evaluator hashes, seed, and raw artifact hash. The deterministic classifier—not the caller—derives method evaluability and outcome. Dataset coverage, required phases, command completion, hashes, raw artifacts, attempt state, and TrialResult completeness must agree.

## Standard Five-Variant Loop

The reducer maintains `0 <= consumed`, `0 <= reserved`, and `consumed + reserved <= target == 5`, and enforces `execution_width=1`.

1. One unchanged direction runs five sequential, semantically unique standard variants.
2. Patch/static/activation failures, resource pauses, and OOM retries do not consume outcomes; their reservation remains until repair or explicit abandonment.
3. Only a verified `method_evaluable=true` TrialResult consumes one slot.
4. Outcomes 1–4 always route `PROPOSE_NEXT_VARIANT`, including accepted outcomes.
5. Outcome 5 atomically creates an aggregate and closes the direction. Any accepted variant yields `FINISH_DIRECTION`; five non-accepted outcomes yield `START_NEW_DIRECTION`.
6. No sixth reservation, patch, run, or outcome can be created. A closed semantic direction cannot be silently reopened.
7. Aggregate selection uses the preregistered primary objective and deterministic tie-break when comparable; otherwise selection is explicitly `inconclusive`.

Scientific tried history contains only standard budget-consuming method outcomes for duplicate prevention. Planner feedback references prior attempt IDs but never transfers an earlier attempt's outcome to a new variant.

## Bootstrap Profile

Bootstrap performs exactly one cached-S0 cheap proxy with `attempt_kind=bootstrap_proxy` and `consumes_direction_budget=false`. Only a verified, method-evaluable proxy can route `FINISH_RUN`; resource, implementation, and integrity failures route `PAUSE_RESOURCE`, `REPAIR_IMPLEMENTATION`, and `BLOCK_INTEGRITY`. Switching the same project to standard creates a separate standard attempt and begins with `consumed=0`.

## Removed Contracts

There is no v1 dual-read, mirror, migration, fallback, or inference from `ideas`, `plan.yaml`, mutable summaries, or route files. Removed runtime authorities include `literature/ideas.json`, both `direction_decision.json` paths, `plan/candidate_ideas.json`, both `next_variant.json` paths, legacy direction/variant readers, legacy route fallback, legacy attempt ledger, and C2C debate execution. Missing v3/v4 authority requires rerunning from S1.

## M1.1.1 Lifecycle and Commit Hardening

- Every attempt carries `lifecycle_generation`. Implementation revision and resource resume advance the generation while preserving attempt, direction, and variant identity.
- Transition and disposition identities bind attempt ID, generation, implementation/input hashes, expected source state, requested operation or failure, and phase. Replays are idempotent only for the same generation and operation.
- `AttemptDispositioned` stores structured failure facts; `AttemptFinalized` stores a validated TrialResult. Callers cannot supply target state, budget changes, route actions, aggregate, or direction status.
- The reducer independently derives all RouteOutcome fields and late replay returns the route associated with the original event, never the global latest route.
- TrialResult validation is repeated inside the same `BEGIN IMMEDIATE` transaction that appends finalization. Identity, frozen TrialSpec, observations, seeds, datasets, roles, phases, raw artifacts, protocol evidence, budget, and duplicate semantic variants are checked again.
- Projection writers take a project-level exclusive lock, reread the latest committed SQLite state, and never publish an older sequence.
