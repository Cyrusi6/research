# Authoritative S1-S3 Contracts

## Canonical Artifacts

| Concept | Version | Canonical path | Purpose |
|---|---|---|---|
| Direction | `DirectionSpec v2` | `literature/direction.json` | Research question, mechanism invariants, falsification conditions, benchmark identity, and legal variant space |
| Variant | `VariantSpec v3` | `plan/variant.json` | One concrete intervention inside the current direction |
| Attempt | `AttemptRecord v1` | `meta/attempts/<attempt_id>.json` | Event-derived implementation/execution lifecycle and budget reservation |
| Result | `TrialResult v1` | `experiment/results/trial_result.json` | Attempt-bound proxy/full/ablation observations and outcome classification |
| Route | `RouteOutcome v1` | `meta/route_outcome.json` | Unified next action, reasons, budget snapshot, hashes, and idempotency key |
| Events | event ledger v1 | `meta/research_events/*.json` | Immutable source of state transitions and budget accounting |

`meta/research_state.json` is a deterministic snapshot rebuilt from the event ledger. Planner candidate pools, scorecards, Gate reports, patch reports, proxy reports, and human-readable result summaries are diagnostics, not alternate truth sources.

## Identity Boundaries

- `direction_hash` excludes the renameable `direction_id` and fingerprints the causal mechanism, invariant mediator, and benchmark-level research identity.
- `variant_spec_hash` includes the direction hash, intervention, algorithm operations, concrete configuration values, variation coordinates, controls, ablation, and expected metric signature.
- `implementation_hash` fingerprints the frozen patch/diff, actual file contents, and implementation manifest. Repair keeps `variant_id` and `variant_spec_hash` unchanged while changing this hash.
- `attempt_input_hash` binds implementation, protocol, datasets/sample manifest, seeds, runtime configuration, and evaluator.
- Canonical hashing sorts object keys. Field order is irrelevant; intervention and configuration changes are hash-sensitive.

## Attempt Lifecycle

Allowed states are `PLANNED`, `IMPLEMENTING`, `IMPLEMENTATION_REPAIR`, `READY`, `PROXY_RUNNING`, `PROXY_COMPLETED`, `FULL_RUNNING`, `METHOD_COMPLETED`, `METHOD_FAILED`, `RESOURCE_PAUSED`, and `INTEGRITY_BLOCKED`.

Reservation, completion, release, and consumption run through one reducer. Events have unique `event_id` values and monotonic sequence numbers, are committed atomically, and are idempotent. Crash/resume after reservation reuses the same `attempt_id`; replaying an event cannot reserve or consume twice.

## Standard Five-Variant Loop

1. S1 emits one DirectionSpec.
2. S2 proposes one VariantSpec at a time (`execution_width=1`).
3. Reservation checks the direction's five-slot budget.
4. Implementation failures, static failures, activation failures, resource pauses, and OOM retries do not consume a slot.
5. A classified real method result consumes one slot exactly when `method_evaluable=true`.
6. Outcomes 1–4 always produce `PROPOSE_NEXT_VARIANT`, even after success.
7. The fifth outcome immediately produces `FINISH_DIRECTION` if any of the five met acceptance, otherwise `START_NEW_DIRECTION`.
8. The reducer refuses a sixth reservation.

Scientific tried history contains only method-evaluable outcomes. Planner rejection and implementation history remain separate. A new variant may cite `feedback_from_attempt_ids`, but an earlier attempt's failure never becomes the new variant's outcome.

## Bootstrap Profile

Bootstrap performs one cached-S0 cheap-proxy traversal. The attempt uses `attempt_kind=bootstrap_proxy` and `consumes_direction_budget=false`, emits `bootstrap_proxy_complete`, and finishes the run without entering the standard five-variant loop.

## Strict Validation

The five authoritative contracts use JSON Schema Draft 2020-12 with fixed schema versions, required identities and lineage, constrained enums/ranges/patterns, and `additionalProperties=false`. Gates recompute hashes, identity relationships, and budget state rather than trusting self-reported pass flags. Missing required stage inputs block before agent execution and before writes.

## Removed Contracts

This is an incompatible switch. Runtime support was removed for:

- `literature/ideas.json`
- `literature/direction_decision.json`
- `literature/c2c/direction_decision.json`
- `plan/candidate_ideas.json`
- `plan/next_variant.json`
- `plan/s2_planner/next_variant.json`
- legacy v1 direction/variant readers
- legacy attempt ledger, route decision, route fallback, and direction-to-idea conversion
- direction/variant inference from old ideas or `plan.yaml`
- the legacy C2C debate execution path

An old project missing canonical v2/v3 artifacts must rerun from S1. The runtime does not mirror, migrate, infer, or repair old artifacts.
