# Regression Test Migration: 71d3875 → 80d6b7e → M1.1

The `71d38750..80d6b7e` range removed or substantially reduced 98 named tests. M1.1 does not restore their legacy readers, route policy, schemas, or artifact mirrors. Valid behavior is covered through the current canonical contracts and SQLite Event v2 state layer.

## Migration Classes

| Removed behavior class | Still required? | Canonical M1.1 coverage | Migration decision |
|---|---|---|---|
| Attempt counters, repair/resource separation, force-new-direction routing | Yes | `tests/test_authoritative_state_machine.py`, `tests/test_attempt_ledger.py`, `tests/test_route_policy.py` | Rewritten as reducer invariants, explicit dispositions, reservations, abandonment, and deterministic RouteOutcome tests. No independent route counters remain. |
| Patch repair, selected-patch locking, activation forward probe, runtime smoke, checkpoint/OOM recovery | Yes | Existing `tests/test_c2c.py`, `tests/test_c2c_execution_hooks.py`, `tests/test_c2c_manifest_hash_strict.py`; full-suite regression | Kept executable behavior. Failures are classified structurally and cannot consume the five-outcome budget. |
| Proxy/full execution, all-zero/neutral proxy, ablation, paired baseline, readiness, artifact lock | Yes | Existing C2C proxy, hook, manifest, Gate, and pipeline tests plus strict TrialResult tests | Retained as execution/Gate evidence. Routing authority moved from mutable proxy/main-results files to the committed attempt transaction. |
| Codex S2/S2.5 session persistence and duplicate session recovery | Yes | Existing C2C planner and code-patch persistence tests in `tests/test_c2c.py` | Retained. A repeated scientific variant is rejected; IDs can no longer disguise duplication. |
| Snapshot pollution, replay, stale route-invalidated artifacts | Yes | Event tamper, crash-before-projection, rebuild, audit, and replay tests | Replaced mutable snapshot authority with hash-chained SQLite events and rebuildable projections. |
| S1/S2/S3 negative Gate paths and strict falsifiability/ablation fields | Yes | `tests/test_validators.py`, `tests/test_stage_contracts.py`, strict schema tests | Rewritten against DirectionSpec v3, VariantSpec v4, TrialResult v2, and pre-agent missing-input blocking. |
| Feedback attribution and adaptive history | Yes | State-machine semantic duplicate tests and existing `tests/test_s2_feedback_policy.py` | Method history reads only verified standard outcomes; implementation/resource history and planner feedback remain separate. |
| Legacy `route_policy` branch-by-branch decisions | No | Unified reducer route tests | Obsolete because multiple route authorities caused conflicting counters and non-atomic decisions. `RouteOutcome v2` is reducer-derived. |
| Legacy direction/idea fallback, v1 variant loading, candidate/next-variant execution inputs | No | Runtime `rg` assertion in `tests/test_authoritative_state_machine.py` | Obsolete under the breaking canonical switch. Old workspaces must rerun from S1. |
| C2C debate timeout/fallback and compatibility-generated candidates | No | None | Obsolete because the legacy debate execution path and compatibility candidate source were removed. Restoring these tests would require forbidden legacy producers/readers. |
| Old result-summary routing and archived-route fallback | No | TrialResult/RouteOutcome transaction tests | Obsolete because `main_results.json`, proxy decisions, and route projections are not authorities. |

## Named Test Inventory

The 98 deleted names were audited from the Git diff. They fall into the classes above: 25 S1/S2/S3 Gate contract tests, 21 S3 proxy/full routing tests, 16 C2C planner/debate/session tests, 10 legacy route-policy tests, 8 patch/activation/resource tests, 6 replay/audit/reporting tests, 5 feedback/adaptive-history tests, and 7 attempt/result/contract tests.

The M1.1 full suite is the migration acceptance criterion. Tests are neither skipped nor weakened to preserve old artifacts; assertions now target Event v2 transactions, semantic/spec identity, typed observations, strict Gate behavior, and the five-outcome reducer invariant.
