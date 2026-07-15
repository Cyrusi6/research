from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.domain_contracts import build_direction_spec, build_variant_spec, canonical_hash
from auto_research.research_state import (
    BreakingSchemaError,
    IntegrityError,
    ResearchEventLedger,
)
from test_authoritative_state_machine import _trial_spec
from test_m113_ledger_closure import (
    _failure_evidence as _canonical_failure_evidence,
    _resume_evidence as _canonical_resume_evidence,
)
from support.authoritative_evidence import start_attempt_phase


def _direction(name: str) -> dict:
    return build_direction_spec({
        "direction_id": name,
        "research_question": "Does the method improve accuracy?",
        "mechanism_invariants": {"causal_hypothesis": f"{name} routing improves", "target_mediator": f"routing-{name}", "invariants": ["benchmark"]},
        "falsification_conditions": ["no improvement"],
        "support_claim_ids": ["s1"], "counter_claim_ids": ["c1"],
        "implementation_surface_ids": ["src/model.py"],
        "metric_signature": {"primary": "accuracy", "direction": "increase"},
        "benchmark_contract_hash": canonical_hash({"dataset": "fake"}),
        "variant_space": {"mutable_axes": ["operation"], "immutable_axes": ["benchmark"], "forbidden_combinations": [{"operation": "forbidden"}]},
        "s2_entry_conditions": ["gate"], "return_to_s1_conditions": ["five rejected"],
        "lineage": {"s1_run_id": name, "iteration": 1, "input_manifest_hash": canonical_hash({"name": name})},
    })


def _variant(direction: dict, name: str) -> dict:
    return build_variant_spec(direction, {
        "variant_id": name, "variation_coordinates": {"operation": name},
        "intervention": {"summary": name, "algorithm_operations": [name], "configuration": {"strength": 1}},
        "hypothesis": "improves", "null_hypothesis": "does not improve", "alternative_hypothesis": "improves",
        "controlled_variables": {"dataset": "fake"}, "nuisance_variables": ["noise"],
        "implementation_surface_ids": ["src/model.py"],
        "expected_metric_signature": {"primary": "accuracy", "direction": "increase"},
        "falsification_conditions": ["no improvement"], "ablation": {"disable": name},
        "resource_budget": {"max_wall_seconds": 60, "max_retries": 3},
        "failure_routing": {"implementation": "REPAIR_IMPLEMENTATION", "method": "PROPOSE_NEXT_VARIANT"},
        "lineage": {"s2_run_id": name, "iteration": 1, "direction_spec_hash": direction["direction_spec_hash"], "feedback_from_attempt_ids": []},
    })


def _reserved(tmp_path: Path) -> tuple[ResearchEventLedger, dict]:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction("direction-a")
    variant = _variant(direction, "variant-a")
    ledger.select_direction(direction)
    ledger.plan_variant(variant)
    attempt = ledger.reserve_attempt(profile="standard", direction=direction, variant=variant, implementation_hash=canonical_hash({"impl": 1}), attempt_kind="full", trial_spec=_trial_spec(project_root=tmp_path))
    return ledger, attempt


def _failure(root: Path, attempt: dict, index: int, *, failure_class: str = "resource_pause") -> dict:
    if failure_class == "resource_pause":
        from test_m115_failure_resume_reducer import _resource_pause
        return _resource_pause(root, attempt)[0]
    return _canonical_failure_evidence(
        root,
        attempt,
        failure_class=failure_class,
        exit_code=137,
        resource_type="system_memory",
    )


def _resume(root: Path, ledger: ResearchEventLedger, attempt: dict, index: int) -> dict:
    from auto_research.failure_validation import canonical_evidence_bytes, evidence_bytes_hash
    from test_m115_failure_resume_reducer import _resume as canonical_resume
    pause_event = next(event for event in reversed(ledger.events()) if event["event_type"] == "AttemptDispositioned")
    pause_failure = pause_event["payload"]["failure_evidence"]
    pause_hash = evidence_bytes_hash(canonical_evidence_bytes(pause_failure))
    return canonical_resume(root, attempt, pause_event, pause_failure, pause_hash)


@pytest.mark.parametrize("forged", ["IMPLEMENTATION_REPAIR", "RESOURCE_PAUSED", "INTEGRITY_BLOCKED", "ABANDONED", "METHOD_COMPLETED"])
def test_public_transition_cannot_forge_authoritative_states(tmp_path: Path, forged: str) -> None:
    ledger, attempt = _reserved(tmp_path)
    before = ledger.events()
    with pytest.raises(IntegrityError, match="cannot enter"):
        ledger.transition_attempt(attempt["attempt_id"], forged)
    assert ledger.events() == before


def test_three_resource_pause_resume_cycles_preserve_attempt_and_reservation(tmp_path: Path) -> None:
    ledger, attempt = _reserved(tmp_path)
    for index in range(3):
        attempt = start_attempt_phase(ledger, attempt, "full")
        paused, route = ledger.disposition_failure(_failure(tmp_path, attempt, index))
        assert paused["state"] == "RESOURCE_PAUSED" and paused["phases"]["full"] == "RUNNING"
        assert route["next_action"] == "PAUSE_RESOURCE" and route["source"]["sequence"] == ledger.events()[-1]["sequence"]
        attempt = ledger.resume_attempt(_resume(tmp_path, ledger, paused, index))
        assert attempt["state"] == "READY" and attempt["phases"]["full"] == "PENDING"
    state = ledger.state()
    assert attempt["attempt_id"] in state["attempts"]
    assert attempt["lifecycle_generation"] == 3
    assert state["directions"][attempt["direction_semantic_hash"]]["budget"] == {"target": 5, "reserved": 1, "consumed": 0}
    assert not state["trial_results"]


def test_explicit_disposition_event_id_late_replay_returns_historical_result(tmp_path: Path) -> None:
    ledger, attempt = _reserved(tmp_path)
    attempt = start_attempt_phase(ledger, attempt, "full")
    evidence = _failure(tmp_path, attempt, 1)
    historical_attempt, historical_route = ledger.disposition_failure(evidence, event_id="pause-explicit-1")
    historical_query = ledger.query_operation_result("pause-explicit-1")
    assert historical_query["attempt"] == historical_attempt
    assert historical_query["route_outcome"] == historical_route
    assert historical_query["state_sequence"] == historical_route["source"]["sequence"]
    resumed = ledger.resume_attempt(_resume(tmp_path, ledger, historical_attempt, 1))
    start_attempt_phase(ledger, resumed, "full")
    before = len(ledger.events())
    replay_attempt, replay_route = ledger.disposition_failure(evidence, event_id="pause-explicit-1")
    assert len(ledger.events()) == before
    assert replay_attempt == historical_attempt and replay_route == historical_route
    changed = deepcopy(evidence); changed["reason"] = "different request"
    with pytest.raises(IntegrityError, match="fingerprint conflict"):
        ledger.disposition_failure(changed, event_id="pause-explicit-1")


def test_project_execution_width_blocks_second_direction_reservation(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    first, second = _direction("first"), _direction("second")
    first_variant, second_variant = _variant(first, "first-v"), _variant(second, "second-v")
    ledger.select_direction(first); ledger.plan_variant(first_variant)
    ledger.select_direction(second); ledger.plan_variant(second_variant)
    ledger.reserve_attempt(profile="standard", direction=first, variant=first_variant, implementation_hash=canonical_hash({"impl": 1}), attempt_kind="full", trial_spec=_trial_spec(project_root=tmp_path))
    before = ledger.events()
    with pytest.raises(IntegrityError, match="project execution_width=1"):
        ledger.reserve_attempt(profile="standard", direction=second, variant=second_variant, implementation_hash=canonical_hash({"impl": 2}), attempt_kind="full", trial_spec=_trial_spec(project_root=tmp_path))
    assert ledger.events() == before


def test_future_and_corrupt_projection_are_quarantined_and_rebuilt(tmp_path: Path) -> None:
    ledger, _ = _reserved(tmp_path)
    authoritative = ledger.state()
    ledger.snapshot_path.write_text(json.dumps({"last_sequence": authoritative["last_sequence"] + 99}), encoding="utf-8")
    assert ledger.state()["last_sequence"] == authoritative["last_sequence"]
    ledger.snapshot_path.write_text("not json", encoding="utf-8")
    assert ledger.state()["last_sequence"] == authoritative["last_sequence"]
    assert len(list(ledger.meta_dir.glob("research_state.json.quarantine.*"))) == 2


def test_old_sqlite_event_schema_is_breaking_error(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    with sqlite3.connect(ledger.db_path) as connection:
        connection.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (1, "legacy", "AuditMarker", '{"index":1}', "0" * 64, "1" * 64, "2026-01-01T00:00:00Z", "auto_research_event_v2"))
    with pytest.raises(BreakingSchemaError, match="unsupported"):
        ResearchEventLedger(tmp_path)
