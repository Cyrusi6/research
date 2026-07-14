from __future__ import annotations

import hashlib
import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.domain_contracts import build_direction_spec, build_variant_spec, canonical_hash
from auto_research.research_state import (
    BreakingSchemaError,
    FAILURE_EVIDENCE_SCHEMA_VERSION,
    RESUME_EVIDENCE_SCHEMA_VERSION,
    IntegrityError,
    ResearchEventLedger,
)


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


def _trial_spec() -> dict:
    runtime = {"device": "cpu", "batch_size": 1}
    return {
        "schema_version": "auto_research_trial_spec_v2",
        "protocol": {"protocol_id": "full-v1", "required_phases": ["full"], "terminal_phases": ["full"], "proxy_terminal_allowed": False, "aggregation": "mean"},
        "sample_manifest": {"manifest_id": "fake-v1", "datasets": ["fake"], "content_hash": canonical_hash({"samples": [1]})},
        "datasets": [{"dataset_id": "fake", "split": "test", "sample_count": 1, "sample_hash": canonical_hash({"sample": 1})}],
        "metrics": [{"metric_id": "accuracy", "objective": "maximize", "aggregation": "mean", "role": "primary"}],
        "primary_metric_id": "accuracy",
        "statistical_testing": {"method": "none", "seeds": [1], "require_complete_seed_coverage": True},
        "required_roles": ["baseline", "candidate"],
        "acceptance_constraints": [{"constraint_id": "primary-delta", "kind": "minimum_mean_delta", "hard": True, "metric_id": "accuracy", "threshold": 0.01, "objective": "maximize"}],
        "execution_contract": {"runtime_config": runtime, "runtime_config_hash": canonical_hash(runtime), "evaluator_hash": canonical_hash({"evaluator": 1}), "command_contract_hash": canonical_hash({"command": "run"})},
        "required_artifacts": ["main_results"], "evidence_requirements": [],
    }


def _reserved(tmp_path: Path) -> tuple[ResearchEventLedger, dict]:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction("direction-a")
    variant = _variant(direction, "variant-a")
    ledger.select_direction(direction)
    ledger.plan_variant(variant)
    attempt = ledger.reserve_attempt(profile="standard", direction=direction, variant=variant, implementation_hash=canonical_hash({"impl": 1}), attempt_kind="full", trial_spec=_trial_spec())
    return ledger, attempt


def _artifact(root: Path, name: str) -> dict:
    path = root / "logs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name, encoding="utf-8")
    return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _failure(root: Path, attempt: dict, index: int, *, failure_class: str = "resource_pause") -> dict:
    return {
        "schema_version": FAILURE_EVIDENCE_SCHEMA_VERSION, "attempt_id": attempt["attempt_id"],
        "lifecycle_generation": attempt["lifecycle_generation"], "implementation_hash": attempt["implementation_hash"], "attempt_input_hash": attempt["attempt_input_hash"],
        "source_state": attempt["state"], "source_phase": "full", "failure_class": failure_class,
        "command_status": "resource_paused", "exit_code": 137, "validator_check": None,
        "artifact": _artifact(root, f"pause-{index}.log"), "details": {"resource_type": "memory", "available": False},
        "reason": "resource unavailable", "observed_at": f"2026-01-01T00:00:0{index}Z",
    }


def _resume(root: Path, attempt: dict, index: int) -> dict:
    return {
        "schema_version": RESUME_EVIDENCE_SCHEMA_VERSION, "attempt_id": attempt["attempt_id"],
        "lifecycle_generation": attempt["lifecycle_generation"], "implementation_hash": attempt["implementation_hash"], "attempt_input_hash": attempt["attempt_input_hash"],
        "resource_type": "memory", "probe_status": "available", "artifact": _artifact(root, f"resume-{index}.json"),
        "observed_at": f"2026-01-01T00:01:0{index}Z",
    }


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
        attempt = ledger.transition_attempt(attempt["attempt_id"], "FULL_RUNNING", phase="full", phase_state="RUNNING")
        paused, route = ledger.disposition_failure(_failure(tmp_path, attempt, index))
        assert paused["state"] == "RESOURCE_PAUSED" and paused["phases"]["full"] == "RUNNING"
        assert route["next_action"] == "PAUSE_RESOURCE" and route["source"]["sequence"] == ledger.events()[-1]["sequence"]
        attempt = ledger.resume_attempt(_resume(tmp_path, paused, index))
        assert attempt["state"] == "READY" and attempt["phases"]["full"] == "PENDING"
    state = ledger.state()
    assert attempt["attempt_id"] in state["attempts"]
    assert attempt["lifecycle_generation"] == 3
    assert state["directions"][attempt["direction_semantic_hash"]]["budget"] == {"target": 5, "reserved": 1, "consumed": 0}
    assert not state["trial_results"]


def test_explicit_disposition_event_id_late_replay_returns_historical_result(tmp_path: Path) -> None:
    ledger, attempt = _reserved(tmp_path)
    attempt = ledger.transition_attempt(attempt["attempt_id"], "FULL_RUNNING", phase="full", phase_state="RUNNING")
    evidence = _failure(tmp_path, attempt, 1)
    historical_attempt, historical_route = ledger.disposition_failure(evidence, event_id="pause-explicit-1")
    historical_query = ledger.query_operation_result("pause-explicit-1")
    assert historical_query["attempt"] == historical_attempt
    assert historical_query["route_outcome"] == historical_route
    assert historical_query["state_sequence"] == historical_route["source"]["sequence"]
    resumed = ledger.resume_attempt(_resume(tmp_path, historical_attempt, 1))
    ledger.transition_attempt(resumed["attempt_id"], "FULL_RUNNING", phase="full", phase_state="RUNNING")
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
    ledger.reserve_attempt(profile="standard", direction=first, variant=first_variant, implementation_hash=canonical_hash({"impl": 1}), attempt_kind="full", trial_spec=_trial_spec())
    before = ledger.events()
    with pytest.raises(IntegrityError, match="project execution_width=1"):
        ledger.reserve_attempt(profile="standard", direction=second, variant=second_variant, implementation_hash=canonical_hash({"impl": 2}), attempt_kind="full", trial_spec=_trial_spec())
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
