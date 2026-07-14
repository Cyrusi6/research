from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.domain_contracts import (
    EXECUTION_OBSERVATION_SCHEMA_VERSION,
    attempt_input_hash,
    build_direction_spec,
    build_variant_spec,
    canonical_hash,
    classify_trial_result,
    implementation_hash,
)
from auto_research.research_state import IntegrityError, ResearchEventLedger, build_route_outcome
from auto_research.utils import write_json


def _direction() -> dict:
    return build_direction_spec(
        {
            "direction_id": "direction-event-v2",
            "research_question": "Does the intervention improve accuracy?",
            "mechanism_invariants": {
                "causal_hypothesis": "The intervention improves the target mediator.",
                "target_mediator": "routing_quality",
                "invariants": ["same benchmark", "same mediator"],
            },
            "falsification_conditions": ["accuracy does not improve"],
            "support_claim_ids": ["support-1"],
            "counter_claim_ids": ["counter-1"],
            "implementation_surface_ids": ["src/router.py"],
            "metric_signature": {"primary": "accuracy", "direction": "increase"},
            "benchmark_contract_hash": canonical_hash({"datasets": ["fake"]}),
            "variant_space": {
                "mutable_axes": ["intervention"],
                "immutable_axes": ["benchmark"],
                "forbidden_combinations": [{"intervention": "forbidden"}],
            },
            "s2_entry_conditions": ["S1 gate passes"],
            "return_to_s1_conditions": ["five outcomes reject the direction"],
            "lineage": {
                "s1_run_id": "s1-run",
                "iteration": 1,
                "input_manifest_hash": canonical_hash({"input": 1}),
            },
        }
    )


def _variant(direction: dict, index: int) -> dict:
    return build_variant_spec(
        direction,
        {
            "variant_id": f"variant-{index}",
            "variation_coordinates": {"intervention": {"strength": index}},
            "intervention": {
                "summary": f"Apply intervention {index}",
                "algorithm_operations": [f"operation-{index}"],
                "configuration": {"strength": index},
            },
            "hypothesis": f"Intervention {index} improves accuracy.",
            "null_hypothesis": f"Intervention {index} does not improve accuracy.",
            "alternative_hypothesis": f"Intervention {index} improves accuracy.",
            "controlled_variables": {"dataset": "fake", "seed_policy": "fixed"},
            "nuisance_variables": ["gpu_noise"],
            "implementation_surface_ids": ["src/router.py"],
            "expected_metric_signature": {"primary": "accuracy", "direction": "increase"},
            "falsification_conditions": ["accuracy does not improve"],
            "ablation": {"switch": f"disable_{index}"},
            "resource_budget": {"max_wall_seconds": 60, "max_retries": 2},
            "failure_routing": {"implementation": "REPAIR_IMPLEMENTATION", "method": "PROPOSE_NEXT_VARIANT"},
            "lineage": {
                "s2_run_id": f"s2-run-{index}",
                "iteration": index,
                "direction_spec_hash": direction["direction_spec_hash"],
                "feedback_from_attempt_ids": [],
            },
        },
    )


def _attempt_values(variant: dict, revision: int = 1) -> dict:
    protocol = {"required_phases": ["full"], "terminal_method_phases": ["full"]}
    sample_manifest = {"datasets": ["fake"]}
    runtime_config = {"batch": 1}
    evaluator_hash = canonical_hash({"evaluator": 1})
    implementation = implementation_hash(
        frozen_patch={"variant": variant["variant_id"], "revision": revision},
        files={"src/router.py": f"revision-{revision}"},
        manifest={"revision": revision},
    )
    return {
        "implementation_hash": implementation,
        "attempt_input_hash": attempt_input_hash(
            implementation_hash_value=implementation,
            protocol=protocol,
            sample_manifest=sample_manifest,
            seeds=[1],
            runtime_config=runtime_config,
            evaluator_hash=evaluator_hash,
        ),
        "protocol_hash": canonical_hash(protocol),
        "sample_manifest_hash": canonical_hash(sample_manifest),
        "runtime_config_hash": canonical_hash(runtime_config),
        "evaluator_hash": evaluator_hash,
        "seeds": [1],
        "required_datasets": ["fake"],
        "required_phases": ["full"],
        "terminal_method_phases": ["full"],
        "required_roles": ["baseline", "candidate"],
        "require_complete_seed_coverage": False,
    }


def _reserve(ledger: ResearchEventLedger, direction: dict, variant: dict, *, profile: str = "standard") -> dict:
    ledger.select_direction(direction)
    ledger.plan_variant(variant)
    values = _attempt_values(variant)
    attempt_kind = "proxy_full"
    phase = "proxy" if profile == "bootstrap" else "full"
    write_json(
        ledger.project_root / "plan" / "trial_spec.json",
        {
            "protocol": {"required_phases": [phase], "terminal_method_phases": [phase]},
            "sample_manifest": {"datasets": ["fake"]},
            "datasets": ["fake"],
            "metrics": [{"name": "accuracy", "primary": True, "higher_is_better": True}],
            "acceptance_criteria": {"minimum_mean_delta": 0.05},
        },
    )
    if profile == "bootstrap":
        protocol = {"required_phases": ["proxy"], "terminal_method_phases": ["proxy"]}
        values["protocol_hash"] = canonical_hash(protocol)
        values["attempt_input_hash"] = attempt_input_hash(
            implementation_hash_value=values["implementation_hash"],
            protocol=protocol,
            sample_manifest={"datasets": ["fake"]},
            seeds=[1],
            runtime_config={"batch": 1},
            evaluator_hash=values["evaluator_hash"],
        )
        values["required_phases"] = ["proxy"]
        values["terminal_method_phases"] = ["proxy"]
        attempt_kind = "bootstrap_proxy"
    return ledger.reserve_attempt(
        profile=profile,
        direction=direction,
        variant=variant,
        attempt_kind=attempt_kind,
        **values,
    )


def _revision_values(attempt: dict, variant: dict, revision: int) -> dict:
    phase = "proxy" if attempt["attempt_kind"] == "bootstrap_proxy" else "full"
    protocol = {"required_phases": [phase], "terminal_method_phases": [phase]}
    implementation = implementation_hash(
        frozen_patch={"variant": variant["variant_id"], "revision": revision},
        files={"src/router.py": f"revision-{revision}"},
        manifest={"revision": revision},
    )
    return {
        "implementation_hash": implementation,
        "attempt_input_hash": attempt_input_hash(
            implementation_hash_value=implementation,
            protocol=protocol,
            sample_manifest={"datasets": ["fake"]},
            seeds=[1],
            runtime_config={"batch": 1},
            evaluator_hash=attempt["evaluator_hash"],
        ),
        "protocol_hash": attempt["protocol_hash"],
        "sample_manifest_hash": attempt["sample_manifest_hash"],
        "runtime_config_hash": attempt["runtime_config_hash"],
        "evaluator_hash": attempt["evaluator_hash"],
    }


def _trial(ledger: ResearchEventLedger, attempt: dict) -> dict:
    phase = "proxy" if attempt["attempt_kind"] == "bootstrap_proxy" else "full"
    artifact = ledger.project_root / "experiment" / "raw" / f"{attempt['attempt_id']}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"verified": true}\n', encoding="utf-8")
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    observations = [
        {
            "schema_version": EXECUTION_OBSERVATION_SCHEMA_VERSION,
            "phase": phase,
            "role": role,
            "command_status": "completed",
            "dataset_id": "fake",
            "metric_id": "accuracy",
            "metric_value": value,
            "sample_manifest_hash": attempt["sample_manifest_hash"],
            "evaluator_hash": attempt["evaluator_hash"],
            "seed": 1,
            "raw_artifact_hash": artifact_hash,
        }
        for role, value in (("baseline", 0.5), ("candidate", 0.7))
    ]
    return classify_trial_result(
        attempt=attempt,
        trial_spec={
            "protocol": {"required_phases": [phase], "terminal_method_phases": [phase]},
            "sample_manifest": {"datasets": ["fake"]},
            "datasets": ["fake"],
            "metrics": [{"name": "accuracy", "primary": True, "higher_is_better": True}],
            "acceptance_criteria": {"minimum_mean_delta": 0.05},
        },
        observations=observations,
        raw_artifacts={str(artifact.relative_to(ledger.project_root)): artifact_hash},
    )


def _start_execution(ledger: ResearchEventLedger, attempt: dict) -> None:
    phase = "proxy" if attempt["attempt_kind"] == "bootstrap_proxy" else "full"
    state = "PROXY_RUNNING" if phase == "proxy" else "FULL_RUNNING"
    ledger.transition_attempt(attempt["attempt_id"], state, phase=phase, phase_state="RUNNING")


def _complete(ledger: ResearchEventLedger, attempt: dict) -> tuple[dict, dict]:
    trial = _trial(ledger, attempt)
    _start_execution(ledger, attempt)
    return ledger.complete_attempt(trial)


def test_repair_revision_can_reenter_same_proxy_running_transition(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    attempt = _reserve(ledger, direction, variant, profile="bootstrap")

    _start_execution(ledger, attempt)
    ledger.disposition_attempt(attempt["attempt_id"], failure_class="activation_failure")
    repaired = ledger.revise_implementation(attempt["attempt_id"], **_revision_values(attempt, variant, revision=2))
    assert repaired["state"] == "READY"

    rerun = ledger.transition_attempt(attempt["attempt_id"], "PROXY_RUNNING", phase="proxy", phase_state="RUNNING")

    assert rerun["state"] == "PROXY_RUNNING"
    assert ledger.state()["attempts"][attempt["attempt_id"]]["state"] == "PROXY_RUNNING"


def test_same_failure_class_in_new_revision_creates_new_disposition_event(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    attempt = _reserve(ledger, direction, variant, profile="bootstrap")
    _start_execution(ledger, attempt)
    ledger.disposition_attempt(attempt["attempt_id"], failure_class="activation_failure")
    first_count = len(ledger.events())

    ledger.revise_implementation(attempt["attempt_id"], **_revision_values(attempt, variant, revision=2))
    ledger.transition_attempt(attempt["attempt_id"], "PROXY_RUNNING", phase="proxy", phase_state="RUNNING")
    failed, route = ledger.disposition_attempt(attempt["attempt_id"], failure_class="activation_failure")

    assert len(ledger.events()) == first_count + 3
    assert failed["state"] == "IMPLEMENTATION_REPAIR"
    assert route["source"]["attempt_id"] == attempt["attempt_id"]


def test_late_finalize_replay_returns_original_attempt_route(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    first = _reserve(ledger, direction, _variant(direction, 1))
    first_trial = _trial(ledger, first)
    _start_execution(ledger, first)
    _, first_route = ledger.complete_attempt(first_trial)

    second = _reserve(ledger, direction, _variant(direction, 2))
    _complete(ledger, second)
    _, replayed_route = ledger.complete_attempt(first_trial)

    assert replayed_route == first_route
    assert replayed_route["source"]["attempt_id"] == first["attempt_id"]


def test_late_disposition_replay_returns_original_attempt_route(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    first = _reserve(ledger, direction, _variant(direction, 1), profile="bootstrap")
    _start_execution(ledger, first)
    _, first_route = ledger.disposition_attempt(first["attempt_id"], failure_class="activation_failure")

    second = _reserve(ledger, direction, _variant(direction, 2), profile="bootstrap")
    _start_execution(ledger, second)
    ledger.disposition_attempt(second["attempt_id"], failure_class="resource_paused")
    _, replayed_route = ledger.disposition_attempt(first["attempt_id"], failure_class="activation_failure")

    assert replayed_route == first_route
    assert replayed_route["source"]["attempt_id"] == first["attempt_id"]


@pytest.mark.parametrize(
    "mutation",
    ["next_action", "source_event_id", "budget_snapshot", "idempotency_key"],
)
def test_low_level_append_rejects_forged_finalization_route(tmp_path: Path, mutation: str) -> None:
    ledger = ResearchEventLedger(tmp_path / mutation)
    direction = _direction()
    attempt = _reserve(ledger, direction, _variant(direction, 1))
    trial = _trial(ledger, attempt)
    _start_execution(ledger, attempt)
    state = ledger.state()
    preview = deepcopy(state)
    preview["directions"][attempt["direction_semantic_hash"]]["budget"] = {"target": 5, "reserved": 0, "consumed": 1}
    event_id = f"forged-finalize:{mutation}"
    route = build_route_outcome(preview, "PROPOSE_NEXT_VARIANT", ["verified_outcome_recorded", "direction_budget_remaining"], attempt, source_event_id=event_id)
    if mutation == "next_action":
        route["next_action"] = "FINISH_RUN"
    elif mutation == "source_event_id":
        route["source"]["event_id"] = "wrong:event"
    elif mutation == "budget_snapshot":
        route["budget_snapshot"] = {"target": 5, "reserved": 0, "consumed": 4}
    else:
        route["idempotency_key"] = "f" * 64

    with pytest.raises(IntegrityError):
        ledger.append(
            "AttemptFinalized",
            {"trial_result": trial, "route_outcome": route, "aggregate": None},
            event_id=event_id,
        )


def test_low_level_append_rejects_failure_target_and_action_mismatch(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    attempt = _reserve(ledger, direction, _variant(direction, 1), profile="bootstrap")
    _start_execution(ledger, attempt)
    event_id = "forged-disposition:mismatch"
    route = build_route_outcome(ledger.state(), "PAUSE_RESOURCE", ["activation_failure"], attempt, source_event_id=event_id)

    with pytest.raises(IntegrityError):
        ledger.append(
            "AttemptDispositioned",
            {
                "attempt_id": attempt["attempt_id"],
                "failure_class": "activation_failure",
                "target_state": "RESOURCE_PAUSED",
                "artifact_hashes": {},
                "route_outcome": route,
            },
            event_id=event_id,
        )


@pytest.mark.parametrize(
    "mutation",
    ["next_action", "source_event_id", "budget_snapshot", "idempotency_key"],
)
def test_low_level_append_rejects_forged_disposition_route(tmp_path: Path, mutation: str) -> None:
    ledger = ResearchEventLedger(tmp_path / mutation)
    direction = _direction()
    attempt = _reserve(ledger, direction, _variant(direction, 1), profile="bootstrap")
    _start_execution(ledger, attempt)
    event_id = f"forged-disposition:{mutation}"
    route = build_route_outcome(ledger.state(), "REPAIR_IMPLEMENTATION", ["activation_failure"], attempt, source_event_id=event_id)
    if mutation == "next_action":
        route["next_action"] = "FINISH_RUN"
    elif mutation == "source_event_id":
        route["source"]["event_id"] = "wrong:event"
    elif mutation == "budget_snapshot":
        route["budget_snapshot"] = {"target": 5, "reserved": 0, "consumed": 4}
    else:
        route["idempotency_key"] = "f" * 64

    with pytest.raises(IntegrityError):
        ledger.append(
            "AttemptDispositioned",
            {
                "attempt_id": attempt["attempt_id"],
                "failure_class": "activation_failure",
                "target_state": "IMPLEMENTATION_REPAIR",
                "artifact_hashes": {},
                "route_outcome": route,
            },
            event_id=event_id,
        )


def test_stale_projection_writer_cannot_overwrite_newer_sequence(tmp_path: Path) -> None:
    first = ResearchEventLedger(tmp_path)
    second = ResearchEventLedger(tmp_path)
    first_committed = threading.Event()
    allow_stale_write = threading.Event()
    original_write = first._write_projections

    def delayed_write(state: dict) -> None:
        first_committed.set()
        assert allow_stale_write.wait(timeout=5)
        original_write(state)

    first._write_projections = delayed_write
    thread = threading.Thread(target=lambda: first.append("AuditMarker", {"index": 1}, event_id="audit:one"))
    thread.start()
    assert first_committed.wait(timeout=5)
    second.append("AuditMarker", {"index": 2}, event_id="audit:two")
    allow_stale_write.set()
    thread.join(timeout=5)
    assert not thread.is_alive()

    snapshot = json.loads(first.snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["last_sequence"] == 2
    assert len(first.events()) == 2


def test_three_repair_generations_keep_one_attempt_and_consume_once(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    attempt = _reserve(ledger, direction, variant)

    for revision in (2, 3):
        _start_execution(ledger, ledger.state()["attempts"][attempt["attempt_id"]])
        ledger.disposition_attempt(attempt["attempt_id"], failure_class="activation_failed")
        ledger.revise_implementation(
            attempt["attempt_id"],
            **_revision_values(ledger.state()["attempts"][attempt["attempt_id"]], variant, revision),
        )

    current = ledger.state()["attempts"][attempt["attempt_id"]]
    _start_execution(ledger, current)
    completed, _ = ledger.complete_attempt(_trial(ledger, ledger.state()["attempts"][attempt["attempt_id"]]))

    state = ledger.state()
    budget = state["directions"][direction["direction_semantic_hash"]]["budget"]
    disposition_events = [event for event in ledger.events() if event["event_type"] == "AttemptDispositioned"]
    assert completed["attempt_id"] == attempt["attempt_id"]
    assert completed["variant_spec_hash"] == attempt["variant_spec_hash"]
    assert completed["lifecycle_generation"] == 2
    assert len(completed["implementation_revisions"]) == 3
    assert len({event["event_id"] for event in disposition_events}) == 2
    assert budget == {"target": 5, "reserved": 0, "consumed": 1}
    assert len(state["method_tried_history"]) == 1


def test_concurrent_plan_and_reserve_enforces_single_execution_slot(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    ledger.select_direction(direction)
    write_json(
        ledger.project_root / "plan" / "trial_spec.json",
        {
            "protocol": {"required_phases": ["full"], "terminal_method_phases": ["full"]},
            "sample_manifest": {"datasets": ["fake"]},
            "datasets": ["fake"],
            "metrics": [{"name": "accuracy", "primary": True, "higher_is_better": True}],
            "acceptance_criteria": {"minimum_mean_delta": 0.05},
        },
    )

    def compete(index: int) -> str | None:
        variant = _variant(direction, index)
        local = ResearchEventLedger(tmp_path)
        try:
            local.plan_variant(variant)
            return local.reserve_attempt(
                profile="standard",
                direction=direction,
                variant=variant,
                attempt_kind="proxy_full",
                **_attempt_values(variant),
            )["attempt_id"]
        except IntegrityError:
            return None

    with ThreadPoolExecutor(max_workers=16) as executor:
        attempt_ids = list(executor.map(compete, range(1, 17)))

    winners = [attempt_id for attempt_id in attempt_ids if attempt_id]
    assert len(winners) == 1
    events = ledger.events()
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    state = ledger.rebuild()
    assert sum(1 for item in state["attempts"].values() if item["reserved_slot"]) == 1

    winner = state["attempts"][winners[0]]
    _start_execution(ledger, winner)
    ledger.complete_attempt(_trial(ledger, ledger.state()["attempts"][winner["attempt_id"]]))
    final = ledger.rebuild()
    assert final["directions"][direction["direction_semantic_hash"]]["budget"] == {"target": 5, "reserved": 0, "consumed": 1}
