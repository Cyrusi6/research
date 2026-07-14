from __future__ import annotations

import hashlib
import sqlite3
import subprocess
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
    direction_spec_hash,
    implementation_hash,
    validate_contract,
    validate_trial_result,
    validate_variant_identity,
    variant_semantic_hash,
    variant_spec_hash,
)
from auto_research.research_state import IntegrityError, ResearchEventLedger


def _direction() -> dict:
    return build_direction_spec(
        {
            "direction_id": "direction-alpha",
            "research_question": "Does mediator-aware routing improve the benchmark outcome?",
            "mechanism_invariants": {
                "causal_hypothesis": "Changing mediator-aware routing improves the target metric.",
                "target_mediator": "routing_quality",
                "invariants": ["same mediator", "same benchmark", "same causal mechanism"],
            },
            "falsification_conditions": ["routing quality does not change", "the target metric does not improve"],
            "support_claim_ids": ["support-1"],
            "counter_claim_ids": ["counter-1"],
            "implementation_surface_ids": ["src/router.py"],
            "metric_signature": {"primary": "accuracy", "direction": "increase"},
            "benchmark_contract_hash": canonical_hash({"datasets": ["fake"]}),
            "variant_space": {
                "mutable_axes": ["intervention"],
                "immutable_axes": ["benchmark", "mediator"],
                "forbidden_combinations": [{"intervention": "forbidden"}],
            },
            "s2_entry_conditions": ["S1 gate passes"],
            "return_to_s1_conditions": ["five outcomes reject the direction"],
            "lineage": {"s1_run_id": "s1-run", "iteration": 1, "input_manifest_hash": canonical_hash({"input": 1})},
        }
    )


def _variant(direction: dict, index: int, feedback: list[str] | None = None) -> dict:
    return build_variant_spec(
        direction,
        {
            "variant_id": f"variant-{index}",
            "variation_coordinates": {"intervention": {"operation": f"operation-{index}", "strength": index}},
            "intervention": {
                "summary": f"Apply operation {index}",
                "algorithm_operations": [f"operation-{index}"],
                "configuration": {"strength": index},
            },
            "hypothesis": f"Operation {index} improves accuracy.",
            "null_hypothesis": f"Operation {index} does not improve accuracy.",
            "alternative_hypothesis": f"Operation {index} improves accuracy.",
            "controlled_variables": {"dataset": "fake", "seed_policy": "fixed"},
            "nuisance_variables": ["gpu_noise"],
            "implementation_surface_ids": ["src/router.py"],
            "expected_metric_signature": {"primary": "accuracy", "direction": "increase"},
            "falsification_conditions": ["accuracy does not improve"],
            "ablation": {"switch": f"disable_operation_{index}"},
            "resource_budget": {"max_wall_seconds": 60, "max_retries": 2},
            "failure_routing": {"implementation": "REPAIR_IMPLEMENTATION", "method": "PROPOSE_NEXT_VARIANT"},
            "lineage": {
                "s2_run_id": f"s2-run-{index}",
                "iteration": index,
                "direction_spec_hash": direction["direction_spec_hash"],
                "feedback_from_attempt_ids": feedback or [],
            },
        },
    )


def _initialize(ledger: ResearchEventLedger, direction: dict, variant: dict) -> None:
    ledger.select_direction(direction)
    ledger.plan_variant(variant, feedback_from_attempt_ids=(variant.get("lineage") or {}).get("feedback_from_attempt_ids") or [])


def _attempt_inputs(variant: dict) -> dict:
    protocol = {"required_phases": ["full"], "terminal_method_phases": ["full"]}
    sample_manifest = {"datasets": ["fake"]}
    runtime_config = {"batch": 1}
    evaluator = canonical_hash({"evaluator": 1})
    implementation = implementation_hash(
        frozen_patch={"variant": variant["variant_id"]},
        files={"src/router.py": variant["variant_id"]},
        manifest={"v": 1},
    )
    return {
        "implementation_hash": implementation,
        "attempt_input_hash": attempt_input_hash(
            implementation_hash_value=implementation,
            protocol=protocol,
            sample_manifest=sample_manifest,
            seeds=[1],
            runtime_config=runtime_config,
            evaluator_hash=evaluator,
        ),
        "protocol_hash": canonical_hash(protocol),
        "sample_manifest_hash": canonical_hash(sample_manifest),
        "runtime_config_hash": canonical_hash(runtime_config),
        "evaluator_hash": evaluator,
        "seeds": [1],
    }


def _reserve(ledger: ResearchEventLedger, direction: dict, variant: dict, *, profile: str = "standard") -> dict:
    values = _attempt_inputs(variant)
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
    return ledger.reserve_attempt(
        profile=profile,
        direction=direction,
        variant=variant,
        attempt_kind="bootstrap_proxy" if profile == "bootstrap" else "proxy_full",
        **values,
    )


def _trial_spec(attempt: dict) -> dict:
    phase = "proxy" if attempt["attempt_kind"] == "bootstrap_proxy" else "full"
    return {
        "protocol": {"required_phases": [phase], "terminal_method_phases": [phase]},
        "sample_manifest": {"datasets": ["fake"]},
        "datasets": ["fake"],
        "metrics": [{"name": "accuracy", "primary": True, "higher_is_better": True}],
        "acceptance_criteria": {"minimum_mean_delta": 0.05},
    }


def _complete(
    ledger: ResearchEventLedger,
    attempt: dict,
    *,
    outcome: str,
    evaluable: bool = True,
    failure: str | None = None,
):
    if not evaluable:
        return ledger.disposition_attempt(attempt["attempt_id"], failure_class=failure or "implementation_failure")
    artifact = ledger.project_root / "experiment" / "raw" / f"{attempt['attempt_id']}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"verified": true}\n', encoding="utf-8")
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    phase = "proxy" if attempt["attempt_kind"] == "bootstrap_proxy" else "full"
    candidate = 0.7 if outcome == "accepted" else 0.51
    observations = []
    for role, value in [("baseline", 0.5), ("candidate", candidate)]:
        observations.append(
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
        )
    trial = classify_trial_result(
        attempt=attempt,
        trial_spec=_trial_spec(attempt),
        observations=observations,
        raw_artifacts={str(artifact.relative_to(ledger.project_root)): artifact_hash},
    )
    if phase == "proxy":
        ledger.transition_attempt(attempt["attempt_id"], "PROXY_RUNNING", phase="proxy", phase_state="RUNNING")
    else:
        ledger.transition_attempt(attempt["attempt_id"], "FULL_RUNNING", phase="full", phase_state="RUNNING")
    return ledger.complete_attempt(trial)


def test_five_outcomes_keep_direction_identity_and_never_create_sixth(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    attempts = []
    for index in range(1, 6):
        variant = _variant(direction, index)
        _initialize(ledger, direction, variant)
        attempt = _reserve(ledger, direction, variant)
        attempts.append(attempt)
        _, route = _complete(ledger, attempt, outcome="accepted")
        assert attempt["direction_spec_hash"] == direction["direction_spec_hash"]
        assert route["next_action"] == ("PROPOSE_NEXT_VARIANT" if index < 5 else "FINISH_DIRECTION")
    state = ledger.state()
    budget = state["directions"][direction["direction_semantic_hash"]]["budget"]
    assert budget == {"target": 5, "reserved": 0, "consumed": 5}
    assert len({item["variant_semantic_hash"] for item in state["method_tried_history"]}) == 5
    with pytest.raises(IntegrityError, match="closed direction"):
        ledger.select_direction(direction, event_id="direction:reopen")


def test_same_scientific_variant_with_new_id_and_lineage_is_duplicate(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    first = _variant(direction, 1)
    _initialize(ledger, direction, first)
    _complete(ledger, _reserve(ledger, direction, first), outcome="rejected")
    duplicate = deepcopy(first)
    duplicate["variant_id"] = "renamed"
    duplicate["lineage"] = {**duplicate["lineage"], "iteration": 999, "s2_run_id": "new-run"}
    duplicate.pop("variant_spec_hash")
    duplicate["variant_spec_hash"] = variant_spec_hash(duplicate)
    assert duplicate["variant_semantic_hash"] == first["variant_semantic_hash"]
    ledger.select_direction(direction)
    with pytest.raises((IntegrityError, ValueError), match="duplicate"):
        ledger.plan_variant(duplicate)


def test_event_id_idempotency_and_conflict(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    first, _ = ledger.append("AuditMarker", {"index": 1}, event_id="audit:one")
    repeated, _ = ledger.append("AuditMarker", {"index": 1}, event_id="audit:one")
    assert first == repeated
    assert len(ledger.events()) == 1
    with pytest.raises(IntegrityError, match="conflict"):
        ledger.append("AuditMarker", {"index": 2}, event_id="audit:one")


def test_concurrent_append_has_unique_contiguous_sequences(tmp_path: Path) -> None:
    def append(index: int) -> None:
        ResearchEventLedger(tmp_path).append("AuditMarker", {"index": index}, event_id=f"audit:{index}")

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(append, range(100)))
    events = ResearchEventLedger(tmp_path).events()
    assert [event["sequence"] for event in events] == list(range(1, 101))
    assert len({event["event_id"] for event in events}) == 100


def test_rebuild_rejects_hash_chain_tampering(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    ledger.append("AuditMarker", {"index": 1}, event_id="audit:1")
    ledger.append("AuditMarker", {"index": 2}, event_id="audit:2")
    with sqlite3.connect(ledger.db_path) as connection:
        connection.execute("UPDATE events SET previous_event_hash = ? WHERE sequence = 2", ("f" * 64,))
    with pytest.raises(IntegrityError, match="hash chain"):
        ledger.rebuild()


def test_commit_crash_before_projection_rebuilds_unique_result_and_route(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    def crash_after_finalization() -> None:
        if ledger.events()[-1]["event_type"] == "AttemptFinalized":
            raise RuntimeError("crash after commit")

    ledger.after_commit_hook = crash_after_finalization
    with pytest.raises(RuntimeError, match="crash after commit"):
        _complete(ledger, attempt, outcome="rejected")
    state = ResearchEventLedger(tmp_path).rebuild()
    assert list(state["trial_results"]) == [attempt["attempt_id"]]
    assert state["last_route_outcome"]["source"]["attempt_id"] == attempt["attempt_id"]
    assert state["directions"][direction["direction_semantic_hash"]]["budget"]["consumed"] == 1


def test_trial_identity_mismatch_writes_nothing(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    before_events = len(ledger.events())
    before_state = ledger.state()
    artifact = tmp_path / "raw.json"
    artifact.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    observation = {
        "schema_version": EXECUTION_OBSERVATION_SCHEMA_VERSION,
        "phase": "full", "role": "candidate", "command_status": "completed", "dataset_id": "fake",
        "metric_id": "accuracy", "metric_value": 0.7, "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"], "seed": 1, "raw_artifact_hash": digest,
    }
    trial = classify_trial_result(attempt=attempt, trial_spec=_trial_spec(attempt), observations=[observation], raw_artifacts={"raw.json": digest})
    trial["attempt_input_hash"] = "f" * 64
    with pytest.raises(IntegrityError, match="attempt_input_hash mismatch"):
        ledger.complete_attempt(trial)
    assert len(ledger.events()) == before_events
    after_state = ledger.state()
    assert after_state["directions"] == before_state["directions"]
    assert after_state["last_route_outcome"] == before_state["last_route_outcome"]


def test_fingerprint_stability_and_sensitivity() -> None:
    direction = _direction()
    variant = _variant(direction, 1)
    reordered = {key: variant[key] for key in reversed(list(variant))}
    assert variant_spec_hash(variant) == variant_spec_hash(reordered)
    changed = deepcopy(variant)
    changed["intervention"]["configuration"]["strength"] = 99
    assert variant_semantic_hash(changed) != variant["variant_semantic_hash"]
    assert variant_spec_hash(changed) != variant["variant_spec_hash"]
    changed_direction = deepcopy(direction)
    changed_direction["research_question"] += " Precisely?"
    assert direction_spec_hash(changed_direction) != direction["direction_spec_hash"]


def test_wrong_lineage_direction_spec_hash_fails() -> None:
    direction = _direction()
    variant = _variant(direction, 1)
    variant["lineage"]["direction_spec_hash"] = "f" * 64
    variant["variant_spec_hash"] = variant_spec_hash(variant)
    with pytest.raises(ValueError, match="lineage.direction_spec_hash"):
        validate_variant_identity(direction, variant)


def test_abandoned_attempt_cannot_revive_after_five_outcomes(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    first_variant = _variant(direction, 1)
    _initialize(ledger, direction, first_variant)
    first_attempt = _reserve(ledger, direction, first_variant)
    ledger.disposition_attempt(first_attempt["attempt_id"], failure_class="activation_failure")
    ledger.abandon_attempt(first_attempt["attempt_id"], reason="replace non-evaluable implementation")
    for index in range(2, 7):
        variant = _variant(direction, index)
        _initialize(ledger, direction, variant)
        _complete(ledger, _reserve(ledger, direction, variant), outcome="rejected")
    assert ledger.state()["directions"][direction["direction_semantic_hash"]]["budget"]["consumed"] == 5
    with pytest.raises(IntegrityError, match="already finalized|cannot finalize|no reserved slot|illegal attempt transition"):
        _complete(ledger, first_attempt, outcome="accepted")


def test_illegal_transition_and_terminal_transition_write_no_event(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    count = len(ledger.events())
    with pytest.raises(IntegrityError, match="illegal attempt transition"):
        ledger.transition_attempt(attempt["attempt_id"], "METHOD_COMPLETED")
    assert len(ledger.events()) == count
    completed, _ = _complete(ledger, attempt, outcome="accepted")
    count = len(ledger.events())
    with pytest.raises(IntegrityError, match="illegal attempt transition"):
        ledger.transition_attempt(completed["attempt_id"], "READY")
    assert len(ledger.events()) == count


def test_method_completed_phases_match_trial_completeness(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    completed, _ = _complete(ledger, attempt, outcome="accepted")
    trial = ledger.state()["trial_results"][attempt["attempt_id"]]
    assert completed["state"] == "METHOD_COMPLETED"
    assert completed["phases"][trial["completeness"]] == "COMPLETED"
    assert {item["phase"] for item in trial["observations"] if item["role"] == "candidate"} == {trial["completeness"]}


def test_bootstrap_then_standard_same_variant_has_independent_identity_and_budget(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    bootstrap = _reserve(ledger, direction, variant, profile="bootstrap")
    _complete(ledger, bootstrap, outcome="accepted")
    ledger.select_direction(direction)
    ledger.plan_variant(variant, event_id="variant:standard:same-science")
    standard = _reserve(ledger, direction, variant, profile="standard")
    assert standard["attempt_id"] != bootstrap["attempt_id"]
    state = ledger.state()
    assert state["directions"][direction["direction_semantic_hash"]]["budget"] == {"target": 5, "reserved": 1, "consumed": 0}


def test_implementation_repair_keeps_attempt_and_variant_identity(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    ledger.disposition_attempt(attempt["attempt_id"], failure_class="activation_failure")
    values = _attempt_inputs(variant)
    new_implementation = implementation_hash(frozen_patch={"repair": True}, files={"src/router.py": "repaired"}, manifest={"v": 2})
    protocol = {"required_phases": ["full"], "terminal_method_phases": ["full"]}
    values["implementation_hash"] = new_implementation
    values["attempt_input_hash"] = attempt_input_hash(
        implementation_hash_value=new_implementation,
        protocol=protocol,
        sample_manifest={"datasets": ["fake"]},
        seeds=[1],
        runtime_config={"batch": 1},
        evaluator_hash=values["evaluator_hash"],
    )
    repaired = ledger.reserve_attempt(profile="standard", direction=direction, variant=variant, attempt_kind="proxy_full", **values)
    assert repaired["attempt_id"] == attempt["attempt_id"]
    assert repaired["variant_spec_hash"] == attempt["variant_spec_hash"]
    assert repaired["implementation_hash"] != attempt["implementation_hash"]
    assert len(repaired["implementation_revisions"]) == 2


@pytest.mark.parametrize(
    ("failure_class", "expected_action"),
    [("resource_paused", "PAUSE_RESOURCE"), ("activation_failure", "REPAIR_IMPLEMENTATION"), ("integrity", "BLOCK_INTEGRITY")],
)
def test_bootstrap_failures_never_finish_run(tmp_path: Path, failure_class: str, expected_action: str) -> None:
    ledger = ResearchEventLedger(tmp_path / failure_class)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant, profile="bootstrap")
    failed, route = ledger.disposition_attempt(attempt["attempt_id"], failure_class=failure_class)
    assert route["next_action"] == expected_action
    assert route["next_action"] != "FINISH_RUN"
    assert failed["method_evaluable"] is False
    assert ledger.state()["directions"][direction["direction_semantic_hash"]]["budget"]["consumed"] == 0


def test_evaluable_trial_rejects_empty_evidence_and_none_completeness(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    completed, _ = _complete(ledger, attempt, outcome="accepted")
    valid = ledger.state()["trial_results"][completed["attempt_id"]]
    mutations = []
    for field, value in [("observed_datasets", []), ("observations", []), ("raw_artifacts", {}), ("completeness", "none")]:
        changed = deepcopy(valid)
        changed[field] = value
        mutations.append(changed)
    for changed in mutations:
        with pytest.raises(ValueError):
            validate_trial_result(changed)


@pytest.mark.parametrize("field", ["variant_id", "variation_coordinates", "intervention", "hypothesis", "null_hypothesis", "alternative_hypothesis", "controlled_variables", "nuisance_variables", "implementation_surface_ids", "expected_metric_signature", "falsification_conditions", "ablation", "resource_budget", "failure_routing", "lineage"])
def test_variant_spec_hash_is_sensitive_to_every_authoritative_field(field: str) -> None:
    direction = _direction()
    variant = _variant(direction, 1)
    changed = deepcopy(variant)
    changed.pop("variant_spec_hash")
    changed.pop("variant_semantic_hash")
    value = changed[field]
    if isinstance(value, str):
        changed[field] = value + "-changed"
    elif isinstance(value, list):
        changed[field] = value + ["changed"]
    else:
        changed[field] = {**value, "changed": True}
    assert variant_spec_hash(changed) != variant["variant_spec_hash"]


@pytest.mark.parametrize("field", ["direction_id", "research_question", "mechanism_invariants", "falsification_conditions", "support_claim_ids", "counter_claim_ids", "implementation_surface_ids", "metric_signature", "benchmark_contract_hash", "variant_space", "exploration_policy", "s2_entry_conditions", "return_to_s1_conditions", "lineage"])
def test_direction_spec_hash_is_sensitive_to_every_authoritative_field(field: str) -> None:
    direction = _direction()
    changed = deepcopy(direction)
    changed.pop("direction_spec_hash")
    changed.pop("direction_semantic_hash")
    value = changed[field]
    if isinstance(value, str):
        changed[field] = value + "-changed"
    elif isinstance(value, list):
        changed[field] = value + ["changed"]
    else:
        changed[field] = {**value, "changed": True}
    assert direction_spec_hash(changed) != direction["direction_spec_hash"]


def test_strict_contract_rejects_missing_extra_and_wrong_version() -> None:
    direction = _direction()
    missing = deepcopy(direction)
    missing.pop("research_question")
    extra = {**direction, "legacy": True}
    wrong = {**direction, "schema_version": "old"}
    for payload in [missing, extra, wrong]:
        with pytest.raises(ValueError):
            validate_contract(payload, "direction_v3.schema.json")


def test_generic_core_state_machine_has_no_c2c_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in ["src/auto_research/domain_contracts.py", "src/auto_research/research_state.py"]:
        assert "c2c" not in (root / rel).read_text(encoding="utf-8").lower()


def test_runtime_source_contains_no_removed_legacy_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    patterns = [
        "literature/ideas.json", "literature/direction_decision.json", "literature/c2c/direction_decision.json",
        "direction_to_legacy_idea", "load_direction_or_legacy_idea", "plan/candidate_ideas.json",
        "plan/next_variant.json", "plan/s2_planner/next_variant.json", "legacy_route_fallback",
    ]
    command = ["rg", "-n", "|".join(pattern.replace(".", r"\.") for pattern in patterns), "src/auto_research", "--glob", "*.py"]
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    assert completed.returncode == 1, completed.stdout
