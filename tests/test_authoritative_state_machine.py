from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from auto_research.domain_contracts import (
    attempt_input_hash,
    build_direction_spec,
    build_variant_spec,
    canonical_hash,
    implementation_hash,
    validate_contract,
    variant_spec_hash,
)
from auto_research.research_state import ResearchEventLedger, build_trial_result


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
            "variation_coordinates": {"intervention": f"operation-{index}"},
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
                "direction_spec_hash": direction["direction_hash"],
                "feedback_from_attempt_ids": feedback or [],
            },
        },
    )


def _reserve(ledger: ResearchEventLedger, direction: dict, variant: dict, *, profile: str = "standard") -> dict:
    impl_hash = implementation_hash(frozen_patch={"variant": variant["variant_id"]}, files={"src/router.py": variant["variant_id"]}, manifest={"v": 1})
    input_hash = attempt_input_hash(
        implementation_hash_value=impl_hash,
        protocol={"phase": "proxy_full"},
        sample_manifest={"datasets": ["fake"]},
        seeds=[1],
        runtime_config={"batch": 1},
        evaluator_hash=canonical_hash({"evaluator": 1}),
    )
    return ledger.reserve_attempt(
        profile=profile,
        direction=direction,
        variant=variant,
        implementation_hash=impl_hash,
        attempt_input_hash=input_hash,
        attempt_kind="bootstrap_proxy" if profile == "bootstrap" else "proxy_full",
    )


def _complete(
    ledger: ResearchEventLedger,
    attempt: dict,
    *,
    outcome: str,
    evaluable: bool = True,
    failure: str | None = None,
):
    trial = build_trial_result(
        attempt=attempt,
        protocol_hash=canonical_hash({"protocol": 1}),
        input_hash=attempt["attempt_input_hash"],
        completeness="full" if evaluable else "partial",
        observed_datasets=["fake"] if evaluable else [],
        raw_artifacts={},
        proxy_observations=[{"accuracy": 0.5}] if evaluable else [],
        full_observations=[{"accuracy": 0.6}] if evaluable else [],
        ablation_observations=[],
        method_evaluable=evaluable,
        outcome_classification=outcome,
        failure_classification=failure,
    )
    return ledger.complete_attempt(trial)


def _initialize(ledger: ResearchEventLedger, direction: dict, variant: dict, feedback: list[str] | None = None) -> None:
    ledger.select_direction(direction, event_id=f"direction:{direction['direction_hash']}")
    ledger.plan_variant(variant, feedback_from_attempt_ids=feedback or [], event_id=f"variant:{variant['variant_spec_hash']}")


def test_same_direction_completes_five_distinct_variants_in_sequence(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    attempt_ids = []
    for index in range(1, 6):
        variant = _variant(direction, index)
        _initialize(ledger, direction, variant)
        attempt = _reserve(ledger, direction, variant)
        attempt_ids.append(attempt["attempt_id"])
        _, route = _complete(ledger, attempt, outcome="accepted")
        assert route["next_action"] == ("PROPOSE_NEXT_VARIANT" if index < 5 else "FINISH_DIRECTION")
    state = ledger.state()
    outcomes = state["method_tried_history"]
    assert len(outcomes) == 5
    assert len({item["variant_spec_hash"] for item in outcomes}) == 5
    assert {item["direction_hash"] for item in outcomes} == {direction["direction_hash"]}
    assert len(set(attempt_ids)) == 5


def test_five_successes_do_not_stop_early(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    routes = []
    for index in range(1, 6):
        variant = _variant(direction, index)
        _initialize(ledger, direction, variant)
        _, route = _complete(ledger, _reserve(ledger, direction, variant), outcome="accepted")
        routes.append(route["next_action"])
    assert routes == ["PROPOSE_NEXT_VARIANT"] * 4 + ["FINISH_DIRECTION"]


def test_mixed_success_failure_still_runs_five(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    outcomes = ["rejected", "accepted", "falsified", "accepted", "rejected"]
    routes = []
    for index, outcome in enumerate(outcomes, 1):
        variant = _variant(direction, index)
        _initialize(ledger, direction, variant)
        _, route = _complete(ledger, _reserve(ledger, direction, variant), outcome=outcome)
        routes.append(route["next_action"])
    assert routes[:4] == ["PROPOSE_NEXT_VARIANT"] * 4
    assert routes[-1] == "FINISH_DIRECTION"
    assert ledger.state()["directions"][direction["direction_hash"]]["budget"]["consumed"] == 5


def test_sixth_variant_is_refused_after_fifth_outcome(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    for index in range(1, 6):
        variant = _variant(direction, index)
        _initialize(ledger, direction, variant)
        _complete(ledger, _reserve(ledger, direction, variant), outcome="rejected")
    sixth = _variant(direction, 6)
    with pytest.raises(RuntimeError, match="sixth"):
        _reserve(ledger, direction, sixth)


@pytest.mark.parametrize(
    ("outcome", "failure"),
    [
        ("implementation_failed", "patch_failure"),
        ("activation_failed", "activation_wiring_failure"),
        ("resource_paused", "resource_paused"),
        ("resource_paused", "oom_retry"),
    ],
)
def test_non_evaluable_failures_do_not_consume_direction_budget(tmp_path: Path, outcome: str, failure: str) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    completed, _ = _complete(ledger, attempt, outcome=outcome, evaluable=False, failure=failure)
    budget = ledger.state()["directions"][direction["direction_hash"]]["budget"]
    assert completed["consumes_direction_budget"] is False
    assert budget == {"target": 5, "reserved": 0, "consumed": 0}


def test_implementation_repair_preserves_variant_hash_and_changes_implementation_hash(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    old_implementation = attempt["implementation_hash"]
    repaired = ledger.transition_attempt(
        attempt["attempt_id"],
        "IMPLEMENTATION_REPAIR",
        changes={"implementation_hash": canonical_hash({"repair": 1})},
    )
    assert repaired["attempt_id"] == attempt["attempt_id"]
    assert repaired["variant_spec_hash"] == variant["variant_spec_hash"]
    assert repaired["implementation_hash"] != old_implementation


def test_failure_attribution_stays_with_a_and_only_feedback_links_to_b(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant_a = _variant(direction, 1)
    _initialize(ledger, direction, variant_a)
    attempt_a = _reserve(ledger, direction, variant_a)
    _complete(ledger, attempt_a, outcome="rejected")
    variant_b = _variant(direction, 2, [attempt_a["attempt_id"]])
    _initialize(ledger, direction, variant_b, [attempt_a["attempt_id"]])
    state = ledger.state()
    assert state["variants"][variant_b["variant_spec_hash"]]["feedback_from_attempt_ids"] == [attempt_a["attempt_id"]]
    assert all(item["variant_id"] != variant_b["variant_id"] for item in state["method_tried_history"])
    assert state["trial_results"][attempt_a["attempt_id"]]["variant_id"] == variant_a["variant_id"]


def test_planned_or_patch_rejected_variant_not_in_method_history(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    ledger.transition_attempt(attempt["attempt_id"], "IMPLEMENTATION_REPAIR")
    assert ledger.state()["method_tried_history"] == []


def test_crash_after_reservation_resumes_same_attempt_without_double_count(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    first = _reserve(ledger, direction, variant)
    resumed = ResearchEventLedger(tmp_path)
    second = _reserve(resumed, direction, variant)
    assert second["attempt_id"] == first["attempt_id"]
    assert resumed.state()["directions"][direction["direction_hash"]]["budget"]["reserved"] == 1


def test_replaying_same_event_id_is_idempotent(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    ledger.select_direction(direction, event_id="same-event")
    ledger.select_direction(direction, event_id="same-event")
    assert len(ledger.events()) == 1
    assert ledger.state()["last_sequence"] == 1


def test_force_new_direction_is_executed_by_reducer(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    ledger.select_direction(direction)
    state = ledger.force_new_direction(reason_codes=["human_force_new_direction"])
    assert state["last_route_outcome"]["next_action"] == "START_NEW_DIRECTION"
    assert state["current_direction_hash"] is None


def test_bootstrap_proxy_finishes_once_without_consuming_standard_budget(tmp_path: Path) -> None:
    ledger = ResearchEventLedger(tmp_path)
    direction = _direction()
    variant = _variant(direction, 1)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant, profile="bootstrap")
    completed, route = _complete(ledger, attempt, outcome="accepted")
    assert route["next_action"] == "FINISH_RUN"
    assert completed["consumes_direction_budget"] is False
    assert ledger.state()["directions"][direction["direction_hash"]]["budget"]["consumed"] == 0


def test_strict_schema_rejects_missing_version_and_extra_fields() -> None:
    direction = _direction()
    missing = dict(direction)
    missing.pop("research_question")
    with pytest.raises(ValueError):
        validate_contract(missing, "direction_v2.schema.json")
    wrong = dict(direction, schema_version="wrong")
    with pytest.raises(ValueError):
        validate_contract(wrong, "direction_v2.schema.json")
    extra = dict(direction, unexpected=True)
    with pytest.raises(ValueError):
        validate_contract(extra, "direction_v2.schema.json")


def test_fingerprint_is_order_stable_and_sensitive_to_intervention_and_config() -> None:
    direction = _direction()
    first = _variant(direction, 1)
    reordered = {key: first[key] for key in reversed(list(first))}
    assert variant_spec_hash(first) == variant_spec_hash(reordered)
    changed_config = dict(first)
    changed_config["intervention"] = {**first["intervention"], "configuration": {"strength": 99}}
    assert variant_spec_hash(changed_config) != first["variant_spec_hash"]
    changed_operation = dict(first)
    changed_operation["intervention"] = {**first["intervention"], "algorithm_operations": ["different-operation"]}
    assert variant_spec_hash(changed_operation) != first["variant_spec_hash"]


def test_generic_core_state_machine_has_no_c2c_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in ["src/auto_research/domain_contracts.py", "src/auto_research/research_state.py"]:
        source = (root / rel).read_text(encoding="utf-8").lower()
        assert "c2c" not in source


def test_runtime_source_contains_no_removed_legacy_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    patterns = [
        "literature/ideas.json",
        "literature/direction_decision.json",
        "literature/c2c/direction_decision.json",
        "direction_to_legacy_idea",
        "load_direction_or_legacy_idea",
        "plan/candidate_ideas.json",
        "plan/next_variant.json",
        "plan/s2_planner/next_variant.json",
        "plan/plan.yaml",
        "legacy_route_fallback",
    ]
    command = ["rg", "-n", "|".join(pattern.replace(".", r"\.") for pattern in patterns), "src/auto_research", "--glob", "*.py"]
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    assert completed.returncode == 1, completed.stdout
