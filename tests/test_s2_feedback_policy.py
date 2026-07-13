from pathlib import Path

from auto_research.s2_feedback_policy import (
    build_s2_adaptive_policy,
    build_s2_dataset_risk_prior,
    build_s2_feedback_context,
    build_s2_variant_failure_prior,
)
from auto_research.utils import write_json


def _direction() -> dict:
    return {
        "direction_id": "direction_x",
        "mechanism_axis": "routing",
        "integration_point": "projector",
        "control_signal": "utility",
    }


def _candidate(**overrides) -> dict:
    payload = {
        "id": "variant_x",
        "mechanism_axis": "routing",
        "integration_point": "projector",
        "control_signal": "utility",
        "expected_files": ["rosetta/model/projector.py"],
        "experiment_contract": {"expected_files": ["rosetta/model/projector.py"], "ablation_switch": "disable_x"},
    }
    payload.update(overrides)
    return payload


def _config() -> dict:
    return {
        "c2c": {"enabled": True, "s2_adaptive_policy": {"enabled": True}},
        "orchestration": {
            "route_policy": {
                "budgets": {
                    "same_direction_proxy_failures": 2,
                    "same_direction_full_s3_failures": 1,
                    "patch_repair_attempts_per_variant": 2,
                    "resource_retries_per_stage": 3,
                }
            }
        },
    }


def test_feedback_context_no_history_writes_insufficient_policy(tmp_path: Path) -> None:
    context = build_s2_feedback_context(project_root=tmp_path, direction=_direction(), config=_config())
    policy = build_s2_adaptive_policy(context, _config())

    assert context["direction_id"] == "direction_x"
    assert context["recent_failures"] == []
    assert policy["history_sufficient"] is False
    assert policy["policy_hash"]


def test_feedback_context_defaults_same_direction_proxy_budget_to_five(tmp_path: Path) -> None:
    context = build_s2_feedback_context(
        project_root=tmp_path,
        direction=_direction(),
        config={"c2c": {"enabled": True}, "orchestration": {"route_policy": {"enabled": True}}},
    )

    assert context["attempt_counters"]["max_same_direction_proxy_failures"] == 5


def test_proxy_failure_penalizes_same_mechanism_and_integration() -> None:
    context = {
        "recent_failures": [
            {
                "mechanism_axis": "routing",
                "integration_point": "projector",
                "control_signal": "utility",
                "failure_class": "proxy_false_positive",
            }
        ],
        "attempt_counters": {"same_direction_proxy_failures": 1, "max_same_direction_proxy_failures": 2},
    }
    policy = build_s2_adaptive_policy({"direction_id": "direction_x", **context}, _config())

    prior = build_s2_variant_failure_prior(_candidate(), context, policy)

    assert prior["components"]["route_history_prior"] < 0
    assert any(item["reason"] == "same_integration_point_proxy_false_positive" for item in prior["applied"])


def test_implementation_failure_only_penalizes_patch_surface() -> None:
    context = {
        "recent_failures": [
            {
                "mechanism_axis": "routing",
                "integration_point": "projector",
                "control_signal": "utility",
                "failure_class": "implementation_failure",
            }
        ],
        "attempt_counters": {},
    }
    policy = build_s2_adaptive_policy({"direction_id": "direction_x", **context}, _config())

    prior = build_s2_variant_failure_prior(_candidate(), context, policy)

    assert prior["components"]["route_history_prior"] == 0
    assert prior["components"]["patch_surface_prior"] < 0


def test_repairable_proxy_is_treated_as_implementation_failure_only() -> None:
    context = {
        "recent_failures": [
            {
                "mechanism_axis": "routing",
                "integration_point": "projector",
                "control_signal": "utility",
                "failure_class": "effect_first_proxy_repair",
            }
        ],
        "attempt_counters": {"same_direction_proxy_failures": 0, "max_same_direction_proxy_failures": 2},
    }
    policy = build_s2_adaptive_policy({"direction_id": "direction_x", **context}, _config())

    prior = build_s2_variant_failure_prior(_candidate(), context, policy)

    assert prior["components"]["route_history_prior"] == 0
    assert prior["components"]["patch_surface_prior"] < 0
    assert not any(item["reason"] == "same_integration_point_proxy_false_positive" for item in prior["applied"])


def test_resource_retry_does_not_penalize_method_or_patch_surface() -> None:
    context = {
        "recent_failures": [{"mechanism_axis": "routing", "integration_point": "projector", "failure_class": "resource_retry"}],
        "attempt_counters": {"resource_retries": 1},
    }
    policy = build_s2_adaptive_policy({"direction_id": "direction_x", **context}, _config())

    prior = build_s2_variant_failure_prior(_candidate(), context, policy)

    assert prior["components"]["route_history_prior"] == 0
    assert prior["components"]["patch_surface_prior"] == 0


def test_dragging_dataset_penalty_and_addressing_bonus() -> None:
    context = {"dragging_datasets": ["openbookqa"], "recent_failures": [{"failure_class": "proxy_negative"}], "attempt_counters": {}}
    policy = build_s2_adaptive_policy({"direction_id": "direction_x", **context}, _config())

    missing = build_s2_dataset_risk_prior(_candidate(), context, policy)
    addressed = build_s2_dataset_risk_prior(
        _candidate(experiment_contract={"expected_files": ["rosetta/model/projector.py"], "ablation_switch": "disable_x", "diagnostics_required": ["dragging_dataset_probe:openbookqa"]}),
        context,
        policy,
    )

    assert missing["components"]["dataset_risk_prior"] < 0
    assert addressed["components"]["dataset_risk_prior"] > 0




def test_feedback_context_reads_only_method_evaluable_history(tmp_path: Path) -> None:
    from auto_research.research_state import ResearchEventLedger
    from test_authoritative_state_machine import _complete, _direction, _initialize, _reserve, _variant

    direction = _direction()
    ledger = ResearchEventLedger(tmp_path)
    first = _variant(direction, 1)
    _initialize(ledger, direction, first)
    _complete(ledger, _reserve(ledger, direction, first), outcome="rejected")
    second = _variant(direction, 2)
    _initialize(ledger, direction, second)
    pending = _reserve(ledger, direction, second)
    ledger.transition_attempt(pending["attempt_id"], "IMPLEMENTATION_REPAIR")

    context = build_s2_feedback_context(project_root=tmp_path, direction=direction, config={})

    assert len(context["recent_failures"]) == 1
    assert context["recent_failures"][0]["variant_id"] == first["variant_id"]
    assert context["attempt_counters"]["direction_budget_consumed"] == 1
    assert context["attempt_counters"]["direction_budget_reserved"] == 1
