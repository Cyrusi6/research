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


def test_policy_force_constraints_from_attempt_counters(tmp_path: Path) -> None:
    write_json(
        tmp_path / "meta" / "attempt_ledger.json",
        {
            "schema_version": "c2c_attempt_ledger_v1",
            "project_id": "proj",
            "records": [],
            "counters": {"by_direction": {"direction_x": {"proxy_failures": 2, "full_s3_failures": 1, "patch_repairs": 0, "resource_retries": 0}}},
        },
    )
    context = build_s2_feedback_context(project_root=tmp_path, direction=_direction(), config=_config())
    policy = build_s2_adaptive_policy(context, _config())

    assert policy["route_constraints"]["force_new_integration_point"] is True
    assert policy["route_constraints"]["force_new_direction"] is True
    assert policy["route_constraints"]["same_direction_budget_remaining"] is False


def test_feedback_context_recomputes_attempt_counters_and_dedupes_proxy_sources(tmp_path: Path) -> None:
    write_json(
        tmp_path / "meta" / "attempt_ledger.json",
        {
            "schema_version": "c2c_attempt_ledger_v1",
            "project_id": "proj",
            "records": [
                {
                    "direction_id": "direction_x",
                    "variant_id": "variant_x",
                    "failure_class": "effect_first_proxy_repair",
                    "route_decision": "route_to_s2",
                    "consumes_same_direction_attempt": True,
                    "consumes_patch_repair_attempt": False,
                    "consumes_resource_retry": False,
                }
            ],
            "counters": {"by_direction": {"direction_x": {"proxy_failures": 2, "full_s3_failures": 0, "patch_repairs": 0, "resource_retries": 0}}},
        },
    )
    write_json(
        tmp_path / "experiment" / "results" / "c2c_proxy_decision_report.json",
        {
            "decision": "proxy_repairable",
            "route_hint": "return_s2",
            "failure_class": "effect_first_proxy_repair",
            "variant_id": "variant_x",
        },
    )
    write_json(
        tmp_path / "experiment" / "results" / "main_results.json",
        {
            "candidate_results": [
                {"id": "variant_x", "decision": "proxy_repairable", "failure_class": "effect_first_proxy_repair"}
            ]
        },
    )

    context = build_s2_feedback_context(project_root=tmp_path, direction=_direction(), config=_config())
    policy = build_s2_adaptive_policy(context, _config())

    assert context["attempt_counters"]["same_direction_proxy_failures"] == 0
    assert policy["route_constraints"]["force_new_integration_point"] is False
    repair_rows = [row for row in context["recent_failures"] if row["failure_class"] == "implementation_failure"]
    assert len(repair_rows) == 1
