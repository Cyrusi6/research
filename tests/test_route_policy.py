from pathlib import Path

from auto_research.route_policy import build_route_context, decide_next_route
from auto_research.utils import write_json


def _config() -> dict:
    return {
        "c2c": {"enabled": True},
        "orchestration": {
            "route_policy": {
                "enabled": True,
                "budgets": {
                    "same_direction_proxy_failures": 2,
                    "same_direction_full_s3_failures": 1,
                    "patch_repair_attempts_per_variant": 2,
                    "resource_retries_per_stage": 3,
                },
            }
        },
    }


def _registry(stage: str = "S3_experiment") -> dict:
    return {"project_id": "proj", "iteration": 2, "current_stage": stage}


def _base_contracts(project: Path) -> None:
    write_json(
        project / "literature" / "direction.json",
        {
            "direction_id": "direction_x",
            "mechanism_axis": "routing",
            "integration_point": "rosetta/model/aligner.py",
            "control_signal": "utility",
        },
    )
    write_json(project / "literature" / "c2c" / "evidence_quality_score.json", {"direction_id": "direction_x", "gate": "pass", "novelty_score": 0.68})
    write_json(project / "plan" / "s2_planner" / "planner_gate_report.json", {"gate": "pass", "selected_variant_id": "variant_x"})
    write_json(project / "plan" / "s2_planner" / "variant_scorecard.json", {"ranking": [{"variant_id": "variant_x", "score": 0.72, "decision": "selected"}]})
    write_json(project / "plan" / "code_patches" / "patch_gate_report.json", {"gate": "pass", "repairable": False})


def _decision(project: Path, registry: dict | None = None, *, stage: str = "S3_experiment") -> dict:
    context = build_route_context(
        project,
        registry or _registry(stage),
        _config(),
        trigger={"stage": stage, "source": "s3_gate", "status": "failed", "reason": "failed"},
    )
    return decide_next_route(context, _config())


def test_proxy_rejected_routes_to_s2_with_budget_remaining(tmp_path: Path) -> None:
    _base_contracts(tmp_path)
    write_json(tmp_path / "experiment" / "results" / "c2c_proxy_decision_report.json", {"decision": "proxy_rejected", "route_hint": "return_s2", "failure_class": "proxy_negative", "variant_id": "variant_x"})

    decision = _decision(tmp_path)

    assert decision["decision"] == "route_to_s2"
    assert decision["next_stage"] == "S2_plan"
    assert decision["budget_effects"]["consumes_same_direction_attempt"] is True
    assert decision["memory_effects"]["write_shared_method_memory"] is True
    assert "same_direction_budget_remaining" in decision["reason_codes"]


def test_route_context_defaults_same_direction_proxy_budget_to_five(tmp_path: Path) -> None:
    _base_contracts(tmp_path)
    context = build_route_context(
        tmp_path,
        _registry(),
        {"c2c": {"enabled": True}, "orchestration": {"route_policy": {"enabled": True}}},
        trigger={"stage": "S3_experiment", "source": "s3_gate", "status": "failed", "reason": "proxy rejected"},
    )

    assert context["budgets"]["max_same_direction_proxy_failures"] == 5


def test_proxy_rejected_routes_to_s1_when_same_direction_budget_exhausted(tmp_path: Path) -> None:
    _base_contracts(tmp_path)
    write_json(tmp_path / "experiment" / "results" / "c2c_proxy_decision_report.json", {"decision": "proxy_rejected", "route_hint": "return_s2", "failure_class": "proxy_negative", "variant_id": "variant_x"})
    write_json(
        tmp_path / "meta" / "attempt_ledger.json",
        {
            "schema_version": "c2c_attempt_ledger_v1",
            "project_id": "proj",
            "records": [],
            "counters": {"by_direction": {"direction_x": {"proxy_failures": 2, "full_s3_failures": 0, "patch_repairs": 0, "resource_retries": 0}}},
        },
    )

    decision = _decision(tmp_path)

    assert decision["decision"] == "route_to_s1"
    assert decision["next_stage"] == "S1_literature"
    assert "same_direction_budget_exhausted" in decision["reason_codes"]


def test_implementation_failure_routes_to_s2_5_without_method_memory(tmp_path: Path) -> None:
    _base_contracts(tmp_path)
    write_json(tmp_path / "experiment" / "results" / "c2c_proxy_decision_report.json", {"decision": "proxy_repairable", "route_hint": "repair_s2_5", "failure_class": "implementation_failure", "variant_id": "variant_x"})

    decision = _decision(tmp_path)

    assert decision["decision"] == "route_to_s2_5"
    assert decision["budget_effects"]["consumes_patch_repair_attempt"] is True
    assert decision["memory_effects"]["write_shared_method_memory"] is False


def test_proxy_repairable_return_s2_hint_is_reclassified_to_patch_repair(tmp_path: Path) -> None:
    _base_contracts(tmp_path)
    write_json(
        tmp_path / "experiment" / "results" / "c2c_proxy_decision_report.json",
        {
            "decision": "proxy_repairable",
            "route_hint": "return_s2",
            "failure_class": "effect_first_proxy_repair",
            "variant_id": "variant_x",
        },
    )

    decision = _decision(tmp_path)

    assert decision["decision"] == "route_to_s2_5"
    assert decision["budget_effects"]["consumes_same_direction_attempt"] is False
    assert decision["budget_effects"]["consumes_patch_repair_attempt"] is True
    assert decision["memory_effects"]["write_shared_method_memory"] is False


def test_resource_retry_pauses_without_attempt_consumption(tmp_path: Path) -> None:
    _base_contracts(tmp_path)
    write_json(tmp_path / "experiment" / "results" / "c2c_proxy_decision_report.json", {"decision": "blocked", "route_hint": "block_resource", "failure_class": "resource_retry", "variant_id": "variant_x"})

    decision = _decision(tmp_path)

    assert decision["decision"] == "pause"
    assert decision["budget_effects"]["consumes_resource_retry"] is True
    assert decision["budget_effects"]["consumes_same_direction_attempt"] is False
    assert decision["memory_effects"]["write_shared_method_memory"] is False


def test_neutral_proxy_worthiness_low_routes_to_s2(tmp_path: Path) -> None:
    _base_contracts(tmp_path)
    write_json(tmp_path / "experiment" / "results" / "c2c_proxy_decision_report.json", {"decision": "proxy_rejected", "route_hint": "return_s2", "failure_class": "full_s3_not_worthy", "variant_id": "variant_x"})
    write_json(tmp_path / "experiment" / "results" / "c2c_full_s3_worthiness.json", {"score": 0.41, "decision": "do_not_run_full_s3"})

    decision = _decision(tmp_path)

    assert decision["decision"] == "route_to_s2"
    assert "neutral_proxy_worthiness_low" in decision["reason_codes"]


def test_proxy_pass_full_s3_failure_routes_to_s2_and_writes_memory(tmp_path: Path) -> None:
    _base_contracts(tmp_path)
    write_json(tmp_path / "experiment" / "results" / "c2c_proxy_decision_report.json", {"decision": "proxy_pass", "route_hint": "run_full_s3", "failure_class": None, "variant_id": "variant_x"})
    write_json(
        tmp_path / "experiment" / "results" / "main_results.json",
        {
            "acceptance": {"passed": False},
            "candidate_results": [{"id": "variant_x", "decision": "accepted_for_full", "metrics": {"mean": 48.0, "datasets": {"mmlu-redux": 47.0}}}],
        },
    )

    decision = _decision(tmp_path)

    assert decision["decision"] == "route_to_s2"
    assert decision["failure_class"] == "full_s3_method_failure"
    assert decision["memory_effects"]["memory_kind"] == "proxy_false_positive_full_s3_failure"


def test_s2_5_patch_gate_failure_routes_to_s2_5(tmp_path: Path) -> None:
    _base_contracts(tmp_path)
    write_json(tmp_path / "plan" / "code_patches" / "patch_gate_report.json", {"gate": "fail", "failure_class": "missing_ablation_switch", "repairable": True})

    decision = _decision(tmp_path, _registry("S2_plan"), stage="S2_plan")

    assert decision["decision"] == "route_to_s2_5"
    assert decision["failure_class"] == "missing_ablation_switch"
