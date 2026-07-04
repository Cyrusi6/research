import json
from pathlib import Path

from auto_research.route_policy import build_route_context, decide_next_route, write_route_artifacts
from auto_research.validators.route_policy_gate import RoutePolicyGateValidator
from auto_research.utils import read_json, write_json


def test_write_route_artifacts_updates_ledger_and_trace(tmp_path: Path) -> None:
    write_json(tmp_path / "literature" / "direction.json", {"direction_id": "direction_x"})
    write_json(tmp_path / "literature" / "c2c" / "evidence_quality_score.json", {"direction_id": "direction_x", "gate": "pass"})
    write_json(tmp_path / "plan" / "code_patches" / "patch_gate_report.json", {"gate": "pass"})
    write_json(tmp_path / "experiment" / "results" / "c2c_proxy_decision_report.json", {"decision": "proxy_rejected", "route_hint": "return_s2", "failure_class": "proxy_negative", "variant_id": "variant_x"})
    registry = {"project_id": "proj", "iteration": 2, "current_stage": "S3_experiment"}
    config = {"c2c": {"enabled": True}, "orchestration": {"route_policy": {"enabled": True, "budgets": {"same_direction_proxy_failures": 2}}}}

    context = build_route_context(tmp_path, registry, config, trigger={"stage": "S3_experiment", "source": "s3_gate", "status": "failed", "reason": "proxy rejected"})
    decision = decide_next_route(context, config)
    bundle = write_route_artifacts(tmp_path, context, decision)

    assert bundle["attempt_record"]["route_decision"] == "route_to_s2"
    ledger = read_json(tmp_path / "meta" / "attempt_ledger.json")
    assert ledger["counters"]["by_direction"]["direction_x"]["proxy_failures"] == 1
    trace_lines = (tmp_path / "meta" / "iteration_trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(trace_lines[-1])["event"] == "route_decision"
    assert read_json(tmp_path / "meta" / "route_decision.json")["attempt_record"]["direction_id"] == "direction_x"

    report = RoutePolicyGateValidator(tmp_path, config).validate()
    assert report.status == "PASS"


def test_attempt_ledger_counts_full_s3_failure_separately(tmp_path: Path) -> None:
    write_json(tmp_path / "literature" / "direction.json", {"direction_id": "direction_x"})
    write_json(tmp_path / "literature" / "c2c" / "evidence_quality_score.json", {"direction_id": "direction_x", "gate": "pass"})
    write_json(tmp_path / "plan" / "code_patches" / "patch_gate_report.json", {"gate": "pass"})
    write_json(tmp_path / "experiment" / "results" / "c2c_proxy_decision_report.json", {"decision": "proxy_pass", "route_hint": "run_full_s3", "variant_id": "variant_x"})
    write_json(tmp_path / "experiment" / "results" / "main_results.json", {"acceptance": {"passed": False}, "candidate_results": [{"id": "variant_x", "metrics": {"mean": 48.0}}]})
    registry = {"project_id": "proj", "iteration": 2, "current_stage": "S3_experiment"}
    config = {"c2c": {"enabled": True}, "orchestration": {"route_policy": {"enabled": True, "budgets": {"same_direction_full_s3_failures": 1}}}}

    context = build_route_context(tmp_path, registry, config, trigger={"stage": "S3_experiment", "source": "s3_gate", "status": "failed", "reason": "full s3 failed"})
    decision = decide_next_route(context, config)
    write_route_artifacts(tmp_path, context, decision)

    counters = read_json(tmp_path / "meta" / "attempt_ledger.json")["counters"]["by_direction"]["direction_x"]
    assert counters["full_s3_failures"] == 1
    assert counters["proxy_failures"] == 0


def test_attempt_ledger_does_not_count_repairable_proxy_as_same_direction_failure(tmp_path: Path) -> None:
    write_json(tmp_path / "literature" / "direction.json", {"direction_id": "direction_x"})
    write_json(tmp_path / "literature" / "c2c" / "evidence_quality_score.json", {"direction_id": "direction_x", "gate": "pass"})
    write_json(tmp_path / "plan" / "code_patches" / "patch_gate_report.json", {"gate": "pass"})
    write_json(
        tmp_path / "experiment" / "results" / "c2c_proxy_decision_report.json",
        {
            "decision": "proxy_repairable",
            "route_hint": "return_s2",
            "failure_class": "effect_first_proxy_repair",
            "variant_id": "variant_x",
        },
    )
    registry = {"project_id": "proj", "iteration": 2, "current_stage": "S3_experiment"}
    config = {"c2c": {"enabled": True}, "orchestration": {"route_policy": {"enabled": True, "budgets": {"same_direction_proxy_failures": 2}}}}

    context = build_route_context(tmp_path, registry, config, trigger={"stage": "S3_experiment", "source": "s3_gate", "status": "failed", "reason": "repairable proxy"})
    decision = decide_next_route(context, config)
    write_route_artifacts(tmp_path, context, decision)

    counters = read_json(tmp_path / "meta" / "attempt_ledger.json")["counters"]["by_direction"]["direction_x"]
    assert counters["proxy_failures"] == 0
    assert counters["patch_repairs"] == 1
