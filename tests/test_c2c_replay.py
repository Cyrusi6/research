from pathlib import Path

from auto_research.c2c_e2e import build_c2c_replay_plan, build_c2c_replay_result
from auto_research.route_policy import build_route_context, decide_next_route, write_route_artifacts
from auto_research.utils import write_json, write_yaml


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


def _write_route_inputs(project: Path, *, route_hint: str = "return_s2") -> None:
    write_yaml(
        project / "meta" / "registry.yaml",
        {
            "project_id": project.name,
            "research_topic": "topic",
            "current_stage": "S3_experiment",
            "iteration": 1,
            "status": "running",
            "stages": {},
        },
    )
    write_json(project / "literature" / "direction.json", {"direction_id": "direction_x", "mechanism_axis": "routing", "integration_point": "projector", "control_signal": "utility"})
    write_json(project / "literature" / "c2c" / "evidence_quality_score.json", {"direction_id": "direction_x", "gate": "pass"})
    write_json(project / "plan" / "s2_planner" / "planner_gate_report.json", {"gate": "pass", "selected_variant_id": "variant_x"})
    write_json(project / "plan" / "s2_planner" / "variant_scorecard.json", {"selected_variant_id": "variant_x", "ranking": [{"variant_id": "variant_x", "score": 0.7, "decision": "selected"}]})
    write_json(project / "plan" / "code_patches" / "patch_gate_report.json", {"gate": "pass", "repairable": False})
    write_json(project / "experiment" / "results" / "c2c_proxy_decision_report.json", {"decision": "proxy_rejected", "route_hint": route_hint, "failure_class": "proxy_negative", "variant_id": "variant_x"})


def _write_expected_route(project: Path, config: dict) -> None:
    registry = {
        "project_id": project.name,
        "current_stage": "S3_experiment",
        "iteration": 1,
        "status": "running",
    }
    context = build_route_context(project, registry, config, trigger={"stage": "S3_experiment", "source": "s3_gate", "status": "failed", "reason": "proxy rejected"})
    decision = decide_next_route(context, config)
    write_route_artifacts(project, context, decision)


def test_c2c_replay_route_policy_matches_frozen_artifacts(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    config = _config()
    _write_route_inputs(project)
    _write_expected_route(project, config)

    plan = build_c2c_replay_plan(project, replay_from="S3_experiment")
    result = build_c2c_replay_result(project, plan, config)

    assert result["status"] == "match"
    assert result["replayed_decisions"]["route_decision"] == "route_to_s2"
    assert result["mismatches"] == []


def test_c2c_replay_detects_changed_proxy_decision(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    config = _config()
    _write_route_inputs(project)
    _write_expected_route(project, config)
    plan = build_c2c_replay_plan(project, replay_from="S3_experiment")
    _write_route_inputs(project, route_hint="return_s1")

    result = build_c2c_replay_result(project, plan, config)

    assert result["status"] == "mismatch"
    assert any(item["kind"] == "input_hash_mismatch" for item in result["mismatches"])


def test_c2c_replay_from_s3_uses_archived_route_when_latest_route_is_later_s2(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    config = _config()
    _write_route_inputs(project)
    _write_expected_route(project, config)

    write_yaml(
        project / "meta" / "registry.yaml",
        {
            "project_id": project.name,
            "research_topic": "topic",
            "current_stage": "S2_plan",
            "iteration": 1,
            "status": "failed",
            "stages": {},
        },
    )
    write_json(project / "plan" / "s2_planner" / "planner_gate_report.json", {"gate": "fail", "selected_variant_id": "variant_x"})
    write_json(
        project / "meta" / "attempt_ledger.json",
        {
            "schema_version": "c2c_attempt_ledger_v1",
            "project_id": project.name,
            "records": [],
            "counters": {"by_direction": {"direction_x": {"proxy_failures": 2, "full_s3_failures": 0, "patch_repairs": 0, "resource_retries": 0}}},
        },
    )
    context = build_route_context(project, {"project_id": project.name, "current_stage": "S2_plan", "iteration": 1}, config, trigger={"stage": "S2_plan", "source": "s2_gate", "status": "failed", "reason": "planner gate failed"})
    write_route_artifacts(project, context, decide_next_route(context, config))

    plan = build_c2c_replay_plan(project, replay_from="S3_experiment")
    result = build_c2c_replay_result(project, plan, config)

    assert plan["expected_decision_source"] == "route_decision_archive"
    assert result["status"] == "match"
    assert result["expected_decisions"]["failure_class"] == "proxy_negative"
