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
