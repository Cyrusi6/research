import json
from pathlib import Path

from auto_research.c2c_e2e import STAGE_ARTIFACT_REQUIREMENTS, build_c2c_artifact_audit_report
from auto_research.utils import write_json, write_yaml


def _config() -> dict:
    return {
        "c2c": {"enabled": True},
        "orchestration": {
            "stop_after_stage": "S3_experiment",
            "c2c_e2e": {
                "require_schema_validation": False,
                "require_stage_manifest_entries": False,
                "require_hash_validation": False,
                "detect_stale_artifacts": False,
            },
        },
    }


def _write_required_files(project: Path, stage: str) -> None:
    for rel_path, _schema in STAGE_ARTIFACT_REQUIREMENTS[stage]:
        if rel_path.endswith(".jsonl"):
            path = project / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"event": "ok"}) + "\n", encoding="utf-8")
        else:
            write_json(project / rel_path, {"schema_version": "test"})


def _write_registry_and_manifest(project: Path, *, current_stage: str, status: str, boundaries: dict) -> None:
    write_yaml(
        project / "meta" / "registry.yaml",
        {
            "project_id": project.name,
            "research_topic": "c2c",
            "current_stage": current_stage,
            "iteration": 1,
            "status": status,
            "stages": {},
        },
    )
    write_json(
        project / "meta" / "c2c_e2e_run_manifest.json",
        {
            "schema_version": "c2c_e2e_run_manifest_v1",
            "project_id": project.name,
            "stage_boundaries": boundaries,
            "final_status": status,
        },
    )


def test_audit_completed_scope_skips_not_reached_s3_requirements(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    _write_required_files(project, "S1_literature")
    _write_registry_and_manifest(
        project,
        current_stage="S2_plan",
        status="blocked",
        boundaries={
            "S1_literature": {"status": "completed"},
            "S2_plan": {"status": "blocked"},
        },
    )

    report = build_c2c_artifact_audit_report(project, _config())

    assert report["gate"] == "pass"
    assert report["audit_scope"] == "completed"
    assert report["expected_stages"] == ["S1_literature"]
    assert "S3_experiment" not in report["by_stage"]
    assert {"stage": "S3_experiment", "reason": "not_reached"} in report["skipped_stages"]


def test_audit_up_to_current_includes_current_stage_but_not_unreached_s2_5(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    _write_required_files(project, "S1_literature")
    _write_registry_and_manifest(
        project,
        current_stage="S2_plan",
        status="blocked",
        boundaries={
            "S1_literature": {"status": "completed"},
            "S2_plan": {"status": "blocked"},
        },
    )

    report = build_c2c_artifact_audit_report(project, _config(), scope="up-to-current")

    assert report["gate"] == "fail"
    assert report["expected_stages"] == ["S1_literature", "S2_plan"]
    assert "plan/s2_planner/candidate_pool.json" in report["by_stage"]["S2_plan"]["missing"]
    assert "S2_5_patch" not in report["by_stage"]
    assert {"stage": "S3_experiment", "reason": "not_reached"} in report["skipped_stages"]


def test_audit_full_scope_keeps_current_full_requirements(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()

    report = build_c2c_artifact_audit_report(project, _config(), scope="full")

    assert report["gate"] == "fail"
    assert report["expected_stages"] == ["S1_literature", "S2_plan", "S2_5_patch", "S3_experiment", "orchestration"]
    assert "experiment/results/c2c_proxy_decision_report.json" in report["by_stage"]["S3_experiment"]["missing"]


def test_audit_uses_current_trial_and_state_schemas() -> None:
    assert ("plan/trial_spec.json", "trial_spec_v5.schema.json") in STAGE_ARTIFACT_REQUIREMENTS["S2_plan"]
    assert ("meta/research_state.json", "research_state_v6.schema.json") in STAGE_ARTIFACT_REQUIREMENTS["orchestration"]
    assert all(schema != "trial_spec_v4.schema.json" for requirements in STAGE_ARTIFACT_REQUIREMENTS.values() for _, schema in requirements)
    assert all(schema != "research_state_v5.schema.json" for requirements in STAGE_ARTIFACT_REQUIREMENTS.values() for _, schema in requirements)
