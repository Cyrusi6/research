from pathlib import Path

from auto_research.c2c_e2e import build_c2c_artifact_audit_report
from auto_research.utils import sha256_file, write_json, write_yaml


def _config() -> dict:
    return {
        "c2c": {"enabled": True},
        "orchestration": {
            "c2c_e2e": {
                "require_schema_validation": False,
                "require_stage_manifest_entries": True,
                "require_hash_validation": True,
                "detect_stale_artifacts": False,
            }
        },
    }


def _write_completed_s2_manifest(project: Path) -> None:
    write_yaml(
        project / "meta" / "registry.yaml",
        {
            "project_id": project.name,
            "research_topic": "c2c",
            "current_stage": "S3_experiment",
            "iteration": 1,
            "status": "running",
            "stages": {},
        },
    )
    write_json(
        project / "meta" / "c2c_e2e_run_manifest.json",
        {
            "schema_version": "c2c_e2e_run_manifest_v1",
            "project_id": project.name,
            "stage_boundaries": {"S2_plan": {"status": "completed"}},
            "final_status": "running",
        },
    )


def test_audit_fails_when_manifest_entry_has_no_sha256(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    target = project / "plan" / "s2_planner" / "candidate_pool.json"
    write_json(target, {"schema_version": "test"})
    write_json(
        project / "plan" / "stage_manifest.json",
        {
            "schema_version": "stage_manifest_v1",
            "artifacts": [{"path": "plan/s2_planner/candidate_pool.json", "type": "test"}],
        },
    )
    _write_completed_s2_manifest(project)

    report = build_c2c_artifact_audit_report(project, _config(), scope="completed")

    assert report["gate"] == "fail"
    assert report["summary"]["missing_manifest_hash"] == 1
    assert report["by_stage"]["S2_plan"]["missing_manifest_hash"][0]["kind"] == "missing_manifest_hash"
    assert "S2_plan:missing_manifest_hash" in report["blocking_reasons"]


def test_audit_accepts_manifest_entry_with_matching_sha256(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    target = project / "plan" / "s2_planner" / "candidate_pool.json"
    write_json(target, {"schema_version": "test"})
    write_json(
        project / "plan" / "stage_manifest.json",
        {
            "schema_version": "stage_manifest_v1",
            "artifacts": [{"path": "plan/s2_planner/candidate_pool.json", "type": "test", "sha256": sha256_file(target)}],
        },
    )
    _write_completed_s2_manifest(project)
    config = _config()
    config["orchestration"]["c2c_e2e"]["require_stage_manifest_entries"] = True

    report = build_c2c_artifact_audit_report(project, config, scope="completed")

    assert report["summary"]["missing_manifest_hash"] == 0
    assert not report["by_stage"]["S2_plan"]["missing_manifest_hash"]
