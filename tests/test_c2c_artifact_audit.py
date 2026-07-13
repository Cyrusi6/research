import json
import time
from pathlib import Path

from auto_research.artifacts import ArtifactManager
from auto_research.c2c_e2e import STAGE_ARTIFACT_REQUIREMENTS, build_c2c_artifact_audit_report
from auto_research.utils import write_json


def _audit_config() -> dict:
    return {
        "c2c": {"enabled": True},
        "orchestration": {
            "c2c_e2e": {
                "require_schema_validation": False,
                "require_stage_manifest_entries": True,
                "require_hash_validation": True,
                "detect_stale_artifacts": True,
            }
        },
    }


def _stage_and_relative_path(rel_path: str) -> tuple[str, str] | None:
    if rel_path.startswith("literature/"):
        return "S1_literature", rel_path.removeprefix("literature/")
    if rel_path.startswith("plan/"):
        return "S2_plan", rel_path.removeprefix("plan/")
    if rel_path.startswith("experiment/"):
        return "S3_experiment", rel_path.removeprefix("experiment/")
    return None


def _write_registered_required_artifacts(project: Path, *, skip: str | None = None) -> None:
    manager = ArtifactManager(project)
    for requirements in STAGE_ARTIFACT_REQUIREMENTS.values():
        for rel_path, _schema in requirements:
            if rel_path == skip or rel_path.startswith("meta/"):
                if rel_path.startswith("meta/") and rel_path != skip:
                    if rel_path.endswith(".jsonl"):
                        path = project / rel_path
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps({"event": "route_decision"}) + "\n", encoding="utf-8")
                    else:
                        write_json(project / rel_path, {"schema_version": "test"})
                continue
            stage_path = _stage_and_relative_path(rel_path)
            if not stage_path:
                continue
            stage, relative = stage_path
            manager.write_json(stage, relative, {"schema_version": "test"}, artifact_type="test_artifact")


def test_c2c_artifact_audit_reports_missing_contract_artifacts(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()

    report = build_c2c_artifact_audit_report(project, _audit_config(), scope="full")

    assert report["gate"] == "fail"
    assert report["summary"]["missing"] > 0
    assert "literature/direction.json" in report["by_stage"]["S1_literature"]["missing"]


def test_c2c_artifact_audit_reports_missing_stage_manifest_entry(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    _write_registered_required_artifacts(project, skip="plan/s2_planner/adaptive_policy.json")
    write_json(project / "plan" / "s2_planner" / "adaptive_policy.json", {"schema_version": "test"})

    report = build_c2c_artifact_audit_report(project, _audit_config(), scope="full")

    assert report["gate"] == "fail"
    assert any(item["path"] == "plan/s2_planner/adaptive_policy.json" for item in report["by_stage"]["S2_plan"]["manifest_missing"])


def test_c2c_artifact_audit_reports_hash_mismatch(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    _write_registered_required_artifacts(project)
    (project / "plan" / "s2_planner" / "variant_scorecard.json").write_text(json.dumps({"changed": True}), encoding="utf-8")

    report = build_c2c_artifact_audit_report(project, _audit_config(), scope="full")

    assert report["gate"] == "fail"
    assert any(item["path"] == "plan/s2_planner/variant_scorecard.json" for item in report["by_stage"]["S2_plan"]["hash_mismatches"])


def test_c2c_artifact_audit_allows_same_stage_failed_route_diagnostics(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    _write_registered_required_artifacts(project)
    stale = project / "plan" / "s2_planner" / "score_adjustment_report.json"
    stale.write_text(json.dumps({"diagnostic": True}), encoding="utf-8")
    time.sleep(0.01)
    write_json(
        project / "meta" / "route_decision.json",
        {
            "schema_version": "c2c_route_decision_v1",
            "trigger_stage": "S2_plan",
            "decision": "route_to_s2",
            "artifact_effects": {
                "invalidate_from": "S2_plan",
                "invalidate_artifacts": ["plan/s2_planner/score_adjustment_report.json"],
            },
        },
    )

    report = build_c2c_artifact_audit_report(project, _audit_config(), scope="full")

    assert report["by_stage"]["orchestration"]["stale_artifacts"] == []
