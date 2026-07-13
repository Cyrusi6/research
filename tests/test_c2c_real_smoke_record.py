from pathlib import Path

from auto_research.c2c_e2e import build_c2c_real_smoke_record, write_c2c_real_smoke_record
from auto_research.utils import read_json, write_json, write_yaml
from auto_research.validators import C2CE2EGateValidator


def test_c2c_real_smoke_record_captures_s1_blocked_smoke(tmp_path: Path) -> None:
    project = tmp_path / "c2c_smoke_blocked"
    _write_registry(
        project,
        current_stage="S1_literature",
        status="blocked",
        blocked_reason="S1c direction agent did not return valid bundle-grounded direction JSON",
    )
    write_json(
        project / "meta" / "c2c_e2e_readiness_report.json",
        _readiness(project, gate="warn", warnings=["deepseek_api_key_missing_for_semantic_enrichment"], blocking=[]),
    )
    write_json(project / "meta" / "c2c_e2e_run_manifest.json", _manifest(project, final_status="blocked", stage="S1_literature"))
    write_json(
        project / "meta" / "c2c_artifact_audit_report.json",
        _audit(project, gate="fail", blocking=["S1_literature:missing:literature/direction.json"]),
    )
    write_json(
        project / "meta" / "c2c_replay_result.json",
        _replay(project, status="blocked", mismatches=[{"kind": "missing_expected_route_decision", "path": "meta/route_decision.json"}]),
    )

    record = build_c2c_real_smoke_record(project, {})

    assert record["readiness_gate"] == "warn"
    assert record["run_manifest_final_status"] == "blocked"
    assert record["artifact_audit_gate"] == "fail"
    assert record["replay_status"] == "blocked"
    assert record["last_stage"] == "S1_literature"
    assert record["s1_evidence_gate"] is None
    assert record["s2_planner_gate"] is None
    assert record["route_decision"] is None
    assert "deepseek_api_key_missing_for_semantic_enrichment" in record["warnings"]
    assert "S1c direction agent did not return valid bundle-grounded direction JSON" in record["blocking_reasons"]
    assert "S1_literature:missing:literature/direction.json" in record["blocking_reasons"]
    assert "replay:missing_expected_route_decision:meta/route_decision.json" in record["blocking_reasons"]


def _write_registry(project: Path, *, current_stage: str, status: str, blocked_reason: str | None = None) -> None:
    write_yaml(
        project / "meta" / "registry.yaml",
        {
            "project_id": project.name,
            "research_topic": "c2c smoke",
            "current_stage": current_stage,
            "iteration": 1,
            "status": status,
            "blocked_reason": blocked_reason,
            "stages": {},
        },
    )


def _readiness(project: Path, *, gate: str, warnings: list[str], blocking: list[str]) -> dict:
    return {
        "schema_version": "c2c_e2e_readiness_report_v1",
        "project_id": project.name,
        "mode": "real",
        "gate": gate,
        "checks": {
            "target_repo_exists": True,
            "ref_paper_exists": True,
            "ref_rebuttal_exists": True,
            "env_python_executable": True,
            "workspace_writable": True,
            "worktree_root_writable": True,
            "llm_config_ready": True,
            "dataset_paths_ready": True,
            "gpu_policy_ready": True,
            "s0_cache_compatible": True,
            "baseline_cache_valid_or_invalidated": True,
            "real_execution_hooks_ready": True,
        },
        "warnings": warnings,
        "blocking_reasons": blocking,
        "recommended_action": "run_c2c" if not blocking else "fix_environment",
    }


def _manifest(project: Path, *, final_status: str, stage: str) -> dict:
    return {
        "schema_version": "c2c_e2e_run_manifest_v1",
        "project_id": project.name,
        "started_at": "2026-07-03T00:00:00+00:00",
        "mode": "real",
        "command": {"name": "run-c2c"},
        "inputs": {
            "target_repo": "/tmp/C2C",
            "ref_paper": "/tmp/paper",
            "ref_rebuttal": "/tmp/rebuttal",
            "env_python": "/usr/bin/python3",
            "project_config_sha256": "abc",
            "root_config_sha256": "def",
        },
        "stage_boundaries": {stage: {"status": final_status}},
        "final_status": final_status,
    }


def _audit(project: Path, *, gate: str, blocking: list[str]) -> dict:
    return {
        "schema_version": "c2c_artifact_audit_report_v1",
        "project_id": project.name,
        "gate": gate,
        "audit_scope": "completed",
        "expected_stages": [],
        "skipped_stages": [],
        "summary": {
            "checked_artifacts": 27,
            "missing": 0 if gate == "pass" else 1,
            "schema_failures": 0,
            "missing_manifest_hash": 0,
            "hash_mismatches": 0,
            "stale_artifacts": 0,
        },
        "by_stage": {},
        "blocking_reasons": blocking,
    }


def _replay(project: Path, *, status: str, mismatches: list[dict]) -> dict:
    return {
        "schema_version": "c2c_replay_result_v1",
        "project_id": project.name,
        "status": status,
        "replayed_decisions": {},
        "expected_decisions": {},
        "mismatches": mismatches,
    }
