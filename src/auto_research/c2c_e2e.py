"""Deterministic C2C real-run readiness, audit, and replay helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import bootstrap_cached_s0_only_enabled
from .research_state import ResearchEventLedger
from .utils import ensure_dir, now_utc, read_json, read_yaml, sha256_file, write_json
from .validators.base import load_schema, validate_min_schema


C2C_E2E_READINESS_SCHEMA_VERSION = "c2c_e2e_readiness_report_v1"
C2C_E2E_RUN_MANIFEST_SCHEMA_VERSION = "c2c_e2e_run_manifest_v1"
C2C_ARTIFACT_AUDIT_SCHEMA_VERSION = "c2c_artifact_audit_report_v1"
C2C_RUNTIME_HEALTH_SCHEMA_VERSION = "c2c_runtime_health_report_v1"
C2C_EXECUTION_HOOKS_SCHEMA_VERSION = "c2c_execution_hooks_report_v1"
C2C_REPLAY_PLAN_SCHEMA_VERSION = "c2c_replay_plan_v1"
C2C_REPLAY_RESULT_SCHEMA_VERSION = "c2c_replay_result_v1"
C2C_REAL_SMOKE_RECORD_SCHEMA_VERSION = "c2c_real_smoke_record_v1"


STAGE_ARTIFACT_REQUIREMENTS = {
    "S1_literature": [
        ("literature/direction.json", "direction_v3.schema.json"),
        ("literature/c2c/evidence_request_plan.json", "s1_evidence_request_plan.schema.json"),
        ("literature/c2c/evidence_bundle.json", "s1_deterministic_evidence_bundle.schema.json"),
        ("literature/c2c/direction_candidate_scorecard.json", "s1_direction_candidate_scorecard.schema.json"),
        ("literature/c2c/evidence_retrieval_trace.json", "s1_evidence_retrieval_trace.schema.json"),
        ("literature/c2c/evidence_quality_score.json", "s1_evidence_quality.schema.json"),
        ("literature/c2c/direction_fingerprint.json", "s1_direction_fingerprint.schema.json"),
    ],
    "S2_plan": [
        ("plan/variant.json", "variant_v4.schema.json"),
        ("plan/trial_spec.json", "trial_spec_v3.schema.json"),
        ("plan/s2_planner/candidate_pool.json", "s2_candidate_pool.schema.json"),
        ("plan/s2_planner/feedback_context.json", "s2_feedback_context.schema.json"),
        ("plan/s2_planner/adaptive_policy.json", "s2_adaptive_policy.schema.json"),
        ("plan/s2_planner/variant_scorecard.json", "s2_variant_scorecard.schema.json"),
        ("plan/s2_planner/score_adjustment_report.json", "s2_score_adjustment_report.schema.json"),
        ("plan/s2_planner/planner_gate_report.json", "s2_planner_gate_report.schema.json"),
    ],
    "S2_5_patch": [
        ("plan/code_patches/implementation_contract.json", "s2_5_implementation_contract.schema.json"),
        ("plan/code_patches/patch_manifest.json", None),
        ("plan/code_patches/patch_gate_report.json", "s2_5_patch_gate_report.schema.json"),
    ],
    "S3_experiment": [
        ("experiment/results/trial_result.json", "trial_result_v4.schema.json"),
        ("experiment/results/c2c_proxy_baseline_fingerprint.json", "c2c_proxy_baseline_fingerprint.schema.json"),
        ("experiment/results/c2c_proxy_cache_report.json", "c2c_proxy_cache_report.schema.json"),
        ("experiment/results/c2c_effective_proxy_policy.json", "c2c_effective_proxy_policy.schema.json"),
        ("experiment/results/c2c_proxy_decision_report.json", "c2c_proxy_decision_report.schema.json"),
        ("experiment/results/c2c_proxy_calibration_policy.json", "c2c_proxy_calibration_policy.schema.json"),
        ("experiment/results/s3_candidate_selection.json", None),
    ],
    "orchestration": [
        ("meta/route_outcome.json", "route_outcome_v3.schema.json"),
        ("meta/research_state.json", "research_state_v4.schema.json"),
        ("meta/iteration_trace.jsonl", None),
    ],
}

STAGE_MANIFESTS = {
    "literature": "literature/stage_manifest.json",
    "plan": "plan/stage_manifest.json",
    "experiment": "experiment/stage_manifest.json",
}

AUDIT_STAGE_ORDER = ["S1_literature", "S2_plan", "S2_5_patch", "S3_experiment", "orchestration"]


def build_c2c_e2e_readiness_report(project_root: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the pre-real-run C2C readiness report without calling any model."""

    config = config if isinstance(config, dict) else {}
    c2c = config.get("c2c") if isinstance(config.get("c2c"), dict) else {}
    mode = "simulate" if bool((config.get("experiment") or {}).get("simulate")) else "real"
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    blocking: list[str] = []

    target_repo = _path(c2c.get("target_repo"))
    ref_paper = _path(c2c.get("ref_paper"))
    ref_rebuttal = _path(c2c.get("ref_rebuttal"))
    env_python = _path(c2c.get("env_python"))
    snapshot_path = project_root / str(c2c.get("snapshot_path") or "external/c2c_snapshot")
    workspace_root = project_root.parent
    code_patch_root = project_root / "plan" / "code_patches"

    checks["target_repo_exists"] = bool(target_repo and target_repo.exists())
    checks["ref_paper_exists"] = bool(ref_paper and ref_paper.exists())
    checks["ref_rebuttal_exists"] = bool(ref_rebuttal and ref_rebuttal.exists())
    checks["env_python_executable"] = bool(env_python and env_python.exists() and os.access(env_python, os.X_OK))
    checks["workspace_writable"] = _writable(workspace_root)
    checks["worktree_root_writable"] = _writable(code_patch_root.parent)
    checks["snapshot_ready"] = snapshot_path.exists() or checks["target_repo_exists"]
    cached_s0_only = bootstrap_cached_s0_only_enabled(config)
    checks["s0_cache_compatible"] = _s0_cache_compatible(project_root, snapshot_path)
    checks["cached_s0_only_ready"] = bool(
        not cached_s0_only
        or (_s0_cache_available(project_root) and checks["s0_cache_compatible"])
    )
    checks["llm_config_ready"] = _llm_config_ready(config, mode=mode, warnings=warnings)
    checks["semantic_enrichment_key_ready"] = True if checks["cached_s0_only_ready"] and cached_s0_only else _semantic_enrichment_ready(config, mode=mode, warnings=warnings)
    checks["dataset_paths_ready"] = _dataset_paths_ready(c2c, mode=mode, warnings=warnings)
    checks["gpu_policy_ready"] = _gpu_policy_ready(config, mode=mode, warnings=warnings)
    checks["baseline_cache_valid_or_invalidated"] = _baseline_cache_ready(project_root, warnings=warnings)
    hooks_report = read_json(project_root / "meta" / "c2c_execution_hooks_report.json", default={}) or {}
    if mode == "simulate":
        checks["real_execution_hooks_ready"] = True
    elif isinstance(hooks_report, dict) and hooks_report:
        checks["real_execution_hooks_ready"] = hooks_report.get("gate") == "pass"
    else:
        checks["real_execution_hooks_ready"] = bool(c2c.get("env_python") and c2c.get("snapshot_path"))
        warnings.append("execution_hooks_report_missing")

    hard_checks = [
        "target_repo_exists",
        "ref_paper_exists",
        "ref_rebuttal_exists",
        "env_python_executable",
        "workspace_writable",
        "worktree_root_writable",
        "snapshot_ready",
        "cached_s0_only_ready",
        "llm_config_ready",
        "dataset_paths_ready",
        "gpu_policy_ready",
        "real_execution_hooks_ready",
    ]
    if mode == "simulate":
        hard_checks = ["workspace_writable", "worktree_root_writable"]
    for key in hard_checks:
        if not checks.get(key):
            blocking.append(key)
    if checks.get("s0_cache_compatible") is False:
        warnings.append("s0_cache_fingerprint_mismatch_or_unchecked")
    gate = "fail" if blocking else ("warn" if warnings else "pass")
    recommended = "run_c2c"
    if blocking:
        recommended = "fix_environment"
    elif "s0_cache_fingerprint_mismatch_or_unchecked" in warnings:
        recommended = "refresh_s0"
    elif "baseline_cache_invalidated" in warnings:
        recommended = "rerun_baseline"
    return {
        "schema_version": C2C_E2E_READINESS_SCHEMA_VERSION,
        "created_at": now_utc(),
        "project_id": project_root.name,
        "mode": mode,
        "gate": gate,
        "checks": checks,
        "warnings": list(dict.fromkeys(warnings)),
        "blocking_reasons": blocking,
        "recommended_action": recommended,
        "paths": {
            "target_repo": str(target_repo) if target_repo else None,
            "ref_paper": str(ref_paper) if ref_paper else None,
            "ref_rebuttal": str(ref_rebuttal) if ref_rebuttal else None,
            "env_python": str(env_python) if env_python else None,
            "snapshot_path": str(snapshot_path),
        },
        "execution_hooks_gate": hooks_report.get("gate") if isinstance(hooks_report, dict) else None,
        "execution_hooks_report": "meta/c2c_execution_hooks_report.json" if hooks_report else None,
    }


def build_c2c_runtime_health_report(project_root: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Capture deterministic runtime health signals for a C2C run."""

    config = config if isinstance(config, dict) else {}
    env_python = _path(((config.get("c2c") or {}).get("env_python") if isinstance(config.get("c2c"), dict) else None))
    python_probe = _python_version_probe(env_python)
    nvidia_smi = shutil.which("nvidia-smi")
    return {
        "schema_version": C2C_RUNTIME_HEALTH_SCHEMA_VERSION,
        "created_at": now_utc(),
        "project_id": project_root.name,
        "python": {
            "current": platform.python_version(),
            "env_python": str(env_python) if env_python else None,
            "env_python_exists": bool(env_python and env_python.exists()),
            "env_python_version": python_probe.get("version"),
            "env_python_probe_status": python_probe.get("status"),
        },
        "gpu": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_smi_available": bool(nvidia_smi),
            "nvidia_smi_path": nvidia_smi,
            "policy": ((config.get("c2c") or {}).get("small_loop") or {}).get("gpu_ids")
            if isinstance((config.get("c2c") or {}).get("small_loop"), dict)
            else None,
        },
        "filesystem": {
            "project_root_writable": _writable(project_root),
            "workspace_writable": _writable(project_root.parent),
        },
    }


def build_c2c_execution_hooks_report(project_root: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run cheap probes that prove the configured C2C repo can execute basic hooks."""

    config = config if isinstance(config, dict) else {}
    c2c = config.get("c2c") if isinstance(config.get("c2c"), dict) else {}
    e2e_cfg = ((config.get("orchestration") or {}).get("c2c_e2e") or {}) if isinstance(config.get("orchestration"), dict) else {}
    env_python = _path(c2c.get("env_python"))
    snapshot_path = project_root / str(c2c.get("snapshot_path") or "external/c2c_snapshot")
    target_repo = snapshot_path if snapshot_path.exists() else _path(c2c.get("target_repo"))
    try:
        timeout_seconds = int(e2e_cfg.get("execution_hook_timeout_seconds") or 30)
    except (TypeError, ValueError):
        timeout_seconds = 30
    commands: list[dict[str, Any]] = []

    version = _run_hook_command("env_python_version", [str(env_python), "--version"] if env_python else [], timeout_seconds=timeout_seconds)
    commands.append(version)
    executable = _run_hook_command(
        "env_python_executable",
        [str(env_python), "-c", "import sys; print(sys.executable)"] if env_python else [],
        timeout_seconds=timeout_seconds,
    )
    commands.append(executable)

    importable = _run_hook_command(
        "target_repo_importable",
        [
            str(env_python),
            "-c",
            (
                "import importlib.util, pathlib, sys; "
                "repo=pathlib.Path(sys.argv[1]).resolve(); "
                "sys.path.insert(0, str(repo)); "
                "ok=importlib.util.find_spec('rosetta') is not None; "
                "print(ok); raise SystemExit(0 if ok else 1)"
            ),
            str(target_repo or ""),
        ]
        if env_python and target_repo
        else [],
        timeout_seconds=timeout_seconds,
        cwd=target_repo,
    )
    commands.append(importable)

    eval_entrypoint = (target_repo / "script" / "evaluation" / "unified_evaluator.py") if target_repo else None
    eval_help = _run_hook_command(
        "eval_help",
        [str(env_python), str(eval_entrypoint), "--help"] if env_python and eval_entrypoint and eval_entrypoint.exists() else [],
        timeout_seconds=timeout_seconds,
        cwd=target_repo,
    )
    commands.append(eval_help)

    checks = {
        "env_python_runs": version.get("returncode") == 0 and executable.get("returncode") == 0,
        "target_repo_importable": importable.get("returncode") == 0,
        "eval_entrypoint_exists": bool(eval_entrypoint and eval_entrypoint.exists()),
        "eval_help_command_passed": eval_help.get("returncode") == 0,
        "dataset_one_example_loadable": _dataset_one_example_loadable(_path(c2c.get("dataset_root"))),
        "output_dir_writable": _writable((target_repo / "local" / "auto_research_runs") if target_repo else project_root / "experiment" / "results"),
        "command_timeout_configured": _command_timeout_configured(c2c),
    }
    blocking = [key for key, ok in checks.items() if not ok]
    return {
        "schema_version": C2C_EXECUTION_HOOKS_SCHEMA_VERSION,
        "created_at": now_utc(),
        "project_id": project_root.name,
        "gate": "fail" if blocking else "pass",
        "checks": checks,
        "commands": commands,
        "blocking_reasons": blocking,
        "warnings": [],
        "paths": {
            "env_python": str(env_python) if env_python else None,
            "target_repo": str(target_repo) if target_repo else None,
            "eval_entrypoint": str(eval_entrypoint) if eval_entrypoint else None,
            "dataset_root": c2c.get("dataset_root"),
        },
    }


def build_c2c_e2e_run_manifest(
    project_root: Path,
    config: dict[str, Any] | None = None,
    *,
    command: dict[str, Any] | None = None,
    existing: dict[str, Any] | None = None,
    stage_event: dict[str, Any] | None = None,
    final_status: str | None = None,
) -> dict[str, Any]:
    """Build or update the immutable-input manifest for a C2C E2E run."""

    config = config if isinstance(config, dict) else {}
    c2c = config.get("c2c") if isinstance(config.get("c2c"), dict) else {}
    manifest = dict(existing or {})
    if manifest.get("schema_version") != C2C_E2E_RUN_MANIFEST_SCHEMA_VERSION:
        manifest = {
            "schema_version": C2C_E2E_RUN_MANIFEST_SCHEMA_VERSION,
            "project_id": project_root.name,
            "started_at": now_utc(),
            "mode": "simulate" if bool((config.get("experiment") or {}).get("simulate")) else "real",
            "command": command or {},
            "inputs": _run_manifest_inputs(project_root, config, c2c),
            "stage_boundaries": {},
            "final_status": "running",
        }
    if command:
        manifest["command"] = command
    if stage_event:
        stage = str(stage_event.get("stage") or "")
        status = str(stage_event.get("status") or "")
        if stage:
            current = dict((manifest.setdefault("stage_boundaries", {}) or {}).get(stage) or {})
            timestamp = stage_event.get("timestamp") or now_utc()
            if status in {"started", "running"}:
                current.setdefault("started_at", timestamp)
            elif status in {"completed", "blocked", "failed", "retryable_paused", "feedback_routed"}:
                current["completed_at"] = timestamp
            current["status"] = status
            if stage_event.get("reason"):
                current["reason"] = stage_event.get("reason")
            manifest["stage_boundaries"][stage] = current
    if final_status:
        manifest["final_status"] = final_status
        manifest["completed_at"] = now_utc()
    manifest["updated_at"] = now_utc()
    return manifest


def build_c2c_artifact_audit_report(project_root: Path, config: dict[str, Any] | None = None, *, scope: str | None = None) -> dict[str, Any]:
    """Audit C2C run artifacts, schemas, stage manifests, hashes, and stale outputs."""

    config = config if isinstance(config, dict) else {}
    e2e_cfg = ((config.get("orchestration") or {}).get("c2c_e2e") or {}) if isinstance(config.get("orchestration"), dict) else {}
    audit_scope = str(scope or e2e_cfg.get("audit_scope") or "completed")
    if audit_scope not in {"completed", "up-to-current", "full"}:
        audit_scope = "completed"
    require_manifest = bool(e2e_cfg.get("require_stage_manifest_entries", True))
    require_schema = bool(e2e_cfg.get("require_schema_validation", True))
    require_hash = bool(e2e_cfg.get("require_hash_validation", True))
    detect_stale = bool(e2e_cfg.get("detect_stale_artifacts", True))
    registry = read_yaml(project_root / "meta" / "registry.yaml", default={}) or {}
    run_manifest = read_json(project_root / "meta" / "c2c_e2e_run_manifest.json", default={}) or {}
    expected_stages, skipped_stages = _expected_audit_stages(project_root, registry, run_manifest, audit_scope, config)
    by_stage: dict[str, dict[str, Any]] = {}
    checked_artifacts = 0
    missing_count = 0
    schema_failure_count = 0
    missing_manifest_hash_count = 0
    hash_mismatch_count = 0
    stale_count = 0
    blocking: list[str] = []

    for stage in expected_stages:
        requirements = STAGE_ARTIFACT_REQUIREMENTS.get(stage, [])
        stage_report = {
            "gate": "pass",
            "missing": [],
            "schema_failures": [],
            "hash_mismatches": [],
            "manifest_missing": [],
            "missing_manifest_hash": [],
            "stale_artifacts": [],
        }
        for rel_path, schema_name in requirements:
            path = project_root / rel_path
            checked_artifacts += 1
            if not path.exists():
                stage_report["missing"].append(rel_path)
                missing_count += 1
                continue
            if require_schema and schema_name:
                schema_errors = _schema_errors(path, schema_name)
                if schema_errors:
                    schema_failure_count += 1
                    stage_report["schema_failures"].append({"path": rel_path, "errors": schema_errors[:8]})
            if require_manifest and _is_stage_artifact(rel_path):
                manifest_error = _manifest_error(project_root, rel_path, require_hash=require_hash)
                if manifest_error:
                    if manifest_error.get("kind") == "hash_mismatch":
                        hash_mismatch_count += 1
                        stage_report["hash_mismatches"].append(manifest_error)
                    elif manifest_error.get("kind") == "missing_manifest_hash":
                        missing_manifest_hash_count += 1
                        stage_report["missing_manifest_hash"].append(manifest_error)
                    else:
                        stage_report["manifest_missing"].append(manifest_error)
        if any(stage_report[key] for key in ["missing", "schema_failures", "hash_mismatches", "manifest_missing", "missing_manifest_hash", "stale_artifacts"]):
            stage_report["gate"] = "fail"
        by_stage[stage] = stage_report

    optional = _optional_artifact_checks(project_root, require_schema=require_schema)
    for item in optional:
        checked_artifacts += 1
        if item.get("schema_errors"):
            schema_failure_count += 1
            by_stage.setdefault(
                item["stage"],
                {"gate": "pass", "missing": [], "schema_failures": [], "hash_mismatches": [], "manifest_missing": [], "missing_manifest_hash": [], "stale_artifacts": []},
            )
            by_stage[item["stage"]]["schema_failures"].append({"path": item["path"], "errors": item["schema_errors"]})
            by_stage[item["stage"]]["gate"] = "fail"

    if detect_stale:
        stale = _stale_artifacts_after_route(project_root)
        stale_count = len(stale)
        if stale:
            by_stage.setdefault(
                "orchestration",
                {"gate": "pass", "missing": [], "schema_failures": [], "hash_mismatches": [], "manifest_missing": [], "missing_manifest_hash": [], "stale_artifacts": []},
            )
            by_stage["orchestration"]["stale_artifacts"].extend(stale)
            by_stage["orchestration"]["gate"] = "fail"

    for stage, report in by_stage.items():
        if report.get("gate") == "fail":
            if report.get("missing"):
                blocking.append(f"{stage}:missing:{','.join(report['missing'][:3])}")
            if report.get("schema_failures"):
                blocking.append(f"{stage}:schema_failure")
            if report.get("hash_mismatches"):
                blocking.append(f"{stage}:hash_mismatch")
            if report.get("manifest_missing"):
                blocking.append(f"{stage}:manifest_missing")
            if report.get("missing_manifest_hash"):
                blocking.append(f"{stage}:missing_manifest_hash")
            if report.get("stale_artifacts"):
                blocking.append(f"{stage}:stale_artifact")
    gate = "fail" if blocking else "pass"
    return {
        "schema_version": C2C_ARTIFACT_AUDIT_SCHEMA_VERSION,
        "created_at": now_utc(),
        "project_id": project_root.name,
        "gate": gate,
        "audit_scope": audit_scope,
        "expected_stages": expected_stages,
        "skipped_stages": skipped_stages,
        "summary": {
            "checked_artifacts": checked_artifacts,
            "missing": missing_count,
            "schema_failures": schema_failure_count,
            "missing_manifest_hash": missing_manifest_hash_count,
            "hash_mismatches": hash_mismatch_count,
            "stale_artifacts": stale_count,
        },
        "by_stage": by_stage,
        "blocking_reasons": blocking,
    }


def build_c2c_replay_plan(project_root: Path, *, replay_from: str = "S3_experiment") -> dict[str, Any]:
    """Freeze immutable event hashes for deterministic reducer replay."""

    ledger = ResearchEventLedger(project_root)
    ledger.events()
    frozen_inputs = ["meta/research_events.sqlite3"]
    input_hashes = {rel: _sha_or_none(project_root / rel) for rel in frozen_inputs}
    state = ledger.state()
    normalized = _normalize_research_state(state)
    return {
        "schema_version": C2C_REPLAY_PLAN_SCHEMA_VERSION,
        "created_at": now_utc(),
        "project_id": project_root.name,
        "replay_from": replay_from,
        "frozen_inputs": frozen_inputs,
        "input_hashes": input_hashes,
        "actions": ["replay_immutable_events", "rebuild_research_state", "compare_state_hash"],
        "expected_decision_hashes": {"research_state": _stable_hash(normalized)},
        "expected_decision_source": "meta/research_events.sqlite3",
        "expected_decision_summary": {
            "last_sequence": state.get("last_sequence"),
            "next_action": ((state.get("last_route_outcome") or {}).get("next_action") if isinstance(state.get("last_route_outcome"), dict) else None),
        },
    }


def build_c2c_replay_result(project_root: Path, replay_plan: dict[str, Any] | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Rebuild state exclusively from immutable events and compare hashes."""

    del config
    plan = replay_plan if isinstance(replay_plan, dict) else read_json(project_root / "meta" / "c2c_replay_plan.json", default={}) or {}
    mismatches: list[dict[str, Any]] = []
    for rel, expected_hash in (plan.get("input_hashes") or {}).items():
        actual_hash = _sha_or_none(project_root / rel)
        if expected_hash != actual_hash:
            mismatches.append({"kind": "input_hash_mismatch", "path": rel, "expected": expected_hash, "actual": actual_hash})
    state = ResearchEventLedger(project_root).rebuild()
    actual_hash = _stable_hash(_normalize_research_state(state))
    expected_hash = ((plan.get("expected_decision_hashes") or {}).get("research_state"))
    if expected_hash and expected_hash != actual_hash:
        mismatches.append({"kind": "research_state_mismatch", "expected_hash": expected_hash, "actual": actual_hash})
    route = state.get("last_route_outcome") if isinstance(state.get("last_route_outcome"), dict) else {}
    return {
        "schema_version": C2C_REPLAY_RESULT_SCHEMA_VERSION,
        "created_at": now_utc(),
        "project_id": project_root.name,
        "status": "match" if not mismatches else "mismatch",
        "replayed_decisions": {"route_decision": route.get("next_action"), "hash": actual_hash},
        "expected_decisions": {"route_decision": (plan.get("expected_decision_summary") or {}).get("next_action"), "hash": expected_hash, "source": "meta/research_events.sqlite3"},
        "mismatches": mismatches,
    }


def _normalize_research_state(state: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(state))
    normalized.pop("updated_at", None)
    for attempt in (normalized.get("attempts") or {}).values():
        if isinstance(attempt, dict):
            attempt.pop("updated_at", None)
    return normalized


def build_c2c_real_smoke_record(project_root: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Summarize the latest real C2C smoke run into a single deterministic record."""

    del config  # Reserved for future policy switches without changing the public builder.
    registry = read_yaml(project_root / "meta" / "registry.yaml", default={}) or {}
    readiness = read_json(project_root / "meta" / "c2c_e2e_readiness_report.json", default={}) or {}
    execution_hooks = read_json(project_root / "meta" / "c2c_execution_hooks_report.json", default={}) or {}
    manifest = read_json(project_root / "meta" / "c2c_e2e_run_manifest.json", default={}) or {}
    audit = read_json(project_root / "meta" / "c2c_artifact_audit_report.json", default={}) or {}
    replay = read_json(project_root / "meta" / "c2c_replay_result.json", default={}) or {}
    evidence_quality = read_json(project_root / "literature" / "c2c" / "evidence_quality_score.json", default={}) or {}
    planner_gate = read_json(project_root / "plan" / "s2_planner" / "planner_gate_report.json", default={}) or {}
    patch_gate = read_json(project_root / "plan" / "code_patches" / "patch_gate_report.json", default={}) or {}
    proxy_decision = read_json(project_root / "experiment" / "results" / "c2c_proxy_decision_report.json", default={}) or {}
    route_decision = ResearchEventLedger(project_root).state().get("last_route_outcome") or {}

    blocking_reasons = _smoke_blocking_reasons(registry, readiness, execution_hooks, audit, replay)
    return {
        "schema_version": C2C_REAL_SMOKE_RECORD_SCHEMA_VERSION,
        "created_at": now_utc(),
        "project_id": project_root.name,
        "readiness_gate": readiness.get("gate") if isinstance(readiness, dict) else None,
        "execution_hooks_gate": execution_hooks.get("gate") if isinstance(execution_hooks, dict) else None,
        "run_manifest_final_status": manifest.get("final_status") if isinstance(manifest, dict) else None,
        "artifact_audit_gate": audit.get("gate") if isinstance(audit, dict) else None,
        "replay_status": replay.get("status") if isinstance(replay, dict) else None,
        "last_stage": _smoke_last_stage(registry, manifest),
        "s1_evidence_gate": _gate_value(evidence_quality),
        "s2_planner_gate": _gate_value(planner_gate),
        "s2_5_patch_gate": _gate_value(patch_gate),
        "s3_proxy_decision": proxy_decision.get("decision") if isinstance(proxy_decision, dict) else None,
        "route_decision": route_decision.get("next_action") if isinstance(route_decision, dict) else None,
        "route_next_stage": route_decision.get("next_action") if isinstance(route_decision, dict) else None,
        "blocked_reason": registry.get("blocked_reason") if isinstance(registry, dict) else None,
        "blocking_reasons": blocking_reasons,
        "warnings": _string_list(readiness.get("warnings") if isinstance(readiness, dict) else []),
        "audit_summary": audit.get("summary") if isinstance(audit.get("summary") if isinstance(audit, dict) else None, dict) else {},
        "replay_mismatches": replay.get("mismatches") if isinstance(replay.get("mismatches") if isinstance(replay, dict) else None, list) else [],
    }


def write_c2c_e2e_readiness_report(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    report = build_c2c_e2e_readiness_report(project_root, config)
    write_json(project_root / "meta" / "c2c_e2e_readiness_report.json", report)
    return report


def write_c2c_runtime_health_report(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    report = build_c2c_runtime_health_report(project_root, config)
    write_json(project_root / "meta" / "c2c_runtime_health_report.json", report)
    return report


def write_c2c_execution_hooks_report(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    report = build_c2c_execution_hooks_report(project_root, config)
    write_json(project_root / "meta" / "c2c_execution_hooks_report.json", report)
    return report


def write_c2c_e2e_run_manifest(
    project_root: Path,
    config: dict[str, Any],
    *,
    command: dict[str, Any] | None = None,
    stage_event: dict[str, Any] | None = None,
    final_status: str | None = None,
) -> dict[str, Any]:
    path = project_root / "meta" / "c2c_e2e_run_manifest.json"
    existing = read_json(path, default={}) or {}
    manifest = build_c2c_e2e_run_manifest(project_root, config, command=command, existing=existing, stage_event=stage_event, final_status=final_status)
    write_json(path, manifest)
    return manifest


def write_c2c_artifact_audit_report(project_root: Path, config: dict[str, Any], *, scope: str | None = None) -> dict[str, Any]:
    report = build_c2c_artifact_audit_report(project_root, config, scope=scope)
    write_json(project_root / "meta" / "c2c_artifact_audit_report.json", report)
    return report


def write_c2c_replay_plan(project_root: Path, *, replay_from: str = "S3_experiment") -> dict[str, Any]:
    plan = build_c2c_replay_plan(project_root, replay_from=replay_from)
    write_json(project_root / "meta" / "c2c_replay_plan.json", plan)
    return plan


def write_c2c_replay_result(project_root: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    plan = read_json(project_root / "meta" / "c2c_replay_plan.json", default={}) or {}
    result = build_c2c_replay_result(project_root, plan, config)
    write_json(project_root / "meta" / "c2c_replay_result.json", result)
    return result


def write_c2c_real_smoke_record(project_root: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    record = build_c2c_real_smoke_record(project_root, config)
    write_json(project_root / "meta" / "c2c_real_smoke_record.json", record)
    return record


def _path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value).expanduser()


def _writable(path: Path) -> bool:
    try:
        ensure_dir(path)
        probe = path / ".auto_research_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _llm_config_ready(config: dict[str, Any], *, mode: str, warnings: list[str]) -> bool:
    if mode == "simulate":
        return True
    llm = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    if llm.get("use_real_api") is False:
        warnings.append("llm_real_api_disabled")
        return True
    providers = {str(llm.get("provider") or "openai"), str(llm.get("reasoning_provider") or llm.get("provider") or "openai")}
    providers.discard("none")
    providers.discard("mock")
    if "openai" in providers and not _has_openai_key(llm):
        return False
    if "codex_cli" in providers and not shutil.which("codex"):
        return False
    return True


def _has_openai_key(llm: dict[str, Any]) -> bool:
    if os.environ.get("OPENAI_API_KEY"):
        return True
    if llm.get("api_key"):
        return True
    keys = llm.get("api_keys")
    return isinstance(keys, list) and any(keys)


def _semantic_enrichment_ready(config: dict[str, Any], *, mode: str, warnings: list[str]) -> bool:
    if mode == "simulate":
        return True
    enrichment = ((config.get("intake") or {}).get("semantic_enrichment") or {}) if isinstance(config.get("intake"), dict) else {}
    if not enrichment.get("enabled"):
        return True
    provider = str(enrichment.get("provider") or "deepseek")
    if provider == "deepseek" and not os.environ.get("DEEPSEEK_API_KEY"):
        warnings.append("deepseek_api_key_missing_for_semantic_enrichment")
        return False
    return True


def _dataset_paths_ready(c2c: dict[str, Any], *, mode: str, warnings: list[str]) -> bool:
    if mode == "simulate":
        return True
    dataset_root = _path(c2c.get("dataset_root"))
    strict = bool(((c2c.get("small_loop") or {}).get("strict_dataset_cache", True)) if isinstance(c2c.get("small_loop"), dict) else True)
    if dataset_root and dataset_root.exists():
        return True
    if strict:
        return False
    warnings.append("dataset_root_missing_but_strict_dataset_cache_disabled")
    return True


def _gpu_policy_ready(config: dict[str, Any], *, mode: str, warnings: list[str]) -> bool:
    if mode == "simulate":
        return True
    proxy_cfg = (((config.get("c2c") or {}).get("small_loop") or {}).get("proxy_screen") or {}) if isinstance((config.get("c2c") or {}).get("small_loop"), dict) else {}
    gpu_policy = proxy_cfg.get("gpu_policy") if isinstance(proxy_cfg.get("gpu_policy"), dict) else {}
    gpu_ids = gpu_policy.get("gpu_ids", ((config.get("c2c") or {}).get("small_loop") or {}).get("gpu_ids"))
    if gpu_ids in (None, "auto", [], ""):
        if shutil.which("nvidia-smi") or os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
            return True
        warnings.append("gpu_auto_policy_without_nvidia_smi")
        return True
    return True


def _baseline_cache_ready(project_root: Path, *, warnings: list[str]) -> bool:
    cache = project_root / "experiment" / "results" / "c2c_proxy_baseline.json"
    fingerprint = project_root / "experiment" / "results" / "c2c_proxy_baseline_fingerprint.json"
    report = project_root / "experiment" / "results" / "c2c_proxy_cache_report.json"
    if not cache.exists():
        return True
    if fingerprint.exists():
        return True
    if report.exists():
        return True
    warnings.append("baseline_cache_invalidated")
    return True


def _run_hook_command(
    name: str,
    command: list[str],
    *,
    timeout_seconds: int,
    cwd: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    if not command:
        return {
            "name": name,
            "command": "",
            "returncode": None,
            "duration_sec": 0.0,
            "status": "skipped",
            "reason": "command_not_configured",
        }
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=max(1, timeout_seconds),
            check=False,
        )
        status = "ok" if result.returncode == 0 else "failed"
        return {
            "name": name,
            "command": " ".join(command),
            "returncode": result.returncode,
            "duration_sec": round(time.monotonic() - started, 3),
            "status": status,
            "stdout": (result.stdout or "").strip()[:1000],
            "stderr": (result.stderr or "").strip()[:1000],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "command": " ".join(command),
            "returncode": None,
            "duration_sec": round(time.monotonic() - started, 3),
            "status": "timeout",
            "stderr": str(exc)[:1000],
        }
    except Exception as exc:
        return {
            "name": name,
            "command": " ".join(command),
            "returncode": None,
            "duration_sec": round(time.monotonic() - started, 3),
            "status": "failed",
            "stderr": str(exc)[:1000],
        }


def _dataset_one_example_loadable(dataset_root: Path | None) -> bool:
    if not dataset_root or not dataset_root.exists():
        return False
    if dataset_root.is_file():
        return _load_one_dataset_example(dataset_root)
    checked = 0
    pending = [dataset_root]
    visited_dirs: set[Path] = set()
    while pending and checked < 25:
        current = pending.pop()
        try:
            resolved = current.resolve()
            if resolved in visited_dirs:
                continue
            visited_dirs.add(resolved)
            children = sorted(current.iterdir(), key=lambda path: path.name, reverse=True)
        except OSError:
            continue
        for path in children:
            if path.is_dir():
                pending.append(path)
                continue
            if not path.is_file() or path.suffix.lower() not in {".jsonl", ".json", ".csv", ".txt"}:
                continue
            checked += 1
            if _load_one_dataset_example(path):
                return True
            if checked >= 25:
                break
    return False


def _load_one_dataset_example(path: Path) -> bool:
    try:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                json.loads(line)
                return True
            return False
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(payload, list):
                return bool(payload)
            if isinstance(payload, dict):
                for key in ["data", "examples", "train", "validation", "test"]:
                    value = payload.get(key)
                    if isinstance(value, list) and value:
                        return True
                return bool(payload)
            return payload is not None
        if suffix == ".csv":
            return bool(path.read_text(encoding="utf-8", errors="ignore").splitlines())
        if suffix == ".txt":
            return bool(path.read_text(encoding="utf-8", errors="ignore").strip())
    except Exception:
        return False
    return False


def _command_timeout_configured(c2c: dict[str, Any]) -> bool:
    small_loop = c2c.get("small_loop") if isinstance(c2c.get("small_loop"), dict) else {}
    proxy_screen = small_loop.get("proxy_screen") if isinstance(small_loop.get("proxy_screen"), dict) else {}
    for key in ["command_timeout_seconds", "train_timeout_seconds", "eval_timeout_seconds", "preflight_timeout_seconds"]:
        try:
            if int(proxy_screen.get(key) or small_loop.get(key) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _s0_cache_compatible(project_root: Path, snapshot_path: Path) -> bool:
    bundle = read_json(project_root / "intake" / "c2c" / "static_bundle.json", default={}) or {}
    if not isinstance(bundle, dict) or not bundle:
        return True
    repo_manifest = bundle.get("repo_manifest") if isinstance(bundle.get("repo_manifest"), dict) else {}
    core_files = repo_manifest.get("core_files") if isinstance(repo_manifest, dict) else []
    if not isinstance(core_files, list) or not core_files:
        return True
    for item in core_files:
        if not isinstance(item, dict):
            continue
        rel = item.get("path")
        expected = item.get("sha256")
        if not rel or not expected:
            continue
        path = snapshot_path / str(rel)
        if not path.exists() or sha256_file(path) != expected:
            return False
    return True


def _s0_cache_available(project_root: Path) -> bool:
    bundle = read_json(project_root / "intake" / "c2c" / "static_bundle.json", default={}) or {}
    if not isinstance(bundle, dict) or bundle.get("schema_version") != "c2c_static_intake_bundle_v1":
        return False
    chunk_index = bundle.get("chunk_index") if isinstance(bundle.get("chunk_index"), dict) else {}
    return bool(chunk_index.get("entries"))


def _expected_audit_stages(
    project_root: Path,
    registry: dict[str, Any],
    run_manifest: dict[str, Any],
    scope: str,
    config: dict[str, Any],
) -> tuple[list[str], list[dict[str, str]]]:
    del config
    if scope == "full":
        return list(AUDIT_STAGE_ORDER), []

    boundaries = run_manifest.get("stage_boundaries") if isinstance(run_manifest, dict) else {}
    boundaries = boundaries if isinstance(boundaries, dict) else {}
    completed = {
        stage
        for stage, payload in boundaries.items()
        if isinstance(payload, dict) and payload.get("status") == "completed"
    }
    expected: list[str] = []
    current_stage = str(registry.get("current_stage") or "")
    current_index = None
    if scope == "up-to-current":
        current_index = _audit_stage_index(current_stage, completed)
        if current_index is None:
            indexes = [_audit_stage_index(stage, completed) for stage in completed]
            current_index = max((idx for idx in indexes if idx is not None), default=-1)
        expected = [stage for idx, stage in enumerate(AUDIT_STAGE_ORDER) if idx <= current_index and stage != "orchestration"]
    else:
        expected = [stage for stage in AUDIT_STAGE_ORDER if stage in completed and stage != "orchestration"]

    s2_5_index = AUDIT_STAGE_ORDER.index("S2_5_patch")
    if scope == "completed" and "S2_plan" in expected:
        expected.append("S2_5_patch")
    if scope == "up-to-current" and ((current_index is not None and current_index >= s2_5_index) or "S2_plan" in completed):
        expected.append("S2_5_patch")
    if "S3_experiment" in expected and "S2_5_patch" not in expected:
        expected.append("S2_5_patch")
    expected = [stage for stage in AUDIT_STAGE_ORDER if stage in set(expected)]

    if _orchestration_required_artifacts_present(project_root):
        expected.append("orchestration")

    skipped = []
    expected_set = set(expected)
    for stage in AUDIT_STAGE_ORDER:
        if stage in expected_set:
            continue
        reason = "not_reached"
        payload = boundaries.get(stage) if isinstance(boundaries.get(stage), dict) else {}
        if scope == "completed" and payload and payload.get("status") != "completed":
            reason = "not_completed"
        if stage == "orchestration" and not _orchestration_required_artifacts_present(project_root):
            reason = "not_reached"
        skipped.append({"stage": stage, "reason": reason})
    return expected, skipped


def _audit_stage_index(current_stage: str, completed: set[str]) -> int | None:
    if current_stage == "DONE":
        indexes = [AUDIT_STAGE_ORDER.index(stage) for stage in completed if stage in AUDIT_STAGE_ORDER]
        return max(indexes) if indexes else None
    if current_stage == "S2_5_patch":
        return AUDIT_STAGE_ORDER.index("S2_5_patch")
    if current_stage in AUDIT_STAGE_ORDER:
        return AUDIT_STAGE_ORDER.index(current_stage)
    return None


def _orchestration_required_artifacts_present(project_root: Path) -> bool:
    return any((project_root / rel_path).exists() for rel_path, _schema in STAGE_ARTIFACT_REQUIREMENTS.get("orchestration", []))


def _run_manifest_inputs(project_root: Path, config: dict[str, Any], c2c: dict[str, Any]) -> dict[str, Any]:
    target_repo = _path(c2c.get("target_repo"))
    ref_paper = _path(c2c.get("ref_paper"))
    ref_rebuttal = _path(c2c.get("ref_rebuttal"))
    return {
        "target_repo": str(target_repo) if target_repo else None,
        "target_repo_head": _git_head(target_repo) if target_repo else None,
        "ref_paper": str(ref_paper) if ref_paper else None,
        "ref_paper_sha256": _sha_or_none(ref_paper) if ref_paper else None,
        "ref_rebuttal": str(ref_rebuttal) if ref_rebuttal else None,
        "ref_rebuttal_sha256": _sha_or_none(ref_rebuttal) if ref_rebuttal else None,
        "env_python": c2c.get("env_python"),
        "project_config_sha256": _sha_or_none(project_root / "meta" / "project_config.yaml"),
        "root_config_sha256": _stable_hash(config),
    }


def _git_head(path: Path | None) -> str | None:
    if not path or not path.exists():
        return None
    try:
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False)
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _python_version_probe(env_python: Path | None) -> dict[str, Any]:
    if not env_python or not env_python.exists():
        return {"status": "missing", "version": None}
    try:
        result = subprocess.run([str(env_python), "--version"], capture_output=True, text=True, timeout=5, check=False)
    except Exception as exc:
        return {"status": "failed", "version": None, "error": str(exc)}
    text = (result.stdout or result.stderr or "").strip()
    return {"status": "ok" if result.returncode == 0 else "failed", "version": text}


def _schema_errors(path: Path, schema_name: str) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid JSON: {exc}"]
    return validate_min_schema(payload, load_schema(schema_name))


def _manifest_error(project_root: Path, rel_path: str, *, require_hash: bool) -> dict[str, Any] | None:
    manifest_rel = _manifest_for_artifact(rel_path)
    if not manifest_rel:
        return None
    manifest = read_json(project_root / manifest_rel, default={}) or {}
    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else []
    entry = next((item for item in artifacts or [] if isinstance(item, dict) and item.get("path") == rel_path), None)
    if not entry:
        return {"kind": "missing_manifest_entry", "path": rel_path, "manifest": manifest_rel}
    if require_hash:
        expected = entry.get("sha256")
        if not expected:
            return {"kind": "missing_manifest_hash", "path": rel_path, "manifest": manifest_rel}
        actual = sha256_file(project_root / rel_path)
        if actual != expected:
            return {"kind": "hash_mismatch", "path": rel_path, "manifest": manifest_rel, "expected": expected, "actual": actual}
    return None


def _manifest_for_artifact(rel_path: str) -> str | None:
    for prefix, manifest in STAGE_MANIFESTS.items():
        if rel_path.startswith(prefix + "/"):
            return manifest
    return None


def _is_stage_artifact(rel_path: str) -> bool:
    return _manifest_for_artifact(rel_path) is not None


def _optional_artifact_checks(project_root: Path, *, require_schema: bool) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    optional = [
        ("experiment/results/c2c_full_s3_worthiness.json", "c2c_full_s3_worthiness.schema.json", "S3_experiment"),
        ("meta/c2c_e2e_readiness_report.json", "c2c_e2e_readiness_report.schema.json", "orchestration"),
        ("meta/c2c_e2e_run_manifest.json", "c2c_e2e_run_manifest.schema.json", "orchestration"),
        ("meta/c2c_execution_hooks_report.json", "c2c_execution_hooks_report.schema.json", "orchestration"),
        ("meta/c2c_replay_plan.json", "c2c_replay_plan.schema.json", "orchestration"),
        ("meta/c2c_replay_result.json", "c2c_replay_result.schema.json", "orchestration"),
        ("meta/c2c_real_smoke_record.json", "c2c_real_smoke_record.schema.json", "orchestration"),
    ]
    proxy = read_json(project_root / "experiment" / "results" / "c2c_proxy_decision_report.json", default={}) or {}
    for rel, schema, stage in optional:
        path = project_root / rel
        if rel.endswith("c2c_full_s3_worthiness.json") and proxy.get("decision") != "neutral_proxy_full_s3" and not path.exists():
            continue
        if not path.exists():
            continue
        checks.append({"path": rel, "stage": stage, "schema_errors": _schema_errors(path, schema) if require_schema else []})
    return checks


def _stale_artifacts_after_route(project_root: Path) -> list[dict[str, Any]]:
    if (project_root / "meta" / "research_events.sqlite3").exists():
        ResearchEventLedger(project_root).state()
    return []


def _route_decision_summary(decision: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(decision, dict):
        return {}
    return {
        "trigger_stage": decision.get("trigger_stage"),
        "failure_class": decision.get("failure_class"),
        "decision": decision.get("decision"),
        "next_stage": decision.get("next_stage"),
        "reason_codes": decision.get("reason_codes") or [],
    }


def _sha_or_none(path: Path | None) -> str | None:
    if not path or not path.exists() or not path.is_file():
        return None
    return sha256_file(path)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _gate_value(payload: dict[str, Any]) -> str | None:
    if not isinstance(payload, dict) or not payload:
        return None
    value = payload.get("gate", payload.get("status"))
    return str(value) if value is not None else None


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(item) for item in values if item is not None]


def _smoke_last_stage(registry: dict[str, Any], manifest: dict[str, Any]) -> str | None:
    if isinstance(registry, dict) and registry.get("current_stage"):
        return str(registry.get("current_stage"))
    boundaries = manifest.get("stage_boundaries") if isinstance(manifest, dict) else {}
    if not isinstance(boundaries, dict) or not boundaries:
        return None
    ordered = [
        (str(stage), str((payload or {}).get("completed_at") or (payload or {}).get("started_at") or ""))
        for stage, payload in boundaries.items()
        if isinstance(payload, dict)
    ]
    if not ordered:
        return None
    ordered.sort(key=lambda item: item[1])
    return ordered[-1][0]


def _smoke_blocking_reasons(
    registry: dict[str, Any],
    readiness: dict[str, Any],
    execution_hooks: dict[str, Any],
    audit: dict[str, Any],
    replay: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if isinstance(registry, dict) and registry.get("blocked_reason"):
        reasons.append(str(registry.get("blocked_reason")))
    if isinstance(readiness, dict):
        reasons.extend(_string_list(readiness.get("blocking_reasons")))
    if isinstance(execution_hooks, dict):
        reasons.extend(f"execution_hooks:{item}" for item in _string_list(execution_hooks.get("blocking_reasons")))
    if isinstance(audit, dict):
        reasons.extend(_string_list(audit.get("blocking_reasons")))
    if isinstance(replay, dict):
        for item in replay.get("mismatches") or []:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind") or "mismatch"
            path = item.get("path")
            reasons.append(f"replay:{kind}:{path}" if path else f"replay:{kind}")
    return list(dict.fromkeys(reasons))


def _normalize_route_decision(decision: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(decision, dict):
        return {}
    return {
        key: decision.get(key)
        for key in [
            "trigger_stage",
            "trigger_source",
            "failure_class",
            "decision",
            "next_stage",
            "next_iteration",
            "reason_codes",
            "budget_effects",
            "memory_effects",
            "artifact_effects",
            "orchestrator_action",
        ]
    }
