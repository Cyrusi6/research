"""Deterministic C2C real-run readiness, audit, and replay helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .route_policy import build_route_context, decide_next_route
from .utils import ensure_dir, now_utc, read_json, read_yaml, sha256_file, write_json
from .validators.base import load_schema, validate_min_schema


C2C_E2E_READINESS_SCHEMA_VERSION = "c2c_e2e_readiness_report_v1"
C2C_E2E_RUN_MANIFEST_SCHEMA_VERSION = "c2c_e2e_run_manifest_v1"
C2C_ARTIFACT_AUDIT_SCHEMA_VERSION = "c2c_artifact_audit_report_v1"
C2C_RUNTIME_HEALTH_SCHEMA_VERSION = "c2c_runtime_health_report_v1"
C2C_REPLAY_PLAN_SCHEMA_VERSION = "c2c_replay_plan_v1"
C2C_REPLAY_RESULT_SCHEMA_VERSION = "c2c_replay_result_v1"


STAGE_ARTIFACT_REQUIREMENTS = {
    "S1_literature": [
        ("literature/direction.json", "direction.schema.json"),
        ("literature/c2c/evidence_request_plan.json", "s1_evidence_request_plan.schema.json"),
        ("literature/c2c/evidence_bundle.json", "s1_deterministic_evidence_bundle.schema.json"),
        ("literature/c2c/evidence_retrieval_trace.json", "s1_evidence_retrieval_trace.schema.json"),
        ("literature/c2c/evidence_quality_score.json", "s1_evidence_quality.schema.json"),
        ("literature/c2c/direction_fingerprint.json", "s1_direction_fingerprint.schema.json"),
    ],
    "S2_plan": [
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
        ("experiment/results/c2c_proxy_baseline_fingerprint.json", "c2c_proxy_baseline_fingerprint.schema.json"),
        ("experiment/results/c2c_proxy_cache_report.json", "c2c_proxy_cache_report.schema.json"),
        ("experiment/results/c2c_effective_proxy_policy.json", "c2c_effective_proxy_policy.schema.json"),
        ("experiment/results/c2c_proxy_decision_report.json", "c2c_proxy_decision_report.schema.json"),
        ("experiment/results/c2c_proxy_calibration_policy.json", "c2c_proxy_calibration_policy.schema.json"),
        ("experiment/results/s3_candidate_selection.json", None),
    ],
    "orchestration": [
        ("meta/route_context.json", "route_context.schema.json"),
        ("meta/route_decision.json", "route_decision.schema.json"),
        ("meta/attempt_ledger.json", "attempt_ledger.schema.json"),
        ("meta/iteration_trace.jsonl", None),
    ],
}

STAGE_MANIFESTS = {
    "literature": "literature/stage_manifest.json",
    "plan": "plan/stage_manifest.json",
    "experiment": "experiment/stage_manifest.json",
}


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
    checks["s0_cache_compatible"] = _s0_cache_compatible(project_root, snapshot_path)
    checks["llm_config_ready"] = _llm_config_ready(config, mode=mode, warnings=warnings)
    checks["semantic_enrichment_key_ready"] = _semantic_enrichment_ready(config, mode=mode, warnings=warnings)
    checks["dataset_paths_ready"] = _dataset_paths_ready(c2c, mode=mode, warnings=warnings)
    checks["gpu_policy_ready"] = _gpu_policy_ready(config, mode=mode, warnings=warnings)
    checks["baseline_cache_valid_or_invalidated"] = _baseline_cache_ready(project_root, warnings=warnings)
    checks["real_execution_hooks_ready"] = mode == "simulate" or bool(c2c.get("env_python") and c2c.get("snapshot_path"))

    hard_checks = [
        "target_repo_exists",
        "ref_paper_exists",
        "ref_rebuttal_exists",
        "env_python_executable",
        "workspace_writable",
        "worktree_root_writable",
        "snapshot_ready",
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


def build_c2c_artifact_audit_report(project_root: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Audit C2C run artifacts, schemas, stage manifests, hashes, and stale outputs."""

    config = config if isinstance(config, dict) else {}
    e2e_cfg = ((config.get("orchestration") or {}).get("c2c_e2e") or {}) if isinstance(config.get("orchestration"), dict) else {}
    require_manifest = bool(e2e_cfg.get("require_stage_manifest_entries", True))
    require_schema = bool(e2e_cfg.get("require_schema_validation", True))
    require_hash = bool(e2e_cfg.get("require_hash_validation", True))
    detect_stale = bool(e2e_cfg.get("detect_stale_artifacts", True))
    by_stage: dict[str, dict[str, Any]] = {}
    checked_artifacts = 0
    missing_count = 0
    schema_failure_count = 0
    hash_mismatch_count = 0
    stale_count = 0
    blocking: list[str] = []

    for stage, requirements in STAGE_ARTIFACT_REQUIREMENTS.items():
        stage_report = {"gate": "pass", "missing": [], "schema_failures": [], "hash_mismatches": [], "manifest_missing": [], "stale_artifacts": []}
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
                    else:
                        stage_report["manifest_missing"].append(manifest_error)
            if require_hash and path.is_file() and rel_path.endswith(".json"):
                pass
        if any(stage_report[key] for key in ["missing", "schema_failures", "hash_mismatches", "manifest_missing", "stale_artifacts"]):
            stage_report["gate"] = "fail"
        by_stage[stage] = stage_report

    optional = _optional_artifact_checks(project_root, require_schema=require_schema)
    for item in optional:
        checked_artifacts += 1
        if item.get("schema_errors"):
            schema_failure_count += 1
            by_stage.setdefault(item["stage"], {"gate": "pass", "missing": [], "schema_failures": [], "hash_mismatches": [], "manifest_missing": [], "stale_artifacts": []})
            by_stage[item["stage"]]["schema_failures"].append({"path": item["path"], "errors": item["schema_errors"]})
            by_stage[item["stage"]]["gate"] = "fail"

    if detect_stale:
        stale = _stale_artifacts_after_route(project_root)
        stale_count = len(stale)
        if stale:
            by_stage.setdefault("orchestration", {"gate": "pass", "missing": [], "schema_failures": [], "hash_mismatches": [], "manifest_missing": [], "stale_artifacts": []})
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
            if report.get("stale_artifacts"):
                blocking.append(f"{stage}:stale_artifact")
    gate = "fail" if blocking else "pass"
    return {
        "schema_version": C2C_ARTIFACT_AUDIT_SCHEMA_VERSION,
        "created_at": now_utc(),
        "project_id": project_root.name,
        "gate": gate,
        "summary": {
            "checked_artifacts": checked_artifacts,
            "missing": missing_count,
            "schema_failures": schema_failure_count,
            "hash_mismatches": hash_mismatch_count,
            "stale_artifacts": stale_count,
        },
        "by_stage": by_stage,
        "blocking_reasons": blocking,
    }


def build_c2c_replay_plan(project_root: Path, *, replay_from: str = "S3_experiment") -> dict[str, Any]:
    """Freeze deterministic decision inputs for later replay."""

    frozen_inputs = [
        "plan/s2_planner/variant_scorecard.json",
        "plan/code_patches/patch_gate_report.json",
        "experiment/results/c2c_proxy_decision_report.json",
        "experiment/results/c2c_full_s3_worthiness.json",
        "experiment/results/c2c_proxy_calibration_policy.json",
        "meta/attempt_ledger.json",
        "meta/route_decision.json",
    ]
    input_hashes = {rel: _sha_or_none(project_root / rel) for rel in frozen_inputs}
    expected = read_json(project_root / "meta" / "route_decision.json", default={}) or {}
    return {
        "schema_version": C2C_REPLAY_PLAN_SCHEMA_VERSION,
        "created_at": now_utc(),
        "project_id": project_root.name,
        "replay_from": replay_from,
        "frozen_inputs": frozen_inputs,
        "input_hashes": input_hashes,
        "actions": ["rebuild_route_context", "recompute_route_decision", "compare_route_decision_hash"],
        "expected_decision_hashes": {"route_decision": _stable_hash(_normalize_route_decision(expected)) if expected else None},
    }


def build_c2c_replay_result(project_root: Path, replay_plan: dict[str, Any] | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Replay deterministic route policy decisions from frozen artifacts."""

    config = config if isinstance(config, dict) else {}
    plan = replay_plan if isinstance(replay_plan, dict) else read_json(project_root / "meta" / "c2c_replay_plan.json", default={}) or {}
    expected = read_json(project_root / "meta" / "route_decision.json", default={}) or {}
    registry = read_yaml(project_root / "meta" / "registry.yaml", default={}) or {}
    mismatches: list[dict[str, Any]] = []
    for rel, expected_hash in (plan.get("input_hashes") or {}).items():
        actual_hash = _sha_or_none(project_root / rel)
        if expected_hash != actual_hash:
            mismatches.append({"kind": "input_hash_mismatch", "path": rel, "expected": expected_hash, "actual": actual_hash})
    if not expected:
        return {
            "schema_version": C2C_REPLAY_RESULT_SCHEMA_VERSION,
            "created_at": now_utc(),
            "project_id": project_root.name,
            "status": "blocked",
            "replayed_decisions": {},
            "expected_decisions": {},
            "mismatches": [{"kind": "missing_expected_route_decision", "path": "meta/route_decision.json"}],
        }
    trigger = {
        "stage": expected.get("trigger_stage") or registry.get("current_stage") or "S3_experiment",
        "source": expected.get("trigger_source") or "replay",
        "status": "failed",
        "reason": expected.get("failure_class") or "replay",
    }
    route_context = build_route_context(project_root, registry, config, trigger=trigger)
    replayed = decide_next_route(route_context, config)
    expected_hash = _stable_hash(_normalize_route_decision(expected))
    replayed_hash = _stable_hash(_normalize_route_decision(replayed))
    if expected_hash != replayed_hash:
        mismatches.append({"kind": "route_decision_mismatch", "expected_hash": expected_hash, "actual_hash": replayed_hash})
    status = "match" if not mismatches else "mismatch"
    return {
        "schema_version": C2C_REPLAY_RESULT_SCHEMA_VERSION,
        "created_at": now_utc(),
        "project_id": project_root.name,
        "status": status,
        "replayed_decisions": {
            "route_decision": replayed.get("decision"),
            "next_stage": replayed.get("next_stage"),
            "failure_class": replayed.get("failure_class"),
            "reason_codes": replayed.get("reason_codes") or [],
            "hash": replayed_hash,
        },
        "expected_decisions": {
            "route_decision": expected.get("decision"),
            "next_stage": expected.get("next_stage"),
            "failure_class": expected.get("failure_class"),
            "reason_codes": expected.get("reason_codes") or [],
            "hash": expected_hash,
        },
        "mismatches": mismatches,
    }


def write_c2c_e2e_readiness_report(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    report = build_c2c_e2e_readiness_report(project_root, config)
    write_json(project_root / "meta" / "c2c_e2e_readiness_report.json", report)
    return report


def write_c2c_runtime_health_report(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    report = build_c2c_runtime_health_report(project_root, config)
    write_json(project_root / "meta" / "c2c_runtime_health_report.json", report)
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


def write_c2c_artifact_audit_report(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    report = build_c2c_artifact_audit_report(project_root, config)
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
    if require_hash and entry.get("sha256") and sha256_file(project_root / rel_path) != entry.get("sha256"):
        return {"kind": "hash_mismatch", "path": rel_path, "manifest": manifest_rel, "expected": entry.get("sha256"), "actual": sha256_file(project_root / rel_path)}
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
        ("meta/c2c_replay_plan.json", "c2c_replay_plan.schema.json", "orchestration"),
        ("meta/c2c_replay_result.json", "c2c_replay_result.schema.json", "orchestration"),
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
    route_path = project_root / "meta" / "route_decision.json"
    route = read_json(route_path, default={}) or {}
    if not isinstance(route, dict) or not route_path.exists():
        return []
    invalidated = ((route.get("artifact_effects") or {}).get("invalidate_artifacts") or []) if isinstance(route.get("artifact_effects"), dict) else []
    if not isinstance(invalidated, list):
        return []
    route_mtime = route_path.stat().st_mtime
    stale = []
    for rel in invalidated:
        if not isinstance(rel, str):
            continue
        path = project_root / rel
        if path.exists() and path.stat().st_mtime <= route_mtime:
            stale.append({"path": rel, "reason": "artifact older than route_decision but listed for invalidation"})
    return stale


def _sha_or_none(path: Path | None) -> str | None:
    if not path or not path.exists() or not path.is_file():
        return None
    return sha256_file(path)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
