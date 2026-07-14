from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

from auto_research.evidence import content_addressed_evidence_path, encode_canonical_evidence


def upgrade_trial_spec_v4(spec: dict) -> dict:
    result = deepcopy(spec)
    result["schema_version"] = "auto_research_trial_spec_v4"
    version_map = {
        "main_results": "auto_research_main_results_v3",
        "proxy_results": "auto_research_proxy_results_v1",
        "ablation_results": "auto_research_ablation_results_v3",
        "coverage_results": "auto_research_coverage_results_v3",
        "matched_control_results": "auto_research_matched_control_results_v3",
        "activation_evidence": "auto_research_activation_evidence_v3",
        "proxy_baseline_fingerprint": "auto_research_proxy_baseline_fingerprint_v3",
        "proxy_cache_report": "auto_research_proxy_cache_report_v3",
        "effective_proxy_policy": "auto_research_effective_proxy_policy_v3",
        "proxy_calibration_policy": "auto_research_proxy_calibration_policy_v3",
        "full_s3_readiness": "auto_research_full_s3_readiness_v3",
        "bootstrap_completion": "auto_research_bootstrap_completion_v3",
    }
    for requirement in result.get("evidence_requirements") or []:
        if set(result["protocol"]["required_phases"]) == {"proxy"} and requirement["kind"] == "main_results":
            requirement["kind"] = "proxy_results"
            requirement["requirement_id"] = "proxy-results"
        requirement["schema_version"] = version_map[requirement["kind"]]
    if set(result["protocol"]["required_phases"]) == {"proxy"}:
        result["required_artifacts"] = ["proxy_results" if item == "main_results" else item for item in result["required_artifacts"]]
    required_phases = list(result["protocol"]["required_phases"])
    datasets = [item["dataset_id"] for item in result["datasets"]]
    seeds = list(result["statistical_testing"]["seeds"])
    metrics = [item["metric_id"] for item in result["metrics"]]
    result["phase_contracts"] = [
        {
            "phase": phase,
            "datasets": datasets,
            "seeds": seeds,
            "roles": ["baseline", "candidate"] if phase == "proxy" else list(result["required_roles"]),
            "metrics": metrics,
            "evidence_kinds": [
                item["kind"]
                for item in result["evidence_requirements"]
                if phase in item["applicable_phases"] or "always" in item["applicable_phases"]
            ],
            "terminal": phase in result["protocol"]["terminal_phases"],
            "consumes_direction_budget": phase in result["protocol"]["terminal_phases"],
        }
        for phase in required_phases
    ]
    return result


def start_attempt_phase(ledger, attempt: dict, phase: str) -> dict:
    phase_execution_id = f"{phase}-{attempt['attempt_id'][:12]}-g{attempt['lifecycle_generation']}"
    producer_run_id = f"producer-{phase}-{attempt['attempt_id'][:8]}-g{attempt['lifecycle_generation']}"
    if phase == "proxy":
        return ledger.start_proxy_phase(attempt["attempt_id"], phase_execution_id=phase_execution_id, producer_run_id=producer_run_id)
    return ledger.start_full_phase(attempt["attempt_id"], phase_execution_id=phase_execution_id, producer_run_id=producer_run_id)


def build_quantitative_completion(
    project_root: Path,
    attempt: dict,
    *,
    role_values: dict[str, float],
    dataset_id: str,
    metric_id: str,
    seed: int,
    phase: str,
    producer_run_id: str | None = None,
    evidence_kind: str | None = None,
) -> dict:
    phase_execution = attempt["phase_executions"][phase]
    if not isinstance(phase_execution, dict):
        raise ValueError("Attempt phase must be started before building evidence")
    producer_run_id = producer_run_id or phase_execution["producer_run_id"]
    if producer_run_id != phase_execution["producer_run_id"]:
        raise ValueError("producer_run_id must match PhaseExecutionManifest")
    evidence_kind = evidence_kind or ("proxy_results" if phase == "proxy" else "main_results")
    schema_version = "auto_research_proxy_results_v1" if evidence_kind == "proxy_results" else "auto_research_main_results_v3"
    evidence_id = f"evidence:{attempt['attempt_id'][:8]}:{evidence_kind}"
    common = {
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "phase": phase,
        "phase_execution_id": phase_execution["phase_execution_id"],
        "phase_start_event_id": phase_execution["phase_start_event_id"],
    }
    rows = [
        {
            "phase": phase,
            "role": role,
            "command_status": "completed",
            "dataset_id": dataset_id,
            "metric_id": metric_id,
            "metric_value": value,
            "sample_manifest_hash": attempt["sample_manifest_hash"],
            "evaluator_hash": attempt["evaluator_hash"],
            "seed": seed,
            "attempt_id": attempt["attempt_id"],
            "variant_semantic_hash": attempt["variant_semantic_hash"],
            "variant_spec_hash": attempt["variant_spec_hash"],
            "trial_spec_hash": attempt["trial_spec_hash"],
            "producer_run_id": producer_run_id,
            **common,
        }
        for role, value in role_values.items()
    ]
    payload = {
        "schema_version": schema_version,
        "evidence_kind": evidence_kind,
        "evidence_id": evidence_id,
        "attempt_id": attempt["attempt_id"],
        "producer_run_id": producer_run_id,
        "direction_semantic_hash": attempt["direction_semantic_hash"],
        "direction_spec_hash": attempt["direction_spec_hash"],
        "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "protocol_hash": attempt["protocol_hash"],
        "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"],
        "cross_references": {},
        **common,
        "rows": rows,
    }
    raw = encode_canonical_evidence(payload)
    content_hash = hashlib.sha256(raw).hexdigest()
    relative_path = content_addressed_evidence_path(
        attempt_id=attempt["attempt_id"],
        producer_run_id=producer_run_id,
        evidence_kind=evidence_kind,
        content_hash=content_hash,
    )
    artifact = project_root / relative_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(raw)
    return {
        "schema_version": "auto_research_completion_evidence_v2",
        "attempt_id": attempt["attempt_id"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "entries": [
            {
                "evidence_id": evidence_id,
                "kind": evidence_kind,
                "relative_path": relative_path,
                "content_hash": content_hash,
                "schema_version": schema_version,
                "attempt_id": attempt["attempt_id"],
                "producer_run_id": producer_run_id,
                "direction_semantic_hash": attempt["direction_semantic_hash"],
                "direction_spec_hash": attempt["direction_spec_hash"],
                "variant_semantic_hash": attempt["variant_semantic_hash"],
                "variant_spec_hash": attempt["variant_spec_hash"],
                "trial_spec_hash": attempt["trial_spec_hash"],
                "protocol_hash": attempt["protocol_hash"],
                "sample_manifest_hash": attempt["sample_manifest_hash"],
                "evaluator_hash": attempt["evaluator_hash"],
                **common,
            }
        ],
    }
