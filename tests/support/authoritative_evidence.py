from __future__ import annotations

import hashlib
import tempfile
from copy import deepcopy
from pathlib import Path

from auto_research.contract_store import ContractStore, canonical_contract_bytes
from auto_research.domain_contracts import TRIAL_SPEC_SCHEMA_VERSION, canonical_hash, validate_trial_spec
from auto_research.evidence import content_addressed_evidence_path, encode_canonical_evidence
from auto_research.proxy_classifier import build_proxy_decision_policy
from auto_research.research_state import FAILURE_EVIDENCE_SCHEMA_VERSION


_DEFAULT_CONTRACT_ROOT = Path(tempfile.mkdtemp(prefix="auto-research-test-contracts-"))


def build_trial_spec_v5(spec: dict, *, project_root: Path | None = None) -> dict:
    """Rebuild a legacy test specification as a strict CAS-backed TrialSpec v5."""

    result = deepcopy(spec)
    root = Path(project_root) if project_root is not None else _DEFAULT_CONTRACT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    store = ContractStore(root)
    result["schema_version"] = TRIAL_SPEC_SCHEMA_VERSION
    legacy_manifest = result["sample_manifest"]
    sample_datasets = []
    dataset_specs = []
    for index, dataset in enumerate(legacy_manifest["datasets"]):
        dataset_id = dataset["dataset_id"]
        ordered_sample_ids = list(dataset["ordered_sample_ids"])
        source_payload = {
            "dataset_id": dataset_id,
            "source_revision": dataset["source_revision"],
            "split": dataset["split"],
            "ordered_sample_ids": ordered_sample_ids,
            "fixture_index": index,
        }
        source_blob = store.put_bytes(canonical_contract_bytes(source_payload))
        content_digest = store.digest_referenced_bytes([source_blob])
        sample_datasets.append(
            {
                "dataset_id": dataset_id,
                "source_revision": dataset["source_revision"],
                "split": dataset["split"],
                "sample_count": len(ordered_sample_ids),
                "ordered_sample_ids": ordered_sample_ids,
                "source_blobs": [source_blob],
                "content_digest": content_digest,
            }
        )
        dataset_specs.append(
            {
                "dataset_id": dataset_id,
                "split": dataset["split"],
                "sample_count": len(ordered_sample_ids),
                "sample_hash": content_digest,
            }
        )
    manifest_id = str(legacy_manifest.get("manifest_id") or "test-sample-manifest")
    if len(manifest_id) < 8:
        manifest_id = f"fixture-{manifest_id}"
    sample_manifest = {
        "schema_version": "auto_research_sample_manifest_v2",
        "manifest_id": manifest_id,
        "provenance_mode": legacy_manifest.get("provenance_mode", "synthetic"),
        "datasets": sample_datasets,
    }
    result["sample_manifest"] = sample_manifest
    result["sample_manifest_ref"] = store.put_contract(
        sample_manifest,
        contract_kind="sample_manifest",
        schema_file="sample_manifest_v2.schema.json",
    )
    result["datasets"] = dataset_specs

    legacy_evaluator = result["execution_contract"]["evaluator_provenance"]
    source_blob = store.put_bytes(canonical_contract_bytes({"source": legacy_evaluator.get("source_digest"), "evaluator_id": legacy_evaluator.get("evaluator_id")}))
    dependency_blob = store.put_bytes(canonical_contract_bytes({"dependencies": legacy_evaluator.get("dependency_digest")}))
    config_blob = store.put_bytes(canonical_contract_bytes({"config": legacy_evaluator.get("config_hash") or legacy_evaluator.get("config_digest")}))
    evaluator_id = str(legacy_evaluator.get("evaluator_id") or "test-evaluator")
    if len(evaluator_id) < 8:
        evaluator_id = f"fixture-{evaluator_id}"
    evaluator_manifest = {
        "schema_version": "auto_research_evaluator_manifest_v2",
        "evaluator_id": evaluator_id,
        "provenance_mode": legacy_evaluator.get("provenance_mode", "synthetic"),
        "source_blobs": [source_blob],
        "dependency_blobs": [dependency_blob],
        "config_blob": config_blob,
        "config_digest": store.digest_referenced_bytes([config_blob]),
        "source_digest": store.digest_referenced_bytes([source_blob]),
        "dependency_digest": store.digest_referenced_bytes([dependency_blob]),
    }
    evaluator_ref = store.put_contract(
        evaluator_manifest,
        contract_kind="evaluator_manifest",
        schema_file="evaluator_manifest_v2.schema.json",
    )
    result["execution_contract"]["evaluator_provenance"] = evaluator_manifest
    result["execution_contract"]["evaluator_manifest_ref"] = evaluator_ref
    result["execution_contract"]["evaluator_hash"] = canonical_hash(evaluator_manifest)

    version_map = {
        "main_results": "auto_research_main_results_v3",
        "proxy_results": "auto_research_proxy_results_v1",
        "ablation_results": "auto_research_ablation_results_v3",
        "coverage_results": "auto_research_coverage_results_v3",
        "matched_control_results": "auto_research_matched_control_results_v3",
        "activation_evidence": "auto_research_activation_evidence_v3",
        "proxy_baseline_fingerprint": "auto_research_proxy_baseline_fingerprint_v3",
        "proxy_cache_report": "auto_research_proxy_cache_report_v3",
        "full_s3_readiness": "auto_research_full_s3_readiness_v3",
        "bootstrap_completion": "auto_research_bootstrap_completion_v3",
    }
    requirements = []
    for requirement in result.get("evidence_requirements") or []:
        if requirement["kind"] in {"effective_proxy_policy", "proxy_calibration_policy", "proxy_decision_report"}:
            continue
        if set(result["protocol"]["required_phases"]) == {"proxy"} and requirement["kind"] == "main_results":
            requirement["kind"] = "proxy_results"
            requirement["requirement_id"] = "proxy-results"
        requirement["schema_version"] = version_map[requirement["kind"]]
        requirements.append(requirement)
    result["evidence_requirements"] = requirements

    required_phases = list(result["protocol"]["required_phases"])
    datasets = [item["dataset_id"] for item in result["datasets"]]
    seeds = list(result["statistical_testing"]["seeds"])
    metrics = [item["metric_id"] for item in result["metrics"]]
    if "proxy" in required_phases:
        mandatory = {
            "proxy_results": ("proxy-results", "auto_research_proxy_results_v1"),
            "activation_evidence": ("activation", "auto_research_activation_evidence_v3"),
            "proxy_baseline_fingerprint": ("proxy-baseline", "auto_research_proxy_baseline_fingerprint_v3"),
            "proxy_cache_report": ("proxy-cache", "auto_research_proxy_cache_report_v3"),
            "bootstrap_completion" if result["protocol"]["proxy_terminal_allowed"] else "full_s3_readiness": (
                "bootstrap" if result["protocol"]["proxy_terminal_allowed"] else "readiness",
                "auto_research_bootstrap_completion_v3" if result["protocol"]["proxy_terminal_allowed"] else "auto_research_full_s3_readiness_v3",
            ),
        }
        present = {item["kind"] for item in requirements}
        for kind, (requirement_id, schema_version) in mandatory.items():
            if kind not in present:
                requirements.append(
                    {
                        "requirement_id": requirement_id,
                        "kind": kind,
                        "required": True,
                        "applicable_phases": ["proxy"],
                        "schema_version": schema_version,
                    }
                )
    result["required_artifacts"] = [item["kind"] for item in requirements]
    result["phase_contracts"] = [
        {
            "phase": phase,
            "datasets": datasets,
            "seeds": seeds,
            "roles": ["baseline", "candidate"] if phase == "proxy" else list(result["required_roles"]),
            "metrics": [result["primary_metric_id"]],
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
    if "proxy" in required_phases:
        proxy_contract = next(item for item in result["phase_contracts"] if item["phase"] == "proxy")
        primary_constraint = next(
            item
            for item in result["acceptance_constraints"]
            if item["kind"] == "minimum_mean_delta" and item.get("metric_id") == result["primary_metric_id"]
        )
        regression_constraint = next(
            (item for item in result["acceptance_constraints"] if item["kind"] == "per_dataset_maximum_regression"),
            None,
        )
        result["proxy_decision_policy"] = build_proxy_decision_policy(
            primary_metric_id=result["primary_metric_id"],
            objective=next(item["objective"] for item in result["metrics"] if item["metric_id"] == result["primary_metric_id"]),
            aggregation="paired_mean",
            datasets=proxy_contract["datasets"],
            seeds=proxy_contract["seeds"],
            metric_ids=proxy_contract["metrics"],
            roles=proxy_contract["roles"],
            aggregate_improvement_threshold=float(primary_constraint["threshold"]),
            per_dataset_maximum_regression=float(regression_constraint["threshold"] if regression_constraint else 0.0),
            activation_surface_ids=["src/model.py"],
            readiness_check_ids=[] if result["protocol"]["proxy_terminal_allowed"] else ["proxy-ready-for-full"],
            evidence_kinds=proxy_contract["evidence_kinds"],
            mode="terminal_bootstrap" if result["protocol"]["proxy_terminal_allowed"] else "gate_to_full",
        )
    else:
        result["proxy_decision_policy"] = None
    validate_trial_spec(result)
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


def build_bootstrap_completion(
    project_root: Path,
    attempt: dict,
    trial_spec: dict,
    *,
    baseline_value: float = 0.5,
    candidate_value: float = 0.7,
) -> dict:
    """Build the complete strict bootstrap inventory through production staging."""

    from auto_research.agents.experiment import _c2c_strict_evidence_inventory, _stage_evidence_inventory

    inventory = _c2c_strict_evidence_inventory(
        project_root=project_root,
        attempt=attempt,
        trial_spec=trial_spec,
        comparison_candidate={
            "metrics": {"mean": candidate_value, "datasets": {"fake": candidate_value}},
            "proxy_screen": {
                "metrics": {"mean": candidate_value, "datasets": {"fake": candidate_value}},
                "baseline_metrics": {"mean": baseline_value, "datasets": {"fake": baseline_value}},
            },
        },
        baseline={"mean": baseline_value, "datasets": {"fake": baseline_value}},
        simulate=True,
    )
    return _stage_evidence_inventory(
        project_root=project_root,
        attempt=attempt,
        trial_spec=trial_spec,
        inventory=inventory,
    )


def build_failure_evidence_v4(
    project_root: Path,
    attempt: dict,
    *,
    failure_class: str,
    suffix: str,
    exit_code: int = 1,
) -> dict:
    """Stage canonical non-resource failure and command-receipt evidence."""

    if failure_class not in {
        "implementation_failure",
        "activation_failure",
        "integrity_failure",
        "safety_failure",
    }:
        raise ValueError("build_failure_evidence_v4 only supports non-resource failures")
    running_phase = next(
        (phase for phase in ("proxy", "full") if attempt["phases"].get(phase) == "RUNNING"),
        None,
    )
    if running_phase is None:
        raise ValueError("Attempt must have an authoritative running phase")
    phase_execution = attempt["phase_executions"][running_phase]
    if not isinstance(phase_execution, dict):
        raise ValueError("Attempt running phase is missing its execution manifest")
    source_phase = "activation" if failure_class == "activation_failure" else running_phase
    producer_run_id = phase_execution["producer_run_id"]
    identity = {
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
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "phase": source_phase,
        "phase_execution_id": phase_execution["phase_execution_id"],
        "phase_start_event_id": phase_execution["phase_start_event_id"],
    }
    receipt = {
        "schema_version": "auto_research_command_result_evidence_v1",
        "evidence_kind": "command_result_evidence",
        "evidence_id": f"command-result-{attempt['lifecycle_generation']}-{suffix}",
        **identity,
        "command_id": f"failure-command-{attempt['lifecycle_generation']}-{suffix}",
        "command": ["python", "failure_fixture.py", "--phase", source_phase],
        "working_directory": "runner",
        "started_at": "2026-07-14T00:00:00Z",
        "finished_at": "2026-07-14T00:00:01Z",
        "command_status": "failed",
        "exit_code": exit_code,
        "stdout_hash": "a" * 64,
        "stderr_hash": "b" * 64,
    }
    receipt_raw = encode_canonical_evidence(receipt)
    receipt_hash = hashlib.sha256(receipt_raw).hexdigest()
    receipt_path = (
        project_root
        / "experiment"
        / "attempts"
        / attempt["attempt_id"]
        / producer_run_id
        / "command_result_evidence"
        / f"{receipt_hash}.json"
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(receipt_raw)

    command_status = "integrity_blocked" if failure_class in {"integrity_failure", "safety_failure"} else "failed"
    evidence = {
        "schema_version": FAILURE_EVIDENCE_SCHEMA_VERSION,
        "evidence_kind": "failure_evidence",
        "evidence_id": f"failure-evidence-{attempt['lifecycle_generation']}-{suffix}",
        **identity,
        "cross_references": {"command_result_evidence_hash": receipt_hash},
        "source_state": attempt["state"],
        "source_phase": source_phase,
        "failure_class": failure_class,
        "command_status": command_status,
        "exit_code": exit_code,
        "reason": f"verified {failure_class}",
        "observed_at": "2026-07-14T00:00:01Z",
        "log_hash": receipt["stderr_hash"],
    }
    evidence_raw = encode_canonical_evidence(evidence)
    evidence_hash = hashlib.sha256(evidence_raw).hexdigest()
    evidence_path = (
        project_root
        / "experiment"
        / "attempts"
        / attempt["attempt_id"]
        / producer_run_id
        / "failure_evidence"
        / f"{evidence_hash}.json"
    )
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_bytes(evidence_raw)
    return evidence


def build_resource_failure_evidence_v4(
    project_root: Path,
    attempt: dict,
    *,
    failure_class: str,
    suffix: str,
    resource_type: str = "system_memory",
    resource_id: str = "memory-0",
    required_capacity: float = 10.0,
    observed_capacity: float = 1.0,
    unit: str = "bytes",
    exit_code: int = 137,
) -> dict:
    """Stage canonical resource-pause failure and its immutable probe."""

    if failure_class not in {"resource_pause", "oom_retry"}:
        raise ValueError("build_resource_failure_evidence_v4 only supports resource failures")
    running_phase = next(
        (phase for phase in ("proxy", "full") if attempt["phases"].get(phase) == "RUNNING"),
        None,
    )
    if running_phase is None:
        raise ValueError("Attempt must have an authoritative running phase")
    phase_execution = attempt["phase_executions"][running_phase]
    if not isinstance(phase_execution, dict):
        raise ValueError("Attempt running phase is missing its execution manifest")
    producer_run_id = phase_execution["producer_run_id"]
    identity = {
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
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "phase": running_phase,
        "phase_execution_id": phase_execution["phase_execution_id"],
        "phase_start_event_id": phase_execution["phase_start_event_id"],
    }
    probe = {
        "schema_version": "auto_research_resource_probe_evidence_v3",
        "evidence_kind": "resource_probe",
        "evidence_id": f"resource-probe-{attempt['lifecycle_generation']}-{suffix}",
        **identity,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "required_capacity": required_capacity,
        "observed_capacity": observed_capacity,
        "unit": unit,
        "probe_status": "insufficient",
        "observed_at": "2026-07-14T00:00:00Z",
    }
    probe_raw = encode_canonical_evidence(probe)
    probe_hash = hashlib.sha256(probe_raw).hexdigest()
    probe_path = project_root / content_addressed_evidence_path(
        attempt_id=attempt["attempt_id"],
        producer_run_id=producer_run_id,
        evidence_kind="resource_probe",
        content_hash=probe_hash,
    )
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    probe_path.write_bytes(probe_raw)

    evidence = {
        "schema_version": FAILURE_EVIDENCE_SCHEMA_VERSION,
        "evidence_kind": "failure_evidence",
        "evidence_id": f"failure-evidence-{attempt['lifecycle_generation']}-{suffix}",
        **identity,
        "cross_references": {"resource_probe_hash": probe_hash},
        "source_state": attempt["state"],
        "source_phase": running_phase,
        "failure_class": failure_class,
        "command_status": "resource_paused",
        "exit_code": exit_code,
        "reason": f"verified {failure_class}",
        "observed_at": "2026-07-14T00:00:01Z",
        "log_hash": probe_hash,
    }
    evidence_raw = encode_canonical_evidence(evidence)
    evidence_hash = hashlib.sha256(evidence_raw).hexdigest()
    evidence_path = project_root / content_addressed_evidence_path(
        attempt_id=attempt["attempt_id"],
        producer_run_id=producer_run_id,
        evidence_kind="failure_evidence",
        content_hash=evidence_hash,
    )
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_bytes(evidence_raw)
    return evidence
