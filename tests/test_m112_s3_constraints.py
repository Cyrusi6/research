from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.domain_contracts import (
    TRIAL_SPEC_SCHEMA_VERSION,
    canonical_hash,
    classify_trial_result,
    trial_spec_hash,
    validate_trial_spec,
)
from auto_research.evidence import EVIDENCE_SCHEMA_VERSIONS, content_addressed_evidence_path, encode_canonical_evidence
from auto_research.s3_validation import S3ValidationError, validate_trial_precommit
from auto_research.utils import write_json
from test_authoritative_state_machine import _direction, _initialize, _reserve, _trial_spec, _variant
from test_m113_evidence_contracts import _attempt as _contract_attempt
from test_m113_evidence_contracts import _main_inventory, _trial_spec as _contract_trial_spec
from support.authoritative_evidence import refresh_phase_command_plans


def _classify(spec: dict, values: dict[tuple[str, int], tuple[float, float]]) -> dict:
    manifest, evidence_bytes = _main_inventory(spec, values)
    return classify_trial_result(
        attempt=_contract_attempt(spec),
        trial_spec=spec,
        evidence_manifest=manifest,
        evidence_bytes=evidence_bytes,
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["protocol"].__setitem__("protocol_id", "protocol-2"),
        lambda value: value["sample_manifest"].__setitem__("manifest_id", "manifest-2"),
        lambda value: value["datasets"][0].__setitem__("sample_count", 9),
        lambda value: value["metrics"][0].__setitem__("objective", "minimize"),
        lambda value: value.__setitem__("primary_metric_id", "secondary"),
        lambda value: value["statistical_testing"].__setitem__("seeds", [8]),
        lambda value: value["required_roles"].append("coverage"),
        lambda value: value["acceptance_constraints"][0].__setitem__("threshold", 0.06),
        lambda value: value["execution_contract"]["runtime_config"].__setitem__("batch_size", 4),
        lambda value: value["required_artifacts"].append("coverage_results"),
        lambda value: value["evidence_requirements"][0].__setitem__("schema_version", "forged-v9"),
        lambda value: value["phase_contracts"][0]["command_plan"]["commands"][0]["argv"].append("--changed"),
    ],
)
def test_trial_spec_hash_changes_for_every_authoritative_field(mutate) -> None:
    original = _contract_trial_spec()
    changed = deepcopy(original)
    mutate(changed)
    assert trial_spec_hash(changed) != trial_spec_hash(original)


def test_trial_spec_v6_is_closed_and_v5_is_rejected() -> None:
    spec = _contract_trial_spec()
    spec["unexpected"] = True
    with pytest.raises(ValueError, match="Additional properties"):
        validate_trial_spec(spec)
    spec = _contract_trial_spec()
    spec["schema_version"] = "auto_research_trial_spec_v5"
    with pytest.raises(ValueError, match=TRIAL_SPEC_SCHEMA_VERSION):
        validate_trial_spec(spec)


def test_mean_improvement_with_dataset_regression_is_rejected() -> None:
    spec = _contract_trial_spec(objective="maximize", datasets=("d1", "d2"))
    result = _classify(spec, {("d1", 7): (0.5, 0.9), ("d2", 7): (0.5, 0.39)})
    assert result["primary_metric_summary"]["delta"] > 0
    assert result["outcome_classification"] == "rejected"
    constraint = next(item for item in result["constraint_results"] if item["constraint_id"] == "dataset-regression")
    assert constraint["status"] == "FAIL"
    assert constraint["observed"]["deltas"]["d2"] == pytest.approx(-0.11)


def _with_role_requirement(role: str, kind: str, artifact: str, threshold: float = 0.0) -> dict:
    spec = _contract_trial_spec()
    spec["required_roles"].append(role)
    spec["acceptance_constraints"].append(
        {
            "constraint_id": role,
            "kind": kind,
            "hard": True,
            "metric_id": "score",
            "threshold": threshold,
            "objective": "maximize",
        }
    )
    spec["required_artifacts"].append(artifact)
    spec["evidence_requirements"].append(
        {
            "requirement_id": artifact,
            "kind": artifact,
            "required": True,
            "applicable_phases": ["full"],
            "schema_version": EVIDENCE_SCHEMA_VERSIONS[artifact],
        }
    )
    full_contract = next(item for item in spec["phase_contracts"] if item["phase"] == "full")
    full_contract["roles"].append(role)
    full_contract["evidence_kinds"].append(artifact)
    refresh_phase_command_plans(spec)
    validate_trial_spec(spec)
    return spec


def _add_role_evidence(
    spec: dict,
    manifest: dict,
    evidence_bytes: dict[str, bytes],
    *,
    kind: str,
    role: str,
    value: float,
) -> None:
    attempt = _contract_attempt(spec)
    phase_execution = attempt["phase_executions"]["full"]
    producer_run_id = phase_execution["producer_run_id"]
    evidence_id = f"evidence:{role}:attempt-1"
    phase = "full"
    common = {
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "phase_execution_id": phase_execution["phase_execution_id"],
        "phase_start_event_id": phase_execution["phase_start_event_id"],
    }
    row = {
        "phase": phase,
        "role": role,
        "dataset_id": "d1",
        "metric_id": "score",
        "seed": 7,
        "metric_value": value,
        "command_status": "completed",
        "attempt_id": attempt["attempt_id"],
        "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"],
        "producer_run_id": producer_run_id,
        **common,
    }
    payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSIONS[kind],
        "evidence_kind": kind,
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
        "phase": phase,
        "rows": [row],
    }
    raw = encode_canonical_evidence(payload)
    digest = hashlib.sha256(raw).hexdigest()
    relative_path = content_addressed_evidence_path(
        attempt_id=attempt["attempt_id"],
        producer_run_id=producer_run_id,
        evidence_kind=kind,
        content_hash=digest,
    )
    manifest["entries"].append(
        {
            "evidence_id": evidence_id,
            "kind": kind,
            "relative_path": relative_path,
            "content_hash": digest,
            "schema_version": payload["schema_version"],
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
            "phase": phase,
            "cross_references": {},
            "command_id": manifest["entries"][0]["command_id"],
            "command_hash": manifest["entries"][0]["command_hash"],
            "command_plan_hash": manifest["entries"][0]["command_plan_hash"],
            "receipt_ref": deepcopy(manifest["entries"][0]["receipt_ref"]),
            "receipt_hash": manifest["entries"][0]["receipt_hash"],
            "output_ref": {
                "schema_version": "auto_research_contract_blob_v1",
                "algorithm": "sha256",
                "digest": digest,
                "size_bytes": len(raw),
                "relative_path": f"meta/contracts/sha256/{digest[:2]}/{digest}.json",
            },
            "completed_event_id": manifest["entries"][0]["completed_event_id"],
        }
    )
    evidence_bytes[evidence_id] = raw


@pytest.mark.parametrize(
    "role,kind,artifact",
    [
        ("ablation", "required_ablation_contrast", "ablation_results"),
        ("matched_control", "matched_control_constraint", "matched_control_results"),
        ("coverage", "coverage_constraint", "coverage_results"),
    ],
)
def test_missing_required_ablation_or_control_is_not_an_outcome(role: str, kind: str, artifact: str) -> None:
    spec = _with_role_requirement(role, kind, artifact)
    manifest, evidence_bytes = _main_inventory(spec, {("d1", 7): (0.5, 0.7)})
    with pytest.raises(ValueError, match=f"required {role} dataset/seed coverage is missing"):
        classify_trial_result(
            attempt=_contract_attempt(spec),
            trial_spec=spec,
            evidence_manifest=manifest,
            evidence_bytes=evidence_bytes,
        )


def test_completed_ablation_hard_failure_is_rejected_outcome() -> None:
    spec = _with_role_requirement("ablation", "required_ablation_contrast", "ablation_results", threshold=0.1)
    manifest, evidence_bytes = _main_inventory(spec, {("d1", 7): (0.5, 0.7)})
    _add_role_evidence(spec, manifest, evidence_bytes, kind="ablation_results", role="ablation", value=0.65)
    result = classify_trial_result(
        attempt=_contract_attempt(spec),
        trial_spec=spec,
        evidence_manifest=manifest,
        evidence_bytes=evidence_bytes,
    )
    assert result["method_evaluable"] is True
    assert result["outcome_classification"] == "rejected"
    assert result["all_hard_constraints_passed"] is False


def test_observation_cannot_fall_back_to_arbitrary_first_artifact() -> None:
    spec = _contract_trial_spec()
    manifest, evidence_bytes = _main_inventory(spec, {("d1", 7): (0.5, 0.7)})
    diagnostic = classify_trial_result(
        attempt=_contract_attempt(spec),
        trial_spec=spec,
        evidence_manifest=manifest,
        evidence_bytes=evidence_bytes,
    )
    diagnostic["observations"][0]["raw_artifact_path"] = "experiment/raw/env.json"
    with pytest.raises(ValueError, match="diagnostic TrialResult"):
        classify_trial_result(
            attempt=_contract_attempt(spec),
            trial_spec=spec,
            evidence_manifest=manifest,
            evidence_bytes=evidence_bytes,
            diagnostic_result=diagnostic,
        )


def _completed_trial(tmp_path: Path) -> tuple[dict, dict, dict, dict]:
    direction = _direction()
    variant = _variant(direction, 1)
    ledger_root = tmp_path
    from auto_research.research_state import ResearchEventLedger
    from test_authoritative_state_machine import _complete

    ledger = ResearchEventLedger(ledger_root)
    _initialize(ledger, direction, variant)
    attempt = _reserve(ledger, direction, variant)
    _complete(ledger, attempt, outcome="accepted")
    state = ledger.state()
    return direction, variant, state["attempts"][attempt["attempt_id"]], state["trial_results"][attempt["attempt_id"]]


@pytest.mark.parametrize("attack", ["projection_drift", "artifact_hash", "manifest_version", "cross_reference"])
def test_shared_s3_validator_fails_closed_for_frozen_contract_and_evidence(tmp_path: Path, attack: str) -> None:
    direction, variant, attempt, trial = _completed_trial(tmp_path)
    spec = attempt["frozen_trial_spec"]
    if attack == "projection_drift":
        changed = deepcopy(spec)
        changed["protocol"]["protocol_id"] = "forged"
        write_json(tmp_path / "plan" / "trial_spec.json", changed)
    elif attack == "artifact_hash":
        artifact_path = tmp_path / next(iter(trial["raw_artifacts"]))
        artifact_path.write_text("{}", encoding="utf-8")
    elif attack == "manifest_version":
        trial["evidence_manifest"]["entries"][0]["schema_version"] = "forged-v9"
    else:
        trial["evidence_manifest"]["entries"][0]["cross_references"] = {"policy_hash": "a" * 64}
    with pytest.raises(S3ValidationError):
        validate_trial_precommit(
            project_root=tmp_path,
            direction=direction,
            variant=variant,
            attempt=attempt,
            trial_spec=spec,
            trial_result=trial,
        )
