from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from auto_research.domain_contracts import (
    EVIDENCE_MANIFEST_SCHEMA_VERSION,
    EXECUTION_OBSERVATION_SCHEMA_VERSION,
    TRIAL_SPEC_SCHEMA_VERSION,
    acceptance_contract_hash,
    build_direction_spec,
    build_variant_spec,
    canonical_hash,
    classify_trial_result,
    trial_spec_hash,
    validate_trial_spec,
)
from auto_research.s3_validation import S3ValidationError, validate_trial_precommit
from auto_research.utils import sha256_file, write_json


def _trial_spec(*, datasets: tuple[str, ...] = ("d1",), roles: tuple[str, ...] = ("baseline", "candidate")) -> dict:
    manifest_body = {"manifest_id": "manifest-1", "datasets": list(datasets)}
    runtime = {"batch_size": 2, "precision": "fp32"}
    spec = {
        "schema_version": TRIAL_SPEC_SCHEMA_VERSION,
        "protocol": {
            "protocol_id": "protocol-1",
            "required_phases": ["full"],
            "terminal_phases": ["full"],
            "proxy_terminal_allowed": False,
            "aggregation": "mean",
        },
        "sample_manifest": {**manifest_body, "content_hash": canonical_hash(manifest_body)},
        "datasets": [
            {"dataset_id": dataset, "split": "test", "sample_count": 8, "sample_hash": canonical_hash({"dataset": dataset})}
            for dataset in datasets
        ],
        "metrics": [{"metric_id": "accuracy", "objective": "maximize", "aggregation": "mean", "role": "primary"}],
        "primary_metric_id": "accuracy",
        "statistical_testing": {"method": "none", "seeds": [7], "require_complete_seed_coverage": True},
        "required_roles": list(roles),
        "acceptance_constraints": [
            {
                "constraint_id": "primary-delta",
                "kind": "minimum_mean_delta",
                "hard": True,
                "metric_id": "accuracy",
                "threshold": 0.05,
                "objective": "maximize",
            }
        ],
        "execution_contract": {
            "runtime_config": runtime,
            "runtime_config_hash": canonical_hash(runtime),
            "evaluator_hash": canonical_hash({"evaluator": "accuracy-v1"}),
            "command_contract_hash": canonical_hash(["evaluate"]),
        },
        "required_artifacts": ["main_results"],
        "evidence_requirements": [
            {
                "requirement_id": "main-results",
                "kind": "main_results",
                "required": True,
                "applicable_phases": ["full"],
                "schema_version": "auto_research_main_results_v1",
            }
        ],
    }
    validate_trial_spec(spec)
    return spec


def _direction() -> dict:
    return build_direction_spec(
        {
            "direction_id": "direction-1",
            "research_question": "Does the intervention improve accuracy?",
            "mechanism_invariants": {"causal_hypothesis": "routing matters", "target_mediator": "routing", "invariants": ["fixed data"]},
            "falsification_conditions": ["no accuracy change"],
            "support_claim_ids": ["s1"],
            "counter_claim_ids": ["c1"],
            "implementation_surface_ids": ["src/model.py"],
            "metric_signature": {"primary": "accuracy", "direction": "increase"},
            "benchmark_contract_hash": canonical_hash({"d": 1}),
            "variant_space": {"mutable_axes": ["operation"], "immutable_axes": ["data"], "forbidden_combinations": []},
            "s2_entry_conditions": ["gate pass"],
            "return_to_s1_conditions": ["five rejects"],
            "lineage": {"s1_run_id": "s1", "iteration": 1, "input_manifest_hash": canonical_hash({"i": 1})},
        }
    )


def _variant(direction: dict) -> dict:
    return build_variant_spec(
        direction,
        {
            "variant_id": "variant-1",
            "variation_coordinates": {"operation": "gated-routing"},
            "intervention": {"summary": "gated routing", "algorithm_operations": ["gate"], "configuration": {"strength": 1}},
            "hypothesis": "Gating improves accuracy.",
            "null_hypothesis": "Gating does not improve accuracy.",
            "alternative_hypothesis": "Gating improves accuracy.",
            "controlled_variables": {"data": "fixed"},
            "nuisance_variables": ["noise"],
            "implementation_surface_ids": ["src/model.py"],
            "expected_metric_signature": {"primary": "accuracy", "direction": "increase"},
            "falsification_conditions": ["accuracy does not improve"],
            "ablation": {"switch": "disable_gate"},
            "resource_budget": {"max_wall_seconds": 60, "max_retries": 1},
            "failure_routing": {"implementation": "REPAIR_IMPLEMENTATION", "method": "PROPOSE_NEXT_VARIANT"},
            "lineage": {"s2_run_id": "s2", "iteration": 1, "direction_spec_hash": direction["direction_spec_hash"], "feedback_from_attempt_ids": []},
        },
    )


def _attempt(spec: dict, direction: dict, variant: dict) -> dict:
    return {
        "attempt_id": "attempt-12345678",
        "profile": "standard",
        "attempt_kind": "full",
        "direction_id": direction["direction_id"],
        "direction_semantic_hash": direction["direction_semantic_hash"],
        "direction_spec_hash": direction["direction_spec_hash"],
        "variant_id": variant["variant_id"],
        "variant_semantic_hash": variant["variant_semantic_hash"],
        "variant_spec_hash": variant["variant_spec_hash"],
        "trial_spec_snapshot": deepcopy(spec),
        "trial_spec_hash": trial_spec_hash(spec),
        "acceptance_contract_hash": acceptance_contract_hash(spec),
        "protocol_hash": canonical_hash(spec["protocol"]),
        "sample_manifest_hash": canonical_hash(spec["sample_manifest"]),
        "runtime_config_hash": spec["execution_contract"]["runtime_config_hash"],
        "evaluator_hash": spec["execution_contract"]["evaluator_hash"],
        "attempt_input_hash": canonical_hash({"attempt": 1, "trial_spec": spec}),
        "implementation_hash": canonical_hash({"implementation": 1}),
        "seeds": spec["statistical_testing"]["seeds"],
        "state": "FULL_RUNNING",
        "phases": {"proxy": "PENDING", "full": "RUNNING"},
        "lifecycle_generation": 0,
        "consumes_direction_budget": True,
        "reserved_slot": True,
        "method_evaluable": False,
    }


def _artifacts(tmp_path: Path, attempt: dict, spec: dict, *, extra_kinds: tuple[str, ...] = ()) -> tuple[dict[str, str], dict]:
    entries = []
    raw_artifacts = {}
    for kind in ("main_results", *extra_kinds):
        relative = f"experiment/raw/{kind}.json"
        write_json(tmp_path / relative, {"schema_version": f"auto_research_{kind}_v1", "attempt_id": attempt["attempt_id"]})
        digest = sha256_file(tmp_path / relative)
        raw_artifacts[relative] = digest
        entries.append(
            {
                "evidence_id": f"evidence:{kind}",
                "kind": kind,
                "relative_path": relative,
                "content_hash": digest,
                "schema_version": f"auto_research_{kind}_v1",
                "attempt_id": attempt["attempt_id"],
                "variant_spec_hash": attempt["variant_spec_hash"],
                "trial_spec_hash": trial_spec_hash(spec),
                "cross_references": {},
            }
        )
    return raw_artifacts, {
        "schema_version": EVIDENCE_MANIFEST_SCHEMA_VERSION,
        "trial_spec_hash": trial_spec_hash(spec),
        "attempt_id": attempt["attempt_id"],
        "entries": entries,
    }


def _observations(spec: dict, attempt: dict, raw_artifacts: dict[str, str], values: dict[str, dict[str, float]]) -> list[dict]:
    path = "experiment/raw/main_results.json"
    result = []
    for dataset_id, role_values in values.items():
        for role, value in role_values.items():
            result.append(
                {
                    "schema_version": EXECUTION_OBSERVATION_SCHEMA_VERSION,
                    "observation_id": f"obs:{dataset_id}:{role}:7",
                    "phase": "ablation" if role == "ablation" else "full",
                    "role": role,
                    "command_status": "completed",
                    "dataset_id": dataset_id,
                    "metric_id": "accuracy",
                    "metric_value": value,
                    "sample_manifest_hash": attempt["sample_manifest_hash"],
                    "evaluator_hash": attempt["evaluator_hash"],
                    "seed": 7,
                    "raw_artifact_path": path,
                    "raw_artifact_hash": raw_artifacts[path],
                }
            )
    return result


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
        lambda value: value["evidence_requirements"][0].__setitem__("schema_version", "auto_research_main_results_v2"),
    ],
)
def test_trial_spec_hash_changes_for_every_authoritative_field(mutate) -> None:
    original = _trial_spec()
    changed = deepcopy(original)
    mutate(changed)
    assert trial_spec_hash(changed) != trial_spec_hash(original)


def test_trial_spec_v2_is_closed_and_old_version_is_rejected() -> None:
    spec = _trial_spec()
    spec["extra"] = True
    with pytest.raises(ValueError, match="Additional properties"):
        validate_trial_spec(spec)
    spec = _trial_spec()
    spec["schema_version"] = "auto_research_trial_spec_v1"
    with pytest.raises(ValueError, match="auto_research_trial_spec_v2"):
        validate_trial_spec(spec)


def test_mean_improvement_with_dataset_regression_is_rejected(tmp_path: Path) -> None:
    spec = _trial_spec(datasets=("d1", "d2"))
    spec["acceptance_constraints"].append(
        {"constraint_id": "dataset-regression", "kind": "per_dataset_maximum_regression", "hard": True, "metric_id": "accuracy", "threshold": 0.05, "objective": "minimize"}
    )
    validate_trial_spec(spec)
    direction, variant = _direction(), None
    variant = _variant(direction)
    attempt = _attempt(spec, direction, variant)
    raw, manifest = _artifacts(tmp_path, attempt, spec)
    observations = _observations(spec, attempt, raw, {"d1": {"baseline": 0.5, "candidate": 0.9}, "d2": {"baseline": 0.5, "candidate": 0.4}})
    result = classify_trial_result(attempt=attempt, trial_spec=spec, observations=observations, raw_artifacts=raw, evidence_manifest=manifest)
    assert result["primary_metric_summary"]["delta"] > 0
    assert result["outcome_classification"] == "rejected"
    assert next(item for item in result["constraint_results"] if item["constraint_id"] == "dataset-regression")["status"] == "FAIL"


@pytest.mark.parametrize("role,kind,artifact", [("ablation", "required_ablation_contrast", "ablation_results"), ("matched_control", "matched_control_constraint", "matched_control_results"), ("coverage", "coverage_constraint", "coverage_results")])
def test_missing_required_ablation_or_control_is_not_an_outcome(tmp_path: Path, role: str, kind: str, artifact: str) -> None:
    spec = _trial_spec(roles=("baseline", "candidate", role))
    spec["acceptance_constraints"].append({"constraint_id": role, "kind": kind, "hard": True, "metric_id": "accuracy", "threshold": 0.0, "objective": "maximize"})
    spec["required_artifacts"].append(artifact)
    spec["evidence_requirements"].append({"requirement_id": artifact, "kind": artifact, "required": True, "applicable_phases": ["full" if role != "ablation" else "ablation"], "schema_version": f"auto_research_{artifact}_v1"})
    validate_trial_spec(spec)
    direction = _direction()
    attempt = _attempt(spec, direction, _variant(direction))
    raw, manifest = _artifacts(tmp_path, attempt, spec, extra_kinds=(artifact,))
    observations = _observations(spec, attempt, raw, {"d1": {"baseline": 0.5, "candidate": 0.7}})
    with pytest.raises(ValueError, match=f"required {role} dataset/seed coverage is missing"):
        classify_trial_result(attempt=attempt, trial_spec=spec, observations=observations, raw_artifacts=raw, evidence_manifest=manifest)


def test_completed_ablation_hard_failure_is_rejected_outcome(tmp_path: Path) -> None:
    spec = _trial_spec(roles=("baseline", "candidate", "ablation"))
    spec["acceptance_constraints"].append({"constraint_id": "ablation", "kind": "required_ablation_contrast", "hard": True, "metric_id": "accuracy", "threshold": 0.1, "objective": "maximize"})
    spec["required_artifacts"].append("ablation_results")
    spec["evidence_requirements"].append({"requirement_id": "ablation", "kind": "ablation_results", "required": True, "applicable_phases": ["ablation"], "schema_version": "auto_research_ablation_results_v1"})
    validate_trial_spec(spec)
    direction = _direction()
    attempt = _attempt(spec, direction, _variant(direction))
    raw, manifest = _artifacts(tmp_path, attempt, spec, extra_kinds=("ablation_results",))
    observations = _observations(spec, attempt, raw, {"d1": {"baseline": 0.5, "candidate": 0.7, "ablation": 0.65}})
    result = classify_trial_result(attempt=attempt, trial_spec=spec, observations=observations, raw_artifacts=raw, evidence_manifest=manifest)
    assert result["method_evaluable"] is True
    assert result["outcome_classification"] == "rejected"
    assert result["all_hard_constraints_passed"] is False


def test_observation_cannot_fall_back_to_arbitrary_first_artifact(tmp_path: Path) -> None:
    spec = _trial_spec()
    direction = _direction()
    attempt = _attempt(spec, direction, _variant(direction))
    raw, manifest = _artifacts(tmp_path, attempt, spec)
    raw["experiment/raw/env.json"] = canonical_hash({"env": 1})
    observations = _observations(spec, attempt, raw, {"d1": {"baseline": 0.5, "candidate": 0.7}})
    observations[0]["raw_artifact_path"] = "experiment/raw/env.json"
    with pytest.raises(ValueError, match="exact registered result artifact"):
        classify_trial_result(attempt=attempt, trial_spec=spec, observations=observations, raw_artifacts=raw, evidence_manifest=manifest)


@pytest.mark.parametrize("attack", ["projection_drift", "artifact_hash", "manifest_version", "cross_reference"])
def test_shared_s3_validator_fails_closed_for_frozen_contract_and_evidence(tmp_path: Path, attack: str) -> None:
    spec = _trial_spec()
    direction = _direction()
    variant = _variant(direction)
    attempt = _attempt(spec, direction, variant)
    raw, manifest = _artifacts(tmp_path, attempt, spec)
    observations = _observations(spec, attempt, raw, {"d1": {"baseline": 0.5, "candidate": 0.7}})
    trial = classify_trial_result(attempt=attempt, trial_spec=spec, observations=observations, raw_artifacts=raw, evidence_manifest=manifest)
    write_json(tmp_path / "plan" / "trial_spec.json", spec)
    if attack == "projection_drift":
        changed = deepcopy(spec)
        changed["protocol"]["protocol_id"] = "forged"
        write_json(tmp_path / "plan" / "trial_spec.json", changed)
    elif attack == "artifact_hash":
        (tmp_path / "experiment/raw/main_results.json").write_text("{}", encoding="utf-8")
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

