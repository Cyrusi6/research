"""Strict authoritative S1-S3 domain contracts and canonical identities."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

DIRECTION_SCHEMA_VERSION = "auto_research_direction_v3"
VARIANT_SCHEMA_VERSION = "auto_research_variant_v4"
ATTEMPT_SCHEMA_VERSION = "auto_research_attempt_v3"
TRIAL_SPEC_SCHEMA_VERSION = "auto_research_trial_spec_v2"
TRIAL_RESULT_SCHEMA_VERSION = "auto_research_trial_result_v3"
ROUTE_OUTCOME_SCHEMA_VERSION = "auto_research_route_outcome_v3"
EXECUTION_OBSERVATION_SCHEMA_VERSION = "auto_research_execution_observation_v2"
CONSTRAINT_RESULT_SCHEMA_VERSION = "auto_research_constraint_result_v1"
EVIDENCE_MANIFEST_SCHEMA_VERSION = "auto_research_evidence_manifest_v1"
DIRECTION_AGGREGATE_SCHEMA_VERSION = "auto_research_direction_outcome_aggregate_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _without(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in payload.items() if key not in set(keys)}


def direction_semantic_hash(payload: dict[str, Any]) -> str:
    return canonical_hash(
        {
            "research_question": payload.get("research_question"),
            "mechanism_invariants": payload.get("mechanism_invariants"),
            "falsification_conditions": payload.get("falsification_conditions"),
            "metric_signature": payload.get("metric_signature"),
            "benchmark_contract_hash": payload.get("benchmark_contract_hash"),
            "variant_space": payload.get("variant_space"),
        }
    )


def direction_spec_hash(payload: dict[str, Any]) -> str:
    return canonical_hash(_without(payload, "direction_semantic_hash", "direction_spec_hash"))


def variant_semantic_hash(payload: dict[str, Any]) -> str:
    intervention = payload.get("intervention") if isinstance(payload.get("intervention"), dict) else {}
    return canonical_hash(
        {
            "direction_semantic_hash": payload.get("direction_semantic_hash"),
            "intervention": {
                "algorithm_operations": intervention.get("algorithm_operations"),
                "configuration": intervention.get("configuration"),
            },
            "hypothesis": payload.get("hypothesis"),
            "null_hypothesis": payload.get("null_hypothesis"),
            "alternative_hypothesis": payload.get("alternative_hypothesis"),
            "controlled_variables": payload.get("controlled_variables"),
            "implementation_surface_ids": payload.get("implementation_surface_ids"),
            "expected_metric_signature": payload.get("expected_metric_signature"),
            "falsification_conditions": payload.get("falsification_conditions"),
            "ablation": payload.get("ablation"),
        }
    )


def variant_spec_hash(payload: dict[str, Any]) -> str:
    return canonical_hash(_without(payload, "variant_semantic_hash", "variant_spec_hash"))


def implementation_hash(*, frozen_patch: Any, files: dict[str, str], manifest: dict[str, Any]) -> str:
    return canonical_hash({"frozen_patch": frozen_patch, "files": files, "manifest": manifest})


def attempt_input_hash(
    *,
    implementation_hash_value: str,
    protocol: dict[str, Any],
    sample_manifest: dict[str, Any],
    seeds: list[int],
    runtime_config: dict[str, Any],
    evaluator_hash: str,
    trial_spec: dict[str, Any] | None = None,
) -> str:
    return canonical_hash(
        {
            "implementation_hash": implementation_hash_value,
            "protocol": protocol,
            "sample_manifest": sample_manifest,
            "seeds": seeds,
            "runtime_config": runtime_config,
            "evaluator_hash": evaluator_hash,
            "trial_spec": trial_spec,
        }
    )


def trial_spec_hash(trial_spec: dict[str, Any]) -> str:
    return canonical_hash(trial_spec)


def acceptance_contract_hash(trial_spec: dict[str, Any]) -> str:
    return canonical_hash(
        {
            "metrics": trial_spec.get("metrics"),
            "primary_metric_id": trial_spec.get("primary_metric_id"),
            "acceptance_constraints": trial_spec.get("acceptance_constraints"),
            "required_roles": trial_spec.get("required_roles"),
            "evidence_requirements": trial_spec.get("evidence_requirements"),
        }
    )


def build_direction_spec(payload: dict[str, Any]) -> dict[str, Any]:
    spec = deepcopy(payload)
    spec["schema_version"] = DIRECTION_SCHEMA_VERSION
    spec.setdefault(
        "exploration_policy",
        {
            "target_outcome_bearing_variants": 5,
            "execution_width": 1,
            "stop_on_success": False,
            "budget_consumption_rule": "consume_only_verified_method_evaluable_outcomes",
            "non_consuming_outcomes": [
                "planner_rejected",
                "implementation_failed",
                "activation_failed",
                "resource_paused",
                "oom_retry",
                "integrity_blocked",
                "bootstrap_proxy",
            ],
        },
    )
    spec["direction_semantic_hash"] = direction_semantic_hash(spec)
    spec["direction_spec_hash"] = direction_spec_hash(spec)
    validate_contract(spec, "direction_v3.schema.json")
    return spec


def build_variant_spec(direction: dict[str, Any], payload: dict[str, Any], *, tried_variants: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    validate_direction_identity(direction)
    spec = deepcopy(payload)
    spec["schema_version"] = VARIANT_SCHEMA_VERSION
    spec["direction_id"] = direction["direction_id"]
    spec["direction_semantic_hash"] = direction["direction_semantic_hash"]
    spec["direction_spec_hash"] = direction["direction_spec_hash"]
    lineage = deepcopy(spec.get("lineage") or {})
    lineage["direction_spec_hash"] = direction["direction_spec_hash"]
    spec["lineage"] = lineage
    _validate_variant_axes(direction, spec)
    spec["variant_semantic_hash"] = variant_semantic_hash(spec)
    spec["variant_spec_hash"] = variant_spec_hash(spec)
    for tried in tried_variants or []:
        if tried.get("method_evaluable") and tried.get("consumes_direction_budget", True) and tried.get("variant_semantic_hash") == spec["variant_semantic_hash"]:
            raise ValueError("variant duplicates a method-evaluable scientific method in the current direction")
    validate_contract(spec, "variant_v4.schema.json")
    return spec


def validate_direction_identity(spec: dict[str, Any]) -> None:
    validate_contract(spec, "direction_v3.schema.json")
    semantic = direction_semantic_hash(spec)
    complete = direction_spec_hash(spec)
    if spec.get("direction_semantic_hash") != semantic:
        raise ValueError(f"direction_semantic_hash mismatch: expected {semantic}")
    if spec.get("direction_spec_hash") != complete:
        raise ValueError(f"direction_spec_hash mismatch: expected {complete}")


def validate_variant_identity(direction: dict[str, Any], spec: dict[str, Any], *, tried_variants: list[dict[str, Any]] | None = None) -> None:
    validate_direction_identity(direction)
    validate_contract(spec, "variant_v4.schema.json")
    for key in ["direction_id", "direction_semantic_hash", "direction_spec_hash"]:
        if spec.get(key) != direction.get(key):
            raise ValueError(f"variant {key} does not match current DirectionSpec")
    if (spec.get("lineage") or {}).get("direction_spec_hash") != direction["direction_spec_hash"]:
        raise ValueError("VariantSpec.lineage.direction_spec_hash mismatch")
    _validate_variant_axes(direction, spec)
    semantic = variant_semantic_hash(spec)
    complete = variant_spec_hash(spec)
    if spec.get("variant_semantic_hash") != semantic:
        raise ValueError(f"variant_semantic_hash mismatch: expected {semantic}")
    if spec.get("variant_spec_hash") != complete:
        raise ValueError(f"variant_spec_hash mismatch: expected {complete}")
    for tried in tried_variants or []:
        if tried.get("method_evaluable") and tried.get("consumes_direction_budget", True) and tried.get("variant_semantic_hash") == semantic:
            raise ValueError("variant duplicates a method-evaluable scientific method in the current direction")


def validate_trial_spec(trial_spec: dict[str, Any]) -> None:
    validate_contract(trial_spec, "trial_spec_v2.schema.json")
    datasets = {item["dataset_id"] for item in trial_spec["datasets"]}
    manifest_datasets = set(trial_spec["sample_manifest"]["datasets"])
    if datasets != manifest_datasets:
        raise ValueError("TrialSpec datasets must exactly match sample_manifest.datasets")
    metrics = {item["metric_id"]: item for item in trial_spec["metrics"]}
    if trial_spec["primary_metric_id"] not in metrics:
        raise ValueError("TrialSpec primary_metric_id is not registered")
    if sum(item["metric_id"] == trial_spec["primary_metric_id"] for item in trial_spec["metrics"]) != 1:
        raise ValueError("TrialSpec must register the primary metric exactly once")
    constraint_ids = [item["constraint_id"] for item in trial_spec["acceptance_constraints"]]
    if len(set(constraint_ids)) != len(constraint_ids):
        raise ValueError("TrialSpec constraint_id values must be unique")
    evidence_ids = [item["requirement_id"] for item in trial_spec["evidence_requirements"]]
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("TrialSpec evidence requirement IDs must be unique")
    for constraint in trial_spec["acceptance_constraints"]:
        metric_id = constraint.get("metric_id")
        if metric_id is not None and metric_id not in metrics:
            raise ValueError(f"constraint references unknown metric: {metric_id}")
    if not any(
        item["kind"] == "minimum_mean_delta"
        and item.get("metric_id") == trial_spec["primary_metric_id"]
        and item["hard"]
        for item in trial_spec["acceptance_constraints"]
    ):
        raise ValueError("TrialSpec requires a hard primary minimum_mean_delta constraint")


def validate_execution_observation(observation: dict[str, Any]) -> None:
    validate_contract(observation, "execution_observation_v2.schema.json")
    value = observation.get("metric_value")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("metric_value must be finite")


def validate_evidence_manifest(manifest: dict[str, Any], *, trial_spec: dict[str, Any]) -> None:
    validate_contract(manifest, "evidence_manifest_v1.schema.json")
    expected_hash = trial_spec_hash(trial_spec)
    if manifest["trial_spec_hash"] != expected_hash:
        raise ValueError("evidence manifest TrialSpec hash mismatch")
    identities = [item["evidence_id"] for item in manifest["entries"]]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate evidence manifest identity")


def validate_trial_evidence(
    result: dict[str, Any],
    *,
    attempt: dict[str, Any] | None = None,
    trial_spec: dict[str, Any] | None = None,
) -> None:
    validate_contract(result, "trial_result_v3.schema.json")
    observations = result["observations"]
    for observation in observations:
        validate_execution_observation(observation)
    identities = [_observation_identity(item) for item in observations]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate execution observation identity")
    if any(item["command_status"] != "completed" for item in observations):
        raise ValueError("method-evaluable observations must all have completed command status")
    for observation in observations:
        artifact_path = observation["raw_artifact_path"]
        if result["raw_artifacts"].get(artifact_path) != observation["raw_artifact_hash"]:
            raise ValueError("observation must bind the exact registered result artifact")
    if set(result["required_datasets"]) != set(result["observed_datasets"]):
        raise ValueError("required and observed dataset coverage mismatch")
    if result["failure_classification"] is not None:
        raise ValueError("evaluable TrialResult failure_classification must be null")
    if not result["observed_datasets"] or not observations or not result["raw_artifacts"]:
        raise ValueError("evaluable TrialResult requires datasets, observations, and raw artifacts")
    if attempt is not None:
        _validate_trial_against_attempt(result, attempt)
    if trial_spec is not None:
        validate_trial_spec(trial_spec)
        validate_evidence_manifest(result["evidence_manifest"], trial_spec=trial_spec)
        _validate_trial_against_spec(result, trial_spec, attempt=attempt)
        expected_constraints, expected_outcome, expected_summary = classify_acceptance(
            trial_spec=trial_spec,
            observations=observations,
            evidence_manifest=result["evidence_manifest"],
        )
        if canonical_json(result["constraint_results"]) != canonical_json(expected_constraints):
            raise ValueError("TrialResult constraint_results do not match deterministic classifier")
        if result["outcome_classification"] != expected_outcome:
            raise ValueError("TrialResult outcome_classification does not match deterministic classifier")
        if canonical_json(result["primary_metric_summary"]) != canonical_json(expected_summary):
            raise ValueError("TrialResult primary_metric_summary does not match deterministic classifier")


def classify_trial_result(
    *,
    attempt: dict[str, Any],
    trial_spec: dict[str, Any],
    observations: list[dict[str, Any]],
    raw_artifacts: dict[str, str],
    evidence_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_trial_spec(trial_spec)
    for observation in observations:
        validate_execution_observation(observation)
    if len({_observation_identity(item) for item in observations}) != len(observations):
        raise ValueError("duplicate execution observation identity")
    if any(item["command_status"] != "completed" for item in observations):
        raise ValueError("method-evaluable observations must all have completed command status")
    for observation in observations:
        if raw_artifacts.get(observation["raw_artifact_path"]) != observation["raw_artifact_hash"]:
            raise ValueError("observation must bind the exact registered result artifact")
    manifest = evidence_manifest or {
        "schema_version": EVIDENCE_MANIFEST_SCHEMA_VERSION,
        "trial_spec_hash": trial_spec_hash(trial_spec),
        "attempt_id": attempt["attempt_id"],
        "entries": [],
    }
    validate_evidence_manifest(manifest, trial_spec=trial_spec)
    required_datasets = sorted(item["dataset_id"] for item in trial_spec["datasets"])
    required_phases = trial_spec["protocol"]["required_phases"]
    terminal_phases = set(trial_spec["protocol"]["terminal_phases"])
    candidate = [item for item in observations if item["role"] == "candidate"]
    observed_datasets = sorted({item["dataset_id"] for item in candidate})
    if set(observed_datasets) != set(required_datasets):
        raise ValueError("execution observations do not satisfy dataset coverage")
    observed_candidate_phases = {item["phase"] for item in candidate}
    if not observed_candidate_phases or not observed_candidate_phases.issubset(terminal_phases):
        raise ValueError("candidate observations must bind a preregistered terminal phase")
    if any(item["phase"] not in terminal_phases | {"ablation"} for item in observations):
        raise ValueError("execution observation phase is not terminally permitted")
    if any(
        item["sample_manifest_hash"] != attempt["sample_manifest_hash"]
        or item["evaluator_hash"] != attempt["evaluator_hash"]
        for item in observations
    ):
        raise ValueError("execution observation identity hash mismatch")
    _validate_required_coverage(trial_spec, observations, manifest)
    constraint_results, outcome, primary_summary = classify_acceptance(
        trial_spec=trial_spec,
        observations=observations,
        evidence_manifest=manifest,
    )
    completeness = "proxy" if set(required_phases) == {"proxy"} else "full"
    result = {
        "schema_version": TRIAL_RESULT_SCHEMA_VERSION,
        "direction_id": attempt["direction_id"],
        "direction_semantic_hash": attempt["direction_semantic_hash"],
        "direction_spec_hash": attempt["direction_spec_hash"],
        "variant_id": attempt["variant_id"],
        "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "attempt_id": attempt["attempt_id"],
        "trial_spec_hash": trial_spec_hash(trial_spec),
        "acceptance_contract_hash": acceptance_contract_hash(trial_spec),
        "protocol_hash": attempt["protocol_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "completeness": completeness,
        "required_datasets": required_datasets,
        "observed_datasets": observed_datasets,
        "raw_artifacts": deepcopy(raw_artifacts),
        "evidence_manifest": deepcopy(manifest),
        "observations": deepcopy(observations),
        "constraint_results": constraint_results,
        "all_hard_constraints_passed": all(item["status"] == "PASS" for item in constraint_results if item["hard"]),
        "method_evaluable": True,
        "outcome_classification": outcome,
        "failure_classification": None,
        "primary_metric_summary": primary_summary,
    }
    validate_trial_evidence(result, attempt=attempt, trial_spec=trial_spec)
    return result


def validate_trial_result(
    result: dict[str, Any],
    *,
    attempt: dict[str, Any] | None = None,
    trial_spec: dict[str, Any] | None = None,
) -> None:
    validate_trial_evidence(result, attempt=attempt, trial_spec=trial_spec)


def validate_contract(payload: Any, schema_name: str) -> None:
    schema = json.loads((_schema_dir() / schema_name).read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda error: list(error.absolute_path))
    if errors:
        messages = []
        for error in errors[:20]:
            location = ".".join(str(item) for item in error.absolute_path) or "$"
            messages.append(f"{location}: {error.message}")
        raise ValueError("; ".join(messages))


def contract_errors(payload: Any, schema_name: str) -> list[str]:
    try:
        validate_contract(payload, schema_name)
    except (ValueError, OSError) as exc:
        return str(exc).split("; ")
    return []


def _validate_variant_axes(direction: dict[str, Any], variant: dict[str, Any]) -> None:
    space = direction.get("variant_space") or {}
    mutable = set(space.get("mutable_axes") or [])
    immutable = set(space.get("immutable_axes") or [])
    coordinates = variant.get("variation_coordinates") or {}
    _reject_pseudo_semantic_coordinates(coordinates)
    changed = set(coordinates)
    disallowed = changed - mutable
    if disallowed:
        raise ValueError(f"variant changes axes outside mutable_axes: {sorted(disallowed)}")
    if changed & immutable:
        raise ValueError(f"variant changes immutable axes: {sorted(changed & immutable)}")
    for combination in space.get("forbidden_combinations") or []:
        if isinstance(combination, dict) and all(coordinates.get(key) == value for key, value in combination.items()):
            raise ValueError(f"variant matches forbidden combination: {combination}")


def _reject_pseudo_semantic_coordinates(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = {"id", "variant_id", "variant_nonce", "nonce", "ordinal", "iteration", "run_id"}
        matches = sorted(key for key in value if str(key).lower() in forbidden)
        if matches:
            raise ValueError(f"variation_coordinates contain non-scientific identity metadata: {matches}")
        for nested in value.values():
            _reject_pseudo_semantic_coordinates(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_pseudo_semantic_coordinates(nested)


def _observation_identity(observation: dict[str, Any]) -> tuple[Any, ...]:
    return (
        observation["phase"],
        observation["role"],
        observation["dataset_id"],
        observation["metric_id"],
        observation["seed"],
    )


def _validate_trial_against_attempt(result: dict[str, Any], attempt: dict[str, Any]) -> None:
    for key in (
        "direction_id",
        "direction_semantic_hash",
        "direction_spec_hash",
        "variant_id",
        "variant_semantic_hash",
        "variant_spec_hash",
        "attempt_id",
        "trial_spec_hash",
        "acceptance_contract_hash",
        "protocol_hash",
        "attempt_input_hash",
    ):
        if result.get(key) != attempt.get(key):
            raise ValueError(f"TrialResult {key} does not match Attempt")
    seeds = set(attempt.get("seeds") or [])
    if not seeds:
        raise ValueError("Attempt must preregister at least one seed")
    for observation in result["observations"]:
        if observation["sample_manifest_hash"] != attempt.get("sample_manifest_hash"):
            raise ValueError("observation sample_manifest_hash does not match Attempt")
        if observation["evaluator_hash"] != attempt.get("evaluator_hash"):
            raise ValueError("observation evaluator_hash does not match Attempt")
        if observation["seed"] not in seeds:
            raise ValueError("observation seed is not preregistered by Attempt")


def _validate_trial_against_spec(
    result: dict[str, Any],
    trial_spec: dict[str, Any],
    *,
    attempt: dict[str, Any] | None,
) -> None:
    required_datasets = {item["dataset_id"] for item in trial_spec["datasets"]}
    if required_datasets != set(result["required_datasets"]):
        raise ValueError("TrialResult dataset coverage does not match TrialSpec")
    if result["trial_spec_hash"] != trial_spec_hash(trial_spec):
        raise ValueError("TrialResult trial_spec_hash mismatch")
    if result["acceptance_contract_hash"] != acceptance_contract_hash(trial_spec):
        raise ValueError("TrialResult acceptance_contract_hash mismatch")
    required_phases = set(trial_spec["protocol"]["required_phases"])
    expected_completeness = "proxy" if required_phases == {"proxy"} else "full"
    if result["completeness"] != expected_completeness:
        raise ValueError("TrialResult completeness does not match TrialSpec required phases")
    _validate_required_coverage(trial_spec, result["observations"], result["evidence_manifest"])


def _dataset_ids(items: list[Any]) -> list[str]:
    result = []
    for item in items:
        value = item.get("name") if isinstance(item, dict) else item
        if value:
            result.append(str(value))
    if len(set(result)) != len(result):
        raise ValueError("required datasets must be unique")
    return sorted(result)


def _validate_required_coverage(
    trial_spec: dict[str, Any],
    observations: list[dict[str, Any]],
    evidence_manifest: dict[str, Any],
) -> None:
    datasets = {item["dataset_id"] for item in trial_spec["datasets"]}
    seeds = set(trial_spec["statistical_testing"]["seeds"])
    primary_metric = trial_spec["primary_metric_id"]
    require_complete_seed_coverage = trial_spec["statistical_testing"]["require_complete_seed_coverage"]
    for role in trial_spec["required_roles"]:
        covered = {
            (item["dataset_id"], item["seed"])
            for item in observations
            if item["role"] == role and item["metric_id"] == primary_metric
        }
        expected = {(dataset_id, seed) for dataset_id in datasets for seed in seeds}
        covered_datasets = {dataset_id for dataset_id, _seed in covered}
        if (require_complete_seed_coverage and covered != expected) or (
            not require_complete_seed_coverage and covered_datasets != datasets
        ):
            raise ValueError(f"required {role} dataset/seed coverage is missing")
    applicable_phases = set(trial_spec["protocol"]["required_phases"])
    manifest_kinds = {item["kind"] for item in evidence_manifest["entries"]}
    missing_artifacts = set(trial_spec["required_artifacts"]) - manifest_kinds
    if missing_artifacts:
        raise ValueError(f"required result artifacts are missing: {sorted(missing_artifacts)}")
    for requirement in trial_spec["evidence_requirements"]:
        applies = "always" in requirement["applicable_phases"] or bool(
            applicable_phases & set(requirement["applicable_phases"])
        )
        if applies and requirement["required"] and requirement["kind"] not in manifest_kinds:
            raise ValueError(f"required evidence is missing: {requirement['kind']}")


def classify_acceptance(
    *,
    trial_spec: dict[str, Any],
    observations: list[dict[str, Any]],
    evidence_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    validate_trial_spec(trial_spec)
    validate_evidence_manifest(evidence_manifest, trial_spec=trial_spec)
    metrics = {item["metric_id"]: item for item in trial_spec["metrics"]}
    primary_metric_id = trial_spec["primary_metric_id"]
    primary = metrics[primary_metric_id]
    primary_summary = _metric_summary(observations, primary_metric_id, primary["objective"])
    results: list[dict[str, Any]] = []
    for constraint in trial_spec["acceptance_constraints"]:
        status, observed, bindings = _evaluate_constraint(constraint, observations, evidence_manifest, metrics)
        result = {
            "schema_version": CONSTRAINT_RESULT_SCHEMA_VERSION,
            "constraint_id": constraint["constraint_id"],
            "kind": constraint["kind"],
            "hard": constraint["hard"],
            "status": status,
            "observed": observed,
            "threshold": constraint.get("threshold"),
            "objective": constraint.get("objective"),
            "observation_ids": sorted(bindings[0]),
            "evidence_ids": sorted(bindings[1]),
        }
        validate_contract(result, "constraint_result_v1.schema.json")
        results.append(result)
    missing_hard = [item["constraint_id"] for item in results if item["hard"] and item["status"] == "MISSING"]
    if missing_hard:
        raise ValueError(f"required hard constraint evidence is missing: {missing_hard}")
    primary_constraints = [
        item
        for item in results
        if item["kind"] == "minimum_mean_delta"
        and next(spec for spec in trial_spec["acceptance_constraints"] if spec["constraint_id"] == item["constraint_id"])["metric_id"] == primary_metric_id
    ]
    accepted = bool(primary_constraints) and all(item["status"] == "PASS" for item in primary_constraints)
    accepted = accepted and all(item["status"] == "PASS" for item in results if item["hard"])
    return results, "accepted" if accepted else "rejected", primary_summary


def _evaluate_constraint(
    constraint: dict[str, Any],
    observations: list[dict[str, Any]],
    evidence_manifest: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
) -> tuple[str, Any, tuple[set[str], set[str]]]:
    kind = constraint["kind"]
    metric_id = constraint.get("metric_id")
    objective = constraint.get("objective") or (metrics.get(metric_id) or {}).get("objective")
    threshold = constraint.get("threshold")
    selected = [item for item in observations if metric_id is None or item["metric_id"] == metric_id]
    observation_ids = {item["observation_id"] for item in selected}
    evidence_ids: set[str] = set()
    if kind == "minimum_mean_delta":
        summary = _metric_summary(observations, metric_id, objective)
        value = summary.get("delta")
        return _threshold_status(value, threshold, "minimum"), value, (observation_ids, evidence_ids)
    if kind == "per_dataset_maximum_regression":
        deltas = _dataset_deltas(observations, metric_id, objective)
        if not deltas:
            return "MISSING", None, (observation_ids, evidence_ids)
        worst = max(max(0.0, -value) for value in deltas.values())
        return _threshold_status(worst, threshold, "maximum"), {"worst_regression": worst, "deltas": deltas}, (observation_ids, evidence_ids)
    if kind == "required_ablation_contrast":
        value = _role_delta(observations, metric_id, "candidate", "ablation", objective)
        return _threshold_status(value, threshold, "minimum"), value, (observation_ids, evidence_ids)
    if kind in {"matched_control_constraint", "coverage_constraint"}:
        role = "matched_control" if kind == "matched_control_constraint" else "coverage"
        if kind == "coverage_constraint":
            values = [float(item["metric_value"]) for item in observations if item["metric_id"] == metric_id and item["role"] == role]
            value = sum(values) / len(values) if values else None
        else:
            value = _role_delta(observations, metric_id, "candidate", role, objective)
        if threshold is None:
            return ("PASS" if value is not None else "MISSING"), value, (observation_ids, evidence_ids)
        return _threshold_status(value, threshold, "minimum"), value, (observation_ids, evidence_ids)
    if kind == "required_artifact_evidence":
        required_kind = constraint["evidence_kind"]
        matching = {item["evidence_id"] for item in evidence_manifest["entries"] if item["kind"] == required_kind}
        return ("PASS" if matching else "MISSING"), {"evidence_kind": required_kind, "count": len(matching)}, (set(), matching)
    if kind in {"primary_metric_threshold", "secondary_metric_threshold"}:
        values = [float(item["metric_value"]) for item in selected if item["role"] == "candidate"]
        value = sum(values) / len(values) if values else None
        mode = "minimum" if objective == "maximize" else "maximum"
        return _threshold_status(value, threshold, mode), value, (observation_ids, evidence_ids)
    raise ValueError(f"unsupported acceptance constraint kind: {kind}")


def _metric_summary(observations: list[dict[str, Any]], metric_id: str, objective: str) -> dict[str, Any]:
    candidates = [float(item["metric_value"]) for item in observations if item["role"] == "candidate" and item["metric_id"] == metric_id]
    baselines = [float(item["metric_value"]) for item in observations if item["role"] == "baseline" and item["metric_id"] == metric_id]
    candidate_mean = sum(candidates) / len(candidates) if candidates else None
    baseline_mean = sum(baselines) / len(baselines) if baselines else None
    delta = None
    if candidate_mean is not None and baseline_mean is not None:
        delta = candidate_mean - baseline_mean if objective == "maximize" else baseline_mean - candidate_mean
    return {
        "metric_id": metric_id,
        "objective": objective,
        "aggregation": "mean",
        "candidate_mean": candidate_mean,
        "baseline_mean": baseline_mean,
        "delta": delta,
    }


def _dataset_deltas(observations: list[dict[str, Any]], metric_id: str, objective: str) -> dict[str, float]:
    datasets = sorted({item["dataset_id"] for item in observations if item["metric_id"] == metric_id})
    result: dict[str, float] = {}
    for dataset_id in datasets:
        candidate = [float(item["metric_value"]) for item in observations if item["dataset_id"] == dataset_id and item["metric_id"] == metric_id and item["role"] == "candidate"]
        baseline = [float(item["metric_value"]) for item in observations if item["dataset_id"] == dataset_id and item["metric_id"] == metric_id and item["role"] == "baseline"]
        if candidate and baseline:
            raw = sum(candidate) / len(candidate) - sum(baseline) / len(baseline)
            result[dataset_id] = raw if objective == "maximize" else -raw
    return result


def _role_delta(observations: list[dict[str, Any]], metric_id: str, left_role: str, right_role: str, objective: str) -> float | None:
    left = [float(item["metric_value"]) for item in observations if item["metric_id"] == metric_id and item["role"] == left_role]
    right = [float(item["metric_value"]) for item in observations if item["metric_id"] == metric_id and item["role"] == right_role]
    if not left or not right:
        return None
    raw = sum(left) / len(left) - sum(right) / len(right)
    return raw if objective == "maximize" else -raw


def _threshold_status(value: float | None, threshold: float | None, mode: str) -> str:
    if value is None or threshold is None:
        return "MISSING"
    passed = value >= threshold if mode == "minimum" else value <= threshold
    return "PASS" if passed else "FAIL"


def _schema_dir() -> Path:
    return Path(__file__).resolve().parent / "schemas"
