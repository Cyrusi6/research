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
ATTEMPT_SCHEMA_VERSION = "auto_research_attempt_v2"
TRIAL_RESULT_SCHEMA_VERSION = "auto_research_trial_result_v2"
ROUTE_OUTCOME_SCHEMA_VERSION = "auto_research_route_outcome_v2"
EXECUTION_OBSERVATION_SCHEMA_VERSION = "auto_research_execution_observation_v1"
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
) -> str:
    return canonical_hash(
        {
            "implementation_hash": implementation_hash_value,
            "protocol": protocol,
            "sample_manifest": sample_manifest,
            "seeds": seeds,
            "runtime_config": runtime_config,
            "evaluator_hash": evaluator_hash,
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


def validate_execution_observation(observation: dict[str, Any]) -> None:
    validate_contract(observation, "execution_observation_v1.schema.json")
    value = observation.get("metric_value")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("metric_value must be finite")


def validate_trial_evidence(
    result: dict[str, Any],
    *,
    attempt: dict[str, Any] | None = None,
    trial_spec: dict[str, Any] | None = None,
) -> None:
    """Pure validation for a method-evaluable TrialResult and its evidence."""
    validate_contract(result, "trial_result_v2.schema.json")
    observations = result["observations"]
    for observation in observations:
        validate_execution_observation(observation)

    identities = [_observation_identity(item) for item in observations]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate execution observation identity")
    if any(item["command_status"] != "completed" for item in observations):
        raise ValueError("method-evaluable observations must all have completed command status")

    registered_hashes = set(result["raw_artifacts"].values())
    if any(item["raw_artifact_hash"] not in registered_hashes for item in observations):
        raise ValueError("observation raw_artifact_hash is not registered")

    required_datasets = set(result["required_datasets"])
    observed_datasets = set(result["observed_datasets"])
    observation_datasets = {item["dataset_id"] for item in observations}
    candidate_datasets = {item["dataset_id"] for item in observations if item["role"] == "candidate"}
    baseline_datasets = {item["dataset_id"] for item in observations if item["role"] == "baseline"}
    if required_datasets != observed_datasets or required_datasets != observation_datasets:
        raise ValueError("required, observed, and observation dataset coverage mismatch")
    if candidate_datasets != required_datasets or baseline_datasets != required_datasets:
        raise ValueError("baseline and candidate role coverage must match required datasets")

    expected_phase = result["completeness"]
    if any(item["phase"] != expected_phase for item in observations if item["role"] != "ablation"):
        raise ValueError("observation phase does not match TrialResult completeness")
    if any(item["role"] == "ablation" and item["phase"] != "ablation" for item in observations):
        raise ValueError("ablation observations must use the ablation phase")

    if result["failure_classification"] is not None:
        raise ValueError("evaluable TrialResult failure_classification must be null")
    if not result["observed_datasets"] or not observations or not result["raw_artifacts"]:
        raise ValueError("evaluable TrialResult requires datasets, observations, and raw artifacts")

    if attempt is not None:
        _validate_trial_against_attempt(result, attempt)
    if trial_spec is not None:
        _validate_trial_against_spec(result, trial_spec, attempt=attempt)


def classify_trial_result(
    *,
    attempt: dict[str, Any],
    trial_spec: dict[str, Any],
    observations: list[dict[str, Any]],
    raw_artifacts: dict[str, str],
) -> dict[str, Any]:
    for observation in observations:
        validate_execution_observation(observation)
    if len({_observation_identity(item) for item in observations}) != len(observations):
        raise ValueError("duplicate execution observation identity")
    if any(item["command_status"] != "completed" for item in observations):
        raise ValueError("method-evaluable observations must all have completed command status")
    if any(item["raw_artifact_hash"] not in set(raw_artifacts.values()) for item in observations):
        raise ValueError("observation raw_artifact_hash is not registered")
    sample_manifest = trial_spec.get("sample_manifest") if isinstance(trial_spec.get("sample_manifest"), dict) else {}
    required_datasets = _dataset_ids(sample_manifest.get("datasets") or trial_spec.get("datasets") or [])
    required_phases = list(((trial_spec.get("protocol") or {}).get("required_phases") or []))
    if not required_phases:
        required_phases = ["proxy"] if attempt["attempt_kind"] in {"proxy", "bootstrap_proxy"} else ["full"]
    allowed_terminal = set(((trial_spec.get("protocol") or {}).get("terminal_method_phases") or required_phases))
    actual_phases = {item["phase"] for item in observations if item["role"] == "candidate"}
    candidate = [item for item in observations if item["role"] == "candidate"]
    observed_datasets = sorted({item["dataset_id"] for item in candidate})
    coverage_ok = bool(required_datasets) and set(required_datasets) == set(observed_datasets)
    phases_ok = set(required_phases).issubset(actual_phases) and actual_phases.issubset(allowed_terminal)
    hashes_ok = all(
        item["sample_manifest_hash"] == attempt["sample_manifest_hash"]
        and item["evaluator_hash"] == attempt["evaluator_hash"]
        for item in observations
    )
    roles_ok = all(
        {item["dataset_id"] for item in observations if item["role"] == role} == set(required_datasets)
        for role in ("baseline", "candidate")
    )
    method_evaluable = coverage_ok and phases_ok and hashes_ok and bool(candidate) and roles_ok and bool(raw_artifacts)
    if not method_evaluable:
        raise ValueError("execution observations do not satisfy method-evaluable protocol coverage")
    outcome, primary_summary = _classify_outcome(trial_spec, observations)
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
        "protocol_hash": attempt["protocol_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "completeness": completeness,
        "required_datasets": required_datasets,
        "observed_datasets": observed_datasets,
        "raw_artifacts": deepcopy(raw_artifacts),
        "observations": deepcopy(observations),
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
    sample_manifest = trial_spec.get("sample_manifest") if isinstance(trial_spec.get("sample_manifest"), dict) else {}
    required_datasets = set(_dataset_ids(sample_manifest.get("datasets") or trial_spec.get("datasets") or []))
    if required_datasets != set(result["required_datasets"]):
        raise ValueError("TrialResult dataset coverage does not match TrialSpec")

    protocol = trial_spec.get("protocol") if isinstance(trial_spec.get("protocol"), dict) else {}
    required_phases = set(protocol.get("required_phases") or [])
    if not required_phases and attempt is not None:
        required_phases = {"proxy"} if attempt.get("attempt_kind") in {"proxy", "bootstrap_proxy"} else {"full"}
    expected_completeness = "proxy" if required_phases == {"proxy"} else "full"
    if result["completeness"] != expected_completeness:
        raise ValueError("TrialResult completeness does not match TrialSpec required phases")

    required_roles = set(protocol.get("required_roles") or ["baseline", "candidate"])
    actual_roles = {item["role"] for item in result["observations"]}
    if not required_roles.issubset(actual_roles):
        raise ValueError("TrialResult does not satisfy TrialSpec role coverage")

    primary = next((item for item in trial_spec.get("metrics") or [] if isinstance(item, dict) and item.get("primary")), None)
    primary_metric = str((primary or {}).get("name") or (primary or {}).get("metric_id") or "primary_metric")
    for role in required_roles & {"baseline", "candidate"}:
        covered = {
            item["dataset_id"]
            for item in result["observations"]
            if item["role"] == role and item["metric_id"] == primary_metric
        }
        if covered != required_datasets:
            raise ValueError(f"{role} primary metric coverage does not match TrialSpec datasets")

    require_seed_coverage = bool(protocol.get("require_seed_coverage") or protocol.get("required_seed_coverage"))
    if require_seed_coverage:
        registered_seeds = set((attempt or {}).get("seeds") or ((trial_spec.get("statistical_testing") or {}).get("seeds") or []))
        for role in required_roles & {"baseline", "candidate"}:
            observed_seeds = {item["seed"] for item in result["observations"] if item["role"] == role}
            if observed_seeds != registered_seeds:
                raise ValueError(f"{role} seed coverage does not match preregistered seeds")

    expected_outcome, expected_summary = _classify_outcome(trial_spec, result["observations"])
    if result["outcome_classification"] != expected_outcome:
        raise ValueError("TrialResult outcome_classification does not match deterministic classifier")
    if canonical_json(result["primary_metric_summary"]) != canonical_json(expected_summary):
        raise ValueError("TrialResult primary_metric_summary does not match deterministic classifier")


def _dataset_ids(items: list[Any]) -> list[str]:
    result = []
    for item in items:
        value = item.get("name") if isinstance(item, dict) else item
        if value:
            result.append(str(value))
    if len(set(result)) != len(result):
        raise ValueError("required datasets must be unique")
    return sorted(result)


def _classify_outcome(trial_spec: dict[str, Any], observations: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    primary = next((item for item in trial_spec.get("metrics") or [] if isinstance(item, dict) and item.get("primary")), None)
    metric_id = str((primary or {}).get("name") or (primary or {}).get("metric_id") or "primary_metric")
    objective = "maximize" if (primary or {}).get("higher_is_better", True) else "minimize"
    candidates = [item for item in observations if item["role"] == "candidate" and item["metric_id"] == metric_id]
    baselines = [item for item in observations if item["role"] == "baseline" and item["metric_id"] == metric_id]
    candidate_mean = sum(float(item["metric_value"]) for item in candidates) / len(candidates) if candidates else None
    baseline_mean = sum(float(item["metric_value"]) for item in baselines) / len(baselines) if baselines else None
    criteria = trial_spec.get("acceptance_criteria") if isinstance(trial_spec.get("acceptance_criteria"), dict) else {}
    minimum_delta = float(criteria.get("minimum_mean_delta") or criteria.get("min_delta_to_pass") or 0.0)
    summary = {"metric_id": metric_id, "objective": objective, "candidate_mean": candidate_mean, "baseline_mean": baseline_mean, "minimum_delta": minimum_delta}
    if candidate_mean is None or baseline_mean is None:
        return "inconclusive", summary
    delta = candidate_mean - baseline_mean if objective == "maximize" else baseline_mean - candidate_mean
    summary["delta"] = delta
    return ("accepted" if delta >= minimum_delta else "rejected"), summary


def _schema_dir() -> Path:
    return Path(__file__).resolve().parent / "schemas"
