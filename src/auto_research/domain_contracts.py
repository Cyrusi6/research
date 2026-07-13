"""Canonical S1-S3 contracts, fingerprints, and strict validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


DIRECTION_SCHEMA_VERSION = "auto_research_direction_v2"
VARIANT_SCHEMA_VERSION = "auto_research_variant_v3"
ATTEMPT_SCHEMA_VERSION = "auto_research_attempt_v1"
TRIAL_RESULT_SCHEMA_VERSION = "auto_research_trial_result_v1"
ROUTE_OUTCOME_SCHEMA_VERSION = "auto_research_route_outcome_v1"

ATTEMPT_STATES = {
    "PLANNED",
    "IMPLEMENTING",
    "IMPLEMENTATION_REPAIR",
    "READY",
    "PROXY_RUNNING",
    "PROXY_COMPLETED",
    "FULL_RUNNING",
    "METHOD_COMPLETED",
    "METHOD_FAILED",
    "RESOURCE_PAUSED",
    "INTEGRITY_BLOCKED",
}

ROUTE_ACTIONS = {
    "IMPLEMENT_VARIANT",
    "REPAIR_IMPLEMENTATION",
    "RUN_METHOD_ATTEMPT",
    "PROPOSE_NEXT_VARIANT",
    "START_NEW_DIRECTION",
    "FINISH_DIRECTION",
    "FINISH_RUN",
    "PAUSE_RESOURCE",
    "BLOCK_INTEGRITY",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def direction_hash(payload: dict[str, Any]) -> str:
    mechanism = payload.get("mechanism_invariants") or {}
    material = {
        "research_question": payload.get("research_question"),
        "mechanism_invariants": mechanism,
        "causal_hypothesis": mechanism.get("causal_hypothesis") if isinstance(mechanism, dict) else None,
        "target_mediator": mechanism.get("target_mediator") if isinstance(mechanism, dict) else None,
        "metric_signature": payload.get("metric_signature"),
        "benchmark_contract_hash": payload.get("benchmark_contract_hash"),
    }
    return canonical_hash(material)


def variant_spec_hash(payload: dict[str, Any]) -> str:
    intervention = payload.get("intervention") or {}
    material = {
        "direction_hash": payload.get("direction_hash"),
        "intervention": intervention,
        "variation_coordinates": payload.get("variation_coordinates"),
        "algorithm_operations": intervention.get("algorithm_operations") if isinstance(intervention, dict) else None,
        "configuration": intervention.get("configuration") if isinstance(intervention, dict) else None,
        "controlled_variables": payload.get("controlled_variables"),
        "ablation": payload.get("ablation"),
        "expected_metric_signature": payload.get("expected_metric_signature"),
    }
    return canonical_hash(material)


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
    spec = dict(payload)
    spec["schema_version"] = DIRECTION_SCHEMA_VERSION
    spec.setdefault(
        "exploration_policy",
        {
            "target_outcome_bearing_variants": 5,
            "execution_width": 1,
            "stop_on_success": False,
            "budget_consumption_rule": "consume_only_terminal_method_evaluable_outcomes",
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
    spec["direction_hash"] = direction_hash(spec)
    validate_contract(spec, "direction_v2.schema.json")
    return spec


def build_variant_spec(direction: dict[str, Any], payload: dict[str, Any], *, tried_variants: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    spec = dict(payload)
    spec["schema_version"] = VARIANT_SCHEMA_VERSION
    spec["direction_id"] = direction["direction_id"]
    spec["direction_hash"] = direction["direction_hash"]
    _validate_variant_axes(direction, spec)
    spec["variant_spec_hash"] = variant_spec_hash(spec)
    for tried in tried_variants or []:
        if tried.get("method_evaluable") and tried.get("variant_spec_hash") == spec["variant_spec_hash"]:
            raise ValueError("variant duplicates a method-evaluable outcome in the current direction")
    validate_contract(spec, "variant_v3.schema.json")
    return spec


def validate_direction_identity(spec: dict[str, Any]) -> None:
    validate_contract(spec, "direction_v2.schema.json")
    expected = direction_hash(spec)
    if spec.get("direction_hash") != expected:
        raise ValueError(f"direction_hash mismatch: expected {expected}")


def validate_variant_identity(direction: dict[str, Any], spec: dict[str, Any], *, tried_variants: list[dict[str, Any]] | None = None) -> None:
    validate_direction_identity(direction)
    validate_contract(spec, "variant_v3.schema.json")
    if spec.get("direction_id") != direction.get("direction_id") or spec.get("direction_hash") != direction.get("direction_hash"):
        raise ValueError("variant direction identity does not match current DirectionSpec")
    _validate_variant_axes(direction, spec)
    expected = variant_spec_hash(spec)
    if spec.get("variant_spec_hash") != expected:
        raise ValueError(f"variant_spec_hash mismatch: expected {expected}")
    for tried in tried_variants or []:
        if tried.get("method_evaluable") and tried.get("variant_spec_hash") == expected:
            raise ValueError("variant duplicates a method-evaluable outcome in the current direction")


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
    except ValueError as exc:
        return str(exc).split("; ")
    return []


def _validate_variant_axes(direction: dict[str, Any], variant: dict[str, Any]) -> None:
    space = direction.get("variant_space") or {}
    mutable = set(space.get("mutable_axes") or [])
    immutable = set(space.get("immutable_axes") or [])
    coordinates = variant.get("variation_coordinates") or {}
    changed = set(coordinates)
    disallowed = changed - mutable
    if disallowed:
        raise ValueError(f"variant changes axes outside mutable_axes: {sorted(disallowed)}")
    if changed & immutable:
        raise ValueError(f"variant changes immutable axes: {sorted(changed & immutable)}")
    invariants = variant.get("mechanism_invariants")
    if invariants is not None and invariants != direction.get("mechanism_invariants"):
        raise ValueError("variant changes mechanism_invariants")
    for combination in space.get("forbidden_combinations") or []:
        if isinstance(combination, dict) and all(coordinates.get(key) == value for key, value in combination.items()):
            raise ValueError(f"variant matches forbidden combination: {combination}")


def _schema_dir() -> Path:
    return Path(__file__).resolve().parent / "schemas"
