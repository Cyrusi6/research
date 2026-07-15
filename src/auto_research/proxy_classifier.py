"""Pure proxy classification from a frozen decision contract and decoded evidence."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from .domain_contracts import canonical_hash


PROXY_DECISION_CONTRACT_SCHEMA_VERSION = "auto_research_proxy_decision_contract_v1"
PROXY_OUTCOME_SCHEMA_VERSION = "auto_research_proxy_outcome_v2"
CONSTRAINT_RESULT_SCHEMA_VERSION = "auto_research_constraint_result_v2"

_PROXY_RESULTS_KIND = "proxy_results"
_ACTIVATION_KIND = "activation_evidence"
_READINESS_KIND = "full_s3_readiness"
_DECISION_CONSTRAINT_KINDS = {
    "minimum_mean_delta",
    "per_dataset_maximum_regression",
}


def build_proxy_decision_contract(
    *,
    attempt: Mapping[str, Any],
    frozen_trial_spec: Mapping[str, Any],
    activation_surface_ids: Sequence[str],
    readiness_check_ids: Sequence[str],
) -> dict[str, Any]:
    """Project the proxy decision inputs once, before any producer evidence exists."""

    phase_contracts = [item for item in frozen_trial_spec.get("phase_contracts", ()) if item.get("phase") == "proxy"]
    if len(phase_contracts) != 1:
        raise ValueError("frozen TrialSpec must contain exactly one proxy phase contract")
    phase_contract = phase_contracts[0]
    primary_metric_id = str(frozen_trial_spec.get("primary_metric_id") or "")
    metrics = {str(item.get("metric_id") or ""): item for item in frozen_trial_spec.get("metrics", ())}
    primary_metric = metrics.get(primary_metric_id)
    if not isinstance(primary_metric, Mapping):
        raise ValueError("frozen TrialSpec primary metric is missing")
    constraints = [
        {
            "constraint_id": item["constraint_id"],
            "kind": item["kind"],
            "hard": item["hard"],
            "metric_id": item["metric_id"],
            "threshold": item["threshold"],
            "objective": item.get("objective") or primary_metric["objective"],
        }
        for item in frozen_trial_spec.get("acceptance_constraints", ())
        if item.get("kind") in _DECISION_CONSTRAINT_KINDS and item.get("metric_id") == primary_metric_id
    ]
    required_kinds = set(phase_contract.get("evidence_kinds", ()))
    required_kinds.update(
        requirement["kind"]
        for requirement in frozen_trial_spec.get("evidence_requirements", ())
        if requirement.get("required")
        and ("always" in requirement.get("applicable_phases", ()) or "proxy" in requirement.get("applicable_phases", ()))
    )
    phase_execution = (attempt.get("phase_executions") or {}).get("proxy") or {}
    contract = {
        "schema_version": PROXY_DECISION_CONTRACT_SCHEMA_VERSION,
        "attempt_id": attempt.get("attempt_id"),
        "lifecycle_generation": attempt.get("lifecycle_generation"),
        "implementation_hash": attempt.get("implementation_hash"),
        "attempt_input_hash": attempt.get("attempt_input_hash"),
        "phase_execution_id": phase_execution.get("phase_execution_id"),
        "phase_start_event_id": phase_execution.get("phase_start_event_id"),
        "primary_metric_id": primary_metric_id,
        "metric_ids": sorted(str(item) for item in phase_contract.get("metrics", ())),
        "objective": primary_metric.get("objective"),
        "datasets": sorted(str(item) for item in phase_contract.get("datasets", ())),
        "seeds": sorted(phase_contract.get("seeds", ())),
        "roles": sorted(str(item) for item in phase_contract.get("roles", ())),
        "evidence_kinds": sorted(required_kinds),
        "activation_surface_ids": sorted(str(item) for item in activation_surface_ids),
        "readiness_check_ids": sorted(str(item) for item in readiness_check_ids),
        "constraints": sorted(constraints, key=lambda item: item["constraint_id"]),
    }
    contract["contract_hash"] = _contract_hash(contract)
    validate_proxy_decision_contract(contract)
    return contract


def validate_proxy_decision_contract(contract: Mapping[str, Any]) -> None:
    """Validate schema and semantic invariants of a frozen proxy contract."""

    _validate_schema(contract, "proxy_decision_contract_v1.schema.json")
    if contract["contract_hash"] != _contract_hash(contract):
        raise ValueError("proxy decision contract hash mismatch")
    if set(contract["roles"]) != {"baseline", "candidate"}:
        raise ValueError("proxy decision contract roles must be exactly baseline and candidate")
    if contract["primary_metric_id"] not in set(contract["metric_ids"]):
        raise ValueError("proxy primary metric is not covered by metric_ids")
    required_kinds = {_PROXY_RESULTS_KIND, _ACTIVATION_KIND, _READINESS_KIND}
    if not required_kinds.issubset(set(contract["evidence_kinds"])):
        raise ValueError("proxy contract must require results, activation, and readiness evidence")
    constraint_kinds = [item["kind"] for item in contract["constraints"]]
    if Counter(constraint_kinds) != Counter(_DECISION_CONSTRAINT_KINDS):
        raise ValueError("proxy contract requires exactly one paired-delta and one dataset-regression constraint")
    constraint_ids = [item["constraint_id"] for item in contract["constraints"]]
    if len(constraint_ids) != len(set(constraint_ids)):
        raise ValueError("proxy constraint IDs must be unique")
    for constraint in contract["constraints"]:
        if constraint["metric_id"] != contract["primary_metric_id"]:
            raise ValueError("proxy decision constraints must bind the primary metric")
        if constraint["objective"] != contract["objective"]:
            raise ValueError("proxy decision constraint objective mismatch")


def classify_proxy_outcome(
    *,
    frozen_contract: Mapping[str, Any],
    decoded_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify proxy evidence without consulting mutable producer policy or state."""

    validate_proxy_decision_contract(frozen_contract)
    evidence = _validate_exact_evidence(frozen_contract, decoded_evidence)
    proxy_payload = evidence[_PROXY_RESULTS_KIND]
    activation_payload = evidence[_ACTIVATION_KIND]
    readiness_payload = evidence[_READINESS_KIND]
    observation_ids, observed_delta, dataset_deltas = _paired_deltas(frozen_contract, proxy_payload)
    worst_regression = max(max(0.0, -value) for value in dataset_deltas.values())
    _validate_activation(frozen_contract, activation_payload)
    _validate_readiness(frozen_contract, readiness_payload)

    proxy_evidence_id = str(proxy_payload["evidence_id"])
    results = []
    for constraint in frozen_contract["constraints"]:
        if constraint["kind"] == "minimum_mean_delta":
            observed: Any = observed_delta
            status = "PASS" if observed_delta >= float(constraint["threshold"]) else "FAIL"
        else:
            observed = {"worst_regression": worst_regression, "deltas": dataset_deltas}
            status = "PASS" if worst_regression <= float(constraint["threshold"]) else "FAIL"
        results.append(
            {
                "schema_version": CONSTRAINT_RESULT_SCHEMA_VERSION,
                "constraint_id": constraint["constraint_id"],
                "kind": constraint["kind"],
                "hard": constraint["hard"],
                "status": status,
                "observed": observed,
                "threshold": constraint["threshold"],
                "objective": constraint["objective"],
                "observation_ids": observation_ids,
                "evidence_ids": [proxy_evidence_id],
            }
        )
    decision = "RUN_FULL" if all(item["status"] == "PASS" for item in results) else "PROPOSE_NEXT_VARIANT"
    reason_codes = [
        "proxy_contract_constraints_pass" if decision == "RUN_FULL" else "proxy_contract_constraints_fail",
        "proxy_exact_coverage_pass",
        "proxy_activation_surfaces_pass",
        "proxy_readiness_checks_pass",
    ]
    outcome = {
        "schema_version": PROXY_OUTCOME_SCHEMA_VERSION,
        "attempt_id": frozen_contract["attempt_id"],
        "lifecycle_generation": frozen_contract["lifecycle_generation"],
        "implementation_hash": frozen_contract["implementation_hash"],
        "attempt_input_hash": frozen_contract["attempt_input_hash"],
        "phase_execution_id": frozen_contract["phase_execution_id"],
        "phase_start_event_id": frozen_contract["phase_start_event_id"],
        "proxy_decision_contract_hash": frozen_contract["contract_hash"],
        "evidence_set_hash": canonical_hash(_json_value(decoded_evidence)),
        "observed_delta": observed_delta,
        "dataset_deltas": dataset_deltas,
        "worst_dataset_regression": worst_regression,
        "constraint_results": results,
        "activation_surface_ids": sorted(activation_payload["implementation_surface_ids"]),
        "readiness_check_ids": sorted(item["check_id"] for item in readiness_payload["checks"]),
        "evidence_ids": sorted(str(payload["evidence_id"]) for payload in decoded_evidence.values()),
        "decision": decision,
        "reason_codes": reason_codes,
    }
    _validate_schema(outcome, "proxy_outcome_v2.schema.json")
    return outcome


def _validate_exact_evidence(
    contract: Mapping[str, Any], decoded_evidence: Mapping[str, Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(decoded_evidence, Mapping) or not decoded_evidence:
        raise ValueError("decoded immutable evidence must be a non-empty mapping")
    by_kind: dict[str, Mapping[str, Any]] = {}
    observed_kinds = []
    for evidence_id, payload in decoded_evidence.items():
        if not isinstance(payload, Mapping):
            raise ValueError("decoded immutable evidence payloads must be mappings")
        if payload.get("evidence_id") != evidence_id:
            raise ValueError("decoded immutable evidence key must equal evidence_id")
        kind = str(payload.get("evidence_kind") or "")
        observed_kinds.append(kind)
        if kind in by_kind:
            raise ValueError(f"duplicate decoded evidence kind: {kind}")
        by_kind[kind] = payload
        for field in (
            "attempt_id",
            "lifecycle_generation",
            "implementation_hash",
            "attempt_input_hash",
            "phase_execution_id",
            "phase_start_event_id",
        ):
            if payload.get(field) != contract[field]:
                raise ValueError(f"decoded evidence {field} mismatch")
        if payload.get("phase") != "proxy":
            raise ValueError("proxy classifier accepts only proxy-phase evidence")
    if Counter(observed_kinds) != Counter(contract["evidence_kinds"]):
        raise ValueError("decoded evidence kinds do not exactly match the frozen proxy contract")
    return by_kind


def _paired_deltas(
    contract: Mapping[str, Any], proxy_payload: Mapping[str, Any]
) -> tuple[list[str], float, dict[str, float]]:
    rows = proxy_payload.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("proxy results rows must be a sequence")
    expected = {
        (dataset_id, seed, metric_id, role)
        for dataset_id in contract["datasets"]
        for seed in contract["seeds"]
        for metric_id in contract["metric_ids"]
        for role in contract["roles"]
    }
    indexed: dict[tuple[str, int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("proxy result rows must be mappings")
        if row.get("phase") != "proxy" or row.get("command_status") != "completed":
            raise ValueError("proxy result rows must be completed proxy observations")
        key = (row.get("dataset_id"), row.get("seed"), row.get("metric_id"), row.get("role"))
        if key in indexed:
            raise ValueError("duplicate proxy result row identity")
        indexed[key] = row
        value = row.get("metric_value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("proxy metric values must be finite non-boolean numbers")
    if set(indexed) != expected:
        raise ValueError("proxy result row coverage does not exactly match the frozen proxy contract")

    primary = contract["primary_metric_id"]
    dataset_values: dict[str, list[float]] = {dataset_id: [] for dataset_id in contract["datasets"]}
    observation_ids = []
    for dataset_id in contract["datasets"]:
        for seed in contract["seeds"]:
            baseline = indexed[(dataset_id, seed, primary, "baseline")]
            candidate = indexed[(dataset_id, seed, primary, "candidate")]
            baseline_value = float(baseline["metric_value"])
            candidate_value = float(candidate["metric_value"])
            delta = candidate_value - baseline_value if contract["objective"] == "maximize" else baseline_value - candidate_value
            dataset_values[dataset_id].append(delta)
            for role, row in (("baseline", baseline), ("candidate", candidate)):
                observation_ids.append(
                    f"obs:{canonical_hash({'evidence_id': proxy_payload['evidence_id'], 'phase': 'proxy', 'phase_execution_id': contract['phase_execution_id'], 'role': role, 'dataset_id': dataset_id, 'metric_id': primary, 'seed': seed})}"
                )
    dataset_deltas = {
        dataset_id: sum(values) / len(values)
        for dataset_id, values in sorted(dataset_values.items())
    }
    all_deltas = [value for values in dataset_values.values() for value in values]
    return sorted(observation_ids), sum(all_deltas) / len(all_deltas), dataset_deltas


def _validate_activation(contract: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    if payload.get("status") != "passed" or payload.get("command_status") != "completed" or payload.get("exit_code") != 0:
        raise ValueError("activation evidence did not pass")
    surfaces = payload.get("implementation_surface_ids")
    if not isinstance(surfaces, Sequence) or isinstance(surfaces, (str, bytes)):
        raise ValueError("activation surfaces must be a sequence")
    if len(surfaces) != len(set(surfaces)) or set(surfaces) != set(contract["activation_surface_ids"]):
        raise ValueError("activation surfaces do not exactly match the frozen proxy contract")


def _validate_readiness(contract: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    if payload.get("ready") is not True:
        raise ValueError("full S3 readiness is not ready")
    checks = payload.get("checks")
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
        raise ValueError("readiness checks must be a sequence")
    check_ids = [item.get("check_id") for item in checks if isinstance(item, Mapping)]
    if len(check_ids) != len(checks) or len(check_ids) != len(set(check_ids)):
        raise ValueError("readiness checks must have unique check IDs")
    if set(check_ids) != set(contract["readiness_check_ids"]):
        raise ValueError("readiness checks do not exactly match the frozen proxy contract")
    if any(item.get("status") != "PASS" for item in checks):
        raise ValueError("a required readiness check did not pass")


def _contract_hash(contract: Mapping[str, Any]) -> str:
    return canonical_hash(_json_value({key: value for key, value in contract.items() if key != "contract_hash"}))


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    return value


def _schema_dir() -> Path:
    return Path(__file__).with_name("schemas")


def _validate_schema(payload: Mapping[str, Any], schema_name: str) -> None:
    import json

    schema = json.loads((_schema_dir() / schema_name).read_text(encoding="utf-8"))
    constraint_schema = json.loads((_schema_dir() / "constraint_result_v2.schema.json").read_text(encoding="utf-8"))
    registry = Registry().with_resource("constraint_result_v2.schema.json", Resource.from_contents(constraint_schema))
    errors = sorted(
        Draft202012Validator(schema, registry=registry).iter_errors(_json_value(payload)),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        messages = []
        for error in errors[:20]:
            location = ".".join(str(item) for item in error.absolute_path) or "$"
            messages.append(f"{location}: {error.message}")
        raise ValueError("; ".join(messages))


create_proxy_decision_contract = build_proxy_decision_contract
classify_proxy = classify_proxy_outcome
classify_proxy_decision = classify_proxy_outcome
