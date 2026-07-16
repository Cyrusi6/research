"""Pure proxy classification from a frozen scientific policy and runtime binding."""

from __future__ import annotations

import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from .domain_contracts import canonical_hash


PROXY_DECISION_POLICY_SCHEMA_VERSION = "auto_research_proxy_decision_policy_v1"
PROXY_EVALUATION_BINDING_SCHEMA_VERSION = "auto_research_proxy_evaluation_binding_v1"
PROXY_OUTCOME_SCHEMA_VERSION = "auto_research_proxy_outcome_v3"
CONSTRAINT_RESULT_SCHEMA_VERSION = "auto_research_constraint_result_v2"

_PROXY_RESULTS_KIND = "proxy_results"
_ACTIVATION_KIND = "activation_evidence"
_READINESS_KIND = "full_s3_readiness"
_BOOTSTRAP_KIND = "bootstrap_completion"
_DECISION_CONSTRAINT_KINDS = {
    "minimum_mean_delta",
    "per_dataset_maximum_regression",
}


def build_proxy_decision_policy(
    *,
    primary_metric_id: str,
    objective: str,
    aggregation: str,
    datasets: Sequence[str],
    seeds: Sequence[int],
    metric_ids: Sequence[str],
    roles: Sequence[str],
    aggregate_improvement_threshold: float,
    per_dataset_maximum_regression: float,
    activation_surface_ids: Sequence[str],
    readiness_check_ids: Sequence[str],
    evidence_kinds: Sequence[str],
    mode: str,
) -> dict[str, Any]:
    policy = {
        "schema_version": PROXY_DECISION_POLICY_SCHEMA_VERSION,
        "primary_metric_id": primary_metric_id,
        "objective": objective,
        "aggregation": aggregation,
        "datasets": sorted(str(item) for item in datasets),
        "seeds": sorted(int(item) for item in seeds),
        "metric_ids": sorted(str(item) for item in metric_ids),
        "roles": sorted(str(item) for item in roles),
        "aggregate_improvement_threshold": float(aggregate_improvement_threshold),
        "per_dataset_maximum_regression": float(per_dataset_maximum_regression),
        "activation_surface_ids": sorted(str(item) for item in activation_surface_ids),
        "readiness_check_ids": sorted(str(item) for item in readiness_check_ids),
        "evidence_kinds": sorted(str(item) for item in evidence_kinds),
        "mode": mode,
        "route_semantics": {
            "science_reject": "PROPOSE_NEXT_VARIANT" if mode == "gate_to_full" else "FINISH_RUN",
            "integrity_failure": "BLOCK_INTEGRITY",
            "resource_failure": "PAUSE_RESOURCE",
        },
    }
    policy["policy_hash"] = _object_hash(policy, "policy_hash")
    validate_proxy_decision_policy(policy)
    return policy


def validate_proxy_decision_policy(policy: Mapping[str, Any]) -> None:
    _validate_schema(policy, "proxy_decision_policy_v1.schema.json")
    if policy["policy_hash"] != _object_hash(policy, "policy_hash"):
        raise ValueError("proxy decision policy hash mismatch")
    if set(policy["roles"]) != {"baseline", "candidate"}:
        raise ValueError("proxy policy roles must be exactly baseline and candidate")
    if policy["primary_metric_id"] not in set(policy["metric_ids"]):
        raise ValueError("proxy primary metric is not registered")
    expected = {_PROXY_RESULTS_KIND, _ACTIVATION_KIND}
    expected.add(_READINESS_KIND if policy["mode"] == "gate_to_full" else _BOOTSTRAP_KIND)
    if not expected.issubset(set(policy["evidence_kinds"])):
        raise ValueError("proxy policy lacks mandatory authoritative evidence kinds")
    forbidden = {"effective_proxy_policy", "proxy_calibration_policy", "proxy_decision_report"}
    if forbidden.intersection(policy["evidence_kinds"]):
        raise ValueError("producer-authored proxy policy cannot be authoritative evidence")


def build_proxy_evaluation_binding(
    *,
    attempt: Mapping[str, Any],
    phase_execution_id: str,
    phase_start_event_id: str,
    producer_run_id: str,
    command_plan_hash: str,
    phase_contract_hash: str,
    sample_contract_ref: Mapping[str, Any],
    evaluator_contract_ref: Mapping[str, Any],
    provenance_mode: str,
) -> dict[str, Any]:
    policy = attempt["frozen_trial_spec"].get("proxy_decision_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("proxy phase requires frozen ProxyDecisionPolicy")
    validate_proxy_decision_policy(policy)
    binding = {
        "schema_version": PROXY_EVALUATION_BINDING_SCHEMA_VERSION,
        "policy_hash": policy["policy_hash"],
        "attempt_id": attempt["attempt_id"],
        "direction_semantic_hash": attempt["direction_semantic_hash"],
        "direction_spec_hash": attempt["direction_spec_hash"],
        "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "phase_execution_id": phase_execution_id,
        "phase_start_event_id": phase_start_event_id,
        "producer_run_id": producer_run_id,
        "command_plan_hash": command_plan_hash,
        "phase_contract_hash": phase_contract_hash,
        "sample_contract_ref": dict(sample_contract_ref),
        "evaluator_contract_ref": dict(evaluator_contract_ref),
        "provenance_mode": provenance_mode,
        "expected_evidence_kinds": list(policy["evidence_kinds"]),
    }
    binding["binding_hash"] = _object_hash(binding, "binding_hash")
    validate_proxy_evaluation_binding(binding, policy=policy, attempt=attempt)
    return binding


def validate_proxy_evaluation_binding(
    binding: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> None:
    _validate_schema(binding, "proxy_evaluation_binding_v1.schema.json")
    validate_proxy_decision_policy(policy)
    if binding["binding_hash"] != _object_hash(binding, "binding_hash"):
        raise ValueError("proxy evaluation binding hash mismatch")
    expected = {
        "policy_hash": policy["policy_hash"],
        "attempt_id": attempt["attempt_id"],
        "direction_semantic_hash": attempt["direction_semantic_hash"],
        "direction_spec_hash": attempt["direction_spec_hash"],
        "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "expected_evidence_kinds": list(policy["evidence_kinds"]),
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError(f"proxy evaluation binding {key} mismatch")


def classify_proxy_outcome(
    *,
    frozen_policy: Mapping[str, Any],
    evaluation_binding: Mapping[str, Any],
    decoded_evidence: Mapping[str, Mapping[str, Any]],
    evidence_manifest_hash: str,
) -> dict[str, Any]:
    """Classify immutable proxy evidence without producer-authored control inputs."""

    validate_proxy_decision_policy(frozen_policy)
    _validate_binding_without_attempt(evaluation_binding, frozen_policy)
    evidence = _validate_exact_evidence(frozen_policy, evaluation_binding, decoded_evidence)
    proxy_payload = evidence[_PROXY_RESULTS_KIND]
    observation_ids, observed_delta, dataset_deltas = _paired_deltas(frozen_policy, evaluation_binding, proxy_payload)
    worst_regression = max(max(0.0, -value) for value in dataset_deltas.values())
    _validate_activation(frozen_policy, evidence[_ACTIVATION_KIND])
    if frozen_policy["mode"] == "gate_to_full":
        _validate_readiness(frozen_policy, evidence[_READINESS_KIND])
    else:
        _validate_bootstrap(evidence[_BOOTSTRAP_KIND])

    proxy_evidence_id = str(proxy_payload["evidence_id"])
    results = [
        _constraint_result(
            constraint_id="proxy-aggregate-improvement",
            kind="minimum_mean_delta",
            observed=observed_delta,
            threshold=frozen_policy["aggregate_improvement_threshold"],
            passed=observed_delta >= float(frozen_policy["aggregate_improvement_threshold"]),
            policy=frozen_policy,
            observation_ids=observation_ids,
            evidence_id=proxy_evidence_id,
        ),
        _constraint_result(
            constraint_id="proxy-per-dataset-regression",
            kind="per_dataset_maximum_regression",
            observed={"worst_regression": worst_regression, "deltas": dataset_deltas},
            threshold=frozen_policy["per_dataset_maximum_regression"],
            passed=worst_regression <= float(frozen_policy["per_dataset_maximum_regression"]),
            policy=frozen_policy,
            observation_ids=observation_ids,
            evidence_id=proxy_evidence_id,
        ),
    ]
    constraints_pass = all(item["status"] == "PASS" for item in results)
    if frozen_policy["mode"] == "terminal_bootstrap":
        decision = "FINISH_RUN"
    else:
        decision = "RUN_FULL" if constraints_pass else "PROPOSE_NEXT_VARIANT"
    outcome = {
        "schema_version": PROXY_OUTCOME_SCHEMA_VERSION,
        "attempt_id": evaluation_binding["attempt_id"],
        "lifecycle_generation": evaluation_binding["lifecycle_generation"],
        "implementation_hash": evaluation_binding["implementation_hash"],
        "attempt_input_hash": evaluation_binding["attempt_input_hash"],
        "phase_execution_id": evaluation_binding["phase_execution_id"],
        "phase_start_event_id": evaluation_binding["phase_start_event_id"],
        "proxy_decision_policy_hash": frozen_policy["policy_hash"],
        "proxy_evaluation_binding_hash": evaluation_binding["binding_hash"],
        "evidence_manifest_hash": evidence_manifest_hash,
        "evidence_set_hash": canonical_hash(sorted((key, value["evidence_id"]) for key, value in evidence.items())),
        "observed_delta": observed_delta,
        "dataset_deltas": dataset_deltas,
        "worst_dataset_regression": worst_regression,
        "constraint_results": results,
        "activation_surface_ids": list(frozen_policy["activation_surface_ids"]),
        "readiness_check_ids": list(frozen_policy["readiness_check_ids"]),
        "evidence_ids": sorted(str(item["evidence_id"]) for item in evidence.values()),
        "decision": decision,
        "reason_codes": [
            "proxy_contract_constraints_pass" if constraints_pass else "proxy_contract_constraints_fail",
            "proxy_exact_coverage_pass",
            "proxy_activation_surfaces_pass",
            "proxy_readiness_checks_pass" if frozen_policy["mode"] == "gate_to_full" else "bootstrap_completion_pass",
        ],
    }
    _validate_schema(outcome, "proxy_outcome_v3.schema.json")
    return outcome


def classify_receipt_bound_proxy_outcome(
    *,
    frozen_policy: Mapping[str, Any],
    evaluation_binding: Mapping[str, Any],
    receipt_bound_evidence: Any,
    evidence_manifest_hash: str,
) -> dict[str, Any]:
    """Classify only evidence already proven against command receipts.

    ``ReceiptBoundEvidence`` is intentionally duck-typed here to keep the pure
    classifier free of project storage concerns.  The lineage map must cover
    the decoded inventory exactly, preventing callers from mixing receipt-bound
    rows with producer-authored qualitative evidence.
    """

    decoded = getattr(receipt_bound_evidence, "decoded_evidence", None)
    lineage = getattr(receipt_bound_evidence, "lineage", None)
    if not isinstance(decoded, Mapping) or not isinstance(lineage, Mapping):
        raise ValueError("proxy classification requires receipt-bound evidence")
    if set(decoded) != set(lineage):
        raise ValueError("proxy evidence lineage does not exactly cover decoded evidence")
    for evidence_id, payload in decoded.items():
        item = lineage[evidence_id]
        if getattr(item, "evidence_id", None) != evidence_id:
            raise ValueError("proxy evidence lineage identity mismatch")
        if getattr(item, "evidence_kind", None) != payload.get("evidence_kind"):
            raise ValueError("proxy evidence lineage kind mismatch")
        if not getattr(item, "command_id", None) or not getattr(item, "receipt_hash", None):
            raise ValueError("proxy evidence lacks completed command receipt identity")
    return classify_proxy_outcome(
        frozen_policy=frozen_policy,
        evaluation_binding=evaluation_binding,
        decoded_evidence=decoded,
        evidence_manifest_hash=evidence_manifest_hash,
    )


def _constraint_result(*, constraint_id: str, kind: str, observed: Any, threshold: float, passed: bool, policy: Mapping[str, Any], observation_ids: list[str], evidence_id: str) -> dict[str, Any]:
    return {
        "schema_version": CONSTRAINT_RESULT_SCHEMA_VERSION,
        "constraint_id": constraint_id,
        "kind": kind,
        "hard": True,
        "status": "PASS" if passed else "FAIL",
        "observed": observed,
        "threshold": threshold,
        "objective": policy["objective"],
        "observation_ids": observation_ids,
        "evidence_ids": [evidence_id],
    }


def _validate_binding_without_attempt(binding: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    _validate_schema(binding, "proxy_evaluation_binding_v1.schema.json")
    if binding["binding_hash"] != _object_hash(binding, "binding_hash"):
        raise ValueError("proxy evaluation binding hash mismatch")
    if binding["policy_hash"] != policy["policy_hash"]:
        raise ValueError("proxy evaluation binding policy mismatch")
    if binding["expected_evidence_kinds"] != policy["evidence_kinds"]:
        raise ValueError("proxy evaluation binding evidence set mismatch")


def _validate_exact_evidence(policy: Mapping[str, Any], binding: Mapping[str, Any], decoded: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for payload in decoded.values():
        kind = str(payload.get("evidence_kind") or "")
        if kind in indexed:
            raise ValueError("duplicate decoded evidence kind")
        indexed[kind] = payload
    if set(indexed) != set(policy["evidence_kinds"]):
        raise ValueError("authoritative proxy evidence kinds do not exactly match frozen policy")
    identity = {
        "attempt_id": binding["attempt_id"],
        "lifecycle_generation": binding["lifecycle_generation"],
        "implementation_hash": binding["implementation_hash"],
        "attempt_input_hash": binding["attempt_input_hash"],
        "phase": "proxy",
        "phase_execution_id": binding["phase_execution_id"],
        "phase_start_event_id": binding["phase_start_event_id"],
        "producer_run_id": binding["producer_run_id"],
    }
    for payload in indexed.values():
        for key, value in identity.items():
            if payload.get(key) != value:
                raise ValueError(f"proxy evidence {key} mismatch")
    return indexed


def _paired_deltas(policy: Mapping[str, Any], binding: Mapping[str, Any], payload: Mapping[str, Any]) -> tuple[list[str], float, dict[str, float]]:
    rows = payload.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("proxy results rows are missing")
    expected = {(dataset, seed, metric, role) for dataset in policy["datasets"] for seed in policy["seeds"] for metric in policy["metric_ids"] for role in policy["roles"]}
    indexed: dict[tuple[str, int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("phase") != "proxy" or row.get("command_status") != "completed":
            raise ValueError("proxy result row is not a completed proxy measurement")
        key = (str(row.get("dataset_id")), row.get("seed"), str(row.get("metric_id")), str(row.get("role")))
        if key in indexed:
            raise ValueError("duplicate proxy result row identity")
        value = row.get("metric_value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("proxy metric values must be finite non-boolean numbers")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("proxy result row coverage does not exactly match frozen policy")
    primary = policy["primary_metric_id"]
    values: dict[str, list[float]] = {dataset: [] for dataset in policy["datasets"]}
    observation_ids: list[str] = []
    for dataset in policy["datasets"]:
        for seed in policy["seeds"]:
            baseline = indexed[(dataset, seed, primary, "baseline")]
            candidate = indexed[(dataset, seed, primary, "candidate")]
            delta = float(candidate["metric_value"]) - float(baseline["metric_value"])
            if policy["objective"] == "minimize":
                delta = -delta
            values[dataset].append(delta)
            for role in ("baseline", "candidate"):
                observation_ids.append(f"obs:{canonical_hash({'evidence_id': payload['evidence_id'], 'phase_execution_id': binding['phase_execution_id'], 'role': role, 'dataset_id': dataset, 'metric_id': primary, 'seed': seed})}")
    dataset_deltas = {dataset: sum(items) / len(items) for dataset, items in sorted(values.items())}
    all_deltas = [value for items in values.values() for value in items]
    return sorted(observation_ids), sum(all_deltas) / len(all_deltas), dataset_deltas


def _validate_activation(policy: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    if payload.get("status") != "passed" or payload.get("command_status") != "completed" or payload.get("exit_code") != 0:
        raise ValueError("activation evidence did not pass")
    surfaces = payload.get("implementation_surface_ids")
    if not isinstance(surfaces, Sequence) or isinstance(surfaces, (str, bytes)) or set(surfaces) != set(policy["activation_surface_ids"]) or len(surfaces) != len(set(surfaces)):
        raise ValueError("activation surfaces do not exactly match frozen policy")


def _validate_readiness(policy: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    checks = payload.get("checks")
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
        raise ValueError("readiness checks are missing")
    check_ids = [item.get("check_id") for item in checks if isinstance(item, Mapping)]
    if len(check_ids) != len(checks) or len(check_ids) != len(set(check_ids)) or set(check_ids) != set(policy["readiness_check_ids"]):
        raise ValueError("readiness checks do not exactly match frozen policy")
    derived_ready = all(item.get("status") == "PASS" for item in checks)
    if not derived_ready or payload.get("ready") is not derived_ready:
        raise ValueError("full S3 readiness did not pass canonical checks")


def derive_readiness_from_receipts(
    *,
    required_check_ids: Sequence[str],
    raw_checks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive readiness solely from frozen check identities and raw command facts."""

    required = list(required_check_ids)
    if len(required) != len(set(required)) or set(raw_checks) != set(required):
        raise ValueError("readiness raw checks do not exactly match the frozen check set")
    checks = []
    for check_id in required:
        raw = raw_checks[check_id]
        status = str(raw.get("status") or "").upper()
        exit_code = raw.get("exit_code")
        passed = status == "PASS" and exit_code == 0 and raw.get("ready", True) is not False
        checks.append({"check_id": check_id, "status": "PASS" if passed else "BLOCKED"})
    return {"ready": all(item["status"] == "PASS" for item in checks), "checks": checks}


def _validate_bootstrap(payload: Mapping[str, Any]) -> None:
    if payload.get("completion_status") != "verified":
        raise ValueError("bootstrap completion is not verified")


def _object_hash(payload: Mapping[str, Any], hash_field: str) -> str:
    return canonical_hash({key: _json_value(value) for key, value in payload.items() if key != hash_field})


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    return value


def _validate_schema(payload: Mapping[str, Any], schema_name: str) -> None:
    validator = _schema_validator(schema_name)
    errors = sorted(validator.iter_errors(_json_value(payload)), key=lambda error: list(error.absolute_path))
    if errors:
        raise ValueError("; ".join(f"{'.'.join(str(item) for item in error.absolute_path) or '$'}: {error.message}" for error in errors[:20]))


@lru_cache(maxsize=None)
def _schema_validator(schema_name: str) -> Draft202012Validator:
    schema_dir = Path(__file__).with_name("schemas")
    schema = json.loads((schema_dir / schema_name).read_text(encoding="utf-8"))
    constraint_schema = json.loads((schema_dir / "constraint_result_v2.schema.json").read_text(encoding="utf-8"))
    registry = Registry().with_resource("constraint_result_v2.schema.json", Resource.from_contents(constraint_schema))
    return Draft202012Validator(schema, registry=registry)


classify_proxy = classify_proxy_outcome
classify_proxy_decision = classify_proxy_outcome
