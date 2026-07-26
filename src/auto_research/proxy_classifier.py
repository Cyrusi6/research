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


PROXY_DECISION_POLICY_SCHEMA_VERSION = "auto_research_proxy_decision_policy_v2"
PROXY_DECISION_POLICY_SCHEMA_FILE = "proxy_decision_policy_v2.schema.json"
PROXY_EVALUATION_BINDING_SCHEMA_VERSION = "auto_research_proxy_evaluation_binding_v1"
PROXY_OUTCOME_SCHEMA_VERSION = "auto_research_proxy_outcome_v4"
PROXY_OUTCOME_SCHEMA_FILE = "proxy_outcome_v4.schema.json"
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
    activation_delta_threshold: float,
    activation_surface_ids: Sequence[str],
    readiness_check_ids: Sequence[str],
    readiness_check_plan_ref: Mapping[str, Any],
    readiness_check_plan_hash: str,
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
        "activation_delta_threshold": float(activation_delta_threshold),
        "activation_surface_ids": sorted(str(item) for item in activation_surface_ids),
        "readiness_check_ids": sorted(str(item) for item in readiness_check_ids),
        "readiness_check_plan_ref": _json_value(readiness_check_plan_ref),
        "readiness_check_plan_hash": readiness_check_plan_hash,
        "evidence_kinds": sorted(str(item) for item in evidence_kinds),
        "mode": mode,
        "route_semantics": {
            "science_reject": "PROPOSE_NEXT_VARIANT" if mode == "gate_to_full" else "FINISH_RUN",
            "implementation_blocked": "REPAIR_IMPLEMENTATION",
            "integrity_failure": "BLOCK_INTEGRITY",
            "resource_failure": "PAUSE_RESOURCE",
        },
    }
    policy["policy_hash"] = _object_hash(policy, "policy_hash")
    validate_proxy_decision_policy(policy)
    return policy


def validate_proxy_decision_policy(policy: Mapping[str, Any]) -> None:
    _validate_schema(policy, PROXY_DECISION_POLICY_SCHEMA_FILE)
    if policy["policy_hash"] != _object_hash(policy, "policy_hash"):
        raise ValueError("proxy decision policy hash mismatch")
    if set(policy["roles"]) != {"baseline", "candidate"}:
        raise ValueError("proxy policy roles must be exactly baseline and candidate")
    if policy["primary_metric_id"] not in set(policy["metric_ids"]):
        raise ValueError("proxy primary metric is not registered")
    if policy["readiness_check_plan_ref"]["digest"] != policy["readiness_check_plan_hash"]:
        raise ValueError("proxy policy readiness plan reference/hash mismatch")
    expected = {_PROXY_RESULTS_KIND, _ACTIVATION_KIND}
    expected.add(_READINESS_KIND if policy["mode"] == "gate_to_full" else _BOOTSTRAP_KIND)
    if not expected.issubset(set(policy["evidence_kinds"])):
        raise ValueError("proxy policy lacks mandatory authoritative evidence kinds")
    forbidden = {"effective_proxy_policy", "proxy_calibration_policy", "proxy_decision_report"}
    if forbidden.intersection(policy["evidence_kinds"]):
        raise ValueError("producer-authored proxy policy cannot be authoritative evidence")
    expected_routes = {
        "science_reject": "PROPOSE_NEXT_VARIANT" if policy["mode"] == "gate_to_full" else "FINISH_RUN",
        "implementation_blocked": "REPAIR_IMPLEMENTATION",
        "integrity_failure": "BLOCK_INTEGRITY",
        "resource_failure": "PAUSE_RESOURCE",
    }
    if policy["route_semantics"] != expected_routes:
        raise ValueError("proxy policy route semantics are not canonical")


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
    derivation_manifest_hash: str,
) -> dict[str, Any]:
    """Classify immutable proxy evidence without producer-authored control inputs."""

    validate_proxy_decision_policy(frozen_policy)
    _validate_binding_without_attempt(evaluation_binding, frozen_policy)
    evidence = _validate_exact_evidence(frozen_policy, evaluation_binding, decoded_evidence)
    proxy_payload = evidence[_PROXY_RESULTS_KIND]
    observation_ids, observed_delta, dataset_deltas = _paired_deltas(frozen_policy, evaluation_binding, proxy_payload)
    worst_regression = max(max(0.0, -value) for value in dataset_deltas.values())
    activation_passed = _activation_passed(frozen_policy, evidence[_ACTIVATION_KIND])
    readiness_passed = True
    if frozen_policy["mode"] == "gate_to_full":
        readiness_passed = _readiness_passed(frozen_policy, evidence[_READINESS_KIND])
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
    if not activation_passed:
        decision = "REPAIR_IMPLEMENTATION"
    elif frozen_policy["mode"] == "gate_to_full" and not readiness_passed:
        decision = "REPAIR_IMPLEMENTATION"
    elif frozen_policy["mode"] == "terminal_bootstrap":
        decision = "FINISH_RUN"
    else:
        decision = "RUN_FULL" if constraints_pass else "PROPOSE_NEXT_VARIANT"
    reason_codes = [
        "proxy_contract_constraints_pass" if constraints_pass else "proxy_contract_constraints_fail",
        "proxy_exact_coverage_pass",
        "proxy_activation_surfaces_pass" if activation_passed else "proxy_activation_blocked",
    ]
    if frozen_policy["mode"] == "gate_to_full":
        reason_codes.append(
            "proxy_readiness_checks_pass" if readiness_passed else "proxy_readiness_blocked"
        )
    else:
        reason_codes.append("bootstrap_completion_pass")
    outcome = {
        "schema_version": PROXY_OUTCOME_SCHEMA_VERSION,
        "attempt_id": evaluation_binding["attempt_id"],
        "lifecycle_generation": evaluation_binding["lifecycle_generation"],
        "implementation_hash": evaluation_binding["implementation_hash"],
        "attempt_input_hash": evaluation_binding["attempt_input_hash"],
        "phase_execution_id": evaluation_binding["phase_execution_id"],
        "phase_start_event_id": evaluation_binding["phase_start_event_id"],
        "proxy_decision_policy_hash": frozen_policy["policy_hash"],
        "readiness_check_plan_hash": frozen_policy["readiness_check_plan_hash"],
        "derivation_manifest_hash": derivation_manifest_hash,
        "proxy_evaluation_binding_hash": evaluation_binding["binding_hash"],
        "evidence_manifest_hash": evidence_manifest_hash,
        "evidence_set_hash": canonical_hash(sorted((key, value["evidence_id"]) for key, value in evidence.items())),
        "observed_delta": observed_delta,
        "dataset_deltas": dataset_deltas,
        "worst_dataset_regression": worst_regression,
        "constraint_results": results,
        "activation_surface_ids": list(frozen_policy["activation_surface_ids"]),
        "activation_status": "ACTIVATED" if activation_passed else "NOT_ACTIVATED",
        "readiness_check_ids": list(frozen_policy["readiness_check_ids"]),
        "readiness_status": "PASS" if readiness_passed else "BLOCKED",
        "evidence_ids": sorted(str(item["evidence_id"]) for item in evidence.values()),
        "decision": decision,
        "reason_codes": reason_codes,
    }
    _validate_schema(outcome, PROXY_OUTCOME_SCHEMA_FILE)
    return outcome


def classify_receipt_bound_proxy_outcome(
    *,
    frozen_policy: Mapping[str, Any],
    readiness_check_plan: Mapping[str, Any],
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
    activation_payload = next(
        (
            payload
            for payload in decoded.values()
            if payload.get("evidence_kind") == _ACTIVATION_KIND
        ),
        None,
    )
    if not isinstance(activation_payload, Mapping):
        raise ValueError("receipt-derived activation evidence is missing")
    _activation_passed(frozen_policy, activation_payload)
    readiness_payload = next(
        (
            payload
            for payload in decoded.values()
            if payload.get("evidence_kind") == _READINESS_KIND
        ),
        None,
    )
    if frozen_policy["mode"] == "gate_to_full":
        derived_readiness = derive_readiness_from_receipts(
            readiness_check_plan=readiness_check_plan,
            receipt_bound_sources=getattr(
                receipt_bound_evidence,
                "receipt_bound_sources",
                receipt_bound_evidence,
            ),
        )
        if derived_readiness["readiness_check_plan_hash"] != frozen_policy["readiness_check_plan_hash"]:
            raise ValueError("receipt-derived readiness plan hash differs from frozen proxy policy")
        plan_check_ids = [
            item["check_id"]
            for item in readiness_check_plan["checks"]
            if item["check_kind"] == "raw_measurement"
        ]
        if plan_check_ids != list(frozen_policy["readiness_check_ids"]):
            raise ValueError("ReadinessCheckPlan raw-measurement identities differ from frozen proxy policy")
        if not isinstance(readiness_payload, Mapping):
            raise ValueError("receipt-derived readiness evidence is missing")
        expected_readiness = {
            "readiness_check_plan_ref": frozen_policy["readiness_check_plan_ref"],
            "readiness_check_plan_hash": derived_readiness["readiness_check_plan_hash"],
            "ready": derived_readiness["ready"],
            "classification": derived_readiness["classification"],
            "checks": derived_readiness["checks"],
        }
        observed_readiness = {
            key: readiness_payload.get(key)
            for key in expected_readiness
        }
        if observed_readiness != expected_readiness:
            raise ValueError("normalized readiness differs from receipt-derived predicates")
    derivation_manifest_hash = receipt_bound_evidence.manifest.get("derivation_hash")
    if not isinstance(derivation_manifest_hash, str) or len(derivation_manifest_hash) != 64:
        raise ValueError("proxy evidence lacks one physical derivation manifest hash")
    if any(
        entry.get("derivation_hash") != derivation_manifest_hash
        for entry in receipt_bound_evidence.manifest.get("entries", [])
    ):
        raise ValueError("proxy evidence entries do not share the physical derivation manifest")
    return classify_proxy_outcome(
        frozen_policy=frozen_policy,
        evaluation_binding=evaluation_binding,
        decoded_evidence=decoded,
        evidence_manifest_hash=evidence_manifest_hash,
        derivation_manifest_hash=derivation_manifest_hash,
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


def _activation_passed(policy: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    if payload.get("status") == "command_failed":
        raise ValueError("failed activation command must use structured failure disposition")
    if payload.get("command_status") != "completed" or payload.get("exit_code") != 0:
        raise ValueError("activation semantic classification requires a completed command receipt")
    required = list(policy["activation_surface_ids"])
    if payload.get("expected_surface_ids") != required:
        raise ValueError("activation expected surfaces differ from frozen proxy policy")
    observed = payload.get("observed_surface_ids")
    if not isinstance(observed, Sequence) or isinstance(observed, (str, bytes)):
        raise ValueError("activation observed surfaces are malformed")
    if len(observed) != len(set(observed)) or any(item not in required for item in observed):
        raise ValueError("activation observed surfaces contain duplicates or unknown identities")
    threshold = policy["activation_delta_threshold"]
    if payload.get("activation_delta_threshold") != threshold:
        raise ValueError("activation evidence threshold differs from frozen proxy policy")
    measurements = payload.get("surface_measurements")
    if not isinstance(measurements, Sequence) or isinstance(measurements, (str, bytes)):
        raise ValueError("activation surface measurements are missing")
    indexed: dict[str, Mapping[str, Any]] = {}
    for measurement in measurements:
        if not isinstance(measurement, Mapping):
            raise ValueError("activation surface measurement is malformed")
        surface_id = measurement.get("surface_id")
        if not isinstance(surface_id, str) or not surface_id or surface_id in indexed:
            raise ValueError("activation surface measurement identity is invalid or duplicated")
        enabled = measurement.get("enabled_value")
        disabled = measurement.get("disabled_value")
        delta = measurement.get("delta")
        values = (enabled, disabled, delta, measurement.get("threshold"))
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
            raise ValueError("activation surface measurement values must be finite numbers")
        expected_delta = float(enabled) - float(disabled)
        if not math.isclose(float(delta), expected_delta, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("activation surface delta is not derived from enabled/disabled values")
        if float(measurement["threshold"]) != float(threshold):
            raise ValueError("activation surface threshold differs from frozen proxy policy")
        expected_measurement_status = "ACTIVATED" if expected_delta >= float(threshold) else "NOT_ACTIVATED"
        if measurement.get("status") != expected_measurement_status:
            raise ValueError("activation surface status differs from its frozen delta predicate")
        indexed[surface_id] = measurement
    if list(indexed) != list(observed):
        raise ValueError("activation measurements do not exactly match observed surface order")
    activated = set(observed) == set(required) and all(
        item["status"] == "ACTIVATED" for item in indexed.values()
    )
    expected_status = "activated" if activated else "not_activated"
    if payload.get("status") != expected_status:
        raise ValueError("activation summary differs from receipt-derived measurements")
    return activated


def _readiness_passed(policy: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    if payload.get("readiness_check_plan_ref") != policy["readiness_check_plan_ref"]:
        raise ValueError("readiness evidence plan reference differs from frozen proxy policy")
    if payload.get("readiness_check_plan_hash") != policy["readiness_check_plan_hash"]:
        raise ValueError("readiness evidence plan hash differs from frozen proxy policy")
    checks = payload.get("checks")
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
        raise ValueError("readiness checks are missing")
    check_ids = [item.get("check_id") for item in checks if isinstance(item, Mapping)]
    if (
        len(check_ids) != len(checks)
        or len(check_ids) != len(set(check_ids))
        or check_ids != list(policy["readiness_check_ids"])
    ):
        raise ValueError("readiness checks do not exactly match frozen policy")
    statuses = [item.get("status") for item in checks]
    if any(status not in {"PASS", "BLOCKED"} for status in statuses):
        raise ValueError("readiness check status is invalid")
    for item in checks:
        passed = _readiness_comparison_passes(
            item.get("comparator"), item.get("measurement"), item.get("threshold")
        )
        if item.get("status") != ("PASS" if passed else "BLOCKED"):
            raise ValueError("readiness check status differs from its measurement predicate")
    derived_ready = all(status == "PASS" for status in statuses)
    if payload.get("ready") is not derived_ready:
        raise ValueError("full S3 readiness summary differs from canonical checks")
    if payload.get("classification") != ("PASS" if derived_ready else "BLOCKED"):
        raise ValueError("full S3 readiness classification differs from canonical checks")
    return derived_ready


def derive_readiness_from_receipts(
    *,
    readiness_check_plan: Mapping[str, Any],
    receipt_bound_sources: Any,
) -> dict[str, Any]:
    """Derive readiness from validated physical receipt facts and a frozen check plan."""

    _validate_schema(readiness_check_plan, "readiness_check_plan_v2.schema.json")
    raw_facts = getattr(receipt_bound_sources, "raw_facts", None)
    raw_lineage = getattr(receipt_bound_sources, "raw_fact_lineage", None)
    if not isinstance(raw_facts, Mapping) or not isinstance(raw_lineage, Mapping):
        raise ValueError("readiness derivation requires receipt-bound physical sources")
    all_checks = readiness_check_plan.get("checks")
    if not isinstance(all_checks, Sequence) or isinstance(all_checks, (str, bytes)):
        raise ValueError("ReadinessCheckPlan checks are missing")
    checks_plan = [item for item in all_checks if item.get("check_kind") == "raw_measurement"]
    if not isinstance(checks_plan, Sequence) or isinstance(checks_plan, (str, bytes)):
        raise ValueError("ReadinessCheckPlan checks are missing")
    check_ids = [item.get("check_id") for item in checks_plan if isinstance(item, Mapping)]
    if len(check_ids) != len(checks_plan) or len(check_ids) != len(set(check_ids)):
        raise ValueError("ReadinessCheckPlan check identities are invalid")
    if list(raw_facts) != list(raw_lineage):
        raise ValueError("readiness raw fact and lineage inventories differ")
    expected_inventory = [
        (_readiness_source_key(binding), str(check_plan["check_id"]))
        for check_plan in checks_plan
        for binding in check_plan["source_bindings"]
    ]
    expected_check_ids = {str(item["check_id"]) for item in checks_plan}
    observed_inventory: list[tuple[tuple[str, str, str], str]] = []
    for key, source in raw_lineage.items():
        if not isinstance(source, Mapping):
            raise ValueError("readiness raw lineage is malformed")
        authority_roles = list(source.get("authority_roles") or [])
        readiness_check_ids = list(source.get("readiness_check_ids") or [])
        if "readiness" not in authority_roles:
            if any(str(check_id) in expected_check_ids for check_id in readiness_check_ids):
                raise ValueError("non-readiness raw fact declares readiness check authority")
            continue
        raw = raw_facts.get(key)
        if not isinstance(raw, Mapping):
            raise ValueError("readiness raw fact is malformed")
        if isinstance(raw.get("check_id"), str):
            observed_check_ids = [str(raw["check_id"])]
        elif isinstance(raw.get("readiness_checks"), Mapping):
            observed_check_ids = [str(check_id) for check_id in raw["readiness_checks"]]
        elif isinstance(raw.get("checks"), Sequence) and not isinstance(raw.get("checks"), (str, bytes)):
            observed_check_ids = [
                str(item.get("check_id"))
                for item in raw["checks"]
                if isinstance(item, Mapping) and isinstance(item.get("check_id"), str)
            ]
        else:
            raise ValueError("readiness-authority raw fact contains no observed check identity")
        if any(check_id not in readiness_check_ids for check_id in observed_check_ids):
            raise ValueError("readiness raw fact declares a check outside its frozen source authority")
        observed_inventory.extend((key, check_id) for check_id in observed_check_ids)
    if observed_inventory != expected_inventory:
        raise ValueError("readiness raw fact inventory differs from the frozen ordered exact set")
    checks = []
    for check_plan in checks_plan:
        check_id = check_plan["check_id"]
        facts: dict[str, Mapping[str, Any]] = {}
        for binding in check_plan["source_bindings"]:
            key = _readiness_source_key(binding)
            raw = raw_facts[key]
            source = raw_lineage[key]
            if not isinstance(raw, Mapping) or not isinstance(source, Mapping):
                raise ValueError("readiness raw fact or lineage is malformed")
            if not _valid_readiness_source_lineage(source, binding):
                raise ValueError("readiness receipt lineage identity is invalid")
            if check_id not in list(source.get("readiness_check_ids") or []):
                raise ValueError("readiness source is bound to a different frozen check")
            facts[binding["output_id"]] = raw
        passed, measurement = _evaluate_readiness_predicate(check_plan, facts)
        checks.append(
            {
                "check_id": check_id,
                "status": "PASS" if passed else "BLOCKED",
                "measurement": measurement,
                "comparator": check_plan["predicate"]["comparator"],
                "threshold": check_plan["predicate"]["threshold"],
            }
        )
    ready = all(item["status"] == "PASS" for item in checks)
    return {
        "readiness_check_plan_hash": canonical_hash(readiness_check_plan),
        "ready": ready,
        "classification": "PASS" if ready else "BLOCKED",
        "checks": checks,
    }


def _readiness_source_key(binding: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(binding["source_phase"]),
        str(binding["command_spec_id"]),
        str(binding["output_id"]),
    )


def _valid_readiness_source_lineage(
    source: Mapping[str, Any], binding: Mapping[str, Any]
) -> bool:
    return (
        source.get("source_phase") == binding["source_phase"]
        and source.get("command_spec_id") == binding["command_spec_id"]
        and source.get("output_id") == binding["output_id"]
        and source.get("output_kind") == binding["output_kind"]
        and set(binding.get("required_authority_roles") or []).issubset(
            set(source.get("authority_roles") or [])
        )
        and binding.get("check_id") in list(source.get("readiness_check_ids") or [])
        and source.get("command_status") == "completed"
        and source.get("exit_code") == 0
        and isinstance(source.get("receipt_hash"), str)
        and isinstance(source.get("receipt_ref"), Mapping)
        and isinstance(source.get("output_ref"), Mapping)
        and bool(source.get("completed_event_id"))
    )


def _evaluate_readiness_predicate(
    check_plan: Mapping[str, Any],
    facts: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, Any]:
    predicate = check_plan.get("predicate")
    if not isinstance(predicate, Mapping):
        raise ValueError("readiness predicate is missing")
    observed = _resolve_readiness_field_path(facts, predicate.get("field_path"))
    if not _readiness_coverage_passes(check_plan, facts):
        return False, observed
    return _readiness_comparison_passes(
        predicate.get("comparator"), observed, predicate.get("threshold")
    ), observed


def _readiness_comparison_passes(comparator: Any, observed: Any, threshold: Any) -> bool:
    if comparator == "eq":
        return observed == threshold
    if comparator == "exact_set":
        return _exact_set_equal(observed, threshold)
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise ValueError("readiness numeric predicate requires a finite non-boolean measurement")
    value = float(observed)
    if not math.isfinite(value) or isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("readiness predicate threshold is invalid")
    limit = float(threshold)
    if comparator in {"gte", "delta_gte"}:
        return value >= limit
    if comparator == "gt":
        return value > limit
    if comparator == "lte":
        return value <= limit
    if comparator == "lt":
        return value < limit
    raise ValueError("unsupported readiness predicate comparator")


def _resolve_readiness_field_path(
    facts: Mapping[str, Mapping[str, Any]], field_path: Any
) -> Any:
    if not isinstance(field_path, str) or not field_path:
        raise ValueError("readiness predicate field_path is invalid")
    matches = [
        output_id
        for output_id in facts
        if field_path == output_id or field_path.startswith(f"{output_id}.")
    ]
    if len(matches) == 1:
        output_id = matches[0]
        suffix = field_path[len(output_id):].removeprefix(".")
        parts = suffix.split(".") if suffix else []
        current: Any = facts[output_id]
    elif len(facts) == 1:
        parts = field_path.split(".")
        current = next(iter(facts.values()))
    elif not matches:
        values = [
            _resolve_readiness_parts(payload, field_path.split("."))
            for payload in facts.values()
        ]
        if not values or any(value != values[0] for value in values[1:]):
            raise ValueError("readiness predicate sources disagree on one logical measurement")
        return values[0]
    else:
        raise ValueError("readiness predicate field_path must name a source output")
    return _resolve_readiness_parts(current, parts)


def _resolve_readiness_parts(current: Any, parts: Sequence[str]) -> Any:
    for part in parts:
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)) and part.isdigit():
            index = int(part)
            if index >= len(current):
                raise ValueError("readiness predicate field_path index is out of range")
            current = current[index]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            matches = [
                item
                for item in current
                if isinstance(item, Mapping)
                and part in {
                    item.get("check_id"),
                    item.get("surface_id"),
                    item.get("output_id"),
                }
            ]
            if len(matches) != 1:
                raise ValueError("readiness predicate list identity is missing or duplicated")
            current = matches[0]
        else:
            raise ValueError("readiness predicate measurement is missing")
    return current


def _readiness_coverage_passes(
    check_plan: Mapping[str, Any], facts: Mapping[str, Mapping[str, Any]]
) -> bool:
    coverage = check_plan.get("required_coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("readiness required coverage is missing")
    expected = coverage.get("expected_surface_ids")
    if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes)):
        raise ValueError("readiness expected surface coverage is malformed")
    expected_set = {str(item) for item in expected}
    observed_candidates = [
        payload["observed_surface_ids"]
        for payload in facts.values()
        if "observed_surface_ids" in payload
    ]
    if not expected_set:
        return True
    if not observed_candidates:
        raise ValueError("readiness raw facts omit observed surface coverage")
    observed = [item for values in observed_candidates for item in values]
    if any(not isinstance(item, str) or not item for item in observed):
        raise ValueError("readiness observed surface identity is malformed")
    if len(observed) != len(set(observed)):
        raise ValueError("readiness observed surface coverage contains duplicates")
    observed_set = set(observed)
    if coverage.get("mode") == "exact":
        return observed_set == expected_set
    if coverage.get("mode") == "at_least":
        return expected_set.issubset(observed_set)
    raise ValueError("readiness coverage mode is invalid")


def _exact_set_equal(observed: Any, expected: Any) -> bool:
    if (
        not isinstance(observed, Sequence)
        or isinstance(observed, (str, bytes))
        or not isinstance(expected, Sequence)
        or isinstance(expected, (str, bytes))
    ):
        raise ValueError("exact_set readiness predicate requires arrays")
    if len(observed) != len(set(observed)) or len(expected) != len(set(expected)):
        raise ValueError("exact_set readiness predicate contains duplicates")
    return set(observed) == set(expected)


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
    registry = Registry()
    for referenced_name in (
        "constraint_result_v2.schema.json",
        "decoder_descriptor_v2.schema.json",
    ):
        referenced = json.loads((schema_dir / referenced_name).read_text(encoding="utf-8"))
        registry = registry.with_resource(referenced_name, Resource.from_contents(referenced))
    return Draft202012Validator(schema, registry=registry)


classify_proxy = classify_proxy_outcome
classify_proxy_decision = classify_proxy_outcome
