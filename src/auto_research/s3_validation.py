"""Shared fail-closed S3 validation over frozen authoritative contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .domain_contracts import (
    acceptance_contract_hash,
    canonical_hash,
    canonical_json,
    contract_errors,
    trial_spec_hash,
    validate_contract,
    validate_direction_identity,
    validate_evidence_manifest,
    validate_trial_result,
    validate_trial_spec,
    validate_variant_identity,
)
from .evidence import EvidenceStore, decode_receipt_bound_evidence_inventory
from .evidence_lineage import validate_receipt_bound_evidence
from .proxy_classifier import classify_receipt_bound_proxy_outcome
from .utils import read_json


class S3ValidationError(ValueError):
    """Raised when an S3 draft or projection violates the frozen contract."""


_IDENTITY_FIELDS = (
    "direction_id",
    "direction_semantic_hash",
    "direction_spec_hash",
    "variant_id",
    "variant_semantic_hash",
    "variant_spec_hash",
    "attempt_id",
)

_FAILURE_ROUTES = {
    "activation_failure": ("IMPLEMENTATION_REPAIR", "REPAIR_IMPLEMENTATION"),
    "implementation_failure": ("IMPLEMENTATION_REPAIR", "REPAIR_IMPLEMENTATION"),
    "resource_pause": ("RESOURCE_PAUSED", "PAUSE_RESOURCE"),
    "oom_retry": ("RESOURCE_PAUSED", "PAUSE_RESOURCE"),
    "integrity_failure": ("INTEGRITY_BLOCKED", "BLOCK_INTEGRITY"),
    "safety_failure": ("INTEGRITY_BLOCKED", "BLOCK_INTEGRITY"),
}


def validate_trial_precommit(
    *,
    project_root: Path,
    direction: dict[str, Any],
    variant: dict[str, Any],
    attempt: dict[str, Any],
    trial_spec: dict[str, Any],
    trial_result: dict[str, Any],
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pure validator used before commit, inside the ledger, and after commit."""

    errors: list[str] = []
    frozen = _frozen_trial_spec(attempt)
    _capture(errors, validate_direction_identity, direction)
    _capture(errors, validate_variant_identity, direction, variant)
    _capture(errors, validate_trial_spec, frozen)
    if canonical_json(trial_spec) != canonical_json(frozen):
        errors.append("supplied TrialSpec differs from frozen Attempt TrialSpec")
    _validate_trial_spec_projection_drift(errors, project_root, attempt, frozen)
    _validate_identity(errors, direction, variant, attempt, trial_result)
    _validate_attempt_hashes(errors, attempt, frozen)
    _capture(errors, validate_trial_result, trial_result, attempt=attempt, trial_spec=frozen)
    _validate_attempt_contract(errors, attempt, state=state, terminal=attempt.get("state") == "METHOD_COMPLETED")
    _validate_execution_state(errors, attempt, trial_result)
    _validate_evidence(errors, project_root, attempt, frozen, trial_result, state=state)
    if errors:
        raise S3ValidationError("; ".join(dict.fromkeys(errors)))
    return {
        "status": "PASS",
        "attempt_id": attempt["attempt_id"],
        "trial_spec_hash": trial_spec_hash(frozen),
        "completeness": trial_result["completeness"],
        "validated_artifact_hashes": dict(trial_result["raw_artifacts"]),
    }


def validate_ledger_trial_precommit(
    *,
    project_root: Path,
    state: dict[str, Any],
    trial_result: dict[str, Any],
) -> dict[str, Any]:
    attempt = (state.get("attempts") or {}).get(trial_result.get("attempt_id"))
    if not isinstance(attempt, dict):
        raise S3ValidationError("TrialResult attempt does not exist in authoritative state")
    direction_state = (state.get("directions") or {}).get(attempt.get("direction_semantic_hash"))
    direction = direction_state.get("spec") if isinstance(direction_state, dict) else None
    variant = (state.get("variants") or {}).get(attempt.get("variant_spec_hash"))
    if not isinstance(direction, dict) or not isinstance(variant, dict):
        raise S3ValidationError("attempt direction or variant contract is missing from authoritative state")
    frozen = _frozen_trial_spec(attempt)
    return validate_trial_precommit(
        project_root=project_root,
        direction=direction,
        variant=variant,
        attempt=attempt,
        trial_spec=frozen,
        trial_result=trial_result,
        state=state,
    )


def validate_failure_precommit(
    *,
    project_root: Path,
    attempt: dict[str, Any],
    failure_class: str,
    result: dict[str, Any],
    artifact_hashes: dict[str, str],
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    frozen = _frozen_trial_spec(attempt)
    _capture(errors, validate_trial_spec, frozen)
    _validate_trial_spec_projection_drift(errors, project_root, attempt, frozen)
    _validate_attempt_hashes(errors, attempt, frozen)
    _validate_attempt_contract(errors, attempt, state=state, terminal=False)
    if failure_class not in _FAILURE_ROUTES:
        errors.append(f"unsupported structured failure_class: {failure_class}")
    if result.get("method_evaluable") is True:
        errors.append("failure disposition cannot be method_evaluable")
    _validate_staged_artifacts(errors, project_root, artifact_hashes)
    explicit = result.get("failure_class") or result.get("failure_classification")
    if explicit != failure_class:
        errors.append("structured failure evidence must explicitly match failure_class")
    if not artifact_hashes:
        errors.append("failure disposition requires hashed raw evidence")
    if errors:
        raise S3ValidationError("; ".join(dict.fromkeys(errors)))
    target_state, next_action = _FAILURE_ROUTES[failure_class]
    return {"status": "PASS", "target_state": target_state, "next_action": next_action}


def validate_committed_s3(
    *,
    project_root: Path,
    direction: dict[str, Any],
    variant: dict[str, Any],
    state: dict[str, Any],
    attempt: dict[str, Any],
    route_outcome: dict[str, Any],
    trial_spec: dict[str, Any],
    trial_result: dict[str, Any] | None,
    proxy_event_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if attempt.get("method_evaluable"):
        if not isinstance(trial_result, dict):
            errors.append("method-evaluable attempt is missing TrialResult")
        else:
            try:
                validate_trial_precommit(
                    project_root=project_root,
                    direction=direction,
                    variant=variant,
                    attempt=attempt,
                    trial_spec=trial_spec,
                    trial_result=trial_result,
                    state=state,
                )
            except S3ValidationError as exc:
                errors.append(str(exc))
    if isinstance(attempt.get("committed_proxy_outcome"), dict):
        _validate_committed_proxy_evidence(
            errors,
            project_root=project_root,
            state=state,
            attempt=attempt,
            proxy_event_payload=proxy_event_payload,
        )
        _validate_proxy_route_and_state(
            errors,
            attempt=attempt,
            route_outcome=route_outcome,
            proxy_event_payload=proxy_event_payload,
        )
    errors.extend(contract_errors(route_outcome, "route_outcome_v4.schema.json"))
    source = route_outcome.get("source") if isinstance(route_outcome.get("source"), dict) else {}
    if source.get("attempt_id") != attempt.get("attempt_id") or not source.get("event_id"):
        errors.append("RouteOutcome source must bind the committed attempt and event")
    for key in _IDENTITY_FIELDS:
        if route_outcome.get("identity", {}).get(key) != attempt.get(key):
            errors.append(f"RouteOutcome {key} mismatch")
    budget = _direction_budget(state, attempt)
    if route_outcome.get("budget_snapshot") != budget:
        errors.append("RouteOutcome budget snapshot mismatch")
    expected_idempotency_key = canonical_hash(
        {
            "source_event_id": source.get("event_id"),
            "source_sequence": source.get("sequence"),
            "attempt_id": attempt.get("attempt_id"),
            "lifecycle_generation": attempt.get("lifecycle_generation"),
            "next_action": route_outcome.get("next_action"),
            "reason_codes": route_outcome.get("reason_codes"),
            "budget": budget,
            "artifact_hashes": attempt.get("artifact_hashes") or {},
            "variant_spec_hash": attempt.get("variant_spec_hash"),
        }
    )
    if route_outcome.get("idempotency_key") != expected_idempotency_key:
        errors.append("RouteOutcome idempotency key mismatch")
    if errors:
        raise S3ValidationError("; ".join(dict.fromkeys(errors)))
    return {"status": "PASS", "attempt_id": attempt["attempt_id"], "route_action": route_outcome["next_action"]}


def _frozen_trial_spec(attempt: dict[str, Any]) -> dict[str, Any]:
    frozen = attempt.get("frozen_trial_spec")
    if not isinstance(frozen, dict):
        raise S3ValidationError("Attempt is missing frozen TrialSpec snapshot")
    return frozen


def _validate_trial_spec_projection_drift(
    errors: list[str], project_root: Path, attempt: dict[str, Any], frozen: dict[str, Any]
) -> None:
    path = (
        project_root
        / "plan"
        / "attempts"
        / attempt["attempt_id"]
        / "trial_spec"
        / f"{attempt['trial_spec_hash']}.json"
    )
    if not path.exists():
        errors.append("canonical TrialSpec projection is missing")
        return
    projected = read_json(path, default=None)
    if not isinstance(projected, dict) or canonical_json(projected) != canonical_json(frozen):
        errors.append("canonical TrialSpec projection drifted from frozen Attempt snapshot")


def _validate_committed_proxy_evidence(
    errors: list[str],
    *,
    project_root: Path,
    state: dict[str, Any],
    attempt: dict[str, Any],
    proxy_event_payload: dict[str, Any] | None,
) -> None:
    committed = attempt.get("committed_proxy_outcome") or {}
    if not isinstance(proxy_event_payload, dict):
        errors.append("committed proxy outcome is missing its authoritative event payload")
        return
    supplied_outcome = proxy_event_payload.get("proxy_outcome")
    manifest = proxy_event_payload.get("evidence_manifest")
    if not isinstance(supplied_outcome, dict) or not isinstance(manifest, dict):
        errors.append("committed proxy event payload is malformed")
        return
    if supplied_outcome.get("attempt_id") != attempt.get("attempt_id"):
        errors.append("committed ProxyOutcome attempt identity mismatch")
        return
    state_outcome = (state.get("proxy_outcomes") or {}).get(attempt.get("attempt_id"))
    if canonical_json(state_outcome) != canonical_json(supplied_outcome):
        errors.append("committed ProxyOutcome projection differs from event payload")
    if supplied_outcome.get("evidence_manifest_hash") != canonical_hash(manifest):
        errors.append("committed ProxyOutcome evidence manifest hash mismatch")
    try:
        bound = validate_receipt_bound_evidence(
            project_root=project_root,
            attempt=attempt,
            trial_spec=_frozen_trial_spec(attempt),
            manifest=manifest,
            phase_commands=state.get("phase_commands") or {},
            phase="proxy",
        )
        if canonical_json(bound.manifest) != canonical_json(manifest):
            raise ValueError("proxy EvidenceManifest lineage is not canonical")
        phase_execution = (attempt.get("phase_executions") or {}).get("proxy") or {}
        binding = phase_execution.get("proxy_evaluation_binding")
        if not isinstance(binding, dict):
            raise ValueError("proxy phase lacks frozen evaluation binding")
        expected = classify_receipt_bound_proxy_outcome(
            frozen_policy=_frozen_trial_spec(attempt)["proxy_decision_policy"],
            readiness_check_plan=_proxy_readiness_check_plan(attempt),
            evaluation_binding=binding,
            receipt_bound_evidence=bound,
            evidence_manifest_hash=canonical_hash(bound.manifest),
        )
        if canonical_json(expected) != canonical_json(supplied_outcome):
            raise ValueError("ProxyOutcome differs from receipt-derived classifier result")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        errors.append(f"committed proxy evidence audit failed: {exc}")


def _proxy_readiness_check_plan(attempt: dict[str, Any]) -> dict[str, Any]:
    matches = [
        item
        for item in _frozen_trial_spec(attempt).get("phase_contracts", [])
        if item.get("phase") == "proxy"
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("readiness_check_plan"), dict):
        raise ValueError("proxy phase lacks frozen ReadinessCheckPlan")
    return matches[0]["readiness_check_plan"]


def _validate_proxy_route_and_state(
    errors: list[str],
    *,
    attempt: dict[str, Any],
    route_outcome: dict[str, Any],
    proxy_event_payload: dict[str, Any] | None,
) -> None:
    outcome = (proxy_event_payload or {}).get("proxy_outcome")
    committed = attempt.get("committed_proxy_outcome")
    if not isinstance(outcome, dict) or not isinstance(committed, dict):
        return
    decision = outcome.get("decision")
    expected = {
        "RUN_FULL": (
            {
                "PROXY_COMPLETED",
                "FULL_RUNNING",
                "RESOURCE_PAUSED",
                "IMPLEMENTATION_REPAIR",
                "INTEGRITY_BLOCKED",
                "ABANDONED",
                "METHOD_COMPLETED",
            },
            "COMPLETED",
            None,
        ),
        "REPAIR_IMPLEMENTATION": ({"IMPLEMENTATION_REPAIR"}, "FAILED", True),
        "PROPOSE_NEXT_VARIANT": ({"ABANDONED"}, "COMPLETED", False),
    }.get(decision)
    if expected is None:
        errors.append(f"unsupported committed ProxyOutcome decision: {decision}")
        return
    expected_states, expected_phase, expected_reserved = expected
    if attempt.get("state") not in expected_states:
        errors.append("Attempt state differs from committed ProxyOutcome decision")
    if (attempt.get("phases") or {}).get("proxy") != expected_phase:
        errors.append("Attempt proxy phase differs from committed ProxyOutcome decision")
    if (
        expected_reserved is not None
        and attempt.get("profile") == "standard"
        and attempt.get("reserved_slot") is not expected_reserved
    ):
        errors.append("Attempt reservation differs from committed ProxyOutcome decision")
    source = route_outcome.get("source") or {}
    if source.get("event_id") == committed.get("event_id"):
        if route_outcome.get("next_action") != decision:
            errors.append("RouteOutcome action differs from committed ProxyOutcome decision")
        if route_outcome.get("reason_codes") != outcome.get("reason_codes"):
            errors.append("RouteOutcome reasons differ from committed ProxyOutcome")


def _validate_attempt_hashes(errors: list[str], attempt: dict[str, Any], frozen: dict[str, Any]) -> None:
    expected = {
        "trial_spec_hash": trial_spec_hash(frozen),
        "protocol_hash": canonical_hash(frozen["protocol"]),
        "sample_manifest_hash": canonical_hash(frozen["sample_manifest"]),
        "acceptance_contract_hash": acceptance_contract_hash(frozen),
    }
    for key, value in expected.items():
        if attempt.get(key) != value:
            errors.append(f"Attempt {key} does not match frozen TrialSpec")
    if attempt.get("evaluator_hash") != frozen["execution_contract"]["evaluator_hash"]:
        errors.append("Attempt evaluator_hash does not match frozen TrialSpec")
    if attempt.get("seeds") != frozen["statistical_testing"]["seeds"]:
        errors.append("Attempt seeds do not match frozen TrialSpec")


def _validate_identity(errors, direction, variant, attempt, trial_result) -> None:
    for key in ("direction_id", "direction_semantic_hash", "direction_spec_hash"):
        if attempt.get(key) != direction.get(key) or trial_result.get(key) != attempt.get(key):
            errors.append(f"S3 {key} identity mismatch")
    for key in ("variant_id", "variant_semantic_hash", "variant_spec_hash"):
        if attempt.get(key) != variant.get(key) or trial_result.get(key) != attempt.get(key):
            errors.append(f"S3 {key} identity mismatch")
    for key in ("attempt_id", "attempt_input_hash", "protocol_hash", "trial_spec_hash", "acceptance_contract_hash"):
        if trial_result.get(key) != attempt.get(key):
            errors.append(f"TrialResult {key} mismatch")


def _validate_attempt_contract(errors, attempt, *, state, terminal) -> None:
    profile = attempt.get("profile")
    kind = attempt.get("attempt_kind")
    if profile == "bootstrap":
        if kind != "bootstrap_proxy" or attempt.get("consumes_direction_budget") is not False or attempt.get("reserved_slot") is not False:
            errors.append("bootstrap attempt profile/kind/budget mapping is invalid")
    elif profile == "standard":
        if kind not in {"proxy", "full", "proxy_full"} or attempt.get("consumes_direction_budget") is not True:
            errors.append("standard attempt profile/kind/budget mapping is invalid")
        expected_reserved = not (terminal and attempt.get("method_evaluable"))
        if attempt.get("reserved_slot") is not expected_reserved:
            errors.append("standard attempt reservation state is invalid")
    else:
        errors.append(f"unsupported attempt profile: {profile}")
    if state is not None:
        budget = _direction_budget(state, attempt)
        if budget.get("target") != 5 or min(int(budget.get("reserved", -1)), int(budget.get("consumed", -1))) < 0:
            errors.append("direction budget values are invalid")
        if int(budget.get("reserved", 0)) + int(budget.get("consumed", 0)) > 5:
            errors.append("direction budget exceeds target")


def _validate_execution_state(errors, attempt, trial_result) -> None:
    completeness = trial_result.get("completeness")
    state = attempt.get("state")
    phases = attempt.get("phases") or {}
    if completeness == "proxy":
        if attempt.get("attempt_kind") not in {"proxy", "bootstrap_proxy"}:
            errors.append("attempt kind does not permit terminal proxy outcome")
        if state not in {"PROXY_RUNNING", "METHOD_COMPLETED"}:
            errors.append("proxy TrialResult requires proxy execution state")
    elif completeness == "full":
        if attempt.get("attempt_kind") not in {"full", "proxy_full"}:
            errors.append("attempt kind does not permit full outcome")
        if state not in {"FULL_RUNNING", "METHOD_COMPLETED"}:
            errors.append("full TrialResult requires full execution state")
        frozen = _frozen_trial_spec(attempt)
        if "proxy" in frozen["protocol"]["required_phases"] and phases.get("proxy") != "COMPLETED":
            errors.append("full TrialResult requires completed preregistered proxy phase")
    if state == "METHOD_COMPLETED" and phases.get(completeness) != "COMPLETED":
        errors.append("METHOD_COMPLETED phase is inconsistent with TrialResult")


def _validate_evidence(errors, project_root, attempt, trial_spec, trial_result, *, state) -> None:
    manifest = trial_result.get("evidence_manifest")
    try:
        validate_evidence_manifest(manifest, trial_spec=trial_spec)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        return
    if manifest.get("attempt_id") != attempt.get("attempt_id"):
        errors.append("evidence manifest attempt identity mismatch")
    raw_artifacts = trial_result.get("raw_artifacts") or {}
    requirements = {item["kind"]: item for item in trial_spec["evidence_requirements"]}
    for entry in manifest["entries"]:
        path = entry["relative_path"]
        if raw_artifacts.get(path) != entry["content_hash"]:
            errors.append(f"evidence entry is not event-bound in raw_artifacts: {path}")
        if entry["attempt_id"] != attempt.get("attempt_id"):
            errors.append(f"evidence attempt identity mismatch: {path}")
        if entry["variant_spec_hash"] != attempt.get("variant_spec_hash"):
            errors.append(f"evidence variant identity mismatch: {path}")
        if entry["trial_spec_hash"] != attempt.get("trial_spec_hash"):
            errors.append(f"evidence TrialSpec identity mismatch: {path}")
        requirement = requirements.get(entry["kind"])
        if requirement is None:
            errors.append(f"evidence kind was not preregistered: {entry['kind']}")
        elif entry["schema_version"] != requirement["schema_version"]:
            errors.append(f"evidence schema version mismatch: {path}")
        try:
            EvidenceStore(project_root).read_entry(entry, attempt)
        except ValueError as exc:
            errors.append(str(exc))
    applicable_phases = {str(trial_result.get("completeness") or "")}
    required_kinds = {
        item["kind"]
        for item in trial_spec["evidence_requirements"]
        if item["required"] and ("always" in item["applicable_phases"] or applicable_phases.intersection(item["applicable_phases"]))
    }
    present_kinds = {item["kind"] for item in manifest["entries"]}
    missing = sorted(required_kinds - present_kinds)
    if missing:
        errors.append(f"required preregistered evidence is missing: {', '.join(missing)}")
    if not isinstance(state, dict) or not isinstance(state.get("phase_commands"), dict):
        errors.append("authoritative phase command state is required for receipt evidence lineage")
        return
    try:
        receipt_bound = decode_receipt_bound_evidence_inventory(
            project_root=project_root,
            attempt=attempt,
            trial_spec=trial_spec,
            manifest=manifest,
            phase_commands=state["phase_commands"],
            phase=str(trial_result.get("completeness")),
        )
        if canonical_json(receipt_bound.observations) != canonical_json(trial_result.get("observations") or []):
            errors.append("TrialResult observations differ from deterministic evidence decoding")
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))


def _direction_budget(state, attempt) -> dict[str, Any]:
    return dict(((state.get("directions") or {}).get(attempt.get("direction_semantic_hash")) or {}).get("budget") or {})


def _validate_staged_artifacts(errors: list[str], project_root: Path, artifact_hashes: dict[str, str]) -> None:
    store = EvidenceStore(project_root)
    for relative_path, expected_hash in artifact_hashes.items():
        try:
            raw = store.read_staged_source(relative_path)
        except ValueError as exc:
            errors.append(f"raw artifact rejected: {relative_path}: {exc}")
            continue
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            errors.append(f"raw artifact hash mismatch: {relative_path}")


def _capture(errors: list[str], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except (S3ValidationError, ValueError) as exc:
        errors.append(str(exc))
